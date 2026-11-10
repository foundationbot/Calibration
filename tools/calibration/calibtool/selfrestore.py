"""The stale-lease self-restore (design plan section 4: "A lease older than N hours with no TUI (proposal 2 h) makes the
robot restore itself"): a calibration session that died with the camera producer at its collection size and OKVIS
stopped is undone by the robot, from a systemd timer, without anyone at the laptop.

`python -m calibtool restore --if-stale-hours 2 --root <dir> [--dry-run]` acts only when ALL hold:
  - <root>/producer.lease exists and its heartbeat is older than N hours (producer.Lease: calibd beats it every
    minute while a session waits);
  - no calibd holds <root>/calibd.lock (daemon.RobotLock): it is taken here, non-blocking, and held for the whole
    restore, so no calibd can start half-way;
  - no TUI heartbeat for N hours (the session's state.json "last_hb_t", and the events of a TUI attaching, keying or
    being lost);
  - the session never started a deploy (flow "deploy_started"): an interrupted or unaccepted deploy changed robot.env
    and the producer's size together, and undoing only the producer would leave OKVIS on a config for another size.
    That case is REFUSED loudly (exit 1, the session's rollback.sh named): a person decides.
Otherwise it does nothing and says why (exit 0): safe to run every 10 minutes.

The restore, in this order (each step reported; the first failure stops the ones after it, exit 1, the lease kept so
the next run retries):
  1. the producer as saved in preflight (the lease's spec_path, the session's producer.saved.yaml): the live spec
     compared first, a producer already as saved is not restarted; else producer.restore's no-overlap cycle (scale 0,
     wait until no producer pod exists, replace with the saved spec, scale back): it never scales up while an old
     producer pod remains (PRODUCER LEFT AT 0 REPLICAS, said as loudly as the cycle says it);
  2. OKVIS started again by the path the session recorded (state.json flow.okvis: the node label back to the value it
     had on the DaemonSet robots, okvis.sh start on the older ones) -- only after the producer is back, as OKVIS needs
     the cameras at their saved size;
  3. the lease released, the session's state.json updated (flow.producer.changed / flow.okvis.stopped false,
     "stale_restore"), and the session's events.jsonl ("stale_restore", "stale_restore_failed",
     "stale_restore_refused") and commands.txt (every command sent) appended to.
A missing saved producer spec refuses the restore (nothing sent). No recorded OKVIS state with a deploy snapshot:
OKVIS is left as it is, said so.

`python -m calibtool install-restore-timer --root <dir> [--hours 2] [--dry-run]` writes calibtool-restore.service +
calibtool-restore.timer (every 10 minutes) to /etc/systemd/system and enables the timer; ExecStart is this
interpreter, this kit (WorkingDirectory, PYTHONPATH) and `restore --if-stale-hours H --root <dir>`. --dry-run prints
the files and commands and changes nothing. `uninstall-restore-timer [--dry-run]` disables the timer and removes both
files."""
import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_HOURS = 2.0
UNIT = "calibtool-restore"
UNIT_DIR = "/etc/systemd/system"
TUI_EVENTS = frozenset({"tui_heartbeat", "client_attached", "controller", "key", "link_lost"})
LEASE_FILE, LOCK_FILE = "producer.lease", "calibd.lock"


def _say(log, text):
    log(f"calibtool restore: {text}")


def last_tui_t(session_dir):
    """the newest sign of a TUI on this session (wall clock s), or None: state.json's last_hb_t and the times of the
    events a TUI leaves (attach, heartbeat start, key, controller, link lost)"""
    from calibtool import daemon
    st = daemon.Store(session_dir)
    ts = []
    state = st.load_state() or {}
    if state.get("last_hb_t") is not None:
        ts.append(float(state["last_hb_t"]))
    ts += [float(e["t"]) for e in st.events() if e.get("kind") in TUI_EVENTS and e.get("t") is not None]
    return max(ts) if ts else None


class Restore:
    """one run. kube_runner: producer.Kube's runner (kubectl_runner() on the robot); shell: deploy.Remote's runner
    (shell_runner()); log: a line printer (the journal); cycle_kw: producer cycle keywords (tests: sleep, clock,
    timeouts)"""

    def __init__(self, root, hours=DEFAULT_HOURS, clock=time.time, kube_runner=None, shell=None, log=print,
                 dry_run=False, cycle_kw=None):
        self.root, self.max_age_s = Path(root), float(hours) * 3600.0
        self.hours = float(hours)
        self.clock, self.log, self.dry_run = clock, log, bool(dry_run)
        self.kube_runner, self.shell, self.cycle_kw = kube_runner, shell, dict(cycle_kw or {})

    # ------------------------------------------------------------ the decision

    def run(self) -> int:
        from calibtool import daemon, producer as pr
        lease = pr.Lease(self.root / LEASE_FILE)
        d = lease.read()
        now = self.clock()
        if d is None:
            _say(self.log, f"no lease at {lease.path}: nothing to restore")
            return 0
        age = now - float(d.get("heartbeat") or 0.0)
        if age <= self.max_age_s:
            _say(self.log, f"lease of session {d.get('session')} is {age / 3600:.2f} h old (stale after "
                           f"{self.hours:g} h): nothing done")
            return 0
        lock = daemon.RobotLock(self.root / LOCK_FILE)
        if not lock.acquire("stale-restore"):
            _say(self.log, f"a calibd holds {lock.path} ({lock.holder().strip()}): nothing done")
            return 0
        try:
            return self._locked(lease, now)
        finally:
            lock.release()

    def _locked(self, lease, now) -> int:
        from calibtool import daemon
        d = lease.read()                     # again under the lock: a calibd may have released or beaten it
        if d is None or now - float(d.get("heartbeat") or 0.0) <= self.max_age_s:
            _say(self.log, "the lease changed before the lock was taken: nothing done")
            return 0
        sid = str(d.get("session") or "")
        sdir = self.root / "sessions" / sid
        if not sid or not sdir.is_dir():
            _say(self.log, f"REFUSED: the lease names session {sid!r}, whose folder {sdir} does not exist: no saved "
                           "producer spec or OKVIS state to restore from; nothing sent")
            return 1
        tui = last_tui_t(sdir)
        if tui is not None and now - tui <= self.max_age_s:
            _say(self.log, f"session {sid}: a TUI was there {(now - tui) / 3600:.2f} h ago: nothing done")
            return 0
        store = daemon.Store(sdir)
        state = store.load_state() or {}
        fl = state.get("flow") or {}
        info = {"session": sid, "lease_age_h": round((now - float(d["heartbeat"])) / 3600, 3),
                "tui_age_h": None if tui is None else round((now - tui) / 3600, 3), "dry_run": self.dry_run}
        if fl.get("deploy_started") and not fl.get("rolled_back"):
            what = "an interrupted deploy" if not fl.get("deploy_done") else "a deploy not accepted at FINISH"
            why = (f"session {sid} has {what}: robot.env and the producer size changed together; not undoing only "
                   f"the producer. A person decides: {sdir / 'deploy' / 'rollback.sh'} rolls the deploy back")
            return self._refuse(store, state, d, why, info)
        spec_path = Path(d.get("spec_path") or sdir / "producer.saved.yaml")
        if not spec_path.is_file():
            return self._refuse(store, state, d, f"no saved producer spec at {spec_path}: nothing restored", info)
        return self._restore(lease, store, state, spec_path, info)

    def _refuse(self, store, state, lease_d, why, info) -> int:
        _say(self.log, f"REFUSED: {why}")
        prev = state.get("stale_restore") or {}
        same = prev.get("refused") == why and prev.get("lease_created") == lease_d.get("created")
        if not self.dry_run and not same:
            store.event("stale_restore_refused", t=self.clock(), why=why, **info)       # once per lease and reason
            state["stale_restore"] = {"t": self.clock(), "refused": why, "lease_created": lease_d.get("created")}
            if (store.dir / "state.json").is_file():
                store.write_state(state)
        return 1

    # ------------------------------------------------------------ the restore

    def _restore(self, lease, store, state, spec_path, info) -> int:
        from calibtool import deploy as dp, producer as pr, sessionctl as sc
        spec = pr.Spec.from_yaml(spec_path.read_text())
        md = spec.doc.get("metadata") or {}
        ns, name = md.get("namespace") or "dma-video", md.get("name") or "producer"
        fl = state.setdefault("flow", {}) if state else {}
        okv = fl.get("okvis") or {}
        snap = (store.dir / "deploy" / "deploy_snapshot.json").is_file()
        rec = sc.Recorder(dry_run=self.dry_run)
        if not self.dry_run:
            rec.attach(store.dir / "commands.txt")
        rec.note(f"stale-lease self-restore (lease {info['lease_age_h']} h old, no calibd, no TUI)")
        if self.dry_run:
            _say(self.log, f"DRY RUN, nothing sent. Would restore session {info['session']}: producer {ns}/{name} "
                           f"as saved in {spec_path} (no-overlap cycle unless already as saved); "
                           + (f"OKVIS started by {okv.get('restart', {}).get('kind')} (node {okv.get('node')}, label "
                              f"back to {okv.get('label_value')})" if okv.get("stopped") else "OKVIS left as it is "
                              "(not recorded stopped)") + "; the lease released")
            return 0
        done, err = [], None
        store.event("stale_restore_start", t=self.clock(), spec=str(spec_path), producer=f"{ns}/{name}", **info)

        def logged(argv, input=None):
            if argv and argv[0] not in pr.READ_VERBS:
                rec.add("producer", ["k0s", "kubectl", *argv], input)
            run = self.kube_runner or pr.kubectl_runner()
            return run(argv, input=input) if input is not None else run(argv)
        kube = pr.Kube(logged, ns, name, read_only=False)
        try:
            live = pr.Spec.from_yaml(kube.get_yaml())
            if (live.args, live.image, live.replicas) == (spec.args, spec.image, spec.replicas) and \
                    any(p["ready"] and p["phase"] == "Running" for p in kube.pods()):
                done.append("producer already as saved and running: not restarted")
            else:
                pr.restore(kube, spec, lease=None, **self.cycle_kw)
                done.append(f"producer restored as saved ({' '.join(map(str, spec.size()))})")
            fl.setdefault("producer", {})["changed"] = False
        except Exception as e:  # noqa: BLE001  (every failure is loud and stops here)
            err = f"producer: {type(e).__name__}: {e}"
        if err is None:
            try:
                if okv.get("stopped"):
                    rem = dp.Remote(self.shell or dp.shell_runner(), log=rec.hook("okvis"))
                    ctl = sc.OkvisCtl.from_dict(rem, okv)
                    ctl.start()
                    fl["okvis"] = ctl.as_dict()
                    done.append(f"OKVIS started ({ctl.kind})")
                elif okv:
                    done.append("OKVIS was not stopped: left as it is")
                else:
                    done.append("no OKVIS state recorded" + (" (deploy snapshot present, OKVIS never stopped by the "
                                                             "session's record)" if snap else "") + ": left as it is")
            except Exception as e:  # noqa: BLE001
                err = f"OKVIS: {type(e).__name__}: {e}"
        if err is None:
            lease.release()
            done.append("lease released")
        state["stale_restore"] = {"t": self.clock(), "done": done, "error": err}
        if (store.dir / "state.json").is_file():
            store.write_state(state)
        if err is not None:
            store.event("stale_restore_failed", t=self.clock(), done=done, error=err, **info)
            _say(self.log, f"FAILED, the robot is NOT restored: {err}. Done before it: {done or 'nothing'}. The lease "
                           "is kept: the next run retries")
            return 1
        store.event("stale_restore", t=self.clock(), done=done, **info)
        _say(self.log, f"session {info['session']} restored: " + "; ".join(done))
        return 0


# ---------------------------------------------------------------- the systemd units


def units(root, hours=DEFAULT_HOURS, python=None, kit=None, pythonpath=None) -> dict:
    """{file name: text} of the service and its timer (every 10 minutes)"""
    python = python or sys.executable
    kit = str(kit or Path(__file__).resolve().parents[1])
    pp = ":".join(dict.fromkeys([kit] + [p for p in (pythonpath if pythonpath is not None
                                                     else os.environ.get("PYTHONPATH", "")).split(":") if p]))
    root = str(Path(root).resolve())
    cmd = shlex.join([python, "-m", "calibtool", "restore", "--if-stale-hours", f"{float(hours):g}", "--root", root])
    service = (f"[Unit]\nDescription=calibtool: restore the camera producer and OKVIS after a calibration session's "
               f"lease went stale ({float(hours):g} h, no calibd, no TUI)\n"
               f"ConditionPathExists={root}/{LEASE_FILE}\n\n"
               f"[Service]\nType=oneshot\nWorkingDirectory={kit}\nEnvironment=PYTHONPATH={pp}\n"
               f"Environment=OMP_NUM_THREADS=1\nNice=10\nExecStart={cmd}\n")
    timer = (f"[Unit]\nDescription=calibtool: stale calibration lease check every 10 minutes\n\n"
             f"[Timer]\nOnCalendar=*:0/10\nAccuracySec=1min\nUnit={UNIT}.service\n\n"
             f"[Install]\nWantedBy=timers.target\n")
    return {f"{UNIT}.service": service, f"{UNIT}.timer": timer}


def _systemctl(run, *args):
    argv = ["systemctl", *args]
    r = run(argv)
    rc = r.returncode if hasattr(r, "returncode") else int(r)
    if rc != 0:
        raise RuntimeError(f"{shlex.join(argv)} failed (rc {rc})")


def install(root, hours=DEFAULT_HOURS, unit_dir=UNIT_DIR, dry_run=False, run=subprocess.run, log=print, **kw) -> int:
    files = units(root, hours, **kw)
    cmds = [["daemon-reload"], ["enable", "--now", f"{UNIT}.timer"]]
    if dry_run:
        for n, text in files.items():
            log(f"# DRY RUN: would write {Path(unit_dir) / n}\n{text}")
        for c in cmds:
            log("# DRY RUN: would run " + shlex.join(["systemctl", *c]))
        return 0
    for n, text in files.items():
        (Path(unit_dir) / n).write_text(text)
        log(f"wrote {Path(unit_dir) / n}")
    for c in cmds:
        _systemctl(run, *c)
        log("ran " + shlex.join(["systemctl", *c]))
    return 0


def uninstall(unit_dir=UNIT_DIR, dry_run=False, run=subprocess.run, log=print) -> int:
    names = [f"{UNIT}.timer", f"{UNIT}.service"]
    if dry_run:
        log(f"# DRY RUN: would run systemctl disable --now {UNIT}.timer, remove "
            + ", ".join(str(Path(unit_dir) / n) for n in names) + ", systemctl daemon-reload")
        return 0
    _systemctl(run, "disable", "--now", f"{UNIT}.timer")
    for n in names:
        try:
            (Path(unit_dir) / n).unlink()
            log(f"removed {Path(unit_dir) / n}")
        except FileNotFoundError:
            pass
    _systemctl(run, "daemon-reload")
    return 0


# ---------------------------------------------------------------- the commands


def main_restore(argv) -> int:
    ap = argparse.ArgumentParser(prog="python -m calibtool restore", description=__doc__.split("\n")[0])
    ap.add_argument("--if-stale-hours", type=float, default=DEFAULT_HOURS, dest="hours",
                    help="act only when the lease and the TUI are older than this (h)")
    ap.add_argument("--root", required=True, help="calibd's root (the lease, the lock, the sessions)")
    ap.add_argument("--dry-run", action="store_true", help="decide and say what would be done; send nothing")
    a = ap.parse_args(argv)
    return Restore(a.root, a.hours, dry_run=a.dry_run, log=lambda s: print(s, flush=True)).run()


def main_install(argv) -> int:
    ap = argparse.ArgumentParser(prog="python -m calibtool install-restore-timer")
    ap.add_argument("--root", required=True)
    ap.add_argument("--hours", type=float, default=DEFAULT_HOURS)
    ap.add_argument("--unit-dir", default=UNIT_DIR)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    return install(a.root, a.hours, a.unit_dir, a.dry_run)


def main_uninstall(argv) -> int:
    ap = argparse.ArgumentParser(prog="python -m calibtool uninstall-restore-timer")
    ap.add_argument("--unit-dir", default=UNIT_DIR)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    return uninstall(a.unit_dir, a.dry_run)
