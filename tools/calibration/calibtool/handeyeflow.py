"""Hand-eye calibration from the `calib` menu (plan docs/plans/2026-11-11-handeye-plan.md, a first version of H5):
the robot's stored arm-wave plans (config/handeye/<bot>.json), one arm after the other, read B and the URDF fit on the
robot, a results table. Laptop side, standard library only; every robot step goes over ssh (injected for the tests).

The session, in order (the operator is told what happens before anything moves):
  1. the kit copied to the robot (cli.bootstrap: calibtool, scripts, config, armcal)
  2. the preflight per arm, read-only (scripts/handeye_wave.py --check): the fixed motion tool, motors on (the tool never
     enables them), nobody else commanding, the body as in the plan's snapshot (else: re-plan first), depth flowing
  3. the operator's Y; then per arm: the wave detached on the robot (it survives a lost link: the arm holds wherever
     it stops), its log followed on screen; S + Enter writes the STOP file (the runner stops before its next command,
     the arm holds); read B over the arm's captures on the robot (scripts/handeye_match.sh)
  4. the fit of both arms on the robot (armcal.urdffit): the corrected URDF written in the session folder, NOT deployed
     (the deploy is milestone H4), and the table: held-out wrist error per arm, new vs the unmodified URDF.
"""
from __future__ import annotations

import json
import select
import shlex
import subprocess
import sys
import time
from pathlib import Path

from calibtool import cli, paths

CONFIG_DIR = paths.CAL / "config" / "handeye"
ARMS = ("Right", "Left")
POLL_S = 2.0


def setup_path(bot) -> Path:
    return CONFIG_DIR / f"{bot}.json"


def available(p) -> bool:
    """offered only for a robot whose profile opts in (handeye: true) and that has a stored setup"""
    return bool(getattr(p, "handeye", False)) and setup_path(p.bot).is_file()


def stdin_line_ready() -> str | None:
    """a line typed on the laptop, without blocking (None when nothing was typed)"""
    try:
        r, _, _ = select.select([sys.stdin], [], [], 0)
    except (OSError, ValueError):
        return None
    return sys.stdin.readline().strip() if r else None


class HandEye:
    def __init__(self, p, run=subprocess.run, out=print, ask=input, sleep=time.sleep, typed=stdin_line_ready,
                 bootstrap=None, now=time.strftime, poll_s=POLL_S, setup=None):
        self.p, self.run, self.out, self.ask, self.sleep, self.typed = p, run, out, ask, sleep, typed
        self.bootstrap = bootstrap if bootstrap is not None else (lambda: cli.bootstrap(p, run=run))
        self.cfg = setup if setup is not None else json.loads(setup_path(p.bot).read_text())
        self.sid = now("%Y%m%d-%H%M%S")
        self.dir = f"{p.root}/handeye/{self.sid}"
        self.poll_s = poll_s

    # ------------------------------------------------------------ robot steps
    def sh(self, line, timeout=60):
        r = self.run(cli.ssh_argv(self.p, line), capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout or ""), (r.stderr or "")

    def py(self, args) -> str:
        """the shell line that runs the kit python (calibtool and armcal on the path) with `args`"""
        return f"{cli.kit_python(self.p)} {args}"

    def plan_file(self, arm) -> str:
        return f"{self.dir}/plan-{arm.lower()}.json"

    def upload_plans(self):
        self.sh(f"mkdir -p {shlex.quote(self.dir)}")
        for arm in ARMS:
            body = json.dumps(dict(robot=self.cfg["robot"], arm=arm, snapshot=self.cfg["snapshot"],
                                   moves=self.cfg["arms"][arm]["moves"]))
            rc, _, err = self.sh(f"cat > {shlex.quote(self.plan_file(arm))} <<'EOF_PLAN'\n{body}\nEOF_PLAN")
            if rc != 0:
                raise RuntimeError(f"could not write the {arm} plan on the robot: {err.strip()}")

    def wave_args(self, arm) -> str:
        c = self.cfg
        a = [f"scripts/handeye_wave.py {shlex.quote(self.plan_file(arm))} --arm {arm}",
             f"--tool {shlex.quote(self.p.motor_tool)}", f"--grab {shlex.quote(c['grab'])}",
             f"--grab-pydeps {shlex.quote(c['grab_pydeps'])}"]
        return " ".join(a)

    def preflight(self, arm) -> dict:
        rc, out, err = self.sh(self.py(self.wave_args(arm) + " --check"), timeout=90)
        try:
            return json.loads(out.strip().splitlines()[-1])
        except (ValueError, IndexError):
            return {"ok": False, "problems": [f"the preflight did not answer: {(err or out).strip()[-300:]}"]}

    def start_wave(self, arm):
        d = f"{self.dir}/{arm.lower()}"
        extra = f" --frames {int(self.cfg.get('frames', 5))}"
        if self.cfg.get("profiles"):
            extra += f" --profiles {shlex.quote(self.cfg['profiles'])}"
        line = (f"mkdir -p {shlex.quote(d)} && ( {self.py(self.wave_args(arm) + ' --out ' + shlex.quote(d) + extra)}"
                f" > {shlex.quote(d)}/run.out 2>&1 < /dev/null & ) ; echo started")
        rc, out, err = self.sh(f"setsid bash -c {shlex.quote(line)}")
        if "started" not in out:
            raise RuntimeError(f"the {arm} wave did not start: {err.strip()[-300:]}")
        return d

    def follow(self, d) -> str:
        """print the wave's log as it grows; S + Enter stops it. -> 'done' | 'stopped' | 'lost'"""
        seen, quiet = 0, 0
        while True:
            t = self.typed()
            if t is not None and t.lower() in ("s", "stop"):
                self.sh(f"touch {shlex.quote(d)}/STOP")
                self.out("  STOP written: the arm stops before its next command and holds there")
            # [h]: the pattern must not match this ssh shell's own command line (design 7: pgrep -f traps)
            rc, out, _ = self.sh(f"tail -n +{seen + 1} {shlex.quote(d)}/log.jsonl 2>/dev/null; "
                                 f"pgrep -f {shlex.quote('[h]andeye_wave.py ' + d.rsplit('/', 1)[0])} >/dev/null "
                                 "&& echo __ALIVE__", timeout=30)
            alive = "__ALIVE__" in out
            for ln in out.splitlines():
                if ln == "__ALIVE__":
                    continue
                try:
                    e = json.loads(ln)
                except ValueError:
                    continue
                seen += 1
                self.out("  " + describe(e))
                if e.get("kind") == "done":
                    return "done"
                if e.get("kind") == "stopped":
                    return "stopped"
            if not alive:
                quiet += 1
                if quiet >= 3:
                    return "lost"
            self.sleep(self.poll_s)

    def match(self, arm, d) -> dict:
        line = (f"CODE={shlex.quote(cli.kit_dir(self.p))} PY={shlex.quote(self.p.python)} bash "
                f"{shlex.quote(cli.kit_dir(self.p))}/scripts/handeye_match.sh {shlex.quote(d)} {arm.lower()} "
                f"{shlex.quote(self.cfg['urdf'])} 4")
        rc, out, err = self.sh(line, timeout=900)
        tail = [ln for ln in out.splitlines() if "captures," in ln or "wall time" in ln]
        return {"rc": rc, "summary": tail, "json": f"{d}/readb-{arm.lower()}.json"}

    def fit(self, rights, lefts) -> dict:
        urdf_out = f"{self.dir}/{self.cfg['robot']}_mk1_handeye-{self.sid[:8]}.urdf"
        rep = f"{self.dir}/fit.json"
        args = (f"-m armcal.urdffit --right {shlex.quote(rights)} --left {shlex.quote(lefts)} "
                f"--urdf {shlex.quote(self.cfg['urdf'])} --report {shlex.quote(rep)} --out-urdf {shlex.quote(urdf_out)} "
                f"--note {shlex.quote('calib hand-eye session ' + self.sid)}")
        rc, out, err = self.sh(self.py(args), timeout=600)
        if rc != 0:
            return {"rc": rc, "error": (err or out).strip()[-400:]}
        rc2, js, _ = self.sh(f"cat {shlex.quote(rep)}")
        d = json.loads(js) if rc2 == 0 and js.strip() else {}
        return {"rc": 0, "report": d, "urdf": urdf_out}

    # ------------------------------------------------------------ the session
    def session(self) -> int:
        say = self.out
        say(f"Hand-eye calibration on {self.cfg['robot']} (session {self.sid}). Nothing moves before your Y.")
        say("  copying the kit to the robot ...")
        self.bootstrap()
        self.upload_plans()
        problems = []
        for arm in ARMS:
            pre = self.preflight(arm)
            for pr in pre.get("problems", []):
                if pr not in problems:
                    problems.append(pr)
        if problems:
            say("Not ready, nothing moved:")
            for pr in problems:
                say(f"  - {pr}")
            return 2
        n = {a: sum(len(m.get("necks", [])) for m in self.cfg["arms"][a]["moves"] if m.get("burst")) for a in ARMS}
        say("Ready. What happens next:")
        say(f"  the RIGHT arm, the neck and the right hand move ({n['Right']} captures, about "
            f"{n['Right'] * 15 // 60 + 2} min), then the LEFT arm ({n['Left']} captures, about "
            f"{n['Left'] * 15 // 60 + 2} min); every other joint is held where it is now")
        say("  stand clear of both arms; S + Enter stops the arm where it is (it holds)")
        if (self.ask("Start? [Y/n]: ") or "y").strip().lower() not in ("y", "yes"):
            say("  nothing moved")
            return 1
        recs = {}
        for arm in ARMS:
            say(f"{arm} arm: starting")
            d = self.start_wave(arm)
            how = self.follow(d)
            if how != "done":
                say(f"{arm} arm: {how}. The arm holds where it is; the session stops here (data kept in {d}).")
                return 3
            say(f"{arm} arm: done, matching the captures on the robot ...")
            m = self.match(arm, d)
            for ln in m["summary"]:
                say("  " + ln.strip())
            if m["rc"] != 0:
                say(f"  read B failed on the {arm} arm; the session stops (data kept)")
                return 4
            recs[arm] = m["json"]
        say("Fitting both arms on the robot ...")
        f = self.fit(recs["Right"], recs["Left"])
        if f["rc"] != 0:
            say(f"  the fit failed: {f['error']}")
            return 5
        for ln in table(f["report"]):
            say(ln)
        say(f"The corrected URDF is on the robot at {f['urdf']} (NOT deployed: the deploy is still to come).")
        return 0


def describe(e) -> str:
    t = time.strftime("%H:%M:%S", time.localtime(e.get("t", 0)))
    k = e.get("kind", "?")
    if k == "burst":
        return f"{t} capture {e.get('name')} ({e.get('dt')} s)"
    if k.startswith("arm:"):
        return f"{t} arm to {k[4:]} ({e.get('dt')} s)"
    if k.startswith("neck:"):
        return f"{t} neck {k[5:]}"
    if k == "stopped":
        return f"{t} STOPPED: {e.get('reason')}"
    if k == "start":
        return f"{t} started"
    return f"{t} {k}"


def table(report) -> list:
    chosen = report.get("chosen")
    cv = (report.get("cv") or {}).get(chosen) or {}
    L = [f"Result (model {chosen}; wrist error on arm poses left out of the fit):",
         "  arm    | captures | median | 9 in 10 under | unmodified URDF",
         "  -------+----------+--------+---------------+----------------"]
    for arm in ARMS:
        c = cv.get(arm) or {}
        if c:
            L.append(f"  {arm:6s} | {c.get('n', 0):8d} | {c.get('median', float('nan')):4.1f} mm | "
                     f"{c.get('p90', float('nan')):10.1f} mm | {c.get('base_median', float('nan')):9.1f} mm")
    L.append("  How to read it: lower is better; 'unmodified URDF' is the same captures before the correction.")
    return L


def run_session(p, **kw) -> int:
    return HandEye(p, **kw).session()
