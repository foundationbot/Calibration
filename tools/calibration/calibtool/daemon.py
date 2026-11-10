"""calibd: the robot-side job runner (design plan sections 3, 4, 7, 10.1).

Store        <session>/state.json (atomic: tmp + os.replace) after every step and <session>/events.jsonl (one JSON line
             per event, flushed and fsynced), so a crash, a reboot or a dropped link loses nothing.
Machine      the session state machine. Phases: PREFLIGHT -> READY (or BLOCKED: a collect-gate check failed) -> TRIAL
             -> DONE; HOLD and ABORTED. Active phases (TRIAL; the M5 grids later) need a TUI heartbeat: none for
             DEADMAN_S (5 s) -> HOLD. HOLD is left only by key G while a TUI is attached (heartbeat live): nothing
             resumes by itself. Key A aborts from any phase. Machine.resume() reloads state.json: an active phase
             comes back as HOLD.
Server       a Unix socket in the session folder; each client gets the state as one JSON line every period and sends
             {"hb": t}, {"key": "A"} or {"cmd": "stop"} lines. The laptop reaches it through
             `ssh robot python -m calibtool.daemon attach`, which bridges stdin/stdout to the socket: when ssh dies the
             heartbeat stops and the machine holds; a new attach picks the session up where it is.
RobotLock    flock on <root>/calibd.lock: one calibration job per robot.
Daemon       one loop step (~2 Hz): deadman tick, capture counters, live detection of the newest pair (process pool),
             the health watch on capture counters + system samples (its events go to events.jsonl and the NOT NORMAL
             banner), the placement trial while TRIAL (paused in HOLD), state saved.
M2 is READ-ONLY on the robot: the producer and motors are only read (producer.Kube(read_only=True)); the banner says so.

CLI (on the robot): python -m calibtool.daemon {start|run|attach|stop|status} --root R [--session S] [--profile P10]
[--trial]. start spawns `run` in its own session (setsid) and returns once the socket answers."""
import argparse
import fcntl
import json
import os
import select
import socket
import sys
import threading
import time
from pathlib import Path

DEADMAN_S = 5.0
ACTIVE = frozenset({"TRIAL"})
TERMINAL = frozenset({"DONE", "ABORTED"})
STEP_S = 0.5
SAMPLE_S = 1.0
HEALTH_KEEP = 30


# ---------------------------------------------------------------- persistence

class Store:
    def __init__(self, d):
        self.dir = Path(d)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write_state(self, d):
        tmp = self.dir / "state.json.tmp"
        with self._lock:
            with open(tmp, "w") as f:
                json.dump(d, f, default=_json_default)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.dir / "state.json")

    def load_state(self):
        try:
            return json.loads((self.dir / "state.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def event(self, kind, t=None, **kw):
        rec = {"t": time.time() if t is None else t, "kind": kind, **kw}
        line = json.dumps(rec, default=_json_default)
        with self._lock:
            with open(self.dir / "events.jsonl", "a") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())
        return rec

    def events(self) -> list:
        out = []
        try:
            for ln in (self.dir / "events.jsonl").read_text().splitlines():
                try:
                    out.append(json.loads(ln))
                except json.JSONDecodeError:
                    continue            # a torn last line (crash mid-write)
        except FileNotFoundError:
            pass
        return out


def _json_default(o):
    import numpy as np
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)


# ---------------------------------------------------------------- the state machine

class Machine:
    def __init__(self, store: Store, clock=time.time, session="", robot="", deadman_s=DEADMAN_S, read_only=True,
                 mode="watch", _state=None):
        self.store, self.clock, self.deadman_s = store, clock, deadman_s
        self._lock = threading.RLock()
        self.pending_keys = []
        self.last_hb = None
        if _state is not None:
            self.state = _state
            return
        t = clock()
        self.state = {"session": session, "robot": robot, "phase": "PREFLIGHT", "status": "running",
                      "read_only": read_only, "mode": mode, "held_from": None, "attached": False, "created": t,
                      "updated": t, "banner": [], "checks": [], "capture": {}, "health": [], "board": None,
                      "trial": None, "say": None, "preflight_green": None, "pid": os.getpid()}
        store.event("session_start", t=t, session=session, robot=robot, read_only=read_only, mode=mode)
        self.save()

    @classmethod
    def resume(cls, store: Store, clock=time.time, deadman_s=DEADMAN_S):
        """the machine as state.json left it: an active phase comes back as HOLD, an unfinished preflight reruns"""
        st = store.load_state()
        if st is None:
            raise FileNotFoundError(f"no state.json in {store.dir}")
        m = cls(store, clock, deadman_s=deadman_s, _state=st)
        before = st["phase"]
        if before in ACTIVE:
            st["held_from"], st["phase"] = before, "HOLD"
        elif before in ("PREFLIGHT", "READY", "BLOCKED"):
            st["phase"] = "PREFLIGHT"
        st["attached"] = False
        st["pid"] = os.getpid()
        store.event("resumed", t=clock(), phase_before=before, phase=st["phase"])
        m.save()
        return m

    @property
    def phase(self):
        return self.state["phase"]

    def save(self):
        with self._lock:
            self.state["updated"] = self.clock()
            self.store.write_state(self.state)

    def snapshot(self) -> dict:
        with self._lock:
            return json.loads(json.dumps(self.state, default=_json_default))

    def to(self, phase, why=""):
        with self._lock:
            if self.phase in TERMINAL or phase == self.phase:
                return
            self.store.event("phase", t=self.clock(), frm=self.phase, to=phase, why=why)
            self.state["phase"] = phase
            if phase == "ABORTED":
                self.state["status"] = "aborted"
            elif phase == "DONE":
                self.state["status"] = "completed"
            self.save()

    def live(self) -> bool:
        return self.last_hb is not None and self.clock() - self.last_hb <= self.deadman_s

    def heartbeat(self):
        with self._lock:
            self.last_hb = self.clock()
            if not self.state["attached"]:
                self.state["attached"] = True
                self.store.event("tui_heartbeat", t=self.last_hb)

    def tick(self):
        with self._lock:
            live = self.live()
            changed = False
            if self.state["attached"] and not live:
                self.state["attached"] = False
                self.store.event("link_lost", t=self.clock(), last_hb=self.last_hb)
                changed = True
            if self.phase in ACTIVE and not live:
                self.state["held_from"] = self.phase
                self.store.event("heartbeat_lost", t=self.clock(), phase=self.phase, deadman_s=self.deadman_s)
                self.to("HOLD", f"no TUI heartbeat for {self.deadman_s:g} s")
                changed = False
            if changed:
                self.save()

    def key(self, k):
        k = str(k).upper()
        with self._lock:
            self.store.event("key", t=self.clock(), key=k, phase=self.phase)
            if self.phase in TERMINAL:
                return
            if k == "A":
                self.to("ABORTED", "abort key")
            elif k == "G":
                if self.phase == "HOLD" and self.live() and self.state.get("held_from"):
                    back = self.state["held_from"]
                    self.store.event("resumed_by_operator", t=self.clock(), to=back)
                    self.state["held_from"] = None
                    self.to(back, "G with the TUI attached")
            elif k == "T":
                if self.phase == "READY":
                    self.to("TRIAL", "T: placement trial")
            else:
                self.pending_keys.append(k)

    def take_keys(self) -> list:
        with self._lock:
            k, self.pending_keys = self.pending_keys, []
            return k


# ---------------------------------------------------------------- the socket link

class Server:
    def __init__(self, path, machine: Machine, period_s=STEP_S, on_stop=None):
        self.path, self.m, self.period, self.on_stop = Path(path), machine, period_s, on_stop
        self._sock = None
        self._stop = threading.Event()
        self._threads = []

    def start(self):
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.bind(str(self.path))
        s.listen(4)
        s.settimeout(0.2)
        self._sock = s
        t = threading.Thread(target=self._accept, name="calibd-accept", daemon=True)
        t.start()
        self._threads.append(t)

    def _accept(self):
        while not self._stop.is_set():
            try:
                c, _ = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            t = threading.Thread(target=self._client, args=(c,), name="calibd-client", daemon=True)
            t.start()
            self._threads.append(t)

    def _client(self, c):
        self.m.store.event("client_attached", t=self.m.clock())
        buf = b""
        try:
            while not self._stop.is_set():
                c.sendall((json.dumps(self.m.snapshot(), default=_json_default) + "\n").encode())
                end = time.monotonic() + self.period
                while not self._stop.is_set():
                    left = end - time.monotonic()
                    if left <= 0:
                        break
                    r, _, _ = select.select([c], [], [], left)
                    if not r:
                        continue
                    data = c.recv(65536)
                    if not data:
                        return
                    buf += data
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        self._handle(line)
        except (OSError, ValueError):
            return
        finally:
            self.m.store.event("client_detached", t=self.m.clock())
            try:
                c.close()
            except OSError:
                pass

    def _handle(self, line):
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            return
        if "hb" in msg:
            self.m.heartbeat()
        if "key" in msg:
            self.m.heartbeat()
            self.m.key(msg["key"])
        if msg.get("cmd") == "stop" and self.on_stop:
            self.m.store.event("stop_requested", t=self.m.clock())
            self.on_stop()

    def stop(self):
        self._stop.set()
        if self._sock is not None:
            self._sock.close()
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


class RobotLock:
    def __init__(self, path):
        self.path = Path(path)
        self._f = None

    def acquire(self, session) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        f = open(self.path, "a+")
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            f.close()
            return False
        f.seek(0)
        f.truncate()
        f.write(json.dumps({"session": session, "pid": os.getpid(), "t": time.time()}))
        f.flush()
        self._f = f
        return True

    def holder(self) -> str:
        try:
            return self.path.read_text()
        except FileNotFoundError:
            return ""

    def release(self):
        if self._f is not None:
            fcntl.flock(self._f, fcntl.LOCK_UN)
            self._f.close()
            self._f = None


# ---------------------------------------------------------------- the loop

class Daemon:
    """detector(pair) -> Future of (left corners, right corners); to_obs(left corners) -> detect.BoardObs | None;
    sampler() -> system facts for the health watch (boot_id, clock_s, load, imu_head, usb, dmesg, producer_log,
    producer, k0s_ok)"""

    def __init__(self, machine: Machine, capture, detector, to_obs, sampler, trial=None, clock=time.time,
                 sample_every_s=SAMPLE_S):
        self.m, self.capture, self.detector, self.to_obs = machine, capture, detector, to_obs
        self.sampler, self.trial, self.clock = sampler, trial, clock
        self.sample_every_s = sample_every_s
        self.watch = None
        self._pending = None
        self._last_t0 = None            # the left stamp of the last pair sent to detection (each pair once)
        self._last_sample = None
        self._trial_started = False
        self.stop_flag = threading.Event()

    def preflight_done(self, res):
        from calibtool import health, preflight as pf
        st = self.m.state
        st["checks"] = [c.as_dict() for c in res.checks]
        st["preflight_green"] = pf.green(res.checks)
        st["motion_gate_green"] = pf.green(res.checks, gate="motion")
        st["snapshot_keys"] = sorted(res.snapshot)
        self.watch = health.Watch(res.baseline)
        failed = [c.name for c in res.checks if c.status == "fail" and c.gate == "collect"]
        self.m.store.event("preflight", t=self.clock(), green=st["preflight_green"], failed=failed,
                           checks=st["checks"])
        if st["preflight_green"]:
            self.m.to("READY", "preflight green")
        else:
            self.m.to("BLOCKED", "preflight failed: " + ", ".join(failed))
        st["banner"] = self.banner()
        self.m.save()

    def banner(self) -> list:
        st = self.m.state
        out = []
        if st.get("read_only"):
            out.append("READ-ONLY session: the tool changes nothing on the robot (producer, motors, OKVIS as found)")
        if st["phase"] == "HOLD":
            out.append(f"HOLD: no TUI heartbeat; nothing resumes by itself: press G (from {st.get('held_from')})")
        if st["phase"] == "BLOCKED":
            out.append("preflight failed: " + ", ".join(c["name"] for c in st["checks"]
                                                        if c["status"] == "fail" and c["gate"] == "collect"))
        if self.watch is not None:
            for e in self.watch.events[-HEALTH_KEEP:]:
                if e.level in ("stop", "suspect"):
                    out.append(f"{e.level.upper()}: {e.message}")
        seen, uniq = set(), []
        for b in out:
            if b not in seen:
                seen.add(b)
                uniq.append(b)
        return uniq

    def _detect(self):
        """-> a fresh BoardObs (or None: no board) when a detection finished this step, else the sentinel False"""
        fresh = False
        if self._pending is not None and self._pending.done():
            f, self._pending = self._pending, None
            try:
                dl, _ = f.result()
                obs = self.to_obs(dl)
            except Exception as e:  # noqa: BLE001
                self.m.store.event("detect_error", t=self.clock(), error=f"{type(e).__name__}: {e}")
                obs = None
            self.m.state["board"] = None if obs is None else {
                "z_m": round(float(obs.z), 3), "u": round(float(obs.centroid[0]), 1),
                "v": round(float(obs.centroid[1]), 1), "tags": int(obs.n_tags), "corners": int(obs.n_corners),
                "cut": list(obs.cut), "t": self.clock()}
            fresh = obs if obs is not None else None
            self.m.state["detections"] = int(self.m.state.get("detections") or 0) + 1
        if self._pending is None:
            p = self.capture.latest()
            if p is not None and p.t0 != self._last_t0:
                self._last_t0 = p.t0
                self._pending = self.detector(p)
        if fresh is False and self._pending is not None and self._pending.done():
            return self._detect()
        return fresh

    def _health(self, t):
        if self.watch is None:
            return
        if self._last_sample is not None and t - self._last_sample < self.sample_every_s:
            return
        self._last_sample = t
        snap = self.m.state.get("capture") or {}
        s = dict(self.sampler() or {})
        s["cams"] = snap.get("cams") or {}
        s["sync_median_ms"] = snap.get("sync_median_ms")
        evs = self.watch.observe(t, s)
        for e in evs:
            self.m.store.event("health", **{k: v for k, v in e.as_dict().items() if k != "t"}, t=e.t)
        if evs:
            self.m.state["health"] = [e.as_dict() for e in self.watch.events[-HEALTH_KEEP:]]
        self.m.state["system"] = {k: v for k, v in s.items() if k not in ("cams", "dmesg", "producer_log")}

    def _say(self, s):
        d = s.as_dict()
        self.m.state["say"] = d
        self.m.store.event("say", t=s.t, say=d)

    def step(self):
        t = self.clock()
        self.m.tick()
        st = self.m.state
        st["capture"] = self.capture.snapshot()
        fresh = self._detect()
        self._health(t)
        if self.trial is not None and self.m.phase == "TRIAL":
            if not self._trial_started:
                self._trial_started = True
                self._say(self.trial.start(t))
            for k in self.m.take_keys():
                for s in self.trial.key(t, k):
                    self._say(s)
            if fresh is not False:
                for s in self.trial.update(t, fresh):
                    self._say(s)
            st["trial"] = self.trial.state()
            if self.trial.finished:
                self.m.store.event("trial_summary", t=t, summary=self.trial.summary())
                self.m.to("DONE", "placement trial finished")
        elif self.trial is not None:
            st["trial"] = self.trial.state() if self._trial_started else st.get("trial")
        st["banner"] = self.banner()
        self.m.save()

    def loop(self, period_s=STEP_S):
        while not self.stop_flag.is_set():
            t0 = time.monotonic()
            self.step()
            self.stop_flag.wait(max(0.0, period_s - (time.monotonic() - t0)))


# ---------------------------------------------------------------- system samples on the robot

class SystemSampler:
    """refreshes the health watch's system facts in its own thread (kubectl calls take ~1 s and must not stall the
    loop): every second the cheap ones (boot id, clock, load, IMU ring head, USB speeds), every slow_s the kubectl and
    dmesg ones"""

    def __init__(self, kube, shm="/dev/shm", sysfs="/sys/bus/usb/devices", slow_s=10.0, run=None):
        self.kube, self.shm, self.sysfs, self.slow_s = kube, Path(shm), sysfs, slow_s
        self.run = run
        self._d = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._t = None
        self._last_slow = 0.0

    def __call__(self):
        with self._lock:
            return dict(self._d)

    def refresh(self):
        from calibtool import motors, preflight as pf
        d = {"clock_s": time.time(), "load": os.getloadavg()[0]}
        try:
            d["boot_id"] = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        except OSError:
            pass
        try:
            d["imu_head"] = motors.ring_head(self.shm / "raw_imu_actuals")
        except OSError:
            d["imu_head"] = None
        try:
            d["usb"] = {s: v[0] for s, v in pf.usb_speeds(self.sysfs).items()}
        except OSError:
            pass
        if time.time() - self._last_slow >= self.slow_s:
            self._last_slow = time.time()
            try:
                pods = self.kube.pods()
                y = self.kube.get_yaml()
                from calibtool import producer as pr
                spec = pr.Spec.from_yaml(y)
                d["producer"] = {"generation": spec.generation, "args": spec.args,
                                 "pod_uid": pods[0]["uid"] if pods else None}
                d["producer_log"] = self.kube.logs(30)
                d["k0s_ok"] = True
            except Exception:  # noqa: BLE001
                d["k0s_ok"] = False
            if self.run is not None:
                rc, out = self.run(["dmesg", "-T"], 10)
                if rc == 0:
                    d["dmesg"] = out.splitlines()[-80:]
        with self._lock:
            keep = {k: v for k, v in self._d.items() if k in ("producer", "producer_log", "k0s_ok", "dmesg")}
            keep.update(d)
            self._d = keep

    def start(self):
        def loop():
            while not self._stop.is_set():
                try:
                    self.refresh()
                except Exception:  # noqa: BLE001
                    pass
                self._stop.wait(1.0)
        self._t = threading.Thread(target=loop, name="sampler", daemon=True)
        self._t.start()

    def stop(self):
        self._stop.set()


# ---------------------------------------------------------------- CLI

def _session_paths(root, sid):
    from calibtool import storage
    d = storage.SessionDirs(root, sid)
    return d, d.root / "calibd.sock"


def _current(root):
    try:
        return (Path(root) / "current").read_text().strip()
    except FileNotFoundError:
        return None


def cmd_run(a):
    """the daemon in the foreground (start runs this detached)"""
    from calibtool import capture as cp, detect, place, preflight as pf, producer as pr, profile, storage
    prof = profile.load(a.profile)
    if prof.pydeps and prof.pydeps not in sys.path:
        sys.path.insert(0, prof.pydeps)
    root = Path(a.root)
    lock = RobotLock(root / "calibd.lock")
    if not lock.acquire(a.session):
        print(f"another calibd holds {root / 'calibd.lock'}: {lock.holder()}", file=sys.stderr)
        return 3
    dirs, sock = _session_paths(root, a.session)
    dirs.create()
    store = Store(dirs.root)
    m = (Machine.resume(store) if store.load_state() is not None else
         Machine(store, session=a.session, robot=prof.bot, read_only=True, mode="trial" if a.trial else "watch"))
    (root / "current").write_text(a.session)
    (dirs.root / "pid").write_text(str(os.getpid()))
    stop = threading.Event()
    srv = Server(sock, m, on_stop=stop.set)
    srv.start()
    import dma_ipc
    import rigcfg
    cams = cp.DmaCameras(dma_ipc, prof.left, prof.right)
    cap = cp.Capture(cams, prof.left, prof.right)
    cap.start()
    kube = pr.Kube(pr.kubectl_runner(), prof.producer_ns, prof.producer_deploy, read_only=True)
    probes = pf.RobotProbes(prof, kube, cap, lambda: (True, f"{root / 'calibd.lock'} held by session {a.session}"))
    time.sleep(1.0)                     # some frames for the fps / sync counters
    res = pf.run(probes, prof)
    (dirs.root / "preflight").mkdir(exist_ok=True)
    for k, name in (("producer_yaml", "producer.yaml"), ("head_camera_json", "head_camera.json"),
                    ("robot_env", "robot.env")):
        if res.snapshot.get(k):
            (dirs.root / "preflight" / name).write_text(res.snapshot[k])
    (dirs.root / "preflight" / "snapshot.json").write_text(json.dumps(
        {k: v for k, v in res.snapshot.items() if k not in ("producer_yaml", "head_camera_json", "robot_env")},
        default=_json_default, indent=1))
    rig = rigcfg.load(prof.rig_path)
    board = rig.board
    cpus = os.cpu_count() or 1
    workers = a.detect_workers or detect.pool_size(cpus, os.getloadavg()[0])
    pool = detect.DetectPool(board, workers)
    m.state["detect_workers"] = workers
    W, H = (int(v) for v in rig.resolution)

    def to_obs(dl):
        return detect.board_obs(dl, board, focal=rig.nominal_focal_px, size=(W, H))
    trial = place.Trial(place.DEFAULTS, dirs.report / "trial.jsonl", focal=rig.nominal_focal_px, size=(W, H)) \
        if a.trial or m.state.get("mode") == "trial" else None
    sampler = SystemSampler(kube, run=lambda argv, timeout: pf._sh(argv, timeout))
    sampler.refresh()
    sampler.start()
    d = Daemon(m, cap, lambda p: pool.submit_pair(p.left.y, p.right.y), to_obs, sampler, trial=trial)
    d.preflight_done(res)
    t = threading.Thread(target=d.loop, name="calibd-loop", daemon=True)
    t.start()
    try:
        while not stop.is_set():
            stop.wait(0.5)
    finally:
        d.stop_flag.set()
        t.join(3.0)
        sampler.stop()
        cap.stop()
        pool.close()
        m.store.event("daemon_stop", t=time.time(), phase=m.phase)
        m.save()
        srv.stop()
        lock.release()
        try:
            (dirs.root / "pid").unlink()
        except FileNotFoundError:
            pass
    return 0


def _connect(sock, timeout=2.0):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.connect(str(sock))
    return s


def cmd_start(a):
    import subprocess
    from calibtool import storage
    a.session = a.session or storage.new_session_id(a.profile)
    dirs, sock = _session_paths(a.root, a.session)
    dirs.create()
    argv = [sys.executable, "-m", "calibtool.daemon", "run", "--root", a.root, "--session", a.session, "--profile",
            a.profile] + (["--trial"] if a.trial else []) + (["--detect-workers", str(a.detect_workers)]
                                                              if a.detect_workers else [])
    log = open(dirs.root / "calibd.log", "a")
    p = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
                         cwd=str(Path(__file__).resolve().parents[1]))
    t_end = time.time() + 30
    while time.time() < t_end:
        if p.poll() is not None:
            print(json.dumps({"session": a.session, "pid": p.pid, "error": f"calibd exited rc {p.returncode}",
                              "log": str(dirs.root / "calibd.log")}))
            return 2
        try:
            _connect(sock).close()
            print(json.dumps({"session": a.session, "pid": p.pid, "sock": str(sock)}))
            return 0
        except OSError:
            time.sleep(0.2)
    print(json.dumps({"session": a.session, "pid": p.pid, "error": "socket not answering after 30 s"}))
    return 2


def cmd_attach(a):
    """bridge stdin/stdout <-> the session socket (what the laptop runs over ssh)"""
    sid = a.session or _current(a.root)
    _, sock = _session_paths(a.root, sid)
    s = _connect(sock)
    s.settimeout(None)
    done = threading.Event()

    def up():
        try:
            for line in sys.stdin:
                s.sendall(line.encode() if line.endswith("\n") else (line + "\n").encode())
        except OSError:
            pass
        done.set()
    threading.Thread(target=up, daemon=True).start()
    f = s.makefile("rb")
    try:
        for line in f:
            sys.stdout.write(line.decode())
            sys.stdout.flush()
            if done.is_set():
                break
    except (OSError, BrokenPipeError):
        pass
    return 0


def cmd_stop(a):
    sid = a.session or _current(a.root)
    dirs, sock = _session_paths(a.root, sid)
    try:
        pid = int((dirs.root / "pid").read_text())
    except (FileNotFoundError, ValueError):
        pid = None
    try:
        s = _connect(sock)
        s.makefile("r").readline()
        s.sendall(b'{"cmd": "stop"}\n')
        s.close()
    except OSError as e:
        print(json.dumps({"session": sid, "stopped": False, "error": str(e)}))
        return 2
    t_end = time.time() + 15
    while pid and time.time() < t_end:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.2)
    else:
        if pid:
            print(json.dumps({"session": sid, "stopped": False, "pid": pid, "error": "still running after 15 s"}))
            return 2
    print(json.dumps({"session": sid, "stopped": True, "pid": pid}))
    return 0


def cmd_status(a):
    sid = a.session or _current(a.root)
    dirs, _ = _session_paths(a.root, sid)
    print(json.dumps(Store(dirs.root).load_state(), default=_json_default))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="calibd", description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["start", "run", "attach", "stop", "status"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--session", default=None)
    ap.add_argument("--profile", default="P10")
    ap.add_argument("--trial", action="store_true", help="placement-trial mode (neck at home, nothing moves)")
    ap.add_argument("--detect-workers", type=int, default=0)
    a = ap.parse_args(argv)
    if a.cmd == "run" and not a.session:
        ap.error("run needs --session")
    return {"start": cmd_start, "run": cmd_run, "attach": cmd_attach, "stop": cmd_stop, "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
