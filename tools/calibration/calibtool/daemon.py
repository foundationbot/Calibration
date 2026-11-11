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
[--trial] [--replace]. start spawns `run` in its own session (setsid) and returns once the session socket sends a state
line from that child (an old calibd answering is not success); it refuses while a calibd runs (this session or the
lock holder's) unless --replace stops that one first. status says live from the socket, with the running kit hash."""
import argparse
import fcntl
import hashlib
import itertools
import json
import os
import select
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

DEADMAN_S = 5.0
# the full session's phases (M5, flow.py drives them; --session-mode full). ACTIVE ones need the TUI heartbeat: the
# operator's link is the deadman wherever the robot moves, could move next (guidance -> countdown -> grid) or deploys
FLOW_PHASES = ("MOTORS", "SETUP", "PLACE", "GRID", "FIT", "IMU_PLACE", "IMU_SWEEP", "DECIDE", "DEPLOY_SIZE",
               "CHECK_PLACE", "CHECK_GRID", "DEPLOY", "POST_PLACE", "POSTCHECK", "REPORT", "FINISH")
ACTIVE = frozenset({"TRIAL", "MOTORS", "PLACE", "GRID", "IMU_PLACE", "IMU_SWEEP", "CHECK_PLACE", "CHECK_GRID", "DEPLOY",
                    "POST_PLACE", "POSTCHECK"})
TERMINAL = frozenset({"DONE", "ABORTED", "STOPPED_LOUD"})
# the keys a phase hands to the flow (anything else is dropped and logged): Y enables the motors only in MOTORS (and
# retries a deploy the operator was asked about), 1/2/? choose the deploy size, C = "can't put it there", R = roll back
# on purpose, D = disable the motors (twice: the warning first), K = keep them on
PHASE_KEYS = {"MOTORS": "YR", "PLACE": "CUF", "IMU_PLACE": "C", "CHECK_PLACE": "C", "POST_PLACE": "C",
              "DEPLOY_SIZE": "12?", "DEPLOY": "YR", "POSTCHECK": "R", "REPORT": "R", "FINISH": "DKREX", "DONE": "D",
              "ABORTED": "D", "STOPPED_LOUD": "D"}
# what an accepted key does, said back to the operator (state["last_key"]; 10 Nov: "i press Y and nothing is shown")
KEY_MEANING = {("MOTORS", "Y"): "enabling the motors (or taking them over when already on)",
               ("MOTORS", "R"): "restarting the motor master (only with the motors off; up to 90 s)",
               ("PLACE", "C"): "skipping this placement",
               ("PLACE", "U"): "using the board where it is (it must be seen now)",
               ("PLACE", "F"): "finishing the collection (extra placements only): on to the IMU sweep and the fit", ("IMU_PLACE", "C"): "skipping the IMU sweep",
               ("CHECK_PLACE", "C"): "skipping the check grid", ("POST_PLACE", "C"): "skipping the depth check",
               ("DEPLOY_SIZE", "1"): "deploy size 1 chosen", ("DEPLOY_SIZE", "2"): "deploy size 2 chosen",
               ("DEPLOY_SIZE", "?"): "this robot's facts on the screen", ("DEPLOY", "Y"): "deploying",
               ("DEPLOY", "R"): "rolling back", ("POSTCHECK", "R"): "rolling back", ("REPORT", "R"): "rolling back",
               ("FINISH", "R"): "rolling back", ("FINISH", "K"): "the motors stay on",
               ("FINISH", "D"): "disable the motors (the warning first; D again disables)",
               ("FINISH", "E"): "more placements, then fit and decide again",
               ("FINISH", "X"): "deciding again with the regions judged (one-off)", ("DONE", "D"): "disable the motors",
               ("ABORTED", "D"): "disable the motors", ("STOPPED_LOUD", "D"): "disable the motors"}


def key_meaning(phase, k) -> str:
    return KEY_MEANING.get((phase, k), f"{k} accepted in {phase}")


STEP_S = 0.15            # 9 Nov trial: guidance felt laggy at 0.5 s, then 0.25 s; guide_corners on a 1280x800 frame
                         # takes ~30 ms single-threaded (Moonstone; phantom10 measured in the report)
SAYS_KEEP = 8            # the recent prompts kept in the state: a step can make two (an end and the next intro)
SAMPLE_S = 1.0
HEALTH_KEEP = 30
POOL_REBUILD_S = 5.0     # a detection pool that keeps breaking is rebuilt at most this often
STALE_S = 1.0            # no new left/right pair for this long = the camera stalled (frozen or dropped)
STALL_TEXT = "The camera stopped. Please wait."


# ---------------------------------------------------------------- the kit this process runs

KIT_DIRS = ("calibtool", "scripts", "config")      # what `calib` copies to the robot (cli.bootstrap)
KIT_SKIP = ("__pycache__", "tests", "results")     # ... and what it leaves out (with *.npz)
_KIT = None


def kit_hash(base=None) -> str:
    """a hash of the kit's code as bootstrap copies it (tools/calibration/{calibtool,scripts,config} without tests,
    results, caches, *.npz): the laptop compares it with the running calibd's, so attaching to a daemon still running
    older code is refused (review 6)"""
    base = Path(base) if base is not None else Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for d in KIT_DIRS:
        for f in sorted((base / d).rglob("*")):
            rel = f.relative_to(base)
            if not f.is_file() or f.suffix == ".npz" or any(x in KIT_SKIP or x.startswith(".") for x in rel.parts):
                continue
            h.update(rel.as_posix().encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()[:16]


def _kit() -> str:
    """this process's kit, hashed once (at start: the code it loaded)"""
    global _KIT
    if _KIT is None:
        _KIT = kit_hash()
    return _KIT


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
                 mode="watch", _state=None, mono=None):
        self.store, self.clock, self.deadman_s = store, clock, deadman_s
        # the deadman runs on a monotonic clock (a wall-clock step must not make a dead link live, or a live one dead);
        # an injected test clock serves both
        self.mono = mono or (time.monotonic if clock is time.time else clock)
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
                      "trial": None, "say": None, "preflight_green": None, "pid": os.getpid(), "kit": _kit()}
        store.event("session_start", t=t, session=session, robot=robot, read_only=read_only, mode=mode)
        self.save()

    @classmethod
    def resume(cls, store: Store, clock=time.time, deadman_s=DEADMAN_S, mono=None):
        """the machine as state.json left it: an active phase comes back as HOLD (and stays there through the new
        preflight: only G leaves it), an unfinished preflight reruns"""
        st = store.load_state()
        if st is None:
            raise FileNotFoundError(f"no state.json in {store.dir}")
        m = cls(store, clock, deadman_s=deadman_s, _state=st, mono=mono)
        before = st["phase"]
        if before in ACTIVE or (st.get("mode") == "full" and before in FLOW_PHASES):
            # a full session comes back in HOLD from every flow phase: the robot may be half-way through a change
            st["held_from"], st["phase"] = before, "HOLD"
        elif before in ("PREFLIGHT", "READY", "BLOCKED"):
            st["phase"] = "PREFLIGHT"
        st["attached"] = False
        st["pid"], st["kit"] = os.getpid(), _kit()
        # the prompts of the dead run are history: a TUI must not take them for new ones, and the new run numbers its
        # prompts after them (review 1: the TUI skips ids it has spoken, and a new run counting from 1 went unheard)
        st["say_id_max"] = max([int(st.get("say_id_max") or 0)] + [int(x["id"]) for x in st.get("says") or ()
                                                                    if x and x.get("id") is not None])
        st["say"], st["says"], st["board"] = None, [], None
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

    def stopped(self):
        """a clean stop: no pid left in state.json (review 6: a stale pid read as a running daemon)"""
        with self._lock:
            self.store.event("daemon_stop", t=self.clock(), phase=self.phase)
            self.state["pid"], self.state["attached"] = None, False
            self.save()

    def snapshot(self) -> dict:
        with self._lock:
            return json.loads(json.dumps(self.state, default=_json_default))

    def to(self, phase, why=""):
        with self._lock:
            if self.phase in TERMINAL or phase == self.phase:
                return
            self.store.event("phase", t=self.clock(), frm=self.phase, to=phase, why=why)
            if self.pending_keys and (self.state.get("mode") == "full" or
                                      (self.phase in ACTIVE and phase not in ACTIVE)):
                for k in self.pending_keys:     # a key belongs to the phase it was pressed in (not kept over a HOLD)
                    self.store.event("key_dropped", t=self.clock(), key=k, phase=phase, why=f"left {self.phase}")
                self.pending_keys = []
            self.state["phase"] = phase
            if phase == "ABORTED":
                self.state["status"] = "aborted"
            elif phase == "DONE":
                self.state["status"] = "completed"
            self.save()

    def live(self) -> bool:
        return self.last_hb is not None and self.mono() - self.last_hb <= self.deadman_s

    def heartbeat(self):
        with self._lock:
            self.last_hb = self.mono()
            self.state["last_hb_t"] = self.clock()      # saved with the next step: the stale-lease restore reads it
            if not self.state["attached"]:
                self.state["attached"] = True
                self.store.event("tui_heartbeat", t=self.clock())

    def tick(self):
        with self._lock:
            live = self.live()
            changed = False
            if self.state["attached"] and not live:
                self.state["attached"] = False
                self.store.event("link_lost", t=self.clock(), hb_age_s=round(self.mono() - self.last_hb, 2)
                                 if self.last_hb is not None else None)
                changed = True
            if self.phase in ACTIVE and not live:
                self.state["held_from"] = self.phase
                self.store.event("heartbeat_lost", t=self.clock(), phase=self.phase, deadman_s=self.deadman_s)
                self.to("HOLD", f"no TUI heartbeat for {self.deadman_s:g} s")
                changed = False
            if changed:
                self.save()

    def hold(self, why):
        """the flow's own HOLD (a foreign command, a missed pose, a refused arm): left only by G, like the deadman's"""
        with self._lock:
            if self.phase in TERMINAL or self.phase == "HOLD":
                return
            self.state["held_from"], self.state["hold_reason"] = self.phase, str(why)
            self.store.event("hold", t=self.clock(), phase=self.phase, why=str(why))
            self.to("HOLD", str(why))

    def key(self, k):
        """one operator key; every key gets an answer in state["last_key"] (10 Nov, phantom11: "i press Y and nothing
        is shown"): accepted or not, and what it does / why not"""
        k = str(k).upper()
        with self._lock:
            phase = self.phase
            accepted, why = self._key(k)
            self.state["last_key"] = {"key": k, "phase": phase, "accepted": bool(accepted), "why": why,
                                      "t": self.clock()}

    def _key(self, k):
        self.store.event("key", t=self.clock(), key=k, phase=self.phase)
        full = self.state.get("mode") == "full"
        if self.phase in TERMINAL:
            if full and k in PHASE_KEYS.get(self.phase, ""):
                self.pending_keys.append(k)
                return True, key_meaning(self.phase, k)
            return False, f"the session has ended ({self.phase}): {k} does nothing"
        if k == "A":
            self.to("ABORTED", "abort key")
            return True, "aborting: what moves finishes its pose, then the robot goes back as it was"
        if k == "S" and self.phase == "BLOCKED":
            self.state["start_master"] = True
            return True, "starting the motor master (if it is not running); the checks run again in a few seconds"
        if k == "G":
            if self.phase != "HOLD":
                return False, f"G only continues after a HOLD (now {self.phase})"
            if self.state.get("preflight_green") is False:
                self.store.event("g_refused", t=self.clock(), why="the preflight after the restart failed")
                return False, "the preflight after the restart failed: see the checks"
            if not (self.live() and self.state.get("held_from")):
                return False, "G needs this screen attached (no heartbeat yet)"
            back = self.state["held_from"]
            self.store.event("resumed_by_operator", t=self.clock(), to=back)
            self.state["held_from"], self.state["hold_reason"] = None, None
            self.to(back, "G with the TUI attached")
            return True, f"continuing ({back}); any motion starts after a spoken countdown"
        if k == "T" and not full:
            if self.phase == "READY":
                self.to("TRIAL", "T: placement trial")
                return True, "placement trial started"
            return False, f"T starts the trial only from READY (now {self.phase})"
        if k == "M" and full:
            if self.phase != "READY":
                return False, f"M opens the motor screen only from READY (now {self.phase})"
            if self.state.get("motion_gate_green") is True:
                self.to("MOTORS", "M: motor screen")
                return True, "motor screen: check it, then Y"
            failed = [c["name"] for c in self.state.get("checks") or ()
                      if c.get("gate") == "motion" and c.get("status") == "fail"]
            self.store.event("motors_refused", t=self.clock(), why="the preflight motion gate is not green",
                             failed=failed)
            return False, "the motion checks are not green: " + (", ".join(failed) or "see the checks")
        if k == "O" and full:
            self.state["okvis_check"] = False
            self.store.event("okvis_check_skipped", t=self.clock(), phase=self.phase)
            self.save()
            return True, "the OKVIS check after the deploy will be skipped"
        if full:
            if k in PHASE_KEYS.get(self.phase, ""):
                self.pending_keys.append(k)
                busy = self.state.get("busy")
                return True, key_meaning(self.phase, k) + (f" (after: {busy['what']})" if busy else "")
            why = f"{k} does nothing in {self.phase}"
            busy = self.state.get("busy")
            if busy:
                why += f": the tool is busy ({busy['what']}), no key needed"
            self.store.event("key_dropped", t=self.clock(), key=k, phase=self.phase, why=why)
            return False, why
        if self.phase in ACTIVE:
            self.pending_keys.append(k)
            return True, f"{k} passed to the trial"
        # review 4: two C pressed in HOLD were replayed after G and skipped two placements
        self.store.event("key_dropped", t=self.clock(), key=k, phase=self.phase,
                         why="trial keys count only during the trial")
        return False, "trial keys count only during the trial"

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
        # one controlling TUI: the newest client to send a heartbeat or key; the others' heartbeats and keys do not
        # count (review 7: a stale second TUI kept the deadman satisfied), except A (abort is live from any TUI)
        self._ids = itertools.count(1)
        self._claimed = set()
        self.controller = None
        self._ctl = threading.Lock()

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
        cid = next(self._ids)
        self.m.store.event("client_attached", t=self.m.clock(), client=cid)
        buf = b""
        try:
            while not self._stop.is_set():
                snap = self.m.snapshot()
                snap["client"] = {"id": cid, "claimed": cid in self._claimed, "controlling": self.controller == cid}
                c.sendall((json.dumps(snap, default=_json_default) + "\n").encode())
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
                        self._handle(line, cid)
        except (OSError, ValueError):
            return
        finally:
            with self._ctl:
                if self.controller == cid:
                    self.controller = None
            self.m.store.event("client_detached", t=self.m.clock(), client=cid)
            try:
                c.close()
            except OSError:
                pass

    def _claim(self, cid) -> bool:
        """a client's heartbeat or key: its first one makes it the controller (the newest TUI wins), and one sent
        while nobody controls takes control back -> whether cid controls now"""
        with self._ctl:
            if cid not in self._claimed or self.controller is None:
                self._claimed.add(cid)
                if self.controller != cid:
                    self.controller = cid
                    self.m.store.event("controller", t=self.m.clock(), client=cid)
            return self.controller == cid

    def _handle(self, line, cid=0):
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            return
        if "hb" in msg or "key" in msg:
            ctl = self._claim(cid)
            if "hb" in msg and ctl:
                self.m.heartbeat()
            if "key" in msg:
                k = str(msg["key"]).upper()
                if ctl:
                    self.m.heartbeat()
                    self.m.key(k)
                elif k == "A":
                    self.m.key(k)
                else:
                    self.m.store.event("key_ignored", t=self.m.clock(), key=k, client=cid,
                                       why=f"client {self.controller} has control")
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

GUIDE_PHASES = frozenset({"READY", "PLACE", "IMU_PLACE", "CHECK_PLACE", "POST_PLACE"})


class ObsBox:
    """the newest live detection for the full session's guidance (flow.Robot.board_obs): put by the loop, taken by
    the flow thread, each one once"""

    def __init__(self):
        self._lock = threading.Lock()
        self._v = (False, None)
        self._miss = None

    def put(self, obs, miss=None):
        """obs: detect.BoardObs or None; miss: why it is None (detect.Miss), if known"""
        with self._lock:
            self._v = (True, obs)
            self._miss = None if obs is not None else miss

    def last_miss(self):
        """why the newest detection found no board (detect.Miss), or None"""
        with self._lock:
            return self._miss

    def take(self):
        with self._lock:
            v, self._v = self._v, (False, None)
            return v


class Daemon:
    """detector(pair) -> Future of (left corners, right corners) (the guidance detection: (left corners, {"brightness"}));
    to_obs(left corners) -> detect.BoardObs | None; to_miss(left corners, the second item) -> detect.Miss: why no
    board was seen, said by the guide;
    sampler() -> system facts for the health watch (boot_id, clock_s, load, imu_head, usb, dmesg, producer_log,
    producer, k0s_ok)"""

    def __init__(self, machine: Machine, capture, detector, to_obs, sampler, trial=None, clock=time.time,
                 sample_every_s=SAMPLE_S, rebuild_detector=None, flow=None, obs_box=None, to_miss=None):
        self.m, self.capture, self.detector, self.to_obs = machine, capture, detector, to_obs
        self.to_miss = to_miss
        self.miss = None                # why the newest detection found no board (detect.Miss), or None
        self.flow, self.obs_box = flow, obs_box     # the full session (flow.FullFlow runs in its own thread)
        self.rebuild_detector = rebuild_detector    # () -> a new detector when the process pool broke
        self._rebuilt_at = None
        self._last_error = None
        self._thread = None
        self.sampler, self.trial, self.clock = sampler, trial, clock
        self.sample_every_s = sample_every_s
        self.watch = None
        self._pending = None
        self._last_t0 = None            # the left stamp of the last pair sent to detection (each pair once)
        self._last_sample = None
        self._trial_started = False
        self._paused_at = None          # when the running trial was last interrupted (HOLD, a stalled camera)
        self._stall_since = None        # when the newest pair got older than STALE_S
        self._stall_said = False
        self.stop_flag = threading.Event()
        # prompt ids continue after the ones a previous run of this session spoke (Machine.resume keeps the max)
        self.ids = itertools.count(int(machine.state.get("say_id_max") or 0) + 1)
        if trial is not None:
            trial.use_ids(self.ids)

    def preflight_done(self, res):
        from calibtool import health, preflight as pf
        st = self.m.state
        st["checks"] = [c.as_dict() for c in res.checks]
        st["preflight_green"] = pf.green(res.checks)
        st["motion_gate_green"] = pf.green(res.checks, gate="motion")
        st["snapshot_keys"] = sorted(res.snapshot)
        if self.trial is not None:
            st["trial"] = self.trial.state(self.clock())
        self.watch = health.Watch(res.baseline)
        failed = [c.name for c in res.checks if c.status == "fail" and c.gate == "collect"]
        self.m.store.event("preflight", t=self.clock(), green=st["preflight_green"], failed=failed,
                           checks=st["checks"])
        if self.m.phase == "HOLD":
            # a restarted calibd resumed in HOLD: only G leaves it (and not while this preflight failed) (review 7)
            self.m.store.event("preflight_in_hold", t=self.clock(), green=st["preflight_green"])
        elif st["preflight_green"]:
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
        out += list(st.get("flow_banner") or ())
        if st["phase"] == "HOLD":
            why = st.get("hold_reason") or "no TUI heartbeat"
            out.append(f"HOLD: {why}; nothing resumes by itself: press G (from {st.get('held_from')})")
        if st.get("step_error_run"):
            out.append(f"calibd step failing ({st['step_error_run']} in a row): {st.get('last_step_error')}")
        if st.get("camera_stalled_s") is not None:
            out.append(f"CAMERA STALLED: no new left/right pair for {st['camera_stalled_s']:.0f} s (frozen or dropped); "
                       "the trial waits")
        if st["phase"] == "BLOCKED" or (st["phase"] == "HOLD" and st.get("preflight_green") is False):
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
            self.miss = None
            try:
                dl, extra = f.result()
                obs = self.to_obs(dl)
                if obs is None and self.to_miss is not None:
                    self.miss = self.to_miss(dl, extra)
            except Exception as e:  # noqa: BLE001
                self.m.store.event("detect_error", t=self.clock(), error=f"{type(e).__name__}: {e}")
                obs = None
                if _broken_pool(e):
                    self._rebuild(e)
            self.m.state["board_miss"] = None if self.miss is None else self.miss.as_dict()
            self.m.state["board"] = None if obs is None else {
                "z_m": round(float(obs.z), 3), "u": round(float(obs.centroid[0]), 1),
                "v": round(float(obs.centroid[1]), 1), "tags": int(obs.n_tags), "corners": int(obs.n_corners),
                "cut": list(obs.cut), "t": self.clock()}
            fresh = obs if obs is not None else None
            self.m.state["detections"] = int(self.m.state.get("detections") or 0) + 1
        if self._pending is None:
            p = self.capture.latest()
            if p is not None and p.t0 != self._last_t0:
                try:
                    self._pending = self.detector(p)
                    self._last_t0 = p.t0
                except Exception as e:  # noqa: BLE001
                    if not _broken_pool(e):
                        raise
                    self._pending = None
                    self._rebuild(e)
        if fresh is False and self._pending is not None and self._pending.done():
            return self._detect()
        return fresh

    def _rebuild(self, e):
        """a broken detection process pool (a worker killed, e.g. by the OOM killer) is replaced, at most every
        POOL_REBUILD_S; detection pauses meanwhile (review 3: BrokenProcessPool from submit ended the loop)"""
        t = time.monotonic()
        if self.rebuild_detector is None or (self._rebuilt_at is not None and t - self._rebuilt_at < POOL_REBUILD_S):
            return
        self._rebuilt_at = t
        self.m.store.event("detect_pool_broken", t=self.clock(), error=f"{type(e).__name__}: {e}")
        try:
            self.detector = self.rebuild_detector()
        except Exception as e2:  # noqa: BLE001
            self.m.store.event("detect_pool_rebuild_failed", t=self.clock(), error=f"{type(e2).__name__}: {e2}")
            return
        self.m.state["pool_rebuilds"] = int(self.m.state.get("pool_rebuilds") or 0) + 1
        self.m.store.event("detect_pool_rebuilt", t=self.clock(), rebuilds=self.m.state["pool_rebuilds"])

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
        own = self.m.state.get("producer_own_t")
        s["own_until"] = None if own is None else float(own) + health.OWN_WINDOW_S
        evs = self.watch.observe(t, s)
        for e in evs:
            self.m.store.event("health", **{k: v for k, v in e.as_dict().items() if k != "t"}, t=e.t)
        if evs:
            self.m.state["health"] = [e.as_dict() for e in self.watch.events[-HEALTH_KEEP:]]
        self.m.state["system"] = {k: v for k, v in s.items() if k not in ("cams", "dmesg", "producer_log")}

    def say(self, s):
        """a prompt from another thread (the flow): numbered from self.ids, saved under the machine's lock"""
        with self.m._lock:
            self._say(s)
            self.m.save()

    def _say(self, s):
        d = s.as_dict()
        self.m.state["say"] = d
        self.m.state["say_id_max"] = max(int(self.m.state.get("say_id_max") or 0), int(d["id"]))
        self.m.state["says"] = (list(self.m.state.get("says") or []) + [d])[-SAYS_KEEP:]
        self.m.store.event("say", t=s.t, say=d)

    def _stall(self, t) -> bool:
        """the camera check of every step: True while the newest pair is older than STALE_S (review 2: a frozen camera
        kept the board 'seen' and the trial waited silently)"""
        st = self.m.state
        age = self.capture.latest_age()
        st["pair_age_s"] = None if age is None else round(age, 2)
        stalled = age is None or age > STALE_S
        if stalled:
            if self._stall_since is None:
                self._stall_since = t
                self.m.store.event("camera_stalled", t=t, pair_age_s=st["pair_age_s"])
            st["camera_stalled_s"] = round(age if age is not None else t - self._stall_since, 1)
            st["board"] = None
        else:
            if self._stall_since is not None:
                self.m.store.event("camera_back", t=t, stalled_s=round(t - self._stall_since, 2))
            self._stall_since, self._stall_said = None, False
            st["camera_stalled_s"] = None
        if st.get("board"):
            st["board"]["age_s"] = round(t - float(st["board"].get("t") or t), 2)
        return stalled

    def step(self):
        t = self.clock()
        st = self.m.state
        st["step_n"] = int(st.get("step_n") or 0) + 1       # the TUI flags a daemon whose counter stops (review 3)
        self.m.tick()
        st["capture"] = self.capture.snapshot()
        full = st.get("mode") == "full"
        # a full session detects live only for the guidance: in a grid the flow owns the detection pool
        fresh = self._detect() if not full or self.m.phase in GUIDE_PHASES else False
        if full and fresh is not False and self.obs_box is not None:
            self.obs_box.put(fresh, self.miss)
        stalled = self._stall(t)
        self._health(t)
        if full and self.flow is not None and self.m.phase in TERMINAL:
            self.flow.handle_terminal_keys()            # D after the end: the motor disable
        if self.trial is not None and self.m.phase == "TRIAL":
            if not self._trial_started:
                self._trial_started = True
                self._say(self.trial.start(t))
                self.m.store.event("trial_gravity", t=t, gravity=self.trial.gravity_status())
            if stalled:
                if self._paused_at is None and not self.trial.finished:
                    self._paused_at = t
                if not self._stall_said:
                    from calibtool import place
                    self._stall_said = True
                    self._say(place.Say(next(self.ids), float(t), "camera", STALL_TEXT))
            elif self._paused_at is not None:
                # back from HOLD or a stalled camera: the time away does not count against the placement, and the
                # operator hears again what to place (review 4)
                self.m.store.event("trial_resumed", t=t, paused_s=round(t - self._paused_at, 2))
                for s in self.trial.resume(t, t - self._paused_at):
                    self._say(s)
                self._paused_at = None
            keys = self.m.take_keys()
            if not stalled:
                for k in keys:
                    for s in self.trial.key(t, k):
                        self._say(s)
                # a fresh detection feeds the guide; without one the placement's clock still runs (review 2)
                for s in (self.trial.update(t, fresh, self.miss) if fresh is not False else self.trial.tick(t)):
                    self._say(s)
            else:
                for k in keys:
                    self.m.store.event("key_dropped", t=t, key=k, phase="TRIAL", why="camera stalled")
            st["trial"] = self.trial.state(t)
            if self.trial.finished:
                self.m.store.event("trial_summary", t=t, summary=self.trial.summary())
                self.m.to("DONE", "placement trial finished")
        elif self.trial is not None:
            if self._trial_started and not self.trial.finished and self._paused_at is None:
                self._paused_at = t
            # always the trial's own state: a stored line from before a restart is stale (9 Nov: "placement 2 of 6")
            st["trial"] = self.trial.state(t)
        st["step_error_run"] = 0
        st["banner"] = self.banner()
        self.m.save()

    def safe_step(self):
        """one step; an exception is logged (once per distinct error, then counted), shown in the banner and the
        loop goes on (review 3: one exception ended the loop thread and nothing said so)"""
        try:
            self.step()
        except Exception as e:  # noqa: BLE001
            import traceback
            st = self.m.state
            err = f"{type(e).__name__}: {e}"
            st["step_errors"] = int(st.get("step_errors") or 0) + 1
            st["step_error_run"] = int(st.get("step_error_run") or 0) + 1
            st["last_step_error"] = err
            if err != self._last_error or st["step_errors"] % 100 == 0:
                self._last_error = err
                try:
                    self.m.store.event("step_error", t=time.time(), error=err, count=st["step_errors"],
                                       trace=traceback.format_exc()[-4000:])
                except Exception:  # noqa: BLE001
                    pass
            try:
                st["banner"] = self.banner()
            except Exception:  # noqa: BLE001
                st["banner"] = [f"calibd step failing: {err}"]
            try:
                self.m.save()
            except Exception:  # noqa: BLE001
                pass

    def loop(self, period_s=STEP_S):
        while not self.stop_flag.is_set():
            t0 = time.monotonic()
            self.safe_step()
            self.stop_flag.wait(max(0.0, period_s - (time.monotonic() - t0)))

    def start_loop(self, period_s=STEP_S):
        self._period = period_s
        self._thread = threading.Thread(target=self.loop, args=(period_s,), name="calibd-loop", daemon=True)
        self._thread.start()

    def loop_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def watch_loop(self) -> bool:
        """the main thread's check: a loop thread that died anyway (an error safe_step does not catch) is logged and
        started again -> True when it was restarted"""
        if self._thread is None or self.stop_flag.is_set() or self._thread.is_alive():
            return False
        st = self.m.state
        st["loop_restarts"] = int(st.get("loop_restarts") or 0) + 1
        self.m.store.event("loop_died", t=time.time(), restarts=st["loop_restarts"])
        self.start_loop(self._period)
        return True


def _broken_pool(e) -> bool:
    from concurrent.futures.process import BrokenProcessPool
    return isinstance(e, BrokenProcessPool)


# ---------------------------------------------------------------- system samples on the robot

class SystemSampler:
    """the health watch's system facts, refreshed by two threads so a slow kubectl call never stalls the loop or ages
    the cheap facts: every second boot id, clock, load, IMU ring head (with imu_t, when it was read), USB speeds; every
    slow_s the producer (kubectl) and dmesg"""

    def __init__(self, kube, shm="/dev/shm", sysfs="/sys/bus/usb/devices", slow_s=10.0, run=None):
        self.kube, self.shm, self.sysfs, self.slow_s = kube, Path(shm), sysfs, slow_s
        self.run = run
        self._d = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()

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
            d["imu_t"] = time.time()
        except OSError:
            d["imu_head"] = None
        try:
            d["usb"] = {s: v[0] for s, v in pf.usb_speeds(self.sysfs).items()}
        except OSError:
            pass
        with self._lock:
            self._d.update(d)

    def refresh_slow(self):
        """the kubectl and dmesg facts (seconds each: their own thread)"""
        from calibtool import producer as pr
        d = {}
        try:
            pods = self.kube.pods()
            spec = pr.Spec.from_yaml(self.kube.get_yaml())
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
            self._d.update(d)

    def start(self):
        def loop(fn, every):
            while not self._stop.is_set():
                try:
                    fn()
                except Exception:  # noqa: BLE001
                    pass
                self._stop.wait(every)
        for fn, every, name in ((self.refresh, 1.0, "sampler-fast"), (self.refresh_slow, self.slow_s,
                                                                         "sampler-slow")):
            threading.Thread(target=loop, args=(fn, every), name=name, daemon=True).start()

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
    from calibtool import capture as cp, detect, gravity, place, preflight as pf, producer as pr, profile, storage
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
    prev = store.load_state()
    full = getattr(a, "session_mode", "watch") == "full" or (prev or {}).get("mode") == "full"
    m = (Machine.resume(store) if prev is not None else
         Machine(store, session=a.session, robot=prof.bot, read_only=not full,
                 mode="full" if full else "trial" if a.trial else "watch"))
    dry_run = bool(getattr(a, "dry_run", False) or (prev or {}).get("dry_run"))
    m.state["dry_run"] = dry_run
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
    cpus = detect.usable_cpus()
    workers = a.detect_workers or detect.pool_size(cpus, os.getloadavg()[0])
    pools = {"pool": detect.DetectPool(board, workers)}
    m.state["detect_workers"] = workers
    W, H = (int(v) for v in rig.resolution)

    def to_obs(dl):
        return detect.board_obs(dl, board, focal=rig.nominal_focal_px, size=(W, H))

    def to_miss(dl, extra):
        b = extra.get("brightness") if isinstance(extra, dict) else None
        return detect.board_miss(dl or {}, (W, H), brightness=b)
    trial = None
    if a.trial or m.state.get("mode") == "trial":
        # 'up' in camera 0 (head IMU + the live OKVIS R_SC, read-only; None -> guidance by camera distance only)
        grav = gravity.from_robot(fallback=prof.live_config)
        grav.up()
        m.store.event("gravity", t=time.time(), **grav.status())
        trial = place.Trial(place.DEFAULTS, dirs.report / "trial.jsonl", timeout_s=prof.placement_timeout_s,
                            focal=rig.nominal_focal_px, size=(W, H), gravity=grav,
                            stand=place.Stand(prof.stand_raise_m, prof.stand_lower_m, prof.stand_auto_skip, prof.stand_steps))
    sampler = SystemSampler(kube, run=lambda argv, timeout: pf._sh(argv, timeout))
    sampler.refresh()
    sampler.refresh_slow()
    sampler.start()
    def submitter():
        pool = pools["pool"]
        # the placement trial and the full session's guidance only need where the board is (left image): the fast
        # guidance detection
        return (lambda p: pool.submit_guide(p.left.y)) if trial is not None or full else \
            (lambda p: pool.submit_pair(p.left.y, p.right.y))

    def rebuild():
        try:
            pools["pool"].close()
        except Exception:  # noqa: BLE001  (a broken pool may not close cleanly)
            pass
        pools["pool"] = detect.DetectPool(board, workers)
        return submitter()
    d = Daemon(m, cap, submitter(), to_obs, sampler, trial=trial, rebuild_detector=rebuild,
               obs_box=ObsBox() if full else None, to_miss=to_miss)
    d.preflight_done(res)
    if full:
        # 10 Nov, phantom10: building the session raised (a file missing from the kit) and calibd died at start with
        # the motors on; the operator saw only "calibd not stepping". Now calibd stays up and says why
        try:
            d.flow = _full_flow(prof, m, d, dirs, cap, pools, rig, (W, H), res, probes, dry_run, a.session)
        except Exception as e:  # noqa: BLE001
            flow_failed(m, "could not set up the session", e)
    d.start_loop()
    if full and d.flow is not None:
        threading.Thread(target=run_flow, args=(m, d.flow), name="calibd-flow", daemon=True).start()
    restart, last_check = False, time.time()
    try:
        while not stop.is_set():
            stop.wait(0.5)
            d.watch_loop()
            if m.phase == "BLOCKED":
                # 10 Nov, phantom10: rebooted, the motor master (not started at boot) inactive -> BLOCKED for good;
                # now S starts the master, the checks run again every BLOCKED_RECHECK_S, and once green calibd
                # starts itself again (a fresh preflight and session setup, the same session)
                if m.state.pop("start_master", None):
                    start_master(m)
                    last_check = 0.0
                if time.time() - last_check >= BLOCKED_RECHECK_S:
                    last_check = time.time()
                    if blocked_recheck(d, m, probes, prof):
                        restart = True
                        stop.set()
    finally:
        d.stop_flag.set()
        if d._thread is not None:
            d._thread.join(3.0)
        sampler.stop()
        cap.stop()
        try:
            pools["pool"].close()
        except Exception:  # noqa: BLE001
            pass
        m.stopped()
        srv.stop()
        lock.release()
        try:
            (dirs.root / "pid").unlink()
        except FileNotFoundError:
            pass
    if restart:
        os.execv(sys.executable, [sys.executable, "-m", "calibtool.daemon", *sys.argv[1:]])
    return 0


BLOCKED_RECHECK_S = 10.0


def blocked_recheck(d, m, probes, prof, run_preflight=None) -> bool:
    """the preflight again while BLOCKED: the screen gets the new checks; True once it is green (the caller restarts
    calibd for a clean setup)"""
    from calibtool import preflight as pf
    try:
        res = (run_preflight or pf.run)(probes, prof)
    except Exception as e:  # noqa: BLE001  (the next round tries again)
        m.store.event("preflight_recheck_error", t=time.time(), error=f"{type(e).__name__}: {e}")
        return False
    m.state["checks"] = [c.as_dict() for c in res.checks]
    green = pf.green(res.checks)
    m.store.event("preflight_recheck", t=time.time(), green=green,
                  failed=[c.name for c in res.checks if c.status == "fail"])
    if green:
        m.state["banner"] = ["preflight green now: calibd starts again for a clean setup (a few seconds)"]
    else:
        m.state["banner"] = d.banner()
    m.save()
    return green


def start_master(m, run=subprocess.run):
    """S while BLOCKED (approved by the operator on 10 Nov with R): systemctl start dma-ethercat when it is inactive"""
    try:
        r = run(["systemctl", "is-active", "dma-ethercat"], capture_output=True, text=True, timeout=10)
        if r.stdout.strip() == "active":
            m.store.event("start_master", t=time.time(), done=False, why="already active")
            return
        r = run(["systemctl", "start", "dma-ethercat"], capture_output=True, text=True, timeout=90)
        m.store.event("start_master", t=time.time(), done=r.returncode == 0,
                      why=(r.stdout or "").strip()[-200:] + (r.stderr or "").strip()[-200:])
    except Exception as e:  # noqa: BLE001
        m.store.event("start_master", t=time.time(), done=False, why=f"{type(e).__name__}: {e}")


def _full_flow(prof, m, d, dirs, cap, pools, rig, size, res, probes, dry_run, session):
    """the full session's flow over this robot (flowrobot.build); runs in its own thread. Robot-only imports here"""
    import dma_ipc
    from calibtool import (coverage, flow, flowrobot, gravity, imusweep, livecfg, neck as nk, place, protocol,
                           refit, replay)
    grav = gravity.from_robot(fallback=prof.live_config)
    R_live, live_path = gravity.live_r_sc(fallback=prof.live_config)
    live = livecfg.load(live_path, at_size=size)
    board = rig.board

    def live_text():
        return Path(live_path).read_text()

    def submit_pair(yl, yr):
        return pools["pool"].submit_pair(yl, yr)
    check_scores, post_scores = flowrobot.scorers(prof.rig_path, live_text, dirs.fit / "scores", submit_pair)
    name = f"{prof.thor}_stereo_kb4t_{time.strftime('%Y%m%d')}_1280.yaml"
    live_fit = refit.LiveFit(prof.rig_path)          # the final fit = its refit on the final split (step 45 if not)
    # the extras planner (10 Nov: never wired, so no extra placement was ever asked for): the head from the live OKVIS
    # config at the collection size, the neck chain, the board
    from calibtool import placement as pl, planner as pn
    plan_next, why = flowrobot.safe_planner_next(
        lambda: pn.Planner(pl.head_from_okvis(live_path, size), pl.chain(), board))
    if why:
        m.store.event("planner_unavailable", t=time.time(), why=why)
        print(f"calibd: no extras planner ({why}); the session runs without extra placements", flush=True)
    robot = flowrobot.build(
        prof, session=session, dirs=dirs, capture=cap, detect_pair=submit_pair,
        producer_yaml=res.snapshot.get("producer_yaml"), probes=probes, baseline=res.baseline, obs_box=d.obs_box,
        guide_kw={"focal": rig.nominal_focal_px, "size": size, "up": grav.up,
                  "stand": place.Stand(prof.stand_raise_m, prof.stand_lower_m, prof.stand_auto_skip, prof.stand_steps)},
        make_collection=lambda: replay.Collection(rig, replay.capture_labeler(rig),
                                                  targets=coverage.Targets.for_per_hold(protocol.PER_HOLD)),
        # jobs 3: the new and the live validation in parallel (phantom11: FIT 40 s with one at a time)
        fit_decide=flowrobot.pipeline_fit_decide(prof.rig_path, lambda: live_path, name, live_fit=live_fit, jobs=3),
        live_fit=live_fit, dry_run=dry_run,
        imu=nk.DmaImu(dma_ipc, gravity.HEAD_IMU),
        sweep_detect=lambda g: imusweep.board_normal(board.detect(g), board, live.model, live.intr[0]),
        R_live=R_live, check_scores=check_scores, post_scores=post_scores, plan_next=plan_next)
    return flow.FullFlow(m, robot, dirs, emit=d.say, ids=d.ids, place_timeout_s=prof.placement_timeout_s)


def flow_failed(m, what, e):
    """an error the session cannot go on from: on the operator's screen (NOT NORMAL), in the events with the
    traceback, the robot holding (nothing moves: no flow sends anything any more)"""
    import traceback
    msg = f"calibd {what}: {type(e).__name__}: {e}"
    m.store.event("flow_error", t=time.time(), what=what, error=f"{type(e).__name__}: {e}",
                  traceback=traceback.format_exc()[-4000:])
    print(msg, flush=True)
    traceback.print_exc()
    m.state["flow_banner"] = list(m.state.get("flow_banner") or []) + [
        msg + ". Nothing moves. Start calib again and choose to continue this session (it restarts calibd), or tell "
        "the agent (calibd.log has the details)."]
    m.state["busy"] = None
    m.save()


def run_flow(m, flow_):
    """the session's thread: an unexpected error is said on the screen, never a silent stop (10 Nov)"""
    try:
        flow_.run()
    except Exception as e:  # noqa: BLE001
        flow_failed(m, "stopped the session on an error", e)


def _connect(sock, timeout=2.0):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.connect(str(sock))
    return s


def read_state_line(sock, timeout=2.0):
    """the first state line calibd sends on a session socket, or None when nothing answers: liveness is the socket
    (review 6: kill -0 on a stored pid can hit another process, and a cleanly stopped daemon left its pid behind)"""
    try:
        s = _connect(sock, timeout)
    except OSError:
        return None
    f = None
    try:
        f = s.makefile("rb")
        line = f.readline()
        return json.loads(line) if line else None
    except (OSError, ValueError):
        return None
    finally:
        for x in (f, s):
            try:
                if x is not None:
                    x.close()
            except OSError:
                pass


def wait_started(sock, child_pid, child_rc, timeout=30.0, sleep=time.sleep, clock=time.monotonic) -> dict:
    """{"ok": True} once the session socket sends a state line from child_pid (the calibd just spawned); an answer from
    another pid is an old daemon still serving this session, not success (review 6)"""
    other = None
    t_end = clock() + timeout

    def why():
        if other is None:
            return ""
        return (f"; another calibd (pid {other.get('pid')}, kit {other.get('kit')}) still runs session "
                f"{other.get('session')}: stop it first (calib ... --replace, or calibd stop)")
    while clock() < t_end:
        line = read_state_line(sock, 1.0)
        if line is not None and line.get("pid") == child_pid:
            return {"ok": True}
        if line is not None:
            other = line
        rc = child_rc()
        if rc is not None:
            return {"ok": False, "error": f"calibd exited rc {rc}" + why()}
        sleep(0.2)
    return {"ok": False, "error": f"no state line from the new calibd (pid {child_pid}) after {timeout:.0f} s" + why()}


def _live_sessions(root, sid):
    """[(session, its state line)] of the calibd answering for sid and of the one holding the robot lock"""
    out, seen = [], set()
    try:
        holder = json.loads(RobotLock(Path(root) / "calibd.lock").holder() or "{}").get("session")
    except ValueError:
        holder = None
    for s in (sid, holder):
        if not s or s in seen:
            continue
        seen.add(s)
        try:
            line = read_state_line(_session_paths(root, s)[1])
        except ValueError:                  # not a session id
            continue
        if line is not None:
            out.append((s, line))
    return out


def run_argv(a) -> list:
    """the detached `run` child's argv for start"""
    full = getattr(a, "session_mode", "watch") == "full"
    return ([sys.executable, "-m", "calibtool.daemon", "run", "--root", a.root, "--session", a.session, "--profile",
             a.profile] + (["--trial"] if a.trial else []) + (["--detect-workers", str(a.detect_workers)]
                                                               if a.detect_workers else [])
            + (["--session-mode", "full"] if full else []) + (["--dry-run"] if getattr(a, "dry_run", False) else []))


def cmd_start(a):
    import subprocess
    from calibtool import storage
    a.session = a.session or storage.new_session_id(a.profile)
    dirs, sock = _session_paths(a.root, a.session)
    for s, line in _live_sessions(a.root, a.session):
        if not getattr(a, "replace", False):
            print(json.dumps({"session": a.session, "error": f"calibd already runs session {s} (pid {line.get('pid')}, "
                              f"kit {line.get('kit')}); attach to it, or --replace stops it first"}))
            return 2
        if cmd_stop(argparse.Namespace(root=a.root, session=s)) != 0:
            print(json.dumps({"session": a.session, "error": f"could not stop the calibd of session {s}"}))
            return 2
    dirs.create()
    argv = run_argv(a)
    log = open(dirs.root / "calibd.log", "a")
    p = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
                         cwd=str(Path(__file__).resolve().parents[1]))
    r = wait_started(sock, p.pid, p.poll)
    if r["ok"]:
        print(json.dumps({"session": a.session, "pid": p.pid, "sock": str(sock), "kit": _kit()}))
        return 0
    print(json.dumps({"session": a.session, "pid": p.pid, "error": r["error"], "log": str(dirs.root / "calibd.log")}))
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
        line = json.loads(s.makefile("r").readline() or "{}")
        pid = line.get("pid") or pid
        s.sendall(b'{"cmd": "stop"}\n')
        s.close()
    except (OSError, ValueError) as e:
        print(json.dumps({"session": sid, "stopped": False, "error": str(e)}))
        return 2
    t_end = time.monotonic() + 15
    while read_state_line(sock, 1.0) is not None:       # stopped = its socket no longer answers
        if time.monotonic() > t_end:
            print(json.dumps({"session": sid, "stopped": False, "pid": pid, "error": "still answering after 15 s"}))
            return 2
        time.sleep(0.2)
    print(json.dumps({"session": sid, "stopped": True, "pid": pid}))
    return 0


def cmd_status(a):
    """the session's state; live (and pid, kit) from its socket, not from state.json"""
    sid = a.session or _current(a.root)
    if not sid:
        print(json.dumps({"live": False}))
        return 0
    dirs, sock = _session_paths(a.root, sid)
    st = Store(dirs.root).load_state() or {}
    line = read_state_line(sock)
    if line is not None:
        st = dict(st, pid=line.get("pid"), kit=line.get("kit"), phase=line.get("phase"))
    st["live"] = line is not None
    print(json.dumps(st, default=_json_default))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="calibd", description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["start", "run", "attach", "stop", "status"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--session", default=None)
    ap.add_argument("--profile", default="P10")
    ap.add_argument("--trial", action="store_true", help="placement-trial mode (neck at home, nothing moves)")
    ap.add_argument("--detect-workers", type=int, default=0)
    ap.add_argument("--replace", action="store_true", help="start: stop a calibd already running first")
    ap.add_argument("--session-mode", default="watch", choices=["watch", "full"],
                    help="full: the whole calibration session (motors, grids, deploy: M5); watch: read-only")
    ap.add_argument("--dry-run", action="store_true",
                    help="full session: every robot change is recorded in <session>/commands.txt, none is sent")
    a = ap.parse_args(argv)
    if a.cmd == "run" and not a.session:
        ap.error("run needs --session")
    return {"start": cmd_start, "run": cmd_run, "attach": cmd_attach, "stop": cmd_stop, "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
