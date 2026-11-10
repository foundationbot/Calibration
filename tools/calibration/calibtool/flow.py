"""The full calibration session in calibd (design plan sections 4, 6, 7, 11; M5 wiring notes), `--session-mode full`.

FullFlow.run() walks the session in its own thread, one stage after the other, each persisted in state.json
(state["flow"]) and events.jsonl so a restarted calibd resumes where it stopped:

  motors       MOTORS (entered from READY by key M, only with the preflight motion gate green: daemon.Machine). The
               pre-enable checks on screen; Y = "the robot hangs on the gantry and nobody is at its head" -> Enabler
               .enable("Y", gantry=True). A refused or unverified enable stays in MOTORS with the reason.
  setup        SETUP: the deploy snapshot (deploy.take_snapshot: robot.env, configs, head_camera.json, OKVIS image and
               restart path) before anything changes; OKVIS stopped for the collection (sessionctl.OkvisCtl); the
               producer to the collection size under the lease (sessionctl.ProducerCtl, no-overlap cycle).
  collect      per placement: PLACE (the spoken guidance until "Good. Step back."; C = can't put it there; a timeout
               skips) -> the spoken 5 s countdown with abort live -> GRID (grids.GridRun over neck.run_grid: HOLD,
               abort and the heartbeat honoured between poses; the neck goes home after the grid). Defaults, then the
               planner's extras until the stop rule, then the redo list once. Every hold is saved in corners/ as it is
               captured; the placement in progress is saved pose by pose. robot.live_fit (refit.LiveFit), when
               given: a background refit after each placement and, once the collection ends, one on the final split
               (it runs while the IMU sweep does); its failures are events, never the session's.
  sweep        IMU_PLACE ("board flat on the floor"; the board seen still -> "Good. Step back."; C skips the sweep) ->
               countdown -> IMU_SWEEP (imusweep.capture_sweep over a NeckSweepRig whose every move is gated like a grid
               pose) -> the IMU fit (robot.imu_fit: imusweep.to_sweep + imu.verdict).
  decide       FIT (robot.fit_decide: split, step 45 -- or the live refit on the final split standing for it, see
               flowrobot.pipeline_fit_decide -- step 66, decide.verdict) -> DECIDE. Anything but DEPLOY keeps the
               live calibration (no deploy stage runs).
  deploy_size  DEPLOY_SIZE: deploy.size_options of the previous default; keys 1 / 2, ? = this robot's facts;
               deploy.config_for_size (a size OKVIS would refuse keeps live).
  check        at a size other than the fit's 1280: the producer to that size, CHECK_PLACE (1.1 m) -> countdown ->
               CHECK_GRID (15 poses) -> deploy.check_verdict on robot.check_scores; not DEPLOY keeps live. At 1280 the
               lens verdict is the check (skipped, said so).
  deploy       DEPLOY: deploy.Deployer.deploy (rollback.sh first; a stopped OKVIS is started as the restart step).
               DeployDrift: asks (Y = the robot's current state becomes the snapshot); DeployRefused: keeps live;
               DeployFailed: rolls back. A deploy interrupted by a calibd restart asks: Y deploys again (only when
               nothing changed), R rolls back.
  postcheck    POST_PLACE (1.1 m) -> POSTCHECK: depth check always (postcheck.depth_check on robot.post_scores), the
               OKVIS check unless opted out (key O, any time), postcheck.decide -> Deployer.rollback when it says so.
  report       REPORT: report/session.json + session.md.
  finish       FINISH: "D = disable the motors (the warning first, D again), K = keep them on"; R = roll back on purpose.
  restore      the producer as saved (unless the deploy left it at the deploy size), OKVIS started again (unless the
               deploy did), the lease released -> DONE.

Abort (key A, any phase): the motion in progress finishes its pose and nothing more is sent; a deploy not yet accepted
is rolled back; the producer and OKVIS are restored; the report is written; the motors stay as they are until the
operator presses D twice (handle_terminal_keys, from calibd's loop). A failed rollback is loud: STOPPED_LOUD, no retry.

HOLD (no heartbeat for 5 s in an active phase, or the flow's own: a foreign command, a missed pose, a refused arm, a
failed producer/OKVIS step): nothing more is sent; only G leaves it, and before the next neck command the countdown
runs again. A calibd restart resumes in HOLD (daemon.Machine.resume); after G a motion stage asks for Y again (motors
off: the full enable; still on: Y confirms going on), and an interrupted placement is guided and counted down again.

Every robot action goes into the session's command log (sessionctl.Recorder -> <session>/commands.txt): in a dry run
(Robot.dry_run, every backend recording instead of sending) that file is the dry-run command list of the session."""
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from calibtool import deploy as dp, grids, imusweep, neck as nk, place, planner, postcheck as pc, protocol, \
    sessionctl as sc
from calibtool.producer import LeaseHeld, ProducerError, Spec

STAGES = ("motors", "setup", "collect", "sweep", "decide", "deploy_size", "check", "deploy", "postcheck", "report",
          "finish", "restore")
MOTION_STAGES = frozenset({"collect", "sweep", "check"})
COUNTDOWN_S = 5
STEP_S = 0.15
MAX_EXTRAS = 6
LEASE_BEAT_S = 60.0
FLAT_MIN_TAGS = 8
FLAT_STILL_S = 1.0
FLAT_STILL_PX = 8.0
POST_FRAMES = 6
CHECK_TARGET = place.target("1.1 m")
POST_TARGET = place.target("1.1 m")
CHECK_POSES = [(p, y) for p in (-12.0, 0.0, 12.0) for y in (-24.0, -12.0, 0.0, 12.0, 24.0)]
NATIVE = dp.NATIVE
MOTORS_TEXT = ("Motor check. If the robot hangs on the gantry and nobody is at its head, press Y to enable the motors.")
FLAT_TEXT = "Put the board flat on the floor in front of the robot. Then step back."
FINISH_TEXT = "The session is done. Press D to disable the motors, or K to keep them on."


class Aborted(Exception):
    """the operator aborted (key A): restore and stop"""


class LoudStop(Exception):
    """a rollback failed: everything stops, nothing is retried"""


@dataclass
class Robot:
    """what the flow drives. On the robot flowrobot builds it (motors.run_cmd, the fixed dma_motor_motion, the
    writable producer Kube, Capture/DetectPool, the IMU slot, the pipeline); the tests pass fakes. In a dry run every
    mutating backend records instead of sending (Enabler(dry_run), neck.DryRun, sessionctl.DryKubectl,
    deploy.Remote(dry_run)) and the reads may still be real.

    fit_decide(collection, holds, out_dir) -> {"decision", "reasons", "config_text" (native 1280 OKVIS yaml), "name"}:
    the split, step 45 (pipeline.fit45), step 66 and decide.verdict. The speed branch's fast solver plugs in there
    (its `solver=` option on the fit), not in this module."""
    run: object                     # (argv, input=None, timeout=None) -> (rc, text): deploy.Remote / OKVIS commands
    dry_run: bool
    recorder: sc.Recorder
    enabler: object                 # motors_enable.Enabler
    motors_on: object               # () -> bool | None (ping motor_state)
    make_neck: object               # () -> neck.Neck
    kube: object                    # producer.Kube(read_only=False)
    lease: object                   # producer.Lease
    producer_yaml: str              # the preflight producer spec (the previous default)
    grab: object                    # Capture.grab(n)
    detect: object                  # DetectPool.submit_pair(yl, yr)
    board_obs: object               # () -> (fresh, BoardObs | None): the newest detection for the guidance
    make_collection: object         # () -> replay.Collection
    fit_decide: object
    check_scores: object            # (holds, size, config_text) -> (new, live) decide.Scores at the deploy size
    post_scores: object             # (pairs, config_text, size) -> (sync_ms, new, old) decide.Scores
    okvis_pods: object              # () -> deploy.okvis_pods()
    deploy_dir: str                 # the session's deploy folder on the robot
    session: str
    make_guide: object = place.PlacementGuide
    guide_kw: dict = field(default_factory=dict)        # focal, size, up, stand
    poses_for: object = None        # (target) -> [(pitch, yaw)]
    predict: object = None          # (target) -> predict(pitch, yaw) -> grids.Visible, or None
    plan_next: object = None        # (collection) -> the next extra target or None
    imu_accel: object = None        # (seconds) -> (k, 3)
    frames0: object = None          # (k) -> camera-0 grey frames
    sweep_detect: object = None     # (frame) -> imusweep.BoardView | None
    R_live: object = None
    imu_fit: object = None          # (results, R_live) -> {"deploy", "reasons", "correction"}
    okvis_probe: object = None      # (sha12) -> postcheck.okvis_check's keyword arguments
    collect_size: tuple = (1280, 800, 15)
    live_fit: object = None         # refit.LiveFit: placement_done / collection_done / close (fit_decide reads it)


def default_poses(t) -> list:
    g = planner.grid()
    return [g[i] for i in protocol.poses_for(t.distance_m, t.tilted)]


def _tdict(t) -> dict:
    d = asdict(t)
    d["accept"] = [None if (isinstance(v, float) and math.isinf(v)) else v for v in d["accept"]]
    return d


def _target(d) -> place.Target:
    d = dict(d)
    d["accept"] = tuple(math.inf if v is None else v for v in d["accept"])
    return place.Target(**d)


class HoldBag:
    """the check grid's collection: keeps the holds, no coverage"""

    def __init__(self):
        self.holds, self.stop = [], None

    def add(self, h):
        self.holds.append(h)
        return "TEST"

    def end_placement(self, name=None):
        return None


class _Persisting:
    """the collection, with every hold saved to corners/ before it is added (a crash loses no capture)"""

    def __init__(self, col, flow):
        self.col, self.flow = col, flow

    @property
    def stop(self):
        return getattr(self.col, "stop", None)

    def add(self, h):
        self.flow._save_hold(h)
        return self.col.add(h)

    def end_placement(self, name=None):
        return self.col.end_placement(name)


class _GatedRig(nk.NeckSweepRig):
    """the IMU sweep's rig: every move waits for the flow's between-poses gate (abort, HOLD + countdown)"""

    def __init__(self, neck, imu_accel, frames, flow):
        super().__init__(neck, imu_accel, frames)
        self.flow = flow

    def move_neck(self, pitch_deg, yaw_deg):
        if self.flow._between_poses() == "abort":
            raise Aborted()
        self.neck.goto(pitch_deg, yaw_deg)


class FullFlow:
    """machine: daemon.Machine (mode full); robot: Robot; dirs: storage.SessionDirs; emit(place.Say): calibd's say
    (state + event; the TUI speaks it); ids: the prompt-id counter calibd numbers from"""

    def __init__(self, machine, robot: Robot, dirs, emit, ids, clock=time.time, mono=time.monotonic,
                 sleep=time.sleep, targets=place.DEFAULTS, countdown_s=COUNTDOWN_S, step_s=STEP_S,
                 max_extras=MAX_EXTRAS, place_timeout_s=place.TIMEOUT_S):
        self.m, self.robot, self.dirs, self.emit, self.ids = machine, robot, dirs, emit, ids
        self.clock, self.mono, self.sleep = clock, mono, sleep
        self.targets = list(targets)
        self.countdown_s, self.step_s, self.max_extras = int(countdown_s), float(step_s), int(max_extras)
        self.place_timeout_s = float(place_timeout_s)
        self.rec = robot.recorder
        self.remote = dp.Remote(robot.run, dry_run=robot.dry_run, log=self.rec.hook("deploy"))
        self.spec = Spec.from_yaml(robot.producer_yaml)
        self.producer = sc.ProducerCtl(robot.kube, self.spec, robot.lease, robot.session,
                                       Path(dirs.root) / "producer.saved.yaml", clock=clock, sleep=sleep)
        self.snap = None
        self.okvis = None
        self.neck = None
        self.collection = None
        self.pcol = None
        self.resumed = False
        self._reconfirmed = False
        self._lease_beat = None
        self._grid_poses, self._pose_done = [], 0

    # ------------------------------------------------------------ state, events, speech

    @property
    def fl(self) -> dict:
        return self.m.state.setdefault("flow", {})

    def _save(self):
        self.m.save()

    def _upd(self, **kw):
        with self.m._lock:
            self.m.state.update(kw)
            self.m.save()

    def _event(self, kind, **kw):
        return self.m.store.event(kind, t=self.clock(), **kw)

    def _say(self, kind, text):
        self.emit(place.Say(next(self.ids), float(self.clock()), kind, text))

    # ------------------------------------------------------------ waiting: HOLD, abort, keys

    def _idle(self):
        self.m.tick()
        t = self.mono()
        if self._lease_beat is None or t - self._lease_beat >= LEASE_BEAT_S:
            self._lease_beat = t
            try:
                self.producer.heartbeat()
            except OSError:
                pass
        self.sleep(self.step_s)

    def _check(self):
        """-> the phase once it is not HOLD (waits for G); Aborted / LoudStop when the session ended"""
        while True:
            ph = self.m.phase
            if ph == "ABORTED":
                raise Aborted()
            if ph == "STOPPED_LOUD":
                raise LoudStop()
            if ph != "HOLD":
                return ph
            self._idle()

    def _enter(self, phase, why=""):
        if self._check() != phase:
            self.m.to(phase, why or f"flow: {phase}")

    def _hold(self, why):
        self._event("flow_hold", why=str(why))
        self.m.hold(why)
        self._check()

    def _wait_keys(self, allowed):
        while True:
            self._check()
            for k in self.m.take_keys():
                if k in allowed:
                    return k
            self._idle()

    def _countdown(self, phase):
        """the spoken step-back + countdown_s countdown, abort live; a HOLD during it starts it over after G"""
        n = self.countdown_s
        while True:
            self._enter(phase)
            self._event("countdown_start", phase=phase)
            self._upd(countdown=n)
            self._say("countdown", f"Step back. Starting in {n}.")
            t0, held = self.mono(), False
            for i in range(n, 0, -1):
                if i < n:
                    self._upd(countdown=i)
                    self._say("countdown", f"{i}.")
                while self.mono() < t0 + (n - i + 1):
                    self.m.tick()
                    ph = self.m.phase
                    if ph == "ABORTED":
                        self._upd(countdown=None)
                        raise Aborted()
                    if ph == "HOLD":
                        held = True
                        break
                    self.sleep(self.step_s)
                if held:
                    break
            self._upd(countdown=None)
            if not held:
                self._event("countdown_done", phase=phase)
                return
            self._event("countdown_interrupted", phase=phase)
            self._check()

    def _between_poses(self):
        """grids/neck should_stop: None to go on; "abort". A HOLD waits here (nothing sent) for G, then counts down"""
        self.m.tick()
        held = False
        while True:
            ph = self.m.phase
            if ph in ("ABORTED", "STOPPED_LOUD"):
                return "abort"
            if ph != "HOLD":
                break
            held = True
            self._idle()
        if held:
            try:
                self._countdown(self.m.phase)
            except Aborted:
                return "abort"
        return None

    # ------------------------------------------------------------ the run

    def run(self):
        self.rec.attach(Path(self.dirs.root) / "commands.txt")
        self._load()
        try:
            self._start()
            while self.fl.get("stage") not in (None, "done"):
                st = self.fl["stage"]
                getattr(self, "_st_" + st)()
                i = STAGES.index(st)
                self.fl["stage"] = STAGES[i + 1] if i + 1 < len(STAGES) else "done"
                self._event("stage", done=st, next=self.fl["stage"])
                self._save()
        except Aborted:
            self._aborted()
        except LoudStop:
            self._event("flow_stopped_loud")
        finally:
            self._save()
            if self.robot.live_fit is not None:
                try:
                    self.robot.live_fit.close()
                except Exception:  # noqa: BLE001  (the background refit's end is not the session's)
                    pass

    # ------------------------------------------------------------ the live refit (never stops the session)

    def _live_fit(self, what, *args):
        lf = self.robot.live_fit
        if lf is None:
            return None
        try:
            fut = getattr(lf, what)(*args)
        except Exception as e:  # noqa: BLE001
            self._event("live_refit_error", step=what, error=f"{type(e).__name__}: {e}")
            return None
        if fut is None:
            return None
        self._event("live_refit_submitted", step=what)

        def done(f):
            try:
                r = f.result()
            except Exception as e:  # noqa: BLE001
                self._event("live_refit_error", step=what, error=f"{type(e).__name__}: {e}")
                return
            d = {"index": r.index, "placement": r.placement, "pairs": r.pairs, "rms_px": r.rms_px,
                 "baseline_mm": r.baseline_mm, "converged": r.converged, "rejected": r.rejected,
                 "seconds": r.seconds, "verdict": getattr(r, "verdict", "")}
            self._event("live_refit", **d)
            self._upd(live_refit=d)
        fut.add_done_callback(done)
        return fut

    def _start(self):
        if self.fl.get("stage") is None:
            self.fl["stage"] = "motors"
            self._save()
            while self._check() != "MOTORS":         # READY -> M (daemon.Machine checks the motion gate)
                self._idle()
            return
        self.resumed = True
        self.rec.note(f"calibd restarted: resuming at stage {self.fl['stage']} (HOLD until G)")
        self._event("flow_resumed", stage=self.fl["stage"])
        self._check()

    def _load(self):
        """the objects a restarted calibd rebuilds from the session folder"""
        fl = self.fl
        p = fl.get("producer")
        if p:
            self.producer.changed, self.producer.size = bool(p.get("changed")), tuple(p.get("size") or
                                                                                       self.spec.size())
        if (Path(self.dirs.deploy) / "deploy_snapshot.json").is_file():
            self.snap = dp.DeploySnapshot.load(self.dirs.deploy)
            self._make_okvis()
        self.collection = self.robot.make_collection()
        holds = self._load_holds()
        if holds:
            from calibtool import replay
            replay.feed(self.collection, holds)
        self.pcol = _Persisting(self.collection, self)

    def _make_okvis(self):
        rem = dp.Remote(self.robot.run, dry_run=self.robot.dry_run, log=self.rec.hook("okvis"))
        d = self.fl.get("okvis")
        self.okvis = sc.OkvisCtl.from_dict(rem, d) if d else sc.OkvisCtl(rem, self.snap.restart)

    def _persist_ctl(self):
        self.fl["producer"] = self.producer.as_dict()
        if self.okvis is not None:
            self.fl["okvis"] = self.okvis.as_dict()
        self._save()

    # ------------------------------------------------------------ holds on disk

    def _save_hold(self, h):
        d = Path(self.dirs.corners)
        d.mkdir(parents=True, exist_ok=True)
        np.savez(d / f"hold_{int(h.hold):06d}.npz", **{k: np.asarray(v) for k, v in h.data.items()})
        (d / f"hold_{int(h.hold):06d}.json").write_text(json.dumps(
            {"hold": int(h.hold), "placement": str(h.placement), "t0": int(h.t0), "pose": int(h.pose)}))
        self._save()

    def _load_holds(self) -> list:
        from calibtool import assign
        out = []
        for j in sorted(Path(self.dirs.corners).glob("hold_*.json")):
            meta = json.loads(j.read_text())
            npz = j.with_suffix(".npz")
            if not npz.is_file():
                continue
            with np.load(npz) as z:
                data = {k: z[k] for k in z.files}
            out.append(assign.CaptureHold(meta["hold"], meta["placement"], meta["t0"], data, meta.get("pose", -1)))
        return sorted(out, key=lambda h: h.hold)

    def _hold_ids(self):
        while True:
            h = int(self.fl.get("next_hold") or 0)
            self.fl["next_hold"] = h + 1
            yield h

    # ------------------------------------------------------------ the neck

    def _neck(self):
        if self.neck is None:
            n = self.robot.make_neck()
            inner = n.run

            def run(argv, timeout=None):
                self.rec.add("neck", argv)
                self._event("neck_command", argv=list(argv))
                return inner(argv, timeout=timeout)
            n.run = run
            self.neck = n
        return self.neck

    def _arm(self):
        n = self._neck()
        while True:
            try:
                n.arm()
                return n
            except nk.NeckRefused as e:
                self._event("neck_refused", why=str(e))
                self._hold(f"neck not armed: {e}")

    def _home(self, n):
        while True:
            out = nk.run_grid(n, [(0.0, 0.0)], lambda d: None, self._between_poses)
            if out.stopped == "abort":
                raise Aborted()
            if out.stopped:
                self._hold(f"going home: {out.stopped}")
                n = self._arm()
                self._countdown(self.m.phase)
                continue
            return

    def _grid_event(self, kind, **kw):
        self._event(kind, **kw)
        if kind == "pose":
            self._pose_done += 1
            cur = self.fl.get("current")
            if cur is not None and self._grid_kind == "collect":
                cur["poses"] = [list(p) for p in self._grid_poses[self._pose_done:]]
            self._upd(grid={"placement": kw.get("placement"), "pose": self._pose_done, "of": len(self._grid_poses),
                            "last": kw.get("record")})

    def _grid(self, target, poses, col, phase, kind):
        """one placement's grid with its countdown; a foreign command / missed pose HOLDs, and after G the poses left
        run (re-armed, counted down). -> the last GridResult"""
        remaining = [tuple(map(float, p)) for p in poses]
        predict = self.robot.predict(target) if self.robot.predict else None
        res = None
        while remaining:
            n = self._arm()
            self._countdown(phase)
            self._grid_poses, self._pose_done, self._grid_kind = remaining, 0, kind
            g = grids.GridRun(lambda ps, on_pose, ss, n=n: nk.run_grid(n, ps, on_pose, ss), self.robot.grab,
                              self.robot.detect, col, target.name, remaining, predict=predict,
                              hold_ids=self._hold_ids(), should_stop=self._between_poses, on_event=self._grid_event,
                              stop_when_rule_met=(kind == "collect"))
            res = g.run()
            remaining = remaining[len(res.records):]
            self._upd(grid=res.as_dict())
            if res.stopped == "abort":
                raise Aborted()
            if isinstance(res.stopped, str) and res.stopped.startswith("error"):
                self._hold(res.stopped[len("error: "):])
                continue
            break
        self._home(self._neck())
        return res

    # ------------------------------------------------------------ the guidance

    def _intro(self, target, kind):
        if kind == "default":
            names = [t.name for t in self.targets]
            i = names.index(target.name) + 1 if target.name in names else 1
            return place.intro(target, i, len(self.targets))
        where = place.intro(target, 1, 1).split(". ", 1)[1]
        lead = {"extra": "Extra placement.", "redo": "Again.", "check": "Check grid.",
                "post": "Post-check."}.get(kind, "")
        return f"{lead} {where}".strip()

    def _guide(self, target, phase, kind):
        """-> "placed" | "unreachable" (C, or the guide's out of reach) | "timeout\""""
        self._enter(phase)
        g = self.robot.make_guide(target, ids=self.ids, **self.robot.guide_kw)
        intro = self._intro(target, kind)
        self._upd(placement={"target": target.name, "kind": kind, "phase": phase})
        self._say("intro", intro)
        t0, paused = self.mono(), 0.0
        while True:
            if self.m.phase == "HOLD":
                th = self.mono()
                self._check()
                paused += self.mono() - th
                g.interrupted()
                self._say("intro", intro)
            self._check()
            for k in self.m.take_keys():
                if k == "C":
                    self._event("placement_cant", placement=target.name)
                    return "unreachable"
            fresh, obs = self.robot.board_obs()
            if fresh is not False:
                s = g.update(self.clock(), obs)
                if s is not None:
                    self.emit(s)
            if g.done:
                return g.result or "placed"
            if self.mono() - t0 - paused > self.place_timeout_s:
                self._say("skip", place.SKIP_TEXT)
                return "timeout"
            self._idle()

    def _flat_wait(self):
        """IMU_PLACE: the board seen (>= FLAT_MIN_TAGS tags) and still for FLAT_STILL_S -> "placed"; C -> "skipped\""""
        self._enter("IMU_PLACE")
        self._upd(placement={"target": "flat on the floor", "kind": "imu", "phase": "IMU_PLACE"})
        self._say("intro", FLAT_TEXT)
        since, c0, t0 = None, None, self.mono()
        while True:
            self._check()
            for k in self.m.take_keys():
                if k == "C":
                    return "skipped"
            fresh, obs = self.robot.board_obs()
            if fresh is not False:
                ok = obs is not None and int(getattr(obs, "n_tags", 0)) >= FLAT_MIN_TAGS
                c = np.asarray(obs.centroid, float) if ok else None
                if not ok or c0 is None or float(np.hypot(*(c - c0))) > FLAT_STILL_PX:
                    since, c0 = (self.mono(), c) if ok else (None, None)
                elif self.mono() - since >= FLAT_STILL_S:
                    self._say("good", place.DONE_TEXT)
                    return "placed"
            if self.mono() - t0 > self.place_timeout_s:
                self._say("skip", place.SKIP_TEXT)
                return "timeout"
            self._idle()

    # ------------------------------------------------------------ stages

    def _reconfirm(self):
        """after a calibd restart a motion stage needs the operator's Y again (design 4: reboot)"""
        if self.resumed and not self._reconfirmed:
            self._st_motors()
            self._reconfirmed = True

    def _st_motors(self):
        self._enter("MOTORS")
        en = self.robot.enabler
        if self.fl.get("motors_by_us") and self.robot.motors_on():
            detail = "The motors are on (this session enabled them before calibd restarted). Y = go on with them on."
            self._upd(motors={"checks": [], "detail": detail, "resumed": True})
            self._say("motors", detail)
            self._wait_keys("Y")
            self._event("motors_reconfirmed")
            return
        self._say("motors", MOTORS_TEXT)
        last = None
        while True:
            checks = [c.as_dict() for c in en.checks(gantry=False)]
            self._upd(motors={"checks": checks, "detail": MOTORS_TEXT if last is None else last.detail,
                              "refused": None if last is None else last.refused})
            self._wait_keys("Y")
            self._upd(gantry_confirmed=True)
            n0 = len(en.commands)
            res = en.enable("Y", gantry=True)
            for a in en.commands[n0:]:
                self.rec.add("motors", a)
            if len(en.commands) > n0:
                self.fl["motors_by_us"] = True
            self._upd(motors=res.as_dict())
            last = res
            if res.ok or res.dry_run:
                self._upd(motors_on=True)
                self._event("motors_enabled", n=res.n, of=res.of, dry_run=res.dry_run)
                return
            if res.refused:
                self._event("motors_enable_refused", checks=[c.as_dict() for c in res.checks if c.status == "fail"],
                            detail=res.detail)
            else:
                self._event("motors_enable_failed", detail=res.detail, n=res.n, of=res.of)
            self._say("motors", "Not enabled. The reason is on the screen.")

    def _try(self, fn, what):
        while True:
            try:
                return fn()
            except (dp.RemoteError, ProducerError, LeaseHeld) as e:
                self._event("setup_failed", what=what, error=str(e))
                self._persist_ctl()
                self._hold(f"{what} failed: {e}")

    def _snapshot(self):
        if self.snap is None:
            self.snap = dp.take_snapshot(self.remote, producer_yaml=self.robot.producer_yaml, clock=self.clock)
            Path(self.dirs.deploy).mkdir(parents=True, exist_ok=True)
            self.snap.save(self.dirs.deploy)
            self._event("deploy_snapshot", restart=self.snap.restart, calibration=self.snap.calibration)
            self._make_okvis()

    def _st_setup(self):
        self._enter("SETUP")
        self._snapshot()
        self._try(self.okvis.stop, "stopping OKVIS")
        self._persist_ctl()
        self._event("okvis_stopped", how=self.okvis.kind, node=self.okvis.node)
        self._try(lambda: self.producer.to_size(*self.robot.collect_size), "the producer to the collection size")
        self._persist_ctl()
        self._event("producer_size", size=list(self.producer.size))

    def _met(self):
        return getattr(self.collection, "stop", None) is not None

    def _placement(self, target, kind, poses=None):
        fl = self.fl
        poses = [tuple(p) for p in (poses if poses is not None else
                                    (self.robot.poses_for or default_poses)(target))]
        fl["current"] = {"target": _tdict(target), "kind": kind, "poses": [list(p) for p in poses]}
        self._save()
        r = self._guide(target, "PLACE", kind)
        if r != "placed":
            fl.setdefault("unreachable", []).append(target.name)
            self._event("placement_skipped", placement=target.name, result=r)
        else:
            res = self._grid(target, poses, self.pcol, "GRID", "collect")
            self._live_fit("placement_done", self.collection, self._load_holds(), target.name)
            if res is not None:
                for s in (res.sentences or [])[:3]:
                    self._say("gap", s)
                if res.early_stop is not None:
                    self._say("early", res.early_stop.sentence)
                    if kind != "redo":
                        fl.setdefault("redo", []).append({"target": _tdict(target), "reason": res.early_stop.reason,
                                                          "poses": [list(p) for p in poses]})
        if kind == "default":
            fl.setdefault("done", []).append(target.name)
        fl["current"] = None
        self._save()

    def _st_collect(self):
        self._reconfirm()
        fl = self.fl
        cur = fl.get("current")
        if cur:
            self._placement(_target(cur["target"]), cur["kind"], cur["poses"])
        for t in self.targets:
            if self._met():
                break
            if t.name not in fl.get("done", []):
                self._placement(t, "default")
        while self.robot.plan_next is not None and not self._met() and int(fl.get("extras") or 0) < self.max_extras:
            t = self.robot.plan_next(self.collection)
            if t is None:
                break
            fl["extras"] = int(fl.get("extras") or 0) + 1
            self._placement(t, "extra")
        if not fl.get("redo_done"):
            items, fl["redo"], fl["redo_done"] = list(fl.get("redo") or []), [], True
            for it in items:
                if self._met():
                    fl["redo"].append(it)
                    continue
                self._placement(_target(it["target"]), "redo", it["poses"])
        self._event("collect_done", stop_met=self._met(), unreachable=fl.get("unreachable", []),
                    redo=fl.get("redo", []), holds=int(fl.get("next_hold") or 0))
        self._live_fit("collection_done", self.collection, self._load_holds(), self.dirs.fit)

    def _st_sweep(self):
        if self.fl.get("imu") is not None:
            return
        self._reconfirm()
        r = self._flat_wait()
        if r != "placed":
            self.fl["imu"] = {"deploy": False, "skipped": r, "reasons": [f"IMU sweep skipped ({r})"],
                              "correction": None}
            self._upd(sweep=self.fl["imu"])
            return
        n = self._arm()
        self._countdown("IMU_SWEEP")
        while True:
            rig = _GatedRig(n, self.robot.imu_accel, self.robot.frames0, self)
            try:
                results = imusweep.capture_sweep(rig, imusweep.serpentine(), self.robot.sweep_detect,
                                                 self.robot.R_live, sleep=self.sleep,
                                                 log=lambda s: self._event("sweep_pose", line=s))
                break
            except (imusweep.SweepAborted, nk.NeckRefused, nk.NeckMissed) as e:
                self._hold(f"IMU sweep stopped: {e}")
                n = self._arm()
                self._countdown("IMU_SWEEP")
        self._home(n)
        try:
            out = dict(self.robot.imu_fit(results, self.robot.R_live))
        except Exception as e:  # noqa: BLE001  (a failed IMU fit keeps the live rotation, said why)
            out = {"deploy": False, "reasons": [f"IMU fit failed: {type(e).__name__}: {e}"], "correction": None}
        corr = out.get("correction")
        self.fl["imu"] = {"deploy": bool(out.get("deploy")), "reasons": list(out.get("reasons") or []),
                          "correction": None if corr is None else np.asarray(corr, float).tolist(),
                          "poses": len(results)}
        self._upd(sweep=self.fl["imu"])
        self._event("sweep_done", poses=len(results), deploy=self.fl["imu"]["deploy"])

    def _config_text(self):
        p = self.fl.get("config_file")
        return Path(p).read_text() if p and Path(p).is_file() else None

    def _st_decide(self):
        fl = self.fl
        if fl.get("decision") is None:
            self._enter("FIT")
            self._say("fit", "Fitting the lens. This takes a few minutes.")
            try:
                out = dict(self.robot.fit_decide(self.collection, self._load_holds(), self.dirs.fit))
            except Exception as e:  # noqa: BLE001  (no result = keep live, with the error)
                out = {"decision": "NOT_ENOUGH_DATA", "reasons": [f"the fit failed: {type(e).__name__}: {e}"]}
            text = out.get("config_text")
            imu_ = fl.get("imu") or {}
            if text and imu_.get("deploy") and imu_.get("correction") is not None:
                from calibtool import imu
                text = imu.apply(text, np.asarray(imu_["correction"], float), note="calibtool IMU sweep")
            if text:
                p = Path(self.dirs.fit) / "new_config.yaml"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(text)
                fl["config_file"] = str(p)
            fl["decision"] = {"decision": out.get("decision"), "reasons": list(out.get("reasons") or []),
                              "name": out.get("name"), "imu_applied": bool(text and imu_.get("deploy")),
                              "fit_source": out.get("fit_source")}
            self._save()
        self._enter("DECIDE")
        dec = fl["decision"]
        self._upd(decision=dec)
        self._event("decision", **dec)
        if dec["decision"] != "DEPLOY" or not fl.get("config_file"):
            fl["outcome"] = "keep_live"
            self._say("decision", "Keeping the live calibration. The reasons are on the screen.")
        else:
            self._say("decision", "The new calibration is better. Choose the deploy size.")

    def _size_help(self, opts) -> list:
        prev = self.spec.size()
        out = [f"The previous default: the producer at {prev[0]}x{prev[1]} @ {prev[2]} fps (saved in preflight)."]
        out += [f"{o.key}: {o.label}" for o in opts]
        deps = [d.path for d in (self.snap.dependents if self.snap else [])]
        out.append("Configs on this robot naming the live calibration (they keep reading it): "
                   + (", ".join(deps) if deps else "none found"))
        out.append("At a size other than 1280 a check grid at that size decides: deploy only when new is shown "
                   "better than live there too.")
        out.append(f"1280x800 means every camera consumer runs at 1280 (the producer, OKVIS, the configs above).")
        return out

    def _st_deploy_size(self):
        fl = self.fl
        if fl.get("outcome"):
            return
        ds = fl.get("deploy_size") or {}
        if not ds.get("chosen"):
            self._enter("DEPLOY_SIZE")
            opts = dp.size_options(self.spec.size()[:2])
            ds = {"prev": list(self.spec.size()[:2]), "help": None, "chosen": None,
                  "options": [{"key": o.key, "size": list(o.size), "scale": o.scale, "label": o.label,
                               "recommended": o.recommended} for o in opts]}
            self._upd(deploy_size=ds)
            while True:
                k = self._wait_keys("12?")
                if k == "?":
                    ds["help"] = self._size_help(opts)
                    self._upd(deploy_size=ds)
                    continue
                o = next((x for x in opts if x.key == k), None)
                if o is not None:
                    break
                self._event("key_dropped", key=k, why="no such deploy size option")
            ds["chosen"] = {"key": o.key, "size": list(o.size), "scale": o.scale, "label": o.label}
            fl["deploy_size"] = ds
            self._upd(deploy_size=ds)
            self._event("deploy_size", **ds["chosen"])
        size = tuple(ds["chosen"]["size"])
        try:
            text = dp.config_for_size(self._config_text(), size)
        except ValueError as e:
            fl["outcome"] = "keep_live"
            fl["decision"]["reasons"].append(str(e))
            self._event("deploy_size_refused", why=str(e))
            return
        p = Path(self.dirs.deploy) / (fl["decision"].get("name") or "calibtool.yaml")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        fl["deploy_config"] = str(p)
        self._save()

    def _deploy_size(self):
        return tuple(self.fl["deploy_size"]["chosen"]["size"])

    def _st_check(self):
        fl = self.fl
        if fl.get("outcome") or (fl.get("check") or {}).get("decision"):
            return
        size = self._deploy_size()
        if size == tuple(NATIVE):
            fl["check"] = {"skipped": "deploying at 1280x800, the size the fit was judged at: the lens verdict is "
                                      "the check"}
            self._upd(check=fl["check"])
            return
        self._reconfirm()
        self._enter("SETUP")
        fps = self.spec.size()[2] or self.robot.collect_size[2]
        self._try(lambda: self.producer.to_size(size[0], size[1], fps), "the producer to the deploy size")
        self._persist_ctl()
        r = self._guide(CHECK_TARGET, "CHECK_PLACE", "check")
        if r != "placed":
            fl["outcome"] = "keep_live"
            fl["check"] = {"decision": "KEEP_LIVE", "reasons": [f"the check grid was not made ({r})"]}
            self._upd(check=fl["check"])
            return
        bag = HoldBag()
        self._grid(CHECK_TARGET, CHECK_POSES, bag, "CHECK_GRID", "check")
        new, live = self.robot.check_scores(bag.holds, size, Path(fl["deploy_config"]).read_text())
        v = dp.check_verdict(new, live)
        fl["check"] = {"decision": v.decision, "reasons": list(v.reasons), "holds": len(bag.holds)}
        self._upd(check=fl["check"])
        self._event("check_verdict", decision=v.decision, reasons=v.reasons)
        if v.decision != "DEPLOY":
            fl["outcome"] = "keep_live"

    def _deployer(self):
        fps = self.spec.size()[2] or self.robot.collect_size[2]
        d = dp.Deployer(self.remote, self.snap, self.robot.deploy_dir,
                        producer_apply=lambda size: self.producer.to_size(size[0], size[1], fps),
                        producer_restore=self.producer.restore, clock=self.mono, sleep=self.sleep,
                        session=self.robot.session)
        left = self.fl.get("deploy_left")
        if left:
            d.left = [tuple(x) for x in left]
        return d

    def _st_deploy(self):
        fl = self.fl
        if fl.get("outcome"):
            return
        self._enter("DEPLOY")
        name = fl["decision"].get("name") or "calibtool.yaml"
        plan = dp.DeployPlan(name, Path(fl["deploy_config"]).read_text(), self._deploy_size())
        if fl.get("deploy_started") and not fl.get("deploy_done"):
            ch = dp.drift(self.snap.expected(), self.remote)
            self._upd(deploy={"interrupted": True, "drift": ch})
            self._say("deploy", "The deploy was interrupted. Y deploys again, R rolls back.")
            while True:
                k = self._wait_keys("YR")
                if k == "R":
                    self._rollback("the operator asked for it after an interrupted deploy (R)", force=True)
                    return
                if not ch:
                    break
                self._event("deploy_retry_refused", drift=ch)
        while True:
            fl["deploy_started"] = True
            self._save()
            start = self.okvis.start_argv() if self.okvis is not None and self.okvis.stopped else None
            d = self._deployer()
            try:
                res = d.deploy(plan, restart_argv=start)
            except dp.DeployDrift as e:
                fl["deploy_started"] = False
                self._upd(deploy={"drift": str(e)})
                self._event("deploy_drift", why=str(e))
                self._say("deploy", "The robot changed since the session started. Look at the screen. Y deploys "
                                    "over the robot as it is now.")
                self._wait_keys("Y")
                self.snap = dp.take_snapshot(self.remote, producer_yaml=self.robot.producer_yaml, clock=self.clock)
                self.snap.save(self.dirs.deploy)
                continue
            except dp.DeployRefused as e:
                fl["deploy_started"], fl["outcome"] = False, "refused"
                self._upd(deploy={"refused": str(e)})
                self._event("deploy_refused", why=str(e))
                return
            except dp.DeployFailed as e:
                self._event("deploy_failed", why=str(e))
                fl["outcome"] = "deployed"
                fl["deploy_left"] = None if self.robot.dry_run else [list(x) for x in (d.left or [])] or None
                self._rollback(f"the deploy failed: {e}")
                return
            break
        if start and self.okvis is not None:
            self.okvis.stopped = False          # the deploy started it on the new robot.env
        fl["deploy_done"], fl["outcome"] = True, "deployed"
        # what the deploy left is what a rollback compares the robot with; a dry run changed nothing, so the
        # rollback compares with the snapshot
        fl["deploy_left"] = None if self.robot.dry_run else [list(x) for x in (d.left or [])]
        fl["sha12"] = res.sha12
        self._persist_ctl()
        self._upd(deploy=res.as_dict())
        self._event("deploy_done", ok=res.ok, dry_run=res.dry_run, sha12=res.sha12, rollback=res.rollback_script,
                    detail=res.detail)

    def _rollback(self, reason, force=False, interactive=True):
        fl = self.fl
        d = self._deployer()
        self._event("rollback_start", reason=reason)
        while True:
            try:
                rb = d.rollback(reason, force=force)
                break
            except dp.RollbackDrift as e:
                self._upd(rollback={"drift": str(e)})
                self._event("rollback_drift", why=str(e))
                if not interactive:
                    self._upd(flow_banner=[f"NOT ROLLED BACK: {e}"])
                    return
                self._say("rollback", "The robot changed since the deploy. Look at the screen. R rolls back anyway.")
                self._wait_keys("R")
                force = True
            except dp.RollbackFailed as e:
                self._event("ROLLBACK_FAILED", message=str(e))
                self._upd(flow_banner=[str(e)], rollback={"ok": False, "message": str(e)})
                self._say("rollback", "Rollback failed. Stop. Look at the screen.")
                self.m.to("STOPPED_LOUD", "rollback failed")
                raise LoudStop() from e
        fl["rolled_back"], fl["outcome"] = True, "rolled_back"
        self._persist_ctl()
        self._upd(rollback={"ok": rb.ok, "reason": reason, "notes": rb.notes, "dry_run": self.robot.dry_run})
        self._event("rollback", reason=reason, ok=rb.ok or self.robot.dry_run, notes=rb.notes)

    def _st_postcheck(self):
        fl = self.fl
        if fl.get("outcome") != "deployed" or fl.get("postcheck"):
            return
        size = self._deploy_size()
        text = Path(fl["deploy_config"]).read_text()
        r = self._guide(POST_TARGET, "POST_PLACE", "post")
        self._enter("POSTCHECK")
        depth = None
        if r == "placed":
            pairs = self.robot.grab(POST_FRAMES)
            sync_ms, new, old = self.robot.post_scores(pairs, text, size)
            depth = pc.depth_check(sync_ms, new, old)
        okv = None
        if self.m.state.get("okvis_check") is not False and self.robot.okvis_probe is not None:
            okv = pc.okvis_check(sha12=fl.get("sha12"), **self.robot.okvis_probe(fl.get("sha12")))
        crash = dp.crash_loop(self.robot.okvis_pods() or [])
        dec = pc.decide(depth, okv, crash)
        fl["postcheck"] = {"depth": depth.as_dict() if depth else {"status": "not_done", "reasons": [
            f"the board was not placed for the depth check ({r})"]}, "okvis": okv.as_dict() if okv else
            {"skipped": self.m.state.get("okvis_check") is False}, "crash_loop": crash, "decision": dec.as_dict()}
        self._upd(postcheck=fl["postcheck"])
        self._event("postcheck", rollback=dec.rollback, reasons=dec.reasons, crash_loop=crash)
        if dec.rollback:
            self._say("rollback", "The post-check failed. Rolling back.")
            self._rollback("post-check: " + dec.reasons[0])
        for k in self.m.take_keys():
            if k == "R" and fl.get("outcome") == "deployed":
                self._rollback("the operator asked for it (R)")

    def _report(self, aborted=False) -> dict:
        fl = self.fl
        st = self.m.state
        rep = {"session": self.robot.session, "dry_run": self.robot.dry_run, "aborted": aborted,
               "outcome": "aborted" if aborted and fl.get("outcome") is None else fl.get("outcome"),
               "decision": fl.get("decision"), "imu": fl.get("imu"), "deploy_size": fl.get("deploy_size"),
               "check": fl.get("check"), "deploy": st.get("deploy"), "postcheck": fl.get("postcheck"),
               "rollback": st.get("rollback"), "restore": st.get("restore"), "motors": st.get("motors"),
               "collection": {"done": fl.get("done", []), "unreachable": fl.get("unreachable", []),
                              "redo": fl.get("redo", []), "holds": int(fl.get("next_hold") or 0)},
               "commands": str(Path(self.dirs.root) / "commands.txt")}
        d = Path(self.dirs.report)
        d.mkdir(parents=True, exist_ok=True)
        from calibtool.daemon import _json_default
        (d / "session.json").write_text(json.dumps(rep, indent=1, default=_json_default))
        md = [f"# Calibration session {rep['session']}", "", f"- outcome: **{rep['outcome']}**"
              + (" (dry run: nothing was sent)" if rep["dry_run"] else "")]
        if rep["decision"]:
            md.append(f"- lens verdict: {rep['decision'].get('decision')}")
            md += [f"  - {r}" for r in rep["decision"].get("reasons") or []]
        if rep["check"]:
            md.append(f"- check grid: {rep['check'].get('decision') or rep['check'].get('skipped')}")
        if rep["postcheck"]:
            md.append(f"- post-check: {rep['postcheck']['decision']}")
        md.append(f"- placements: {', '.join(rep['collection']['done']) or 'none'}; unreachable: "
                  f"{', '.join(rep['collection']['unreachable']) or 'none'}; holds {rep['collection']['holds']}")
        md.append(f"- robot commands: {rep['commands']}")
        (d / "session.md").write_text("\n".join(md) + "\n")
        self._event("report", outcome=rep["outcome"], aborted=aborted)
        return rep

    def _st_report(self):
        self._enter("REPORT")
        self._report()

    def _motor_key(self) -> bool:
        """one D: the warning first, the disable on the second -> True once disabled (or nothing to disable)"""
        fl = self.fl
        if not fl.get("motors_by_us"):
            self._upd(motors_prompt="The motors were not enabled by this session: nothing to disable.")
            return True
        if not fl.get("disable_armed"):
            fl["disable_armed"] = True
            self._upd(motors_prompt=me_warning())
            self._say("motors", me_warning())
            return False
        fl["disable_armed"] = False
        en = self.robot.enabler
        n0 = len(en.commands)
        res = en.disable(True)
        for a in en.commands[n0:]:
            self.rec.add("motors", a)
        fl["motors_disabled"] = bool(res.ok or res.dry_run)
        self._upd(motors=res.as_dict(), motors_on=not fl["motors_disabled"], motors_prompt=res.detail)
        self._event("motors_disabled", ok=res.ok, dry_run=res.dry_run, detail=res.detail)
        return True

    def _st_finish(self):
        self._enter("FINISH")
        self.fl["final"] = True
        self._upd(motors_prompt="D = disable the motors (the warning comes first), K = keep them on")
        self._say("finish", FINISH_TEXT)
        while True:
            k = self._wait_keys("DKR")
            if k == "R":
                if self.fl.get("outcome") == "deployed":
                    self._rollback("the operator asked for it (R)")
                    self._report()
                continue
            if k == "K":
                self._upd(motors_prompt="The motors stay on.")
                self._event("motors_kept")
                return
            if self._motor_key():
                return

    def _restore_robot(self):
        fl, errs, done = self.fl, [], []
        deployed = fl.get("outcome") == "deployed" and not fl.get("rolled_back")
        try:
            if deployed:
                self.producer.changed = False       # the deploy set the producer's size: it stays
                self.robot.lease.release()
                done.append("producer left at the deploy size, lease released")
            elif self.producer.restore():
                done.append("producer restored as saved")
            else:
                self.robot.lease.release()
        except Exception as e:  # noqa: BLE001  (each step reported; the next one still runs)
            errs.append(f"producer: {type(e).__name__}: {e}")
        try:
            if self.okvis is not None and self.okvis.stopped:
                self.okvis.start()
                done.append("OKVIS started again")
        except Exception as e:  # noqa: BLE001
            errs.append(f"OKVIS: {type(e).__name__}: {e}")
        self._persist_ctl()
        self._upd(restore={"done": done, "errors": errs}, flow_banner=[f"RESTORE FAILED: {x}" for x in errs])
        self._event("restored", done=done, errors=errs)

    def _st_restore(self):
        self._restore_robot()
        self.m.to("DONE", "session finished")

    def _aborted(self):
        fl = self.fl
        self._event("abort_restore", stage=fl.get("stage"))
        self.rec.note("aborted: restoring the robot")
        try:
            if fl.get("outcome") == "deployed" and not fl.get("rolled_back") and not fl.get("final"):
                self._rollback("aborted by the operator (A)", interactive=False)
        except LoudStop:
            self._report(aborted=True)
            return
        self._restore_robot()
        self._report(aborted=True)

    def handle_terminal_keys(self):
        """D after the session ended (DONE / ABORTED / STOPPED_LOUD): the motor disable, the warning first"""
        for k in self.m.take_keys():
            if k == "D" and not self.fl.get("motors_disabled"):
                self._motor_key()


def me_warning() -> str:
    from calibtool import motors_enable
    return motors_enable.DISABLE_WARNING
