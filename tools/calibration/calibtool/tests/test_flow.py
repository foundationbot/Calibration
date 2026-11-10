"""calibtool.flow: the full calibration session in calibd (M5 wiring), end to end against a FAKE robot in dry run.

The fake robot: the motor master (ping, journal) and an all-green pre-enable probe; the neck through neck.DryRun (the
exact dma_motor_motion argv, simulated angles, /dev/shm/desired a counter a test can advance = a foreign command);
the producer through sessionctl.DryKubectl; OKVIS, robot.env and the configs through test_deploy's FakeRobot (plus
the node label and okvis.sh stop/start); capture + detection from test_grids' synthetic boards; a scripted placement
guide; fake IMU, fit, scores and OKVIS pods. The operator is a script: heartbeats and keys pressed when the machine
reaches a phase. Time is one fake clock; the flow's waits advance it.

Asserted: the order of robot actions; the refusals (motion gate red -> no MOTORS, no Y -> no enable, a failed
pre-enable check -> no enable, a foreign command -> the grid stops and HOLDs); abort at each phase restores the
producer and OKVIS and leaves the motors as the operator chooses; heartbeat loss in GRID -> HOLD between poses with no
further command; resume after a crash in GRID / DEPLOY -> HOLD, nothing continues by itself; a deploy only after a
DEPLOY verdict; the rollback on a crash-looping post-check; the session's command list written to its folder."""
import importlib.util
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
yaml = pytest.importorskip("yaml")

from calibtool import (daemon as dm, deploy as dp, flow as fl, imusweep, motors_enable as me, neck as nk,  # noqa: E402
                       place, producer as pr, sessionctl as sc, storage)
from calibtool import decide  # noqa: E402

HERE = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(f"_t_flow_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TD = _load("test_deploy")
TG = _load("test_grids")

PRODUCER_YAML = TD.PRODUCER_YAML.replace('"1280", "--height", "800"', '"960", "--height", "600"')
NAMES = ["NeckPitch", "NeckYaw", "LeftShoulderPitch", "RightShoulderPitch", "SpineYaw", "LeftHipPitch",
         "RightHipPitch", "LeftAnklePitch"]
TARGETS = (place.target("0.75 m"), place.target("1.6 m"))
POSES = [(-8.0, -11.0), (0.0, 0.0), (8.0, 11.0)]


class Crash(BaseException):
    """calibd dies here (kill -9, a power cut): nothing in the flow catches it"""


class Clock:
    def __init__(self):
        self.t = 1.79e9
        self.on_sleep = None
        self.describe = None
        self.limit = self.t + 50_000

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += max(float(s), 1e-3)
        if self.t > self.limit:
            raise AssertionError("the flow did not finish (stuck waiting): "
                                 + (self.describe() if self.describe else ""))
        if self.on_sleep is not None:
            self.on_sleep()


class Operator:
    """heartbeats while hb is True; presses keys[phase] (in order) once the machine is in that phase"""

    def __init__(self, m, keys=None):
        self.m, self.hb = m, True
        self.keys = {k: list(v) for k, v in (keys or {}).items()}
        self.pressed = []
        self.hooks = []         # callables run every tick (tests' own scripted events)

    def tick(self):
        if self.hb:
            self.m.heartbeat()
        self.m.tick()
        for h in list(self.hooks):
            h()
        q = self.keys.get(self.m.phase)
        if q:
            k = q.pop(0)
            self.pressed.append((self.m.phase, k))
            self.m.key(k)


class Robot(TD.FakeRobot):
    """test_deploy's robot plus the OKVIS node label, okvis.sh stop/start, hostname"""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.label = "true"

    def __call__(self, argv, input=None, timeout=None):
        a = list(argv)
        if a[:2] == ["env", f"KUBECONFIG={dp.KUBECONFIG}"]:
            k = a[4:]
            if "jsonpath={.items[0].spec.nodeName}" in k:
                self.calls.append((a, input))
                return 0, "phantom10"
            if k[:2] == ["get", "node"]:
                self.calls.append((a, input))
                return 0, self.label
            if k[:2] == ["label", "node"]:
                self.calls.append((a, input))
                self.label = k[3].split("=")[1]
                return 0, "labeled"
        if a[0] == "env" and a[-2:] in ([dp.OKVIS_SH, "stop"], [dp.OKVIS_SH, "start"]):
            self.calls.append((a, input))
            return 0, ""
        return super().__call__(argv, input, timeout)


class Motors:
    """the motor master: ping, journal; enable/disable flip motor_state (only reached outside a dry run)"""

    def __init__(self):
        self.on = False
        self.calls = []

    def __call__(self, argv, timeout=None):
        self.calls.append(list(argv))
        if argv == me.ENABLE_ARGV:
            self.on = True
            return 0, "ok"
        if argv == me.DISABLE_ARGV:
            self.on = False
            return 0, "ok"
        if argv[-1] == "ping":
            return 0, f"PONG uptime=10.0s motor_state={1 if self.on else 0}"
        if argv[0] == "journalctl":
            return 0, "50 of 50 motors enabled" if self.on else ""
        return 1, "?"


def good_probe(**over):
    s = {"autonomy": None, "pods": [], "desired_still": True,
         "master": SimpleNamespace(active=True, error=None, version="2.1.2", estop=False, motors_on=False,
                                   motor_state=0)}
    s.update(over)
    return lambda: dict(s)


class Guide:
    """a placement guide that judges the board placed on its first fresh view (or as scripted per target)"""

    results = {}

    def __init__(self, target, ids, **kw):
        self.target, self.ids, self.done, self.result, self.says = target, ids, False, None, []
        self.views = 0

    def update(self, t, obs, miss=None):
        if obs is None or self.done:
            return None
        self.views += 1
        if self.views < 3:                              # the operator carries the board for a moment
            return None
        self.done = True
        self.result = self.results.get(self.target.name, "placed")
        self.near_z = 0.88 if self.result == "placed_near" else None
        text = place.DONE_TEXT if self.result.startswith("placed") else place.UNREACHABLE_TEXT
        return place.Say(next(self.ids), t, "good", text)

    def interrupted(self):
        pass


def scores(err_new, err_live, n_holds=15, per=2, seed=1):
    rng = np.random.default_rng(seed)
    n = n_holds * per
    src = np.arange(n)
    hold = np.repeat(np.arange(n_holds), per)

    def one(e, dy):
        return decide.Scores(src, hold, np.array(["x"] * n), e + rng.normal(0, 0.05, n), np.full(n, 1.0),
                             np.repeat(src, 20), dy + rng.normal(0, 0.05, n * 20))
    return one(err_new, 0.2), one(err_live, 0.8 if err_live > err_new else 0.2)


class Session:
    """one fake robot + machine + flow; run() drives the flow to its end (or a Crash)"""

    def __init__(self, tmp_path, keys=None, decision="DEPLOY", motion_green=True, probe=None, crash_loop=False,
                 targets=TARGETS, store=None, robot=None, clock=None, motors=None, kube=None, resume=False,
                 neck_backend=None, live_fit=None):
        self.tmp = tmp_path
        self.clock = clock or Clock()
        self.dirs = storage.SessionDirs(tmp_path / "root", "20261110-010203-P10").create()
        store = store or dm.Store(self.dirs.root)
        if resume:
            self.m = dm.Machine.resume(store, self.clock, mono=self.clock)
        else:
            self.m = dm.Machine(store, self.clock, session=self.dirs.sid, robot="P10", read_only=False, mode="full",
                                mono=self.clock)
            self.m.state["motion_gate_green"] = motion_green
            self.m.state["preflight_green"] = True
            self.m.to("READY", "preflight green")
        self.op = Operator(self.m, keys)
        self.clock.on_sleep = self.op.tick
        self.clock.describe = lambda: f"phase {self.m.phase}, last events " + str(
            [(e["kind"], e.get("to") or e.get("key") or e.get("why")) for e in self.m.store.events()[-12:]])
        self.robot = robot or Robot(okvis_sh=False)
        self.motors = motors or Motors()
        self.rec = sc.Recorder(dry_run=True)
        self.kube = kube or sc.DryKubectl(PRODUCER_YAML)
        self.kube.recorder = self.rec
        self.neck_backend = neck_backend or nk.DryRun(NAMES, [0.0] * len(NAMES))
        self.world = TG.World(200)
        self.grab_hook = None
        self.crash_loop = crash_loop
        self.decision = decision
        self.says = []
        self.fit_calls = 0
        self.lease = pr.Lease(tmp_path / "root" / "producer.lease")
        ids = itertools.count(int(self.m.state.get("say_id_max") or 0) + 1)
        nb = self.neck_backend

        def make_neck():
            return nk.Neck("/root/p11-cal/dma_motor_motion", nb.run, nb.robot, nb.tool_info, clock=nb.clock,
                           sleep=nb.clock.sleep)

        def grab(n):
            if self.grab_hook is not None:
                self.grab_hook(self)
            self.world.pose = min(self.world.pose + 1, 199)
            return self.world.grab(n)

        def fit_decide(col, holds, out_dir):
            self.fit_calls += 1
            return {"decision": self.decision, "reasons": [f"fake verdict {self.decision}"], "config_text": TD.NEW,
                    "name": TD.NEW_NAME}

        self.robot_side = fl.Robot(
            run=self.robot, dry_run=True, recorder=self.rec,
            enabler=me.Enabler(self.motors, probe or good_probe(), clock=self.clock, sleep=self.clock.sleep,
                               wall=self.clock, dry_run=True),
            motors_on=lambda: self.motors.on, make_neck=make_neck,
            kube=pr.Kube(self.kube, "dma-video", "producer", read_only=False), lease=self.lease,
            producer_yaml=PRODUCER_YAML, grab=grab, detect=self.world.detect,
            board_obs=lambda: (True, SimpleNamespace(z=1.0, centroid=(640.0, 400.0), n_tags=30, cut=())),
            make_collection=lambda: TG.FakeCollection(), make_guide=Guide, poses_for=lambda t: list(POSES),
            imu_accel=lambda s: np.tile([0.0, 0.0, 9.81], (20, 1)), frames0=lambda k: [None] * k,
            sweep_detect=lambda g: imusweep.BoardView(np.array([0.0, 0.0, 1.0]), 40, 0.3), R_live=np.eye(3),
            imu_fit=lambda results, R: {"deploy": False, "reasons": [f"{len(results)} poses: fake"],
                                        "correction": None},
            fit_decide=fit_decide, check_scores=lambda holds, size, text: scores(0.2, 0.9),
            post_scores=self._post_scores,
            okvis_probe=lambda sha: {"log_lines": [f"({sha})", "KB4T pinhole camera 0"]},
            okvis_pods=self._pods, deploy_dir=f"/data/calib/{self.dirs.sid}/deploy", session=self.dirs.sid,
            live_fit=live_fit)
        self.flow = fl.FullFlow(self.m, self.robot_side, self.dirs, emit=self.says.append, ids=ids, clock=self.clock,
                                mono=self.clock, sleep=self.clock.sleep, targets=targets)

    def _post_scores(self, pairs, text, size):
        self.clock.sleep(1.0)                           # scoring the post-check frames takes a moment
        return (0.1,) + scores(0.2, 0.25)

    def _pods(self):
        if self.crash_loop:
            return [{"name": "okvis2x-1", "phase": "Running", "restarts": 3, "waiting": "CrashLoopBackOff"}]
        return [{"name": "okvis2x-1", "phase": "Running", "restarts": 0, "waiting": None}]

    def run(self):
        self.flow.run()
        return self

    def events(self):
        return self.m.store.events()

    def kinds(self):
        return [e["kind"] for e in self.events()]

    def lines(self, source=None):
        return [" ".join(a) for a in self.rec.commands(source)]

    def neck_cmds(self):
        return self.rec.commands("neck")


FULL_KEYS = {"READY": ["M"], "MOTORS": ["Y"], "DEPLOY_SIZE": ["1"], "FINISH": ["D", "D"]}


def first(lines, needle):
    return next(i for i, x in enumerate(lines) if needle in x)


def last(lines, needle):
    return max(i for i, x in enumerate(lines) if needle in x)


# ---------------------------------------------------------------- the whole session


def test_a_whole_dry_run_session_in_order(tmp_path):
    s = Session(tmp_path, keys=FULL_KEYS).run()
    assert s.m.phase == "DONE", s.kinds()[-20:]
    L = s.lines()
    enable = first(L, "enable-motors")
    okvis_stop = first(L, "has-okvis=false")
    prod_1280 = first(L, '"1280"')
    neck0 = first(L, "dma_motor_motion")
    deploy_env = first(L, dp.ROBOT_ENV)
    disable = first(L, "disable-motors")
    assert enable < okvis_stop < prod_1280 < neck0 < deploy_env < disable
    # the check grid at the deploy size (960x600, the previous default) runs after the producer went back to it
    prod_960 = first(L, '"960"')
    assert prod_1280 < prod_960 < last(L, "dma_motor_motion") < deploy_env
    # the deploy starts the stopped OKVIS (no rollout restart of a DaemonSet with no pod)
    assert any("has-okvis=true" in x for x in L[deploy_env:])
    assert not any("--grace-period" in x or "--kd" in x for x in L)
    # every grid pose and the IMU sweep went through the neck: 2 placements x 3 poses + homes + 15 sweep + check grid
    assert len(s.neck_cmds()) >= 2 * 3 + 15 + 15
    # the command list is in the session folder
    txt = (s.dirs.root / "commands.txt").read_text()
    assert txt.startswith("# DRY RUN") and "enable-motors" in txt and "disable-motors" in txt
    k = s.kinds()
    for ev in ("motors_enabled", "grid_end", "deploy_done", "postcheck", "motors_disabled", "report"):
        assert ev in k, ev
    assert (s.dirs.report / "session.json").is_file()
    assert s.lease.read() is None


def test_each_grid_starts_after_the_spoken_countdown(tmp_path):
    s = Session(tmp_path, keys=FULL_KEYS).run()
    texts = [x.text for x in s.says]
    i = texts.index("Step back. Starting in 5.")
    assert texts[i + 1:i + 5] == ["4.", "3.", "2.", "1."]
    ev = s.events()
    cd = [e["t"] for e in ev if e["kind"] == "countdown_done"]
    nc = [e["t"] for e in ev if e["kind"] == "neck_command"]
    assert cd and nc and cd[0] <= nc[0]
    starts = [e["t"] for e in ev if e["kind"] == "countdown_start"]
    assert cd[0] - starts[0] >= 5.0


# ---------------------------------------------------------------- refusals


def test_a_red_motion_gate_never_reaches_motors(tmp_path):
    s = Session(tmp_path, keys={"READY": ["M", "A"]}, motion_green=False).run()
    assert "motors_refused" in s.kinds() and "MOTORS" not in [e.get("to") for e in s.events()]
    assert s.lines() == []


def test_without_y_nothing_is_enabled(tmp_path):
    s = Session(tmp_path, keys={"READY": ["M"], "MOTORS": ["C", "1", "A"]}).run()
    assert s.m.phase == "ABORTED"
    assert s.lines() == [] and s.motors.calls == []


def test_a_failed_pre_enable_check_refuses_the_enable(tmp_path):
    s = Session(tmp_path, keys={"READY": ["M"], "MOTORS": ["Y", "A"]},
                probe=good_probe(autonomy="state: ARMED")).run()
    assert "motors_enable_refused" in s.kinds()
    assert not any("enable-motors" in x for x in s.lines())
    assert s.m.state["motors"]["refused"] is True


def test_a_foreign_command_stops_the_grid_and_holds(tmp_path):
    def hook(sess):
        if len(sess.neck_cmds()) == 2:
            sess.neck_backend.robot.head += 1          # another program wrote /dev/shm/desired
    keys = dict(FULL_KEYS)
    keys["HOLD"] = ["A"]
    s = Session(tmp_path, keys=keys)
    s.grab_hook = hook
    s.run()
    k = s.kinds()
    assert "hold" in k
    hold = next(e for e in s.events() if e["kind"] == "hold")
    assert hold["phase"] == "GRID" and "desired" in hold["why"]
    assert len(s.neck_cmds()) == 2                     # nothing after the foreign command
    assert s.m.phase == "ABORTED"


# ---------------------------------------------------------------- heartbeat


def test_heartbeat_loss_in_grid_holds_between_poses_and_g_counts_down_again(tmp_path):
    s = Session(tmp_path, keys=FULL_KEYS)
    state = {"lost": None}

    def hook(sess):
        if len(sess.neck_cmds()) == 2 and state["lost"] is None:
            sess.op.hb = False                         # the link drops during this pose's capture ...
            sess.clock.t += 6.0                        # ... which outlasts the 5 s deadman
            state["lost"] = sess.clock.t

    def back():
        if s.m.phase == "HOLD" and not s.op.hb and s.clock.t - state["lost"] > 60:
            state["cmds_in_hold"] = len(s.neck_cmds())
            s.op.hb = True                             # the TUI is back ...
            s.m.heartbeat()
            s.op.keys["HOLD"] = ["G"]                  # ... and the operator presses G
    s.grab_hook = hook
    s.op.hooks.append(back)
    s.run()
    ev = s.events()
    lost = next(e for e in ev if e["kind"] == "heartbeat_lost")
    assert lost["phase"] == "GRID"
    assert state["cmds_in_hold"] == 2                   # no command while held
    g = next(e["t"] for e in ev if e["kind"] == "resumed_by_operator")
    cd_after = [e["t"] for e in ev if e["kind"] == "countdown_done" and e["t"] > g]
    nc_after = [e["t"] for e in ev if e["kind"] == "neck_command" and e["t"] > g]
    assert cd_after and nc_after and cd_after[0] <= nc_after[0]
    assert s.m.phase == "DONE"


# ---------------------------------------------------------------- abort at each phase


@pytest.mark.parametrize("phase", ["MOTORS", "PLACE", "GRID", "IMU_SWEEP", "DEPLOY_SIZE", "CHECK_GRID", "POSTCHECK",
                                   "FINISH"])
def test_abort_at_each_phase_restores_producer_and_okvis_and_leaves_the_motors(tmp_path, phase):
    keys = {k: list(v) for k, v in FULL_KEYS.items()}
    keys.pop("FINISH")
    keys[phase] = ["A"]
    s = Session(tmp_path, keys=keys).run()
    assert s.m.phase == "ABORTED"
    assert s.kube.args[s.kube.args.index("--width") + 1] == "960" and s.kube.replicas == 1
    assert s.robot.label == "true"                     # OKVIS back (never stopped, or started again)
    assert s.lease.read() is None
    L = s.lines()
    assert not any("disable-motors" in x for x in L)  # the motors stay as they are ...
    if phase == "POSTCHECK":
        assert "rollback" in s.kinds()                 # ... and a deploy made before the abort is rolled back
    if phase == "MOTORS":
        assert L == []                                 # nothing was changed, nothing to restore
        return
    s.m.key("D")
    s.flow.handle_terminal_keys()
    assert not any("disable-motors" in x for x in s.lines())
    assert s.m.state["motors_prompt"] == me.DISABLE_WARNING
    s.m.key("D")
    s.flow.handle_terminal_keys()                      # ... until the operator presses D twice
    assert any("disable-motors" in x for x in s.lines())


# ---------------------------------------------------------------- resume after a crash


def test_a_crash_in_grid_resumes_in_hold_and_continues_only_after_g_and_y(tmp_path):
    def crash(sess):
        if len(sess.neck_cmds()) == 5:                # the second placement's first pose
            raise Crash()
    s = Session(tmp_path, keys=FULL_KEYS)
    s.grab_hook = crash
    with pytest.raises(Crash):
        s.run()
    before = len(s.neck_cmds())
    holds_before = sorted(p.name for p in s.dirs.corners.glob("hold_*.npz"))
    assert holds_before
    s.motors.on = True                                 # calibd died, the motors did not
    s2 = Session(tmp_path, keys={}, store=dm.Store(s.dirs.root), robot=s.robot, kube=s.kube, motors=s.motors,
                 resume=True, neck_backend=s.neck_backend, clock=s.clock)
    assert s2.m.phase == "HOLD" and s2.m.state["held_from"] == "GRID"
    waited = {"n": 0}

    def later():
        waited["n"] += 1
        if waited["n"] == 300:                         # ~45 s in HOLD with the TUI attached: nothing moves
            assert s2.m.phase == "HOLD" and s2.neck_cmds() == []
            s2.op.keys.update({"HOLD": ["G"], "MOTORS": ["Y"], "DEPLOY_SIZE": ["1"], "FINISH": ["K"]})
    s2.op.hooks.append(later)
    s2.run()
    assert s2.m.phase == "DONE"
    ev = s2.events()
    g = next(e["t"] for e in ev if e["kind"] == "resumed_by_operator")
    y = next(e["t"] for e in ev if e["kind"] == "key" and e["key"] == "Y" and e["t"] > g)
    first_cmd = next(e["t"] for e in ev if e["kind"] == "neck_command" and e["t"] > g)
    assert g < y < first_cmd
    assert not any("enable-motors" in x for x in s2.lines())   # motors were on: Y confirms, nothing re-sent
    # the holds of the crashed run are back in the collection and the interrupted placement finished its poses
    assert len(s2.flow.collection.added) >= len(holds_before)
    assert before == 5


def test_a_crash_in_deploy_resumes_in_hold_and_waits_for_the_operator(tmp_path):
    s = Session(tmp_path, keys=FULL_KEYS)
    orig = s.robot.__call__

    class Boom:
        armed = True

    def run(argv, input=None, timeout=None):
        if Boom.armed and s.m.phase == "DEPLOY" and argv[0] == "cat":
            raise Crash()                               # calibd dies at the deploy's first read
        return orig(argv, input, timeout)
    s.robot_side.run = run
    s.flow = fl.FullFlow(s.m, s.robot_side, s.dirs, emit=s.says.append, ids=itertools.count(1), clock=s.clock,
                         mono=s.clock, sleep=s.clock.sleep, targets=TARGETS)
    with pytest.raises(Crash):
        s.run()
    Boom.armed = False
    s.motors.on = True
    s2 = Session(tmp_path, keys={}, store=dm.Store(s.dirs.root), robot=s.robot, kube=s.kube, motors=s.motors,
                 resume=True, neck_backend=s.neck_backend, clock=s.clock)
    assert s2.m.phase == "HOLD" and s2.m.state["held_from"] == "DEPLOY"
    n = {"i": 0}

    def script():
        n["i"] += 1
        if n["i"] == 200:
            assert s2.m.phase == "HOLD" and not s2.lines("deploy")
            s2.op.keys["HOLD"] = ["G"]
        if n["i"] == 600:                               # back in DEPLOY: it asks, it does not continue by itself
            assert s2.m.phase == "DEPLOY" and s2.m.state["deploy"]["interrupted"] is True
            assert not s2.lines("deploy")
            s2.op.keys.update({"DEPLOY": ["Y"], "FINISH": ["K"]})
    s2.op.hooks.append(script)
    s2.run()
    assert s2.m.phase == "DONE" and "deploy_done" in s2.kinds()


# ---------------------------------------------------------------- deploy only on DEPLOY; rollback


@pytest.mark.parametrize("decision", ["KEEP_LIVE", "NOT_ENOUGH_DATA"])
def test_no_deploy_without_a_deploy_verdict(tmp_path, decision):
    s = Session(tmp_path, keys=FULL_KEYS, decision=decision).run()
    assert s.m.phase == "DONE"
    assert s.lines("deploy") == []
    assert "DEPLOY_SIZE" not in [e.get("to") for e in s.events()]
    assert s.kube.args[s.kube.args.index("--width") + 1] == "960" and s.robot.label == "true"
    rep = json.loads((s.dirs.report / "session.json").read_text())
    assert rep["outcome"] == "keep_live" and decision in json.dumps(rep["decision"])


def test_a_crash_looping_postcheck_rolls_back(tmp_path):
    s = Session(tmp_path, keys=FULL_KEYS, crash_loop=True).run()
    assert s.m.phase == "DONE"
    rb = next(e for e in s.events() if e["kind"] == "rollback")
    assert "crash" in rb["reason"].lower()
    L = s.lines("deploy")
    env_writes = [i for i, x in enumerate(L) if dp.ROBOT_ENV in x and x.startswith("sh -c")]
    assert len(env_writes) == 2                        # the deploy's, then the rollback's
    rep = json.loads((s.dirs.report / "session.json").read_text())
    assert rep["outcome"] == "rolled_back"


def test_r_rolls_back_on_purpose_after_the_postcheck(tmp_path):
    keys = dict(FULL_KEYS)
    keys["FINISH"] = ["R", "K"]
    s = Session(tmp_path, keys=keys).run()
    rb = next(e for e in s.events() if e["kind"] == "rollback")
    assert "operator" in rb["reason"]


def test_deploy_size_question_mark_explains_and_2_deploys_at_1280_without_a_check_grid(tmp_path):
    keys = dict(FULL_KEYS)
    keys["DEPLOY_SIZE"] = ["?", "2"]
    s = Session(tmp_path, keys=keys).run()
    assert s.m.state["deploy_size"]["help"]
    assert s.m.state["deploy_size"]["chosen"]["size"] == [1280, 800]
    assert "CHECK_GRID" not in [e.get("to") for e in s.events()]
    assert s.m.state["check"]["skipped"]
    assert "deploy_done" in s.kinds()


# ---------------------------------------------------------------- the live refit (refit.LiveFit) beside the session


class LiveFitSpy:
    """refit.LiveFit's flow side: records the calls (and when, by fit_decide calls so far); error: raised by every
    call; bad_future: a Future whose refit failed"""

    def __init__(self, session_ref, error=None, bad_future=False):
        self.s, self.error, self.bad_future = session_ref, error, bad_future
        self.calls, self.closed = [], False

    def _future(self, placement):
        from concurrent.futures import Future
        f = Future()
        if self.bad_future:
            f.set_exception(RuntimeError("refit process died"))
        else:
            f.set_result(SimpleNamespace(index=len(self.calls), placement=placement, pairs=10, rms_px=0.4,
                                         baseline_mm=44.0, converged=True, rejected=False, seconds=0.5,
                                         verdict="converged: fake"))
        return f

    def placement_done(self, col, holds, placement):
        if self.error:
            raise self.error
        self.calls.append(("placement_done", placement, len(holds), self.s[0].fit_calls))
        return self._future(placement)

    def collection_done(self, col, holds, out_dir):
        if self.error:
            raise self.error
        self.calls.append(("collection_done", str(out_dir), len(holds), self.s[0].fit_calls))
        return self._future("final split")

    def close(self):
        self.closed = True


def test_the_live_refit_runs_after_each_placement_and_on_the_final_split_before_the_fit(tmp_path):
    ref = []
    spy = LiveFitSpy(ref)
    s = Session(tmp_path, keys=FULL_KEYS, live_fit=spy)
    ref.append(s)
    s.run()
    assert s.m.phase == "DONE"
    names = [c[1] for c in spy.calls if c[0] == "placement_done"]
    ev = s.events()
    upto = next(i for i, e in enumerate(ev) if e["kind"] == "collect_done")
    assert names and names == [e["placement"] for e in ev[:upto] if e["kind"] == "grid_end"]
    assert spy.calls[-1][0] == "collection_done" and spy.calls[-1][1] == str(s.dirs.fit)
    assert all(c[3] == 0 for c in spy.calls) and s.fit_calls == 1        # every refit asked for before FIT
    assert [c[2] for c in spy.calls] == sorted(c[2] for c in spy.calls) and spy.calls[-1][2] > 0
    k = s.kinds()
    assert k.count("live_refit") == len(spy.calls) and "live_refit_error" not in k
    assert s.m.state["live_refit"]["placement"] == "final split"
    assert spy.closed


@pytest.mark.parametrize("how", ["call", "future"])
def test_a_failing_live_refit_never_stops_the_session(tmp_path, how):
    ref = []
    spy = LiveFitSpy(ref, error=RuntimeError("no process") if how == "call" else None, bad_future=how == "future")
    s = Session(tmp_path, keys=FULL_KEYS, live_fit=spy)
    ref.append(s)
    s.run()
    assert s.m.phase == "DONE" and s.fit_calls == 1
    errs = [e for e in s.events() if e["kind"] == "live_refit_error"]
    assert errs and all(("no process" if how == "call" else "died") in e["error"] for e in errs)


@pytest.mark.parametrize("limit", [None, 0])
def test_without_a_time_limit_only_the_operators_c_ends_a_placement(tmp_path, limit):
    """10 Nov (phantom10): the full session's PLACE gave up after 3 minutes ("Skipping this one.") while the operator
    was still trying; with the profile's limit off nothing skips by itself"""
    s = Session(tmp_path)
    s.robot_side.board_obs = lambda: (True, None)
    flow = fl.FullFlow(s.m, s.robot_side, s.dirs, emit=s.says.append, ids=itertools.count(1), clock=s.clock,
                       mono=s.clock, sleep=s.clock.sleep, targets=TARGETS, place_timeout_s=limit)
    s.m.heartbeat()
    t0 = s.clock.t
    s.op.hooks.append(lambda: s.clock.t - t0 > 900 and s.m.phase == "PLACE" and s.m.key("C"))
    assert flow._guide(place.target("0.75 m"), "PLACE", "default") == "unreachable"
    assert s.clock.t - t0 > 900
    assert place.SKIP_TEXT not in [x.text for x in s.says]


def test_the_full_session_keeps_the_time_limit_when_one_is_set(tmp_path):
    s = Session(tmp_path)
    s.robot_side.board_obs = lambda: (True, None)
    flow = fl.FullFlow(s.m, s.robot_side, s.dirs, emit=s.says.append, ids=itertools.count(1), clock=s.clock,
                       mono=s.clock, sleep=s.clock.sleep, targets=TARGETS, place_timeout_s=60.0)
    s.m.heartbeat()
    assert flow._guide(place.target("0.75 m"), "PLACE", "default") == "timeout"
    assert s.says[-1].text == place.SKIP_TEXT


def test_a_close_enough_placement_runs_its_grid_and_the_report_says_where(tmp_path, monkeypatch):
    """10 Nov: a near placement the operator cannot quite reach is used where the board is ("Close enough. Using 88
    centimetres."); its holds count toward the band they measured, and the report states the distance"""
    monkeypatch.setattr(Guide, "results", {"0.75 m": "placed_near"})
    s = Session(tmp_path, keys=FULL_KEYS).run()
    assert s.m.phase == "DONE"
    assert "0.75 m" in [e["placement"] for e in s.events() if e["kind"] == "grid_end"]
    assert "placement_skipped" not in s.kinds()
    rep = json.loads((s.dirs.report / "session.json").read_text())
    assert rep["collection"]["done"][:2] == ["0.75 m", "1.6 m"] and rep["collection"]["unreachable"] == []
    assert rep["collection"]["near"] == {"0.75 m": 0.88}
    assert "0.75 m at 88 centimetres" in (s.dirs.report / "session.md").read_text()


def test_the_full_sessions_guide_hears_why_the_board_is_not_seen(tmp_path):
    """10 Nov (phantom10): PLACE says the reason, not only 'I cannot see the board' after 6 s"""
    from calibtool import detect
    s = Session(tmp_path)
    s.robot_side.board_obs = lambda: (True, None)
    s.robot_side.board_miss = lambda: detect.Miss("partial", 3, 30.0, 120.0, ("bottom",))
    s.robot_side.make_guide = place.PlacementGuide
    s.robot_side.guide_kw = {"focal": 568.0, "size": (1280, 800)}
    s.m.heartbeat()
    t0 = s.clock.t
    s.op.hooks.append(lambda: s.clock.t - t0 > 10 and s.m.phase == "PLACE" and s.m.key("C"))
    assert s.flow._guide(place.target("1.1 m"), "PLACE", "default") == "unreachable"
    assert "I can see only part of the board. Raise the board." in [x.text for x in s.says]
