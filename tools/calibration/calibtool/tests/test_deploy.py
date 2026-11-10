"""calibtool.deploy: the deploy transaction and its rollback (design plan 6.3-6.4), against a FAKE robot only.

FakeRobot is a runner over an in-memory file system plus a fake kubectl / okvis.sh: an OKVIS "restart" reads
robot.env, loads the calibration it names and logs what the entrypoint and the reader would (the config's sha256
prefix, 'KB4T pinhole camera'), and crash-loops when the config fails devel PR #930's checks at the producer's size.
Nothing here touches a robot."""
import hashlib
import importlib.util
import json
import re
import shlex
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
yaml = pytest.importorskip("yaml")

from calibtool import decide, deploy as dp  # noqa: E402

HERE = Path(__file__).resolve().parent


def _okvis_yaml(**kw):
    spec = importlib.util.spec_from_file_location("_t_livecfg_d", HERE / "test_livecfg.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.okvis_yaml(**kw)


KB4T = dict(dist_type="kb4t", dist=(0.01, -0.02, 0.003, -0.001, 1e-4, -2e-4))
OLD_NAME = "phantom11_stereo_kb4t_20261106_1280_imu.yaml"
NEW_NAME = "phantom10_stereo_kb4t_20261110_1280.yaml"
OLD = _okvis_yaml(**KB4T, f=((583.0, 582.0), (582.0, 581.0)))
NEW = _okvis_yaml(**KB4T)
ENV = ("# per-robot OKVIS values\nOKVIS_DMA_CAMERAS=3,0\nOKVIS_CALIBRATION=/etc/okvis/" + OLD_NAME
       + "\nOKVIS_MODE=slam  # keep\n")
PRODUCER_YAML = """
apiVersion: apps/v1
kind: Deployment
metadata: {name: producer, namespace: dma-video, generation: 7, resourceVersion: "9"}
spec:
  replicas: 1
  strategy: {type: Recreate}
  selector: {matchLabels: {app: producer}}
  template:
    metadata: {labels: {app: producer}}
    spec:
      containers:
      - name: producer
        image: foundationbot/dma-video:x
        args: ["--width", "1280", "--height", "800", "--fps", "25", "--camera-config", "/opt/configs/cameras.json"]
status: {replicas: 1}
"""
DEPLOY_DIR = "/data/calib/20261110-010203-P10/deploy"


def sha12(t):
    return hashlib.sha256(t.encode()).hexdigest()[:12]


class FakeRobot:
    def __init__(self, ds_desired=1, okvis_sh=False, producer_size=(1280, 800)):
        self.files = {dp.ROBOT_ENV: ENV, f"{dp.HOST_CFG_DIR}/{OLD_NAME}": OLD, dp.HEAD_CAMERA_JSON: '{"cams": 1}',
                      "/etc/phantom/vla/rig.env": f"RIG_YAML=/etc/okvis/{OLD_NAME}\n",
                      "/etc/phantom/other.yaml": "nothing here\n"}
        if okvis_sh:
            self.files[dp.OKVIS_SH] = "#!/bin/bash\n"
        self.ds_desired = ds_desired
        self.image = "foundationbot/okvis2x:thor-98349e6c"
        self.producer_size = producer_size
        self.calls = []
        self.restarts = 0
        self.pod = {"phase": "Running", "restarts": 0, "waiting": None}
        self.log = []
        self.fail_write = set()
        self._boot_okvis()

    # ------------------------------------------------------------ OKVIS
    def _boot_okvis(self):
        env = dp.parse_env(self.files.get(dp.ROBOT_ENV, ""))
        cal = env.get("OKVIS_CALIBRATION", "")
        host = dp.host_path(cal)
        text = self.files.get(host)
        if text is None or dp.okvis_startup_checks(text, self.producer_size):
            self.pod = {"phase": "Running", "restarts": 3, "waiting": "CrashLoopBackOff"}
            self.log = [f"[entrypoint] robot.env: {cal}", "F1110 frames arrive at ... image_dimension"]
            return
        base = Path(cal).name
        self.pod = {"phase": "Running", "restarts": 0, "waiting": None}
        self.log = [f"[entrypoint] robot.env: OKVIS_CALIBRATION={cal}",
                    f"[entrypoint] config: {base} explicit path {cal} ({sha12(text)})"]
        if "kb4t" in text:
            self.log += ["I1110 KB4T pinhole camera 0 with T_SC=...", "I1110 KB4T pinhole camera 1 with T_SC=..."]

    # ------------------------------------------------------------ the runner
    def __call__(self, argv, input=None, timeout=None):
        self.calls.append((list(argv), input))
        a = list(argv)
        if a[:2] == ["cat", a[1]] and len(a) == 2:
            return (0, self.files[a[1]]) if a[1] in self.files else (1, f"cat: {a[1]}: No such file")
        if a[:2] == ["test", "-e"]:
            return (0, "") if a[2] in self.files else (1, "")
        if a[:2] == ["mkdir", "-p"]:
            return 0, ""
        if a[:2] == ["sh", "-c"] and "calibtool-tmp" in a[2]:
            path, mode = a[4], a[5]
            if path in self.fail_write:
                return 1, "write failed"
            self.files[path] = input
            return 0, ""
        if a[0] == "grep":
            name, roots = a[a.index("--") + 1], a[a.index("--") + 2:]
            out = []
            for p, t in sorted(self.files.items()):
                if any(p.startswith(r) for r in roots):
                    for i, ln in enumerate(t.splitlines(), 1):
                        if name in ln:
                            out.append(f"{p}:{i}:{ln}")
            return (0 if out else 1), "\n".join(out)
        if a[:2] == ["env", f"KUBECONFIG={dp.KUBECONFIG}"]:
            return self._kubectl(a[4:])
        if a[0] == "env" and a[-2:] == [dp.OKVIS_SH, "restart"]:
            self.restarts += 1
            self._boot_okvis()
            return 0, "restarted"
        return 127, f"unknown command {a}"

    def _kubectl(self, k):
        if "--grace-period=0" in k:
            raise AssertionError("grace period 0 reached the robot")
        if k[:4] == ["-n", dp.OKVIS_NS, "get", "ds"]:
            if self.ds_desired is None:
                return 1, 'Error from server (NotFound): daemonsets.apps "okvis2x" not found'
            jp = k[-1]
            if "desiredNumberScheduled" in jp:
                return 0, str(self.ds_desired)
            return 0, self.image
        if k[:3] == ["-n", dp.OKVIS_NS, "rollout"]:
            self.restarts += 1
            self._boot_okvis()
            return 0, "daemonset.apps/okvis2x restarted"
        if k[:2] == ["get", "pods"]:
            st = {"phase": self.pod["phase"], "containerStatuses": [
                {"name": "okvis2x", "restartCount": self.pod["restarts"], "ready": not self.pod["waiting"],
                 "state": {"waiting": {"reason": self.pod["waiting"]}} if self.pod["waiting"] else {"running": {}}}]}
            item = {"metadata": {"name": f"okvis2x-{self.restarts}", "namespace": dp.OKVIS_NS}, "status": st}
            return 0, json.dumps({"items": [item]})
        if "logs" in k:
            return 0, "\n".join(self.log)
        return 1, f"fake kubectl: unknown {k}"


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def snapshot(robot, **kw):
    return dp.take_snapshot(dp.Remote(robot), producer_yaml=PRODUCER_YAML, clock=lambda: 1.79e9, **kw)


def deployer(robot, snap=None, dry_run=False, **kw):
    c = Clock()
    snap = snap or snapshot(robot)
    applied = []

    def apply(size):
        applied.append(("apply", tuple(size)))
        robot.producer_size = tuple(size)

    def restore():
        applied.append(("restore",))
        robot.producer_size = (1280, 800)
    d = dp.Deployer(dp.Remote(robot, dry_run=dry_run), snap, DEPLOY_DIR, producer_apply=apply,
                    producer_restore=restore, clock=c, sleep=c.sleep, session="20261110-010203-P10", **kw)
    return d, applied


def mutating_calls(robot):
    return [a for a, _ in robot.calls if (a[:2] == ["sh", "-c"]) or "rollout" in a or a[-1:] == ["restart"]]


# ---------------------------------------------------------------- D1: Remote, robot.env, #930


def test_remote_dry_run_records_mutations_runs_reads():
    r = FakeRobot()
    rem = dp.Remote(r, dry_run=True)
    assert rem.read(dp.ROBOT_ENV) == ENV
    rem.write("/etc/x", "hello")
    assert "/etc/x" not in r.files and len(rem.planned) == 1
    assert "/etc/x" in rem.text() and "5 bytes" in rem.text()


@pytest.mark.parametrize("argv", [["k0s", "kubectl", "delete", "pod", "x", "--grace-period=0"],
                                  ["k0s", "kubectl", "delete", "pod", "x", "--grace-period", "0"],
                                  ["k0s", "kubectl", "delete", "pod", "x", "--force"]])
@pytest.mark.parametrize("dry", [False, True])
def test_forbidden_flags_refused_before_running(argv, dry):
    r = FakeRobot()
    with pytest.raises(dp.ForbiddenCommand):
        dp.Remote(r, dry_run=dry).call(argv, mutating=True)
    assert not r.calls


def test_set_env_keeps_everything_else():
    out = dp.set_env(ENV, "OKVIS_CALIBRATION", "/etc/okvis/new.yaml")
    assert out.splitlines()[0] == "# per-robot OKVIS values" and "OKVIS_DMA_CAMERAS=3,0" in out
    assert "OKVIS_CALIBRATION=/etc/okvis/new.yaml" in out and OLD_NAME not in out
    assert "OKVIS_MODE=slam  # keep" in out
    assert dp.set_env("A=1\n", "B", "2") == "A=1\nB=2\n"
    assert dp.parse_env(out)["OKVIS_CALIBRATION"] == "/etc/okvis/new.yaml"
    assert dp.parse_env('export X="a b"\n# Y=1\n') == {"X": "a b"}


def test_okvis_startup_checks_930():
    assert dp.okvis_startup_checks(NEW, (1280, 800)) == []
    p = dp.okvis_startup_checks(NEW, (960, 600))
    assert len(p) == 2 and all("960x600" in x for x in p)
    bad = _okvis_yaml(dist_type="kb4t", dist=(0.01, -0.02, 0.003, -0.001, 1e-4))
    assert any("exactly 6" in x for x in dp.okvis_startup_checks(bad, (1280, 800)))
    unk = _okvis_yaml(dist_type="fisheye624", dist=(0.0,) * 4)
    assert any("unknown distortion_type" in x for x in dp.okvis_startup_checks(unk, (1280, 800)))
    rt = _okvis_yaml()     # radialtangential, 4 coefficients
    assert dp.okvis_startup_checks(rt, (1280, 800)) == []


# ---------------------------------------------------------------- D2: snapshot, restart path, drift, dependents


def test_restart_path_daemonset():
    rp = dp.restart_path(dp.Remote(FakeRobot(ds_desired=1, okvis_sh=True)))
    assert rp.kind == "daemonset" and rp.argv[-3:] == ["rollout", "restart", "ds/okvis2x"]
    assert "--grace-period" not in " ".join(rp.argv)


def test_restart_path_okvis_sh_and_none():
    rp = dp.restart_path(dp.Remote(FakeRobot(ds_desired=None, okvis_sh=True)), origin="calibtool test")
    assert rp.kind == "okvis_sh" and rp.argv == ["env", "OKVIS_LAUNCH_ORIGIN=calibtool test", "bash", dp.OKVIS_SH,
                                                 "restart"]
    assert dp.restart_path(dp.Remote(FakeRobot(ds_desired=0))).kind == "none"


def test_snapshot_holds_what_rollback_needs():
    r = FakeRobot()
    s = snapshot(r)
    assert s.robot_env == ENV and s.calibration == f"/etc/okvis/{OLD_NAME}"
    assert s.calibration_host == f"{dp.HOST_CFG_DIR}/{OLD_NAME}" and s.config_text == OLD
    assert s.dma_cameras == "3,0" and s.producer_yaml == PRODUCER_YAML and s.head_camera_json == '{"cams": 1}'
    assert s.okvis_image == r.image and s.restart["kind"] == "daemonset"
    assert [d.path for d in s.dependents] == ["/etc/phantom/vla/rig.env"]
    assert not mutating_calls(r)


def test_snapshot_save_load(tmp_path):
    s = snapshot(FakeRobot())
    s.save(tmp_path)
    t = dp.DeploySnapshot.load(tmp_path)
    assert t == s


def test_drift_names_each_change():
    r = FakeRobot()
    s = snapshot(r)
    assert dp.drift(s.expected(), dp.Remote(r)) == []
    r.files[dp.ROBOT_ENV] = ENV + "OKVIS_EXTRA=1\n"
    r.files[f"{dp.HOST_CFG_DIR}/{OLD_NAME}"] = OLD + "# edited\n"
    r.image = "foundationbot/okvis2x:thor-other"
    d = dp.drift(s.expected(), dp.Remote(r))
    assert len(d) == 3 and any("robot.env" in x for x in d) and any("image" in x for x in d)


def test_find_dependents_reports_only():
    r = FakeRobot()
    deps = dp.find_dependents(dp.Remote(r), OLD_NAME, roots=("/etc/phantom",))
    assert [(x.path, x.line) for x in deps] == [("/etc/phantom/vla/rig.env", 1)]
    assert "RIG_YAML" in deps[0].text and not mutating_calls(r)


# ---------------------------------------------------------------- D3: the transaction


def plan(name=NEW_NAME, text=NEW, size=(1280, 800)):
    return dp.DeployPlan(name, text, size)


def test_deploy_order_and_verify():
    r = FakeRobot()
    d, applied = deployer(r)
    res = d.deploy(plan())
    assert res.ok, res.as_dict()
    steps = [s["step"] for s in res.steps]
    assert steps == ["drift", "checks", "backup", "rollback_script", "config", "robot_env", "producer", "restart",
                     "verify"]
    assert r.files[f"{dp.HOST_CFG_DIR}/{NEW_NAME}"] == NEW
    assert dp.parse_env(r.files[dp.ROBOT_ENV])["OKVIS_CALIBRATION"] == f"/etc/okvis/{NEW_NAME}"
    assert r.files[f"{DEPLOY_DIR}/robot.env.before"] == ENV
    assert r.files[f"{DEPLOY_DIR}/{OLD_NAME}.before"] == OLD
    assert "rollback.sh" in res.rollback_script and r.files[res.rollback_script].startswith("#!/bin/bash")
    assert applied == [("apply", (1280, 800))]
    assert res.verify["log_sha"] and res.verify["kb4t_line"] and not res.crash_loop
    assert res.sha12 == sha12(NEW)
    # the rollback script is written before the first change
    order = [a[4] for a, _ in r.calls if a[:2] == ["sh", "-c"]]
    assert order.index(res.rollback_script) < order.index(f"{dp.HOST_CFG_DIR}/{NEW_NAME}")
    assert order.index(res.rollback_script) < order.index(dp.ROBOT_ENV)


def test_failed_930_check_stops_before_any_write():
    r = FakeRobot()
    d, applied = deployer(r)
    with pytest.raises(dp.DeployRefused, match="960x600"):
        d.deploy(plan(size=(960, 600)))
    assert not mutating_calls(r) and not applied


def test_consumer_check_failure_refuses():
    r = FakeRobot()
    d, applied = deployer(r, checks=[lambda: (False, "box_pick on this bake cannot read kb4t (needs 58e96414)")])
    with pytest.raises(dp.DeployRefused, match="box_pick"):
        d.deploy(plan())
    assert not mutating_calls(r)


def test_drift_before_deploy_refuses_nothing_written():
    r = FakeRobot()
    s = snapshot(r)
    r.files[dp.ROBOT_ENV] = ENV.replace("3,0", "2,0")
    d, applied = deployer(r, snap=s)
    with pytest.raises(dp.DeployDrift, match="robot.env"):
        d.deploy(plan())
    assert not mutating_calls(r)


def test_existing_different_file_of_the_same_name_is_not_overwritten():
    r = FakeRobot()
    r.files[f"{dp.HOST_CFG_DIR}/{NEW_NAME}"] = "someone else's file"
    d, _ = deployer(r)
    with pytest.raises(dp.DeployRefused, match="already exists"):
        d.deploy(plan())
    assert r.files[f"{dp.HOST_CFG_DIR}/{NEW_NAME}"] == "someone else's file"


def test_crash_loop_is_reported_not_hidden():
    r2 = FakeRobot()
    d2, _ = deployer(r2)
    d2.producer_apply = lambda size: setattr(r2, "producer_size", (960, 600))     # someone else's producer size
    res2 = d2.deploy(plan())
    assert not res2.ok and res2.crash_loop


def test_dry_run_prints_exact_commands_runs_nothing():
    r = FakeRobot()
    d, applied = deployer(r, dry_run=True)
    res = d.deploy(plan())
    assert res.dry_run and not res.ok and not applied
    assert not mutating_calls(r)
    txt = d.remote.text()
    assert f"{dp.HOST_CFG_DIR}/{NEW_NAME}" in txt and dp.ROBOT_ENV in txt
    assert "rollout restart ds/okvis2x" in txt and "--grace-period" not in txt
    assert "scale deploy producer --replicas=0" in txt


def test_rollback_restores_and_verifies():
    r = FakeRobot()
    d, applied = deployer(r)
    d.deploy(plan())
    rb = d.rollback("new worse than old on depth")
    assert rb.ok, rb.as_dict()
    assert r.files[dp.ROBOT_ENV] == ENV and applied[-1] == ("restore",)
    assert rb.verify["log_sha"] and sha12(OLD) in "\n".join(r.log)


def test_rollback_with_drift_asks():
    r = FakeRobot()
    d, applied = deployer(r)
    d.deploy(plan())
    r.files[dp.ROBOT_ENV] += "OKVIS_SOMEONE=1\n"
    with pytest.raises(dp.RollbackDrift):
        d.rollback("test")
    assert ("restore",) not in applied


def test_failed_rollback_is_loud_and_not_retried():
    r = FakeRobot()
    d, applied = deployer(r)
    d.deploy(plan())
    r.fail_write.add(dp.ROBOT_ENV)
    n_before = len([a for a, _ in r.calls if a[:2] == ["sh", "-c"] and a[4] == dp.ROBOT_ENV])
    with pytest.raises(dp.RollbackFailed) as e:
        d.rollback("test")
    assert "ROLLBACK FAILED" in str(e.value) and "rollback.sh" in str(e.value)
    n_after = len([a for a, _ in r.calls if a[:2] == ["sh", "-c"] and a[4] == dp.ROBOT_ENV])
    assert n_after - n_before == 1


def test_rollback_note_when_target_failed_today():
    r = FakeRobot()
    d, _ = deployer(r)
    d.deploy(plan())
    rb = d.rollback("crash loop", target_failed_today=True)
    assert rb.ok and any("FAILED on today's data" in n for n in rb.notes)


def test_rollback_script_contents():
    s = snapshot(FakeRobot())
    txt = dp.rollback_script(s, DEPLOY_DIR, NEW_NAME, producer_ns="dma-video", producer_deploy="producer")
    assert txt.startswith("#!/bin/bash") and "set -euo pipefail" in txt
    assert f'cp -a {DEPLOY_DIR}/robot.env.before {dp.ROBOT_ENV}' in txt
    assert "scale deploy producer --replicas=0" in txt and "replace -f" in txt
    assert txt.index("--replicas=0") < txt.index("replace -f") < txt.index("--replicas=1")
    assert "rollout restart ds/okvis2x" in txt and "--grace-period" not in txt and "--force" not in txt
    assert "ROLLBACK" in txt


def test_rollback_script_okvis_sh_path():
    s = snapshot(FakeRobot(ds_desired=None, okvis_sh=True))
    txt = dp.rollback_script(s, DEPLOY_DIR, NEW_NAME)
    assert f"bash {dp.OKVIS_SH} restart" in txt


def test_producer_cycle_argv_no_overlap():
    cmds = dp.producer_cycle_argv("dma-video", "producer", replicas=1, between=[["patch", "x"]])
    flat = [" ".join(c) for c in cmds]
    assert "scale deploy producer --replicas=0" in flat[0]
    assert any("wait" in f or "get pods" in f for f in flat[1:3])
    assert flat[-1].endswith("--replicas=1")


# ---------------------------------------------------------------- D4: deploy size, check grid


def test_size_options():
    o = dp.size_options((960, 600, 25))
    assert [(x.size, x.recommended) for x in o] == [((960, 600), True), ((1280, 800), False)]
    assert o[0].scale == pytest.approx(0.75) and "960x600" in o[0].label
    assert [x.size for x in dp.size_options((1280, 800, 25))] == [(1280, 800)]


def test_config_for_size_passes_930_at_its_size():
    t = dp.config_for_size(NEW, (960, 600))
    assert dp.okvis_startup_checks(t, (960, 600)) == []
    assert "image_dimension: [960, 600]" in t
    assert dp.config_for_size(NEW, (1280, 800)) == NEW


def _scores(err_new, err_live, n_holds=15, per=2, seed=1):
    rng = np.random.default_rng(seed)
    n = n_holds * per
    src = np.arange(n)
    hold = np.repeat(np.arange(n_holds), per)
    pl = np.array(["check"] * n)
    z = np.full(n, 1.0)

    def sc(e, dy):
        return decide.Scores(src, hold, pl, e + rng.normal(0, 0.05, n), z, np.repeat(src, 20),
                             dy + rng.normal(0, 0.05, n * 20))
    return sc(err_new, 0.2), sc(err_live, 0.8 if err_live > err_new else 0.2)


def test_check_verdict_deploy_and_keep():
    new, live = _scores(0.2, 0.9)
    v = dp.check_verdict(new, live)
    assert v.decision == "DEPLOY", v.reasons
    new, live = _scores(0.3, 0.3)
    v = dp.check_verdict(new, live)
    assert v.decision == "KEEP_LIVE" and any("not shown better" in r for r in v.reasons)


def test_dry_run_text_is_shell():
    r = FakeRobot()
    d, _ = deployer(r, dry_run=True)
    d.deploy(plan())
    for line in d.remote.text().splitlines():
        if line and not line.startswith("#"):
            shlex.split(line)
    assert re.search(r"<<< \d+ bytes", d.remote.text())


def test_shell_runner_write_and_read_for_real(tmp_path):
    """the atomic write snippet through a real shell (on the test box, in tmp_path only)"""
    rem = dp.Remote(dp.shell_runner())
    p = tmp_path / "robot.env"
    rem.write(str(p), "A=1\nB=two words\n")
    assert p.read_text() == "A=1\nB=two words\n" and oct(p.stat().st_mode & 0o777) == "0o644"
    assert rem.read(str(p)) == "A=1\nB=two words\n" and rem.exists(str(p))
    assert rem.read(str(tmp_path / "nope")) is None and not rem.exists(str(tmp_path / "nope"))
    assert not list(tmp_path.glob("*.calibtool-tmp"))
