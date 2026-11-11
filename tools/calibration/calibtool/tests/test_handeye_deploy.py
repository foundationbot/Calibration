"""calibtool.handeye_deploy (hand-eye H4): the deploy transaction, its checks and rollback, the T_BS convention, the
post-check rule and the head tool's "stale" line -- against a FAKE robot only (test_deploy.FakeRobot, extended with the
URDF reader's log lines, the host name, the machine-id and a directory listing). Nothing here touches a robot.

The fit is synthetic: captures of both arms seen through a robot with known URDF errors and a known camera offset in
the head (as armcal's test_urdffit), written as readb outputs, fitted by armcal.urdffit.run."""
import hashlib
import importlib.util
import json
import re
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
yaml = pytest.importorskip("yaml")

from armcal import REPO_URDF, urdffit  # noqa: E402
from armcal.urdf import Urdf  # noqa: E402
from calibtool import deploy as dp, handeye_deploy as he, imu, scaling  # noqa: E402
from rigmath import rot_axis, transform  # noqa: E402

HERE = Path(__file__).resolve().parent
CAL = HERE.parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TD = _load("_t_deploy_he", HERE / "test_deploy.py")

# ---------------------------------------------------------------- the robot's files

URDF_C = "/etc/okvis/mk1_1.urdf"
URDF_H = "/etc/phantomos/okvis/mk1_1.urdf"
BASE_TEXT = REPO_URDF.read_text()
BASE = Urdf(BASE_TEXT, URDF_H)
T_BS = np.array([[3.26794897e-07, 0.84339145, 0.537299603, 0.00954301],
                 [-1.0, 2.75616021e-07, 1.75586768e-07, -0.0389529],
                 [0.0, -0.537299603, 0.84339145, 0.08296029], [0.0, 0.0, 0.0, 1.0]])
T_SC_DEPTH = np.array([[0.9998607039451599, -0.01611066423356533, 0.004365500062704086, -0.0536164753139019],
                       [-0.0022230164613574743, 0.13068029284477234, 0.9914220571517944, 0.1373654454946518],
                       [-0.0165429525077343, -0.9912936687469482, 0.13062627613544464, 0.025456219911575317],
                       [0.0, 0.0, 0.0, 1.0]])
T_BC = T_BS @ T_SC_DEPTH
FIT_NAME = "phantom10_stereo_kb4t_20261110_1280_s2m2Lprofiles.yaml"
LIVE_NAME = "phantom10_stereo_kb4t_20261110_640.yaml"
MID = "60776b088362" + "0" * 20          # phantom10's machine-id prefix (robots.yaml)


def _tbs_text(T):
    rows = [", ".join(repr(float(v)) for v in T[i]) for i in range(4)]
    return "        [" + ",\n         ".join(rows) + "]"


def config_1280(T=T_BS):
    t = (CAL / "config" / "okvis-template-phantom12.yaml").read_text()
    t = t[t.index("%YAML:1.0"):]
    t = re.sub(r"(\n\s*T_BS:\s*\n)\s*\[[^\]]*\]", lambda m: m.group(1) + _tbs_text(T), t, count=1)
    return t.replace('urdf: "/etc/okvis/20250206_phantom_mk1.urdf"', f'urdf: "{URDF_C}"')


FIT_CFG = config_1280()
LIVE_CFG = scaling.scale_config(FIT_CFG, 0.5)          # someone switched the robot to 640 after the captures
ENV = f"# per-robot OKVIS values\nOKVIS_DMA_CAMERAS=3,0\nOKVIS_CALIBRATION=/etc/okvis/{LIVE_NAME}\n"


def sha12(t):
    return hashlib.sha256(t.encode()).hexdigest()[:12]


class Robot(TD.FakeRobot):
    """test_deploy's fake robot with phantom10's hand-eye files; OKVIS logs the URDF reader's line for the URDF its
    config names (`Failed to load URDF` when the file is missing or urdf_fails), and crash-loops on crash_on configs"""

    def __init__(self, ds_desired=1, okvis_sh=False, crash_on=(), urdf_fails=False, env=ENV, live=LIVE_CFG,
                 urdf_text=BASE_TEXT, machine_id=MID, hostname="phantom10"):
        self.crash_on, self.urdf_fails = set(crash_on), urdf_fails
        super().__init__(ds_desired=ds_desired, okvis_sh=okvis_sh)
        self.files = {dp.ROBOT_ENV: env, f"{dp.HOST_CFG_DIR}/{LIVE_NAME}": live,
                      f"{dp.HOST_CFG_DIR}/{FIT_NAME}": FIT_CFG, URDF_H: urdf_text, dp.HEAD_CAMERA_JSON: '{"cams": 1}',
                      "/etc/hostname": hostname + "\n", "/etc/machine-id": machine_id + "\n"}
        if okvis_sh:
            self.files[dp.OKVIS_SH] = "#!/bin/bash\n"
        self._boot_okvis()

    def _boot_okvis(self):
        if not hasattr(self, "crash_on"):
            return super()._boot_okvis()
        env = dp.parse_env(self.files.get(dp.ROBOT_ENV, ""))
        cal = env.get("OKVIS_CALIBRATION", "")
        text = self.files.get(dp.host_path(cal))
        if text is None or Path(cal).name in self.crash_on or dp.okvis_startup_checks(text, he.image_size(text)):
            self.pod = {"phase": "Running", "restarts": 3, "waiting": "CrashLoopBackOff"}
            self.log = [f"[entrypoint] robot.env: {cal}", "F1111 crash"]
            return
        self.pod = {"phase": "Running", "restarts": 0, "waiting": None}
        self.log = [f"[entrypoint] config: {Path(cal).name} explicit path {cal} ({sha12(text)})"]
        if "kb4t" in text:                                     # deploy.py's reader check keys on the word
            self.log.append("I1111 KB4T (VIO: equidistant k1..k4) pinhole camera 0 with T_SC=...")
        u = env.get("OKVIS_URDF") or he.dma_urdf(text)
        if u and dp.host_path(u) in self.files and not self.urdf_fails:
            self.log.append(f"I1111 UrdfTfModel.cpp:129] [Urdf] loaded '{u}': 40 links, 39 joints, root 'Pelvis'.")
        elif u:
            self.log.append(f"W1111 okvis2x_app_dma_live.cpp:620] Failed to load URDF '{u}'; TF tree AND ...")

    def __call__(self, argv, input=None, timeout=None):
        a = list(argv)
        if a[:2] == ["ls", "-1"]:
            self.calls.append((a, input))
            d = a[2].rstrip("/") + "/"
            return 0, "\n".join(sorted(p[len(d):] for p in self.files if p.startswith(d) and "/" not in p[len(d):]))
        return super().__call__(argv, input, timeout)


def mutations(robot):
    return [a for a, _ in robot.calls if a[:2] == ["sh", "-c"] or "rollout" in a or a[-1:] == ["restart"]]


# ---------------------------------------------------------------- the synthetic fit

TRUE = {"NeckYaw": {"dxyz": [-8.0, 12.0, -20.0]}, "RightShoulderRoll": {"post_axis_deg": -1.2},
        "LeftShoulderRoll": {"post_axis_deg": -0.9}, "HEAD": {"dxyz": [-15.0, 4.0, -6.0]}}
NECKS = [(-0.2, 0.17), (0.0, 0.38), (0.2, 0.45), (0.1, 0.25)]
Q0 = {"Right": {"RightShoulderPitch": -1.149872, "RightShoulderRoll": -0.347025, "RightShoulderYaw": 0.598868,
                "RightElbowPitch": -1.167933, "RightWristYaw": 1.000419, "RightWristRoll": 0.356826,
                "RightGripperYaw": -0.548542},
      "Left": {"LeftShoulderPitch": -1.131819, "LeftShoulderRoll": 0.198874, "LeftShoulderYaw": -0.542669,
               "LeftElbowPitch": -1.031988, "LeftWristYaw": -0.726068, "LeftWristRoll": -0.117381,
               "LeftGripperYaw": 1.117396}}


def records(side, change=TRUE, poses=4, T_BC_rec=T_BC, noise_mm=0.0, seed=0, base=BASE, prefix="p"):
    """a readb output of `side`: captures seen through the true robot (change), stored as readb would (T_TC, E, wrist
    under the base URDF; the camera-in-head the burst carried, T_BC_rec)"""
    rng = np.random.default_rng(seed)
    ch = dict(change)
    head = ch.pop("HEAD", None)
    true = base.changed(ch)
    D = np.eye(4)
    if head is not None:
        D[:3, 3] = 1e-3 * np.asarray(head["dxyz"])
    res = []
    for k in range(poses):
        q = {n: v + rng.normal(0, 0.15) for n, v in Q0[side].items()}
        for i, (yaw, pitch) in enumerate(NECKS):
            a = {"NeckYaw": yaw, "NeckPitch": pitch}
            T_true = true.chain("Torso", "Head").T(a) @ D @ T_BC_rec
            w_true = true.chain("Torso", side + "WristRollLink").T(q)[:3, 3] + 1e-3 * rng.normal(0, noise_mm, 3)
            m = np.linalg.inv(T_true) @ np.r_[w_true, 1.0]
            T_TC = base.chain("Torso", "Head").T(a) @ T_BC_rec
            w = base.chain("Torso", side + "WristRollLink").T(q)[:3, 3]
            E = transform(np.eye(3), m[:3] - (np.linalg.inv(T_TC) @ np.r_[w, 1.0])[:3])
            off = 1e3 * ((T_TC @ m)[:3] - w)
            res.append(dict(name=f"{k + 1:02d}-{prefix}{k:02d}-{i}", side=side, off=off.tolist(), clean=True,
                            E=E.tolist(), T_TC=T_TC.tolist(), T_BC=T_BC_rec.tolist(), q=q, yaw=yaw, pitch=pitch,
                            wrist=w.tolist()))
    return {"urdf": URDF_H, "urdf_md5": hashlib.md5(BASE_TEXT.encode()).hexdigest(), "side": side, "res": res}


@pytest.fixture(scope="module")
def fit_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("fit")
    (d / "readb-right.json").write_text(json.dumps(records("Right", seed=1)))
    (d / "readb-left.json").write_text(json.dumps(records("Left", seed=2)))
    urdf = d / "base.urdf"
    urdf.write_text(BASE_TEXT)
    rep, _ = urdffit.run(d / "readb-right.json", d / "readb-left.json", urdf, models=("M12+head", "M12"))
    rep["chosen"] = "M12+head"
    (d / "fit.json").write_text(json.dumps(rep, indent=1))
    geo = he.FitGeometry("P10", "p10-test", f"/etc/phantomos/okvis/{FIT_NAME}",
                         hashlib.sha256(FIT_CFG.encode()).hexdigest(), T_BS, T_SC_DEPTH)
    (d / "geometry.json").write_text(json.dumps(geo.as_dict()))
    return d


@pytest.fixture
def fit(fit_dir):
    return he.HandeyeFit.load(fit_dir)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


DEPLOY_DIR = "/data/calib/handeye-20261111/deploy"


def deployer(robot, fit, dry_run=False, mode="head", model=None, snap=None, **kw):
    remote = he.dry_run_remote(robot) if dry_run else dp.Remote(robot)
    snap = snap or he.take_snapshot(remote, fit, clock=lambda: 1.79e9)
    c = Clock()
    return he.HandeyeDeployer(remote, fit, snap, DEPLOY_DIR, mode, model, session="h4-test", date="20261111",
                              clock=c, sleep=c.sleep, up_timeout_s=20, **kw)


NEW_CFG = f"{dp.HOST_CFG_DIR}/phantom10_stereo_kb4t_20261110_640_handeye20261111.yaml"
NEW_URDF = f"{dp.HOST_CFG_DIR}/phantom10_mk1_handeye_20261111.urdf"


# ---------------------------------------------------------------- the config surgery


def test_apply_head_moves_only_the_three_translation_numbers():
    head = np.array([-21.354813, -1.648261, -5.559096])
    new = he.apply_head(LIVE_CFG, head)
    T0, T1 = he.t_bs(LIVE_CFG), he.t_bs(new)
    D = np.eye(4)
    D[:3, 3] = 1e-3 * head
    assert np.allclose(T1, D @ T0, atol=1e-15, rtol=0) and np.array_equal(T1[:3, :3], T0[:3, :3])
    a, b = LIVE_CFG.splitlines(), new.splitlines()
    changed = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    assert len(a) == len(b) and len(changed) == 3          # one per T_BS row; all else byte-identical
    assert he.t_sc(new)[0].tolist() == he.t_sc(LIVE_CFG)[0].tolist()


def test_set_urdf_keeps_quotes_and_comment_and_only_the_dma_key():
    t = "%YAML:1.0\nfoo:\n    urdf: x\ndma:\n    imu_name: \"N\"\n    urdf: \"/etc/okvis/a.urdf\"  # c\n    body: H\n"
    out = he.set_urdf(t, "/etc/okvis/b.urdf")
    assert he.dma_urdf(out) == "/etc/okvis/b.urdf"
    assert '    urdf: "/etc/okvis/b.urdf"  # c' in out and "    urdf: x" in out


def test_the_template_config_reads():
    assert he.dma_urdf(LIVE_CFG) == URDF_C and he.image_size(LIVE_CFG) == (640, 400)
    assert he.t_bs(LIVE_CFG).tolist() == T_BS.tolist()


# ---------------------------------------------------------------- the T_BS convention: the proof


def test_new_urdf_and_new_tbs_reproduce_the_fits_prediction(fit):
    """the camera in the torso composed from the WRITTEN files (the new URDF's neck chain, the new config's T_BS, the
    depth camera's T_SC) predicts every fitted capture's wrist exactly as the fit does; the other order (HEAD in the
    IMU frame, T_BS D) does not"""
    robot = Robot()
    hd = deployer(robot, fit, dry_run=True)
    hd.deploy()
    b = hd.build
    new_urdf, T_BS_new = Urdf(b.urdf_text), he.t_bs(b.config_text)
    model, change, head = he.choose(fit, "head")
    ch = dict(change, HEAD={"dxyz": head})
    worst, worst_wrong = 0.0, 0.0
    D_wrong = np.eye(4)
    D_wrong[:3, 3] = 1e-3 * head
    for side in ("Right", "Left"):
        obs = fit.selected(side)
        assert len(obs) == 16
        want = urdffit.off_change(obs, BASE, ch)
        for o, w in zip(obs, want):
            T_TC = new_urdf.chain("Torso", "Head").T({"NeckYaw": o.yaw, "NeckPitch": o.pitch}) @ T_BS_new @ T_SC_DEPTH
            got = 1e3 * ((T_TC @ o.m)[:3] - new_urdf.chain("Torso", side + "WristRollLink").T(o.q)[:3, 3])
            worst = max(worst, np.abs(got - w).max())
            T_wrong = new_urdf.chain("Torso", "Head").T({"NeckYaw": o.yaw, "NeckPitch": o.pitch}) @ T_BS @ D_wrong \
                @ T_SC_DEPTH
            bad = 1e3 * ((T_wrong @ o.m)[:3] - new_urdf.chain("Torso", side + "WristRollLink").T(o.q)[:3, 3])
            worst_wrong = max(worst_wrong, np.abs(bad - w).max())
    assert worst < 1e-6 and b.predict_mm < 1e-6
    assert worst_wrong > 5.0                                  # the convention matters at the mm level
    # and the fit recovered the synthetic robot's true terms (noise-free captures)
    assert np.allclose(head, TRUE["HEAD"]["dxyz"], atol=0.05)


def test_head_is_in_the_head_frame_and_moves_the_camera_by_exactly_head(fit):
    robot = Robot()
    hd = deployer(robot, fit, dry_run=True)
    hd.deploy()
    _, _, head = he.choose(fit, "head")
    T_new = he.t_bs(hd.build.config_text)
    moved = (T_new @ T_SC_DEPTH)[:3, 3] - (T_BS @ T_SC_DEPTH)[:3, 3]
    assert np.allclose(1e3 * moved, head, atol=1e-9)


# ---------------------------------------------------------------- the transaction


def test_dry_run_records_every_command_and_sends_none(fit):
    robot = Robot()
    before = dict(robot.files)
    remote = he.dry_run_remote(robot)
    snap = he.take_snapshot(remote, fit)
    c = Clock()
    hd = he.HandeyeDeployer(remote, fit, snap, DEPLOY_DIR, session="h4", date="20261111", clock=c, sleep=c.sleep)
    res = hd.deploy()
    assert res.dry_run and not res.ok and robot.files == before and mutations(robot) == []
    txt = remote.text()
    for want in (NEW_URDF, NEW_CFG, dp.ROBOT_ENV, f"{DEPLOY_DIR}/rollback.sh", "rollout restart ds/okvis2x"):
        assert want in txt, want
    assert "grace-period" not in txt and "--force" not in txt
    assert txt.index(NEW_URDF) < txt.index(NEW_CFG) < txt.index("rollout restart")


def test_deploy_switches_okvis_and_verifies_the_load(fit, tmp_path):
    robot = Robot()
    before = dict(robot.files)
    res = deployer(robot, fit, publish_dir=tmp_path / "pub").deploy()
    assert res.ok, res.detail
    assert dp.parse_env(robot.files[dp.ROBOT_ENV])["OKVIS_CALIBRATION"] == dp.container_path(Path(NEW_CFG).name)
    new = robot.files[NEW_CFG]
    assert he.dma_urdf(new) == dp.container_path(Path(NEW_URDF).name)
    assert he.marker(new)["fit"] == he.HandeyeFit.load(fit.path).fit_id and he.marker(new)["mode"] == "head"
    assert robot.files[NEW_URDF].startswith("<?xml") and "md5 " + hashlib.md5(BASE_TEXT.encode()).hexdigest() in \
        robot.files[NEW_URDF]
    assert f"T_BS sha {he.geom_sha(he.t_bs(new))}" in robot.files[NEW_URDF]
    for p in (f"{dp.HOST_CFG_DIR}/{LIVE_NAME}", URDF_H, f"{dp.HOST_CFG_DIR}/{FIT_NAME}"):
        assert robot.files[p] == before[p]                    # the old files are never changed
    assert robot.files[dp.ROBOT_ENV] != before[dp.ROBOT_ENV]
    assert robot.files[dp.ROBOT_ENV].replace("_handeye20261111", "") == before[dp.ROBOT_ENV]
    assert res.deploy["verify"]["expect"] is True and res.deploy["verify"]["fail"] == []
    assert robot.files[f"{DEPLOY_DIR}/robot.env.before"] == before[dp.ROBOT_ENV]
    text = "\n".join(res.consumers)
    assert "OKVIS: switched" in text and "burst point cloud: NOT switched" in text
    assert "frames.py must select" in text and "cuRobo" in text and "not updated" in text
    assert (tmp_path / "pub" / Path(NEW_URDF).name).read_text() == robot.files[NEW_URDF]
    assert "rollback" in res.report()


def test_head_is_never_applied_twice(fit):
    robot = Robot()
    assert deployer(robot, fit).deploy().ok
    with pytest.raises(dp.DeployRefused) as e:
        deployer(robot, fit).deploy()
    msg = str(e.value)
    assert "already carries this fit" in msg and "never applied twice" in msg
    assert "hand-eye stale for this camera geometry" in msg and "not the one the captures were matched" in msg


def test_refuses_when_the_live_tbs_is_not_the_fits(fit):
    T = T_BS.copy()
    T[0, 3] += 0.002                                           # a head calibration moved T_BS since the captures
    robot = Robot(live=scaling.scale_config(config_1280(T), 0.5))
    with pytest.raises(dp.DeployRefused, match="hand-eye stale for this camera geometry: the live config's T_BS"):
        deployer(robot, fit).deploy()
    assert mutations(robot) == []


def test_refuses_when_the_live_camera_tsc_is_not_the_fits(fit):
    live = imu.apply(LIVE_CFG, rot_axis(np.array([1.0, 0, 0]), np.radians(0.3)))
    robot = Robot(live=live)
    with pytest.raises(dp.DeployRefused, match="camera T_SC"):
        deployer(robot, fit).deploy()
    assert mutations(robot) == []


def test_refuses_when_the_fit_time_config_is_gone_or_changed(fit):
    robot = Robot()
    robot.files[f"{dp.HOST_CFG_DIR}/{FIT_NAME}"] = FIT_CFG + "# edited\n"
    with pytest.raises(dp.DeployRefused, match="changed since the captures"):
        deployer(robot, fit).deploy()
    del robot.files[f"{dp.HOST_CFG_DIR}/{FIT_NAME}"]
    with pytest.raises(dp.DeployRefused, match="not on the robot any more"):
        deployer(robot, fit).deploy()


def test_the_live_config_switched_since_the_fit_is_fine_when_its_geometry_is_the_same(fit):
    """phantom10, 11 Nov: robot.env went from the 1280 config to the 640 one between the fit and the deploy"""
    robot = Robot()
    res = deployer(robot, fit).deploy()
    assert res.ok and he.image_size(robot.files[NEW_CFG]) == (640, 400)
    assert he.marker(robot.files[NEW_CFG])["from"].startswith(LIVE_NAME)


def test_robot_env_changed_under_us_refuses_and_writes_nothing(fit):
    robot = Robot()
    hd = deployer(robot, fit)
    robot.files[dp.ROBOT_ENV] = ENV.replace(LIVE_NAME, FIT_NAME)   # someone switched it after the snapshot
    with pytest.raises(dp.DeployDrift, match="robot.env"):
        hd.deploy()
    assert mutations(robot) == []


def test_the_urdf_changed_under_us_refuses(fit):
    robot = Robot()
    hd = deployer(robot, fit)
    robot.files[URDF_H] = BASE_TEXT + "<!-- edited -->\n"
    with pytest.raises(dp.DeployDrift, match="the URDF OKVIS reads"):
        hd.deploy()


def test_rollback_restores_byte_identical_files(fit):
    robot = Robot()
    before = dict(robot.files)
    hd = deployer(robot, fit)
    assert hd.deploy().ok
    rb = hd.rollback()
    assert rb.ok
    for p, t in before.items():
        assert robot.files[p] == t, p
    assert f"({sha12(LIVE_CFG)})" in "\n".join(robot.log) and f"[Urdf] loaded '{URDF_C}'" in "\n".join(robot.log)
    sh = robot.files[f"{DEPLOY_DIR}/rollback.sh"]
    assert "robot.env.before" in sh and "rollout restart ds/okvis2x" in sh and "grace-period" not in sh


def test_a_crash_loop_rolls_back_by_itself(fit):
    robot = Robot(crash_on={Path(NEW_CFG).name})
    before = dict(robot.files)
    res = deployer(robot, fit).deploy()
    assert not res.ok and res.rollback and res.rollback["ok"]
    assert "crash_loop" in res.detail and "rolled back automatically" in res.detail
    assert robot.files[dp.ROBOT_ENV] == before[dp.ROBOT_ENV] and robot.pod["waiting"] is None


def test_a_failed_urdf_load_rolls_back_by_itself(fit):
    robot = Robot(urdf_fails=True)
    res = deployer(robot, fit).deploy()
    assert not res.ok and res.rollback and res.rollback["ok"] is not None
    assert "Failed to load URDF" in res.detail
    assert robot.files[dp.ROBOT_ENV] == ENV


def test_model_only_keeps_tbs_byte_for_byte_and_uses_m12(fit):
    robot = Robot()
    hd = deployer(robot, fit, mode="model_only")
    res = hd.deploy()
    assert res.ok
    new = robot.files[NEW_CFG]
    assert re.search(r"T_BS:\s*\[[^\]]*\]", new).group(0) == re.search(r"T_BS:\s*\[[^\]]*\]", LIVE_CFG).group(0)
    mk = he.marker(new)
    assert mk["mode"] == "model_only" and mk["model"] == "M12" and mk["head_mm"] is None
    assert hd.build.predict_mm < 1e-6
    with pytest.raises(ValueError, match="fitted together with the head term"):
        he.choose(fit, "model_only", "M12+head")


def test_refuses_another_robot_and_an_unknown_machine_id(fit):
    with pytest.raises(dp.DeployRefused, match="robot identity"):
        deployer(Robot(machine_id="9851aca02a7d" + "0" * 20, hostname="phantom12"), fit).deploy()
    with pytest.raises(dp.DeployRefused, match="robot identity"):
        deployer(Robot(machine_id="ffffffffffff" + "0" * 20), fit).deploy()


def test_refuses_a_urdf_that_is_not_the_fits_base(fit):
    robot = Robot(urdf_text=BASE_TEXT.replace("<robot", "<!-- another file -->\n<robot", 1))
    with pytest.raises(dp.DeployRefused, match="not the one the captures were matched with"):
        deployer(robot, fit).deploy()


def test_refuses_okvis_urdf_in_robot_env(fit):
    robot = Robot(env=ENV + f"OKVIS_URDF={URDF_C}\n")
    with pytest.raises(dp.DeployRefused, match="OKVIS_URDF"):
        deployer(robot, fit).deploy()


def test_okvis_sh_robot_restarts_through_okvis_sh(fit):
    """phantom10 on 11 Nov: a bare okvis/okvis2x pod, the DaemonSet at 0 pods: okvis.sh restart"""
    robot = Robot(ds_desired=0, okvis_sh=True)
    res = deployer(robot, fit).deploy()
    assert res.ok
    assert any(a[-2:] == [dp.OKVIS_SH, "restart"] for a, _ in robot.calls)
    assert not any("rollout" in a for a, _ in robot.calls)


def test_a_second_deploy_the_same_day_gets_a_free_name(fit):
    robot = Robot()
    robot.files[NEW_URDF] = "someone else's file"
    res = deployer(robot, fit).deploy()
    assert res.ok and res.build["urdf_name"] == "phantom10_mk1_handeye_20261111b.urdf"
    assert robot.files[NEW_URDF] == "someone else's file"


def test_a_huge_head_term_is_refused(fit):
    rep = json.loads(json.dumps(fit.report))
    rep["models"]["M12+head"]["change"]["HEAD"]["dxyz"] = [-213.5, 0.0, 0.0]
    with pytest.raises(ValueError, match="units or sign slip"):
        he.choose(he.HandeyeFit(fit.path, rep, fit.geometry, fit.records, fit.sha12), "head")


def test_a_wrong_geometry_json_is_caught(fit):
    g = he.FitGeometry(**{**fit.geometry.__dict__, "T_SC_depth": np.eye(4)})
    bad = he.HandeyeFit(fit.path, fit.report, g, fit.records, fit.sha12)
    assert any("geometry.json" in p for p in bad.problems())


# ---------------------------------------------------------------- the head tool's coupling


def test_stale_line_after_a_head_deploy_that_changes_tsc_or_tbs(fit):
    robot = Robot()
    assert deployer(robot, fit).deploy().ok
    live = robot.files[NEW_CFG]
    assert he.stale_line(live, live) == ""
    rot = imu.apply(live, rot_axis(np.array([0, 0, 1.0]), np.radians(0.2)))      # the head tool's IMU rotation
    line = he.stale_line(rot, live)
    assert line.startswith("hand-eye STALE") and "camera T_SC changed" in line
    fresh = FIT_CFG                                                               # a step-70 config: no marker
    assert "urdf: is /etc/okvis/mk1_1.urdf" in he.stale_line(fresh, live)
    assert he.stale_line(FIT_CFG, LIVE_CFG) == ""                                 # no hand-eye deployed


def test_the_head_tools_deploy_reports_the_stale_line(fit):
    robot = Robot()
    assert deployer(robot, fit).deploy().ok
    live = robot.files[NEW_CFG]
    snap = dp.take_snapshot(dp.Remote(robot), producer_yaml=None, clock=lambda: 1.79e9)
    c = Clock()
    d = dp.Deployer(dp.Remote(robot), snap, "/data/calib/head/deploy", clock=c, sleep=c.sleep, up_timeout_s=10)
    res = d.deploy(dp.DeployPlan("head_new.yaml", imu.apply(live, rot_axis(np.array([1.0, 0, 0]), np.radians(0.3))),
                                 (640, 400)))
    assert res.handeye.startswith("hand-eye STALE")
    assert any(s.get("handeye") for s in res.steps if s["step"] == "checks")


# ---------------------------------------------------------------- the post-check rule (pure)


def test_postcheck_passes_when_paired_better_and_within_the_gate():
    rng = np.random.default_rng(0)
    old = 40 + 5 * rng.standard_normal(12)
    new = 4 + rng.random(12)
    v = he.postcheck_verdict(old, new, ["Right"] * 6 + ["Left"] * 6)
    assert v.decision == "PASS" and v.passed and not v.rollback


def test_postcheck_one_bad_capture_decides_nothing():
    old = np.full(10, 40.0) + np.arange(10)
    new = np.full(10, 4.0) + 0.1 * np.arange(10)
    new[3] = 60.0                      # one capture far off (a wrong match): no single-capture threshold
    assert he.postcheck_verdict(old, new).decision == "PASS"


def test_postcheck_gate_missed_is_reported_not_rolled_back():
    old, new = np.full(8, 40.0) + np.arange(8), np.full(8, 7.0) + 0.1 * np.arange(8)
    v = he.postcheck_verdict(old, new, ["Right"] * 4 + ["Left"] * 4)
    assert v.decision == "GATE_MISSED" and not v.passed and not v.rollback
    assert "Right median" in v.reasons[1] and "Left median" in v.reasons[1]


def test_postcheck_worse_advises_rollback_and_noise_does_not():
    old = np.full(10, 5.0) + 0.1 * np.arange(10)
    v = he.postcheck_verdict(old, old + 6.0 + 0.05 * np.arange(10))
    assert v.decision == "WORSE" and v.rollback
    rng = np.random.default_rng(3)
    a = 5 + rng.standard_normal(10)
    v = he.postcheck_verdict(a, a + 0.2 * rng.standard_normal(10))
    assert v.decision == "NOT_BETTER" and not v.rollback


def test_postcheck_needs_enough_captures():
    assert he.postcheck_verdict([40, 41, 42], [4, 4, 4]).decision == "NOT_ENOUGH_DATA"


def test_postcheck_scores_old_and_new_geometry_on_fresh_captures(fit):
    """fresh captures taken AFTER the deploy (the bursts carry T_BS'): the deployed geometry explains them, the old
    one does not"""
    robot = Robot()
    hd = deployer(robot, fit, dry_run=True)
    hd.deploy()
    T_new = he.t_bs(hd.build.config_text)
    new_urdf = Urdf(hd.build.urdf_text)
    D = T_new @ np.linalg.inv(T_BS)
    true_after = {k: v for k, v in TRUE.items() if k != "HEAD"}
    # after the deploy the burst's T_BC = T_BS' T_SC; the camera's true place is still D(TRUE) T_BS T_SC
    head_left = {"dxyz": (np.asarray(TRUE["HEAD"]["dxyz"]) - 1e3 * D[:3, 3]).tolist()}
    recs = [records(s, {**true_after, "HEAD": head_left}, poses=2, T_BC_rec=T_new @ T_SC_DEPTH, seed=7 + i,
                    prefix="q") for i, s in enumerate(("Right", "Left"))]
    sc = he.postcheck_scores(recs, old=(BASE, T_BS), new=(new_urdf, T_new), T_BS_records=T_new)
    assert len(sc["old_mm"]) == 16 and np.median(sc["new_mm"]) < 0.5 < 10 < np.median(sc["old_mm"])
    assert he.postcheck_verdict(sc["old_mm"], sc["new_mm"], sc["side"]).decision == "PASS"


# ---------------------------------------------------------------- consumers


def test_consumers_are_data(tmp_path):
    cs = he.load_consumers()
    assert [c.kind for c in cs].count("okvis") == 1 and any(c.kind == "offline_file" for c in cs)
    p = tmp_path / "c.yaml"
    p.write_text("consumers:\n  - {name: OKVIS, kind: okvis}\n  - {name: X, kind: magic}\n")
    with pytest.raises(ValueError, match="kind 'magic'"):
        he.load_consumers(p)
    p.write_text("consumers:\n  - {name: A, kind: okvis}\n  - {name: B, kind: okvis}\n")
    with pytest.raises(ValueError, match="exactly one"):
        he.load_consumers(p)


# ---------------------------------------------------------------- the first real fit (phantom10, 11 Nov)

LIVE_1111A = CAL / "results" / "handeye-p10-2026-11-11" / "live-1111a"


def test_live_1111a_deploy_reproduces_the_fit_and_its_urdf(data_root):
    base_p = data_root / "p10arm" / "p10_mk1_1.urdf"
    if not base_p.exists():
        pytest.skip(f"{base_p} missing (Moonstone: a link to /data/kaustabh/p10arm/p10_mk1_1.urdf)")
    fit = he.HandeyeFit.load(LIVE_1111A)
    assert fit.problems() == [] and fit.base_md5 == "cbf58ef15e990ebd40d2e33ca6d4d685"
    base_text = base_p.read_text()
    ds = dp.DeploySnapshot(ENV, f"/etc/okvis/{LIVE_NAME}", f"{dp.HOST_CFG_DIR}/{LIVE_NAME}", LIVE_CFG, "3,0", None,
                           "{}", "img", {"kind": "daemonset", "argv": ["x"], "detail": ""})
    snap = he.HandeyeSnapshot(ds, "phantom10", MID, None, URDF_C, URDF_H, base_text, None, [])
    b = he.build(snap, fit, "head", date="20261111", session="test")
    assert b.model == "M12+head" and b.predict_mm < 1e-6
    assert np.allclose(b.head_mm, [-21.354813, -1.648261, -5.559096], atol=1e-5)
    stored = Urdf.load(LIVE_1111A / "phantom10_mk1_handeye-20261111.urdf")
    new = Urdf(b.urdf_text)
    for name, j in stored.joints.items():
        assert np.abs(new.joint(name).origin - j.origin).max() < 1e-9, name
    # the in-sample medians of fit.json, recomputed from the written files
    T_new = he.t_bs(b.config_text)
    for side in ("Right", "Left"):
        e = he.wrist_errors(fit.selected(side), new, T_new, T_BS)
        assert abs(np.median(e) - fit.report["models"]["M12+head"]["in_sample_median_mm"][side]) < 1e-6
