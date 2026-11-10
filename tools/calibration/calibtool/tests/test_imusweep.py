"""calibtool.imusweep: the neck sweep behind SweepRig (the kit's imucam_fast.py without robot imports).

Fakes only: a synthetic AprilGrid view projected with the KB4T lens for board_normal, and a FakeRig (neck kinematics,
a true cam0 rotation, IMU noise, a foreign-command counter) for the capture loop. Nothing here touches a robot."""
import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("yaml")

from calibtool import imu, imusweep  # noqa: E402

INTR = np.array([583.5, 582.9, 643.4, 388.9, -0.01, 0.002, 0.0003, -0.0001, 1e-5, -2e-5])


# ---------------------------------------------------------------- task 8: poses and the board normal


def test_serpentine_default():
    poses = imusweep.serpentine()
    assert len(poses) == 15
    assert poses[:5] == [(8.0, y) for y in (-30.0, -15.0, 0.0, 15.0, 30.0)]
    assert poses[5:10] == [(16.0, y) for y in (30.0, 15.0, 0.0, -15.0, -30.0)]
    assert poses[10:] == [(24.0, y) for y in (-30.0, -15.0, 0.0, 15.0, 30.0)]
    assert imusweep.serpentine((5,), (1, 2)) == [(5.0, 1.0), (5.0, 2.0)]


def _view(tilt_deg=30.0, tags=36):
    """a 6x6 AprilGrid (8 cm tags) 1.2 m in front of camera 0, tilted; its detection dict and true normal"""
    pytest.importorskip("cv2")
    from calibtool import paths  # noqa: F401
    import lensmodel
    import rigcfg
    board = rigcfg.AprilGrid(6, 6, 0.08, 0.3)
    Rb = imu.axis_rotation([1, 0.2, 0], 180.0 - tilt_deg)
    centre = np.array([2.5 * board.pitch + 0.04, 2.5 * board.pitch + 0.04, 0.0])
    t = np.array([0.05, -0.03, 1.2]) - Rb @ centre
    det = {}
    for tag in range(tags):
        P = (Rb @ board.obj_points(tag).T).T + t
        for k, uv in enumerate(lensmodel.project("KB4T", INTR, P)):
            det[(tag, k)] = uv
    n = Rb[:, 2] if Rb[:, 2] @ t < 0 else -Rb[:, 2]
    return board, det, n


def test_board_normal_from_synthetic_view():
    board, det, n = _view()
    v = imusweep.board_normal(det, board, "KB4T", INTR)
    assert v is not None and v.corners == 144
    assert np.allclose(v.normal, n, atol=1e-6) and np.isclose(np.linalg.norm(v.normal), 1)
    assert v.rms_px < 1e-3


def test_board_normal_too_few_tags():
    board, det, _ = _view(tags=7)
    assert imusweep.board_normal(det, board, "KB4T", INTR) is None
    board, det, _ = _view(tags=8)
    assert imusweep.board_normal(det, board, "KB4T", INTR) is not None


# ---------------------------------------------------------------- task 9: the capture loop behind SweepRig

R_HC = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])   # camera (x right, y down, z fwd) in head
R_CFG = R_HC.T @ imu.axis_rotation([1, 2, 3], 7.0)


def up_cam(pitch_deg, yaw_deg):
    p, y = np.radians(pitch_deg), np.radians(yaw_deg)
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return R_HC.T @ (Rz @ Ry).T @ np.array([0.0, 0.0, 1.0])


class FakeRig:
    """SweepRig for tests: the neck lands within 0.3 deg of the command, the IMU reads 9.81 x R_true @ up plus noise,
    frames are (pose, frame) tokens; foreign_at = pose index before whose move the foreign counter advances"""

    def __init__(self, R_true, foreign_at=None, seed=5):
        self.R_true, self.foreign_at = R_true, foreign_at
        self.rng = np.random.default_rng(seed)
        self.moves, self.seq, self.checks, self.pose = [], 0, 0, None

    def foreign_seq(self):
        self.checks += 1
        if self.foreign_at is not None and len(self.moves) == self.foreign_at and self.checks % 2 == 0:
            self.seq += 1
        return self.seq

    def move_neck(self, pitch_deg, yaw_deg):
        self.moves.append((pitch_deg, yaw_deg))
        self.pose = (pitch_deg + self.rng.uniform(-0.3, 0.3), yaw_deg + self.rng.uniform(-0.3, 0.3))

    def neck_deg(self):
        return self.pose

    def imu_accel(self, seconds):
        g = 9.81 * (self.R_true @ up_cam(*self.pose))
        return g + self.rng.normal(0, 0.05, size=(int(200 * seconds), 3))

    def frames(self, k):
        return [(len(self.moves) - 1, i) for i in range(k)]


def detect_from(rig, missing=()):
    def detect(token):
        pose, _ = token
        if pose in missing:
            return None
        return imusweep.BoardView(up_cam(*rig.pose), 120, 0.2)
    return detect


def test_capture_then_fit_recovers_rotation():
    rig = FakeRig(R_CFG @ imu.rot_xz(0.4, -0.2))
    slept = []
    res = imusweep.capture_sweep(rig, imusweep.serpentine(), detect_from(rig), R_CFG, sleep=slept.append)
    assert len(res) == 15 and rig.moves == imusweep.serpentine()
    assert all(r.view is not None and r.n_views == 3 for r in res)
    s = imusweep.to_sweep(res, R_CFG)
    f = imu.fit(s)
    assert np.abs(f.angles_deg - [0.4, -0.2]).max() < 0.02
    assert all(abs(r.gap_deg - g) < 1e-9 for r, g in zip(res, imu.floor_gap_deg(s, R_CFG)))


def test_capture_uses_injected_sleep():
    rig = FakeRig(R_CFG)
    slept = []
    imusweep.capture_sweep(rig, imusweep.serpentine(), detect_from(rig), R_CFG, sleep=slept.append)
    assert slept == [imusweep.FOREIGN_WINDOW_S, imusweep.SETTLE_S] * 15


def test_capture_aborts_on_foreign_command():
    rig = FakeRig(R_CFG, foreign_at=3)
    with pytest.raises(imusweep.SweepAborted, match="pose 4"):
        imusweep.capture_sweep(rig, imusweep.serpentine(), detect_from(rig), R_CFG, sleep=lambda s: None)
    assert len(rig.moves) == 3


def test_capture_skips_pose_without_board():
    rig = FakeRig(R_CFG @ imu.rot_xz(0.4, -0.2))
    log = []
    res = imusweep.capture_sweep(rig, imusweep.serpentine(), detect_from(rig, missing={6}), R_CFG,
                                 sleep=lambda s: None, log=log.append)
    assert len(res) == 15 and res[6].view is None and res[6].gap_deg is None and res[6].n_views == 0
    assert len(imusweep.to_sweep(res, R_CFG)) == 14
    assert len(log) == 15 and "no board" in log[6]


def test_save_sweep_is_rscfit_format(tmp_path):
    rig = FakeRig(R_CFG @ imu.rot_xz(0.4, -0.2))
    res = imusweep.capture_sweep(rig, imusweep.serpentine(), detect_from(rig, missing={2}), R_CFG,
                                 sleep=lambda s: None)
    p = tmp_path / "imucam.npz"
    imusweep.save_sweep(p, res, R_CFG)
    with np.load(p) as z:
        assert set(z.files) == {"neck", "acc", "acc_std", "nb", "n", "R_SC"}
        assert z["nb"].shape == (14, 3) and np.array_equal(z["nb"], z["n"]) and np.array_equal(z["R_SC"], R_CFG)
    a, b = imu.load_sweep(p), imusweep.to_sweep(res, R_CFG)
    assert np.allclose(a.normal, b.normal) and np.allclose(a.acc, b.acc) and np.allclose(a.neck, b.neck)
