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
