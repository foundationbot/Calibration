"""rigcfg.Lens('pinhole-kb4t'): OpenCV-shaped calls through lensmodel, so PnP starts (steps 60, 81) work with KB4T."""
import importlib
import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
import rigcfg  # noqa: E402
from test_calib_board import APRILGRID  # noqa: E402

s81 = importlib.import_module("81_tbs_handeye")
FIT_A = json.loads((HERE / "data" / "kb4t_reference.json").read_text())["intr"]
K = np.array([[FIT_A[0][0], 0, FIT_A[0][2]], [0, FIT_A[0][1], FIT_A[0][3]], [0, 0, 1.0]])
D6 = np.array(FIT_A[0][4:]).reshape(6, 1)
D6_NO_P = np.r_[D6[:4, 0], 0.0, 0.0].reshape(6, 1)
KB4T = rigcfg.Lens("pinhole-kb4t")


def _obj(n=80, seed=2):
    return np.random.default_rng(seed).uniform([-0.6, -0.4, 0.6], [0.6, 0.4, 1.4], (n, 3))


def test_the_name_is_known_and_is_a_fisheye_angle():
    assert KB4T.kind == "kb4t" and not KB4T.fisheye and abs(KB4T.off_axis_rad(565.0, 565.0) - 1.0) < 1e-12


def test_without_tangential_terms_it_is_opencvs_fisheye():
    obj = _obj().reshape(-1, 1, 3)
    rv, tv = np.array([[0.1], [-0.2], [0.05]]), np.array([[0.02], [-0.01], [0.1]])
    ref, _ = cv2.fisheye.projectPoints(obj, rv, tv, K, D6[:4])
    got, _ = KB4T.project(obj, rv, tv, K, D6_NO_P)
    assert got.shape == ref.shape and np.abs(got - ref).max() < 1e-6
    pix = ref.reshape(-1, 1, 2)
    R = cv2.Rodrigues(np.array([0.01, -0.02, 0.005]))[0]
    P = np.array([[300.0, 0, 256.0], [0, 300.0, 160.0], [0, 0, 1.0]])
    for kw in ({}, {"R": R}, {"R": R, "P": P}):
        ref_u = cv2.fisheye.undistortPoints(pix, K, D6[:4], **kw)      # OpenCV iterates to ~1e-8: compare at 1e-5
        assert np.abs(KB4T.undistort(pix, K, D6_NO_P, **kw) - ref_u).max() < 1e-5, kw


def test_pnp_recovers_a_board_seen_through_kb4t():
    board = rigcfg.board_from_target(APRILGRID)
    keys = [(i, k) for i in range(board.n_tags) for k in range(4)]
    X = np.array([board.obj_points(i)[k] for i, k in keys]).reshape(-1, 1, 3)
    rv, tv = np.array([[0.2], [-0.1], [0.05]]), np.array([[-0.35], [-0.25], [1.1]])
    uv = KB4T.project(X, rv, tv, K, D6)[0].reshape(-1, 2)
    T = s81.pose_from_corners({kk: tuple(v) for kk, v in zip(keys, uv)}, board, KB4T, K, D6,
                              centre=(640, 400), max_radius_px=900)
    assert np.allclose(T[:3, :3], cv2.Rodrigues(rv)[0], atol=1e-6) and np.allclose(T[:3, 3], tv.ravel(), atol=1e-6)


def test_wrong_coefficient_counts_and_skew_are_refused():
    with pytest.raises(ValueError, match="6 distortion numbers"):
        KB4T.project(_obj().reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, D6[:4])
    Ks = K.copy()
    Ks[0, 1] = 0.5
    with pytest.raises(ValueError, match="no skew"):
        KB4T.undistort(np.zeros((1, 1, 2)), Ks, D6)
    with pytest.raises(NotImplementedError):
        KB4T.stereo_rectify(K, D6, K, D6, (1280, 800), np.eye(3), np.zeros(3), 0)
