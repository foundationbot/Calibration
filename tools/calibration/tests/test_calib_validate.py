"""Step 60's aprilgrid path: match corners by tag id, measure rectified row error."""
import importlib
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("yaml")
cv2 = pytest.importorskip("cv2")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
sys.path.insert(0, str(HERE))
import rigcfg  # noqa: E402
from test_calib_board import APRILGRID  # noqa: E402

validate = importlib.import_module("60_validate_stereo")

RES = (1280, 800)
K = np.array([[568.0, 0, 640.0], [0, 568.0, 400.0], [0, 0, 1.0]])
D = np.zeros((4, 1))
LENS = rigcfg.Lens("pinhole-equi")
BASELINE = 0.08


def _T10(rot_deg=(0.0, 0.0, 0.0)):
    """cam0 -> cam1 for a LEFT cam0 and a RIGHT cam1 BASELINE metres along +x."""
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.radians(np.array(rot_deg, float)))[0]
    T[:3, 3] = [-BASELINE, 0.0, 0.0]
    return T


def _views(T10, board_rv=(0.3, -0.2, 0.05), board_t=(-0.35, -0.25, 1.0)):
    """Project every aprilgrid corner into both cameras: {(id, k): (u, v)} each."""
    b = rigcfg.board_from_target(APRILGRID)
    keys = [(i, k) for i in range(b.n_tags) for k in range(4)]
    X = np.array([b.obj_points(i)[k] for i, k in keys]).reshape(-1, 1, 3)
    R0 = cv2.Rodrigues(np.array(board_rv, float))[0]
    t0 = np.array(board_t, float)
    p0 = cv2.fisheye.projectPoints(X, cv2.Rodrigues(R0)[0], t0.reshape(3, 1), K, D)[0]
    R1 = T10[:3, :3] @ R0
    t1 = T10[:3, :3] @ t0 + T10[:3, 3]
    p1 = cv2.fisheye.projectPoints(X, cv2.Rodrigues(R1)[0], t1.reshape(3, 1), K, D)[0]
    return ({kk: tuple(v) for kk, v in zip(keys, p0.reshape(-1, 2))},
            {kk: tuple(v) for kk, v in zip(keys, p1.reshape(-1, 2))})


def test_corners_are_paired_by_tag_and_corner_only_where_both_cameras_saw_them():
    d0 = {(1, 0): (10.0, 20.0), (1, 1): (11.0, 21.0), (2, 0): (30.0, 40.0)}
    d1 = {(2, 0): (25.0, 40.5), (1, 0): (5.0, 20.2), (7, 3): (1.0, 1.0)}
    keys, p0, p1 = validate.match_tags(d0, d1)
    assert keys == [(1, 0), (2, 0)]
    assert np.allclose(p0, [[10.0, 20.0], [30.0, 40.0]])
    assert np.allclose(p1, [[5.0, 20.2], [25.0, 40.5]])


def test_true_calibration_puts_matches_on_the_same_row():
    T10 = _T10()
    d0, d1 = _views(T10)
    _, p0, p1 = validate.match_tags(d0, d1)
    rect = validate.rectify(LENS, K, D, K, D, T10, RES)
    dy, dx = validate.rectified_rows(LENS, p0, p1, K, D, K, D, rect)
    assert np.median(dy) < 0.01
    assert np.median(dx) > 0          # kalibr cam0 is the LEFT eye


def test_a_half_degree_rotation_error_breaks_row_alignment():
    d0, d1 = _views(_T10(rot_deg=(0.5, 0.0, 0.0)))      # the truth
    _, p0, p1 = validate.match_tags(d0, d1)
    rect = validate.rectify(LENS, K, D, K, D, _T10(), RES)  # a calibration that missed it
    dy, _ = validate.rectified_rows(LENS, p0, p1, K, D, K, D, rect)
    assert np.median(dy) > 1.0
