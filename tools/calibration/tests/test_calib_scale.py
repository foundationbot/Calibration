"""Step 65: stereo-triangulated board pitch vs the steel-rule pitch."""
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
from test_calib_validate import K, D, LENS, _T10, _views  # noqa: E402

scale = importlib.import_module("65_scale_check")
validate = importlib.import_module("60_validate_stereo")


def _ratios(T10_truth, T10_calib):
    board = rigcfg.board_from_target(APRILGRID)
    d0, d1 = _views(T10_truth)
    keys, p0, p1 = validate.match_tags(d0, d1)
    X = scale.triangulate(LENS, p0, p1, K, D, K, D, T10_calib)
    return scale.pitch_ratios(keys, X, board, board.pitch)


def test_true_calibration_reproduces_the_pitch():
    r = _ratios(_T10(), _T10())
    assert len(r) > 100
    assert np.median(r) == pytest.approx(1.0, abs=1e-4)


def test_a_one_percent_long_baseline_reads_one_percent_large():
    # the declared board size scales Kalibr's baseline; a baseline 1 % too long
    # triangulates every point, and so every spacing, 1 % too far/large
    long = _T10(); long[:3, 3] *= 1.01
    assert np.median(_ratios(_T10(), long)) == pytest.approx(1.01, abs=1e-3)


def test_only_same_corner_pairs_in_one_row_or_column_are_compared():
    board = rigcfg.board_from_target(APRILGRID)
    keys = [(0, 0), (1, 0), (7, 0), (8, 1)]
    X = np.array([board.obj_points(t)[k] for t, k in keys])
    r = scale.pitch_ratios(keys, X, board, board.pitch)
    # (0,0)-(1,0): same row; (0,0)-(7,0): same column; (8,1) shares no corner index
    assert sorted(r) == pytest.approx([1.0, 1.0])
