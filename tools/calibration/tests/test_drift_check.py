"""driftcheck: camera 1's roll and pitch against a calibration from one still board's epipolar coplanarity; the depth
error at the board; coverage; the verdict (plan 2026-10-22-phantom12-drift-check.md)."""
import importlib
import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("cv2")
pytest.importorskip("yaml")

sys.path.insert(0, str(Path(__file__).resolve().parent))
S = importlib.import_module("lens_synth")
dc = importlib.import_module("driftcheck")
I = [S.I0, S.I1]
CENTRE = [(643, 385)]


def turned(C, rotvec_deg):
    """C with camera 1 turned by rotvec (camera-0 axes, deg): R' = Exp(rotvec) R"""
    out = np.array(C, float)
    out[:3, :3] = S.sf.rotmat(np.radians([rotvec_deg]))[0] @ out[:3, :3]
    return out


def board(truth, n=12, seed=3, z=(0.7, 0.9), targets=CENTRE):
    return S.views(I, truth, n, seed=seed, z_range=z, noise=0.15, targets=targets, jitter=(40, 30))


def test_a_planted_roll_and_pitch_are_read_back():
    d = board(turned(S.C_TRUE, [0.03, 0.0, -0.36]))
    r = dc.roll_pitch("KB4T", I, S.C_TRUE, d, S.BOARD)
    assert r["roll_deg"] == pytest.approx(-0.36, abs=0.02) and r["pitch_deg"] == pytest.approx(0.03, abs=0.01)
    assert r["pairs"] == 12 and r["roll_p10_p90"][0] <= r["roll_deg"] <= r["roll_p10_p90"][1]


def test_the_true_calibration_reads_zero_and_yaw_is_not_mistaken_for_roll():
    for truth in (S.C_TRUE, turned(S.C_TRUE, [0.0, 0.10, 0.0])):      # a 0.1 deg yaw: not a roll, not a pitch
        r = dc.roll_pitch("KB4T", I, S.C_TRUE, board(truth), S.BOARD)
        assert abs(r["roll_deg"]) < 0.02 and abs(r["pitch_deg"]) < 0.01
