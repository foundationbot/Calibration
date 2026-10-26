"""Phase C of docs/superpowers/plans/2026-10-25-tagfree-selfcal.md (stereo-depth-eval branch): the plate yardstick's
scoring (yardstick.py): a hand pose against the plate's, the held-out-only verdict, the frozen set's manifest."""
import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
import rigmath as rm  # noqa: E402
import yardstick as ys  # noqa: E402

T_REF = rm.transform(rm.exp_rot(np.radians([10.0, -20.0, 5.0])), (0.05, -0.1, 0.45))
P_W = np.array([0.0, 0.0, -0.12])           # the wrist point in the hand frame


def test_the_reference_scores_zero():
    mm, deg = ys.pose_error(T_REF, T_REF, P_W)
    assert mm < 1e-9 and deg < 1e-9


def test_a_known_shift_and_a_turn_about_the_wrist_read_as_given():
    shift = rm.transform(np.eye(3), (0.002, 0.0, 0.0))                       # 2 mm in the camera frame
    mm, deg = ys.pose_error(T_REF, shift @ T_REF, P_W)
    assert abs(mm - 2.0) < 1e-6 and deg < 1e-9
    turn = rm.transform(np.eye(3), P_W) @ rm.transform(rm.rot_axis((0, 0, 1), np.radians(0.1)), (0, 0, 0)) \
        @ rm.transform(np.eye(3), -P_W)                                        # 0.1 deg about the wrist point
    mm, deg = ys.pose_error(T_REF, T_REF @ turn, P_W)
    assert mm < 1e-9 and abs(deg - 0.1) < 1e-6


def _ref(names):
    return {n: {"T_C0_hand": T_REF.tolist(), "wrist_in_hand": P_W.tolist()} for n in names}


def test_only_held_out_captures_count_and_missing_ones_are_listed():
    names = ["a", "b", "c", "d", "e"]
    bad = (rm.transform(np.eye(3), (0.004, 0, 0)) @ T_REF).tolist()             # 4 mm: fails 3 mm
    cand = {"a": {"T_C0_hand": T_REF.tolist(), "in_sample": False},
            "b": {"T_C0_hand": T_REF.tolist(), "in_sample": False},
            "c": {"T_C0_hand": T_REF.tolist(), "in_sample": False},
            "d": {"T_C0_hand": bad, "in_sample": True}}                          # in-sample: shown, never counted
    out = ys.score(_ref(names), cand)
    assert out["missing"] == ["e"]
    assert out["held_out"] == 3 and out["held_out_pass"] == 3
    assert out["captures"]["d"]["pass"] is False and out["captures"]["d"]["counted"] is False
    assert out["verdict"] == "PASS"


def test_the_verdict_needs_80_percent_of_at_least_three_held_out():
    names = ["a", "b", "c", "d", "e"]
    bad = (rm.transform(np.eye(3), (0.004, 0, 0)) @ T_REF).tolist()
    cand = {n: {"T_C0_hand": T_REF.tolist(), "in_sample": False} for n in names}
    cand["e"]["T_C0_hand"] = bad
    assert ys.score(_ref(names), cand)["verdict"] == "PASS"                     # 4 of 5 = 80 %
    cand["d"]["T_C0_hand"] = bad
    assert ys.score(_ref(names), cand)["verdict"] == "FAIL"                     # 3 of 5
    two = {n: {"T_C0_hand": T_REF.tolist(), "in_sample": False} for n in ("a", "b")}
    assert ys.score(_ref(names), two)["verdict"] == "TOO FEW HELD-OUT"


def test_a_turn_about_the_wrist_fails_on_angle_alone():
    turn = rm.transform(np.eye(3), P_W) @ rm.transform(rm.rot_axis((1, 0, 0), np.radians(0.5)), (0, 0, 0)) \
        @ rm.transform(np.eye(3), -P_W)
    out = ys.score(_ref(["a"]), {"a": {"T_C0_hand": (T_REF @ turn).tolist(), "in_sample": False}})
    assert out["captures"]["a"]["wrist_mm"] < 1e-6 and out["captures"]["a"]["pass"] is False


def test_a_capture_without_in_sample_is_refused():
    with pytest.raises(ValueError):
        ys.score(_ref(["a"]), {"a": {"T_C0_hand": T_REF.tolist()}})


def test_the_manifest_refuses_a_changed_file(tmp_path):
    (tmp_path / "x.json").write_text(json.dumps({"k": 1}))
    (tmp_path / "y.npz").write_bytes(b"0123")
    ys.write_manifest(tmp_path, ["x.json", "y.npz"])
    ys.check_manifest(tmp_path)
    (tmp_path / "y.npz").write_bytes(b"0124")
    with pytest.raises(ValueError, match="y.npz"):
        ys.check_manifest(tmp_path)
