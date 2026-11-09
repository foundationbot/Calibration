"""Step 08's pure parts (joint names, pins, grid, neck errors) and step 15b on synthetic neck-grid take folders."""
import importlib
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
ng = importlib.import_module("08_neck_grid_direct")

NAMES = ["LeftShoulderPitch", "LeftShoulderRoll", "LeftShoulderYaw", "LeftElbowPitch", "LeftWristYaw", "LeftWristRoll",
         "LeftGripperYaw", "RightShoulderPitch", "RightShoulderRoll", "RightShoulderYaw", "RightElbowPitch",
         "RightWristYaw", "RightWristRoll", "RightGripperYaw", "NeckPitch", "NeckYaw", "SpineYaw", "LeftHipRoll",
         "LeftHipYaw", "LeftHipPitch", "LeftKneePitch", "RightHipRoll", "RightHipYaw", "RightHipPitch",
         "RightKneePitch", "LeftAnkleRoll", "LeftAnklePitch", "RightAnkleRoll", "RightAnklePitch"]


def test_joint_names_come_out_of_a_config_entry_in_order():
    entry = b"\x00\x07" + b"\x00\x00".join(n.encode() for n in NAMES + ["NeckPitch", "LeftWristYaw"]) + b"\x00\x01ab"
    assert ng.joint_names(entry) == NAMES


def test_pins_hold_arms_spine_and_legs_but_never_the_neck_or_ankles():
    q = [float(i) for i in range(29)]
    held = dict(kv.split("=") for kv in ng.pins(NAMES, q).split(","))
    assert set(held) == set(NAMES[:14]) | {"SpineYaw"} | set(NAMES[17:25])
    assert "NeckPitch" not in held and "LeftAnkleRoll" not in held
    assert held["SpineYaw"] == "16.00" and held["RightKneePitch"] == "24.00"


def test_grid_is_serpentine_and_clamped():
    assert ng.grid([-30, 0], [-40, 0, 40]) == [(-26, -34), (-26, 0), (-26, 34), (0, 34), (0, 0), (0, -34)]


def test_neck_errors_keep_only_the_neck_joints():
    out = ("RESULT targets joint=14 target_deg=5 ss_err=-0.4 rms=0.4 max=0.5\n"
           "RESULT targets joint=15 target_deg=2 ss_err=0.2 rms=0.2 max=0.3\n"
           "RESULT targets joint=3 target_deg=-60 ss_err=-10.7 rms=10 max=11\n")
    assert ng.neck_errors(out, (14, 15)) == {14: -0.4, 15: 0.2}


class FakeBoard:
    """detect() reads a code from the image: left images carry 100 + motion px, right images 200; 8 tags x 4"""

    def detect(self, img):
        code = int(img[0, 0])
        if code == 0:
            return {}
        shift = code - 100 if code < 200 else 0.0
        return {(t, k): (10.0 * t + shift, 20.0 * k) for t in range(8) for k in range(4)}


def _take(folder, poses):
    """poses: list of (cmd, list of left codes); right images always carry 200"""
    folder.mkdir()
    for i, (cmd, codes) in enumerate(poses):
        L = np.stack([np.full((4, 4), c, np.uint8) for c in codes])
        R = np.stack([np.full((4, 4), 200, np.uint8) for _ in codes])
        np.savez(folder / f"pose_{i:03d}.npz", cmd=cmd, neck=[cmd[0] + 0.5, cmd[1]], tl=np.arange(len(codes)) * 10,
                 tr=np.arange(len(codes)) * 10 + 1, L=L, R=R)


def test_15b_keeps_still_pairs_per_pose_and_writes_the_step15_layout(tmp_path):
    d15b = importlib.import_module("15b_dump_neck_take")
    # pose 0: five still pairs; pose 1: the board moved 5 px in pairs 2-4 (head still settling) -> 2 kept;
    # pose 2: no board
    _take(tmp_path / "t", [([0, 0], [100] * 5), ([8, -11], [100, 100, 105, 105, 105]), ([16, 0], [0] * 5)])
    n, held, kept = d15b.dump_take(tmp_path / "t", FakeBoard(), tmp_path / "out.npz", still_px=1.0, per_hold=3)
    assert (n, held, kept) == (3, 2, 5)
    z = np.load(tmp_path / "out.npz")
    assert z["img_hold"].tolist() == [0, 0, 0, 1, 1]
    assert set(z["det_cam"].tolist()) == {0, 1} and len(z["det_u"]) == 5 * 2 * 32
    assert np.allclose(np.degrees(z["img_pitch"][3:]), 8.5) and z["img_ap"][3] == 1 and z["img_ay"][3] == -1
    assert z["img_t0"].tolist()[:3] == [10, 20, 30]          # the middle three of pose 0's five pairs


def test_15b_parallel_poses_give_the_same_output(tmp_path):
    d15b = importlib.import_module("15b_dump_neck_take")
    _take(tmp_path / "t", [([0, 0], [100] * 5), ([8, -11], [100, 100, 105, 105, 105]), ([16, 0], [0] * 5),
                           ([16, 11], [100] * 4)])
    one = d15b.dump_take(tmp_path / "t", FakeBoard(), tmp_path / "a.npz", procs=1)
    many = d15b.dump_take(tmp_path / "t", FakeBoard(), tmp_path / "b.npz", procs=3)
    assert one == many
    a, b = np.load(tmp_path / "a.npz"), np.load(tmp_path / "b.npz")
    assert all(np.array_equal(a[k], b[k]) for k in a.files)
