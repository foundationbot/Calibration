"""Arm session, Task 2: step 90's safety glue (review of 23 Oct, S1-S5 and Q1) against a fake MJCF body: the joint
ranges of both models, the gravity command's range, every pose and path check, path sampling by distance, and a
failed gate writing no runnable list."""
import importlib
import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "scripts"))
import armchain  # noqa: E402

step90 = importlib.import_module("90_arm_poses")
URDF = ROOT / "config/foundation_rig/20250206_phantom_mk1.urdf"
START = np.radians([-42.6, -4.8, 47.5, -92.3, -135.7, 21.2, 19.8])


class FakeBody:
    """MJCF stand-in: RightShoulderYaw's range is the MJCF's (-110..+70 deg), the others wide; the gravity torque a
    constant; the clearance 10 cm except where the elbow is beyond -100 deg (1 cm)."""

    def __init__(self, torque=(-8.0, 0, 0, -6.0, 0, 0, 0)):
        self.t = np.array(torque, float)

    def ranges(self):
        lo, hi = np.full(7, -10.0), np.full(7, 10.0)
        lo[2], hi[2] = np.radians(-110), np.radians(70)
        return lo, hi

    def gravity(self, Q, g):
        return np.tile(self.t, (len(np.atleast_2d(Q)), 1))

    def clearance(self, Q):
        Q = np.atleast_2d(Q)
        return np.where(Q[:, 3] < np.radians(-100), 0.01, 0.10)


def _chk(**kw):
    return step90.Checks(armchain.ArmChain(URDF), FakeBody(**kw), up=(0, 0, 1), g_pelvis=(0, 0, -1))


def test_limits_are_the_intersection_of_both_models():
    c = _chk()
    assert np.isclose(np.degrees(c.upper[2]), 70.0) and np.isclose(np.degrees(c.lower[2]), -70.0)   # URDF -70..110
    q = START.copy()
    q[2] = np.radians(65.0)                                          # inside the URDF, 5 deg from the MJCF's limit
    assert any("range" in w for w in c.pose(q))


def test_the_gravity_command_must_stay_inside_the_ranges():
    c = _chk(torque=(0, 0, 0, -14.0, 0, 0, 0))                        # elbow command 8.0 deg beyond the pose
    q = START.copy()
    q[3] = np.radians(-123.0)                       # 12 deg inside the URDF's -135; commanded -131, inside by 4
    why = c.pose(q)
    assert any("command" in w for w in why)


def test_a_pose_near_the_body_and_its_sag_band_are_named():
    c = _chk()
    q = START.copy()
    q[3] = np.radians(-95.0)                                         # 10 cm, but the band reaches -103 deg: 1 cm
    why = c.pose(q)
    assert not any(w.startswith("clearance 1") for w in why) and any("sag band 1.0" in w for w in why)
    q[3] = np.radians(-105.0)
    assert any(w.startswith("clearance 1.0") for w in c.pose(q))


def test_paths_are_sampled_no_more_than_a_centimetre_apart_and_checked():
    c = _chk()
    a, b = START.copy(), START.copy()
    b[0] += np.radians(30.0)
    Q = c.samples(a, b)
    pts = np.stack([f[:, :3, 3] for f in c.arm.frames(Q)], 1)
    assert np.linalg.norm(np.diff(pts, axis=0), axis=2).max() <= step90.PATH_MM / 1e3
    b2 = START.copy()
    b2[3] = np.radians(-80.0)
    a2 = START.copy()
    a2[3] = np.radians(-80.0)
    a2[0] -= 0.3
    assert c.path(a2, b2) == []                                      # elbow at -80: clear all the way
    a3 = START.copy()
    a3[3] = np.radians(-110.0)                                       # through the close zone
    assert any("path clearance 1.0" in w for w in c.path(a3, b2))


def test_a_failed_gate_writes_no_runnable_list(tmp_path, monkeypatch):
    out = tmp_path / "poses.json"
    fake = {"gate": {"pass": False, "fails": ["x"], "unsafe": [], "coverage": [], "sigma": {}}, "stops": [{"k": 1}]}
    monkeypatch.setattr(step90, "run", lambda a: dict(fake))
    monkeypatch.setattr(step90, "report", lambda r: "")
    argv = ["--urdf", "u", "--mjcf", "m", "--config", "c", "--snapshot", "s", "--palm-corners", "p", "--out", str(out)]
    out.write_text('{"gate": {"pass": true}, "stops": [{"k": 1}]}')      # an older list that passed
    assert step90.main(argv) == 1
    assert not out.exists()
    rej = json.loads(out.with_suffix(".REJECTED.json").read_text())
    assert rej["stops"] == []


def test_the_sag_band_covers_shoulder_roll_too():
    """Shoulder roll gets gravity commands of up to 10 deg at kp 100 (the 05:40 list), so it lands +-several deg
    off like pitch and elbow: a pose whose roll 8 deg further out is close to the body is flagged."""
    class RollBody(FakeBody):
        def clearance(self, Q):
            Q = np.atleast_2d(Q)
            return np.where(Q[:, 1] < np.radians(-40), 0.01, 0.10)
    c = step90.Checks(armchain.ArmChain(URDF), RollBody(), up=(0, 0, 1), g_pelvis=(0, 0, -1))
    q = START.copy()
    q[1] = np.radians(-35.0)
    assert any("sag band 1.0" in w for w in c.pose(q))
