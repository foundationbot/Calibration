"""frames.py: a burst's camera in the Head, Torso and Pelvis through the URDF given (not a hard-coded one)."""
import numpy as np
import pytest

from armcal import REPO_URDF, burst_io
from armcal.frames import Frames
from armcal.urdf import Urdf
import neckfk
from rigmath import rot_axis, transform

NAMES = ["NeckYaw", "NeckPitch", "SpineYaw", "RightShoulderRoll"]


def burst(T_BS=None, joints=None, n=3):
    stamps = np.arange(n, dtype=np.int64) * 1_000_000_000 + 10_000_000_000
    if joints is None:
        t = np.linspace(stamps[0] - 1e9, stamps[-1] + 1e9, 50)
        joints = np.column_stack([t, np.full(50, 0.1), np.full(50, 0.3), np.full(50, -0.05), np.full(50, 0.2)])
    T_SC = transform(rot_axis((0, 1, 0), 0.1), [0.01, 0.02, 0.03])
    return burst_io.Burst(depth=np.ones((n, 4, 5), np.float32), rgb=np.zeros((n, 4, 5, 3), np.uint8),
                          K=np.array([100.0, 100, 2, 2]), T_SC=T_SC, stamp_ns=stamps, seq=np.arange(n),
                          info={"joint_names": NAMES}, joints=joints,
                          T_BS=transform(rot_axis((1, 0, 0), -0.4), [0.1, 0, 0.05]) if T_BS is None else T_BS)


def test_head_torso_pelvis_follow_the_chain():
    b = burst()
    f = Frames(b, REPO_URDF)
    T_BC = b.T_BS @ b.T_SC
    assert np.allclose(f.T_FC("camera"), np.eye(4))
    assert np.allclose(f.T_FC("head"), T_BC)
    assert np.allclose(f.T_FC("torso"), neckfk.NeckChain(REPO_URDF).T_TB(0.1, 0.3) @ T_BC, atol=1e-12)
    sp = neckfk.load_joint(REPO_URDF, "SpineYaw")
    want = sp.origin @ transform(rot_axis(sp.axis, -0.05), np.zeros(3)) @ f.T_FC("torso")
    assert np.allclose(f.T_FC("pelvis"), want, atol=1e-12)


def test_the_urdf_given_is_the_one_used():
    b = burst()
    moved = Urdf.load(REPO_URDF).changed({"NeckYaw": {"dxyz": [0.0, 0.0, 10.0]}})
    d = Frames(b, moved).T_FC("torso")[:3, 3] - Frames(b, REPO_URDF).T_FC("torso")[:3, 3]
    assert np.allclose(d, [0, 0, 0.010], atol=1e-12)


def test_angles_are_the_median_at_the_frame_stamps_and_motion_is_reported():
    b = burst(n=5)
    t = np.array([0, 5e9, 30e9], float)
    b.joints = np.column_stack([t, [0.0, 0.5, 3.0], np.zeros(3), np.zeros(3), np.zeros(3)])
    f = Frames(b, REPO_URDF)
    s = b.stamp_ns.astype(float)
    yaw = np.interp(s, t, [0.0, 0.5, 3.0])
    assert f.angles()["NeckYaw"] == pytest.approx(np.median(yaw))
    assert f.angles(frames_used=2)["NeckYaw"] == pytest.approx(np.median(yaw[:2]))
    assert f.neck_motion_deg()["NeckYaw"] == pytest.approx(np.degrees(np.ptp(yaw)))


def test_missing_inputs_are_named():
    b = burst()
    b.T_BS = None
    with pytest.raises(ValueError, match="T_BS"):
        Frames(b, REPO_URDF).T_FC("head")
    b = burst()
    b.info = {"joint_names": ["SpineYaw", "NeckPitch", "A", "B"]}
    with pytest.raises(ValueError, match="NeckYaw"):
        Frames(b, REPO_URDF).T_FC("torso")


def test_a_mirror_image_is_refused():
    with pytest.raises(ValueError, match="not a rotation"):
        Frames(burst(T_BS=np.diag([1.0, 1, -1, 1])), REPO_URDF).T_FC("head")
