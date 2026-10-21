"""T_BS steps: rotation and transform helpers."""
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import rigmath as rm  # noqa: E402

U = np.array([0.6, 0.0, 0.8])


@pytest.mark.parametrize("w", [[0.0, 0.0, 0.0], [1e-13, -2e-13, 0.0], [0.3, -0.2, 0.1], [1e-9, 0, 0],
                               [3e-8, -4e-8, 1e-8], [2.0, 1.0, -0.5], [0, 0, np.pi - 1e-9],
                               list(-(np.pi - 1e-3) * U), list(-(np.pi - 1e-7) * U)])
def test_log_inverts_exp_at_every_angle(w):
    R = rm.exp_rot(w)
    assert np.allclose(R @ R.T, np.eye(3), rtol=0, atol=1e-12)
    assert np.allclose(rm.exp_rot(rm.log_rot(R)), R, rtol=0, atol=1e-9)
    assert np.allclose(rm.log_rot(R), w, rtol=1e-6, atol=1e-15)


def test_half_turn_axis_is_recovered():
    a = np.array([0.6, 0.0, 0.8])
    w = rm.log_rot(rm.rot_axis(a, np.pi))
    assert abs(np.linalg.norm(w) - np.pi) < 1e-9 and abs(abs(w @ a) / np.pi - 1.0) < 1e-9


@pytest.mark.parametrize("axis", [np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0), np.array([-4.0, 4.0, -7.0]) / 9.0])
def test_a_rounded_rotation_near_a_half_turn_keeps_its_axis(axis):
    """A matrix printed with six decimals, 1e-5 rad short of a half turn: the antisymmetric part
    is mostly rounding there, so the axis must come from the symmetric part. The second axis rounds
    to a trace just above -1, so it also pins where the two branches switch."""
    R = np.round(rm.exp_rot(-(np.pi - 1e-5) * axis), 6)
    assert rm.rot_angle_deg(rm.exp_rot(rm.log_rot(R)).T @ rm.nearest_rotation(R)) < 1e-3


def test_angle_ignores_yaml_rounding():
    """A rotation printed with six decimals is orthonormal only to ~1e-6. arccos would read the
    angle between it and its cleaned self as ~0.06 deg; atan2 reads ~0."""
    R = np.round(rm.rot_axis([0.2, 0.9, -0.4], 0.97), 6)
    assert np.abs(R @ R.T - np.eye(3)).max() > 1e-7
    assert rm.rot_angle_deg(rm.nearest_rotation(R).T @ R) < 1e-4
    assert abs(rm.rot_angle_deg(rm.rot_axis([1, 1, 0], np.radians(0.3))) - 0.3) < 1e-9
    assert abs(rm.rot_angle_deg(rm.rot_axis([0, 0, 1], np.radians(179.9))) - 179.9) < 1e-9


def test_nearest_rotation_is_a_rotation_not_a_reflection():
    R = rm.rot_axis([0.3, -0.5, 0.8], 0.7)
    Q = rm.nearest_rotation(R @ np.diag([1.0, 1.0, -1.0]))
    assert np.allclose(Q @ Q.T, np.eye(3), rtol=0, atol=1e-12) and abs(np.linalg.det(Q) - 1.0) < 1e-12


def test_inverse_mean_pose_and_spread():
    T = rm.transform(rm.rot_axis([0, 1, 0], 0.4), [0.1, -0.2, 0.3])
    assert np.allclose(rm.inv(T) @ T, np.eye(4), rtol=0, atol=1e-12)
    Ts = [rm.transform(rm.rot_axis([0, 0, 1], a), [a, 0, 0]) for a in (-0.3, 0.0, 0.3)]
    assert np.allclose(rm.mean_pose(Ts), np.eye(4), rtol=0, atol=1e-12)
    s = rm.spread(Ts)
    assert s["n"] == 3 and abs(s["trans_max_mm"] - 300.0) < 1e-9 and abs(s["rot_max_deg"] - np.degrees(0.3)) < 1e-9
    assert abs(s["trans_rms_mm"] - 300 * np.sqrt(2 / 3)) < 1e-9 and abs(s["rot_rms_deg"] - np.degrees(0.3) * np.sqrt(2 / 3)) < 1e-9
    two = [T, T @ rm.transform(rm.rot_axis([1, 0, 0], np.radians(0.2)), [0.002, 0, 0])]
    s = rm.spread(two)
    assert abs(s["trans_rms_mm"] - 1.0) < 0.05 and abs(s["rot_rms_deg"] - 0.1) < 0.005
    assert rm.spread([T, T])["rot_rms_deg"] < 1e-9
