"""Arm session: the right arm's chain read from the URDF (armchain.py).

phantom12's arm joints have zero rpy below the shoulder and unit axes, which hides frame-convention bugs, so the
chain is checked twice: on the real URDF against the hand-written FK used on the robot on 22 Oct
(arm-session/tools/armfk.py), and on a made-up chain with rotated origins and tilted, unnormalised axes against
scipy's rotations."""
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "scripts"))
import armchain  # noqa: E402
import neckfk  # noqa: E402
import rigmath as rm  # noqa: E402

URDF = ROOT / "config/foundation_rig/20250206_phantom_mk1.urdf"
SYN = """<robot name="t">
  <joint name="A" type="revolute">
    <origin rpy="0.3 -0.2 0.5" xyz="0.01 -0.1 0.2"/>
    <axis xyz="0 2 0.1"/>
    <limit lower="-1.0" upper="0.5" effort="10" velocity="1"/>
    <parent link="Torso"/><child link="L1"/>
  </joint>
  <joint name="B" type="revolute">
    <origin rpy="-0.1 0.4 0.0" xyz="0 -0.15 -0.01"/>
    <axis xyz="1 0.05 -0.02"/>
    <limit lower="-2.0" upper="2.0" effort="10" velocity="1"/>
    <parent link="L1"/><child link="L2"/>
  </joint>
  <joint name="C" type="continuous">
    <origin rpy="0 0 0.2" xyz="0.02 0 -0.3"/>
    <axis xyz="0 0 1"/>
    <parent link="L2"/><child link="L3"/>
  </joint>
  <joint name="Other" type="revolute">
    <origin rpy="0 0 0" xyz="0 0 0.1"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
    <parent link="Torso"/><child link="Elsewhere"/>
  </joint>
</robot>"""


@pytest.fixture
def syn(tmp_path):
    p = tmp_path / "arm.urdf"
    p.write_text(SYN)
    return p


def _ref_T(xyz, rpy, axis, q):
    from scipy.spatial.transform import Rotation as R
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    T = np.eye(4)
    T[:3, :3] = R.from_euler("xyz", rpy).as_matrix() @ R.from_rotvec(a * q).as_matrix()
    T[:3, 3] = xyz
    return T


def test_synthetic_chain_matches_scipy(syn):
    pytest.importorskip("scipy")
    c = armchain.ArmChain(syn, ("A", "B", "C"), base="Torso")
    q = np.array([0.3, -0.7, 1.9])
    ref = (_ref_T((0.01, -0.1, 0.2), (0.3, -0.2, 0.5), (0, 2, 0.1), q[0])
           @ _ref_T((0, -0.15, -0.01), (-0.1, 0.4, 0.0), (1, 0.05, -0.02), q[1])
           @ _ref_T((0.02, 0, -0.3), (0, 0, 0.2), (0, 0, 1), q[2]))
    assert np.allclose(c.T_base_tip(q), ref, atol=1e-12)
    fr = c.frames(q)
    assert len(fr) == 3 and np.allclose(fr[-1], ref, atol=1e-12)


def test_batched_fk_equals_one_at_a_time(syn):
    c = armchain.ArmChain(syn, ("A", "B", "C"), base="Torso")
    Q = np.random.default_rng(1).uniform(-1, 1, (5, 3))
    Tb = c.T_base_tip(Q)
    assert Tb.shape == (5, 4, 4)
    for i in range(5):
        assert np.allclose(Tb[i], c.T_base_tip(Q[i]), atol=1e-12)


def test_limits_are_read_and_continuous_is_unbounded(syn):
    c = armchain.ArmChain(syn, ("A", "B", "C"), base="Torso")
    assert np.allclose(c.lower[:2], [-1.0, -2.0]) and np.allclose(c.upper[:2], [0.5, 2.0])
    assert c.lower[2] == -np.inf and c.upper[2] == np.inf


def test_a_broken_chain_is_refused(syn):
    with pytest.raises(ValueError, match="not one chain"):
        armchain.ArmChain(syn, ("A", "C"), base="Torso")
    with pytest.raises(ValueError, match="base"):
        armchain.ArmChain(syn, ("B", "C"), base="Torso")


def _armfk_816(sp, sr, sy, ep, wy, wr, gy):
    """arm-session/tools/armfk.py's arm(), written out by hand from the URDF numbers with #816's RightShoulderPitch
    origin (measured 23 Oct; was -0.02 -0.1 0.214): the gripper link in the torso."""
    def T(xyz=(0, 0, 0), R=np.eye(3)):
        M = np.eye(4); M[:3, :3] = R; M[:3, 3] = xyz; return M  # noqa: E702
    def Rx(a): c, s = np.cos(a), np.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])  # noqa: E702
    def Ry(a): c, s = np.cos(a), np.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])  # noqa: E702
    def Rz(a): c, s = np.cos(a), np.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])  # noqa: E702
    M = T((-0.015, -0.1034, 0.2823), Rx(-0.698132)) @ T(R=Ry(sp))
    M = M @ T((0, -0.15676, -0.013), Rx(0.698132)) @ T(R=Rx(sr))
    M = M @ T((0, -0.035, 0)) @ T(R=Rz(sy))
    M = M @ T((0.015, 0, -0.3127)) @ T(R=Ry(ep))
    M = M @ T((-0.015, 0, 0)) @ T(R=Rz(wy))
    M = M @ T((0, 0.013, -0.25)) @ T(R=Rx(wr))
    return M @ T((0, -0.013, -0.031)) @ T(R=Rz(gy))


def test_phantom_right_arm_matches_the_hand_written_fk():
    c = armchain.ArmChain(URDF, armchain.RIGHT_ARM, base="Torso")
    for q in np.random.default_rng(2).uniform(-1.5, 1.5, (6, 7)):
        assert np.allclose(c.T_base_tip(q), _armfk_816(*q), atol=1e-12)
    assert c.tip == "RightGripperYawLink"
    assert np.isclose(c.lower[3], -2.35619) and np.isclose(c.upper[3], 0.174532925)


def test_camera_in_torso_composes_neck_head_and_camera():
    neck = neckfk.NeckChain(URDF)
    T_BS = rm.transform(rm.rot_axis((1, 2, 3), 0.4), (0.01, -0.05, 0.09))
    T_SC = rm.transform(rm.rot_axis((0, 1, 1), 1.2), (-0.06, 0.15, 0.02))
    got = armchain.camera_in_torso(neck, T_BS, T_SC, -0.007, 0.354)
    assert np.allclose(got, neck.T_TB(-0.007, 0.354) @ T_BS @ T_SC, atol=1e-14)
