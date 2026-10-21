"""T_BS steps: the neck chain read from the URDF.

phantom12's URDF has zero rpy and unit axes on both neck joints, which hides frame-convention bugs,
and the later tasks make their synthetic data with this same chain. So the chain is also checked
on a made-up neck with rotated origins and tilted, unnormalised axes, against independent
references."""
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "scripts"))
import neckfk  # noqa: E402
import rigmath as rm  # noqa: E402

URDF = ROOT / "config/foundation_rig/20250206_phantom_mk1.urdf"
ZUPT_GATE = ROOT / "tools/zupt_gate.py"   # mirrors the robot's C++ UrdfTfModel; not in Thor's copy
SYN = """<robot name="t">
  <joint name="NeckYaw" type="revolute">
    <origin rpy="0.07 -0.11 0.25" xyz="0.005 0.002 0.31"/>
    <axis xyz="0.05 -0.03 2"/>
    <parent link="Torso"/><child link="NeckYawLink"/>
  </joint>
  <joint name="NeckPitch" type="revolute">
    <origin rpy="0.3 -0.2 0.4" xyz="0.025 0.01 0.207"/>
    <axis xyz="0.1 1 -0.05"/>
    <parent link="NeckYawLink"/><child link="Head"/>
  </joint>
  <joint name="HeadImuMount" type="fixed">
    <origin rpy="0 0 0" xyz="0 0 0.05"/>
    <parent link="Head"/><child link="HeadImu"/>
  </joint>
</robot>"""


@pytest.fixture
def syn(tmp_path):
    p = tmp_path / "neck.urdf"
    p.write_text(SYN)
    return neckfk.NeckChain(p)


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def test_chain_reads_the_two_neck_joints():
    c = neckfk.NeckChain(URDF)
    assert c.yaw.parent == "Torso" and c.yaw.child == c.pitch.parent and c.pitch.child == "Head"
    assert np.allclose(c.yaw.axis, [0, 0, 1]) and np.allclose(c.pitch.axis, [0, 1, 0])


def test_zero_pose_is_the_sum_of_the_joint_offsets():
    T = neckfk.NeckChain(URDF).T_TB(0.0, 0.0)
    assert np.allclose(T[:3, :3], np.eye(3))
    assert np.allclose(T[:3, 3], [0.005 + 0.025, 0.0 + 0.01, 0.307953 + 0.207252])


def test_yaw_comes_before_pitch():
    """Yaw turns the pitch joint's offset about the torso's z; pitch turns about Head y."""
    c = neckfk.NeckChain(URDF)
    T = c.T_TB(0.4, 0.3)
    assert np.allclose(T[:3, 3], [0.005, 0, 0.307953] + rm.rot_axis([0, 0, 1], 0.4) @ [0.025, 0.01, 0.207252])
    assert np.allclose(T[:3, :3], rm.rot_axis([0, 0, 1], 0.4) @ rm.rot_axis([0, 1, 0], 0.3))
    assert np.allclose(c.T_TB(0, 0)[:3, :3].T @ c.T_TB(0, 0.3)[:3, :3], rm.rot_axis([0, 1, 0], 0.3))


def test_rpy_known_answer():
    """The ROS optical-frame rotation rpy = (-pi/2, 0, -pi/2): optical z -> parent x, x -> -y, y -> -z."""
    assert np.allclose(neckfk.rpy(-np.pi / 2, 0, -np.pi / 2), [[0, 0, 1], [-1, 0, 0], [0, -1, 0]])


def test_rotated_origins_compose_as_written(syn):
    """T_TB on the made-up neck, rebuilt from the literal URDF numbers: origin, then joint rotation."""
    expect = (rm.transform(neckfk.rpy(0.07, -0.11, 0.25), [0.005, 0.002, 0.31])
              @ rm.transform(rm.rot_axis(_unit([0.05, -0.03, 2]), 0.4), np.zeros(3))
              @ rm.transform(neckfk.rpy(0.3, -0.2, 0.4), [0.025, 0.01, 0.207])
              @ rm.transform(rm.rot_axis(_unit([0.1, 1, -0.05]), -0.3), np.zeros(3)))
    assert np.allclose(syn.T_TB(0.4, -0.3), expect, rtol=0, atol=1e-12)
    assert np.allclose(syn.yaw.axis, _unit([0.05, -0.03, 2])) and np.allclose(syn.pitch.axis, _unit([0.1, 1, -0.05]))


@pytest.mark.skipif(not ZUPT_GATE.exists(), reason="tools/zupt_gate.py not in this copy")
def test_matches_the_robot_side_urdf_fk(syn, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ZUPT_GATE.parent))
    import zupt_gate
    R, t = zupt_gate.UrdfChain(str(tmp_path / "neck.urdf")).pose_in_root(
        "Head", {"NeckYaw": 0.4, "NeckPitch": -0.3}, base="Torso")
    T = syn.T_TB(0.4, -0.3)
    assert np.allclose(T[:3, :3], np.array(R)) and np.allclose(T[:3, 3], t)


@pytest.mark.parametrize("urdf", ["real", "made-up"])
def test_omega_B_matches_a_numerical_derivative(urdf, syn):
    c = neckfk.NeckChain(URDF) if urdf == "real" else syn
    qy, qp, qdy, qdp, h = 0.2, -0.35, 0.7, -0.4, 1e-6
    R0 = c.T_TB(qy, qp)[:3, :3]
    R1 = c.T_TB(qy + qdy * h, qp + qdp * h)[:3, :3]
    W = R0.T @ (R1 - R0) / h                             # skew(omega_B)
    w_num = np.array([W[2, 1], W[0, 2], W[1, 0]])
    assert np.allclose(c.omega_B(qp, qdy, qdp), w_num, atol=1e-5)
    qps = np.array([-0.4, -0.1, 0.0, 0.3])               # different rows: the array path per row
    many = c.omega_B(qps, np.full(4, qdy), np.full(4, qdp))
    assert many.shape == (4, 3)
    for i, q in enumerate(qps):
        assert np.allclose(many[i], c.omega_B(q, qdy, qdp))


def test_yaw_axis_and_R_YB_with_rotated_origins(syn):
    qps = np.array([-0.4, -0.1, 0.0, 0.3])
    many = syn.yaw_axis_in_B(qps)
    for i, q in enumerate(qps):
        assert np.allclose(many[i], syn.R_YB(q).T @ syn.yaw.axis)
        assert np.allclose(syn.R_YB(q), rm.inv(syn.yaw.origin)[:3, :3] @ syn.T_TB(0, q)[:3, :3])


def test_axes_in_the_torso_frame_are_the_turns_axes(syn):
    """Turning one joint a little turns the head about that joint's axis, seen in torso axes."""
    qy, qp, d = 0.3, -0.2, 1e-6
    a_y, a_p = syn.axes_in_T(qy, qp)
    turn_y = rm.log_rot(syn.T_TB(qy + d, qp)[:3, :3] @ syn.T_TB(qy, qp)[:3, :3].T) / d
    turn_p = rm.log_rot(syn.T_TB(qy, qp + d)[:3, :3] @ syn.T_TB(qy, qp)[:3, :3].T) / d
    assert np.allclose(turn_y, a_y, atol=1e-5) and np.allclose(turn_p, a_p, atol=1e-5)
    assert not np.allclose(a_y, syn.yaw.axis, atol=1e-3)   # the origin's rotation matters here


def test_a_joint_that_does_not_rotate_is_refused_and_axes_are_unit(tmp_path):
    p = tmp_path / "neck.urdf"
    p.write_text(SYN)
    with pytest.raises(ValueError, match="fixed"):
        neckfk.load_joint(p, "HeadImuMount")
    j = neckfk.Joint("J", "A", "B", np.eye(4), np.array([0.0, 2.0, 0.0]))
    assert np.allclose(j.axis, [0, 1, 0])
    with pytest.raises(ValueError, match="zero axis"):
        neckfk.Joint("J", "A", "B", np.eye(4), np.zeros(3))
