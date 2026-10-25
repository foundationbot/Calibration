"""Arm session, Task 2: the MJCF clearance and gravity checks (armsafety.py, decisions T2-4 and T2-5). MuJoCo is
optional (not in CI's calibration job): these tests skip without it."""
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("mujoco")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
import armsafety  # noqa: E402

# a post on a floating base, and a one-joint arm 0.3 m from it: capsule radii 0.05 and 0.02
XML = """<mujoco>
  <worldbody>
    <body name="Base" pos="0 0 2">
      <freejoint name="root"/>
      <geom name="post" type="capsule" size="0.05" fromto="0 0 0 0 0 0.5" mass="1"/>
      <body name="Arm" pos="0.3 0 0.5">
        <joint name="J1" axis="0 1 0" range="-3 3"/>
        <geom name="link" type="capsule" size="0.02" fromto="0 0 0 0 0 -0.3" mass="2"/>
        <geom name="ghost" type="sphere" size="0.01" contype="0" conaffinity="0" mass="0.001"/>
      </body>
    </body>
  </worldbody>
</mujoco>"""
M_L = 2 * 9.81 * 0.15        # the link's weight times its centre's distance from the joint


@pytest.fixture
def body(tmp_path):
    p = tmp_path / "t.xml"
    p.write_text(XML)
    return armsafety.Body(p, root="Arm", joints=("J1",))


def test_clearance_is_the_capsules_gap(body):
    assert np.isclose(body.clearance(np.array([[0.0]]))[0], 0.3 - 0.05 - 0.02, atol=1e-6)


def test_clearance_goes_negative_through_the_post_and_names_the_pair(body):
    d = body.clearance(np.array([[0.0], [np.pi / 2]]))
    assert d[1] < 0 < d[0]
    assert body.closest(np.array([np.pi / 2])) == ("link", "post")


def test_geoms_without_contacts_are_ignored(body):
    assert "ghost" not in body.arm_geoms and "ghost" not in body.other_geoms


def test_gravity_is_the_torque_to_hold_the_arm(body):
    a = np.array([[0.3], [-0.8]])
    assert np.allclose(body.gravity(a, (0, 0, -1))[:, 0], M_L * np.sin(a[:, 0]), atol=1e-6)


def test_gravity_follows_a_tilted_base(body):
    """The base pitched by b about the joint's axis: gravity in the base is Ry(b)^T (0, 0, -1) and the arm hangs
    as if its angle were a + b."""
    b, a = 0.2, 0.3
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    got = body.gravity(np.array([[a]]), Ry.T @ np.array([0, 0, -1.0]))[0, 0]
    assert np.isclose(got, M_L * np.sin(a + b), atol=1e-6)


def test_an_unknown_joint_is_refused(tmp_path):
    p = tmp_path / "t.xml"
    p.write_text(XML)
    with pytest.raises(KeyError, match="J9"):
        armsafety.Body(p, root="Arm", joints=("J9",))


SIGHT = """<mujoco>
  <worldbody>
    <geom name="blocker" type="sphere" size="0.05" pos="0.5 0 0" group="2" contype="0" conaffinity="0"/>
    <geom name="hidden" type="sphere" size="0.05" pos="-0.5 0 0" group="1" contype="0" conaffinity="0"/>
    <body name="Base">
      <freejoint name="root"/>
      <geom name="eye" type="sphere" size="0.05" group="2" contype="0" conaffinity="0" mass="1"/>
      <body name="Arm" pos="0 0 0">
        <joint name="J1" axis="0 0 1" range="-3 3"/>
        <geom name="palm" type="box" size="0.01 0.05 0.05" pos="1.0 0 0" group="2" contype="0" conaffinity="0"
              mass="0.1"/>
      </body>
    </body>
  </worldbody>
</mujoco>"""


def test_sightline_is_blocked_by_a_body_part_but_not_by_see_through_ones(tmp_path):
    """From the origin (inside 'eye', see-through) to the near face of 'palm' (x = 0.99): with J1 = 0 a fixed blocker
    sits on the line; with J1 = 90 deg the palm and the line turn to +y, clear. Only visual geoms (group 2) block:
    a group-1 sphere on the -x line does not."""
    p = tmp_path / "s.xml"
    p.write_text(SIGHT)
    b = armsafety.Body(p, root="Arm", joints=("J1",))
    eye = np.zeros(3)
    assert not b.sightline_clear(np.array([0.0]), eye, np.array([[0.985, 0.0, 0.0]]), see_through=("Base",))
    assert b.sightline_clear(np.array([np.pi / 2]), eye, np.array([[0.0, 0.985, 0.0]]), see_through=("Base",))
    assert b.sightline_clear(np.array([0.0]), eye, np.array([[0.4, 0.0, 0.0]]), see_through=("Base",))
    assert b.sightline_clear(np.array([0.0]), eye, np.array([[-0.9, 0.0, 0.0]]), see_through=("Base",))
    assert not b.sightline_clear(np.array([0.0]), eye, np.array([[0.4, 0.0, 0.0], [0.985, 0.0, 0.0]]),
                                 see_through=("Base",))


CUBES = """<mujoco>
  <compiler angle="radian"/>
  <asset>
    <mesh name="cube" vertex="-0.05 -0.05 -0.05  0.05 -0.05 -0.05  0.05 0.05 -0.05  -0.05 0.05 -0.05
                               -0.05 -0.05 0.05  0.05 -0.05 0.05  0.05 0.05 0.05  -0.05 0.05 0.05"/>
  </asset>
  <worldbody>
    <body name="Base">
      <freejoint name="root"/>
      <geom name="wall" type="mesh" mesh="cube" group="2" contype="0" conaffinity="0" mass="1"/>
      <body name="Mount" pos="0 0 0">
        <joint name="J0" axis="0 0 1" range="-3 3"/>
        <geom name="mount" type="mesh" mesh="cube" pos="0.06 0 0" group="2" contype="0" conaffinity="0" mass="0.1"/>
        <body name="Arm" pos="0 0 0">
          <joint name="J1" type="slide" axis="1 0 0" range="0 1"/>
          <geom name="hand" type="mesh" mesh="cube" pos="0.3 0 0" group="2" contype="0" conaffinity="0" mass="0.1"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>"""


def test_surface_clearance_is_the_gap_between_mesh_surfaces_skipping_the_mount(tmp_path):
    """Two 10 cm cubes, 'hand' slid along x: the faces are 0.3 - 0.1 + J1 apart. 'mount' touches 'wall' (0.01 m
    apart) but is skipped, as the shoulder's first link is on the robot."""
    p = tmp_path / "c.xml"
    p.write_text(CUBES)
    b = armsafety.Body(p, root="Mount", joints=("J0", "J1"), surface_step=0.004, skip_surface=("Mount",),
                       surface_cap=0.5)
    d = b.surface_clearance(np.array([[0.0, 0.0], [0.0, 0.15]]))
    assert np.allclose(d, [0.2, 0.35], atol=0.004)
    capped = armsafety.Body(p, root="Mount", joints=("J0", "J1"), surface_step=0.004, skip_surface=("Mount",))
    assert np.allclose(capped.surface_clearance(np.array([[0.0, 0.0], [0.0, 0.15]])), [0.1, 0.1])
    assert np.isclose(b.clearance(np.array([[0.0, 0.0]]))[0], 0.2, atol=0.004)   # no contact geoms: the surfaces
    b2 = armsafety.Body(p, root="Mount", joints=("J0", "J1"), surface_step=0.004)
    assert b2.surface_clearance(np.array([[0.0, 0.0]]))[0] < 0.012


def test_joint_ranges_come_from_the_model(tmp_path):
    p = tmp_path / "c.xml"
    p.write_text(CUBES)
    lo, hi = armsafety.Body(p, root="Mount", joints=("J0", "J1")).ranges()
    assert np.allclose(lo, [-3, 0]) and np.allclose(hi, [3, 1])
