"""urdf.py: chains equal the head calibration's neck FK; origin changes mean what the phantom14 fit meant; the writer
changes only those origins and its file's FK equals the in-memory model."""
import numpy as np
import pytest

from armcal import REPO_URDF
from armcal.urdf import Urdf, write_variant
import neckfk
from rigmath import rot_axis, transform

U = Urdf.load(REPO_URDF)
Q = {"RightShoulderPitch": -1.15, "RightShoulderRoll": -0.35, "RightShoulderYaw": 0.6, "RightElbowPitch": -1.17,
     "RightWristYaw": 1.0, "RightWristRoll": 0.36, "NeckYaw": 0.1, "NeckPitch": 0.3}


def test_the_neck_chain_is_neckfks():
    nc = neckfk.NeckChain(REPO_URDF)
    assert np.allclose(U.chain("Torso", "Head").T(Q), nc.T_TB(Q["NeckYaw"], Q["NeckPitch"]), atol=1e-12)


def test_at_zero_a_chain_is_the_product_of_its_origins():
    ch = U.chain("Torso", "RightWristRollLink")
    T = np.eye(4)
    for j in ch.steps:
        T = T @ j.origin
    assert np.allclose(ch.T({n: 0.0 for n in ch.joint_names}), T)
    assert ch.joint_names == ["RightShoulderPitch", "RightShoulderRoll", "RightShoulderYaw", "RightElbowPitch",
                              "RightWristYaw", "RightWristRoll"]


def test_a_missing_angle_and_a_link_not_below_the_root_are_named():
    with pytest.raises(KeyError, match="RightWristRoll"):
        U.chain("Torso", "RightWristRollLink").T({k: v for k, v in Q.items() if k != "RightWristRoll"})
    with pytest.raises(ValueError, match="not below"):
        U.chain("RightWristRollLink", "Torso")


def test_dxyz_moves_the_origin_in_the_parent_frame():
    V = U.changed({"NeckYaw": {"dxyz": [1.0, -2.0, 3.0]}})
    d = V.chain("Torso", "Head").T(Q)[:3, 3] - U.chain("Torso", "Head").T(Q)[:3, 3]
    assert np.allclose(d, [0.001, -0.002, 0.003], atol=1e-12)
    assert U.joint("NeckYaw").origin[0, 3] == pytest.approx(-0.015)        # the base is unchanged


def test_a_zero_offset_equals_adding_to_the_encoder():
    V = U.changed({"RightShoulderRoll": {"post_axis_deg": 1.5}})
    q2 = dict(Q, RightShoulderRoll=Q["RightShoulderRoll"] + np.radians(1.5))
    ch = "Torso", "RightWristRollLink"
    assert np.allclose(V.chain(*ch).T(Q), U.chain(*ch).T(q2), atol=1e-12)


def test_pre_rotates_in_the_parent_frame():
    V = U.changed({"NeckYaw": {"pre": [0.0, 0.0, 2.0]}})
    O = U.joint("NeckYaw").origin
    want = transform(rot_axis((0, 0, 1), np.radians(2.0)) @ O[:3, :3], O[:3, 3])
    assert np.allclose(V.joint("NeckYaw").origin, want)


def test_unknown_change_keys_and_fixed_zero_offsets_are_refused():
    with pytest.raises(ValueError, match="unknown"):
        U.changed({"NeckYaw": {"dz": 1.0}})
    fixed = next(j.name for j in U.joints.values() if not j.moves)
    with pytest.raises(ValueError, match="fixed"):
        U.changed({fixed: {"post_axis_deg": 1.0}})


def test_the_writer_changes_only_the_named_origins_and_says_where_it_came_from(tmp_path):
    ch = {"NeckYaw": {"dxyz": [-8.4, 14.9, -13.1]}, "RightShoulderRoll": {"post_axis_deg": -1.8},
          "LeftShoulderRoll": {"post_axis_deg": 0.48}}
    out = write_variant(U, ch, tmp_path / "x.urdf", ["session S -- test", "T_BS sha abc"])
    text = out.read_text()
    assert text.startswith("<?xml")
    assert U.md5 in text and "session S - - test" in text            # no "--" inside an XML comment
    W = Urdf.load(out)
    V = U.changed(ch)
    for name, j in U.joints.items():
        assert np.allclose(W.joint(name).origin, V.joint(name).origin, atol=1e-8), name
        if name not in ch:
            assert np.allclose(W.joint(name).origin, j.origin, atol=1e-12), name
    old, new = U.text.splitlines(), text.splitlines()
    changed = [line for line in new if line not in old and "<origin" in line]
    assert len(changed) == 3
