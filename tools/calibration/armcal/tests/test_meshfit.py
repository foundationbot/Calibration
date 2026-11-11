"""meshfit.py (B0): a known correction of the arm in the camera is found back from a synthetic depth image, for both
arms; the fast computation equals the reference one where it claims to be exact."""
import numpy as np
import pytest

import synth
from armcal import meshfit

pytest.importorskip("cv2")
pytest.importorskip("scipy")
K, W, H = synth.K, synth.W, synth.H


def fit(side, E_true, cfg=meshfit.FAST, voxel=0.001, seed=0):
    q, _, _, T_TC = synth.pose(side)
    depth = synth.depth_image(side, E_true, seed=seed)
    return meshfit.fit_pose(synth.arm(side, voxel), q, T_TC, meshfit.unproject(depth, K), K, W, H, cfg), T_TC, q


@pytest.mark.parametrize("side", ["Right", "Left"])
def test_the_arm_is_in_view_at_the_stored_pose(side):
    q, _, _, T_TC = synth.pose(side)
    v = meshfit.view_gate(synth.arm(side), q, T_TC, K, W, H)
    assert v["ok"], v["why"]


@pytest.mark.parametrize("side", ["Right", "Left"])
def test_a_5mm_2deg_correction_is_found(side):
    E_true = synth.rigid(2.0, (1, 2, 0.5), (4.0, -3.0, 1.0))
    f, T_TC, q = fit(side, E_true)
    assert f.ok and f.converged
    assert meshfit.angle_deg(np.linalg.inv(E_true) @ f.delta_c) < 0.3
    want = meshfit.wrist_offset(synth.arm(side), q, T_TC, E_true)
    assert np.linalg.norm(f.wrist_offset_T - want) < 0.0007
    assert f.coverage > 0.6


def test_no_correction_reads_under_a_millimetre():
    """the synthetic depth is a nearest-pixel splat at 640x400: up to half a pixel (~1.3 mm at 0.6 m) of its own bias"""
    f, _, _ = fit("Right", np.eye(4))
    assert np.linalg.norm(f.wrist_offset_T) < 0.001 and f.rotation_deg < 0.3


def test_the_background_never_pairs():
    f, _, _ = fit("Right", np.eye(4))
    assert np.all(f.data[f.inlier][:, 2] < 1.3)          # the wall is at 1.4 m


def test_exact_speedups_equal_the_reference():
    """per-stage KD-tree, one z-buffer per stage, coverage by index: the same numbers as the phantom14 algorithm"""
    E_true = synth.rigid(1.0, (0, 1, 1), (2.0, 1.0, -2.0))
    a, _, _ = fit("Right", E_true, meshfit.REFERENCE, voxel=0.002)
    b, _, _ = fit("Right", E_true, meshfit.EXACT, voxel=0.002)
    assert b.steps == a.steps
    assert np.allclose(a.delta_c, b.delta_c, atol=1e-9)
    assert b.coverage == pytest.approx(a.coverage, abs=1e-12)
    assert b.rms_mm == pytest.approx(a.rms_mm, rel=1e-9)


def test_a_far_off_start_is_not_called_ok():
    E_true = synth.rigid(40.0, (1, 0, 0), (60.0, 0.0, 0.0))
    f, _, _ = fit("Right", E_true)
    assert not (f.ok and f.rotation_deg < 12 and f.coverage >= 0.6)


def test_model_voxel_thins_the_points_per_link():
    full, thin = synth.arm("Right", 0.0005), synth.arm("Right", 0.002)
    assert len(thin.P) < len(full.P) / 4
    assert set(np.unique(thin.link_id)) == {0, 1, 2}
