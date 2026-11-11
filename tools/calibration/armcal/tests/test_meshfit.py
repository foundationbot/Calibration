"""meshfit.py (B0): a known correction of the arm in the camera is found back from a synthetic depth image, for both
arms; the fast computation equals the reference one where it claims to be exact."""
import numpy as np
import pytest

import synth
from armcal import meshfit, readb

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


# ------------------------------------------------------------------------------------- the robust path (readb2)
def test_even_sampling_is_even_on_a_long_thin_triangle():
    from armcal import mesh
    V = np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.002, 0]], float)      # 100 x 2 mm: area 1e-4 m^2
    F = np.array([[0, 1, 2]])
    P, N, w = mesh.sample_even(V, F, 0.001)
    assert len(P) == 100                                                # one per mm^2 (grid: ceil(100)^2 = 10000)
    assert len(mesh.sample(V, F, 0.001)[0]) >= 10000
    assert w.sum() == pytest.approx(1e-4)
    assert np.all(P[:, 0] >= 0) and np.all(P[:, 1] >= 0) and np.all(P[:, 0] / 0.1 + P[:, 1] / 0.002 <= 1 + 1e-12)
    assert np.allclose(N, [0, 0, 1])
    Q, _, _ = mesh.sample_even(V, F, 0.001, phase=0.25)
    assert len(Q) == len(P) and not np.allclose(P, Q)
    assert np.array_equal(P, mesh.sample_even(V, F, 0.001)[0])         # deterministic


def test_even_model_is_twenty_times_lighter():
    from armcal import MESHES, mesh
    n_grid = 0
    for n in readb.LINKS:              # what mesh.sample would make: ceil(longest edge / spacing)^2 per triangle
        V, F = mesh.load_obj(MESHES / f"{n}.obj")
        e = np.stack([np.linalg.norm(V[F[:, i]] - V[F[:, (i + 1) % 3]], axis=1) for i in range(3)]).max(0)
        n_grid += int((np.maximum(1, np.ceil(e / 0.001 - 1e-9)) ** 2).sum())
    even = _even("Right")
    assert len(even.P) < n_grid / 15
    assert set(np.unique(even.link_id)) == {0, 1, 2}


def test_fill_holes_closes_see_through_pinholes_only():
    from armcal import mesh
    d = np.zeros((9, 9))
    d[2:7, 2:7] = 0.5
    d[4, 4] = 0.6                     # a pinhole: the back surface shows through
    d[3, 3] = 0.49
    d[2:7, 6] = 0.8                   # a depth edge: the column behind the front surface
    f = mesh.fill_holes(d)
    assert f[4, 4] == pytest.approx(0.49)
    assert np.array_equal(f[:, 6], d[:, 6]) and np.array_equal(f[0], d[0])
    assert (f == d).sum() == d.size - 1


def robust_fit(side, E_true, cfg=None, seed=0, **kw):
    cfg = cfg or meshfit.ROBUST
    q, _, _, T_TC = synth.pose(side)
    depth = synth.depth_image(side, E_true, seed=seed)
    arm = _even(side, **kw)
    return meshfit.fit_pose(arm, q, T_TC, meshfit.unproject(depth, K), K, W, H, cfg), T_TC, q, arm


_EVEN = {}


def _even(side, **kw):
    key = (side, tuple(sorted(kw.items())))
    if key not in _EVEN:
        _EVEN[key] = readb.arm_model(synth.URDF, side, cfg=meshfit.ROBUST, **kw)
    return _EVEN[key]


@pytest.mark.parametrize("side", ["Right", "Left"])
def test_robust_finds_a_5mm_2deg_correction(side):
    """the pull against the forearm's roll keeps part of a true roll out of the answer (here up to 0.7 deg of 2):
    the wrist point, which sits near that axis, moves less than a millimetre"""
    E_true = synth.rigid(2.0, (1, 2, 0.5), (4.0, -3.0, 1.0))
    f, T_TC, q, arm = robust_fit(side, E_true)
    assert f.ok and f.converged
    assert meshfit.angle_deg(np.linalg.inv(E_true) @ f.delta_c) < 1.0
    want = meshfit.wrist_offset(arm, q, T_TC, E_true)
    assert np.linalg.norm(f.wrist_offset_T - want) < 0.001
    assert f.coverage > 0.6 and len(f.candidates) == 1 + len(meshfit.ROBUST.starts)


@pytest.mark.parametrize("side", ["Right", "Left"])
def test_robust_without_the_roll_pull_is_exact(side):
    E_true = synth.rigid(2.0, (1, 2, 0.5), (4.0, -3.0, 1.0))
    f, T_TC, q, arm = robust_fit(side, E_true, meshfit.with_config(meshfit.ROBUST, rot_prior=0.0))
    assert f.ok and meshfit.angle_deg(np.linalg.inv(E_true) @ f.delta_c) < 0.3
    assert np.linalg.norm(f.wrist_offset_T - meshfit.wrist_offset(arm, q, T_TC, E_true)) < 0.0007


def test_robust_finds_a_20mm_correction():
    """the URDF is typically 20-35 mm off at the wrist: the translation-only coarse stages bring it in"""
    E_true = synth.rigid(2.0, (0, 1, 1), (15.0, -12.0, 6.0))
    f, T_TC, q, arm = robust_fit("Right", E_true)
    want = meshfit.wrist_offset(arm, q, T_TC, E_true)
    assert f.ok and np.linalg.norm(f.wrist_offset_T - want) < 0.001


def test_robust_does_not_care_which_points_sample_the_mesh():
    E_true = synth.rigid(1.5, (1, 0, 1), (3.0, 2.0, -2.0))
    a, *_ = robust_fit("Left", E_true)
    b, *_ = robust_fit("Left", E_true, phase=0.25)
    c, *_ = robust_fit("Left", E_true, jitter=0.0003, seed=1)
    assert np.linalg.norm(a.wrist_offset_T - b.wrist_offset_T) < 0.0003
    assert np.linalg.norm(a.wrist_offset_T - c.wrist_offset_T) < 0.0003


def test_the_line_search_never_raises_a_stage_cost():
    q, _, _, T_TC = synth.pose("Right")
    depth = synth.depth_image("Right", synth.rigid(2.0, (1, 2, 0.5), (4.0, -3.0, 1.0)))
    tr = []
    meshfit._solve2(_even("Right"), q, T_TC, meshfit.unproject(depth, K), K, W, H, meshfit.ROBUST, trace=tr)
    for (g0, _, c0, _), (g1, _, c1, _) in zip(tr, tr[1:]):
        if g0 == g1:
            assert c1 <= c0


def test_robust_far_off_start_is_not_called_ok():
    E_true = synth.rigid(40.0, (1, 0, 0), (60.0, 0.0, 0.0))
    f, *_ = robust_fit("Right", E_true)
    assert not (f.ok and f.rotation_deg < 12 and f.coverage >= 0.6)
