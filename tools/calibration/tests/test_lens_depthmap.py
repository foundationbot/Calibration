"""depthmap: one calibration's depth reading of another's scene, and the fitted KB4T's reading of the truth."""
import importlib
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("cv2")
pytest.importorskip("yaml")

sys.path.insert(0, str(Path(__file__).resolve().parent))
S = importlib.import_module("lens_synth")
dm = importlib.import_module("depthmap")
TRUTH = ("KB4T", [S.I0, S.I1], S.C_TRUE)
RES = (1280, 800)


def _rings(uv, pct, pp=(642.83, 384.95)):
    r = np.hypot(uv[:, 0] - pp[0], uv[:, 1] - pp[1])
    return {k: np.abs(pct[m]).max() for k, m in (("<=450", r <= 450), ("450-600", (r > 450) & (r <= 600)),
                                                  (">600", r > 600)) if m.any()}


def test_the_radius_limit_keeps_only_points_within_it():
    uv = np.array([[642.83, 384.95], [642.83 + 449.0, 384.95], [642.83 + 451.0, 384.95]])
    u2, p2 = dm.limit_radius(uv, np.array([1.0, 2.0, 3.0]), (642.83, 384.95), 450.0)
    assert p2.tolist() == [1.0, 2.0] and len(u2) == 2


def test_a_calibration_reads_its_own_scene_exactly():
    uv, pct = dm.depth_diff_map(TRUTH, TRUTH, RES)
    assert len(uv) > 500 and np.abs(pct).max() < 1e-6


def test_a_baseline_one_percent_long_reads_one_percent_far_everywhere():
    Cb = S.C_TRUE.copy()
    Cb[:3, 3] *= 1.01
    _, pct = dm.depth_diff_map(TRUTH, ("KB4T", [S.I0, S.I1], Cb), RES)
    assert np.allclose(pct, 1.0, atol=1e-6)


def test_the_cells_cover_the_whole_image():
    uv, pct = dm.depth_diff_map(TRUTH, TRUTH, RES)
    assert set(dm.by_cell(uv, pct, (642.83, 384.95))) == set(dm.lenscells.CELLS)


def test_the_nominal_start_fit_reads_the_truth_within_tolerance():
    """measured 20 Oct (48 holds, 0.3 px): up to 0.06-0.16 % within 450 px, 0.18-0.27 % to 600 px, 0.6-0.8 % beyond
    (the far corners, where a sideways ray sees a foreshortened baseline): +-0.1 % is below this noise floor"""
    _, r = S.nominal_fit("KB4T")
    uv, pct = dm.depth_diff_map(TRUTH, ("KB4T", r["intr"], r["C"]), RES)
    rings = _rings(uv, pct)
    assert rings["<=450"] < 0.3 and rings["450-600"] < 0.6 and rings[">600"] < 1.5, rings


def test_mutation_p1_p2_forced_to_zero_is_caught():
    """KB4 = KB4T with p1 = p2 = 0. Refitted from the nominal start on the same corners, it absorbs the tangential
    terms into the principal point and the stereo pose and reads depth wrong (20 Oct scratch run: 3.4 % at the
    image centre)"""
    _, r4 = S.nominal_fit("KB4")
    _, r = S.nominal_fit("KB4T")
    assert r4["success"]
    uv, p4 = dm.depth_diff_map(TRUTH, ("KB4", r4["intr"], r4["C"]), RES)
    _, pt = dm.depth_diff_map(TRUTH, ("KB4T", r["intr"], r["C"]), RES)
    near = np.hypot(uv[:, 0] - 642.83, uv[:, 1] - 384.95) <= 450
    assert np.abs(p4[near]).max() > 0.3
    assert np.abs(p4[near]).max() > 3 * np.abs(pt[near]).max()


def _turnover():
    """Fit A with camera 0's k1 = -0.2: its KB4 polynomial turns over ~490 px from the principal point, inside the
    image (as step 71's drop-cost test). Beyond the turn no ray exists, and unproject returns a finite wrong one"""
    lens = S.I0.copy()
    lens[4] = -0.2
    return ("KB4T", [lens, S.I1], S.C_TRUE)


def test_a_lens_that_turns_over_in_the_image_reads_its_own_scene_where_it_has_rays():
    """a grid point counts only if camera 0's ray projects back to it within 1e-3 px; the others are left out and
    counted (145 of the 640 at 40 px). Before this guard, 33 of them were also in front of the camera and inside
    camera 1, and the calibration's reading of its own scene there (exactly 0 where a ray exists) hit a singular
    matrix at 14 (LinAlgError for the whole map) and read up to 4e12 % at the other 19"""
    cal = _turnover()
    uv, pct = dm.depth_diff_map(cal, cal, RES)
    assert len(uv) > 300 and np.abs(pct).max() < 1e-6
    Cb = S.C_TRUE.copy()
    Cb[:3, 3] *= 1.01
    _, pct = dm.depth_diff_map(cal, (cal[0], cal[1], Cb), RES)
    assert np.allclose(pct, 1.0, atol=1e-6)
    assert dm.points_without_ray(cal, RES) == 145


def test_where_the_second_calibration_has_no_ray_the_reading_is_infinite_and_its_cell_flagged():
    """b's camera 1 turns over inside the image (step 46's half fits can): b cannot read a's scene at those pixels.
    Before the guard they went into triangulate with b's wrong rays: NaN or a LinAlgError, and by_cell's NaN maximum
    passed every gate"""
    lens = S.I1.copy()
    lens[4] = -0.2
    uv, pct = dm.depth_diff_map(TRUTH, ("KB4T", [S.I0, lens], S.C_TRUE), RES)
    assert not np.isnan(pct).any() and np.isinf(pct).sum() > 10 and np.isfinite(pct).sum() > 300
    cells = dm.by_cell(uv, pct, S.I0[2:4])
    lab = dm.lenscells.cell_of(uv[:, 0] - S.I0[2], uv[:, 1] - S.I0[3])
    for c in set(lab[np.isinf(pct)]):
        assert np.isinf(cells[c]["max_abs_pct"]) and cells[c]["max_abs_pct"] > 0.5, c      # beyond any gate


def test_fit_a_has_a_ray_at_every_grid_point():
    """the guard leaves Fit A's maps as they were: no grid point of either camera (as camera 0) fails the round trip"""
    assert dm.points_without_ray(TRUTH, RES) == 0
    assert dm.points_without_ray(("KB4T", [S.I1, S.I0], S.C_TRUE), RES) == 0


def test_the_command_line_names_the_points_without_a_ray(tmp_path, capsys):
    a = S.write_calibration(tmp_path / "a.json", *_turnover())
    assert dm.main([str(a), str(a), "--rig", str(S.RIG12)]) == 0
    assert "145 of 640 grid points left out" in capsys.readouterr().out
    t = S.write_calibration(tmp_path / "t.json", *TRUTH)
    assert dm.main([str(t), str(t), "--rig", str(S.RIG12)]) == 0
    assert "left out" not in capsys.readouterr().out
