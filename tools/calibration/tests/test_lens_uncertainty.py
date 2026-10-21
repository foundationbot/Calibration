"""Step 46: split-half sigmas (labelled one-sample) and the repeatability map's cells and flags."""
import importlib
import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("cv2")
pytest.importorskip("yaml")

sys.path.insert(0, str(Path(__file__).resolve().parent))
S = importlib.import_module("lens_synth")
m46 = importlib.import_module("46_lens_uncertainty")


def test_halves_split_the_kept_holds_by_rank():
    u = json.loads((S.pipeline() / "unc.json").read_text())
    fit = json.loads((S.pipeline() / "fit.json").read_text())
    a, b = (set(h["holds"]) for h in u["halves"])
    assert not a & b and sorted(a | b) == fit["kept_holds"]
    assert u["sigma_kind"].startswith("one-sample")


def test_sigmas_are_half_the_split_difference_and_plausible():
    u = json.loads((S.pipeline() / "unc.json").read_text())
    h0, h1 = u["halves"]
    assert u["sigma"]["intr"][0][0] == pytest.approx(abs(h0["intr"][0][0] - h1["intr"][0][0]) / 2)
    assert 0 < u["sigma"]["baseline_mm"] < 0.2              # 24 synthetic holds a half: a fraction of 0.1 mm
    assert max(u["sigma"]["intr"][0][:4]) < 1.5              # px


def test_repeatability_cells_and_flags():
    u = json.loads((S.pipeline() / "unc.json").read_text())
    cells = u["repeat_cells"]
    assert set(cells) == set(importlib.import_module("lenscells").CELLS)
    assert cells["centre"]["max_abs_pct"] < 0.5 and "centre" not in u["unconstrained_cells"]
    assert u["unconstrained_cells"] == [c for c, v in cells.items() if v["max_abs_pct"] > 0.5]


def test_identical_halves_flag_nothing_and_a_one_percent_baseline_flags_everything():
    dm = importlib.import_module("depthmap")
    cal = ("KB4T", [S.I0, S.I1], S.C_TRUE)
    C2 = S.C_TRUE.copy()
    C2[:3, 3] *= 1.01
    for other, want in ((cal, 0), (("KB4T", [S.I0, S.I1], C2), 9)):
        uv, pct = dm.depth_diff_map(cal, other, (1280, 800))
        assert sum(v["max_abs_pct"] > 0.5 for v in dm.by_cell(uv, pct, S.I0[2:4]).values()) == want


def test_a_fit_report_of_another_corner_file_is_refused(tmp_path):
    other = tmp_path / "other.npz"
    np.savez_compressed(other, **S.holdout_sweep())
    with pytest.raises(SystemExit) as e:
        m46.main([str(other), str(S.pipeline() / "fit.json"), str(tmp_path / "u.json"), "--rig", str(S.RIG12)])
    assert "not the corner file" in str(e.value)


def test_the_pipeline_files_are_strict_json():
    def refuse(c):
        raise ValueError(f"non-finite number {c} in the JSON")
    for name in ("fit.json", "unc.json"):
        json.loads((S.pipeline() / name).read_text(), parse_constant=refuse)
