"""calibtool.score: lensscore.score's contract on the fast solver (M3)."""
import importlib
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("cv2")
pytest.importorskip("yaml")

CAL = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CAL / "tests"))
S = importlib.import_module("lens_synth")
sf = S.sf
ls = importlib.import_module("lensscore")
from calibtool import score, solver  # noqa: E402


def _data():
    return sf.Data(S.holdout_sweep(), S.BOARD)


def _close_to_the_reference(got, want, typical, worst):
    """per-pair numbers that agree with lensscore's to `typical` in the median and to `worst` on every pair.
    Why not every pair at `typical`: scipy stops early on pairs whose pose and board scale are nearly degenerate
    (measured 9 Nov on holdout_sweep with the baseline x 1.004, pair 28: scale off the fast result by 8.0e-5 at scipy's
    defaults, 3.9e-5 at tol 1e-10, 1.7e-5 at 1e-12 where it stops unconverged at 2000 evaluations) while the fast
    solve's cost is lower on every pair (a RADTAN config on the centre holds likewise: the 95th percentile of the
    scale difference 8.6e-6, the largest 9.8e-6, at scipy's defaults) and the fast solve started from scipy's result
    returns to its own (pinned by test_solver.test_with_the_camera_fixed_each_pair_is_a_minimum_at_or_below_scipys)"""
    d = np.abs(np.asarray(got) - np.asarray(want))
    assert np.median(d) < typical and d.max() < worst, (np.median(d), d.max())


def test_self_check_passes_with_the_fast_scorer():
    data = _data()
    rec, _, _ = score.score("KB4T", [S.I0, S.I1], S.C_TRUE, data)
    chk = ls.self_check("KB4T", [S.I0, S.I1], S.C_TRUE, data, rec, scorer=score.score)
    assert chk["passed"] and all(abs(b["median_pct"] - 1.0) < 0.02 for b in chk["bins"] if b["pairs"])


@pytest.mark.parametrize("stops", [1, 2])
def test_a_scoring_solve_that_stopped_short_is_restarted_once_from_its_result(monkeypatch, stops):
    """lensscore._solve_fixed's contract: stereofit.MAX_NFEV a round, one restart from the result, no third try"""
    data = _data()
    real = solver.solve
    calls = []

    def capped(d, model, intr, C, poses0, **k):
        assert k["max_nfev"] == sf.MAX_NFEV and not k["free_intr"] and not k["free_C"]
        before = [c for c in calls if c["scale"] == k["scale"]]
        if before:
            assert np.array_equal(poses0, before[-1]["r"]["poses"])
            if k["scale"]:
                assert np.array_equal(k["ls0"], before[-1]["r"]["log_scale"])
        r = real(d, model, intr, C, poses0, **k)
        calls.append({"scale": k["scale"], "r": r})
        return {**r, "success": len(before) >= stops}
    monkeypatch.setattr(solver, "solve", capped)
    rec, _, solves = score.score("KB4T", [S.I0, S.I1], S.C_TRUE, data)
    assert [c["scale"] for c in calls] == [False, False, True, True]
    assert ls.solved(solves) is (stops == 1)


def test_radtan_live_configs_score_on_the_fast_path():
    """a RADTAN live config (point derivative by central differences) scores as lensscore does at its own tolerances
    (at tol 1e-12 lensscore's pose solve stops unconverged at its cap: measured 9 Nov, 190 s)"""
    from calibtool import livecfg
    data = _data()
    data = data.select(data.hold < 12)                  # the centre target: where radtan_like fits the lens
    rt = list(S.radtan_like())
    with livecfg.radtan_scope():
        rec, _, solves = score.score("RADTAN", rt, S.C_TRUE, data)
        rrec, _, rsolves = ls.score("RADTAN", rt, S.C_TRUE, data)
    assert solves == rsolves and rec["src"] == rrec["src"] and ls.solved(solves)
    _close_to_the_reference(rec["scale"], rrec["scale"], 1e-6, 1e-4)
