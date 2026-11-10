"""calibtool.score: lensscore.score's contract on the fast solver (M3), and the CALIB_SOLVER switch of stereofit.solve
and lensscore.score (default: scipy, unchanged)."""
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


# ---------------------------------------------------------------- the switch


def _spy(monkeypatch):
    seen = {"scipy": 0, "fast": 0, "fast_score": 0}
    real_ls, real_solve, real_score = sf.least_squares, solver.solve, score.score

    def ls_(*a, **k):
        seen["scipy"] += 1
        return real_ls(*a, **k)

    def solve_(*a, **k):
        seen["fast"] += 1
        return real_solve(*a, **k)

    def score_(*a, **k):
        seen["fast_score"] += 1
        return real_score(*a, **k)
    monkeypatch.setattr(sf, "least_squares", ls_)
    monkeypatch.setattr(solver, "solve", solve_)
    monkeypatch.setattr(score, "score", score_)
    return seen


def test_default_is_scipy(monkeypatch):
    monkeypatch.delenv(solver.ENV, raising=False)
    seen = _spy(monkeypatch)
    ls.score("KB4T", [S.I0, S.I1], S.C_TRUE, _data())
    assert seen["scipy"] > 0 and seen["fast"] == 0 and seen["fast_score"] == 0
    monkeypatch.setenv(solver.ENV, "scipy")
    ls.score("KB4T", [S.I0, S.I1], S.C_TRUE, _data())
    assert seen["fast"] == 0 and seen["fast_score"] == 0


def test_fast_switch_routes_solve_and_score(monkeypatch):
    monkeypatch.setenv(solver.ENV, "fast")
    seen = _spy(monkeypatch)
    data = _data()
    poses, ok = sf.pnp_poses(data, "KB4T", [S.I0, S.I1], S.C_TRUE)
    r = sf.solve(data, "KB4T", [S.I0, S.I1], S.C_TRUE, poses, free_intr=False, free_C=False)
    assert r["success"] and seen == {"scipy": 0, "fast": 1, "fast_score": 0}
    ls.score("KB4T", [S.I0, S.I1], S.C_TRUE, data)
    assert seen["scipy"] == 0 and seen["fast_score"] == 1


def test_fast_switch_keeps_scipy_for_free_intrinsics_of_other_models(monkeypatch):
    from calibtool import livecfg
    monkeypatch.setenv(solver.ENV, "fast")
    seen = _spy(monkeypatch)
    data = _data()
    rt = [np.r_[565.0, 564.0, 642.0, 385.0, -0.01, 0.002, 0.0005, -0.0003]] * 2
    with livecfg.radtan_scope():
        poses, ok = sf.pnp_poses(data, "RADTAN", rt, S.C_TRUE)
        d = data.select(ok)
        sf.solve(d, "RADTAN", rt, S.C_TRUE, poses[ok], max_nfev=3, rounds=1)
    assert seen["scipy"] == 1 and seen["fast"] == 0


def test_under_the_switch_scoring_solves_still_go_through_stereofit_solve(monkeypatch):
    """step 66's NO RESULT guards and lensscore's restart tests patch stereofit.solve: under CALIB_SOLVER=fast the
    fast scorer's solves must pass through it too (they then reach calibtool.solver by stereofit's hook)"""
    monkeypatch.setenv(solver.ENV, "fast")
    seen = _spy(monkeypatch)
    calls, real = [], sf.solve

    def spy(*a, **k):
        calls.append(k["scale"])
        return real(*a, **k)
    monkeypatch.setattr(sf, "solve", spy)
    ls.score("KB4T", [S.I0, S.I1], S.C_TRUE, _data())
    assert calls == [False, True] and seen["fast"] == 2 and seen["scipy"] == 0 and seen["fast_score"] == 1
