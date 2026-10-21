"""lensscore: the depth scorer, its self-check, the cell tables and the fixed pass rule."""
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
ls = importlib.import_module("lensscore")
GATES = S.rig12().lens_gates()
SOLVED = {"pose_success": True, "scale_success": True, "pnp_failed_pairs": 0}


def _test_data():
    return S.sf.Data(S.holdout_sweep(), S.BOARD)


def test_the_true_calibration_reads_zero_and_a_long_baseline_reads_plus_one_percent():
    data = _test_data()
    rec, hrms, _ = ls.score("KB4T", [S.I0, S.I1], S.C_TRUE, data)
    assert len(rec["scale"]) == data.n and abs(100 * (np.median(rec["scale"]) - 1)) < 0.1
    assert max(hrms.values()) < 0.6
    chk = ls.self_check("KB4T", [S.I0, S.I1], S.C_TRUE, data, rec)
    assert chk["passed"] and sum(b["pairs"] > 0 for b in chk["bins"]) >= 2
    assert all(abs(b["median_pct"] - 1.0) < 0.02 for b in chk["bins"] if b["pairs"])


def test_the_self_check_catches_a_scorer_that_ignores_the_baseline():
    data = _test_data()
    rec, _, _ = ls.score("KB4T", [S.I0, S.I1], S.C_TRUE, data)

    def deaf(model, intr, C, data):                       # rescales any baseline back to the true length
        C = np.array(C, float, copy=True)
        C[:3, 3] *= np.linalg.norm(S.C_TRUE[:3, 3]) / np.linalg.norm(C[:3, 3])
        return ls.score(model, intr, C, data)
    assert not ls.self_check("KB4T", [S.I0, S.I1], S.C_TRUE, data, rec, scorer=deaf)["passed"]


def test_the_self_check_rule_is_plus_one_percent_within_a_tenth_in_every_bin_with_data():
    """the rule alone, with a scorer that reads chosen ratios (no fit): 0.92 and 1.08 pass, 0.88 or 1.12 in one bin
    fails, and records in no distance bin cannot pass"""
    base = {"src": list(range(6)), "hold": list(range(6)), "scale": [1.0] * 6, "z": [0.5] * 3 + [0.7] * 3,
            "u": [643.0] * 6, "v": [385.0] * 6}

    def passes(near, mid, rec=base):                      # the scorer reads ratio `near` below 0.6 m, `mid` above
        def scorer(model, intr, C, data):
            scale = [s * (near if z < 0.6 else mid) for s, z in zip(rec["scale"], rec["z"])]
            return {**rec, "scale": scale}, {}, SOLVED
        return ls.self_check("KB4T", None, np.eye(4), None, rec, scorer=scorer)["passed"]
    assert passes(1.0092, 1.0108) and not passes(1.0088, 1.01) and not passes(1.01, 1.0112)
    assert not passes(1.01, 1.01, {**base, "z": [1.5] * 6})


def test_the_scorer_reports_its_solves():
    data = _test_data()
    _, _, solves = ls.score("KB4T", [S.I0, S.I1], S.C_TRUE, data)
    assert solves == SOLVED


@pytest.mark.parametrize("stops", [1, 2])
def test_a_scoring_solve_that_stopped_short_is_restarted_once_from_its_result(monkeypatch, stops):
    """each of the two solves runs at stereofit's MAX_NFEV evaluations a round (step 45's cap, not solve()'s default
    300) and, when scipy does not report success, once more from its own result: the poses, and in the second solve
    the board scales too. A solve that stops short once and converges when restarted gives a result; one that stops
    short twice does not (step 66: NO RESULT), and there is no third try"""
    data = _test_data()
    real = ls.sf.solve
    calls = []

    def capped(d, model, intr, C, poses0, **k):
        assert k["max_nfev"] == ls.sf.MAX_NFEV
        before = [c for c in calls if c["scale"] == k["scale"]]
        if before:                                        # a restart: from the previous result, not the first start
            assert np.array_equal(poses0, before[-1]["r"]["poses"])
            if k["scale"]:
                assert np.array_equal(k["ls0"], before[-1]["r"]["log_scale"])
        r = real(d, model, intr, C, poses0, **k)
        calls.append({"scale": k["scale"], "r": r})
        return {**r, "success": len(before) >= stops}
    monkeypatch.setattr(ls.sf, "solve", capped)
    rec, _, solves = ls.score("KB4T", [S.I0, S.I1], S.C_TRUE, data)
    assert [c["scale"] for c in calls] == [False, False, True, True]
    assert ls.solved(solves) is (stops == 1)
    if stops == 1:
        assert abs(100 * (np.median(rec["scale"]) - 1)) < 0.1


def test_a_self_check_whose_rescoring_did_not_converge_fails():
    """the rescoring with the long baseline reads exactly +1 % but its least squares stopped short: not passed"""
    base = {"src": [0, 1], "hold": [0, 1], "scale": [1.0, 1.0], "z": [0.7, 0.7], "u": [643.0] * 2, "v": [385.0] * 2}
    for solves, want in ((SOLVED, True), ({**SOLVED, "scale_success": False}, False),
                         ({**SOLVED, "pose_success": False}, False)):
        def scorer(model, intr, C, data, solves=solves):
            return {**base, "scale": [1.01, 1.01]}, {}, solves
        chk = ls.self_check("KB4T", None, np.eye(4), None, base, scorer=scorer)
        assert chk["passed"] is want and chk["solved"] is want


def test_intervals_resample_whole_holds():
    """8 holds of 10 identical pairs: resampling holds gives the median's spread over 8 values, resampling pairs a
    spread 5x narrower (19 Oct review: clusters too small made intervals too narrow)"""
    rec = {"src": list(range(80)), "hold": [i // 10 for i in range(80)],
           "scale": [1 + (i // 10 - 3.5) / 100 for i in range(80)], "z": [0.7] * 80, "u": [643.0] * 80,
           "v": [385.0] * 80}
    ci = ls.cell_dist_table(rec, (642.83, 384.95), 8)[0]["ci95_pct"]
    assert ci[1] - ci[0] >= 4.0                                        # pairs resampled: 1.0


def test_cells_holds_and_intervals():
    rec = {"src": list(range(20)), "hold": [i // 2 for i in range(20)], "scale": [1.001] * 10 + [1.02] * 10,
           "z": [0.7] * 20, "u": [643.0] * 10 + [1100.0] * 10, "v": [385.0] * 20}
    tab = {(r["cell"], tuple(r["dist_m"])): r for r in ls.cell_dist_table(rec, (642.83, 384.95), 5)}
    c, rgt = tab[("centre", (0.6, 0.9))], tab[("right", (0.6, 0.9))]
    assert (c["pairs"], c["holds"]) == (10, 5) and c["median_pct"] == pytest.approx(0.1)
    assert c["ci95_pct"] == pytest.approx([0.1, 0.1])
    assert rgt["median_pct"] == pytest.approx(2.0)
    assert ls.cell_dist_table(rec, (642.83, 384.95), 6)[0]["ci95_pct"] is None    # 5 holds < 6


def test_the_fixed_pass_rule():
    row = lambda cell, med, holds: {"cell": cell, "dist_m": [0.6, 0.9], "holds": holds, "median_pct": med}
    assert ls.verdict([row("centre", 0.49, 8), row("top-left", -0.99, 8)], GATES)[0] == "PASS"
    assert ls.verdict([row("centre", 0.51, 8)], GATES)[0] == "FAIL"
    assert ls.verdict([row("top", -0.6, 8)], GATES)[0] == "FAIL"                 # sides keep the 0.5 % limit
    v, val, notval = ls.verdict([row("centre", 0.2, 8), row("bottom", 3.0, 7)], GATES)
    assert v == "PASS" and len(val) == 1 and notval[0]["cell"] == "bottom"      # 7 holds: not judged
    assert ls.verdict([row("centre", 0.0, 7)], GATES)[0] == "NO RESULT"
