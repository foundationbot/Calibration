"""Step 66 on the synthetic test sweep: PASS for the fitted KB4T, FAIL for a KB4 refit, NO RESULT for a bad recording."""
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
m66 = importlib.import_module("66_validate_lens")


@pytest.fixture(scope="module")
def passed(tmp_path_factory):
    out = tmp_path_factory.mktemp("v")
    P = S.pipeline()
    code = m66.main([str(P / "fit.json"), str(P / "test.npz"), str(out), "--fit-report", str(P / "fit.json"),
                     "--uncertainty", str(P / "unc.json"), "--rig", str(S.RIG12)])
    return code, out


def test_the_fitted_calibration_passes_its_test_sweep(passed):
    code, out = passed
    rep = json.loads((out / "report.json").read_text())
    assert code == 0 and rep["verdict"] == "PASS"
    assert rep["self_check"]["passed"]
    val = {(r["cell"], tuple(r["dist_m"])) for r in rep["table"] if "ok" in r}
    assert ("centre", (0.6, 0.9)) in val and len(val) >= 2
    assert "top-left" in rep["not_validated_cells"]                 # the test sweep never went there
    assert rep["calibration"]["fit_verdict"].startswith("converged")


def test_the_report_reads_as_the_user_asks(passed):
    md = (passed[1] / "report.md").read_text()
    for heading in ("## Verdict", "## Scorer self-check", "## Coverage of the test sweep",
                    "## Depth error by cell and distance", "## How well each number is pinned down",
                    "## Repeatability at 1 m", "## Holds over the quality gate"):
        assert heading in md
    tables = sum(1 for line in md.splitlines() if line.startswith("|---"))
    assert md.count("How to read it:") == tables == 7 and "one-sample" in md                 # one list per table


def test_a_kb4_refit_fails_the_same_test_sweep(tmp_path):
    """the p1, p2 mutation seen by the validation itself (20 Oct scratch run: -2.1 % at the centre). This code's KB4
    refit reads depth +1.6 % at 0.5 m and -2.4 % at 1 m near the centre, crossing zero near 0.7 m, so the test sweep's
    0.6-0.9 m cells read little (centre -0.2 %, right -0.8 %), and the near bottom-right corner reads +1.5 %. Its
    holds there fit a rigid board worse than 0.6 px with this calibration: they are flagged and still judged
    (dropping them, as step 66 first did, left the corner "not validated" and hid its failure)"""
    _, r4 = S.nominal_fit("KB4")
    cal = S.write_calibration(tmp_path / "kb4.json", "KB4", r4["intr"], r4["C"])
    assert m66.main([str(cal), str(S.pipeline() / "test.npz"), str(tmp_path / "v"), "--rig", str(S.RIG12)]) == 1
    rep = json.loads((tmp_path / "v" / "report.json").read_text())
    assert rep["verdict"] == "FAIL"
    tab = {(r["cell"], tuple(r["dist_m"])): r for r in rep["table"]}
    br, rgt = tab[("bottom-right", (0.4, 0.6))], tab[("right", (0.6, 0.9))]
    assert br["holds"] >= 8 and br["ok"] is False and br["median_pct"] > br["limit_pct"] == 1.0
    assert rgt["ok"] is False
    assert br["flagged_holds"] and any("bottom-right" in f["cells"] for f in rep["test_flagged"])


def test_a_flagged_test_hold_is_listed_and_still_judged(tmp_path):
    """3 of 10 holds noisier than the 0.6 px gate (30 %, more than step 45 would accept on a fit sweep): flagged per
    cell, counted and judged; the test recording is not rejected"""
    d = S.views([S.I0, S.I1], S.C_TRUE, 10, seed=41, targets=[(643, 385)], z_range=(0.6, 0.9),
                noisy_holds=(0, 3, 6))
    np.savez_compressed(tmp_path / "t.npz", **d)
    cal = S.write_calibration(tmp_path / "true.json", "KB4T", [S.I0, S.I1], S.C_TRUE)
    assert m66.main([str(cal), str(tmp_path / "t.npz"), str(tmp_path / "v"), "--rig", str(S.RIG12)]) == 0
    rep = json.loads((tmp_path / "v" / "report.json").read_text())
    centre = [r for r in rep["table"] if r["cell"] == "centre"]
    assert [(r["dist_m"], r["holds"], r["flagged_holds"], r["ok"]) for r in centre] == [([0.6, 0.9], 10, [0, 3, 6], True)]
    assert [(f["hold"], f["cells"]) for f in rep["test_flagged"]] == [(0, ["centre"]), (3, ["centre"]), (6, ["centre"])]
    assert all(f["rms_px"] > 0.6 for f in rep["test_flagged"]) and "rejected" not in rep["reason"]
    md = (tmp_path / "v" / "report.md").read_text()
    assert md.count("flagged: board or calibration bad here") == 3 and "| 10 (3 flagged) |" in md


def test_a_failed_scorer_self_check_gives_no_result_and_no_table(tmp_path, monkeypatch):
    """the self-check runs before any table: with a scorer deaf to the baseline, step 66 trusts no depth number"""
    real = m66.ls.self_check

    def deaf(model, intr, C, data):                       # rescales any baseline back to the true length
        C = np.array(C, float, copy=True)
        C[:3, 3] *= np.linalg.norm(S.C_TRUE[:3, 3]) / np.linalg.norm(C[:3, 3])
        return m66.ls.score(model, intr, C, data)
    monkeypatch.setattr(m66.ls, "self_check", lambda *a: real(*a, scorer=deaf))
    np.savez_compressed(tmp_path / "t.npz", **S.holdout_sweep())
    cal = S.write_calibration(tmp_path / "true.json", "KB4T", [S.I0, S.I1], S.C_TRUE)
    assert m66.main([str(cal), str(tmp_path / "t.npz"), str(tmp_path / "v"), "--rig", str(S.RIG12)]) == 2
    rep = json.loads((tmp_path / "v" / "report.json").read_text())
    assert rep["verdict"] == "NO RESULT" and "self-check" in rep["reason"] and "table" not in rep
    assert "Self-check FAILED." in (tmp_path / "v" / "report.md").read_text()


def test_an_unconverged_fit_is_not_judged(tmp_path):
    """a step-45 file whose verdict is not "converged" (stopped at its cap, rejected, wrong sides) still carries
    numbers; they are not a calibration, so step 66 gives NO RESULT before scoring them. A calibration from elsewhere
    (no verdict) is judged as before (the other tests here)"""
    fit = json.loads((S.pipeline() / "fit.json").read_text())
    fit["verdict"] = "unconverged: 3 attempts of at most 3 did not reach scipy's convergence"
    cal = tmp_path / "fit.json"
    cal.write_text(json.dumps(fit))
    assert m66.main([str(cal), str(S.pipeline() / "test.npz"), str(tmp_path / "v"), "--rig", str(S.RIG12)]) == 2
    rep = json.loads((tmp_path / "v" / "report.json").read_text())
    assert rep["verdict"] == "NO RESULT" and "unconverged" in rep["reason"] and "converged fit" in rep["reason"]
    assert "table" not in rep and "self_check" not in rep
    assert rep["calibration"]["fit_verdict"] == fit["verdict"]
    assert rep["reason"] in (tmp_path / "v" / "report.md").read_text()


@pytest.mark.parametrize("stop", ["pose", "scale", "self-check rescoring"])
def test_a_scoring_solve_that_did_not_converge_gives_no_result(tmp_path, monkeypatch, stop):
    """pose: the scoring's pose solve with the camera fixed; scale: its pose-and-scale solve, whose board scale is the
    depth error; self-check rescoring: both solves of the self-check's scoring with the baseline x 1.01. Only that
    solve stops short of scipy's convergence, at every call, and no depth number is trusted. Each NO RESULT must come
    from its own guard: with every solve stopped, the self-check's guard alone would also give NO RESULT for a
    scoring that did not converge, and a scoring guard that never fires would go unseen"""
    test = S.pipeline() / "test.npz"                      # before patching: the cached pipeline must stay unpatched
    real = m66.sf.solve

    def stopped(d, model, intr, C, *a, **k):
        r = real(d, model, intr, C, *a, **k)
        rescoring = not np.allclose(C, S.C_TRUE)          # the self-check scores the baseline x 1.01
        hit = rescoring if stop == "self-check rescoring" else (
            not rescoring and k.get("scale", False) == (stop == "scale"))
        return {**r, "success": False} if hit else r
    monkeypatch.setattr(m66.sf, "solve", stopped)
    cal = S.write_calibration(tmp_path / "true.json", "KB4T", [S.I0, S.I1], S.C_TRUE)
    assert m66.main([str(cal), str(test), str(tmp_path / "v"), "--rig", str(S.RIG12)]) == 2
    rep = json.loads((tmp_path / "v" / "report.json").read_text())
    assert rep["verdict"] == "NO RESULT" and "table" not in rep
    if stop == "self-check rescoring":
        assert rep["reason"].startswith("the scorer self-check's rescoring did not converge")
        assert m66.ls.solved(rep["scoring"]) and rep["self_check"]["solved"] is False
    else:
        assert rep["reason"].startswith("the scorer's least squares did not converge on the test sweep")
        assert "self_check" not in rep                    # refused before the self-check ran
        assert rep["scoring"][f"{stop}_success"] is False


def test_pairs_whose_pnp_start_failed_are_counted(tmp_path, monkeypatch):
    """the scorer leaves out a pair whose PnP start failed; how many is in the report, not silent"""
    test = S.pipeline() / "test.npz"                      # before patching: the cached pipeline must stay unpatched
    real = m66.sf.pnp_poses

    def failing(data, *a, **k):
        poses, ok = real(data, *a, **k)
        return poses, ok & ~np.isin(data.src_pairs, [0, 1])
    monkeypatch.setattr(m66.sf, "pnp_poses", failing)
    cal = S.write_calibration(tmp_path / "true.json", "KB4T", [S.I0, S.I1], S.C_TRUE)
    assert m66.main([str(cal), str(test), str(tmp_path / "v"), "--rig", str(S.RIG12)]) == 0
    rep = json.loads((tmp_path / "v" / "report.json").read_text())
    assert rep["test"]["pnp_failed_pairs"] == 2 and rep["scoring"]["pnp_failed_pairs"] == 2
    assert "2 of them left out of the scoring (their PnP start failed)" in (tmp_path / "v" / "report.md").read_text()


def test_reports_are_strict_json_when_a_hold_has_no_rms(tmp_path, monkeypatch):
    """a test hold with no corner within GROSS_PX (rms inf) is flagged and written as null, and a fit report whose
    dropped hold carries null (step 45's file) still renders"""
    P = S.pipeline()                                      # before patching: the cached pipeline must stay unpatched
    real = m66.sf.hold_rms
    monkeypatch.setattr(m66.sf, "hold_rms", lambda *a: {h: (np.inf if h == 3 else v) for h, v in real(*a).items()})
    fit = json.loads((P / "fit.json").read_text())
    fit["dropped_holds"] = [[7, None]]
    report = tmp_path / "fit-report.json"
    report.write_text(json.dumps(fit))
    cal = S.write_calibration(tmp_path / "true.json", "KB4T", [S.I0, S.I1], S.C_TRUE)
    assert m66.main([str(cal), str(P / "test.npz"), str(tmp_path / "v"), "--fit-report", str(report),
                     "--rig", str(S.RIG12)]) == 0

    def refuse(c):
        raise ValueError(f"non-finite number {c} in the JSON")
    rep = json.loads((tmp_path / "v" / "report.json").read_text(), parse_constant=refuse)
    assert [(f["hold"], f["rms_px"]) for f in rep["test_flagged"]] == [(3, None)]
    md = (tmp_path / "v" / "report.md").read_text()
    assert "| fit | 7 | - | no corner within 20 px | dropped by step 45 |" in md
    assert "| no corner within 20 px | flagged: board or calibration bad here |" in md
