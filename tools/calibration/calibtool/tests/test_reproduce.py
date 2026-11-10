"""calibtool.pipeline (wrappers of steps 16, 45, 66 and run_lens_kb4t's summary row) and the M1 baseline: replaying
the two stored sessions through them reproduces what the recorded runs concluded by hand.

Fast tests (no marker): the wrappers read verdicts from the steps' JSON, never from exit codes (fake steps); step 16 on
the stored corner files gives the recorded FIT/TEST files array for array. `replay` tests (minutes each, -m replay):
the fit, both validations and the summary rows against the recorded numbers (sessions.Recorded), and the reference
numbers of phantom21's live radtan config (P21_LIVE_RADTAN below), which phase B's livecfg must reproduce."""
import json
import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("cv2")
pytest.importorskip("yaml")

from calibtool import paths, pipeline, sessions  # noqa: E402
from calibtool.session import Session  # noqa: E402

NAMES = ["p21_1105", "p10_1106"]
ENTRY_TOL_PCT = 0.02          # per-entry step-66 median (plan A3)
RMS_TOL_PX, BASELINE_TOL_MM = 0.005, 0.01
SUMMARY_TOL_PCT, ROWS_TOL_PX = 0.02, 0.005

# phantom21's LIVE radtan config (p21/etc-okvis/mk12_stereo_holevisor-radtan-pre-daksh.yaml) scored by step 66 and the
# summary row on the TEST pairs of split16(p21_1105) (= the recorded p21/kit/final/test.npz), through a temporary
# path-patched copy of the kit's radtan_shim.py + run66.py (made by _radtan_runner below, never committed).
# Source: test_p21_live_radtan_reference, first run on Moonstone 9 Nov 2026 (x86, OMP_NUM_THREADS=1); the same numbers
# as the recorded p21/kit/final/validate-live/report.json and final21.log (Thor, 5 Nov) to the printed digits.
# Step 66: FAIL, 2 of 6 validated entries outside the limits, 16 not validated, 74 of 174 TEST holds flagged.
P21_LIVE_RADTAN = {
    "verdict": "FAIL",
    "entries": {("centre", (0.4, 0.6)): -0.0566, ("centre", (0.6, 0.9)): -0.0006, ("right", (0.6, 0.9)): 0.5375,
                ("bottom", (0.6, 0.9)): 0.4034, ("bottom", (0.9, 1.3)): 3.9857, ("bottom-right", (0.6, 0.9)): 0.4744},
    "summary": {"baseline_mm": 43.2798, "n_pairs": 499, "median_pct": 0.7739, "iqr_pct": (0.0654, 1.7674),
                "rows_px": 0.8203,
                "bands": {(0.0, 1.0): (289, 0.2850), (1.0, 1.3): (92, 2.8398), (1.3, 1.7): (51, 1.3912),
                          (1.7, 9.9): (67, 2.1313)}},     # (lo, hi] m: (pairs, median %)
}


# ---------------------------------------------------------------- helpers


def _write(path, text):
    path.write_text(textwrap.dedent(text))
    return path


def _live_json(stored, out_json):
    """the stored session's live OKVIS config as a step-66 calibration entry, the model taken from the registry (never
    inferred from the coefficient count); phase B's livecfg.load replaces this"""
    import okviscfg
    c = okviscfg.load(stored.live_config)
    intr = [np.r_[K[0, 0], K[1, 1], K[0, 2], K[1, 2], np.ravel(D)] for K, D in zip(c.K, c.D)]
    C = np.linalg.inv(c.T_SC[0]) @ c.T_SC[1]
    out_json.write_text(json.dumps({"model": stored.live_model, "intr": [v.tolist() for v in intr], "C": C.tolist(),
                                    "source": str(stored.live_config)}))
    return out_json


def _radtan_runner(root, tmp):
    """temporary copies of the p21 kit's radtan_shim.py and run66.py with their /root/p21-cal paths pointed at this
    tree; -> (shim dir, the step-66 runner). The kit files must still carry the paths patched here"""
    kit = root / "p21" / "kit"
    shim = (kit / "radtan_shim.py").read_text()
    old = 'sys.path.insert(0, "/root/p21-cal/code/tools/calibration/scripts")'
    assert old in shim, "the kit's radtan_shim.py changed: re-check the patch"
    (tmp / "radtan_shim.py").write_text(shim.replace(old, f"sys.path.insert(0, {str(paths.SCRIPTS)!r})"))
    run = (kit / "run66.py").read_text()
    for a, b in (('sys.path.insert(0, "/root/p21-cal")', f"sys.path.insert(0, {str(tmp)!r})"),
                 ('S = "/root/p21-cal/code/tools/calibration/scripts/66_validate_lens.py"',
                  f"S = {str(paths.script('66_validate_lens'))!r}")):
        assert a in run, "the kit's run66.py changed: re-check the patch"
        run = run.replace(a, b)
    (tmp / "run66.py").write_text(run)
    return tmp, tmp / "run66.py"


def _summary_with_shim(shim_dir, calib_json, test_npz, rig_path):
    """pipeline.summary in its own process with the radtan shim loaded (it patches lensmodel process-wide)"""
    code = (f"import sys, json; sys.path[:0] = [{str(shim_dir)!r}, {str(paths.CAL)!r}]\n"
            "import radtan_shim\nfrom calibtool import pipeline\n"
            f"r = pipeline.summary({str(calib_json)!r}, {str(test_npz)!r}, {str(rig_path)!r})\n"
            "print('JSON', json.dumps(r.as_dict()))\n")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    line = [ln for ln in p.stdout.splitlines() if ln.startswith("JSON ")][-1]
    return json.loads(line[5:])


def _entries_off(got, want, tol):
    """mismatch lines between two {(cell, dist): median %} dicts: keys differ, or a median further than tol"""
    out = []
    if set(got) != set(want):
        out.append(f"entries differ: only reproduced {sorted(set(got) - set(want))}, "
                   f"only recorded {sorted(set(want) - set(got))}")
    for k in sorted(set(got) & set(want)):
        if abs(got[k] - want[k]) > tol:
            out.append(f"entry {k[0]} {k[1][0]}-{k[1][1]} m: {got[k]:+.4f} % vs recorded {want[k]:+.2f} %")
    return out


def _summary_off(tag, got, want):
    out = []
    for name, g, w, tol in (("median", got["median_pct"], want.median_pct, SUMMARY_TOL_PCT),
                            ("IQR low", got["iqr_pct"][0], want.iqr_pct[0], SUMMARY_TOL_PCT),
                            ("IQR high", got["iqr_pct"][1], want.iqr_pct[1], SUMMARY_TOL_PCT),
                            ("rows", got["rows_px"], want.rows_px, ROWS_TOL_PX)):
        if abs(g - w) > tol:
            out.append(f"summary {tag} {name}: {g:+.4f} vs recorded {w:+.3f}")
    return out


# ---------------------------------------------------------------- fast: verdicts parsed, never exit codes


def test_validate66_reads_the_verdict_from_report_json(tmp_path):
    """a step 66 that FAILs with exit 0 (the p21 kit's runner did) is still a FAIL"""
    fake = _write(tmp_path / "fake66.py", """
        import json, sys
        from pathlib import Path
        out = Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
        rep = {"verdict": "FAIL", "reason": "1 of 2 validated entries are outside the limits: right 0.6-0.9 m +0.54 %",
               "table": [{"cell": "right", "dist_m": [0.6, 0.9], "median_pct": 0.54, "holds": 9, "ok": False,
                          "limit_pct": 0.5},
                         {"cell": "centre", "dist_m": [0.4, 0.6], "median_pct": -0.05, "holds": 13, "ok": True,
                          "limit_pct": 0.5},
                         {"cell": "top", "dist_m": [0.9, 1.3], "median_pct": 2.0, "holds": 3}],
               "validated_cells": {"right": [[0.6, 0.9]], "centre": [[0.4, 0.6]]}}
        (out / "report.json").write_text(json.dumps(rep))
        print("FAIL: " + rep["reason"])
        sys.exit(int(sys.argv[-1]) if sys.argv[-1].isdigit() else 0)
        """)
    r = pipeline.validate66(tmp_path / "cal.json", tmp_path / "test.npz", tmp_path / "v", rig="rig.yaml", script=fake)
    assert r.verdict == "FAIL" and r.returncode == 0
    assert r.entries == {("right", (0.6, 0.9)): 0.54, ("centre", (0.4, 0.6)): -0.05}
    assert r.failing == {("right", (0.6, 0.9)): 0.54}
    assert r.n_validated == 2
    assert r.not_validated == [("top", (0.9, 1.3))]
    assert r.text.startswith("FAIL: 1 of 2")
    assert r.report_path == tmp_path / "v" / "report.json"


def test_validate66_without_report_raises(tmp_path):
    fake = _write(tmp_path / "fake66.py", "import sys; print('boom'); sys.exit(0)\n")
    with pytest.raises(pipeline.PipelineError, match="boom"):
        pipeline.validate66(tmp_path / "c.json", tmp_path / "t.npz", tmp_path / "v", rig="r.yaml", script=fake)


def test_validate66_text_and_json_must_agree(tmp_path):
    fake = _write(tmp_path / "fake66.py", """
        import json, sys
        from pathlib import Path
        out = Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
        (out / "report.json").write_text(json.dumps({"verdict": "PASS", "reason": "x", "table": []}))
        print("FAIL: x")
        """)
    with pytest.raises(pipeline.PipelineError, match="disagree"):
        pipeline.validate66(tmp_path / "c.json", tmp_path / "t.npz", tmp_path / "v", rig="r.yaml", script=fake)


def test_fit45_reads_the_fit_json(tmp_path):
    fake = _write(tmp_path / "fake45.py", """
        import json, sys
        from pathlib import Path
        verdict = sys.argv[-1]
        Path(sys.argv[2]).write_text(json.dumps({"verdict": verdict, "rms_px": 0.4, "baseline_mm": 44.0,
                                                 "fit_pairs": 10, "fit_holds": 4, "model": "KB4T"}))
        print(verdict); sys.exit(0 if verdict.startswith("converged") else 1)
        """)
    r = pipeline.fit45(tmp_path / "fit.npz", "rig.yaml", tmp_path / "f", script=fake,
                       extra=["converged: rms 0.400 px over 10 pairs, baseline 44.00 mm"])
    assert r.converged and r.rms_px == 0.4 and r.baseline_mm == 44.0 and r.fit_pairs == 10 and r.fit_holds == 4
    assert r.json_path == tmp_path / "f" / "fit-kb4t.json"
    r = pipeline.fit45(tmp_path / "fit.npz", "rig.yaml", tmp_path / "g", script=fake, extra=["rejected: too few"])
    assert not r.converged and r.verdict == "rejected: too few" and r.returncode == 1


def test_fit45_without_json_raises(tmp_path):
    fake = _write(tmp_path / "fake45.py", "import sys; print('ERROR: nothing'); sys.exit(1)\n")
    with pytest.raises(pipeline.PipelineError, match="ERROR: nothing"):
        pipeline.fit45(tmp_path / "fit.npz", "rig.yaml", tmp_path / "f", script=fake)


def test_load_rig_is_quiet(capsys):
    rig = pipeline.load_rig(sessions.get("p21_1105").rig_path)
    assert rig.name == "phantom21"
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------- fast, stored data: step 16


@pytest.mark.parametrize("name", NAMES)
def test_split16_reproduces_the_recorded_split(name, data_root, tmp_path):
    st = sessions.get(name)
    sp = pipeline.split16(Session.load(st), tmp_path, st.rig_path, max_fit_z=3.0, holdout_every=2)
    assert (sp.fit_pairs, sp.test_pairs) == (st.recorded.fit_pairs_16, st.recorded.test_pairs_16)
    assert sp.fit_npz == tmp_path / "fit.npz" and sp.test_npz == tmp_path / "test.npz"
    for mine, rec in ((sp.fit_npz, st.recorded.files["fit_npz"]), (sp.test_npz, st.recorded.files["test_npz"])):
        a, b = np.load(mine), np.load(data_root / rec)
        assert sorted(a.files) == sorted(b.files)
        for k in b.files:
            if k == "img_z":       # the rough PnP distance: recorded on Thor (arm64), ~1e-8 m apart on x86
                assert np.allclose(a[k], b[k], rtol=0, atol=1e-6, equal_nan=True), k
            else:
                assert np.array_equal(a[k], b[k]), (rec, k)


# ---------------------------------------------------------------- replay: the recorded runs reproduced


@pytest.mark.replay
@pytest.mark.parametrize("name", NAMES)
def test_summary_of_the_recorded_files(name, data_root, tmp_path):
    """the summary row of the recorded fit on the recorded TEST file (p10: also the live KB4T config) reads the
    recorded numbers: the summary math alone, before any fit is reproduced"""
    st = sessions.get(name)
    f = st.recorded.files
    got = pipeline.summary(data_root / f["fit_json"], data_root / f["test_npz"], st.rig_path)
    off = _summary_off("new", got.as_dict(), st.recorded.summary_new)
    assert got.n_pairs == st.recorded.test_pairs_16
    if st.live_model != "RADTAN":
        live = _live_json(st, tmp_path / "live.json")
        rec = json.loads((data_root / f["live_json"]).read_text())
        mine = json.loads(live.read_text())
        assert mine["model"] == rec["model"]
        assert np.allclose(mine["intr"], rec["intr"], atol=1e-9) and np.allclose(mine["C"], rec["C"], atol=1e-9)
        off += _summary_off("live", pipeline.summary(live, data_root / f["test_npz"], st.rig_path).as_dict(),
                            st.recorded.summary_live)
    assert not off, "\n".join(off)


@pytest.mark.replay
@pytest.mark.parametrize("name", NAMES)
def test_replay_reproduces_the_recorded_run(name, data_root, tmp_path):
    """split16 -> fit45 -> validate66 (new; and live where it is not radtan) -> summary rows, against the recorded
    run. Every mismatch is collected and reported together"""
    st, rec = sessions.get(name), sessions.get(name).recorded
    sp = pipeline.split16(Session.load(st), tmp_path / "split", st.rig_path, max_fit_z=3.0, holdout_every=2)
    fit = pipeline.fit45(sp.fit_npz, st.rig_path, tmp_path / "fit", max_z=3.0)
    off, table = [], {"session": name, "fit": fit.as_dict()}
    assert fit.converged, fit.verdict
    if fit.fit_pairs != rec.fit_used_pairs:
        off.append(f"fit pairs {fit.fit_pairs} vs recorded {rec.fit_used_pairs}")
    if abs(fit.rms_px - rec.fit_rms_px) > RMS_TOL_PX:
        off.append(f"fit rms {fit.rms_px:.5f} px vs recorded {rec.fit_rms_px:.5f} px")
    if abs(fit.baseline_mm - rec.baseline_mm) > BASELINE_TOL_MM:
        off.append(f"baseline {fit.baseline_mm:.4f} mm vs recorded {rec.baseline_mm:.4f} mm")
    jobs = {"new": (fit.json_path, dict(fit_report=fit.json_path))}
    if st.live_model != "RADTAN":
        jobs["live"] = (_live_json(st, tmp_path / "live.json"), {})
    with ThreadPoolExecutor(len(jobs)) as ex:
        futs = {k: ex.submit(pipeline.validate66, cal, sp.test_npz, tmp_path / f"validate-{k}", rig=st.rig_path,
                             **kw) for k, (cal, kw) in jobs.items()}
        val = {k: f.result() for k, f in futs.items()}
    table["validate_new"] = val["new"].as_dict()
    if val["new"].verdict != rec.new_verdict:
        off.append(f"66 new: {val['new'].verdict} vs recorded {rec.new_verdict} ({val['new'].text})")
    off += _entries_off(val["new"].entries, rec.new_entries, ENTRY_TOL_PCT)
    sm = pipeline.summary(fit.json_path, sp.test_npz, st.rig_path).as_dict()
    table["summary_new"] = sm
    off += _summary_off("new", sm, rec.summary_new)
    if "live" in val:
        v = val["live"]
        table["validate_live"] = v.as_dict()
        if (v.verdict, v.n_validated) != (rec.live_verdict, rec.live_n_validated):
            off.append(f"66 live: {v.verdict} on {v.n_validated} vs recorded {rec.live_verdict} on "
                       f"{rec.live_n_validated} ({v.text})")
        off += [f"live {x}" for x in _entries_off(v.failing, rec.live_failing, ENTRY_TOL_PCT)]
        sl = pipeline.summary(jobs["live"][0], sp.test_npz, st.rig_path).as_dict()
        table["summary_live"] = sl
        off += _summary_off("live", sl, rec.summary_live)
    print(f"REPRODUCED {name}: " + json.dumps(table, default=str))
    assert not off, "\n".join(off)


@pytest.mark.replay
def test_p21_live_radtan_reference(data_root, tmp_path):
    """phantom21's live radtan config on the replayed TEST split, scored as the recorded run did (the kit's shim):
    the recorded verdict and summary row, and the reference numbers P21_LIVE_RADTAN"""
    st, rec = sessions.get("p21_1105"), sessions.get("p21_1105").recorded
    sp = pipeline.split16(Session.load(st), tmp_path / "split", st.rig_path, max_fit_z=3.0, holdout_every=2)
    live = _live_json(st, tmp_path / "live-radtan.json")
    recorded_json = json.loads((data_root / rec.files["live_json"]).read_text())
    mine = json.loads(live.read_text())
    assert mine["model"] == recorded_json["model"] == "RADTAN"
    assert np.allclose(mine["intr"], recorded_json["intr"], atol=1e-9)
    assert np.allclose(mine["C"], recorded_json["C"], atol=1e-9)
    shim_dir, runner = _radtan_runner(data_root, tmp_path)
    v = pipeline.validate66(live, sp.test_npz, tmp_path / "validate-live", rig=st.rig_path, script=runner)
    sm = _summary_with_shim(shim_dir, live, sp.test_npz, st.rig_path)
    print("P21 LIVE RADTAN: " + json.dumps({"validate": v.as_dict(), "summary": sm}, default=str))
    off = []
    if (v.verdict, v.n_validated) != (rec.live_verdict, rec.live_n_validated):
        off.append(f"66 live: {v.verdict} on {v.n_validated} vs recorded {rec.live_verdict} on {rec.live_n_validated}")
    off += _entries_off(v.failing, rec.live_failing, ENTRY_TOL_PCT)
    off += _summary_off("live", sm, rec.summary_live)
    # the recorded report's full table (every validated entry)
    rep = json.loads((data_root / rec.files["validate_live"] / "report.json").read_text())
    rec_entries = {(e["cell"], tuple(e["dist_m"])): e["median_pct"] for e in rep["table"] if "ok" in e}
    off += _entries_off(v.entries, {k: round(x, 2) for k, x in rec_entries.items()}, ENTRY_TOL_PCT)
    ref = P21_LIVE_RADTAN
    if v.verdict != ref["verdict"]:
        off.append(f"reference verdict {ref['verdict']} vs {v.verdict}")
    off += [f"reference {x}" for x in _entries_off(v.entries, ref["entries"], ENTRY_TOL_PCT)]
    rs = ref["summary"]
    got = {"median_pct": sm["median_pct"], "iqr low": sm["iqr_pct"][0], "iqr high": sm["iqr_pct"][1],
           "rows_px": sm["rows_px"], "baseline_mm": sm["baseline_mm"]}
    want = {"median_pct": rs["median_pct"], "iqr low": rs["iqr_pct"][0], "iqr high": rs["iqr_pct"][1],
            "rows_px": rs["rows_px"], "baseline_mm": rs["baseline_mm"]}
    off += [f"reference summary {k}: {got[k]:.4f} vs {want[k]:.4f}" for k in got if abs(got[k] - want[k]) > 1e-3]
    if sm["n_pairs"] != rs["n_pairs"]:
        off.append(f"reference summary pairs {sm['n_pairs']} vs {rs['n_pairs']}")
    for b in sm["bands"]:
        n, med = rs["bands"][(b["lo"], b["hi"])]
        if b["n"] != n or abs(b["median_pct"] - med) > ENTRY_TOL_PCT:
            off.append(f"reference band {b['lo']}-{b['hi']} m: {b['median_pct']:+.4f} % (n {b['n']}) vs {med:+.4f} "
                       f"(n {n})")
    assert not off, "\n".join(off)
