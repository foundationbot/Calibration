"""calibtool.pipeline's solver choice and the speed of its scoring.

The solver: calibtool runs the steps with the fast solver (calibtool.solver, CALIB_SOLVER=fast) unless told
solver="scipy"; the choice reaches step 45 / 66 (their own processes, through their environment) and the in-process
scoring (pipeline.summary, decide.score, through stereofit.solver_scope), and never changes the caller's environment.
The standalone scripts keep scipy by default (stereofit reads CALIB_SOLVER, unset = scipy)."""
import importlib
import importlib.util
import json
import os
import sys
import textwrap
import threading
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("cv2")
pytest.importorskip("yaml")

from calibtool import decide, pipeline, sessions  # noqa: E402

CAL = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CAL / "tests"))
S = importlib.import_module("lens_synth")
sf = S.sf
RIG = sessions.get("p21_1105").rig_path
ENV = "CALIB_SOLVER"


def _write(path, text):
    path.write_text(textwrap.dedent(text))
    return path


def _fake45(tmp_path):
    """a step 45 that records the CALIB_SOLVER it ran with in its fit JSON"""
    return _write(tmp_path / "fake45.py", """
        import json, os, sys
        from pathlib import Path
        Path(sys.argv[2]).write_text(json.dumps({"verdict": "converged: x", "rms_px": 0.4, "baseline_mm": 44.0,
                                                 "fit_pairs": 1, "fit_holds": 1, "solver": os.environ.get("CALIB_SOLVER")}))
        """)


def _fake66(tmp_path):
    """a step 66 that records the CALIB_SOLVER it ran with as its reason"""
    return _write(tmp_path / "fake66.py", """
        import json, os, sys
        from pathlib import Path
        out = Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
        (out / "report.json").write_text(json.dumps({"verdict": "PASS", "reason": str(os.environ.get("CALIB_SOLVER")),
                                                     "table": []}))
        print("PASS: x")
        """)


# ---------------------------------------------------------------- the steps' processes


def test_steps_run_with_the_fast_solver_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    f = pipeline.fit45(tmp_path / "fit.npz", RIG, tmp_path / "f", script=_fake45(tmp_path))
    assert f.entry["solver"] == "fast"
    v = pipeline.validate66(tmp_path / "c.json", tmp_path / "t.npz", tmp_path / "v", RIG, script=_fake66(tmp_path))
    assert v.reason == "fast"
    assert ENV not in os.environ


def test_scipy_is_passed_to_the_steps_and_the_callers_environment_is_kept(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV, "fast")
    f = pipeline.fit45(tmp_path / "fit.npz", RIG, tmp_path / "f", script=_fake45(tmp_path), solver="scipy")
    assert f.entry["solver"] == "scipy"
    v = pipeline.validate66(tmp_path / "c.json", tmp_path / "t.npz", tmp_path / "v", RIG, script=_fake66(tmp_path),
                            solver="scipy")
    assert v.reason == "scipy"
    assert os.environ[ENV] == "fast"


def test_an_unknown_solver_is_refused(tmp_path):
    with pytest.raises(ValueError, match="solver"):
        pipeline.fit45(tmp_path / "fit.npz", RIG, tmp_path / "f", script=_fake45(tmp_path), solver="ceres")


# ---------------------------------------------------------------- in-process: stereofit.solver_scope


def test_solver_scope_overrides_the_environment_in_its_thread_only(monkeypatch):
    monkeypatch.setenv(ENV, "fast")
    seen = {}

    def other():
        seen["other"] = sf.fast_solver("KB4T") is not None

    with sf.solver_scope("scipy"):
        assert sf.fast_solver("KB4T") is None
        t = threading.Thread(target=other)
        t.start()
        t.join()
        with sf.solver_scope("fast"):                       # nested: the inner one wins, then the outer is back
            assert sf.fast_solver("KB4T") is not None
        assert sf.fast_solver("KB4T") is None
    assert seen["other"] is True
    assert sf.fast_solver("KB4T") is not None
    assert os.environ[ENV] == "fast"
    monkeypatch.delenv(ENV)
    with sf.solver_scope("fast"):
        assert sf.fast_solver("KB4T") is not None
    assert sf.fast_solver("KB4T") is None                   # the scripts' default: scipy
    with pytest.raises(ValueError):
        with sf.solver_scope("ceres"):
            pass


class _Seen(Exception):
    pass


@pytest.mark.parametrize("solver", ["fast", "scipy"])
def test_summary_scores_with_the_chosen_solver(solver, tmp_path, monkeypatch):
    import lensscore
    monkeypatch.setenv(ENV, "scipy" if solver == "fast" else "fast")    # the environment says the other one

    def spy(model, intr, C, data):
        raise _Seen(sf.fast_solver(model) is not None)

    monkeypatch.setattr(lensscore, "score", spy)
    np.savez_compressed(tmp_path / "t.npz", **S.holdout_sweep())
    cal = S.write_calibration(tmp_path / "c.json", "KB4T", [S.I0, S.I1], S.C_TRUE)
    with pytest.raises(_Seen) as e:
        pipeline.summary(cal, tmp_path / "t.npz", S.RIG12, solver=solver)
    assert e.value.args[0] is (solver == "fast")
    with pytest.raises(_Seen) as e:
        decide.score(cal, tmp_path / "t.npz", S.RIG12, None, solver=solver)
    assert e.value.args[0] is (solver == "fast")


# ---------------------------------------------------------------- the solver reaches every step of a replay


def _test_module(name):
    spec = importlib.util.spec_from_file_location(f"_tp_{name}", Path(__file__).with_name(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _recording_steps(monkeypatch, tmp_path):
    """replay.evaluate's steps replaced by recorders of their solver argument -> (calls, evaluate's arguments)"""
    from types import SimpleNamespace
    from calibtool import livecfg
    TD, TL = _test_module("test_decide"), _test_module("test_livecfg")
    calls = {}
    fit = SimpleNamespace(converged=True, json_path=tmp_path / "fit.json", verdict="converged",
                          as_dict=lambda: {"verdict": "converged"})
    v66 = TD.step66()
    v66.as_dict = lambda: {"verdict": "PASS"}

    def rec(name, value):
        def f(*a, **k):
            calls.setdefault(name, []).append(k.get("solver"))
            calls.setdefault("kwargs", []).append((name, k))
            return value() if callable(value) else value
        return f

    monkeypatch.setattr(pipeline, "fit45", rec("fit45", fit))
    monkeypatch.setattr(pipeline, "validate66", rec("validate66", v66))
    monkeypatch.setattr(livecfg, "validate66", rec("live66", v66))
    monkeypatch.setattr(decide, "placements", lambda *a, **k: None)
    monkeypatch.setattr(decide, "score", rec("score", lambda: TD.synthetic()))
    live = tmp_path / "live.yaml"
    live.write_text(TL.okvis_yaml("equidistant", (0.01, -0.02, 0.003, -0.001)))
    (tmp_path / "out").mkdir(exist_ok=True)
    split = SimpleNamespace(fit_npz=tmp_path / "fit.npz", test_npz=tmp_path / "test.npz")
    return calls, (None, RIG, split, live, tmp_path / "out", decide.Gates())


@pytest.mark.parametrize("solver", [None, "scipy"])
def test_evaluate_passes_the_solver_to_every_step(solver, tmp_path, monkeypatch):
    from calibtool import replay
    calls, args = _recording_steps(monkeypatch, tmp_path)
    replay.evaluate(*args, jobs=1, **({} if solver is None else {"solver": solver}))
    want = solver or "fast"
    kw = calls.pop("kwargs")
    assert calls == {"fit45": [want], "validate66": [want], "live66": [want], "score": [want, want]}
    # each calibration's scores come from its own step-66 run (no second scoring)
    assert [k["step66"].name for n, k in kw if n == "score"] == ["validate-new", "validate-live"]


def test_run_checks_the_solver_first_and_passes_it_to_evaluate(tmp_path, monkeypatch):
    from calibtool import replay
    TR = _test_module("test_replay")
    stored = TR.synth_stored(pipeline.load_rig(RIG), tmp_path)
    real_load = replay.load_session
    monkeypatch.setattr(replay, "load_session", lambda *a, **k: (_ for _ in ()).throw(_Seen()))
    with pytest.raises(ValueError, match="solver"):         # before any data is read
        replay.run(stored, out_dir=tmp_path / "o", solver="ceres")
    monkeypatch.setattr(replay, "load_session", real_load)
    seen = []
    monkeypatch.setattr(replay, "evaluate", lambda *a, **k: seen.append(k.get("solver")) or {"timings": {}})
    a = replay.run(stored, out_dir=tmp_path / "o", solver="scipy")
    b = replay.run(stored, out_dir=tmp_path / "p")
    assert seen == ["scipy", "fast"]
    assert (a["solver"], b["solver"]) == ("scipy", "fast")              # the report says which one ran


# ---------------------------------------------------------------- row errors: read once, vectorised


def _rows_reference(model, intr, C, test_npz, rig):
    """the row error as pipeline.summary computed it up to 9 Nov 2026 (a loop over pairs, a dict per camera):
    -> (pair per corner, dy per corner)"""
    import cv2
    import lensmodel
    rig = pipeline.load_rig(rig)
    T10 = np.linalg.inv(np.asarray(C, float))
    R0, R1, *_ = cv2.stereoRectify(np.eye(3), np.zeros(5), np.eye(3), np.zeros(5), tuple(rig.resolution),
                                   np.ascontiguousarray(T10[:3, :3]), np.ascontiguousarray(T10[:3, 3].reshape(3, 1)),
                                   flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    z = dict(np.load(test_npz))
    srcs, dys = [], []
    for j in range(len(z["img_t0"])):
        m = z["det_img"] == j
        d = [{(t, k): (u, v) for t, k, u, v, c in zip(z["det_tag"][m], z["det_k"][m], z["det_u"][m], z["det_v"][m],
                                                       z["det_cam"][m]) if c == cam} for cam in (0, 1)]
        keys = sorted(set(d[0]) & set(d[1]))
        if keys:
            b0 = (R0 @ lensmodel.unproject(model, intr[0], [d[0][k] for k in keys]).T).T
            b1 = (R1 @ lensmodel.unproject(model, intr[1], [d[1][k] for k in keys]).T).T
            dys.append(rig.nominal_focal_px * (b0[:, 1] / b0[:, 2] - b1[:, 1] / b1[:, 2]))
            srcs.append(np.full(len(keys), j))
    return np.concatenate(srcs), np.concatenate(dys)


def _sweep_with_one_sided_corners(tmp_path):
    """holdout_sweep() with some corners seen by one camera only (dropped from camera 1) -> npz path"""
    d = {k: np.asarray(v) for k, v in S.holdout_sweep().items()}
    drop = (d["det_cam"] == 1) & (np.arange(len(d["det_cam"])) % 7 == 0)
    for k in [k for k in d if k.startswith("det_")]:
        d[k] = d[k][~drop]
    np.savez_compressed(tmp_path / "t.npz", **d)
    return tmp_path / "t.npz"


def test_row_errors_equal_the_per_pair_loop(tmp_path):
    npz = _sweep_with_one_sided_corners(tmp_path)
    intr = [S.I0, S.I1]
    want_src, want_dy = _rows_reference("KB4T", intr, S.C_TRUE, npz, S.RIG12)
    src, dy = pipeline.row_errors("KB4T", intr, S.C_TRUE, npz, S.RIG12)
    assert np.array_equal(src, want_src)
    assert np.allclose(dy, want_dy, rtol=0, atol=1e-9)
    s2, d2 = decide.row_errors("KB4T", intr, S.C_TRUE, npz, S.RIG12)        # decide's name for the same numbers
    assert np.array_equal(s2, src) and np.array_equal(d2, dy)


def test_row_errors_read_each_array_of_the_test_file_once(tmp_path, monkeypatch):
    npz = _sweep_with_one_sided_corners(tmp_path)
    reads = []
    real = np.lib.npyio.NpzFile.__getitem__

    def counting(self, key):
        reads.append(key)
        return real(self, key)

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", counting)
    pipeline.row_errors("KB4T", [S.I0, S.I1], S.C_TRUE, npz, S.RIG12)
    assert len(reads) == len(set(reads)), sorted(reads)


def test_summary_rows_are_the_median_of_the_row_errors_it_keeps(tmp_path):
    npz = _sweep_with_one_sided_corners(tmp_path)
    cal = S.write_calibration(tmp_path / "c.json", "KB4T", [S.I0, S.I1], S.C_TRUE)
    sm = pipeline.summary(cal, npz, S.RIG12)
    src, dy = pipeline.row_errors("KB4T", [S.I0, S.I1], S.C_TRUE, npz, S.RIG12)
    assert np.array_equal(sm.corner_src, src) and np.array_equal(sm.corner_dy, dy)
    assert sm.rows_px == float(np.median(np.abs(dy)))
    assert "corner_dy" not in sm.as_dict()


# ---------------------------------------------------------------- the decision: one bootstrap


def test_decision_bootstraps_once_for_both_verdicts(monkeypatch):
    from calibtool import replay
    TD = _test_module("test_decide")
    new, live, v66 = TD.synthetic(0.3, 0.3), TD.synthetic(), TD.step66()
    want = {k: decide.verdict(new, live, v66, decide.Gates(required_edges=e)).as_dict()
            for k, e in replay.VERDICT_KEYS.items()}
    want_paired = decide.paired(new, live, decide.Gates()).as_dict()
    n = []
    real = decide.paired
    monkeypatch.setattr(decide, "paired", lambda *a, **k: n.append(1) or real(*a, **k))
    dec = replay.decision(new, live, v66, decide.Gates())
    assert len(n) == 1
    assert dec["verdicts"] == replay._js(want) and dec["paired"] == replay._js(want_paired)


def test_verdict_takes_a_paired_comparison_made_with_other_required_edges_only():
    TD = _test_module("test_decide")
    new, live, v66 = TD.synthetic(0.3, 0.3), TD.synthetic(), TD.step66()
    pr = decide.paired(new, live, decide.Gates())
    v = decide.verdict(new, live, v66, decide.Gates(required_edges=frozenset(decide.EDGES)), paired_=pr)
    assert v.paired is pr and v.decision == "DEPLOY"
    with pytest.raises(ValueError, match="gates"):
        decide.verdict(new, live, v66, decide.Gates(boot=100), paired_=pr)


# ---------------------------------------------------------------- step 66's scores reused (no second scoring)


def _scored66(tmp_path, solver="fast"):
    """step 66 (in this process) of the true calibration on holdout_sweep() -> (calibration, test npz, report dir)"""
    np.savez_compressed(tmp_path / "t.npz", **S.holdout_sweep())
    cal = S.write_calibration(tmp_path / "c.json", "KB4T", [S.I0, S.I1], S.C_TRUE)
    v66 = importlib.import_module("66_validate_lens")
    with sf.solver_scope(solver):
        v66.main([str(cal), str(tmp_path / "t.npz"), str(tmp_path / "v"), "--rig", str(S.RIG12)])
    return cal, tmp_path / "t.npz", tmp_path / "v"


def test_step66_writes_the_scores_it_judged(tmp_path):
    import lensscore
    cal, npz, out = _scored66(tmp_path)
    js = json.loads((out / "scores.json").read_text())
    data = sf.Data(str(npz), S.BOARD)
    with sf.solver_scope("fast"):
        rec, _, solves = lensscore.score("KB4T", [S.I0, S.I1], S.C_TRUE, data)
    assert js["records"] == rec and js["solves"] == solves and js["solver"] == "fast"
    assert js["calibration_sha256"] == sf.calibration_digest("KB4T", [S.I0, S.I1], S.C_TRUE)
    assert js["test_sha256"] == sf.file_sha256(npz)


def test_summary_from_step66_scores_equals_a_fresh_scoring_without_scoring_again(tmp_path, monkeypatch):
    import lensscore
    cal, npz, out = _scored66(tmp_path)
    fresh = pipeline.summary(cal, npz, S.RIG12)
    got = pipeline.step66_scores(out, cal, npz, solver="fast")
    assert got is not None

    def no(*a, **k):
        raise AssertionError("scored again")

    monkeypatch.setattr(lensscore, "score", no)
    sm = pipeline.summary(cal, npz, S.RIG12, scores=got)
    assert sm.as_dict(records=True) == fresh.as_dict(records=True)
    sc = decide.score(cal, npz, S.RIG12, np.zeros(400, int), step66=out)
    assert sc.summary.as_dict(records=True) == fresh.as_dict(records=True)


def test_step66_scores_of_another_calibration_test_file_or_solver_are_not_used(tmp_path):
    cal, npz, out = _scored66(tmp_path)
    other = S.write_calibration(tmp_path / "o.json", "KB4T", [S.I0 * 1.001, S.I1], S.C_TRUE)
    assert pipeline.step66_scores(out, other, npz) is None
    np.savez_compressed(tmp_path / "u.npz", **S.holdout_sweep(seed=12))
    assert pipeline.step66_scores(out, cal, tmp_path / "u.npz") is None
    assert pipeline.step66_scores(out, cal, npz, solver="scipy") is None
    assert pipeline.step66_scores(tmp_path / "nowhere", cal, npz) is None


def _scheduled(monkeypatch, tmp_path, jobs, live_done_first):
    """evaluate() with fake steps that log start/end: step 45 waits (up to 5 s) for step 66 of the live config to end
    when live_done_first, else step 66 of the live config waits for step 66 of the new fit to start -> the log
    [(event, step, --jobs)]"""
    from calibtool import livecfg, replay
    _, args = _recording_steps(monkeypatch, tmp_path)
    TD = _test_module("test_decide")
    log, live_end, new_start = [], threading.Event(), threading.Event()
    fit = pipeline.fit45

    def fit45(*a, **k):
        log.append(("start", "45", 1))
        if jobs >= 2 and live_done_first:
            live_end.wait(5)
        log.append(("end", "45", 1))
        return fit(*a, **k)

    def step(name):
        def f(*a, **k):
            log.append(("start", name, k.get("jobs", 1)))
            if name == "new":
                new_start.set()
            elif jobs >= 2 and not live_done_first:
                new_start.wait(5)
            log.append(("end", name, k.get("jobs", 1)))
            if name == "live":
                live_end.set()
            v = TD.step66()
            v.as_dict = lambda: {"verdict": "PASS"}
            return v
        return f

    monkeypatch.setattr(pipeline, "fit45", fit45)
    monkeypatch.setattr(pipeline, "validate66", step("new"))
    monkeypatch.setattr(livecfg, "validate66", step("live"))
    ev = replay.evaluate(*args, jobs=jobs)
    assert ev["verdicts"]
    return log


def test_one_process_runs_the_steps_one_after_the_other(tmp_path, monkeypatch):
    log = _scheduled(monkeypatch, tmp_path, 1, True)
    assert log == [("start", "45", 1), ("end", "45", 1), ("start", "new", 1), ("end", "new", 1),
                   ("start", "live", 1), ("end", "live", 1)]


@pytest.mark.parametrize("jobs, live_jobs", [(2, 1), (3, 2), (4, 2)])
def test_step66_of_the_live_config_runs_beside_the_fit(jobs, live_jobs, tmp_path, monkeypatch):
    """it needs only the TEST split: it starts with step 45 (live_jobs processes beside step 45's one); step 66 of the
    new fit then gets two processes (its self-check forked) when the live one is done"""
    log = _scheduled(monkeypatch, tmp_path, jobs, True)
    assert log.index(("end", "live", live_jobs)) < log.index(("end", "45", 1)) < log.index(("start", "new", 2))


@pytest.mark.parametrize("jobs, new_jobs", [(2, 1), (3, 1), (4, 2)])
def test_step66_of_the_new_fit_shares_the_processes_with_a_live_one_still_running(jobs, new_jobs, tmp_path,
                                                                                   monkeypatch):
    log = _scheduled(monkeypatch, tmp_path, jobs, False)
    assert ("start", "new", new_jobs) in log
    assert log.index(("start", "new", new_jobs)) < log.index(("end", "live", min(2, jobs - 1)))


# ---------------------------------------------------------------- step 66 --jobs 2: the self-check in a second process


def test_step66_with_its_self_check_in_a_second_process_reports_the_same(tmp_path):
    """66_validate_lens --jobs 2 (as its own process, as calibtool runs it): the same report, self-check included,
    and the same scores as --jobs 1"""
    np.savez_compressed(tmp_path / "t.npz", **S.holdout_sweep())
    cal = S.write_calibration(tmp_path / "c.json", "KB4T", [S.I0, S.I1], S.C_TRUE)
    out = {}
    for j in (1, 2):
        v = pipeline.validate66(cal, tmp_path / "t.npz", tmp_path / f"v{j}", S.RIG12, jobs=j)
        out[j] = (v, json.loads((tmp_path / f"v{j}" / "scores.json").read_text()))
    (v1, s1), (v2, s2) = out[1], out[2]
    assert v1.verdict == v2.verdict and v1.entries == v2.entries
    assert v1.report["self_check"] == v2.report["self_check"] and v2.report["self_check"]["passed"]
    assert s1 == s2
    assert v2.report.get("jobs") == 2 and v1.report.get("jobs", 1) == 1
