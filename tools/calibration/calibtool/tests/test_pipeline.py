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
    assert calls == {"fit45": [want], "validate66": [want], "live66": [want], "score": [want, want]}


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
