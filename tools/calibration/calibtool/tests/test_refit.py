"""calibtool.refit: the live refit during collection (design plan 4.3: "refit in the background after each grid").

After each placement the FIT corners so far are fitted again, warm from the previous placement's solution: its camera
and stereo pose, and its board poses for the pairs it already had (PnP with its camera only for the new pairs). Then
step 45's quality gate (holds over the rms or depth gate dropped, warm refit). The last refit, on the session's final
FIT corners, reaches step 45's fit from nominal. BackgroundRefit runs the same in one process of its own (spawned),
its state kept there from one placement to the next."""
import importlib
import importlib.util
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
from calibtool import refit  # noqa: E402


def _placements(d, cuts=(16, 32)):
    """fit_sweep()'s pairs as three placements by hold (0-15, 16-31, 32-47): the corners of placements 1..k, k = 1..3"""
    d = {k: np.asarray(v) for k, v in d.items()}
    out = []
    for hi in list(cuts) + [int(d["img_hold"].max()) + 1]:
        keep = np.flatnonzero(d["img_hold"] < hi)
        m = np.isin(d["det_img"], keep)
        remap = np.full(len(d["img_hold"]), -1)
        remap[keep] = np.arange(len(keep))
        out.append({"img_hold": d["img_hold"][keep], "img_t0": d["img_t0"][keep], "img_t1": d["img_t1"][keep],
                    **{k: d[k][m] for k in ("det_cam", "det_tag", "det_k", "det_u", "det_v")},
                    "det_img": remap[d["det_img"][m]]})
    return out


def _grid_diff(model, a, b, cam):
    g = np.stack(np.meshgrid(np.arange(0, S.W, 64.0), np.arange(0, S.H, 64.0)), -1).reshape(-1, 2)
    ray = S.lensmodel.unproject(model, a[cam], g)
    return float(np.abs(S.lensmodel.project(model, b[cam], ray) - g).max())


def test_each_placement_refits_warm_from_the_previous_one_and_ends_at_the_cold_fit(monkeypatch):
    parts = _placements(S.fit_sweep())
    pnp_sizes = []
    real = sf.pnp_poses
    monkeypatch.setattr(sf, "pnp_poses", lambda data, *a, **k: pnp_sizes.append(data.n) or real(data, *a, **k))
    lr = refit.LiveRefit(S.RIG12)
    out = [lr.update(p, placement=f"P{i + 1}") for i, p in enumerate(parts)]
    assert [r.warm for r in out] == [False, True, True]
    assert [r.placement for r in out] == ["P1", "P2", "P3"]
    assert [r.pairs for r in out] == [len(p["img_t0"]) for p in parts]
    assert [r.new_pairs for r in out] == [len(parts[0]["img_t0"])] + [len(b["img_t0"]) - len(a["img_t0"])
                                                                      for a, b in zip(parts, parts[1:])]
    assert all(r.converged for r in out)
    # PnP only for the pairs a refit had not seen (the first: every pair)
    assert pnp_sizes == [r.new_pairs for r in out]
    with sf.solver_scope("fast"):          # the same solver (scipy at its default tolerance stops ~0.01 px short)
        cold_data, cold = sf.fit_from_nominal(sf.Data(parts[-1], S.BOARD), "KB4T",
                                              *sf.nominal_start(S.rig12(), "KB4T"), log=lambda *a: None)
    last = out[-1]
    for c in (0, 1):
        assert _grid_diff("KB4T", last.intr, cold["intr"], c) < 1e-3
    assert abs(last.baseline_mm - 1e3 * np.linalg.norm(cold["C"][:3, 3])) < 1e-3
    assert np.allclose(last.C, cold["C"], atol=1e-6)


def test_the_quality_gate_drops_a_bad_hold_as_step_45_does():
    d = {k: np.asarray(v).copy() for k, v in S.fit_sweep().items()}
    bad = d["img_hold"][5]
    m = np.isin(d["det_img"], np.flatnonzero(d["img_hold"] == bad)) & (d["det_cam"] == 0)
    d["det_u"][m] += np.linspace(-6, 6, m.sum())             # a bent board: ~3.5 px rms in camera 0
    r = refit.LiveRefit(S.RIG12).update(d)
    assert bad in r.dropped_holds and r.converged
    assert r.holds == len(np.unique(d["img_hold"])) - len(r.dropped_holds)


def test_a_refit_with_fewer_pairs_than_before_keeps_the_ones_it_has():
    """the session's end may move held-back FIT holds to TEST: the last refit then has fewer pairs, all known"""
    parts = _placements(S.fit_sweep())
    lr = refit.LiveRefit(S.RIG12)
    lr.update(parts[-1])
    r = lr.update(parts[1])
    assert r.warm and r.new_pairs == 0 and r.pairs == len(parts[1]["img_t0"]) and r.converged


def test_background_refit_keeps_its_state_in_its_own_process():
    parts = _placements(S.fit_sweep())
    with refit.BackgroundRefit(S.RIG12) as bg:
        f1 = bg.submit(parts[0], placement="P1")
        f2 = bg.submit(parts[1], placement="P2")
        r1, r2 = f1.result(timeout=300), f2.result(timeout=300)
    here = refit.LiveRefit(S.RIG12)
    h1, h2 = here.update(parts[0], placement="P1"), here.update(parts[1], placement="P2")
    assert (r1.warm, r2.warm) == (False, True)
    assert r2.pid == r1.pid and r2.pid != h2.pid
    assert np.allclose(r2.C, h2.C, atol=1e-9) and np.allclose(r2.intr, h2.intr, atol=1e-9)


def test_the_solver_is_chosen_per_refit_and_scipy_works(monkeypatch):
    monkeypatch.setenv("CALIB_SOLVER", "fast")
    parts = _placements(S.fit_sweep(), cuts=(40,))
    seen = []
    real = sf.solve
    monkeypatch.setattr(sf, "solve", lambda *a, **k: seen.append(sf.solver_name()) or real(*a, **k))
    r = refit.LiveRefit(S.RIG12, solver="scipy").update(parts[0])
    assert r.converged and set(seen) == {"scipy"}


# ---------------------------------------------------------------- a stored session, placement by placement


def _test_module(name):
    spec = importlib.util.spec_from_file_location(f"_tr_{name}", Path(__file__).with_name(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_on_session_refits_after_each_placement_and_once_on_the_final_split(tmp_path, monkeypatch):
    """refit.on_session: the session's holds through the capture-time assignment in capture order; after each
    placement a refit on the FIT pairs so far (capture-time labels, pairs within max_z); at the end one more on the
    final split's FIT file (held-back holds released to TEST)"""
    from calibtool import pipeline
    TR = _test_module("test_replay")
    rig = pipeline.load_rig(TR.RIG_PATH)
    stored = TR.synth_stored(rig, tmp_path)
    seen = []

    def fake(self, fit_corners, placement=""):
        seen.append((placement, np.asarray(fit_corners["img_t0"]).copy()))
        return refit.Refit(len(seen), placement, len(fit_corners["img_t0"]), 0, 0, len(seen) > 1, True, False, 0.0,
                           0.0, [], 0, 0, 0.0, "KB4T", [], np.eye(4))

    monkeypatch.setattr(refit.LiveRefit, "update", fake)
    out = refit.on_session(stored, tmp_path / "o")
    assert [p for p, _ in seen] == TR.CAPTURE + ["final split"]
    assert [r["placement"] for r in out["refits"]] == TR.CAPTURE + ["final split"]
    fit_t0 = np.load(tmp_path / "o" / "split" / "fit.npz")["img_t0"]
    assert np.array_equal(np.sort(seen[-1][1]), np.sort(fit_t0))
    for (_, a), (_, b) in zip(seen, seen[1:-1]):
        assert set(a.tolist()) <= set(b.tolist())                # FIT pairs only grow while capturing


@pytest.mark.replay
@pytest.mark.parametrize("name", ["p21_1105", "p10_1106"])
def test_live_refit_on_a_stored_session_ends_at_step_45s_fit(name, data_root, tmp_path):
    """every placement's refit converges within the fit budget (design 10.4: 30 s); the refit on the final split is
    step 45's fit from nominal on the same FIT file (fast solver: the same minimum, measured 9 Nov to 1e-12 px), in
    less time. Measured on Moonstone, one thread: p21 0.8-9.1 s per placement, final 4.7 s vs step 45 11.3 s; p10
    1.2-5.2 s, final 4.9 s vs 11.1 s"""
    import lensmodel
    from calibtool import pipeline, sessions
    st = sessions.get(name)
    res = refit.on_session(st, tmp_path, log=print)
    f = pipeline.fit45(res["split"]["fit_npz"], st.rig_path, tmp_path / "fit", max_z=refit.MAX_Z)
    rows = [(r["placement"], r["pairs"], r["new_pairs"], round(r["seconds"], 1), r["nfev"]) for r in res["refits"]]
    print(f"REFIT {name}: {rows}; step 45 {f.seconds:.1f} s")
    assert all(r["converged"] and not r["rejected"] for r in res["refits"])
    assert all(r["seconds"] < 30 for r in res["refits"])
    last = res["refits"][-1]
    assert last["placement"] == "final split" and last["pairs"] == f.fit_pairs and f.converged
    g = np.stack(np.meshgrid(np.arange(0, 1280, 32.0), np.arange(0, 800, 32.0)), -1).reshape(-1, 2)
    for c in (0, 1):
        ray = lensmodel.unproject("KB4T", f.entry["intr"][c], g)
        assert np.abs(lensmodel.project("KB4T", last["intr"][c], ray) - g).max() < 1e-3
    assert np.abs(np.array(last["C"])[:3, 3] - np.array(f.entry["C"])[:3, 3]).max() < 1e-6       # m
    assert last["seconds"] < f.seconds
