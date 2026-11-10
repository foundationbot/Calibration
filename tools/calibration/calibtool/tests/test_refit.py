"""calibtool.refit: the live refit during collection (design plan 4.3: "refit in the background after each grid").

After each placement the FIT corners so far are fitted again, warm from the previous placement's solution: its camera
and stereo pose, and its board poses for the pairs it already had (PnP with its camera only for the new pairs). Then
step 45's quality gate (holds over the rms or depth gate dropped, warm refit). The last refit, on the session's final
FIT corners, reaches step 45's fit from nominal. BackgroundRefit runs the same in one process of its own (spawned),
its state kept there from one placement to the next."""
import importlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

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
    lr = refit.LiveRefit(S.RIG12)
    r = lr.update(d)
    assert bad in r.dropped_holds and r.converged
    assert r.holds == len(np.unique(d["img_hold"])) - len(r.dropped_holds)
    r2 = lr.update(d)                       # judged again on the next refit's fit (step 45 judges its whole data)
    assert r2.warm and bad in r2.dropped_holds and r2.holds == r.holds


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
    step 45's fit from nominal on the same FIT file (fast solver: the same minimum, measured 9 Nov to 2e-12 px).
    Measured on Moonstone, one thread: p21 0.5-6.1 s per placement, final 6.0 s vs step 45 7.0 s; p10 0.9-7.1 s,
    final 7.0 s vs 8.8 s (the gate's second fit each time: the holds are judged afresh at every refit)"""
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


# ---------------------------------------------------------------- the final fit = the last live refit (10 Nov 2026)


def _reordered(d, seed=0):
    """the same corners in another layout: pairs in another order (renumbered), corner rows shuffled, hold ids moved"""
    d = {k: np.asarray(v) for k, v in d.items()}
    rng = np.random.default_rng(seed)
    p = rng.permutation(len(d["img_t0"]))
    new_of_old = np.argsort(p)
    out = {k: d[k][p] for k in ("img_t0", "img_t1")}
    out["img_hold"] = d["img_hold"][p] + 1000
    c = rng.permutation(len(d["det_u"]))
    out.update({k: d[k][c] for k in ("det_cam", "det_tag", "det_k", "det_u", "det_v")})
    out["det_img"] = new_of_old[d["det_img"][c]]
    return out


def test_the_corner_digest_names_the_corners_not_their_layout():
    d = {k: np.asarray(v) for k, v in S.fit_sweep().items()}
    a = refit.corner_digest(d)
    assert refit.corner_digest(_reordered(d)) == a
    moved = dict(d, det_u=d["det_u"] + np.r_[0.01, np.zeros(len(d["det_u"]) - 1)])
    assert refit.corner_digest(moved) != a
    assert refit.corner_digest(_placements(d)[1]) != a
    regroup = dict(d, img_hold=np.where(d["img_hold"] == d["img_hold"][0], d["img_hold"][-1], d["img_hold"]))
    assert refit.corner_digest(regroup) != a


def _npz(tmp_path, d, name="fit.npz"):
    p = tmp_path / name
    np.savez_compressed(p, **{k: np.asarray(v) for k, v in d.items()})
    return p


def test_the_refit_on_the_final_split_is_written_as_step_45s_result(tmp_path, monkeypatch):
    """final_fit: step 45's JSON from the refit (every key step 45 writes, plus "source"), the same fit as step 45
    from nominal on the same corner file (same solver), and a FitResult replay.evaluate takes"""
    monkeypatch.setenv("CALIB_SOLVER", "fast")
    m45 = importlib.import_module("45_fit_lens")
    npz = _npz(tmp_path, S.fit_sweep())
    assert m45.main([str(npz), str(tmp_path / "ref.json"), "--rig", str(S.RIG12), "--max-z", str(refit.MAX_Z)]) == 0
    ref = json.loads((tmp_path / "ref.json").read_text())
    r = refit.LiveRefit(S.RIG12).update(dict(np.load(npz)), "final split")
    fit, why = refit.final_fit(r, npz, tmp_path / "fit", S.RIG12)
    assert why is None and fit.converged and fit.returncode == 0 and fit.json_path.is_file()
    e = json.loads(fit.json_path.read_text())          # strict JSON
    assert set(e) == set(ref) | {"source"}
    assert e["source"]["placement"] == "final split" and e["inputs"]["sha256"] == ref["inputs"]["sha256"]
    assert e["verdict"].startswith("converged") and e["fit_pairs"] == ref["fit_pairs"] == fit.fit_pairs
    assert e["kept_holds"] == ref["kept_holds"] and e["dropped_holds"] == ref["dropped_holds"]
    assert e["gates"] == ref["gates"] and e["trim_settled"]
    for c in (0, 1):
        assert _grid_diff("KB4T", e["intr"], ref["intr"], c) < 1e-6
    assert abs(e["baseline_mm"] - ref["baseline_mm"]) < 1e-6
    assert "step 45 not run" in fit.log_path.read_text()


def test_a_refit_that_does_not_cover_the_split_or_failed_a_check_is_not_the_final_fit(tmp_path):
    from dataclasses import replace
    parts = _placements(S.fit_sweep())
    r = refit.LiveRefit(S.RIG12).update(parts[1], "P2")
    assert refit.final_fit_refusal(r, _npz(tmp_path, parts[1], "same.npz")) is None
    assert refit.final_fit_refusal(r, _npz(tmp_path, _reordered(parts[1]), "layout.npz")) is None
    fit, why = refit.final_fit(r, _npz(tmp_path, parts[2], "other.npz"), tmp_path / "fit", S.RIG12)
    assert fit is None and "does not cover exactly" in why and not (tmp_path / "fit").exists()
    npz = tmp_path / "same.npz"
    for bad, word in ((replace(r, rejected=True), "gate"), (replace(r, converged=False), "gate"),
                      (replace(r, step45={**r.step45, "verdict": "camera sides: camera 1 at x = -44 mm"}), "sides"),
                      (replace(r, step45={**r.step45, "verdict": "unconverged: 3 attempts"}), "unconverged"),
                      (None, "no live refit")):
        assert word in refit.final_fit_refusal(bad, npz)


class _Done:
    """a finished Future"""

    def __init__(self, value=None, error=None):
        self.value, self.error = value, error

    def result(self, timeout=None):
        if self.error is not None:
            raise self.error
        return self.value

    def add_done_callback(self, fn):
        fn(self)


class FakeBg:
    """BackgroundRefit's interface over a recording LiveRefit.update stand-in (no fit)"""

    def __init__(self, error=None):
        self.seen, self.closed, self.error = [], False, error

    def submit(self, corners, placement=""):
        self.seen.append((placement, {k: np.asarray(v) for k, v in corners.items()}))
        r = refit.Refit(len(self.seen), placement, len(corners["img_t0"]), 0, 0, len(self.seen) > 1, True, False,
                        0.4, 44.0, [], 0, 10, 0.1, "KB4T", [np.zeros(10)] * 2, np.eye(4),
                        digest=refit.corner_digest(corners),
                        step45={"verdict": "converged: fake", "rms_px": 0.4, "baseline_mm": 44.0, "fit_pairs":
                                len(corners["img_t0"]), "fit_holds": 1, "gates": {}})
        return _Done(error=self.error) if self.error else _Done(r)

    def close(self):
        self.closed = True


class Labeler:
    """the capture labeler's interface over a session: FIT = every other pair of the placements seen so far; the
    split releases the first FIT pair to TEST (as finish() releases held-back holds)"""

    def __init__(self, session):
        self.s, self.seen = session, []

    def pair_labels(self):
        from calibtool import assign
        t0, take = self.s.pairs["t0"], self.s.pairs["take"]
        idx = np.flatnonzero(np.isin(take, self.seen))
        return {int(t0[i]): (assign.FIT if k % 2 == 0 else assign.TEST) for k, i in enumerate(idx)}

    def write_split(self, session, out_dir):
        from calibtool import assign
        out_dir.mkdir(parents=True, exist_ok=True)
        lab = self.pair_labels()
        fit = [t for t, v in lab.items() if v == assign.FIT][1:]
        mask = np.isin(session.pairs["t0"], fit)
        f = out_dir / "fit.npz"
        np.savez_compressed(f, **session.to_npz_dict(mask))
        return SimpleNamespace(fit_npz=f)


def _collection(tmp_path):
    from calibtool import pipeline, replay
    TR = _test_module("test_replay")
    session = replay.load_session(TR.synth_stored(pipeline.load_rig(TR.RIG_PATH), tmp_path))
    return TR, session, SimpleNamespace(labeler=Labeler(session))


def test_live_fit_refits_after_each_placement_and_once_on_the_final_split(tmp_path):
    TR, session, col = _collection(tmp_path)
    bg = FakeBg()
    lf = refit.LiveFit(TR.RIG_PATH, make=lambda: bg, to_session=lambda holds: session)
    for name in session.take_order:
        col.labeler.seen.append(name)
        assert lf.placement_done(col, None, name) is not None
        n_fit = sum(v == "FIT" for v in col.labeler.pair_labels().values())
        assert len(bg.seen[-1][1]["img_t0"]) == n_fit
    assert lf.final(tmp_path / "x.npz", tmp_path / "fit") == (None, "no live refit on the final split")
    lf.collection_done(col, None, tmp_path / "work")
    assert [p for p, _ in bg.seen] == list(session.take_order) + ["final split"]
    live_split = np.load(tmp_path / "work" / "live-split" / "fit.npz")
    assert np.array_equal(np.sort(bg.seen[-1][1]["img_t0"]), np.sort(live_split["img_t0"]))
    assert len(live_split["img_t0"]) == len(bg.seen[-2][1]["img_t0"]) - 1          # one pair released
    # fit_decide writes the split again (its own folder): the same corners, so the refit stands for step 45
    split = col.labeler.write_split(session, tmp_path / "decide" / "split")
    fit, why = lf.final(split.fit_npz, tmp_path / "decide" / "fit")
    assert why is None and fit.entry["source"]["placement"] == "final split" and fit.converged
    # the last placement's refit (before the release) does not cover the split: step 45 would run
    other = _npz(tmp_path, bg.seen[-2][1], "other.npz")
    fit, why = lf.final(other, tmp_path / "other-fit")
    assert fit is None and "does not cover exactly" in why
    # a placement after the final refit (a redo) makes the final refit stale until the next collection_done
    col.labeler.seen.append(session.take_order[0])
    lf.placement_done(col, None, "redo")
    assert lf.final(split.fit_npz, tmp_path / "f2") == (None, "no live refit on the final split")
    lf.close()
    assert bg.closed


def test_a_failed_final_refit_falls_back_to_step_45(tmp_path):
    TR, session, col = _collection(tmp_path)
    col.labeler.seen.extend(session.take_order)
    lf = refit.LiveFit(TR.RIG_PATH, make=lambda: FakeBg(error=RuntimeError("worker died")),
                       to_session=lambda holds: session)
    lf.collection_done(col, None, tmp_path / "work")
    fit, why = lf.final(tmp_path / "work" / "live-split" / "fit.npz", tmp_path / "fit")
    assert fit is None and "worker died" in why


@pytest.mark.replay
@pytest.mark.parametrize("name", ["p21_1105", "p10_1106"])
def test_the_final_live_refit_stands_for_step_45_on_a_stored_session(name, data_root, tmp_path):
    """the live flow's path on a stored session: a background refit after each placement (capture order, capture-time
    labels), one on the final split once the collection ends; at FIT that refit covers exactly the split fit_decide
    writes, and its step-45 JSON is step 45's fit from nominal on that file (projection < 1e-3 px; measured 2e-12).
    Printed (LIVEFIT line): step 45's time, the final refit's own time and the wait for it at FIT"""
    import time as _time
    from calibtool import assign, pipeline, replay, sessions
    st = sessions.get(name)
    session = replay.load_session(st)
    rig = pipeline.load_rig(st.rig_path)
    col = replay.Collection.for_session(session, rig, "capture", max_fit_z=refit.MAX_Z)
    lf = refit.LiveFit(st.rig_path, to_session=lambda holds: session)
    cur = None
    try:
        for h in assign.capture_holds(session):
            if cur is not None and h.placement != cur:
                col.end_placement(cur)
                lf.placement_done(col, None, cur)
            col.add(h)
            cur = h.placement
        col.end_placement(cur)
        lf.placement_done(col, None, cur)
        lf.collection_done(col, None, tmp_path / "work")
        split = col.labeler.write_split(session, tmp_path / "decide" / "split")
        t = _time.time()
        fit, why = lf.final(split.fit_npz, tmp_path / "decide" / "fit")
        waited = _time.time() - t
    finally:
        lf.close()
    assert why is None, why
    ref = pipeline.fit45(split.fit_npz, st.rig_path, tmp_path / "ref", max_z=refit.MAX_Z)
    print(f"LIVEFIT {name}: step 45 {ref.seconds:.1f} s; final refit {fit.entry['source']['seconds']:.1f} s "
          f"(warm {fit.entry['source']['warm']}), waited {waited:.1f} s at FIT; pairs {fit.fit_pairs}")
    assert ref.converged and fit.fit_pairs == ref.fit_pairs and fit.entry["kept_holds"] == ref.entry["kept_holds"]
    g = np.stack(np.meshgrid(np.arange(0, 1280, 32.0), np.arange(0, 800, 32.0)), -1).reshape(-1, 2)
    for c in (0, 1):
        ray = S.lensmodel.unproject("KB4T", ref.entry["intr"][c], g)
        assert np.abs(S.lensmodel.project("KB4T", fit.entry["intr"][c], ray) - g).max() < 1e-3
