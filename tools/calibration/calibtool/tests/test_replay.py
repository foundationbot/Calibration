"""calibtool.replay: a stored session replayed as if live. Holds go through the FIT/TEST assignment and the coverage
tallies in capture order, a snapshot (tallies, deficits, gap sentences, a progress line) is taken after each
placement, the planner suggests the next placements after the last one, and then the split is fitted (45), validated
(66, new and live) and decided (decide.verdict with the default required edges and with all four) into report.json
+ report.md.

Fast tests use synthetic takes (test_assign's generator) and a synthetic OKVIS config (test_livecfg's); stored-session
tests need CALIBTOOL_DATA; `replay` tests (-m replay) run the whole thing, fit included (minutes per session)."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("cv2")
pytest.importorskip("yaml")

from calibtool import assign, coverage as cv, decide, pipeline, sessions  # noqa: E402
from calibtool.session import Session  # noqa: E402

HERE = Path(__file__).resolve().parent
RIG_PATH = sessions.get("p21_1105").rig_path
P21_LEFT = "Left camera, left edge: 176 corners vs 1,099 on the right edge (6x fewer)."
P21_BOTTOM_LEFT = "Left camera, bottom-left corner: 50 corners vs 588 in the bottom-right corner (12x fewer)."
DEFAULT, ALL4 = "required_edges=left,right,bottom", "required_edges=all"     # the report's two verdicts
REPORT_KEYS = {"session", "robot", "assignment", "exclude_takes", "take_order", "capture_order", "placements",
               "final", "suggestions", "split", "fit", "validate_new", "validate_live", "live", "stats", "paired",
               "verdicts", "timings", "stop", "targets", "per_hold"}


def _load_test_module(name):
    spec = importlib.util.spec_from_file_location(f"_t_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TA = _load_test_module("test_assign")         # synth_take: synthetic step-15b takes, distance and cell by construction
TL = _load_test_module("test_livecfg")        # okvis_yaml: a minimal OKVIS config
TD = _load_test_module("test_decide")         # synthetic paired scores, step-66 stand-in


@pytest.fixture(scope="module")
def rig():
    return pipeline.load_rig(RIG_PATH)


def replay():
    from calibtool import replay as rp
    return rp


# three placements; recorded in the order C, A, B (capture order) while the take order lists A, B, C
TAKES = {"A": [("centre", 0.5, 2), ("left", 0.75, 2), ("right", 1.1, 2), ("top", 0.75, 2)],
         "B": [("bottom", 0.75, 2), ("left", 0.5, 2), ("centre", 3.5, 2), ("right", 0.75, 3)],
         "C": [("centre", 1.1, 2), ("bottom-left", 0.75, 2), ("left", 1.1, 1)]}
CAPTURE = ["C", "A", "B"]


def synth_files(board, tmp_path=None):
    """[(take, corner dict or npz path)] in take order A, B, C, recorded C first"""
    out = []
    for name in ("A", "B", "C"):
        d = TA.synth_take(board, TAKES[name], t_start=(CAPTURE.index(name) + 1) * 10**12, seed=ord(name))
        if tmp_path is not None:
            p = Path(tmp_path) / f"{name}.npz"
            np.savez_compressed(p, **d)
            d = p
        out.append((name, d))
    return out


def synth_stored(rig, tmp_path):
    """a stored-session stand-in: corner files and a live OKVIS config (equidistant) written under tmp_path"""
    files = synth_files(rig.board, tmp_path)
    live = tmp_path / "live.yaml"
    live.write_text(TL.okvis_yaml("equidistant", (0.01, -0.02, 0.003, -0.001)))
    return SimpleNamespace(name="synth", robot="synthbot", rig_path=RIG_PATH, take_order=[t for t, _ in files],
                           corner_files=[p for _, p in files], live_config=live, live_model="KB4")


def fed(rig, session, assignment="capture"):
    """a Collection fed every hold of session in capture order, a snapshot after each placement"""
    rp = replay()
    col = rp.Collection.for_session(session, rig, assignment=assignment)
    snaps = rp.feed(col, assign.capture_holds(session))
    return col, snaps


# ---------------------------------------------------------------- the per-placement step (Collection)


def test_snapshots_one_per_placement_in_capture_order(rig):
    s = Session.from_files(synth_files(rig.board), name="synth")
    col, snaps = fed(rig, s)
    assert [x.placement for x in snaps] == CAPTURE
    assert [x.index for x in snaps] == [1, 2, 3]
    n = {k: sum(len(z) if isinstance(z, (list, tuple)) else nn for _, z, nn in v) for k, v in TAKES.items()}
    assert [x.placement_pairs for x in snaps] == [n["C"], n["A"], n["B"]]
    assert [x.pairs for x in snaps] == list(np.cumsum([n["C"], n["A"], n["B"]]))
    assert [x.placement_holds for x in snaps] == [len(TAKES[k]) for k in CAPTURE]
    assert snaps[-1].pairs == s.n_pairs
    for x in snaps:
        assert x.fit_pairs + x.test_pairs == x.pairs


def test_tallies_equal_coverage_on_the_whole_session(rig):
    """the incremental tallies are coverage's own, pair for pair: all corners, FIT corners, TEST entries"""
    s = Session.from_files(synth_files(rig.board), name="synth")
    col, snaps = fed(rig, s)
    z = cv.session_z(s, rig)
    allt = cv.fit_tally(s, rig=rig, z=z)
    labels = col.labels()
    near = np.nan_to_num(z, nan=np.inf) <= 3.0                   # a far pair is TEST whatever its hold (step 16)
    fit = np.array([labels[int(h)] == "FIT" for h in s.pairs["hold"]]) & near
    fitt = cv.fit_tally(s, mask=fit, rig=rig, z=z)
    test = cv.test_tally(s, labels, col.pp, z)
    last = col.tallies()
    for c in (0, 1):
        for mine, ref in ((last["all"].cams[c], allt.cams[c]), (last["fit"].cams[c], fitt.cams[c])):
            assert mine.n == ref.n and mine.rim == ref.rim and mine.edges == ref.edges and mine.boxes == ref.boxes
            assert mine.thirds.tolist() == ref.thirds.tolist()
    assert (last["all"].far_pairs, last["all"].n_pairs) == (allt.far_pairs, allt.n_pairs)
    assert (last["fit"].far_pairs, last["fit"].n_pairs) == (fitt.far_pairs, fitt.n_pairs)
    assert last["test"].entries == test.entries and last["test"].far_bands == test.far_bands


def test_capture_labels_are_the_assigners(rig):
    s = Session.from_files(synth_files(rig.board), name="synth")
    col, _ = fed(rig, s, "capture")
    a = assign.Assigner(rig, (rig.resolution[0] / 2, rig.resolution[1] / 2), rig.nominal_focal_px)
    ref = {h.hold: a.observe(h) for h in assign.capture_holds(s)}
    assert {h: i.captured for h, i in col.labeler.holds.items()} == ref
    a.finish()
    col.finish()
    assert col.labels() == {h: i.label for h, i in a.holds.items()}


def test_split16_labels_are_step16s(rig):
    s = Session.from_files(synth_files(rig.board), name="synth")
    col, _ = fed(rig, s, "split16")
    assert col.labels() == cv.split16_assignment(s, cv.session_z(s, rig))


def test_snapshot_has_progress_line_deficits_and_sentences(rig):
    s = Session.from_files(synth_files(rig.board), name="synth")
    _, snaps = fed(rig, s)
    for x in snaps:
        assert x.placement in x.progress and f"{x.pairs:,} pairs" in x.progress
        assert len(x.sentences) == len(x.deficits) > 0
        assert x.sentences == cv.sentences(x.deficits)
        d = x.as_dict()
        json.dumps(d)
        assert d["placement"] == x.placement and d["sentences"] == x.sentences
        assert {"all", "fit", "test"} <= set(d["tallies"])


def test_a_hold_after_the_last_snapshot_needs_its_own(rig):
    """end_placement closes what was added since the previous one; a second call with nothing new is refused"""
    rp = replay()
    s = Session.from_files(synth_files(rig.board), name="synth")
    col = rp.Collection.for_session(s, rig)
    holds = list(assign.capture_holds(s))
    for h in holds[:2]:
        col.add(h)
    x = col.end_placement()
    assert x.placement == "C" and x.placement_holds == 2
    with pytest.raises(ValueError, match="no hold"):
        col.end_placement()


def test_collection_needs_no_files(rig, tmp_path, monkeypatch):
    """the step M2 calls as holds arrive: holds as in-memory dicts, nothing read or written"""
    monkeypatch.chdir(tmp_path)
    rp = replay()
    files = synth_files(rig.board)                                # dicts, no npz
    s = Session.from_files(files, name="mem")
    col = rp.Collection(rig, rp.capture_labeler(rig), pp=(rig.resolution[0] / 2, rig.resolution[1] / 2))
    snaps = rp.feed(col, assign.capture_holds(s))
    assert len(snaps) == 3 and list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------- the stop rule


def only_pairs(n):
    """targets met as soon as the FIT pairs reach n (every other target 0)"""
    return cv.Targets.for_per_hold(3, pairs_min=n, third_min=0, rim_min=0, far_pairs_min=0, strip_min={},
                                   judged_min=0, side_holds=0, far_band_min=0)


def fit_pairs_in_capture_order(rig, s):
    """FIT pairs (step 16's split, near pairs of FIT holds) after each hold, capture order"""
    z = cv.session_z(s, rig)
    a = cv.split16_assignment(s, z)
    near = np.nan_to_num(z, nan=np.inf) <= 3.0
    return list(np.cumsum([int((near[h.pairs]).sum()) if a[h.hold] == "FIT" else 0 for h in s.holds()]))


def test_stop_point_is_the_first_hold_that_meets_the_rule(rig):
    rp = replay()
    s = Session.from_files(synth_files(rig.board), name="synth")
    cum = fit_pairs_in_capture_order(rig, s)
    n = cum[-1] // 2
    want = next(i for i, c in enumerate(cum) if c >= n)
    hs = list(s.holds())
    col = rp.Collection.for_session(s, rig, "split16", targets=only_pairs(n))
    snaps = rp.feed(col, assign.capture_holds(s))
    st = col.stop
    assert st is not None and st.index == want + 1 and st.hold == hs[want].hold
    assert (st.placement, st.pose) == (hs[want].take, hs[want].pose)
    assert st.placement_index == CAPTURE.index(hs[want].take) + 1
    assert np.isfinite(st.neck_pitch_deg) and np.isfinite(st.neck_yaw_deg)
    met = [x.stop_met for x in snaps]
    assert met == sorted(met) and met[-1] and met[st.placement_index - 1] and not any(met[:st.placement_index - 1])
    col = rp.Collection.for_session(s, rig, "split16", targets=only_pairs(10**6))
    snaps = rp.feed(col, assign.capture_holds(s))
    assert col.stop is None and not any(x.stop_met for x in snaps)


def test_stop_rule_uses_the_labels_finish_would_leave(rig):
    """the capture-time assignment holds holds back while one placement feeds an entry; if the session stopped now,
    finish() would release some to TEST: the stop rule counts with those labels (the split it would write)"""
    rp = replay()
    s = TA.synth_session(rig.board, [("P1", [("centre", 0.5, 2)] * 14)])
    col = rp.Collection.for_session(s, rig, "capture", targets=only_pairs(0))
    rp.feed(col, assign.capture_holds(s))
    assert col.stop_labels() == col.labeler.labels_if_finished() != col.labels()
    t = col.tallies(col.stop_labels())
    assert t["test"].entries[("centre", (0.4, 0.6))] == 7


def test_snapshot_counts_with_the_labels_finish_would_leave(rig):
    """what the operator is told after a placement (counts, TEST entries, deficits, sentences) is the split the session
    would write if it ended there: the stop rule's labels, so the sentences and the stop rule never disagree"""
    rp = replay()
    s = TA.synth_session(rig.board, [("P1", [("centre", 0.5, 2)] * 14)])
    col = rp.Collection.for_session(s, rig, "capture")
    snap = rp.feed(col, assign.capture_holds(s))[-1]
    assert snap.tallies["test"].entries[("centre", (0.4, 0.6))] == 7
    lab = col.stop_labels()
    assert snap.fit_pairs == sum(len(h.pairs) for h in s.holds() if lab[h.hold] == "FIT")
    assert snap.deficits == col.coverage(col.tallies(lab)).deficits(col.targets)


def test_run_stops_at_the_rule(rig, tmp_path):
    """stop_at_rule=True: the session is cut at the stop point (the holds fed until the rule was met) and everything
    after (snapshots, planner, split) runs on that part; without it the report still says where the rule was met"""
    rp = replay()
    st = synth_stored(rig, tmp_path)
    s = rp.load_session(st)
    cum = fit_pairs_in_capture_order(rig, s)
    n = cum[-1] // 2
    want = next(i for i, c in enumerate(cum) if c >= n) + 1
    full = rp.run(st, assignment="split16", out_dir=tmp_path / "full", fit=False, targets=only_pairs(n))
    assert full["stop"]["index"] == want and full["stop"]["holds_total"] == len(cum)
    assert full["stop"]["stopped"] is False and full["placements"][-1]["holds"] == len(cum)
    cut = rp.run(st, assignment="split16", out_dir=tmp_path / "cut", fit=False, targets=only_pairs(n),
                 stop_at_rule=True)
    assert cut["stop"]["stopped"] is True and cut["stop"]["index"] == want
    assert cut["placements"][-1]["holds"] == want and cut["placements"][-1]["stop_met"]
    assert cut["capture_order"] == CAPTURE[:full["stop"]["placement_index"]]
    assert cut["targets"]["pairs_min"] == n and cut["per_hold"] == 3
    md = (tmp_path / "cut" / "report.md").read_text()
    assert "Stop rule met" in md
    never = rp.run(st, assignment="split16", out_dir=tmp_path / "never", fit=False, targets=only_pairs(10**6),
                   stop_at_rule=True)
    assert never["stop"]["index"] is None and never["placements"][-1]["holds"] == len(cum)


def test_run_targets_default_to_the_stored_sessions_per_hold(rig, tmp_path):
    js = replay().run(synth_stored(rig, tmp_path), out_dir=tmp_path / "o", fit=False)
    assert js["per_hold"] == cv.STORED_PER_HOLD == 3
    assert js["targets"]["pairs_min"] == cv.Targets.for_per_hold(3).pairs_min


# ---------------------------------------------------------------- run(): the report (no fit)


def test_run_without_fit_writes_report(rig, tmp_path):
    rp = replay()
    st = synth_stored(rig, tmp_path)
    rep = rp.run(st, out_dir=tmp_path / "out", fit=False)
    js = json.loads((tmp_path / "out" / "report.json").read_text())
    assert set(js) == REPORT_KEYS
    assert js == json.loads(json.dumps(rep, default=str))
    assert js["capture_order"] == CAPTURE and [p["placement"] for p in js["placements"]] == CAPTURE
    assert js["take_order"] == ["A", "B", "C"] and js["exclude_takes"] == []
    assert js["fit"] is None and js["verdicts"] is None
    assert isinstance(js["suggestions"], list)
    for s in js["suggestions"]:
        assert {"say", "distance_m", "bearing_deg", "tilted", "target", "poses"} <= set(s)
    md = (tmp_path / "out" / "report.md").read_text()
    for p in js["placements"]:
        assert p["progress"] in md
        for sent in p["sentences"][:3]:
            assert sent in md
    assert "Next placements" in md and "not run" in md


def test_exclude_takes_drops_them(rig, tmp_path):
    rp = replay()
    st = synth_stored(rig, tmp_path)
    js = rp.run(st, out_dir=tmp_path / "out", fit=False, exclude_takes=("A",))
    assert js["exclude_takes"] == ["A"] and js["capture_order"] == ["C", "B"]
    n = sum(len(z) if isinstance(z, (list, tuple)) else nn for k in ("B", "C") for _, z, nn in TAKES[k])
    assert js["placements"][-1]["pairs"] == n
    with pytest.raises(ValueError, match="nope"):
        rp.run(st, out_dir=tmp_path / "out2", fit=False, exclude_takes=("nope",))


def test_unknown_assignment_is_refused(rig, tmp_path):
    with pytest.raises(ValueError, match="assignment"):
        replay().run(synth_stored(rig, tmp_path), assignment="random", out_dir=tmp_path / "o", fit=False)


# ---------------------------------------------------------------- the decision part of the report


def test_decision_section_both_gates_and_paired_table():
    rp = replay()
    new, live = TD.synthetic(0.3, 0.3), TD.synthetic()
    dec = rp.decision(new, live, TD.step66(), decide.Gates())
    assert set(dec["verdicts"]) == {DEFAULT, ALL4}
    for k, want in ((DEFAULT, ["left", "right", "bottom"]), (ALL4, ["left", "right", "top", "bottom"])):
        assert dec["verdicts"][k]["decision"] == "DEPLOY"
        assert dec["verdicts"][k]["gates"]["required_edges"] == want
    names = [s["name"] for s in dec["paired"]["stats"]]
    assert names[:3] == ["depth", "spread", "rows"] and any(n.startswith("band ") for n in names)
    md = "\n".join(rp.decision_md(dec))
    assert "DEPLOY" in md and "| depth |" in md and "| rows |" in md
    for r in dec["verdicts"][DEFAULT]["reasons"]:
        assert r in md


def test_decision_without_live_scores():
    rp = replay()
    dec = rp.decision(TD.synthetic(), None, TD.step66(), decide.Gates(), live_error="live config not loaded: X")
    assert dec["paired"] is None
    assert {v["decision"] for v in dec["verdicts"].values()} == {"NOT_ENOUGH_DATA"}
    assert all(any("not loaded: X" in r for r in v["reasons"]) for v in dec["verdicts"].values())
    dec = rp.decision(TD.synthetic(), None, TD.step66(), decide.Gates(allow_deploy_without_live=True))
    assert {v["decision"] for v in dec["verdicts"].values()} == {"LIVE_NOT_SCORABLE_DEPLOY"}
    assert "LIVE_NOT_SCORABLE_DEPLOY" in "\n".join(rp.decision_md(dec))


def _fake_steps(monkeypatch, tmp_path, live_score_error=None):
    """evaluate() without the slow steps: step 45 and 66 stand-ins, decide.score of synthetic scores (the live config's
    JSON scores as TD.synthetic(), the new fit as 0.3 x it; live_score_error: raised when the live config is scored)
    -> evaluate's arguments after the session"""
    from calibtool import livecfg
    fit = SimpleNamespace(converged=True, json_path=tmp_path / "fit.json", verdict="converged",
                          as_dict=lambda: {"verdict": "converged"})
    v66 = TD.step66()
    v66.as_dict = lambda: {"verdict": "PASS"}
    monkeypatch.setattr(pipeline, "fit45", lambda *a, **k: fit)
    monkeypatch.setattr(pipeline, "validate66", lambda *a, **k: v66)
    monkeypatch.setattr(livecfg, "validate66", lambda *a, **k: v66)
    monkeypatch.setattr(decide, "placements", lambda *a, **k: None)

    def score(calib_json, *a, **k):
        if Path(calib_json).name.startswith("live"):
            if live_score_error is not None:
                raise live_score_error
            return TD.synthetic()
        return TD.synthetic(0.3, 0.3)

    monkeypatch.setattr(decide, "score", score)
    (tmp_path / "out").mkdir(exist_ok=True)
    split = SimpleNamespace(fit_npz=tmp_path / "fit.npz", test_npz=tmp_path / "test.npz")
    return RIG_PATH, split


def _live(tmp_path, dtype="equidistant"):
    p = tmp_path / f"live-{dtype}.yaml"
    p.write_text(TL.okvis_yaml(dtype, (0.01, -0.02, 0.003, -0.001)))
    return p


def test_evaluate_scores_a_readable_live_config(monkeypatch, tmp_path):
    rig_path, split = _fake_steps(monkeypatch, tmp_path)
    ev = replay().evaluate(None, rig_path, split, _live(tmp_path), tmp_path / "out", decide.Gates(), jobs=1)
    assert ev["live"]["scorable"] and ev["live"]["error"] is None
    assert ev["verdicts"][DEFAULT]["decision"] == "DEPLOY"


def test_evaluate_unsupported_live_config_is_not_enough_data_with_the_error(monkeypatch, tmp_path):
    """a live config the package cannot score (a distortion type it does not know): NOT_ENOUGH_DATA by default, the
    error text in every verdict's reasons; LIVE_NOT_SCORABLE_DEPLOY only with allow_deploy_without_live"""
    rig_path, split = _fake_steps(monkeypatch, tmp_path)
    live = _live(tmp_path, "omni")
    ev = replay().evaluate(None, rig_path, split, live, tmp_path / "out", decide.Gates(), jobs=1)
    assert not ev["live"]["scorable"] and "omni" in ev["live"]["error"]
    for v in ev["verdicts"].values():
        assert v["decision"] == "NOT_ENOUGH_DATA"
        assert any("omni" in r for r in v["reasons"]), v["reasons"]
    ev = replay().evaluate(None, rig_path, split, live, tmp_path / "out", decide.Gates(allow_deploy_without_live=True),
                           jobs=1)
    assert ev["verdicts"][DEFAULT]["decision"] == "LIVE_NOT_SCORABLE_DEPLOY"
    assert any("omni" in r for r in ev["verdicts"][DEFAULT]["reasons"])


def test_evaluate_a_step_error_while_scoring_live_is_not_scorable(monkeypatch, tmp_path):
    rig_path, split = _fake_steps(monkeypatch, tmp_path, live_score_error=pipeline.PipelineError("no JSON"))
    ev = replay().evaluate(None, rig_path, split, _live(tmp_path), tmp_path / "out", decide.Gates(), jobs=1)
    assert not ev["live"]["scorable"] and "no JSON" in ev["live"]["error"]
    assert all(any("no JSON" in r for r in v["reasons"]) for v in ev["verdicts"].values())


@pytest.mark.parametrize("where", ["load", "score"])
def test_evaluate_an_unexpected_error_stops_the_run(monkeypatch, tmp_path, where):
    """a bug (KeyError, ImportError, ...) while loading or scoring the live config is not "live not scorable": it
    stops the run"""
    from calibtool import livecfg
    rig_path, split = _fake_steps(monkeypatch, tmp_path, live_score_error=KeyError("bug") if where == "score" else None)
    if where == "load":
        def broken(*a, **k):
            raise ImportError("bug")
        monkeypatch.setattr(livecfg, "load", broken)
    with pytest.raises((KeyError, ImportError), match="bug"):
        replay().evaluate(None, rig_path, split, _live(tmp_path), tmp_path / "out", decide.Gates(), jobs=1)


# ---------------------------------------------------------------- stored sessions, collection only (fast)


def test_p21_gap_sentences_over_all_takes(data_root, tmp_path):
    """after the last of the 8 takes the gap sentences carry the handoff's numbers (all corners captured)"""
    js = replay().run(sessions.get("p21_1105"), out_dir=tmp_path, fit=False)
    assert js["capture_order"] == sessions.get("p21_1105").take_order
    last = js["placements"][-1]
    assert last["pairs"] == 1011
    assert any(s.startswith(P21_LEFT) for s in last["sentences"])
    assert any(s.startswith(P21_BOTTOM_LEFT) for s in last["sentences"])
    md = (tmp_path / "report.md").read_text()
    assert P21_LEFT in md and P21_BOTTOM_LEFT in md


def test_p21_counterfactual_first_suggestion_is_left_edge_toward_robots_left(data_root, tmp_path):
    js = replay().run(sessions.get("p21_1105"), out_dir=tmp_path, fit=False, exclude_takes=("dleft", "dtiltleft"))
    top = js["suggestions"][0]
    print("COUNTERFACTUAL", json.dumps(js["suggestions"], default=str)[:2000])
    assert top["target"]["kind"] == "edge" and top["target"]["region"] == "left" and top["target"]["cam"] == 0
    assert top["bearing_deg"] > 0
    assert "to YOUR right (the robot's left)" in top["say"]


# ---------------------------------------------------------------- replay: the whole session, fit included


@pytest.mark.replay
@pytest.mark.parametrize("assignment", ["capture", "split16"])
@pytest.mark.parametrize("name", ["p21_1105", "p10_1106"])
def test_replay_new_beats_live_and_deploys(name, assignment, data_root, tmp_path):
    st = sessions.get(name)
    js = replay().run(st, assignment=assignment, out_dir=tmp_path)
    md = (tmp_path / "report.md").read_text()
    print(f"REPLAY {name} {assignment} timings {json.dumps(js['timings'])}")
    print(md)
    assert json.loads((tmp_path / "report.json").read_text())["session"] == name
    assert js["fit"]["converged"], js["fit"]["verdict"]
    assert js["validate_new"]["verdict"] == "PASS", js["validate_new"]["text"]
    worse = [s["name"] for s in js["paired"]["stats"] if s["judged"] and not s["diff"] < 0]
    assert not worse, f"new not better on {worse}"
    v = js["verdicts"][DEFAULT]
    assert v["decision"] == "DEPLOY", v["reasons"]
    # all four edges required: p21 has no judged top entry (measured 9 Nov, both assignments)
    assert js["verdicts"][ALL4]["decision"] == {"p21_1105": "NOT_ENOUGH_DATA", "p10_1106": "DEPLOY"}[name]
    if name == "p21_1105":
        last = js["placements"][-1]["sentences"]
        assert any(s.startswith(P21_LEFT) for s in last) and any(s.startswith(P21_BOTTOM_LEFT) for s in last)


@pytest.mark.replay
@pytest.mark.parametrize("name", ["p21_1105", "p10_1106"])
def test_replay_cut_at_the_stop_rule_still_passes(name, data_root, tmp_path):
    """C3's check of the re-based targets: the session cut where the stop rule was first met (capture-time
    assignment, the tool's) still fits, PASSes step 66, the paired comparison still shows the new fit better on depth
    and rows, and the default verdict is DEPLOY: the rule does not stop before the data the recorded PASS needed"""
    js = replay().run(sessions.get(name), assignment="capture", out_dir=tmp_path, stop_at_rule=True)
    st = js["stop"]
    print(f"STOP {name} {json.dumps(st)} capture_order {js['capture_order']} split {json.dumps(js['split'])}")
    print((tmp_path / "report.md").read_text())
    assert st["index"] is not None and st["stopped"], st
    assert js["fit"]["converged"], js["fit"]["verdict"]
    assert js["validate_new"]["verdict"] == "PASS", js["validate_new"]["text"]
    pr = {s["name"]: s for s in js["paired"]["stats"]}
    assert pr["depth"]["judged"] and pr["depth"]["ci"][1] < 0 and pr["rows"]["judged"] and pr["rows"]["ci"][1] < 0
    worse = [s["name"] for s in js["paired"]["stats"] if s["judged"] and not s["diff"] < 0]
    assert not worse, f"new not better on {worse}"
    assert js["verdicts"][DEFAULT]["decision"] == "DEPLOY", js["verdicts"][DEFAULT]["reasons"]
