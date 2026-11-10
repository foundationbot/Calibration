"""calibtool.replay: a stored session replayed as if live. Holds go through the FIT/TEST assignment and the coverage
tallies in capture order, a snapshot (tallies, deficits, gap sentences, a progress line) is taken after each
placement, the planner suggests the next placements after the last one, and then the split is fitted (45), validated
(66, new and live) and decided (decide.verdict with and without require_all_edges) into report.json + report.md.

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
REPORT_KEYS = {"session", "robot", "assignment", "exclude_takes", "take_order", "capture_order", "placements",
               "final", "suggestions", "split", "fit", "validate_new", "validate_live", "live", "stats", "paired",
               "verdicts", "timings"}


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
    assert set(dec["verdicts"]) == {"require_all_edges=False", "require_all_edges=True"}
    for k, want in (("require_all_edges=False", False), ("require_all_edges=True", True)):
        assert dec["verdicts"][k]["decision"] == "DEPLOY"
        assert dec["verdicts"][k]["gates"]["require_all_edges"] is want
    names = [s["name"] for s in dec["paired"]["stats"]]
    assert names[:3] == ["depth", "spread", "rows"] and any(n.startswith("band ") for n in names)
    md = "\n".join(rp.decision_md(dec))
    assert "DEPLOY" in md and "| depth |" in md and "| rows |" in md
    for r in dec["verdicts"]["require_all_edges=False"]["reasons"]:
        assert r in md


def test_decision_without_live_scores():
    rp = replay()
    dec = rp.decision(TD.synthetic(), None, TD.step66(), decide.Gates())
    assert dec["paired"] is None
    assert {v["decision"] for v in dec["verdicts"].values()} == {"LIVE_NOT_SCORABLE_DEPLOY"}
    assert "LIVE_NOT_SCORABLE_DEPLOY" in "\n".join(rp.decision_md(dec))


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
    loose = js["verdicts"]["require_all_edges=False"]
    assert loose["decision"] == "DEPLOY", loose["reasons"]
    assert js["verdicts"]["require_all_edges=True"]["decision"] in decide.DECISIONS
    if name == "p21_1105":
        last = js["placements"][-1]["sentences"]
        assert any(s.startswith(P21_LEFT) for s in last) and any(s.startswith(P21_BOTTOM_LEFT) for s in last)
