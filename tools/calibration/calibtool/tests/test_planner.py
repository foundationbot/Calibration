"""calibtool.planner: candidate board placements (distance x bearing, plus the tilted-low one), each simulated over the
49-pose neck grid with the placement model, and the greedy choice of the next placements for the current deficits.

Synthetic tests use a nominal forward-looking head; the counterfactual (`replay`) removes phantom21's dleft and
dtiltleft takes and asks the planner what to add."""
import contextlib
import io

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")

from calibtool import coverage as cv, paths, placement as pl, planner as pn  # noqa: E402

W, H = 1280, 800


def _rig():
    import rigcfg
    with contextlib.redirect_stdout(io.StringIO()):
        return rigcfg.load(paths.CONFIG / "rig-phantom21.yaml")


def _head():
    """forward-looking camera 9 cm ahead of the Head origin, 10 deg nose-down; equidistant f 565 px at 1280x800"""
    import rigmath
    c, s = np.cos(np.radians(10)), np.sin(np.radians(10))
    R0 = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
    Rp = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    cam = pl.Camera("pinhole-equi", np.array([[565.0, 0, 640], [0, 565.0, 400], [0, 0, 1]]), np.zeros(4), (W, H))
    return pl.Head(X=rigmath.transform(Rp @ R0, [0.09, 0.0, 0.2]), cams=[cam, cam],
                   C=[np.eye(4), rigmath.transform(np.eye(3), [0.044, 0, 0])])


@pytest.fixture(scope="module")
def planner():
    return pn.Planner(_head(), pl.chain(), _rig().board)


def test_candidates_distance_by_bearing_plus_tilted():
    cs = pn.candidates()
    upright = [c for c in cs if not c.tilted]
    assert sorted({c.distance_m for c in upright}) == [0.5, 0.75, 1.1, 1.6, 2.0]
    assert sorted({c.bearing_deg for c in upright}) == [-40, -20, 0, 20, 40]
    assert len(upright) == 25
    tilted = [c for c in cs if c.tilted]
    assert tilted and all(c.distance_m == 0.75 for c in tilted)


def test_grid_is_49_poses():
    g = pn.grid()
    assert len(g) == 49
    p, y = np.array(g).T
    assert sorted(set(np.round(p, 6))) == [-24, -16, -8, 0, 8, 16, 24]
    assert len(set(np.round(y, 6))) == 7 and min(y) == -34 and max(y) == 34


def test_simulate_straight_ahead_fills_centre(planner):
    sim = planner.simulate(pn.Candidate(1.1, 0, False))
    t = sim.fit.cams[0]
    assert sim.n_detected > 20
    assert t.thirds[1, 1] > 0
    assert sim.fit.n_pairs == sim.n_detected * pn.PAIRS_PER_POSE
    assert sum(sim.entries.values()) > 0 and all(b in ((0.4, 0.6), (0.6, 0.9), (0.9, 1.3)) for _, b in sim.entries)


def test_board_toward_robots_left_fills_image_left(planner):
    lt = planner.simulate(pn.Candidate(0.75, 40, False)).fit.cams[0]
    rt = planner.simulate(pn.Candidate(0.75, -40, False)).fit.cams[0]
    assert lt.edges["left"] > lt.edges["right"]
    assert rt.edges["right"] > rt.edges["left"]


def test_far_candidate_gives_far_pairs(planner):
    assert planner.simulate(pn.Candidate(2.0, 0, False)).fit.far_pairs > 0
    assert planner.simulate(pn.Candidate(0.75, 0, False)).fit.far_pairs == 0


def _edge_deficit(region, cam=0, have=100, ref=1000):
    kind = "box" if "-" in region else "edge"
    return cv.Deficit(kind, cam, region, have, 0.5 * ref, cv.MIRROR[region], ref, 3.0)


def test_suggest_left_edge_bears_left(planner):
    s = planner.suggest([_edge_deficit("left")], k=3)
    assert 1 <= len(s) <= 3
    top = s[0]
    assert top.target.region == "left" and top.candidate.bearing_deg > 0
    assert top.gains[0] > 0
    assert "to YOUR right (the robot's left)" in top.say()


def test_suggest_right_edge_bears_right(planner):
    top = planner.suggest([_edge_deficit("right")], k=1)[0]
    assert top.candidate.bearing_deg < 0
    assert "to YOUR left (the robot's right)" in top.say()


def test_one_suggestion_per_top_deficit(planner):
    """C3: pick i serves the most important deficit no earlier pick served (deficits in Coverage.deficits order), so a
    second deficit is not drowned by the first; a candidate is used once; once every short deficit has been served, a
    new round starts"""
    left, right = _edge_deficit("left", ref=40000), _edge_deficit("right", ref=40000)
    s = planner.suggest([left, right], k=3)
    assert len(s) == 3
    assert [x.target.region for x in s[:2]] == ["left", "right"]
    assert s[0].candidate.bearing_deg > 0 and s[1].candidate.bearing_deg < 0
    assert len({x.candidate for x in s}) == len(s)
    s = planner.suggest([right, left], k=2)
    assert [x.target.region for x in s] == ["right", "left"]
    # one deficit met by the first pick is not asked for again
    one = planner.suggest([_edge_deficit("left")], k=3)
    assert len(one) == 1


def test_a_suggestion_is_the_best_fill_of_its_deficit(planner):
    """the candidate picked for a deficit adds the most to it (predicted gain, not capped: of two boards that both
    cover what is short, the one that fills more); the rest of its value breaks ties"""
    d = _edge_deficit("left", have=990, ref=2000)                     # 10 short: any left-bearing board covers it
    g = {c: planner.simulate(c).fit.cams[0].edges["left"] for c in planner.candidates}
    assert sum(v >= 10 for v in g.values()) > 1
    assert g[planner.suggest([d], k=1)[0].candidate] == max(g.values())
    d = cv.Deficit("far_band", None, (1.7, 2.3), 0, 1000)            # more than any candidate gives: no cap
    s = planner.suggest([_edge_deficit("left"), d], k=2)
    assert [x.target.kind for x in s] == ["edge", "far_band"]
    g = {c: pn.gain(planner.simulate(c).fit, planner.simulate(c).entries, planner.simulate(c).far_bands, d)
         for c in planner.candidates if c != s[0].candidate}
    assert g[s[1].candidate] == max(g.values()) > 0


def test_a_deficit_no_candidate_fills_is_skipped(planner):
    nowhere = cv.Deficit("entry", None, ("centre", (5.0, 6.0)), 0, 10)
    s = planner.suggest([nowhere, _edge_deficit("left")], k=2)
    assert [x.target.region for x in s] == ["left"]


def test_suggest_nothing_for_no_deficit(planner):
    assert planner.suggest([], k=3) == []


def test_pose_subset_left_edge_turns_head_right(planner):
    c = pn.Candidate(0.75, 40, False)
    poses = planner.pose_subset(c, _edge_deficit("left"))
    assert 0 < len(poses) < 49
    assert np.mean([y for _, y in poses]) < 0          # -yaw = head to the robot's right: board to the image left
    assert set(poses) <= set(pn.grid())


def test_side_gain_counts_a_corner_cell_for_both_its_edges():
    """the same edge rule as coverage's side deficit and decide's verdict: holds in a corner cell fill both its sides"""
    ent = {("bottom-left", (0.6, 0.9)): 10, ("left", (0.9, 1.3)): 4}
    side = lambda r: cv.Deficit("side", None, r, 0, 10)            # noqa: E731
    assert pn.gain(None, ent, {}, side("left")) == 10 * pn.TEST_FRAC
    assert pn.gain(None, ent, {}, side("bottom")) == 10 * pn.TEST_FRAC
    assert pn.gain(None, ent, {}, side("top")) == 0


def test_describe_words():
    assert "to YOUR right (the robot's left)" in pn.Candidate(1.1, 20, False).describe()
    assert "to YOUR left (the robot's right)" in pn.Candidate(1.1, -40, False).describe()
    assert "straight ahead" in pn.Candidate(0.5, 0, False).describe()
    t = pn.Candidate(0.75, 20, True).describe()
    assert "tilted" in t and "to YOUR right (the robot's left)" in t
    for c in pn.candidates():
        d = c.describe()
        assert "robot's right" not in d or "to YOUR left (the robot's right)" in d


# ---------------------------------------------------------------- the counterfactual on phantom21


_CF = {}


def _counterfactual():
    """phantom21 5 Nov without dleft and dtiltleft, as the replay counts it: replay.Collection with step 16's split
    (every 2nd hold per take), regions over all corners, the stored sessions' targets (per_hold 3); its deficits after
    the last placement and the planner's 3 suggestions at 3 pairs a pose (cached for the module)"""
    if not _CF:
        from calibtool import assign, pipeline, replay, sessions
        st = sessions.get("p21_1105")
        rig = pipeline.load_rig(st.rig_path)
        s = replay.load_session(st, exclude_takes=("dleft", "dtiltleft"))
        col = replay.Collection.for_session(s, rig, "split16", targets=cv.Targets.for_per_hold(3))
        d = replay.feed(col, assign.capture_holds(s))[-1].deficits
        head = pl.head_from_okvis(st.live_config, rig.resolution)
        _CF.update(deficits=d, suggestions=pn.Planner(head, pl.chain(), rig.board, pairs_per_pose=3).suggest(d, k=3))
        for x in d:
            print("DEFICIT", "blocks" if x.blocks else "asked", cv.sentence(x))
        for x in _CF["suggestions"]:
            print("SUGGEST", x.say(), "| fills:",
                  [(f.kind, f.cam, f.region, round(g)) for f, g in zip(x.fills, x.gains)])
    return _CF


@pytest.mark.replay
def test_p21_counterfactual_top_suggestion_is_left_edge_toward_robots_left(data_root):
    cf = _counterfactual()
    d = cf["deficits"]
    assert (d[0].kind, d[0].cam, d[0].region) in {("edge", 0, "left"), ("box", 0, "bottom-left")}
    top = cf["suggestions"][0]
    assert top.target.kind == "edge" and top.target.region == "left" and top.target.cam == 0
    assert top.candidate.bearing_deg > 0
    assert "to YOUR right (the robot's left)" in top.say()


@pytest.mark.replay
def test_p21_counterfactual_asks_for_a_tilted_placement_for_the_bottom(data_root):
    """one suggestion per top deficit (C3): the bottom edge strip gets its own pick, and the tilted board is its best
    fill (M1 B2's greedy summed value preferred upright boards that also fill edges and other entries, so no tilted
    placement made the top 3; the real dtiltleft take added 726 corners on the bottom edge)"""
    sug = _counterfactual()["suggestions"]
    low = [x for x in sug if x.candidate.tilted and any(
        f.region in ("bottom", "bottom-left") or (f.kind == "entry" and f.region[0] in ("bottom", "bottom-left"))
        for f in x.fills[:1])]
    assert low, "no tilted placement for the bottom / bottom-left deficit"
    assert low[0].target.kind == "strip" and low[0].target.region == "bottom"
    assert sug[0].target.kind == "edge" and sug[0].target.region == "left"
