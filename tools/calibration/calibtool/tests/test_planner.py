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


def test_suggest_is_greedy_without_repeats(planner):
    s = planner.suggest([_edge_deficit("left", ref=40000), _edge_deficit("right", ref=40000)], k=3)
    assert len(s) == 3
    assert len({x.candidate for x in s}) == len(s)
    assert all(a.score >= b.score for a, b in zip(s, s[1:]))
    # one big deficit met by a first pick is not asked for again: the picks change with the deficits
    one = planner.suggest([_edge_deficit("left")], k=3)
    assert len(one) == 1


def test_suggest_nothing_for_no_deficit(planner):
    assert planner.suggest([], k=3) == []


def test_pose_subset_left_edge_turns_head_right(planner):
    c = pn.Candidate(0.75, 40, False)
    poses = planner.pose_subset(c, _edge_deficit("left"))
    assert 0 < len(poses) < 49
    assert np.mean([y for _, y in poses]) < 0          # -yaw = head to the robot's right: board to the image left
    assert set(poses) <= set(pn.grid())


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
    """phantom21 5 Nov without dleft and dtiltleft: FIT tally of the FIT holds and TEST tally of the TEST holds of step
    16's rule (every 2nd hold per take), deficits, and the planner's 3 suggestions (cached for the module)"""
    if not _CF:
        from calibtool import pipeline, sessions
        from calibtool.session import Session
        st = sessions.get("p21_1105")
        rig = pipeline.load_rig(st.rig_path)
        keep = [t for t in st.take_order if t not in ("dleft", "dtiltleft")]
        files = dict(zip(st.take_order, st.corner_files))
        s = Session.from_files([(t, files[t]) for t in keep], name="p21-no-left")
        z = cv.session_z(s, rig)
        a = cv.split16_assignment(s, z)
        fit = np.array([a[int(h)] == "FIT" for h in s.pairs["hold"]])
        head = pl.head_from_okvis(st.live_config, rig.resolution)
        cov = cv.Coverage(fit=cv.fit_tally(s, mask=fit, rig=rig, z=z),
                          test=cv.test_tally(s, a, tuple(head.cams[0].K[:2, 2]), z))
        d = cov.deficits(cv.Targets())
        _CF.update(deficits=d, suggestions=pn.Planner(head, pl.chain(), rig.board).suggest(d, k=3))
        for x in d:
            print("DEFICIT", cv.sentence(x))
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
@pytest.mark.xfail(strict=True, reason=(
    "M1 B2 measured 9 Nov: no tilted placement in the top 3. Without dleft/dtiltleft the comparative rule finds no "
    "bottom-edge deficit (cam0, all pairs: bottom 2,674 vs top 2,574 corners); for the bottom-left box the model "
    "predicts the tilted board adds 45-54 corners vs 120-168 for an upright board to the robot's left (the real "
    "dtiltleft take added 3 there, 726 on the bottom edge). The tilted board is the best single fill of the bottom "
    "0.9-1.3 m TEST entry (20-23 holds vs 10-12) but greedy summed value prefers upright boards that also fill edges "
    "and other entries. Decision for the lead (C3): a bottom-edge target, or one suggestion per top deficit."))
def test_p21_counterfactual_asks_for_a_tilted_placement_for_the_bottom(data_root):
    sug = _counterfactual()["suggestions"]
    low = [x for x in sug if x.candidate.tilted and any(
        f.region in ("bottom", "bottom-left") or (f.kind == "entry" and f.region[0] in ("bottom", "bottom-left"))
        for f in x.fills[:1])]
    assert low, "no tilted placement for the bottom / bottom-left deficit"
