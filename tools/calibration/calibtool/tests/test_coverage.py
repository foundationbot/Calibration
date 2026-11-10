"""calibtool.coverage: FIT tallies (image thirds, rim, far pairs, edge strips, corner boxes; per camera), TEST tallies
(per step-66 entry and far band), deficits and the comparative, directional gap sentences.

The reference numbers are the stored kit scripts' output on the stored corner files (Moonstone ~/calibtool/data, run
9 Nov 2026 with only their paths changed): p21/kit/cov21.py and edge21.py on p21/kit/out, p10/kit/cov1280.py on
p10/kit/out1280. Those tests are skipped without CALIBTOOL_DATA."""
import re

import pytest

np = pytest.importorskip("numpy")

from calibtool import coverage as cv  # noqa: E402

W, H = 1280, 800


# ---------------------------------------------------------------- corner tallies (synthetic)


def test_corner_tally_thirds_rim_edges_boxes():
    u = np.array([10.0, 70, 100, 1250, 640, 640, 150, 1200])
    v = np.array([10.0, 400, 790, 790, 10, 400, 700, 100])
    t = cv.corner_tally(u, v, (W, H))
    assert t.n == 8
    # thirds: row = min(int(v / H * 3), 2), col = min(int(u / W * 3), 2)
    g = np.zeros((3, 3), int)
    for a, b in zip(u, v):
        g[min(int(b / H * 3), 2), min(int(a / W * 3), 2)] += 1
    assert (t.thirds == g).all()
    r = np.hypot(u - W / 2, v - H / 2) / np.hypot(W / 2, H / 2)
    assert t.rim == int((r > 0.8).sum())
    assert t.edges == {"left": 2, "right": 1, "top": 2, "bottom": 2}
    assert t.boxes == {"top-left": 1, "top-right": 1, "bottom-left": 2, "bottom-right": 1}


def test_corner_tally_empty():
    t = cv.corner_tally(np.zeros(0), np.zeros(0), (W, H))
    assert t.n == 0 and t.rim == 0 and t.thirds.sum() == 0
    assert set(t.edges.values()) == {0} and set(t.boxes.values()) == {0}


# ---------------------------------------------------------------- TEST tallies (synthetic)


def _mini_session():
    """3 holds of 1 pair each: hold 0 board at the image left, hold 1 centre, hold 2 centre"""
    from calibtool.session import Session
    us = [100.0, 640.0, 640.0]
    z = {"img_hold": np.array([0, 1, 2]), "img_t0": np.array([10, 20, 30]), "img_t1": np.array([11, 21, 31]),
         "det_img": np.repeat([0, 1, 2], 4), "det_cam": np.zeros(12, int), "det_tag": np.zeros(12, int),
         "det_k": np.tile([0, 1, 2, 3], 3), "det_u": np.repeat(us, 4), "det_v": np.full(12, 400.0)}
    return Session.from_files([("t", z)], name="mini")


def test_test_tally_counts_holds_per_entry_and_far_band():
    s = _mini_session()
    z = np.array([0.5, 0.7, 1.5])
    tt = cv.test_tally(s, {0: "TEST", 1: "TEST", 2: "TEST"}, pp=(640.0, 400.0), z=z)
    assert tt.entries == {("left", (0.4, 0.6)): 1, ("centre", (0.6, 0.9)): 1}
    assert tt.far_bands == {(1.3, 1.7): 1, (1.7, 2.3): 0}
    tf = cv.test_tally(s, {0: "FIT", 1: "TEST", 2: "FIT"}, pp=(640.0, 400.0), z=z)
    assert tf.entries == {("centre", (0.6, 0.9)): 1}
    assert tf.far_bands == {(1.3, 1.7): 0, (1.7, 2.3): 0}


def test_fit_tally_mask_selects_pairs():
    s = _mini_session()
    z = np.array([0.5, 1.8, 2.5])
    f = cv.fit_tally(s, z=z, size=(W, H))
    assert f.cams[0].n == 12 and f.cams[1].n == 0
    assert f.far_pairs == 2
    f1 = cv.fit_tally(s, mask=np.array([True, False, False]), z=z, size=(W, H))
    assert f1.cams[0].n == 4 and f1.far_pairs == 0
    assert f1.cams[0].edges["left"] == 0 and f1.cams[0].thirds[1, 0] == 4


# ---------------------------------------------------------------- deficits and sentences (synthetic)


def _tally(edges, boxes, n=10000, rim=2000, thirds=6000):
    return cv.CornerTally(n=n, thirds=np.full((3, 3), thirds), rim=rim, edges=dict(edges), boxes=dict(boxes))


P21_E0 = {"left": 176, "right": 1099, "top": 2983, "bottom": 3648}
P21_B0 = {"top-left": 89, "top-right": 155, "bottom-left": 50, "bottom-right": 588}
FINE_E = {"left": 1000, "right": 1000, "top": 1000, "bottom": 1000}
FINE_B = {"top-left": 500, "top-right": 500, "bottom-left": 500, "bottom-right": 500}


def _cov(e0=P21_E0, b0=P21_B0, e1=FINE_E, b1=FINE_B, far=100, **kw):
    fit = cv.FitTally(cams={0: _tally(e0, b0, **kw), 1: _tally(e1, b1, **kw)}, far_pairs=far, n_pairs=1000)
    return cv.Coverage(fit=fit)


def test_deficits_p21_numbers_left_edge_and_bottom_left_box_first():
    d = _cov().deficits(cv.Targets())
    assert [(x.kind, x.cam, x.region) for x in d[:2]] == [("edge", 0, "left"), ("box", 0, "bottom-left")]
    left = next(x for x in d if x.region == "left")
    assert (left.have, left.ref_region, left.ref_have) == (176, "right", 1099)
    # top-left box 89 vs 155: within the default ratio, not a deficit; top/bottom edges fine
    assert not any(x.region in ("top-left", "top", "bottom", "right") for x in d)


def test_weights_edges_then_boxes_then_the_rest():
    w = cv.Targets().weights
    assert w["edge"] > w["box"] > 1.0
    d = _cov().deficits(cv.Targets())
    assert {x.kind: x.weight for x in d} == {"edge": w["edge"], "box": w["box"]}


def test_deficits_edges_then_boxes_then_others():
    e = dict(FINE_E, right=50)
    b = dict(FINE_B, **{"top-left": 10})
    d = _cov(e0=e, b0=b, thirds=1000).deficits(cv.Targets())
    assert [x.kind for x in d[:2]] == ["edge", "box"]


def test_deficits_edges_before_other_kinds():
    d = _cov(thirds=1000, rim=100, far=10).deficits(cv.Targets())
    kinds = [x.kind for x in d]
    assert {"third", "rim", "far"} <= set(kinds)
    first_other = min(i for i, k in enumerate(kinds) if k not in ("edge", "box"))
    assert all(k in ("edge", "box") for k in kinds[:first_other])
    assert all(k not in ("edge", "box") for k in kinds[first_other:])


def test_no_deficit_when_balanced():
    assert _cov(e0=FINE_E, b0=FINE_B).deficits(cv.Targets()) == []


def test_sentence_p21_left_edge_verbatim():
    d = _cov().deficits(cv.Targets())
    left = next(x for x in d if x.kind == "edge" and x.region == "left")
    s = cv.sentence(left)
    assert s.startswith("Left camera, left edge: 176 corners vs 1,099 on the right edge (6x fewer).")
    assert "to YOUR right (the robot's left)" in s


def test_sentence_bottom_left_box():
    d = _cov().deficits(cv.Targets())
    bl = next(x for x in d if x.kind == "box")
    s = cv.sentence(bl)
    assert s.startswith("Left camera, bottom-left corner: 50 corners vs 588 in the bottom-right corner (12x fewer).")
    assert "to YOUR right (the robot's left)" in s
    assert "lower" in s and "tilted" in s


# every edge and box: the direction named from the operator's side first
EDGE_WORDS = {"left": "to YOUR right (the robot's left)", "right": "to YOUR left (the robot's right)",
              "top": "higher", "bottom": "lower"}


@pytest.mark.parametrize("edge", ["left", "right", "top", "bottom"])
@pytest.mark.parametrize("cam", [0, 1])
def test_edge_sentence_direction(edge, cam):
    e = dict(FINE_E)
    e[edge] = 100
    b = FINE_B
    cov = _cov(e0=e, b0=b) if cam == 0 else _cov(e1=e, b1=b, e0=FINE_E, b0=FINE_B)
    d = cov.deficits(cv.Targets())
    assert [(x.kind, x.cam, x.region) for x in d] == [("edge", cam, edge)]
    s = cv.sentence(d[0])
    cam_word = "Left camera" if cam == 0 else "Right camera"
    opp = cv.MIRROR[edge]
    assert s.startswith(f"{cam_word}, {edge} edge: 100 corners vs 1,000 on the {opp} edge (10x fewer).")
    assert EDGE_WORDS[edge] in s
    _check_wording(s)


@pytest.mark.parametrize("box", ["top-left", "top-right", "bottom-left", "bottom-right"])
def test_box_sentence_direction(box):
    b = dict(FINE_B)
    b[box] = 50
    d = _cov(e0=FINE_E, b0=b).deficits(cv.Targets())
    assert [(x.kind, x.region) for x in d] == [("box", box)]
    s = cv.sentence(d[0])
    vert, horiz = box.split("-")
    assert ("higher" if vert == "top" else "lower") in s
    assert EDGE_WORDS[horiz] in s
    _check_wording(s)


def _check_wording(s):
    """the operator's side is named, the robot's only in brackets after it; never "robot's right" alone"""
    for m in re.finditer(r"robot's (left|right)", s):
        side = m.group(1)
        mine = "right" if side == "left" else "left"
        assert f"to YOUR {mine} (the robot's {side})" in s, s


def test_sentences_wording_for_every_kind():
    cov = _cov(thirds=1000, rim=100, far=10)
    cov.test = cv.TestTally(entries={("left", (0.9, 1.3)): 3, ("right", (0.6, 0.9)): 12},
                            far_bands={(1.3, 1.7): 2, (1.7, 2.3): 0})
    d = cov.deficits(cv.Targets())
    ss = cv.sentences(d)
    assert len(ss) == len(d) and all(isinstance(s, str) and s for s in ss)
    for s in ss:
        _check_wording(s)
        assert "robot's right" not in s or "YOUR left" in s
    assert any(s.startswith("Left camera, top-left third: 1,000 corners of 5,000 wanted") for s in ss)
    assert any("Far boards" in s for s in ss)
    # an unfinished TEST entry, with its side named from the operator
    ent = [s for s, x in zip(ss, d) if x.kind == "entry"]
    assert ent and "left cell at 0.9-1.3 m: 3 of 10 held-out holds" in ent[0]
    assert "to YOUR right (the robot's left)" in ent[0]


def test_ratio_words():
    assert cv.ratio_words(176, 1099) == "6x fewer"
    assert cv.ratio_words(50, 588) == "12x fewer"
    assert cv.ratio_words(100, 160) == "1.6x fewer"
    assert cv.ratio_words(0, 588) == "none there"


# ---------------------------------------------------------------- stored sessions: the kit's numbers


def _load(name):
    from calibtool import pipeline, sessions
    from calibtool.session import Session
    st = sessions.get(name)
    return Session.load(st), pipeline.load_rig(st.rig_path)


P21_KIT = {  # p21/kit/cov21.py ALL + edge21.py all 8 takes
    "thirds": [[3665, 12675, 5873], [8019, 24645, 15515], [4458, 14638, 9296]], "rim": 912, "far": 79,
    "edges": {0: {"left": 176, "right": 1099, "top": 2983, "bottom": 3648},
              1: {"left": 272, "right": 1003, "top": 3011, "bottom": 4428}},
    "boxes": {0: {"top-left": 89, "top-right": 155, "bottom-left": 50, "bottom-right": 588},
              1: {"top-left": 145, "top-right": 192, "bottom-left": 125, "bottom-right": 689}},
    "n": {0: 98784, 1: 101768}}
P10_KIT = {  # p10/kit/cov1280.py ALL
    "thirds": [[9108, 17347, 6008], [13567, 26556, 12032], [7492, 11528, 5814]], "rim": 486, "far": 127}


def test_p21_fit_tally_matches_kit(data_root):
    s, rig = _load("p21_1105")
    f = cv.fit_tally(s, rig=rig)
    assert f.n_pairs == 1011
    assert f.cams[0].thirds.tolist() == P21_KIT["thirds"]
    assert f.cams[0].rim == P21_KIT["rim"]
    assert f.far_pairs == P21_KIT["far"]
    for c in (0, 1):
        assert f.cams[c].n == P21_KIT["n"][c]
        assert f.cams[c].edges == P21_KIT["edges"][c]
        assert f.cams[c].boxes == P21_KIT["boxes"][c]


def test_p10_fit_tally_matches_kit(data_root):
    s, rig = _load("p10_1106")
    f = cv.fit_tally(s, rig=rig)
    assert f.n_pairs == 1159
    assert f.cams[0].thirds.tolist() == P10_KIT["thirds"]
    assert f.cams[0].rim == P10_KIT["rim"]
    assert f.far_pairs == P10_KIT["far"]


def test_p21_gap_sentences_name_left_edge_and_bottom_left(data_root):
    s, rig = _load("p21_1105")
    cov = cv.Coverage(fit=cv.fit_tally(s, rig=rig))
    ss = cv.sentences(cov.deficits(cv.Targets()))
    assert "Left camera, left edge: 176 corners vs 1,099 on the right edge (6x fewer)." in " ".join(ss)
    assert "Left camera, bottom-left corner: 50 corners vs 588 in the bottom-right corner (12x fewer)." \
        in " ".join(ss)


# ---------------------------------------------------------------- step 16's rule as a hold assignment


def test_split16_assignment_every_second_near_hold_and_far_holds():
    s = _mini_session()
    a = cv.split16_assignment(s, z=np.array([0.5, 0.7, 3.5]), max_fit_z=3.0, holdout_every=2)
    assert a == {0: "FIT", 1: "TEST", 2: "TEST"}


@pytest.mark.parametrize("name,n_test", [("p21_1105", 499), ("p10_1106", 572)])
def test_split16_assignment_reproduces_step16_test_pairs(data_root, name, n_test):
    s, rig = _load(name)
    a = cv.split16_assignment(s, z=cv.session_z(s, rig))
    test = np.array([a[int(h)] == "TEST" for h in s.pairs["hold"]])
    assert int(test.sum()) == n_test
