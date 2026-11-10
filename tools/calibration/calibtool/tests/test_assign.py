"""calibtool.assign: FIT/TEST assigned at capture time, one hold at a time, stratified per step-66 entry (lenscells cell
x distance bin), with each entry's TEST holds drawn from more than one placement.

Synthetic holds are made by projecting a 2 x 2-tag patch of the rig's board with the plain equidistant lens step 16's
rough distance assumes (nominal focal, nominal principal point), so a hold's distance and cell are known by
construction. Stored-session tests need CALIBTOOL_DATA; `replay` tests (-m replay) fit and validate the split."""
import json
import math
from concurrent.futures import ThreadPoolExecutor

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("cv2")
pytest.importorskip("yaml")

from calibtool import pipeline, sessions  # noqa: E402
from calibtool.session import Session  # noqa: E402

RIG_PATH = sessions.get("p21_1105").rig_path
PP, FOCAL = (640.0, 400.0), 565.0
PATCH_TAGS = (8, 9, 15, 16)              # 2 x 2 tags in the middle of the 7 x 5 board: 16 corners per camera
OFFSET = {"centre": (0, 0), "left": (-320, 0), "right": (320, 0), "top": (0, -220), "bottom": (0, 220),
          "top-left": (-380, -230), "top-right": (380, -230), "bottom-left": (-380, 230), "bottom-right": (380, 230)}
SPLIT_KEYS = {"img_hold", "img_t0", "img_t1", "img_z", "det_img", "det_cam", "det_tag", "det_k", "det_u", "det_v"}


@pytest.fixture(scope="module")
def rig():
    return pipeline.load_rig(RIG_PATH)


def _project(P):
    """camera-frame points -> pixels, plain equidistant lens (r = f theta) at FOCAL and PP"""
    rr = np.hypot(P[:, 0], P[:, 1])
    th = np.arctan2(rr, P[:, 2])
    s = np.where(rr > 1e-12, FOCAL * th / np.maximum(rr, 1e-12), FOCAL / P[:, 2])
    return np.c_[PP[0] + s * P[:, 0], PP[1] + s * P[:, 1]]


def _patch_pixels(board, cell, z, tags=PATCH_TAGS, jitter=0.0, rng=None):
    """(tag, k, u, v) of the patch facing the camera at depth z with its corners' mean pixel in `cell`"""
    obj = np.concatenate([board.obj_points(t) for t in tags])
    du, dv = OFFSET[cell]
    d = math.hypot(du, dv)
    lat = np.zeros(2) if d == 0 else z * math.tan(d / FOCAL) * np.array([du, dv]) / d
    P = obj - obj.mean(0) + np.r_[lat, z]
    uv = _project(P)
    if jitter and rng is not None:
        uv = uv + rng.normal(0, jitter, uv.shape)
    tag = np.repeat(np.asarray(tags), 4)
    k = np.tile(np.arange(4), len(tags))
    return tag, k, uv


def synth_take(board, holds, t_start, seed=0):
    """a step-15b corner dict. holds: [(cell, z or [z per pair], n_pairs)] in time order, pose = position in the
    list; z None: the pair sees one tag only (too few corners for a distance: nan)"""
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ("img_hold", "img_t0", "img_t1", "img_yaw", "img_pitch", "img_ay", "img_ap", "det_img",
                           "det_cam", "det_tag", "det_k", "det_u", "det_v")}
    t = t_start
    for pose, (cell, z, n) in enumerate(holds):
        zs = list(z) if isinstance(z, (list, tuple)) else [z] * n
        for zp in zs:
            j = len(out["img_t0"])
            out["img_hold"].append(pose)
            out["img_t0"].append(t)
            out["img_t1"].append(t + 3)
            out["img_yaw"].append(0.01 * pose)
            out["img_pitch"].append(-0.01 * pose)
            out["img_ay"].append(0)
            out["img_ap"].append(0)
            t += 50_000_000
            if zp is None:
                tag, k, uv = _patch_pixels(board, cell, 0.75, tags=PATCH_TAGS[:1])
            else:
                tag, k, uv = _patch_pixels(board, cell, zp, jitter=0.05, rng=rng)
            zz = 0.75 if zp is None else zp
            for cam, shift in ((0, 0.0), (1, -FOCAL * 0.044 / zz)):
                for (tg, kk), (u, v) in zip(zip(tag, k), uv):
                    out["det_img"].append(j)
                    out["det_cam"].append(cam)
                    out["det_tag"].append(int(tg))
                    out["det_k"].append(int(kk))
                    out["det_u"].append(float(u + shift))
                    out["det_v"].append(float(v))
        t += 1_000_000_000
    ints = {"img_hold", "img_t0", "img_t1", "img_ay", "img_ap", "det_img", "det_cam", "det_tag", "det_k"}
    return {k: np.asarray(v, np.int64 if k in ints else float) for k, v in out.items()}


def synth_session(board, takes):
    """takes: [(name, holds)], recorded one after the other in list order"""
    files = [(name, synth_take(board, holds, t_start=(i + 1) * 10**12, seed=i)) for i, (name, holds) in
             enumerate(takes)]
    return Session.from_files(files, name="synth")


def assigner(rig, **kw):
    from calibtool import assign
    return assign.Assigner(rig, PP, FOCAL, **kw)


def run(rig, session, **kw):
    """an Assigner fed every hold of session in capture order -> (assigner, [(hold, placement, label)])"""
    from calibtool import assign
    a = assigner(rig, **kw)
    seen = [(h.hold, h.placement, a.observe(h)) for h in assign.capture_holds(session)]
    return a, seen


# ---------------------------------------------------------------- 1. far pairs are TEST (as step 16)


def test_far_and_unmeasurable_pairs_are_test(rig, tmp_path):
    s = synth_session(rig.board, [("A", [("centre", 0.5, 2), ("centre", 3.5, 2), ("left", [2.0, 3.4], 2),
                                         ("right", 0.75, 2), ("top", [0.75, None], 2), ("bottom", 0.5, 2)])])
    a, seen = run(rig, s)
    assert seen[1][2] == "TEST"                       # every pair beyond 3 m: the hold is TEST
    sp = a.write_split(s, tmp_path)
    fit, test = np.load(sp.fit_npz), np.load(sp.test_npz)
    far_or_nan = lambda zz: np.nan_to_num(zz, nan=np.inf) > 3.0          # noqa: E731
    assert not far_or_nan(fit["img_z"]).any()
    tz = dict(zip(test["img_t0"].tolist(), test["img_z"].tolist()))
    p = s.pairs
    for i in range(s.n_pairs):
        zz = tz.get(int(p["t0"][i]))
        if p["hold"][i] == 1 or (p["hold"][i] == 2 and i % 2 == 1) or (p["hold"][i] == 4 and i % 2 == 1):
            assert zz is not None and far_or_nan(np.array([zz]))[0], (i, zz)
    assert math.isclose(tz[int(p["t0"][s.pairs["hold"] == 1][0])], 3.5, rel_tol=1e-3)


# ---------------------------------------------------------------- 5. the written split


def test_write_split_has_step16s_layout_and_loads_in_stereofit(rig, tmp_path):
    import stereofit
    s = synth_session(rig.board, [("A", [(c, 0.75, 2) for c in ("centre", "left", "right", "top")] +
                                   [("centre", 3.5, 2)]),
                                  ("B", [(c, 1.1, 3) for c in ("bottom", "left", "centre")])])
    a, _ = run(rig, s)
    sp = a.write_split(s, tmp_path / "mine")
    ref = pipeline.split16(s, tmp_path / "ref", rig, max_fit_z=3.0, holdout_every=2)
    assert (sp.fit_npz, sp.test_npz) == (tmp_path / "mine" / "fit.npz", tmp_path / "mine" / "test.npz")
    mine = {k: np.load(f) for k, f in (("fit", sp.fit_npz), ("test", sp.test_npz))}
    theirs = {k: np.load(f) for k, f in (("fit", ref.fit_npz), ("test", ref.test_npz))}
    for k in ("fit", "test"):
        assert set(mine[k].files) == SPLIT_KEYS
        for key in SPLIT_KEYS:
            assert mine[k][key].dtype == theirs[k][key].dtype, (k, key)
    t_fit, t_test = set(mine["fit"]["img_t0"].tolist()), set(mine["test"]["img_t0"].tolist())
    assert not t_fit & t_test and t_fit | t_test == set(s.pairs["t0"].tolist())
    assert (sp.fit_pairs, sp.test_pairs) == (len(t_fit), len(t_test))
    # hold ids and rough distances are step 16's, pair for pair
    ref_all = {int(t): (int(h), float(z)) for f in theirs.values() for t, h, z in zip(f["img_t0"], f["img_hold"],
                                                                                      f["img_z"])}
    for f in mine.values():
        for t, h, z in zip(f["img_t0"], f["img_hold"], f["img_z"]):
            assert ref_all[int(t)][0] == int(h)
            assert math.isclose(ref_all[int(t)][1], float(z), abs_tol=1e-9)
    for k, f in (("fit", sp.fit_npz), ("test", sp.test_npz)):
        d = stereofit.Data(str(f), rig.board)
        assert d.n == len(mine[k]["img_t0"])


def test_write_split_refuses_a_session_with_unobserved_holds(rig, tmp_path):
    from calibtool import assign
    s = synth_session(rig.board, [("A", [("centre", 0.75, 2), ("left", 0.75, 2)])])
    a = assigner(rig)
    a.observe(next(assign.capture_holds(s)))
    with pytest.raises(ValueError, match="not observed"):
        a.write_split(s, tmp_path)


# ---------------------------------------------------------------- 2-4. the stratified rule

# three placements feeding four entries, cells interleaved within each take
MIXED = [("P1", [("centre", 0.5, 2), ("left", 0.75, 2)] * 6 + [("centre", 0.5, 2)]),
         ("P2", [("centre", 0.5, 2), ("left", 0.75, 2), ("right", 1.1, 2)] * 5 + [("left", 0.75, 2)]),
         ("P3", [("right", 1.1, 2), ("centre", 0.5, 2), ("top", 1.1, 1)] * 3)]


def _per_entry(a, labels=None):
    """{entry: [(placement, label)]} in observation order (labels: {hold: label} overriding the final ones)"""
    out = {}
    for h in a.holds.values():
        if h.entry is not None:
            out.setdefault(h.entry, []).append((h.placement, (labels or {}).get(h.hold, h.label)))
    return out


def test_entries_are_known_by_construction(rig):
    a, _ = run(rig, synth_session(rig.board, MIXED))
    assert set(_per_entry(a)) == {("centre", (0.4, 0.6)), ("left", (0.6, 0.9)), ("right", (0.9, 1.3)),
                                  ("top", (0.9, 1.3))}


@pytest.mark.parametrize("every", [2, 3])
def test_test_fraction_per_entry_within_one_hold(rig, every):
    s = synth_session(rig.board, MIXED)
    a, seen = run(rig, s, holdout_every=every)
    a.finish()
    for e, rows in _per_entry(a).items():
        k, t = len(rows), sum(lab == "TEST" for _, lab in rows)
        assert abs(t - k / every) <= 1, (e, k, t)
    # with no single-placement hold-back in play (target_test large) the bound holds at every hold, as captured
    a, seen = run(rig, s, holdout_every=every, target_test=1000)
    cap = {h: lab for h, _, lab in seen}
    for e, rows in _per_entry(a, cap).items():
        for i in range(1, len(rows) + 1):
            t = sum(lab == "TEST" for _, lab in rows[:i])
            assert abs(t - i / every) <= 1, (e, i, t)


def test_entry_counts_match_the_labels(rig):
    a, _ = run(rig, synth_session(rig.board, MIXED))
    a.finish()
    ec = a.entry_counts()
    per = _per_entry(a)
    assert set(ec) == set(per)
    for e, rows in per.items():
        c = ec[e]
        assert c.holds == len(rows)
        assert c.test == sum(lab == "TEST" for _, lab in rows) and c.fit == c.holds - c.test
        assert c.placements == len({p for p, _ in rows})
        assert c.test_placements == len({p for p, lab in rows if lab == "TEST"})


def test_every_entry_fed_by_two_placements_tests_both(rig):
    a, _ = run(rig, synth_session(rig.board, MIXED))
    a.finish()
    for e, rows in _per_entry(a).items():
        if len({p for p, _ in rows}) >= 2:
            assert len({p for p, lab in rows if lab == "TEST"}) >= 2, (e, rows)


@pytest.mark.parametrize("m", range(1, 14))
def test_a_second_placements_single_hold_is_tested(rig, m):
    """m holds of one entry from placement P1, then one from P2: TEST holds come from both"""
    a, seen = run(rig, synth_session(rig.board, [("P1", [("centre", 0.5, 2)] * m), ("P2", [("centre", 0.5, 2)])]))
    assert seen[-1][2] == "TEST"
    a.finish()
    rows = _per_entry(a)[("centre", (0.4, 0.6))]
    assert {p for p, lab in rows if lab == "TEST"} == {"P1", "P2"}
    assert abs(sum(lab == "TEST" for _, lab in rows) - (m + 1) / 2) <= 1


@pytest.mark.parametrize("target,cap", [(10, 5), (6, 3), (7, 4)])
def test_one_placement_holds_back_at_most_half_the_target(rig, target, cap):
    """while one placement feeds an entry, at most ceil(target_test / 2) of its holds are TEST as captured; a second
    placement then takes TEST until the entry is back at its share"""
    s = synth_session(rig.board, [("P1", [("centre", 0.5, 2)] * 14), ("P2", [("centre", 0.5, 2)] * 4)])
    a, seen = run(rig, s, target_test=target)
    p1 = [lab for _, p, lab in seen if p == "P1"]
    assert p1.count("TEST") == cap
    assert [lab for _, p, lab in seen if p == "P2"] == ["TEST"] * 4
    a.finish()
    rows = _per_entry(a)[("centre", (0.4, 0.6))]
    assert abs(sum(lab == "TEST" for _, lab in rows) - 18 / 2) <= 1


def test_entry_never_given_a_second_placement(rig):
    """at the end of the session, an entry still fed by one placement gets the holds it held back released to TEST up
    to its share (release_at_end=True, the default); without release it keeps ceil(target_test / 2)"""
    s = synth_session(rig.board, [("P1", [("centre", 0.5, 2)] * 14)])
    a, seen = run(rig, s)
    assert [lab for *_, lab in seen].count("TEST") == 5
    a.finish()
    labs = [h.label for h in a.holds.values()]
    assert labs.count("TEST") == 7
    assert a.entry_counts()[("centre", (0.4, 0.6))].released == 2
    a, _ = run(rig, s, release_at_end=False)
    a.finish()
    assert [h.label for h in a.holds.values()].count("TEST") == 5


def test_observe_after_finish_is_refused(rig):
    from calibtool import assign
    s = synth_session(rig.board, [("P1", [("centre", 0.5, 2)] * 2)])
    a = assigner(rig)
    hs = list(assign.capture_holds(s))
    a.observe(hs[0])
    a.finish()
    with pytest.raises(RuntimeError, match="finish"):
        a.observe(hs[1])


def test_deterministic_for_the_same_input(rig, tmp_path):
    s = synth_session(rig.board, MIXED)
    out = []
    for i in range(2):
        a, seen = run(rig, s)
        sp = a.write_split(s, tmp_path / str(i))
        out.append((seen, {h.hold: h.label for h in a.holds.values()},
                    {k: np.load(f)[k].tolist() for f in (sp.fit_npz, sp.test_npz) for k in ("img_t0", "img_hold")}))
    assert out[0][0] == out[1][0] and out[0][1] == out[1][1]
    assert out[0][2] == out[1][2]


def test_recell_names_the_holds_another_principal_point_moves(rig):
    """the cell is set against the nominal principal point at capture; recell(pp) reports the holds whose cell
    differs against another one (the fitted principal point, after the fit)"""
    a, _ = run(rig, synth_session(rig.board, [("A", [("centre", 0.75, 2), ("left", 0.75, 2), ("bottom", 0.75, 2)])]))
    assert a.recell(PP) == {}
    moved = a.recell((PP[0], PP[1] + 300))          # pp 300 px lower: the centre hold now reads top
    assert moved == {0: ("centre", "top"), 1: ("left", "top-left"), 2: ("bottom", "centre")}


# ---------------------------------------------------------------- the stored sessions (fast: no fit)

NAMES = ["p21_1105", "p10_1106"]
MIN_HOLDS = 8                     # gates.lens_min_holds: holds a step-66 entry needs to be judged


def _judged(counts):
    """step-66 entries (DIST_BINS) with >= MIN_HOLDS TEST holds"""
    import lenscells
    return sorted(e for e, n in counts.items() if e[1] in lenscells.DIST_BINS and n >= MIN_HOLDS)


@pytest.mark.parametrize("name", NAMES)
def test_stored_session_capture_split(name, data_root, rig, tmp_path):
    st = sessions.get(name)
    s = Session.load(st)
    from calibtool import assign
    r = pipeline.load_rig(st.rig_path)
    a = assign.Assigner(r, (r.resolution[0] / 2, r.resolution[1] / 2), r.nominal_focal_px)
    for h in assign.capture_holds(s):
        a.observe(h)
    sp = a.write_split(s, tmp_path)
    mine = {k: np.load(f) for k, f in (("fit", sp.fit_npz), ("test", sp.test_npz))}
    rec = {k: np.load(data_root / st.recorded.files[f"{k}_npz"]) for k in ("fit", "test")}
    # the rough distance is step 16's, pair for pair; the far pairs are 16's far pairs
    z16 = {int(t): float(z) for f in rec.values() for t, z in zip(f["img_t0"], f["img_z"])}
    zme = {int(t): float(z) for f in mine.values() for t, z in zip(f["img_t0"], f["img_z"])}
    assert set(z16) == set(zme) == set(s.pairs["t0"].tolist())
    assert all((math.isnan(z16[t]) and math.isnan(zme[t])) or abs(z16[t] - zme[t]) <= 1e-6 for t in z16)
    far = lambda f: set(f["img_t0"][~(np.nan_to_num(f["img_z"], nan=np.inf) <= 3.0)].tolist())   # noqa: E731
    assert far(mine["test"]) == far(rec["test"]) and not far(mine["fit"])
    # the rule's properties on real holds
    ec = a.entry_counts()
    for e, c in ec.items():
        assert abs(c.test - c.holds / 2) <= 1, (e, c)
        if c.placements >= 2:
            assert c.test_placements >= 2, (e, c)
    # entries with >= 8 TEST holds, counted with the capture-time entries: this split vs step 16's
    lab16 = {int(t): "TEST" for t in rec["test"]["img_t0"]}
    n16, nme = {}, {}
    for h in a.holds.values():
        if h.entry is None:
            continue
        nme[h.entry] = nme.get(h.entry, 0) + (h.label == "TEST")
        near_t0 = h.pair_t0[h.near]
        n16[h.entry] = n16.get(h.entry, 0) + any(lab16.get(int(t)) == "TEST" for t in near_t0)
    j16, jme = _judged(n16), _judged(nme)
    print(f"CAPTURE SPLIT {name}: FIT {sp.fit_pairs} TEST {sp.test_pairs} pairs; judged at capture: mine {len(jme)} "
          f"{jme}; step 16 {len(j16)} {j16}; held back {sum(c.held_back for c in ec.values())}, released "
          f"{sum(c.released for c in ec.values())}")
    assert len(jme) >= len(j16)


# ---------------------------------------------------------------- 6. replay: the capture split fits and validates


@pytest.mark.replay
@pytest.mark.parametrize("name", NAMES)
def test_replay_capture_split_validates(name, data_root, tmp_path):
    """the capture-time split through step 45 and step 66: the new fit PASSes, judged on at least as many entries
    (>= 8 TEST holds after the fit) as step 16's split was (the recorded run: p21 7, p10 9). Also reported: the holds
    whose cell moves from the nominal to the fitted principal point, and the TEST holds whose step-66 entry differs
    from their capture-time one"""
    import lenscells
    from calibtool import assign
    st = sessions.get(name)
    s = Session.load(st)
    r = pipeline.load_rig(st.rig_path)
    a = assign.Assigner(r, (r.resolution[0] / 2, r.resolution[1] / 2), r.nominal_focal_px)
    for h in assign.capture_holds(s):
        a.observe(h)
    sp = a.write_split(s, tmp_path / "split")
    fit = pipeline.fit45(sp.fit_npz, st.rig_path, tmp_path / "fit", max_z=3.0)
    assert fit.converged, fit.verdict
    v = pipeline.validate66(fit.json_path, sp.test_npz, tmp_path / "validate", rig=st.rig_path,
                            fit_report=fit.json_path)
    sm = pipeline.summary(fit.json_path, sp.test_npz, st.rig_path)
    ppf = fit.entry["intr"][0][2:4]
    moved = a.recell(ppf)
    rec = sm.records
    cells = lenscells.cell_of(np.asarray(rec["u"]) - ppf[0], np.asarray(rec["v"]) - ppf[1])
    bins = lenscells.dist_bin(rec["z"])
    after = {}
    for hold, c, b in zip(rec["hold"], cells, bins):
        after.setdefault(int(hold), set()).add((str(c), lenscells.DIST_BINS[b] if b >= 0 else None))
    test_holds = [h for h in a.holds.values() if h.label == "TEST" and h.entry is not None and h.hold in after]
    changed = [h.hold for h in test_holds if after[h.hold] != {h.entry if h.entry[1] in lenscells.DIST_BINS
                                                               else (h.entry[0], None)}]
    n16 = len(st.recorded.new_entries)
    out = {"session": name, "split": {"fit_pairs": sp.fit_pairs, "test_pairs": sp.test_pairs},
           "fit": fit.as_dict(), "validate": v.as_dict(),
           "summary": {k: getattr(sm, k) for k in ("median_pct", "iqr_pct", "rows_px", "n_pairs")},
           "judged_16": n16, "judged_capture": v.n_validated,
           "holds_cell_moved_by_fitted_pp": len(moved), "holds_with_entry": sum(h.entry is not None
                                                                                for h in a.holds.values()),
           "test_holds_entry_changed_after_fit": len(changed), "test_holds_scored": len(test_holds),
           "held_back": sum(c.held_back for c in a.entry_counts().values()),
           "released": sum(c.released for c in a.entry_counts().values())}
    print("CAPTURE REPLAY " + json.dumps(out, default=str))
    assert v.verdict == "PASS", v.text
    assert v.n_validated >= n16, f"{v.n_validated} judged entries vs {n16} on step 16's split"
