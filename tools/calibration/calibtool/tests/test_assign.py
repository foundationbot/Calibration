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
