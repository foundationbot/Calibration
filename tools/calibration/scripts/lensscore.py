"""The depth scorer (step 66, the acceptance test): a calibration held fixed, per stereo pair the board's pose, then
pose and board scale; board scale - 1 is the stereo depth error at the board (a calibration that reads depth 1 % long
needs a 1 % larger board to explain the pixels: + = reads too far).

Ported from sweep_590.evaluate_pp (19 Oct), take_eval.py (cell_dist_table, job_inject part (a)) and bands590.py
(boot_median, B = 2000, MIN_CLUSTERS = 8). Changed: board start poses by PnP with the scored calibration (not
phantom12's deployed one); clusters = the corner file's holds; cells and bins from lenscells (relative to the scored
calibration's principal point); the summary numbers sweep.evaluate kept for lens-sweep.json dropped."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lenscells  # noqa: E402
import stereofit as sf  # noqa: E402

SEED = 20261020
B = 2000
REC = ("src", "hold", "scale", "z", "u", "v")


def _solve_fixed(d, model, intr, C, poses, scale):
    """stereofit.solve with the camera fixed, at stereofit.MAX_NFEV evaluations a round (step 45's cap; solve()'s
    default is 300), and once more from its own result (poses, board scales) when scipy did not report success: a
    solve stopped at its cap is not yet a failure. No third try: still short, its success stays false"""
    r = sf.solve(d, model, intr, C, poses, free_intr=False, free_C=False, scale=scale, max_nfev=sf.MAX_NFEV)
    if not r["success"]:
        r = sf.solve(d, model, intr, C, r["poses"], free_intr=False, free_C=False, scale=scale, max_nfev=sf.MAX_NFEV,
                     ls0=r["log_scale"])
    return r


def score(model, intr, C, data):
    """-> (records {src, hold, scale, z, u, v} per pair, {hold: pixel rms with the camera fixed}, solves): board scale,
    the board centre's camera-0 depth z (m; the centre of the corners seen) and pixel (u, v: their camera-0 mean);
    solves = {"pose_success", "scale_success": scipy's success of the pose solve and of the pose-and-scale solve,
    each restarted once from its result if it stopped short (_solve_fixed), "pnp_failed_pairs": pairs left out
    because their PnP start failed}. A solve that did not converge leaves board scales that are not depth errors:
    check solved(solves) before using the records"""
    if sf.fast_solver(model) is not None:                         # CALIB_SOLVER=fast: calibtool.score, same contract
        from calibtool import score as fast_score
        return fast_score.score(model, intr, C, data)
    poses, ok = sf.pnp_poses(data, model, intr, C)
    d = data.select(ok)
    r = _solve_fixed(d, model, intr, C, poses[ok], scale=False)
    s = _solve_fixed(d, model, intr, C, r["poses"], scale=True)
    scale = np.exp(s["log_scale"])
    R = sf.rotmat(s["poses"][:, :3])
    rec = {k: [] for k in REC}
    for p in range(d.n):
        m = d.pair == p
        pc = R[p] @ (d.P[m].mean(0) * scale[p]) + s["poses"][p, 3:]          # as sweep_590.evaluate_pp
        u, v = d.uv[m & (d.cam == 0)].mean(0)
        for k, val in zip(REC, (int(d.src_pairs[p]), int(d.hold[p]), float(scale[p]), float(pc[2]), float(u),
                                float(v))):
            rec[k].append(val)
    return rec, sf.hold_rms(d, r["res"]), {"pose_success": bool(r["success"]), "scale_success": bool(s["success"]),
                                            "pnp_failed_pairs": int((~ok).sum())}


def solved(solves):
    """both scoring solves reached scipy's convergence"""
    return bool(solves["pose_success"] and solves["scale_success"])


def keep_holds(rec, drop):
    """the records without the holds in drop (pairs are independent with the camera fixed: no rescoring needed)"""
    m = ~np.isin(np.asarray(rec["hold"]), list(drop))
    return {k: np.asarray(v)[m].tolist() for k, v in rec.items()}


def self_check(model, intr, C, data, base, scorer=score):
    """take_eval.job_inject (a): the stereo baseline x 1.01 must read +1.00 % +- 0.10 in every distance bin with data
    (median over the pairs both runs kept of scale ratio - 1), and at least ONE bin must have data. A similarity
    identity: it catches only a scorer that ignores the baseline, the one failure a scorer can have with no other
    symptom. base = score()'s records of the unchanged calibration on the same data. The rescoring's solves must
    converge too ("solved"); otherwise the check has not passed. -> {"bins": [...], "solved": bool, "passed": bool}

    One bin, where the reference needed data in >= 3 of its 4 bins (0.4-2.0 m): there part (a) ran beside a camera-1
    0.2 px shift compared with an off-axis prediction and its slope with distance, which need a spread of distances.
    Part (a) alone does not: a 1 % longer baseline scales every triangulated point by 1.01 at any distance, so one bin
    with data already tells a scorer that reads the baseline from one that ignores it. The bins here span 0.4-1.3 m, the
    whole test sweep; demanding all three would refuse a test sweep that covers only part of that range, whose empty
    bins the pass rule already leaves "not validated"."""
    Cb = np.array(C, float, copy=True)
    Cb[:3, 3] *= 1.01
    rec1, _, solves = scorer(model, intr, Cb, data)
    b = dict(zip(rec1["src"], rec1["scale"]))
    rows = [(100 * (b[s] / sc - 1), z) for s, sc, z in zip(base["src"], base["scale"], base["z"]) if s in b]
    ratio = np.array([x[0] for x in rows])
    z = np.array([x[1] for x in rows])
    bins = []
    for lo, hi in lenscells.DIST_BINS:
        m = (z >= lo) & (z < hi)
        bins.append({"bin_m": [lo, hi], "pairs": int(m.sum()),
                     "median_pct": float(np.median(ratio[m])) if m.any() else None})
    with_data = [x for x in bins if x["pairs"]]
    ok = solved(solves)
    return {"bins": bins, "solved": ok,
            "passed": ok and bool(with_data) and all(abs(x["median_pct"] - 1.0) <= 0.10 for x in with_data)}


def boot_median(x, cl, rng, min_clusters):
    """95 % cluster-bootstrap interval of the median, or None below min_clusters clusters (bands590.boot_median)"""
    ids = np.unique(cl)
    if len(ids) < min_clusters:
        return None
    groups = [x[cl == c] for c in ids]
    meds = np.empty(B)
    for i in range(B):
        meds[i] = np.median(np.concatenate([groups[j] for j in rng.integers(0, len(ids), len(ids))]))
    return [float(v) for v in np.percentile(meds, [2.5, 97.5])]


def cell_dist_table(rec, pp, min_holds, seed=SEED):
    """per cell x distance bin with data: pairs, holds, median depth error (%), 95 % cluster-bootstrap interval from
    min_holds holds up (take_eval.cell_dist_table; clusters = holds)"""
    lab = lenscells.cell_of(np.asarray(rec["u"]) - pp[0], np.asarray(rec["v"]) - pp[1])
    bins = lenscells.dist_bin(rec["z"])
    e = 100 * (np.asarray(rec["scale"]) - 1)
    cl = np.asarray(rec["hold"])
    rng = np.random.default_rng(seed)
    out = []
    for c in lenscells.CELLS:
        for i, (lo, hi) in enumerate(lenscells.DIST_BINS):
            m = (lab == c) & (bins == i)
            if m.any():
                out.append({"cell": c, "dist_m": [lo, hi], "pairs": int(m.sum()), "holds": int(len(np.unique(cl[m]))),
                            "median_pct": float(np.median(e[m])), "ci95_pct": boot_median(e[m], cl[m], rng, min_holds)})
    return out


def verdict(table, gates):
    """The fixed pass rule: every entry with >= lens_min_holds holds has |median| <= lens_pass_pct (corner cells
    lens_pass_corner_pct); entries with fewer holds are not validated; no validated entry = NO RESULT.
    -> ("PASS" | "FAIL" | "NO RESULT", validated entries with limit and ok, entries not validated)"""
    n = int(gates["lens_min_holds"])
    val, notval = [], []
    for r in table:
        if r["holds"] >= n:
            lim = gates["lens_pass_corner_pct"] if r["cell"] in lenscells.CORNERS else gates["lens_pass_pct"]
            val.append({**r, "limit_pct": lim, "ok": bool(abs(r["median_pct"]) <= lim)})
        else:
            notval.append(r)
    if not val:
        return "NO RESULT", val, notval
    return ("PASS" if all(r["ok"] for r in val) else "FAIL"), val, notval
