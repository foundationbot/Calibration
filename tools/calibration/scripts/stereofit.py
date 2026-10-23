"""The KB4 / KB4T stereo lens fit (steps 45, 46, 66): corner data, the stereo bundle adjustment, the nominal start
and the converged-fit loop.

Ported from the phantom12 analysis scripts of 17-20 Oct (~/calib/phantom12/results/tbs-2026-10-17/lens/, not in git):
lensfit.py (Data, rotmat, residuals, sparsity, _unpack, _pack, _solve_once, solve, ring_rms), sweep.py (init_poses),
sweep_590.py (fit_cost, the 2000-evaluation cap, 3 attempts) and fits_take.py (the restart loop). Changed: only KB4 and
KB4T (projection from lensmodel.py); no phantom12 constants (lensfit.CENTRE, sweep.cfg / C_cfg / KB4_cfg): the start is
the rig file's nominal camera, never a deployed calibration; every pair carries its hold, the cluster for error bars;
no max_radius cut (Fit A used every corner). numpy + scipy; cv2 only for the PnP start."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lensmodel  # noqa: E402
import rigmath as rm  # noqa: E402

# MAX_NFEV caps ONE solve round: solve() repeats its least squares up to TRIM_ROUNDS times while the trimmed corner
# set settles, so one attempt can take up to TRIM_ROUNDS x MAX_NFEV evaluations; fit_converged restarts up to
# MAX_ATTEMPTS times until scipy reports success (sweep_590.py: 2000 evaluations, 3 attempts)
MAX_NFEV, MAX_ATTEMPTS, TRIM_ROUNDS = 2000, 3, 4
TRIM_PX = 3.0                        # corners beyond this cut and the fit repeated (lensfit.solve)
GROSS_PX = 20.0                      # residuals beyond this are detection failures, not noise (lensfit.ring_rms)


class Data:
    """Corners of stereo pairs from step 15's npz: per corner its pair (0..n-1), camera, board point and pixel; per
    pair its index in the npz (src_pairs) and its hold (img_hold: one still stretch). Pairs with fewer than
    min_corners corners in either camera are left out. AprilGrid boards only (corners carry tag ids)."""

    def __init__(self, npz, board, min_corners=12):
        d = dict(np.load(npz)) if not isinstance(npz, dict) else npz
        img, cam = np.asarray(d["det_img"], int), np.asarray(d["det_cam"], int)
        size = int(img.max()) + 1 if img.size else 0
        n0 = np.bincount(img[cam == 0], minlength=size)
        n1 = np.bincount(img[cam == 1], minlength=size)
        ok = np.flatnonzero((n0 >= min_corners) & (n1 >= min_corners))
        m = np.isin(img, ok)
        remap = np.full(size, -1, int)
        remap[ok] = np.arange(len(ok))
        obj = np.array([board.obj_points(t) for t in range(board.n_tags)])          # (tags, 4, 3)
        self.src_pairs = ok
        self.hold = np.asarray(d["img_hold"], int)[ok]
        self.pair = remap[img[m]]
        self.cam = cam[m]
        self.P = obj[np.asarray(d["det_tag"], int)[m], np.asarray(d["det_k"], int)[m]].reshape(-1, 3)
        self.uv = np.c_[d["det_u"], d["det_v"]][m].astype(float)
        self.n = len(ok)

    def subset(self, mask):
        """the same pairs (numbering kept), only the corners in mask (lensfit.Data.subset)"""
        out = object.__new__(Data)
        out.src_pairs, out.hold, out.n = self.src_pairs, self.hold, self.n
        out.pair, out.cam, out.P, out.uv = self.pair[mask], self.cam[mask], self.P[mask], self.uv[mask]
        return out

    def select(self, keep):
        """only the pairs where keep (bool per pair), renumbered 0..n'-1: a pair without corners would leave six
        parameters with no residual in the fit"""
        keep = np.asarray(keep, bool)
        new = np.cumsum(keep) - 1
        m = keep[self.pair]
        out = object.__new__(Data)
        out.src_pairs, out.hold, out.n = self.src_pairs[keep], self.hold[keep], int(keep.sum())
        out.pair, out.cam, out.P, out.uv = new[self.pair[m]], self.cam[m], self.P[m], self.uv[m]
        return out


def rotmat(r):
    """Rotation vectors (N, 3) -> matrices (N, 3, 3) (Rodrigues, vectorised; lensfit.rotmat)."""
    r = np.atleast_2d(r)
    th = np.linalg.norm(r, axis=1)
    k = r / np.where(th > 1e-12, th, 1.0)[:, None]
    K = np.zeros((len(r), 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 0], K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = (-k[:, 2], k[:, 1], k[:, 2], -k[:, 0],
                                                                                -k[:, 1], k[:, 0])
    s, c = np.sin(th)[:, None, None], np.cos(th)[:, None, None]
    R = np.eye(3) + s * K + (1 - c) * (K @ K)
    R[th < 1e-12] = np.eye(3)
    return R


def _unpack(x, model, n_pairs, free_intr, free_C, scale, intr_fixed, C0):
    ni = 4 + lensmodel.MODELS[model]
    i = 0
    if free_intr:
        intr = [x[0:ni], x[ni:2 * ni]]
        i = 2 * ni
    else:
        intr = intr_fixed
    if free_C:
        dC = x[i:i + 6]
        i += 6
    else:
        dC = np.zeros(6)
    poses = x[i:i + 6 * n_pairs].reshape(n_pairs, 6)
    i += 6 * n_pairs
    ls = x[i:i + n_pairs] if scale else np.zeros(n_pairs)
    C = C0 @ np.r_[np.c_[rotmat(dC[:3])[0], dC[3:]], [[0, 0, 0, 1]]]
    return intr, C, poses, ls


def residuals(x, data: Data, model, free_intr, free_C, scale, intr_fixed, C0):
    intr, C, poses, ls = _unpack(x, model, data.n, free_intr, free_C, scale, intr_fixed, C0)
    R = rotmat(poses[:, :3])[data.pair]
    Pw = data.P * np.exp(ls)[data.pair][:, None]
    Pc0 = np.einsum("nij,nj->ni", R, Pw) + poses[data.pair, 3:]
    Ci = np.linalg.inv(C)                                        # camera 0 points -> camera 1
    out = np.empty((len(data.P), 2))
    for c in (0, 1):
        m = data.cam == c
        Pc = Pc0[m] if c == 0 else Pc0[m] @ Ci[:3, :3].T + Ci[:3, 3]
        r = lensmodel.project(model, intr[c], Pc) - data.uv[m]
        r[~np.isfinite(r)] = 1e3
        r[Pc[:, 2] <= 1e-6] = 1e3
        out[m] = r
    return out.ravel()


def sparsity(data: Data, model, free_intr, free_C, scale):
    ni = 4 + lensmodel.MODELS[model]
    rows, cols = [], []
    i0 = 0
    nres = 2 * len(data.P)
    ridx = np.arange(nres)
    corner = ridx // 2
    if free_intr:
        for c in (0, 1):
            rr = ridx[data.cam[corner] == c]
            for j in range(ni):
                rows.append(rr)
                cols.append(np.full(len(rr), c * ni + j))
        i0 = 2 * ni
    if free_C:
        rr = ridx[data.cam[corner] == 1]
        for j in range(6):
            rows.append(rr)
            cols.append(np.full(len(rr), i0 + j))
        i0 += 6
    for j in range(6):
        rows.append(ridx)
        cols.append(i0 + 6 * data.pair[corner] + j)
    i0 += 6 * data.n
    if scale:
        rows.append(ridx)
        cols.append(i0 + data.pair[corner])
        i0 += data.n
    r, c = np.concatenate(rows), np.concatenate(cols)
    return coo_matrix((np.ones(len(r)), (r, c)), shape=(nres, i0)).tocsr(), i0


def _pack(intr, poses, ls, free_intr, scale):
    x = []
    if free_intr:
        x += [np.asarray(intr[0], float), np.asarray(intr[1], float)]
    x.append(np.asarray(poses, float).ravel())
    if scale:
        x.append(np.asarray(ls, float))
    return np.concatenate(x)


def _solve_once(data: Data, model, intr0, C0, poses0, free_intr, free_C, scale, max_nfev, ls0=None):
    x0 = []
    if free_intr:
        x0 += [np.asarray(intr0[0], float), np.asarray(intr0[1], float)]
    if free_C:
        x0.append(np.zeros(6))
    x0.append(np.asarray(poses0, float).ravel())
    if scale:
        x0.append(np.zeros(data.n) if ls0 is None else np.asarray(ls0, float))
    x0 = np.concatenate(x0)
    S, n = sparsity(data, model, free_intr, free_C, scale)
    assert n == len(x0)
    args = (data, model, free_intr, free_C, scale, intr0, C0)
    sol = least_squares(residuals, x0, jac_sparsity=S, args=args, method="trf", x_scale="jac", max_nfev=max_nfev,
                        tr_solver="lsmr")
    intr, C, poses, ls = _unpack(sol.x, model, data.n, free_intr, free_C, scale, intr0, C0)
    return {"intr": [np.array(v) for v in intr], "C": C, "poses": poses, "log_scale": ls,
            "res": residuals(sol.x, *args).reshape(-1, 2), "success": bool(sol.success), "nfev": int(sol.nfev)}


def solve(data: Data, model, intr0, C0, poses0, free_intr=True, free_C=True, scale=False, trim_px=TRIM_PX,
          rounds=TRIM_ROUNDS, max_nfev=300, ls0=None):
    """Least squares (a robust loss from a far start stalls: tested 19 Oct), then corners beyond trim_px of the fit
    cut and the fit repeated until the cut set settles (lensfit.solve). ls0: the board scales' start (log, per pair;
    None = 1) when scale. -> dict(intr, C, poses, log_scale, res (all corners, N x 2), kept (mask), success, nfev)"""
    keep = np.ones(len(data.P), bool)
    intr, C, poses, ls = intr0, C0, np.asarray(poses0, float), ls0
    nfev, ok = 0, True
    for _ in range(rounds):
        r = _solve_once(data.subset(keep), model, intr, C, poses, free_intr, free_C, scale, max_nfev, ls)
        intr, C, poses, ls = r["intr"], r["C"], r["poses"], r["log_scale"]
        nfev += r["nfev"]
        ok = ok and r["success"]
        x = _pack(intr, poses, ls, free_intr, scale)
        res = residuals(x, data, model, free_intr, False, scale, intr, C).reshape(-1, 2)
        new = np.linalg.norm(res, axis=1) <= trim_px
        if np.array_equal(new, keep):
            break
        keep = new
    return {"intr": intr, "C": C, "poses": poses, "log_scale": ls, "res": res, "kept": keep, "success": ok,
            "nfev": nfev}


def fit_cost(r):
    """sum of squares over the corners the fit kept, and their rms (px) (sweep_590.fit_cost)"""
    e = np.linalg.norm(r["res"][r["kept"]], axis=1)
    return float(np.sum(e ** 2)), float(np.sqrt(np.mean(e ** 2)))


def ring_rms(data: Data, res, centre, edges=(0, 150, 300, 450, 600, 800)):
    """[(lo, hi, rms px, corners)] by distance of the corner from centre (lensfit.ring_rms; centre is a parameter)"""
    r = np.hypot(*(data.uv - np.asarray(centre)).T)
    e = np.linalg.norm(res, axis=1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (r >= lo) & (r < hi) & (e < GROSS_PX)
        out.append((lo, hi, float(np.sqrt(np.mean(e[m] ** 2))) if m.sum() else float("nan"), int(m.sum())))
    return out


def pnp_poses(data: Data, model, intr, C, min_corners=12):
    """Board pose in camera 0 per pair (rotation vector, translation): PnP on camera 0's corners, else camera 1's
    carried through C. Rays from lensmodel.unproject, so any KB4 / KB4T camera works, the zero-distortion nominal one
    included (sweep.init_poses did this with phantom12's deployed calibration). -> (poses (n, 6), ok (n,))"""
    import cv2
    poses, ok = np.zeros((data.n, 6)), np.zeros(data.n, bool)
    for p in range(data.n):
        for c in (0, 1):
            m = (data.pair == p) & (data.cam == c)
            ray = lensmodel.unproject(model, intr[c], data.uv[m])
            front = ray[:, 2] > 0.1                   # within ~84 deg of the axis, where a pinhole direction exists
            if front.sum() < min_corners:
                continue
            und = np.ascontiguousarray(ray[front, :2] / ray[front, 2:3]).reshape(-1, 1, 2)
            obj = np.ascontiguousarray(data.P[m][front]).reshape(-1, 1, 3)
            good, rv, tv = cv2.solvePnP(obj, und, np.eye(3), None)
            if not good or tv[2, 0] <= 0:
                continue
            T = rm.transform(cv2.Rodrigues(rv)[0], tv.ravel())
            if c == 1:
                T = C @ T
            poses[p] = np.r_[rm.log_rot(T[:3, :3]), T[:3, 3]]
            ok[p] = True
            break
    return poses, ok


def load_entry(spec):
    """'file.json' or 'file.json:key' -> that JSON entry (dict)"""
    path, _, key = str(spec).partition(":")
    e = json.loads(Path(path).read_text())
    return e[key] if key else e


def converged_fit(entry):
    """None if the entry may be used as a calibration, else why not: a step-45 file (it has a "verdict") is a
    calibration only when that verdict starts with "converged"; an entry without a verdict comes from elsewhere"""
    if "verdict" not in entry or str(entry["verdict"]).startswith("converged"):
        return None
    return f"step 45's verdict {entry['verdict']!r}: only a converged fit is a calibration"


def load_calibration(spec):
    """'file.json' or 'file.json:key' -> (model, [intr0, intr1], C) from any entry with model, intr and C: step 45's
    output, or a lensfit-era result such as final-fit-nofar.json:KB4T@all@800"""
    e = load_entry(spec)
    return e["model"], [np.asarray(v, float) for v in e["intr"]], np.asarray(e["C"], float)


def calibration_digest(model, intr, C):
    """sha256 of a calibration's numbers: a report names the calibration it judged, and step 71 refuses another"""
    s = json.dumps({"model": model, "intr": [np.asarray(v, float).tolist() for v in intr],
                    "C": np.asarray(C, float).tolist()})
    return hashlib.sha256(s.encode()).hexdigest()


def json_safe(o):
    """o as strict JSON can hold it (81_tbs_handeye._json_safe): numpy types as Python ones, non-finite numbers (a hold
    with no corner within GROSS_PX, an empty ring) as null, never Infinity or NaN"""
    if isinstance(o, dict):
        return {k: json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    if isinstance(o, np.ndarray):
        return json_safe(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    return o


def write_json(path, o):
    """o written as strict JSON (json_safe; allow_nan=False makes a miss an error, not an Infinity in the file)"""
    Path(path).write_text(json.dumps(json_safe(o), indent=1, allow_nan=False))


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_sha():
    """the package's commit: CALIB_GIT_SHA when set (a container mount has no .git), else git, else 'unknown'"""
    if os.environ.get("CALIB_GIT_SHA"):
        return os.environ["CALIB_GIT_SHA"]
    try:
        return subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


# ---- the fit from a nominal start ----------------------------------------------------------------------------------
def hold_rms(data: Data, res):
    """{hold: pixel rms of its corners, both cameras, every pair of the hold} with the camera at a fit: how well the
    hold's corners fit a rigid board of the declared geometry (residuals beyond GROSS_PX left out as lensfit.ring_rms).
    phantom12, 22 Oct, Fit C on 98 held-out neck poses: median 0.42, 95th 1.05, max 1.78 px; the 20 Oct far take
    (1.4-2.0 m) median 1.30 px: the two overlap (docs/superpowers/plans/2026-10-22-lens-hold-gate.md)."""
    e = np.linalg.norm(res, axis=1)
    h = data.hold[data.pair]
    out = {}
    for k in np.unique(data.hold):
        m = (h == k) & (e < GROSS_PX)
        out[int(k)] = float(np.sqrt(np.mean(e[m] ** 2))) if m.any() else float("inf")
    return out


def nominal_start(rig, model="KB4T"):
    """The start of a fit from the rig file alone: both cameras f = cameras.nominal_focal_px, the principal point at
    the image's geometric centre ((W - 1) / 2, (H - 1) / 2: OpenCV and Kalibr put pixel centres on integer coordinates,
    and step 15's corners are in that convention), zero distortion (a pure equidistant lens); the stereo pose C
    (camera-1 points into camera 0) = the identity rotation and t = (s * cameras.nominal_baseline_m, 0, 0) with
    s = rig.expected_dx_sign() (+1: camera 0 = kalibr cam0 is the left eye, so camera 1 sits at +x).
    -> ([intr0, intr1], C)"""
    if rig.lens.kind not in ("equi", "kb4t"):
        raise ValueError(f"{rig.path}: cameras.model {rig.model!r}: the KB4T steps start from an equidistant lens "
                         "(fisheye heads); calibrate a pinhole-radtan rig with Kalibr (step 40)")
    if rig.nominal_baseline_m is None:
        raise ValueError(f"{rig.path}: no cameras.nominal_baseline_m (the stereo baseline before calibration, m, "
                         "from CAD or a past calibration)")
    W, H = rig.resolution
    f = rig.nominal_focal_px
    intr = [np.r_[f, f, (W - 1) / 2.0, (H - 1) / 2.0, np.zeros(lensmodel.MODELS[model])] for _ in (0, 1)]
    C = np.eye(4)
    C[0, 3] = rig.expected_dx_sign() * rig.nominal_baseline_m
    return intr, C


def fit_converged(data: Data, model, intr, C, poses, max_nfev=MAX_NFEV, attempts=MAX_ATTEMPTS, log=print):
    """solve() restarted from its own result until scipy reports success, at most `attempts` times (fits_take.job_fit):
    a fit stopped at its evaluation cap (max_nfev per solve round, up to TRIM_ROUNDS rounds an attempt) is
    unconverged, and its numbers are not a calibration. -> solve()'s dict with "attempts" (per attempt: nfev,
    success, cost, rms_px, kept, seconds), "nfev" summed over them, and the caps it ran with ("max_attempts",
    "max_nfev")"""
    hist, t0 = [], time.time()
    for a in range(attempts):
        r = solve(data, model, intr, C, poses, max_nfev=max_nfev)
        cost, rms = fit_cost(r)
        hist.append({"attempt": a + 1, "nfev": r["nfev"], "success": r["success"], "cost": cost, "rms_px": rms,
                     "kept": int(r["kept"].sum()), "seconds": round(time.time() - t0)})
        log(f"  attempt {a + 1}: nfev {r['nfev']} success {r['success']} rms {rms:.4f} px, kept {hist[-1]['kept']} "
            f"({hist[-1]['seconds']} s)")
        intr, C, poses = r["intr"], r["C"], r["poses"]
        if r["success"]:
            break
    r["attempts"], r["nfev"] = hist, int(sum(h["nfev"] for h in hist))
    r["max_attempts"], r["max_nfev"] = int(attempts), int(max_nfev)
    return r


def _pnp_kept(data: Data, model, intr, C, log):
    """pnp_poses, and the pairs whose PnP start failed left out, counted and logged -> (data kept, poses, failed)"""
    poses, ok = pnp_poses(data, model, intr, C)
    failed = int((~ok).sum())
    if failed:
        log(f"  PnP start failed for {failed} of {data.n} pairs: left out")
    return data.select(ok), poses[ok], failed


def fit_from_nominal(data: Data, model, intr0, C0, log=print):
    """The whole fit from a nominal start: board poses by PnP with the nominal camera, the stereo pose alone with the
    cameras fixed (no trimming: a start, not a result), then everything by fit_converged. The middle step halved the
    time from an identity start to a pair rolled 21 deg (synthetic, 20 Oct). -> (the pairs used, the fit, with
    "pnp_failed_pairs": the pairs left out because their PnP start failed)"""
    data, poses, failed = _pnp_kept(data, model, intr0, C0, log)
    pre = solve(data, model, intr0, C0, poses, free_intr=False, free_C=True, trim_px=np.inf, max_nfev=MAX_NFEV)
    log(f"  stereo pose alone from the nominal start: nfev {pre['nfev']}, success {pre['success']}")
    r = fit_converged(data, model, intr0, pre["C"], pre["poses"], log=log)
    r["pnp_failed_pairs"] = failed
    return data, r


def warm_fit(data: Data, model, intr, C, log=print):
    """a converged fit started from a calibration: board poses by PnP, refitted with the camera fixed, then
    fit_converged (fits_take.start_from + job_fit); "pnp_failed_pairs" as fit_from_nominal"""
    data, poses, failed = _pnp_kept(data, model, intr, C, log)
    r0 = solve(data, model, intr, C, poses, free_intr=False, free_C=False, max_nfev=MAX_NFEV)
    r = fit_converged(data, model, intr, C, r0["poses"], log=log)
    r["pnp_failed_pairs"] = failed
    return data, r
