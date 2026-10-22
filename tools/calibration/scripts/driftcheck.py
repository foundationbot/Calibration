"""Stereo drift check (step 67): does the head still match its calibration? From one still board: camera 1's roll and
pitch against the calibration (epipolar coplanarity of the corners' rays, R' = Exp(d) R with d = (pitch, 0, roll) in
camera-0 axes, translation fixed: yaw moves horizontal disparity only, which coplanarity cannot see), and the depth error
at the board (board scale - 1, lensscore). Plan docs/superpowers/plans/2026-10-22-phantom12-drift-check.md."""
from __future__ import annotations

import numpy as np

import lensmodel
import stereofit as sf

F_SCALE = 1e-4                 # soft_l1 scale of the coplanarity residual (normalised; ~0.06 px at f 565)
MIN_CORNERS = 24


def _common(d, img):
    """(uv0, uv1) of the corners both cameras saw in image img, matched by (tag, corner)"""
    m = np.asarray(d["det_img"]) == img
    by = ({}, {})
    for c, t, k, u, v in zip(np.asarray(d["det_cam"])[m], np.asarray(d["det_tag"])[m], np.asarray(d["det_k"])[m],
                             np.asarray(d["det_u"])[m], np.asarray(d["det_v"])[m]):
        by[int(c)][(int(t), int(k))] = (float(u), float(v))
    keys = sorted(set(by[0]) & set(by[1]))
    return np.array([by[0][q] for q in keys]).reshape(-1, 2), np.array([by[1][q] for q in keys]).reshape(-1, 2)


def _residual(x, r0, r1, R, t):
    Rp = sf.rotmat(np.array([[x[0], 0.0, x[1]]]))[0] @ R
    n = np.cross(t, r1 @ Rp.T)                                       # t x (R' r1), per corner
    return np.einsum("ij,ij->i", r0, n) / np.linalg.norm(t)


def _fit(r0, r1, R, t):
    from scipy.optimize import least_squares
    s = least_squares(_residual, np.zeros(2), args=(r0, r1, R, t), loss="soft_l1", f_scale=F_SCALE,
                      x_scale=1e-3, xtol=1e-12, ftol=1e-12, gtol=1e-12)
    return np.degrees(s.x), bool(s.success)


def roll_pitch(model, intr, C, d, board=None) -> dict:
    """camera 1's pitch and roll against (model, intr, C) on the still pairs of corner dump d: pooled over every pair,
    and per pair (p10..p90)"""
    C = np.asarray(C, float)
    R, t = C[:3, :3], C[:3, 3]
    rays, per = [], []
    for img in range(len(d["img_t0"])):
        uv0, uv1 = _common(d, img)
        if len(uv0) < MIN_CORNERS:
            continue
        r0, r1 = lensmodel.unproject(model, intr[0], uv0), lensmodel.unproject(model, intr[1], uv1)
        ok = np.isfinite(r0).all(1) & np.isfinite(r1).all(1)
        if ok.sum() < MIN_CORNERS:
            continue
        rays.append((r0[ok], r1[ok], uv0[ok]))
        per.append(_fit(r0[ok], r1[ok], R, t)[0])
    if not rays:
        return {"pairs": 0}
    (pitch, roll), ok = _fit(np.vstack([r[0] for r in rays]), np.vstack([r[1] for r in rays]), R, t)
    per = np.array(per)
    return {"pairs": len(rays), "pitch_deg": float(pitch), "roll_deg": float(roll), "converged": ok,
            "pitch_p10_p90": np.percentile(per[:, 0], [10, 90]).tolist(),
            "roll_p10_p90": np.percentile(per[:, 1], [10, 90]).tolist(),
            "corners": int(sum(len(r[0]) for r in rays)), "uv0": [r[2] for r in rays]}


# D2 (coverage) and D3 (verdict), fixed in the plan before any data
MIN_PAIRS = 10
MIN_SPAN_U, MIN_SPAN_V = 400.0, 150.0          # raw px spanned by the median pair's camera-0 corners
MAX_DU, MAX_DV = 300.0, 200.0                  # the board centre's distance from the principal point, raw px
DEPTH_BAND_M = (0.4, 1.3)                      # the depth part is judged only here
MAX_ROLL_DEG, MAX_PITCH_DEG, MAX_DEPTH_PCT = 0.10, 0.03, 1.0


def check(model, intr, C, d, board) -> dict:
    """the drift check of one still-board corner dump d against (model, intr, C): numbers, reasons and the verdict
    (OK / DRIFTED / NO RESULT); a JSON-safe dict"""
    import lensscore as ls
    intr = [np.asarray(v, float) for v in intr]
    rp = roll_pitch(model, intr, C, d)
    uv0 = rp.pop("uv0", [])
    out = {"model": model, **rp, "reasons": [], "depth_judged": False}
    if rp["pairs"] < MIN_PAIRS:
        out["reasons"].append(f"{rp['pairs']} still pairs with >= {MIN_CORNERS} common corners (needs {MIN_PAIRS})")
        out["verdict"] = "NO RESULT"
        return out
    spans = np.array([np.ptp(u, axis=0) for u in uv0])
    span_u, span_v = (float(v) for v in np.median(spans, axis=0))
    centre = np.median(np.array([u.mean(0) for u in uv0]), axis=0)
    du, dv = float(centre[0] - intr[0][2]), float(centre[1] - intr[0][3])
    out.update(span_px=[span_u, span_v], centre_px=centre.tolist(), centre_offset_px=[du, dv])
    if span_u < MIN_SPAN_U or span_v < MIN_SPAN_V:
        out["reasons"].append(f"the board spans {span_u:.0f} x {span_v:.0f} raw px (needs {MIN_SPAN_U:.0f} x "
                              f"{MIN_SPAN_V:.0f}): move it closer or into the middle of the image")
    if abs(du) > MAX_DU or abs(dv) > MAX_DV:
        out["reasons"].append(f"the board centre is {du:+.0f}, {dv:+.0f} raw px from the image centre (limits "
                              f"+-{MAX_DU:.0f}, +-{MAX_DV:.0f}): centre the board (at an edge roll and yaw mix)")
    if out["reasons"]:
        out["verdict"] = "NO RESULT"
        return out
    rec, _, solves = ls.score(model, intr, C, sf.Data(dict(d), board))
    if ls.solved(solves):
        e, z = 100 * (np.asarray(rec["scale"]) - 1), np.asarray(rec["z"])
        out.update(depth_pct=float(np.median(e)), depth_p10_p90=np.percentile(e, [10, 90]).tolist(),
                   z_m=float(np.median(z)))
        out["depth_judged"] = bool(DEPTH_BAND_M[0] <= out["z_m"] <= DEPTH_BAND_M[1])
    else:
        out["depth_note"] = "the board-scale solve did not converge: depth not judged"
    drift = []
    if abs(out["roll_deg"]) > MAX_ROLL_DEG:
        drift.append(f"roll {out['roll_deg']:+.3f} deg (limit {MAX_ROLL_DEG})")
    if abs(out["pitch_deg"]) > MAX_PITCH_DEG:
        drift.append(f"pitch {out['pitch_deg']:+.3f} deg (limit {MAX_PITCH_DEG})")
    if out["depth_judged"] and abs(out["depth_pct"]) > MAX_DEPTH_PCT:
        drift.append(f"depth error {out['depth_pct']:+.2f} % at {out['z_m']:.2f} m (limit {MAX_DEPTH_PCT} %)")
    out["reasons"] = drift
    out["verdict"] = "DRIFTED" if drift else "OK"
    return out


def load_okvis(path):
    """the calibration an OKVIS config makes OKVIS run: (model, [intr0, intr1], C) with C = inv(T_SC0) T_SC1 (camera-1
    points into camera 0); KB4T for a kb4t file, KB4 for an equidistant one"""
    import okviscfg
    import rigmath as rm
    cfg = okviscfg.load(path)
    model = {"pinhole-kb4t": "KB4T", "pinhole-equi": "KB4"}.get(cfg.lens)
    if model is None:
        raise ValueError(f"{path}: lens {cfg.lens}; the drift check reads kb4t or equidistant configs")
    intr = [np.r_[cfg.K[c][0, 0], cfg.K[c][1, 1], cfg.K[c][0, 2], cfg.K[c][1, 2], np.ravel(cfg.D[c])] for c in (0, 1)]
    return model, intr, rm.inv(cfg.T_SC[0]) @ cfg.T_SC[1]
