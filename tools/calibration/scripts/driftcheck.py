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
