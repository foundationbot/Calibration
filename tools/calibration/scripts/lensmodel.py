"""The lens models the calibration package fits and uses, in numpy (no OpenCV needed).

  KB4   equidistant fisheye (Kannala-Brandt), k1..k4: OpenCV's cv2.fisheye model and OKVIS's "equidistant".
  KB4T  KB4 plus Brown tangential terms p1, p2 on the bent normalised point. phantom12's lens needs them: the
        15 Oct Kalibr KB4 calibration read depth up to +4.75 % toward the image sides; KB4T fits it (Fit A, 20 Oct).
        Neither OpenCV nor OKVIS has it.

intr = [fx, fy, cx, cy, *dist]; dist = k1..k4 (KB4) or k1..k4, p1, p2 (KB4T); no skew. The arithmetic is the
analysis code's that fitted Fit A (lensfit.project, 18-20 Oct), operation for operation, so a calibration fitted
there projects identically here (pinned by tests/data/kb4t_reference.json)."""
from __future__ import annotations

import numpy as np

MODELS = {"KB4": 4, "KB4T": 6}
NEWTON_STEPS = 30


def _check(model: str, intr) -> np.ndarray:
    if model not in MODELS:
        raise ValueError(f"lens model {model!r}: one of {sorted(MODELS)}")
    intr = np.asarray(intr, float).ravel()
    n = 4 + MODELS[model]
    if intr.size != n:
        raise ValueError(f"{model} takes {n} numbers (fx fy cx cy + {MODELS[model]} distortion), got {intr.size}")
    return intr


def _tangential(mx, my, p1, p2):
    r2 = mx * mx + my * my
    return mx + 2 * p1 * mx * my + p2 * (r2 + 2 * mx * mx), my + p1 * (r2 + 2 * my * my) + 2 * p2 * mx * my


def project(model: str, intr, P) -> np.ndarray:
    """Camera-frame points (N, 3) -> pixels (N, 2). The ray's angle theta off the axis is bent by the KB4
    polynomial, then (KB4T) the tangential terms move the bent normalised point."""
    intr = _check(model, intr)
    fx, fy, cx, cy = intr[:4]
    d = intr[4:]
    X, Y, Z = np.asarray(P, float).reshape(-1, 3).T
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.hypot(X, Y)
        th = np.arctan2(rho, Z)
        t2 = th * th
        thd = th * (1 + d[0] * t2 + d[1] * t2 ** 2 + d[2] * t2 ** 3 + d[3] * t2 ** 4)
        s = np.where(rho > 1e-12, thd / np.maximum(rho, 1e-12), 1.0 / Z)
        mx, my = s * X, s * Y
        if model == "KB4T":
            mx, my = _tangential(mx, my, d[4], d[5])
    return np.c_[fx * mx + cx, fy * my + cy]


def unproject(model: str, intr, uv) -> np.ndarray:
    """Pixels (N, 2) -> unit rays (N, 3) in the camera frame: project's inverse by Newton steps (the tangential
    terms first, then the KB4 polynomial for theta). Exact to ~1e-12 rad wherever the polynomial still rises,
    which for phantom12's lenses is beyond the raw image's corners (~77 deg)."""
    intr = _check(model, intr)
    fx, fy, cx, cy = intr[:4]
    d = intr[4:]
    uv = np.asarray(uv, float).reshape(-1, 2)
    mx, my = (uv[:, 0] - cx) / fx, (uv[:, 1] - cy) / fy
    if model == "KB4T":
        p1, p2 = d[4], d[5]
        a, b = mx.copy(), my.copy()
        for _ in range(NEWTON_STEPS):
            ta, tb = _tangential(a, b, p1, p2)
            ea, eb = ta - mx, tb - my
            j11, j12 = 1 + 2 * p1 * b + 6 * p2 * a, 2 * p1 * a + 2 * p2 * b
            j21, j22 = 2 * p1 * a + 2 * p2 * b, 1 + 6 * p1 * b + 2 * p2 * a
            det = j11 * j22 - j12 * j21
            a, b = a - (j22 * ea - j12 * eb) / det, b - (j11 * eb - j21 * ea) / det
        mx, my = a, b
    thd = np.hypot(mx, my)
    th = thd.copy()
    for _ in range(NEWTON_STEPS):
        t2 = th * th
        f = th * (1 + d[0] * t2 + d[1] * t2 ** 2 + d[2] * t2 ** 3 + d[3] * t2 ** 4) - thd
        df = 1 + 3 * d[0] * t2 + 5 * d[1] * t2 ** 2 + 7 * d[2] * t2 ** 3 + 9 * d[3] * t2 ** 4
        th = th - f / df
    s = np.where(thd > 1e-12, np.sin(th) / np.maximum(thd, 1e-12), 1.0)
    return np.c_[s * mx, s * my, np.cos(th)]
