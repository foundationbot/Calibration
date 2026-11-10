"""The fast stereo lens solve (M3): stereofit.solve's problem and contract, solved by Levenberg-Marquardt with an exact
Jacobian and the per-pair board poses eliminated by Schur complement.

Same model and parameters as stereofit (per camera fx fy cx cy k1..k4 [p1 p2]; the stereo pose C, camera-1 points into
camera 0; one board pose per stereo pair; per pair a log board scale when scale=True), the same residuals (pixels,
1e3 for a non-finite component or a corner at z <= 1e-6, with a zero Jacobian there as finite differences see it) and
the same trim rounds. Only the minimiser differs: scipy's TRF with a finite-difference Jacobian is replaced by

- the exact Jacobian: KB4/KB4T derivatives in closed form (project_jac); any other lens model lensmodel knows (RADTAN
  registered by calibtool.livecfg) only with the camera fixed, its 2x3 point derivative by central differences;
- local updates: C <- C [Exp(dr) | dt] (stereofit's C0 @ [rotmat(dC[:3]) | dC[3:]] at dC = 0), board R <- Exp(dr) R,
  t <- t + dt, log scale += d;
- Marquardt damping on the Jacobi-scaled normal equations (the x_scale="jac" equivalent) with Nielsen's update;
- the 6 (7 with scale) numbers of each pair eliminated per iteration: the linear system is the global block only
  (2 x 10 intrinsics + 6 stereo pose = 26 for KB4T). With the camera fixed there is no global block: the pairs are
  independent and solved as a vector of small LMs, each with its own damping and stop.

A solve "succeeds" as scipy's does: a convergence test met before max_nfev residual evaluations (one evaluation per
trial step). The tests are tighter than scipy's defaults (relative cost decrease < 1e-12 on a good step, scaled
gradient or step at rounding level), so a converged fast solve sits on the minimum scipy approaches.

Switched on by CALIB_SOLVER=fast (stereofit.solve and lensscore.score read it at each call); default scipy."""
from __future__ import annotations

import numpy as np

from calibtool import paths  # noqa: F401  (puts scripts/ on sys.path)
import lensmodel  # noqa: E402

ENV = "CALIB_SOLVER"
KB_MODELS = ("KB4", "KB4T")
TRIM_PX, TRIM_ROUNDS = 3.0, 4                      # stereofit.TRIM_PX, TRIM_ROUNDS
FTOL = 1e-12                                       # relative cost decrease on a good step
GTOL = 1e-10                                       # scaled gradient (px), relative to the residual norm
XTOL = 1e-12                                       # scaled step (px), relative to the residual norm
LAM0, LAM_MAX = 1e-3, 1e16
FD_REL = 1e-6                                      # central-difference step of the point derivative (relative)


# ---------------------------------------------------------------- projection with derivatives


def project_jac(model, intr, P):
    """Camera-frame points (..., 3) -> (uv (..., 2), d uv / d P (..., 2, 3), d uv / d intr (..., 2, ni) or None).
    KB4/KB4T: closed form, uv computed as lensmodel.project computes it. Other models (registered in lensmodel): uv by
    lensmodel.project, d uv / d P by central differences, no intrinsic derivative."""
    P = np.asarray(P, float)
    if model not in KB_MODELS:
        return _project_jac_generic(model, intr, P)
    intr = np.asarray(intr, float)
    fx, fy, cx, cy = intr[..., 0], intr[..., 1], intr[..., 2], intr[..., 3]
    k1, k2, k3, k4 = intr[..., 4], intr[..., 5], intr[..., 6], intr[..., 7]
    X, Y, Z = P[..., 0], P[..., 1], P[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        rho2 = X * X + Y * Y
        rho = np.hypot(X, Y)
        th = np.arctan2(rho, Z)
        t2 = th * th
        thd = th * (1 + k1 * t2 + k2 * t2 ** 2 + k3 * t2 ** 3 + k4 * t2 ** 4)
        s = np.where(rho > 1e-12, thd / np.maximum(rho, 1e-12), 1.0 / Z)
        mx, my = s * X, s * Y
        r2t = rho2 + Z * Z
        dthd = 1 + 3 * k1 * t2 + 5 * k2 * t2 ** 2 + 7 * k3 * t2 ** 3 + 9 * k4 * t2 ** 4
        on = rho > 1e-12
        rs = np.where(on, rho, 1.0)
        a = np.where(on, dthd * Z / (rs * rs * r2t) - thd / (rs * rs * rs), 0.0)      # ds/dX = a X, ds/dY = a Y
        sZ = -dthd / r2t                                                              # ds/dZ * rho / rho
        dm = np.empty(P.shape[:-1] + (2, 3))
        dm[..., 0, 0], dm[..., 0, 1], dm[..., 0, 2] = s + a * X * X, a * X * Y, X * sZ
        dm[..., 1, 0], dm[..., 1, 1], dm[..., 1, 2] = a * X * Y, s + a * Y * Y, Y * sZ
        # d thd / d k_i = th t2^i; mx = (X / rho) thd
        ux, uy = np.where(on, X / rs, 0.0), np.where(on, Y / rs, 0.0)
        dk = np.stack([th * t2, th * t2 ** 2, th * t2 ** 3, th * t2 ** 4], axis=-1)    # (..., 4)
        dmk = np.stack([ux[..., None] * dk, uy[..., None] * dk], axis=-2)              # (..., 2, 4)
    ni = intr.shape[-1]
    if model == "KB4T":
        p1, p2 = intr[..., 8], intr[..., 9]
        r2 = mx * mx + my * my
        mxt = mx + 2 * p1 * mx * my + p2 * (r2 + 2 * mx * mx)
        myt = my + p1 * (r2 + 2 * my * my) + 2 * p2 * mx * my
        Jt = np.empty(P.shape[:-1] + (2, 2))
        Jt[..., 0, 0] = 1 + 2 * p1 * my + 6 * p2 * mx
        Jt[..., 0, 1] = Jt[..., 1, 0] = 2 * p1 * mx + 2 * p2 * my
        Jt[..., 1, 1] = 1 + 6 * p1 * my + 2 * p2 * mx
        dm = Jt @ dm
        dmk = Jt @ dmk
        dmp = np.empty(P.shape[:-1] + (2, 2))
        dmp[..., 0, 0], dmp[..., 1, 0] = 2 * mx * my, r2 + 2 * my * my              # d / d p1
        dmp[..., 0, 1], dmp[..., 1, 1] = r2 + 2 * mx * mx, 2 * mx * my              # d / d p2
        mx, my = mxt, myt
    f = np.stack([fx * np.ones_like(X), fy * np.ones_like(X)], axis=-1)[..., None]    # (..., 2, 1)
    uv = np.stack([fx * mx + cx, fy * my + cy], axis=-1)
    dI = np.zeros(P.shape[:-1] + (2, ni))
    dI[..., 0, 0], dI[..., 1, 1] = mx, my
    dI[..., 0, 2] = dI[..., 1, 3] = 1.0
    dI[..., 4:8] = f * dmk
    if model == "KB4T":
        dI[..., 8:10] = f * dmp
    return uv, f * dm, dI


def _project_jac_generic(model, intr, P):
    uv = lensmodel.project(model, intr, P.reshape(-1, 3)).reshape(P.shape[:-1] + (2,))
    dP = np.empty(P.shape[:-1] + (2, 3))
    h = FD_REL * np.maximum(np.linalg.norm(P, axis=-1), 1e-9)
    for j in range(3):
        e = np.zeros_like(P)
        e[..., j] = h
        up = lensmodel.project(model, intr, (P + e).reshape(-1, 3)).reshape(uv.shape)
        um = lensmodel.project(model, intr, (P - e).reshape(-1, 3)).reshape(uv.shape)
        dP[..., j] = (up - um) / (2 * h[..., None])
    return uv, dP, None
