"""The fast depth scorer (M3): lensscore.score's contract and steps on calibtool.solver.

The calibration held fixed, per stereo pair the board's pose, then pose and board scale (board scale - 1 = the stereo
depth error at the board). Steps exactly as lensscore.score: PnP start with the scored calibration (failed pairs left
out and counted), the pose solve, the pose-and-scale solve from its poses, each at stereofit.MAX_NFEV evaluations a
round and restarted once from its own result when it did not converge (lensscore._solve_fixed). With the camera fixed
the pairs are independent, so calibtool.solver solves them as one vector of 6x6 / 7x7 Levenberg-Marquardt problems.
Any lens model lensmodel knows works (a RADTAN live config: its point derivative by central differences).

Switched on by CALIB_SOLVER=fast in lensscore.score."""
from __future__ import annotations

import numpy as np

from calibtool import paths  # noqa: F401  (puts scripts/ on sys.path)
from calibtool import solver
import lensscore  # noqa: E402
import stereofit as sf  # noqa: E402


def _solver(model):
    """the solve to call: stereofit.solve when CALIB_SOLVER=fast routes it to calibtool.solver anyway (lensscore.score's
    hook), so callers and tests that patch stereofit.solve (step 66's NO RESULT guards) see every scoring solve on
    either path; calibtool.solver.solve when called directly"""
    return sf.solve if sf.fast_solver(model) is not None else solver.solve


def _solve_fixed(d, model, intr, C, poses, scale):
    """lensscore._solve_fixed on the fast solver"""
    solve = _solver(model)
    r = solve(d, model, intr, C, poses, free_intr=False, free_C=False, scale=scale, max_nfev=sf.MAX_NFEV)
    if not r["success"]:
        r = solve(d, model, intr, C, r["poses"], free_intr=False, free_C=False, scale=scale, max_nfev=sf.MAX_NFEV,
                  ls0=r["log_scale"])
    return r


def score(model, intr, C, data):
    """lensscore.score: -> (records {src, hold, scale, z, u, v} per pair, {hold: pixel rms with the camera fixed},
    {"pose_success", "scale_success", "pnp_failed_pairs"})"""
    poses, ok = sf.pnp_poses(data, model, intr, C)
    d = data.select(ok)
    r = _solve_fixed(d, model, intr, C, poses[ok], scale=False)
    s = _solve_fixed(d, model, intr, C, r["poses"], scale=True)
    scale = np.exp(s["log_scale"])
    R = sf.rotmat(s["poses"][:, :3])
    n = np.bincount(d.pair, minlength=d.n)
    centre = np.stack([np.bincount(d.pair, weights=d.P[:, j], minlength=d.n) for j in range(3)], -1) / n[:, None]
    pc = np.einsum("pij,pj->pi", R, centre * scale[:, None]) + s["poses"][:, 3:]       # as sweep_590.evaluate_pp
    m0 = d.cam == 0
    n0 = np.bincount(d.pair[m0], minlength=d.n)
    uv = np.stack([np.bincount(d.pair[m0], weights=d.uv[m0, j], minlength=d.n) for j in range(2)], -1) / n0[:, None]
    rec = {"src": [int(x) for x in d.src_pairs], "hold": [int(x) for x in d.hold], "scale": [float(x) for x in scale],
           "z": [float(x) for x in pc[:, 2]], "u": [float(x) for x in uv[:, 0]], "v": [float(x) for x in uv[:, 1]]}
    assert tuple(rec) == lensscore.REC
    return rec, sf.hold_rms(d, r["res"]), {"pose_success": bool(r["success"]), "scale_success": bool(s["success"]),
                                           "pnp_failed_pairs": int((~ok).sum())}
