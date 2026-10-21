#!/usr/bin/env python3
"""One stereo calibration's depth reading of another's scene, pixel by pixel (step 46's repeatability map; the
acceptance test's new-vs-Fit-A map).

For a camera-0 pixel (u, v): the point at depth z (m) on calibration a's ray, its pixels in both cameras under a,
triangulated with calibration b; the reading is |X_b| / |P| - 1 (%): how far a depth image built with b would misread
a scene that a describes, at that pixel. Port of the 20 Oct A/B review (compare_ab.py part 2: scipy least_squares per
point on a 40 px grid within 600 px of phantom12's (640, 400)) as vectorised Gauss-Newton on the whole image grid,
rays from lensmodel.unproject. A grid point where a's camera 0 has no ray (its lens polynomial turns over inside the
image) is left out and counted (points_without_ray). Usage: python depthmap.py <cal-a.json[:key]> <cal-b.json[:key]>
[--z 1.0] [--rig PATH]"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lenscells  # noqa: E402
import lensmodel  # noqa: E402

RAY_TOL_PX = 1e-3                    # a grid point's ray must project back to it within this (as step 71's)


def grid(resolution, step=40):
    """camera-0 pixels (N, 2) on a step-px grid over the image, half a step in from its edges"""
    W, H = resolution
    return np.array([(u, v) for v in np.arange(step / 2, H, step) for u in np.arange(step / 2, W, step)], float)


def rays(model, intr, uv):
    """-> (unproject's rays (N, 3), has_ray (N,)): a pixel has a ray only if the ray projects back to it within
    RAY_TOL_PX. A lens whose KB4 polynomial turns over inside the image has none beyond the turn, where unproject
    returns NaN or, more often, a finite wrong ray (Fit A's camera 0 with k1 = -0.2: 145 of 640 grid points, where
    the calibration's reading of its own scene, 0 wherever a ray exists, hit a singular matrix or read up to 4e12 %)"""
    with np.errstate(all="ignore"):
        ray = lensmodel.unproject(model, intr, uv)
        back = np.linalg.norm(lensmodel.project(model, intr, ray) - uv, axis=1)
    return ray, np.isfinite(ray).all(1) & np.isfinite(back) & (back <= RAY_TOL_PX)


def points_without_ray(cal, resolution, step=40):
    """how many grid points depth_diff_map(cal, ..., step) leaves out because cal's camera 0 has no ray there"""
    model, intr, _ = cal
    return int((~rays(model, intr[0], grid(resolution, step))[1]).sum())


def triangulate(model, intr, C, uv0, uv1, iters=5):
    """camera-0 points from pixel pairs (C: camera-1 points into camera 0): the rays' midpoint, then Gauss-Newton
    on both cameras' pixel residuals (compare_ab.py's least squares, vectorised)"""
    r0 = lensmodel.unproject(model, intr[0], uv0)
    r1 = lensmodel.unproject(model, intr[1], uv1) @ C[:3, :3].T
    o1 = C[:3, 3]
    a, b, c = (r0 * r0).sum(1), (r0 * r1).sum(1), (r1 * r1).sum(1)
    d, e = -(r0 @ o1), -(r1 @ o1)
    den = a * c - b * b
    s, t = (b * e - c * d) / den, (a * e - b * d) / den
    X = 0.5 * (r0 * s[:, None] + o1 + r1 * t[:, None])
    Ci = np.linalg.inv(C)

    def res(X):
        return np.c_[lensmodel.project(model, intr[0], X) - uv0,
                     lensmodel.project(model, intr[1], X @ Ci[:3, :3].T + Ci[:3, 3]) - uv1]

    h = 1e-6
    for _ in range(iters):
        r = res(X)
        J = np.stack([(res(X + h * np.eye(3)[j]) - res(X - h * np.eye(3)[j])) / (2 * h) for j in range(3)], axis=2)
        JT = np.transpose(J, (0, 2, 1))
        # pinv, not solve: near a lens's turn the projection is flat along the ray and JᵀJ singular for that point
        # alone (solve raised LinAlgError for the whole map); such a point comes out wrong, not as an exception
        X = X - (np.linalg.pinv(JT @ J) @ (JT @ r[:, :, None]))[:, :, 0]
    return X


def depth_diff_map(cal_a, cal_b, resolution, z=1.0, step=40):
    """cal = (model, [intr0, intr1], C). -> (uv (N, 2) camera-0 pixels on a step-px grid, pct (N,) b's depth reading
    of a's scene at depth z, %), keeping only pixels where a's camera 0 has a ray (rays(); points_without_ray counts
    the others) and whose point lands inside camera 1's image under a (where a depth image exists). Where b cannot
    read the point (a's camera 1 has no ray at the point's pixel there, or either of b's cameras has none at its
    pixel), pct is +inf: a cell holding such a point is beyond every gate (step 46 flags it), never NaN, which none
    catches"""
    (ma, ia, Ca), (mb, ib, Cb) = cal_a, cal_b
    W, H = resolution
    g = grid(resolution, step)
    ray, has_ray = rays(ma, ia[0], g)
    front = has_ray & (ray[:, 2] > 0.05)
    g, ray = g[front], ray[front]
    P = ray * (z / ray[:, 2:3])
    Ci = np.linalg.inv(Ca)
    uv1 = lensmodel.project(ma, ia[1], P @ Ci[:3, :3].T + Ci[:3, 3])
    inside = (uv1[:, 0] >= 0) & (uv1[:, 0] <= W - 1) & (uv1[:, 1] >= 0) & (uv1[:, 1] <= H - 1)
    g, P, uv1 = g[inside], P[inside], uv1[inside]
    readable = rays(ma, ia[1], uv1)[1] & rays(mb, ib[0], g)[1] & rays(mb, ib[1], uv1)[1]
    pct = np.full(len(g), np.inf)
    X = triangulate(mb, ib, Cb, g[readable], uv1[readable])
    pct[readable] = 100 * (np.linalg.norm(X, axis=1) / np.linalg.norm(P[readable], axis=1) - 1)
    pct[~np.isfinite(pct)] = np.inf
    return g, pct


def by_cell(uv, pct, pp):
    """{cell: {"max_abs_pct", "points"}}, cells relative to the principal point pp (lenscells)"""
    lab = lenscells.cell_of(uv[:, 0] - pp[0], uv[:, 1] - pp[1])
    return {c: {"max_abs_pct": float(np.abs(pct[lab == c]).max()), "points": int((lab == c).sum())}
            for c in lenscells.CELLS if (lab == c).any()}


def limit_radius(uv, pct, pp, max_r):
    """the grid points (and their differences) within max_r px of the principal point pp"""
    keep = np.hypot(uv[:, 0] - pp[0], uv[:, 1] - pp[1]) <= max_r
    return uv[keep], pct[keep]


def main(argv=None):
    import rigcfg
    import stereofit as sf
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cal_a")
    ap.add_argument("cal_b")
    ap.add_argument("--z", type=float, default=1.0)
    ap.add_argument("--max-radius-px", type=float, default=None,
                    help="only grid points within this distance of camera 0's principal point")
    rigcfg.add_arg(ap)
    a = ap.parse_args(argv)
    rig = rigcfg.load(a.rig)
    ca, cb = sf.load_calibration(a.cal_a), sf.load_calibration(a.cal_b)
    uv, pct = depth_diff_map(ca, cb, rig.resolution, a.z)
    no_ray = points_without_ray(ca, rig.resolution)
    if a.max_radius_px is not None:
        uv, pct = limit_radius(uv, pct, ca[1][0][2:4], a.max_radius_px)
    within = "" if a.max_radius_px is None else f" (points within {a.max_radius_px:g} px of the principal point)"
    print(f"{a.cal_b} reading {a.cal_a}'s scene at {a.z} m depth, largest |difference| per image cell{within}:")
    if no_ray:
        print(f"({no_ray} of {len(grid(rig.resolution))} grid points left out: {a.cal_a}'s camera 0 has no ray there, "
              "its lens polynomial turns over inside the image)")
    print("| cell | largest difference (%) | grid points |")
    print("|---|---|---|")
    for c, v in by_cell(uv, pct, ca[1][0][2:4]).items():
        print(f"| {c} | {v['max_abs_pct']:.2f} | {v['points']} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
