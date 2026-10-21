#!/usr/bin/env python3
"""Step 20 — measure target COVERAGE before you spend an hour on a fit.

WHY THIS EXISTS
---------------
On 2026-08-22 the first calibration attempt was run on a capture where the board
was kept visible to BOTH cameras at all times. That confines it to the stereo
overlap region, and for a 129 deg x 81 deg fisheye pair that leaves 28-32 deg of
each field never observed. The fit still "succeeded":

    frame coverage      36% / 32%
    distortion coeffs   ~30x too large, and SIGN-FLIPPED between two
                        physically identical lenses
    reproj sigma        0.47 px          <- looks fine!
    held-out epipolar   2423 px          <- catastrophic

The reprojection error did not notice, because a distortion model can fit the
observed region beautifully while diverging everywhere else. Only held-out
epipolar error caught it. The fix was to sweep each camera's FULL frame
separately and only then add a short shared segment for the baseline:

    frame coverage      68%
    periphery sampled   63-65 deg off-axis

So: run this BEFORE kalibr, on the tree from step 10. It costs ~1 min and can
save an hour plus a wrong calibration shipped to a robot.

WHAT IT MEASURES, per camera
----------------------------
  detection rate   fraction of frames where the board was found: the full
                   corner grid for a checkerboard, at least MIN_TAGS tags for an
                   aprilgrid (target file named in config/rig.yaml)
  grid coverage    of a 16x10 cell grid over the image, the fraction of cells
                   that ever contained a detected corner  (>= 60% wanted)
  radial reach     the largest off-axis angle any corner reached, using the
                   NOMINAL lens model with f = cameras.nominal_focal_px
                   (equidistant r = f*theta; radtan r = f*tan(theta)). For the
                   mk13 head: 1280 px over 129 deg -> f ~ 568 px. This is a
                   pre-calibration sanity figure, not a calibrated one.
                   (>= gates.reach_min_deg wanted; 60 deg for the mk13 fisheye)
  a heatmap PNG    so you can SEE which part of the field was never visited

Thresholds come from config/rig.yaml `gates` and are printed with the mk13
reference numbers beside them. Exit status is 1 if any camera fails, so it can
gate a script.

Usage
-----
    python3 20_check_coverage.py TREE [--grid 16x10] [--nominal-focal 568]
                                      [--limit 0] [--jobs 8] [--no-heatmap]
                                      [--rig config/rig.yaml]
    python3 20_check_coverage.py tree/stereo
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

import rigcfg

FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
MIN_TAGS = 4                                   # an aprilgrid view counts from 4 tags

_BOARD = None                                  # per-worker, set by _init


def _init(board):
    global _BOARD
    _BOARD = board


def corners_from_image(board, gray):
    """Detected corner pixels (N x 2), or None if the view does not count.

    checkerboard: the full internal grid or nothing.
    aprilgrid:    any view with >= MIN_TAGS tags; partial views are kept because
                  tags near the image edge are what constrain distortion.
    """
    if board.kind == "aprilgrid":
        found = board.detect(gray)
        if len({tid for tid, _ in found}) < MIN_TAGS:
            return None
        return np.array(list(found.values()), np.float32)
    ok, cor = cv2.findChessboardCorners(gray, board.size, flags=FLAGS)
    return cor.reshape(-1, 2).astype(np.float32) if ok else None


def detect(path: str):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None, None
    return corners_from_image(_BOARD, im), im.shape[:2]


def heatmap(pts: np.ndarray, hw: tuple[int, int], gw: int, gh: int, out: Path) -> None:
    h, w = hw
    grid = np.zeros((gh, gw), np.float32)
    gx = np.clip((pts[:, 0] / w * gw).astype(int), 0, gw - 1)
    gy = np.clip((pts[:, 1] / h * gh).astype(int), 0, gh - 1)
    np.add.at(grid, (gy, gx), 1.0)
    vis = grid / max(grid.max(), 1.0)
    vis = cv2.resize((vis * 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    vis = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
    vis[cv2.resize((grid == 0).astype(np.uint8), (w, h),
                   interpolation=cv2.INTER_NEAREST) > 0] = (40, 40, 40)  # never visited
    for i in range(1, gw):
        cv2.line(vis, (i * w // gw, 0), (i * w // gw, h), (90, 90, 90), 1)
    for i in range(1, gh):
        cv2.line(vis, (0, i * h // gh), (w, i * h // gh), (90, 90, 90), 1)
    cv2.imwrite(str(out), vis)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tree", type=Path)
    ap.add_argument("--grid", default=None, help="cells, WxH (default: rig gates.coverage_grid)")
    ap.add_argument("--nominal-focal", type=float, default=None,
                    help="px (default: rig cameras.nominal_focal_px)")
    ap.add_argument("--limit", type=int, default=0, help="sample at most N frames (0=all)")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--no-heatmap", action="store_true")
    rigcfg.add_arg(ap)
    args = ap.parse_args()

    rig = rigcfg.load(args.rig)
    WANT_GRID = float(rig.gates["coverage_min"])
    WANT_DEG = float(rig.gates["reach_min_deg"])
    if args.nominal_focal is None:
        args.nominal_focal = rig.nominal_focal_px
    if args.grid:
        gw, gh = (int(v) for v in args.grid.lower().split("x"))
    else:
        gw, gh = (int(v) for v in rig.gates["coverage_grid"])
    cams = sorted(d for d in args.tree.iterdir() if d.is_dir() and d.name.startswith("cam"))
    if not cams:
        sys.exit(f"ERROR: no cam*/ dirs under {args.tree}")

    print(f"coverage check on {args.tree}   grid {gw}x{gh}, "
          f"nominal f={args.nominal_focal:.0f} px\n")
    failed = False
    for cam in cams:
        files = sorted(str(p) for p in cam.glob("*.png"))
        if args.limit and len(files) > args.limit:
            files = files[:: max(1, len(files) // args.limit)]
        if not files:
            print(f"{cam.name}: no PNGs"); failed = True; continue

        with Pool(args.jobs, initializer=_init, initargs=(rig.board,)) as pool:
            res = pool.map(detect, files, chunksize=8)

        hw = next((s for _, s in res if s is not None), None)
        found = [c for c, _ in res if c is not None]
        rate = len(found) / len(res)
        if not found:
            print(f"{cam.name}: 0/{len(res)} detections -- wrong board size, or "
                  f"the target is not in frame"); failed = True; continue

        pts = np.vstack(found)
        h, w = hw
        cells = {(int(min(x / w * gw, gw - 1)), int(min(y / h * gh, gh - 1)))
                 for x, y in pts}
        grid_cov = len(cells) / (gw * gh)
        r = np.hypot(pts[:, 0] - w / 2.0, pts[:, 1] - h / 2.0)
        deg = np.degrees(rig.lens.off_axis_rad(r, args.nominal_focal))

        ok_g = grid_cov >= WANT_GRID
        ok_d = deg.max() >= WANT_DEG
        failed |= not (ok_g and ok_d)
        print(f"{cam.name}:  {len(found)}/{len(res)} frames detected ({rate:6.1%})   "
              f"{len(pts)} corners")
        print(f"   grid coverage {grid_cov:6.1%}  {'OK ' if ok_g else 'LOW'}"
              f"  (want >= {WANT_GRID:.0%}; 2026-08-22 good run 68%, bad run 36%/32%)")
        print(f"   radial reach  {deg.max():5.1f} deg max, "
              f"{np.percentile(deg, 99.5):5.1f} deg p99.5  {'OK ' if ok_d else 'LOW'}"
              f"  (want >= {WANT_DEG:.0f} max; good run 63-65)")
        if not args.no_heatmap:
            out = args.tree / f"coverage_{cam.name}.png"
            heatmap(pts, hw, gw, gh, out)
            print(f"   heatmap -> {out}   (dark grey = never observed)")
        print()

    if failed:
        print("VERDICT: coverage is INSUFFICIENT. Re-capture before fitting.")
        print("  Sweep each camera's FULL frame separately -- corners, edges, near and")
        print("  far, in-plane and out-of-plane tilt. Keeping the board in BOTH cameras")
        print("  the whole time is exactly what caused the 2026-08-22 divergence.")
        sys.exit(1)
    print("VERDICT: coverage OK -- proceed to the stereo fit (step 40).")
    print("  After the fit, check the two lenses' distortion coefficients have the")
    print("  SAME SIGNS. Sign-flipped between two identical lenses is the residual")
    print("  signature of this failure even when coverage passes.")


if __name__ == "__main__":
    main()
