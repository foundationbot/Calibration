#!/usr/bin/env python3
"""Step 65 — does the stereo calibration get METRIC SCALE right?

Step 60's row error is blind to scale: a baseline 1 % too long leaves every row
aligned and still puts every depth 1 % too far. The declared board size is what
sets that scale inside Kalibr, so it is checked here against the physical board.

How: triangulate each matched aprilgrid corner from the two cameras using ONLY
the stereo calibration (lens model + intrinsics + cam0->cam1 transform; the board
size does not enter), then compare the 3D distance between the same corner of
two tags in one row or column with |delta| x pitch. Same-corner pairs make the
check independent of the tag corner convention.

Compare against the pitch MEASURED with a steel rule (--pitch-m, or the target
file's pitch once rig `target_measured: true`). Then the ratio tests the declared
board size and the fit together:

    ratio = triangulated spacing / (|delta| x pitch)    1.000 = scale right
    depth error at distance z ~= (ratio - 1) x z

Gate: |median ratio - 1| < gates.scale_max_pct. Run it on step 10's held-out
pairs, or better on a separate still take at grasping distance (0.4-1.2 m).

Usage:  65_scale_check.py CAMCHAIN [--tree heldout2] [--pitch-m 0.1200] [--rig RIG]
"""
from __future__ import annotations

import argparse, glob, importlib, re, sys

import cv2
import numpy as np
import yaml

import rigcfg

MIN_TAGS = 4


def triangulate(lens, p0, p1, K0, D0, K1, D1, T10) -> np.ndarray:
    """N x 3 points in cam0's frame from matched pixel pairs (cam0 -> cam1 = T10)."""
    n0 = lens.undistort(np.asarray(p0, np.float64).reshape(-1, 1, 2), K0, D0).reshape(-1, 2)
    n1 = lens.undistort(np.asarray(p1, np.float64).reshape(-1, 1, 2), K1, D1).reshape(-1, 2)
    P0 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P1 = np.hstack([T10[:3, :3], T10[:3, 3].reshape(3, 1)])
    Xh = cv2.triangulatePoints(P0, P1, n0.T, n1.T)
    return (Xh[:3] / Xh[3]).T


def pitch_ratios(keys, X, board, pitch: float) -> list[float]:
    """measured / expected spacing for every pair sharing a corner index and a
    tag row or column."""
    pos = {key: x for key, x in zip(keys, np.asarray(X))}
    ks = list(pos)
    out = []
    for i, (ta, ka) in enumerate(ks):
        ra, ca = board.tag_rc(ta)
        for tb, kb in ks[i + 1:]:
            if kb != ka:
                continue
            rb, cb = board.tag_rc(tb)
            steps = abs(cb - ca) if rb == ra else abs(rb - ra) if cb == ca else 0
            if steps:
                out.append(float(np.linalg.norm(pos[(tb, kb)] - pos[(ta, ka)]) / (steps * pitch)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("camchain", help="STEREO camchain from step 40")
    ap.add_argument("--tree", default="heldout2",
                    help="dir with cam0/ and cam1/ PNGs named by stamp (default ./heldout2)")
    ap.add_argument("--pitch-m", type=float, default=None,
                    help="steel-rule tag pitch in metres (default: the target file's)")
    rigcfg.add_arg(ap)
    args = ap.parse_args()
    rig = rigcfg.load(args.rig)
    board = rig.board
    if board.kind != "aprilgrid":
        sys.exit("ERROR: step 65 needs an aprilgrid target (corners identified by tag id).")
    validate = importlib.import_module("60_validate_stereo")

    pitch = args.pitch_m if args.pitch_m else board.pitch
    if not args.pitch_m and not rig.target_measured:
        print("  ⚠ comparing against the target file's PLACEHOLDER pitch: this checks the fit "
              "against\n    the declared size, not the declared size against the board. "
              "Pass --pitch-m.")
    CC = yaml.safe_load(open(args.camchain))
    lens = rig.camchain_lens(CC, "cam0"); rig.camchain_lens(CC, "cam1")
    K0, D0 = validate.KD(CC["cam0"]); K1, D1 = validate.KD(CC["cam1"])
    T10 = np.array(CC["cam1"]["T_cn_cnm1"])

    f0s = sorted(glob.glob(f"{args.tree}/cam0/*.png"))
    t1 = {int(re.search(r"(\d+)\.png", f).group(1)): f for f in glob.glob(f"{args.tree}/cam1/*.png")}
    if not f0s or not t1:
        sys.exit(f"ERROR: no PNGs under {args.tree}/cam0 and /cam1")
    k1 = np.array(sorted(t1))
    ratios, depths, used = [], [], 0
    for f0 in f0s:
        t0 = int(re.search(r"(\d+)\.png", f0).group(1))
        j = k1[np.argmin(np.abs(k1 - t0))]
        if abs(j - t0) > 5_000_000:
            continue
        im0 = cv2.imread(f0, cv2.IMREAD_GRAYSCALE); im1 = cv2.imread(t1[j], cv2.IMREAD_GRAYSCALE)
        keys, p0, p1 = validate.match_tags(board.detect(im0), board.detect(im1))
        if len({tid for tid, _ in keys}) < MIN_TAGS:
            continue
        X = triangulate(lens, p0, p1, K0, D0, K1, D1, T10)
        ratios += pitch_ratios(keys, X, board, pitch)
        depths.append(float(np.median(X[:, 2])))
        used += 1
    if not ratios:
        sys.exit("ERROR: no pair with enough shared tags")
    r = np.array(ratios)
    med = float(np.median(r)); mad = float(1.4826 * np.median(np.abs(r - med)))
    print(f"  pairs used {used}, spacings compared {len(r)}, board depth "
          f"{min(depths):.2f}-{max(depths):.2f} m")
    print(f"  triangulated / steel-rule pitch: median {med:.5f}  (robust sd {mad:.5f})")
    print(f"  => scale error {100*(med-1):+.3f} %:  {1000*(med-1)*0.85:+.1f} mm of depth at "
          f"0.85 m, {1000*(med-1)*0.5:+.1f} mm at 0.5 m")
    gate = float(rig.gates.get("scale_max_pct", 0.5))
    if abs(med - 1) * 100 < gate:
        print(f"\n  VERDICT: PASS (|scale error| < {gate:g} %)")
    else:
        print(f"\n  VERDICT: FAIL (|scale error| >= {gate:g} %). Check the declared tag size "
              "against the board, then refit.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
