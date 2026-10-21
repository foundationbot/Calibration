#!/usr/bin/env python3
"""Keep only imucam frames whose plain-checkerboard corner ordering matches the
ArUco-ID-anchored canonical ordering.

Kalibr's checkerboard target wraps cv::findChessboardCorners, so it inherits the
180-degree ordering ambiguity of a rectangular corner grid. Frames where the
ordering is flipped imply a target pose rotated 180 deg, which is irreconcilable
with a single fixed target -> the camera-IMU solve diverges (we measured a 1.03 m
lever arm and std-32 px residuals).

The ChArUco markers give an absolute orientation reference. For each frame we
match the checkerboard corners against the ID-anchored charuco corners and
decide identity (canonical) vs reversed (k <-> N-1-k). Only canonical frames are
kept, so both cameras are mutually consistent by construction.

The corner grid comes from the target file and the ChArUco layout from the
`charuco` block of config/rig.yaml.

Usage:  31_label_canonical_charuco.py SRC_TREE DST_TREE [--jobs 8] [--rig RIG]
"""
import argparse, cv2, glob, numpy as np, os, sys
from collections import Counter
from multiprocessing import Pool

import rigcfg

_CORNERS = _N = _BOARD = None                    # per-worker, set by _init


def _init(corners, charuco):
    global _CORNERS, _N, _BOARD
    _CORNERS, _N = corners, corners[0] * corners[1]
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, charuco["dictionary"]))
    _BOARD = cv2.aruco.CharucoBoard(tuple(charuco["squares"]), float(charuco["square_m"]),
                                    float(charuco["marker_m"]), d)


def classify(f):
    im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    if im is None: return (f, "unreadable")
    ok, cor = cv2.findChessboardCorners(im, _CORNERS,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH|cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok: return (f, "no_checkerboard")
    cor = cor.reshape(-1,2)
    det = cv2.aruco.CharucoDetector(_BOARD)
    cc, ci, _, _ = det.detectBoard(im)
    if ci is None or len(ci) < 6: return (f, "no_charuco")
    cc = cc.reshape(-1,2); ci = ci.reshape(-1)
    # nearest checkerboard corner index for each charuco corner
    idn, idr = 0, 0
    for pt, cid in zip(cc, ci):
        k = int(np.argmin(((cor-pt)**2).sum(axis=1)))
        if k == int(cid): idn += 1
        if k == _N-1-int(cid): idr += 1
    if idn > idr and idn >= max(3, 0.6*len(ci)):  return (f, "canonical")
    if idr > idn and idr >= max(3, 0.6*len(ci)):  return (f, "flipped")
    return (f, "ambiguous")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--jobs", type=int, default=int(os.environ.get("NPROC", "8")))
    rigcfg.add_arg(ap)
    a = ap.parse_args()
    rig = rigcfg.load(a.rig)
    if not rig.charuco:
        sys.exit(f"ERROR: {rig.path} has no 'charuco' block")
    for cam in ("cam0","cam1"):
        fs = sorted(glob.glob(f"{a.src}/{cam}/*.png"))
        if not fs:
            sys.exit(f"ERROR: no PNGs under {a.src}/{cam}")
        with Pool(a.jobs, initializer=_init, initargs=(rig.board.size, rig.charuco)) as p:
            res = p.map(classify, fs, chunksize=8)
        cnt = Counter(v for _,v in res)
        keep = [f for f,v in res if v=="canonical"]
        d = f"{a.dst}/{cam}"; os.makedirs(d, exist_ok=True)
        for f in keep:
            o = f"{d}/{os.path.basename(f)}"
            if not os.path.exists(o): os.link(f,o)
        print(f"{cam}: {dict(cnt)}")
        print(f"  -> kept {len(keep)}/{len(fs)} canonical ({100*len(keep)/len(fs):.0f}%)")


if __name__ == "__main__":
    main()
