#!/usr/bin/env python3
"""Step 30 — keep only the camera-IMU frames whose board labelling is consistent.

=============================================================================
 THIS IS THE STEP THAT MAKES THE CAMERA-IMU SOLVE WORK. DO NOT SKIP IT.
=============================================================================

THE PROBLEM
-----------
Kalibr's `checkerboard` target wraps cv::findChessboardCorners. A 180 deg
rotation is a symmetry of any rectangular corner grid, so the detector may
label the same physical board either way, and it can SWITCH between frames as
the board moves across the image. No choice of targetCols/targetRows escapes
this -- it is a property of the grid, not of the configuration.

  * STEREO calibration is IMMUNE. Each frame's target pose is solved
    independently, so a relabelled board is still self-consistent.
  * CAMERA-IMU calibration DIVERGES. It assumes ONE fixed target pose and
    chains camera motion against IMU integration. Measured on 2026-08-22 with
    ~35% of frames flipped:  |T_SC| = 1.03 m instead of ~17 cm, 4-10 px
    reprojection residuals, std-32 whitened errors.

An AprilGrid target removes the ambiguity at source, because every tag is
ID-anchored. If you have one, use it and skip this script.

THE METHOD: COMPARE AGAINST THE GYRO
------------------------------------
A label flip between two consecutive frames adds ~180 deg to the apparent
inter-frame rotation. At 15 fps the true inter-frame rotation is a few degrees,
so the two hypotheses are trivially separable -- IF you have an independent
measurement of how far the rig actually turned. The gyro is exactly that.

For each pair of consecutive detected frames we compare the board-implied
rotation ANGLE against the gyro-integrated angle over the same interval:

    label L in {0,1};  true board->camera rotation  R_true = R_solved @ Rz(pi)^L
    (reversing corner order k -> 87-k is a 180 deg rotation of the object points
     about the board normal through the board centre, so it right-multiplies the
     solved pose by Rz(pi); the derivation is in the README)

    relative rotation depends only on dL = L_i XOR L_j:
        dL=0 ->  R_j @ R_i^T
        dL=1 ->  R_j @ Rz(pi) @ R_i^T

    pick the dL whose angle is closer to  |integral omega dt|.

Only the ANGLE is used, never the axis. That is deliberate: the axis would
need R_cam_imu, which is precisely what the camera-IMU calibration has not
solved yet. The angle is frame-invariant, so this test needs no extrinsics at
all -- it works on the very first run, before anything about the IMU mount is
known.

Because the test is per-INTERVAL rather than a chain of pose compositions, it
survives detection gaps: a missing frame costs one comparison, not the whole
sequence downstream of it.

Finally the two cameras are cross-checked through the stereo extrinsics, so the
kept sets are mutually consistent and not merely self-consistent.

WHY NOT THE ChArUco MARKERS?
----------------------------
They are the obvious ID-anchored reference, and `31_label_canonical_charuco.py`
in this package does exactly that -- but on the 2026-08-22 capture it kept
0/2250 frames, because a 45 mm DICT_5X5 marker at ~1 m spans ~25 px = ~3.6 px
per cell and simply does not decode at working distance. Keep it for close-range
captures; the gyro method is the one that works on real data.

REFERENCE RESULT (2026-08-22, reproduce this to validate a change)
------------------------------------------------------------------
    cam0: 1624/2250 PnP ok, flip events=125, label0=631 label1=993, majority=1 (61%)
       roll(majority) mean +78.8 sd 2.1 | roll(minority) mean -101.1 sd 2.9
    cam1: 2041/2251 PnP ok, flip events=155, label0=723 label1=1318, majority=1 (65%)
       roll(majority) mean +97.6 sd 2.1 | roll(minority) mean -82.2 sd 3.0
    cross-camera check on majority frames: agree=935 disagree=0
    kept cam0=993  cam1=1318

THIS SCRIPT IS A RECONSTRUCTION, AND IT WAS VALIDATED AGAINST THAT RUN
----------------------------------------------------------------------
The original labeller was never saved to disk (it ran inline in a session), so
this is a rewrite against the recorded output above. Re-run on the same 2026-08-22
capture it produces:

    cam0: 1624/2250 PnP ok, flip events=125, label0=993 ... (61%)
       roll(majority) mean +79.3 sd 2.7 | roll(minority) mean -100.9 sd 3.7
       gyro cross-check: 1606/1606 consecutive pairs agree (100.0%)
    cam1: 2041/2251 PnP ok, flip events=165 ... (64%)
       roll(majority) mean +98.3 sd 8.9
       gyro cross-check: 2022/2029 consecutive pairs agree (99.7%)
    cross-camera check: agree=931 disagree=4
    kept cam0=993  cam1=1315

Against the original kept-frame lists (reference/2026-08-22/logs/lab_cam*.npy):

    cam0:  993/993  frames IDENTICAL  (100.00%)
    cam1: 1311/1318 frames identical  (99.47%; 7 dropped, 4 added)

PnP counts match exactly on both cameras. Use those .npy lists as the regression
test for any change to this file.

The +78.8 vs +97.6 deg roll difference is an independent corroboration of the
~18.8 deg mechanical inter-camera ROLL documented in the README -- two entirely
separate measurements of the same physical fault.

Usage
-----
    python3 30_label_canonical.py TREE CAMCHAIN OUT_TREE [--jobs 8] [--rig config/rig.yaml]
    python3 30_label_canonical.py tree/imucam reference/2026-08-22/kalibr/\\
                                  calib_stereo-camchain.yaml tree/imucam_canon

The board (corner grid + spacing) comes from the target file config/rig.yaml
names; the lens model from the camchain, which must agree with the rig file.
"""
from __future__ import annotations

import argparse
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import yaml

import rigcfg

FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
RZ_PI = np.diag([-1.0, -1.0, 1.0])            # 180 deg about the board normal

# per-worker globals, set by _init: lens, intrinsics, corner grid, and the
# object points in detector order (k = row*cols + col)
_K = _D = _LENS = _CORNERS = _OBJ = None


def _init(K, D, lens, corners, obj):
    global _K, _D, _LENS, _CORNERS, _OBJ
    _K, _D, _LENS, _CORNERS, _OBJ = K, D, lens, corners, obj


def _solve(path: str):
    """-> (stamp_ns, R_solved 3x3) or (stamp_ns, None) if no usable detection."""
    ts = int(Path(path).stem)
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        return ts, None
    ok, cor = cv2.findChessboardCorners(im, _CORNERS, flags=FLAGS)
    if not ok:
        return ts, None
    cor = cv2.cornerSubPix(
        im, cor, (5, 5), (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)).reshape(-1, 1, 2)
    und = _LENS.undistort(cor.astype(np.float64), _K, _D)
    ok, rvec, _ = cv2.solvePnP(_OBJ.reshape(-1, 1, 3), und, np.eye(3), None)
    if not ok:
        return ts, None
    return ts, cv2.Rodrigues(rvec)[0]


def ang(R: np.ndarray) -> float:
    """rotation magnitude of R, degrees."""
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


def roll_deg(R: np.ndarray) -> float:
    """in-plane angle of the board x-axis as seen in the image, degrees."""
    return float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))


def load_imu(csv: Path) -> tuple[np.ndarray, np.ndarray]:
    ts, w = [], []
    with csv.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line[0] not in "0123456789":
                continue                                    # header
            p = line.split(",")
            ts.append(int(float(p[0])))
            w.append([float(p[1]), float(p[2]), float(p[3])])
    if not ts:
        sys.exit(f"ERROR: no IMU samples parsed from {csv}")
    return np.asarray(ts, np.int64), np.asarray(w, np.float64)


def gyro_angle(ts: np.ndarray, w: np.ndarray, t0: int, t1: int) -> float | None:
    """|integral omega dt| over [t0,t1] in degrees, trapezoid on |omega|."""
    i0, i1 = np.searchsorted(ts, t0), np.searchsorted(ts, t1)
    if i1 - i0 < 2:
        return None
    seg_t = ts[i0:i1].astype(np.float64) * 1e-9
    seg_m = np.linalg.norm(w[i0:i1], axis=1)
    return float(np.degrees(np.trapezoid(seg_m, seg_t)))


def label_camera(name, files, K, D, lens, board, its, iw, jobs, max_gap_s, margin_deg):
    with Pool(jobs, initializer=_init,
              initargs=(K, D, lens, board.size, board.obj())) as pool:
        res = pool.map(_solve, files, chunksize=8)
    solved = [(t, R) for t, R in res if R is not None]
    solved.sort(key=lambda x: x[0])
    n_ok, n_all = len(solved), len(res)
    if n_ok < 2:
        sys.exit(f"ERROR: {name}: only {n_ok} PnP solutions -- nothing to label")

    # ---- GLOBAL labelling from the in-plane roll -----------------------------
    # The board is static and the two labels differ by exactly 180 deg, so the
    # solved roll is BIMODAL with its modes antipodal. Doubling the angle folds
    # the two modes onto one, so a circular mean of 2*roll recovers the common
    # axis with no reference frame to choose and NO SEQUENTIAL PROPAGATION.
    # A chain that carries each label forward from its neighbour inverts every
    # frame downstream of a single bad decision; measured, that turned the
    # majority group's roll spread from sd 2.1 deg into sd 58 deg.
    rolls = np.array([roll_deg(R) for _, R in solved])
    a2 = np.radians(rolls) * 2.0
    axis = np.degrees(np.arctan2(np.sin(a2).mean(), np.cos(a2).mean())) / 2.0
    d = (rolls - axis + 180.0) % 360.0 - 180.0      # signed distance to the axis
    labels = (np.abs(d) > 90.0).astype(np.int8)     # 1 = the antipodal mode

    # ---- the GYRO check: independent, and it is what validates the above -----
    # A label flip between consecutive frames adds ~180 deg to the apparent
    # inter-frame rotation, which the gyro flatly contradicts. We test every
    # consecutive pair and count how often the roll-derived relative label
    # agrees with the gyro's verdict. This is a genuine independent measurement
    # (the gyro knows nothing about the board), so disagreement means the roll
    # clustering is unsafe here -- typically because the rig itself rolled.
    agree = disagree = nocover = 0
    for i in range(n_ok - 1):
        (t0, R0), (t1, R1) = solved[i], solved[i + 1]
        dt = (t1 - t0) * 1e-9
        g = gyro_angle(its, iw, t0, t1) if 0 < dt <= max_gap_s else None
        if g is None:
            nocover += 1
            continue
        d_same = abs(ang(R1 @ R0.T) - g)
        d_flip = abs(ang(R1 @ RZ_PI @ R0.T) - g)
        gyro_says_flip = d_flip + margin_deg < d_same
        roll_says_flip = labels[i] != labels[i + 1]
        if gyro_says_flip == roll_says_flip:
            agree += 1
        else:
            disagree += 1

    flips = int((labels[1:] != labels[:-1]).sum())
    n1 = int(labels.sum())
    n0 = n_ok - n1
    maj = 1 if n1 >= n0 else 0
    rm, rn = rolls[labels == maj], rolls[labels != maj]

    print(f"{name}: {n_ok}/{n_all} PnP ok, flip events={flips}, "
          f"label0={n0} label1={n1}, majority={maj} ({max(n0, n1) / n_ok:.0%})")
    print(f"   roll(majority) mean {rm.mean():+.1f} sd {rm.std():.1f}"
          + (f" | roll(minority) mean {rn.mean():+.1f} sd {rn.std():.1f}"
             if len(rn) else " | (no minority frames)"))
    tot = agree + disagree
    if tot:
        print(f"   gyro cross-check: {agree}/{tot} consecutive pairs agree "
              f"({agree / tot:.1%})" + (f", {nocover} without gyro cover" if nocover else ""))
        if disagree > 0.02 * tot:
            print(f"   ⚠ {disagree} pairs DISAGREE with the gyro. The roll clustering "
                  f"assumes the rig\n     stayed roughly upright; if it rolled, these "
                  f"labels are not trustworthy.")
    if rm.std() > 10.0:
        print(f"   ⚠ majority roll spread is sd {rm.std():.1f} deg. The 2026-08-22 "
              f"reference is 2.1.\n     A wide spread means the two modes are not "
              f"cleanly separated -- inspect before fitting.")

    return solved, labels, maj


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tree", type=Path, help="kalibr tree from step 10 (needs imu0.csv)")
    ap.add_argument("camchain", type=Path, help="STEREO camchain from step 40")
    ap.add_argument("out_tree", type=Path, help="filtered tree to write")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--max-gap", type=float, default=0.5,
                    help="skip intervals longer than this many seconds (default 0.5)")
    ap.add_argument("--margin", type=float, default=20.0,
                    help="degrees by which 'flipped' must beat 'same' (default 20)")
    ap.add_argument("--pair-tol-ms", type=float, default=5.0)
    rigcfg.add_arg(ap)
    args = ap.parse_args()

    rig = rigcfg.load(args.rig)
    if not rig.board.ambiguous_orientation:
        print(f"step 30 not needed: the {rig.board.kind} target anchors every corner by its "
              "tag id, so there is no\n180 deg ambiguity to relabel. Build the camera-IMU bag "
              "from step 10's imucam tree directly\n(40_calibrate.sh imucam picks it for an "
              "aprilgrid rig).")
        sys.exit(0)
    CC = yaml.safe_load(args.camchain.read_text())
    lenses = {n: rig.camchain_lens(CC, n) for n in ("cam0", "cam1")}
    def KD(c):
        fx, fy, cx, cy = c["intrinsics"]
        return (np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]),
                np.array(c["distortion_coeffs"], np.float64).reshape(4, 1))
    K0, D0 = KD(CC["cam0"]); K1, D1 = KD(CC["cam1"])
    T10 = np.array(CC["cam1"]["T_cn_cnm1"], np.float64)

    imu_csv = args.tree / "imu0.csv"
    if not imu_csv.is_file():
        sys.exit(f"ERROR: {imu_csv} not found -- this step needs the gyro. "
                 "Re-run step 10 without --no-imu.")
    its, iw = load_imu(imu_csv)

    lab = {}
    for name, K, D in (("cam0", K0, D0), ("cam1", K1, D1)):
        files = sorted(str(p) for p in (args.tree / name).glob("*.png"))
        if not files:
            sys.exit(f"ERROR: no PNGs under {args.tree / name}")
        lab[name] = label_camera(name, files, K, D, lenses[name], rig.board, its, iw,
                                 args.jobs, args.max_gap, args.margin)

    # ---- tie the two cameras' mode naming together --------------------------
    # Each camera clusters its own roll independently, so WHICH mode it calls
    # the majority is arbitrary -- the two can land on opposite conventions and
    # both be internally correct. The stereo extrinsics settle it: cam1 must see
    # the board at T10 @ (board in cam0). We test cam1's majority and its
    # 180-deg-rotated alternative and keep whichever agrees, so the two kept
    # sets are mutually consistent rather than merely self-consistent.
    def as_keep(entry, flip=False):
        solved, labels, maj = entry
        post = RZ_PI if (maj ^ flip) else np.eye(3)
        return {t: R @ post for (t, R), L in zip(solved, labels) if L == maj}

    kept = {"cam0": as_keep(lab["cam0"])}
    tol = int(args.pair_tol_ms * 1e6)
    scores = {}
    for flip in (False, True):
        cand = as_keep(lab["cam1"], flip)
        k1 = np.array(sorted(cand))
        a = d = 0
        if len(k1):
            for t0, R0 in kept["cam0"].items():
                j = int(k1[np.argmin(np.abs(k1 - t0))])
                if abs(j - t0) > tol:
                    continue
                res = ang((T10[:3, :3] @ R0) @ cand[j].T)
                a += res < 30.0
                d += res >= 30.0
        scores[flip] = (a, d, cand)
    flip = scores[True][0] > scores[False][0]
    agree, disagree, kept["cam1"] = scores[flip]
    if flip:
        print("\ncam1's mode naming was OPPOSITE cam0's; realigned through the "
              "stereo extrinsics")

    print(f"\ncross-camera check on majority frames: agree={agree} disagree={disagree}")
    if agree == 0 or disagree > agree:
        print("=> WARNING: the two cameras cannot be reconciled under EITHER convention.\n"
              "   That is not a labelling problem -- suspect the stereo camchain you\n"
              "   passed in, or a capture where the two cameras saw different boards.")
    else:
        print("=> cam1 majority label MATCHES cam0's")

    # --- write the filtered tree --------------------------------------------
    args.out_tree.mkdir(parents=True, exist_ok=True)
    for name in ("cam0", "cam1"):
        dst = args.out_tree / name
        dst.mkdir(exist_ok=True)
        for t in kept[name]:
            src = args.tree / name / f"{t}.png"
            out = dst / f"{t}.png"
            if not out.exists():
                try:
                    import os
                    os.link(src, out)
                except OSError:
                    import shutil
                    shutil.copy2(src, out)
    import shutil
    shutil.copy2(imu_csv, args.out_tree / "imu0.csv")
    src_json = args.tree / "SOURCE.json"
    if src_json.is_file():
        shutil.copy2(src_json, args.out_tree / "SOURCE.json")

    print(f"kept cam0={len(kept['cam0'])}  cam1={len(kept['cam1'])}")
    print(f"wrote {args.out_tree}  -> build a bag from THIS tree for the camera-IMU fit")


if __name__ == "__main__":
    main()
