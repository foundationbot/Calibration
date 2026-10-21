#!/usr/bin/env python3
"""High-yield stereo validation on held-out frames.

The judge of a stereo calibration is the RECTIFIED EPIPOLAR ERROR |dy| on frames
the fit never saw: after rectification a corner must sit on the same row in both
images. Reprojection error cannot see an extrinsics fault (README, trap 1).

checkerboard  ChArUco marker decoding is distance-limited (45 mm marker at 1 m ~=
              25 px = 3.6 px per DICT_5X5 cell), so the ID-anchored validator only
              forms a handful of stereo pairs. The plain chessboard detector finds
              all 88 corners far more often, but its corner ORDER is 180-degree
              ambiguous -- correspondence k<->k is only valid if both cameras happen
              to be labelled the same way. So: solve each camera's board pose, test
              them against the calibrated stereo extrinsics, and if they disagree by
              ~180 deg, reverse cam1's corner order (k -> N-1-k) and re-test. Pairs
              that reconcile under neither ordering are dropped. A consistently-
              flipped PAIR is fine -- only disagreement breaks correspondence.
aprilgrid     every corner is identified by (tag id, corner), so the two cameras
              are matched directly on that key; no flip handling. A pair counts
              when both cameras see at least MIN_TAGS of the same tags.

Board, lens model, resolution, thresholds and the expected disparity SIGN come
from config/rig.yaml (the lens model and resolution are cross-checked against
the camchain). The sign check is what verifies the rig file's left/right claim.

Usage:  60_validate_stereo.py CAMCHAIN [--heldout heldout2] [--rig RIG]
"""
import argparse, cv2, glob, numpy as np, re, sys, yaml

import rigcfg

MIN_TAGS = 4
CB_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE


def KD(c):
    fx, fy, cx, cy = c["intrinsics"]
    return (np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]]),
            np.array(c["distortion_coeffs"]).reshape(4, 1))


def rectify(lens, K0, D0, K1, D1, T10, resolution):
    """-> (R0, R1, P0, P1) for the camchain's cam0 -> cam1 transform T10."""
    return lens.stereo_rectify(
        K0, D0, K1, D1, resolution,
        np.ascontiguousarray(T10[:3, :3]), np.ascontiguousarray(T10[:3, 3].reshape(3, 1)),
        flags=cv2.CALIB_ZERO_DISPARITY)


def match_tags(d0: dict, d1: dict):
    """Corners both cameras saw, paired on (tag id, corner) -> (keys, pts0, pts1)."""
    keys = sorted(set(d0) & set(d1))
    return (keys, np.array([d0[k] for k in keys], np.float64).reshape(-1, 2),
            np.array([d1[k] for k in keys], np.float64).reshape(-1, 2))


def rectified_rows(lens, pts0, pts1, K0, D0, K1, D1, rect):
    """|dy| and dx (u0 - u1) of matched corners after undistortion + rectification."""
    R0, R1, P0, P1 = rect
    u0 = lens.undistort(np.asarray(pts0, np.float64).reshape(-1, 1, 2), K0, D0, R=R0, P=P0)
    u1 = lens.undistort(np.asarray(pts1, np.float64).reshape(-1, 1, 2), K1, D1, R=R1, P=P1)
    return np.abs(u0[:, 0, 1] - u1[:, 0, 1]), u0[:, 0, 0] - u1[:, 0, 0]


def _read_grey(p, resolution):
    im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if im.shape[::-1] != resolution:
        sys.exit(f"ERROR: {p} is {im.shape[1]}x{im.shape[0]}, rig config says "
                 f"{resolution[0]}x{resolution[1]}")
    return im


def _detect_checkerboard(im, board):
    ok, c = cv2.findChessboardCorners(im, board.size, flags=CB_FLAGS)
    if not ok:
        return None
    return cv2.cornerSubPix(im, c, (5, 5), (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)).reshape(-1, 2)


def _pose(lens, c, obj, K, D):
    und = lens.undistort(c.reshape(-1, 1, 2), K, D)
    ok, rv, tv = cv2.solvePnP(obj.reshape(-1, 1, 3), und, np.eye(3), None)
    if not ok:
        return None, None
    return cv2.Rodrigues(rv)[0], tv.ravel()


def _rms_reproj(lens, c, obj, K, D):
    R, t = _pose(lens, c, obj, K, D)
    if R is None:
        return None
    pr, _ = lens.project(obj.reshape(-1, 1, 3), cv2.Rodrigues(R)[0], t.reshape(3, 1), K, D)
    return float(np.sqrt(((pr.reshape(-1, 2) - c) ** 2).sum(axis=1).mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("camchain", help="STEREO camchain from step 40")
    ap.add_argument("--heldout", default="heldout2",
                    help="held-out tree from step 10 (default: ./heldout2)")
    rigcfg.add_arg(ap)
    args = ap.parse_args()
    rig = rigcfg.load(args.rig)
    board = rig.board

    CC = yaml.safe_load(open(args.camchain))
    lens = rig.camchain_lens(CC, "cam0"); rig.camchain_lens(CC, "cam1")
    K0, D0 = KD(CC["cam0"]); K1, D1 = KD(CC["cam1"])
    T10 = np.array(CC["cam1"]["T_cn_cnm1"])
    rect = rectify(lens, K0, D0, K1, D1, T10, rig.resolution)

    f0s = sorted(glob.glob(f"{args.heldout}/cam0/*.png"))
    t1 = {int(re.search(r"(\d+)\.png", f).group(1)): f
          for f in glob.glob(f"{args.heldout}/cam1/*.png")}
    if not f0s or not t1:
        sys.exit(f"ERROR: no held-out PNGs under {args.heldout}/cam0 and /cam1")
    k1 = np.array(sorted(t1))
    mono, epi, dxs = [], [], []
    used = flipped = dropped = nodet = 0
    for f0 in f0s:
        t0 = int(re.search(r"(\d+)\.png", f0).group(1))
        j = k1[np.argmin(np.abs(k1 - t0))]
        if abs(j - t0) > 5_000_000:
            continue
        im0, im1 = _read_grey(f0, rig.resolution), _read_grey(t1[j], rig.resolution)

        if board.kind == "aprilgrid":
            d0, d1 = board.detect(im0), board.detect(im1)
            keys, c0, cc = match_tags(d0, d1)
            if len({tid for tid, _ in keys}) < MIN_TAGS:
                nodet += 1; continue
            obj = np.array([board.obj_points(tid)[k] for tid, k in keys])
            for c, K, D in ((c0, K0, D0), (cc, K1, D1)):
                m = _rms_reproj(lens, c, obj, K, D)
                if m is not None:
                    mono.append(m)
        else:
            obj = board.obj()
            c0 = _detect_checkerboard(im0, board); c1 = _detect_checkerboard(im1, board)
            if c0 is None or c1 is None:
                nodet += 1; continue
            for m in (_rms_reproj(lens, c0, obj, K0, D0), _rms_reproj(lens, c1, obj, K1, D1)):
                if m is not None:
                    mono.append(m)
            Ra, _ = _pose(lens, c0, obj, K0, D0)
            best = None
            for tag, cand in (("same", c1), ("rev", c1[::-1])):
                Rb, _ = _pose(lens, cand, obj, K1, D1)
                if Rb is None:
                    continue
                res = np.degrees(np.arccos(np.clip(
                    (np.trace((T10[:3, :3] @ Ra) @ Rb.T) - 1) / 2, -1, 1)))
                if best is None or res < best[0]:
                    best = (res, tag, cand)
            if best is None or best[0] > 30:
                dropped += 1; continue
            if best[1] == "rev":
                flipped += 1
            cc = best[2]

        dy, dx = rectified_rows(lens, c0, cc, K0, D0, K1, D1, rect)
        epi.extend(dy.tolist()); dxs.extend(dx.tolist())
        used += 1

    if not epi:
        sys.exit(f"ERROR: no usable held-out pairs ({nodet} without a shared detection)")
    epi = np.array(epi); mono = np.array(mono); dxs = np.array(dxs)
    if board.kind == "aprilgrid":
        print(f"  pairs used {used}  (fewer than {MIN_TAGS} shared tags: {nodet})")
    else:
        print(f"  pairs used {used}  (order-reversed {flipped}, dropped-inconsistent "
              f"{dropped}, no-detection {nodet})")
    print(f"  epipolar samples: {len(epi)}")
    if len(mono):
        print(f"  mono reproj RMS   : mean {mono.mean():.3f}  p95 {np.percentile(mono,95):.3f}  "
              f"worst {mono.max():.3f} px  (n={len(mono)})")
    print(f"  epipolar |dy|     : mean {epi.mean():.3f}  median {np.median(epi):.3f}  "
          f"p95 {np.percentile(epi,95):.3f}  max {epi.max():.3f} px")
    print(f"  horizontal dx     : median {np.median(dxs):+.2f} px  "
          f"({100*np.mean(dxs>0):.1f}% positive)")

    # ---- guards -------------------------------------------------------------
    # |dy| == 0 is NOT a perfect calibration. cv2.fisheye.stereoRectify can return a
    # DEGENERATE rectification when the distortion model is badly wrong, collapsing
    # both views onto one another -- the 2026-08-22 bad camchain (sign-flipped
    # distortion, 34% coverage) scores exactly 0.000 mean/median/p95/max here while
    # being unusable. A real calibration always has a small nonzero residual.
    DEGEN = float(rig.gates["degenerate_disparity_px"])
    EPI_MAX = float(rig.gates["epipolar_max_px"])
    if np.median(np.abs(dxs)) < DEGEN or np.median(epi) < 1e-3:
        print("\n  *** DEGENERATE RECTIFICATION -- THIS RESULT IS MEANINGLESS ***")
        print(f"  median |disparity| is {np.median(np.abs(dxs)):.3f} px. A real stereo baseline at")
        print("  working distance gives tens of pixels, never a fraction of one -- the two")
        print("  views have collapsed onto each other. The distortion model is almost")
        print("  certainly diverged; check")
        print("  that the two lenses' distortion coefficients share the same signs,")
        print("  and re-run step 20 on the capture that produced this camchain.")
        raise SystemExit(2)
    # ---- camera identity: the rig file's left/right claim, MEASURED ------------
    # kalibr cam0 = physical LEFT  =>  median(u0 - u1) > 0 after rectification.
    # Checked by SIGN and its consistency, never magnitude (disparity scales 1/Z).
    want = rig.expected_dx_sign()
    share = np.mean(np.sign(dxs) == want)
    side0 = rig.kalibr[0]["side"].upper()
    print(f"\n  Disparity sign: {rig.path.name} says kalibr cam0 is the physical {side0} eye,")
    print(f"  so a {'POSITIVE' if want > 0 else 'NEGATIVE'} median dx is expected: measured "
          f"{np.median(dxs):+.2f} px, {100*share:.1f}% with that sign.")
    if np.sign(np.median(dxs)) != want:
        print("\n  VERDICT: FAIL -- the measured disparity sign CONTRADICTS cameras.kalibr[].side.")
        print("  Either the sides in the rig config are swapped, or the record dirs feeding")
        print("  kalibr cam0/cam1 are. Fix that before step 70, or OKVIS is fed mirrored stereo.")
        raise SystemExit(3)
    if np.median(epi) < EPI_MAX:
        print(f"\n  VERDICT: PASS (median |dy| < {EPI_MAX:g} px). 2026-08-22 mk13 reference: median 0.290.")
    else:
        print(f"\n  VERDICT: FAIL (median |dy| >= {EPI_MAX:g} px). Do not deploy this calibration.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
