"""Independent validation of the Kalibr stereo calibration on HELD-OUT frames,
using the ChArUco detector (ID-anchored correspondences, partial views OK).

 1. Monocular reprojection: solvePnP per frame with the calibrated equidistant
    model -> RMS(reprojected vs detected) in pixels.
 2. Stereo geometry: for corners with the same ID in both images, rectify with
    the calibrated T_cn_cnm1 -> vertical disparity must be ~0 (epipolar error).

ChArUco layout, lens model and resolution come from config/rig.yaml.
Usage:  61_validate_stereo_charuco.py [CAMCHAIN] [--heldout heldout2] [--rig RIG]
"""
import argparse
import glob
import cv2
import numpy as np
import re
import yaml

import rigcfg

ap = argparse.ArgumentParser()
ap.add_argument("camchain", nargs="?", default="calib-camchain.yaml")
ap.add_argument("--heldout", default="heldout2")
rigcfg.add_arg(ap)
args = ap.parse_args()
rig = rigcfg.load(args.rig)
CH = rig.charuco

CC = yaml.safe_load(open(args.camchain))
lens = rig.camchain_lens(CC, "cam0"); rig.camchain_lens(CC, "cam1")
def K_D(c):
    fx, fy, cx, cy = c["intrinsics"]
    return np.array([[fx,0,cx],[0,fy,cy],[0,0,1]]), np.array(c["distortion_coeffs"]).reshape(4,1)
K0, D0 = K_D(CC["cam0"]); K1, D1 = K_D(CC["cam1"])
T10 = np.array(CC["cam1"]["T_cn_cnm1"]); R, t = T10[:3,:3], T10[:3,3].reshape(3,1)

# layout from the rig config's charuco block
def make_detector(dic):
    d = cv2.aruco.getPredefinedDictionary(dic)
    board = cv2.aruco.CharucoBoard(tuple(CH["squares"]), float(CH["square_m"]),
                                   float(CH["marker_m"]), d)
    return board, cv2.aruco.CharucoDetector(board)

best = None
probe = cv2.imread(sorted(glob.glob(f"{args.heldout}/cam0/*.png"))[0], cv2.IMREAD_GRAYSCALE)
for name in [CH["dictionary"]]:
    board, det = make_detector(getattr(cv2.aruco, name))
    cc, ci, mc, mi = det.detectBoard(probe)
    n = 0 if ci is None else len(ci)
    if best is None or n > best[3]:
        best = (name, board, det, n)
print(f"dictionary: {best[0]} ({best[3]} corners on probe frame)")
_, board, det, _ = best
allobj = board.getChessboardCorners()  # (88,3) object points indexed by charuco id

def detect(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    cc, ci, _, _ = det.detectBoard(img)
    if ci is None or len(ci) < 10:
        return None, None
    return cc.reshape(-1, 2).astype(np.float64), ci.reshape(-1)

def paired(d0, d1, tol_ns=5_000_000):
    ts0 = {int(re.search(r"(\d+)\.png", f).group(1)): f for f in glob.glob(d0 + "/*.png")}
    ts1 = {int(re.search(r"(\d+)\.png", f).group(1)): f for f in glob.glob(d1 + "/*.png")}
    out = []
    for t0, f0 in sorted(ts0.items()):
        t1 = min(ts1, key=lambda t: abs(t - t0))
        if abs(t1 - t0) <= tol_ns:
            out.append((f0, ts1[t1], abs(t1 - t0)))
    print(f"paired {len(out)} frames, max stamp gap {max(p[2] for p in out)/1e6:.2f} ms")
    return [p[0] for p in out], [p[1] for p in out]

f0s, f1s = paired(f"{args.heldout}/cam0", f"{args.heldout}/cam1")
mono, epi, pairs = [], [], 0
R0, R1, P0, P1 = lens.stereo_rectify(K0,D0,K1,D1,rig.resolution,np.ascontiguousarray(R),np.ascontiguousarray(t),flags=cv2.CALIB_ZERO_DISPARITY)
for p0, p1 in zip(f0s, f1s):
    c0, i0 = detect(p0); c1, i1 = detect(p1)
    for c, i, K, D in [(c0,i0,K0,D0),(c1,i1,K1,D1)]:
        if c is None: continue
        obj = allobj[i].reshape(-1,1,3).astype(np.float64)
        und = lens.undistort(c.reshape(-1,1,2), K, D)
        ok, rvec, tvec = cv2.solvePnP(obj, und, np.eye(3), None)
        if not ok: continue
        proj, _ = lens.project(obj, rvec, tvec, K, D)
        mono.append(float(np.sqrt(((proj.reshape(-1,2) - c)**2).sum(axis=1).mean())))
    if c0 is None or c1 is None: continue
    common = np.intersect1d(i0, i1)
    if len(common) < 10: continue
    pairs += 1
    a = c0[np.searchsorted(i0, common)].reshape(-1,1,2)
    b = c1[np.searchsorted(i1, common)].reshape(-1,1,2)
    u0 = lens.undistort(a, K0, D0, R=R0, P=P0)
    u1 = lens.undistort(b, K1, D1, R=R1, P=P1)
    epi.extend(np.abs(u0[:,0,1] - u1[:,0,1]).tolist())

print(f"frames with detections: mono solves={len(mono)}/{2*len(f0s)}, stereo pairs={pairs}/{len(f0s)}")
print(f"monocular reprojection RMS: mean={np.mean(mono):.3f} px, worst={max(mono):.3f} px")
print(f"rectified epipolar |dy|: mean={np.mean(epi):.3f} px, p95={np.percentile(epi,95):.3f} px, max={np.max(epi):.3f} px")
