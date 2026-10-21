"""OKVIS depth rectification with the KB4T lens: the map in numpy, its checks, and the C++ test's fixture.

Task 1 of docs/superpowers/plans/2026-10-21-okvis-kb4t-depth-rectification.md. `twin_map` is the per-pixel
rectification map OKVIS builds for a `distortion_type: kb4t` camera: OpenCV 4.6's cv::fisheye::initUndistortRectifyMap
loop (calib3d/src/fisheye.cpp: SVD inverse of P_3x3 R, incremental x/y/w along a row, the w <= 0 branch, CV_16SC2 by
rounding u * INTER_TAB_SIZE) with lensmodel's KB4T lens in place of the 4-term one: the tangential terms p1, p2 move the
bent normalised point. The C++ `okvis::kb4tInitUndistortRectifyMap` (okvis_common/src/Kb4tRectify.cpp) mirrors it line
by line.

    cd tools/calibration/results/phantom12-2026-10-21-okvis-kb4t
    uv run --python 3.11 --with "numpy<2" --with "opencv-python-headless==4.6.0.66" python make_fixture.py \\
        > make_fixture.out 2>&1

Checks (any failure stops before the fixture is written):
  (b) p1 = p2 = 0 against cv2.fisheye.initUndistortRectifyMap for Fit A's two cameras, with R1/P1, R2/P2 from
      cv2.fisheye.stereoRectify exactly as OKVIS calls it (zero distortion, CALIB_ZERO_DISPARITY, 1280x800 -> 512x320,
      balance 0, fov_scale 1; T_rl = inverse of Fit A's C): CV_16SC2 identical, CV_32FC1 within 1e-4 px;
      and (b') a synthetic R/P with pixels behind the camera (w <= 0): the same +-inf, the same CV_16SC2 entries;
  (c) with p1, p2: the map against lensmodel.project("KB4T") of the same rays, 1e-9 px;
  (d) mutation: the map with p1, p2 swapped misses Fit A's by more than 0.1 px on both cameras.
Writes okvis_common/test/kb4t_rectify_fixture.hpp and formats it with clang-format-22 (CI's format job is --Werror)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "tools" / "calibration" / "scripts"))
import lensmodel  # noqa: E402

FIT_A = REPO / "tools/calibration/results/phantom12-2026-10-20-lens/fit-a-kb4t.json"
REFERENCE = REPO / "tools/calibration/tests/data/kb4t_reference.json"
HEADER = REPO / "okvis_common/test/kb4t_rectify_fixture.hpp"
RAW_SIZE = (1280, 800)  # (width, height): phantom12's stereo cameras
RECT_SIZE = (512, 320)  # the stereo network's input (OKVIS's 512-wide default for a 1280x800 sensor)
INTER_BITS = 5  # cv::INTER_BITS
INTER_TAB_SIZE = 1 << INTER_BITS  # cv::INTER_TAB_SIZE
INT_MIN, INT_MAX = -(2 ** 31), 2 ** 31 - 1
N_SAMPLES = 64
N_POINTS = 8
SEED = 20261021
# the C++ test's "the p-terms are applied" check: some sample moves by more than this (camera 0's p-terms move the map
# by up to 2.5 px, camera 1's by at most 0.48 px: see make_fixture.out)
MIN_SAMPLE_EFFECT = (0.5, 0.25)


def stop(msg: str) -> None:
    print("STOP:", msg)
    sys.exit(1)


def twin_map(lens, R, P, size):
    """The KB4T rectification map: rectified pixel (j, i) -> raw pixel (u, v), doubles, shape (H, W) each.

    lens = [fx, fy, cx, cy, k1, k2, k3, k4, p1, p2]; R, P = the rectifying rotation and the 3x3 part of the rectified
    camera matrix (cv::fisheye::initUndistortRectifyMap's R and P.colRange(0, 3)); size = (width, height).
    Mirrors OpenCV 4.6's loop operation for operation; p1 = p2 = 0 is its map."""
    fx, fy, cx, cy, k1, k2, k3, k4, p1, p2 = (float(a) for a in lens)
    width, height = size
    iR = np.linalg.pinv(np.asarray(P, float) @ np.asarray(R, float))  # (P R).inv(cv::DECOMP_SVD)
    U = np.empty((height, width))
    V = np.empty((height, width))
    for i in range(height):
        # hx, hy, hw: OpenCV's _x, _y, _w, the homogeneous ray of pixel (j, i), summed along the row as OpenCV does
        hx = np.add.accumulate(np.r_[i * iR[0, 1] + iR[0, 2], np.full(width - 1, iR[0, 0])])
        hy = np.add.accumulate(np.r_[i * iR[1, 1] + iR[1, 2], np.full(width - 1, iR[1, 0])])
        hw = np.add.accumulate(np.r_[i * iR[2, 1] + iR[2, 2], np.full(width - 1, iR[2, 0])])
        behind = hw <= 0
        with np.errstate(divide="ignore", invalid="ignore"):
            x = hx / hw
            y = hy / hw
            r = np.sqrt(x * x + y * y)
            theta = np.arctan(r)
            theta2 = theta * theta
            theta4 = theta2 * theta2
            theta6 = theta4 * theta2
            theta8 = theta4 * theta4
            theta_d = theta * (1 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8)
            scale = np.where(r == 0, 1.0, theta_d / np.where(r == 0, 1.0, r))
            # KB4T: the tangential terms move the bent normalised point (lensmodel._tangential)
            mx = x * scale
            my = y * scale
            r2 = mx * mx + my * my
            tx = mx + 2 * p1 * mx * my + p2 * (r2 + 2 * mx * mx)
            ty = my + p1 * (r2 + 2 * my * my) + 2 * p2 * mx * my
            u = fx * tx + cx
            v = fy * ty + cy
        U[i] = np.where(behind, np.where(hx > 0, -np.inf, np.inf), u)
        V[i] = np.where(behind, np.where(hy > 0, -np.inf, np.inf), v)
    return U, V


def cv_round(t):
    """cv::saturate_cast<int>(double) = cvRound: round half to even; on x86-64 (_mm_cvtsd_si32, this wheel's build)
    a value not representable as int (incl. +-inf) gives INT_MIN. On aarch64 it is (int) lrint: identical for every
    finite value in range."""
    with np.errstate(invalid="ignore"):
        q = np.rint(t)
        ok = np.isfinite(q) & (q >= INT_MIN) & (q <= INT_MAX)
        return np.where(ok, q, INT_MIN).astype(np.int64)


def to_16sc2(U, V):
    """OpenCV's CV_16SC2 + CV_16UC1 pair: map1 = (iu >> INTER_BITS, iv >> INTER_BITS) as short, map2 = the
    sub-pixel table index, with iu = saturate_cast<int>(u * INTER_TAB_SIZE)."""
    iu = cv_round(U * INTER_TAB_SIZE)
    iv = cv_round(V * INTER_TAB_SIZE)
    m1 = np.stack([iu >> INTER_BITS, iv >> INTER_BITS], -1).astype(np.int16)  # (short): wraps like C
    m2 = ((iv & (INTER_TAB_SIZE - 1)) * INTER_TAB_SIZE + (iu & (INTER_TAB_SIZE - 1))).astype(np.uint16)
    return m1, m2


def float_diff(a, b):
    """Max |a - b| over the map; equal infinities count 0, any other non-finite mismatch counts inf."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    both = np.isfinite(a) & np.isfinite(b)
    same_inf = ~np.isfinite(a) & (a == b)
    if not np.all(both | same_inf):
        return np.inf
    return float(np.abs(a[both] - b[both]).max()) if both.any() else 0.0


def count_16sc2_diff(mine, theirs):
    (a1, a2), (b1, b2) = mine, theirs
    return int(np.sum(np.any(a1 != b1, -1) | (a2 != b2)))


def lens_k(intr):
    return np.array([[intr[0], 0, intr[2]], [0, intr[1], intr[3]], [0, 0, 1.0]])


def lit(v) -> str:
    s = "%.17g" % float(v)
    return s if any(c in s for c in ".en") else s + ".0"


def c_array(values) -> str:
    a = np.asarray(values)
    if a.ndim == 1:
        return "{" + ", ".join(lit(v) if a.dtype.kind == "f" else str(int(v)) for v in a) + "}"
    return "{" + ", ".join(c_array(row) for row in a) + "}"


def main() -> None:
    print("cv2", cv2.__version__, "numpy", np.__version__)
    if cv2.__version__ != "4.6.0" or int(np.__version__.split(".")[0]) >= 2:
        stop("needs opencv-python-headless 4.6.0.66 (CI's OpenCV is Ubuntu 24.04's 4.6) and numpy < 2")

    fit = json.loads(FIT_A.read_text())["KB4T@all@800"]
    intr = [np.array(v, float) for v in fit["intr"]]
    C = np.array(fit["C"], float)
    ref = json.loads(REFERENCE.read_text())
    if not np.array_equal(np.array(ref["intr"], float), np.array(intr)):
        stop("kb4t_reference.json is not Fit A")

    # OKVIS: T_SC1 = T_SC0 C, computeRectifyMap's T_rl = T_SC1^-1 T_SC0 = C^-1
    T_rl = np.linalg.inv(C)
    zero = np.zeros((4, 1))
    R1, R2, P1, P2, _ = cv2.fisheye.stereoRectify(lens_k(intr[0]), zero, lens_k(intr[1]), zero, RAW_SIZE,
                                                  T_rl[:3, :3].copy(), T_rl[:3, 3:].copy(),
                                                  flags=cv2.CALIB_ZERO_DISPARITY, newImageSize=RECT_SIZE,
                                                  balance=0.0, fov_scale=1.0)
    rect = [(R1, P1), (R2, P2)]
    print(f"rectified: fx {P1[0, 0]:.6f} cx {P1[0, 2]:.6f} cy {P1[1, 2]:.6f}; baseline -P2[0,3]/fx "
          f"{-P2[0, 3] / P2[0, 0]:.6f} m; Fit A's |t| {np.linalg.norm(C[:3, 3]):.6f} m")

    failures = []

    def gate(ok: bool, what: str) -> None:
        print(("  ok: " if ok else "FAIL: ") + what)
        if not ok:
            failures.append(what)

    maps, maps0 = [], []
    for cam, (R, P) in enumerate(rect):
        lens = intr[cam]
        lens0 = lens.copy()
        lens0[8:] = 0.0
        print(f"camera {cam}: p1 {lens[8]:.6g} p2 {lens[9]:.6g}")
        # (b) p = 0 against OpenCV
        U0, V0 = twin_map(lens0, R, P[:, :3], RECT_SIZE)
        cu, cv_ = cv2.fisheye.initUndistortRectifyMap(lens_k(lens), lens[4:8].reshape(4, 1).copy(), R, P, RECT_SIZE,
                                                      cv2.CV_32FC1)
        s1, s2 = cv2.fisheye.initUndistortRectifyMap(lens_k(lens), lens[4:8].reshape(4, 1).copy(), R, P, RECT_SIZE,
                                                     cv2.CV_16SC2)
        n16 = count_16sc2_diff(to_16sc2(U0, V0), (s1, s2))
        gate(n16 == 0, f"(b) cam{cam} p=0 vs cv2.fisheye CV_16SC2: {n16} differing entries of {U0.size}")
        # the CV_32FC1 map holds (float) u: compare the twin rounded the same way (a double beyond FLT_MAX is inf)
        fd = max(float_diff(U0.astype(np.float32), cu), float_diff(V0.astype(np.float32), cv_))
        gate(fd <= 1e-4, f"(b) cam{cam} p=0 vs cv2.fisheye CV_32FC1: max |diff| {fd:.3g} px (tol 1e-4)")
        fd64 = max(float_diff(U0, cu), float_diff(V0, cv_))
        print(f"  (the twin's doubles vs OpenCV's floats: max |diff| {fd64:.3g} px, float rounding)")
        # (c) p != 0 against lensmodel's KB4T on the same rays
        U, V = twin_map(lens, R, P[:, :3], RECT_SIZE)
        jj, ii = np.meshgrid(np.arange(RECT_SIZE[0]), np.arange(RECT_SIZE[1]))
        rays = (np.linalg.inv(P[:, :3] @ R) @ np.stack([jj.ravel(), ii.ravel(), np.ones(jj.size)])).T
        uv = lensmodel.project("KB4T", lens, rays)
        ld = max(float(np.abs(uv[:, 0] - U.ravel()).max()), float(np.abs(uv[:, 1] - V.ravel()).max()))
        gate(ld <= 1e-9, f"(c) cam{cam} KB4T map vs lensmodel.project('KB4T'): max |diff| {ld:.3g} px (tol 1e-9)")
        # (d) mutation: p1, p2 swapped
        swapped = lens.copy()
        swapped[8], swapped[9] = lens[9], lens[8]
        Us, Vs = twin_map(swapped, R, P[:, :3], RECT_SIZE)
        miss = float(np.hypot(Us - U, Vs - V).max())
        gate(miss > 0.1, f"(d) cam{cam} p1/p2 swapped misses the KB4T map by {miss:.3f} px (must be > 0.1)")
        effect = np.hypot(U - U0, V - V0)
        print(f"  p-terms move the map by up to {effect.max():.3f} px (raw); > 0.5 px at {int(np.sum(effect > 0.5))} "
              f"of {effect.size} pixels; map u {U.min():.1f}..{U.max():.1f}, v {V.min():.1f}..{V.max():.1f}")
        maps.append((U, V))
        maps0.append((U0, V0))

    # (b') pixels behind the camera (w <= 0): OpenCV's +-inf branch, synthetic R/P (80 deg yaw, short focal). P's
    # principal point is off the pixel grid on purpose: on a rectified row or column through it the ray's x or y is
    # ~0 and the sign of the +-inf there is the sign of the inverse's rounding noise (numpy's pinv and OpenCV's SVD
    # disagree on it at cy = 240: 303 entries of row 240 flip).
    Ks = np.array([[500.0, 0, 320], [0, 500, 240], [0, 0, 1]])
    Ds = np.array([0.01, 0.0, 0.0, 0.0]).reshape(4, 1)
    th = np.radians(80.0)
    Rs = np.array([[np.cos(th), 0, np.sin(th)], [0, 1.0, 0], [-np.sin(th), 0, np.cos(th)]])
    Ps = np.array([[100.0, 0, 320.25], [0, 100, 240.25], [0, 0, 1]])
    lens_s = [500.0, 500.0, 320.0, 240.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0]
    Us0, Vs0 = twin_map(lens_s, Rs, Ps, (640, 480))
    cu, cv_ = cv2.fisheye.initUndistortRectifyMap(Ks, Ds, Rs, Ps, (640, 480), cv2.CV_32FC1)
    s1, s2 = cv2.fisheye.initUndistortRectifyMap(Ks, Ds, Rs, Ps, (640, 480), cv2.CV_16SC2)
    n_inf = int(np.sum(np.isinf(cu)))
    gate(n_inf > 0, f"(b') synthetic: OpenCV maps {n_inf} pixels to inf (the w <= 0 branch is exercised)")
    fd = max(float_diff(Us0.astype(np.float32), cu), float_diff(Vs0.astype(np.float32), cv_))
    gate(fd <= 1e-4, f"(b') synthetic p=0 vs cv2.fisheye CV_32FC1 (same +-inf): max |diff| {fd:.3g} px")
    n16 = count_16sc2_diff(to_16sc2(Us0, Vs0), (s1, s2))
    gate(n16 == 0, f"(b') synthetic p=0 vs cv2.fisheye CV_16SC2: {n16} differing entries of {Us0.size}")

    if failures:
        stop(f"{len(failures)} check(s) failed; fixture not written")

    # sample rectified pixels: the four corners, the centre, then seeded random ones (shared by both cameras)
    W, H = RECT_SIZE
    fixed = [(0, 0), (W - 1, 0), (0, H - 1), (W - 1, H - 1), (W // 2, H // 2)]
    rng = np.random.default_rng(SEED)
    samples = list(fixed)
    while len(samples) < N_SAMPLES:
        p = (int(rng.integers(0, W)), int(rng.integers(0, H)))
        if p not in samples:
            samples.append(p)
    samples = np.array(samples, int)
    sample_kb4t = [np.stack([U[samples[:, 1], samples[:, 0]], V[samples[:, 1], samples[:, 0]]], -1) for U, V in maps]
    sample_p0 = [np.stack([U[samples[:, 1], samples[:, 0]], V[samples[:, 1], samples[:, 0]]], -1) for U, V in maps0]
    for cam in range(2):
        big = float(np.hypot(*(sample_kb4t[cam] - sample_p0[cam]).T).max())
        need = MIN_SAMPLE_EFFECT[cam]
        gate(big > need, f"fixture cam{cam}: the largest sample p-effect is {big:.3f} px (must be > {need})")

    # camera-frame points: 8 of kb4t_reference.json's 60, spread over the angle off the axis
    pts = np.array(ref["points"], float)
    angle = np.arctan2(np.hypot(pts[:, 0], pts[:, 1]), pts[:, 2])
    order = np.argsort(angle)
    pick = order[np.round(np.linspace(0, len(order) - 1, N_POINTS)).astype(int)]
    points = pts[pick]
    pixels = [np.array(ref["pixels"][cam], float)[pick] for cam in range(2)]
    for cam in range(2):
        d = float(np.abs(lensmodel.project("KB4T", intr[cam], points) - pixels[cam]).max())
        gate(d <= 1e-9, f"fixture cam{cam}: lensmodel.project reproduces the reference pixels, max |diff| {d:.3g} px")
    print("points: angle off the axis (deg)", np.round(np.degrees(angle[pick]), 1).tolist())
    if failures:
        stop(f"{len(failures)} check(s) failed; fixture not written")

    lenses = np.array([intr[0], intr[1]])
    text = f"""// Generated by tools/calibration/results/phantom12-2026-10-21-okvis-kb4t/make_fixture.py (OpenCV {cv2.__version__},
// numpy {np.__version__}); do not edit. Fit A (results/phantom12-2026-10-20-lens/fit-a-kb4t.json, KB4T@all@800), its
// stereo rectification as OKVIS computes it (cv::fisheye::stereoRectify, zero distortion, CALIB_ZERO_DISPARITY,
// 1280x800 -> 512x320, balance 0, fov_scale 1, T_rl = inverse of Fit A's C) and the KB4T map at sample pixels.
#pragma once

namespace kb4t_fixture {{

constexpr int kRectWidth = {W};
constexpr int kRectHeight = {H};

/// Per camera: fx, fy, cx, cy, k1, k2, k3, k4, p1, p2.
constexpr double kLens[2][10] = {c_array(lenses)};

/// Per camera: the rectifying rotation R1 / R2 (3x3, row-major).
constexpr double kR[2][9] = {c_array(np.array([R1.ravel(), R2.ravel()]))};

/// Per camera: the rectified camera matrix P1 / P2 (3x4, row-major).
constexpr double kP[2][12] = {c_array(np.array([P1.ravel(), P2.ravel()]))};

constexpr int kNumSamples = {N_SAMPLES};

/// Rectified pixels (column, row), shared by both cameras: the four corners, the centre, then seeded random ones.
constexpr int kSamplePixels[{N_SAMPLES}][2] = {c_array(samples)};

/// Per camera: the KB4T map (raw pixel u, v) at each sample pixel, doubles from the numpy twin.
constexpr double kSampleKb4t[2][{N_SAMPLES}][2] = {c_array(np.array(sample_kb4t))};

/// Per camera: the same with p1 = p2 = 0 (OpenCV's fisheye map).
constexpr double kSampleP0[2][{N_SAMPLES}][2] = {c_array(np.array(sample_p0))};

/// Per camera: some sample's KB4T value differs from its p = 0 value by more than this [px] (the p-terms are applied;
/// camera 1's p-terms move its whole map by at most 0.48 px).
constexpr double kMinSampleEffect[2] = {c_array(np.array(MIN_SAMPLE_EFFECT, float))};

constexpr int kNumPoints = {N_POINTS};

/// Camera-frame points (shared by both cameras), from tools/calibration/tests/data/kb4t_reference.json.
constexpr double kPoints[{N_POINTS}][3] = {c_array(points)};

/// Per camera: the points' pixels by lensmodel.project("KB4T") (= the reference file's pixels).
constexpr double kPointPixels[2][{N_POINTS}][2] = {c_array(np.array(pixels))};

}} // namespace kb4t_fixture
"""
    HEADER.write_text(text)
    print("wrote", HEADER.relative_to(REPO))
    fmt = next((p for p in (os.path.expanduser("~/.local/bin/clang-format-22"), shutil.which("clang-format-22"))
                if p and os.path.exists(p)), None)
    if fmt is None:
        stop("clang-format-22 not found: the header is written but unformatted (CI's format job is --Werror)")
    subprocess.run([fmt, "-i", str(HEADER)], check=True, cwd=REPO)
    print("formatted with", Path(fmt).name)
    print("all checks passed")


if __name__ == "__main__":
    main()
