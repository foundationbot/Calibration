"""A burst's depth frames fused into one depth image: the numpy port of OKVIS2-X okvis_burst_fusion's fuseDepth()
(src/DepthFuse.cpp, c5b4c6b0), which is the only part of the C++ module _burstfusion that read B uses (hand_check's
load_pose took run(...).depth.depth; the fused RGB fed only the overlay picture, the cloud and voxels nothing).

Per pixel, over the first `frames_used` frames (0 = all):
  - a sample is valid if finite and z_min <= d <= z_max; with a sigma plane also if sigma is finite and
    max(sigma, sigma_floor) <= sigma_max_rel * d;
  - fewer than ceil(k_min_frac * N) valid samples: depth 0 (TOO_FEW);
  - m = median, MAD = median |d - m|, scale = max(1.4826 MAD, s_floor_abs + s_floor_rel m);
  - Tukey weights t_j = (1 - r²)² for |r| < 1, r = (d_j - m) / (tukey_c scale), times 1 / sigma_j² with sigma;
  - MAD > edge_mad_rel m (an edge or a bimodal pixel) or all weights zero: the lower median (MEDIAN), else the weighted
    mean (MEAN).
The arithmetic follows the C++ types (float medians, double weights), so the result matches the C++ to float rounding
(tests: the stored phantom14 fusions)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

TOO_FEW, MEDIAN, MEAN = 1, 2, 4


@dataclass(frozen=True)
class DepthParams:                 # okvis/burst/Params.hpp DepthFuseParams, the defaults read B ran with
    z_min: float = 0.1
    z_max: float = 6.0
    use_sigma: bool = True
    sigma_floor: float = 0.0005
    sigma_max_rel: float = 0.5
    k_min_frac: float = 0.5
    s_floor_abs: float = 0.0005
    s_floor_rel: float = 0.001
    edge_mad_rel: float = 0.02
    tukey_c: float = 4.685


@dataclass
class Fused:
    frames: int
    depth: np.ndarray       # (H,W) float32 m, 0 = none
    count: np.ndarray       # (H,W) uint8 valid samples
    mad: np.ndarray         # (H,W) float32 m
    flags: np.ndarray       # (H,W) uint8: TOO_FEW / MEDIAN / MEAN


def min_valid(frac: float, n: int) -> int:
    """ceil(frac * n) clamped to 1..n (Params.hpp minValid)"""
    if not math.isfinite(frac) or frac < 0:
        raise ValueError("the fraction must be finite and not negative")
    k = math.ceil(float(np.float32(frac)) * n - 1e-4)
    return int(max(1, min(k, max(1, n))))


def _median_sorted(s, k):
    """float32 median of the first k (>= 1) values of each column of s (sorted along axis 0, NaN last)"""
    lo = np.take_along_axis(s, np.maximum((k - 1) // 2, 0)[None], 0)[0]
    hi = np.take_along_axis(s, np.maximum(k // 2, 0)[None], 0)[0]
    return np.float32(0.5) * (lo + hi), lo


def fuse_depth(depth, sigma=None, frames_used: int = 0, p: DepthParams = DepthParams()) -> Fused:
    depth = np.asarray(depth, np.float32)
    if depth.ndim != 3 or depth.shape[0] < 1:
        raise ValueError(f"depth must be (N,H,W) with N >= 1, got {depth.shape}")
    N = depth.shape[0]
    if not 0 <= frames_used <= N:
        raise ValueError(f"frames_used {frames_used} outside 0..{N}")
    n = frames_used or N
    d = depth[:n]
    f32 = np.float32
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(d) & ~(d < f32(p.z_min)) & ~(d > f32(p.z_max))
        weigh = p.use_sigma and sigma is not None
        if weigh:
            s = np.asarray(sigma, np.float32)[:n]
            valid &= np.isfinite(s)
            s = np.maximum(s, f32(p.sigma_floor))
            valid &= ~(s > f32(p.sigma_max_rel) * d)
    k = valid.sum(0)
    needed = min_valid(p.k_min_frac, n)
    dv = np.where(valid, d, f32(np.nan))
    m, lower = _median_sorted(np.sort(dv, axis=0), k)
    dev = np.where(valid, np.abs(d - m), f32(np.nan))
    mad, _ = _median_sorted(np.sort(dev, axis=0), k)
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.maximum(1.4826 * mad.astype(np.float64),
                           float(f32(p.s_floor_abs)) + (f32(p.s_floor_rel) * m).astype(np.float64))
        r = (d - m).astype(np.float64) / (scale * float(f32(p.tukey_c)))
        t = np.where(np.abs(r) < 1.0, (1.0 - r * r) * (1.0 - r * r), 0.0)
        t = np.where(valid, t, 0.0)
        w = np.where(valid, t / (s.astype(np.float64) * s), 0.0) if weigh else t
        wsum = w.sum(0)
        mean = ((w * np.where(valid, d, f32(0)).astype(np.float64)).sum(0) / wsum).astype(np.float32)
        use_median = (mad > f32(p.edge_mad_rel) * m) | ~(wsum > 0.0)
    out = np.where(use_median, lower, mean).astype(np.float32)
    flags = np.where(use_median, MEDIAN, MEAN).astype(np.uint8)
    few = k < needed
    out[few] = 0.0
    flags[few] = TOO_FEW
    mad = np.where(few, f32(0), mad).astype(np.float32)
    return Fused(n, out, k.astype(np.uint8), mad, flags)


def fuse_rgb(rgb, frames_used: int = 0) -> np.ndarray:
    """(H,W,3) uint8: the per-pixel median of the frames (for the overlay picture only; the C++ median's
    even-count rounding is not reproduced)"""
    rgb = np.asarray(rgb)
    n = frames_used or rgb.shape[0]
    return np.median(rgb[:n], axis=0).round().astype(np.uint8)
