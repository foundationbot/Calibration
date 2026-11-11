"""fuse.py: the numpy port of okvis_burst_fusion's fuseDepth (DepthFuse.cpp), checked against a line-by-line scalar
transcription of the C++ loop and on the cases the C++ tests pin."""
import math

import numpy as np
import pytest

from armcal import fuse

P = fuse.DepthParams()


def cpp_pixel(d, s, p=P, n=None):
    """the C++ per-pixel loop, transcribed (float32 where the C++ uses float, double where it uses double)"""
    f = np.float32
    n = len(d) if n is None else n
    needed = fuse.min_valid(p.k_min_frac, n)
    dd, ss = [], []
    for i in range(n):
        di = f(d[i])
        if not math.isfinite(di) or di < f(p.z_min) or di > f(p.z_max):
            continue
        si = f(1.0)
        if s is not None:
            si = f(s[i])
            if not math.isfinite(si):
                continue
            si = max(si, f(p.sigma_floor))
            if si > f(p.sigma_max_rel) * di:
                continue
        dd.append(di)
        ss.append(si)
    k = len(dd)
    if k < needed:
        return 0.0, fuse.TOO_FEW
    srt = sorted(dd)
    med = lambda v: v[k // 2] if k % 2 else f(0.5) * (v[k // 2 - 1] + v[k // 2])   # noqa: E731
    m = med(srt)
    lower = srt[(k - 1) // 2]
    dev = sorted(f(abs(x - m)) for x in dd)
    mad = med(dev)
    scale = max(1.4826 * float(mad), float(f(p.s_floor_abs)) + float(f(p.s_floor_rel) * m))
    tsum = wsum = wd = 0.0
    for x, si in zip(dd, ss):
        r = float(f(x - m)) / (scale * float(f(p.tukey_c)))
        t = (1 - r * r) ** 2 if abs(r) < 1 else 0.0
        w = t / (float(si) * float(si)) if s is not None else t
        tsum += t
        wsum += w
        wd += w * float(x)
    if mad > f(p.edge_mad_rel) * m or wsum <= 0:
        return float(lower), fuse.MEDIAN
    return float(f(wd / wsum)), fuse.MEAN


def test_min_valid_is_the_cpps():
    assert [fuse.min_valid(0.5, n) for n in (1, 2, 3, 4, 5)] == [1, 1, 2, 2, 3]
    assert fuse.min_valid(0.3, 10) == 3
    assert fuse.min_valid(0.0, 5) == 1
    with pytest.raises(ValueError):
        fuse.min_valid(float("nan"), 3)


@pytest.mark.parametrize("with_sigma", [True, False])
@pytest.mark.parametrize("frames_used", [0, 1, 2, 3])
def test_matches_the_cpp_loop_on_random_stacks(with_sigma, frames_used):
    rng = np.random.default_rng(7)
    N, Hh, Ww = 5, 12, 15
    base = rng.uniform(0.3, 2.0, (1, Hh, Ww))
    d = (base + rng.normal(0, 0.002, (N, Hh, Ww))).astype(np.float32)
    d[rng.random(d.shape) < 0.1] = 0.0                                  # no return
    d[rng.random(d.shape) < 0.05] = np.nan
    d[rng.random(d.shape) < 0.05] += 0.3                                # flyers / a second surface
    d[0, 0, 0] = 7.0                                                    # beyond z_max
    s = (rng.uniform(0.0, 0.02, d.shape)).astype(np.float32)
    s[rng.random(d.shape) < 0.05] = 100.0                               # masked by the network
    s[rng.random(d.shape) < 0.02] = np.inf
    out = fuse.fuse_depth(d, s if with_sigma else None, frames_used)
    n = frames_used or N
    assert out.frames == n
    for v in range(Hh):
        for u in range(Ww):
            want, flag = cpp_pixel(d[:, v, u], s[:, v, u] if with_sigma else None, n=n)
            assert out.flags[v, u] == flag, (v, u)
            assert out.depth[v, u] == pytest.approx(want, abs=2e-7), (v, u)


def test_identical_frames_give_the_value_and_an_outlier_is_rejected():
    d = np.full((5, 2, 2), 1.0, np.float32)
    d[3, 0, 0] = 1.2                                    # one wild sample: Tukey weight 0
    out = fuse.fuse_depth(d)
    assert out.depth[1, 1] == pytest.approx(1.0)
    assert out.depth[0, 0] == pytest.approx(1.0, abs=1e-6)
    assert (out.flags == fuse.MEAN).all()


def test_too_few_valid_samples_give_no_depth():
    d = np.zeros((5, 1, 2), np.float32)
    d[:2, 0, 0] = 1.0                                   # 2 of 5 < ceil(2.5) = 3
    d[:3, 0, 1] = 1.0
    out = fuse.fuse_depth(d)
    assert out.depth[0, 0] == 0 and out.flags[0, 0] == fuse.TOO_FEW
    assert out.depth[0, 1] == pytest.approx(1.0) and out.count[0, 1] == 3


def test_a_bimodal_pixel_takes_the_lower_median():
    d = np.array([1.0, 1.0, 1.0, 1.2, 1.2, 1.2], np.float32).reshape(6, 1, 1)
    out = fuse.fuse_depth(d)
    assert out.flags[0, 0] == fuse.MEDIAN
    assert out.depth[0, 0] == pytest.approx(1.0)


def test_frames_used_takes_the_first_frames_only():
    d = np.stack([np.full((1, 1), v, np.float32) for v in (1.0, 1.0, 1.0, 3.0, 3.0)])
    assert fuse.fuse_depth(d, frames_used=3).depth[0, 0] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        fuse.fuse_depth(d, frames_used=6)
