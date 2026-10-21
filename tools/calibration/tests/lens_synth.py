"""Synthetic stereo corners for the KB4T steps' tests (a helper, not a test file): the 7x5 AprilGrid seen by a KB4T
pair whose truth is phantom12's Fit A, in step 15's npz layout. Board centres are aimed at chosen camera-0 pixels, by
default all nine image cells including the four corners and the bottom (what the hand-held sweep is for). The cached
results are shared by several test files: do not modify what they return."""
import atexit
import functools
import importlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG / "scripts"))
import lensmodel  # noqa: E402
import rigcfg  # noqa: E402
import stereofit as sf  # noqa: E402

RIG12 = PKG / "config" / "rig-phantom12.yaml"
W, H = 1280, 800
# Fit A (final-fit-nofar.json, KB4T@all@800): fx, fy, cx, cy, k1..k4, p1, p2
I0 = np.array([564.88, 563.93, 642.83, 384.95, -0.00828, 0.01186, -0.00050, -0.00142, 0.00085, -0.00111])
I1 = np.array([567.21, 566.15, 642.52, 383.06, -0.00610, 0.00664, 0.00309, -0.00234, -0.00037, -0.00007])
BOARD = rigcfg.AprilGrid(7, 5, 0.1, 0.2)
NINE = [(u, v) for v in (110, 385, 680) for u in (170, 643, 1110)]      # camera-0 pixels, one per image cell


def stereo_pose(rotvec_deg, t):
    C = np.eye(4)
    C[:3, :3] = sf.rotmat(np.radians(rotvec_deg))[0]
    C[:3, 3] = t
    return C


C_TRUE = stereo_pose([0.0, 0.0, 0.9], [0.044425, -0.000226, 0.000302])     # Fit A's t, R turned 0.9 deg about z


def views(intr, C, n_holds, seed, noise=0.3, per_hold=1, z_range=(0.45, 1.2), targets=NINE, jitter=(60, 50),
          noisy_holds=(), noisy=0.6, board=BOARD):
    """n_holds holds of per_hold pairs each (independent pixel noise per pair, sigma `noise` px per axis; holds in
    noisy_holds get `noisy` instead). Hold h aims the board centre at targets[h % len(targets)] (+- jitter px) at a
    camera-0 depth drawn from z_range (m), the board turned up to 25 deg out of plane. A pair is kept when both
    cameras see >= 24 corners (6 tags) inside the 1280 x 800 image. -> step 15's npz layout (dict)"""
    rng = np.random.default_rng(seed)
    Ci = np.linalg.inv(C)
    obj = np.array([board.obj_points(t) for t in range(board.n_tags)]).reshape(-1, 3)
    centre = obj.mean(0)
    rows, hold_of, h, tries = [], [], 0, 0
    while h < n_holds and tries < 50 * n_holds:
        tries += 1
        u, v = targets[h % len(targets)]
        uv = np.array([[u + rng.uniform(-jitter[0], jitter[0]), v + rng.uniform(-jitter[1], jitter[1])]])
        ray = lensmodel.unproject("KB4T", intr[0], uv)[0]
        xa = np.cross([0.0, 1.0, 0.0], ray)
        xa /= np.linalg.norm(xa)
        R = (np.c_[xa, np.cross(ray, xa), ray] @ sf.rotmat(np.radians(rng.uniform(-25, 25, 3) * [1, 1, 0.5]))[0]
             @ sf.rotmat([np.pi, 0.0, 0.0])[0])
        t = ray * (rng.uniform(*z_range) / ray[2]) - R @ centre
        P0 = obj @ R.T + t
        P1 = P0 @ Ci[:3, :3].T + Ci[:3, 3]
        sigma = noisy if h in noisy_holds else noise
        pairs = []
        for _ in range(per_hold):
            cams = []
            for c, P in ((0, P0), (1, P1)):
                px = lensmodel.project("KB4T", intr[c], P) + rng.normal(0, sigma, (len(P), 2))
                inside = (P[:, 2] > 0.05) & (px[:, 0] >= 0) & (px[:, 0] <= W - 1) & (px[:, 1] >= 0) & (px[:, 1] <= H - 1)
                cams.append([(c, i // 4, i % 4, px[i, 0], px[i, 1]) for i in np.flatnonzero(inside)])
            if min(len(x) for x in cams) < 24:
                break
            pairs.append(cams[0] + cams[1])
        if len(pairs) < per_hold:
            continue
        for corners in pairs:
            rows += [(len(hold_of),) + x for x in corners]
            hold_of.append(h)
        h += 1
    r = np.array(rows, float)
    n = len(hold_of)
    return {"img_hold": np.array(hold_of, np.int64), "img_t0": np.arange(n, dtype=np.int64) * 66_666_667,
            "img_t1": np.arange(n, dtype=np.int64) * 66_666_667 + 30_000, "det_img": r[:, 0].astype(np.int64),
            "det_cam": r[:, 1].astype(np.int64), "det_tag": r[:, 2].astype(np.int64), "det_k": r[:, 3].astype(np.int64),
            "det_u": r[:, 4], "det_v": r[:, 5]}


@functools.lru_cache(maxsize=None)
def rig12():
    return rigcfg.load(RIG12)


@functools.lru_cache(maxsize=None)
def fit_sweep():
    """the standard fit sweep: 48 holds over the nine cells at 0.45-1.2 m, 0.3 px noise"""
    return views([I0, I1], C_TRUE, 48, seed=1)


@functools.lru_cache(maxsize=None)
def nominal_fit(model="KB4T"):
    """step 45's fit of fit_sweep() from phantom12's nominal start -> (Data used, fit dict)"""
    intr0, C0 = sf.nominal_start(rig12(), model)
    return sf.fit_from_nominal(sf.Data(fit_sweep(), BOARD), model, intr0, C0)


TEST_TARGETS = [(643, 385), (1000, 385), (985, 625)]        # centre, right, bottom-right


def holdout_sweep(seed=11, n_per_target=12):
    """an independent test sweep: 12 holds of 2 pairs aimed at each of the centre and the right cell at 0.6-0.9 m and
    at the bottom-right corner at 0.45-0.58 m; nothing in the other cells (they must come out "not validated")"""
    parts = [views([I0, I1], C_TRUE, n_per_target, seed=seed + i, per_hold=2, z_range=z, targets=[tg], jitter=(30, 25))
             for i, (tg, z) in enumerate(zip(TEST_TARGETS, ((0.6, 0.9), (0.6, 0.9), (0.45, 0.58))))]
    out, off_img, off_hold = {}, 0, 0
    for d in parts:
        d = dict(d)
        d["det_img"] = d["det_img"] + off_img
        d["img_hold"] = d["img_hold"] + off_hold
        off_img += len(d["img_t0"])
        off_hold += n_per_target
        for k, v in d.items():
            out.setdefault(k, []).append(v)
    return {k: np.concatenate(v) for k, v in out.items()}


@functools.lru_cache(maxsize=None)
def pipeline():
    """steps 45 and 46 run once on fit_sweep(), and holdout_sweep() saved, in a temporary directory shared by the step
    tests (fit.npz, test.npz, fit.json, unc.json) -> Path"""
    out = Path(tempfile.mkdtemp(prefix="lens-pipeline-"))
    atexit.register(shutil.rmtree, out, ignore_errors=True)
    np.savez_compressed(out / "fit.npz", **fit_sweep())
    np.savez_compressed(out / "test.npz", **holdout_sweep())
    rig = ["--rig", str(RIG12)]
    assert importlib.import_module("45_fit_lens").main([str(out / "fit.npz"), str(out / "fit.json")] + rig) == 0
    assert importlib.import_module("46_lens_uncertainty").main(
        [str(out / "fit.npz"), str(out / "fit.json"), str(out / "unc.json")] + rig) == 0
    return out


def write_calibration(path, model, intr, C):
    """a calibration file 66 and 71 read (model, intr, C)"""
    Path(path).write_text(json.dumps({"model": model, "intr": [np.asarray(v).tolist() for v in intr],
                                      "C": np.asarray(C).tolist()}))
    return Path(path)
