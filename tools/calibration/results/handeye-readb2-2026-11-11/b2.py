"""readb2 bench: B0 on all 48 stored phantom14 captures (right wave 1 + wave 2, left wave), one capture per process
(one BLAS thread each), under a config and a perturbation that should not matter.

  python3 b2.py --config fast|robust|<name>[,key=value...] --variant base|jit|phase|f2|f2b|start --out OUT.json [-j 8]

variants (each against "base" = the numpy fusion of all 5 frames, the model as the config samples it, E0 = I):
  jit    every model point moved in its face's plane by a Gaussian offset (0.3 mm sd): the same surface, other points
  phase  the even sampler's R2 sequence started elsewhere (0.25 instead of 0.5): other, equally even points
  f2     only the first 2 depth frames fused (the joints still from all 5)
  f2b    only the last 2 depth frames fused
  f3     only the first 3 depth frames fused
  start  the solver started from a small correction (1 deg about a random axis, 2 mm in a random direction; seeded
         per capture) instead of the encoders' pose
Data: $CALIBTOOL_DATA/p14arm (Moonstone ~/calibtool/data/p14arm)."""
import argparse
import json
import os
import sys
import time
import zlib
from multiprocessing import get_context
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from armcal import fuse, meshfit, readb  # noqa: E402
from armcal.urdf import Urdf  # noqa: E402
from rigmath import rot_axis, transform  # noqa: E402

DATA = Path(os.environ.get("CALIBTOOL_DATA", Path.home() / "calibtool" / "data")).expanduser() / "p14arm"
STORED = {"base_w1.json": "1103a-right", "base_w2.json": "1104a-right", "base_L.json": "1104b-left"}
CONV = {"data_stride": int, "max_steps_final": int, "final_step_m": float, "final_step_deg": float,
        "max_steps": int, "model_voxel": float, "tree_leafsize": int, "spacing": float, "sampler": str,
        "trans_gate": float, "score_tau": float, "ambiguous_mm": float, "rot_prior": float, "prior_axis": str, "creep_cos": float, "ls_adaptive": lambda v: v == "1", "creep_cos": float, "ls_adaptive": lambda v: v == "1",
        "strides": lambda v: tuple(int(x) for x in v.split(":")),
        "robust": lambda v: v == "1", "line_search": lambda v: v == "1", "fill_holes": lambda v: v == "1",
        "starts": lambda v: tuple(tuple(float(x) for x in p.split(":")) for p in v.split("/") if p)}


def config(name):
    head, *kv = name.split(",")
    base = {"fast": meshfit.FAST, "exact": meshfit.EXACT, "reference": meshfit.REFERENCE}.get(head)
    if base is None:
        base = getattr(meshfit, head.upper())
    kw = dict(x.split("=", 1) for x in kv)
    return meshfit.with_config(base, **{k: CONV[k](v) for k, v in kw.items()})


def jobs(only=None):
    out = []
    for f, tag in STORED.items():
        D = json.loads((DATA / f).read_text())
        for r in D["res"]:
            path = next(DATA.glob(f"bursts/p14-armwave-{tag}-{r['name']}.npz"))
            if only and not any(o in path.name for o in only.split(",")):
                continue
            out.append((str(path), D["side"], {k: r[k] for k in ("off", "rot", "cov", "clean", "ok")}))
    return out


_ARM = {}


def wave(burst):
    """p14-armwave-1104a-right-07-p09-10-0.npz -> 1104a-right"""
    return "-".join(burst.split("-")[2:4])


def one(args):
    path, side, stored, cfgname, variant, med, gates = args
    urdf = Urdf.load(DATA / "phantom_mk1.urdf")
    cfg = config(cfgname)
    kw = {}
    if variant == "jit":
        kw = dict(jitter=0.0003, seed=1)
    elif variant == "phase":
        kw = dict(phase=0.25)
    key = (side, cfg.sampler, cfg.spacing, cfg.model_voxel, tuple(sorted(kw.items())))
    if key not in _ARM:
        _ARM[key] = readb.arm_model(urdf, side, cfg=cfg, **kw)
    arm = _ARM[key]
    t0 = time.perf_counter()
    depth = None
    if variant in ("f2", "f2b", "f3"):
        from armcal import burst_io
        b = burst_io.load(path)
        sl = {"f2": slice(0, 2), "f3": slice(0, 3)}.get(variant, slice(b.frames - 2, b.frames))
        t1 = time.perf_counter()
        depth = fuse.fuse_depth(np.asarray(b.depth)[sl], None if b.sigma is None else np.asarray(b.sigma)[sl]).depth
        fuse_s = time.perf_counter() - t1
    cap = readb.load_capture(path, urdf, side, 0, depth=depth)
    if depth is not None:
        cap.fuse_s = fuse_s
    load_s = time.perf_counter() - t0
    E0 = None
    if variant == "start":
        rng = np.random.default_rng(zlib.crc32(Path(path).name.encode()))
        ax, d = rng.normal(size=3), rng.normal(size=3)
        E0 = transform(rot_axis(ax / np.linalg.norm(ax), np.radians(1.0)), 0.002 * d / np.linalg.norm(d))
    H, W = cap.depth.shape
    t0 = time.perf_counter()
    data = meshfit.unproject(cap.depth, cap.K, cfg.data_stride)
    kw = {}
    if med is not None:                # pass 2: from the arm's median offset of pass 1 (a translation)
        Ec = readb.consensus_start(cap.T_TC, med)
        E0 = Ec if E0 is None else E0 @ Ec
        kw["gates"] = gates
    f = meshfit.fit_pose(arm, cap.q, cap.T_TC, data, cap.K, W, H, cfg, E0=E0, **kw)
    fit_s = time.perf_counter() - t0
    off = 1e3 * f.wrist_offset_T
    rec = dict(name=cap.name, side=cap.side, off=off.tolist(), norm=float(np.linalg.norm(off)), rot=f.rotation_deg,
               cov=f.coverage, ok=f.ok, E=np.asarray(f.delta_c).tolist(), T_TC=np.asarray(cap.T_TC).tolist(),
               T_BC=np.asarray(cap.T_BC).tolist(), q=cap.q, yaw=cap.angles["NeckYaw"],
               pitch=cap.angles["NeckPitch"], wrist=arm.wrist_point_T(cap.q).tolist(), rms=f.rms_mm, cond=f.cond,
               steps=f.steps, frames=cap.frames, fit_s=round(fit_s, 3), fuse_s=round(cap.fuse_s, 3),
               load_s=round(load_s, 3), score=f.score, candidates=f.candidates, margin=f.margin,
               model_points=len(arm.P))
    rec["clean"] = readb.is_clean(rec)
    rec["burst"] = Path(path).name
    rec["stored"] = stored
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="fast")
    ap.add_argument("--variant", default="base")
    ap.add_argument("--out", required=True)
    ap.add_argument("-j", type=int, default=8)
    ap.add_argument("--only")
    ap.add_argument("--pass1", help="pass 2: start each capture at its wave's median offset in this bench output")
    ap.add_argument("--gates", default="0.015,0.008,0.005", help="pass 2's stages")
    a = ap.parse_args()
    med = {}
    if a.pass1:
        P1 = json.loads(Path(a.pass1).read_text())["res"]
        for key in {wave(r["burst"]) for r in P1}:
            med[key] = readb.wave_median([r for r in P1 if wave(r["burst"]) == key])
    gates = tuple(float(g) for g in a.gates.split(","))
    J = [(p, s, r, a.config, a.variant, med.get(wave(Path(p).name)) if a.pass1 else None, gates)
         for p, s, r in jobs(a.only)]
    t0 = time.time()
    with get_context("fork").Pool(min(a.j, len(J))) as pool:
        recs = pool.map(one, J, chunksize=1)
    t = np.array([r["fit_s"] for r in recs])
    summary = dict(config=a.config, variant=a.variant, cfg=repr(config(a.config)), n=len(recs), j=a.j,
                   wall_s=round(time.time() - t0, 1), fit_s_mean=float(t.mean()), fit_s_max=float(t.max()),
                   steps_median=float(np.median([r["steps"] for r in recs])),
                   clean=int(sum(r["clean"] for r in recs)))
    Path(a.out).write_text(json.dumps(dict(summary=summary, res=recs), indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
