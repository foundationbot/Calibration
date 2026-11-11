"""H0 bench: read B (B0) on every stored phantom14 capture (48: right wave 1 + wave 2, left), one capture per process,
one BLAS thread each; compared with the stored per-capture records (urdfx/base_w1, base_w2, base_L.json) or with
another bench output.

  python3 bench.py --config reference|exact|fast|<name> --depth cpp|numpy [--frames N] --out OUT.json [-j 8]
                   [--only GLOB] [--profile]

--depth cpp: the stored C++ fusion of each burst (fused_cpp/, what base_*.json were fitted on); numpy: handeye.fuse.
Data: $CALIBTOOL_DATA/p14arm (Moonstone ~/calibtool/data/p14arm)."""
import argparse
import cProfile
import io
import json
import os
import pstats
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from armcal import meshfit, readb  # noqa: E402
from armcal.urdf import Urdf  # noqa: E402

DATA = Path(os.environ.get("CALIBTOOL_DATA", Path.home() / "calibtool" / "data")).expanduser() / "p14arm"
STORED = {"base_w1.json": "1103a-right", "base_w2.json": "1104a-right", "base_L.json": "1104b-left"}

CONFIGS = {
    "reference": meshfit.REFERENCE,
    "exact": meshfit.EXACT,
}


def config(name):
    if name in CONFIGS:
        return CONFIGS[name]
    if name == "fast":
        return meshfit.FAST
    kw = dict(kv.split("=") for kv in name.split(",")) if "=" in name else {}
    base = meshfit.EXACT
    conv = {"data_stride": int, "max_steps_final": int, "final_step_m": float, "final_step_deg": float,
            "roi_px": int, "max_steps": int, "model_voxel": float, "tree_leafsize": int,
            "tree_fast": lambda v: v == "1", "fast_render": lambda v: v == "1"}
    return meshfit.with_config(base, **{k: conv[k](v) for k, v in kw.items()})


def jobs(only=None):
    out = []
    for f, tag in STORED.items():
        D = json.loads((DATA / f).read_text())
        for r in D["res"]:
            path = next(DATA.glob(f"bursts/p14-armwave-{tag}-{r['name']}.npz"))
            if only and not Path(path).match(only):
                continue
            out.append((str(path), D["side"], r))
    return out


_ARM = {}


def one(args):
    path, side, stored, cfgname, depth_src, frames, prof = args
    urdf = Urdf.load(DATA / "phantom_mk1.urdf")
    cfg = config(cfgname)
    if (side, cfg.model_voxel) not in _ARM:
        _ARM[side, cfg.model_voxel] = readb.arm_model(urdf, side, cfg=cfg)
    arm = _ARM[side, cfg.model_voxel]
    t0 = time.perf_counter()
    depth = None
    if depth_src == "cpp":
        depth = np.load(DATA / "fused_cpp" / Path(path).name)["depth"]
    cap = readb.load_capture(path, urdf, side, frames, depth=depth)
    load_s = time.perf_counter() - t0
    pr = cProfile.Profile() if prof else None
    if pr:
        pr.enable()
    rec, f = readb.match(arm, cap, cfg, view=False)
    if pr:
        pr.disable()
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(25)
        rec["profile"] = s.getvalue()
    rec["load_s"] = round(load_s, 3)
    rec["burst"] = Path(path).name
    rec["stored"] = {k: stored[k] for k in ("off", "rot", "cov", "clean", "ok")}
    return rec


def compare(recs):
    d_off = np.array([np.linalg.norm(np.subtract(r["off"], r["stored"]["off"])) for r in recs])
    d_rot = np.array([abs(r["rot"] - r["stored"]["rot"]) for r in recs])
    d_cov = np.array([abs(r["cov"] - r["stored"]["cov"]) for r in recs])
    flips = [r["burst"] for r in recs if r["clean"] != r["stored"]["clean"]]
    t = np.array([r["fit_s"] for r in recs])
    return dict(n=len(recs), max_off_mm=float(d_off.max()), median_off_mm=float(np.median(d_off)),
                max_rot_deg=float(d_rot.max()), max_cov=float(d_cov.max()), clean_flips=flips,
                fit_s_mean=float(t.mean()), fit_s_median=float(np.median(t)), fit_s_max=float(t.max()),
                fuse_s_mean=float(np.mean([r["fuse_s"] for r in recs])),
                steps_median=float(np.median([r["steps"] for r in recs])),
                worst=sorted(((float(o), r["burst"]) for o, r in zip(d_off, recs)), reverse=True)[:5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="fast")
    ap.add_argument("--depth", choices=("cpp", "numpy"), default="cpp")
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("-j", type=int, default=8)
    ap.add_argument("--only")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--against", help="compare with this bench output instead of the stored records")
    a = ap.parse_args()
    J = [(p, s, r, a.config, a.depth, a.frames, a.profile) for p, s, r in jobs(a.only)]
    if a.against:
        ref = {r["burst"]: r for r in json.loads(Path(a.against).read_text())["res"]}
        J = [(p, s, dict(ref[Path(p).name], **{"off": ref[Path(p).name]["off"]}), *rest) for p, s, _, *rest in J]
    t0 = time.time()
    with get_context("fork").Pool(min(a.j, len(J))) as pool:
        recs = pool.map(one, J, chunksize=1)
    summary = compare(recs)
    summary.update(config=a.config, depth=a.depth, frames=a.frames, wall_s=round(time.time() - t0, 1),
                   cfg=repr(config(a.config)), against=a.against or "stored base_w1/base_w2/base_L")
    Path(a.out).write_text(json.dumps(dict(summary=summary, res=recs), indent=1))
    print(json.dumps(summary, indent=1))
    if a.profile:
        print(recs[0]["profile"])


if __name__ == "__main__":
    main()
