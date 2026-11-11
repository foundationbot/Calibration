"""diagnostic: per-step trajectory of B0 on a few captures (FAST, and the model thinned to 0.5 mm voxels)"""
import json, os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from armcal import meshfit, readb  # noqa
from armcal.urdf import Urdf  # noqa

DATA = Path(os.environ.get("CALIBTOOL_DATA", "~/calibtool/data")).expanduser() / "p14arm"
urdf = Urdf.load(DATA / "phantom_mk1.urdf")
names = sys.argv[1:]
for nm in names:
    path = next(DATA.glob(f"bursts/p14-armwave-{nm}.npz"))
    side = "Left" if "-left-" in nm else "Right"
    cap = readb.load_capture(path, urdf, side, depth=np.load(DATA / "fused_cpp" / path.name)["depth"])
    H, W = cap.depth.shape
    data = meshfit.unproject(cap.depth, cap.K)
    for vox in (None, 0.0005):
        cfg = meshfit.with_config(meshfit.FAST, model_voxel=vox)
        arm = readb.arm_model(urdf, side, cfg=cfg)
        tr = []
        t0 = time.time()
        f = meshfit.fit_pose(arm, cap.q, cap.T_TC, data, cap.K, W, H, cfg, trace=tr)
        print(f"== {nm} voxel {vox}: off {np.round(1e3*f.wrist_offset_T,1)} rot {f.rotation_deg:.2f} cov {f.coverage:.3f} "
              f"rms {f.rms_mm:.2f} inl {f.inlier.sum()} steps {f.steps} {time.time()-t0:.1f}s")
        prev = None
        for k, (g, n, cost, E) in enumerate(tr):
            off = 1e3 * meshfit.wrist_offset(arm, cap.q, cap.T_TC, E)
            if k % 5 == 0 or (prev is not None and prev != g) or k == len(tr) - 1:
                print(f"   {k:4d} gate {g*1e3:4.0f} n {n:6d} cost/n {1e6*cost/max(n,1):7.3f} off {np.round(off,1)} rot {meshfit.angle_deg(E):6.2f}")
            prev = g
