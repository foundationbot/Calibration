"""Is the 5 mm between the cameras' own pixel-level fits a systematic or noise? Fit each camera alone on the
even holds and on the odd holds of a take (corner dumps from dump_corners.py) with handeye_px: a systematic
shows the same difference in both halves; noise would differ between them by about its own size.
Usage (Thor, calib-py:2, PYTHONPATH = the reference package's scripts): python split_half.py <corners.npz> <qc.json>"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np

import handeye_px as hp
import neckfk
import okviscfg
import rigcfg
import rigmath as rm

s81 = importlib.import_module("81_tbs_handeye")
T = Path("/t")
cfg = okviscfg.load(T / "tools/calibration/results/phantom12-2026-10-15/imucam/phantom12_stereo.yaml")
rig = rigcfg.load(T / "tools/calibration/config/rig-phantom12.yaml")
chain = neckfk.NeckChain(T / "config/foundation_rig/20250206_phantom_mk1.urdf")
d = dict(np.load(sys.argv[1]))
drop = set(json.loads(Path(sys.argv[2]).read_text())["drop_holds"])
kind = rigcfg.Lens(cfg.lens).kind
lens = rigcfg.Lens(cfg.lens)
C = [rm.inv(cfg.T_SC[0]) @ Ts for Ts in cfg.T_SC]
X0 = cfg.T_BS @ cfg.T_SC[0]
centre = np.array(rig.resolution, float) / 2
ims, z0 = [], None
for n in range(len(d["img_hold"])):
    h = int(d["img_hold"][n])
    if h in drop:
        continue
    A = chain.T_TB(d["img_yaw"][n], d["img_pitch"][n])
    for c in (0, 1):
        m = (d["det_img"] == n) & (d["det_cam"] == c)
        uv = np.c_[d["det_u"][m], d["det_v"][m]]
        ok = np.hypot(*(uv - centre).T) <= s81.MAX_RADIUS_PX
        keys = list(zip(d["det_tag"][m][ok], d["det_k"][m][ok]))
        if len({t for t, _ in keys}) < s81.MIN_TAGS:
            continue
        if z0 is None:
            det = {(int(t), int(k)): tuple(p) for (t, k), p in zip(keys, uv[ok])}
            Tcw = s81.pose_from_corners(det, rig.board, lens, cfg.K[c], cfg.D[c], tuple(centre))
            if Tcw is not None:
                z0 = A @ X0 @ C[c] @ Tcw
        ims.append(hp.Image(0, c, A, np.array([rig.board.obj_points(int(t))[int(k)] for t, k in keys]), uv[ok], str(h)))
print(f"{len(ims)} images, holds {len({im.group for im in ims})}")
for half in (0, 1):
    X = {}
    for c in (0, 1):
        sel = [im for im in ims if int(im.group) % 2 == half and im.cam == c]
        r = hp.fit(sel, X0, [z0], C, cfg.K, cfg.D, kind)
        X[c] = r["X"]
        print(f"  holds {'even' if half == 0 else 'odd'}, camera {c}: X - config (Head, mm) "
              + " ".join(f"{v:+.2f}" for v in 1e3 * (r['X'][:3, 3] - X0[:3, 3]))
              + f"  +- {np.linalg.norm(r['sigma_trans_head_mm']):.2f} mm, {r['rms_px']:.3f} px")
    dt = 1e3 * (X[0][:3, 3] - X[1][:3, 3])
    dr = np.degrees(rm.log_rot(X[0][:3, :3] @ X[1][:3, :3].T))
    print(f"  holds {'even' if half == 0 else 'odd'}: camera 0 - camera 1 = " + " ".join(f"{v:+.2f}" for v in dt)
          + f" mm ({np.linalg.norm(dt):.2f}), rotation " + " ".join(f"{v:+.3f}" for v in dr) + " deg")
