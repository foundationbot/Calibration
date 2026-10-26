"""T2 cross-neck test (plan 2026-10-25-tagfree-selfcal.md, fixed before data): the correction trans-only read B fitted
at one neck pitch, applied to another session's captures, scored against chain + B1 (score_chain_b1.py).

The correction is recovered from the source session's own candidate poses as the HEAD-side translation each capture
implies, X = T_TB^-1 · (T_tip · T_C0_hand^-1) · T_HC0^-1 (25 Oct: its mean equals read B's reported X_t to 0.03 mm;
torso- and camera-side do not match). It must be one constant translation across captures up to the joint-reading
noise (median vs per-frame joints: spread <= 0.3 mm, angle <= 0.05 deg), else nothing is written.

usage: cross_neck.py src_bursts src_candidate.json dst_bursts dst_held(comma) config.yaml urdf out_candidate.json
"""
import json
import sys
from pathlib import Path

import numpy as np

import armchain
import armfit as af
import neckfk
import okviscfg
import rigmath as rm

sb, sc, db, held, cfg, urdf, outp = sys.argv[1:8]
arm, neck = armchain.ArmChain(urdf), neckfk.NeckChain(urdf)
rig = af.rig_from_config(okviscfg.load(cfg))


def joints(path):
    d = np.load(path, allow_pickle=False)
    names = json.loads(str(d["info"]))["joint_names"]
    q = np.median(d["joints"][:, 1:], axis=0)
    return q[[names.index(j) for j in armchain.RIGHT_ARM]], (q[names.index("NeckYaw")], q[names.index("NeckPitch")])


src = json.loads(Path(sc).read_text())["captures"]
dts = []
for n, c in src.items():
    q, qn = joints(Path(sb) / f"{n}.npz")
    T_TC = arm.T_base_tip(q) @ np.linalg.inv(np.asarray(c["T_C0_hand"], float))
    dts.append(np.linalg.inv(neck.T_TB(*qn)) @ T_TC @ np.linalg.inv(rig.T_HC[0]))
t = np.array([d[:3, 3] for d in dts]) * 1e3
a = [rm.rot_angle_deg(d[:3, :3]) for d in dts]
print(f"implied head-side X over {len(dts)} source captures: t mean {t.mean(0).round(3).tolist()} mm, spread "
      f"{(t.max(0) - t.min(0)).round(3).tolist()} mm, angle {np.mean(a):.4f} deg (spread {max(a) - min(a):.4f})")
if (t.max(0) - t.min(0)).max() > 0.3 or max(a) > 0.05:
    raise SystemExit("not one constant head-side translation: convention differs; nothing written")
X = rm.transform(np.eye(3), t.mean(0) * 1e-3)
held = set(held.split(","))
out = {}
for p in sorted(Path(db).glob("*.npz")):
    q, qn = joints(p)
    out[p.stem] = {"T_C0_hand": (np.linalg.inv(neck.T_TB(*qn) @ X @ rig.T_HC[0]) @ arm.T_base_tip(q)).tolist(),
                   "in_sample": p.stem not in held}
Path(outp).write_text(json.dumps({"method": f"trans-only X from {Path(sc).name} applied across neck", "captures": out},
                                 indent=1))
print(f"{len(out)} captures written, held-out {sorted(held)}")
