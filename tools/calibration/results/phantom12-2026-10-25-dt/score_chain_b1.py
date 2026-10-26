"""Score a tag-free candidate on a NEW right-arm session against the D6 reference fixed before data (plan
2026-10-25-tagfree-selfcal.md, 25 Oct ~00:20): the hand from the #816 chain at each burst's own joints (median over
the burst) with B1, in Fit C's camera 0 — no plate needed. Also shown: the chain alone (M0-like) and today's-old dT.

usage: score_chain_b1.py bursts_dir candidate.json config.yaml urdf out.json [Right|Left]
  Left: the left chain (mirrored shoulder origin, unmeasured) with B1 -- not a reference, a measure of the left chain
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
import yardstick as ys

B1 = np.array([[1.0, 0.000257, 0.000018, -0.001391],                  # README.md here; rotation re-orthonormalised
               [-0.000257, 0.999995, -0.003281, 0.002994],
               [-0.000019, 0.003281, 0.999995, 0.001852],
               [0, 0, 0, 1.0]])
B1[:3, :3] = rm.nearest_rotation(B1[:3, :3])
OLD = rm.transform(rm.exp_rot(np.radians([0.219, 0.011, -0.032])), np.array([-1.37, 2.32, 1.40]) * 1e-3)

bursts, cand_p, cfg, urdf, outp = sys.argv[1:6]
SIDE = sys.argv[6] if len(sys.argv) > 6 else "Right"
JOINTS = tuple(n.replace("Right", SIDE) for n in armchain.RIGHT_ARM)
cand = json.loads(Path(cand_p).read_text())
arm, neck = armchain.ArmChain(urdf, JOINTS), neckfk.NeckChain(urdf)
assert arm.joints[5].child == f"{SIDE}WristRollLink" and arm.tip == f"{SIDE}GripperYawLink", (arm.joints[5].child, arm.tip)
rig = af.rig_from_config(okviscfg.load(cfg))
refs = {"B1": {}, "chain": {}, "old_dT": {}}
for n in cand["captures"]:
    d = np.load(Path(bursts) / f"{n}.npz", allow_pickle=False)
    names = json.loads(str(d["info"]))["joint_names"]
    q_all = np.median(d["joints"][:, 1:], axis=0)
    q = q_all[[names.index(j) for j in JOINTS]]
    qn = (q_all[names.index("NeckYaw")], q_all[names.index("NeckPitch")])
    fr = arm.frames(q)
    w = (np.linalg.inv(fr[-1]) @ fr[5])[:3, 3]
    for k, dT in (("B1", B1), ("chain", np.eye(4)), ("old_dT", OLD)):
        refs[k][n] = {"T_C0_hand": (np.linalg.inv(dT @ neck.T_TB(*qn) @ rig.T_HC[0]) @ fr[-1]).tolist(),
                      "wrist_in_hand": w.tolist()}
res = {k: ys.score(r, cand["captures"]) for k, r in refs.items()}     # the CANDIDATE against each reference
res["method"] = cand.get("method")
# the do-nothing baseline: the plain chain itself scored against chain + B1 (25 Oct: the "chain" column above is the
# candidate vs a chain-only reference, not this; reading it as the baseline was wrong)
res["baseline_plain_chain_vs_B1"] = ys.score(refs["B1"], {n: {"T_C0_hand": refs["chain"][n]["T_C0_hand"],
                                                            "in_sample": c["in_sample"]} for n, c in cand["captures"].items()})
Path(outp).write_text(json.dumps(res, indent=1))
print(f"{res['method']}: verdict {res['B1']['verdict']} vs chain+B1 ({res['B1']['held_out_pass']}/{res['B1']['held_out']})")
for n, c in sorted(res["B1"]["captures"].items()):
    o, ch = res["old_dT"]["captures"][n], res["chain"]["captures"][n]
    bl = res["baseline_plain_chain_vs_B1"]["captures"][n]
    print(f"  {n:30s} {'held-out ' if c['counted'] else 'in-sample'} B1 {c['wrist_mm']:5.2f} mm {c['rot_deg']:5.3f} deg "
          f"{'pass' if c['pass'] else 'FAIL'} | vs old dT {o['wrist_mm']:5.2f} | vs chain-only ref {ch['wrist_mm']:5.2f} "
          f"| do-nothing baseline (plain chain vs B1) {bl['wrist_mm']:5.2f} mm {'pass' if bl['pass'] else 'FAIL'}")
print("do-nothing baseline verdict:", res["baseline_plain_chain_vs_B1"]["verdict"])
