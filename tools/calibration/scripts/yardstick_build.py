#!/usr/bin/env python3
"""Freeze phantom12's four right-arm plate sessions as the yardstick (yardstick.py; plan
docs/superpowers/plans/2026-10-25-tagfree-selfcal.md, Phase C, stereo-depth-eval branch).

Writes into out_dir:
  frames.npz      every plate frame of every session (armfit.Frame, flattened), for scoring a camera correction dT
  yardstick.json  per session the fit / unseen holds; per capture (burst) its hold, whether the plate fits used it,
                  and the hand (RightGripperYawLink) in Fit C's camera 0 under each plate model:
                    B1   the shared dT of all four sessions (dt_all.json), the reference a method must pass
                    M1   that session's own dT (armfit M1 on its fit holds)
                    M0   no correction (the chain alone)
                  T_C0_hand = (dT · T_TB(q_neck) · T_HC0)^-1 · T_Torso,GripperYaw(q_arm), q = the hold's median over
                  camera-0 frames; wrist_in_hand = RightWristRollLink's origin in the hand frame at q
  MANIFEST.json   sha256 of the two files above, of every input, and of every burst file

usage: yardstick_build.py task2_dir rv_dir rl_dir config.yaml urdf dt_all.json bursts_dir out_dir
  bursts_dir/<session>/<capture>.npz (1023c: arm-1023c-NN-pK, rv1024-*, rl1024-*)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import armchain  # noqa: E402
import armfit as af  # noqa: E402
import armfit_multi as am  # noqa: E402
import neckfk  # noqa: E402
import okviscfg  # noqa: E402
import plate_sessions  # noqa: E402
import yardstick as ys  # noqa: E402


def hand_in_cam0(obs, h, dT, arm, neck, rig):
    sel = [x for x in obs if x.hold == h and x.cam == 0]
    q = np.median([x.q_arm for x in sel], axis=0); qn = np.median([x.q_neck for x in sel], axis=0)
    fr = arm.frames(q)
    T = np.linalg.inv(dT @ neck.T_TB(*qn) @ rig.T_HC[0]) @ fr[-1]
    wrist = (np.linalg.inv(fr[-1]) @ fr[5])[:3, 3]                          # RightWristRollLink's origin
    return T, wrist, q, qn, len(sel)


def main(task2, rv, rl, cfg, urdf, dt_all, bursts, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    arm, neck = armchain.ArmChain(urdf), neckfk.NeckChain(urdf)
    assert arm.joints[5].child == "RightWristRollLink", arm.joints[5].child
    rig = af.rig_from_config(okviscfg.load(cfg))
    S = plate_sessions.load_all(task2, rv, rl)
    dB1 = am.fit_all([s.obs for s in S], arm, neck, rig, S[0].X0, holds=[s.fit for s in S])["dT"][0]
    logged = json.loads(Path(dt_all).read_text())["B1"]["dT"]["None"]          # the same fit, rounded
    assert np.allclose(dB1[:3, 3] * 1e3, logged["t_mm"], atol=0.01), (dB1, logged)
    sess, caps, flat = {}, {}, {k: [] for k in ("session", "hold", "cam", "t", "q_arm", "q_neck", "n")}
    P, uv = [], []
    for i, s in enumerate(S):
        m1 = af.fit(s.obs, arm, neck, rig, "M1", s.X0, holds=s.fit)
        sess[s.name] = {"fit_holds": sorted(int(h) for h in s.fit), "unseen_holds": [int(h) for h in s.unseen],
                        "M1_dT": m1["dT"].tolist(), "M1_rms_px": m1["rms_px"]}
        for n, h in sorted(s.captures.items()):
            row = {"session": s.name, "hold": int(h), "in_plate_fit": h in s.fit,
                   "burst": f"{s.name}/{n}.npz"}                                   # under bursts_dir
            for model, dT in (("B1", dB1), ("M1", m1["dT"]), ("M0", np.eye(4))):
                T, w, q, qn, nf = hand_in_cam0(s.obs, h, dT, arm, neck, rig)
                row[model] = {"T_C0_hand": T.tolist(), "wrist_in_hand": w.tolist()}
            row["q_deg"] = dict(zip(armchain.RIGHT_ARM, np.degrees(q).round(3).tolist()))
            row["neck_deg"] = {"NeckYaw": round(float(np.degrees(qn[0])), 3), "NeckPitch": round(float(np.degrees(qn[1])), 3)}
            row["n_frames_cam0"] = nf
            caps[n] = row
        for x in s.obs:
            for k, v in (("session", i), ("hold", x.hold), ("cam", x.cam), ("t", x.t), ("q_arm", x.q_arm),
                         ("q_neck", x.q_neck), ("n", len(x.P))):
                flat[k].append(v)
            P.append(x.P); uv.append(x.uv)
    np.savez_compressed(out / "frames.npz", **{k: np.asarray(v) for k, v in flat.items()},
                        P=np.concatenate(P), uv=np.concatenate(uv), sessions=np.array([s.name for s in S]))
    doc = {"what": "phantom12 right-arm plate yardstick v1 (plan 2026-10-25-tagfree-selfcal.md, Phase C)",
           "frame": "T_C0_hand = RightGripperYawLink in Fit C's camera 0 (T_BS·T_SC[0]) placed by dT·NeckFK",
           "reference": "B1", "B1_dT": dB1.tolist(), "gate": {"mm": ys.GATE_MM, "deg": ys.GATE_DEG,
                                                                 "share": ys.PASS_SHARE, "min_held_out": ys.MIN_HELD},
           "sessions": sess, "captures": caps}
    (out / "yardstick.json").write_text(json.dumps(doc, indent=1))
    inputs = [f"{task2}/{f}" for f in ("sweep_1023c_corners.npz", "sweep_1023d_corners.npz", "poses-1023c.json",
                                       "poses-1023d.json", "arm-session-1023c.jsonl", "arm-session-1023d.jsonl")]
    inputs += [f"{rv}/rv1024_corners.npz", f"{rv}/rv-log.txt", f"{rl}/rl1024_corners.npz", f"{rl}/rl-log.txt",
               cfg, urdf, dt_all] + [str(Path(bursts) / c["burst"]) for c in caps.values()]
    missing = [f for f in inputs if not Path(f).exists()]
    assert not missing, missing
    ys.write_manifest(out, ["frames.npz", "yardstick.json"]                   # relative: no home folder in git
                      + [os.path.relpath(Path(f).resolve(), out.resolve()) for f in inputs])
    print(f"{len(caps)} captures, {sum(not c['in_plate_fit'] for c in caps.values())} outside the plate fits; "
          f"{len(flat['t'])} frames -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:9])
