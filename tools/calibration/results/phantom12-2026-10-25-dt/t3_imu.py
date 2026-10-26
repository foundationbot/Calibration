"""T3 of plan 2026-10-25-tagfree-selfcal.md (decisions fixed 25 Oct before data): can the body IMUs check the head
camera's rotation without tags?

Both IMUs see the same gravity. With the head on the neck chain, the NeckPitch IMU (OKVIS's IMU, S; T_BS = S in the
Head link) should read
    s = R_HS^T · R_TB(q_neck)^T · R^T · g_T,     g_T = R_T,TorsoIMU · u
where u is the Torso IMU's gravity direction, R_T,TorsoIMU the URDF mount (TorsoIMUJoint), and R a constant
torso-side rotation (the camera correction's rotation if the error is in the torso/neck chain; NOTE: a Torso-IMU mount
error is also torso-side and constant, so R = dT_R ∘ (mount error): not separable by gravity).
Fit R (rotation vector; its component about the torso's up axis is weakly observable); report it, the per-burst
residual angle and its spread across neck pitch/yaw (decision: usable only if flat, ≤ 0.05° p-p across neck bins),
and compare R's tilt with B1's rotation (0.188, 0.001, −0.015)°.

usage: t3_imu.py urdf out.json burst.npz ...   (bursts with body_imu: Pelvis, Torso, NeckPitch blocks of 13)
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

import neckfk
import rigmath as rm

B1_ROTVEC_DEG = np.array([0.188, 0.001, -0.015])
urdf, outp, files = sys.argv[1], sys.argv[2], sys.argv[3:]
neck = neckfk.NeckChain(urdf)
R_T_TIMU = rm.rot_axis((0, 0, 1), -np.pi / 2)                     # URDF TorsoIMUJoint rpy (0, 0, -90 deg)


def block(b, names, name):
    i = names.index(name)
    return b[:, 1 + 13 * i: 1 + 13 * (i + 1)]


rows = []
for f in files:
    d = np.load(f, allow_pickle=False)
    if "body_imu" not in d.files:
        continue
    info = json.loads(str(d["info"]))
    names = info.get("body_imu_names", ["Pelvis", "Torso", "NeckPitch"])
    b = d["body_imu"]
    aT = np.median(block(b, names, "Torso")[:, 7:10], axis=0)
    aS = np.median(block(b, names, "NeckPitch")[:, 7:10], axis=0)
    gyro = max(np.abs(block(b, names, "Torso")[:, 10:13]).max(), np.abs(block(b, names, "NeckPitch")[:, 10:13]).max())
    jn = info["joint_names"]
    q = np.median(d["joints"][:, 1:], axis=0)
    qn = (q[jn.index("NeckYaw")], q[jn.index("NeckPitch")])
    rows.append({"file": Path(f).name, "u": aT / np.linalg.norm(aT), "s": aS / np.linalg.norm(aS), "qn": qn,
                 "R_HS": np.asarray(d["T_BS"], float).reshape(4, 4)[:3, :3], "gyro_max": float(gyro),
                 "g_norm": [float(np.linalg.norm(aT)), float(np.linalg.norm(aS))]})


def pred(r, w):
    R = rm.exp_rot(w)
    return r["R_HS"].T @ neck.T_TB(*r["qn"])[:3, :3].T @ R.T @ R_T_TIMU @ r["u"]


def ang(a, b):
    return float(np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))


fit = least_squares(lambda w: np.concatenate([np.cross(pred(r, w), r["s"]) for r in rows]), np.zeros(3))
w = fit.x
res0 = [ang(pred(r, np.zeros(3)), r["s"]) for r in rows]
res = [ang(pred(r, w), r["s"]) for r in rows]
bins = {}
for r, e in zip(rows, res):
    k = (int(round(np.degrees(r["qn"][1]) / 5) * 5), int(round(np.degrees(r["qn"][0]) / 10) * 10))
    bins.setdefault(k, []).append(e)
means = {f"pitch {k[0]} yaw {k[1]}": round(float(np.mean(v)), 4) for k, v in sorted(bins.items())}
pp = max(np.mean(v) for v in bins.values()) - min(np.mean(v) for v in bins.values())
out = {"n_bursts": len(rows), "fit_rotvec_deg": np.degrees(w).round(4).tolist(), "fit_angle_deg": round(float(np.degrees(np.linalg.norm(w))), 4),
       "B1_rotvec_deg": B1_ROTVEC_DEG.tolist(),
       "tilt_diff_deg (x,y vs B1)": round(float(np.linalg.norm(np.degrees(w)[:2] - B1_ROTVEC_DEG[:2])), 4),
       "residual_deg_before": {"rms": round(float(np.sqrt(np.mean(np.square(res0)))), 4), "max": round(max(res0), 4)},
       "residual_deg_after": {"rms": round(float(np.sqrt(np.mean(np.square(res)))), 4), "max": round(max(res), 4)},
       "residual_by_neck_bin": means, "residual_bin_pp_deg": round(float(pp), 4),
       "flat (<= 0.05 deg p-p)": bool(pp <= 0.05),
       "gyro_max_rad_s": round(max(r["gyro_max"] for r in rows), 4),
       "per_burst": [{"file": r["file"], "neck_pitch": round(float(np.degrees(r["qn"][1])), 2),
                      "neck_yaw": round(float(np.degrees(r["qn"][0])), 2), "res_before": round(a, 4), "res_after": round(e, 4),
                      "g_norm": [round(x, 3) for x in r["g_norm"]]} for r, a, e in zip(rows, res0, res)]}
Path(outp).write_text(json.dumps(out, indent=1))
print({k: v for k, v in out.items() if k != "per_burst"})
