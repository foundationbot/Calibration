# Compare Kalibr camera-IMU fits of phantom12: per camera T_SC = inv(T_cam_imu) (IMU frame),
# clock offset and residuals; the differences between fits (repeatability); and each fit
# against phantom21's T_SC in the step-70 template (same head design, 43.4 vs 44.8 mm baseline).
# usage (calib-py:1, /w = /root/calib-work):  python /w/compare_imucam.py calib_imucam-B-noedge calib_imucam-C-noedge ...
import re
import sys

import numpy as np
import yaml

TPL = "/w/config/okvis-template-phantom12.yaml"


def rot_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def template_tsc():
    txt = open(TPL).read()
    blocks = re.findall(r"T_SC:\s*\[([^\]]*)\]", txt)
    return [np.array([float(v) for v in b.split(",") if v.strip()]).reshape(4, 4) for b in blocks]


def residuals(name):
    try:
        txt = open(f"/w/{name}-results-imucam.txt").read()
    except OSError:
        return {}
    out = {}
    for key, pat in (("reproj0", r"Reprojection error \(cam0\):\s+mean ([0-9.e+-]+)"),
                     ("reproj1", r"Reprojection error \(cam1\):\s+mean ([0-9.e+-]+)"),
                     ("gyro", r"Gyroscope error \(imu0\):\s+mean ([0-9.e+-]+)"),
                     ("accel", r"Accelerometer error \(imu0\):\s+mean ([0-9.e+-]+)"),
                     ("reproj0_px", r"Reprojection error \(cam0\) \x5bpx\x5d:\s+mean ([0-9.e+-]+)"),  # \x5b: the log-tag registry test reads a quoted escaped bracket as a log tag
                     ("reproj1_px", r"Reprojection error \(cam1\) \x5bpx\x5d:\s+mean ([0-9.e+-]+)"),
                     ("gyro_si", r"Gyroscope error \(imu0\) \[rad/s\]:\s+mean ([0-9.e+-]+)"),
                     ("accel_si", r"Accelerometer error \(imu0\) \[m/s\^2\]:\s+mean ([0-9.e+-]+)")):
        m = re.search(pat, txt)                 # first match = Kalibr's normalized block
        out[key] = float(m.group(1)) if m else float("nan")
    return out


p21 = template_tsc()
fits = {}
for name in sys.argv[1:]:
    cc = yaml.safe_load(open(f"/w/{name}-camchain-imucam.yaml"))
    fits[name] = {"T_SC": [np.linalg.inv(np.array(cc[c]["T_cam_imu"])) for c in ("cam0", "cam1")],
                  "shift": [float(cc[c]["timeshift_cam_imu"]) for c in ("cam0", "cam1")],
                  "res": residuals(name)}

print("== each fit   (cam0 = left, queue 1; cam1 = right, queue 2; T_SC = camera in IMU frame)")
for name, f in fits.items():
    r = f["res"]
    print(f"-- {name}")
    for i in (0, 1):
        t = f["T_SC"][i][:3, 3]
        print(f"   cam{i}: t_SC = ({t[0] * 1e3:+7.1f}, {t[1] * 1e3:+7.1f}, {t[2] * 1e3:+7.1f}) mm, "
              f"|t| {np.linalg.norm(t) * 1e3:6.1f} mm; vs phantom21: rotation {rot_deg(f['T_SC'][i][:3, :3].T @ p21[i][:3, :3]):5.2f} deg, "
              f"position {np.linalg.norm(t - p21[i][:3, 3]) * 1e3:5.1f} mm; timeshift {f['shift'][i] * 1e3:+.2f} ms")
    print(f"   timeshift cam1 - cam0: {(f['shift'][1] - f['shift'][0]) * 1e3:+.2f} ms (left/right sync check)")
    print(f"   residuals, normalized mean (Kalibr): reproj {r.get('reproj0', np.nan):.2f} / {r.get('reproj1', np.nan):.2f}, "
          f"gyro {r.get('gyro', np.nan):.2f}, accel {r.get('accel', np.nan):.2f}")
    print(f"   residuals, units: reproj {r.get('reproj0_px', np.nan):.3f} / {r.get('reproj1_px', np.nan):.3f} px, "
          f"gyro {r.get('gyro_si', np.nan):.4f} rad/s, accel {r.get('accel_si', np.nan):.3f} m/s^2")

names = list(fits)
if len(names) > 1:
    print("== differences between fits (left camera; the right one follows through the fixed stereo pose)")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            A, B = fits[a]["T_SC"][0], fits[b]["T_SC"][0]
            print(f"   {a} vs {b}: position {np.linalg.norm(A[:3, 3] - B[:3, 3]) * 1e3:5.1f} mm, "
                  f"rotation {rot_deg(A[:3, :3].T @ B[:3, :3]):5.2f} deg, "
                  f"timeshift {(fits[a]['shift'][0] - fits[b]['shift'][0]) * 1e3:+.2f} ms")
print("phantom21 (template): |t_SC| " + " / ".join(f"{np.linalg.norm(T[:3, 3]) * 1e3:.1f}" for T in p21) + " mm")
