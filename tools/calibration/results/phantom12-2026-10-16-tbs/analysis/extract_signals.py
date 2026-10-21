"""Read-only on the takes: joints (full rate), holds, head IMU (export), torso + pelvis IMUs (recorder
copy) of one take, saved to /w/tbs/<take>-signals.npz for offline analysis."""
import sys
from pathlib import Path
import numpy as np
import dmarec_imu, necktake
take = sys.argv[1]; W = Path("/w/takes")
j = necktake.read_joints(W / take / "session.mcap"); holds = necktake.find_holds(j)
th, wh, ah = necktake.read_imu(W / f"{take}-euroc") if hasattr(necktake, "read_imu") else (None, None, None)
segs = sorted((W / f"{take}-dmarec").glob("recording_*.mcap"))
out = {"jt": j.t, "yaw": j.yaw, "pitch": j.pitch,
       "holds": np.array([[h.t0, h.t1, h.yaw, h.pitch, h.approach_yaw, h.approach_pitch, h.drift_rad] for h in holds], float),
       "ht": np.asarray(th), "hg": np.asarray(wh)}
d = np.loadtxt(W / f"{take}-euroc" / "imu0" / "data.csv", delimiter=",", comments="#")
out["ht"], out["hg"], out["ha"] = d[:, 0].astype(np.int64), d[:, 1:4], d[:, 4:7]
for idx, nm in ((9, "torso"), (10, "pelvis")):
    t, g, a, err = dmarec_imu.read_raw_imu(segs, idx, int(j.t[0]) - int(5e9), int(j.t[-1]) + int(5e9))
    out[f"{nm}_t"], out[f"{nm}_g"], out[f"{nm}_a"] = t, g, a
np.savez_compressed(f"/w/tbs/{take}-signals.npz", **out)
print(take, {k: np.shape(v) for k, v in out.items()})
