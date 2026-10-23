#!/usr/bin/env python3
"""The arm session's read-only snapshot, run ON the robot (needs dma_ipc): one /actuals sample (every joint's angle
in deg and torque, the body IMUs' accelerometers), whether anything writes /desired (its last write's age and
whether it advances over one second) and the free disk under /root. Prints one JSON line. Writes nothing.
Used by step 90 (the start pose, gravity) and step 91 (the pre-checks)."""
import json
import math
import os
import time

import dma_ipc


def main():
    cfg = dma_ipc.ConfigQueue("/config", wait=False).read_latest()
    if cfg is None:
        raise SystemExit("/config not published")
    names = [cfg.joint_names(i) for i in range(cfg.num_joints)]
    imus = [str(n) for n in cfg.get_imu_names()][: int(cfg.num_imus)]
    s = dma_ipc.SensorQueue("/actuals", wait=False).read_latest()
    if s is None:
        raise SystemExit("/actuals not published")
    out = {"t_ns": s.timestamp_ns, "names": names,
           "q_deg": {n: math.degrees(float(s.joint_positions[i])) for i, n in enumerate(names)},
           "joint_torques": {n: float(s.joint_torques[i]) for i, n in enumerate(names)}, "imu": {}}
    for k, n in enumerate(imus):
        m = s.imu(k)
        out["imu"][n] = {a: float(getattr(m, a)) for a in ("accel_x", "accel_y", "accel_z")}
    q = dma_ipc.CommandDataQueue("/desired", wait=False)
    a = q.read_latest()
    time.sleep(1.0)
    b = q.read_latest()
    ta = a.timestamp_ns if a is not None else 0
    tb = b.timestamp_ns if b is not None else 0
    out["desired"] = {"age_s": (time.time_ns() - tb) / 1e9 if tb else float("inf"), "advancing": tb != ta}
    st = os.statvfs("/root")
    out["disk_free_gb"] = st.f_bavail * st.f_frsize / 1e9
    print(json.dumps(out))


if __name__ == "__main__":
    main()
