"""A record-app neck take: neck angles from session.mcap, IMU and frames from its EuRoC export
(tools/mcap_to_euroc.py), and the held poses, the stretches where the neck is still.
All times are host-clock nanoseconds, the clock every recorded stream shares."""
from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]   # repo (or /root/calib-tools) root: has interface/


@dataclass
class Joints:
    t: np.ndarray       # int64 ns
    yaw: np.ndarray     # rad, NeckYaw
    pitch: np.ndarray   # rad, NeckPitch


@dataclass
class Hold:
    t0: int                 # ns, after trimming
    t1: int
    yaw: float              # median over the hold, rad
    pitch: float
    approach_yaw: int = 0   # +1: last moved up into the hold, -1: down, 0: it did not move
    approach_pitch: int = 0
    drift_rad: float = 0.0  # largest peak-to-peak of either joint over the hold: sag, plus tremor


def joints_from_messages(messages, names, yaw="NeckYaw", pitch="NeckPitch") -> Joints:
    iy, ip = names.index(yaw), names.index(pitch)
    rows = sorted((int(m["timestamp_ns"]), float(m["positions"][iy]), float(m["positions"][ip]))
                  for m in messages)
    return Joints(np.array([r[0] for r in rows], np.int64),
                  np.array([r[1] for r in rows]), np.array([r[2] for r in rows]))


def read_joints(session: Path, names: list | None = None) -> Joints:
    """NeckYaw and NeckPitch from the record app's /joints. Joint names come from the session's
    'robot' metadata record unless given (the record app writes none if /config was unreadable)."""
    if str(ROOT / "interface") not in sys.path:
        sys.path.insert(0, str(ROOT / "interface"))
    import mcap_reader
    msgs = []

    def on(ch, log_t, pub_t, data):
        if ch.topic == "/joints":
            msgs.append(json.loads(bytes(data)))

    s = mcap_reader.read_session(str(session), on_message=on)
    if not msgs:
        raise ValueError(f"{session}: no /joints messages (the record app could not open /actuals)")
    if names is None:
        if "robot" not in s.metadata or "joint_names" not in s.metadata["robot"]:
            raise ValueError(f"{session}: no joint names (metadata 'robot'/'joint_names'): "
                             "the record app could not read /config during the take; pass names")
        names = s.metadata["robot"]["joint_names"].split(",")
    return joints_from_messages(msgs, names)


def read_imu(export: Path):
    """-> (t int64 ns, gyro N x 3 rad/s, accel N x 3 m/s^2) from imu0/data.csv."""
    t, w, a = [], [], []
    with open(Path(export) / "imu0" / "data.csv") as f:
        for r in csv.reader(f):
            if not r or r[0].startswith("#"):
                continue
            t.append(int(r[0]))
            w.append([float(v) for v in r[1:4]])
            a.append([float(v) for v in r[4:7]])
    return np.array(t, np.int64), np.array(w), np.array(a)


def read_frames(export: Path, record_dir: str):
    """-> (t int64 ns, list of png paths) for one camera folder of the export (cam1, cam2)."""
    with open(Path(export) / record_dir / "data.csv") as f:
        rows = [r for r in csv.reader(f) if r and not r[0].startswith("#")]
    return (np.array([int(r[0]) for r in rows], np.int64),
            [Path(export) / record_dir / "data" / r[1].strip() for r in rows])


def _sec(j: Joints, t_ns) -> np.ndarray:
    return (np.asarray(t_ns, np.int64) - j.t[0]) / 1e9


def at(j: Joints, t_ns):
    """Neck angles at arbitrary stamps, by linear interpolation -> (yaw, pitch)."""
    tj, t = _sec(j, j.t), _sec(j, t_ns)
    return np.interp(t, tj, j.yaw), np.interp(t, tj, j.pitch)


def rates_at(j: Joints, t_ns, half_window_s: float = 0.02):
    """Central-difference joint speeds over +-half_window_s at arbitrary stamps -> (qd_yaw, qd_pitch).
    Within half_window_s of the take's ends the clamped interpolation lowers them; outside it, 0."""
    tj, t, h = _sec(j, j.t), _sec(j, t_ns), half_window_s
    qdy = (np.interp(t + h, tj, j.yaw) - np.interp(t - h, tj, j.yaw)) / (2 * h)
    qdp = (np.interp(t + h, tj, j.pitch) - np.interp(t - h, tj, j.pitch)) / (2 * h)
    return qdy, qdp


def _approach(j: Joints, t_ns: int, back_s: float, min_rad: float):
    """Side each joint last moved from into the pose reached at t_ns: the sign of (settled angle -
    the latest sample within back_s that was at least min_rad away), 0 if none was. The last
    motion, not the net change: a hand overshoots and comes back."""
    m = (j.t >= t_ns - round(back_s * 1e9)) & (j.t <= t_ns)
    y1, p1 = at(j, np.array([t_ns]))

    def side(q, q1):
        d = q1 - q[::-1]
        far = np.flatnonzero(np.abs(d) >= min_rad)
        return 0 if far.size == 0 else int(np.sign(d[far[0]]))
    return side(j.yaw[m], y1[0]), side(j.pitch[m], p1[0])


def find_holds(j: Joints, rate_max: float = 0.03, half_window_s: float = 0.15, min_s: float = 1.0,
               trim_s: float = 0.2, back_s: float = 1.0, approach_min_rad: float = 0.02) -> list[Hold]:
    """Stretches where both neck joints turn slower than rate_max [rad/s] for at least min_s after
    trimming trim_s off each end (the hand settling). Speeds are taken over +-half_window_s, so a
    hand's tremor (~10 Hz, a few hundredths of a degree) does not break a hold. Each hold also
    records the side each joint last moved from (see _approach) and its drift."""
    qdy, qdp = rates_at(j, j.t, half_window_s)
    still = np.maximum(np.abs(qdy), np.abs(qdp)) < rate_max
    holds, i, n = [], 0, len(still)
    while i < n:
        if not still[i]:
            i += 1
            continue
        k = i
        while k < n and still[k]:
            k += 1
        t0, t1 = int(j.t[i]) + round(trim_s * 1e9), int(j.t[k - 1]) - round(trim_s * 1e9)
        if (t1 - t0) / 1e9 >= min_s:
            m = (j.t >= t0) & (j.t <= t1)
            ay, ap = _approach(j, int(j.t[i]), back_s, approach_min_rad)
            drift = max(np.ptp(j.yaw[m]), np.ptp(j.pitch[m]))
            holds.append(Hold(t0, t1, float(np.median(j.yaw[m])), float(np.median(j.pitch[m])), ay, ap, float(drift)))
        i = k
    return holds
