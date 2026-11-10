"""'Up' in camera-0 coordinates, for the placement guidance (read-only: nothing on the robot is changed).

9 Nov trial (phantom10): the head camera sits ~1.5 m above the floor and at its home pose looks steeply down, so a board
standing at the robot's feet is already ~1.1 m from the camera and "bring it closer" was the wrong advice. With 'up' in
the camera frame, the board's position from PnP splits into a height relative to the camera and a distance along the
floor (place.py).

The head IMU: DMA.common's ShmRingBuffer<RawImuFrame, 1024> at /dev/shm/raw_imu_actuals (the queue the robot kit's
imucam_fast.py reads through dma_ipc.RawImuQueue, IMU slot 0). Parsed here directly from the file so calibd needs no
consumer and the reader can be tested on a synthetic ring:
  header  192 bytes: u64 magic "DMA_QUEU", u64 version, u64 slot_size, u64 capacity; u64 head (writer position) at 64
  slot k  at 192 + k x slot_size: u64 sequence (= position + 1 once written), then RawImuFrame: i64 timestamp_ns,
          u32 cycle_counter, u8 num_imus, and from byte 16 of the frame 32 x (accel xyz, gyro xyz) float32
The accelerometer at rest measures the reaction to gravity, i.e. 'up' in the IMU frame; it is averaged over the newest
window_s of the ring (500 Hz: ~250 samples, read at once, no waiting).

R_SC (camera -> IMU) of camera 0 comes from the live OKVIS config: robot.env's OKVIS_CALIBRATION names it as the OKVIS
container sees it (/etc/okvis/...), which is /etc/phantomos/okvis/... on the host. up_cam = R_SC^T up_imu.

Gravity(read_acc, R_SC) reads once on first use and again every refresh_s (the neck stays at home in the trial); when
the IMU or the config is not available it gives None and says why (status()), and the guidance falls back to the
camera distance alone."""
import json
import re
import struct
import time
from pathlib import Path

import numpy as np

RING = "/dev/shm/raw_imu_actuals"
MAGIC = 0x444D415F51554555          # "DMA_QUEU" (ShmHeader::MAGIC)
HEADER_BYTES = 192                  # sizeof(ShmHeader), pinned in shm_queue.h
HEAD_OFFSET = 64
FRAME_OFFSET = 8                    # RawImuFrame after the slot's u64 sequence
IMUS_OFFSET = 16                    # imus[] inside RawImuFrame (i64 + u32 + u8, padded to 4)
IMU_BYTES = 24                      # 6 x float32
MAX_IMUS = 32
HEAD_IMU = 0                        # the head IMU's slot (imucam_fast.py: f.imu(0))
WINDOW_S = 0.5
REFRESH_S = 30.0
G = 9.81
G_TOL = 1.0                         # |acc| further than this from G (m/s^2): not a still reading of gravity
ENV_FILES = ("/etc/phantomos/okvis/robot.env", "/etc/phantom/robot.env")
MOUNTS = (("/etc/okvis", "/etc/phantomos/okvis"),)   # OKVIS container path -> host path
FALLBACK = "placement guidance uses the camera distance only"


def ring_accel(path=RING, slot=HEAD_IMU, window_s=WINDOW_S):
    """(mean accelerometer xyz of IMU `slot` over the newest window_s of the ring, samples used); ValueError for a
    file that is not a DMA queue or holds no samples"""
    b = Path(path).read_bytes()
    magic, _, slot_size, cap = struct.unpack_from("<QQQQ", b, 0)
    if magic != MAGIC:
        raise ValueError(f"{path}: magic {magic:#x} is not a DMA queue")
    if not 0 <= slot < MAX_IMUS or slot_size < FRAME_OFFSET + IMUS_OFFSET + IMU_BYTES * (slot + 1):
        raise ValueError(f"{path}: slot {slot} does not fit a {slot_size}-byte slot")
    head = struct.unpack_from("<Q", b, HEAD_OFFSET)[0]
    t, acc = [], []
    for pos in range(head - 1, max(-1, head - 1 - cap), -1):
        off = HEADER_BYTES + (pos % cap) * slot_size
        if off + slot_size > len(b):
            raise ValueError(f"{path}: shorter than its header says")
        if struct.unpack_from("<Q", b, off)[0] != pos + 1:
            continue                                    # torn or overwritten while we read
        ts = struct.unpack_from("<q", b, off + FRAME_OFFSET)[0]
        if t and ts <= t[0] - window_s * 1e9:
            break
        t.append(ts)
        acc.append(struct.unpack_from("<3f", b, off + FRAME_OFFSET + IMUS_OFFSET + IMU_BYTES * slot))
    if not acc:
        raise ValueError(f"{path}: no samples")
    return np.mean(np.array(acc, float), axis=0), len(acc)


def okvis_config_path(env_text, mounts=MOUNTS):
    """the host path of robot.env's OKVIS_CALIBRATION (None when it is not set)"""
    for ln in str(env_text).splitlines():
        m = re.match(r"\s*(?:export\s+)?OKVIS_CALIBRATION\s*=\s*(.*)$", ln)
        if not m:
            continue
        v = m.group(1).split("#")[0].strip().strip("'\"")
        if not v:
            return None
        for inside, host in mounts:
            if v == inside or v.startswith(inside.rstrip("/") + "/"):
                return Path(str(host).rstrip("/") + v[len(inside.rstrip("/")):])
        return Path(v)
    return None


def live_r_sc(env_files=ENV_FILES, mounts=MOUNTS):
    """(R_SC of camera 0, config path) from the live OKVIS config; FileNotFoundError / ValueError says what is missing"""
    for e in env_files:
        e = Path(e)
        if e.is_file():
            cfg = okvis_config_path(e.read_text(), mounts)
            if cfg is None:
                raise ValueError(f"{e}: no OKVIS_CALIBRATION")
            from calibtool import paths  # noqa: F401  (scripts folder on sys.path)
            import okviscfg
            return np.asarray(okviscfg.load(cfg).T_SC[0])[:3, :3], cfg
    raise FileNotFoundError("robot.env not found (" + ", ".join(str(e) for e in env_files) + ")")


def up_in_camera(acc_imu, R_SC):
    """unit 'up' in camera coordinates (x right, y down, z forward) from the IMU's still acceleration"""
    a = np.asarray(acc_imu, float)
    u = np.asarray(R_SC, float).T @ (a / np.linalg.norm(a))
    return u / np.linalg.norm(u)


def pitch_down_deg(up) -> float:
    """how far the optical axis points below the horizon (deg; negative: above)"""
    return float(-np.degrees(np.arcsin(np.clip(np.asarray(up, float)[2], -1.0, 1.0))))


class Gravity:
    """up() -> unit 'up' in camera 0, or None (status() says why)"""

    def __init__(self, read_acc, R_SC, clock=time.monotonic, refresh_s=REFRESH_S, note=""):
        self.read_acc, self.R_SC, self.clock, self.refresh_s = read_acc, R_SC, clock, refresh_s
        self._up = None
        self._t = None
        self._n = 0
        self._acc = None
        self.note = note
        self._base_note = note

    def _read(self):
        try:
            acc, n = self.read_acc()
            acc = np.asarray(acc, float)
            g = float(np.linalg.norm(acc))
            if abs(g - G) > G_TOL:
                raise ValueError(f"|acc| {g:.2f} m/s^2, not 9.81: the head is not still")
            return up_in_camera(acc, self.R_SC), acc, int(n)
        except Exception as e:  # noqa: BLE001
            return None, f"{type(e).__name__}: {e}", 0

    def up(self):
        if self.R_SC is None:
            self.note = (self._base_note or "no camera-to-IMU rotation") + "; " + FALLBACK
            return None
        t = self.clock()
        if self._t is not None and t - self._t < self.refresh_s:
            return self._up
        up, acc, n = self._read()
        self._t = t
        if up is not None:
            self._up, self._acc, self._n = up, acc, n
            self.note = f"IMU slot {HEAD_IMU}, {n} samples"
        elif self._up is not None:
            self.note = f"IMU read failed ({acc}); kept the earlier reading"
        else:
            self.note = f"IMU not read ({acc}); {FALLBACK}"
        return self._up

    def status(self) -> dict:
        u = self._up
        return {"ok": u is not None, "up": None if u is None else [round(float(x), 4) for x in u],
                "pitch_down_deg": None if u is None else round(pitch_down_deg(u), 1), "note": self.note}


def from_robot(ring=RING, env_files=ENV_FILES, mounts=MOUNTS, clock=time.monotonic, refresh_s=REFRESH_S) -> Gravity:
    """the robot's Gravity (never raises: a missing IMU or config gives a Gravity whose up() is None)"""
    try:
        R_SC, cfg = live_r_sc(env_files, mounts)
        note = f"R_SC from {cfg}"
    except Exception as e:  # noqa: BLE001
        R_SC, note = None, f"no OKVIS config ({type(e).__name__}: {e})"
    return Gravity(lambda: ring_accel(ring), R_SC, clock=clock, refresh_s=refresh_s, note=note)


def _main():
    """read-only probe: python -m calibtool.gravity -> 'up' in camera 0 and the camera pitch it implies"""
    g = from_robot()
    up = g.up()
    print(json.dumps(g.status()))
    return 0 if up is not None else 1


if __name__ == "__main__":
    raise SystemExit(_main())
