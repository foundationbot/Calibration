"""(Ported from OKVIS2-X tools/burst_pointcloud/burst_io.py, c5b4c6b0; the C++ to_native() dropped: the
depth fusion is armcal/fuse.py.) The burst file: N consecutive depth + RGB frames of one rectified camera, one .npz (design §4.1).

The live grabber, make_fixture.py (recordings) and every consumer use this one format, so live and offline differ only
in where the file comes from. Keys: depth (N,H,W) float32 m (0 = no return); rgb (N,H,W,3) uint8 RGB, pixel-aligned
with depth; K (4,) fx fy cx cy; T_SC (4,4) camera -> IMU; stamp_ns, seq (N,) int64; optional sigma (N,H,W) float32 m;
optional imu (M,7) float64: t_ns, gyro x y z (rad/s), acc x y z (m/s^2), in the camera's IMU (S) frame, any M >= 1 (save()
puts its summary, info['imu'], beside the other stats);
optional joints (M,1+J) float64: column 0 = sample time ns (wall clock, as stamp_ns), columns 1.. = joint angles (rad)
in the order of info['joint_names'] (J names, required with joints), sampled during the capture (frames.py interpolates
them at the frame stamps); optional T_BS (4,4) float64: IMU S -> Head link B (p_B = T_BS p_S), from the OKVIS config
named in info['okvis_config'] (its sha256 in info['okvis_config_sha256']);
optional body_imu (M,1+13K) float64: column 0 = sample time ns (as joints), then for each IMU in info['body_imu_names']
the 13 BODY_IMU_FIELDS from /actuals' ImuSensorData: projected gravity gx gy gz (unit, IMU frame, the IMU's own
filter), orientation qw qx qy qz (the IMU in the world), accelerometer ax ay az (m/s^2, specific force), gyro
wx wy wz (rad/s);
info: a JSON string (source, pids, offsets, motion, ...). No size is fixed: H and W come from the arrays."""
from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REQUIRED = ("depth", "rgb", "K", "T_SC", "stamp_ns", "seq")
BODY_IMU_FIELDS = ("gx", "gy", "gz", "qw", "qx", "qy", "qz", "ax", "ay", "az", "wx", "wy", "wz")


@dataclass
class Burst:
    depth: np.ndarray
    rgb: np.ndarray
    K: np.ndarray
    T_SC: np.ndarray
    stamp_ns: np.ndarray
    seq: np.ndarray
    sigma: np.ndarray | None = None
    info: dict = field(default_factory=dict)
    imu: np.ndarray | None = None
    joints: np.ndarray | None = None
    T_BS: np.ndarray | None = None
    body_imu: np.ndarray | None = None

    @property
    def frames(self) -> int:
        return int(np.asarray(self.depth).shape[0])


def _bad(key: str, why: str):
    raise ValueError(f"burst: {key}: {why}")


def validate(b: Burst) -> Burst:
    """b with canonical dtypes, or a ValueError naming the first bad key"""
    depth = np.asarray(b.depth)
    if depth.ndim != 3 or 0 in depth.shape:
        _bad("depth", f"expected (N,H,W) with N, H, W >= 1, got {depth.shape}")
    n, h, w = depth.shape
    rgb = np.asarray(b.rgb)
    if rgb.shape != (n, h, w, 3):
        _bad("rgb", f"expected shape {(n, h, w, 3)}, got {rgb.shape}")
    if rgb.dtype != np.uint8:
        _bad("rgb", f"expected uint8, got {rgb.dtype}")
    sigma = None if b.sigma is None else np.asarray(b.sigma)
    if sigma is not None and sigma.shape != (n, h, w):
        _bad("sigma", f"expected shape {(n, h, w)}, got {sigma.shape}")
    K = np.asarray(b.K, np.float64).reshape(-1)
    if K.shape != (4,) or not (K[0] > 0 and K[1] > 0):
        _bad("K", f"expected fx fy cx cy with positive focal lengths, got {K.tolist()}")
    T = np.asarray(b.T_SC, np.float64)
    if T.shape != (4, 4):
        _bad("T_SC", f"expected (4, 4), got {T.shape}")
    stamp, seq = np.asarray(b.stamp_ns, np.int64), np.asarray(b.seq, np.int64)
    for key, a in (("stamp_ns", stamp), ("seq", seq)):
        if a.shape != (n,):
            _bad(key, f"expected ({n},), got {a.shape}")
    imu = None
    if b.imu is not None:
        imu = np.asarray(b.imu)
        if imu.ndim != 2 or imu.shape[1] != 7 or imu.shape[0] < 1:
            _bad("imu", f"expected (M,7) with M >= 1, got {imu.shape}")
        if imu.dtype != np.float64:
            _bad("imu", f"expected float64, got {imu.dtype}")
        imu = np.ascontiguousarray(imu)
    joints = None
    if b.joints is not None:
        joints = np.asarray(b.joints)
        if joints.ndim != 2 or joints.shape[0] < 1 or joints.shape[1] < 2:
            _bad("joints", f"expected (M,1+J) with M, J >= 1, got {joints.shape}")
        names = b.info.get("joint_names")
        if not isinstance(names, list) or not all(isinstance(x, str) for x in names):
            _bad("joints", "info['joint_names'] (a list of names, one per angle column) is required with joints")
        if len(names) != joints.shape[1] - 1:
            _bad("joints", f"{joints.shape[1] - 1} angle columns but {len(names)} joint_names")
        joints = np.ascontiguousarray(joints, np.float64)
    T_BS = None
    if b.T_BS is not None:
        T_BS = np.asarray(b.T_BS, np.float64)
        if T_BS.shape != (4, 4):
            _bad("T_BS", f"expected (4, 4), got {T_BS.shape}")
    body_imu = None
    if b.body_imu is not None:
        body_imu = np.asarray(b.body_imu)
        names = b.info.get("body_imu_names")
        if not isinstance(names, list) or not names or not all(isinstance(x, str) for x in names):
            _bad("body_imu", "info['body_imu_names'] (a non-empty list of IMU names) is required with body_imu")
        want = 1 + len(BODY_IMU_FIELDS) * len(names)
        if body_imu.ndim != 2 or body_imu.shape[0] < 1 or body_imu.shape[1] != want:
            _bad("body_imu", f"expected (M,{want}) with M >= 1 for {len(names)} IMU(s), got {body_imu.shape}")
        if body_imu.dtype != np.float64:
            _bad("body_imu", f"expected float64, got {body_imu.dtype}")
        body_imu = np.ascontiguousarray(body_imu)
    return Burst(np.ascontiguousarray(depth, np.float32), np.ascontiguousarray(rgb), K, T, stamp, seq,
                 None if sigma is None else np.ascontiguousarray(sigma, np.float32), dict(b.info), imu, joints, T_BS,
                 body_imu)


def imu_summary(imu) -> dict:
    """summary stats of an (M,7) imu array, for info['imu']"""
    imu = np.asarray(imu, np.float64)
    return {"samples": int(len(imu)), "span_ms": float(imu[-1, 0] - imu[0, 0]) / 1e6,
            "gyro_rms_rad_s": float(np.sqrt(np.mean(np.sum(imu[:, 1:4] ** 2, axis=1)))),
            "acc_mean_norm_m_s2": float(np.mean(np.linalg.norm(imu[:, 4:7], axis=1)))}


def _arrays(b: Burst) -> dict:
    """what the .npz holds for b (validated); the one writer behind save() and to_bytes()"""
    b = validate(b)
    arrays = {"depth": b.depth, "rgb": b.rgb, "K": b.K, "T_SC": b.T_SC, "stamp_ns": b.stamp_ns, "seq": b.seq,
              "info": np.array(json.dumps(b.info))}
    if b.sigma is not None:
        arrays["sigma"] = b.sigma
    if b.imu is not None:
        arrays["imu"] = b.imu
        arrays["info"] = np.array(json.dumps({**b.info, "imu": imu_summary(b.imu)}))
    if b.joints is not None:
        arrays["joints"] = b.joints
    if b.T_BS is not None:
        arrays["T_BS"] = b.T_BS
    if b.body_imu is not None:
        arrays["body_imu"] = b.body_imu
    return arrays


def save(path, b: Burst) -> Path:
    """write b (validated) to path, which must end in .npz; uncompressed: float depth barely compresses"""
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError(f"burst: {path}: the file name must end in .npz")
    arrays = _arrays(b)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return path


def to_bytes(b: Burst) -> bytes:
    """the burst .npz in memory"""
    buf = io.BytesIO()
    np.savez(buf, **_arrays(b))
    return buf.getvalue()


def _read(z, where) -> Burst:
    missing = [k for k in REQUIRED if k not in z.files]
    if missing:
        raise ValueError(f"burst: {where}: missing {', '.join(missing)}")
    info = json.loads(str(z["info"])) if "info" in z.files else {}
    return validate(Burst(z["depth"], z["rgb"], z["K"], z["T_SC"], z["stamp_ns"], z["seq"],
                          z["sigma"] if "sigma" in z.files else None, info,
                          z["imu"] if "imu" in z.files else None,
                          z["joints"] if "joints" in z.files else None,
                          z["T_BS"] if "T_BS" in z.files else None,
                          z["body_imu"] if "body_imu" in z.files else None))


def load(path) -> Burst:
    with np.load(path, allow_pickle=False) as z:
        return _read(z, path)


def from_bytes(data: bytes) -> Burst:
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        return _read(z, "<bytes>")
