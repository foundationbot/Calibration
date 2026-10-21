"""Camera and IMU transforms of an OKVIS config (step 70's output), read with OpenCV's
FileStorage, the parser OKVIS itself uses (the `%YAML:1.0` header defeats PyYAML).

Rotations are cleaned to the nearest true rotation on load: a config prints them with a few
decimals (phantom21's T_BS is orthonormal to 8e-7 only), and the T_BS steps compare rotations to
0.1 deg. `max_nonortho` says how far the file's rotations were from orthonormal. A matrix the
cleaning would change by more than CLEAN_MAX (a typo, a mirror image, a column-major paste) is
refused rather than silently replaced by some other rotation.

Lenses: equidistant and radialtangential blocks give D = the first four numbers (what OKVIS reads); a kb4t block
(OKVIS's KB4T lens: equidistant k1..k4 plus tangential p1, p2, used whole by the stereo network's rectification, k1..k4
by VIO) gives all six, and must have exactly six, as OKVIS requires."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rigmath import nearest_rotation  # noqa: E402

LENS = {"equidistant": "pinhole-equi", "radialtangential": "pinhole-radtan", "kb4t": "pinhole-kb4t"}
KB4T_N = 6              # a kb4t block's distortion_coefficients: k1, k2, k3, k4, p1, p2, exactly
CLEAN_MAX = 1e-4        # largest element change the rotation cleaning may make (files: ~1e-6)


@dataclass
class OkvisRig:
    T_SC: list          # per OKVIS camera block: 4x4, camera in the IMU frame
    K: list             # per camera: 3x3
    D: list             # per camera: 4x1, the four coefficients OKVIS uses; kb4t: 6x1 (k1..k4, p1, p2)
    lens: str           # rigcfg lens name
    T_BS: np.ndarray    # IMU in the body (Head link) frame
    max_nonortho: float  # max |R R^T - I| over the file's rotations, before cleaning


def _seq(node, where: str, n: int | None = None) -> np.ndarray:
    if node.empty() or not node.isSeq():
        raise ValueError(f"{where}: missing or not a list")
    v = np.array([node.at(i).real() for i in range(node.size())], float)
    if n is not None and v.size != n:
        raise ValueError(f"{where}: {v.size} numbers, want {n}")
    return v


def _transform(node, where: str):
    """A row-major 4x4 rigid transform -> (cleaned 4x4, max |R R^T - I| of the file's rotation)."""
    T = _seq(node, where, 16).reshape(4, 4)
    if not np.allclose(T[3], [0, 0, 0, 1]):
        raise ValueError(f"{where}: bottom row {T[3].tolist()} is not [0, 0, 0, 1] (column-major?)")
    R = T[:3, :3]
    Rc = nearest_rotation(R)
    if np.abs(Rc - R).max() > CLEAN_MAX:
        raise ValueError(f"{where}: not a rotation (cleaning would change it by "
                         f"{np.abs(Rc - R).max():.2g}; a typo or a mirror image?)")
    T = T.copy()
    T[:3, :3] = Rc
    return T, float(np.abs(R @ R.T - np.eye(3)).max())


def load(path) -> OkvisRig:
    import cv2
    fs = cv2.FileStorage(str(Path(path)), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(path)
    cams = fs.getNode("cameras")
    if cams.empty() or cams.size() == 0:
        raise ValueError(f"{path}: no cameras")
    T_SC, K, D, kinds, errs = [], [], [], set(), []
    for i in range(cams.size()):
        c, where = cams.at(i), f"{path}: cameras[{i}]"
        model = c.getNode("cam_model")
        # an absent cam_model is accepted (OKVIS itself builds no camera from such a block): the
        # step-83 and end-to-end test configs omit it
        if not model.empty() and model.string() != "pinhole":
            raise ValueError(f"{where}: cam_model {model.string()!r}, only pinhole is supported")
        T, e = _transform(c.getNode("T_SC"), f"{where}.T_SC")
        T_SC.append(T)
        errs.append(e)
        f = _seq(c.getNode("focal_length"), f"{where}.focal_length", 2)
        pp = _seq(c.getNode("principal_point"), f"{where}.principal_point", 2)
        K.append(np.array([[f[0], 0.0, pp[0]], [0.0, f[1], pp[1]], [0.0, 0.0, 1.0]]))
        dtype = c.getNode("distortion_type")
        if dtype.empty() or not dtype.string():
            raise ValueError(f"{where}.distortion_type: missing")
        kinds.add(dtype.string())
        d = _seq(c.getNode("distortion_coefficients"), f"{where}.distortion_coefficients")
        if dtype.string() == "kb4t":
            if d.size != KB4T_N:
                raise ValueError(f"{where}.distortion_coefficients: {d.size} numbers; kb4t takes exactly {KB4T_N} "
                                 "[k1, k2, k3, k4, p1, p2]")
            D.append(d.reshape(KB4T_N, 1))
        else:
            if d.size < 4:
                raise ValueError(f"{where}.distortion_coefficients: {d.size} numbers, want at least 4")
            D.append(d[:4].reshape(4, 1))
    if len(kinds) != 1 or next(iter(kinds)) not in LENS:
        raise ValueError(f"{path}: distortion types {kinds} (want one of {sorted(LENS)})")
    T_BS, e = _transform(fs.getNode("imu_parameters").getNode("T_BS"), f"{path}: imu_parameters.T_BS")
    return OkvisRig(T_SC, K, D, LENS[kinds.pop()], T_BS, max(errs + [e]))
