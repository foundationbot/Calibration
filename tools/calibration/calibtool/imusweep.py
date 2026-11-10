"""The neck sweep that feeds calibtool.imu: the kit's imucam_fast.py without its robot-only imports (design plan 9.3).

imucam_fast.py exec'd the robot kit's neckgrid script and read dma_ipc queues directly. Here the robot side is the
`SweepRig` protocol, which M5's neck driver implements (dma_motor_motion moves with the neck only and the body pinned,
the /dev/shm/desired sequence counter, the IMU slot ring, camera-0 frames), and board detection is a `detect(frame)`
callable (M5: `lambda g: board_normal(board.detect(g), board, "KB4T", intr)`). This module never imports a robot
module and never moves anything by itself.

Per pose, as imucam_fast.py: check for a foreign command source over FOREIGN_WINDOW_S (abort if the counter
advances), move, settle SETTLE_S, average the IMU acceleration over IMU_WINDOW_S, read the measured neck angles, take
FRAMES_PER_POSE frames and keep the median board normal of those with >= MIN_TAGS tags. `save_sweep` writes rscfit's
npz format (neck, acc, acc_std, nb, n, R_SC); `to_sweep` gives an imu.Sweep directly."""
import math
import time
from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Sequence

import numpy as np

from calibtool import imu

DEFAULT_PITCH_DEG = (8.0, 16.0, 24.0)
DEFAULT_YAW_DEG = (-30.0, -15.0, 0.0, 15.0, 30.0)
FOREIGN_WINDOW_S = 0.2      # watch the foreign-command counter this long before each move
SETTLE_S = 1.5              # after the move returns, before the IMU window
IMU_WINDOW_S = 2.0          # IMU samples averaged per pose
FRAMES_PER_POSE = 3
MIN_TAGS = 8                # distinct tags a frame needs for its PnP to count


def serpentine(pitches: Sequence[float] = DEFAULT_PITCH_DEG, yaws: Sequence[float] = DEFAULT_YAW_DEG):
    """(pitch, yaw) poses row by row, the yaw order reversed on every other row (no long yaw swings)"""
    ys = [float(y) for y in yaws]
    return [(float(p), y) for k, p in enumerate(pitches) for y in (ys if k % 2 == 0 else ys[::-1])]


@dataclass
class BoardView:
    normal: np.ndarray      # board normal in camera 0, unit, toward the camera
    corners: int
    rms_px: float           # PnP reprojection rms, in pixels (normalised error x fx)


def board_normal(det: dict, board, model: str, intr, min_tags: int = MIN_TAGS) -> Optional[BoardView]:
    """imucam_fast.py's PnP: corners {(tag, k): (u, v)} -> unit rays by the lens -> solvePnP + refine on normalised
    coordinates -> the board's z axis, flipped toward the camera. None below min_tags distinct tags"""
    import cv2
    from calibtool import paths  # noqa: F401  (scripts folder on sys.path)
    import lensmodel
    if len({t for t, _ in det}) < min_tags:
        return None
    keys = list(det)
    uv = np.array([det[k] for k in keys], float)
    obj = np.array([board.obj_points(t)[k] for t, k in keys], float)
    ray = lensmodel.unproject(model, intr, uv)
    xy = np.ascontiguousarray(ray[:, :2] / ray[:, 2:3]).reshape(-1, 1, 2)
    ok, rv, tv = cv2.solvePnP(obj, xy, np.eye(3), None)
    if not ok:
        return None
    rv, tv = cv2.solvePnPRefineLM(obj, xy, np.eye(3), None, rv, tv)
    Rb = cv2.Rodrigues(rv)[0]
    n = Rb[:, 2].copy()
    if n @ tv.ravel() > 0:
        n = -n
    pr = (Rb @ obj.T + tv).T
    rms = math.sqrt(np.mean(np.sum((pr[:, :2] / pr[:, 2:3] - xy.reshape(-1, 2)) ** 2, axis=1))) * float(intr[0])
    return BoardView(n / np.linalg.norm(n), len(keys), rms)
