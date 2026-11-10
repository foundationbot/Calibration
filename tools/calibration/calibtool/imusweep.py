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


# ---------------------------------------------------------------- the capture loop


class SweepAborted(RuntimeError):
    """another command source moved the robot during the sweep; nothing more was commanded"""


class SweepRig(Protocol):
    """what M5's neck driver provides to capture_sweep"""

    def foreign_seq(self) -> int:
        """a counter that advances whenever another source commands the robot (/dev/shm/desired's sequence)"""

    def move_neck(self, pitch_deg: float, yaw_deg: float) -> None:
        """move the neck there (neck only, every other joint pinned, kp only) and return once it has arrived"""

    def neck_deg(self) -> tuple:
        """the measured (pitch, yaw), degrees"""

    def imu_accel(self, seconds: float) -> np.ndarray:
        """(k, 3) acceleration samples of the head IMU slot gathered over the next `seconds`"""

    def frames(self, k: int) -> list:
        """k fresh camera-0 grey frames (newer than the call)"""


@dataclass
class PoseResult:
    command: tuple                  # (pitch, yaw) commanded
    neck: np.ndarray                # (pitch, yaw) measured
    acc_mean: np.ndarray
    acc_std: np.ndarray
    view: Optional[BoardView]       # median over the frames with a board; None: no board in any frame
    n_views: int
    gap_deg: Optional[float]        # floor-vs-gravity gap with the config's R_SC (the live check imucam printed)


def capture_sweep(rig: SweepRig, poses, detect: Callable, R_SC, *, frames_per_pose: int = FRAMES_PER_POSE,
                  settle_s: float = SETTLE_S, imu_window_s: float = IMU_WINDOW_S,
                  foreign_window_s: float = FOREIGN_WINDOW_S, sleep: Callable = time.sleep,
                  log: Optional[Callable] = None) -> list:
    """imucam_fast.py's loop over `poses` [(pitch, yaw)]; raises SweepAborted (before moving) if the foreign-command
    counter advances. R_SC = the live config's cam0 rotation (for the per-pose gap only)"""
    R_SC = np.asarray(R_SC, float)
    out = []
    for i, (p, y) in enumerate(poses):
        h0 = rig.foreign_seq()
        sleep(foreign_window_s)
        if rig.foreign_seq() != h0:
            raise SweepAborted(f"pose {i + 1}/{len(poses)}: another source is commanding the robot "
                               "(the foreign-command counter advanced); sweep stopped before moving")
        rig.move_neck(p, y)
        sleep(settle_s)
        A = np.asarray(rig.imu_accel(imu_window_s), float).reshape(-1, 3)
        acc = A.mean(axis=0) if len(A) else np.full(3, np.nan)
        std = A.std(axis=0) if len(A) else np.full(3, np.nan)
        meas = np.asarray(rig.neck_deg(), float)
        views = [v for v in (detect(g) for g in rig.frames(frames_per_pose)) if v is not None]
        view, gap = None, None
        if views:
            nb = np.median([v.normal for v in views], axis=0)
            view = BoardView(nb / np.linalg.norm(nb), int(np.median([v.corners for v in views])),
                             float(np.median([v.rms_px for v in views])))
            if np.isfinite(acc).all():
                up = acc / np.linalg.norm(acc)
                gap = math.degrees(math.acos(float(np.clip(view.normal @ (R_SC.T @ up), -1.0, 1.0))))
        out.append(PoseResult((float(p), float(y)), meas, acc, std, view, len(views), gap))
        if log is not None:
            head = f"pose {i + 1:2d}/{len(poses)} cmd {p:+3.0f}/{y:+4.0f} meas {meas[0]:+5.1f}/{meas[1]:+5.1f}"
            if view is None:
                log(f"{head}: no board")
            else:
                log(f"{head} acc-std {np.nanmax(std):.2f} board {len(views)} fr, {view.corners} corners, PnP "
                    f"{view.rms_px:.2f} px | gap {gap if gap is not None else float('nan'):.2f} deg")
    return out


def _arrays(results):
    kept = [r for r in results if r.view is not None]
    return (np.array([r.neck for r in kept]).reshape(-1, 2), np.array([r.acc_mean for r in kept]).reshape(-1, 3),
            np.array([r.acc_std for r in kept]).reshape(-1, 3), np.array([r.view.normal for r in kept]).reshape(-1, 3))


def to_sweep(results, R_SC, max_acc_std: float = imu.MAX_ACC_STD) -> imu.Sweep:
    """the poses with a board as an imu.Sweep (rscfit's stillness filter applied)"""
    neck, acc, std, nb = _arrays(results)
    return imu.sweep_from_arrays(neck, acc, std, nb, R_SC, max_acc_std)


def save_sweep(path, results, R_SC):
    """rscfit's npz format, as imucam_fast.py wrote it (poses with a board only; nb and n the same normals)"""
    neck, acc, std, nb = _arrays(results)
    np.savez(path, neck=neck, acc=acc, acc_std=std, nb=nb, n=nb, R_SC=np.asarray(R_SC, float))
