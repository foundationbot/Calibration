"""Camera-to-IMU rotation v1 from an AprilGrid lying flat on the floor during a neck sweep (design plan 9.3).

Data (`Sweep`): per still neck pose, the IMU slot's mean acceleration (the reaction to gravity: "up" in the IMU frame)
and the board's normal from PnP in camera 0, flipped toward the camera (the floor's "up" in the camera frame). With
the config's cam0 rotation R_SC (camera -> IMU), R_SC.T @ up_imu should equal the board normal; the angle between them
is the pose's floor-vs-gravity gap.

Model (the kit's rscfit.py, reproduced): R_new = R_SC @ Rx(ax) @ Rz(az), two angles about camera x and z; robust
soft-L1 least squares on the stacked cross products (f_scale 0.02) from (0, 0); formal 1-sigma from pinv(J^T J) x
median(r^2). The third angle, about the gravity direction, is (nearly) unobservable from floor sweeps: neck yaw turns
about a near-vertical axis and changes neither gravity in the IMU frame nor the floor normal in the camera frame, so
only the pitch span separates the up directions (one pitch row makes it exactly unobservable). `observability`
measures it; v1 keeps the live config's value.

Scoring is on held-out poses (fitting and scoring on the same poses would make "improves" circular): leave one yaw
column out (fit on the other columns, score the left-out one), pooled over the columns; the two pitch halves (fit on
one, score the other) are reported too. `verdict` deploys the rotation only if the pooled held-out median gap improves
on the live config's, on the same poses, by more than k x the fit's combined sigma (hypot of the two formal sigmas);
the thresholds (`Gates`) are named placeholders until fixed with the user.

`apply` is the kit's mkfinal.py rule: cam0 T_SC rotation R0 -> R0 @ R_correction, cam1 = new cam0 x the config's own
stereo pose inv(T0) @ T1; cam0 translation, T_BS and every other byte kept. Only the changed numbers are rewritten,
each in its own place (repr of the float), so a diff shows nothing else. The kit's 83_tbs_apply.py is not used (it
writes T_BS). Changing T_SC with T_BS kept moves the camera in the Head frame for other consumers (camera-to-arm): the
verdict says so."""
import re
from dataclasses import dataclass, field

import numpy as np

MAX_ACC_STD = 0.4        # m/s^2: a pose whose IMU std reaches this on any axis was not still (rscfit)
SOFT_L1_SCALE = 0.02     # rscfit's f_scale, in |cross| units (sin of the gap: ~1.1 deg)
COLUMN_GAP_DEG = 5.0     # neck yaw values further apart than this start a new yaw column
ROW_GAP_DEG = 3.0        # neck pitch values further apart than this start a new pitch row
MIN_FIT_POSES = 3        # two angles from fewer poses is not a fit


# ---------------------------------------------------------------- rotations


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def rot_xz(ax_deg, az_deg):
    """rscfit's correction: Rx(ax) @ Rz(az), angles in degrees about camera x and z"""
    return _rx(np.radians(ax_deg)) @ _rz(np.radians(az_deg))


def axis_rotation(axis, deg):
    """rotation by `deg` about `axis` (any length)"""
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    t = np.radians(deg)
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def rotation_angle_deg(Ra, Rb):
    """the angle of the rotation taking Ra to Rb"""
    c = (np.trace(np.asarray(Ra).T @ np.asarray(Rb)) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


# ---------------------------------------------------------------- the sweep


@dataclass
class Sweep:
    """still neck poses kept for the fit (rscfit's filter applied)"""
    neck: np.ndarray        # (N, 2) measured neck pitch, yaw (deg)
    acc: np.ndarray         # (N, 3) IMU acceleration mean (IMU frame, m/s^2): "up" when still
    acc_std: np.ndarray     # (N, 3) its standard deviation over the window
    normal: np.ndarray      # (N, 3) board normal in camera 0, unit, toward the camera ("up")
    R_SC: np.ndarray        # (3, 3) the cam0 rotation of the config the sweep ran with
    n_recorded: int = 0     # poses recorded before the filter

    @property
    def up_imu(self):
        return self.acc / np.linalg.norm(self.acc, axis=1, keepdims=True)

    def __len__(self):
        return len(self.normal)


def sweep_from_arrays(neck, acc, acc_std, normal, R_SC, max_acc_std=MAX_ACC_STD) -> Sweep:
    """keep the poses rscfit keeps: every acc_std axis below max_acc_std and a finite normal"""
    neck, acc, acc_std, normal = (np.asarray(a, float).reshape(-1, k) for a, k in
                                  ((neck, 2), (acc, 3), (acc_std, 3), (normal, 3)))
    ok = (acc_std.max(axis=1) < max_acc_std) & np.isfinite(normal).all(axis=1) & np.isfinite(acc).all(axis=1)
    return Sweep(neck[ok], acc[ok], acc_std[ok], normal[ok], np.asarray(R_SC, float).reshape(3, 3), int(len(ok)))


def load_sweep(path, key="nb", max_acc_std=MAX_ACC_STD) -> Sweep:
    """an imucam_fast.py / imusweep.save_sweep npz (neck, acc, acc_std, nb, n, R_SC); `key` picks the normals"""
    with np.load(path) as z:
        return sweep_from_arrays(z["neck"], z["acc"], z["acc_std"], z[key], z["R_SC"], max_acc_std)


def _idx(sweep, idx):
    return np.arange(len(sweep)) if idx is None else np.asarray(idx, int)


def floor_gap_deg(sweep: Sweep, R, idx=None):
    """per pose: the angle between the board normal and R.T @ up_imu (deg)"""
    idx = _idx(sweep, idx)
    v = sweep.up_imu[idx] @ np.asarray(R, float)            # rows (R.T @ u).T
    n = sweep.normal[idx]
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(v, n), axis=1), np.einsum("ij,ij->i", v, n)))


# ---------------------------------------------------------------- the fit


@dataclass
class Fit:
    """rscfit's two-angle correction fitted on the poses `idx`"""
    angles_deg: np.ndarray      # (2,) about camera x, about camera z
    sigma_deg: np.ndarray       # (2,) formal 1-sigma
    gap_deg: np.ndarray         # per fitted pose, after the correction
    idx: np.ndarray

    @property
    def correction(self):
        """R_SC -> R_SC @ correction"""
        return rot_xz(*self.angles_deg)

    @property
    def combined_sigma_deg(self):
        return float(np.hypot(*self.sigma_deg))


def fit(sweep: Sweep, idx=None) -> Fit:
    """rscfit.fit: soft-L1 least squares on the cross products of R_new.T @ up_imu and the board normal"""
    from scipy.optimize import least_squares
    idx = _idx(sweep, idx)
    if len(idx) < MIN_FIT_POSES:
        raise ValueError(f"{len(idx)} poses: the two-angle fit needs at least {MIN_FIT_POSES}")
    U, N, R = sweep.up_imu[idx], sweep.normal[idx], sweep.R_SC

    def f(p):
        return np.cross(U @ (R @ _rx(p[0]) @ _rz(p[1])), N).ravel()

    s = least_squares(f, [0.0, 0.0], loss="soft_l1", f_scale=SOFT_L1_SCALE)
    cov = np.linalg.pinv(s.jac.T @ s.jac) * np.median(s.fun ** 2)
    angles = np.degrees(s.x)
    return Fit(angles, np.degrees(np.sqrt(np.diag(cov))), floor_gap_deg(sweep, R @ rot_xz(*angles), idx), idx)


def R_new(sweep: Sweep, f: Fit):
    """the cam0 rotation the fit gives"""
    return sweep.R_SC @ f.correction
