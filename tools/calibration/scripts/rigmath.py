"""Rotation and rigid-transform helpers for the T_BS steps (75, 81, 82, 83).

T_AB maps points from frame B into frame A (p_A = T_AB p_B); R_AB is its rotation. Angles are
computed with atan2, never arccos: arccos near zero turns a matrix that is orthonormal to only
1e-6 (a config printed with six decimals) into a fake 0.06 deg."""
from __future__ import annotations

import numpy as np


def skew(v) -> np.ndarray:
    x, y, z = np.asarray(v, float)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def rot_axis(axis, angle: float) -> np.ndarray:
    """Rotation by `angle` [rad] about `axis` (any length), Rodrigues."""
    a = np.asarray(axis, float)
    K = skew(a / np.linalg.norm(a))
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def exp_rot(w) -> np.ndarray:
    """Rotation vector (axis * angle) -> rotation matrix."""
    w = np.asarray(w, float)
    a = np.linalg.norm(w)
    return np.eye(3) + skew(w) if a < 1e-12 else rot_axis(w, a)


def log_rot(R) -> np.ndarray:
    """Rotation matrix -> rotation vector (axis * angle), accurate at every angle. Under 90 deg the
    antisymmetric part of R gives the axis; over 90 deg its symmetric part does, which stays well
    conditioned up to a half turn where the antisymmetric part vanishes, and the antisymmetric part
    only picks the sign. No angle threshold: a threshold is how a tiny rotation once came back as a
    half turn and stalled the fit."""
    R = np.asarray(R, float)
    w = 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s, c = np.linalg.norm(w), (np.trace(R) - 1.0) / 2.0
    a = np.arctan2(s, c)
    if c > 0.0:                                    # under 90 deg: s is a well-conditioned sin(a)
        return w if s == 0.0 else w * (a / s)
    B = (R + R.T) / 2.0 - c * np.eye(3)            # = (1 - c) n n^T
    k = int(np.argmax(np.diag(B)))
    n = B[:, k] / np.linalg.norm(B[:, k])
    return a * n if n @ w >= 0.0 else -a * n


def rot_angle_deg(R) -> float:
    """Angle of a rotation in degrees, well conditioned near 0 and 180."""
    R = np.asarray(R, float)
    s = 0.5 * np.linalg.norm([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return float(np.degrees(np.arctan2(s, (np.trace(R) - 1.0) / 2.0)))


def nearest_rotation(M) -> np.ndarray:
    """The rotation closest to M (Frobenius), e.g. to clean a matrix read from a yaml."""
    U, _, Vt = np.linalg.svd(np.asarray(M, float))
    return U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt


def transform(R, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def inv(T) -> np.ndarray:
    T = np.asarray(T, float)
    return transform(T[:3, :3].T, -T[:3, :3].T @ T[:3, 3])


def mean_pose(Ts) -> np.ndarray:
    """Translation mean; rotation = the rotation nearest the mean matrix (chordal mean)."""
    return transform(nearest_rotation(np.mean([T[:3, :3] for T in Ts], axis=0)),
                     np.mean([T[:3, 3] for T in Ts], axis=0))


def spread(Ts) -> dict:
    """How far a set of poses that should be one pose scatter: rms and max of the translation
    [mm] and of the rotation [deg] about their mean."""
    ref = mean_pose(Ts)
    dt = np.array([np.linalg.norm(T[:3, 3] - ref[:3, 3]) * 1e3 for T in Ts])
    dr = np.array([rot_angle_deg(ref[:3, :3].T @ T[:3, :3]) for T in Ts])
    return {"n": len(Ts), "trans_rms_mm": float(np.sqrt(np.mean(dt ** 2))), "trans_max_mm": float(dt.max()),
            "rot_rms_deg": float(np.sqrt(np.mean(dr ** 2))), "rot_max_deg": float(dr.max())}
