"""Forward kinematics of an arm, base link -> ... -> tip link, read from the URDF (the arm session's chain).

With the torso still, the tip link's pose in the base frame is

    T_base,tip(q) = O_1 · Rot(a_1, q_1) · O_2 · Rot(a_2, q_2) · ... · O_n · Rot(a_n, q_n)

with the joints' fixed <origin> transforms O and axes a, as neckfk.py reads them (the axis is in the child link's
frame). For the right arm the tip is RightGripperYawLink: the hand (and the plate on its palm) is rigid on it.
The camera reaches the same torso frame through the neck (camera_in_torso), so a camera pixel and an arm pose
meet in one frame:  T_TC(q_neck) · T_C,plate = T_T,tip(q_arm) · X.
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from neckfk import load_joint  # noqa: E402

RIGHT_ARM = ("RightShoulderPitch", "RightShoulderRoll", "RightShoulderYaw", "RightElbowPitch",
             "RightWristYaw", "RightWristRoll", "RightGripperYaw")


def load_limits(urdf: Path, name: str) -> tuple[float, float]:
    """(lower, upper) in rad of a revolute joint; a continuous joint is unbounded."""
    for j in ET.parse(urdf).getroot().iter("joint"):
        if j.get("name") != name:
            continue
        if j.get("type") == "continuous":
            return -np.inf, np.inf
        lim = j.find("limit")
        if lim is None or lim.get("lower") is None or lim.get("upper") is None:
            raise ValueError(f"joint {name!r} has no <limit lower upper>")
        return float(lim.get("lower")), float(lim.get("upper"))
    raise KeyError(f"joint {name!r} not in {urdf}")


def _rot(axis: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Rodrigues for a unit axis and angles of any shape (...,) -> (..., 3, 3)."""
    q = np.asarray(q, float)
    x, y, z = axis
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    s, c = np.sin(q)[..., None, None], np.cos(q)[..., None, None]
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


class ArmChain:
    def __init__(self, urdf: Path, joints=RIGHT_ARM, base: str = "Torso"):
        urdf = Path(urdf)
        self.names = tuple(joints)
        self.joints = [load_joint(urdf, n) for n in self.names]
        if self.joints[0].parent != base:
            raise ValueError(f"{self.names[0]} hangs from {self.joints[0].parent!r}, not the base {base!r}")
        for a, b in zip(self.joints, self.joints[1:]):
            if a.child != b.parent:
                raise ValueError(f"{a.name} -> {b.name} is not one chain ({a.child} != {b.parent})")
        self.base, self.tip = base, self.joints[-1].child
        lim = np.array([load_limits(urdf, n) for n in self.names])
        self.lower, self.upper = lim[:, 0], lim[:, 1]

    def frames(self, q) -> list:
        """Every joint's child link in the base frame at angles q (rad): n of (..., 4, 4); q is (n,) or (..., n)."""
        q = np.asarray(q, float)
        if q.shape[-1] != len(self.joints):
            raise ValueError(f"{q.shape[-1]} angles for {len(self.joints)} joints")
        M = np.broadcast_to(np.eye(4), q.shape[:-1] + (4, 4)).copy()
        out = []
        for i, j in enumerate(self.joints):
            R = np.broadcast_to(np.eye(4), q.shape[:-1] + (4, 4)).copy()
            R[..., :3, :3] = _rot(j.axis, q[..., i])
            M = M @ j.origin @ R
            out.append(M)
        return out

    def T_base_tip(self, q) -> np.ndarray:
        """The tip link in the base frame (maps tip points into the base) at angles q (rad)."""
        return self.frames(q)[-1]


def camera_in_torso(neck, T_BS: np.ndarray, T_SC: np.ndarray, q_yaw: float, q_pitch: float) -> np.ndarray:
    """T_TC: a camera in the torso frame at neck angles (rad), through the Head link (neckfk.NeckChain), the IMU
    in the Head (T_BS) and the camera in the IMU (T_SC), all as OKVIS's config writes them."""
    return neck.T_TB(q_yaw, q_pitch) @ T_BS @ T_SC
