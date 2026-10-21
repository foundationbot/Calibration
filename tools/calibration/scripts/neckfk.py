"""Forward kinematics of the neck, Torso -> NeckYawLink -> Head, read from the URDF.

With the torso still, the Head link's pose in the torso frame is

    T_TB(q) = O_yaw · Rot(a_yaw, q_yaw) · O_pitch · Rot(a_pitch, q_pitch)

where O are the joints' fixed <origin> transforms (xyz, rpy) and a their axes (URDF convention:
the axis is expressed in the joint frame, which is the child link's frame). Used by the T_BS
steps (75, 81, 82).
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rigmath import rot_axis, transform  # noqa: E402


def rpy(r: float, p: float, y: float) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw: R = Rz(y) Ry(p) Rx(r)."""
    return rot_axis((0, 0, 1), y) @ rot_axis((0, 1, 0), p) @ rot_axis((1, 0, 0), r)


@dataclass(frozen=True, eq=False)
class Joint:
    name: str
    parent: str
    child: str
    origin: np.ndarray  # 4x4: parent link -> joint frame at q = 0
    axis: np.ndarray    # axis in the joint (child) frame, made unit on construction

    def __post_init__(self):
        a = np.asarray(self.axis, float)
        if np.linalg.norm(a) < 1e-12:
            raise ValueError(f"joint {self.name!r} has a zero axis")
        object.__setattr__(self, "axis", a / np.linalg.norm(a))


def load_joint(urdf: Path, name: str) -> Joint:
    for j in ET.parse(urdf).getroot().iter("joint"):
        if j.get("name") != name:
            continue
        if j.get("type") not in ("revolute", "continuous"):
            raise ValueError(f"joint {name!r} is {j.get('type')!r}, not a rotation")
        o, ax = j.find("origin"), j.find("axis")
        xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
        r, p, y = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
        axis = np.array([float(v) for v in (ax.get("xyz") if ax is not None else "1 0 0").split()])
        return Joint(name, j.find("parent").get("link"), j.find("child").get("link"),
                     transform(rpy(r, p, y), xyz), axis)
    raise KeyError(f"joint {name!r} not in {urdf}")


class NeckChain:
    def __init__(self, urdf: Path, yaw: str = "NeckYaw", pitch: str = "NeckPitch"):
        self.yaw = load_joint(Path(urdf), yaw)
        self.pitch = load_joint(Path(urdf), pitch)
        if self.yaw.child != self.pitch.parent:
            raise ValueError(f"{yaw} -> {pitch} is not one chain ({self.yaw.child} != {self.pitch.parent})")

    def R_YB(self, q_pitch: float) -> np.ndarray:
        """Head link orientation in the NeckYawLink frame."""
        return self.pitch.origin[:3, :3] @ rot_axis(self.pitch.axis, q_pitch)

    def T_TB(self, q_yaw: float, q_pitch: float) -> np.ndarray:
        """The Head link in the torso frame (maps Head points into the torso) at neck angles in rad,
        with the URDF's signs."""
        return (self.yaw.origin @ transform(rot_axis(self.yaw.axis, q_yaw), np.zeros(3))
                @ self.pitch.origin @ transform(rot_axis(self.pitch.axis, q_pitch), np.zeros(3)))

    def axes_in_T(self, q_yaw: float, q_pitch: float):
        """(NeckYaw's axis, NeckPitch's axis), both in torso axes, at the given neck angles."""
        return (self.yaw.origin[:3, :3] @ self.yaw.axis,
                self.T_TB(q_yaw, q_pitch)[:3, :3] @ self.pitch.axis)

    def yaw_axis_in_B(self, q_pitch) -> np.ndarray:
        """NeckYaw's axis in Head axes, R_YB(q_pitch)^T a_yaw; scalar -> 3, array -> N x 3."""
        qp = np.asarray(q_pitch, float)
        k = self.pitch.axis
        v = self.pitch.origin[:3, :3].T @ self.yaw.axis        # yaw axis in the pitch joint frame
        c, s = np.cos(-qp)[..., None], np.sin(-qp)[..., None]   # Rot(k, q)^T = Rot(k, -q), Rodrigues
        return v * c + np.cross(k, v) * s + k * np.dot(k, v) * (1.0 - c)

    def omega_B(self, q_pitch, qd_yaw, qd_pitch) -> np.ndarray:
        """Head angular velocity relative to the torso, in Head axes:
        qd_pitch * a_pitch + qd_yaw * R_YB(q_pitch)^T a_yaw. Scalars give a 3-vector, arrays of
        equal length N x 3 (vectorised: step 82 evaluates it ~10^5 times per lag)."""
        qp, qdy, qdp = np.broadcast_arrays(*(np.asarray(v, float) for v in (q_pitch, qd_yaw, qd_pitch)))
        return qdp[..., None] * self.pitch.axis + qdy[..., None] * self.yaw_axis_in_B(qp)
