"""Frames of a burst: the rectified depth camera C -> the Head link (B) -> the Torso -> the Pelvis.

    p_F = T_FB(q) · T_BS · T_SC · p_C       (T_AB maps B coordinates into A; row-major 4x4)

(Ported from OKVIS2-X tools/burst_pointcloud/frames.py, c5b4c6b0, with two changes: the URDF is an explicit argument --
the robot's own, which the hand-eye fit corrects -- instead of the checkout's file or the BURST_URDF patch; and the
gravity-levelled frame L of box-pick is left out: read B does not use it.)

T_SC comes with every depth frame; T_BS (IMU -> Head) from the live OKVIS config (the grabber stores it); the neck
and spine from the URDF at the joint samples interpolated to each frame's stamp. The burst's pose is the median over
its frames; neck_motion_deg() says how far the joints moved inside the burst (a moving head blurs the fusion and the
frame alike). A config prints T_BS with ~6 decimals, orthonormal only to ~1e-6; T_FC() returns the nearest true
rotation (and refuses a matrix that is not close to one: a typo, a mirror image)."""
from __future__ import annotations

import numpy as np

from armcal import SCRIPTS  # noqa: F401  (scripts/ on sys.path)
from armcal.urdf import Urdf
from rigmath import nearest_rotation, rot_axis, transform  # noqa: E402

FRAMES = ("camera", "head", "torso", "pelvis")
NEEDS = {"camera": (), "head": ("T_BS",), "torso": ("T_BS", "joints"), "pelvis": ("T_BS", "joints")}
JOINTS = {"camera": (), "head": (), "torso": ("NeckYaw", "NeckPitch"), "pelvis": ("NeckYaw", "NeckPitch", "SpineYaw")}
CLEAN_MAX = 1e-4   # largest element change the rotation cleaning may make (as scripts/okviscfg.py)


class Frames:
    def __init__(self, burst, urdf):
        self.b, self.urdf = burst, Urdf.of(urdf)

    def available(self) -> list:
        return [f for f in FRAMES if self._missing(f) is None]

    def _missing(self, frame):
        for k in NEEDS[frame]:
            if getattr(self.b, k, None) is None:
                return f"frame {frame!r} needs the burst's {k} (captured before Plan 4, or not recorded)"
        names = self.b.info.get("joint_names", []) if JOINTS[frame] else []
        missing = [n for n in JOINTS[frame] if n not in names]
        if missing:
            return f"frame {frame!r} needs the joint(s) {', '.join(missing)}; the burst has {names}"
        return None

    def angles(self, frames_used: int = 0) -> dict:
        """median over the burst's (first frames_used) frames of each joint, interpolated at the frame stamps (rad)"""
        j = np.asarray(self.b.joints, float)
        t, s = j[:, 0], self._stamps(frames_used)
        if s.min() < t.min() or s.max() > t.max():
            raise ValueError("the joint samples do not cover the burst's frame stamps")
        return {n: float(np.median(np.interp(s, t, j[:, 1 + i]))) for i, n in enumerate(self.b.info["joint_names"])}

    def neck_motion_deg(self, frames_used: int = 0) -> dict:
        """max - min of each joint over the burst's (first frames_used) frames, degrees"""
        j = np.asarray(self.b.joints, float)
        s = self._stamps(frames_used)
        return {n: float(np.degrees(np.ptp(np.interp(s, j[:, 0], j[:, 1 + i]))))
                for i, n in enumerate(self.b.info["joint_names"])}

    def _stamps(self, frames_used):
        s = np.asarray(self.b.stamp_ns, float)
        return s[:frames_used] if frames_used else s

    def T_FC(self, frame: str, frames_used: int = 0) -> np.ndarray:
        """the rectified depth camera in `frame` (camera / head / torso / pelvis), its rotation cleaned"""
        if frame not in FRAMES:
            raise ValueError(f"frame must be one of {FRAMES}, got {frame!r}")
        why = self._missing(frame)
        if why is not None:
            raise ValueError(why)
        T = self._T_FC(frame, frames_used)
        R = T[:3, :3]
        Rc = nearest_rotation(R)
        if np.abs(Rc - R).max() > CLEAN_MAX or np.linalg.det(R) <= 0:
            raise ValueError(f"T_FC({frame!r}) is not a rotation (T_BS or T_SC a typo or a mirror image?)")
        T = T.copy()
        T[:3, :3] = Rc
        return T

    def T_BC(self) -> np.ndarray:
        """the camera in the Head link, T_BS · T_SC (not cleaned: the stage-2 fit uses it as stored)"""
        return np.asarray(self.b.T_BS, float) @ np.asarray(self.b.T_SC, float)

    def _T_FC(self, frame: str, frames_used: int) -> np.ndarray:
        if frame == "camera":
            return np.eye(4)
        T = self.T_BC()
        if frame == "head":
            return T
        a = self.angles(frames_used)
        T = self.urdf.chain("Torso", "Head").T(a) @ T
        if frame == "torso":
            return T
        spine = self.urdf.joint("SpineYaw")
        return (spine.origin @ transform(rot_axis(spine.axis, a["SpineYaw"]), np.zeros(3))) @ T
