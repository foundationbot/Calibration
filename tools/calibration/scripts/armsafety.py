"""The arm session's safety and sag checks on the robot's MuJoCo model (MJCF; decisions T2-4, T2-5 of
docs/superpowers/plans/2026-10-22-phantom12-arm-session.md rev 3).

clearance: the least distance between the arm's contact geometry (every geom with contacts in the subtree from
`root`) and every other contact geom of the robot, the rest of the body held at a baseline pose. gravity: the
joint torques that hold the arm still against gravity (MuJoCo's bias force at zero velocity), with gravity given
in the floating base's frame (the robot hangs, leaning; the pelvis IMU gives the direction).

Uses the MJCF's own kinematics and geometry, not the URDF's (T2-1): the two agree below the shoulder. Needs
`mujoco` (not part of the calibration CI job).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from armchain import RIGHT_ARM  # noqa: E402


class Body:
    def __init__(self, xml: Path, root: str = "RightShoulderPitchLink", joints=RIGHT_ARM, distmax: float = 0.3):
        import mujoco
        self.mj = mujoco
        self.m = mujoco.MjModel.from_xml_path(str(xml))
        self.d = mujoco.MjData(self.m)
        self.distmax = distmax
        self.qadr, self.dadr = [], []
        for n in joints:
            j = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)
            if j < 0:
                raise KeyError(f"joint {n!r} not in {xml}")
            self.qadr.append(int(self.m.jnt_qposadr[j]))
            self.dadr.append(int(self.m.jnt_dofadr[j]))
        rb = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, root)
        if rb < 0:
            raise KeyError(f"body {root!r} not in {xml}")
        sub = {rb}
        for b in range(self.m.nbody):                       # bodies are numbered parents first
            if int(self.m.body_parentid[b]) in sub and b != 0:
                sub.add(b)
        live = [g for g in range(self.m.ngeom) if self.m.geom_contype[g] or self.m.geom_conaffinity[g]]
        self._arm = [g for g in live if int(self.m.geom_bodyid[g]) in sub]
        self._other = [g for g in live if int(self.m.geom_bodyid[g]) not in sub]
        self.arm_geoms = [self._name(g) for g in self._arm]
        self.other_geoms = [self._name(g) for g in self._other]
        self._free = [int(self.m.jnt_qposadr[j]) for j in range(self.m.njnt)
                      if self.m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        self.baseline = self.d.qpos.copy()

    def _name(self, g) -> str:
        return self.m.geom(g).name or f"geom{g}"

    def set_baseline(self, angles: dict) -> None:
        """Hold every named joint (rad) at this value; names the model lacks are ignored."""
        for n, v in angles.items():
            j = self.mj.mj_name2id(self.m, self.mj.mjtObj.mjOBJ_JOINT, n)
            if j >= 0:
                self.baseline[self.m.jnt_qposadr[j]] = v

    def _pose(self, q) -> None:
        self.d.qpos[:] = self.baseline
        self.d.qpos[self.qadr] = q
        self.mj.mj_kinematics(self.m, self.d)

    def _pairs(self):
        fromto = np.zeros(6)
        for a in self._arm:
            for b in self._other:
                yield a, b, self.mj.mj_geomDistance(self.m, self.d, a, b, self.distmax, fromto)

    def clearance(self, Q) -> np.ndarray:
        """(N,) least arm-to-body distance (m) at arm angles Q (N, n); distmax when nothing is closer; < 0 inside."""
        out = []
        for q in np.atleast_2d(Q):
            self._pose(q)
            out.append(min((dist for _, _, dist in self._pairs()), default=self.distmax))
        return np.array(out)

    def closest(self, q) -> tuple:
        """(arm geom, other geom) of the least distance at arm angles q."""
        self._pose(q)
        a, b, _ = min(self._pairs(), key=lambda t: t[2])
        return self._name(a), self._name(b)

    def gravity(self, Q, g_base) -> np.ndarray:
        """(N, n) torques (Nm) the arm's joints must give to hold angles Q still, gravity along g_base (any length) in
        the floating base's frame."""
        g = np.asarray(g_base, float)
        g = g / np.linalg.norm(g)
        z = -g                                                # world up, in the base frame
        x = np.cross([0.0, 1.0, 0.0], z)
        if np.linalg.norm(x) < 1e-6:
            x = np.cross([1.0, 0.0, 0.0], z)
        x /= np.linalg.norm(x)
        R_BW = np.column_stack([x, np.cross(z, x), z])        # world axes in the base
        quat = np.zeros(4)
        self.mj.mju_mat2Quat(quat, np.ascontiguousarray(R_BW.T).ravel())
        out = []
        for q in np.atleast_2d(Q):
            self.d.qpos[:] = self.baseline
            for a in self._free:
                self.d.qpos[a + 3:a + 7] = quat
            self.d.qpos[self.qadr] = q
            self.d.qvel[:] = 0.0
            self.mj.mj_forward(self.m, self.d)
            out.append(self.d.qfrc_bias[self.dadr].copy())
        return np.array(out)

    def frame(self, q, body: str) -> np.ndarray:
        """A body's frame in MuJoCo's world (4x4) at arm angles q, the rest at the baseline."""
        b = self.mj.mj_name2id(self.m, self.mj.mjtObj.mjOBJ_BODY, body)
        if b < 0:
            raise KeyError(f"body {body!r} not in the model")
        self._pose(q)
        T = np.eye(4)
        T[:3, :3] = self.d.xmat[b].reshape(3, 3)
        T[:3, 3] = self.d.xpos[b]
        return T

    def sightline_clear(self, q, origin, targets, see_through=(), clear_m: float = 0.02, group: int = 2) -> bool:
        """True when no geom of `group` (2: the MJCF's visual meshes) outside the bodies `see_through` lies on the
        straight line from origin to any target (world points, at arm angles q) closer than clear_m before it."""
        keep = {self.mj.mj_name2id(self.m, self.mj.mjtObj.mjOBJ_BODY, n) for n in see_through}
        mask = np.zeros(6, np.uint8)
        mask[group] = 1
        self._pose(q)
        gid = np.zeros(1, np.int32)
        o = np.asarray(origin, float)
        for t in np.atleast_2d(targets):
            v = np.asarray(t, float) - o
            L = float(np.linalg.norm(v))
            u = v / L
            start, gone = o.copy(), 0.0
            while True:
                dist = self.mj.mj_ray(self.m, self.d, start, u, mask, 1, -1, gid)
                if dist < 0 or gone + dist >= L - clear_m:
                    break
                if int(self.m.geom_bodyid[gid[0]]) not in keep:
                    return False
                step = dist + 1e-4                           # through the see-through geom's surface, and on
                start = start + u * step
                gone += step
        return True

