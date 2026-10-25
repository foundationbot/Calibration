"""The arm session's safety and sag checks on the robot's MuJoCo model (MJCF; decisions T2-4, T2-5 of
docs/superpowers/plans/2026-10-22-phantom12-arm-session.md rev 3).

clearance: the least distance between the arm and the rest of the robot (held at a baseline pose), the smaller of
two: between the contact geoms (capsules and boxes; they leave ~9 cm of wrist uncovered), and, with surface_step,
between points on the visual meshes' surfaces (every surface_step m; the arm's links in skip_surface excepted: the
shoulder's first link sits 1.3-1.7 cm from the torso shell in every pose). gravity: the
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
    def __init__(self, xml: Path, root: str = "RightShoulderPitchLink", joints=RIGHT_ARM, distmax: float = 0.3,
                 surface_step=None, skip_surface=("RightShoulderPitchLink",), surface_group: int = 2,
                 surface_cap: float = 0.1):
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
        self.jnt = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joints]
        self._surf, self._tree, self.surface_cap = None, None, surface_cap
        if surface_step:
            skip = {mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, n) for n in skip_surface}
            meshes = [g for g in range(self.m.ngeom) if self.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                      and self.m.geom_group[g] == surface_group]
            self._surf = {g: _surface_points(self.m, g, surface_step) for g in meshes}
            self._arm_mesh = [g for g in meshes if int(self.m.geom_bodyid[g]) in sub
                              and int(self.m.geom_bodyid[g]) not in skip]
            self._other_mesh = [g for g in meshes if int(self.m.geom_bodyid[g]) not in sub]

    def _name(self, g) -> str:
        return self.m.geom(g).name or f"geom{g}"

    def set_baseline(self, angles: dict) -> None:
        """Hold every named joint (rad) at this value; names the model lacks are ignored."""
        for n, v in angles.items():
            j = self.mj.mj_name2id(self.m, self.mj.mjtObj.mjOBJ_JOINT, n)
            if j >= 0:
                self.baseline[self.m.jnt_qposadr[j]] = v
        self._tree = None

    def _pose(self, q) -> None:
        self.d.qpos[:] = self.baseline
        self.d.qpos[self.qadr] = q
        self.mj.mj_kinematics(self.m, self.d)

    def _pairs(self):
        fromto = np.zeros(6)
        for a in self._arm:
            for b in self._other:
                yield a, b, self.mj.mj_geomDistance(self.m, self.d, a, b, self.distmax, fromto)

    def ranges(self) -> tuple:
        """(lower, upper) rad of the arm's joints in the model; an unlimited joint is unbounded."""
        lim = np.array([self.m.jnt_range[j] if self.m.jnt_limited[j] else (-np.inf, np.inf) for j in self.jnt])
        return lim[:, 0], lim[:, 1]

    def _world(self, g) -> np.ndarray:
        return self._surf[g] @ self.d.geom_xmat[g].reshape(3, 3).T + self.d.geom_xpos[g]

    def surface_clearance(self, Q) -> np.ndarray:
        """(N,) least distance (m) between surface points of the arm's meshes and of the rest of the robot's;
        surface_cap when nothing is closer (the search stops there: only centimetres matter)."""
        from scipy.spatial import cKDTree
        if self._surf is None:
            raise ValueError("no surface points: construct with surface_step")
        if self._tree is None:
            self._pose(np.zeros(len(self.qadr)))            # the rest of the body does not move with the arm
            self._tree = cKDTree(np.vstack([self._world(g) for g in self._other_mesh]))
        out = []
        for q in np.atleast_2d(Q):
            self._pose(q)
            d = self._tree.query(np.vstack([self._world(g) for g in self._arm_mesh]),
                                 distance_upper_bound=self.surface_cap)[0]
            out.append(float(min(d.min(), self.surface_cap)))
        return np.array(out)

    def contact_clearance(self, Q) -> np.ndarray:
        """(N,) least distance (m) between the arm's and the body's contact geoms; distmax when nothing is closer;
        < 0 inside."""
        out = []
        for q in np.atleast_2d(Q):
            self._pose(q)
            out.append(min((dist for _, _, dist in self._pairs()), default=self.distmax))
        return np.array(out)

    def clearance(self, Q) -> np.ndarray:
        """(N,) least arm-to-body distance (m) at arm angles Q (N, n): contact_clearance and, with surface points,
        the smaller of that and surface_clearance."""
        out = self.contact_clearance(Q)
        return np.minimum(out, self.surface_clearance(Q)) if self._surf is not None else out

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


def _surface_points(m, g, step: float) -> np.ndarray:
    """A mesh geom's vertices and points spread over its triangles about every `step` m, thinned to one per
    step-sized cell, in the geom's frame."""
    mid = m.geom_dataid[g]
    V = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    F = m.mesh_face[m.mesh_faceadr[mid]:m.mesh_faceadr[mid] + m.mesh_facenum[mid]]
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(B - A, C - A), axis=1)
    n = np.maximum(1, np.ceil(2.0 * area / step ** 2)).astype(int)
    idx = np.repeat(np.arange(len(F)), n)
    r1, r2 = np.random.default_rng(0).random((2, len(idx)))
    s = np.sqrt(r1)
    P = np.vstack([V, (1 - s)[:, None] * A[idx] + (s * (1 - r2))[:, None] * B[idx] + (s * r2)[:, None] * C[idx]])
    _, keep = np.unique(np.floor(P / step).astype(np.int64), axis=0, return_index=True)
    return P[np.sort(keep)]

