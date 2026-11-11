"""Read B's per-capture match (B0): an arm's forearm and wrist CAD meshes fitted to the head camera's depth.

(Ported from OKVIS2-X tools/burst_pointcloud/meshfit.py, c5b4c6b0: fit_pose (B0), view_gate and their helpers. Left
out: the joint fits B1/B2, B-neck and the RGB edge term -- the hand-eye plan's stage 2 (armcal/urdffit.py) replaces
them.)

The model: each link's mesh, area-sampled (mesh.sample), placed by the URDF chain from the Torso at the measured
joints and brought into the rectified depth camera by T_CT = T_TC^-1. The data: the fused depth's camera-frame points.

The fit moves the model, never the data: m -> E · T_CT · T_TL(q) · p, with E a rigid correction in the camera frame,
by robust point-to-plane Gauss-Newton:
  - per stage (correspondence gate 30 -> 15 -> 8 -> 5 mm): the model's visible points (mesh.visible: nearest in their
    pixel, not edge-on) and the data inside the model's silhouette dilated by 8 px;
  - per step: each data point's nearest visible model point within the gate (a KD-tree), residual r = n · (d - m),
    Huber weights (3 mm), the 6 unknowns from the normal equations; at most 30 steps a stage (1000 in the last),
    a stage ends at |step| < 0.01 mm and 0.001 deg.
Only the last stage must settle; the reported statistics are at the final E. The result is said at the WRIST POINT
(the <side>WristRoll joint origin): where the depth puts it minus where the encoders + URDF put it, Torso frame (m).

Speed (FitConfig, measured on all 48 stored phantom14 captures: tools/calibration/results/handeye-h0-2026-11-11/):
REFERENCE is the algorithm as it ran on phantom14; FAST (the default) changes only how the same quantities are
computed, plus the measured-safe approximations named in FitConfig."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from armcal import mesh
from armcal.urdf import Urdf

GATES = (0.030, 0.015, 0.008, 0.005)   # m, the correspondence gate per stage
HUBER = 0.003                          # m
DILATE_PX, ERODE_PX = 8, 3             # data inside the silhouette dilated by this; the coverage counts it eroded
MAX_STEPS = 30
MAX_STEPS_FINAL = 1000                 # the last stage settles slowly along a near-flat direction
STEP_M, STEP_DEG = 1e-5, 1e-3          # a stage ends below both
INLIER_M = 0.005                       # an inlier pairs within the last gate and |r| <= this
MIN_INLIERS = 1500
MAX_COND = 1e6                         # the 6x6 normal matrix, rotation scaled by COND_L
COND_L = 0.1                           # m
SIDES = ("Right", "Left")              # the left arm's links and joints are the right's names with "Left"
VIEW_MIN_POINTS = 300                  # per link, visible model points inside the image
VIEW_BORDER_PX = 2


@dataclass(frozen=True)
class FitConfig:
    """how B0 computes; REFERENCE = as it ran on phantom14 (4 Nov 2026)"""
    tree_per_stage: bool = False   # one KD-tree per stage on the E-free model points, data moved by E^-1 (exact)
    shared_render: bool = False    # one z-buffer per stage for visibility and data selection (exact)
    index_coverage: bool = False   # coverage by data index, not a KD-tree lookup (exact)
    tree_fast: bool = False        # the stage's KD-tree unbalanced and not compacted (scipy cKDTree options): the same
                                   # nearest neighbours, 7-12x faster queries on these dense CAD samples (exact up to
                                   # the order among exactly coincident model points)
    tree_leafsize: int = 16
    fast_render: bool = False      # the z-buffer by ufunc minimum.at instead of a lexsort, one projection shared with
                                   # the visibility test (exact: the same depth and the same winning point per pixel)
    roi_px: int | None = None      # unproject only the depth inside the encoders' silhouette grown by this (exact
                                   # when it covers every stage's dilated silhouette)
    data_stride: int = 1           # use every k-th depth pixel in u and v (approximation)
    final_step_m: float = STEP_M   # the last stage's settle thresholds (approximation when looser)
    final_step_deg: float = STEP_DEG
    max_steps_final: int = MAX_STEPS_FINAL
    max_steps: int = MAX_STEPS     # the coarse stages' step cap (approximation when lower)
    model_voxel: float | None = None   # m: the sampled mesh points thinned to one per voxel of this size, per link
                                       # (approximation; the sampler puts ceil(edge/spacing)^2 points on every long
                                       # thin CAD triangle: 5.2 M points for 0.24 m^2 of surface at 1 mm)


REFERENCE = FitConfig()
EXACT = FitConfig(tree_per_stage=True, shared_render=True, index_coverage=True, fast_render=True, tree_fast=True,
                  tree_leafsize=64)
FAST = EXACT


def side_of(links) -> str:
    sides = {next((p for p in SIDES if n.startswith(p)), None) for n in links}
    if len(sides) != 1 or None in sides:
        raise ValueError(f"the links must all be one arm's (Right* or Left*): {list(links)}")
    return sides.pop()


def unproject(depth, K, stride: int = 1, mask=None):
    """(n,3) camera points of the valid (> 0) pixels of a depth image (camera z, pinhole K); stride: every k-th pixel
    in u and v; mask: only pixels where it is true"""
    fx, fy, cx, cy = K
    ok = depth > 0
    if mask is not None:
        ok &= mask
    if stride > 1:
        sub = np.zeros_like(ok)
        sub[::stride, ::stride] = True
        ok &= sub
    v, u = np.nonzero(ok)
    z = depth[v, u]
    return np.column_stack([(u - cx) / fx * z, (v - cy) / fy * z, z])


def _rot(w):
    a = float(np.linalg.norm(w))
    if a < 1e-15:
        return np.eye(3)
    k = w / a
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * Kx + (1 - np.cos(a)) * Kx @ Kx


def _about(c, w, v):
    """the 4x4 of a rotation w about the point c followed by the translation v"""
    T = np.eye(4)
    R = _rot(w)
    T[:3, :3] = R
    T[:3, 3] = c - R @ c + v
    return T


def angle_deg(T) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(np.asarray(T)[:3, :3]) - 1) / 2, -1, 1))))


class ArmModel:
    """the fitted links: points P and normals N in their own link frames, link_id per point, and their chains"""

    def __init__(self, urdf, link_meshes: dict, spacing: float = 0.001, voxel: float | None = None):
        self.urdf = Urdf.of(urdf)
        self.voxel = voxel
        self.links = list(link_meshes)
        self.meshes = {n: (np.asarray(V, float), np.asarray(F, np.int64)) for n, (V, F) in link_meshes.items()}
        self.chains = {n: self.urdf.chain("Torso", n) for n in self.links}
        self.side = side_of(self.links)
        self.wrist = self.urdf.chain("Torso", f"{self.side}WristRollLink")
        P, N, lid = [], [], []
        for k, n in enumerate(self.links):
            p, nn, _ = mesh.sample(*link_meshes[n], spacing)
            if voxel:
                keep = np.sort(np.unique(np.floor(p / voxel).astype(np.int64), axis=0, return_index=True)[1])
                p, nn = p[keep], nn[keep]
            P.append(p)
            N.append(nn)
            lid.append(np.full(len(p), k))
        self.P, self.N, self.link_id = np.vstack(P), np.vstack(N), np.concatenate(lid)
        ends = np.cumsum([len(p) for p in P])
        self._slices = [slice(a, b) for a, b in zip(np.r_[0, ends[:-1]], ends)]   # each link's points, in order

    @classmethod
    def load(cls, urdf, mesh_dir, links, spacing: float = 0.001, voxel: float | None = None):
        return cls(urdf, {n: mesh.load_obj(Path(mesh_dir) / f"{n}.obj") for n in links}, spacing, voxel)

    def with_urdf(self, urdf) -> "ArmModel":
        """the same sampled meshes placed by another URDF's chains (no resampling)"""
        new = object.__new__(ArmModel)
        new.__dict__.update(self.__dict__)
        new.urdf = Urdf.of(urdf)
        new.chains = {n: new.urdf.chain("Torso", n) for n in self.links}
        new.wrist = new.urdf.chain("Torso", f"{self.side}WristRollLink")
        return new

    def link_T(self, q: dict) -> list:
        return [self.chains[n].T(q) for n in self.links]

    def place(self, q: dict, T_CT, E=None, idx=None):
        """(P (k,3), N (k,3), link_id) in the camera: E · T_CT · T_TL(q); idx: only those points"""
        M = np.asarray(T_CT, float) if E is None else np.asarray(E, float) @ np.asarray(T_CT, float)
        P0, N0, lid = (self.P, self.N, self.link_id) if idx is None else (self.P[idx], self.N[idx], self.link_id[idx])
        P, N = np.empty_like(P0), np.empty_like(N0)
        for k, T in enumerate(self.link_T(q)):
            A = M @ T
            m = self._slices[k] if idx is None else lid == k
            P[m] = P0[m] @ A[:3, :3].T + A[:3, 3]
            N[m] = N0[m] @ A[:3, :3].T
        return P, N, lid

    def place_meshes(self, q: dict, T_CT, E=None) -> list:
        """[(V (n,3) in the camera, F)] per link, placed as place()"""
        M = np.asarray(T_CT, float) if E is None else np.asarray(E, float) @ np.asarray(T_CT, float)
        out = []
        for n, T in zip(self.links, self.link_T(q)):
            A = M @ T
            V, F = self.meshes[n]
            out.append((V @ A[:3, :3].T + A[:3, 3], F))
        return out

    def wrist_point_T(self, q: dict):
        return self.wrist.T(q)[:3, 3]


def wrist_offset(arm: ArmModel, q: dict, T_TC, E_c):
    """where E (camera frame) puts the wrist point minus where the encoders put it, Torso frame (m)"""
    T_TC = np.asarray(T_TC, float)
    w = arm.wrist_point_T(q)
    seen = T_TC @ np.asarray(E_c, float) @ np.linalg.inv(T_TC) @ np.r_[w, 1.0]
    return seen[:3] - w


def zbuffer(P, K, W: int, H: int):
    """mesh.render(P, I, K, W, H) computed with ufunc minimum.at (the same depth and, among equal depths, the same
    lowest point index per pixel), plus the projection (pc, u, v, ok) for the visibility test"""
    pc, u, v, ok = mesh._project(P, np.eye(4), K, W, H)
    k = np.nonzero(ok)[0]
    lin = v[k] * W + u[k]
    z = pc[k, 2]
    flat = np.full(W * H, np.inf)
    np.minimum.at(flat, lin, z)
    win = z == flat[lin]
    big = np.iinfo(np.int64).max
    idx = np.full(W * H, big, np.int64)
    np.minimum.at(idx, lin[win], k[win])
    depth = np.where(np.isfinite(flat), flat, 0.0).reshape(H, W)
    idx = np.where(idx == big, -1, idx).reshape(H, W)
    return depth, idx, (pc, u, v, ok)


def visible_from(N, depth, proj, tol: float = 0.003, facing: float = -0.1):
    """mesh.visible(P, N, I, K, W, H, two_sided=True, depth=depth) from zbuffer()'s projection"""
    pc, u, v, ok = proj
    H, W = depth.shape
    view = pc / np.linalg.norm(pc, axis=1, keepdims=True)
    nv = np.einsum("ij,ij->i", np.asarray(N, np.float64), view)
    zb = np.where(ok, depth[np.clip(v, 0, H - 1), np.clip(u, 0, W - 1)], 0.0)
    return ok & (np.abs(nv) > -facing) & (zb > 0) & (pc[:, 2] <= zb + tol)


def visible(P, N, K, W: int, H: int, depth=None):
    """the camera-frame model points nearest in their pixel and not seen edge-on, on EITHER side: the CAD meshes are
    open and partly wound inward, so a facing test would drop real surface; the point-to-plane residual does not care
    about a normal's sign"""
    return mesh.visible(P, N, np.eye(4), K, W, H, two_sided=True, depth=depth)


def view_gate(arm: ArmModel, q: dict, T_TC, K, W: int, H: int) -> dict:
    """{"ok", "why", "points" {link: n}, "hand_px"}: at the encoders' FK, each fitted link projects at least
    VIEW_MIN_POINTS visible model points VIEW_BORDER_PX inside the image, and the hand (the GripperYaw link's origin)
    projects inside it"""
    P, N, lid = arm.place(q, np.linalg.inv(np.asarray(T_TC, float)))
    vis = visible(P, N, K, W, H)
    fx, fy, cx, cy = K
    with np.errstate(divide="ignore", invalid="ignore"):
        u, v = fx * P[:, 0] / P[:, 2] + cx, fy * P[:, 1] / P[:, 2] + cy
    b = VIEW_BORDER_PX
    inside = vis & (P[:, 2] > 0) & (u >= b) & (u <= W - 1 - b) & (v >= b) & (v <= H - 1 - b)
    points = {n: int((inside & (lid == k)).sum()) for k, n in enumerate(arm.links)}
    why = [f"{n} {c} points in view (< {VIEW_MIN_POINTS})" for n, c in points.items() if c < VIEW_MIN_POINTS]
    hand = arm.urdf.chain("Torso", f"{arm.side}GripperYawLink").T(q)[:3, 3]
    h = np.linalg.inv(np.asarray(T_TC, float)) @ np.r_[hand, 1.0]
    hu, hv = (fx * h[0] / h[2] + cx, fy * h[1] / h[2] + cy) if h[2] > 0 else (np.nan, np.nan)
    if not (h[2] > 0 and 0 <= hu <= W - 1 and 0 <= hv <= H - 1):
        why.append("the hand is out of view")
    return {"ok": not why, "why": "not all in view: " + "; ".join(why) if why else "", "points": points,
            "hand_px": [float(hu), float(hv)]}


def _pixels(P, K, W, H):
    fx, fy, cx, cy = K
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.rint(fx * P[:, 0] / P[:, 2] + cx).astype(np.int64)
        v = np.rint(fy * P[:, 1] / P[:, 2] + cy).astype(np.int64)
    ok = (P[:, 2] > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return u, v, ok


def _grow(sil, px):
    import cv2
    if px > 0:
        return cv2.dilate(sil, np.ones((2 * px + 1, 2 * px + 1), np.uint8))
    if px < 0:
        return cv2.erode(sil, np.ones((-2 * px + 1, -2 * px + 1), np.uint8))
    return sil


def _select(data, model_P, K, W, H, px, depth=None):
    """the data points inside the model's silhouette grown by px (shrunk when px < 0); depth: its z-buffer if
    already rendered"""
    if depth is None:
        depth, _ = mesh.render(model_P, np.eye(4), K, W, H)
    sil = _grow((depth > 0).astype(np.uint8), px)
    u, v, ok = _pixels(data, K, W, H)
    keep = ok.copy()
    keep[ok] = sil[v[ok], u[ok]] > 0
    return keep


@dataclass
class _Pairs:
    d: np.ndarray       # data points
    m: np.ndarray       # their model points
    n: np.ndarray       # model normals
    mi: np.ndarray      # model point index
    r: np.ndarray       # n . (d - m)
    di: np.ndarray | None = None   # the data points' index in the stage's data


def _pairs(data, P, N, iv, gate, tree=None, Einv=None):
    """each data point's nearest model point within gate; P, N are the model points iv (indices into the model).
    tree/Einv: a KD-tree on the E-free model points of the stage and E^-1 -- the same pairs, found by moving the data
    instead of rebuilding the tree on the moved model"""
    from scipy.spatial import cKDTree
    if len(iv) == 0 or len(data) == 0:
        z = np.zeros((0, 3))
        return _Pairs(z, z, z, np.zeros(0, np.int64), np.zeros(0), np.zeros(0, np.int64))
    if tree is None:
        dist, j = cKDTree(P).query(data, distance_upper_bound=gate)
    else:
        dist, j = tree.query(data @ Einv[:3, :3].T + Einv[:3, 3], distance_upper_bound=gate)
    ok = np.isfinite(dist)
    jj = j[ok]
    d, m, n = data[ok], P[jj], N[jj]
    return _Pairs(d, m, n, iv[jj], np.einsum("ij,ij->i", n, d - m), np.nonzero(ok)[0])


def _pairs_moved(data, P0, N0, iv, gate, tree, E):
    """_pairs against the E-free model points P0 (the stage's tree) moved by E: only the paired ones are moved"""
    if len(iv) == 0 or len(data) == 0:
        z = np.zeros((0, 3))
        return _Pairs(z, z, z, np.zeros(0, np.int64), np.zeros(0), np.zeros(0, np.int64))
    Ei = np.linalg.inv(E)
    X = data @ Ei[:3, :3].T + Ei[:3, 3]
    dist, j = tree.query(X, distance_upper_bound=gate)
    ok = np.isfinite(dist)
    jj = j[ok]
    R, t = E[:3, :3], E[:3, 3]
    d, m, n = data[ok], P0[jj] @ R.T + t, N0[jj] @ R.T
    return _Pairs(d, m, n, iv[jj], np.einsum("ij,ij->i", n, d - m), np.nonzero(ok)[0])


def _solve(arm: ArmModel, q, T_TC, data_all, K, W, H, cfg: FitConfig, gates=GATES, huber=HUBER,
           dilate_px=DILATE_PX):
    """E (camera frame) of one capture; returns (E, converged, steps, last pairs, cond, last gate, stage data)"""
    from scipy.spatial import cKDTree
    E = np.eye(4)
    steps, converged = 0, True
    last, cond = None, np.inf
    T_CT = np.linalg.inv(T_TC)
    base = arm.place(q, T_CT)                          # E-free model points and normals, all of them
    for gate in gates:
        P, N, _ = arm.place(q, T_CT, E)
        if cfg.fast_render:
            zbuf, _, proj = zbuffer(P, K, W, H)
            vis = np.nonzero(visible_from(N, zbuf, proj))[0]
            data = data_all[_select(data_all, P, K, W, H, dilate_px, depth=zbuf)]
        elif cfg.shared_render:
            zbuf, _ = mesh.render(P, np.eye(4), K, W, H)
            vis = np.nonzero(visible(P, N, K, W, H, depth=zbuf))[0]
            data = data_all[_select(data_all, P, K, W, H, dilate_px, depth=zbuf)]
        else:
            vis = np.nonzero(visible(P, N, K, W, H))[0]
            data = data_all[_select(data_all, P, K, W, H, dilate_px)]
        tree = None
        if cfg.tree_per_stage and len(vis):
            tree = cKDTree(base[0][vis], leafsize=cfg.tree_leafsize,
                           **(dict(balanced_tree=False, compact_nodes=False) if cfg.tree_fast else {}))
        if tree is not None:
            P0, N0 = base[0][vis], base[1][vis]
        final = gate == gates[-1]
        step_m, step_deg = (cfg.final_step_m, cfg.final_step_deg) if final else (STEP_M, STEP_DEG)
        done = False
        for _ in range(cfg.max_steps_final if final else cfg.max_steps):
            if tree is not None:
                pr = _pairs_moved(data, P0, N0, vis, gate, tree, E)
            else:
                Pv, Nv, _ = arm.place(q, T_CT, E, vis)
                pr = _pairs(data, Pv, Nv, vis, gate)
            last = pr
            if len(pr.r) < 12:
                converged = False
                break
            c = pr.m.mean(0)
            A = np.column_stack([np.cross(pr.m - c, pr.n), pr.n])     # rotation about c, translation
            a = np.abs(pr.r)
            w = np.where(a <= huber, 1.0, huber / np.maximum(a, 1e-12))
            Hm = A.T @ (A * w[:, None])
            g = A.T @ (w * pr.r)
            try:
                x = np.linalg.solve(Hm, g)
            except np.linalg.LinAlgError:
                converged = False
                break
            S = np.diag([1 / COND_L] * 3 + [1.0] * 3)
            cond = float(np.linalg.cond(S @ Hm @ S))
            E = _about(c, x[:3], x[3:6]) @ E
            steps += 1
            if np.linalg.norm(x[3:6]) < step_m and np.degrees(np.linalg.norm(x[:3])) < step_deg:
                done = True
                break
        if not converged:
            break
        if not done and final:
            converged = False          # only the last stage must settle; a coarse one at its cap hands on
    if last is not None:               # the statistics at the final estimate, not at the one before its last step
        if tree is not None:
            last = _pairs_moved(data, P0, N0, vis, gate, tree, E)
        else:
            Pv, Nv, _ = arm.place(q, T_CT, E, vis)
            last = _pairs(data, Pv, Nv, vis, gate)
    return E, converged, steps, last, cond, gate


@dataclass
class PoseFit:
    delta_c: np.ndarray        # E: the model moved by this (camera frame) matches the depth
    wrist_offset_T: np.ndarray  # m, Torso frame
    rotation_deg: float
    converged: bool
    ok: bool                   # converged, enough inliers, conditioned
    cond: float
    steps: int
    data: np.ndarray           # the data points of the last pairs
    inlier: np.ndarray         # (len(data),) bool
    residual_mm: np.ndarray    # (len(data),) n . (d - m)
    model_index: np.ndarray    # (len(data),) the paired model point
    per_link: dict
    rms_mm: float
    coverage: float            # inliers / data points inside the silhouette eroded by ERODE_PX


def _stats(arm, pr, K, W, H, data_all, model_P, gate, cfg: FitConfig):
    inl = (np.linalg.norm(pr.d - pr.m, axis=1) <= gate) & (np.abs(pr.r) <= INLIER_M)
    per = {}
    for k, n in enumerate(arm.links):
        m = inl & (arm.link_id[pr.mi] == k)
        per[n] = {"inliers": int(m.sum()), "rms_mm": float(1e3 * np.sqrt(np.mean(pr.r[m] ** 2))) if m.any() else None}
    inner = _select(data_all, model_P, K, W, H, -ERODE_PX,
                    depth=zbuffer(model_P, K, W, H)[0] if cfg.fast_render else None)
    if cfg.index_coverage:
        coverage = _coverage_by_index(pr, inl, data_all, inner, K, W, H, model_P)
    else:
        inner_pts = data_all[inner]
        if len(inner_pts) and inl.any():
            from scipy.spatial import cKDTree
            hit = cKDTree(pr.d[inl]).query(inner_pts, distance_upper_bound=1e-9)[0]
            coverage = float(np.isfinite(hit).mean())
        else:
            coverage = 0.0
    rms = float(1e3 * np.sqrt(np.mean(pr.r[inl] ** 2))) if inl.any() else float("nan")
    return inl, per, rms, coverage


def _coverage_by_index(pr, inl, data_all, inner, K, W, H, model_P):
    """the fraction of the inner data points that are inliers. The reference asks a KD-tree whether each inner point
    coincides with an inlier; here the inliers are marked on the pixel grid (data points are one per pixel, and the
    stage's data is a subset of data_all), which is the same set"""
    if not inner.any() or not inl.any():
        return 0.0
    u, v, _ = _pixels(pr.d[inl], K, W, H)
    hit = np.zeros((H, W), bool)
    hit[v, u] = True
    ui, vi, _ = _pixels(data_all[inner], K, W, H)
    return float(hit[vi, ui].mean())


def fit_pose(arm: ArmModel, q: dict, T_TC, data, K, W: int, H: int, cfg: FitConfig = FAST, **kw) -> PoseFit:
    """B0: the rigid correction E (camera frame) of one capture's model to its depth points"""
    T_TC = np.asarray(T_TC, float)
    data = np.asarray(data, float)
    E, converged, steps, last, cond, gate = _solve(arm, q, T_TC, data, K, W, H, cfg, **kw)
    pr = last if last is not None else _Pairs(*(np.zeros((0, 3)),) * 3, np.zeros(0, np.int64), np.zeros(0))
    P, _, _ = arm.place(q, np.linalg.inv(T_TC), E)
    inl, per, rms, coverage = _stats(arm, pr, K, W, H, data, P, gate, cfg)
    ok = bool(converged and inl.sum() >= MIN_INLIERS and np.isfinite(cond) and cond <= MAX_COND)
    return PoseFit(delta_c=E, wrist_offset_T=wrist_offset(arm, q, T_TC, E), rotation_deg=angle_deg(E),
                   converged=converged, ok=ok, cond=cond, steps=steps, data=pr.d, inlier=inl,
                   residual_mm=1e3 * pr.r, model_index=pr.mi, per_link=per, rms_mm=rms, coverage=coverage)


def with_config(cfg: FitConfig, **kw) -> FitConfig:
    return replace(cfg, **kw)
