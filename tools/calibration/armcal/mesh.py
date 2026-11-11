"""(Ported verbatim from OKVIS2-X tools/burst_pointcloud/mesh.py, c5b4c6b0.) Triangle meshes for the arm session's read B (Task 5 plan Task 2): OBJ files -> area-sampled surface points with
outward normals, and their z-buffer splat into the rectified depth camera.

  load_obj(path)                 -> V (n,3), F (m,3) vertex indices (polygons fanned into triangles)
  sample(V, F, spacing)          -> P (k,3), N (k,3) unit outward normals, w (k,) the area each point stands for (m²)
  render(P, T_CM, K, W, H)       -> depth (H,W) camera z, 0 = nothing; idx (H,W) the nearest point per pixel, -1 none
  visible(P, N, T_CM, K, W, H)   -> (k,) bool: the point is nearest in its pixel (within tol) and faces the camera

Sampling is deterministic: each triangle is cut into n² equal sub-triangles (n = ceil(longest edge / spacing)) and
each sub-triangle's centroid is one point carrying area / n², so no point is farther than `spacing` from its
neighbours and the weights sum to the mesh's area. Normals are the faces' (right-hand winding), all flipped when the
mesh's signed volume is negative (inward winding). The splat puts each point at its nearest pixel (pinhole K = fx, fy,
cx, cy; pixel centres at integers): sample at well under the pixel footprint (z / fx: ~2 mm at 0.4 m) for a filled
silhouette. T_CM maps model coordinates into the camera."""
from __future__ import annotations

import numpy as np


def load_obj(path):
    V, F = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                V.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                idx = [int(t.split("/")[0]) for t in line.split()[1:]]
                idx = [i - 1 if i > 0 else len(V) + i for i in idx]           # OBJ: 1-based, negative = from the end
                F.extend([idx[0], idx[k], idx[k + 1]] for k in range(1, len(idx) - 1))
    return np.asarray(V, np.float64).reshape(-1, 3), np.asarray(F, np.int64).reshape(-1, 3)


def _bary(n: int) -> np.ndarray:
    """(n², 2) barycentric (s, t) of the centroids of a triangle's n² equal sub-triangles"""
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    up = (i + j <= n - 1)
    down = (i + j <= n - 2)
    return np.vstack([np.column_stack([(i[up] + 1 / 3) / n, (j[up] + 1 / 3) / n]),
                      np.column_stack([(i[down] + 2 / 3) / n, (j[down] + 2 / 3) / n])])


def sample(V, F, spacing: float):
    V, F = np.asarray(V, np.float64), np.asarray(F, np.int64)
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cr = np.cross(B - A, C - A)
    area2 = np.linalg.norm(cr, axis=1)
    keep = area2 > 1e-14
    A, B, C, cr, area2 = A[keep], B[keep], C[keep], cr[keep], area2[keep]
    sign = 1.0 if np.einsum("ij,ij->i", A, np.cross(B, C)).sum() >= 0 else -1.0     # 6 x the signed volume
    normal = sign * cr / area2[:, None]
    longest = np.max(np.stack([np.linalg.norm(B - A, axis=1), np.linalg.norm(C - B, axis=1),
                               np.linalg.norm(A - C, axis=1)]), axis=0)
    ns = np.maximum(1, np.ceil(longest / spacing - 1e-9)).astype(np.int64)
    P, N, w = [], [], []
    for n in np.unique(ns):
        m = ns == n
        st = _bary(int(n))
        a, ab, ac = A[m][:, None], (B - A)[m][:, None], (C - A)[m][:, None]
        P.append((a + st[None, :, :1] * ab + st[None, :, 1:] * ac).reshape(-1, 3))
        N.append(np.repeat(normal[m], len(st), axis=0))
        w.append(np.repeat(area2[m] / 2 / n ** 2, len(st)))
    if not P:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    return np.vstack(P), np.vstack(N), np.concatenate(w)


def _project(P, T_CM, K, W, H):
    """camera points, pixel (u, v) ints, and the mask of those in front of the camera and inside the image"""
    T = np.asarray(T_CM, np.float64)
    pc = np.asarray(P, np.float64) @ T[:3, :3].T + T[:3, 3]
    z = pc[:, 2]
    fx, fy, cx, cy = K
    ok = z > 1e-6
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.rint(np.where(ok, fx * pc[:, 0] / z + cx, -1)).astype(np.int64)
        v = np.rint(np.where(ok, fy * pc[:, 1] / z + cy, -1)).astype(np.int64)
    ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return pc, u, v, ok


def render(P, T_CM, K, W: int, H: int):
    pc, u, v, ok = _project(P, T_CM, K, W, H)
    depth = np.zeros((H, W))
    idx = np.full((H, W), -1, np.int64)
    k = np.nonzero(ok)[0]
    if len(k):
        lin = v[k] * W + u[k]
        order = np.lexsort((pc[k, 2], lin))                       # by pixel, nearest first
        first = np.unique(lin[order], return_index=True)[1]
        win = k[order[first]]
        depth.flat[lin[order[first]]] = pc[win, 2]
        idx.flat[lin[order[first]]] = win
    return depth, idx


def raster(V, F, T_CM, K, W: int, H: int):
    """(H,W) camera z of the nearest triangle along each pixel centre's ray, 0 = none: exact, as a stereo depth image
    is sampled (render() puts a sample at its nearest pixel, which on a slanted face reads up to half a pixel's
    footprint too near: for synthetic depth use this). Triangles with a vertex behind the camera are skipped."""
    T = np.asarray(T_CM, np.float64)
    Vc = np.asarray(V, np.float64) @ T[:3, :3].T + T[:3, 3]
    F = np.asarray(F, np.int64)
    fx, fy, cx, cy = K
    depth = np.full((H, W), np.inf)
    zs = Vc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = np.column_stack([fx * Vc[:, 0] / zs + cx, fy * Vc[:, 1] / zs + cy])
    for a, b, c in F:
        if min(zs[a], zs[b], zs[c]) <= 1e-6:
            continue
        p = uv[[a, b, c]]
        u0, v0 = max(int(np.ceil(p[:, 0].min())), 0), max(int(np.ceil(p[:, 1].min())), 0)
        u1, v1 = min(int(np.floor(p[:, 0].max())), W - 1), min(int(np.floor(p[:, 1].max())), H - 1)
        if u0 > u1 or v0 > v1:
            continue
        n = np.cross(Vc[b] - Vc[a], Vc[c] - Vc[a])
        uu, vv = np.meshgrid(np.arange(u0, u1 + 1, dtype=np.float64), np.arange(v0, v1 + 1, dtype=np.float64))
        e = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            e.append((p[j, 0] - p[i, 0]) * (vv - p[i, 1]) - (p[j, 1] - p[i, 1]) * (uu - p[i, 0]))
        inside = ((e[0] >= -1e-9) & (e[1] >= -1e-9) & (e[2] >= -1e-9)) | ((e[0] <= 1e-9) & (e[1] <= 1e-9) & (e[2] <= 1e-9))
        nd = n[0] * (uu - cx) / fx + n[1] * (vv - cy) / fy + n[2]
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (n @ Vc[a]) / nd
        inside &= np.isfinite(z) & (z > 0)
        if not inside.any():
            continue
        sub = depth[v0:v1 + 1, u0:u1 + 1]
        np.minimum(sub, np.where(inside, z, np.inf), out=sub)
    depth[~np.isfinite(depth)] = 0.0
    return depth


def visible(P, N, T_CM, K, W: int, H: int, tol: float = 0.003, facing: float = -0.1, depth=None,
            two_sided: bool = False):
    """points nearest in their pixel (within tol m of the z-buffer; `depth` = a z-buffer already rendered, e.g. with
    other occluders) whose normal faces the camera (n · view direction < facing); two_sided: either side, only not
    edge-on (|n · view| > -facing), for meshes whose winding cannot be trusted"""
    if depth is None:
        depth = render(P, T_CM, K, W, H)[0]
    pc, u, v, ok = _project(P, T_CM, K, W, H)
    R = np.asarray(T_CM, np.float64)[:3, :3]
    view = pc / np.linalg.norm(pc, axis=1, keepdims=True)
    nv = np.einsum("ij,ij->i", np.asarray(N, np.float64) @ R.T, view)
    faces = np.abs(nv) > -facing if two_sided else nv < facing
    zb = np.where(ok, depth[np.clip(v, 0, H - 1), np.clip(u, 0, W - 1)], 0.0)
    return ok & faces & (zb > 0) & (pc[:, 2] <= zb + tol)


_R2 = 1.32471795724474602596          # the plastic number: the R2 low-discrepancy sequence's base


def sample_even(V, F, spacing: float, phase: float = 0.5):
    """(P, N, w) as sample(), with about one point per spacing² of surface on every triangle, however long and thin
    (sample() puts ceil(longest edge / spacing)² points on each: 5.2 M points for an arm's 0.24 m² at 1 mm, 1.1 M of
    them visible, where this gives about 0.25 M). Each triangle gets k = max(1, round(area / spacing²)) points at
    the first k terms of the R2 sequence (folded into the triangle), started at `phase`: deterministic, evenly
    spread inside each triangle; another phase is another, equally even, choice of the points"""
    V, F = np.asarray(V, np.float64), np.asarray(F, np.int64)
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cr = np.cross(B - A, C - A)
    area2 = np.linalg.norm(cr, axis=1)
    keep = area2 > 1e-14
    A, B, C, cr, area2 = A[keep], B[keep], C[keep], cr[keep], area2[keep]
    if not len(A):
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    sign = 1.0 if np.einsum("ij,ij->i", A, np.cross(B, C)).sum() >= 0 else -1.0
    normal = sign * cr / area2[:, None]
    k = np.maximum(1, np.rint(area2 / 2 / spacing ** 2)).astype(np.int64)
    tri = np.repeat(np.arange(len(A)), k)
    j = np.arange(len(tri)) - np.repeat(np.cumsum(k) - k, k) + 1            # 1..k within each triangle
    s = np.mod(phase + j / _R2, 1.0)
    t = np.mod(phase + j / _R2 ** 2, 1.0)
    fold = s + t > 1.0
    s[fold], t[fold] = 1.0 - s[fold], 1.0 - t[fold]
    P = A[tri] + s[:, None] * (B - A)[tri] + t[:, None] * (C - A)[tri]
    return P, normal[tri], (area2 / 2 / k)[tri]


def fill_holes(depth, gap: float = 0.01, min_nearer: int = 6):
    """a z-buffer with its see-through pinholes closed: a pixel at least `gap` behind `min_nearer` or more of its 8
    neighbours (a gap in a sparse splat, through which a surface behind shows) gets the nearest of them; real depth
    edges (at most about 5 nearer neighbours) stay. Empty pixels (0) stay empty"""
    H, W = depth.shape
    z = np.where(depth > 0, depth, np.inf)
    pad = np.pad(z, 1, constant_values=np.inf)
    nb = np.stack([pad[1 + dv:1 + dv + H, 1 + du:1 + du + W] for dv in (-1, 0, 1) for du in (-1, 0, 1)
                   if dv or du])
    nearer = (nb < z - gap).sum(0)
    zmin = nb.min(0)
    return np.where((depth > 0) & (nearer >= min_nearer), zmin, depth)
