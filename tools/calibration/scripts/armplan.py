"""The arm session's pose list (docs/superpowers/plans/2026-10-22-phantom12-arm-session.md, Task 2, rev 3's
decisions T2-2..T2-9): what the head cameras see of the palm plate at an arm pose, how much the pose tells the
camera-to-arm fit, which poses to take and in what order.

One frame for everything, the torso (T). A camera c sits at T_TC_c (armchain.camera_in_torso); the plate at

    T_TP(q) = T_T,tip(q + δ) · X,     seen by camera c at   T_CP = (ΔT · T_TC_c)^-1 · T_TP

with X the plate on the gripper link, ΔT the camera-to-torso correction (both cameras move together) and δ the
arm's joint zero offsets. Parameters p, in this order: X's rotation vector and translation (right perturbation,
plate frame), ΔT's rotation vector and translation (left perturbation, torso frame), then δ of the joints listed in
`offsets` (default INNER: at one neck pose shoulder pitch's offset is ΔT and gripper yaw's is X, decision T2-2).
Angles in rad, lengths in m, pixels in the native 1280x800 image.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lensmodel  # noqa: E402
import rigmath as rm  # noqa: E402
from armchain import RIGHT_ARM, camera_in_torso  # noqa: E402

INNER = (1, 2, 3, 4, 5)                    # observable joint offsets at one neck pose (T2-2)
BIG = (0, 1, 2, 3)                         # the shoulder and elbow: where approach sides and sag matter
BOTTOM_CORNERS = (6, 8)                    # 3x3 image cells, row-major, row 0 at the top
MARGIN_PX, RANGE_M, MAX_FACING_DEG, MIN_TAG_PX = 40.0, (0.35, 0.80), 50.0, 40.0    # T2-3
MIN_RAMP_S, MAX_RATE = 6.0, 0.3            # T2-7: seconds, rad/s
LIM_MM, LIM_DEG, LIM_OFFSET_DEG, LIM_PLATE_MM = 0.5, 0.05, 0.05, 1.0              # T2-9
FD_STEP = 1e-6                             # finite-difference step for the Jacobians (rad, m)
MIN_Z_M = 0.05                             # a corner closer to the camera plane than this is not projected
ELBOW_BELOW_M, MIN_ROLL_DEG, MAX_PITCH_DEG = 0.05, -60.0, 0.0     # T2-4b: the arm low and forward


@dataclass
class Camera:
    T_TC: np.ndarray       # the camera in the torso
    model: str             # lensmodel name: KB4T or KB4
    intr: np.ndarray       # fx fy cx cy + distortion
    size: tuple            # (width, height) px


def cameras_from_config(cfg, neck, q_yaw: float, q_pitch: float, size=(1280, 800)) -> list:
    """Both cameras of an OKVIS config (okviscfg.OkvisRig) in the torso at the given neck angles (rad)."""
    model = {"pinhole-kb4t": "KB4T", "pinhole-equi": "KB4"}[cfg.lens]
    return [Camera(camera_in_torso(neck, cfg.T_BS, T_SC, q_yaw, q_pitch), model,
                   np.r_[K[0, 0], K[1, 1], K[0, 2], K[1, 2], np.asarray(D, float).ravel()], tuple(size))
            for T_SC, K, D in zip(cfg.T_SC, cfg.K, cfg.D)]


def plate_points(grid):
    """(16 x 3 corners on the board, [(tag, k)]), tag by tag, each tag's 4 corners in order around the tag."""
    keys = [(t, k) for t in range(grid.n_tags) for k in range(4)]
    return np.array([grid.obj_points(t)[k] for t, k in keys], float), keys


def _apply(T, P):
    """T (..., 4, 4) applied to points P (M, 3) -> (..., M, 3)."""
    return np.einsum("...ij,mj->...mi", T[..., :3, :3], P) + T[..., None, :3, 3]


def _inv(T):
    R = np.swapaxes(T[..., :3, :3], -1, -2)
    out = np.broadcast_to(np.eye(4), T.shape).copy()
    out[..., :3, :3] = R
    out[..., :3, 3] = -np.einsum("...ij,...j->...i", R, T[..., :3, 3])
    return out


def _project(cam: Camera, Pc: np.ndarray) -> np.ndarray:
    """Camera-frame points (..., M, 3) -> pixels (..., M, 2); NaN for points too close to or behind the camera."""
    sh = Pc.shape
    uv = lensmodel.project(cam.model, cam.intr, Pc.reshape(-1, 3)).reshape(sh[:-1] + (2,))
    uv[Pc[..., 2] < MIN_Z_M] = np.nan
    return uv


def view(cams, T_TP: np.ndarray, P: np.ndarray, grid) -> dict:
    """What each camera sees of the plate at poses T_TP (N, 4, 4). Per pose and camera (N, C): range_m (camera to
    plate centre), facing_deg (the plate's printed side vs the direction to the camera: 0 square on, > 90 its
    back), tag_px (the shortest projected tag side), margin_px (the corners' least distance to an image edge;
    -inf if a corner is behind the camera); uv (N, C, 16, 2)."""
    T_TP = np.asarray(T_TP, float)
    c = P.mean(0)
    out = {k: [] for k in ("uv", "range_m", "facing_deg", "tag_px", "margin_px")}
    for cam in cams:
        T_CP = _inv(cam.T_TC) @ T_TP
        Pc = _apply(T_CP, P)
        uv = _project(cam, Pc)
        cc = T_CP[:, :3, :3] @ c + T_CP[:, :3, 3]
        n = T_CP[:, :3, 2]                     # board +z (Kalibr: x right, y up on the print) = out of the print
        cosf = np.einsum("ni,ni->n", n, -cc) / np.linalg.norm(cc, axis=1)
        q = uv.reshape(len(T_TP), grid.n_tags, 4, 2)
        side = np.linalg.norm(np.roll(q, -1, axis=2) - q, axis=3).min(axis=(1, 2))
        w, h = cam.size
        m = np.minimum.reduce([uv[..., 0], w - 1 - uv[..., 0], uv[..., 1], h - 1 - uv[..., 1]]).min(axis=1)
        bad = np.isnan(uv).any(axis=(1, 2))
        out["uv"].append(uv)
        out["range_m"].append(np.linalg.norm(cc, axis=1))
        out["facing_deg"].append(np.degrees(np.arccos(np.clip(cosf, -1, 1))))
        out["tag_px"].append(np.where(bad, 0.0, side))
        out["margin_px"].append(np.where(bad, -np.inf, m))
    return {"uv": np.stack(out["uv"], 1), **{k: np.stack(out[k], 1) for k in out if k != "uv"}}


def visible(v: dict, margin_px=MARGIN_PX, range_m=RANGE_M, max_facing_deg=MAX_FACING_DEG,
            min_tag_px=MIN_TAG_PX) -> np.ndarray:
    """T2-3, in every camera: (N,) bool."""
    ok = ((v["margin_px"] >= margin_px) & (v["range_m"] >= range_m[0]) & (v["range_m"] <= range_m[1])
          & (v["facing_deg"] <= max_facing_deg) & (v["tag_px"] >= min_tag_px))
    return ok.all(axis=1)


def workspace_ok(arm, Q, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """T2-4b (rev 3 amendment): (N,) the elbow ELBOW_BELOW_M under the shoulder joint, the wrist not above it,
    shoulder roll >= MIN_ROLL_DEG and shoulder pitch <= MAX_PITCH_DEG (forward), so the arm stays out of the
    unmodelled hoist rigging over the shoulders. Heights along `up`, world up in the torso frame (the robot hangs
    leaning: minus gravity, from the IMU)."""
    Q = np.atleast_2d(np.asarray(Q, float))
    u = np.asarray(up, float) / np.linalg.norm(up)
    fr = arm.frames(Q)
    zs, ze, zw = (fr[i][:, :3, 3] @ u for i in (0, 3, 5))
    return ((ze <= zs - ELBOW_BELOW_M) & (zw <= zs) & (Q[:, 1] >= np.radians(MIN_ROLL_DEG))
            & (Q[:, 0] <= np.radians(MAX_PITCH_DEG)))


def plate_in_torso(arm, X, Q, p=None, offsets=INNER) -> np.ndarray:
    """T_TP (N, 4, 4) at arm angles Q (N, 7), with parameters p (module docstring) applied to X and the joints."""
    Q = np.atleast_2d(np.asarray(Q, float))
    if p is not None:
        X = X @ rm.transform(rm.exp_rot(p[0:3]), p[3:6])
        Q = Q.copy()
        Q[:, list(offsets)] += p[12:]
    return arm.T_base_tip(Q) @ X


def pixels(arm, cams, X, Q, P, p=None, offsets=INNER) -> np.ndarray:
    """Every corner's pixels in every camera, (N, C * 16 * 2), at parameters p (zeros: the nominal chain)."""
    offsets = tuple(offsets)
    if p is None:
        p = np.zeros(12 + len(offsets))
    T_TP = plate_in_torso(arm, X, Q, p, offsets)
    dT = rm.transform(rm.exp_rot(p[6:9]), p[9:12])
    out = [_project(c, _apply(_inv(dT @ c.T_TC) @ T_TP, P)) for c in cams]
    return np.stack(out, 1).reshape(len(T_TP), -1)


def jacobian(arm, cams, X, Q, P, offsets=INNER) -> np.ndarray:
    """d pixels / d p at p = 0, central differences: (N, C * 16 * 2, 12 + len(offsets))."""
    offsets = tuple(offsets)
    n = 12 + len(offsets)
    cols = []
    for i in range(n):
        d = np.zeros(n)
        d[i] = FD_STEP
        cols.append((pixels(arm, cams, X, Q, P, d, offsets) - pixels(arm, cams, X, Q, P, -d, offsets)) / (2 * FD_STEP))
    return np.stack(cols, -1)


def information(arm, cams, X, Q, P, sigma_px: float, offsets=INNER) -> np.ndarray:
    """Per pose J^T J / sigma^2, (N, n, n): one image pair per pose, every corner seen."""
    J = jacobian(arm, cams, X, Q, P, offsets)
    return np.einsum("nri,nrj->nij", J, J) / sigma_px ** 2


def plate_sigma_mm(cams, T_TP, P, sigma_px: float) -> np.ndarray:
    """(N,) the plate centre's position error (largest axis, 1 sigma, mm) when its pose is measured from one image
    pair: both cameras, every corner, sigma_px per coordinate (rev 2, item 6)."""
    T_TP = np.asarray(T_TP, float)
    c = P.mean(0)

    def obs(d):
        T = T_TP @ rm.transform(rm.exp_rot(d[:3]), d[3:])
        return (np.stack([_project(cam, _apply(_inv(cam.T_TC) @ T, P)) for cam in cams], 1).reshape(len(T), -1),
                T[:, :3, :3] @ c + T[:, :3, 3])

    J, Jc = [], []
    for i in range(6):
        d = np.zeros(6)
        d[i] = FD_STEP
        (a, ca), (b, cb) = obs(d), obs(-d)
        J.append((a - b) / (2 * FD_STEP))
        Jc.append((ca - cb) / (2 * FD_STEP))
    J, Jc = np.stack(J, -1), np.stack(Jc, -1)
    cov = np.linalg.inv(np.einsum("nri,nrj->nij", J, J)) * sigma_px ** 2
    cc = np.einsum("nai,nij,nbj->nab", Jc, cov, Jc)
    return np.sqrt(np.linalg.eigvalsh(cc)[:, -1]) * 1e3


def image_cell(uv: np.ndarray, size=(1280, 800)) -> np.ndarray:
    """3x3 cell (row-major, row 0 at the top) of pixels (N, 2)."""
    col = np.clip((uv[:, 0] / (size[0] / 3)).astype(int), 0, 2)
    row = np.clip((uv[:, 1] / (size[1] / 3)).astype(int), 0, 2)
    return row * 3 + col


def select(info, cell, third, Q, n_pick: int, min_per_cell: int, min_bottom: int, min_per_third: int, prior,
           min_sep_deg: float = 5.0) -> list:
    """T2-6: greedy D-optimal choice of n_pick candidates (information (N, n, n)). While a coverage rule is unmet,
    only candidates meeting the most unmet rules compete; among them the largest log det of the information so far
    wins. Rules: min_per_cell in every image cell that at least min_per_cell candidates reach; min_bottom in
    BOTTOM_CORNERS together; min_per_third in each range third (each capped by what the candidates allow). Two
    picks differ by at least min_sep_deg in some joint. coverage() re-checks the rules on the result."""
    cell, third, Q = np.asarray(cell), np.asarray(third), np.asarray(Q, float)
    n = len(info)
    reach = {c: int((cell == c).sum()) for c in range(9)}
    need_cell = {c: min(min_per_cell, reach[c]) for c in range(9) if min_per_cell > 0 and reach[c] > 0}
    need_bottom = min(min_bottom, int(np.isin(cell, BOTTOM_CORNERS).sum()))
    need_third = {t: min(min_per_third, int((third == t).sum())) for t in range(3)}
    A = np.array(prior, float).copy()
    taken = np.zeros(n, bool)
    sel = []
    for _ in range(n_pick):
        avail = ~taken
        if sel:
            gap = np.degrees(np.abs(Q[:, None, :] - Q[sel][None]).max(axis=2)).min(axis=1)
            avail &= gap >= min_sep_deg
        if not avail.any():
            break
        got_cell = {c: int((cell[sel] == c).sum()) for c in range(9)}
        got_bottom = int(np.isin(cell[sel], BOTTOM_CORNERS).sum())
        got_third = {t: int((third[sel] == t).sum()) for t in range(3)}
        helps = np.zeros(n, int)
        for c, k in need_cell.items():
            helps += (cell == c) & (got_cell[c] < k)
        helps += np.isin(cell, BOTTOM_CORNERS) & (got_bottom < need_bottom)
        for t, k in need_third.items():
            helps += (third == t) & (got_third[t] < k)
        pool = avail & (helps == helps[avail].max()) if helps[avail].max() > 0 else avail
        idx = np.flatnonzero(pool)
        gain = np.linalg.slogdet(A[None] + info[idx])[1]
        if not np.isfinite(gain).any():
            raise ValueError("no candidate has a finite information gain (NaN pixels?)")
        best = int(idx[np.nanargmax(np.where(np.isfinite(gain), gain, np.nan))])
        sel.append(best)
        taken[best] = True
        A = A + info[best]
    return sel


def coverage(cell, third, sel, min_per_cell: int, min_bottom: int, min_per_third: int) -> list:
    """T2-6's rules on a selection, capped by what the candidates (cell, third: (N,)) reach: the unmet ones, named."""
    cell, third, sel = np.asarray(cell), np.asarray(third), list(sel)
    miss = []
    for c in range(9):
        need = min(min_per_cell, int((cell == c).sum()))
        got = int((cell[sel] == c).sum())
        if got < need:
            miss.append(f"cell {c}: {got} of {need}")
    need = min(min_bottom, int(np.isin(cell, BOTTOM_CORNERS).sum()))
    got = int(np.isin(cell[sel], BOTTOM_CORNERS).sum())
    if got < need:
        miss.append(f"bottom corners: {got} of {need}")
    for t in range(3):
        need, got = min(min_per_third, int((third == t).sum())), int((third[sel] == t).sum())
        if got < need:
            miss.append(f"third {t}: {got} of {need}")
    return miss


def verify(stops_, start, pose_problems, path_problems) -> tuple:
    """Every stop's pose (pose_problems(q) -> [reasons]; the return is the start pose, where the arm already is,
    so only its path) and every move's path from the previous stop (path_problems(a, b) -> [reasons]).
    -> (problems, poses whose repeat must be dropped: a failing pre-pose, repeat, or path into either)."""
    problems, drop = [], set()
    prev = np.asarray(start, float)
    for i, s in enumerate(stops_):
        q = np.asarray(s["q"], float)
        why = ([] if s["kind"] == "return" else list(pose_problems(q))) + list(path_problems(prev, q))
        if why:
            problems.append(f"stop {i} ({s['kind']}{'' if 'pose' not in s else ' of pose %d' % s['pose']}): "
                            + "; ".join(why))
            if s["kind"] in ("pre", "repeat_same", "repeat_opp"):
                drop.add(s["pose"])
        prev = q
    return problems, drop


def move_cost(a, b) -> float:
    """The largest joint change (rad): the ramp's driver."""
    return float(np.abs(np.asarray(b, float) - np.asarray(a, float)).max())


def ramp_s(a, b) -> float:
    """T2-7: max(6 s, largest change / 0.3 rad/s)."""
    return max(MIN_RAMP_S, move_cost(a, b) / MAX_RATE)


def tour_cost(Q, start, order) -> float:
    pts = [start] + [Q[k] for k in order] + [start]
    return sum(move_cost(a, b) for a, b in zip(pts, pts[1:]))


def tour(Q, start, two_opt: bool = True, allowed=None) -> list:
    """T2-7: an order of the poses Q (N, 7) from start and back to start: nearest neighbour by the largest joint
    change, then 2-opt. allowed(a, b) -> bool forbids moves (a, b are pose indices, -1 = start); a forbidden move
    costs infinity."""
    Q = np.asarray(Q, float)
    ok = allowed or (lambda a, b: True)

    def cost(a, b):
        if not ok(a, b):
            return np.inf
        return move_cost(start if a < 0 else Q[a], start if b < 0 else Q[b])

    left, cur, order = set(range(len(Q))), -1, []
    while left:
        nxt = min(left, key=lambda k: cost(cur, k))
        order.append(nxt)
        left.discard(nxt)
        cur = nxt
    if not two_opt:
        return order
    route = [-1] + order + [-1]
    better = True
    while better:
        better = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route) - 1):
                old = cost(route[i - 1], route[i]) + cost(route[j], route[j + 1])
                new = cost(route[i - 1], route[j]) + cost(route[i], route[j + 1])
                if new < old - 1e-12:
                    route[i:j + 1] = route[i:j + 1][::-1]
                    better = True
    return route[1:-1]


def stops(Q, order, start, held_every: int = 4, n_same: int = 3, n_opp: int = 4, pre_deg: float = 6.0,
          big=BIG, repeatable=None, pre: str = "tour") -> list:
    """T2-8: the session's stops. The tour (every held_every-th pose in travel order held out), then n_same +
    n_opp repeats of fit poses spread along the tour (alternating same / opposite side, extra opposite ones last),
    each after a pre-pose. pre="inline": see _stops_inline (no move that is not the tour's own). pre="tour": the pose before it in the tour (same side: the same move again) or
    the one after it (opposite side: the same path reversed), so every pre-pose and the move from it are the tour's
    own, already checked; pre="offset": pre_deg away in the big joints, back along the original approach (same side)
    or beyond it (opposite side). Only fit poses in `repeatable` (None: all) are repeated. Last, the return to start.
    Each stop: kind, pose (index into Q), q (rad)."""
    Q, start = np.asarray(Q, float), np.asarray(start, float)
    if pre == "inline":
        return _stops_inline(Q, order, start, held_every, n_same, n_opp, repeatable)
    out, prev, came_from = [], start, {}
    for pos, k in enumerate(order):
        out.append({"kind": "heldout" if pos % held_every == held_every - 1 else "fit", "pose": int(k),
                    "q": Q[k].tolist()})
        came_from[k] = prev
        prev = Q[k]
    fit = [s["pose"] for s in out if s["kind"] == "fit" and (repeatable is None or s["pose"] in repeatable)]
    n_rep = n_same + n_opp
    if len(fit) < n_rep:
        raise ValueError(f"{n_rep} repeats asked, {len(fit)} fit poses may be repeated")
    picks = [fit[int(round(i))] for i in np.linspace(0, len(fit) - 1, n_rep)] if n_rep else []
    kinds = ["repeat_same" if (i % 2 == 0 and i // 2 < n_same) else "repeat_opp" for i in range(n_rep)]
    while kinds.count("repeat_same") < n_same:          # more same than opposite asked: fill from the end
        kinds[len(kinds) - 1 - kinds[::-1].index("repeat_opp")] = "repeat_same"
    big = list(big)
    order = list(order)
    for k, kind in zip(picks, kinds):
        if pre == "tour":
            pos = order.index(k)
            nxt = Q[order[pos + 1]] if pos + 1 < len(order) else start
            q_pre = came_from[k] if kind == "repeat_same" else nxt
        else:
            side = np.sign(Q[k][big] - came_from[k][big])
            side[side == 0] = 1.0
            q_pre = Q[k].copy()
            q_pre[big] += (-1.0 if kind == "repeat_same" else 1.0) * np.radians(pre_deg) * side
        out.append({"kind": "pre", "pose": int(k), "q": np.asarray(q_pre, float).tolist()})
        out.append({"kind": kind, "pose": int(k), "q": Q[k].tolist()})
    out.append({"kind": "return", "q": start.tolist()})
    return out


def _stops_inline(Q, order, start, held_every, n_same, n_opp, repeatable) -> list:
    """pre='inline': each repeat right where the tour visits its pose, using only the tour's own moves: same side
    prev -> k -> prev -> k (the same move again), opposite side k -> next -> k -> next (the same path reversed)."""
    order = list(order)
    kind_at = ["heldout" if pos % held_every == held_every - 1 else "fit" for pos in range(len(order))]
    cand = [pos for pos in range(1, len(order) - 1) if kind_at[pos] == "fit"
            and (repeatable is None or order[pos] in repeatable)]
    n_rep = n_same + n_opp
    if len(cand) < n_rep:
        raise ValueError(f"{n_rep} repeats asked, {len(cand)} fit poses may be repeated")
    picks = []
    for x in np.linspace(0, len(cand) - 1, n_rep):          # spread, never two in neighbouring tour positions
        c = [p_ for p_ in cand if all(abs(p_ - q_) > 1 for q_ in picks)]
        if not c:
            raise ValueError("no room for the repeats along the tour")
        picks.append(min(c, key=lambda p_: abs(cand.index(p_) - x)))
    picks.sort()
    kinds = ["repeat_same" if (i % 2 == 0 and i // 2 < n_same) else "repeat_opp" for i in range(n_rep)]
    while kinds.count("repeat_same") < n_same:
        kinds[len(kinds) - 1 - kinds[::-1].index("repeat_opp")] = "repeat_same"
    rep = dict(zip(picks, kinds))
    q_at = lambda pos: start if pos < 0 or pos >= len(order) else Q[order[pos]]       # noqa: E731
    out = []
    for pos, k in enumerate(order):
        out.append({"kind": kind_at[pos], "pose": int(k), "q": Q[k].tolist()})
        if rep.get(pos) == "repeat_same":
            out.append({"kind": "pre", "pose": int(k), "q": np.asarray(q_at(pos - 1), float).tolist()})
            out.append({"kind": "repeat_same", "pose": int(k), "q": Q[k].tolist()})
        if rep.get(pos - 1) == "repeat_opp":
            j = order[pos - 1]
            out.append({"kind": "repeat_opp", "pose": int(j), "q": Q[j].tolist()})
            out.append({"kind": "pre", "pose": int(j), "q": Q[k].tolist()})
    out.append({"kind": "return", "q": np.asarray(start, float).tolist()})
    return out


def param_names(offsets=INNER, joints=RIGHT_ARM) -> list:
    return ([f"X r{a}" for a in "xyz"] + [f"X t{a}" for a in "xyz"] + [f"dT r{a}" for a in "xyz"]
            + [f"dT t{a}" for a in "xyz"] + [f"offset {joints[j]}" for j in offsets])


def list_gate(info, plate_sigma_mm, offsets=INNER) -> dict:
    """T2-9 on the summed information of the fit poses: every parameter's 1 sigma (rotations and offsets in deg,
    translations in mm) within its limit, and every pose's plate sigma within LIM_PLATE_MM."""
    names = param_names(offsets)
    ev = np.linalg.eigvalsh(info)
    if ev[0] <= ev[-1] * 1e-14:
        return {"pass": False, "fails": ["information singular"], "sigma": {}}
    sd = np.sqrt(np.diag(np.linalg.inv(info)))
    sigma, fails = {}, []
    for i, nm in enumerate(names):
        rot = nm.startswith("offset") or nm[-2] == "r"
        val, lim = (np.degrees(sd[i]), LIM_OFFSET_DEG if nm.startswith("offset") else LIM_DEG) if rot \
            else (sd[i] * 1e3, LIM_MM)
        sigma[nm] = float(val)
        if val > lim:
            fails.append(nm)
    ps = np.asarray(plate_sigma_mm, float)
    fails += [f"plate sigma, pose {i}" for i in np.flatnonzero(ps > LIM_PLATE_MM)]
    return {"pass": not fails, "fails": fails, "sigma": sigma, "plate_sigma_mm_max": float(ps.max())}


PLATE_MIN_TAGS = 3              # two tags of another grid with the same ids can sit side by side like the plate's
PLATE_FIT_FRAC = 0.08           # layout check: homography rms under this fraction of the tag's side in pixels


def detect_plate(grid, gray) -> dict:
    """The plate's corners {(tag, k): (u, v)} when another grid with the same tag ids is in view too (phantom12: the
    7x5 board's tags 0-8). AprilGrid.detect keeps one detection per id, so it can mix the two. Here every ArUco quad
    with a plate id is a candidate; the set of at least PLATE_MIN_TAGS tags, one per id, spanning two rows and two
    columns of the plate (a row of another grid's tags fits any one row), whose raw corners fit the plate's layout by
    one homography (rms < PLATE_FIT_FRAC of a tag side) wins, most tags first. Only its quads are refined, on the
    image as it is (AprilGrid.refine): nothing is painted, so a neighbour cannot pull a corner. {} when no set fits."""
    import itertools

    import cv2
    p = cv2.aruco.DetectorParameters()
    p.markerBorderBits = grid.border_bits
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), p)
    corners, ids, _ = det.detectMarkers(gray)
    if ids is None:
        return {}
    quads = [(int(i), c.reshape(4, 2).astype(float)) for c, i in zip(corners, ids.ravel())]
    cand = {t: [q for i, q in quads if i == t] for t in range(grid.n_tags)}
    best = None
    for pick in itertools.product(*[[None] + cand[t] for t in range(grid.n_tags)]):
        tags = [t for t in range(grid.n_tags) if pick[t] is not None]
        rc = [grid.tag_rc(t) for t in tags]
        if len(tags) < PLATE_MIN_TAGS or len({r for r, _ in rc}) < 2 or len({c for _, c in rc}) < 2:
            continue
        obj = np.concatenate([grid.obj_points(t)[:, :2] for t in tags])
        img = np.concatenate([pick[t] for t in tags])
        H, _ = cv2.findHomography(obj, img, 0)
        if H is None:
            continue
        pr = cv2.perspectiveTransform(obj.reshape(-1, 1, 2), H).reshape(-1, 2)
        side = np.mean([np.linalg.norm(np.roll(pick[t], -1, 0) - pick[t], axis=1).mean() for t in tags])
        rms = float(np.sqrt(np.mean(np.sum((pr - img) ** 2, axis=1))))
        if rms < PLATE_FIT_FRAC * side and (best is None or (len(tags), -rms) > (len(best[0]), -best[1])):
            best = (tags, rms, pick)
    if best is None:
        return {}
    tags, _, pick = best
    return {(t, k): uv for t in tags for k, uv in grid.refine(gray, pick[t]).items()}
