"""The arm session's camera-to-arm fit (docs/superpowers/plans/2026-10-22-phantom12-arm-session.md, Task 7A; rev 2's
models, rev 3's T2-2).

Every image of a hold closes the loop between the head cameras and the arm:

    pixel = project_c( (ΔT · T_TB(q_neck) · T_HC_c)^-1 · T_T,tip(q_arm + δ) · X · P )

with P a plate corner on the board, T_HC_c camera c in the Head link (T_BS · T_SC_c), X the plate on the gripper
link, ΔT a correction of the cameras in the torso and δ zero offsets of the inner arm joints (armplan.INNER).
Models: M0 = X; M1 = X, ΔT; M2 = X, ΔT, δ. Each is fitted to the pixels of the fit holds (robust loss); a held-out
hold is judged at the plate: the plate pose the cameras see (through the fitted ΔT) against the one the arm's
joints and the fitted X give, as the move of the plate's centre (mm). That is what a grasp would miss.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lensmodel  # noqa: E402
import rigmath as rm  # noqa: E402
from armplan import INNER  # noqa: E402

MODELS = {"M0": 6, "M1": 12, "M2": 12 + len(INNER)}
ORDER = ("M0", "M1", "M2")
CHOOSE_MM = 0.5                 # rev 2: the smallest model within this of the best held-out RMS wins
LOSS_PX = 1.0                   # soft-L1 scale: a corner off by more than ~1 px counts less
GAP_NS = 20_000_000             # an image whose nearest /joints sample is further than this is not used


@dataclass
class Rig:
    T_HC: list           # per camera: the camera in the Head link (T_BS · T_SC)
    lens: list           # per camera: (lensmodel name, intr)
    size: tuple


@dataclass
class Frame:
    hold: int
    cam: int
    t: int
    q_arm: np.ndarray    # rad, armchain.RIGHT_ARM order, as the encoders read them at the image's stamp
    q_neck: tuple        # (yaw, pitch) rad
    P: np.ndarray        # (n, 3) corners on the board
    uv: np.ndarray       # (n, 2) where they were found


def rig_from_config(cfg, size=(1280, 800)) -> Rig:
    model = {"pinhole-kb4t": "KB4T", "pinhole-equi": "KB4"}[cfg.lens]
    return Rig([cfg.T_BS @ T for T in cfg.T_SC],
               [(model, np.r_[K[0, 0], K[1, 1], K[0, 2], K[1, 2], np.asarray(D, float).ravel()])
                for K, D in zip(cfg.K, cfg.D)], tuple(size))


def holds_from_joints(t, q, min_s: float = 1.5, max_deg_s: float = 0.5, window_s: float = 0.02) -> list:
    """[(t0, t1)] ns: stretches of at least min_s in which no arm joint moves faster than max_deg_s (speed by a
    centred difference over window_s)."""
    t, q = np.asarray(t, np.int64), np.asarray(q, float)
    half = window_s / 2 * 1e9
    i = np.arange(len(t))
    lo = np.minimum(np.searchsorted(t, t - half), np.maximum(i - 1, 0))          # at least the neighbours
    hi = np.maximum(np.searchsorted(t, t + half, side="right") - 1, np.minimum(i + 1, len(t) - 1))
    dt = (t[hi] - t[lo]) / 1e9
    with np.errstate(invalid="ignore", divide="ignore"):
        speed = np.degrees(np.abs(q[hi] - q[lo]).max(axis=1)) / np.where(dt > 0, dt, np.nan)
    still = np.nan_to_num(speed, nan=np.inf) <= max_deg_s                        # unknown speed: not still
    out, i = [], 0
    while i < len(t):
        if not still[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(t) and still[j + 1]:
            j += 1
        if (t[j] - t[i]) / 1e9 >= min_s:
            out.append((int(t[i]), int(t[j])))
        i = j + 1
    return out


def frames_from_dump(d: dict, holds: list, names: list, arm_names, grid) -> list:
    """arm_corners.py's npz -> Frames of the images inside the holds (hold = its index in `holds`), each with the
    /joints sample nearest its stamp (none further than GAP_NS)."""
    ia = [names.index(n) for n in arm_names]
    iy, ip = names.index("NeckYaw"), names.index("NeckPitch")
    jt, jq = d["jt"], d["jq"]
    out = []
    for cam in (0, 1):
        m = d["cam"] == cam
        for t in np.unique(d["t"][m]):
            h = [k for k, (a, b) in enumerate(holds) if a <= t <= b]
            if not h:
                continue
            j = int(np.clip(np.searchsorted(jt, t), 1, len(jt) - 1))
            j = j if abs(jt[j] - t) < abs(jt[j - 1] - t) else j - 1
            if abs(int(jt[j]) - int(t)) > GAP_NS:
                continue
            k = m & (d["t"] == t)
            P = np.array([grid.obj_points(int(a))[int(b)] for a, b in zip(d["tag"][k], d["k"][k])])
            out.append(Frame(h[0], cam, int(t), jq[j][ia].astype(float), (float(jq[j][iy]), float(jq[j][ip])), P,
                             np.c_[d["u"][k], d["v"][k]]))
    return out


class _Stack:
    """The frames as flat arrays for a fast residual."""

    def __init__(self, obs, neck, rig):
        self.q = np.array([f.q_arm for f in obs])
        self.cam = np.array([f.cam for f in obs])
        self.T_TC = np.array([neck.T_TB(*f.q_neck) @ rig.T_HC[f.cam] for f in obs])
        self.fi = np.concatenate([np.full(len(f.P), i) for i, f in enumerate(obs)])
        self.P = np.concatenate([f.P for f in obs])
        self.uv = np.concatenate([f.uv for f in obs])
        self.hold = np.array([f.hold for f in obs])
        self.rig = rig

    def pixels(self, arm, X, dT, off) -> np.ndarray:
        q = self.q.copy()
        q[:, list(INNER)] += off
        T = _inv(dT[None] @ self.T_TC) @ arm.T_base_tip(q) @ X
        Pc = np.einsum("nij,nj->ni", T[self.fi, :3, :3], self.P) + T[self.fi, :3, 3]
        uv = np.empty((len(Pc), 2))
        for c, (model, intr) in enumerate(self.rig.lens):
            r = self.cam[self.fi] == c
            uv[r] = lensmodel.project(model, intr, Pc[r])
        return uv


def _inv(T):
    R = np.swapaxes(T[..., :3, :3], -1, -2)
    out = np.broadcast_to(np.eye(4), T.shape).copy()
    out[..., :3, :3] = R
    out[..., :3, 3] = -np.einsum("...ij,...j->...i", R, T[..., :3, 3])
    return out


def _unpack(p, model, X0):
    X = X0 @ rm.transform(rm.exp_rot(p[0:3]), p[3:6])
    dT = rm.transform(rm.exp_rot(p[6:9]), p[9:12]) if model != "M0" else np.eye(4)
    off = p[12:] if model == "M2" else np.zeros(len(INNER))
    return X, dT, off


def fit(obs, arm, neck, rig, model: str, X0, holds=None, fixed=None) -> dict:
    """Fit a model to the frames of `holds` (None: all). fixed: {'dT':..., 'offsets':...} held at given values
    (for the plate-slip check: X alone, the rest from the full fit). -> X, dT, offsets, rms_px, n_corners."""
    from scipy.optimize import least_squares
    sel = [f for f in obs if holds is None or f.hold in holds]
    S = _Stack(sel, neck, rig)
    n = MODELS[model]
    fx = fixed or {}

    def res(p):
        X, dT, off = _unpack(np.r_[p, np.zeros(MODELS["M2"] - n)], model, X0)
        dT = fx.get("dT", dT)
        off = fx.get("offsets", off)
        return (S.pixels(arm, X, dT, off) - S.uv).ravel()

    r = least_squares(res, np.zeros(n), loss="soft_l1", f_scale=LOSS_PX, x_scale=np.r_[[0.01] * 3, [0.005] * 3,
                      [0.01] * 3, [0.005] * 3, [0.01] * len(INNER)][:n], method="trf")
    X, dT, off = _unpack(np.r_[r.x, np.zeros(MODELS["M2"] - n)], model, X0)
    e = r.fun.reshape(-1, 2)
    return {"model": model, "X": X, "dT": fx.get("dT", dT), "offsets": fx.get("offsets", off),
            "rms_px": float(np.sqrt(np.mean(np.sum(e ** 2, axis=1)))), "n_corners": len(e),
            "within_1px": float(np.mean(np.hypot(e[:, 0], e[:, 1]) <= 1.0)), "x": r.x.tolist()}


def plate_offsets(obs, arm, neck, rig, params: dict, holds=None) -> dict:
    """{hold: (3,) mm}: per hold, the rigid correction E (plate frame) that makes T_T,tip(q_f + δ) · X · E fit the
    hold's pixels in both cameras through ΔT; reported as where E moves the plate's centre, in the torso frame at the
    hold's mean joints. Zero when the model is right; what a grasp would miss when it is not."""
    from scipy.optimize import least_squares
    out = {}
    hs = sorted({f.hold for f in obs}) if holds is None else holds
    X, dT, off = params["X"], params["dT"], params["offsets"]
    for h in hs:
        sel = [f for f in obs if f.hold == h]
        if not sel:
            continue
        S = _Stack(sel, neck, rig)
        c = np.mean(sel[0].P, axis=0) if len(sel[0].P) == 16 else np.array([0.0495, 0.0495, 0.0])
        r = least_squares(lambda e: (S.pixels(arm, X @ rm.transform(rm.exp_rot(e[:3]), e[3:]), dT, off)
                                     - S.uv).ravel(), np.zeros(6), loss="soft_l1", f_scale=LOSS_PX)
        E = rm.transform(rm.exp_rot(r.x[:3]), r.x[3:])
        q = S.q.mean(axis=0).copy()
        q[list(INNER)] += off
        T = arm.T_base_tip(q) @ X
        out[h] = (T[:3, :3] @ (E[:3, :3] @ c + E[:3, 3] - c)) * 1e3
    return out


def plate_errors(obs, arm, neck, rig, params: dict, holds=None) -> dict:
    """{hold: mm}: the length of plate_offsets."""
    return {h: float(np.linalg.norm(v)) for h, v in plate_offsets(obs, arm, neck, rig, params, holds).items()}


def choose(heldout_rms_mm: dict) -> str:
    """The smallest model whose held-out RMS is within CHOOSE_MM of the best."""
    best = min(heldout_rms_mm.values())
    return next(m for m in ORDER if m in heldout_rms_mm and heldout_rms_mm[m] <= best + CHOOSE_MM)


def _triangulate(r0, r1, T01):
    """Midpoints of the closest approach of rays r0 (camera 0) and r1 (camera 1, T01 = camera 1 in camera 0)."""
    o1 = T01[:3, 3]
    d1 = (T01[:3, :3] @ r1.T).T
    w = -o1
    a, b, c = np.sum(r0 * r0, 1), np.sum(r0 * d1, 1), np.sum(d1 * d1, 1)
    d, e = np.sum(r0 * w, 1), np.sum(d1 * w, 1)
    den = a * c - b * b
    s, u = (b * e - c * d) / den, (a * e - b * d) / den
    return 0.5 * (r0 * s[:, None] + (o1 + d1 * u[:, None]))


def board_scale_pct(obs, rig, grid, holds=None) -> dict:
    """{hold: %}: per hold the plate's corners triangulated in stereo through the rig (pairs of a camera-0 and a
    camera-1 image with the same stamp, corners seen in both) and their pairwise distances against the board's:
    (scale - 1) * 100, the median over the hold's pairs. Positive = the plate measures too large = depth too far."""
    T01 = rm.inv(rig.T_HC[0]) @ rig.T_HC[1]
    by = {}
    for f in obs:
        if holds is None or f.hold in holds:
            by.setdefault((f.hold, f.t), {})[f.cam] = f
    out = {}
    for (h, _), pair in sorted(by.items()):
        if 0 not in pair or 1 not in pair:
            continue
        a, b = pair[0], pair[1]
        ka = {tuple(np.round(p, 6)): i for i, p in enumerate(a.P)}
        common = [(ka[tuple(np.round(p, 6))], j) for j, p in enumerate(b.P) if tuple(np.round(p, 6)) in ka]
        if len(common) < 4:
            continue
        ia, ib = np.array(common).T
        r0 = lensmodel.unproject(rig.lens[0][0], rig.lens[0][1], a.uv[ia])
        r1 = lensmodel.unproject(rig.lens[1][0], rig.lens[1][1], b.uv[ib])
        X = _triangulate(r0, r1, T01)
        Pb = a.P[ia]
        i, j = np.triu_indices(len(ia), 1)
        dm, dt = np.linalg.norm(X[i] - X[j], axis=1), np.linalg.norm(Pb[i] - Pb[j], axis=1)
        out.setdefault(h, []).append((dm.sum() / dt.sum() - 1.0) * 100.0)
    return {h: float(np.median(v)) for h, v in out.items()}


MEASURED = ("fit", "heldout", "repeat_same", "repeat_opp")
MATCH_DEG = 0.5                 # a logged stop and a hold in /joints are the same when every joint agrees this well


def match_holds(hold_q, stops) -> dict:
    """{hold index: logged stop} for the measured stops kept (step 91's 'stop' rows). Walks the log in order; every
    logged position (a correction's first landing, pre-poses, skipped stops too) takes the next hold of /joints whose
    mean joints agree within MATCH_DEG, so a pose visited twice is matched to the right visit."""
    hold_q = np.degrees(np.asarray(hold_q, float))
    out, cur = {}, 0
    for s in stops:
        targets = ([s["reached_first_deg"]] if s.get("corrections") and s.get("reached_first_deg") else []) \
            + [s["reached_deg"]]
        for k, q in enumerate(targets):
            hit = next((h for h in range(cur, len(hold_q)) if np.abs(hold_q[h] - np.asarray(q)).max() <= MATCH_DEG),
                       None)
            if hit is None:
                continue
            cur = hit + 1
            if k == len(targets) - 1 and s["kind"] in MEASURED and s["status"] == "ok":
                out[hit] = s
    return out


def verdict(d: dict) -> dict:
    """Decision 5 (rev 2): the session valid (noise floor <= 1 mm; plate slip <= 1 mm / 0.2 deg), then M0 within
    3 mm = the URDF suffices; else the chosen model within 3 mm = calibrated, within 5 mm = acceptable, beyond =
    the model misses something."""
    if d["slip_mm"] > 1.0 or d["slip_deg"] > 0.2:
        return {"verdict": "VOID (plate slipped)", "model": None}
    if d["noise_floor_mm"] > 1.0:
        return {"verdict": "INCONCLUSIVE", "model": None}
    h = d["heldout_rms_mm"]
    if h.get("M0", np.inf) <= 3.0:
        return {"verdict": "URDF suffices", "model": "M0"}
    m = choose(h)
    if h[m] <= 3.0:
        return {"verdict": f"calibrated ({m})", "model": m}
    if h[m] <= 5.0:
        return {"verdict": f"acceptable ({m})", "model": m}
    return {"verdict": f"model missing something ({m} {h[m]:.1f} mm held out)", "model": m}
