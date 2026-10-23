#!/usr/bin/env python3
"""Step 90 — the arm session's pose list (docs/superpowers/plans/2026-10-22-phantom12-arm-session.md, Task 2;
rev 3's decisions T2-1..T2-9).

From the robot's current joints (a /actuals snapshot), the live OKVIS config and a recording of the palm plate at
the current pose, it finds arm poses at which both head cameras see the plate well (T2-3) and the arm stays clear
of the body (T2-4), picks the 26 that tell the camera-to-arm fit the most under the coverage rules (T2-6), orders
them (T2-7), adds the held-out marks and the repeats (T2-8), computes each command with the gravity sag (T2-5),
and checks the list against T2-9 before anyone moves the arm. Writes poses.json and prints a report.

The plate's place on the hand (X0) comes from the palm recording through this same chain: it is only the
planning prior; the session's fit (Task 7) fits X anew.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import armchain  # noqa: E402
import armplan as ap  # noqa: E402
import rigmath as rm  # noqa: E402

KEEP_IN_DEG, CMD_IN_DEG = 10.0, 5.0        # T2-4: desired / commanded joints this far inside both models' ranges
CLEAR_M, BAND_CLEAR_M, BAND_DEG = 0.03, 0.015, 8.0
BAND_FRICTION_DEG, BAND_GRAVITY_SHARE, FLAT_ABORT_DEG = 5.0, 0.3, 5.0     # the per-pose landing band (Checks.band_deg)
BAND_JOINTS = (0, 1, 3)                     # shoulder pitch, shoulder roll, elbow: gravity commands up to ~10 deg
PATH_STEP_DEG, PATH_MM = 2.0, 10.0          # path samples: at most this many degrees and millimetres apart
SIGMA_PX = 0.5
N_POSES, N_FIT = 26, 20
N_SAME, N_OPP = 3, 4                        # T2-8: same-side repeats (the noise floor) and opposite-side (play)
KP = 100.0
TIP_BODY = "RightGripperYawLink"                    # the arm chain's tip in both files (armchain: arm.tip)
SEE_THROUGH = ("Head", "NeckYawLink", "RightHand", "RightGripperYawLink")  # the cameras' own head; the plate's hand
PELVIS_IMU_R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])   # MJCF site pelvis_imu: +90 deg z


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plate_prior(corners: dict, names: list, arm, neck, cfg, grid) -> tuple:
    """X0 = T_tip,plate: per image the plate's pose in its camera (PnP through the config's lens, step 81's
    pose_from_corners) carried into the gripper link with the joints nearest the image's stamp. -> (X0, report)."""
    import rigcfg
    s81 = importlib.import_module("81_tbs_handeye")
    lens = rigcfg.Lens(cfg.lens)
    ia = [names.index(n) for n in armchain.RIGHT_ARM]
    iy, ip = names.index("NeckYaw"), names.index("NeckPitch")
    Xs = {0: [], 1: []}
    for cam in (0, 1):
        m = corners["cam"] == cam
        for t in np.unique(corners["t"][m]):
            k = m & (corners["t"] == t)
            det = {(int(a), int(b)): (u, v) for a, b, u, v in
                   zip(corners["tag"][k], corners["k"][k], corners["u"][k], corners["v"][k])}
            T_CP = s81.pose_from_corners(det, grid, lens, cfg.K[cam], cfg.D[cam], (640.0, 400.0),
                                         max_radius_px=1e9, min_tags=grid.n_tags)
            if T_CP is None:
                continue
            q = corners["jq"][int(np.argmin(np.abs(corners["jt"] - t)))]
            T_TC = armchain.camera_in_torso(neck, cfg.T_BS, cfg.T_SC[cam], q[iy], q[ip])
            Xs[cam].append(rm.inv(arm.T_base_tip(q[ia])) @ T_TC @ T_CP)
    if not Xs[0] or not Xs[1]:
        raise SystemExit("plate prior: the palm recording gives no plate pose in one camera")
    per = [rm.mean_pose(Xs[c]) for c in (0, 1)]
    rep = {"frames": [len(Xs[0]), len(Xs[1])], "spread": [rm.spread(Xs[c]) for c in (0, 1)],
           "cameras_apart_mm": float(np.linalg.norm(per[0][:3, 3] - per[1][:3, 3]) * 1e3),
           "cameras_apart_deg": rm.rot_angle_deg(per[0][:3, :3].T @ per[1][:3, :3])}
    return rm.mean_pose(Xs[0] + Xs[1]), rep


def cache_key(a) -> str:
    return "|".join([sha(a.urdf), sha(a.mjcf), sha(a.config), sha(a.snapshot), sha(a.palm_corners), str(a.seed),
                     str(a.samples), str(a.surface_step), Path(__file__).read_text().split("def run(")[0][-4000:]])


def gravity_in_pelvis(snap: dict) -> np.ndarray:
    """Gravity's direction in the pelvis from the pelvis IMU's accelerometer (specific force = -gravity)."""
    a = snap["imu"]["Pelvis"]
    f = PELVIS_IMU_R @ np.array([a["accel_x"], a["accel_y"], a["accel_z"]])
    return -f / np.linalg.norm(f)


class Checks:
    """T2-4 / T2-4b / T2-5 for one pose and for the straight joint path between two (review of 23 Oct, S1-S5):
    joints inside the URDF's and the MJCF's ranges both; the gravity command inside them; the arm low and forward
    along world up; clearance to the contact geoms and to the mesh surfaces; the same within the sag band. Paths
    are sampled so no chain point moves more than PATH_MM between samples."""

    def __init__(self, arm, body, up, g_pelvis, kp=KP):
        self.arm, self.body, self.up, self.g, self.kp = arm, body, np.asarray(up, float), g_pelvis, kp
        blo, bhi = body.ranges()
        self.lower, self.upper = np.maximum(arm.lower, blo), np.minimum(arm.upper, bhi)
        self._path = {}

    def command(self, Q) -> np.ndarray:
        Q = np.atleast_2d(Q)
        return Q + self.body.gravity(Q, self.g) / self.kp

    def limits(self, Q, keep_deg=KEEP_IN_DEG) -> np.ndarray:
        Q = np.atleast_2d(Q)
        k = np.radians(keep_deg)
        return ((Q >= self.lower + k) & (Q <= self.upper - k)).all(axis=1)

    def band_deg(self, q) -> np.ndarray:
        """(7,) how far each joint may land from the desired pose: for BAND_JOINTS, BAND_FRICTION_DEG plus
        BAND_GRAVITY_SHARE of the joint's gravity command at q, at most BAND_DEG (22 Oct-23 Oct: the friction band is
        ~5 deg at kp 100; the gravity model is off by up to ~30 %); the other joints FLAT_ABORT_DEG (step 91 aborts
        there; they carry little gravity and are not swept in the clearance band)."""
        g = np.degrees(np.abs(self.body.gravity(np.asarray(q, float)[None], self.g)[0] / self.kp))
        out = np.full(7, FLAT_ABORT_DEG)
        j = list(BAND_JOINTS)
        out[j] = np.minimum(BAND_DEG, BAND_FRICTION_DEG + BAND_GRAVITY_SHARE * g[j])
        return out

    def band(self, q, need=BAND_CLEAR_M, scale: float = 1.0) -> float:
        """least clearance with the BAND_JOINTS moved by 0 or +- scale * their band_deg each (every corner, edge and
        face of the box)"""
        w = scale * self.band_deg(q)[list(BAND_JOINTS)]
        Q = []
        for d in itertools.product((-1.0, 0.0, 1.0), repeat=len(BAND_JOINTS)):
            if any(d):
                x = np.array(q, float).copy()
                x[list(BAND_JOINTS)] += np.radians(np.array(d) * w)
                Q.append(x)
        return float(self.body.clearance(np.array(Q)).min())

    def pose(self, q) -> list:
        q = np.asarray(q, float)
        why = []
        if not self.limits(q)[0]:
            why.append(f"a joint within {KEEP_IN_DEG:.0f} deg of its range")
        if not self.limits(self.command(q), CMD_IN_DEG)[0]:
            why.append(f"the command within {CMD_IN_DEG:.0f} deg of a range")
        if not ap.workspace_ok(self.arm, q[None], self.up)[0]:
            why.append("outside the workspace (T2-4b)")
        c = float(self.body.clearance(q[None])[0])
        if c < CLEAR_M:
            why.append(f"clearance {c * 100:.1f} cm")
        b = self.band(q)
        if b < BAND_CLEAR_M:
            why.append(f"clearance in the sag band {b * 100:.1f} cm")
        return why

    def samples(self, a, b) -> np.ndarray:
        n = max(2, int(np.ceil(np.degrees(ap.move_cost(a, b)) / PATH_STEP_DEG)) + 1)
        while True:
            Q = np.linspace(a, b, n)
            pts = np.stack([f[:, :3, 3] for f in self.arm.frames(Q)], 1)              # (n, 7, 3)
            tip = self.arm.T_base_tip(Q) @ np.array([0.0, 0.0, -0.25, 1.0])            # past the fingertips
            pts = np.concatenate([pts, tip[:, None, :3]], 1)
            if np.linalg.norm(np.diff(pts, axis=0), axis=2).max() <= PATH_MM / 1e3 or n > 4000:
                return Q
            n = 2 * n

    def path(self, a, b, band=True, known_a=False, known_b=False) -> list:
        """problems on the straight joint path a -> b. known_a / known_b: that end is a pose the arm actually stands
        at (measured), with no landing band: the band tapers linearly to nothing there."""
        key = (tuple(np.round(a, 9)), tuple(np.round(b, 9)), band, known_a, known_b)
        if key in self._path:
            return self._path[key]
        Q = self.samples(np.asarray(a, float), np.asarray(b, float))
        why = []
        ws = ap.workspace_ok(self.arm, Q, self.up)
        if not ws[1:-1].all():                     # the ends are judged as poses (the start is where the arm is)
            why.append(f"{int((~ws[1:-1]).sum())} path samples outside the workspace")
        c = self.body.clearance(Q)
        if c.min() < CLEAR_M:
            why.append(f"path clearance {c.min() * 100:.1f} cm")
        if band:
            t = np.linspace(0.0, 1.0, len(Q))
            w = (t if known_a else 1.0) * ((1.0 - t) if known_b else 1.0) * np.ones(len(Q))
            bmin = min(self.band(q, scale=x) for q, x in zip(Q[1:-1], w[1:-1])) if len(Q) > 2 else np.inf
            if bmin < BAND_CLEAR_M:
                why.append(f"path clearance in the sag band {bmin * 100:.1f} cm")
        self._path[key] = why
        return why


_W = {}


def _init_worker(urdf, mjcf, surface_step, baseline, up, g):
    import armsafety
    body = armsafety.Body(mjcf, surface_step=surface_step)
    body.set_baseline(baseline)
    _W["chk"] = Checks(armchain.ArmChain(urdf), body, up, g)


def _pose_ok(q) -> bool:
    return not _W["chk"].pose(q)


def _pose_why(q) -> list:
    return _W["chk"].pose(q)


def _path_why(leg) -> list:
    a, b, ka, kb = leg
    return _W["chk"].path(a, b, known_a=ka, known_b=kb)


def parse_freeze(items) -> dict:
    """['RightWristRoll=15.4', ...] -> {joint index: rad}: joints held at one angle for the whole sweep (a plate taped
    across the wrist is rigid only while wrist roll and gripper yaw stay put)."""
    out = {}
    for it in items or []:
        name, _, val = it.partition("=")
        if name not in armchain.RIGHT_ARM or not val:
            raise SystemExit(f"--freeze {it!r}: want <right-arm joint>=<deg>, joints {armchain.RIGHT_ARM}")
        out[armchain.RIGHT_ARM.index(name)] = float(np.radians(float(val)))
    return out


def observable_offsets(frozen: dict) -> tuple:
    """armplan.INNER without the frozen joints: a joint that never moves has its zero offset absorbed by X."""
    return tuple(j for j in ap.INNER if j not in frozen)


def sample(arm, n: int, rng, keep_in_deg=KEEP_IN_DEG, frozen=None, lower=None, upper=None) -> np.ndarray:
    lo = (arm.lower if lower is None else lower) + np.radians(keep_in_deg)
    hi = (arm.upper if upper is None else upper) - np.radians(keep_in_deg)
    Q = rng.uniform(lo, hi, (n, len(lo)))
    for j, v in (frozen or {}).items():
        Q[:, j] = v
    return Q


def in_sight(body, q, cfg, X, P) -> bool:
    """Both cameras see every plate corner past the robot's own meshes (MJCF frames: the Head body carries
    T_BS · T_SC, the gripper link carries X)."""
    T_H, T_G = body.frame(q, "Head"), body.frame(q, TIP_BODY)
    pts = (T_G @ X @ np.c_[P, np.ones(len(P))].T).T[:, :3]
    return all(body.sightline_clear(q, (T_H @ cfg.T_BS @ T_SC)[:3, 3], pts, see_through=SEE_THROUGH)
               for T_SC in cfg.T_SC)


def gravity_in_torso(g_pelvis, spine_yaw: float) -> np.ndarray:
    """The torso hangs from the pelvis through SpineYaw (URDF: zero rpy, axis z)."""
    c, s = np.cos(spine_yaw), np.sin(spine_yaw)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]]) @ np.asarray(g_pelvis, float)


def run(a) -> dict:
    import armsafety
    import neckfk
    import okviscfg
    import rigcfg
    snap = json.loads(Path(a.snapshot).read_text())
    names = snap.get("names", list(snap["q_deg"]))
    arm, neck = armchain.ArmChain(a.urdf), neckfk.NeckChain(a.urdf)
    cfg = okviscfg.load(a.config)
    grid = rigcfg.AprilGrid(a.grid_cols, a.grid_rows, a.tag_size, a.tag_spacing)
    P, _ = ap.plate_points(grid)
    qy, qp = np.radians(snap["q_deg"]["NeckYaw"]), np.radians(snap["q_deg"]["NeckPitch"])
    cams = ap.cameras_from_config(cfg, neck, qy, qp)
    corners = dict(np.load(a.palm_corners))
    if corners["jq"].shape[1] != len(names):
        raise SystemExit(f"palm corners: {corners['jq'].shape[1]} joints per sample, the snapshot names {len(names)}")
    X0, prior_rep = plate_prior(corners, names, arm, neck, cfg, grid)
    start = np.radians([snap["q_deg"][n] for n in armchain.RIGHT_ARM])
    body = armsafety.Body(a.mjcf, surface_step=a.surface_step)
    baseline = {n: np.radians(v) for n, v in snap["q_deg"].items()}
    body.set_baseline(baseline)
    g = gravity_in_pelvis(snap)
    up = -gravity_in_torso(g, np.radians(snap["q_deg"]["SpineYaw"]))
    chk = Checks(arm, body, up, g)
    rng = np.random.default_rng(a.seed)
    frozen = parse_freeze(a.freeze)
    offsets = observable_offsets(frozen)
    pool = None
    if a.workers > 1:                           # the slow checks (mesh surfaces, the sag band) in parallel
        import multiprocessing as mp
        pool = mp.get_context("fork").Pool(a.workers, initializer=_init_worker,
                                            initargs=(a.urdf, a.mjcf, a.surface_step, baseline, up, g))

    # the feasible set: cheap rules first, the mesh surfaces and the sag band last
    cache = Path(a.feasible_cache) if a.feasible_cache else None
    feas, counts = [], {k_: 0 for k_ in ("sampled", "in_view", "workspace", "contact_geoms", "plate_sigma",
                                         "commandable", "surfaces_and_band", "in_sight")}
    if cache and cache.exists():                # the same inputs and seed give the same set: reuse it
        z = np.load(cache, allow_pickle=False)
        if str(z["key"]) == cache_key(a):
            feas, counts = [z["Q"]], json.loads(str(z["counts"]))
    while counts["sampled"] < a.samples:
        Q = sample(arm, min(a.chunk, a.samples - counts["sampled"]), rng, frozen=frozen, lower=chk.lower,
                   upper=chk.upper)
        counts["sampled"] += len(Q)
        Q = Q[ap.visible(ap.view(cams, ap.plate_in_torso(arm, X0, Q), P, grid))]
        counts["in_view"] += len(Q)
        Q = Q[ap.workspace_ok(arm, Q, up)]
        counts["workspace"] += len(Q)
        if len(Q):
            Q = Q[body.contact_clearance(Q) >= CLEAR_M]
        counts["contact_geoms"] += len(Q)
        if len(Q):
            Q = Q[ap.plate_sigma_mm(cams, ap.plate_in_torso(arm, X0, Q), P, SIGMA_PX) <= ap.LIM_PLATE_MM]
        counts["plate_sigma"] += len(Q)
        if len(Q):
            Q = Q[chk.limits(chk.command(Q), CMD_IN_DEG)]
        counts["commandable"] += len(Q)
        if len(Q):
            ok = pool.map(_pose_ok, list(Q), chunksize=4) if pool else [not chk.pose(q) for q in Q]
            Q = Q[np.array(ok, bool)]
        counts["surfaces_and_band"] += len(Q)
        Q = Q[[in_sight(body, q, cfg, X0, P) for q in Q]] if len(Q) else Q
        counts["in_sight"] += len(Q)
        feas.append(Q)
    Q = np.concatenate(feas)
    if cache and not cache.exists():
        np.savez(cache, Q=Q, counts=json.dumps(counts), key=cache_key(a))
    if len(Q) < N_POSES:
        raise SystemExit(f"only {len(Q)} feasible poses of {counts['sampled']} sampled: {counts}")
    v = ap.view(cams, ap.plate_in_torso(arm, X0, Q), P, grid)
    cell = ap.image_cell(v["uv"][:, 0].mean(axis=1))
    lo, hi = ap.RANGE_M[0], float(v["range_m"][:, 0].max())      # thirds of the range the feasible set reaches
    third = np.clip(((v["range_m"][:, 0] - lo) / max((hi - lo) / 3, 1e-6)).astype(int), 0, 2)
    info = ap.information(arm, cams, X0, Q, P, SIGMA_PX, offsets=offsets)
    prior = np.diag(np.r_[[1 / 0.1 ** 2] * 3, [1 / 0.05 ** 2] * 3, [1 / 0.1 ** 2] * 3, [1 / 0.05 ** 2] * 3,
                          [1 / 0.1 ** 2] * len(offsets)])
    rules = dict(min_per_cell=2, min_bottom=4, min_per_third=5)
    sel = ap.select(info, cell, third, Q, n_pick=N_POSES, prior=prior, **rules)
    Qs = Q[sel]

    # order on the cheap path check, then verify every stop and path in full; a failing tour edge is forbidden
    # and the tour redone, a failing repeat is drawn from another pose
    def lookups(st):
        """every stop's pose problems and every move's path problems, in parallel when a pool is up"""
        qs = [np.asarray(x["q"], float) for x in st]
        legs = list(zip([start] + qs[:-1], qs))
        at_start = lambda q: bool(np.allclose(q, start))                            # noqa: E731
        legs4 = [(x, y, at_start(x), at_start(y)) for x, y in legs]                 # the start pose is measured
        if pool:
            pp = pool.map(_pose_why, qs, chunksize=1)
            lp = pool.map(_path_why, legs4, chunksize=1)
        else:
            pp, lp = [chk.pose(q) for q in qs], [chk.path(*leg[:2], known_a=leg[2], known_b=leg[3]) for leg in legs4]
        key = lambda *v: tuple(np.round(np.concatenate(v), 9))                     # noqa: E731
        pd = {key(q): w for q, w in zip(qs, pp)}
        ld = {key(x, y): w for (x, y), w in zip(legs, lp)}
        return (lambda q: pd[key(np.asarray(q, float))]), (lambda x, y: ld[key(np.asarray(x, float),
                                                                                 np.asarray(y, float))])

    forbid, repeatable, problems, n_opp = set(), set(range(len(Qs))), [], N_OPP
    for it in range(4 * len(Qs)):
        order = ap.tour(Qs, start, allowed=lambda i, j: (i, j) not in forbid and not chk.path(
            start if i < 0 else Qs[i], start if j < 0 else Qs[j], band=False))
        st = None
        for n_opp in range(N_OPP, -1, -1):                   # opposite-side repeats (play) give way first
            try:
                st = ap.stops(Qs, order, start, n_same=N_SAME, n_opp=n_opp, repeatable=repeatable, pre="inline")
                break
            except ValueError:
                continue
        if st is None:
            problems = [f"no room for {N_SAME} same-side repeats among the poses that can be repeated safely"]
            break
        problems, drop = ap.verify(st, start, *lookups(st))
        print(f"verify {it}: {len(problems)} problems, {len(repeatable)} repeatable, {n_opp} opposite repeats",
              *("  " + w for w in problems), sep="\n", file=sys.stderr, flush=True)
        if not problems:
            break
        route = [-1] + order + [-1]
        new = set()
        for i, j in zip(route, route[1:]):
            if (i, j) not in forbid and chk.path(start if i < 0 else Qs[i], start if j < 0 else Qs[j],
                                                 known_a=i < 0, known_b=j < 0):
                new |= {(i, j), (j, i)}
        if not drop and not new:
            break                                             # nothing left to change: the gate reports it
        forbid |= new
        repeatable -= drop
    if pool:
        pool.close()
    fit_idx = [s["pose"] for s in st if s["kind"] == "fit"]
    sig = ap.plate_sigma_mm(cams, ap.plate_in_torso(arm, X0, Qs), P, SIGMA_PX)
    gate = ap.list_gate(info[np.array(sel)[fit_idx]].sum(0), sig, offsets=offsets)
    gate["unsafe"] = problems
    gate["coverage"] = ap.coverage(cell, third, sel, **rules)
    gate["pass"] = gate["pass"] and not problems and not gate["coverage"]
    vs = ap.view(cams, ap.plate_in_torso(arm, X0, Qs), P, grid)
    R0 = (np.linalg.inv(cams[0].T_TC) @ ap.plate_in_torso(arm, X0, Qs))[:, :3, :3]
    ang = [rm.rot_angle_deg(R0[i].T @ R0[j]) for i in range(len(Qs)) for j in range(i + 1, len(Qs))]
    prev, total = start, 0.0
    for s in st:
        q = np.array(s["q"])
        s["ramp_s"] = round(ap.ramp_s(prev, q), 1)
        s["q_deg"] = np.round(np.degrees(q), 3).tolist()
        s["command_deg"] = np.round(np.degrees(chk.command(q)[0]), 3).tolist()
        s["abort_deg"] = np.round(chk.band_deg(q), 2).tolist()      # the landing band the checks covered here
        s["clearance_m"] = round(float(body.clearance(q[None])[0]), 3)
        if "pose" in s and s["kind"] != "pre":
            kk = s["pose"]
            s["view"] = {"cell": int(cell[sel[kk]]), "range_m": np.round(vs["range_m"][kk], 3).tolist(),
                         "facing_deg": np.round(vs["facing_deg"][kk], 1).tolist(),
                         "tag_px": np.round(vs["tag_px"][kk], 1).tolist(),
                         "margin_px": np.round(vs["margin_px"][kk], 1).tolist(),
                         "plate_sigma_mm": round(float(sig[kk]), 3)}
        total += s["ramp_s"] + a.hold_s + a.overhead_s
        prev = q
        del s["q"]
    return {
        "step": "90_arm_poses", "joints": list(armchain.RIGHT_ARM), "names": names, "kp": KP, "hold_s": a.hold_s,
        "frozen_deg": {armchain.RIGHT_ARM[j]: float(np.degrees(v)) for j, v in frozen.items()},
        "offsets": list(offsets),
        "inputs": {"urdf": sha(a.urdf)[:12], "mjcf": sha(a.mjcf)[:12], "config": sha(a.config)[:12],
                   "snapshot": sha(a.snapshot)[:12], "palm_corners": sha(a.palm_corners)[:12], "seed": a.seed,
                   "mujoco": __import__("mujoco").__version__},
        "neck_deg": {"NeckYaw": snap["q_deg"]["NeckYaw"], "NeckPitch": snap["q_deg"]["NeckPitch"]},
        "start_deg": np.round(np.degrees(start), 3).tolist(),
        "limits_deg": [np.degrees(chk.lower).round(2).tolist(), np.degrees(chk.upper).round(2).tolist()],
        "gravity_in_pelvis": np.round(g, 4).tolist(), "up_in_torso": np.round(up, 4).tolist(),
        "baseline_deg": snap["q_deg"],
        "grid": {"cols": a.grid_cols, "rows": a.grid_rows, "tag_size": a.tag_size, "tag_spacing": a.tag_spacing},
        "cameras": [{"T_TC": c.T_TC.tolist(), "model": c.model, "intr": c.intr.tolist(), "size": list(c.size)}
                    for c in cams],
        "X0": X0.tolist(), "plate_prior": prior_rep,
        "feasible": counts, "range_thirds_m": [lo, hi], "cells": {int(c): int((cell == c).sum()) for c in range(9)},
        "selected_cells": {int(c): int((cell[sel] == c).sum()) for c in range(9)},
        "selected_thirds": {int(t): int((third[sel] == t).sum()) for t in range(3)},
        "orientation_pairwise_deg": {"min": float(min(ang)), "median": float(np.median(ang))},
        "repeats": {"same": N_SAME, "opposite": n_opp, "pre": "inline: the tour's own moves"},
        "gate": gate, "motion_s": round(total, 1), "stops": st,
    }


def report(r: dict) -> str:
    g = r["gate"]
    lines = [f"feasible: {r['feasible']}", f"cells reached (feasible): {r['cells']}",
             f"selected per cell: {r['selected_cells']}, per range third: {r['selected_thirds']}",
             f"plate prior: frames {r['plate_prior']['frames']}, cameras apart "
             f"{r['plate_prior']['cameras_apart_mm']:.2f} mm / {r['plate_prior']['cameras_apart_deg']:.2f} deg",
             f"plate orientations pairwise: min {r['orientation_pairwise_deg']['min']:.1f} deg, "
             f"median {r['orientation_pairwise_deg']['median']:.1f} deg",
             "predicted 1 sigma (20 fit poses, 0.5 px): " + ", ".join(f"{k} {v:.3f}" for k, v in g["sigma"].items()),
             f"plate sigma max {g.get('plate_sigma_mm_max', float('nan')):.3f} mm",
             f"stops {len(r['stops'])}, motion {r['motion_s']:.0f} s (computed: ramps + holds + overhead)",
             f"GATE T2-9: {'PASS' if g['pass'] else 'FAIL'} {g['fails'] + g['unsafe'] + g['coverage']}"]
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--urdf", required=True)
    p.add_argument("--mjcf", required=True)
    p.add_argument("--config", required=True, help="the live OKVIS config (cameras, T_BS)")
    p.add_argument("--snapshot", required=True, help="/actuals snapshot JSON: q_deg by joint name, imu.Pelvis")
    p.add_argument("--palm-corners", required=True, help="npz: the plate's corners at the current pose + /joints")
    p.add_argument("--grid-cols", type=int, default=2)
    p.add_argument("--grid-rows", type=int, default=2)
    p.add_argument("--tag-size", type=float, default=0.045)
    p.add_argument("--tag-spacing", type=float, default=0.2)
    p.add_argument("--samples", type=int, default=400000)
    p.add_argument("--chunk", type=int, default=50000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hold-s", type=float, default=3.0)
    p.add_argument("--overhead-s", type=float, default=1.0, help="per move, on top of ramp and hold (22 Oct)")
    p.add_argument("--surface-step", type=float, default=0.008, help="mesh-surface sampling for the clearance, m")
    p.add_argument("--workers", type=int, default=1, help="processes for the per-pose and path checks")
    p.add_argument("--freeze", action="append", default=[], help="<right-arm joint>=<deg>, held for the whole sweep")
    p.add_argument("--feasible-cache", default=None, help="npz to keep the feasible set in (reused if inputs match)")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    r = run(a)
    print(report(r))
    if not r["gate"]["pass"]:                   # a failed list carries no stops: nothing can run it (review S4)
        r["stops"] = []
        Path(a.out).unlink(missing_ok=True)     # nor an older list left at the same path (review 2, S6)
        Path(a.out).with_suffix(".REJECTED.json").write_text(json.dumps(r, indent=1))
        return 1
    Path(a.out).write_text(json.dumps(r, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
