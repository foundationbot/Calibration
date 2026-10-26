"""Route 2 (plan 2026-10-25-tagfree-selfcal.md, entry 25 Oct 23:25): the camera's rotation in the torso from the
robot's own torso CAD seen in the head depth, compared with the plate's B1. Rules fixed in the plan before data.

Per capture: M0 (the mesh at E = I: median |point-to-plane| of the silhouette's depth points paired within 30 mm),
then meshfit's B0 (fit_pose) with the torso as the only link, converted to the plate's convention
dT = T_TC . E^-1 . T_CT (T_Torso_cam(true) = dT . T_Torso_cam(chain)); per session the median over captures that are
`ok` with coverage >= 0.70; on 3 of them the cost profile about each torso axis. Pixels in the right arm's chain
silhouette (dilated 10 px) are removed first.

Run on Moonstone (cores 0-7) from the point-cloud code copy (meshfit, _burstfusion: feat/pointcloud-agent-burst-fusion):
  PYTHONPATH=<copy>/build:<copy>/repo/tools/burst_pointcloud venv/bin/python torso_fit.py --mesh-dir <mjcf/meshes> OUT"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import armfk
import frames
import hand_check
import mesh
import meshfit

TORSO_OBJS = ("MK1_1_Torso_Shell.obj", "BWP_0095_Torso_Back.obj", "BWP_0096_Torso_Front.obj")
B1_ROTVEC_DEG = np.array([0.188, 0.001, -0.015])     # the plate's dT (results/phantom12-2026-10-25-dt)
SESSIONS = ("sc1025d-right", "sc1025i-right23", "sc1025j-right18")
M0_GATE, M0_MAX_MM = 0.030, 10.0
ARM_DILATE_PX = 10
MIN_COVERAGE, MIN_CAPTURES = 0.70, 3
TOL_DEG = 0.1
PROFILE_DEG = tuple(round(0.05 * k, 2) for k in range(-20, 21) if k)   # +-1 deg in 0.05 deg steps (the plan)
FLAT_RISE = 0.01                  # flat: the cost at -0.2 or at +0.2 deg rises < 1 % (either side: the stricter reading)


class BodyModel(meshfit.ArmModel):
    """meshfit's model for links that are not an arm's (the torso): no side, an identity 'wrist' at the Torso origin"""

    def __init__(self, urdf, link_meshes: dict, spacing: float = 0.001):
        self.links = list(link_meshes)
        self.urdf = urdf
        self.meshes = {n: (np.asarray(V, float), np.asarray(F, np.int64)) for n, (V, F) in link_meshes.items()}
        self.chains = {n: armfk.Chain(urdf, "Torso", n) for n in self.links}
        self.side = None
        self.wrist = armfk.Chain(urdf, "Torso", "Torso")
        P, N, lid = [], [], []
        for k, n in enumerate(self.links):
            p, nn, _ = mesh.sample(*link_meshes[n], spacing)
            P.append(p)
            N.append(nn)
            lid.append(np.full(len(p), k))
        self.P, self.N, self.link_id = np.vstack(P), np.vstack(N), np.concatenate(lid)


def load_torso(mesh_dir, spacing: float = 0.001) -> BodyModel:
    """the three torso OBJs (vertices in the MJCF Torso body frame = the #816 URDF Torso frame) as one link 'Torso'"""
    Vs, Fs, n = [], [], 0
    for name in TORSO_OBJS:
        V, F = mesh.load_obj(Path(mesh_dir) / name)
        Vs.append(V)
        Fs.append(F + n)
        n += len(V)
    return BodyModel(frames.URDF, {"Torso": (np.vstack(Vs), np.vstack(Fs))}, spacing)


def merged_mesh(body: BodyModel):
    return body.meshes["Torso"]


def dT_from_E(T_TC, E):
    """the plate's convention from B0's camera-frame E: E . T_CT(chain) = T_CT(true) and T_TC(true) = dT . T_TC(chain)"""
    T_TC = np.asarray(T_TC, float)
    return T_TC @ np.linalg.inv(np.asarray(E, float)) @ np.linalg.inv(T_TC)


def rotvec(R):
    return Rotation.from_matrix(np.asarray(R, float)).as_rotvec()


def arm_mask(arm, q, T_CT, K, W, H):
    import cv2
    P, _, _ = arm.place(q, T_CT)
    sil = (mesh.render(P, np.eye(4), K, W, H)[0] > 0).astype(np.uint8)
    return cv2.dilate(sil, np.ones((2 * ARM_DILATE_PX + 1,) * 2, np.uint8)) > 0


def drop_masked(data, mask, K, W, H):
    u, v, ok = meshfit._pixels(data, K, W, H)
    keep = ok.copy()
    keep[ok] = ~mask[v[ok], u[ok]]
    return data[keep]


def m0(body, T_CT, data, K, W, H):
    """the mesh at E = I: model points visible, data in its silhouette, median |point-to-plane| paired within 30 mm"""
    P, N, _ = body.place({}, T_CT)
    vis = np.nonzero(meshfit.visible(P, N, K, W, H))[0]
    if len(vis) == 0:
        return {"visible_model_points": 0, "silhouette_points": 0, "paired": 0, "median_abs_mm": None}
    sel = data[meshfit._select(data, P[vis], K, W, H, meshfit.DILATE_PX)]
    pr, _ = meshfit._pairs(sel, P[vis], N[vis], vis, M0_GATE)
    return {"visible_model_points": int(len(vis)), "silhouette_points": int(len(sel)), "paired": int(len(pr.r)),
            "median_abs_mm": float(1e3 * np.median(np.abs(pr.r))) if len(pr.r) else None}


def cost_profile(body, T_TC, E, data, K, W, H):
    """the robust cost (meshfit's last gate and Huber; unpaired at the gate) with the camera rotated about each TORSO
    axis through its own centre, relative to the cost at the fit; data selection and visible points held at the fit"""
    gate, huber = meshfit.GATES[-1], meshfit.HUBER
    T_TC = np.asarray(T_TC, float)
    T_true = T_TC @ np.linalg.inv(E)                  # the fitted camera in the torso
    P0, N0, _ = body.place({}, np.eye(4))             # torso frame
    Pc, Nc, _ = body.place({}, np.linalg.inv(T_true))
    vis = np.nonzero(meshfit.visible(Pc, Nc, K, W, H))[0]
    sel = data[meshfit._select(data, Pc[vis], K, W, H, meshfit.DILATE_PX)]

    def cost(T_cam):
        Ti = np.linalg.inv(T_cam)
        P, N = P0[vis] @ Ti[:3, :3].T + Ti[:3, 3], N0[vis] @ Ti[:3, :3].T
        pr, _ = meshfit._pairs(sel, P, N, vis, gate)
        a = np.abs(pr.r)
        tot = float(np.sum(np.where(a <= huber, 0.5 * a ** 2, huber * (a - 0.5 * huber))))
        tot += (len(sel) - len(a)) * huber * (gate - 0.5 * huber)
        return tot / max(len(sel), 1)

    c0 = max(cost(T_true), 1e-30)
    out = {"cost_at_fit": c0, "points": int(len(sel)), "axes": {}, "flat": []}
    for k, name in enumerate(("rx", "ry", "rz")):
        row = []
        for d in PROFILE_DEG:
            T = T_true.copy()
            T[:3, :3] = Rotation.from_rotvec(np.eye(3)[k] * np.radians(d)).as_matrix() @ T_true[:3, :3]
            row.append((d, cost(T) / c0))
        out["axes"][name] = row
        rise = min(v for d, v in row if abs(abs(d) - 0.2) < 1e-9) - 1.0
        if rise < FLAT_RISE:
            out["flat"].append(name)
    return out


def one(path, body, arm, bf):
    try:
        return _one(path, body, arm, bf)
    except Exception as e:                            # one bad capture must not lose the session: it counts as not ok
        return {"name": Path(path).stem, "m0": {"visible_model_points": 0, "silhouette_points": 0, "paired": 0,
                                                "median_abs_mm": None},
                "ok": False, "coverage": 0.0, "why": f"error: {type(e).__name__}: {e}"}, None


def _one(path, body, arm, bf):
    t0 = time.time()
    p = hand_check.load_pose(path, bf, None, "Right")
    K, (H, W) = p["K"], p["depth"].shape
    T_TC = np.asarray(p["T_TC"], float)
    T_CT = np.linalg.inv(T_TC)
    data = drop_masked(meshfit.unproject(p["depth"], K), arm_mask(arm, p["q"], T_CT, K, W, H), K, W, H)
    row = {"name": p["name"], "neck_pitch_deg": float(np.degrees(p["angles"]["NeckPitch"])),
           "neck_yaw_deg": float(np.degrees(p["angles"]["NeckYaw"])), "m0": m0(body, T_CT, data, K, W, H)}
    if row["m0"]["visible_model_points"] < meshfit.VIEW_MIN_POINTS:     # the torso is not in view: nothing to fit
        row.update({"ok": False, "converged": False, "cond": float("nan"), "coverage": 0.0, "rms_mm": float("nan"),
                    "inliers": 0, "why": "torso not in view", "seconds": round(time.time() - t0, 1)})
        return row, None
    f = meshfit.fit_pose(body, {}, T_TC, data, K, W, H)
    dT = dT_from_E(T_TC, f.delta_c)
    row.update({"ok": bool(f.ok), "converged": bool(f.converged), "cond": float(f.cond), "coverage": float(f.coverage),
                "rms_mm": float(f.rms_mm),
                "inliers": int(f.inlier.sum()), "E": f.delta_c.tolist(), "dT": dT.tolist(),
                "dT_rotvec_deg": np.degrees(rotvec(dT[:3, :3])).tolist(), "dT_t_mm": (1e3 * dT[:3, 3]).tolist(),
                "seconds": round(time.time() - t0, 1)})
    return row, (T_TC, f.delta_c, data, K, W, H)


_CTX = {}                                             # set in main() before the fork


def verdict(sessions: dict) -> dict:
    """the plan's rule over the sessions' summaries: a session PASSES if its M0 passed, it has >= MIN_CAPTURES counted
    captures, none of its profiles is flat and its median rotation is within TOL_DEG of B1; route 2 is usable if >= 2
    sessions pass and every two sessions with a median rotation agree within TOL_DEG"""
    per, meds = {}, {}
    for s, x in sessions.items():
        if not x.get("m0_pass"):
            per[s] = "M0 FAIL"
        elif "median_rotvec_deg" not in x:
            per[s] = "TOO FEW"
        elif any(pf["flat"] for pf in x.get("profiles", {}).values()):
            per[s] = "FLAT " + ",".join(sorted({a for pf in x["profiles"].values() for a in pf["flat"]}))
        else:
            per[s] = "PASS" if x["deg_from_b1"] <= TOL_DEG else f"FAIL ({x['deg_from_b1']:.3f} deg from B1)"
        if per[s] != "M0 FAIL" and "median_rotvec_deg" in x:
            meds[s] = np.asarray(x["median_rotvec_deg"], float)
    names = sorted(meds)
    spread = max((float(np.linalg.norm(meds[a] - meds[b])) for i, a in enumerate(names) for b in names[i + 1:]),
                 default=None)
    passed = [s for s, v in per.items() if v == "PASS"]
    usable = len(passed) >= 2 and spread is not None and spread <= TOL_DEG
    return {"per_session": per, "sessions_spread_deg": spread, "passed": passed,
            "route2": "USABLE" if usable else "NOT USABLE"}


def _job(path):
    try:
        import cv2
        cv2.setNumThreads(1)
    except ImportError:
        pass
    if not _CTX["m0_only"]:
        return one(path, _CTX["body"], _CTX["arm"], _CTX["bf"])
    p = hand_check.load_pose(path, _CTX["bf"], None, "Right")
    K, (H, W) = p["K"], p["depth"].shape
    T_CT = np.linalg.inv(np.asarray(p["T_TC"], float))
    data = drop_masked(meshfit.unproject(p["depth"], K), arm_mask(_CTX["arm"], p["q"], T_CT, K, W, H), K, W, H)
    return {"name": p["name"], "m0": m0(_CTX["body"], T_CT, data, K, W, H)}, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--mesh-dir", required=True)
    ap.add_argument("--bursts", default=str(Path.home() / "burst-pc/arm"))
    ap.add_argument("--sessions", default=",".join(SESSIONS))
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--m0-only", action="store_true", help="stop after M0 (no fit)")
    cli = ap.parse_args()
    import _burstfusion as bf
    _CTX["bf"], _CTX["m0_only"] = bf, cli.m0_only
    body = load_torso(cli.mesh_dir, 0.001)
    arm = meshfit.ArmModel.load(frames.URDF, Path.home() / "burst-pc/meshes/cycloidal_arm",
                                hand_check.sided(hand_check.LINKS, "Right"), 0.001)
    _CTX["body"], _CTX["arm"] = body, arm
    V = merged_mesh(body)[0]
    out = {"started": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "mesh_extent_mm": (1e3 * np.ptp(V, axis=0)).tolist(),
           "b1_rotvec_deg": B1_ROTVEC_DEG.tolist(), "do_nothing_deg_from_b1": float(np.linalg.norm(B1_ROTVEC_DEG)),
           "sessions": {}}
    print("torso mesh extent mm", np.round(out["mesh_extent_mm"], 1).tolist(), flush=True)
    for s in cli.sessions.split(","):
        paths = sorted(Path(cli.bursts, s).glob(f"{s}-*.npz"))
        import multiprocessing as mp
        with mp.get_context("fork").Pool(cli.jobs) as pool:
            res = pool.map(_job, paths)
        rows = [r for r, _ in res]
        for r in rows:
            m = r["m0"]
            line = f"{s} {r['name']} M0 vis {m['visible_model_points']} sil {m['silhouette_points']} " \
                   f"paired {m['paired']} median {m['median_abs_mm'] if m['median_abs_mm'] is None else round(m['median_abs_mm'], 2)} mm"
            if "dT_rotvec_deg" in r:
                line += f" | B0 ok {r['ok']} cov {r['coverage']:.2f} rms {r['rms_mm']:.2f} inl {r['inliers']} " \
                        f"cond {r['cond']:.2e} rot {np.round(r['dT_rotvec_deg'], 3).tolist()} t {np.round(r['dT_t_mm'], 2).tolist()}"
            print(line, flush=True)
        med = [r["m0"]["median_abs_mm"] for r in rows if r["m0"]["median_abs_mm"] is not None]
        sess = {"captures": rows, "m0_median_of_captures_mm": float(np.median(med)) if med else None,
                "m0_captures_with_torso": len(med), "m0_captures": len(rows)}
        sess["m0_pass"] = bool(med) and bool(sess["m0_median_of_captures_mm"] <= M0_MAX_MM)   # verdict(): M0 FAIL ignores the fit
        if not cli.m0_only:
            good = [k for k, r in enumerate(rows) if r["ok"] and r["coverage"] >= MIN_COVERAGE]
            sess["counted"] = [rows[k]["name"] for k in good]
            if len(good) >= MIN_CAPTURES:
                rv = np.median([rows[k]["dT_rotvec_deg"] for k in good], axis=0)
                sess.update({"median_rotvec_deg": rv.tolist(),
                             "median_t_mm": np.median([rows[k]["dT_t_mm"] for k in good], axis=0).tolist(),
                             "spread_rotvec_deg": np.std([rows[k]["dT_rotvec_deg"] for k in good], axis=0).tolist(),
                             "deg_from_b1": float(np.linalg.norm(rv - B1_ROTVEC_DEG))})
                pick = [good[0], good[len(good) // 2], good[-1]]
                sess["profiles"] = {rows[k]["name"]: cost_profile(body, *res[k][1][:2], *res[k][1][2:]) for k in pick}
                for n, pf in sess["profiles"].items():
                    print(s, "profile", n, {a: [(d, round(v, 4)) for d, v in row if abs(d) in (0.2, 1.0)]
                                            for a, row in pf["axes"].items()}, "flat", pf["flat"], flush=True)
            print(s, "counted", len(good), "median rot", np.round(sess.get("median_rotvec_deg", []), 3).tolist(),
                  "deg from B1", sess.get("deg_from_b1"), flush=True)
        out["sessions"][s] = sess
        Path(cli.out).write_text(json.dumps(out, indent=1))
    out["verdict"] = verdict(out["sessions"])
    print("VERDICT", json.dumps(out["verdict"]), flush=True)
    out["finished"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    Path(cli.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
