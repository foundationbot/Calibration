"""H-1: offline go/no-go for the hand-eye URDF route on phantom14's stored wave 2 (4 Nov 2026).

Uses the stored per-capture read-B (B0) results under the base URDF (urdfx/base_w2.json right, base_L.json left) and
the stage-2 prediction of urdfx/predict.py (unchanged). No matching is re-run.

Checks:
  1. full fit per model: parameters +- standard error, condition number, correlations, per-capture residuals;
  2. leave-one-arm-pose-out cross-validation (every capture of one arm pose held out, the rest of BOTH arms fitted);
     held-out wrist error, paired against the base URDF on the same captures;
  3. noise floor: residual scatter within an arm pose (neck angles only differ) vs the arm-pose mean (pose bias).
Capture selection (decided before fitting, usable online): clean (rot <= 12 deg, coverage >= 0.6) and within 15 mm of
the arm's median base-URDF offset.
"""
import json, sys
from multiprocessing import get_context
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, "/data/kaustabh/p14arm/urdfx")
import predict  # noqa: E402

RS, LS = "RightShoulderPitch", "LeftShoulderPitch"


def P_xyz(j, i):
    def f(v, ch):
        ch.setdefault(j, {}).setdefault("dxyz", np.zeros(3))[i] += v
    return f


def P_zero(j):
    def f(v, ch):
        ch.setdefault(j, {})["post_axis_deg"] = ch.get(j, {}).get("post_axis_deg", 0.0) + v
    return f


NY = [("NY." + a, P_xyz("NeckYaw", i)) for i, a in enumerate("xyz")]
HEAD = [("HEAD." + a, P_xyz("HEAD", i)) for i, a in enumerate("xyz")]
MODELS = {
    "M12": NY + [("Rroll0", P_zero("RightShoulderRoll")), ("Lroll0", P_zero("LeftShoulderRoll"))],
    "M9": NY + [("Rroll0", P_zero("RightShoulderRoll"))],
    "NY3": NY,
    "X1": HEAD,
}


def build(spec, x):
    ch = {}
    for (_, f), v in zip(spec, x):
        f(v, ch)
    return ch


def pose_of(name):
    return name.split("-")[1]  # 01-p07-10-m12 -> p07 ; 01-pL07-10-0 -> pL07


def select(caps):
    base = np.array([c["off"] for c in caps])
    med = np.median(base, 0)
    keep = [c for c, o in zip(caps, base) if np.linalg.norm(o - med) <= 15.0]
    drop = [c["name"] for c, o in zip(caps, base) if np.linalg.norm(o - med) > 15.0]
    return keep, drop


def resid(spec, x, R, L):
    ch = build(spec, x)
    parts = []
    if R:
        parts.append(predict.off_change(R, "Right", ch).ravel())
    if L:
        parts.append(predict.off_change(L, "Left", ch).ravel())
    return np.concatenate(parts)


def fit(spec, R, L):
    fn = lambda x: resid(spec, x, R, L)

    def jac(x):
        f0 = fn(x)
        return np.column_stack([(fn(x + 0.1 * np.eye(len(x))[i]) - f0) / 0.1 for i in range(len(x))])

    r = least_squares(fn, np.zeros(len(spec)), jac=jac)
    J = r.jac
    dof = max(len(r.fun) - len(spec), 1)
    s2 = r.fun @ r.fun / dof
    cov = s2 * np.linalg.pinv(J.T @ J)
    se = np.sqrt(np.diag(cov))
    corr = cov / np.outer(se, se)
    return r.x, se, float(np.linalg.cond(J)), corr, float(np.sqrt(s2))


def norms(spec, x, caps, side):
    if not caps:
        return np.zeros(0)
    return np.linalg.norm(predict.off_change(caps, side, build(spec, x)), axis=1)


def fold(args):
    mname, side, pose, R, L = args
    spec = MODELS[mname]
    if mname == "M9" and side == "Left":
        pass
    hold = [c for c in (R if side == "Right" else L) if pose_of(c["name"]) == pose]
    Rf = [c for c in R if not (side == "Right" and pose_of(c["name"]) == pose)]
    Lf = [c for c in L if not (side == "Left" and pose_of(c["name"]) == pose)]
    x, se, cond, _, _ = fit(spec, Rf, Lf)
    new = norms(spec, x, hold, side)
    base = norms(spec, np.zeros(len(spec)), hold, side)
    return dict(model=mname, side=side, pose=pose, n=len(hold), new=new.tolist(), base=base.tolist(),
                x=x.tolist(), se=se.tolist())


def main():
    out = {}
    Rall, _ = predict.load("/data/kaustabh/p14arm/urdfx/base_w2.json")
    Lall, _ = predict.load("/data/kaustabh/p14arm/urdfx/base_L.json")
    R, dR = select(Rall)
    L, dL = select(Lall)
    print("clean right", len(Rall), "kept", len(R), "dropped (>15 mm from median)", dR)
    print("clean left ", len(Lall), "kept", len(L), "dropped (>15 mm from median)", dL)
    out["selection"] = dict(right=[c["name"] for c in R], left=[c["name"] for c in L], drop_right=dR, drop_left=dL)

    # 1 + 3: full fits
    out["full"] = {}
    for mname, spec in MODELS.items():
        x, se, cond, corr, rms = fit(spec, R, L)
        nR, nL = norms(spec, x, R, "Right"), norms(spec, x, L, "Left")
        print("\n== %s  rms %.2f mm/axis  cond %.0f" % (mname, rms, cond))
        print("   " + ", ".join("%s=%.2f+-%.2f" % (n, v, e) for (n, _), v, e in zip(spec, x, se)))
        big = [(spec[i][0], spec[j][0], round(float(corr[i, j]), 2)) for i in range(len(spec)) for j in range(i + 1, len(spec))
               if abs(corr[i, j]) > 0.7]
        print("   |corr| > 0.7:", big)
        print("   in-sample wrist error R median %.1f p90 %.1f | L median %.1f p90 %.1f mm" % (
            np.median(nR), np.percentile(nR, 90), np.median(nL), np.percentile(nL, 90)))
        floor = {}
        for side, caps in (("Right", R), ("Left", L)):
            O = predict.off_change(caps, side, build(spec, x))
            groups = {}
            for c, o in zip(caps, O):
                groups.setdefault(pose_of(c["name"]), []).append(o)
            within = np.concatenate([np.linalg.norm(np.array(g) - np.mean(g, 0), axis=1) for g in groups.values() if len(g) > 1])
            between = np.array([np.linalg.norm(np.mean(g, 0)) for g in groups.values()])
            floor[side] = dict(within_median=float(np.median(within)), pose_bias=between.round(1).tolist())
            print("   %s noise: within-pose median %.1f mm; arm-pose mean residual %s mm" % (
                side, np.median(within), between.round(1).tolist()))
        out["full"][mname] = dict(x=x.tolist(), se=se.tolist(), cond=cond, rms=rms, inR=nR.tolist(), inL=nL.tolist(),
                                  floor=floor, names=[n for n, _ in spec])

    # 2: leave-one-arm-pose-out
    jobs = []
    for mname in MODELS:
        for side, caps in (("Right", R), ("Left", L)):
            for pose in sorted({pose_of(c["name"]) for c in caps}):
                jobs.append((mname, side, pose, R, L))
    with get_context("fork").Pool(8) as pool:
        res = pool.map(fold, jobs)
    out["lopo"] = res
    print("\n== leave-one-arm-pose-out (held-out wrist error, mm; base = the unmodified URDF on the same captures)")
    for mname in MODELS:
        for side in ("Right", "Left"):
            rr = [r for r in res if r["model"] == mname and r["side"] == side]
            new = np.concatenate([r["new"] for r in rr])
            base = np.concatenate([r["base"] for r in rr])
            per = ", ".join("%s %.1f" % (r["pose"], np.median(r["new"])) for r in rr)
            print("%-4s %-5s median %.1f mean %.1f p90 %.1f max %.1f | base median %.1f | better %d/%d | per pose medians: %s" % (
                mname, side, np.median(new), np.mean(new), np.percentile(new, 90), new.max(), np.median(base),
                int(np.sum(new < base)), len(new), per))
    json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "h1.json", "w"), indent=1)


if __name__ == "__main__":
    main()
