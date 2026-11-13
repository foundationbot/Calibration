"""Stage 2 of the hand-eye fit: a few URDF origins (and the camera's position in the head) fitted jointly over the
clean read-B captures of both arms, with standard errors, the condition number, leave-one-arm-pose-out
cross-validation, and the corrected URDF.

(Ported from the phantom14 stage-2 scripts, Moonstone p14arm/urdfx/{predict,urdfvar,twoarm}.py of 4 Nov 2026, and the
H-1 analysis tools/calibration/results/handeye-h1-2026-11-11/h1.py; the URDF changes are applied in memory instead of
through temporary files.)

The prediction: each capture's matched wrist point is held fixed in the camera, m = E · T_CT · [w, 1] (E the B0
correction, w the encoders' wrist point). Under changed origins, its offset is

    off = T_TB'(yaw, pitch) · D · T_BC · m  -  w'(q)        (mm, Torso frame)

with T_TB' the neck chain and w' the wrist point of the changed URDF and D = head_transform(HEAD, HEADR) the camera
moved in the head: D = [R | t], R the rotation vector HEADR (deg) and t the translation HEAD (mm), both in the head
frame; D · p = R · p + t (the camera turned about the head's origin, then moved). Valid for changes upstream of the
matched links (neck, shoulders) and in the camera chain.

The candidate models (MODELS: the hand-eye plan, section 4, H-1's result and phantom10's 11 + 13 Nov data), parameters
in mm and deg; run() fits each, cross-validates each and keeps the one with the smallest held-out error:
  M12+headR (DEFAULT, 8): NeckYaw origin xyz, a ShoulderRoll zero per arm, the camera's ROTATION in the head
           (HEADR). On phantom10 it beat M12+head on both days (held-out median right 7.0 -> 3.9 mm and left
           5.1 -> 3.6 mm on 11 Nov; right 9.1 -> 5.7 and left 17.1 -> 11.2 mm on 13 Nov): the head calibration's
           camera rotation is a few degrees off, and a degree at the wrist's ~0.6 m is ~10 mm;
  M12+head (8): the same with the camera's POSITION in the head (HEAD) instead -- H-1's model (phantom14);
  M12 (5, FALLBACK): no camera term -- phantom14's deployed model, and the model-only deploy's;
  M9 (4): NeckYaw origin + the right shoulder roll;  X1 (3): the camera's position in the head only.
Not a candidate unless asked for (EXTRA_MODELS, --models): M12+headRT (11: position AND rotation) overfits (13 Nov:
HEAD.y +42.6 mm).
Selection of the captures (before fitting): clean, and within OUTLIER_MM of the arm's median offset.

The camera terms (HEAD, HEADR) are not URDF origins: the camera sits in the Head link through T_BS, which the head
calibration owns. write_urdf() therefore writes the URDF terms only and states the camera terms in the header; the
deploy (calibtool/handeye_deploy.py) puts them into T_BS, once: T_BS' = D T_BS (see README.md)."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from armcal.urdf import Urdf, write_variant
from rigmath import exp_rot  # noqa: E402  (scripts/ on sys.path through armcal)

OUTLIER_MM = 15.0
FD_STEP = 0.1                      # mm or deg: the finite-difference Jacobian step (as H-1 and phantom14)
SIDES = ("Right", "Left")


# ------------------------------------------------------------------------------------------------- parameters
def P_xyz(joint: str, i: int):
    """the origin's i-th coordinate of `joint` (mm); joint "HEAD" = the camera's position in the head"""
    def f(v, ch):
        ch.setdefault(joint, {}).setdefault("dxyz", np.zeros(3))[i] += v
    return f


def P_zero(joint: str):
    """the zero offset of `joint` (deg)"""
    def f(v, ch):
        ch.setdefault(joint, {})["post_axis_deg"] = ch.get(joint, {}).get("post_axis_deg", 0.0) + v
    return f


def P_rot(joint: str, i: int):
    """the i-th component of a rotation vector (deg) of `joint`; only "HEAD" (the camera in the head) takes one"""
    def f(v, ch):
        ch.setdefault(joint, {}).setdefault("drot", np.zeros(3))[i] += v
    return f


NY = [("NY." + a, P_xyz("NeckYaw", i)) for i, a in enumerate("xyz")]
HEAD = [("HEAD." + a, P_xyz("HEAD", i)) for i, a in enumerate("xyz")]
# 13 Nov phantom10: with that day's refitted camera rotations the offsets grew with neck yaw (+-15 mm over +-12 deg)
# and no translation model fitted them (M12+head 9.6 / 16 mm held out): the camera's ROTATION in the head
HEADR = [("HEADR." + a, P_rot("HEAD", i)) for i, a in enumerate("xyz")]
RROLL, LROLL = ("Rroll0", P_zero("RightShoulderRoll")), ("Lroll0", P_zero("LeftShoulderRoll"))
MODELS = {                       # the default candidates, in order (a tie in choose() goes to the earlier)
    "M12+headR": NY + [RROLL, LROLL] + HEADR,
    "M12+head": NY + [RROLL, LROLL] + HEAD,
    "M12": NY + [RROLL, LROLL],
    "M9": NY + [RROLL],
    "X1": HEAD,
}
EXTRA_MODELS = {                 # fitted only when asked for (--models): 13 Nov, HEAD.y +42.6 mm -- overfits
    "M12+headRT": NY + [RROLL, LROLL] + HEAD + HEADR,
}
ALL_MODELS = {**MODELS, **EXTRA_MODELS}
DEFAULT, FALLBACK = "M12+headR", "M12"


def build(spec, x) -> dict:
    """the change {joint: {...}} of parameter vector x (HEAD included as a pseudo-joint)"""
    ch: dict = {}
    for (_, f), v in zip(spec, x):
        f(float(v), ch)
    return ch


# ------------------------------------------------------------------------------------------------- captures
@dataclass
class Obs:
    name: str
    side: str
    pose: str
    m: np.ndarray            # (4,) the matched wrist point in the camera
    T_BC: np.ndarray
    yaw: float
    pitch: float
    q: dict
    off: np.ndarray          # the record's offset under the URDF it was matched with (mm)


def pose_of(name: str) -> str:
    """the arm pose of a capture name: 01-p07-10-m12 -> p07 (the neck angles follow it)"""
    return name.split("-")[1]


def observations(records, side: str, clean_only: bool = True) -> list:
    out = []
    for r in records:
        if clean_only and not r["clean"]:
            continue
        T_TC, E = np.asarray(r["T_TC"], float), np.asarray(r["E"], float)
        m = E @ np.linalg.inv(T_TC) @ np.r_[np.asarray(r["wrist"], float), 1.0]
        out.append(Obs(r["name"], side, pose_of(r["name"]), m, np.asarray(r["T_BC"], float), float(r["yaw"]),
                       float(r["pitch"]), dict(r["q"]), np.asarray(r["off"], float)))
    return out


def load_records(path, clean_only: bool = True):
    """(observations, side) from a readb output (or phantom14's stored base_*.json, the same schema)"""
    D = json.loads(Path(path).read_text())
    return observations(D["res"], D["side"], clean_only), D["side"]


def select(obs, max_mm: float = OUTLIER_MM):
    """(kept, dropped names): captures within max_mm of the arm's median offset"""
    if not obs:
        return [], []
    off = np.array([o.off for o in obs])
    med = np.median(off, 0)
    d = np.linalg.norm(off - med, axis=1)
    return [o for o, x in zip(obs, d) if x <= max_mm], [o.name for o, x in zip(obs, d) if x > max_mm]


# ------------------------------------------------------------------------------------------------- prediction
def head_transform(head_dxyz=None, head_drot=None) -> np.ndarray:
    """D (4x4, m) = [R | t]: R the rotation vector head_drot (deg, head frame; Rodrigues), t = head_dxyz (mm, head
    frame). It maps the camera's place in the head: T_BC' = D @ T_BC, i.e. a point p of the head frame goes to
    R p + t -- turned about the head's origin first, then moved. No term: the identity."""
    D = np.eye(4)
    if head_drot is not None:
        D[:3, :3] = exp_rot(np.radians(np.asarray(head_drot, float)))
    if head_dxyz is not None:
        D[:3, 3] = 1e-3 * np.asarray(head_dxyz, float)
    return D


def offsets(obs, urdf: Urdf, head_dxyz=None, head_drot=None) -> np.ndarray:
    """(n,3) mm: each capture's wrist offset under `urdf` (and the camera moved by head_dxyz mm / turned by head_drot
    deg in the head)"""
    if not obs:
        return np.zeros((0, 3))
    neck = urdf.chain("Torso", "Head")
    D = head_transform(head_dxyz, head_drot)
    out = []
    for o in obs:
        wrist = urdf.chain("Torso", o.side + "WristRollLink")
        T_TC = neck.T({"NeckYaw": o.yaw, "NeckPitch": o.pitch}) @ D @ o.T_BC
        out.append(1e3 * ((T_TC @ o.m)[:3] - wrist.T(o.q)[:3, 3]))
    return np.array(out)


def off_change(obs, base: Urdf, change: dict) -> np.ndarray:
    change = dict(change)
    head = change.pop("HEAD", None) or {}
    return offsets(obs, base.changed(change), head.get("dxyz"), head.get("drot"))


# ------------------------------------------------------------------------------------------------- fit
@dataclass
class Fit:
    model: str
    names: list
    x: np.ndarray
    se: np.ndarray
    cond: float
    corr: np.ndarray
    rms: float               # mm per axis
    n_right: int
    n_left: int
    change: dict
    resid_right: np.ndarray = field(default_factory=lambda: np.zeros(0))   # mm, |offset| after the fit
    resid_left: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def undetermined(self) -> list:
        """parameters whose standard error exceeds their value (the plan: fixed at the URDF and said so)"""
        return [n for n, v, e in zip(self.names, self.x, self.se) if e > abs(v)]

    def summary(self) -> dict:
        return dict(model=self.model, params={n: [float(v), float(e)] for n, v, e in zip(self.names, self.x, self.se)},
                    cond=self.cond, rms_mm_per_axis=self.rms, n_right=self.n_right, n_left=self.n_left,
                    high_corr=[[self.names[i], self.names[j], round(float(self.corr[i, j]), 2)]
                               for i in range(len(self.names)) for j in range(i + 1, len(self.names))
                               if abs(self.corr[i, j]) > 0.7],
                    undetermined=self.undetermined(),
                    in_sample_median_mm={"Right": _med(self.resid_right), "Left": _med(self.resid_left)},
                    change=_jsonable(self.change))


def _med(a):
    return float(np.median(a)) if len(a) else None


def _jsonable(ch):
    return {j: {k: (np.asarray(v).tolist() if not np.isscalar(v) else float(v)) for k, v in d.items()}
            for j, d in ch.items()}


def residuals(spec, x, R, L, base: Urdf) -> np.ndarray:
    ch = build(spec, x)
    parts = [off_change(c, base, ch).ravel() for c in (R, L) if c]
    return np.concatenate(parts) if parts else np.zeros(0)


def fit(spec, R, L, base: Urdf, model: str = "") -> Fit:
    """robust-free least squares as H-1 and phantom14 (scipy least_squares, forward differences of FD_STEP)"""
    from scipy.optimize import least_squares
    fn = lambda x: residuals(spec, x, R, L, base)          # noqa: E731

    def jac(x):
        f0 = fn(x)
        return np.column_stack([(fn(x + FD_STEP * np.eye(len(x))[i]) - f0) / FD_STEP for i in range(len(x))])

    r = least_squares(fn, np.zeros(len(spec)), jac=jac)
    J = r.jac
    dof = max(len(r.fun) - len(spec), 1)
    s2 = r.fun @ r.fun / dof
    cov = s2 * np.linalg.pinv(J.T @ J)
    se = np.sqrt(np.diag(cov))
    corr = cov / np.outer(se, se)
    ch = build(spec, r.x)
    nR = np.linalg.norm(off_change(R, base, ch), axis=1) if R else np.zeros(0)
    nL = np.linalg.norm(off_change(L, base, ch), axis=1) if L else np.zeros(0)
    return Fit(model, [n for n, _ in spec], r.x, se, float(np.linalg.cond(J)), corr, float(np.sqrt(s2)),
               len(R), len(L), ch, nR, nL)


def fit_identified(spec, R, L, base: Urdf, model: str = ""):
    """(Fit, dropped names): parameters whose standard error exceeds their value are fixed at the URDF (left out)
    and the rest refitted, until none is"""
    dropped = []
    while True:
        f = fit(spec, R, L, base, model)
        bad = f.undetermined()
        if not bad or len(bad) == len(spec):
            return f, dropped
        dropped += bad
        spec = [p for p in spec if p[0] not in bad]


def lopo(spec, R, L, base: Urdf, model: str = "") -> list:
    """leave one arm pose out: every capture of one pose held out, the rest of BOTH arms fitted; per fold the held-out
    wrist errors (mm) under the fit ("new") and under the base URDF ("base")"""
    out = []
    for side, caps in (("Right", R), ("Left", L)):
        for pose in sorted({o.pose for o in caps}):
            hold = [o for o in caps if o.pose == pose]
            Rf = [o for o in R if not (side == "Right" and o.pose == pose)]
            Lf = [o for o in L if not (side == "Left" and o.pose == pose)]
            f = fit(spec, Rf, Lf, base, model)
            new = np.linalg.norm(off_change(hold, base, f.change), axis=1)
            old = np.linalg.norm(offsets(hold, base), axis=1)
            out.append(dict(model=model, side=side, pose=pose, n=len(hold), new=new.tolist(), base=old.tolist(),
                            x=f.x.tolist(), se=f.se.tolist()))
    return out


def cv_summary(folds) -> dict:
    s = {}
    for side in SIDES:
        rr = [r for r in folds if r["side"] == side]
        if not rr:
            continue
        new, old = np.concatenate([r["new"] for r in rr]), np.concatenate([r["base"] for r in rr])
        s[side] = dict(median=float(np.median(new)), p90=float(np.percentile(new, 90)), max=float(new.max()),
                       base_median=float(np.median(old)), better=int(np.sum(new < old)), n=len(new))
    allnew = np.concatenate([r["new"] for r in folds]) if folds else np.zeros(0)
    s["pooled_median"] = float(np.median(allnew)) if len(allnew) else None
    return s


def choose(cv: dict, models=tuple(MODELS)) -> str:
    """the candidate with the smallest pooled held-out median (cross-validation inside the fit data); a tie goes to
    the earlier in `models`"""
    return min((m for m in models if m in cv), key=lambda m: cv[m]["pooled_median"])


# ------------------------------------------------------------------------------------------------- output
def write_urdf(base: Urdf, f: Fit, out, provenance: list[str] = ()) -> Path:
    """the corrected URDF: base with the fitted origins; the camera terms HEAD / HEADR (not URDF origins) stated in
    the header"""
    ch = dict(f.change)
    head = ch.pop("HEAD", None)
    lines = [f"model {f.model}: " + ", ".join(f"{n} = {v:+.3f} +- {e:.3f}" for n, v, e in zip(f.names, f.x, f.se))
             + " (mm, deg)",
             f"condition {f.cond:.0f}; rms {f.rms:.2f} mm per axis; captures right {f.n_right}, left {f.n_left}"]
    if head is not None and "dxyz" in head:
        lines.append("camera-in-head translation HEAD = %s mm (head frame) is NOT in this file: apply it once, to "
                     "T_BS (T_BS' = D T_BS), or not at all" % np.round(head["dxyz"], 3).tolist())
    if head is not None and "drot" in head:
        lines.append("camera-in-head rotation HEADR = %s deg (rotation vector, head frame) is NOT in this file: apply "
                     "it once, to T_BS (T_BS' = D T_BS, D = [R(HEADR) | HEAD]), or not at all"
                     % np.round(head["drot"], 4).tolist())
    return write_variant(base, ch, out, lines + list(provenance))


def run(right_json, left_json, urdf, models=tuple(MODELS), held_out=(), max_mm=OUTLIER_MM) -> dict:
    """the whole stage 2 on readb outputs: selection, full fit and LOPO per model, the choice, held-out scoring"""
    unknown = [m for m in models if m not in ALL_MODELS]
    if unknown:
        raise ValueError(f"unknown models {unknown} (known: {', '.join(ALL_MODELS)})")
    base = Urdf.of(urdf)
    R0, _ = load_records(right_json) if right_json else ([], "Right")
    L0, _ = load_records(left_json) if left_json else ([], "Left")
    R0, dR = select(R0, max_mm)
    L0, dL = select(L0, max_mm)
    hold = set(held_out)
    R = [o for o in R0 if o.pose not in hold]
    L = [o for o in L0 if o.pose not in hold]
    HR = [o for o in R0 if o.pose in hold]
    HL = [o for o in L0 if o.pose in hold]
    rep = dict(selection=dict(right=[o.name for o in R], left=[o.name for o in L], drop_right=dR, drop_left=dL,
                              held_right=[o.name for o in HR], held_left=[o.name for o in HL]), models={}, cv={})
    fits = {}
    for m in models:
        spec = ALL_MODELS[m]
        f = fit(spec, R, L, base, m)
        fits[m] = f
        rep["models"][m] = f.summary()
        rep["cv"][m] = cv_summary(lopo(spec, R, L, base, m))
    best = choose(rep["cv"], models)
    rep["chosen"] = best
    if HR or HL:
        f = fits[best]
        rep["held_out"] = {side: dict(new=np.linalg.norm(off_change(H, base, f.change), axis=1).tolist(),
                                      live=np.linalg.norm(offsets(H, base), axis=1).tolist())
                           for side, H in (("Right", HR), ("Left", HL)) if H}
    return rep, fits


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--right", help="readb records of the right arm (JSON)")
    ap.add_argument("--left", help="readb records of the left arm (JSON)")
    ap.add_argument("--urdf", required=True, help="the URDF the captures were matched with")
    ap.add_argument("--models", default=",".join(MODELS),
                    help=f"candidates, comma-separated (default: {','.join(MODELS)}; also: {','.join(EXTRA_MODELS)})")
    ap.add_argument("--held-out", default="", help="comma-separated arm poses (e.g. p08,pL09) kept out of the fit")
    ap.add_argument("--report", required=True)
    ap.add_argument("--out-urdf", help="write the chosen model's URDF here")
    ap.add_argument("--note", action="append", default=[], help="a provenance line for the URDF header")
    a = ap.parse_args(argv)
    rep, fits = run(a.right, a.left, a.urdf, tuple(a.models.split(",")), [s for s in a.held_out.split(",") if s])
    if a.out_urdf:
        write_urdf(Urdf.load(a.urdf), fits[rep["chosen"]], a.out_urdf,
                   [f"CV (leave one arm pose out): {json.dumps(rep['cv'][rep['chosen']])}"] + a.note)
    Path(a.report).write_text(json.dumps(rep, indent=1))
    for m, s in rep["cv"].items():
        print(m, json.dumps(s))
    print("chosen", rep["chosen"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
