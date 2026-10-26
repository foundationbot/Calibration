"""Several plate sessions, one camera correction (plan docs/superpowers/plans/2026-10-25-tagfree-selfcal.md, Phase B,
on the stereo-depth-eval branch).

armfit's M1 per session fits a plate placement X and a camera correction ΔT together. Across sessions the plate moved
(re-taped, slipped), so here every session s keeps its own X_s while ΔT is shared:

    pixel = project_c( (ΔT_g · T_TB(q_neck) · T_HC_c)^-1 · T_T,tip(q_arm) · X_s · P )

B1: one ΔT for every frame. B2: one ΔT per head-pitch band g (BANDS, nearest band). A NeckPitch zero offset is not a
model here: at fixed yaw it is exactly a constant torso-side ΔT (the plan's amendment of 24 Oct 23:33).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import armfit as af  # noqa: E402
import rigmath as rm  # noqa: E402

BANDS = (18, 23, 28)            # head pitch, deg: the pitches the 23-24 Oct sessions held


def band(q_pitch: float) -> int:
    """The pitch band (deg) nearest a neck pitch (rad)."""
    return min(BANDS, key=lambda b: abs(b - np.degrees(q_pitch)))


def _pose(p):
    return rm.transform(rm.exp_rot(p[:3]), p[3:6])


def _groups(sessions, bands):
    """[(s, g, frames)]: each session's frames split by ΔT group (all 0 for B1), and the band of each group."""
    keys = sorted({band(f.q_neck[1]) for obs in sessions for f in obs}) if bands else [None]
    out = []
    for s, obs in enumerate(sessions):
        for g, b in enumerate(keys):
            sel = [f for f in obs if b is None or band(f.q_neck[1]) == b]
            if sel:
                out.append((s, g, sel))
    return out, keys


def fit_all(sessions, arm, neck, rig, X0, bands=False, fixed_dT=None, holds=None) -> dict:
    """Fit one X per session and a shared ΔT (B1), or one ΔT per pitch band (bands=True, B2), to the frames of
    `holds` (per session a set of hold ids; None: all). fixed_dT: ΔT held at this value, only the plates fitted.
    -> X (per session), dT (per group), bands (per group; [None] for B1), rms_px, n_corners."""
    from scipy.optimize import least_squares
    sel = [[f for f in obs if holds is None or f.hold in holds[s]] for s, obs in enumerate(sessions)]
    parts, keys = _groups(sel, bands)
    stacks = [(s, g, af._Stack(fr, neck, rig)) for s, g, fr in parts]
    nx, ng = 6 * len(sessions), (0 if fixed_dT is not None else 6 * len(keys))

    def unpack(p):
        X = [X0 @ _pose(p[6 * s:6 * s + 6]) for s in range(len(sessions))]
        dT = [fixed_dT] if fixed_dT is not None else [_pose(p[nx + 6 * g:nx + 6 * g + 6]) for g in range(len(keys))]
        return X, dT

    def res(p):
        X, dT = unpack(p)
        return np.concatenate([(S.pixels(arm, X[s], dT[0 if fixed_dT is not None else g], np.zeros(0), ())
                                - S.uv).ravel() for s, g, S in stacks])

    r = least_squares(res, np.zeros(nx + ng), loss="soft_l1", f_scale=af.LOSS_PX, method="trf",
                      x_scale=np.tile([0.01] * 3 + [0.005] * 3, (nx + ng) // 6))
    X, dT = unpack(r.x)
    e = r.fun.reshape(-1, 2)
    return {"X": X, "dT": dT, "bands": keys if fixed_dT is None else [None],
            "rms_px": float(np.sqrt(np.mean(np.sum(e ** 2, axis=1)))), "n_corners": len(e)}


def unseen_errors(sessions, arm, neck, rig, fit: dict, holds=None) -> list:
    """Per session {hold: mm}: armfit.plate_errors of each hold in `holds` (per session a list; None: all) under the
    fit's X_s and the ΔT of the hold's pitch band."""
    out = []
    for s, obs in enumerate(sessions):
        hs = sorted({f.hold for f in obs}) if holds is None else holds[s]
        d = {}
        for h in hs:
            fr = [f for f in obs if f.hold == h]
            if not fr:
                continue
            b = band(np.median([f.q_neck[1] for f in fr]))
            g = fit["bands"].index(b) if fit["bands"] != [None] else 0
            prm = {"X": fit["X"][s], "dT": fit["dT"][g], "offsets": np.zeros(0), "offset_joints": []}
            d.update(af.plate_errors(fr, arm, neck, rig, prm, holds=[h]))
        out.append(d)
    return out
