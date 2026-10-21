"""Gravity test, refined. Per hold, the head IMU's gravity direction against the encoders' prediction,
with three model variants so the residual's cause can be told apart:
  base     R_SB (3) + gravity in the torso (2), torso assumed still
  torso    + the torso's own tilt per hold from the torso IMU (its mount R_T,Timu fitted, 3)
  scale    + per-axis accelerometer scale of the head IMU (3; a scale mismatch fakes pose-dependent tilt)
The residual is split along the head's pitch direction (what a pitch encoder error would move) and
across it, and the pitch component is compared between holds reached from above and below
(approach_pitch) and for drift: backlash shows as a sign flip with the approach side.
Usage: python gravity_fit2.py <signals.npz>..."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from gravity_fit import CFG, URDF, hold_means, lm  # noqa: E402
import neckfk  # noqa: E402
import okviscfg  # noqa: E402
import rigmath as rm  # noqa: E402


def predict(p, rows, chain, R_SB0, variant):
    R_SB = R_SB0 @ rm.exp_rot(p[:3])
    if variant == "base":
        th, ph = p[3], p[4]
        gT = [np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])] * len(rows)
    else:
        R_TI = rm.exp_rot(p[3:6])                  # torso IMU mount: torso <- torso IMU
        gT = [R_TI @ r["ut"] for r in rows]
    out = []
    for r, g in zip(rows, gT):
        u = R_SB @ chain.T_TB(r["yaw"], r["pitch"])[:3, :3].T @ g
        if variant == "scale":
            u = u / (1.0 + p[6:9])                 # the IMU reads a_i (1 + s_i): undo it on the model side
            u = u / np.linalg.norm(u)
        out.append(u)
    return out, R_SB


def main():
    cfg, chain = okviscfg.load(CFG), neckfk.NeckChain(URDF)
    R_SB0 = cfg.T_BS[:3, :3].T
    for path in sys.argv[1:]:
        rows = hold_means(dict(np.load(path)))
        print(f"== {Path(path).stem}: {len(rows)} holds")
        for variant, n in (("base", 5), ("torso", 6), ("scale", 9)):
            f = lambda p: np.array([np.cross(u, r["u"]) for u, r in zip(predict(p, rows, chain, R_SB0, variant)[0], rows)]).ravel()  # noqa: E731
            best = None
            starts = ([np.r_[0, 0, 0, th, ph] for ph in np.linspace(-np.pi, np.pi, 8, endpoint=False) for th in (0.05, 0.3, 2.8, 3.1)]
                      if variant == "base" else [np.r_[np.zeros(3), rm.log_rot(R), np.zeros(n - 6)]
                                                  for R in [rm.rot_axis(ax, ang) for ax in ([1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0])
                                                            for ang in (0, 1.5, 3.1, -1.5)]])
            for p0 in starts:
                try:
                    p, r = lm(f, np.array(p0, float))
                except np.linalg.LinAlgError:        # the scale variant can be singular on few holds
                    continue
                if best is None or r @ r < best[1] @ best[1]:
                    best = (p, r)
            if best is None:
                print(f"  [{variant}] not determined by these holds")
                continue
            p, r = best
            e = r.reshape(-1, 3)
            res = np.degrees(np.linalg.norm(e, axis=1))
            us, R_SB = predict(p, rows, chain, R_SB0, variant)
            # pitch direction in the IMU frame: the pitch axis (Head y) seen from the IMU
            ax = R_SB @ chain.pitch.axis
            pitch_comp = np.degrees(e @ ax)                    # signed rotation about the pitch axis
            ap = np.array([q["ap"] for q in rows])
            extra = f", accel scale {np.round(100 * p[6:9], 2)} %" if variant == "scale" else ""
            print(f"  [{variant}] R_SB vs config {np.degrees(np.linalg.norm(p[:3])):.3f} deg; residual rms {np.sqrt(np.mean(res ** 2)):.3f}, "
                  f"max {res.max():.3f} deg; about the pitch axis rms {np.sqrt(np.mean(pitch_comp ** 2)):.3f} deg{extra}")
            for s in (1, -1):
                if (ap == s).any():
                    print(f"      reached from pitch {'+' if s > 0 else '-'}: mean residual about the pitch axis "
                          f"{pitch_comp[ap == s].mean():+.3f} deg over {(ap == s).sum()} holds")
            if variant == "scale":
                print("      hold  yaw pitch ap drift | about pitch | total")
                for q, pc, rr in zip(rows, pitch_comp, res):
                    print(f"      {np.degrees(q['yaw']):+5.0f} {np.degrees(q['pitch']):+5.0f} {int(q['ap']):+d} {q['drift']:5.2f} | {pc:+.3f} | {rr:.3f}")


if __name__ == "__main__":
    main()
