# phantom12 right-arm grasp correction dT, refitted on all plate data (25 Oct 2026; reviewed)

Plan: `docs/superpowers/plans/2026-10-25-tagfree-selfcal.md` (stereo-depth-eval branch), Phase B (decisions fixed
before data, amended 24 Oct 23:33 before data). Code: `dt_all.py` (this folder), `scripts/armfit_multi.py`,
`scripts/plate_sessions.py`. Run log `b3.log`, numbers `dt_all.json`.

**Use (grasping, right arm, #816 URDF only):** `T_Torso_cam(true) = dT · T_Torso_cam(URDF chain, live neck, Fit C T_BS·T_SC)`

    dT = [[ 1.      ,  0.000257,  0.000018, -0.001391],
          [-0.000257,  0.999995, -0.003281,  0.002994],
          [-0.000019,  0.003281,  0.999995,  0.001852],
          [ 0.      ,  0.      ,  0.      ,  1.      ]]

t = (−1.39, +2.99, +1.85) mm in the torso, 0.189° (rotation vector (0.188, 0.001, −0.015)°).

| | dT | error at poses the fit did not see (64 holds, RMS / max) |
|---|---|---|
| URDF chain alone (23 Oct) | none | 3.9 mm / 6.7 mm (1023c + 1023d) |
| today's dT (1023c + 1023d), held fixed, plates refitted | (−1.37, +2.32, +1.40) mm, 0.222° | 2.27 / 5.63 mm |
| **B1: one dT, all four sessions (adopted)** | **(−1.39, +2.99, +1.85) mm, 0.189°** | **1.75 / 5.01 mm**, 63/64 within 5 mm |
| B2: one dT per head-pitch band 18/23/28° | within ~0.6 mm / 0.2° of each other | 1.70 / 4.78 mm (not adopted) |

- **Moved from today's dT:** 0.81 mm, 0.037°. **Good to ±1.06 mm, ±0.16°** (bootstrap over holds, 95 %); leaving any one
  session out moves it ≤ 0.93 mm / 0.085°. The move is inside that ±: the parameters alone do not separate the two dTs.
  It is adopted by the rule fixed before data (unseen RMS lower by ≥ 0.3 mm: 0.52 mm).
- **Review (one Sonnet subagent): SOUND WITH NOTES.** Near-repeat poses flatter the gain by ~0.1 mm (without them
  0.43-0.46 mm); B1 is better on 56/64 holds; fitted without the session it is scored on, it beats today's dT on
  1023c (1.32 vs 1.98 mm) and rv1024 (1.15 vs 1.89), ties on rl1024 (2.24 vs 2.25), and is slightly better on 1023d
  (2.75 vs 2.98). No dependence on head pitch shows (B2 gains 0.05 mm; its 18° band rests on 2 holds).
- **Limits:** right arm only; head pitch 18-28°, neck yaw ≈ −0.4°; the plate fits leave ~2 px residual, so expect
  ~2 mm RMS (worst ~5 mm) at the hand, not better.
- A NeckPitch zero offset is not a separate model: at fixed yaw it equals a constant torso-side dT (checked to 1e-16).

## The plate yardstick (Phase C): `yardstick-v1/`
38 captures (1023c 12, 1023d 8, rv1024 9, rl1024 9) with the plate's hand pose in Fit C's camera 0 under B1 (the
reference), each session's M1, and M0; frames on Moonstone (`/data/<user folder>/yardstick-v1/frames.npz`, sha in
`MANIFEST.json`, paths relative to that folder). Score: `scripts/yardstick_score.py <dir> candidate.json`.
Baseline (`baseline/`): read B blind on rl1024 (e60ce2e6) FAILS on rotation, ~1.04° vs B1 (wrist 0.6-2.4 mm); its
numbers vs M1 reproduce the point-cloud agent's reviewed score exactly; the plain chain vs M0 reads 0.01 mm.
