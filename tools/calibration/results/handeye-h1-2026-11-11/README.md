# H-1: offline go/no-go for hand-eye calibration (11 Nov 2026) — verdict: GO

The hand-eye plan's first milestone (`docs/plans/2026-11-11-handeye-plan.md`, section 8). It answers, on data already
recorded, whether the URDF route can promise an accuracy on arm poses it has **not** seen. No robot was used.

- **Data:** phantom14's arm wave 2 of 4 Nov 2026:
  - 5 arm poses per arm, each captured at 3-4 neck angles;
  - right arm: 14 clean captures;
  - left arm: 15 clean captures, of which 14 kept.
- **The per-capture matches were not re-run.** The stored read-B (B0) results under the unmodified URDF were used
  (Moonstone `/data/kaustabh/p14arm/urdfx/base_w2.json`, `base_L.json`), with the stage-2 prediction of
  `urdfx/predict.py` unchanged.
- **Capture selection,** fixed before fitting and usable online:
  - clean = match rotation ≤ 12° and coverage ≥ 0.6;
  - within 15 mm of the arm's median offset;
  - dropped: left `01-pL07-10-0` (48 mm off).
- **Run:** `h1.py` and `h1b.py` on Moonstone; 5.6 s for 40 fits. Logs: `h1.log`, `h1b.log`; numbers: `h1*.json`.

## Result in plain words

1. **The correction works on unseen arm poses.** Every held-out capture (28 of 28) is better than with the unmodified
   URDF: typically 4-5 mm instead of 22-36 mm.
2. **The best model**, "M12+head" (8 terms), reaches a held-out median of **3.6 mm (right) and 5.3 mm (left)**,
   with 90 % of captures within **7.1 mm**. That is under the user's 6 mm aim on the median.
   - The terms: NeckYaw origin, both shoulder-roll zeros, and the camera's position in the head.
3. **Both shoulder-roll corrections are real:**
   - right −1.77 ± 0.18°;
   - left +0.52 ± 0.18° (about 3 standard errors).
4. **The extra joint-zero terms don't help on unseen poses:** shoulder yaw, elbow and shoulder pitch. Shoulder pitch
   also makes the fit ill-conditioned (condition 39, |correlations| up to 0.94). They stay out.
5. **The camera-in-head and head-mount terms can be told apart only through neck variety.** Their correlations are
   0.93-0.97 on this data. The session therefore moves the neck at every arm pose (decided, 11 Nov), and the planner
   should choose neck angles that separate them.
6. **The limit is pose-dependent, not noise.**
   - Scatter between neck angles of one arm pose (the matcher's noise floor) is about 3.7-3.9 mm.
   - Some arm poses stay 6-13 mm off. The worst is left pose L10, at 12-14 mm in every model. More, and more varied,
     arm poses are the way to lower the p90.

## Leave-one-arm-pose-out cross-validation

Every capture of one arm pose is held out, the rest of both arms is fitted, and the held-out wrist error is measured.
This repeats for each of the 5 poses of each arm.

| Model (terms) | Right median / p90 / max | Left median / p90 / max | Better than unmodified |
|---|---|---|---|
| unmodified URDF | 35.8 mm | 22.0 mm | — |
| X1: camera position in head only (3) | 9.4 / 10.9 / 13.8 | 9.4 / 13.0 / 13.6 | 28 / 28 |
| NY3: NeckYaw origin only (3) | 10.5 / 15.4 / 17.3 | 9.7 / 14.6 / 15.9 | 28 / 28 |
| M9: NY3 + right roll (4) | 4.6 / 11.4 / 12.5 | 6.3 / 11.5 / 12.3 | 28 / 28 |
| M12: NY3 + both rolls (5), phantom14's deployed URDF model | 5.3 / 10.3 / 11.4 | 5.3 / 11.0 / 13.9 | 28 / 28 |
| M12 + shoulder-yaw zeros (7) | 5.0 / 9.9 / 10.7 | 6.0 / 10.8 / 13.9 | 28 / 28 |
| M12 + elbow zeros (7) | 6.6 / 10.6 / 13.9 | 6.0 / 10.3 / 15.7 | 28 / 28 |
| M12 + shoulder-pitch zeros (7) | 5.9 / 10.5 / 11.1 | 6.3 / 10.5 / 14.3 | 28 / 28 |
| **M12 + camera position in head (8)** | **3.6 / 7.1 / 10.4** | **5.3 / 7.1 / 12.0** | 28 / 28 |

How to read it:
- one row per candidate model;
- each cell is the wrist-point error in mm on arm poses the fit did not see;
- lower is better;
- "better than unmodified" counts the held-out captures that improved.

## Full-data fit of the chosen model

From `h1b.log`, M12+head, both arms (condition number 71):

| Term | Value ± standard error |
|---|---|
| NeckYaw origin x / y / z (mm) | 2.7 ± 2.0 / 4.6 ± 2.4 / 7.0 ± 4.5 |
| Camera position in head x / y / z (mm) | −3.7 ± 2.5 / 11.1 ± 2.2 / −23.3 ± 4.2 |
| Right shoulder-roll zero (°) | −1.67 ± 0.13 |
| Left shoulder-roll zero (°) | +0.30 ± 0.14 |

How to read it:
- each row is one fitted correction with its uncertainty;
- a value well above its uncertainty is a real correction;
- the NeckYaw and camera-in-head rows trade off against each other (correlated), so their sum is better determined
  than either alone.

## What this does not show

- **No independent truth.** The error is read B against the corrected arm chain. A bias of the matcher itself (the mesh
  against the real arm covers, the depth scale) is invisible here. H0 scores the ported matcher against the
  (old) phantom12 plate data.
- **Only one robot (phantom14) and 5 arm poses per arm.**
- **Where the "head" term goes in the deployed files** (the URDF's head link, or T_BS) is decided in H0. It must not be
  applied twice.
