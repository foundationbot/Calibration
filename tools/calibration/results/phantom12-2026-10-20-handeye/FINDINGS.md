# phantom12 camera-in-Head with Fit A's lens (20-21 Oct 2026): findings

Plan: `docs/superpowers/plans/2026-10-20-phantom12-handeye-fit-a.md` (decisions fixed before any result). Data: the
17 Oct powered-neck takes P1, P2 (their corner files), refitted offline on Thor with step 81 (pixel model).
Independent results review (Fable, 21 Oct ~01:00) checked every number below against the report files.

## In plain words

- **Outcome A of the plan's Decision 5.** With Fit A's lens both takes say IMPROVE: camera 0 sits **13.9 mm / 0.48 deg
  (P1) and 14.2 mm / 0.51 deg (P2) from where the config puts it**, with a 1-sigma error of 2.8 / 2.5 mm. The config is
  shown wrong by 5-6 sigma, and the two takes agree with each other (2.2 mm apart, within their errors).
- **The old lens caused the cameras' disagreement.** Each camera fitted alone: 5.2 mm apart with the old lens (18 Oct),
  1.4 mm apart with Fit A, now inside the 3 mm gate.
- **For grasping:** the config's camera pose is off by ~21 mm at arm's reach (0.85 m). The measured pose is good to
  ~8 mm there at 2 sigma. That is far better than the config, but not the 3 mm / 0.3 deg budget.
- **Why not the gate:** the statistical error did not fall with the better lens (2.4-2.7 mm, 1 sigma; it was 2.3-2.5
  on 18 Oct). The hold-to-hold scatter is not lens-driven. Reaching the gate needs ~3x the holds (a longer powered-neck
  take; sigma ~ 1/sqrt(holds)) or camera-to-arm calibration with the board in the gripper.
- **Deliverable:** `T_BC-fita.json` = camera 0 in the Head link (mean of the two takes) with its error bars. **No
  `T_BS` was written:** X is valid only with Fit A's lens (the lens change alone moved X by ~6 mm, 5.2 of it along the
  optical axis). `T_BS = X · T_SC0^-1` is computed by step 83 against the config that carries Fit A's lens, once that
  exists (rerun step 81 with `--config <that config> --lens-json ...`, about a minute).

## The numbers (primary: corners within 450 px)

How to read it: "X vs config" = how far the measured camera pose is from the config's (rotation, distance); "at
0.85 m" = the worst-case error that makes at a point 0.85 m from the camera; sigma = 1-sigma error of X (statistical,
clustered by neck hold; systematic = half the distance between the two cameras' own fits); "cameras apart" = the two
cameras' own fits compared.

| | P1 | P2 |
|---|---|---|
| verdict | IMPROVE | IMPROVE |
| X vs config | 0.480 deg / 13.91 mm | 0.510 deg / 14.15 mm |
| X vs config, Head x y z (mm) | -2.87 -12.38 +5.67 | -2.00 -11.80 +7.55 |
| at 0.85 m | 21.0 mm | 21.7 mm |
| sigma total (stat / sys) | 0.084 deg / 2.80 mm (2.72 / 0.68) | 0.079 deg / 2.53 mm (2.42 / 0.71) |
| cameras apart | 1.36 mm / 0.109 deg | 1.42 mm / 0.105 deg |
| pixel rms after fit | 0.713 px | 0.674 px |
| corners cut | 0.42 % | 0.40 % |
| holds / images | 45 / 446 | 48 / 472 |
| P2 vs P1 | | 2.16 mm / 0.039 deg: agree (within 2 combined sigma) |

Verdict arithmetic (step 81's rule, recomputed by the reviewer): KEEP needs angle + 2 sigma <= 0.3 deg (0.65 / 0.67:
no); REPLACE needs 2 sigma <= 3 mm (5.60 / 5.05: no); IMPROVE: the config shown outside the gate (13.91 - 5.60 = 8.31,
14.15 - 5.05 = 9.10 mm > 3) and the fit better than the config in translation (13.91 > 4 sigma = 11.21; 14.15 > 10.10)
and good to the gate in rotation (2 sigma 0.168 / 0.159 deg <= 0.3). **P1's IMPROVE is the tighter one:** a 24 % larger
translation sigma would have made it INCONCLUSIVE.

## Against 18 Oct (same takes, old lens)

| | P1 18 Oct -> today | P2 18 Oct -> today |
|---|---|---|
| X vs config | 9.95 mm / 0.375 deg -> 13.91 / 0.480 | 10.31 / 0.410 -> 14.15 / 0.510 |
| sigma (stat, sys) | 3.62 (2.54, 2.58) -> 2.80 (2.72, 0.68) | 3.43 (2.26, 2.59) -> 2.53 (2.42, 0.71) |
| cameras apart | 5.17 -> 1.36 mm | 5.17 -> 1.42 mm |
| pixel rms | 0.773 -> 0.713 | 0.744 -> 0.674 |

- **Correction to 18 Oct's FINDINGS (`phantom12-2026-10-17-tbs/FINDINGS.md`):** "camera 0 ~10 mm / 0.4 deg from the
  config" is now **~14 mm / 0.5 deg**, and its Head x component changed sign (+1.4 / +2.2 -> -2.9 / -2.0 mm). The old
  lens pulled X ~5.2 mm along the optical axis in both takes (Fit A's focal length is ~0.35 % longer: 564.9 vs 562.9
  px, 567.2 vs 565.4, worth ~3.9 mm of board distance at 1.1 m, same sign). "Not settled that the config is outside the
  gate" is now settled (8.3 / 9.1 mm beyond 3 mm at 2 sigma). Confirmed: the 5 mm between the cameras was the
  calibration; "mostly along Head y" holds.
- Checks before any Fit A number was read: the new corner-file route with the old lens reproduced 18 Oct to 0.05 mm /
  0.003 deg (P1) and 0.01 mm / 0.001 deg (P2) (Decision 3); the config's own lens through the new `--lens-json` path gave
  the same X to 1.3e-11 (3b).

## Secondary (corners within 560 px; reported, never deciding)

Same verdicts (IMPROVE, IMPROVE), takes agree (1.35 mm), cameras 1.35 / 1.32 mm apart: outcome A too. X moves by 2.45
mm (P1) and 1.78 mm (P2) against the 450 px fit, in the same direction in both takes and mostly along Head z (the
weakest axis, sigma_z 2.4-2.5 mm). A radius-dependent shift of that size sits outside the statistical error: one more
reason the gate is not reached, and a thing camera-to-arm calibration can check.

## Caveats

- Fit A was fitted partly on P1/P2's even-ranked holds (lens and stereo pose, board pose free per pair). X is not in
  that fit; the reviewer found no sign it matters (the pixel rms dropped equally on even and odd holds).
- The reports cannot prove the tangential terms were applied (K, D, C are not stored); the evidence is indirect (rms
  down, the cameras' gap closed 5.2 -> 1.4 mm) plus the code review and the unit tests.
- Two takes of one session (17 Oct); both after the ~0.2 px session shift like Fit A.

## Files

`fita450-P{1,2}.json` (primary), `fita560-P{1,2}.json` (secondary), `repro-P{1,2}.json` + `cfglens-P1.json` (the
gates), `T_BC-fita.json` (the deliverable), `run.sh` (the exact commands), `cfg-as-fit.json`, `fita.log`, `repro.log`.
Runs: Thor, `calib-py:3`, package snapshot `ed73ddc8`, 42-50 s per take.
