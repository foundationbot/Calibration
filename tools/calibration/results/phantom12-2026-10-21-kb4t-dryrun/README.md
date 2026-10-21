# KB4T steps, real-data dry run on Thor (21 Oct 2026): step 45 from the nominal start on Fit A's own data

Productising plan `docs/superpowers/plans/2026-10-20-calibration-kb4t-productise.md`, Task 17 (rules written before the
run). Package snapshot `ed6af4ff` on Thor (`/root/calib-tools-kb4t`), `calib-py:3`, CPU only, no robot motion.
Input: Fit A's fit set (`final_fit.sets()`, FF_NOFAR=1, half 0) as one corner file: 1793 pairs in 220 holds (the
extraction's printout, not kept in a file); 1777 pairs / 218 holds after step 45's >= 12-corner filter (= Fit A's
1777); 1741 pairs / 216 holds got a starting pose (36 pairs in 2 holds failed the nominal-start PnP; where in the
image they were is not recorded). Run 21 Oct 00:40-00:55 (Thor clock).

## In plain words

- **The new fitter, started from nothing but the rig file's nominal numbers, lands on Fit A.** Focal lengths within
  0.04 px, image centres within 0.03 px, tangential terms practically equal, stereo pose within 0.002 deg, baseline
  44.433 vs 44.427 mm. Converged on its first attempt (170 evaluations, 14.5 min on Thor), 0.391 px rms.
- **Depth agreement with Fit A at 1 m:** 0.03-0.11 % in every image cell within 450 px of the image centre; 5-6 % in
  the image corners beyond that (as Fit A vs Fit B on 20 Oct: these data barely reach the corners, so the high-order
  lens terms k2-k4 are free to trade off there).
- **Pre-registered rule (b) failed narrowly** (+-0.10 % asked; the left cell read 0.106 %). **The user loosened the bound
  to +-0.11 % after seeing the result (21 Oct ~12:30), so rule (b) passes** and the acceptance session is not blocked by it.
- **The quality gate is too strict for the 19 Oct stand take (rule (c), report only):** with the gate on (0.6 px per
  hold), step 45 REJECTED the recording: 60 of 216 holds (28 %) fit a rigid board worse than 0.6 px (40 at 0.6-0.8,
  13 at 0.8-1.0, 4 at 1.0-1.3, 3 above 1.3 px). They are spread across the stand take's segments (50 of the 60; 2-7
  holds per segment, segment medians 0.39-0.61 px); the three 1.6 m segments lose 10 (medians 0.85-0.98 px); the four
  neck takes (medians 0.38-0.45 px, the source of the 0.30-0.42 px reference) lose 3. So it is not shown to be too
  strict for real data in general: hand-held fixed-exposure sweeps may look like the neck takes. A 1.0 px gate would
  drop 7 holds (3 %) here. Note: the gate's per-hold rms includes corners the fit trimmed (3-20 px), unlike the fit's
  rms; whether the 20 Oct reference used the same convention is not recorded. Decide the threshold on the acceptance
  session's hand-held data. (Independent results review, 21 Oct ~04:40.)

## Pre-registered rules and outcomes

| rule (written before the run) | outcome |
|---|---|
| (a) step 45 converges (judged on the gate-off run) | **PASS**: success on attempt 1, 170 nfev, 863 s, rms 0.391 px |
| (b) gate off: depth new vs Fit A at 1 m within +-0.10 % in every cell of the 450 px table | **FAIL (narrow)** as registered: left 0.106 %; centre 0.04, top 0.05, bottom 0.03, right 0.09, corners 0.03-0.10. **PASS under the user's loosened bound +-0.11 %** (decided 21 Oct ~12:30, after the result) |
| (c) gate on: dropped holds listed, map by cell, no pass rule | the recording was **rejected** (28 % of holds over 0.6 px); the map was then made by refitting with the rejection limit lifted (below) |

## Numbers (camera 0 / camera 1; Fit A -> this fit)

| | camera 0 | camera 1 |
|---|---|---|
| fx, fy (px) | 564.88 -> 564.89, 563.93 -> 563.93 | 567.21 -> 567.24, 566.15 -> 566.19 |
| cx, cy (px) | 642.83 -> 642.84, 384.95 -> 384.95 | 642.52 -> 642.54, 383.06 -> 383.04 |
| p1, p2 | 0.000852 -> 0.000841, -0.001113 -> -0.001120 | -0.000371 -> -0.000370, -0.0000727 -> -0.0000613 |
| k1..k4 | -0.00828, 0.01186, -0.00050, -0.00142 -> -0.00855, 0.01267, -0.00130, -0.00119 | -0.00610, 0.00664, 0.00309, -0.00235 -> -0.00714, 0.01008, -0.00102, -0.00074 |

Stereo pose: rotation 0.0018 deg apart; translation (mm) [44.425, -0.226, 0.302] -> [44.431, -0.237, 0.310].

Depth difference at 1 m by cell (`depthmap-nogate.txt`), largest |difference| %:

| cell | whole raw image | within 450 px |
|---|---|---|
| centre | 0.04 | 0.04 |
| top / bottom | 0.05 / 0.03 | 0.05 / 0.03 |
| left / right | 0.84 / 1.46 | **0.11** / 0.09 |
| corners (TL, TR, BL, BR) | 5.03, 5.54, 6.23, 6.16 | 0.09, 0.08, 0.03, 0.10 |

## Rule (c)'s map: the gate's holds dropped (added 21 Oct ~05:10, after the results review asked for it)

Same input, the gate at 0.6 px, the recording-rejection limit lifted (`rig-gatedrop.yaml`, `dry-gatedrop.sh`): step 45
dropped the 60 holds and refit: converged (attempt 1 of each fit; 702 s + 452 s on Thor), 0.313 px rms over 1180 pairs
in 156 holds, **baseline 44.25 mm** (Fit A 44.43). Depth against Fit A at 1 m (`depthmap-gatedrop.txt`), largest
|difference| %:

| cell | whole raw image | within 450 px |
|---|---|---|
| centre | 0.21 | 0.21 |
| top / bottom | 1.62 / 0.38 | 1.62 / 0.37 |
| left / right | 8.38 / 4.37 | 1.14 / 0.27 |
| corners (TL, TR, BL, BR) | 18.62, 9.31, 2.75, 2.70 | 1.88, 0.66, 0.81, 0.37 |

Reading: dropping the 0.6-0.8 px holds removes data that constrains the outer image, and the fit moves away from Fit A
(validated within +-0.5 % on held-out holds) by up to 1.9 % within 450 px: these holds carry information, they are not
noise to be cleaned out. With no held-out test here this does not say which fit is right, but together with the counts
above it is a second reason not to adopt the 0.6 px gate as it stands.

## What this means for the acceptance session (plan Task 18)

The fitter reproduces a known calibration from a nominal start; what limits agreement is the data's coverage (the
corners), which is exactly what the hand-held protocol with the live coverage map (step 06) is for. Before that session:
the user decides on rule (b)'s bound and the quality gate's threshold.

## Files

`fit-nogate.json` / `fit-gate.json` (step 45 outputs; the gate one is the rejection record with the 60 holds),
`fit-*.log`, `depthmap-nogate.txt`, `dry-fits.sh` (the launch), `rig-nogate.yaml` (the rig copy with the gate off),
`fita_fit_set.py` (the input extraction, run in the 20 Oct analysis environment). The corner file itself stays on Thor
(`/root/calib-work/kb4t-dryrun/fita-fit-corners.npz`, 3.4 MB).
