# First live hand-eye session: phantom10, 11 Nov 2026, 08:58-09:16 (H3)

Both arms waved by `scripts/handeye_wave.py`. Read B (`armcal.readb`, the robust matcher) and the stage-2 fit
(`armcal.urdffit`) ran **on the robot**. The robot's consumer URDF was the base:
`/etc/phantomos/okvis/mk1_1.urdf`, md5 cbf58ef1.

**Plain result.**
- **The arms were 38-45 mm off.** With the unmodified URDF, the arm and the head camera disagreed by 38 mm (left) and
  45 mm (right) at the wrist.
- **After the fit:** about 5 mm (left) and 7 mm (right) on arm poses held out of the fit.
- **Every held-out capture got better** (25 of 25).
- **The left arm meets the < 6 mm aim; the right arm doesn't yet.** Only 9 of its 20 captures were usable: pose p07
  gave none and p04 one.
- Nothing has been deployed: the URDF is written, the deploy (H4) isn't built.

## Accuracy (leave one arm pose out)

| Model | Right median / p90 | Left median / p90 | Unmodified URDF, R / L |
|---|---|---|---|
| **M12+head** (chosen) | **7.0 / 11.5 mm** | **5.1 / 8.4 mm** | 45.0 / 37.8 mm |
| M12 | 7.1 / 9.2 | 7.1 / 10.9 | |
| M9 | 9.3 / 10.1 | 6.5 / 11.0 | |
| X1 | 7.9 / 14.6 | 5.8 / 7.7 | |

How to read it:
- one row per model;
- each cell is the wrist error on arm poses the fit did not see (median, and the value 9 in 10 captures stay under);
- lower is better.

## Fitted corrections (M12+head, both arms jointly)

Condition number 99; residual 3.6 mm per axis.

| Term | Value ± standard error |
|---|---|
| NeckYaw origin x / y / z | −9.4 ± 4.3 / −12.1 ± 5.8 / −27.1 ± 7.4 mm |
| Right shoulder-roll zero | −1.19 ± 0.23° |
| Left shoulder-roll zero | −0.94 ± 0.23° |
| Camera in head x / y / z | −21.4 ± 4.9 / −1.6 ± 5.3 / −5.6 mm (not written into the URDF; where it goes is still open) |

How to read it:
- a value well above its uncertainty is a real correction;
- both shoulder rolls are;
- the head terms trade off against each other, which is why their uncertainties are large.

## Captures

| Arm | Captures | Clean (two passes agree) | Used in the fit | Dropped |
|---|---|---|---|---|
| Right | 20 | 12 | 9 | p07 22/12 and 26/0 (30-60 mm from the rest: wrong matches), p08 22/−12 |
| Left | 20 | 16 | 16 | none |

How to read it: "clean" = the matcher settled on one answer; "used" = clean and within 15 mm of the arm's median.

## Time (measured from `log-*.jsonl`)

| Arm | Robot time | Depth bursts | Arm moves | Neck moves | Read B on the robot |
|---|---|---|---|---|---|
| Right | 430 s (7.2 min) | 21, 9.1 s median | 9, 107 s in all | 1.9 s median | 30 s (4 processes) |
| Left | 448 s (7.5 min) | 21, 9.1 s median | 11, 122 s in all | 1.9 s median | 27 s |

How to read it: robot time is from the first command to the arm back at rest (part A + part B). The bursts are about
45 % of it, at the default depth rate (one 1280 frame every 1.93 s).

## What was found live

- **Sag with the motion tool's own gains is large.** No `--kp` was passed (the user's "don't overdo kp"). At p07 the
  right shoulder pitch reached −34.7° for −58.2°, roll −8.5° for −21.2°, elbow −31.5° for −42.6°. Clearance was
  re-checked with 1.3x this sag before part B: ≥ 3.8 cm. The fit uses measured angles, so sag doesn't bias it. It does
  change the views: the planned p07 view is not the one the camera saw.
- **The joint conventions match the model.** The rest-pose and first-pose captures matched at 2.6-3.7° rotation, 89-95 %
  coverage.
- **The left arm's shoulder-yaw departure** (lift first) worked as planned.

## Files

- `readb-{right,left}.json`: read B records.
- `fit.json`: the fit, CV and selection.
- `phantom10_mk1_handeye-20261111.urdf`: the corrected URDF, not deployed.
- `log-{right,left}.jsonl`: every command with its duration.
- `wave-p10-*-{A,B}.json`: the plans as run.
- `snap-phantom10-1111-0857.json`: the start pose.
- The bursts (40 x 57 MB): phantom10 `/data/calib/handeye/p10-wave-1111a-{right,left}/` and Moonstone
  `/data/kaustabh/p10arm/live-1111a/`.
