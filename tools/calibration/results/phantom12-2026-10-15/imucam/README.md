# phantom12 camera-IMU calibration (15 Oct 2026, evening PDT)

**Result: `phantom12_stereo.yaml`**, the OKVIS config written by step 70 from fit
`calib_imucam-B-loose` on `config/okvis-template-phantom12.yaml`. Only `T_SC`, intrinsics,
distortion (equidistant) and `image_delay` differ from the template; `T_BS` and `imu_parameters`
are the template's (phantom21's, unverified on phantom12, see below). OpenCV `FileStorage` parses
it. Its header carries the same provenance as this file.

## Takes

Recorded with the OKVIS record app (`okvis2x:thor-0594dfc5…`, cameras 1,2, IMU index 2 =
NeckPitch, 180 s), fixed exposure **1500 us / ISO 1600** (1000 us was estimated too dark for the
evening light: about 56 of 255), board fixed at ~1.1 m, robot hanging on its gantry and moved by
hand. Quick check (`qc_imucam.py` via `qc-take.sh`, ~4 min per take) after each take:

| take | fps | gyro rms x / y / z (deg/s) | std \|a\| (m/s^2) | frames inside 450 px | use |
|---|---|---|---|---|---|
| A | 14.0 | 9 / 5 / 7 | 0.38 | 77 % | weak motion; comparison only |
| B | 13.6 | 26 / 24 / 34 | 1.98 | 67 % | **the result** |
| C | 13.5 | 25 / 29 / 37 | 1.54 | 56 % | repeat of B |

All three: IMU 500 Hz with 0 dropped, board found in every frame, brightness ~79 of 255. B and C
were flagged RETAKE only because the board sat outside the 450 px zone during the opening still
(the robot's resting pose); Kalibr starts at the first usable frame, so it costs nothing.

**Left/right sync at this exposure** (`sync_diag.py 6`, recommended stereo camchain):
+0.04 +- 0.02 ms (B, 384 pairs), +0.04 +- 0.03 ms (C, 374 pairs). Kalibr's per-camera timeshifts
agree to 0.01 ms, a second, independent confirmation.

## Processing (`calib-run-imucam.sh`, `calib-run-imucam2.sh`, on Thor)

Export -> step 10 `--mode imucam` -> `filter_edges.py 450` (now filters each camera on its own
corners: in imucam trees step 10 keeps every frame of each camera, unpaired, and the old pairwise
filter refused them) -> step 40 imucam with `CAM=calib_stereo3noedge-camchain.yaml` (stereo pose
held fixed) -> step 70 -> step 80 `--offline --compare`. Each Kalibr fit took 20-30 min on 2 CPUs.

## Fits (`compare_imucam.py`)

| fit | IMU noise file | reproj (px) | normalized gyro / accel | \|t_SC\| cam0 / cam1 (mm) | timeshift (ms) |
|---|---|---|---|---|---|
| A, edges filtered | `imu-phantom12-neckpitch.yaml` | 0.29 / 0.26 | 0.27 / 0.52 | 161.0 / 148.5 | +5.67 |
| B, edges filtered | same | 0.29 / 0.26 | 1.40 / 3.04 | 162.3 / 149.7 | +5.59 |
| B, unfiltered | same | 0.39 / 0.34 | 1.29 / 2.75 | 162.8 / 149.0 | +5.56 |
| C, edges filtered | same | 0.30 / 0.27 | 1.37 / 3.19 | 163.2 / 151.6 | +5.58 |
| **B, filtered, loose** | `imu-phantom12-neckpitch-loose.yaml` | **0.29 / 0.26** | **1.00 / 1.22** | **162.4 / 149.7** | **+5.61** |
| C, filtered, loose | same | 0.30 / 0.27 | 0.98 / 1.28 | 162.9 / 151.2 | +5.59 |

- **Repeatability:** every pair of fits within 1.6-7.8 mm, 0.07-0.31 deg and 0.11 ms; the final
  pair (B-loose vs C-loose) 4.4 mm, 0.08 deg, 0.02 ms.
- **Accelerometer residuals scale with motion, not with a wrong model:** gentle take A gives 0.52,
  the brisk takes ~3 (jolts up to 57 m/s^2, a hanging robot that flexes), yet A lands within
  1.6 mm of B. Loosening the noise model (accel x2.5, gyro x1.4) brings the normalized residuals
  to ~1 (calibration.md section 6) and moves the result 0.5 mm / 0.08 deg.
- **image_delay -0.0056 s** = -mean(timeshift): an event stamped t on the camera clock is seen at
  t + 5.6 ms in the IMU stream. Host-clock stamps, no rebase.

## Step 80 (offline, with phantom21's T_BS)

- Check 1 passes: baseline 44.83 mm; the 0.98 deg between the cameras is 93.5 % roll about the
  optical axis, true vergence 0.35 deg.
- Check 4 passes on everything it judges: baseline 100.0 % lateral, centre 1.4 mm off the head
  centreline, stagger 0.1 mm, azimuth 0.2 deg (a +0.6 deg yaw on T_BS would zero them; C asks +1.2).
- **Not judged, and the open item:** the optical axes look **30.1 deg down** in the Head frame and
  the cameras sit 0.14 m forward, 0.014 m up. phantom12's `T_SC` differs from phantom21's by
  ~20 deg and ~106 mm (all six fits), so phantom21's `T_BS` probably does not describe this head.
  Depth and VIO do not use T_BS; camera-in-Head (`T_BC = T_BS T_SC`, grasping) does. Measuring it
  is a hand-eye calibration (calibration.md section 9).

## Files

`phantom12_stereo.yaml` (final), `calib_imucam-*-{camchain-imucam,results-imucam,imu}.*` (Kalibr
outputs of the six fits), `sync-calib-imucam-{B,C}.csv`, `calib-run-imucam*.{sh,log}` (runners and
their logs), `qc_imucam.py` + `qc-take.sh` (take quick check), `cam-exposure.sh` (fixed/auto
exposure with the right restarts), `compare_imucam.py`, `orient_probe.py` (tag positions in one
frame). Not committed: the Kalibr report PDFs (~25 MB each) and step-40 logs (~17 MB each, progress
bars), on Thor in `/root/calib-work/` and on the workstation in
`~/calib/phantom12/results/imucam-2026-10-15/`. The takes, exports and trees are on phantom12 in
`/data/okvis-recordings/`.
