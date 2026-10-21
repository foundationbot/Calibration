# Head camera calibration: what we know

What it takes to calibrate a Phantom head (OAK stereo pair + the NeckPitch IMU) with
`tools/calibration/`, what went wrong on phantom12 and how it was fixed, the results, and the
open work. `README.md` in this folder is the package manual (per-step usage); this file is the
experience on top of it.

**How to extend this file.** Add a row to the status table, a section under §7 for a new
robot's results, a numbered entry under §8 for a lesson, or a bullet under §9 for an idea.
Give every number its date and source (a file, a log line or a commit subject), and mark
anything not measured as a hypothesis. Commits are cited by subject, because rebases change
their IDs.

## 1. Status by robot

| robot | stereo | camera–IMU | OKVIS config | where |
|---|---|---|---|---|
| **phantom12** | **done 2026-10-15** (`calib_stereo3noedge`, §7) | **done 2026-10-15** (`calib_imucam-B-loose`, §7) | `results/phantom12-2026-10-15/imucam/phantom12_stereo.yaml`; **deployed on phantom12 2026-10-16** as a transplant (`imucam/deploy/`, §7); its `T_BS` is phantom21's: **refitted 21 Oct with Fit A's lens (KB4T): camera 0 ~14 mm / 0.5° from where the config says**, ±2.5-2.8 mm (1σ), IMPROVE on both 17 Oct takes, the cameras now agreeing to 1.4 mm; not good to the 3 mm gate; deliverable `T_BC-fita.json`, `T_BS` follows with a config that carries Fit A's lens (§6b, `results/phantom12-2026-10-20-handeye/FINDINGS.md`; the 18 Oct old-lens answer ~10 mm / 0.4° is superseded) | `results/phantom12-2026-10-15/` |
| phantom21 | done 2026-08 (radial-tangential, 43.37 mm) | done | `config/foundation_rig/mk12_stereo_holevisor.yaml` | repo |
| phantom13 (mk13) | done 2026-08-22 (the package's origin) | done | `config/rig.yaml` lineage | `reference/` |

## 2. Terms

- **Stereo calibration:** each lens (focal length, image centre, distortion) and the pose of the
  right camera relative to the left (**baseline**: the distance between them).
- **Camera–IMU calibration:** where the cameras sit relative to the head IMU (`T_SC`, camera in
  IMU frame) and the time offset between their clocks. Needs the head to move.
- **`T_BS`:** the IMU's pose in the **Head link** frame (B, the URDF/planner's body frame).
  Not observable from a camera–IMU calibration, because moving the whole head shows the camera
  and IMU only relative to each other; it comes from the step-70 template. Camera in head is
  `T_BC = T_BS · T_SC`.
- **Hand-eye calibration:** finding the fixed transform between a robot link and a sensor on
  it from paired motions (link motion from the joint encoders, sensor motion from the sensor).
  Measuring `T_BS` (or `T_BC` directly) is one, with the Head link as the "hand" (§9).
- **Row error (step 60):** after rectification the same point must land on the same row in both
  images; the median vertical mismatch on held-out frames.
- **Scale check (step 65):** triangulate the board with the stereo calibration and compare tag
  spacing with the steel-rule pitch.
- **Scale against distance:** the same ratio, binned by board distance, on frame pairs no fit
  used (§5). It catches distance-dependent errors that steps 60 and 65 miss.
- **Edge frames:** frames with tag corners far from the image centre; on phantom12, beyond
  450 px (~46° off-axis) the lens models stop fitting (§8.2).

## 3. The pipeline at a glance

```
 robot (Thor)                                                         workstation / anywhere
 ─────────────────────────────────────────────────────────────────    ──────────────────────
 05 record (OKVIS record app pod)  → session.mcap (lossless PNG + IMU)
 mcap_to_euroc.py                  → per-frame folders + timestamps
 10 build trees                    → tree/<take>/{cam0,cam1[,imu0.csv]}, heldout2/
 20 coverage gate (≥ 60 % grid, ≥ 60° reach)
 40 Kalibr stereo                  → <bag>-camchain.yaml
 60 held-out row error (< 0.5 px)
 65 scale vs steel rule (< 0.5 %)
 scale against distance (depth_scale2.py, §5)
 KB4T lens, hand-held (§4b, §11): 06 live coverage + 05 record, a fit sweep and a test sweep
   → 15 corners → 45 fit (nominal start) → 46 split halves → 66 validation (fixed rule) → 71 export
 10 imucam tree → 40 Kalibr imucam → 70 OKVIS yaml → 80 checks          (camera–IMU, §6)
```

**Everything heavy runs on Thor, in capped containers** (§8.9): `--cpus=2 --cpu-shares=2
--memory=16g` (lowest CPU weight, so it only takes idle CPU), never on isolated cores 10–13, and
long jobs started with `setsid nohup` so they survive a dropped session. The WiFi link from the
robot measured 0.8 MB/s, so exports and trees are built on the robot and only results travel.

Images on phantom12: `calib-py:2` (also tagged `calib-py:1`; python 3.11 + numpy,
opencv-python-headless 5.0, pyyaml, pillow, pytest, and for MCAP/dma-recorder reading zstandard,
mcap, protobuf, mcap-protobuf-support), rebuilt by `/root/calib-py-build2.sh` on 2026-10-16 after
all images were pruned (§8.18). The Kalibr image (`kalibr:1f60227-arm64`, §8.5) was pruned too and
has not been rebuilt (43 min on Thor). `calib-py:3` (phantom12, in use from 19 Oct: python 3.11,
numpy, scipy, opencv 5; source: the 19 Oct lens-take plan) is the image the lens, hand-eye and
KB4T steps (15, 45, 46, 66) run in. The runner scripts used on 2026-10-15 are `results/phantom12-2026-10-15/calib3/calib-run*.sh`
(stereo) and `results/phantom12-2026-10-15/imucam/calib-run-imucam*.sh` (camera–IMU): copy and
adapt one rather than retyping commands.

## 4. Before any take: checklist

1. **Cameras in sync.** Record ~1 min of a board moved up and down and run `sync_diag.py`
   (`results/.../diag/`): the fitted delay must be under 0.5 ms. Do this whenever the camera
   producer's config, the DMA.video version or the set of running cameras changed (§8.1).
   A camera–IMU take is its own sync check: run `sync_diag.py` on it, and Kalibr's two
   per-camera timeshifts must agree.
2. **Only the cameras you need, or every camera on fixed exposure.** On phantom12 the bottom
   camera is removed from `/etc/phantom/head_camera.json` (§8.1).
3. **Fixed exposure** on left and right for the take, restored afterwards: stereo 3000 µs /
   ISO 800 worked indoors; camera–IMU used 1500 µs / ISO 1600 in evening light (image mean
   ~79/255; 1000 µs would have given ~56). Each change is a producer restart: scale 0 → 1, never
   a rolling restart. `results/.../imucam/cam-exposure.sh fixed [µs] [iso] | auto` does the edit
   and every restart in order, and refuses unless the config is the known auto file.
4. **After every producer restart, restart the viewer and the RTSP streamer** too (scale 0 → 1):
   otherwise they keep showing frozen images (§8.7).
5. **Record pod CPU limit 4** (`05_record.sh` on the robot, its `CPU` default): 2 CPUs gave 10–12 fps, 4 gave ~12–13.
6. **IMU index:** `python3 display_imu.py` on the robot (repo `deployment/`) prints the roster;
   NeckPitch was index 2 on phantom12 (2026-10-15). The roster can change.
7. **Board measured:** `target-aprilgrid.yaml` and `target_measured: true` (phantom12: 720 mm
   over 6 pitches, 480 mm over 4, pitch 0.120 m).
8. **Live view on a phone** (`http://<robot>:9299` or `:8889/left`) for whoever holds the board:
   without it, nobody reaches the picture edges (§8.4).

## 4b. KB4T lens calibration: the hand-held recording

What the KB4T fit (step 45) and its validation (step 66) need from the recording, and why. Steps and commands:
README "KB4T lens calibration"; a new robot: §11. The hold counts and pixel and depth limits here are `gates.lens_*`
values of the rig file (defaults shown); the distance bins are fixed (`scripts/lenscells.py`).

1. **Two separate sweeps:** one recording to fit, a second one only to test. The test sweep is the independent
   hold-out: no fit sees it, so its depth errors are a fair measure. (Until 20 Oct the held-out data were alternate
   holds of the same recordings, which cannot catch anything that changed between recordings.)
2. **Fixed short exposure on both cameras** (`/root/cam-exposure.sh fixed 1500 1600` on phantom12: a producer restart,
   the operator's yes; `auto` afterwards; §4 items 3-4). On auto the left/right exposure offset reached 0.47 ms on
   phantom12 (§8.1): a board moving at 300 px/s is then seen 0.14 px apart by the two cameras, ~0.5 % of depth at
   1 m. On fixed exposure it measured +0.04..+0.10 ms (phantom12).
3. **Hand-held board, slow moves, 2-3 s still pauses.** Only still stretches are used (step 15); a still stretch is a
   "hold", the unit the error bars and the 8-hold rule count. A pause that drifts can split into two stretches and
   count twice: hold still, and use `--still-px 3` in step 15 for hand-held sweeps (on fixed exposure a 3 px drift
   costs under 0.01 px of disparity). Vary the board's tilt between pauses (the 19 Oct take turned it ~30 deg).
4. **Board 0.4-1.2 m from the cameras**, in all three distance bins (0.4-0.6, 0.6-0.9, 0.9-1.3 m). Not beyond 1.3 m:
   phantom12's far boards (1.4-2.0 m, 20 Oct) fit a rigid board 2-4x worse than near boards in the same image areas
   and read -25..+40 % per hold, cause unknown. Step 66 has no distance bin past 1.3 m, and in a fit sweep step 45's
   quality gate (item 7) would drop such holds anyway.
5. **The whole image, the four corners and the bottom included.** A cell the fit sweep never covers is fitted by
   extrapolation, and one the test sweep never covers is not validated (step 71 marks it). On phantom12 the corners and
   the bottom stayed unvalidated: the board stood on a stand and only the neck moved it across the image. The stand's
   three heights and the neck's reach (head-down limits yaw at the coupled stop; grid rows 18-34 deg off-axis) kept
   board centres in the middle band and at the sides; only the noisy 1.6 m boards were centred in a corner, and the
   bottom strip beyond 300 px came from one placement and never reached 8 holds (`handoff-lens-eval.md`
   §0, §4, on branch `feat/kaustabh-stereo-depth-eval`). There, two fits on nearly the same data differ by up to 6 % at 1 m. A hand-held board goes wherever the
   map is empty.
6. **Watch the live coverage map** (step 06) during each sweep; one run per sweep. A cell is done at `--target` holds
   in at least one distance bin, by default `lens_min_holds` + 4 = 12: step 66 judges an entry from 8 holds
   (`lens_min_holds`), but it counts holds seen by both cameras and places them around the fitted principal point,
   while the map counts the left camera's live view around the nominal one, so the live count needs a margin. Stop
   the sweep when the map says it is complete; the corner and bottom cells, never validated on phantom12, are worth
   a minute more. The person holding the board needs the live view on a phone (§4 item 8); the map is read to them.
7. **The quality gate: dropped from the fit, flagged in the test.** A hold whose corners fit a rigid board worse than
   0.6 px rms (`lens_hold_rms_max_px`; phantom12: near boards 0.30-0.42 px, the bad far boards 0.67-1.13 px) is a bad
   board in the fit sweep: step 45, the only step that drops holds, drops it, lists it and refits; more than 20 % of
   the holds dropped (`lens_bad_hold_max_frac`) rejects the recording (exit 2: record again). Step 66 drops no test
   hold: measured with the calibration being judged, a bad hold can be the calibration's fault, and dropping it would
   turn that cell's FAIL into "not validated" (21 Oct, the KB4 refit's bottom-right corner). It lists such holds per
   cell as flagged and judges them with the rest. Its NO RESULT has four causes, none of them a rejected test
   recording: a fit that did not converge (step 45's verdict in the file it judges), a scoring solve that did not
   converge (restarted once), a failed scorer self-check, or no cell x distance entry with 8 holds.

## 5. Validation: what "good" means

| check | gate | why it is not enough on its own |
|---|---|---|
| Kalibr reprojection σ | well under 0.5 px (phantom13: 0.27/0.23) | blind to extrinsics and to timing |
| step 20 coverage / reach | ≥ 60 % / ≥ 60° | says nothing about the fit |
| step 60 held-out row error | median < 0.5 px | blind to scale and to a disparity offset |
| step 65 scale | \|error\| < 0.5 % | one number over the held-out distances only |
| **scale against distance** | flat within ±1 % from 0.4 to 2 m | none: this is the check that decides |
| **step 66: depth by image cell × distance** (KB4T, §4b) | every entry with ≥ 8 holds within ±0.5 % (corner cells ±1.0 %) | entries with fewer holds are not judged: read the report's "not validated" cells; holds over the 0.6 px gate are flagged and judged, not dropped |

Run the distance check on a **common set of unused pairs** (pairs not in any candidate's fit),
so candidates are compared on identical data: `depth_scale2.py <camchain> 4 <out.csv>
<export dirs>` with `USED_TREE` set to the union tree. A distance-dependent error that grows
with distance means a disparity offset (a tiny relative yaw between the cameras), and appears
when the fit is pulled by data the model cannot fit (§8.2).

## 6. Camera–IMU: how it was done (phantom12, 2026-10-15)

- **Board** fixed on a wall or stand, well lit, **1.0–1.2 m** from the head, and **the whole
  board inside the central zone** of the live view (every corner within 450 px of the image
  centre: roughly the middle 70 % of the width, away from the corners). The edge filter drops
  any image with a corner outside; a centred board at 0.8 m already reaches ~350 px, leaving
  ~10° of swing, against 17–21° at 1.0–1.2 m (estimate, f = 563 px).
- **Exposure** 1500 µs / ISO 1600 under evening indoor light; check brightness on the live view
  (`cam-exposure.sh`, §4.3).
- **Motion**, 180 s from when the record log shows frame counts: 10 s still, then nods (pitch),
  turns (yaw), side tilts (roll: push one shoulder down, then the other), push/pull, side to
  side, up/down, mixed, 10 s still. **Quick, not wide:** ±10–15°, 1–2 back-and-forths per
  second, no pauses; short sharp pushes of 10–15 cm. phantom12 hangs on a gantry and is moved by
  the torso; never force a powered, holding neck. Slow motion (take A: 5–9 °/s rms) is too weak.
- **Quick check before leaving the robot:** `bash /root/qc-take.sh <take>` (export + check,
  ~4 min; `results/.../imucam/qc_imucam.py`). Heuristic targets: ≥ 17 °/s rms about each IMU axis,
  ≥ 0.5 m/s² std of |a|, ≥ 60 % of frames inside 450 px. It does not leave out the opening
  still, so a board outside the zone there is flagged, but costs nothing.
- **Processing** (`results/.../imucam/calib-run-imucam.sh`): export → `10_build_trees.py --mode
  imucam` → `filter_edges.py <tree> <out> 450` (`results/.../calib3/`; filters each camera on its
  own corners, §8.15) → step 40 imucam with `CAM=calib_stereo3noedge-camchain.yaml` (stereo pose
  held fixed) → step 70 with `config/okvis-template-phantom12.yaml` → step 80 `--offline
  --compare <second take's yaml>`. ~20–30 min per fit on 2 CPUs.
- **Acceptance:** Kalibr's normalized gyro/accel residuals average ~1 (else loosen the IMU noise
  file and refit: `config/imu-phantom12-neckpitch-loose.yaml`, §8.16); `|timeshift_cam_imu|` a few
  ms (step 70 warns above 10 ms); **the two cameras' timeshifts within 0.5 ms** (left/right
  sync); `|t_SC|` inside the rig gate 0.05–0.40 m (phantom12 ~0.16 m, phantom21 ~0.20 m: not
  "a few cm"); **a second take within ~5 mm, 0.3°, 1 ms**. The stereo pose is held fixed, so the
  two cameras' `T_SC` are consistent with the baseline by construction: that is not a check.
- **Template:** `okvis-template-phantom12.yaml` = repo `mk12_stereo_holevisor.yaml` with
  `dma_queue` 1, 2. It keeps `imu_name: "NeckPitch"` and phantom21's `T_BS` (head-design value),
  which phantom12's results put in doubt (§7).

## 6b. T_BS measurement (phantom12): powered neck 2026-10-17, ~10 mm off the config, not certified

**Update 21 Oct (refit with Fit A's KB4T lens, `results/phantom12-2026-10-20-handeye/FINDINGS.md`): camera 0 sits ~14 mm / 0.5° from the config (P1 13.91 mm / 0.480°, P2 14.15 mm / 0.510°), 1σ 2.8 / 2.5 mm, IMPROVE on both takes, takes agreeing (2.16 mm); the two cameras' own fits now 1.4 mm apart (5.2 with the old lens: the old lens caused it). Not good to the 3 mm / 0.3° gate: the statistical error (2.4-2.7 mm) did not fall. The old lens had pulled X ~5.2 mm along the optical axis, so the ~10 mm / 0.4° below is superseded. Deliverable `T_BC-fita.json` (camera in Head, Fit A's camera frame); `T_BS` is computed by step 83 against the config that carries Fit A's lens. The section below is the 17-18 Oct record.**

`T_BS` (the NeckPitch IMU in the Head link) is phantom21's on phantom12. Hand-held neck takes (16 Oct,
`results/phantom12-2026-10-16-tbs/`) could not decide: a hand loading the unpowered neck puts the head
0.3–0.5° from where the encoders say. With the neck held by its motors (17 Oct,
`results/phantom12-2026-10-17-tbs/FINDINGS.md`) the encoders agree with the head IMU's gravity to
0.09° per hold (after step 75's drops), and both takes put camera 0 **~10 mm (mostly Head y, −9 mm) and 0.4°** from where the
config says, agreeing with each other to 2.1 mm. X's error bar is ±3.5 mm (1σ), half of it a 5 mm
disagreement between the two cameras' own calibrations (the same in every half of both takes): the
config is ~3σ off, but the fit is not shown better than it at 2σ, so step 81 says INCONCLUSIVE and
step 83 writes nothing. Settling it: a check of the stereo calibration (then refit the takes
offline), or a camera-to-arm calibration. Also settled: encoder signs as the URDF; the NeckPitch IMU
is the head's; `R_BS` within ~2° of the config (gyro, 16 Oct).

- **Step 81, the decision:** the hand-eye fit `T_TB(q) · X · C = Z · B` over held poses measures
  camera 0 in the Head link, `X`. `--model pixels` (default, `handeye_px.py`) fits every board
  corner's pixels in both cameras (`--model poses`, per-image board poses, put the two cameras 12 mm
  apart on the same images). X's σ: the fit's statistical σ (clustered by hold) and, in quadrature, half
  the distance between each camera's fit alone on the holds both saw. KEEP if `X` ± 2σ is inside
  0.3° and 3 mm of the config's `T_BS · T_SC0`; REPLACE if X's 2σ is inside the gate; IMPROVE if
  the config is shown outside the gate at 2σ and in rotation and translation the fit is good to the
  gate or over 4σ from the config; otherwise INCONCLUSIVE, always with a reason (`verdict_reason`
  carries the arithmetic), as with fewer than 20 holds, a direction the data cannot see, a pixel rms
  over 1.5 px or more than 2 % of the corners cut, or no cross-check between the cameras. Step 83
  writes `T_BS = mean(X) · T_SC0⁻¹` only from REPLACE or IMPROVE reports of two takes step 75 passed
  that agree (within 0.1° / 1 mm, or within 2 combined statistical σ when each is within the gate and
  they are within the gate of each other), and its note gives each take's σ against the gate.
- **Step 82, the check:** the gyro against the joint rates gives `R_BS` with no camera: PASS / FAIL /
  INCONCLUSIVE at 1° with a within-take jackknife σ, and the angle between the neck axes from
  single-joint strokes faster than 20°/s (the 17 Oct nod, 16°/s, was too slow for it).
- **Step 75, at the robot:** holds graded over their own range (a 3 × 3 grid over it, yaw ≥ 15°,
  pitch ≥ 10°) and approach sides, encoder signs seen by the camera, sweeps, streams, the head IMU,
  each camera (a camera that stopped: RETAKE; frames missing only in recorder gaps: WARN), the board's
  spread in the torso frame, and the torso per hold; verdict GOOD / RETAKE / STOP (exit 0/1/2) with an
  action per finding; writes `tbs/<take>-qc.json`, whose `drop_holds` run-tbs.sh passes to step 81.

**Take** (`results/phantom12-2026-10-17-tbs/powered/README.md`; `OUT_HOST=/root/calib-work/takes`):
board fixed at ~1.1 m; the robot hanging on its gantry, motors on with `dma_cmd_client enable-motors`
(the walker stays OFF: it is not needed), the neck walked through a 5 × 5 grid by DMA.ethercat's
`dma_motor_motion` targets mode (`tbs-grid.sh`: yaw −30…30° × pitch −24…0°, the range in which the
board is seen; snake there and back, 49 holds of 3 s, 1.5 s ramps; optional single-joint sweeps);
record app 420 s started by the agent, which confirms both cameras and the IMU before the grid starts;
exposure auto. Run `bash /root/calib-work/run-tbs.sh <take> [<earlier take>]` (`CALIB_TOOLS` picks the
package copy).

## 7. Results

### phantom12, stereo, 2026-10-15 (`results/phantom12-2026-10-15/`)

Takes: night 14 Oct (`calib-stereo`, unusable: §8.1), and 15 Oct with the bottom camera off and
fixed exposure: `calib-stereo-2` (210 s), `-2b` (125 s), `-3-250` (250 s, near range and edges
with live view). Merged tree `stereo3`: 1350 calibration + 75 held-out pairs, coverage
95.0 / 95.6 %, reach 67.5 / 66.0°.

**Recommended: `calib3/calib_stereo3noedge-camchain.yaml`** (equidistant, all three takes,
pairs with any corner beyond 450 px left out: 1103 pairs).

| | left (queue 1, socket B) | right (queue 2, socket D) |
|---|---|---|
| focal fx / fy (px) | 562.91 / 561.68 | 565.36 / 563.77 |
| centre cx / cy (px) | 642.75 / 384.22 | 641.35 / 383.57 |
| distortion k1..k4 | −0.0124, 0.0174, 0.0116, −0.0252 | −0.0121, 0.0315, −0.0606, 0.0422 |
| reprojection σ (px) | 0.29 / 0.26 | 0.27 / 0.26 |

Baseline 44.83 mm (phantom21: 43.37). Step 60: 0.204 px. Step 65: +0.21 %. k3 and k4 differ
between the lenses because the outer ring that constrains them is left out.

Scale error (%) on the same 722 unused pairs:

| fit | 0.4 m | 0.6 m | 0.8 m | 1.05 m | 1.4 m | 2.1 m |
|---|---|---|---|---|---|---|
| calib 2: takes 2+2b, equidistant | +1.7 | +1.2 | +0.7 | +0.4 | −0.0 | −0.4 |
| calib 3: all takes, equidistant | +0.6 | +0.0 | −0.9 | −2.0 | −3.4 | −5.0 |
| calib 3r: all takes, radial-tangential | +0.7 | +0.6 | +1.0 | +0.9 | +0.7 | +0.7 |
| take 3 only, equidistant | +0.5 | −0.4 | −1.6 | −3.2 | −5.2 | −7.3 |
| all minus near (< 0.6 m) | +1.0 | +0.3 | −0.8 | −2.1 | −3.6 | −5.6 |
| **all minus edges (> 450 px)** | **+1.0** | **+1.0** | **+0.6** | **+0.4** | **−0.4** | **−1.0** |

Fallback if the full field matters (e.g. SLAM features at the border): `calib_stereo3r`
(radial-tangential): flat +0.7 % but a poorer lens fit (held-out reprojection 1.13 px against
0.59).

### phantom12, camera–IMU, 2026-10-15 (`results/phantom12-2026-10-15/imucam/`)

Takes `calib-imucam-{A,B,C}` (180 s each, 1500 µs / ISO 1600, board at ~1.1 m, robot swung by
hand on its gantry): A too gentle (5–9 °/s rms), B and C good (24–37 °/s, 1.5–2.0 m/s² std of
|a|). Left/right exposure delay at this exposure +0.04 ± 0.02 ms (`sync_diag.py`, 384 and 374
pairs).

**Result: `imucam/phantom12_stereo.yaml`**, step 70 on fit `calib_imucam-B-loose` (take B,
edges filtered, `imu-phantom12-neckpitch-loose.yaml`):

| | left (cam0) | right (cam1) |
|---|---|---|
| `t_SC` (mm, IMU frame) | (−67.2, +147.0, +15.9), \|t\| 162.4 | (−22.3, +147.3, +15.4), \|t\| 149.7 |
| timeshift_cam_imu | +5.61 ms | +5.60 ms |
| reprojection (px) | 0.29 | 0.26 |

Normalized residuals gyro 1.00, accel 1.22; `image_delay` −0.0056 s. Six fits of three takes
(A; B filtered, unfiltered and loose; C filtered and loose) agree within 1.6–7.8 mm, 0.07–0.31°
and 0.11 ms; B-loose vs C-loose: 4.4 mm, 0.08°, 0.02 ms. Full table: `imucam/README.md`.

**Open: `T_BS`.** Every fit puts phantom12's cameras ~20° and ~106 mm away from phantom21's
relative to the IMU. With phantom21's `T_BS` (the template's), step 80 check 4 passes on what it
judges (baseline 100 % lateral, 1.4 mm off the centreline, azimuth 0.2°) but the optical axes
look **30.1° down** in the Head frame (phantom21: ~10°). Either phantom12's camera mount or its
IMU mount differs from phantom21's; a camera–IMU fit cannot tell which. Depth and VIO do not use
`T_BS`; camera-in-Head for grasping does. Measure it (§9) before trusting `T_BC`.
**Measured 17 Oct (powered neck, §6b; superseded 21 Oct by the Fit A refit: ~14 mm / 0.5°, IMPROVE):** camera 0 sits ~10 mm (Head y −9 mm) and 0.4° from where
phantom21's `T_BS` puts it, ±3.5 mm (1σ, half of it the two cameras' 5 mm calibration disagreement):
not certified to 3 mm, not replaced (`results/phantom12-2026-10-17-tbs/FINDINGS.md`).

**Deployed 2026-10-16 16:12 PDT** (`imucam/deploy/README.md`): phantom12's OKVIS runs
`/etc/phantomos/okvis/phantom12_stereo_dense_thor.yaml`, the robot's own 9 Aug
`okvis_stereo_dense_thor.yaml` with only the calibration transplanted from `phantom12_stereo.yaml`
(`T_SC`, intrinsics, distortion, `dma_queue`, `image_delay`, `T_BS`, `dma.imu_name: NeckPitch`);
all its tuning kept, the old file untouched beside it, `robot.env` pointed at the new one
(backup `robot.env.bak-20261016-161136`). The pod logs `Resolved DMA IMU 'NeckPitch' -> slot 2`
and pairs cam 0 <- DMA 1, cam 1 <- DMA 2; SLAM latency p50 149–163 ms (old config 186–189).

## 8. Lessons learned

1. **Check that the cameras expose together before calibrating.** On phantom12 the right
   camera exposed 8.5 ms after the left (rectified row error against board vertical speed:
   slope −8.47 ms, r −0.89, 205 pairs) while the frame stamps agreed to 0.03 ms and DepthAI's
   own per-frame exposure timestamps said 0.02 ms. Cause, found by elimination with a capture
   tool (`calib2/sync_capture.py`): the **bottom camera on auto exposure** delays the right
   camera by 3.5–8.5 ms, whichever camera sends the sync pulse and whether or not the bottom
   camera is on the sync line; with it on fixed exposure (+0.3 ms) or removed (+0.3 ms, and
   +0.47 ms with left/right on auto) the pair is in sync. Fix applied: bottom camera removed
   from `head_camera.json`. DMA.video request: `results/.../dma-video-sync-request.md`.
2. **This lens's outer ring does not fit a 4-parameter model.** Edge frames (corners beyond
   ~46° off-axis) pull an equidistant fit into a distance-dependent scale error (−5 % at
   2.1 m); radial-tangential shifts the error into a flat +0.7 % and a worse lens fit. Dropping
   the edge frames removes the drift (§7). Close-range frames are not the cause (tested).
3. **Steps 60 and 65 can pass while depth is wrong at other distances.** Calibration 3 passed
   step 60 (0.25 px) and its error was −5 % at 2.1 m. Always run the scale-against-distance
   check, on frames no fit used, from 0.4 m to 2 m.
4. **Coverage of the edges needs the live view.** Two takes without it reached 53–57°; the one
   with it reached 66–68° and 95 %. Close range (0.4–0.7 m) also needs a deliberate segment.
5. **Kalibr on ARM64 (Thor):** `osrf/ros:noetic-desktop-full` is amd64-only; build on
   `ros:noetic-ros-base` with `catkin build -j2 -p1` (43 min), add `python3-opencv
   ros-noetic-cv-bridge`, and run with `LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1`
   (otherwise `import cv_bridge` before `cv2` fails: "initialization of cv_bridge_boost raised
   unreported exception"). Pipe a focal length on stdin (`yes 568 |`) so Kalibr cannot block on
   its manual-focal prompt.
6. **`kalibr_bagcreater` read every image twice** when the tree had a `heldout2/cam*` folder:
   fixed in step 40 (commit "step 40 bags each image once", with a test).
7. **The viewer and RTSP streamer freeze after a producer restart:** they keep the deleted
   shared-memory files mapped and show old frames as live. OKVIS and DMA.streams re-attach.
8. **Verify what is actually running before citing code.** phantom12 ran `dma-video:main` whose
   producer was byte-identical to DMA.video commit 6e5d172 (Feb), not `main`'s tip; later a
   `fix-SOF-1206-canonical-8b-meta` image with the same producer file. Compare the file inside
   the container with `git hash-object`.
9. **The development workstation is a 14 GB laptop already ~9.5 GB used:** a Kalibr build with
   `-j$(nproc)` (24 compiles) OOM-killed the editor twice. Heavy work belongs on Thor.
10. **Shared-deployment edits are blocked for agents** (correctly). For experiments with camera
    settings, scale the producer to 0, drive the cameras with a standalone tool built from
    DMA.video's own functions (`sync_capture.py`), and scale it back to 1: no config changes.
    On 2026-10-15 the auto-mode permission check also blocked the exposure change for a
    calibration take; the user switched the session to manual mode and approved each command.
11. **Recording:** the OKVIS record app gets ~12–13 fps with 4 CPUs (PNG encoding); the capture
    tool saved the full 15 fps. Frames dropped by the record app are whole frames, never torn.
12. **Shell traps that cost time:** zsh treats `$var:h`/`$var:s` as modifiers (write
    `${var}:path`); an apostrophe inside a single-quoted ssh heredoc ends the quoting (write the
    script locally and `scp` it); `pkill -f <name>` over ssh kills its own shell (kill by PID).
13. **phantom12's OKVIS ran a config that was not its calibration (found 2026-10-15, fixed
    2026-10-16).** `okvis_stereo_dense_thor.yaml` described another camera pair (85.75 mm baseline,
    f ≈ 585 px), had identity `T_BS` (OKVIS's front = IMU +x, 90° off the cameras) and no IMU name,
    so OKVIS fused the head cameras with slot 0 = **RightFoot**. Stereo depth from it was scaled by
    ~1.9×. The handoff said "tell the deployer"; the note never went out, and a day later someone
    else reported the stale file and read the 21° between it and our result as the cameras being
    re-angled. Fixed by the transplant in §7. Lesson: a result that is not deployed and not
    announced is invisible; whoever runs the robot learns of it only by tripping over the old file.
14. **Kalibr's "Orientation prior camera-imu found as: (T_i_c)" prints the IMU-to-camera
    rotation**, despite its label. Read as camera-in-IMU it suggested an upside-down camera and
    a 153° difference from phantom21 (2026-10-15, retracted within the hour). Use the final
    camchain's `T_cam_imu` (step 70 inverts it into `T_SC`), and sanity-check orientation against
    one image: which way is up, where tag 0 is.
15. **Step 10's imucam mode keeps every frame of each camera, unpaired** (take A: 2509 left,
    2520 right). The pairwise edge filter refused such trees; `filter_edges.py` now filters each
    camera on its own corners when the tree has `imu0.csv` (commit "tools/calibration:
    camera-IMU calibration of phantom12").
16. **Brisk motion inflates Kalibr's accelerometer residuals without biasing the result.**
    Hand-swinging a hanging robot (jolts up to 57 m/s²) gave normalized accel residuals of ~3;
    gentle take A gave 0.5 and a `T_SC` within 1.6 mm of B's. Loosening the noise model (accel
    ×2.5, gyro ×1.4) brought them to ~1 and moved the result 0.5 mm / 0.08°.
17. **The first camera–IMU take is usually too gentle.** Take A (slow, wide-ish moves, 30 s
    still while the message to start reached the operator) reached 5–9 °/s rms. Asking for
    1–2 back-and-forths per second gave 24–37 °/s. Check every take with `qc-take.sh` before
    leaving the robot, and let the operator see the record log start.
18. **The robot's shared folders are not ours to keep.** On 2026-10-16 at 14:59 someone else
    emptied `/data/okvis-recordings` and pruned every Docker image on phantom12: all stereo and
    camera–IMU takes, exports and trees went, a day after the fits. Nothing needed was lost because
    results were already in git and on the workstation. Bring results home the same day, keep
    takes under `/root/calib-work/takes` (pass `OUT_HOST=` to `05_record.sh`, whose default is
    the shared folder), and expect to rebuild images.
19. **Deploy a calibration by transplant, and check it in the startup log.** A step-70 file is
    built on a template robot's tuning; swapping it in whole also changes the deployed estimator
    (on phantom12: `realtime_time_limit` 0.035 → 0.015 s, final BA off, online extrinsics on,
    `a_max` 176 → 78). Replace only the calibration keys in the robot's own file (`imucam/deploy/
    transplant.py`), compare the two with OpenCV FileStorage, and after the restart read the pod's
    `[entrypoint] config:` hash, the camera pairing and `Resolved DMA IMU '<name>' -> slot N`. A
    missing `dma.imu_name` silently means slot 0, which is a foot on these robots. And before
    calling a camera "re-angled", check that the old file's baseline is this pair's: different
    camera pairs and different heads also disagree by tens of degrees.

## 9. Ideas and extensions

- **Record with DMA.streams instead of the OKVIS record app**, once it has the changes in the
  DMA.streams request (`~/calib/phantom12/dma-streams-request.md`; memory
  `dma-streams-recorder-gaps`): a lossless mode for chosen cameras at the full frame rate, the
  capture stamp written unchanged (today +29 ms), a sequence-checked copy so no torn frames,
  sessions kept outside the 7-day cleanup. Then no extra pod is needed and every take has the
  whole robot's telemetry beside it. Needed on our side: a converter from its MCAP to the
  step-10 layout, and the IMU taken from its `/raw_imu` (every IMU at 500 Hz).
- **Or a lossless recorder inside DMA.video itself**, which owns the frames: fewest copies,
  and it can stamp mid-exposure. It would also need to fill `exposure_time_ms` / `gain_db`
  (present in `VideoFrameMeta`, written as 0.0 today) so exposure mismatches are visible.
- **Promote the diagnostics into the package:** a sync step (`sync_diag.py` with a 0.5 ms gate),
  an edge filter as a step-10 option (`--max-radius-px`), and the scale-against-distance check
  as a numbered step with a ±1 % gate.
- **Package the Thor recipes:** a Dockerfile for Kalibr on ARM64 with the fixes in §8.5, and one
  runner script that takes a take name and runs export → 10 → 20 → 40 → 60 → 65 → distance.
- **A lens model that fits the whole field** (e.g. equidistant with tangential terms, double
  sphere, EUCM) would let the edges be used. Kalibr fits some of these; OKVIS would need to
  support the chosen one, including the depth-mode rectification.
- **Hand-eye / `T_BS` for phantom12 (needed: §7). Decided 2026-10-16: test before fixing;
  T_BS tests planned (`docs/superpowers/plans/2026-10-16-phantom12-tbs-tests.md` on branch `feat/kaustabh-stereo-depth-eval`). Revised after
  the plan's review the same evening (`handoff-tbs.md` §8 on that branch; protocol §6b; outcome 16 Oct: hand-held cannot reach 0.3°, the per-hold head-vs-encoder error is ~0.4°, powered neck next; the rest of this bullet is the superseded first plan): test 1 becomes the hand-eye fit itself
  (its spread was ~4× less sensitive than its gate implied), the decision gate is 0.3° and 3 mm on
  the fitted camera-in-Head against the config (~5 mm at the gripper at 0.85 m), two takes, poses
  approached from both sides; test 2 stays as a ~1° independent rotation check.** One neck take (board fixed, torso braced still, neck moved by hand through ~15 held
  pitch/yaw poses plus a pitch-only and a yaw-only sweep; the record app logs the joint angles).
  Test 1: the board, carried into the torso frame through `T_TB(q) · T_BS · T_SC`, must land in one
  place for every pose. Test 2: the gyro's reading of the sweeps must match the joint axes through
  `R_SB` (step 80's check 5 idea, rotation only). Only if they fail: the full fit, robot-world
  AX = ZB (X = camera in Head, Z = board in torso) by reprojection least squares, which gives
  `T_BC` and `T_BS = T_BC · T_SC⁻¹`. Encoder zeros are unobservable there (a pitch zero is absorbed
  into X, a yaw zero into Z); the neck geometry is observable. **The gyro cannot see an encoder's
  sign:** a reversed pitch encoder fits exactly with `R_SB` turned 180° about Head z (yaw: Head y;
  both: Head x), so test 2 names the joint and a hand check decides. For grasping, the strongest
  version is camera to **arm** (the gripper holds the board), which absorbs errors in the neck and
  torso kinematics. Background: the explainer's Part VI (§10).
- **Re-check sync on auto exposure in dim light:** the auto-exposure check (+0.47 ms) was 75 s in
  daylight. Fixed 1500 µs in evening light measured +0.04 ms (2026-10-15, camera–IMU takes).

## 10. Where things are (2026-10-16, afternoon)

- **T_BS (added 18 Oct):** results `results/phantom12-2026-10-16-tbs/` (by hand) and
  `results/phantom12-2026-10-17-tbs/` (powered); on phantom12 the takes in `/root/calib-work/takes/tbs-neck-*`,
  reports in `/root/calib-work/tbs/`, the branch's package in `/root/calib-tools-ref`.

- **Branch** `feat/*-stereo-depth-eval`: this folder; results in `results/phantom12-2026-10-15/`
  (night attempt, `calib2/` sync tests, `calib3/` the stereo result and all candidate camchains,
  `imucam/` the camera–IMU result, fits and tools); `config/okvis-template-phantom12.yaml`,
  `config/imu-phantom12-neckpitch-loose.yaml`.
- **phantom12** `/data/okvis-recordings/` was emptied by someone else on 2026-10-16 (§8.18): the
  takes, exports and trees are gone; disk 51 %. `/root/calib-work/`: config, scripts, camchains,
  Kalibr bags, reports and logs, the final `phantom12_stereo.yaml`, a copy of dma-recorder
  segments (`dmarec-imucam-2026-10-15/`: 15 Oct 19:37–20:05 UTC, 7 h *before* the camera–IMU takes
  of 16 Oct 02:38–03:03 UTC; the takes' own segments aged out of `/root/recordings`, which keeps
  ~60 h, checked 19 Oct), and `takes/` for new takes. `/root/head_camera.json.with-bottom` restores the bottom
  camera; `/root/head_camera.json.auto-noBottom` is the current auto config;
  `/root/cam-exposure.sh`, `/root/qc-take.sh`. `/etc/phantomos/okvis/`: OKVIS's calibration
  `phantom12_stereo_dense_thor.yaml` (deployed 2026-10-16), the old `okvis_stereo_dense_thor.yaml`
  untouched, `robot.env` and its backup `robot.env.bak-20261016-161136`.
- **Workstation** `~/calib/phantom12/`: `NOTES.md` (full run log), the 14 Oct recording, both
  requests, `phantom12-camera-delay.html`, `results/imucam-2026-10-15/` (every camera–IMU fit
  with Kalibr reports and logs), `imu-allan/` (the raw still IMU data and Allan scripts),
  `explainer/` (the explainer's sources and assembler).
- **Artifacts:** the camera-delay page https://claude.ai/artifact/QhJnyafncmqyo3Td6KmdKn (made
  before the cause was found: its "why" section is out of date); the calibration explainer
  https://claude.ai/artifact/HodUnMRCFHdwfre8kKFpbs (Part VI: `T_BS`, hand-eye, solvers, the two
  tests); the OKVIS2-X field guide
  https://claude.ai/artifact/WVC6ZGi4gz5ckP1BK8vE9R.

## 11. New robot: calibrating a head stereo pair (KB4T)

1. **Rig file.** Copy `config/rig.yaml` to `config/rig-<robot>.yaml` and fill it: `cameras.resolution`,
   `nominal_focal_px` (image width / field of view in rad for an equidistant lens), `nominal_baseline_m` (CAD or the
   camera's spec; the fit refines it), `kalibr[]` (record dir, DMA queue, physical side per camera; step 15 reads each
   camera from the record app's channel `/cam<dma_queue>/image`), `okvis_order`, the target file, `gates.lens_*`
   (keep the defaults). Check it: `python3 scripts/rigcfg.py config/rig-<robot>.yaml`.
2. **Board measured** with a steel rule (README "Before the session" item 1), `target_measured: true`: a 1 % size
   error is a 1 % depth error everywhere.
3. **Cameras in sync** (§4 item 1): `sync_diag.py`, under 0.5 ms; only the cameras needed, or all on fixed exposure.
4. **Fixed short exposure** on both cameras for the session (§4b item 2), restored afterwards.
5. **Fit sweep, then test sweep**, each with its own run of the coverage map (step 06) until the map calls every cell
   done (by default 12 holds in one distance bin: `gates.lens_min_holds` + 4, §4b item 6).
6. **Corners:** step 15 on each recording (`--still-px 3`).
7. **Fit:** step 45 on the fit sweep. Exit 2: unconverged, or more than 20 % of the holds over the 0.6 px quality gate
   (record again); exit 3: the rig file's left/right is wrong.
8. **Uncertainty:** step 46. "Unconstrained" cells: record more there, or mask them in the depth image.
9. **Validate:** step 66 on the test sweep. Only a PASS goes on. NO RESULT (exit 2) has four causes, named in the
   report's verdict: step 45's fit did not converge (back to item 7); a scoring solve did not converge, or the scorer
   self-check failed (no depth number of that run is trusted); or no cell x distance entry reached 8 holds (sweep
   again). Step 66 drops no test hold; read the flagged holds per cell (board or calibration bad there) and the "not
   validated" cells (step 71 lists them, for masking).
10. **Export:** step 71: the KB4T JSON and a Kalibr camchain (k1..k4 only). With `--okvis-template` it writes an OKVIS
    config only when `--allow-4term-okvis` is also given, and labels it NOT FOR STEREO DEPTH: dropping p1, p2 is not
    an approximation for depth (phantom12's Fit A at 1 m: 0.58 % in the centre, 4.5 % within 350 px, 9.3 % within
    450 px, 25 % to 560 px, the depth image's extent). Step 71 prints that cost for the calibration it exports.
    **For OKVIS stereo depth:** `--okvis-template <config> --okvis-kb4t` writes the whole lens, `distortion_type: kb4t`
    with `distortion_coefficients: [k1, k2, k3, k4, p1, p2]`: the stereo network's rectification (depth) uses all six,
    VIO k1..k4 (OKVIS2-X #796, in AI-1184 since fc255aaa). It needs an OKVIS image with
    that change; an older image refuses the file. Deploy check: the OKVIS log's `KB4T rectification (stereo network)`
    line at startup. `--okvis-kb4t` with `--allow-4term-okvis` is refused.
11. **Camera-IMU:** Kalibr (step 40 imucam) with the exported camchain fixed (the 4-term lens costs pixels there:
    Fit A camera 0 0.90 px within 350 px, 4.2 px in the far corners), then step 70 and step 80 (§6). Step 70 copies
    the camchain's 4-term lens into its OKVIS config: that config carries the camera-IMU result (`T_SC`, timing)
    with a lens that is not for stereo depth either; for depth, write the lens with step 71 `--okvis-kb4t` into it.

The heavy steps run on Thor in capped containers (§3). Record the result in §1 and §7 with dates and sources.
