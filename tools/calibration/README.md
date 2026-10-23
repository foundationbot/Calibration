# phantom12 with an AprilGrid — read this first

This package came from phantom13 (mk13 head, checkerboard target). In OKVIS2-X it
also calibrates **phantom12** against a Kalibr **aprilgrid**, with these changes
(see git log for each):

| change | why |
|---|---|
| **No camera clock rebase** (step 10) | DMA.video already stamps capture time on the host clock (publish − capture = 29–30 ms on phantom12). The old rebase shifted frames ~24–29 ms late; Kalibr reported it as `timeshift_cam_imu` (phantom13: −23.6 / −24.6 ms) and step 70 baked it into `image_delay: 0.0241`, which current OKVIS (no live rebase since 2026-09-16) would apply as a real error. Step 70 now warns when \|timeshift\| > 10 ms. |
| **Aprilgrid targets** (rigcfg, steps 20, 30, 40, 60, 70) | Tag ids anchor every corner: no 180° ambiguity, so step 30 is a no-op and the camera–IMU fit uses step 10's tree directly; partial views count toward coverage. Corners are re-found at Kalibr's X-junctions (OpenCV's ArUco quads alone are ~2 px off on a Kalibr board). |
| **Step 65 scale check** (new) | Step 60's row error is blind to a wrong baseline. Step 65 triangulates the board with the stereo calibration alone and compares tag spacing to the steel-rule pitch. |
| **phantom12 config** | `config/rig-phantom12.yaml`, `config/target-aprilgrid.yaml` (phantom21's 7×5 board, pitch measured 0.120 m on 2026-10-14), `config/imu-phantom12-neckpitch.yaml` (Allan deviation of 2.35 h of still NeckPitch data, ×10). No borrowed `T_BS`: step 70 takes phantom12's from its template. |

### Before the session
1. **Board: done 2026-10-14.** Steel rule, left edge of the first tag's black square to the left edge of the last one's: 720 mm along a 7-tag row (6 pitches), 480 mm down a 5-tag column (4 pitches), so pitch 0.120 m; `target_measured: true`. For a different board, measure the same way and put `tagSize`, `tagSpacing` (= gap / tagSize) and `tagCols` (tags along the direction ids 0,1,2… run) in the target file.
2. **Re-read NeckPitch's IMU index** on phantom12 (it was 2 on 2026-10-14; the roster is dynamic).
3. **Fixed short exposure** for the session (`/etc/phantom/head_camera.json`, `"exposure": <µs>, "iso": <n>` for left and right; producer restart = scale 0 → 1), restored afterwards. 15 min camera warm-up.
4. **Kalibr image** built once on the workstation (step 40).

### Record (on phantom12) and export (workstation)
The current record app writes `session.mcap`; `tools/mcap_to_euroc.py` (repo root) turns it into the per-frame layout step 10 reads, bit-identical PNGs plus `timestamps_full.csv`.

```bash
# robot, as root: phantom12 caches okvis2x:thor-0594dfc5... (AI-1184 line); never :latest
IMAGE=foundationbot/okvis2x:thor-0594dfc58c471db44233526bf525965cb1bb10b5 \
  CAMERAS=1,2 IMU_INDEX=2 DURATION=180 OUT_SUBDIR=calib-stereo  scripts/05_record.sh
IMAGE=... CAMERAS=1,2 IMU_INDEX=2 DURATION=150 OUT_SUBDIR=calib-imucam scripts/05_record.sh
IMAGE=... CAMERAS=1,2 IMU_INDEX=2 DURATION=60  OUT_SUBDIR=calib-check  scripts/05_record.sh  # still board, 0.4-1.2 m
# workstation, per take
python3 tools/mcap_to_euroc.py <take>/session.mcap <take>-euroc/
```
The record pod is capped at 2 CPUs; lossless PNG for two cameras may drop below 15 fps. The record log prints per-camera frame counts every 5 s.

### Run order (phantom12)
```bash
export CALIB_RIG=$PWD/config/rig-phantom12.yaml
python3 scripts/10_build_trees.py --src <stereo-euroc> --dst tree/stereo --mode stereo
python3 scripts/20_check_coverage.py tree/stereo                       # gate
# docker: RIG=/data/config/rig-phantom12.yaml bash /data/scripts/40_calibrate.sh stereo
python3 scripts/60_validate_stereo.py calib_stereo-camchain.yaml --heldout tree/stereo/heldout2   # gate
python3 scripts/65_scale_check.py calib_stereo-camchain.yaml --tree <check-take tree> --pitch-m <measured>  # gate
python3 scripts/10_build_trees.py --src <imucam-euroc> --dst tree/imucam --mode imucam
# step 30 is not needed for an aprilgrid
# docker: RIG=... CAM=/data/calib_stereo-camchain.yaml bash /data/scripts/40_calibrate.sh imucam
python3 scripts/70_make_mk13_yaml.py <phantom12 template> calib_imucam-camchain-imucam.yaml phantom12_stereo.new.yaml --tree tree/imucam
python3 scripts/80_validate_extrinsics.py phantom12_stereo.new.yaml --offline
```
Camera–IMU acceptance: Kalibr's normalized gyro/accel residuals average ~1 (else rescale the IMU noise file and refit); `|timeshift_cam_imu|` a few ms.

### KB4T lens calibration (hand-held; steps 06, 15, 45, 46, 66, 71)
The stereo lens calibration that replaced Kalibr's 4-term lens on phantom12 (Fit A, 20 Oct): KB4T = equidistant
k1..k4 plus tangential p1, p2 for both cameras, and the stereo pose. Why each item matters: `calibration.md` §4b; a
new robot: §11. The heavy steps (15, 45, 46, 66) run on Thor in capped `calib-py:3` containers (§3). Every step takes
`--rig <rig file>` (or `CALIB_RIG`); the hold counts and the 1.8 px, 1.3 m and 20 % limits below are `gates.lens_*` values of
the rig file (defaults shown).
1. Rig file: `cameras.nominal_baseline_m` and `gates.lens_*` set; the board measured (`target_measured: true`).
2. Fixed short exposure on both cameras for the session; the live view on a phone for whoever holds the board.
3. **Fit sweep:** `06_coverage_live.py <dir>/coverage-fit` running, then `05_record.sh`. Hand-held board, 0.4-1.2 m,
   slow moves, 2-3 s pauses, until the map says every cell (corners and bottom too) is done: `--target` holds in at
   least one distance bin, by default `lens_min_holds` + 4 = 12. The margin over the 8 holds step 66 needs: step 66
   counts holds seen by both cameras, placed around the fitted principal point; the map counts the left camera's live
   view around the nominal one.
4. **Test sweep:** the same again (`<dir>/coverage-test`), a separate recording that no fit sees.
5. `15_dump_corners.py <session.mcap> <corners.npz> --still-px 3 --rig <rig>` for each recording (`PYTHONPATH` = this
   folder's `scripts/` + the repo's `interface/`).
6. `45_fit_lens.py fit-corners.npz fit.json` (exit 0; 2 = unconverged, or the recording rejected; 3 = the rig file's
   left/right is wrong), then `46_lens_uncertainty.py fit-corners.npz fit.json unc.json`. Step 45 is the only step
   that drops holds: fit-sweep holds whose board sits beyond 1.3 m or whose corners fit a rigid board worse than
   1.8 px rms are dropped, listed and the fit repeated; more than 20 % dropped rejects the recording (record again).
7. `66_validate_lens.py fit.json test-corners.npz report/ --fit-report fit.json --uncertainty unc.json`: PASS (exit 0)
   is required; 1 = FAIL; 2 = NO RESULT, which has four causes: step 45's fit did not converge (its verdict in
   fit.json), a scoring solve did not converge (restarted once), the scorer self-check failed, or no cell x distance
   entry has 8 holds. It drops no test hold: holds over 1.8 px are flagged per cell and still judged. Read the
   report's "not validated", "unconstrained" and flagged cells.
8. `71_export_lens.py fit.json report/report.json export/`: the KB4T JSON and `export/camchain-kb4t-equi4.yaml`
   (Kalibr, k1..k4 only), then Kalibr camera-IMU (step 40 imucam) with that camchain fixed, step 70, step 80. Dropping
   p1, p2 costs the camera-IMU step pixels (Fit A camera 0: 0.90 px within 350 px, 4.2 px in the far corners) but
   stereo depth far more (Fit A at 1 m: 0.58 % in the centre, 4.5 % within 350 px, 9.3 % within 450 px, 25 % to
   560 px): neither step 71's 4-term OKVIS config (written only with `--okvis-template <config> --allow-4term-okvis`,
   and labelled NOT FOR STEREO DEPTH) nor step 70's output (it copies the camchain's 4-term lens) is for OKVIS stereo
   depth. Step 71 prints both costs for the calibration it exports. **The export for OKVIS stereo depth** is
   `--okvis-template <config> --okvis-kb4t`: `distortion_type: kb4t` with all six coefficients (depth: the whole KB4T
   lens; VIO: k1..k4), for an OKVIS image with the kb4t change
   (OKVIS2-X #796, in AI-1184 since fc255aaa; older images refuse the file; deploy
   check: the OKVIS log's `KB4T rectification (stereo network)` line).

### Not imported from the original package
`scripts/extra/` (MuJoCo viewers), `reference/robot-originals/` (provenance copies incl. `monitor_imu.py`), the Kalibr report PDFs, `reference/2026-08-22/logs/` (incl. the step-30 regression lists `lab_cam*.npy`). The sections below that mention them describe the phantom13 package as received.

### Tests
`tests/` run with `python3 -m pytest tools/calibration` and need numpy, scipy, opencv (with aruco) and pyyaml (plus mcap and mcap-protobuf-support for the T_BS tests that write an MCAP); without them they skip, as in CI's pytest-only job `python-tests`. CI's `calibration-tests` job installs them and runs this folder for real. It is NOT one of `main`'s required status checks (those are `build-thor`, `build-amd64`, `clang-format`, `build-models`, `base-thor`, `base-amd64`; `devel` requires none; read 21 Oct), so a red `calibration-tests` does not block a merge: look at it before merging a change here. They include rendered fisheye stereo images of a Kalibr-style board (steps 60 and 65 end to end) and synthetic KB4T recordings whose truth is phantom12's Fit A (steps 45, 46, 66 and 71 end to end; `tests/lens_synth.py`). Locally, with 2 BLAS threads: 306 tests in 266 s (21 Oct); 313 with the kb4t tests (21 Oct, run file by file: the whole folder in one process was OOM-killed on a laptop with ~1 GB free).

---

# START HERE — configure the rig before running anything

Every rig-specific input of this package lives in **`config/rig.yaml`**. No
script carries its own camera, board, IMU or robot constants any more. The
shipped file describes the mk13 head (phantom13's Thor, now hw-thor01). For any
other rig, copy it and edit the copy:

```bash
cp config/rig.yaml config/rig-myrobot.yaml      # edit it, then:
python3 scripts/rigcfg.py config/rig-myrobot.yaml   # validates it and prints what every step will derive
export CALIB_RIG=$PWD/config/rig-myrobot.yaml      # or pass --rig PATH to each step
# step 40 (docker): RIG=/data/config/rig-myrobot.yaml bash /data/scripts/40_calibrate.sh stereo
```

| key | what it sets | used by |
|---|---|---|
| `target` | the Kalibr target file. The corner grid and spacing are read **from it** | 20, 30, 31, 40, 60, 61 |
| `charuco` | ChArUco layout and dictionary (ID-anchored variants only) | 31, 61 |
| `cameras.model` | `pinhole-equi` or `pinhole-radtan`. Chooses `cv2.fisheye.*` vs plain `cv2.*`. 30/60/61/70 **refuse to run** if the camchain disagrees | 20, 30, 40, 60, 61, 70 |
| `cameras.resolution` | image size. Checked against the camchain and the images | 30, 60, 61, 70 |
| `cameras.nominal_focal_px` | pre-calibration focal for the coverage reach figure | 20 |
| `cameras.kalibr[]` | per camera, in Kalibr order: `record_dir`, `dma_queue` (step 15 reads the record app's MCAP channel `/cam<dma_queue>/image`), physical `side` | 05, 10, 15, 60, 70 |
| `cameras.okvis_order` | OKVIS block order (`[1, 0]` = the mk13 stereo-role swap) | 70 |
| `imu.*` | IMU name + index, Kalibr noise model, fallback `T_BS` file | 05, 40, 70, 80 |
| `robot.*` | ssh host, `display_imu.py` dir on the robot, the tuner tools dir, Head axes | 80 |
| `capture.*` | pair tolerance, calibration and held-out pair counts | 10 |
| `gates.*` | coverage, reach, epipolar, degenerate-disparity and `T_SC` thresholds | 20, 60, 70 |

Command-line flags still override the config where they existed (`--host`,
`--grid`, `--nominal-focal`, `--target`, `--heldout`; `TARGET`/`IMU`/`MODELS`
env vars in step 40).

**What the config is checked against, so a wrong value fails instead of shipping:**
- **`side`** is *measured*: step 60 compares the rectified disparity sign with
  the sides you declared and exits 3 if they disagree.
- **`okvis_order` + `dma_queue`**: step 70 refuses a template whose `dma_queue`
  values are not the queues the config routes to each OKVIS block. It also
  refuses a tree whose `SOURCE.json` was built with a different mapping.
- **`cameras.model` / `resolution`**: checked against every camchain they meet.
  Step 70 rewrites `distortion_type` from the camchain.

**Still fixed by design:** exactly **two** cameras (stereo-only). The target may
be a Kalibr **checkerboard** or **aprilgrid** (aprilgrid support added in OKVIS2-X;
with an aprilgrid step 30 is a no-op). The fallback `T_BS` is one
robot's IMU-in-Head transform: on another robot, keep `T_BS` in the template or
point `imu.T_BS_default_file` at that robot's value. `00_make_imu_stub.py` holds
the DMA `/raw_imu_actuals` ABI, not a rig setting. The reference numbers printed
beside results (68 % coverage, 0.290 px, 79 mm …) are the **mk13 2026-08-22
values**, shown for comparison only.

⚠ `robot.host` is `root@hw-thor01`. Before step 80's live checks, confirm
`robot.display_imu_dir` exists on that robot and that `imu.index` matches the
live roster.

---

# calib-phantom13 — stereo + camera-IMU calibration for the phantom13 / mk13 head

Everything needed to recalibrate the mk13 stereo head and emit a deployable
`mk13_stereo.yaml`: the Kalibr target and IMU configs, a numbered script per
stage, the known-good 2026-08-22 outputs to compare against, and the traps that
cost days the first time.

**If you are a new Claude session: read [For a new Claude session](#for-a-new-claude-session)
at the bottom first.** It tells you what is load-bearing and what is not.

---

## The one-minute version

```bash
# on the robot: record two datasets (see "Capture")
sudo IMU_INDEX=2 DURATION=90  OUT_SUBDIR=calib-stereo scripts/05_record.sh
sudo IMU_INDEX=2 DURATION=150 OUT_SUBDIR=calib-imucam scripts/05_record.sh

# on the workstation
python3 scripts/10_build_trees.py --src <run>/calib-stereo --dst tree/stereo --mode stereo
python3 scripts/20_check_coverage.py tree/stereo            # STOP if this fails
# docker: scripts/40_calibrate.sh stereo                    -> camchain-calib_stereo.yaml

python3 scripts/10_build_trees.py --src <run>/calib-imucam --dst tree/imucam --mode imucam
python3 scripts/30_label_canonical.py tree/imucam camchain-calib_stereo.yaml tree/imucam_canon
# docker: CAM=... scripts/40_calibrate.sh imucam            -> camchain-imucam-*.yaml

python3 scripts/60_validate_stereo.py camchain-calib_stereo.yaml   # held-out epipolar
python3 scripts/70_make_mk13_yaml.py <template> <camchain-imucam> mk13_stereo.new.yaml \
        --tree tree/imucam_canon
python3 scripts/80_validate_extrinsics.py mk13_stereo.new.yaml --host root@phantom13
```

**Acceptance:** held-out rectified epipolar `|dy| < 0.5 px`, `|T_SC| ≈ 17 cm`,
baseline ≈ 79 mm, `R_SC` vs gravity < ~3°. Anything else, do not deploy.

---

## The four traps

Every one of these was paid for in full. They are the reason this package exists
rather than a two-line "run Kalibr" note.

### 1. Reprojection error cannot see an extrinsics fault

On identical frames, a calibration **58× worse** in epipolar terms scored
**0.423 px vs 0.422 px** monocular RMS. `solvePnP` absorbs a focal error into the
estimated board distance, so the residual stays flat while the geometry rots.

> **Judge a stereo calibration by rectified epipolar `|dy|` on HELD-OUT frames.
> Never by reprojection RMS.** That is what step 60 is for.

### 2. Coverage, not frame count, decides distortion quality

Keeping the board visible to *both* cameras confines it to the overlap region and
leaves 28–32° of each fisheye field unobserved. The fit still "succeeds":

| | bad capture | good capture |
|---|---|---|
| frame coverage | 36% / 32% | **68%** |
| periphery reached | — | **63–65° off-axis** |
| distortion coeffs | ~30× too large, **sign-flipped between two identical lenses** | sane |
| Kalibr reproj sigma | 0.47 px — *looks fine* | 0.266 / 0.234 px |
| held-out epipolar | **2423 px** | **0.195 px** |

Sweep each camera's **full frame separately**, then add a short shared segment
for the baseline. Step 20 measures this and exits 1 if it is short — run it
**before** you spend an hour on a fit.

### 3. Checkerboard corner ordering is 180° ambiguous

A 180° rotation is a symmetry of any rectangular corner grid, so
`cv::findChessboardCorners` (which Kalibr's `checkerboard` target wraps) may
label the same board either way, and can **switch between frames**. No
`targetCols`/`targetRows` choice escapes it.

- **Stereo is immune** — each frame's target pose is solved independently.
- **Camera-IMU diverges** — it assumes one fixed target pose and chains camera
  motion against IMU integration. With ~35% of frames flipped:
  **`|T_SC| = 1.03 m` instead of ~17 cm**, 4–10 px residuals, std-32 whitened errors.

Step 30 fixes it. **Do not skip it.** An AprilGrid target would remove the
ambiguity at source (every tag is ID-anchored) — use one if you can get one.

### 4. `|T_SC|` and `T_BS` are different animals

`T_SC` (camera ← IMU) is what Kalibr solves. `T_BS` (IMU ← Head) is **not
observable from this pipeline at all** — nothing in the capture references the
Head link. See [T_BS](#t_bs--the-neckpitch-imu) below; getting this wrong hid a
31° yaw error through two full recalibrations.

---

## Camera identity

Two **independent** swaps live here. Conflating them is the single most
expensive mistake available in this system.

| swap | where it lives | can a camera-map edit touch it? |
|---|---|---|
| kalibr cam0/cam1 ↔ OKVIS cam0/cam1 | `mk13_stereo.yaml` **block order** | **No** |
| physical camera ↔ `dma_queue` number | `/etc/phantom/head_camera.json` `socket → queue_id` | Yes — this is the only axis it moves |

The chain this package implements:

```
record dir cam1 ──► kalibr cam0 ──► OKVIS cam1 ──► dma_queue 1 ──► socket D ──► physical RIGHT
record dir cam2 ──► kalibr cam1 ──► OKVIS cam0 ──► dma_queue 2 ──► socket B ──► physical LEFT
```

Step 10 does the first arrow, step 70 inverts it, so they compose to the
identity and **OKVIS cam0 is the physical LEFT eye**. `SOURCE.json` records the
mapping and step 70 refuses to run if it is not what it expects.

> **Verify by measurement, never by reading config.** De-roll each image by the
> frontend's half-angle, match, take the median horizontal offset. OKVIS cam0 =
> physical LEFT ⇒ `median(u_cam0 − u_cam1)` must be **positive**.
> Measured 2026-08-28: **+73.56 px, 99.8% positive, median `|dy|` 0.53 px**.
> Disparity scales with 1/Z, so compare the **sign and its consistency**, never
> the magnitude — the same test at Z ≈ 2 m in camchain ordering reads −22.40 px.

⚠ To re-label a camera in `head_camera.json`, move the **whole entry** (rename
the keys) so `queue_id`, `socket`, `intrinsics` and `extrinsics` travel together.
Swapping `socket` alone inverts `socket → queue`: OKVIS cam0 is then fed the
right eye while holding the left eye's `T_SC`, disparity sign inverts, stereo
triangulates behind the camera, and **the de-roll doubles to ~41.6° instead of
cancelling 20.79°** — with the startup banner still printing normally.
`/etc/phantom/*` is off limits to Claude; propose the edit, let the operator run it.

---

## The rig, as measured

### It is a 21° ROLL, not a splay

The calibrated inter-camera rotation is 21.34°, but **decompose the axis before
believing that number**: it is `[-0.029, +0.239, +0.971]` — 97% along the
optical axis.

| | |
|---|---|
| true vergence (angle between optical axes) | **5.11°** → ~96–98% image overlap |
| **in-plane ROLL about the optical axis** | **20.74°** |
| baseline | 43.0 → **79.0 mm** |

Corroborated independently: the board's in-plane angle measures **+78.8° in cam0
vs +97.6° in cam1**, an 18.8° difference, sd 0.02° over 25 pairs — a rigid
mechanical roll, not noise. A 20.7° *splay* would have given 84% overlap, which
the operator's ~98% observation rules out. Step 30 reproduces the same two
numbers from a completely different computation.

**Consequence:** SuperPoint has no orientation-normalisation step (it reads a
256-D descriptor straight off a CNN feature map, and ordinary convolutions are
not rotation-equivariant), so a fixed 20.74° inter-camera roll destroys
**inter-camera** matching while leaving temporal matching untouched. OKVIS
compensates in software by de-rolling each image by half the relative roll before
description. **The angle is derived from the live `T_SC` pair, never configured**,
so it cannot go stale against a recalibration — which is also why a hard-coded
constant would have had the OKVIS↔kalibr swap backwards.

> If the mount is ever physically corrected, the derived angle goes to zero, the
> de-roll becomes a no-op and the ~12–15% corner FOV it costs comes back.
> Fixing the mount is strictly better than calibrating around it.

### Lenses

`pinhole-equi` (equidistant fisheye), **129° H × 81° V** (152° diagonal),
1280×800. **Not `pinhole-radtan`** — an older version of the repo doc said
radtan and it is wrong.

---

## T_BS — the NeckPitch IMU

Full detail, with the superseded value and the seven verification measurements,
is in **`config/neckpitch_imu.yaml`**. The short version:

```
T_BS = [-0.03735085764393119,  0.839173404349257,   0.5425798658871718,  0.018,
        -0.9989078032051125,  -0.01610002806750228, -0.04386330803942975,-0.026,
        -0.02807337046382329, -0.543625594071052,   0.8388582117030844,  0.077,
         0, 0, 0, 1]
```

- **Do not expect to find this in a URDF.** Checked 2026-09-03: **no URDF in the
  workspace declares a NeckPitch IMU link** — every variant declares 16 IMU links
  and NeckPitch is not among them, while the live roster has 17. None declares a
  head camera link either. **If the URDF does not have it, use the value above
  verbatim.** That is the normal case, not the exception.
- The old header claimed T_BS was "derived by chaining the URDF-validated
  `head_T_cam` with the calibrated `T_SC`". That chain has no URDF end to anchor
  to, and the claim is how a **+31.31° yaw error survived two recalibrations**.
- **Gravity cannot see a yaw about the vertical**, so the camera-IMU fit never
  complained. It was caught by stereo geometry instead: a rigidly mounted pair
  must give a lateral baseline, left/right symmetry, no forward stagger and a
  forward optical axis. The bad value scored 86.3% lateral / +39.1 mm centre /
  **+39.9 mm forward stagger** / +30.2° azimuth. *A 40 mm front-to-back stagger
  between two cameras bolted into one rigid head is the tell — nothing physical
  can produce it.*
- Body frame B := the **Head** URDF link. Axes **x forward, y left, z up**.
  `T_BS` is `Body_T_Sensor`, row-major 4×4, translation in metres.
- Step 70 **preserves** an existing `T_BS` and **injects this default** if the
  template has none, printing a warning either way.

### Which IMU, and where it is on the wire

IMU **NeckPitch** (head-fixed, so rigid with the cameras), on
`/dev/shm/raw_imu_actuals`, **index 2** → `OKVIS_DMA_IMU_INDEX=2`, and
`05_record.sh IMU_INDEX=2`.

> ⚠ **Upstream `deployment/record.sh` defaults to `IMU_INDEX=0`, which is a FOOT
> IMU.** The packaged `scripts/05_record.sh` takes `imu.index` from
> `config/rig.yaml` instead, and refuses to run if it cannot read it. Passing it
> explicitly is still the safe habit.
>
> ⚠ **The index is not stable.** The roster is dynamic and has grown 5 → 17.
> NeckPitch has held index 2 by luck; Torso moved 3 → 9 and Pelvis 4 → 10.
> **Verify every session:** `python3 deployment/display_imu.py` (reads
> `imu_names[]` from `/dev/shm/config`). There is no `read_imu.py` on phantom13.

⚠ The OKVIS `imu_parameters` are **not** identical to the Kalibr ones in
`config/imu.yaml`: `sigma_g_c` and `sigma_gw_c` are deliberately inflated
(gyro noise 0.00264 vs 0.00164, gyro drift 10×). Kalibr wants the true noise
model to weight its fit; OKVIS wants headroom to stay robust on a walking robot.
**Do not "fix" one to match the other.**

---

## Capture

Record **two** datasets with `scripts/05_record.sh`, on the robot. It is
OKVIS2-X `deployment/record.sh` @ `0f32385b`, with one change: the `CAMERAS`
and `IMU_INDEX` defaults come from `config/rig.yaml`. Copy the whole package
to the robot so it can find the rig file, or pass `RIG=`. Reading the rig file
needs python3 + PyYAML on the robot host. Without them, pass both values
explicitly.

```bash
sudo IMU_INDEX=2 DURATION=90  OUT_SUBDIR=calib-stereo scripts/05_record.sh
sudo IMU_INDEX=2 DURATION=150 OUT_SUBDIR=calib-imucam scripts/05_record.sh
```

⚠ Its default recorder `IMAGE` is upstream's
`foundationbot/okvis2x:feat-mapping-integration-260710.1.0`. Check that the tag
still exists, or pass `IMAGE=`.

| dataset | how to move | why |
|---|---|---|
| **stereo** | target static; move the head slowly through the **full field of each camera separately** — all corners, near and far, in-plane and out-of-plane tilt. Then a short shared segment for the baseline. No blur. | intrinsics + distortion need periphery coverage (trap 2) |
| **camera-IMU** | target static and always in view; excite **all 6 axes** — rotate about x/y/z, translate along x/y/z. Moderately dynamic, but no blur and never lose the target. | the IMU leg is only observable under real excitation |

Good even lighting; low exposure time helps. Recordings land in
`/data/okvis-recordings/<subdir>/`. Copy to the workstation and run step 10.

Notes:
- **Kill the okvis2x SLAM pod first** or the two contend on `/dev/shm` + GPU.
- **OKVIS/recorder needs EtherCAT queues up.** Check the QUEUES exist
  (`raw_imu_actuals`, `actuals`, `config`), not the systemd unit state — the unit
  can read `active` before it has published them, and the recorder then exits.
- For a **stereo-only** capture on a box with no `dma-ethercat`, the recorder
  still hard-exits if it cannot open the IMU queue. `scripts/00_make_imu_stub.py`
  creates an empty, correctly-shaped ring so it can start. Never use a stub run
  for camera-IMU calibration.
- `monitor_imu.py` (in `reference/robot-originals/`) tails the IMU csv during a
  capture so you can confirm you are actually exciting all six axes.

---

## Scripts

Run them in numeric order. Steps **20** and **30** are the ones people skip and
then lose a day to. The KB4T lens steps (06, 15, 45, 46, 66, 71) are a chain of their
own, which hands over to 40 imucam, 70 and 80: "KB4T lens calibration" at the top.

| | script | where | what |
|---|---|---|---|
| 00 | `00_make_imu_stub.py` | robot | empty `/raw_imu_actuals` ring so the recorder can start without EtherCAT (**stereo-only**) |
| 05 | `05_record.sh` | robot | launches the `okvis2x-record` pod; cameras + IMU index from `config/rig.yaml` |
| 06 | `06_coverage_live.py` | robot (calib-py:3, `--network host`) | live coverage map for the hand-held KB4T sweeps: still holds per image cell x distance from the left camera's RTSP stream; a text map every few s and `coverage.png`; a cell is done at `--target` holds in one distance bin (default `gates.lens_min_holds` + 4 = 12, a margin over step 66's 8); one run per sweep |
| 10 | `10_build_trees.py` | either | `record.sh` run → Kalibr tree. Camera clock check (no rebase on host-clock stamps), pairing, held-out reservation, greyscale |
| 15 | `15_dump_corners.py` | robot (calib-py:3) | stereo corners of still pairs straight from a record-app `session.mcap` (npz), cameras from the rig file. A hold = one still stretch. Far boards' edge tags flicker and break the 1 px stillness; hand-held sweeps use `--still-px 3` |
| 20 | `20_check_coverage.py` | workstation | **gate.** detection rate, grid coverage, radial reach, heatmap PNG. Exits 1 if short |
| 30 | `30_label_canonical.py` | workstation | **gate.** gyro-vs-board flip labelling; writes the `_canon` tree |
| 31 | `31_label_canonical_charuco.py` | workstation | the ChArUco-anchored alternative. **Kept 0/2250 on real data** — close range only |
| 40 | `40_calibrate.sh` | docker | `kalibr_bagcreater` + both fits, with sanity checks printed |
| 45 | `45_fit_lens.py` | robot (calib-py:3) | **KB4T stereo lens fit** from the rig file's nominal camera: converged or exit 2; the only step that drops holds: fit-sweep holds above 1.8 px rigid-board rms or beyond 1.3 m dropped and listed, more than 20 % rejects the recording (exit 2); camera sides against the rig file (exit 3) |
| 46 | `46_lens_uncertainty.py` | robot (calib-py:3) | split-half refits: per-number sigma (a one-sample estimate) and the repeatability map at 1 m; cells beyond 0.5 % flagged "unconstrained" |
| 60 | `60_validate_stereo.py` | workstation | **the acceptance test.** held-out epipolar `|dy|` over all 88 corners (aprilgrid: corners paired by tag id) |
| 65 | `65_scale_check.py` | workstation | **gate (aprilgrid).** stereo-triangulated tag pitch vs the steel-rule pitch: catches a wrong baseline / board size |
| 61 | `61_validate_stereo_charuco.py` | workstation | ID-anchored variant; fewer pairs, no flip handling needed |
| 66 | `66_validate_lens.py` | robot (calib-py:3) | **the KB4T acceptance test.** depth error per image cell x distance on the test sweep; scorer self-check first; fixed rule (+-0.5 %, corners +-1.0 %, entries with 8 holds); drops no hold: test holds above 1.8 px are flagged per cell and still judged; `report.md` + `report.json`; PASS / FAIL / NO RESULT = exit 0 / 1 / 2 (NO RESULT, four causes: an unconverged fit (its step-45 verdict), an unconverged scoring solve, a failed self-check, or no entry with 8 holds) |
| 70 | `70_make_mk13_yaml.py` | workstation | camchain → `mk13_stereo.yaml`, preserving `T_BS` |
| 71 | `71_export_lens.py` | either | a PASSed KB4T (step 66's report of the same numbers) -> its JSON and a Kalibr camchain (k1..k4, for the camera-IMU fit, where dropping p1, p2 costs pixels); an OKVIS config only with `--okvis-template` and `--allow-4term-okvis`, labelled NOT FOR STEREO DEPTH (Fit A at 1 m: up to 9.3 % within 450 px); both costs printed and written |
| 75 | `75_qc_necktake.py` | robot (calib-py:2) | quick check of a neck take: holds over their own range and approach sides, encoder signs seen by the camera, sweeps, streams, head IMU, camera coverage (a stopped camera RETAKE, recorder gaps WARN), the board's spread in the torso frame, torso per hold; GOOD / RETAKE / STOP (exit 0/1/2) with an action per finding; `--qc-out` with the holds step 81 drops |
| 80 | `80_validate_extrinsics.py` | workstation | validates `T_SC`/`T_BS` against **gravity and stereo geometry**, live |
| 81 | `81_tbs_handeye.py` | robot (calib-py:2) | **T_BS decision.** hand-eye fit of the camera in the Head link over held neck poses, on every board corner's pixels in both cameras (`handeye_px.py`; `--model poses`: per-image board poses); X's σ = the fit's (clustered by hold) and the two cameras' disagreement as a systematic; KEEP / REPLACE / IMPROVE / INCONCLUSIVE (exit 0/1/1/2) at 0.3° / 3 mm and 2σ, ≥ 20 holds, every INCONCLUSIVE with its reason; take-to-take agreement |
| 82 | `82_tbs_gyro_axes.py` | robot (calib-py:2) | independent check: `R_BS` from the gyro and the neck joint rates, PASS / FAIL / INCONCLUSIVE at 1° with a within-take σ; the angle between the neck axes ± bootstrap; encoder-sign half turns |
| 83 | `83_tbs_apply.py` | workstation | writes a measured `T_BS` (mean of two agreeing step-81 REPLACE or IMPROVE reports of one model, from takes step 75 passed, fitted with the target's `T_SC0`) into an OKVIS config, changing nothing else; the note gives each take's σ against the gate |

`scripts/extra/` holds MuJoCo frame viewers (`view_head_frames.py`) from the
ghost-wbc work — useful for seeing the head frames, not part of the flow.

### Why 60 and 61 both exist

ChArUco marker decoding is **distance-limited**: a 45 mm DICT_5X5 marker at 1 m
spans ~25 px ≈ 3.6 px per cell. `61` formed only **4 stereo pairs out of 80**.
`60` uses all 88 plain chessboard corners and resolves the flip ambiguity per
pair through the stereo extrinsics — **7040 epipolar samples from 80/80 pairs**.
Prefer `60`.

### How step 30 works

A label flip between consecutive frames adds ~180° to the apparent inter-frame
rotation. At 15 fps the true inter-frame rotation is a few degrees, so the
hypotheses are trivially separable — given an independent measure of how far the
rig actually turned. **The gyro is that measure.**

Reversing corner order `k → 87-k` maps object point `(j·s, i·s, 0)` to
`((7-j)·s, (10-i)·s, 0)` — a 180° rotation about the board normal through its
centre. So `solvePnP` returns `R_solved = R_true · Rz(π)^L`, and the relative
rotation between two frames depends **only on `ΔL = L_i XOR L_j`**:

```
ΔL=0 →  R_j · R_iᵀ            ΔL=1 →  R_j · Rz(π) · R_iᵀ
```

Pick the `ΔL` whose **angle** is closer to `|∫ω dt|`, propagate labels, keep the
majority, cross-check the two cameras through the stereo extrinsics.

Only the **angle** is used, never the axis — the axis would need `R_cam_imu`,
which is exactly what has not been solved yet. The angle is frame-invariant, so
the test needs no extrinsics and works on the very first run. Being
per-**interval** rather than a chain, it also survives detection gaps: a missing
frame costs one comparison, not everything downstream.

**Reference result to reproduce:**

```
cam0: 1624/2250 PnP ok, flip events=125, label0=631 label1=993, majority=1 (61%)
   roll(majority) mean +78.8 sd 2.1 | roll(minority) mean -101.1 sd 2.9
cam1: 2041/2251 PnP ok, flip events=155, label0=723 label1=1318, majority=1 (65%)
   roll(majority) mean +97.6 sd 2.1 | roll(minority) mean -82.2 sd 3.0
cross-camera check on majority frames: agree=935 disagree=0
kept cam0=993  cam1=1318
```

The +78.8 vs +97.6 roll difference is an independent corroboration of the 18.8°
mechanical inter-camera roll — two unrelated measurements of one physical fault.

**Validation of the reconstruction.** Re-run on the same capture, this script
reproduces the PnP counts exactly and keeps:

| | original | reconstruction | identical frames |
|---|---|---|---|
| cam0 | 993 | 993 | **993 — 100.00%** |
| cam1 | 1318 | 1315 | 1311 — 99.47% |

with the gyro cross-check agreeing on **1606/1606** consecutive pairs for cam0
and 2022/2029 for cam1. `reference/2026-08-22/logs/lab_cam*.npy` are the original
kept-frame lists — use them as the regression test for any change.

Note the two cameras cluster their modes independently, so which mode each calls
"majority" is arbitrary; the script reconciles them through the stereo
extrinsics rather than assuming they agree.

---

## Reference results (2026-08-22, the calibration in force)

`reference/2026-08-22/` holds the camchains, Kalibr report PDFs, results txt and
the resulting `mk13_stereo.yaml`. Compare any new fit against these.

> ⚠ **`reference/2026-08-22/kalibr/` contains BOTH a failed and a good stereo
> fit. Use the `_d` one.**
>
> | | `calib_stereo-*` | `calib_stereo_d-*` |
> |---|---|---|
> | capture | `tree/stereo` | `tree-d/stereo` |
> | grid coverage (step 20) | 37.5% / 34.4% — **FAILS** | **79.4% / 76.9%** — passes |
> | distortion cam0 | `[-0.181, 0.704, -1.047, 0.524]` | `[-0.019, 0.047, -0.048, 0.018]` |
> | distortion cam1 | `[+0.117, -0.501, 0.747, -0.361]` | `[-0.018, 0.041, -0.040, 0.016]` |
> | | **signs differ between two identical lenses** | consistent |
> | held-out `\|dy\|` (step 60) | **degenerate — exits 2** | median 0.290 px |
> | shipped? | no | **yes** |
>
> This is trap 2 preserved as a worked example. The failed fit's Kalibr
> reprojection sigma was 0.47 px — it does not look broken from the inside.
> Step 20 rejects its capture and step 60 rejects its camchain.

| | 2026-08-22 | previous (2026-08-14) |
|---|---|---|
| stereo Kalibr sigma | 0.266 / 0.234 px on 450 pairs | — |
| held-out mono RMS | 0.268 px | — |
| **held-out epipolar `\|dy\|`** | **0.195 px** | 0.246 px |
| cam-IMU reprojection | 0.237 / 0.254 px | 0.28 px |
| gyro / accel residual | 0.0030 rad/s / 0.085 m/s² | — |
| `\|T_SC\|` | 16.7 / 17.9 cm | 13.8 / 14.9 cm |
| timeshift | −23.6 / −24.6 ms → `image_delay: 0.0241` (an artefact of step 10's old clock rebase; see the top section) | — |
| baseline | **79.0 mm** | 43.0 mm |
| projection uncertainty | ±[0.07, 0.07, 0.22, 0.10] | 5–10× looser |

Independent validation on a **separate 40 s static-head take never used in the
fit**: 80/80 stereo pairs, 7040 epipolar samples → `|dy|` mean 0.338 px, median
0.290, p95 0.817, max 1.341. The old pre-move calibration scores **19.72 px mean
on those same frames (58× worse)** while mono reproj RMS is identical
(0.422 vs 0.423 px) — trap 1, demonstrated.

`reference/2026-09-03/mk13_stereo.current.yaml` is the current rig file: same
camera calibration, **corrected `T_BS`**, plus a 35.3 mm translation offset
applied to both `t_SC` (baseline preserved to 79.0028 mm exactly).

`reference/robot-originals/` holds the scripts as they were found on the robot at
`/root/terrain_aware/calibration/`, unmodified, for provenance.

---

## Dependencies

- **Workstation:** Python 3.10+, `numpy`, `scipy` (steps 45, 46, 66, 71: new with the KB4T
  steps), `opencv-python` (with `cv2.aruco`), `pyyaml`, `pillow`. Step 80 also needs ssh to
  the robot for its live checks. On Thor the KB4T steps run in `calib-py:3`.
- **Kalibr:** the ROS1 docker image (`Dockerfile_ros1_20_04`). Not pip-installable.
- **Robot:** nothing beyond what the OKVIS image ships. `00_make_imu_stub.py` is
  stdlib-only.

```bash
pip install numpy scipy opencv-python pyyaml pillow
```

---

## For a new Claude session

**Read this section before touching anything.**

1. **This package is self-contained.** Everything needed is here. The wider
   project context (`CLAUDE.md` in OKVIS2-X) is useful but not required to run
   the pipeline.
2. **Steps 20 and 30 are gates, not suggestions.** They exist because skipping
   them produced, respectively, a 2423 px held-out epipolar error and a
   `|T_SC| = 1.03 m`. Both fits reported healthy residuals while being wrong.
3. **Never judge a calibration by reprojection error.** Trap 1. Use step 60's
   held-out epipolar `|dy|`, and step 80's gravity checks.
4. **`T_BS` is not an output of this pipeline.** If a template lacks it, use the
   default in `config/neckpitch_imu.yaml` verbatim. Do not go looking for it in a
   URDF — it is not there, and pretending otherwise hid a 31° error twice.
5. **Verify camera identity by disparity sign, never by reading config.** See
   [Camera identity](#camera-identity).
6. **The reconstructed step 30.** The original gyro labeller was never saved to
   disk (it ran inline); `30_label_canonical.py` is a reconstruction written
   against the recorded output in `reference/2026-08-22/logs/label.log`, which is
   its acceptance target. `lab_cam0.npy` / `lab_cam1.npy` beside it are the exact
   frame lists the original kept — diff against them to validate a change.
   Everything else in `scripts/` is the original code, repackaged.
7. **Do not modify `/etc/phantom*` on the robot.** Diagnose and propose; the
   operator applies. `/etc/phantomos/okvis` and `/etc/phantomos/okvis-models` are
   the granted exceptions.
8. **A repo edit does nothing until it is synced.** The deployed robot reads
   `/etc/phantomos/okvis/*.yaml`. Hand-edit the staged copy where it has
   deliberately drifted from the repo, or you silently revert someone's
   experiment.
9. **Upstream `record.sh` defaults `IMU_INDEX=0`** — a foot IMU. The packaged
   `05_record.sh` reads `imu.index` from the rig file; either way, re-verify the
   index against the live roster first.
10. **If something disagrees with this README, trust a live measurement over any
    document, this one included** — then fix the README.

---

## Verification record (2026-09-14)

Every script here was run against the real 2026-08-22 capture before this
package was considered done.

| check | result |
|---|---|
| step 30 vs the original kept-frame lists, cam0 | **993 / 993 frames identical (100.00%)** |
| step 30 vs the original kept-frame lists, cam1 | **1318 / 1318 frames identical (100.00%)** |
| step 30 flip events | 125 / 155 — exact match |
| step 30 roll statistics | `+78.8 sd 2.1` / `+97.6 sd 2.1` — exact match |
| step 30 cross-camera | 935 agree / 0 disagree — exact match |
| step 30 gyro cross-check | **1606/1606 and 2029/2029 consecutive pairs agree (100%)** |
| step 20 on the good capture | 79.4% / 76.9% → passes |
| step 20 on the failed capture | 37.5% / 34.4% → exits 1 |
| step 60 on the shipped camchain | `\|dy\|` mean 0.338, median 0.290, p95 0.817, max 1.341 px over 7040 samples — matches the documented figures |
| step 60 disparity sign | median **−22.31 px, 0.0% positive** — matches the documented −22.40 px / 100% negative |
| step 60 on the failed camchain | degenerate rectification detected, exits 2 |
| step 70 vs the shipped `mk13_stereo.yaml` | `focal_length`, `principal_point`, `distortion_coefficients`, `image_delay`, **both `T_SC`** — all MATCH |
| step 70 `T_BS` present | preserved unchanged |
| step 70 `T_BS` absent | default injected, with a warning |
| step 70 given a stereo-only camchain | refused with a clear error |

Two things this testing found, now folded in:

1. **The original labeller used the `_d` camchain.** With the failed one, cam1's
   reproduction drops to 99.47%; with `_d` it is exact. The intrinsics you pass
   step 30 matter.
2. **A degenerate rectification scores `|dy| = 0.000`** and reads as a *perfect*
   calibration. Step 60 now checks median disparity magnitude and exits 2 rather
   than reporting a flawless-looking result on a broken camchain.

Not yet exercised end-to-end: `40_calibrate.sh` (needs the Kalibr docker image)
and `80_validate_extrinsics.py`'s live checks (need the robot streaming). Both
are original code carried over unmodified; only their packaging is new.
