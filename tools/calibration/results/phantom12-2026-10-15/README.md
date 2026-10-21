# phantom12 stereo calibration attempt, 2026-10-14/15: NOT deployable

**Verdict: no calibration file from this run is fit to deploy.** The cameras expose ~8.5 ms
apart (left first), so every moving-board view is smeared. The views where the board was
nearly still only span 0.8–1.5 m. As a result the stereo geometry is ambiguous at grasping
range: the two fits below disagree by about ±7 % in depth at 0.5 m.

**Next:** a stop-and-go stereo take, described below, can be recorded now. The real fix is
synchronized exposures (`dma-video-sync-request.md`, for DMA.video's owner).

## The recording

- **Head:** phantom12, one OAK board, 15 fps.
  - queue 1 = left (socket B, FSYNC master);
  - queue 2 = right (socket D, FSYNC slave).
- **Exposure:** fixed at 3000 µs / ISO 800 for the take, restored to auto afterwards.
- **Recorder:** OKVIS2-X record app (`okvis2x:thor-0594dfc5…`), 210 s, lossless PNG,
  8.7 GB `session.mcap`.
- **Frame rate:** the record pod's 2-CPU limit dropped frames, giving 11.6 / 10.4 fps.
- **Pairs:** step 10 found 1738 synchronized pairs; stamps agree to 0.03 ms mean.
- **Board:** 7×5 AprilGrid, pitch measured 0.120 m with a steel rule.
  - No still holds: board tracking found none.

## Pipeline problems found and fixed on the way

- **Step 40 bagged every image twice.** `kalibr_bagcreater` finds `cam*` folders anywhere
  under the tree and keeps only their names, so `heldout2/cam0` re-read `cam0`. Fixed:
  step 40 now bags a staged view of the tree (commit "step 40 bags each image once", test
  `test_calib_bag_staging.py`).
- **Kalibr on ARM64 (Thor)** needs two changes from the stock image:
  - `python3-opencv ros-noetic-cv-bridge` on top of `ros:noetic-ros-base`, because
    `osrf/ros:noetic-desktop-full` is amd64-only;
  - `LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1`. Without it, `import cv_bridge`
    before `cv2` fails with "initialization of cv_bridge_boost raised unreported exception"
    (a static-TLS problem).

## Results

| | all-frames fit (`calib_stereo-*`) | slow-frames fit (`calib_slow-*`, board < 25 px/s) |
|---|---|---|
| views | 450 pairs, 75 used by Kalibr | 312 pairs, 88 used |
| step 20 coverage / reach | 67.5 % ✓ / 55.9° ✗ (gate 60°) | 65.6 % ✓ / 57.4° ✗ |
| reprojection σ (px) | 0.51 / 0.41, 0.50 / 0.41 | **0.29 / 0.22, 0.30 / 0.28** |
| baseline | 43.42 mm | 47.27 mm |
| step 60 held-out median \|dy\| | 0.511 px ✗ (all held-out) | **0.222 px ✓** (slow held-out) |
| step 65 scale, slow held-out | +0.80 % | +0.54 % |
| scale at 0.8–1.0 / 1.0–1.2 / 1.2–1.5 m | −2.0 / +0.2 / +2.4 % | +1.2 / +0.3 / −1.2 % |
| extrapolated to 0.5 m | **−8 %** | **+7 %** |

- **Left/right check:** step 60's disparity sign confirms queue 1 is the physical left
  (dx +18 px, 100 % positive).
- **Root cause:** `diag/sync_diag.py` fits signed rectified `dy` against board vertical speed
  over 205 pairs: slope −8.47 ms, r = −0.89.
- **Why the baselines differ:** baseline and the toe-in angle between the cameras trade off
  when all usable views sit at one distance. `diag/depth_scale.py` (172 independent slowish
  pairs) shows both fits drifting with distance in opposite directions. The truth lies
  between them; a flat-scale fit would come out near 45–46 mm.
- **phantom21's baseline:** 43.37 mm, same head design.

## Next take: stop-and-go stereo (works despite the offset)

- **Hold the board still for about 1 s at every pose.** Only those frames are used, so the
  8.5 ms offset costs nothing.
- **Distances:** 0.4–0.6 m (grasping range) and 1.5–2.5 m, plus 0.8–1.2 m. Spread in
  distance is what separates the baseline from the toe-in angle.
- **Edges and corners:** hold poses where tags sit within ~50 px of the left and right
  edges. Both runs missed the 60° reach gate there.
- **Tilts** at each distance.
- **Frame rate:** raise the record pod's CPU limit, or record fewer seconds and more poses.
- **Step 10 selection:** select the still frames first (`diag/select_slow.py`).

## Step-70 template ("B")

phantom12's `/etc/phantomos/okvis/` files are not this camera pair: `mk12_stereo.yaml`
implies a 66 mm baseline, and `okvis_stereo_dense_thor.yaml` 86 mm with T_BS = identity.
- **Use** the repo's `config/foundation_rig/mk12_stereo_holevisor.yaml` as the template,
  with `dma_queue` set to 1 then 2 (queue 1 = left, confirmed above).
- **Its T_BS is phantom21's** (head-design value, unverified on phantom12).
- **Step 70 also needs** the camera–IMU take, which has not been recorded.

## Where the data is

- **phantom12:**
  - `/data/okvis-recordings/calib-stereo/` (mcap) and `calib-stereo-euroc/` (export);
  - `tree/stereo`, `tree/slow`;
  - `/root/calib-work` (Kalibr bags, camchains, reports);
  - images `kalibr:1f60227-arm64`, `calib-py:1`.
- **Workstation:** `~/calib/phantom12/`: `takes/calib-stereo/session.mcap` (8.7 GB archive),
  `work/tree/stereo`, `results/`, `NOTES.md` (full run log).
- **`diag/`** holds the ad-hoc scripts as run (container paths `/w`, `/e`).
