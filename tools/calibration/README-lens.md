# KB4T lens calibration (head stereo pair)

Collecting the recordings (what to ask, the two routes, the on-the-spot check): **COLLECT-lens-data.md**.

One command from record-app sessions to a validated KB4T lens and its OKVIS config:

    cd tools/calibration/scripts
    python run_lens_kb4t.py --rig ../config/rig-<robot>.yaml --out <dir> \
        [--max-blur 0.5 --exposure-ms <ms>] \
        --okvis-template <robot's current OKVIS config> --baseline-okvis <robot's current OKVIS config> \
        <take>/session.mcap ...           # or step-08 take folders: <takes>/d050 <takes>/d075 ...

Steps (each also runs on its own; see each script's docstring):

| Step | Script | What it does |
|---|---|---|
| 08 | `08_neck_grid_direct.py` | ON THE ROBOT, no recorder: neck pitch x yaw grid of still holds, grabbing synced pairs from the DMA camera queues into `<take>/pose_NNN.npz`. Refuses a motor tool without `targets` (2.1.1 zeroes the legs), refuses while another program commands the motors, pins arms/spine/legs. |
| 15b | `15b_dump_neck_take.py` | Corners from a step-08 take folder, in step 15's layout (median-displacement stillness per pose). `run_lens_kb4t.py` uses it for take folders. |
| 15 | `15_dump_corners.py` | Corners of both cameras per stereo pair. Default: still holds only. `--max-blur PX --exposure-ms MS`: moving-board mode for a handheld board. It keeps pairs whose blur within the exposure is ≤ PX, spaced `--gap-s` apart, and refuses cameras that are not hardware-synced. |
| 16 | `16_split_distance.py` | Splits pairs into FIT and held-out TEST. TEST gets every 4th hold in time order, plus every pair beyond the fit range. The driver fits up to 3 m; run alone, step 16 uses the rig's 1.3 m. |
| 45 | `45_fit_lens.py` | KB4T fit on FIT (k1..k4 + p1, p2 per camera, plus the stereo pose), with its quality gates. `--max-z` overrides the rig's 1.3 m far-board gate; the driver passes 3.0. |
| 66 | `66_validate_lens.py` | Depth error on TEST per image cell × distance against the fixed pass rule (exit 0 PASS / 1 FAIL / 2 NO RESULT). The driver also scores `--baseline-okvis` on the same pairs. |
| 71 | `71_export_lens.py` | OKVIS config with `distortion_type: kb4t`, written only on a step-66 PASS. Keeps the template's cam0 T_SC, T_BS and tuning. |

## Why a held-out set split by distance

A fit always looks good on the data it was fitted to. Fit rms cannot tell a correct lens from a biased one. On
phantom09 (4 Nov 2026) a plain KB4 refit converged at 0.55 px rms and still read depth 3.7 % long on held-out
boards: without p1, p2 the fit absorbed the lens's decentering into a shorter baseline. Holding out the FAR boards
tests the part that matters for a robot, depth at range, which no near board can show.

## Moving-board mode: when it is valid

- **Cameras hardware-synced.** Step 15 checks the median left/right stamp offset (`--max-sync-ms`, 0.5 ms). On the
  OAK heads it is about 0.015 ms.
- **Exposure known.** Blur = board speed × exposure, so `--exposure-ms` must be the real one. Too short a value
  makes the filter too loose.
- phantom09: 704 sharp pairs out of 3,700. The still-only mode found 45 in the same takes.

## Deploying

The OKVIS image must read `distortion_type: kb4t` (`okvis_common/src/Kb4tRectify.cpp`). VIO uses k1..k4; p1 and
p2 feed only the stereo network's rectification.

## Results so far

These are `run_lens_kb4t.py` defaults on 4 Nov 2026, Terminus: fit to 3 m, every 4th hold held out, moving-board
mode. The takes were handheld A/F/E sessions. phantom11 and 19 used a 0.15 px blur limit; phantom09 and 15 used 0.5 px.

TEST depth error, KB4T vs the robot's previous config on the same held-out pairs:

| Robot | Pairs FIT / TEST | Fit rms | Baseline | KB4T: median (IQR) | KB4T by distance (<1.0 / 1.0–1.3 / 1.3–1.7 / >1.7 m) | Previous config by distance | Step 66 |
|---|---|---|---|---|---|---|---|
| phantom09 | 529 / 175 | 0.45 px | 43.76 mm | −0.05 % (−0.37 … +0.20) | −0.04 / −0.21 / −0.16 / +0.82 % | −0.39 / +0.50 / +2.06 / +9.85 % | PASS, 7 entries |
| phantom11 | 284 / 94 | 0.36 px | 43.91 mm | −0.04 % (−0.17 … +0.09) | −0.02 / −0.23 / −0.05 / − % | +0.04 / +0.36 / +0.00 / − % | PASS on 4 entries (too few) |
| phantom15 | 634 / 210 | 0.44 px | 43.52 mm | −0.15 % (−0.72 … +0.46) | −0.29 / −0.37 / +0.40 / +0.62 % | −2.15 / −1.09 / +7.69 / +16.48 % | FAIL: left 0.9–1.3 m −0.85 % |
| phantom19 | 387 / 128 | 0.47 px | 43.72 mm | +0.00 % (−0.21 … +0.19) | +0.00 / −0.40 / +0.25 / +0.56 % | −0.42 / −2.33 / −2.73 / −4.47 % | PASS on 5 entries (too few) |

How to read it:
- **Depth error:** + means stereo reads too far. Rows are the median per distance band.
- **Step 66:** judges only boards up to 1.3 m, per image cell, and needs at least 8 held-out holds per cell × distance
  entry. Thin TEST sets judge few entries. The driver exports a config only on a PASS over at least 6 entries.
- **Data limits** (pairs past 1.7 m, and corners in the image's outer rim): phantom11 has the least (12 pairs past
  1.7 m, a quarter of the others' rim corners). No robot has boards past 2.3 m, so depth beyond about 2 m is untested.
- **Two independent pipelines agree.** phantom11 and phantom15 were also fitted separately, with different scripts
  and pair selection. The lenses agree within 0.03–0.2 px median.
