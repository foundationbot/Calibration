# phantom12: right camera exposes ~8.5 ms after the left, although FSYNC is on and the stamps agree

## What we found

While calibrating phantom12's head stereo pair (2026-10-14), we found that the two cameras
do not expose at the same instant:

- **The right image's content lags the left by about 8.5 ms** (7.4–9.8 ms when each quarter
  of a 3.5-minute recording is fitted separately; no trend).
- **The frame stamps hide it.** `capture_timestamp_ns` of each left/right pair agree to
  0.03 ms on average (0.31 ms max), so nothing downstream can see the offset.

Setup at the time:
- One OAK board (`19443010F1BC622E00`), 15 fps, 1280×800 NV12.
- **Software actually running:** image `foundationbot/dma-video:main`, digest
  `sha256:517d2a291a127eb240816d81de8fae8a1fe962bd02066c44007e517cc7c25ad8`. Its
  `src/multi_camera_producer.py` is byte-identical to commit **6e5d172** (2026-02-19), i.e.
  `main` before 5f0b4d9 (`--fsync-master-board`, 2026-08-11). Current `main` is a0f1d84.
  **Line numbers below refer to 6e5d172**, the code on the robot.
- `head_camera.json`: left = queue 1, socket B; right = queue 2, socket D; bottom = queue 0,
  socket C.
- Fixed exposure on left and right (3000 µs, ISO 800); bottom on auto.
- Producer log: `FSYNC: enabled (master=B)`, `CAM_B -> cam1: MASTER (OV9282, NV12)`. Socket D
  is therefore an `INPUT` slave (`src/multi_camera_producer.py:606-609`, master chosen at
  `:681-685` from `--master`, default `B`; GPIO 6 bridge at `:167-176`).

## Update 2026-10-15: the bottom camera's auto exposure triggers it

We then drove the cameras with our own capture tool, built from the producer's own
`create_camera_node`/`setup_gpio_bridge` (no configuration changed). Left and right were on
fixed 3000 µs / ISO 800 throughout, and the board was moving in every run:

| socket B sends the pulse unless noted | bottom camera C | right camera's delay | r |
|---|---|---|---|
| DMA.video producer (14 Oct) | auto exposure, sync listener | +8.5 ms | −0.89 |
| capture tool | not running | +0.3 ms | −0.13 |
| capture tool, **D** sends | not running | +1.1 ms | −0.26 |
| capture tool | auto exposure, sync listener | **+5.0 ms** | −0.80 |
| capture tool | auto exposure, **not** on the sync pulse | **+3.5 ms** | −0.75 |
| capture tool | **fixed** 3000 µs, sync listener | **+0.3 ms** | −0.27 |

- Neither the sender nor C's sync membership matters. **C running on auto exposure makes D
  expose late**, by a varying amount (3.5–8.5 ms). With C on fixed exposure the pair is in sync.
- DepthAI's own per-frame timestamps (start, middle and end of exposure) say B and D expose
  within 0.02 ms in every case: the firmware does not see the shift.
- In the producer, auto exposure also sets C's 3A to full rate (`isp3a_fps = 0`, `:620`) while
  fixed exposure runs it at 1 Hz, so "auto exposure" and "full-rate 3A on C" are confounded.

**Confirmed in the real producer (15 Oct, 15:10–15:21):** with the bottom camera removed from
`head_camera.json` (phantom12 does not use it), the pair is in sync both ways:

| producer, bottom camera off | pairs | delay | r |
|---|---|---|---|
| left/right auto exposure (normal operation) | 177 | +0.47 ms | −0.16 |
| left/right fixed 3000 µs (calibration take) | 499 | +0.10 ms | −0.04 |

So for phantom12 the immediate fix is done (bottom camera off). What remains for DMA.video is the
underlying behaviour: a camera on auto exposure on the same board shifts another camera's exposure
without any trace in the timestamps. Any head that needs the third camera will hit it; giving that
camera a fixed exposure avoided it in our test (+0.3 ms).

## How it was measured

An AprilGrid was moved by hand in front of the head and recorded losslessly with OKVIS2-X's
record app. For 205 synchronized pairs spread over the recording, we compared:
- the vertical offset between the same board corner in the rectified left and right images
  (`dy`), with
- how fast the board was moving vertically in the image (`vy`).

If both cameras exposed together, `dy` would not depend on `vy`. It does:
`dy = −vy · 8.5 ms`, correlation **−0.89**.

| board vertical speed | pairs | median \|dy\| |
|---|---|---|
| < 50 px/s | 137 | 0.29 px |
| 50–150 px/s | 56 | 0.81 px |
| 150–261 px/s (fastest seen) | 12 | 1.49 px |

On frames where the board was nearly still (< 25 px/s), our left/right row check passes with
either calibration (0.22–0.28 px). On moving frames the same calibration fails (0.51 px). The
error is in the moving images, not in the calibration.

## Why it matters

- **Live stereo depth.** Horizontal motion becomes disparity error: `vx · 8.5 ms`. At
  100 px/s that is 0.85 px: about **3 % of depth (2.5 cm) at 0.85 m**, 4 % (5 cm) at 1.2 m. OKVIS's S2M2 depth on phantom12
  is wrong whenever the head turns or something in view moves.
- **Calibration.** Any moving-board stereo calibration absorbs the offset as noise or bias.
  Until it is fixed we can only calibrate with the board held still at every pose.
- **SLAM.** Stereo triangulation during motion carries the same error.

## What we need

Three changes, in order of importance. The first is the one that matters; the other two make
any remaining error visible instead of silent.

### 1. Left and right expose at the same instant (required)

**Done when:** a 1-minute recording of a board moved up and down gives a `dy`-vs-`vy` slope
under 0.5 ms (8.5 ms today). Check this with fixed exposure **and** with auto exposure, since
the robot normally runs auto.

We don't know the cause. **A clue (hypothesis, untested):** 8.5 ms (7.4–9.8 ms per quarter) is
close to 8.3 ms, the time an OV9282 needs to read out one 1280×800 frame (it can run at
120 fps). That fits a listener that exposes one readout period after the sender, which is
sensor and DepthAI-firmware timing rather than DMA.video's Python. If the experiments below
point that way and DepthAI offers no setting for it, the question goes to Luxonis with this
measurement (DepthAI 3.3.0, OV9282, listener ~8.5 ms late).

Things to try, cheapest first. Each is a producer restart, a 1-minute moving-board take and
our script:

- **Swap the sender: run the producer with `--master D`.** Today socket B sends the sync
  pulse (`OUTPUT`) and D listens (`INPUT`), so the two sensors run in different sync modes.
  If after the swap the *left* picture is the one ~8.5 ms late, the listening mode is the
  cause. (Making every camera a listener is not an option on one board: with no sender
  there is no pulse and no frames.)
- **Measure the delay at two or three frame rates** (e.g. 10, 15, 30 fps). A delay that stays
  ~8.3 ms whatever the rate points to the readout-period explanation above.
- **Measure the delay at two or three exposure times.** If it changes with exposure, the
  two sensor modes place the exposure differently relative to the frame start. That is fixable
  in sensor configuration or with a compensating trigger delay.
- **Updating the robot to current `main` will not fix this.** The only code change since
  6e5d172 is `--fsync-master-board` (5f0b4d9), which is off by default and only matters with
  several boards; `depthai` is pinned to 3.3.0 in both. On phantom12's single board the new
  code takes the same path (B sends, D listens). Running the experiments on current `main`
  is still sensible if that is what will ship.
- **Give both stereo cameras the same exposure in auto mode.** With auto exposure each camera
  runs its own controller (`:620` sets full-rate 3A per camera), so their exposure times can
  differ. Exposures of different length have different midpoints even with a perfect trigger.
  One auto-exposure result shared by both, or locked exposure for the stereo pair, avoids that.

### 2. Frame metadata that says when the picture was really taken

This does not fix stereo depth on its own: the two pictures would still show the scene at
different moments. It lets software see and account for whatever delay remains.


- **`capture_timestamp_ns` = the middle of that camera's own exposure.** Today the two frames
  of a pair carry stamps 0.03 ms apart while their exposures are 8.5 ms apart.
- **Fill `exposure_time_ms` and `gain_db`.** Both fields already exist in `VideoFrameMeta`
  (`include/video_frame_meta.hpp`), but the producer writes them as `0.0`
  (`src/multi_camera_producer.py:773`). Consumers need the exposure time to find the
  midpoint and to check that both cameras use the same exposure. DepthAI frames report both.
- **If a fixed delay has to remain** after step 1, apply it per camera in the stamp, so the
  stamps match the exposures.

### 3. Correct the header comment (minor)

`include/video_frame_meta.hpp:51` says `capture_timestamp_ns` is a "Monotonic clock". The
producer writes host wall-clock time (`time.time()` minus device latency,
`src/multi_camera_producer.py:750-751`). That comment already misled one consumer:
DMA.streams' video sidecar adds a ~29 ms "correction" to every stamp.

## How to check a fix

Record ~1 minute of a board moved up and down in front of the head with OKVIS2-X's record
app, then rerun the same measurement. We have the script (`sync_diag.py`) and are happy to
run it on a new recording.
