# Collecting data for a KB4T lens calibration (guide for agents)

This guide covers how to get recordings that `run_lens_kb4t.py` (README-lens.md) can turn into a calibration that
passes validation. The targets come from the 4 Nov 2026 runs on phantom09, 11, 15 and 19.

## What to ask the human, and nothing more

Ask only for what the data cannot give you. The board's known size gives you its distance, so you need no floor
marks or tape measure.

1. **The robot, and permission to act on it.** Moving the neck needs an explicit yes. Motors are enabled by the
   operator, never by you. Never touch the walker or locomotion.
2. **Which route** (below): someone moving the board by hand, or the board on a stand while the robot moves its head.
3. **For the handheld route only: the cameras' exposure time.** The blur filter needs it, and recordings do not store
   it. phantom09's 2 ms came from a config header note.
4. **For the stand route: where the board stands.** One distance at a time. You tell them when to move it.

## Before recording: check the robot

- **Cameras streaming.** Both stereo cameras at 15 fps over SuperSpeed (producer log: `C1:15fps C2:15fps`, a
  10 Gbps link). A frozen producer looks healthy in `kubectl`, so check that the frame files in `/dev/shm` are
  updating.
  - To restart the cameras, scale the `dma-video` producer to 0, wait until it has fully stopped, then scale it back
    to 1. A rolling restart overlaps the two pods and the old one deletes the new one's `/dev/shm` segments
    (RUNBOOK §3).
  - Afterwards `ls /dev/shm/video_* | wc -l` must be 6.
- **Cameras hardware-synced.** Left/right timestamps should match within about 0.02 ms (OAK heads: about 0.015 ms).
  Step 15's moving-board mode refuses a recording with a median offset over 0.5 ms.
- **The rig file.** `config/rig-<robot>.yaml` maps cameras to DMA queues. For the left/right sockets, read
  `/etc/phantom/head_camera.json` and the robot's live OKVIS config (`dma_queue` per camera); robots differ
  (phantom09 1/2, phantom11 3/0, phantom14 2/1). Set the resolution the producer actually records (phantom11 is
  960x600).

## Recording

Use the robot's own record path where it has one: the record app via the console agent / `okvis.sh`, or the
`calib_record.py` the takes on phantom09 used. `05_record.sh` (ported from AI-1184) is the fallback. Its default
image and namespace are dated, so pass `IMAGE=` and check `NS=` against the robot. Every take must produce a
`session.mcap` with both cameras' `/cam<queue>/image` channels.

## Route A — handheld board (moving-board mode)

Three takes, robot still, board held by a person. Use `06_coverage_live.py` (live coverage map) on the robot while
they move.

| Take | Board distance | What the person does |
|---|---|---|
| A, near/mid | 0.5–2 m | Slow sweeps covering the whole image, pausing briefly at each spot |
| F, far | 2–3.5 m | Same, and walk sideways until the board touches each image edge. Past 3 m it is often skipped (no robot has data beyond 2.3 m); ask for it explicitly |
| E, edges | 1–2.5 m | Slide the board along the left edge top to bottom, then the right; it must touch the edge in BOTH cameras. Then the four corners |

Ask for **slow movement**: blur = speed × exposure. On these takes about half the frames with the board passed
0.5 px of blur.

Process with `--max-blur 0.5 --exposure-ms <ms>`. phantom09 and 15 used 0.5 px and validated well; 0.15 px
(phantom11, 19) halves the data for no measured gain.

## Route B — board on a stand, robot moves its neck (still holds)

**Use `08_neck_grid_direct.py`** (no recorder; this is how phantom11 and 15 were done). The flow:
- Copy a kit to the robot: `pydeps/` with dma_ipc, a `dma_motor_motion` WITH `targets`, a venv with
  numpy/scipy/opencv-python-headless/pyyaml, and this folder.
- `08_neck_grid_direct.py check --left Q --right Q --tool ... --pydeps ...`
- One `grid --take dNNN` per distance.
- On the robot: `run_lens_kb4t.py --rig ... --out run --holdout-every 2 takes/d050 takes/d075 ...`
`07_neck_grid.py` below is the variant for robots with a recorder running.

This needs no hand-eye skill from the person: they only move the stand. Each neck pose is a still hold, so step 15's
default still mode extracts it with no blur question at all.

1. **Board at about 0.5 m**, facing the robot. Start the recorder.
2. **Plan, then run the grid:** `python3 07_neck_grid.py --pitch=-15,-5,5,15 --yaw=-27,-13,0,13,27`.
   - Read the plan, then add `--go`.
   - The script clamps pitch to ±26° and yaw to ±34°. phantom14's limits were pitch ±27.5° and a yaw stop at 35.9°.
   - It stops at the first pose a joint misses by more than 3°.
3. **Repeat at about 0.75, 1.1, 1.6, 2.0 and 3.0 m.** Tell the human the next distance and wait for "done".
   - When the board is low, tilt the head UP to keep it in view. A grid that only looks down covers half the image.
   - Ask for 3 m explicitly; it is what every robot so far lacks.
4. **Hold the body still.** `dma_motor_motion` re-holds every joint it is not given at its measured pose, so a robot
   on bent knees sags with every command (phantom14: spine −1 → −25° over the grids).
   - Pass `--also "SpineYaw=<start>,LeftKneePitch=<start>,..."` with the start angles.
   - Stop if any body joint drifts more than 3–5°.

## Check on the spot, before the human leaves

Copy the takes off the robot, then run steps 15 and 16 on a compute box (Terminus or Moonstone, `nice -n 10`,
`OMP_NUM_THREADS=1`; never on the user's laptop). Then compare against the table below.

| Target (per robot, all takes) | Why | 4 Nov |
|---|---|---|
| ≥ 500 usable pairs | Fit and held-out test both need hundreds | phantom09 704, phantom11 378, phantom15 844, phantom19 515 |
| ≥ 50 pairs at 1.7–3 m | Depth at range; the fit runs to 3 m | 12–54; nobody has any beyond 2.3 m |
| Every image corner cell ≥ about 5,000 left-camera corners, outer rim ≥ about 1,500 | The lens model is decided at the edges | phantom11 rim 363, corners ≤ 3,300: its weak spot |
| Step 66 judges ≥ 6 cell × distance entries | Otherwise no config is exported | phantom09 7; phantom11 4, phantom19 5: too thin |

If a target is missed, re-record that part while the human is still there: more corners/edges, or farther boards.

## Field lessons (phantom15, phantom11, 4-5 Nov 2026)

- **Check the motor tool before any neck move.** `dma_motor_motion` from dma-ethercat 2.1.1 (no `targets` signal in
  `--help`) sends commands that cover only joints 0..highest listed. For the neck (14/15) that leaves spine and legs
  (16-28) at **position 0 and zero stiffness**. DMA.ethercat `a66805d` fixed this. Use a build that has it:
  `07_neck_grid.py` needs `targets`. phantom14's 2.1.0 build ran as a private copy on phantom11/15; the shared-memory
  layouts were byte-identical (`/dev/shm/{config,actuals,desired}`).
- **Never command joints while a locomotion policy streams.** Check that `/dev/shm/desired` is not advancing.
  The robot hangs on the gantry, with the policy off.
- **Direct grabbing instead of the record app** (archived in the team handoff branch, `calib-kit-20261105/`): the
  neck-grid script read synced pairs straight from the DMA camera queues at each still pose (5 pairs per pose). Corners
  came from those frames. It needs only `dma_ipc` and numpy on the robot.
- **Process on the robot, not over WiFi.** Copying a 49-pose take (480 MB) ran at 2.3 MB/s, slower than recording one.
  A private venv (numpy, scipy, opencv-python-headless, pyyaml) on the Thor did corners and the fit; a fit took about
  3 min.
- **Resolution** is set by the producer deployment's `--width/--height` args, not by `head_camera.json`. Calibrate at
  1280x800: at 640x400 a board at 3 m has ~10 px tags. Change it with scale 0 → patch → scale 1, keep the old spec, and
  put it back afterwards. Writing a 1280x800 fit at a lower resolution by scaling fx, fy, cx, cy is an assumption: it
  held on one 960x600 sample on phantom11. Check it on native frames.
- **Exposure** is in `/etc/phantom/head_camera.json` per camera (phantom09/15: 2000 us, phantom11: 8000 us).
- **Range:** with this 7x5 board, 2.2 m is the practical maximum. At 3 m at most 2 tags were found (small and dark).
- **The bottom of the image** needs the board below the camera and facing it. A board on the floor is seen too
  edge-on. Lean it back toward the head at about 0.75 m and run head pitch 0..+24.
- **Camera board crash:** the producer reopens the board, but the frame-pool sequence then desyncs from the metadata
  and every read fails. A full no-overlap producer restart fixed it.
- **phantom11 result** (neck grids at 1280x800, 800 pairs at 0.5-2.2 m): held-out depth −0.03 % (IQR −0.25 … +0.35),
  rows 0.21 px. Its previous config read −18.7 % / 18 px rows on the same day's frames.

## Then

`run_lens_kb4t.py` (README-lens.md) on the compute box. Deploy only to an OKVIS image that reads
`distortion_type: kb4t`: `grep -c -a kb4t /home/OKVIS2-X/build/okvis2x_app_dma_live` inside the OKVIS pod must be
> 0. On an older image the config fails to start OKVIS. Back up the robot's config and `robot.env` first.
