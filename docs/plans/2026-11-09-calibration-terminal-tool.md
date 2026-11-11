# Head-stereo calibration terminal tool — design and milestone plan (rev 2)

> **For agentic workers:** this is the DESIGN + MILESTONE plan. It spans several subsystems, so each milestone
> (M0-M6, section 13) gets its own step-by-step TDD plan (`docs/superpowers/plans/<date>-calibtool-M<n>-<name>.md`,
> written with superpowers:writing-plans) when the user starts it. Execute those with
> superpowers:subagent-driven-development or superpowers:executing-plans. Nothing here is implemented.
>
> rev 1 (574963b2) was reviewed by three independent reviewers (technical, safety/operations, operator workflow) on
> 9 Nov 2026; section 15 lists what changed and which rev-1 claims were wrong.

**Goal:** one command, `calib phantomNN`, that takes a robot's head stereo pair to a validated, deployed KB4T lens +
camera-to-IMU rotation: it guides the operator's board placements in closed loop, runs the neck grids, shows progress
and coverage in plain words, fits, proves the new calibration beats the live one, deploys it as one transaction with a
rollback, and checks it afterwards. No Kalibr.

**Architecture:** a thin terminal UI on the operator's laptop (`calib`) drives a job runner on the robot (`calibd`) over
ssh. Capture and compute run on the robot (Thor CPU, niced). `calibd` keeps its state on disk and resumes after a
dropped link or reboot, but **never resumes motion by itself**. The package lives in `tools/calibration/calibtool/`
and imports only a declared list of existing `tools/calibration` modules, so it can be PR'd to devel as a unit.

**Tech stack:** Python 3 (NumPy, SciPy, OpenCV headless with aruco/AprilTag, PyYAML), `textual`/`rich` for the TUI,
`dma_ipc` for camera rings, `k0s kubectl`, the fixed `dma_motor_motion` (2.1.0 build) for the neck.

---

## 1. Decisions taken with the user (9 Nov 2026)

| # | Decision |
|---|---|
| D1 | Neck-grid collection only (no handheld moving-board mode). |
| D2 | All collection at 1280x800 for uniformity; the deploy size is asked at the end (D7). |
| D3 | Mostly Python; speed from algorithms + parallelism, with per-step budgets (section 10); a compiled hot path only where a step misses its budget after the algorithm work. |
| D4 | The tool may start/restart the camera producer itself (no-overlap) and must tell the operator when it suspects hardware (section 8). |
| D5 | The operator starts `dma-ethercat`; the **tool enables the motors** after an on-screen check and a keypress (section 7). Tool only; agents outside it still never enable motors. |
| D6 | Deploy only when the new calibration is shown better than the live one on the same fresh data (section 6), as one transaction with a rollback; post-check **depth first, OKVIS second and optional** (opt-out when OKVIS is not used). |
| D7 | Deploy size asked at the end: match the robot's previous default by automatic scaling (recommended) or 1280; "not sure" shows this robot's facts. The scaled config is verified at the deploy size before the decision. |
| D8 | Default placements + adaptive extras; talkative, comparative messages; edge coverage never traded away. |
| D9 | Too dark: exposure first, gain second, a contrast-stretched copy only to *find* tags (corners refined on the raw image); camera-side contrast/sharpening/denoise never touched. |
| D10 | Compute on Thor by default (`--compute` override deferred to v2, section 14). Images never leave the robot during a session. |
| D11 | Data management is part of the tool (section 11). |
| D12 | Develop on `feat/calibration-agent-calib-tui` (off AI-1184, the calibration development branch); merge into AI-1184 on the user's word; at the end PR the package into devel from a devel-based branch (section 12). |

## 2. What already exists (reuse, do not rewrite)

| Piece | Where | Reuse as |
|---|---|---|
| KB4T projection, pinned by `tests/data/kb4t_reference.json` | `scripts/lensmodel.py` (both branches) | the model in the fast solver |
| Neck-grid capture with refusals (motor tool without `targets`, another program commanding) and clamps (+-26 pitch, +-34 yaw; phantom14 hit a stop at 35.9) | `scripts/08_neck_grid_direct.py` (**devel only**, PR #909) | grid driver core; every pose, default or adaptive, goes through its clamps |
| Take folder -> corners (process pool), writes `img_yaw/img_pitch/img_ay/img_ap` | `scripts/15b_dump_neck_take.py` (devel only) | detection worker; its output feeds step 81 directly |
| FIT/TEST split: TEST = every `--holdout-every`-th near hold in time order (default 4) + every pair beyond `--max-fit-z` | `scripts/16_split_distance.py` (devel only) | replaced by capture-time stratified assignment (section 5), kept as the replay reference |
| KB4T fit + quality gate (hold rms 1.8 px, `--max-z`, 20 % reject, warm refit, camera-side check) | `scripts/45_fit_lens.py` `main` (devel has `--max-z`) | gate logic around the fast solver |
| Validation per image cell x distance bin; `DIST_BINS = (0.4-0.6, 0.6-0.9, 0.9-1.3 m)`; >= 8 held-out holds per entry; exits 0 even on FAIL | `scripts/66_validate_lens.py`, `lenscells.py` (identical on both branches) | TEST targets and the decide rule; verdict parsed from text, never the exit code |
| Driver: fit to 3 m, far bands reported per band, `--min-validated 6` | `scripts/run_lens_kb4t.py` (devel only) | reference pipeline for replay |
| Current solver: scipy `least_squares` TRF, finite-difference Jacobian with `jac_sparsity`, `x_scale="jac"`, one board pose **per stereo pair**, PnP start dropping failed pairs, stereo-pose-only presolve, `warm_fit`, trim 3 px x 4 rounds, success only if every round converges, caps 2000 x 4 x 3, residual clamp 1e3 for non-finite/behind-camera points | `scripts/stereofit.py` | the **reference** the fast solver must reproduce |
| Scorer: `lensscore.score` runs `solve(scale=True)` with the camera fixed, twice per score; `self_check` re-scores | `scripts/lensscore.py` | second slow path to replace (per-pair independent) |
| Export with `distortion_type: kb4t` on PASS | `scripts/71_export_lens.py` (identical) | export |
| Camera-in-Head X from neck takes (hand-eye, pixel model, KEEP/REPLACE/IMPROVE); **requires `--urdf`** for the neck chain (`neckfk.NeckChain`); X lives in the URDF's Head frame | `scripts/81_tbs_handeye.py` (**AI-1184 only**) | camera mount (reported in v1) |
| IMU rotation from gyro + joint rates; needs record-app `session.mcap` + EuRoC export (full-rate IMU + `/joints`), joint rates > 0.3 rad/s; lag = joints<->IMU on +-50 ms in 2 ms steps; gyro error ~0.3 deg, gate 1.0 deg | `scripts/82_tbs_gyro_axes.py` (AI-1184 only) | v2 third-angle route (section 9) |
| Writes `imu_parameters.T_BS` only, fail-closed, needs two step-75-GOOD takes | `scripts/83_tbs_apply.py` (AI-1184 only) | not used for T_SC |
| Floor-board rotation: `imucam_fast.py` (sweep incl. yaw, IMU slot means, board PnP; execs the robot-only `/root/p11-cal/neckgrid11.py`) + `rscfit.py` (**2 angles: about camera x and z; "yaw about the gravity-free axis is unobservable"**, soft-L1, leave-out splits); `mkfinal.py` (cam0 T_SC rotated, cam1 = cam0 x stereo pose, translation and T_BS kept); `rowcheck.py`, `rowfeat2.py`, `boardtruth.py`, `worldtilt.py`; `p11_1280.sh`, `deploy1106.sh` | `calib-kit-20261106/` on branch `feat/*-stereo-depth-eval` | gravity rotation, T_SC surgery, post-checks; ported into the package (no robot-only imports) |
| Rectified-pair row check on OKVIS's network input (`/dev/shm/okvis-viz/rect_cam{0,1}.jpg`), size-aware `rowfeat3.py` | phantom11 `/root/p11-cal/` only (not in git) | OKVIS post-check |
| CI job with numpy/scipy/opencv-headless/pyyaml/pillow/mcap + a `cv2.aruco` check | `calibration-tests` in AI-1184's `.circleci/config.yml` | port to devel in M6 |

M0 facts (measured by the technical reviewer): of 51 files shared by AI-1184 and devel under `tools/calibration`, only
`15_dump_corners.py` (+106/-0) and `45_fit_lens.py` (+5/-0, `--max-z`) differ, both pure additions on devel. devel-only:
`08`, `15b`, `16`, `run_lens_kb4t.py`, `07_neck_grid.py`, top-level `calib_coverage.py`/`calib_record.py`,
`rig-phantom09.yaml`, tests `test_calib_neck_take`/`moving_split`. **Trap:** devel's `tools/calibration/test_calib_coverage.py`
next to AI-1184's `tests/test_calib_coverage.py` (no `__init__.py`) is a pytest import-file-mismatch: rename one.
Rig files for phantom11/15/21 are not in git (needed by replay): M0 adds them.

## 3. Package layout (`tools/calibration/calibtool/`)

| File | Responsibility |
|---|---|
| `cli.py` | `calib phantomNN [--resume ID] [--replay DIR] [--restore]`: bootstrap the robot runtime (section 10.3), start or attach `calibd`, run the TUI. `--restore` returns the robot to its pre-session state without the TUI |
| `tui.py` | Renders state and sends operator keys only. Persistent header: robot identity, motor state, **NOT NORMAL banner** (e.g. "producer 1280 since 13:02, OKVIS stopped"), abort key always visible |
| `daemon.py` (`calibd`) | Session state machine (section 4); `state.json` + `events.jsonl` after every step; TUI heartbeat; robot-side lock (one calibration job per robot) |
| `profile.py` + `config/robots/phantomNN.yaml` | Host, **machine-id** (identity check; hostnames lie, e.g. phantom14 = `hw-thor01`), producer namespace/deploy, robot.env path, IMU slot, motor tool path + md5, neck joint indices, rig file. Camera queues verified live, never trusted from `head_camera.json` |
| `capture.py` | Camera rings via `dma_ipc` (re-attaches its own consumer on desync before anything else), pair matching by capture stamp, stillness, frame-health counters; one thread, never blocked |
| `detect.py` | AprilGrid detection pool; contrast-stretched copy for finding tags; sub-pixel refinement on the raw image |
| `coverage.py` | FIT targets and TEST targets (section 5), deficits, comparative sentences |
| `place.py` | Closed-loop placement guidance (section 4.1), default placement list, adaptive extras, redo list, stop rule |
| `neck.py` | Pose execution via the fixed `dma_motor_motion` with 08's clamps, all other joints pinned, kp only; per-command foreign-command check |
| `motors.py` | Pre-enable checks, enable after keypress, master/version/fault watch, disable at end (default) |
| `producer.py` | Lease, spec backup, no-overlap restart/resize/exposure, restore on every exit path, post-restart verification |
| `health.py` | Baseline-relative watchers and hardware-suspicion rules (section 8) |
| `solver.py`, `score.py` | Fast KB4T stereo bundle adjustment and fast scorer (section 9.1), plugged in behind `stereofit.solve` / `lensscore.score` |
| `imu.py` | Gravity rotation (2 angles), held-out scoring, T_SC surgery (mkfinal rule) |
| `decide.py` | Paired new-vs-live statistics, deploy-size scaling and verification |
| `deploy.py` | Transactional deploy + rollback (section 6.4), dependent-config finder, consumer compatibility checks |
| `postcheck.py` | Depth check (always), OKVIS check (opt-out) |
| `storage.py` | Session folders, retention, protected paths, disk checks |
| `tests/` | Unit + replay tests; every test file `importorskip`s numpy/scipy/cv2 (one module-level import breaks devel's whole python job) |
| `DEPS.txt`, `tests/devel_compat.sh` | Declared imports from `tools/calibration`; the devel-compatibility check (section 12) |

## 4. Session flow

**0. Preflight** (nothing moves): robot identity (machine-id) and kit version; robot-side lock taken; clock sane (not
1970), k0s API answering; **autonomy not armed** (`nav-cli status`) and no loco/policy pods, else refuse; producer spec,
robot.env, `head_camera.json`, OKVIS image digest saved (= the "previous default" and the rollback snapshot); USB link
speed of the head cameras (USB2 fallback is a hardware fault a restart cannot fix); L/R capture-stamp offset < 0.5 ms;
left/right by disparity sign; frame reads not desynced (re-attach first); IMU queue present (silent IMU: check the head
power switch first); master active, its version and the motor tool's md5/`targets` recorded; `/dev/shm` segment
count; disk free on the target filesystem; robot on the gantry (operator confirms). The robot is **reserved for the
session** by definition (no locomotion policy can run while the tool moves the neck); the tool lists, once, everything
else using the cameras (OKVIS, recorders, VLA) and then restarts the producer whenever it needs to. Then: lease file
written, **OKVIS stopped** for the whole collection (decided 9 Nov: its config would not match 1280); producer to 1280x800 @ 15 (head cameras only where the build supports
per-camera sizes); motors (section 7).

**1. Lens collection: default placements** 0.5, 0.75, 1.1, 1.6, 2.0 m plus one tilted placement at 0.75 m for the image
bottom. **Fixture:** the board stands on its own on a chair-shaped stand (no armrests); nobody holds it during a sweep
(a hand-held board follows the head and adds no edge coverage). **Tilted placement = the board leaning back against the
chair's backrest**, facing up toward the head; it is asked for only if the bottom edge is still short after the
defaults. For each placement:
- **Placement guidance: spoken, pause-then-speak (4.1).** The tool watches the camera continuously (~2 Hz, neck at home)
  but **speaks only once the board has been still for ~1 s**: either "Good. Step back." or **one** short correction
  ("about 20 centimetres further back"); after the operator adjusts and stops, it checks again. No narration while the
  board is being carried; no Enter key. Tolerance is loose for default placements (a capture counts toward the band it
  *measured*, so 1.0-1.2 m is fine for "1.1 m"); it corrects only for the wrong band or a board out of view; adaptive
  extras have tighter targets. Spoken prompts are the default; the TUI screen (thumbnail with board outline and target
  box, measured distance, the same hint as text) is an extra, so the laptop can sit anywhere. Directions name **the
  operator's side** ("to YOUR right = the robot's left"), never "robot's right" alone (misread several times). The
  operator can say/press **C** "can't put it there" (asks for the next-best position and records the region as
  unreachable in this room). In speech the word "hold" is not used (inside the tool it means a still neck pose).
- **Go: spoken countdown.** "Step back. Starting in 5... 4... 3... 2... 1": the sweep starts by itself unless the
  operator aborts (abort key / spoken abort always live). This is the explicit go: the person has been told to step
  away from the head and given 5 s.
- **Grid.** The 49-pose grid (pitch -24..24, yaw -34..34, clamped); corners detected live; per-pose quality (tags,
  brightness at the predicted board region, blur, sync). **Early stop** only when the model predicts the board visible
  and it is not found on most such poses; darkness (low brightness at the predicted region) vs absence is told apart.
  Then D9 levers, retry, or skip into the redo list. A bumped board (its pose drifting across poses) is detected and
  the grid paused.
- **End.** Sound; gap report for that placement in comparative words while the operator moves the board.

**2. Adaptive extras + redo list:** after the defaults, the tool lists what is still short, most important first,
comparing to the best region (*"Left camera, left edge: 176 corners vs 1,099 on the right edge (6x fewer)"*), and
guides the next placement in closed loop. The lever is the **board's position in the room** (distance x bearing),
because edges need the board off-axis beyond what +-34 deg of neck yaw reaches; the pose subset is chosen afterwards.
Stop rule: section 5.

**3. Fit + held-out score:** refit in the background after each grid (live score in the TUI); final fit; new and live
config scored on the same TEST pairs (section 6).

**4. Camera-to-IMU rotation:** "board flat on the floor" -> ~90 s neck sweep -> the two observable angles (section 9.3),
scored on held-out poses against the live config.

**5. Decide** (section 6) -> deploy size (D7) -> producer back to the chosen size -> check grid at that size ->
verdict at that size.

**6. Deploy + post-check** (section 6.4/6.5) -> report -> motors disabled by default (with the warning that the head
and arms drop; only with the operator present) -> lease released, OKVIS restored, banner cleared.

**Abort (key A, any time):** current motion finishes its pose and holds; producer, OKVIS and configs return to the
preflight snapshot; report written. **Link loss / laptop crash:** `calibd` misses the TUI heartbeat (5 s), finishes the
current pose and **holds**; it never starts another grid until the TUI is back and the operator confirms (key G).
**Reboot:** motors are off; on resume preflight re-runs, the motor check and Y are required again, and the board pose is
re-measured before any grid. A lease older than N hours with no TUI (proposal 2 h) makes the robot restore itself.

## 5. Coverage targets, the stop rule and the FIT/TEST assignment

Targets re-based in M1 C3 (9 Nov 2026, decided with the user; `coverage.Targets`, constants `REACHED_*` in
`coverage.py`). The 4 Nov handheld-route targets (`COLLECT-lens-data.md`: corner thirds ~5,000, rim >= 1,500, >= 50 far
pairs, >= 6 entries at 10 TEST holds) were met by **neither** stored neck-grid session, yet both gave the best results
so far (held-out median -0.04 % / +0.04 %, rows 0.28 / 0.21 px). So each target is now **what both sessions reached,
the lower of the two**, counted the way the stop rule counts (`replay.Collection` at the end of the session:
capture-time assignment, labels as its `finish()` leaves them; regions and far pairs over all corners captured, pairs
= FIT pairs), and pair/corner counts are scaled by 2/3 for the protocol's 2 pairs per pose (13.1; protocol results
section 2). Values at 2 pairs per pose, with the stored sessions' 3-per-pose value and its source in brackets:
- **FIT targets** (for both cameras; camera 1's edges matter for the fit even though step 66 judges camera 0 only):
  FIT pairs >= 317 (476: p21 476, p10 557); corners in each image-corner third >= 2,443 (3,665: p21 cam0 top-left;
  p10's lowest 5,814); rim >= 324 (486: p10 cam0; p21 912); far pairs (1.7-3 m) >= 52 (79: p21; p10 127); **bottom edge
  strip** (80 px) >= 2,002 corners (3,004: p10 cam0; p21 3,648), new in C3: the tilted placement's job.
- **TEST targets**: >= 6 step-66 entries (`lenscells` cell x `DIST_BINS`) judged with >= 10 TEST holds (p21 6, p10 7;
  10 = 8 required + margin, because a hold's cell is set against the *nominal* principal point and can move after the
  fit); each **required side (left, right, bottom)** with >= 9 TEST holds in one of its cells, a corner cell counting
  for both its sides (the weakest either session reached: p21 left 9; step 66 needs 8); far bands 1.3-1.7 and
  1.7-2.3 m >= 21 TEST holds each (p21 21 / 21; p10 51 / 25).
- **Asked for, not blocking** (a deficit and its sentence, but the session can end without it): the comparative
  edge/box balance (a strip or box with < 0.5 x its mirror's corners; p21's left edge 176 vs 1,099 drove its `dleft`
  take, yet neither passing session met the rule: p10's right edge 143 vs 408), each single entry short of 10 TEST
  holds, and the **top** side (decision 1: the top does not block a deploy).
- **Assignment at capture time**, stratified per (cell, bin): the tool marks each hold FIT or TEST as it is captured,
  so it can steer the TEST side; each entry's TEST holds come from **at least 2 placements** (holds from one placement
  are correlated with its FIT holds). The split file is handed to the fit (step 16 is the replay reference only).
- **Stop rule** (`Coverage.stop_met`): no blocking deficit, counted with the labels the assignment's `finish()` would
  leave if the session ended now (the split it would write; the placement snapshots and their sentences use the same
  labels). Regions marked unreachable (key C, or a dark strip proven to be the room) are "not judged" with the reason.
- **Replay check (C3):** fed hold by hold through `replay.Collection` (capture-time assignment, 3 pairs per pose), the
  rule is first met at hold 352 of 354 on p21 5 Nov (`dtiltleft`, its last placement, pose 46: 2 poses before the
  end) and at hold 387 of 398 on p10 6 Nov (`n6_far`, its last placement, pose 20: 28 poses, about 1.5 min at 3.13
  s/pose, before the end). By construction (the lower of the two sessions) each session's binding targets are met
  only near its end; the rule is a floor proven sufficient, not a saving. With step 16's split instead of the
  capture-time one, neither session meets it (5 judged entries at 10 TEST holds; p21's 1.7-2.3 m band 19 holds).
  Fit + validate on the data up to each stop point still PASS and deploy (13.0).

## 6. Deciding, deploy size, deploy, post-check

### 6.1 Statistics (one definition each)
- Depth error per TEST pair = stereo depth of the board / board-truth depth - 1; **median |error|** = median of the
  absolute values; per band on one band table: 0.4-0.6, 0.6-0.9, 0.9-1.3 (step-66 bins), 1.3-1.7, 1.7-2.3 m.
- **Paired difference** new - live per TEST hold; 95 % interval by bootstrap over holds clustered by placement.
- Live config scored on the 1280 data: a live config at another size is scaled to 1280 by the inverse of section 6.3;
  a radtan live config goes through `livecfg.radtan_scope` (phantom21); a live config the tool cannot score -> "live
  not scorable": **no deploy** (NOT_ENOUGH_DATA, the error on the screen), because "deploy only if shown better than
  live" cannot be checked. Deploying on the new fit's step-66 PASS alone (LIVE_NOT_SCORABLE_DEPLOY) needs an explicit
  choice (`Gates.allow_deploy_without_live=True`; default False; design change from the M1 review).

### 6.2 Lens verdict
Deploy only if (`decide.verdict`, thresholds `decide.Gates`, fixed in M1 C3 with the user on 9 Nov 2026):
- step-66 PASS on the new fit with >= 6 validated entries, including a judged entry on the **left, right and bottom**
  image edges (`Gates.required_edges`; a corner cell counts for both its edges, `coverage.edges_of_cell`). The **top**
  is asked for during collection but does not block; the reasons say when it is not judged. `required_edges` can be
  set to all four. (p21 5 Nov judged left, right and bottom, no top; p10 6 Nov all four; both deploy.)
- the paired 95 % interval (bootstrap over placements) shows new better than live on median |depth error| **and** on
  row error (both upper ends < 0; pinned by tests: either one not shown better keeps live);
- no judged band where the interval shows new worse; a band is judged from >= 10 pairs from **>= 3 placements**
  (`band_min_placements`, 2 before C3);
- the new fit scored at least 95 % of the pairs the live config scored;
- live scores exist, or `allow_deploy_without_live` was set explicitly (6.1).
Otherwise **keep live** (or NOT_ENOUGH_DATA) and say why.

### 6.3 Deploy size (D7)
"Previous default" = the producer spec saved in preflight. Scaling rule, pinned by a test: focal x s; principal point
**(c + 0.5) x s - 0.5** (matches OKVIS's own resize, `ViParametersReader.cpp:343-357`; the old `final11.sh scaled()`
used c x s, 0.125 px different at s = 0.75); `image_dimension` and resize_factor such that OKVIS's `int(dim x s)`
truncation (`ThreadedSlam.cpp:305`) yields exactly 640x400 for VIO (the test pins the exact output sizes). The "not
sure" screen shows **this robot's facts**: consumers found (dependent-config grep, producer arg history), fps at each
size, the check-grid result, and what would have to be reconfigured for 1280. **Check grid** at the deploy size: a
short default (proposal 15 poses at ~1 m) scored with the 6.1 paired statistics (step 66 cannot be met by it). If the
scaled config fails it: keep live, or offer 1280 if 1280 was verified.

### 6.4 Deploy and rollback: one transaction
Snapshot (taken in preflight, re-checked just before deploy): robot.env (`OKVIS_CALIBRATION`, `OKVIS_DMA_CAMERAS`),
the config file, the saved producer YAML (applied as saved, not hard-coded args), `head_camera.json`, every dependent
config changed, the OKVIS image digest. Before deploy: robot identity again; image digest still the same and reads
`kb4t`; the rig's other consumers can read KB4T (e.g. box_pick needs commit 58e96414 on phantom21-style bakes); OKVIS's
restart path detected (`okvis.sh` agent vs `rollout restart ds/okvis2x`; never `--grace-period=0`); autonomy not
armed. Rollback: first compare the robot's current state with what the tool left; if someone changed it since, ask.
After rollback, verify (config sha in the OKVIS log, producer args, `/dev/shm` segments). **A failed rollback stops
everything and alerts loudly**; no retry loop. If the rollback target itself FAILED on today's data, say so.

### 6.5 Post-check (order fixed by the user)
1. **Depth check, always:** board at one fresh distance; new vs old on the same frames by corner triangulation + row
   error; sync checked first.
2. **OKVIS check, opt-out:** OKVIS up, log shows the KB4T line, depth latency normal; row offset on OKVIS's rectified
   pair, **new vs old by the same method on the same frames** (the rect-pair method read a steady +1.78 px on
   phantom11's good config, so no absolute threshold); network depth on the board new vs old (the network reads ~1.7 %
   short at 1.6 m regardless); floor tilt only after VIO has settled (its gravity wandered 0.3-2.6 deg for minutes after
   restarts). Automatic rollback only on a crash-loop or when new is worse than old by the same method; everything else
   reported for the operator.

## 7. Safety and permissions (built in)

- The tool never starts the motor master; it tells the operator and waits.
- **Before enable:** gantry (operator confirms), autonomy off, no loco/policy pods, `/dev/shm/desired` still over 1 s,
  e-stop false, 48 V bus charged, head power switch on, slave operational count vs a preflight baseline, no per-joint
  fault latch (e.g. phantom11's 0x7300). Operator presses **Y**; the tool enables, then checks `dma_cmd_client ping`
  motor_state and the journal's "N of N motors enabled" (`state` prints zeros on the 2.1.0 master).
- **During motion:** before every own command, `/dev/shm/desired` must be still for >= 0.5 s while no own command runs
  (the tool's own writes move it, so a single start-time check is not enough); loco/policy pods appearing, e-stop, the
  master going inactive or changing version, a fault latch, rising tracking error or motor temperature -> stop and hold.
- Neck only through the fixed `dma_motor_motion` (2.1.1 zeroes unlisted joints), 08's clamps, every other joint pinned,
  **kp only, never kd**. Safe state = **hold** (08's behaviour); disabling drops head and arms, so it is done only at the
  end, with the operator present.
- An explicit go before every grid: the spoken step-back + 5 s countdown with abort live (the operator stands next to
  the head between grids); abort key always live; heartbeat deadman;
  no motion resumes by itself.
- Never touches loco/the walker; never pauses the fleet recorders without a prompt; never deploys without the section-6
  verdict; kills only its own children by PID (a `pgrep -f` pattern also matches the agent's own ssh shell).
- Producer restarts whenever needed (the robot is reserved for the session); after every restart: segment count, frame size, sync,
  disparity sign re-checked.

## 8. Health watch and hardware suspicion (baseline-relative, v1 short list)

| Signal | Action |
|---|---|
| Frame reads None | re-attach own consumer; if it persists, producer restart, counted |
| Frame freeze (stamps stop, same frame repeats), fps drop **not explained by CPU load** | restart; again within 10 min -> hardware suspicion |
| Head-camera USB link below SuperSpeed; `dmesg` reset **on the head camera's port** (filtered; phantom11's USB-ethernet dongle is not a camera); producer log "Device crashed" | hardware suspicion, quoting the lines; no restart loop for a link-speed fault |
| L/R offset > 0.5 ms | restart once; persists -> refuse to collect |
| Slaves leaving OP **vs the preflight baseline** (46/50 can be normal); fault latch; e-stop; master inactive | stop, hold, hardware suspicion |
| Someone else changes the producer or the queue mapping | stop collecting, tell the operator |
| boot id changed; clock 1970; k0s refused | say so; preflight again; motor check + Y again |

Message form: *"3rd producer restart in 20 min; USB reset on the head camera's port at 15:41 each time -> suspect the
head camera cable or hub, not software."* All events go into the session report.

## 9. Algorithms

### 9.1 Fast solver and scorer
Same model and parameters as `stereofit.py` (per camera fx fy cx cy k1..k4 p1 p2; stereo pose; **one board pose per
stereo pair**), and everything around it carried over: PnP start with failed pairs dropped, stereo-pose-only presolve
(`trim_px=inf`), `warm_fit`, trimming 3 px x 4 rounds with success only if every round converges, restarts,
`x_scale="jac"`-equivalent scaling, residual clamps for non-finite/behind-camera points with a zero Jacobian there.
Method: Levenberg-Marquardt, exact Jacobian (analytic KB4T derivatives, checked against finite differences), board
poses eliminated per iteration (6x6 blocks, 26-parameter reduced system). Plugged in behind `stereofit.solve` (a flag),
so 45/66/`run_lens_kb4t` use it unchanged. The scorer (`lensscore.score`, `solve(scale=True)` with the camera fixed)
is per-pair independent: vectorised/parallel. **Equivalence acceptance** (scipy reference run to tight tolerance):
max projection difference over an image grid at 0.5/1/2 m < 0.02 px per camera; stereo pose and baseline within
0.01 mm / 0.001 deg; per-entry step-66 medians within 0.02 %; identical verdicts except entries within 0.02 % of a
threshold (listed, not failed).

### 9.2 Placement model
From the first frames at the home pose: board pose; with the measured joint angles (kp-only control sags) and the
camera-in-Head from the URDF chain (refined by step 81 after the first default grid), predict which (cell, bin) and
which FIT regions each neck pose fills. Adaptive extras choose a board position from a discrete set (distance x bearing
the operator can carry out), then the pose subset. **M1 tests:** predicted vs measured board centroid per pose on stored
takes; counterfactual: with phantom21's `dleft` and `dtiltleft` takes removed, the planner must ask for those
placements.

### 9.3 Camera-to-IMU
- **v1, rotation, gravity:** board flat on the floor, IMU slot means per still pose (`imucam_fast.py`, ported into
  `neck.py`/`imu.py` without its robot-only import) + `rscfit.py`: **2 observable angles** (about camera x and z). The
  third angle (about the gravity direction) is unobservable from floor sweeps: neck yaw turns about a near-vertical
  axis and changes neither gravity in the IMU frame nor the floor normal in the camera frame; only a wider pitch span
  helps the other two. The live value of the third angle is kept and the report says so. Scored on **held-out poses**
  (fitting on the same poses would make "residual improves" circular); gate from the fit's combined sigma.
- **T_SC surgery:** `mkfinal.py`'s rule (cam0 T_SC rotated, cam1 = cam0 x stereo pose, translation and T_BS kept); 83
  is not used (it writes T_BS). Changing T_SC with T_BS kept moves camera-in-Head for other consumers (e.g. camera-to-arm):
  the report flags it.
- **Reported, not deployed, in v1:** camera-in-Head X from step 81 on the lens grids (needs `--urdf`; X is in the URDF's
  Head frame, so any CAD IMU position must be given in that frame); translation and time offset.
- **v2 (section 14):** the third angle and a camera<->IMU time offset need the gyro route: full-rate IMU + joint capture
  in `capture.py` (82 reads record-app mcap/EuRoC today), sweeps faster than 0.3 rad/s (imucam ramps run 0.12-0.22),
  and a camera<->IMU lag method (82's lag is joints<->IMU, +-50 ms in 2 ms steps).
- IMU noise values: fixed per IMU model, not per robot.

## 10. Concurrency, compute, runtime, speed budgets

### 10.1 Concurrency
Capture thread (never blocked); detection in a process pool sized to the robot's **free** CPU (phantom11 ran load
20-25 with OKVIS + recorders) and shrinking under load; fit/scoring in its own process; neck, health, TUI link in
threads. Until the fast solver lands (M3) the scipy fit runs in the background during grids. Laptop: rendering only.

### 10.2 Compute placement
Images stay on the robot: a 49-pose grid is ~500 MB of raw pairs; ~20 s at the 25 MB/s measured Thor->Moonstone,
~3.5 min at the 2.3 MB/s measured over phantom21's WiFi. Only corners (KB) would ever travel.

### 10.3 Robot runtime
The OKVIS image has only `python3-numpy` and `dma_ipc`; robot hosts have no OpenCV and no system pip; past kits used
hand-made venvs (`/root/p11-cal/venv`, `pydeps`). The tool needs SciPy, OpenCV with aruco/AprilTag, PyYAML and textual
on aarch64 without building images on Thor. Decided (9 Nov): a CI-built `calib` image per arch, pulled like other
images.

### 10.4 Budgets (targets, to be measured)
| Step | Measured today | Budget |
|---|---|---|
| Neck grid, 49 poses | ~2.5 min (~3.1 s/pose) | <= 2.5 min; adaptive extras ~10-15 poses |
| Placement (operator) | not measured (phantom21: 1 h 15 min session, mostly placements and re-placements) | measured in M2's timed trials; target < 1 min each |
| Corner detection | a separate step after recording | done within one pose time |
| Fit | 274 s KB4T on Thor (phantom21) | < 30 s |
| Scoring (step 66 + self-check, new + live) | several minutes (`solve` per score) | < 30 s |
| Camera-to-IMU sweep + fit | 87 s + ~30 s | same sweep, fit < 10 s |
| Session, motors-on to live | 1 h 15 min (phantom21) | measured in M5; target < 40 min |

A step that misses its budget after the algorithm work gets, in order: JAX/Numba JIT, Ceres via its Python binding, a
small C++ module.

## 11. Data management

- One folder per session: `/data/calib/<session-id>/{raw,corners,fit,report,deploy}`; raw stored lossless-compressed.
- Kept forever (MB): corners, fits, reports, deployed configs, health log.
- Raw (GB): kept for the session behind the **live deployment** and the last 2 **completed-deploy** sessions (aborted
  and read-only sessions do not count toward the 2); older deleted.
- **Protected paths, automatic:** anything a `rollback.sh` or the tool references (e.g. `/root/p21-cal/dma_motor_motion`,
  `/root/p21-cal/backup/dma-pre-210-*`, `/root/p07-stack`, `/root/p09-*-backup*`) and the M1-M3 datasets
  (`/root/p{11,21}-cal/takes`) until archived.
- Free space measured on the filesystem the session writes to (`/data` is shared with the recorders); preflight refuses
  below a minimum (proposal 20 GB) and capture stops if it falls below 5 GB.
- Legacy kit cleanup and archiving to Moonstone: v2.

## 12. Branches, devel PR, CI

- Development on `feat/calibration-agent-calib-tui` (off AI-1184); merge into AI-1184 on the user's word.
- `tests/devel_compat.sh` copies `calibtool/` + `DEPS.txt` modules onto a devel worktree (under `~/wt`, never /tmp,
  `GIT_LFS_SKIP_SMUDGE=1`, reused between runs) and runs the tool's tests there.
- Release: `feat/calibration-agent-calib-tui-devel` off devel, one reviewed commit (package + the modules devel lacks,
  e.g. step 81 math), PR into devel. Python only; no OKVIS C++ changes.
- **CI:** devel's `python-tests` installs `pytest pytest-xdist` only and collects `tools/`, so calibration tests skip
  via `importorskip`; `pytest.ini` has `-n auto`. M6 ports AI-1184's `calibration-tests` job (numpy, scipy,
  opencv-headless, pyyaml, pillow, mcap, aruco check) to devel and adds `pytest-xdist`.

## 13. Milestones (ordered by risk; each gets its own TDD plan when the user starts it)

| M | Deliverable | Acceptance (measured, logged) |
|---|---|---|
| M0 | Neck-grid pipeline onto this branch (the 2 differing files + devel-only files; the `test_calib_coverage` clash renamed); rig files for phantom11/15/21; `DEPS.txt`, `devel_compat.sh` | AI-1184's and devel's calibration suites both pass here; `kb4t_reference.json` test passes |
| M1 | `calibd` in **replay** mode: capture-time FIT/TEST assignment, both target sets, gap sentences, placement model + planner, decide statistics, scaling rule + test | Reproduces the hand-reached verdicts (phantom21 5 Nov: new PASS 7/7 vs live FAIL 2/6; phantom11 6 Nov: new PASS 9 vs 5 Nov config FAIL); placement-model centroid error reported; phantom21 counterfactual (drop `dleft`/`dtiltleft`) makes the planner ask for them; poses-per-target measured and targets set to fit the time budget; **collection protocol measured (13.1)** |
| M2 | Robot runtime (10.3), `capture`, `producer`, `motors` (checks only), `health`, `storage`, `tui` skeleton; **read-only live run** + **timed closed-loop placement trials** with the neck at home (no motion) | Preflight all green on one robot; producer snapshot/restore exercised; >= 5 placements, success rate and seconds per placement logged against the budget |
| M3 | `solver.py`, `score.py` | Section 9.1 equivalence on phantom21 (8 takes), phantom11 (`n6_*`), phantom15 (corners on Terminus `<user folder>/p15cal/out`); fit and scoring times on Thor vs the budgets (can run in parallel with M1/M2) |
| M4 | `imu.py` (v1: 2 angles, held-out scoring, T_SC surgery; X/translation/time offset reported) on stored sweeps | phantom11 6 Nov sweep: rotation reproduced vs the deployed `..._1280_imu.yaml` values; held-out residual vs live reported; phantom21's -16 deg fix re-scored if its data allows |
| M5 | `neck`, `deploy`, `postcheck`, the full TUI; one full session on a robot with the user | **Pass:** deploy decision correct by section 6, rollback exercised once on purpose and verified, abort/heartbeat/reboot paths each exercised once, timings vs 10.4 logged. **Fail** otherwise |
| M6 | devel PR branch + CI job | devel CI green with the tool's tests actually running (not skipped) |

### 13.0 Status

- **M0 done (9 Nov 2026):** devel's neck-grid pipeline imported (15 added files, `15_dump_corners.py` +95 and
  `45_fit_lens.py` +5, pure additions; nothing of AI-1184 removed); devel's `test_calib_coverage.py` renamed
  `test_calib_coverage_stream.py` (clash with AI-1184's `tests/test_calib_coverage.py`); rig files for phantom11 (now
  phantom10; files keep the old name), its 960 variant, phantom15, phantom21 (host fixed: it carried phantom11's);
  `calibtool/` with `DEPS.txt`, `tests/test_deps.py`, `tests/test_rigs.py`, `tests/devel_compat.sh`. Calibration suite
  on Moonstone (CI-mirroring venv): **488 passed, 2 skipped (mujoco), 55 s**. devel_compat on devel da8b1ae6: devel lacks
  4 DEPS modules (`81_tbs_handeye.py`, `handeye_px.py`, `handeye.py`, `neckfk.py`), calibtool's tests pass there.
- **devel PR #930 (merged 9 Nov):** OKVIS now stops at startup when the calibration does not match the cameras
  (C++: `ViParametersReader`, `Stereo2DepthProcessor`, `ThreadedSlam`). The deploy step (M5) must satisfy those checks
  on robots whose image carries #930, and the deploy-size scaling (6.3) must produce a config they accept.

- **M1 C3 (9 Nov 2026), thresholds and protocol decided with the user:** sections 5 and 6.2 (required edges left,
  right, bottom; targets = the lower of what both stored sessions reached, x 2/3 for 2 pairs per pose, plus a bottom
  edge strip target; stop rule; band_min_placements 3; no deploy without live scores unless explicitly allowed), the
  planner gives one suggestion per top deficit (the p21 counterfactual now asks for the tilted board for the bottom
  strip), the protocol's defaults are constants (`protocol.py`: per_hold 2 from 3 frames, far placement 1.8 m asked /
  1.7-2.0 accepted / > 2.1 refused, pose lists 49/41/44, hold ends at capture), robot identity by machine-id
  (`calibtool/config/robots.yaml`, `robots.identify`). Replay (capture-time assignment): full data DEPLOY on both
  (p21 new 0.255 % median |depth error|, rows 0.204 px vs live 0.790 %, 0.833 px; p10 0.364 %, 0.283 px vs 1.997 %,
  3.969 px); cut at the stop rule (p21 hold 352/354, p10 387/398) still converges, PASSes step 66 (8 / 10 entries) and
  DEPLOYs (p21 0.254 %, 0.204 px; p10 0.356 %, 0.283 px). With all four edges required p21 is NOT_ENOUGH_DATA (no
  top entry). Calibration suite 746 passed, 2 skipped; replay suite 19 passed.

- **9-10 Nov 2026: M2-M5 built, first live sessions on phantom10 (P10).** M3 fast solver (fit 15-23 s vs 129-296 s on
  Moonstone; 23 s vs 444 s on phantom10's Thor CPU; identical verdicts), now the calibtool default; settle-trim (6-7
  rounds, cap 12); the final fit is the last live refit when it covers the final split. M4 IMU v1 (2 angles; the third
  is weakly observable, sigma ~0.08 deg). M5: neck grids, motor enable/takeover, deploy transaction + rollback,
  post-check, full-session wiring (`calib <robot> --full`), stale-lease self-restore (not installed). Live: placement
  trials (9-10 Nov) and the first full session (10 Nov 12:32, `20261110-123252-P10`, PAUSED in HOLD during the tilted
  grid): 5 placements, 225 poses in 30 min, live refit stable at 0.33-0.37 px rms / baseline 44.2 mm; deploy not
  reached. Fixes from operator feedback are in the code (short slow sentences, natural Piper voice, feedback wording,
  no automatic skips or time limits on P10, reasons when the board is not seen, spoken grid/hold progress, snake pose
  order). Full state, open issues and the successor prompt: `tools/calibration/calibtool/HANDOFF.md`.

### 13.1 Collection protocol, measured in M1 (so placements rarely need repeating)

Past re-measures came from placement/visibility problems found only after a grid was processed (a "0.5 m" board at
0.7 m, a board not square to the head, a room-dark strip, ~27 px tags at ~2.15 m with 22/49 poses detected, the upright
board too edge-on for the image bottom, thin left edge / bottom-left needing two extra takes on phantom21). Live
detection, distance measured before the sweep, early stop and the immediate gap report address finding them late; M1
fixes the protocol itself from the stored phantom10 (ex-phantom11) and phantom21 takes:
1. **Pose list per distance:** for each default distance, which of the 49 poses actually detect the board (enough tags)
   and add coverage; drop poses that never see it there (e.g. extreme yaw at 0.5 m), keep or densify those that fill
   edges. Output: one pose list per distance, with its predicted coverage.
2. **Frames per pose:** today 5 pairs per still pose; measure the corner-noise and fit/TEST effect of 1-5 pairs per
   pose; use the smallest number with no measurable loss.
3. **Far limit:** detection rate and corner noise vs distance; cap the far placement (~1.8 m, unless the data shows
   2.0 m works with this board).
4. **Poses per coverage target:** how many poses each FIT/TEST target costs, so the default set meets the targets
   without extras in a typical room.
Result: each placement's duration and the whole session's expected length become measured numbers, told to the
operator at the start ("about N minutes, M placements").

Copying stored datasets off the robots (M1, M3, M4) needs robot access and is done only when the user allows it.

### 13.2 Link to a future calibration health monitor (separate project, not in this tool)

A later OKVIS-side monitor could watch each robot's stereo rows from the left/right keypoint matches OKVIS already makes,
and alert when a robot drifts (phantom10's 5 Nov config read 18 px rows and nobody knew). Absolute row values differ by
method (the rectified-pair method read a steady +1.78 px on phantom10's fresh config), so such a monitor must compare
against a **reference taken right after calibration by the same method**. This tool therefore records, at every deploy
(M5's post-check already measures it): the row-health reference = method, median row offset, its pattern across the
image (left/centre/right thirds, edges), the frame conditions (head still, L/R sync offset), next to the deployed config
and in the session report. Nothing else of the monitor is in scope here; its own plan (monitor read-only first, fleet
data on what drifts, only then any bounded, logged, reversible pitch/roll correction) is future work. Note for that
plan: phantom10's 5->6 Nov change differed by 0.19 deg in stereo rotation **and** 44.99 -> 44.06 mm baseline between the
two fits, which a rows-only correction would not fix.

## 14. Deferred to v2

`--compute auto|thor|moonstone`; the gyro route (third rotation angle, camera<->IMU time offset); translation
deployment (needs Q1); legacy kit cleanup and raw-data archiving; hardware heuristics beyond section 8's list;
handheld moving-board mode.

## 15. Review log (rev 1 -> rev 2, 9 Nov 2026)

Corrections of rev-1 claims:
- "Neck-yaw poses make the third rotation angle observable": **wrong** (physics; `rscfit.py` already says so). v1 fits 2 angles.
- "The URDF is not used": **wrong**; step 81 requires it for the neck chain, and X is in the URDF's Head frame.
- "One board pose per hold": **wrong**; one per stereo pair.
- "Step 82 gives the time offset": it gives joints<->IMU lag at 2 ms steps, not camera<->IMU.
- "Coverage bands 0.4-2.3 m judged per cell": step 66 judges per cell only to 1.3 m.
- "devel CI installs only pytest": it installs pytest + pytest-xdist and skips calibration tests via `importorskip`;
  AI-1184 already has a `calibration-tests` job.
- 0.2 deg IMU agreement gate and "residual improves on live": unmeetable / circular; replaced (9.3).

Added from the reviews: closed-loop placement with operator-side directions, sound cues, a "can't put it there" key
and an explicit G per grid; FIT and TEST target sets with capture-time stratified assignment from >= 2 placements and a
realistic stop rule with edge entries; heartbeat deadman, abort key, no self-resuming motion, hold as safe state,
per-command foreign-command check, master/version/fault watch, fuller pre-enable checks, motors disabled at the end by
default; producer lease + `--restore` + self-restore, reserved/shared choice, OKVIS stopped during collection,
post-restart verification; one-transaction deploy/rollback with drift check and loud failure; robot identity by
machine-id; consumer compatibility at deploy; same-method new-vs-old post-checks; baseline-relative health rules;
protected paths; robot runtime question; scorer acceleration; stronger solver equivalence; scaling truncation pinned;
milestones reordered by risk (replay and placement trials before the solver); v2 deferrals.

## 16. Open questions for the user

- ~~Q1~~ Answered 9 Nov: **measure** the IMU lever arm (URDFs have no IMU link): a brisk neck oscillation (~+-10 deg at
  ~1 Hz; ~0.35 m/s^2 at a 5 cm arm, back-of-envelope) read by the accelerometer against encoder angles/rates gives the IMU
  position in the Head frame, the same frame as step 81's camera X; translation = their difference. One v2 "wiggle
  sweep" with full-rate IMU + joints also yields the gyro third angle and the time offset.
- **Q2** Thresholds (section 6, 9.3) and disk minimums: fixed in M1/M4 from past sessions, then shown to you.
- ~~Q3~~ Answered 9 Nov: the robot is always reserved during calibration; no shared mode.
- ~~Q4~~ Answered 9 Nov: `root@phantom10` = the robot previously named phantom11 (P10), renamed.
- ~~Q5~~ Answered 9 Nov: preflight runs a 30 s 1280 test and falls back in order: (1) head cameras at 1280, others at
  their own size; (2) other cameras off during collection (phantom15's crash was at 1280 with 4 cameras running);
  (3) lower fps for USB2 robots (two 1280x800 streams at 15 fps need ~46 MB/s, above USB2's practical ~35-40 MB/s;
  holds are still, so ~5-8 fps suffices); (4) collect at the live size, last resort.
- ~~Q6~~ Answered 9 Nov: OKVIS is stopped during collection.
- ~~Q7~~ Answered 9 Nov: a CI-built calib image (10.3 option a).
- ~~Q8~~ Answered 9 Nov: a chair-shaped stand without armrests; the tilt = board leaning against its backrest.
- ~~Q9~~ Answered 9 Nov: spoken prompts by default, so the laptop's position does not matter.
- ~~Q10~~ Answered 9 Nov: v1 keeps the third rotation angle from the live config; the gyro route stays in v2, to be
  revisited only if VIO shows yaw-correlated drift on turns.
