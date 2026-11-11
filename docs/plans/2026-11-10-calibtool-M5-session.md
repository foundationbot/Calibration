# calibtool M5 — the session's moving and deploying parts (code against fakes) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:test-driven-development for every task (test written and
> seen failing, then code, then green). Steps use `- [ ]`. Same format as the M1/M2 plans
> (`2026-11-09-calibtool-M2-live.md`): each task names its file, its API and the exact tests; the code is written
> test-first in the task. One agent executes A -> E in order (C uses A; D/E are independent of A-C).

**Goal:** the five modules a full calibration session still lacks, each a clean API the daemon drives: the neck grid
through the fixed `dma_motor_motion` (`neck.py`), the motor enable/disable step after the operator's Y
(`motors_enable.py`), the per-placement collection controller (`grids.py`), the deploy transaction with its rollback
and the deploy-size choice (`deploy.py`), and the post-deploy checks (`postcheck.py`); plus the wiring notes the lead
needs to add the daemon phases GRID, IMU_SWEEP, DECIDE, DEPLOY_SIZE, DEPLOY, POSTCHECK.

**Architecture:** every robot action goes through an injected runner (`run(argv, timeout) -> (rc, text)`) or an
injected callable, so each module is tested with fakes and has a dry-run mode that records the exact commands it would
send. No module imports a robot-only module (`dma_ipc` is passed in, as in `capture.DmaCameras`). The modules do not
know the daemon: they take callables (`should_stop`, `on_event`) and return dataclasses; the daemon (`daemon.py`, owned
by the lead) maps them to phases, keys and events (section "M5 wiring notes").

**Tech stack:** Python 3.12 stdlib, NumPy, OpenCV (postcheck's row features only), PyYAML (producer spec); tests with
pytest + xdist on Moonstone only (`tests/run_moonstone.sh <wt> m5 tools/calibration -q -n 4`; at most 4 processes).
Every new test file `importorskip`s numpy (and cv2/yaml where used).

Design: `2026-11-09-calibration-terminal-tool.md` sections 4 (session flow), 5 (targets, stop rule), 6 (decide,
deploy size, deploy transaction, post-check), 7 (safety), 8 (health), 11 (data), 13 (M5 row), 13.1 (protocol), 13.2
(row-health reference).

**Tonight's rule (9-10 Nov):** nothing moves, enables, deploys or changes on any robot; no robot access at all, not
even read-only. Everything below is built and tested against fakes; the robot acceptance (M5 row of section 13) is a
later session with the user.

**Not edited here** (another agent edits them tonight): `place.py`, `detect.py`, `daemon.py`, `tui.py`, `cli.py`.
M1/M2 modules are only imported.

---

## Facts every task relies on

- **Motor tool** (step 08, `scripts/08_neck_grid_direct.py`; the phantom12 powered-neck notes): the fixed build
  answers `--help` with `targets`; a move is
  `<tool> "NeckPitch=<p>,NeckYaw=<y>,<pins>" targets position --ramp <s> --hold <s> --no-return [--kp <Nm/rad>]`;
  every listed joint ramps to its target, unlisted joints hold their measured pose (2.1.1's tool zeroes them: the
  reason for the fixed build), defaults Kp 30 / Kd 3; with `--no-return` the master keeps holding the last pose after
  the tool exits. It prints `RESULT targets joint=<i> target_deg=<t> ss_err=<e> ...` per listed joint and exits 1
  when a hold misses its 1 deg tolerance (expected under gravity sag: the tool's rc is not the verdict, 08's
  `--max-err` 3 deg on the neck's ss_err is). `+NeckPitch` = head down, `+NeckYaw` = head left.
- **08's safety rules**, ported as they are: refused without `targets` in `--help`; refused when `/dev/shm/desired`
  advances over 1 s before the grid; every command pins the arm, spine and leg joints (names from `/dev/shm/config`,
  prefixes `PIN_PREFIXES`) at their angles **at the grid's start**; pitch clamped to +-26 (limit +-27.5; +26 commanded
  reached +27.7 on phantom15), yaw to +-34 (a stop at 35.9 on phantom14); stop at the first pose a neck joint misses
  by > 3 deg. `/dev/shm` rings: header `<QQ` entry size/capacity at 16, head `<Q` at 64, entries from byte 128;
  `actuals` entry: 29 joint angles `<29f` (rad) at byte 88; `config` entry: joint names as CamelCase words.
- **Design 7 adds**: before EVERY own command `/dev/shm/desired` must have been still for >= 0.5 s while no own command
  ran (the tool's own writes move it); kp only, never kd; safe state = hold (no command: the master keeps the pose).
- **Protocol** (`protocol.py`): `CAPTURE_FRAMES` 3 frames per pose, `PER_HOLD` 2 pairs kept (15b's still rule:
  `protocol.still_frames` + `per_hold_slice`), `HOLD_ENDS_AT_CAPTURE`: the hold ends once the frames are in;
  `poses_for(distance, tilted)` -> indices of `planner.grid()` (pitch-major 49 poses).
- **Enable** (design 7, the phantom12 notes): `dma_cmd_client enable-motors` / `disable-motors`; after enable,
  `dma_cmd_client ping` motor_state != 0 and the master's journal (`journalctl -u dma-ethercat`) says
  "N of N motors enabled" (`state` prints zeros on 2.1.x). A latched FF10 on every drive with ~3 V bus = motor power
  off (e-stop in). `motors.py` stays read-only (its test forbids enable/disable there).
- **OKVIS on the robot** (`deployment/RUNBOOK.md` 4a): per-robot values in `/etc/phantomos/okvis/robot.env`
  (`OKVIS_CALIBRATION=/etc/okvis/<file>` = host `/etc/phantomos/okvis/<file>`, `OKVIS_DMA_CAMERAS`); restart =
  `k0s kubectl -n positronic rollout restart ds/okvis2x`; the older robots run `/data/okvis-perf/deployment/okvis.sh
  restart` (kit `deploy1106.sh`, `p11_1280.sh`); **never `--grace-period=0`**. The entrypoint logs
  `[entrypoint] config: <file> ... (<sha256 first 12>)`; the reader logs `KB4T pinhole camera <i> with T_SC=...`.
- **devel PR #930**: OKVIS stops at startup unless (a) every pinhole camera's `distortion_type` is one of
  equidistant 4, kb4t 6, radialtangential 4, plumb_bob 4, radialtangential8 8, plumb_bob8 8 coefficients, exactly,
  and (b) frames arrive at the yaml's `image_dimension` (before resize_factor). The deploy refuses a config that would
  fail either at the producer size it deploys with.
- **Kit deploy scripts**: back up robot.env, write `rollback.sh` before changing anything, `install -m 0644` the
  config, `sed` the `OKVIS_CALIBRATION` line, restart; the producer through the no-overlap cycle
  (`producer.restart/resize/restore`).

## File map (all in `tools/calibration/calibtool/`)

| File | Responsibility | Test |
|---|---|---|
| `neck.py` | clamps, pins, the motion argv (kp only), `/dev/shm` reader, `Neck` (arm refusals, per-pose foreign check, move, measured angles after settle), `run_grid` (abort/hold between poses, hold ends at capture), dry-run backend, `NeckSweepRig` (imusweep.SweepRig), `DmaImu` | `tests/test_neck.py` |
| `motors_enable.py` | pre-enable checks, `Enabler.enable(key)` / `disable(confirmed)`, journal parse, verification | `tests/test_motors_enable.py` |
| `grids.py` | `GridRun` (one placement: neck grid + capture + detection + CaptureHold -> Collection, early stop, bump watch, snapshot + gap sentences), `EarlyStopRule`, `BumpWatch`, `Collector` (placements, stop rule, redo list) | `tests/test_grids.py` |
| `deploy.py` | `Remote` (runner + dry run + forbidden flags), robot.env edits, #930 checks, OKVIS restart path, `DeploySnapshot`, drift check, dependent-config finder, `rollback.sh`, `Deployer.deploy/rollback`, deploy-size options + check-grid verdict | `tests/test_deploy.py` |
| `postcheck.py` | depth check (sync first, new vs old on the same frames), row features (`rowfeat`), OKVIS check (log lines, latency, rect rows new vs old, floor tilt), crash loop, the auto-rollback rule, the row-health reference | `tests/test_postcheck.py` |

---

## Phase A — neck (`neck.py`)

API: `MAX_PITCH 26.0`, `MAX_YAW 34.0`, `RAMP_S 1.0`, `SETTLE_S 0.8` (passed as the tool's `--hold`: the tool exits once
the neck has settled; the master keeps the pose), `FOREIGN_STILL_S 0.5`, `START_STILL_S 1.0`, `MAX_ERR_DEG 3.0`,
`STILL_DEG 0.1`; `clamp(p, y) -> (p, y, clamped)`; `pins(names, q_deg)`; `neck_errors(text, idx)`;
`motion_argv(tool, p, y, pins, ramp, hold, kp=None)`; `ShmRobot(shm)`: `desired_head()`, `joints() -> (names, deg)`;
`Neck(tool, run, robot, tool_info, clock, sleep, kp=None)`: `arm()`, `goto(p, y) -> PoseDone`, `commands`;
`run_grid(neck, poses, on_pose, should_stop) -> GridOutcome(done, stopped)`; `DryRun(names, q0)` (`.run`, `.robot`,
`.commands`); `NeckSweepRig(neck, imu_accel, frames)`; `DmaImu(d, slot)`.

### Task A1: clamps, pins, argv
- [ ] Tests: `clamp(30, -40)` -> `(26, -34, True)`; `clamp(-8, 11)` unchanged, False; `pins` pins every arm/spine/leg
  name and not the neck or ankles, two decimals; `motion_argv` = 08's argv, `--kp 60` appended only when kp given;
  `kp=0` or negative refused; no argv ever contains `--kd` (and `motion_argv` has no kd parameter: signature test);
  `neck_errors` reads 08's RESULT lines for the neck joints only.
- [ ] Implement.

### Task A2: `/dev/shm` reader
- [ ] Tests (fake ring files written in tmp_path with the UEUQ layout): `desired_head` reads `<Q` at 64; `joints`
  reads names from `config`'s newest entry and 29 angles (deg) from `actuals`' newest entry at byte 88.
- [ ] Implement `ShmRobot` (08's `ring`/`latest`/`joint_names`/`actuals_deg`, path injectable).

### Task A3: `Neck.arm` refusals
- [ ] Tests (FakeRobot: scripted desired head and joints; fake runner): refused (`NeckRefused`, nothing run) when the
  tool has no `targets`, when `/dev/shm/desired` advances during 1 s, when NeckPitch/NeckYaw are missing from the
  names; `goto` before `arm` refused; `arm` pins at the start angles (a later change of the arm angles does not change
  the pins).

### Task A4: `Neck.goto`
- [ ] Tests: the argv sent = clamped targets + the start pins + `targets position --ramp 1.0 --hold 0.8 --no-return`;
  before each command the foreign check needs `desired` unchanged for >= 0.5 s since the end of the previous own
  command (sleeps only the remainder; a head that moved since -> `ForeignCommand`, nothing run); a neck ss_err > 3 deg
  or a missing RESULT line -> `NeckMissed` (the pose is held: no further command); the tool's rc 1 with good RESULT
  lines is fine; the measured angles are read after the tool exits, until two reads agree within 0.1 deg (bounded);
  `PoseDone` carries cmd, clamped flag, measured (pitch, yaw), errors, the command's seconds.

### Task A5: `run_grid` (abort / hold / hold ends at capture)
- [ ] Tests: `on_pose(done)` is called after each pose and the next command waits for it (the hold ends when it
  returns: the capture); `should_stop()` returning "abort" or "hold" before a pose stops the grid between poses with
  no further command (`GridOutcome.stopped`), the last pose kept; an exception from `goto` stops the grid
  (`stopped="error: ..."`), no command after it.

### Task A6: dry run, SweepRig, IMU
- [ ] Tests: `DryRun` records the exact argv of a 3-pose grid, its RESULT lines say ss_err 0 and its measured neck
  follows the command; `NeckSweepRig` satisfies `imusweep.capture_sweep` (a full fake sweep through it: commands
  recorded, `foreign_seq` = desired head); `DmaImu.accel(seconds)` against a fake `dma_ipc` (`RawImuQueue`,
  consumer `available/pop`, `imu(slot).accel_*`) returns (k, 3).
- [ ] Commit A.

## Phase B — motor enable (`motors_enable.py`)

API: `ENABLE_ARGV`, `DISABLE_ARGV`, `DISABLE_WARNING`; `parse_enabled(text) -> (n, of) | None`;
`pre_enable_checks(s, baseline) -> [preflight.Check]` over a state dict: `gantry` (operator flag), `autonomy` (text or
None = no nav-cli), `pods`, `desired_still`, `master` (`motors.MasterStatus`), optional `bus_v`, `head_power`,
`slaves_op`, `fault_latches` (None = not available on this master: reported, not blocking);
`Enabler(run, probe, clock, sleep, dry_run=False)`: `checks(gantry)`, `enable(key, gantry) -> EnableResult`,
`disable(confirmed) -> DisableResult`, `commands`.

### Task B1: checks
- [ ] Tests: all good -> every check ok; each of: no gantry confirmation, autonomy armed, a policy pod, desired moving,
  master inactive, e-stop engaged, e-stop unknown, motors already on, bus below 44 V, head power off, fewer slaves in
  OP than the baseline, a fault latch -> that check fails with a plain detail; unavailable optional probes -> "info"
  "not available on this master", not blocking.

### Task B2: enable / disable
- [ ] Tests: key other than "Y" -> refused, no command; any failed check -> refused, no command (parametrised over
  every check); all ok + Y -> exactly `dma_cmd_client enable-motors`, then verified by ping motor_state != 0 and the
  journal's "N of N"; "48 of 50" -> `EnableResult.ok` False with the numbers (no automatic disable: the operator
  decides); `disable(False)` refused (the warning text returned), `disable(True)` runs `disable-motors` and verifies
  motor_state 0; dry run records the commands and runs nothing; a source scan: `motors_enable.py` is the only
  calibtool module naming `enable-motors`.
- [ ] Commit B.

## Phase C — grids (`grids.py`)

API: `PoseRecord`, `EarlyStop(reason in {"dark", "absent", "bumped"}, sentence, poses_judged, misses)`,
`EarlyStopRule(min_judged=6, miss_frac=0.5, dark_level=40)`, `BumpWatch(px=60, run=3)`, `GridResult(placement,
records, holds, stopped, early_stop, snapshot, sentences, redo)`, `RedoItem(placement, reason, poses)`;
`GridRun(neck_grid, grab, detect, collection, placement, poses, predict=None, hold_ids, per_hold, frames,
should_stop, on_event, stop_when_rule_met=False).run()`; `Collector(collection, make_grid, await_placement,
plan_next=None, max_extras=6).run(placements) -> CollectorResult(results, redo, stop_met, unreachable, stopped)`.
`neck_grid` is `neck.run_grid` bound to a Neck (fakes in tests); `grab(n)` = `Capture.grab`; `detect(yl, yr)` =
`DetectPool.submit_pair` (a Future or a plain result); `predict(pitch, yaw)` -> `Visible(visible, bbox, centroid)` or
None (no model: no early stop).

### Task C1: one pose -> one CaptureHold
- [ ] Tests: 3 captured pairs with board corners -> one `CaptureHold` with 2 pairs (15b's middle two of the still
  ones), img_pitch/img_yaw = measured angles in rad, `img_ap/img_ay` = sign of the command change, `pose` = grid index,
  hold ids consecutive; a pose whose frames have < 6 tags -> no hold, a record with found False;
  `Collection.add` receives the holds (a fake collection records them) and its label lands in the record.

### Task C2: early stop, darkness vs absence, bump
- [ ] Tests: poses the model predicts not visible are never judged; after >= 6 judged poses with most misses ->
  `EarlyStop`; misses whose predicted region is darker than `dark_level` -> "dark" (sentence names exposure/gain),
  else "absent"; found poses whose observed centroid sits > px from the prediction for `run` poses in a row after
  agreeing -> "bumped" (grid paused, sentence says the board moved); the grid stops at once (no further command), the
  rest of the poses go to the redo item.

### Task C3: end of placement + collector
- [ ] Tests: the end calls `collection.end_placement(name)` once, the result carries its sentences; abort/hold from
  `should_stop` -> `stopped` and no snapshot of an empty placement; `stop_when_rule_met` ends the grid once
  `collection.stop` is set; `Collector` runs the defaults, then extras from `plan_next` until the stop rule is met,
  records `await_placement` -> False as unreachable, retries the redo list once at the end, stops on abort/hold.
- [ ] Test with stored-session holds (`data_root`): the holds of p21's first take fed through `GridRun` (fake neck and
  capture replaying them) give the same `Collection` tallies as `replay.feed` of the same holds.
- [ ] Commit C.

## Phase D — deploy (`deploy.py`)

API: paths `ROBOT_ENV`, `HOST_CFG_DIR`, `CONTAINER_CFG_DIR`, `OKVIS_SH`, `OKVIS_NS`, `OKVIS_DS`, `HEAD_CAMERA_JSON`;
`Remote(run, dry_run=False)` (`read`, `exists`, `write`, `call(argv, mutating)`, `planned`; `ForbiddenCommand` for
`--grace-period=0` / `--force`, checked before anything runs, dry run included); `parse_env`, `set_env`;
`okvis_startup_checks(text, frame_size) -> [problem]`; `restart_path(remote) -> RestartPath(kind, argv)`;
`DeploySnapshot` + `take_snapshot(remote, producer_yaml, ...)` + `save/load`; `drift(expected, remote) -> [what]`;
`find_dependents(remote, old_name, roots) -> [Dependent]`; `rollback_script(snapshot, new_name, restart) -> str`;
`Deployer(remote, snapshot, deploy_dir, producer_apply, producer_restore, okvis_status, clock, sleep).deploy(plan)
-> DeployResult` / `.rollback(reason) -> RollbackResult`; `RollbackFailed` (loud); `size_options(prev_size)`,
`config_for_size(text, size)`, `check_verdict(new, live, gates)`.

### Task D1: Remote, robot.env, #930
- [ ] Tests: mutating calls in dry run are recorded and not run, reads run; `--grace-period=0` anywhere -> refused even
  in dry run; `set_env` replaces one key, keeps every other line and comment, appends a missing key; #930: a kb4t
  config with 5 coefficients, an unknown type, or `image_dimension` != the producer size -> one problem each, a good
  config -> none.

### Task D2: snapshot, restart path, drift, dependents
- [ ] Tests (FakeRobot filesystem + fake kubectl): the DaemonSet with desired >= 1 -> `rollout restart ds/okvis2x`;
  no DaemonSet but `okvis.sh` -> `okvis.sh restart` with `OKVIS_LAUNCH_ORIGIN`; neither -> kind "none"; the snapshot
  holds robot.env, the calibration's host path and text, producer YAML as given, head_camera.json, the OKVIS image;
  `drift` names each changed item (robot.env edited by someone, config replaced, image changed); `find_dependents`
  returns files naming the old calibration (e.g. a VLA `RIG_YAML`) and writes nothing.

### Task D3: deploy transaction + rollback
- [ ] Tests: order = drift check -> #930 checks -> rollback.sh + backups written -> config installed -> robot.env
  edited -> producer applied -> OKVIS restarted -> verify (robot.env read back, config sha, log has the new
  `(<sha12>)` and `KB4T pinhole camera`); a failed check stops before any write; drift -> `DeployDrift`, nothing
  written; dry run -> the exact command list, nothing run; `rollback` checks drift vs what the tool left (someone
  changed it -> `RollbackDrift`, nothing done), restores robot.env, the producer as saved, restarts OKVIS, verifies
  (log has the old sha); a rollback whose verification fails -> `RollbackFailed` with a loud message, no retry
  (runner called once per step); the generated `rollback.sh` restores robot.env, replaces the producer with the saved
  YAML through the no-overlap cycle, restarts by the detected path, and never contains `--grace-period`.

### Task D4: deploy size + check grid
- [ ] Tests: previous default 960x600 -> options [match 960x600 (recommended, scale 0.75), 1280x800]; 1280 previous
  -> one option; `config_for_size` = `scaling.scale_config` and passes `okvis_startup_checks` at its size;
  `check_verdict` (holds as the bootstrap clusters): new better on depth and rows -> DEPLOY, not shown better ->
  KEEP_LIVE with the reason.
- [ ] Commit D.

## Phase E — post-check (`postcheck.py`)

API: `DEPTH_WORSE_PCT 0.10`, `ROWS_WORSE_PX 0.10`, `RECT_WORSE_PX 0.30`, `LATENCY_FACTOR 1.5`; `depth_check(sync_ms,
new, old) -> DepthCheck`; `rowfeat(img0, img1) -> RowHealth`; `okvis_check(log_lines, sha12, latency_s,
baseline_latency_s, rect_new, rect_old, floor_tilt_deg) -> OkvisCheck`; `crash_loop(pods) -> bool`;
`decide(depth, okvis, crash) -> PostDecision(rollback, reasons)`; `row_reference(row, conditions, config_sha12)`;
`score_holds(calib_json, holds, rig, work_dir, scorer=decide.score)`.

### Task E1: depth check + decision
- [ ] Tests: sync 0.6 ms -> not checked ("sync"), no rollback on it; new worse by > 0.1 % median |depth| or > 0.1 px
  rows -> worse; better -> fine; `decide` rolls back only on a crash loop or a "worse by the same method" (depth, or
  OKVIS rect rows new vs old); a missing KB4T line, slow latency, a large floor tilt -> reported, no rollback.

### Task E2: row features + OKVIS check + reference
- [ ] Tests (cv2): a textured image and its copy shifted by (dx -12, dy +2) -> median dy 2 +- 0.2 px with the
  left/centre/right thirds; the KB4T line and the entrypoint's `(<sha12>)` are found in a log; `crash_loop` on
  CrashLoopBackOff or >= 2 restarts; `row_reference` has method, median, thirds, edges, conditions, sha.
- [ ] Commit E.

## Acceptance (this agent, tonight)
Phases A-E green on Moonstone (fast suite `tools/calibration -q -n 4`), no robot touched; the dry-run command lists of
one grid pose, one enable and one deploy + rollback in the report. The robot acceptance (section 13 M5 row: one full
session with the user, rollback exercised on purpose, abort/heartbeat/reboot paths, timings) is the lead's, later.

---

## M5 wiring notes (for the lead: what `daemon.py` must add)

New phases (all but DECIDE and DEPLOY_SIZE are **ACTIVE**: no heartbeat for 5 s -> HOLD; HOLD left only by G with a TUI
attached; A aborts):

| Phase | Entered by | Drives | Leaves to |
|---|---|---|---|
| `MOTORS` | READY + preflight motion gate green, key `M` | shows `Enabler.checks(gantry)`; key `Y` -> `Enabler.enable("Y", gantry=state["gantry_confirmed"])` | `PLACE` on `EnableResult.ok`; stays with the reason otherwise |
| `GRID` (ACTIVE) | placement guidance said "Good. Step back." + the 5 s spoken countdown with abort live | `GridRun(...).run()` in a worker thread; `should_stop` reads the machine | `PLACE` (next target), `HOLD` (heartbeat/hold), `ABORTED`, `FIT` when `Collector` says `stop_met` |
| `IMU_SWEEP` (ACTIVE) | after the final fit, operator puts the board flat, countdown | `imusweep.capture_sweep(NeckSweepRig(neck, DmaImu(d, prof.imu_slot).accel, frames0), imusweep.serpentine(), detect, R_SC_live, sleep=...)` | `DECIDE`; `imusweep.SweepAborted` -> `HOLD` |
| `DECIDE` | sweep done | `decide.verdict(...)` (M1) | `DEPLOY_SIZE` on DEPLOY, `DONE` with the reasons otherwise |
| `DEPLOY_SIZE` | DEPLOY verdict | `deploy.size_options(prev_size)`; key `1`/`2`/`?`; then producer to that size, check grid (`GridRun` with a 15-pose list at ~1 m), `deploy.check_verdict(new, live)` | `DEPLOY` on DEPLOY; `DONE` (keep live) otherwise |
| `DEPLOY` (ACTIVE) | check verdict DEPLOY | `Deployer.deploy(DeployPlan)` | `POSTCHECK`; `DeployDrift` -> stays, operator asked; `RollbackFailed` -> `STOPPED_LOUD` |
| `POSTCHECK` (ACTIVE) | deploy verified | `postcheck.depth_check` (always), `okvis_check` unless opted out (key `O`), `postcheck.decide` -> `Deployer.rollback(reason)` when it says so | `DONE` (report), motors: `Enabler.disable(confirmed)` after key `D` and the warning |

Keys to add: `M` (motor screen), `Y` (enable; only in MOTORS), `D` (disable after the warning; DONE/ABORTED), `O`
(skip the OKVIS check), `1`/`2`/`?` (deploy size), `R` (rollback on purpose: M5 acceptance). State keys: `motors`
(`EnableResult.as_dict()` / `DisableResult`), `grid` (`GridResult.as_dict()` of the running placement: pose i/N, last
`PoseRecord`), `collector` (`CollectorResult.as_dict()`), `sweep`, `decision`, `deploy_size`, `deploy`
(`DeployResult.as_dict()` incl. the rollback script path), `postcheck` (`PostDecision.as_dict()`), `gantry_confirmed`.
Events: `motors_enable_refused{checks}`, `motors_enabled{n,of}`, `motors_disabled`, `neck_command{argv,t}`,
`pose{PoseRecord}`, `grid_early_stop{reason,sentence}`, `grid_end{snapshot}`, `deploy_step{step}`, `deploy_done`,
`rollback{reason,ok}`, `ROLLBACK_FAILED{message}` (banner, loud, no retry), `postcheck{...}`, `row_reference{...}`.

Function signatures the daemon calls:

```python
# neck.py
neck = Neck(tool=prof.motor_tool, run=motors.run_cmd, robot=ShmRobot("/dev/shm"),
            tool_info=motors.motor_tool(prof.motor_tool), kp=None)
neck.arm()                                        # NeckRefused -> MOTORS phase shows why
out = run_grid(neck, poses_deg, on_pose=cb, should_stop=lambda: machine_stop_reason())  # GridOutcome
# motors_enable.py
en = Enabler(run=motors.run_cmd, probe=EnableProbe(...))
en.checks(gantry=True) -> [preflight.Check];  en.enable("Y", gantry=True) -> EnableResult
en.disable(confirmed=True) -> DisableResult
# grids.py
GridRun(neck_grid=lambda poses, on_pose, should_stop: run_grid(neck, poses, on_pose, should_stop),
        grab=capture.grab, detect=pool.submit_pair, collection=col, placement=target.name,
        poses=[planner.grid()[i] for i in protocol.poses_for(target.distance_m, target.tilted)],
        predict=model_predict, hold_ids=itertools.count(next_hold), should_stop=stop_fn,
        on_event=store.event).run() -> GridResult
Collector(col, make_grid, await_placement, plan_next).run(place.DEFAULTS) -> CollectorResult
# deploy.py
remote = Remote(motors.run_cmd, dry_run=False)
snap = take_snapshot(remote, producer_yaml=preflight.snap["producer_yaml"])     # in preflight
d = Deployer(remote, snap, deploy_dir, producer_apply=lambda size: producer.resize(kube, spec, *size, fps),
             producer_restore=lambda: producer.restore(kube, spec), okvis_status=...)
d.deploy(DeployPlan(name, config_text, size)) -> DeployResult;  d.rollback(reason) -> RollbackResult
# postcheck.py
depth_check(sync_ms, new_scores, old_scores) -> DepthCheck
okvis_check(log_lines, sha12, latency_s, baseline_latency_s, rect_new, rect_old, floor_tilt_deg) -> OkvisCheck
decide(depth, okvis, crash) -> PostDecision
```

Not in these modules (lead): stopping OKVIS during collection and bringing it back on abort (design 4); the producer's
`Kube(read_only=False)`; the spoken countdown; mapping health events to `should_stop`.

---

## Status (10 Nov 2026, night; against fakes only, no robot touched)

Phases A-E implemented test-first (`neck.py`, `motors_enable.py`, `grids.py`, `deploy.py`, `postcheck.py`; 137 new
tests in 5 files, one of them a stored-session replay through `GridRun` that matches `replay.feed`'s tallies and labels
on p21's first take). Fast calibration suite on Moonstone (`-n 4`): **1158 passed, 2 skipped**. `DEPS.txt` unchanged
(the new modules port step 08's code instead of importing it). Not edited: `place.py`, `detect.py`, `daemon.py`,
`tui.py`, `cli.py`.

Things to verify on the robot (behaviour that fakes cannot prove):
- the capture now happens after the motor tool has exited (its `--hold` is the 0.8 s settle only), relying on
  `--no-return` keeping the pose, as step 08 relied on between poses; step 08 captured while the tool still held;
- the journal line "N of N motors enabled" and its wording on the 2.1.x master;
- on `okvis.sh` robots the OKVIS pod label (`app.kubernetes.io/name=okvis2x`) and container name used for its log;
- `grep -rnIF` over `DEPENDENT_ROOTS` finishes quickly on a robot (roots exclude `/root` and `/data` takes).

Open questions for the user:
1. robot.env: OKVIS reads `/etc/phantomos/okvis/robot.env` (RUNBOOK 4a, kit scripts); `preflight.RobotProbes`
   defaults to `/etc/phantom/robot.env`. `deploy.take_snapshot` reads the OKVIS one; preflight's default should match.
2. Pre-enable facts with no known source on the 2.1.x master (48 V bus, head power switch, slaves in OP, fault
   latches): reported "not available", not blocking. Should any of them block when unavailable?
3. Thresholds (placeholders): post-check worse = +0.10 % depth / +0.10 px rows / +0.30 px rectified rows, latency
   1.5x, floor tilt reported above 1 deg; early stop after 6 judged poses with more than half missed, dark below grey
   40; bumped = 60 px off for 3 poses.
4. Check grid at the deploy size: DEPLOY only when new is shown better than live there too (bootstrap over holds), or
   is "not shown worse" enough after the 1280 verdict?
5. Neck kp: step 08 used the tool's default (30); the phantom12 neck moves used `--kp 60`. Default left at the tool's.
6. A partial enable (N of M, N < M) is reported and not undone automatically (disabling drops the head and arms).

---

## M5 wiring status (10 Nov 2026, night; fakes and dry runs only, no robot touched)

`calib <robot> --full [--dry-run]` -> `calibd run --session-mode full [--dry-run]`: `flow.FullFlow` runs the session
in its own thread beside calibd's loop (`flow.py`, `sessionctl.py`, `flowrobot.py`; daemon/tui/cli extended). Phase
graph: READY -M (motion gate green)-> MOTORS -Y-> SETUP (snapshot, OKVIS stopped by node label / `okvis.sh stop`,
producer 1280x800@15 under the lease) -> [PLACE -> countdown -> GRID]* (defaults, extras, redo once) -> IMU_PLACE ->
countdown -> IMU_SWEEP -> FIT -> DECIDE -(DEPLOY)-> DEPLOY_SIZE (1/2/?) -> [SETUP -> CHECK_PLACE -> countdown ->
CHECK_GRID, at a size other than 1280] -> DEPLOY -> POST_PLACE -> POSTCHECK -> REPORT -> FINISH (D,D / K; R) ->
restore -> DONE. Not DEPLOY anywhere -> REPORT with the live calibration kept. HOLD from every motion phase (deadman,
foreign command, missed pose, refused arm); G then a countdown before the next command; a restart resumes in HOLD and
asks for Y again before any motion; an interrupted deploy asks Y/R. Every robot command: `<session>/commands.txt`.

Decisions taken here that the user should confirm: (1) one Y in MOTORS = "on the gantry, nobody at the head" + enable
(no separate gantry key); (2) at 1280 no check grid (the lens verdict is the check); (3) the normal end asks D/K
before restoring, an abort restores at once and leaves the motors until D, D; (4) abort after FINISH keeps the
deploy; (5) OKVIS stop = node label `foundation.bot/has-okvis=false` (RUNBOOK 7), restored to the value it had; (6) a
dead calibd leaves OKVIS stopped and the producer at 1280 until a restart or the lease's stale-restore, which nothing
on the robot runs yet; (7) the board-flat IMU placement is "board seen still for 1 s" (C skips the sweep).

Still needs the robot (M5 live run): the countdown's speech timing against the 5 s, OKVIS by node label on a
DaemonSet robot (pod gone and back, its 60 s grace), Capture across a producer resize (1280 -> deploy size),
`flowrobot.pipeline_fit_decide` (split -> replay.evaluate -> step 71 export) and `scorers` on real holds, the flat
board seen at neck home, the IMU slot read through DmaImu.

---

## Status (10 Nov 2026, evening): settle-trim, final fit = last live refit, stale-lease self-restore

Approved by the user on 10 Nov; test-first, fakes and stored sessions only, no robot touched. Fast suite on
Moonstone (`-n 4`): **1390 passed, 2 skipped**; replay tier: 27 of 28 passed in one `-n 4` run (the failure:
`test_live_refit_on_a_stored_session_ends_at_step_45s_fit[p10]`, one placement refit over its 30 s budget while four
replays shared the box; rerun alone with the new replays: 6 of 6 passed, slowest p10 refit 17.2 s).

1. **Settle-trim** (M3 plan, Results): trim rounds until the kept set repeats, cap 12, unsettled = warning in step 45.
   Settles at 6 (p21) / 7 (p10) rounds; same verdicts on both sessions.
2. **Final fit = last live refit.** `refit.LiveFit`: the flow (`robot.live_fit`, built in `daemon._full_flow`) asks
   for a background refit after each placement and, once the collection ends, one on the final split (it runs while
   the IMU sweep does). At FIT, `flowrobot.pipeline_fit_decide` uses it as step 45's result (`refit.final_fit`:
   step 45's JSON, `"source"` naming the refit; `replay.evaluate(fit=...)` skips step 45) only when its corner digest
   equals the split's FIT file and it converged, passed the gate and has the rig file's camera sides; otherwise step
   45 runs. The decision records `fit_source`; refit failures are `live_refit_error` events, never the session's.
   Stored sessions (`test_the_final_live_refit_stands_for_step_45_on_a_stored_session`): the refit covers the split
   and equals step 45 within 1e-3 px (measured 2e-12 earlier); step 45 there took 17.9 s (p21) / 13.6 s (p10), the
   final refit itself 14.7 / 11.2 s. With the IMU sweep (~90 s) between the end of collection and FIT the refit is
   done by then: **the whole step 45 is saved, 13-18 s on Moonstone** (longer on a Thor; to be measured there). A
   session whose IMU sweep is skipped (C) waits for the refit instead: ~3 s saved.
3. **Stale-lease self-restore** (`calibtool/selfrestore.py`, `python -m calibtool restore --if-stale-hours 2 --root
   <dir>`): acts only with the lease older than 2 h, `<root>/calibd.lock` free (taken for the restore) and no TUI for
   2 h (calibd now saves `last_hb_t`); restores the producer as saved (no-overlap cycle; not restarted when already as
   saved), then OKVIS by the recorded path, then releases the lease; logs to the session's events.jsonl and
   commands.txt. Refuses (exit 1, nothing sent) a session that started a deploy and did not roll it back (a person
   runs its `deploy/rollback.sh`), a missing saved spec or session folder. A failure keeps the lease; the next run
   retries. 23 tests with fakes (`tests/test_selfrestore.py`).

**Install on phantom10 (the lead, with the user's yes; not done by this agent).** The robot's kit must contain this
branch's commit first (any `calib phantom10` start rsyncs it to `<root>/kit`). As root on phantom10, check with the
dry run, then install:

```bash
cd /tmp/calibtool-m2/kit/tools/calibration && PYTHONPATH=/tmp/calibtool-m2/kit/tools/calibration:/root/p11-cal/pydeps \
  /root/p11-cal/venv/bin/python -m calibtool install-restore-timer --root /tmp/calibtool-m2 --hours 2 --dry-run
cd /tmp/calibtool-m2/kit/tools/calibration && PYTHONPATH=/tmp/calibtool-m2/kit/tools/calibration:/root/p11-cal/pydeps \
  /root/p11-cal/venv/bin/python -m calibtool install-restore-timer --root /tmp/calibtool-m2 --hours 2
systemctl list-timers calibtool-restore.timer      # next run within 10 min
journalctl -u calibtool-restore.service -n 20      # "no lease ...: nothing to restore" when idle
```

Undo: the same prefix with `uninstall-restore-timer`. A decision check without a timer: `... -m calibtool restore
--if-stale-hours 2 --root /tmp/calibtool-m2 --dry-run`. Caveat: P10's profile root is `/tmp/calibtool-m2` (the M2
read-only folder): a reboot clears the lease and the kit, after which the timer finds nothing to restore while the
producer keeps whatever size the session left. A full session should use `/data/calib` (profiles.yaml) before the
timer is relied on; re-run the install with that `--root` then.
