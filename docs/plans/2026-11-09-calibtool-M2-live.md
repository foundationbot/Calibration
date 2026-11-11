# calibtool M2 — live robot runtime (read-only) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:test-driven-development for every task (test written and
> seen failing, then code, then green). One agent executes the tasks in order (B depends on A; C on A+B). Steps use
> `- [ ]`. Same format as the M1 plan (`2026-11-09-calibtool-M1-replay.md`): each task names its file, its API and the
> exact tests; the code is written test-first in the task.

**Goal:** `calib phantom10` on the laptop starts (or re-attaches to) `calibd` on the robot over ssh; `calibd` runs a
preflight with every M2 check green or a plain reason, reads the head cameras (queue 3 = left, 0 = right) into matched
pairs with frame-health counters, detects the board live, watches health (section 8 v1), keeps `state.json` +
`events.jsonl` and resumes after a dropped link; the TUI renders it and speaks pause-then-speak placement guidance. A
placement-trial mode (neck at home, nothing moves) guides the operator to the default placements by voice and logs
seconds and success per placement.

**Architecture:** robot side = `calibd` (`daemon.py`): one state machine thread, one capture thread (`capture.py`,
never blocked), a detection process pool (`detect.py`), a health watch (`health.py`) fed by capture counters and
injected system probes, read-only producer and motor checks (`producer.py`, `motors.py`), session folders
(`storage.py`); clients connect over a Unix socket in the session folder, one JSON line per message. Laptop side =
`cli.py` (ssh bootstrap, start/attach) + `tui.py` (render, keys, heartbeat, `spd-say`). The ssh link is just
`ssh root@robot python -m calibtool.daemon attach ...` bridging stdin/stdout to the socket: when it dies, the
heartbeat stops and calibd holds; a new attach resumes.

**Tech stack:** Python 3.12 stdlib (threading, socket, json, subprocess, fcntl), NumPy, OpenCV (aruco), PyYAML;
`dma_ipc` on the robot only (imported lazily, injected in tests). Tests: pytest + xdist on Moonstone only
(`tests/run_moonstone.sh <wt> m2 tools/calibration -q -n 4`; at most 4 processes). Every new test file
`importorskip`s numpy (and cv2/yaml where used).

Design: `2026-11-09-calibration-terminal-tool.md` sections 3, 4, 7, 8, 10, 11, 13 (M2 row).

---

## Facts every task relies on (measured on phantom10, 9 Nov 2026, read-only)

- Identity: hostname `phantom10`, machine-id prefix `60776b088362` (= `robots.yaml` P10). 9 CPUs visible, load ~9.
- Runtime: `/root/p11-cal/venv/bin/python` 3.12 (numpy 2.5.4, cv2 5.0.0, scipy 1.18.1, yaml) + `/root/p11-cal/pydeps`
  (`dma_ipc`). Writes allowed only under `/tmp/calibtool-m2/`.
- Cameras: `dma_ipc.VideoMetaQueue(video_meta_queue_name(c), create=False, wait=True).make_consumer(max_lag=8)`;
  consumer `.next(timeout)` -> meta or None; `VideoFramePool(video_frame_pool_name(c), create=False).read(meta)` ->
  buffer or None (None = pool desync after a producer restart). Meta fields: `camera_id, capture_timestamp_ns,
  publish_timestamp_ns, frame_sequence, width, height, format (3 = NV12: Y plane = first w*h bytes),
  frame_size_bytes, exposure_time_ms, gain_db`. Pool `.stats()` -> `{recycled, in_flight, torn, rejected,
  reopen_count}`. Producer: 1280x800 @ 25 fps; ~26 frames/s/queue read.
- Producer: deployment `producer` in namespace `dma-video`, strategy Recreate, replicas 1, args
  `--width 1280 --height 800 --fps 25 --camera-config /opt/configs/cameras.json --master C`; `k0s kubectl` with
  `KUBECONFIG=/var/lib/k0s/pki/admin.conf` (no `kubectl` on PATH). The kit's `prodrestart.sh` is the no-overlap
  reference: scale 0, wait for no pod, 3 s, scale 1, wait `1/1 Running`, 8 s, check reads.
- `/etc/phantom/head_camera.json`: cameras `ego_left` queue 3 board 0, `ego_right` queue 0 board 0; board 0 mxid
  `19443010A1F94D2E00` = USB `2-3.1`, speed 10000 (sysfs `/sys/bus/usb/devices/*/{idVendor=03e7,serial,speed}`).
- Rings in `/dev/shm` (`UEUQ_AMD` header): entry size/capacity `<QQ` at 16, head counter `<Q` at 64.
  `raw_imu_actuals` and `actuals` advance at 500/s; `desired` head 1 (nothing commanding).
- Master: `systemctl is-active dma-ethercat` active; `dma_cmd_client -V` -> `dma_cmd_client 2.1.1 (build ...)`;
  `dma_cmd_client ping` -> `PONG  uptime=6065.3s  motor_state=0`; `state` prints zeros on 2.1.x. Motor tool
  `/root/p11-cal/dma_motor_motion` md5 `208231a2...`, `--help` mentions `targets`; `/usr/bin/dma_motor_motion` does not.
- No `nav-cli` on phantom10; no `okvis2x` pod (DaemonSet desired 0). Policy-like pods running: `ghost-wbc-teleop`,
  `foundation-vla`, `vla-cli` (motion gate is M5; M2 reports them).

## File map (all in `tools/calibration/calibtool/`)

| File | Responsibility | Test |
|---|---|---|
| `profile.py`, `config/profiles.yaml` | per-robot runtime: host, python, pydeps, root, queues, producer ns/deploy, motor tool | `tests/test_profile.py` |
| `capture.py` | `PairMatcher`, `FrameCounters`, `Capture` thread over a camera source, `DmaCameras` adapter | `tests/test_capture.py` |
| `detect.py` | contrast stretch, find-on-stretched/refine-on-raw, `pool_size`, `DetectPool`, `board_obs`, `hold_dict` | `tests/test_detect.py` |
| `health.py` | section 8 v1 rules -> `Event`s; hardware-suspicion sentences | `tests/test_health.py` |
| `producer.py` | `Kube` runner (read-only guard), `Spec`, `Lease`, no-overlap `restart`/`resize`/`restore` | `tests/test_producer.py` |
| `motors.py` | read-only master/ping/desired/motor-tool checks | `tests/test_motors.py` |
| `storage.py` | session folders, retention, protected paths, disk checks | `tests/test_storage.py` |
| `preflight.py` | the preflight checks, injectable probes, `Check` list | `tests/test_preflight.py` |
| `place.py` | `PlacementGuide` (pause-then-speak), default placements, `Trial` log | `tests/test_place.py` |
| `daemon.py` | `Store` (state.json/events.jsonl), `Machine` (phases, heartbeat, keys, resume), socket server, CLI | `tests/test_daemon.py` |
| `tui.py` | `render(state)`, `Speaker`/`SpdSay`, `Tui` loop (heartbeat, keys, speak new prompts once) | `tests/test_tui.py` |
| `cli.py` | `calib <robot>`: bootstrap (rsync kit), start/attach/stop over an injected ssh runner | `tests/test_cli.py` |

M1 modules are not changed.

## Phase A — pure building blocks

### Task A1: profile
- [ ] Tests: `profile.load("P10")` -> host `root@phantom10`, python `/root/p11-cal/venv/bin/python`, pydeps
  `/root/p11-cal/pydeps`, root `/tmp/calibtool-m2`, left 3, right 0, producer (`dma-video`, `producer`), motor tool
  `/root/p11-cal/dma_motor_motion`, rig path = robots' rig; unknown bot -> KeyError; `profile.for_host("phantom10")` -> P10.
- [ ] Implement `profile.py` + `config/profiles.yaml`.

### Task A2: capture
API: `Frame(cam, seq, t_ns, t_pub_ns, y)`; `PairMatcher(left, right, tol_ns=500_000, keep=6).push(frame) -> [Pair]`
(`Pair(t0, t1, left, right)`, dt = t1 - t0); `FrameCounters` per camera (frames, none_reads, none_run, seq_gaps,
repeats, last_seq, last_t_ns, fps(now)) + pair stats (pairs, sync_ns window, max/median |dt|); `Capture(source,
left, right, clock)`: `step()` one poll of each camera (testable), `start()/stop()` thread, `latest()` newest pair
(1-slot mailbox), `subscribe(maxsize)` -> queue fed with `put_nowait` (full -> dropped, counted), `grab(n, timeout)`
the next n pairs; re-attach a camera after `REATTACH_AFTER` consecutive None reads (counted `reattaches`).
- [ ] Tests (fake source yielding scripted metas/None):
  1. pairs match by capture stamp within tol; an unmatched stamp ages out after `keep`; dt sign is right - left;
  2. counters: seq gap counted, repeated seq counted as repeat (freeze), none_run resets on a good read, fps from stamps;
  3. `REATTACH_AFTER` None reads in a row -> `source.reattach(cam)` once, counter 1, counts restart;
  4. a slow subscriber never blocks `step()` (queue full -> drop counted, `step` returns at once);
  5. `grab(2)` returns the next 2 pairs only (not old ones); `latest()` is the newest;
  6. `DmaCameras` against a fake `dma_ipc` module: Y plane sliced from an NV12 buffer of w*h*1.5 bytes; `reattach`
     makes new queue/pool objects.

### Task A3: detect
API: `stretch(gray, lo=1, hi=99)` u8; `find_refine(board, gray, use_stretch=True)` -> `{(tag,k): (u,v)}` (ArUco on the
stretched copy, `board.refine(raw gray, quad)`); `pool_size(ncpu, load1, reserve=2, cap=4)`; `DetectPool(board,
workers, ctx="fork")` `.submit(gray) -> Future`; `board_obs(det, board, focal, pp, min_tags=6)` -> `BoardObs(centroid,
z, n_tags, n_corners, border)` (border = corners within 8 px of the image edge) via KB4 unproject + `solvePnP`;
`hold_dict(pairs_dets, hold, neck=(nan,nan))` -> step-15b corner dict (for `assign.CaptureHold`).
- [ ] Tests (synthetic board image rendered with `cv2.aruco.generateImageMarker` at known positions):
  1. `stretch` maps the 1st/99th percentiles to 0/255, constant image -> unchanged;
  2. on a low-contrast copy (pixels in 100..130) plain `board.detect` finds fewer tags than `find_refine`, and the
     corners `find_refine` returns equal `board.refine(raw, quad)` on the raw image (refined on raw, not stretched);
  3. `pool_size`: 9 cpus load 9 -> 1 (never 0); 32 cpus load 2 -> cap; monotone in load;
  4. `DetectPool` returns the same corners as a direct call;
  5. `board_obs` z within 3 % of the rendered distance for a pinhole render; < min_tags -> None; border flag;
  6. `hold_dict` has the 15b keys, `det_cam` 0 = left, and `replay.Collection.add(CaptureHold(...))` accepts it.

### Task A4: health
API: `Event(t, level in {"info","warn","stop","suspect"}, code, message, action)`; `Watch(baseline, clock)`
`.observe(sample) -> [Event]` where `sample` = dict of counters + probe results (`boot_id`, `clock_s`, `k0s_ok`,
`imu_head`, `usb` {serial: speed}, `dmesg` lines, `producer_log` lines, `producer` {generation, pod_uid, args},
`cpu_load`); restarts are recorded by `Watch.note_restart(t, reason)`.
Rules (section 8 v1): all reads None on a camera for >= 1 s -> warn `desync` action `reattach`; still after a
re-attach -> `producer_restart` action; stamps stop / same seq for >= 2 s -> `freeze` action `producer_restart`; a 2nd
freeze/restart within 10 min -> `suspect`; fps < 0.7 x baseline with load < 1.5 x baseline -> `fps_drop`; sync
median |dt| > 0.5 ms -> warn `sync` action `producer_restart` once, persists after a restart -> `stop` (refuse to
collect); boot id changed -> `stop` `reboot` action `preflight`; clock < 2020 -> `stop`; k0s refused -> warn; IMU head
not advancing for >= 1 s -> warn `imu_silent` "check the head power switch first"; head-camera USB speed < 5000 ->
`suspect` `usb_link` (no restart action); dmesg reset/disconnect on the head camera's port, producer log "Device
crashed" -> `suspect` quoting the line; producer generation/pod uid/args changed while no own restart is running ->
`stop` `producer_changed`. Each rule fires once per episode (re-arms when the condition clears).
- [ ] Tests: one per rule (fires, words, action, once per episode, re-arm), plus `suspicion_sentence(events, now)`
  -> "3rd producer restart in 20 min; USB reset on the head camera's port at 15:41 each time -> suspect the head
  camera cable or hub, not software." for the matching history, and the dmesg filter ignores other ports (e.g. the
  USB-ethernet dongle `2-1.4`).

### Task A5: producer
API: `Kube(run, ns, deploy, read_only=True)`: `get_yaml()`, `pods()`, `logs(tail)`, `scale(n)`, `patch_args(args)`,
`replace(yaml_text)`; a mutating call with `read_only` -> `ReadOnlyRefused` (never reaches `run`). `Spec.from_yaml(text)`
(args, image, replicas, generation, strategy, `clean_yaml()` without status/resourceVersion/uid/managedFields/
creationTimestamp/generation). `Lease(path)`: `acquire(session, spec_path, now)` (refuses a live lease of another
session), `heartbeat(now)`, `stale(now, max_age_s=7200)`, `release()`. `restart(kube, wait, now)`, `resize(kube, w, h,
fps)`, `restore(kube, spec)`: scale 0 -> wait until no producer pod -> settle -> (patch/replace) -> scale back ->
wait Running; never a scale-up while a pod exists (no overlap).
- [ ] Tests with a FakeKube (records calls; pods disappear after N polls):
  1. read-only guard: scale/patch/replace raise and are not run; get/pods/logs run;
  2. `restart` call order: scale 0, pods polled until empty, scale 1, polled until Running; a pod that never goes ->
     `ProducerError` and no scale 1;
  3. `resize` patches `--width/--height/--fps` in the saved args only (other args untouched), in the no-pod window;
  4. `restore` replaces with the cleaned saved YAML (no status/resourceVersion), then replicas as saved; a lease is
     released only after a successful restore;
  5. `Lease`: acquire/heartbeat/stale/release; a second session refused while live; a stale lease reported.

### Task A6: motors (read-only)
API: `parse_ping(text)` -> `{uptime_s, motor_state}`; `parse_version(text)`; `master(run)` -> `MasterStatus(active,
version, uptime_s, motor_state, error)`; `ring_head(path)`; `desired_still(read_head, sleep, s=1.0)`; `motor_tool(path,
run)` -> `{md5, has_targets}`.
- [ ] Tests: parse the recorded phantom10 outputs above; inactive master; ping timeout -> error text; desired advancing
  -> not still; md5 of a temp file; `--help` without `targets` -> has_targets False. No function in `motors.py` calls
  `enable-motors`/`disable-motors` (source scan test).

### Task A7: storage
API: `new_session_id(bot, now)` `YYYYMMDD-HHMMSS-<bot>`; `SessionDirs(root, sid).create()` -> raw, corners, fit,
report, deploy; `disk_free_gb(path)`; `disk_check(path, min_gb)`; `PROTECTED` glob list + `is_protected(path,
refs=())`; `retention(sessions, live_sid, keep_completed=2)` -> session ids whose raw goes (aborted/read-only never
count; the live one and the last 2 completed-deploy kept); `delete_raw(dirs)` refuses protected paths.
- [ ] Tests on tmp_path for each.

## Phase B — preflight, placement guidance, daemon

### Task B1: preflight
API: `Check(name, status in {"ok","warn","fail","info"}, detail, gate in {"collect","motion"})`; `run(probes, profile,
robots) -> [Check]`; `green(checks, gate="collect")`. Probes (injectable dict of callables): hostname/machine-id,
lock, clock, k0s, autonomy (nav-cli or "absent"), policy pods (motion gate), producer spec, head_camera.json, robot.env
(if any), OKVIS state, USB speed of the head board, frame reads + sync + left/right sign (from a `Capture.grab`),
IMU advancing, master, motor tool, `/dev/shm` count, disk.
- [ ] Tests: all-good fake probes -> every collect check ok; each failing probe -> its check fails with a plain detail;
  policy pods running -> motion-gate warn but collect green; left/right sign wrong (board left corner u < right u) ->
  fail "queues swapped"; sync 0.6 ms -> fail.

### Task B2: placement guidance + trial (`place.py`)
API: `Target(name, distance_m, accept=(lo,hi), tilted=False, bearing_deg=None, tol_deg=None)`; `DEFAULTS` = 0.5,
0.75, 1.1, 1.6, far 1.8 (accept `FAR_ACCEPT_M`), tilted 0.75 (accept = the nominal band of `protocol.band_of`);
`PlacementGuide(target, still_s=1.0, still_px=6.0, lost_s=3.0)` `.update(t, obs) -> Say | None`; `Say(id, text,
kind in {"correction","good","lost","intro"})`; `Trial(targets, clock, log_path)`: `start()` intro, `update(t, obs)`,
`key("C")` can't-place, timeout 180 s -> failed; writes `trial.jsonl` (target, seconds, success, n_says) and
`summary()`.
- [ ] Tests:
  1. no speech while the board moves; one `Say` after it has been still >= 1 s; nothing more until it moves and stops
     again (pause-then-speak);
  2. one correction at a time, priority out-of-view (border) > distance > bearing;
  3. distance wording "about 20 centimetres further back" / "closer", rounded to 10 cm; far accepts 1.7-2.0, refuses
     2.15 ("too far");
  4. direction words name the operator's side: board too far to the image left -> "to YOUR left (the robot's right)";
     no sentence says "robot's right" without "YOUR left"; the word "hold" never appears in speech;
  5. inside the accept band -> "Good. Step back." and the target is done;
  6. board lost for 3 s -> one "I can't see the board" prompt;
  7. `Trial` logs seconds per placement and success, C -> logged unreachable, timeout -> failed; summary success rate
     and mean/median seconds vs the 60 s budget.

### Task B3: daemon (`daemon.py`)
API: `Store(dir)`: `write_state(dict)` atomic (tmp + `os.replace`), `event(kind, **kw)` append + flush + fsync,
`load_state()`, `events()`. `Machine(store, clock, read_only=True)`: phases `PREFLIGHT -> READY -> TRIAL -> DONE`,
`HOLD` (heartbeat lost: entered from any active phase; leaves only on key `G` with a TUI attached), `ABORTED` (key `A`);
`heartbeat(t)`, `tick(t)` (5 s deadman), `key(k)`, `resume()` (from state.json: active phases come back as `HOLD`,
events `resumed`), `banner` (NOT NORMAL lines). `Daemon(machine, capture, health, guide/trial, server)` wires them;
`serve(sock_path)` (Unix socket, JSON lines: daemon -> client state every 0.5 s; client -> `{"hb":t}`, `{"key":"A"}`,
`{"cmd":"stop"}`). CLI: `python -m calibtool.daemon start|attach|stop|status --root R --session S --profile P10
[--trial] [--fake-cameras]`; `start` daemonises (`setsid`), takes `fcntl` lock `R/calibd.lock` (one job per robot),
writes `pid`.
- [ ] Tests:
  1. `Store`: state survives a crash mid-write (tmp file left, state.json intact); events appended in order, each a
     JSON line with t and kind;
  2. heartbeat: no hb for 5 s in TRIAL -> HOLD + event; hb returns -> still HOLD until `G`; `G` without a TUI ignored;
  3. `A` from any phase -> ABORTED, event, nothing restarts by itself;
  4. resume: a state.json in TRIAL -> after `resume()` phase HOLD, counters kept, `resumed` event appended;
  5. socket round trip on tmp_path: a client gets state lines, its hb keeps the phase, closing the client -> HOLD after
     5 s (fake clock), a new client + `G` -> back to TRIAL; the events file keeps every transition;
  6. the lock refuses a second daemon on the same root.

## Phase C — laptop side and the robot run

### Task C1: tui
API: `render(state, width=100) -> str` (header: robot, motors, NOT NORMAL banner, abort key always shown; checks;
capture counters; health; placement prompt); `SpdSay(path="/usr/bin/spd-say", run=subprocess.run)` `.say(text)`
(`spd-say -w -r -10 text`); `Tui(link, speaker, clock, out)`: `step(line)` handles one state line (speaks each new
`say.id` once, renders), `heartbeat()` every 1 s, `key(k)` forwards.
- [ ] Tests: render shows the NOT NORMAL banner and "A = abort" in every phase; a state with `say` id 3 spoken once
  even when the line repeats; ids 3 then 4 -> two calls; keys forwarded as JSON lines; heartbeat lines at 1 Hz on a
  fake clock; `SpdSay` builds the expected argv (fake run).

### Task C2: cli
API: `calib <robot> [--session ID] [--trial] [--headless] [--stop] [--status]`; `Link(argv_prefix, popen)` (ssh);
`bootstrap(profile, rsync)` copies `calibtool/` + DEPS modules to `<root>/kit/tools/calibration/`; `start_or_attach`.
- [ ] Tests (fake ssh/popen): command lines for start (nice, OMP_NUM_THREADS=2, PYTHONPATH kit:pydeps), attach, stop;
  an existing session id -> attach only, no new start; `--headless` renders N states and exits.

### Task C3: the read-only run on phantom10 (acceptance, M2 part without a person)
- [ ] Unit suite green on Moonstone.
- [ ] Copy the kit to `/tmp/calibtool-m2/kit`; `calib phantom10 --headless` (or the daemon CLI by ssh): preflight
  green (collect gate), pairs from 3/0 with sync, live detection (board if present), state.json/events.jsonl written.
- [ ] Link loss: kill the attach ssh -> HOLD after 5 s in events; re-attach -> state intact, `G` -> back.
- [ ] Health: counters and rules report correctly on the live stream (no desync, sync < 0.5 ms, IMU advancing).
- [ ] TUI renders it from the laptop; one short spoken test prompt.
- [ ] `stop`; no process left (own PIDs only); nothing written outside `/tmp/calibtool-m2`.

## Acceptance (M2 part of this agent)
Phase A-C tests green on Moonstone; the read-only phantom10 run as in C3, with exact commands and results in the
report. The timed placement trials with the operator are run by the lead with the user (`calib phantom10 --trial`).

## Status (9 Nov 2026, evening)

Phases A-C implemented test-first on `feat/calibration-agent-calib-tui-m2`; M1 modules unchanged; `DEPS.txt` unchanged
(the new modules import only `lensmodel` and `rigcfg`, already listed). Tests: 189 new (12 files), fast calibration
suite on Moonstone **935 passed, 2 skipped** (`-n 4`).

**Read-only run on phantom10** (kit rsynced to `/tmp/calibtool-m2/kit`, calibd started and stopped by `calib`'s
functions over ssh; sessions `20261109-194918-P10`, `20261109-195141-P10` under `/tmp/calibtool-m2/sessions/`):
- preflight: every **collect** check ok (identity by machine-id, lock, clock, k0s, autonomy (no nav-cli), producer
  1280x800 @ 25 saved, head_camera.json saved, USB 10000 Mbit/s on 2-3.1, 3/3 pairs from queues 3/0, sync 0.013-0.014
  ms, left/right by disparity sign (queue 3 left, 20 px), IMU advancing, 241 GB free). **Motion** gate: master 2.1.1
  active, motors off, e-stop clear, `/dev/shm/desired` still, motor tool md5 `208231a2...` with `targets`; **fails** on
  running policy pods `positronic/foundation-vla`, `positronic/ghost-wbc-teleop` (to be off before M5 enables motors).
- capture: 25.0 fps both queues, 0 empty reads, 0 sequence gaps, 0 repeats, 0 unpaired over ~700 pairs; sync median
  0.0148 ms, max 0.0197 ms.
- live detection: 1 worker (9 usable cpus, load ~7), ~2 detections/s; no board stood in front of the robot, so the
  trial said the intro and, after 6 s, "I can't see the board" once. Detection checked on a stored phantom10 frame
  instead (`test_live_detection_on_a_stored_phantom10_frame`, `test_a_dark_stored_frame_is_still_found`: 35/35 tags;
  at 1/8 brightness plain 31, stretched 34).
- link loss: attach ssh killed -> `link_lost`, `heartbeat_lost`, phase HOLD (held_from TRIAL) within the 5 s deadman;
  re-attach -> HOLD shown, G -> TRIAL (`resumed_by_operator`); events.jsonl has every transition.
- daemon restart under the same session id (stands in for a crash/reboot): came back in HOLD, waited for G.
- health: run 1 raised a false `imu_silent` (sampler reading went stale during its kubectl calls; fixed: the rule
  judges the reading's own time, slow probes in their own thread); run 2: no health event in 25 s.
- `calib phantom10 --headless 2 --no-speech --session ...` rendered the state on the laptop; `calib phantom10 --say
  "Calibration tool: speech test."` spoke once through spd-say (rc 0).

**Placement trials with the operator (lead + user):** `cd tools/calibration && python3 -m calibtool.cli phantom10
--trial`, then T starts; the robot needs nothing changed (read-only). Results land in
`/tmp/calibtool-m2/sessions/<id>/report/trial.jsonl` (+ `trial_summary` in events.jsonl).

### Review fixes of the live code (9 Nov, night; no robot access)

An independent review of the live-robot code found 9 problems; all fixed test-first on
`feat/calibration-agent-calib-tui` (fast suite on Moonstone 1092 passed, 2 skipped, `-n 4`). What the operator notices:
- a restarted calibd keeps speaking (prompt ids continue across runs; the dead run's prompts are cleared on resume);
- a camera that stops delivering pairs for > 1 s shows **CAMERA STALLED** in the banner, says "The camera stopped.
  Please wait." once, pauses the placement's clock and repeats the intro when frames come back; the placement times
  out even when no detection finishes;
- a failing loop step shows "calibd step failing ..." and the loop goes on; a broken detection pool is rebuilt; the
  TUI shows "calibd not stepping" when calibd's step counter stops for 3 s;
- keys other than A/G/T pressed outside TRIAL are dropped (logged `key_dropped`); time in HOLD does not count against
  the placement; G repeats the current intro;
- `calib` asks calibd's status before copying the kit: a live calibd running other code (kit hash differs), or another
  session, is refused with the way out (`--replace` stops it first); `start` succeeds only on its own child's state line;
- after a restart calibd stays in HOLD through its preflight (G needed; refused while that preflight failed); the
  deadman runs on the monotonic clock; only the newest TUI controls (A works from any); the motor tool's `targets` is
  read from its file, never run; read-only kubectl is an allow-list (get/logs/describe); a failed producer cycle
  scales the producer back (or says PRODUCER LEFT AT 0 REPLICAS when the old pod is still there: no overlap);
- the speech queue drops a waiting correction when a newer one comes (never an intro, skip, end, "out of reach");
- "out of reach" needs two separate rests of 1 s, or one of 3 s (was: one rest of 1 s).

**Morning check, not yet done (a robot change: needs the user's yes): the camera re-attach after a producer restart.**
The re-attach is tested only against fake rings (segment files replaced, a hanging attach). The live check restarts the
producer pod while calibd runs and watches calibd pick up the new segments by itself:
1. Laptop: `cd tools/calibration && python3 -m calibtool.cli phantom10 --status` (is a calibd live? which kit?). Then
   `python3 -m calibtool.cli phantom10 --trial` (add `--replace` if the status shows a live calibd: it runs older code).
   Leave it in READY (do not press T); note the session id in the header.
2. Robot (as root), before: `k0s kubectl -n dma-video get pods -l app=producer -o wide; ls -li /dev/shm | grep -i video`
3. Robot, the no-overlap restart (what `producer.restart` and the kit's prodrestart.sh do):
   ```
   k0s kubectl -n dma-video scale deploy producer --replicas=0
   until [ -z "$(k0s kubectl -n dma-video get pods -l app=producer -o name)" ]; do sleep 1; done
   sleep 3
   k0s kubectl -n dma-video scale deploy producer --replicas=1
   k0s kubectl -n dma-video rollout status deploy producer --timeout=90s
   ls -li /dev/shm | grep -i video      # other inode numbers than in step 2 = the segments were recreated
   ```
4. Expected in the TUI: CAMERA STALLED while the pod is down; health `freeze` / `desync`; a STOP line "Someone else
   changed the camera producer" (the pod uid changed and the tool did not do it: expected in this check). Within ~10 s
   of the pod Running: the cameras line back at ~25 fps on both queues, CAMERA STALLED gone, no calibd restart.
5. Robot, the evidence (session id from step 1):
   ```
   S=/tmp/calibtool-m2/sessions/<id>
   grep -E '"kind": "(camera_stalled|camera_back|health|step_error|loop_died)"' $S/events.jsonl | tail -20
   python3 -c "import json; s=json.load(open('$S/state.json')); print(s['pair_age_s'], {q: (c['fps'], c['reattaches'], c['last_reattach']) for q, c in s['capture']['cams'].items()})"
   ```
   Pass: `camera_back` after `camera_stalled`, `pair_age_s` < 1, both queues ~25 fps, `reattaches` >= 1 per queue
   (`last_reattach` `segment_recreated` or `no_frames`), no `step_error`. Fail: `reattaches` still rising with no
   frames 30 s after the pod is Running -> keep `$S/calibd.log` and events.jsonl, stop calibd, report.
6. Laptop: `python3 -m calibtool.cli phantom10 --stop`.
