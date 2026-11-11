# calibtool handoff — 10 Nov 2026 ~13:10 PST

Written by the calibration agent session `okvis2-x-65 [988d7f]` for the next one. Branch
`feat/calibration-agent-calib-tui` (off AI-1184; worktree `~/wt/calib-tui`). Read `README.md` (this folder) first,
then the design plan `docs/superpowers/plans/2026-11-09-calibration-terminal-tool.md` (D1-D12 = the user's decisions,
section 13.0 = status).

## Running it (since 10 Nov evening)

`calib` (installed with `python3 -m calibtool.cli --install` in `tools/calibration`; re-run after pulling) asks which
robot and offers what fits: continue an unfinished session, start new (the robot's own restore first if a dead
session holds the producer), trial, status, stop, or "new robot (setup)" (`calibtool.onboard`). The long flag forms
below still work for scripts. phantom10's tool folder is `/data/calib` now. phantom11 (P11) is onboarded and got the
first calibtool deploy (session 20261110-145300-P11).

## In one paragraph

The tool exists and runs on a real robot.
- **Code:** M0-M5 are built and tested (fast suite ~1460 tests on Moonstone; stored-session replays reproduce the past hand-run calibrations' verdicts).
- **Live sessions on phantom10 (P10):** read-only preflight, placement trials, a 10-pose grid smoke test, then the first full session on 10 Nov.
- **That full session is paused in HOLD:** 5 placements, 225 poses in 30 min, live refit stable at 0.33-0.37 px rms and baseline 44.2 mm.
- **Never run live:** the IMU floor sweep, the final fit/decide, deploy size, deploy, post-check, rollback.
- **The operator found the guidance frustrating** (silences, impossible instructions, automatic skips). Much is fixed in code; the list below is what is left.

## Robot state (phantom10 = P10, machine-id 60776b088362)

Re-checked read-only on 10 Nov 13:10 PST by `CalibrationV10`: no reboot since 9 Nov 12:37, the session folder is intact
in `/tmp` (state HOLD, motors_on), calibd not running, no orphan workers, master 2.1.1 motor_state 2, e-stop false, no
loco pods, producer 1280x800 @ 15 (lease held by the session), head camera on 10 Gb/s USB, no OKVIS pod. The fleet
recorders (dma-recorder, dma-mcap-recorder, dma-video-recorder, universal-recorder) run since ~12:03.

| Item | State |
|---|---|
| Motors | **ON**, enabled by the user (`dma_cmd_client enable-motors`) to hold the head level; neck held at about pitch +1.7 / yaw -0.5 deg by the fixed tool |
| Loco (`ghost-wbc-teleop`), `foundation-vla` | **Stopped by the user in Cortana** for the session. Restarting them is the user's (loco is operator-only) |
| Camera producer | Head-only (wrist cameras removed from `/etc/phantom/head_camera.json` at the user's request), **1280x800 @ 15** (set by the tool's SETUP under its lease). Rollback to wrists + 25 fps: `bash /root/p11-cal/backup/headonly-20261109-204447/rollback.sh`; the pre-session producer spec is in the session folder (`producer.saved.yaml`) |
| OKVIS | Not running (it was not running before either) |
| dma-ethercat | 2.1.1, restarted twice on 10 Nov (once by the agent at the user's request, once by Cortana's loco stop) |
| Paused session | `20261110-123252-P10`: `/tmp/calibtool-m2/sessions/` (wiped by a reboot!), copies in `/root/calibtool-sessions/` and Moonstone `~/calibtool/data/p10/live-sessions/` |
| calibd | Not running; no orphan workers |

**Resume:** `python3 -m calibtool.cli phantom10 --full --session 20261110-123252-P10 --replace --voice piper`.
- It syncs the new code and comes back in HOLD.
- The tilted placement's last poses remain.
- Then come the extras, the IMU sweep (board flat on the floor), fit, decide, and the deploy question (Y).

If `/tmp` was wiped, copy the session back from `/root/calibtool-sessions/` first.

## How the user works (read before talking to them)

- **They are at the robot, not at the laptop.** Tell them in chat what is happening before any key press ("tell me instead of pressing a button"). They want short plain status, never silence.
- **Voice:** the Piper voice (`--voice piper`, `length_scale` 1.3) was approved; espeak is "ridiculously artificial".
  - Prompts must be short, one instruction each, say the remaining amount ("Not yet. 30 more centimetres closer.") and give positive feedback.
  - Prompts must never skip automatically, and must explain why the board is not seen.
- **Their stand is chair-shaped.** Height moves in coarse steps, so the tool says "At least N centimetres" for up/down. **The user decides reachability (C), not the tool.**
- **Head level:** with the motors off the head droops (camera 66 deg down) and placement is impossible. Hold the neck level, motors on, before guiding placements.

## Open issues, most important first

1. **Never run live:**
   - the IMU floor sweep (`neck.NeckSweepRig` + `imusweep`, `nk.DmaImu`);
   - fit/decide via the final live refit;
   - deploy size and the check grid at 960;
   - the deploy transaction (rollback.sh first; OKVIS restart via `okvis.sh` on P10, confirmed it has start/stop);
   - the post-check, and rollback.

   Do them in that order on the paused session, telling the user at every step. The deploy needs their Y. phantom10's previous default size is 960x600 @ 25 (the crate demo); the deploy-size question decides.
2. **Spoken coverage reports between placements confuse the operator.** After each grid the tool speaks gap sentences ("right edge: 1 corner vs 57…"); the operator took them as instructions. Show them on screen only, and speak them only when the planner actually asks for that extra placement. Not fixed yet.
3. **Health false alarms during the tool's own actions:**
   - its own producer resize is reported "STOP: Someone else changed the camera producer";
   - its own producer restart gives "SUSPECT: head camera not on the USB bus" / "USB disconnect";
   - during grids "Camera queue N: 2.9 fps against 25" (unexplained; captures were fine).

   Teach health about the tool's own producer changes (the lease generation), and investigate the fps reading.
4. **Automatic TUI re-attach** after a network drop. phantom10's link dropped mid-grid; the robot went to HOLD correctly but the TUI exited. Reconnect with backoff, say "Connection lost / back", still require G (and maybe a way for the operator to give the go without the laptop).
5. **Orphaned detection workers after `--stop`** (2 forkserver processes left each time). Fix the stop path to shut down the DetectPool.
6. **Motor takeover oddity:** inside calibd a second `dma_cmd_client ping` answered `motor_state 0` while the probe and a shell ping read 2. Worked around (takeover uses the checks' state); root cause unknown.
7. **Pre-enable checks: wire the real signals.** The 2.1.1 master reports these in its journal:
   - e-stop (`dma_cmd_client estop-status`);
   - "no drive fault, no latch (29 motors, enabled 29/29, wkc 144/144)";
   - "SAFE_OP phase: 46/46 reached";
   - "ESC baseline: 0 of 52 slaves carry non-zero … counters".

   `EnableProbe` still shows bus_v / head_power / slaves_op / fault_latches as "not available". The user agreed these should block once available. 48 V has no source; the head IMU ticking proves head power.
8. **Near placements (0.5 / 0.75 m) and the tilted placement are hard with this stand.**
   - "Close enough" (accept within 0.15 m after two non-improving corrections) is built but not tried live.
   - The tilted placement's guidance does not model a board leaning back; it gave 15 "move closer" prompts and timed out once.
   - Consider modelling tilt, or replacing the tilted pass by neck pitch.
9. **The stale-lease self-restore timer is built but not installed.** The user agreed (10 Nov) to move P10's root to
   `/data/calib` AFTER the paused session ends (not mid-session: the resume reads `/tmp/calibtool-m2`). The user approved installing it on phantom10. P10's profile root is `/tmp`, so a reboot clears the kit and lease: move the root to `/data/calib`, then run `python -m calibtool install-restore-timer --root /data/calib --hours 2 [--dry-run]` (exact command in the M5 plan status).
10. **Thresholds AGREED by the user on 10 Nov (in the code since this handoff's update):**
    - deploy only if new is better on depth AND rows (95 % interval, as coded);
    - post-check rollback if new is worse by > 0.15 % depth or > 0.15 px rows (same method);
    - OKVIS rectified rows > 0.5 px;
    - latency > 1.5x;
    - check grid at the deploy size "not worse than live" (`deploy.check_verdict`: keep live only when shown worse).

    `postcheck.py`: DEPTH_WORSE_PCT 0.15, ROWS_WORSE_PX 0.15, RECT_WORSE_PX 0.50, LATENCY_FACTOR 1.5.
11. **Kept for the next version:**
    - **M6:** the devel PR. Branch off devel, carry calibtool + `DEPS.txt` files, port AI-1184's `calibration-tests` CI job with pytest-xdist; `devel_compat.sh` passes today.
    - **Stereo health monitor:** separate project, plan 13.2.
    - **The gyro-based third IMU angle, translation and time offset:** v2, Q1.
    - **The `--compute moonstone` option.**

## Measured numbers worth keeping

- **Neck grid:** 2.55 s/pose with the tool's neck driver (was 3.13 s); kp 30 sags ~1.4 deg pitch, ~0.6 deg yaw.
- **Fast solver:** step 45 7.5-23 s; whole fit+decide 15-23 s on Moonstone; Thor 23 s fit vs 444 s scipy.
- **Live session:** 30 min for 5 placements including 4 HOLDs at the old non-snake corner pose (now fixed) and one network drop.
- **Gravity on phantom10:** camera 65.9 deg below horizontal with the head drooping, 40.5-41 deg with the neck level.
- **Detection:** the guidance detector takes 29-36 ms per frame; capture runs at 25 fps (15 fps in the session), L/R sync 0.014 ms.

## Successor prompt (paste into a new session)

```
You are the calibration agent for the head-stereo calibration terminal tool `calibtool` (OKVIS2-X repo, branch
feat/calibration-agent-calib-tui, worktree ~/wt/calib-tui; never /tmp; GIT_LFS_SKIP_SMUDGE=1). The tool calibrates a
robot's head stereo cameras end to end: spoken board-placement guidance, neck grids, an IMU floor sweep, fit, a
decision against the live calibration, deploy with rollback, post-check.

First, only get up to date — do not touch any robot and do not change code yet:
1. Read tools/calibration/calibtool/HANDOFF.md (state, how the user works, open issues), then README.md in the same
   folder, then docs/superpowers/plans/2026-11-09-calibration-terminal-tool.md (D1-D12 decisions; sections 4-7 and
   13.0) and the M5 plan docs/superpowers/plans/2026-11-10-calibtool-M5-session.md (status section).
2. Read your memory index, especially calib-terminal-tool-plan, calib-tool-overnight-2026-11-09,
   robot-rename-phantom11-to-phantom10 (Thor names changed on 9 Nov: P10 = phantom10, P11 = phantom11, P12 = phantom12
   pending), moonstone-workspace (run tests ONLY on Moonstone with run_moonstone.sh; one BLAS thread per xdist worker),
   workstation-ram-limit, ask-in-chat-not-popups, plain-summary-first.
3. Run the fast suite once on Moonstone to confirm the branch is green.
Then give the user one short plain summary of where the tool is, and propose the next steps from HANDOFF.md's open
issues (the paused live session on phantom10 first: IMU sweep -> fit/decide -> deploy question; then the operator
experience fixes). Ask before any robot action; the user is often at the robot and wants to be told what is happening
before any key is pressed. Commits carry Co-Authored-By plus `Claude-Session: <your name> [<ref>]`; never --no-verify;
never write the user's name in files, branches or commits.
```
