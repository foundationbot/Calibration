# calibtool handoff — 13 Nov 2026

Written by the calibration agent session `CalibrationV11 [2cedf9]` (it was `[9c385e]` on 11 Nov; the name is the
address) for the next one. Read this first, then:
- `README.md` (repo root): the user guide;
- `tools/calibration/calibtool/README.md`: modules and keys;
- `tools/calibration/armcal/README.md`: read B and the hand-eye fit;
- `docs/plans/2026-11-11-handeye-plan.md` (rev 1, with the user's answers in 9.1) and its review
  `docs/plans/2026-11-11-handeye-plan-review.md`;
- `docs/plans/2026-11-09-calibration-terminal-tool.md` (the head tool's decisions D1-D12).

## In one paragraph

`calib` does the head-stereo calibration end to end, and now also a **hand-eye (camera-to-arm) calibration** from its
menu:
- plan → arm waves → read B on the robot → URDF fit → deploy with rollback.

Hand-eye has run live on phantom10 twice (11 and 13 Nov). Nothing hand-eye has been deployed yet.
- **The current default model is M12+headR:** NeckYaw origin, both shoulder-roll zeros, and the **camera's rotation
  in the head**, found on 13 Nov.
- **Accuracy on arm poses left out of the fit:** 3.9 / 3.6 mm (right / left) on 11 Nov's data; 5.7 / 11.2 mm on 13 Nov's.
  The left arm on 13 Nov had only 7 usable captures.

## 1. Where things are

- **Repository:** `foundationbot/Calibration` (public), local clone `~/CODE/Calibration`, branch `main`. Agents may push
  (the user said so).
- **CircleCI:** runs on every push (tests + the single-file `calib` artifact). Releases are made by hand with
  `tools/calibration/release.sh`; the last one is v2026.11.10. **None contains the 12-13 Nov fixes yet.**
- **Tests:** only on Moonstone:
  ```
  bash tools/calibration/calibtool/tests/run_moonstone.sh ~/CODE/Calibration <name> tools/calibration -q -n 8
  ```
  Result at handoff: 1925 passed, 3 skipped. Never run suites on the laptop (14 GB, it crashes).
- **Merged branches** (their worktrees in `~/wt/calib-{h0,readb2,h4,cam,headr}` can be removed):
  - `feat/calibration-agent-handeye-h0`
  - `-readb-robust`
  - `-handeye-deploy`
  - `-camera-recovery`
  - `-headr`
- **On Moonstone:**
  - the test data is under `~/calibtool/data` (`p14arm/`, `p10arm/` symlinks);
  - phantom10's arm planning is in `/data/kaustabh/p10arm/` (`plan_p10*.py`, snapshots, plans, the 11 Nov bursts in
    `live-1111a/`);
  - the MuJoCo venv and MJCF are in `/data/kaustabh/p21arm/arm-venv` and `/data/kaustabh/p21arm/mjcf`.
- **Results in the repo:** `tools/calibration/results/`:
  - `handeye-h1-2026-11-11` (offline go/no-go);
  - `handeye-h0-2026-11-11` (port, speed);
  - `handeye-readb2-2026-11-11` (robust matcher);
  - `handeye-p10-2026-11-11` (first live session, plans, Thor timing);
  - `handeye-p10-2026-11-13` (second live session, the camera-rotation finding).

## 2. Hand-eye: what exists

| Piece | Where | State |
|---|---|---|
| Read B (match the arm mesh in head depth) | `armcal/readb.py`, `meshfit.py` (robust matcher, two passes + settled check) | about 2 s per capture on Thor; on the robot over a whole wave: `scripts/handeye_match.sh` (30-36 s per arm, 4 processes) |
| Stage-2 URDF fit | `armcal/urdffit.py` | default M12+headR; candidates M12+head, M12 (fallback), M9, X1; leave-one-arm-pose-out CV chooses; 15 mm outlier cut |
| Arm-wave runner (robot side) | `scripts/handeye_wave.py` | every joint listed and pinned; motor, foreign-writer and STOP checks; slow ramps (≤ 5°/s, ≥ 6 s); burst retries 2+10+20 s; `--profiles`; `--pins` (legs); `--check` (preflight, read-only) |
| Per-robot plan | `config/handeye/P10.json` (only P10) | made on Moonstone with MuJoCo (`results/handeye-p10-2026-11-11/plan_p10c.py`): clearance ≥ 3 cm incl. 1.3x the measured sag, from a snapshot; the preflight refuses when the body moved > 3° from it (neck excepted) |
| Menu session | `calibtool/handeyeflow.py` | profile opt-in `handeye: true` (P10); kit copy → preflight → Y → waves detached with a live log (S stops) → read B → fit → deploy offer (dry run, Y, R) |
| Deploy | `calibtool/handeye_deploy.py` | URDF + camera term (rotation and translation) into a copy of the LIVE OKVIS config's T_BS, robot.env switch, OKVIS restart, load check, rollback; all checks in its docstring; **live dry run OK on 13 Nov**, never deployed for real |
| Post-check after a deploy | `handeye_deploy.postcheck_scores/verdict` | rule exists (paired, no single-capture threshold); **not wired** into the menu |

## 3. Results so far (phantom10)

| Data | Model | Right median / p90 | Left median / p90 | Unmodified URDF |
|---|---|---|---|---|
| 11 Nov (10 Nov head geometry) | M12+head | 7.0 / 11.5 | 5.1 / 8.4 | 45 / 38 |
| 11 Nov | **M12+headR** | **3.9 / 10.4** | **3.6 / 8.4** | 45 / 38 |
| 13 Nov (13 Nov head geometry) | M12+head | 9.1 / 12.8 | 17.1 / 22.3 | 32 / 32 |
| 13 Nov | **M12+headR** | **5.7 / 10.5** | **11.2 / 14.8** (7 captures) | 32 / 32 |

How to read it:
- mm at the wrist on arm poses left out of the fit;
- the user's aim is < 6 mm.

**Fitted values:**
- **Camera rotation in the head (HEADR, deg):**
  - 11 Nov: (−0.3, +3.4, 0.0);
  - 13 Nov: (−2.7, +2.1, −1.5).
  The steady +3.4° about y appears on both days. The extra −2.4° (x) and −1.5° (z) appeared with **13 Nov's head
  refit** (another session's, `/data/calib/scratch-claude`, config `phantom10_stereo_kb4t_20261113_640_imu.yaml`).
  That refit's camera rotation looks 2-3° off: tell its owner / the user, since it affects SLAM too.
- **Shoulder-roll zeros:** stable across days: right −1.2 to −1.7°, left −0.9 to −1.2°.

## 4. phantom10 (P10) at handoff

- **OKVIS is depth-only, for the hand-eye night** (the user's go):
  - DaemonSet `positronic/okvis2x`, `OKVIS_MODE=depth`;
  - config `/etc/phantomos/okvis/phantom10_stereo_kb4t_20261113_1280_s2m2Lprofiles.yaml`: 13 Nov's 640 calibration scaled
    to 1280 by `calibtool.scaling`, plus the point-cloud agent's S2M2-L profiles 1280/1024/768/512, switched through
    `/dev/shm/okvis_depth_profile`;
  - producer at 1280x800 @ 15;
  - node label `has-okvis=true`; the `nav-agent` DaemonSet deleted.
- **To put the SLAM stack back:** `bash /root/handeye-night-backup-20261113-123227/restore-slam.sh`. It restores robot.env,
  the okvis-env ConfigMap, nav-agent, the label and the 640x400 @ 25 producer, then runs `okvis.sh start`.
- **Motors are on** (enabled 13:43 through `motors_enable.Enabler`, all checks passed); the arms are at rest.
- **The calib kit** at `/data/calib/kit` is at the current `main`.
- **Data:**
  - `/data/calib/handeye/20261113-125417/`: right, left, `r768/` captures, records, fits, `fitR/` (the dry-run deploy
    folder);
  - `/data/calib/handeye/p10-wave-1111a-*` (11 Nov).
- **The SLAM snapshot** of 13 Nov 12:32 is in `/root/handeye-night-backup-20261113-123227/`.

## 5. Hardware on phantom10

The 48 V brownouts and the head-camera USB drops of 11-13 Nov are **fixed** (the user, 13 Nov). Still worth knowing:
the motor master's 30 s stale-command fallback ("holding last/current") moved the ankles by itself on 11 Nov; the
tool's keep-alive now avoids it during sessions (the user said: keep a note only).

## 6. Root causes found 11-13 Nov (don't re-learn them)

- **No camera restart:**
  - `daemon._health` raised NameError (health imported only locally) after the tool's own producer restart (fixed);
  - nothing executed health's `producer_restart` action; `camrecover.py` does now, in **full sessions only, not the
    trial**.
- **Frames all "empty" after a producer restart without a camera reset:** the frame pool's slot sequence ran a constant
  offset (−52036 left, −52113 right) from the meta queue's frame sequence, so dma_ipc's `read(meta)` refused every
  frame. `capture.DmaCameras` learns the offset (fresh frames only; recycled and torn reads still refused).
- **Placements accepted silently:** a default placement accepted its whole band (0.75 m = 0.63-0.93 m). They now aim
  within max(6 cm, 8 %) and "Good" says the distance.
- **"Arm moving while the head sweeps":** pins were re-read every grid, so a displaced arm sank grid by grid (kp 30).
  The body is now pinned once per session, a displaced joint returns slowly after the countdown, and a motionless
  keep-alive (20 s) stops the master's 30 s fallback. Measured: a pinned hanging body moves < 0.1° during a 66° yaw
  sweep.
- **"I cannot see the board, so I cannot use it":** the head now searches (8 small slow moves, said first) when the
  board is not seen for 10 s or on U, and says where to move the board.
- **Hand-eye session plumbing:**
  - the detached wave must redirect its whole group (else ssh hangs and the laptop loses the session);
  - an old log must be moved aside before a retry (else follow() reads the old STOP);
  - `pgrep -f <pattern>` matches your own ssh shell: use the `[h]` trick.
- **Depth profile switch:** the grabber's timeout follows the recent rate; right after 768 → 1280 it can time out
  (retries cover it).
- **phantom10 joint conventions:**
  - its own URDF `/etc/phantomos/okvis/mk1_1.urdf` (MJCF-generated, md5 cbf58ef1) differs from #816 (left wrist 7 mm);
    match and fit with it;
  - the system `/usr/bin/dma_motor_motion` lacks `targets`: use `/root/p11-cal/dma_motor_motion`;
  - `dma_cmd_client state` prints zeros: use `calibtool.motors.master()`.
- **Sag with the tool's default gains (kp 30)** is large (shoulder pitch up to 24°); the fit uses measured angles, but
  views and clearance must include sag.

## 7. How the user works

- **Plain summary first**, then details; tables with a "how to read it".
- Ask in chat, never with pop-up questions.
- **Tell them about hardware issues** (they asked).
- **At the robot:** say what happens before any key press or motion; never silent; Piper voice; only the operator skips.
- **Thorough first:** find the root cause and check it on the real robot read-only before saying "try again".
- **Permissions given:**
  - phantom10 for overnight / free use;
  - may enable the motors, restart the motor master, move arms, head and hands;
  - kp ≤ 100 (joints oscillate), kp only, never kd;
  - loco is the operator's;
  - no tags or plates on robots.
- **Never write the user's name in files, branches or commits.** Commits end with `Co-Authored-By: Claude …` and
  `Claude-Session: <name> [<ref>]` (from `ListAgents`).
- Quote times only from measurements; label estimates.
- They want the tool **finished**, including a decent hand-eye calibration.

## 8. Next steps (proposals: evaluate them, don't just follow)

1. **Finish hand-eye on phantom10:**
   - redo the left wave (13 Nov had only 7 usable left captures), fit M12+headR;
   - decide the deploy target with the user: the temporary depth config, or the SLAM config after `restore-slam.sh`
     (the geometry check requires the same T_BS/T_SC as the fit);
   - deploy, roll back once (R), deploy again;
   - wire and run the post-check (2 poses per arm, `postcheck_*`).
2. **The 13 Nov head refit's rotation (2-3°):** raise it with the user. Consider a fresh head calibration with `calib`
   (with all the fixes) before the final hand-eye, since hand-eye depends on the head geometry.
3. **Live-check the 12-13 Nov head-tool fixes** in a real session with the operator:
   - camera recovery, keep-alive, session pins, head search, aim placements;
   - the scripted full-session stall test (`/data/calib/handeye/fulltest.py`) was blocked by the auto-mode
     classifier: the user must allow or run it.
4. **Faster depth:** analyse 1280 vs 768 on the 13 Nov right arm (24 poses at both, `right/r768/`; matching it was cut
   by a power loss, since fixed). Then set `profiles` in `config/handeye/P10.json` (now "1280,768", which doubles captures).
5. **Make it general:**
   - automatic per-robot planning (H1: today `plan_p10c.py` by hand on Moonstone; the menu refuses when the body moved);
   - other robots' setups;
   - camera recovery in the trial.
6. **Consumers:**
   - the point cloud's `frames.py` hard-codes the #816 URDF (its owner, Pointcloundv11, must select a per-robot URDF);
   - cuRobo and box-pick are not updated;
   - shoulder-roll zeros help policies only if put into the master's joint offsets (later; the user's decision).
7. **A release** with the 12-13 Nov fixes (`release.sh`); a colleague is waiting to run `calib`.

## 9. Successor prompt (paste into a new session)

```
You are the calibration agent for `calib` (foundationbot/Calibration, local clone ~/CODE/Calibration): the head-stereo
calibration tool, now also doing hand-eye (camera-to-arm) calibration from its menu. Start by getting up to date --
touch no robot, change no code, push nothing:
1. Read tools/calibration/calibtool/HANDOFF.md (state at 13 Nov, results, robot state, root causes,
   how the user works, next steps), then the repo README.md, tools/calibration/calibtool/README.md,
   tools/calibration/armcal/README.md, docs/plans/2026-11-11-handeye-plan.md (+ its review) and
   tools/calibration/results/handeye-p10-2026-11-13/README.md.
2. Read your memory index, especially handeye-autonomy-2026-11-11, calib-terminal-tool-plan,
   calib-operator-communication, moonstone-workspace, workstation-ram-limit, ask-in-chat-not-popups,
   plain-summary-first, kp-only-gain-overrides, no-plates-on-robots, robot-rename-phantom11-to-phantom10.
3. `git -C ~/CODE/Calibration status -sb`; run the full suite once on Moonstone
   (tools/calibration/calibtool/tests/run_moonstone.sh); read-only, look at phantom10's state (is it reachable, which
   OKVIS runs, motors) and say what you find.
Then evaluate HANDOFF.md section 8 (next steps) critically -- order, what is missing, what you would change -- and give
the user a short plain summary: where the tool stands and your proposed plan. Start no robot work
and no milestone without the user's go. Commits carry Co-Authored-By plus `Claude-Session: <your name> [<ref>]`; never
write the user's name in files, branches or commits.
```
