# calibtool handoff — 11 Nov 2026

Written by the calibration agent session `CalibrationV10 [9d7ba3]` for the next one. Read this first, then
`README.md` (repo root, the user-facing guide), then `tools/calibration/calibtool/README.md` (modules and keys), then
`docs/plans/2026-11-09-calibration-terminal-tool.md` (decisions D1-D12) and `docs/plans/2026-11-11-handeye-plan.md`
(the next project, for review).

## Where things are

- **Repository:** `foundationbot/Calibration` (public, the user's choice), local clone **`~/CODE/Calibration`**.
  - It was split out of OKVIS2-X's `tools/calibration` (with history, same layout) on 10 Nov.
  - The old branch OKVIS2-X `feat/calibration-agent-calib-tui` stops at 44c51a04 and is frozen; work here.
- **Pushing:** the user pushed `main` on 10 Nov; the last docs commit (this HANDOFF, the hand-eye plan) may still be
  unpushed. The agent's pushes to this public repo are blocked by Claude Code's safety check: the user pushes
  (`git push origin main`) and makes releases (`tools/calibration/release.sh`; first one v2026.11.10-f57cb3c9), unless they add a
  permission rule. Check `git status -sb` and `gh release list --repo foundationbot/Calibration` first.
- **CircleCI:** about to be set up, not done yet.
  - The org uses the CircleCI **GitHub App**; an org owner must give the app access to the Calibration repo, then
    "Create Project" in CircleCI.
  - `.circleci/config.yml` is ready: the tests with `tools/calibration/requirements-dev.txt`, and the `calib` file as
    an artifact. It has never run.
- **The `calib` command:**
  - one self-contained file (python3 only; PyYAML bundled), installed on the user's laptop at `~/.local/bin/calib`;
  - `calib` opens the robot menu;
  - `calib --update` installs the latest GitHub release (none published yet); `calib --install-voice` installs Piper;
  - after code changes, `python3 -m calibtool.cli --install` in `tools/calibration` refreshes the local copy.
- **Tests run only on Moonstone** (the laptop has 14 GB and is usually nearly full):
  `tools/calibration/calibtool/tests/run_moonstone.sh <worktree> <name> tools/calibration -q -n 8`. 1699 passed,
  3 skipped at handoff.

## What the tool does now

- **The head calibration end to end:**
  - voice-guided placements, neck grids (1.33 s per pose), extras from the planner, the IMU floor sweep;
  - the fit (about 20 s on Thor) and the decision against the live calibration;
  - deploy (size choice, transaction, rollback), post-check, report.
- **Robots onboarded:** phantom10 (P10) and phantom11 (P11). `calib` → *new robot (setup)* onboards others
  (`calibtool/onboard.py`, one summary and one Y, machine-id-fenced writes).
- **Deployed with the tool on 10 Nov:**

  | Robot | Session | New depth error | Live depth error | New row error | Live row error |
  |---|---|---|---|---|---|
  | phantom11 | `20261110-145300-P11` | 0.22 % | stand-in 15.5 % | 0.21 px | 4.56 px |
  | phantom10 | `20261110-183533-P10` | 0.21 % | 4.71 % | 0.23 px | 5.92 px |

  How to read it: one row per deployed robot; "live" is the calibration that was on the robot before.

## Robot state at handoff

- **phantom10:**
  - OKVIS runs the new calibration;
  - motors off (the master restarted 19:03);
  - producer 1280x800 @ 25;
  - the tool folder is `/data/calib`.
  - The left shoulder pitch drive had a standing encoder fault after a power cycle. It cleared after a later reboot
    and re-power; watch for it.
- **phantom11:**
  - OKVIS was not running at the last check;
  - its robot.env points at the tool's calibration;
  - loco was running again (the operator's);
  - its producer was left at 1280x800 @ **15** fps by an old bug. The user said to leave it.
  - Kit: `/root/calib-kit`.

## Root causes found on 10 Nov (don't re-learn them)

- **The motor master's ping prints the reply to the PREVIOUS request:** blank right after the master starts, else
  stale. `motors.ping` asks until a fresh reply. Every "motors off / someone else enabled them / not verified"
  contradiction came from this.
- **OKVIS on okvis.sh robots:** the pod is `okvis/okvis2x` with the label `app.kubernetes.io/name=okvis`, not
  `=okvis2x`. With only the latter, the tool never found OKVIS and every deploy waited 3 min.
- **This OKVIS build's KB4T line** reads `KB4T (VIO: equidistant k1..k4) pinhole camera N`.
- **A drive that faults on enable** (an encoder) makes the master leave all motors off. The motor screen names it;
  **R** restarts the master.
- **`dma-ethercat` is not started at boot** on phantom10. **S** in BLOCKED starts it, and the preflight re-checks
  every 10 s.
- **From phantom10, the name `phantom11` resolves to phantom10 itself** (an old name). Copy robot-to-robot by IP and
  check the machine-id.

## How the user works (read before talking to them)

- **At the robot:** they stand at the robot, not at the laptop. Say what happens before any key press; short plain
  status; never silence. Ask questions in chat (no pop-ups).
- **Thorough first:** they were frustrated by repeated small fixes. **Investigate thoroughly and verify against the
  real robot read-only before saying "try again".** Find the root cause, check it with the real master / OKVIS / logs,
  then fix.
- **Voice:** Piper; prompts short; the countdown is spoken as "Step back. 3, 2, 1" only; any key cuts the speech.
- **The operator decides:** no automatic skips. **U** uses the board where it is; **C** skips; **F** finishes
  collecting; **E/X** after not enough data.
- **Robot permissions:**
  - neck moves inside the tool are fine;
  - restarting the motor master via the tool's **R/S** was approved on 10 Nov;
  - loco is the operator's, never touched;
  - kp only, never kd;
  - no tags or plates on robots.
- **Writing:** never write the user's name in files, branches or commits. Commits end with
  `Co-Authored-By: Claude …` and `Claude-Session: <name> [<ref>]` (read the name off `ListAgents`).
- **Timing:** quote times only from measurements; label estimates as estimates.

## Open issues, most important first

1. **Push and first release** (the user), then CircleCI's first run (an org owner gives the app access).
2. **Hand-eye calibration:** `docs/plans/2026-11-11-handeye-plan.md` is at rev 1 after an independent review
   (`docs/plans/2026-11-11-handeye-plan-review.md`, session CalibrationV11). It waits for the user's answers (its
   section 9). The first milestone is now H-1, an offline go/no-go on Moonstone with no robot.
3. **Not yet measured live:** the deploy time after the OKVIS-label fix (should be seconds), and the
   step-scaled-ramp neck timing on a robot other than phantom10.
4. **The menu's look** doesn't yet list sessions under a profile's old root (`/tmp/calibtool-m2` on phantom10, all
   old).
5. **phantom11's producer** at 15 fps (left by the user's choice).
6. **Kept for later:**
   - the stereo health monitor (separate project);
   - the gyro-based third IMU angle and time offset;
   - the robot-side calib image (robots use the copied kit);
   - CircleCI auto-releases on tags.

## Successor prompt (paste into a new session)

```
You are the calibration agent for `calib`, the head-stereo (and next: hand-eye) calibration tool, now in its own
public repo foundationbot/Calibration, local clone ~/CODE/Calibration (not ~/wt, not OKVIS2-X; the OKVIS2-X branch
feat/calibration-agent-calib-tui is frozen). The tool runs from the user's laptop (`calib`, a single file in
~/.local/bin) and drives robots over ssh.

First, only get up to date — touch no robot, change no code, push nothing:
1. Read tools/calibration/calibtool/HANDOFF.md (state, root causes found, how the user works, open issues), the repo
   README.md, tools/calibration/calibtool/README.md, docs/plans/2026-11-09-calibration-terminal-tool.md (D1-D12) and
   docs/plans/2026-11-11-handeye-plan.md.
2. Read your memory index, especially calib-terminal-tool-plan, calib-operator-communication, moonstone-workspace
   (tests run ONLY on Moonstone via tools/calibration/calibtool/tests/run_moonstone.sh; one BLAS thread per worker),
   workstation-ram-limit, ask-in-chat-not-popups, plain-summary-first, no-plates-on-robots, tagfree-selfcal-thread,
   phantom14-calibration-rowcheck (the URDF-route hand-eye that worked), arm-moves-allowed, kp-only-gain-overrides.
3. Check `git -C ~/CODE/Calibration status -sb` (was main pushed?) and run the full suite once on Moonstone.
Note: CircleCI is about to be set up (an org owner must give the CircleCI GitHub App access to the repo) but is not
done yet; the config is in .circleci/config.yml and has never run.

Then review docs/plans/2026-11-11-handeye-plan.md critically before anything is built: get a second reviewer's
independent read (a subagent given the plan and the code, not your conclusions), check its claims against the code
and data it cites (blind_b.py in OKVIS2-X's point-cloud branches, Moonstone /data/kaustabh/p14arm/, the scripts in
tools/calibration/scripts/arm*.py, selfcal*.py), and write the review findings and a rev 1 of the plan. Give the
user a short plain summary, the plan's open questions (its section 9) and your recommendations; start no milestone
without their go. Ask before any robot action; tell the user what happens before any key is pressed. Commits carry
Co-Authored-By plus `Claude-Session: <your name> [<ref>]`; never write the user's name in files, branches or commits.
```
