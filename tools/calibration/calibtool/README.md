# calibtool — head-stereo calibration terminal tool

One command takes a robot's head stereo pair through a full calibration:
1. guided board placements, with spoken prompts;
2. neck grids;
3. a camera-to-IMU floor sweep;
4. the fit (KB4T lens + stereo pose);
5. a decision against the robot's live calibration, deploy with a rollback, and a post-check.

The user guide (install, run, troubleshooting) is the repository's `README.md`; this file is for developers.

Design and milestones: `docs/plans/2026-11-09-calibration-terminal-tool.md` (decisions D1-D12, sections 4-13).
Per-milestone plans: `docs/plans/2026-11-09-calibtool-M{1,2,3,4}-*.md` and `2026-11-10-calibtool-M5-session.md`.
Next project (for review): `docs/plans/2026-11-11-handeye-plan.md`. State and open issues: `HANDOFF.md` (this folder).

## Running it (from a laptop)

**Users** install the released file (repository README: `gh release download …` or `curl …`; `calib --update`
later; `calib --install-voice` for the Piper voice). **Developers** build it from the tree and put it on the PATH
(again after pulling new code):
```
cd tools/calibration && python3 -m calibtool.cli --install      # -> ~/.local/bin/calib  [--install-dir DIR]
```
It is a zipapp holding the whole kit (`calibtool`, `scripts`, `config`, the robot URDF); it unpacks once to
`~/.cache/calibtool/kit-<hash>` and runs from there. `calib --version` prints the kit hash, the same as from the worktree.

**Then just `calib`** (or `calib phantom10`). It asks, in plain numbered prompts:
1. which robot (number, host name or bot name; Enter = last time's), or "new robot (setup)";
2. after one read-only look at the robot (live calibd and its kit, a held producer lease, unfinished sessions), only
   what makes sense: continue a session, start a new calibration, the placement trial, status, stop the running one;
   - a calibd running older code, or a dead session, is restarted with this laptop's kit (it comes back in HOLD: G);
   - a dead session still holding the producer lease is undone first by the robot's own restore (output shown);
   - never a second session beside a live one: stop it from the menu first;
3. the voice, once: the Piper voice (recommended) or text on screen (text without asking when Piper is not
   installed); remembered in `~/.config/calibtool/laptop.json`.

The menu also offers **discard session X** for an unfinished session (the robot's own restore first when it still
holds the producer; its folder is kept; phase DISCARDED, no longer offered).

The kit is always copied before calibd starts. If the network drops, the screen stays: it says "Connection lost",
re-attaches by itself (1, 2, 4, 8, then every 10 s) and says "Connection back"; the robot holds meanwhile (press G
when ready). Keys pressed while it is down are not sent. Q leaves; calibd keeps running.

**The scripted path** (every flag as before; `calib` here = `python3 -m calibtool.cli` in `tools/calibration/`):
```
calib phantom10 --trial            # placement trial: guidance only, nothing moves
calib phantom10 --full --dry-run   # the whole session against the robot, recording commands only
calib phantom10 --full             # the real session (motors, neck, producer, deploy on Y)
calib phantom10 --watch            # read-only watch session (a bare robot name did this before the menu)
    [--voice piper|espeak|text] [--session ID] [--replace] [--status] [--stop] [--no-speech] [--no-sync]
```

**The laptop side is standard library only;** the built `calib` file carries the pure-Python PyYAML it needs.
- It copies the kit to the robot (`<root>/kit`, root from `config/profiles.yaml`).
- It starts or attaches `calibd` over ssh, renders the state, and speaks prompts:
  - `espeak` = `spd-say` (the scripted path's default);
  - `piper` uses the local neural voice in `~/.local/share/calib-voice` (Piper binary + `en_US-lessac-medium`);
  - `text` speaks nothing: a PROMPT panel at the top of the screen shows the newest prompt (highlighted) and
    the 3 before it with their age, including the prompt that is current when the TUI attaches.

**On the robot, calibd runs with the kit venv** named by the robot's profile (P10: `/root/p11-cal/venv` + `pydeps`;
P11: `/root/calib-kit`; new robots get it from `onboard.py`). The robot image (`docker/calib.Dockerfile`, `RUNTIME.md`)
is not used yet.

**Keys (each works only in its phase; every key is answered on screen at once and cuts the speech):**
- T: start the trial
- M: motor screen
- Y: enable or take over the motors / deploy
- R on the motor screen: restart the motor master (motors off only); elsewhere: roll back
- S in BLOCKED: start the motor master (the preflight re-checks every 10 s and calibd restarts itself once green)
- G: continue after HOLD
- C: can't put it there, skip
- U: use the board where it is (placements)
- F: finish collecting (extra placements)
- 1 / 2 / ?: deploy size
- D: disable the motors
- K: keep the motors on
- E: after "not enough data", more placements (the planner asks), then fit and decide again
- X: after "not enough data" from too few judged regions (5 or more of 6), decide again accepting them (a one-off
  exception, recorded in the report; every other rule still applies)
- R (after a deploy): roll back
- O: skip the OKVIS check
- A: abort
- Q: leave the TUI (calibd keeps running)

**Safety.** Nothing in a motion phase resumes by itself: after a link loss, a crash or a reboot it comes back in HOLD and needs G (and Y). Design section 7 has the full list.

## Hand-eye calibration (first version, 11 Nov 2026)

`calib` → robot → **hand-eye calibration**, offered for robots whose profile says `handeye: true` and that have a
stored setup `config/handeye/<bot>.json` (the arm-wave plans, checked for clearance in MuJoCo against the snapshot in
the file; P10 only so far). `calibtool/handeyeflow.py`:
1. copies the kit (now with `armcal`);
2. runs the preflight per arm, read-only (`scripts/handeye_wave.py --check`): the fixed motion tool, motors on (it
   never enables them), nobody else commanding, the body within 3° of the plan's snapshot (else: re-plan on Moonstone
   first), depth flowing;
3. asks Y, then runs each arm's wave detached on the robot and shows its log; **S + Enter** stops the arm where it is
   (it holds);
4. runs read B (`scripts/handeye_match.sh`) and the fit (`armcal.urdffit`) on the robot and prints the table.

The corrected URDF stays in `<root>/handeye/<session>/`: nothing is deployed yet (milestone H4). Depth profiles:
`profiles` in the setup file (e.g. `"768"`) switches OKVIS's depth profile for the waves and restores it after.

## Developing

**Tests run only on Moonstone** (the laptop has 14 GB of RAM):
```
bash tools/calibration/calibtool/tests/run_moonstone.sh <worktree> <name> [pytest args]
```
- `tools/calibration -q -n 4` runs the fast suite.
- `-m replay` runs the slow stored-session replays, about 20 min.
- Stored sessions (`sessions.py`) live on Moonstone under `~/calibtool/data`:
  - `p21_1105`, P12 (formerly phantom21);
  - `p10_1106`, P10 (formerly phantom11);
  - the live session `p10/live-sessions/20261110-123252-P10`.

**devel PR contract.** `DEPS.txt` lists every `tools/calibration` file calibtool needs. `tests/devel_compat.sh <devel tree>` reports what devel lacks and runs the tests there.

## Layout (one responsibility per file)

| Area | Modules |
|---|---|
| data / replay | `paths`, `sessions`, `session`, `pipeline` (steps 16/45/66 wrappers, `solver="fast"`), `assign` (capture-time FIT/TEST), `coverage`, `placement`, `planner`, `replay` (`Collection` = the live per-hold step), `protocol` (constants) |
| decision | `livecfg` (OKVIS configs by `distortion_type`, `radtan_scope`), `scaling`, `decide` (paired bootstrap, gates) |
| speed | `solver`, `score`, `refit` (live refit after each placement) |
| IMU | `imu`, `imusweep`, `gravity` |
| live robot | `capture`, `detect`, `health`, `preflight`, `motors` (read-only), `motors_enable`, `neck`, `producer`, `storage`, `profile` + `config/{profiles,robots}.yaml` |
| session | `daemon` (calibd: state machine, persistence, deadman), `place` (spoken guidance), `grids`, `flow` + `flowrobot` + `sessionctl` (the full session), `deploy`, `postcheck`, `selfrestore` |
| laptop | `cli`, `menu` (the robot menu, continue / new / discard), `tui` (render, keys, speech, progress, quality), `exe` (the single-file build, `--update`), `voice` (`--install-voice`) |
| robots | `onboard` (a new robot: identity, kit, rig, profile), `quality` (per-section data status on screen) |
