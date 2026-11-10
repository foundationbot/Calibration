# calibtool — head-stereo calibration terminal tool

One command takes a robot's head stereo pair through a full calibration:
1. guided board placements, with spoken prompts;
2. neck grids;
3. a camera-to-IMU floor sweep;
4. the fit (KB4T lens + stereo pose);
5. a decision against the robot's live calibration, deploy with a rollback, and a post-check.

Design and milestones: `docs/superpowers/plans/2026-11-09-calibration-terminal-tool.md` (decisions D1-D12, sections 4-13).
Per-milestone plans: `docs/superpowers/plans/2026-11-09-calibtool-M{1,2,3,4}-*.md` and `2026-11-10-calibtool-M5-session.md`.
State and open issues: `HANDOFF.md` (this folder).

## Running it (from a laptop, in `tools/calibration/`)

```
python3 -m calibtool.cli phantom10 --trial            # placement trial: guidance only, nothing moves
python3 -m calibtool.cli phantom10 --full --dry-run    # the whole session against the robot, recording commands only
python3 -m calibtool.cli phantom10 --full              # the real session (motors, neck, producer, deploy on Y)
    [--voice piper|text] [--session ID] [--replace] [--status] [--stop] [--no-speech]
```

**The laptop side is standard library only.**
- It copies the kit to the robot (`/tmp/calibtool-m2/kit` for P10, from `config/profiles.yaml`).
- It starts or attaches `calibd` over ssh, renders the state, and speaks prompts:
  - `spd-say` by default;
  - `--voice piper` uses the local neural voice in `~/.local/share/calib-voice` (Piper binary + `en_US-lessac-medium`).
  - `--voice text` speaks nothing: a PROMPT panel at the top of the screen shows the newest prompt (highlighted) and
    the 3 before it with their age, including the prompt that is current when the TUI attaches.

**On the robot, calibd runs with the kit venv** (`/root/p11-cal/venv` + `pydeps` on P10). The CI-built image `foundationbot/okvis2x:calib-<arch>-<sha>` is the fleet runtime; see `RUNTIME.md`.

**Keys (each works only in its phase):**
- T: start the trial
- M: motor screen
- Y: enable or take over the motors / deploy
- G: continue after HOLD
- C: can't put it there, skip
- 1 / 2 / ?: deploy size
- D: disable the motors
- K: keep the motors on
- E: after "not enough data", more placements (the planner asks), then fit and decide again
- X: after "not enough data" from too few judged regions (5 or more of 6), decide again accepting them (a one-off
  exception, recorded in the report; every other rule still applies)
- R: roll back
- O: skip the OKVIS check
- A: abort
- Q: leave the TUI (calibd keeps running)

**Safety.** Nothing in a motion phase resumes by itself: after a link loss, a crash or a reboot it comes back in HOLD and needs G (and Y). Design section 7 has the full list.

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
| laptop | `cli`, `tui` (render, keys, speech) |
