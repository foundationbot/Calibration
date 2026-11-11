# Calibration

Tool for voice assisted stereo, imu and autonomous hand eye calibration.

`calib` calibrates a robot's **head stereo cameras** from start to finish:
- it tells you, by voice, where to put the calibration board;
- it moves the head over the board;
- it fits the lens and stereo calibration and the camera-to-IMU rotation;
- it compares the result with the robot's current (live) calibration;
- it deploys the new one only if it is better, keeping a rollback;
- it checks the result after deploying.

A typical run takes **13–16 minutes**: about 13 with no extra placements, plus about 1.5 per extra.

You run `calib` on your **laptop**. It talks to the robot over ssh and runs its robot-side part (`calibd`) there. You
stand at the robot with the board, and the tool speaks every instruction aloud. The same instruction is always on
the laptop screen too.

---

## 1. What you need

**On your laptop (Linux):**
- `python3` 3.10 or newer (every current Ubuntu has it). Nothing else to install: `calib` is one self-contained file.
- **root ssh to the robot without a password**: `ssh root@phantom10 true` must work. Ask whoever manages robot access
  to add your key.
- Optional, for the spoken instructions: the Piper voice (section 2) and a sound player (`pw-play` or `aplay`).
  Without them the instructions are shown on screen only.

**On the robot:** it must be set up for the tool once ("onboarded", section 5). phantom10 and phantom11 already are.

**At the robot:**
- the robot hanging on its gantry;
- the calibration board (the 7 x 5 AprilTag grid) on a chair-like stand;
- the robot's motor master (`dma-ethercat`) running. If it isn't, the tool offers to start it.
- Loco / walking policies stopped. The tool refuses to move the head while they run, and it never stops them for you.

---

## 2. Install

**Download `calib`** into `~/.local/bin` (any folder on your `PATH` works):

```bash
mkdir -p ~/.local/bin
# with the GitHub CLI (works whether the repository is public or private):
gh release download --repo foundationbot/Calibration --pattern calib --dir ~/.local/bin --clobber
# or, while the repository is public, without gh:
curl -L -o ~/.local/bin/calib https://github.com/foundationbot/Calibration/releases/latest/download/calib
chmod +x ~/.local/bin/calib
calib --version
```

If `calib --version` says "command not found", add `~/.local/bin` to your `PATH`: put
`export PATH="$HOME/.local/bin:$PATH"` in `~/.bashrc` or `~/.zshrc` and open a new terminal.

**The voice (optional, recommended):**

```bash
calib --install-voice        # downloads the Piper voice (~60 MB) into ~/.local/share/calib-voice
calib --voice piper --say "Voice test"     # you should hear it
```

**Updating** later:

```bash
calib --update
```

---

## 3. Run a calibration

In a terminal on your laptop:

```bash
calib
```

1. **Which robot?** Type its number or name (e.g. `phantom10`).
2. **What now?** The tool looks at the robot first. It offers only what makes sense:
   - *start a new calibration*;
   - *continue session …*: an earlier session that did not finish. Its data is kept.
   - *discard session …*: an unfinished session you don't want. Its data stays on the robot, and it is no longer
     offered.
   - *placement trial*: practise board placement; nothing moves.
   - *status*, *quit*.
3. **Voice:** Enter takes Piper. "Text on screen" gives no voice and a bigger prompt panel.

Then the screen opens. From here the tool guides you; you mostly press a key when it asks.

### What happens, in order

| Step | What you do | What the tool does |
|---|---|---|
| Checks | nothing (about 10 s) | checks the robot: identity, cameras, sync, IMU, motor master |
| **M** | press M | opens the motor screen |
| **Y** | press Y once the robot hangs on the gantry and nobody is at its head | switches the motors on (or takes them over if they're already on) |
| Setup | step away from the head | sets up the cameras for the collection, then **turns the head to the middle** (after a countdown) |
| 6 placements | put the board where the voice says, wait for "Good", step back | counts down ("Step back. 3, 2, 1"), sweeps the head over the board (~65 s per placement) |
| Extras (0–6) | the same, if asked | only when a gap would make the calibration unreliable; each says what it's for |
| IMU sweep | lay the board **flat on the floor** in front of the robot | sweeps the head for about 80 s |
| Fit + decision | nothing (~20 s) | fits, and compares with the live calibration |
| Deploy | choose the size (**1** = the robot's usual size), then **Y** | writes the new calibration, restarts OKVIS, checks that OKVIS loaded it |
| Post-check | board at about 1.1 m (or **C** to skip) | checks depth with the new calibration against the old one |
| End | **K** keeps the motors on; **D** twice switches them off | puts the cameras back as they were and writes the report |

How to read it: one row per step, in order. The middle column is what you do, the right column what the tool does
meanwhile. The tool never moves the head without a spoken countdown first.

### Keys

Each key only works where the screen offers it. The bottom line of the screen always lists the keys that work now.

- **A**: abort, at any time. What is moving finishes its pose; then the robot is put back as it was.
- **G**: continue after a HOLD. The tool holds when something is not right (e.g. the laptop's link dropped); it says
  why. A countdown runs again before anything moves.
- **C**: can't put the board there. Skips this placement.
- **U**: use the board where it is (e.g. the near placement can't get closer than 60 cm).
- **F**: finish collecting (during extra placements). Goes on to the IMU sweep and the fit.
- **R**: on the motor screen: restart the motor master (only while the motors are off). At the end: roll back the
  deploy.
- **S**: when the checks are blocked because the motor master isn't running: start it.
- **E / X**: if the result is "not enough data": **E** asks for more placements; **X** decides with what there is
  (a one-off exception, written in the report).
- **Q**: leave the screen. The robot holds; run `calib` again and choose *continue* to come back.

Every key you press shows at once on the screen, and then what it did (or why it was ignored).

### The screen

- **Top:** robot, session, phase, motors, link, and any NOT NORMAL warnings.
- **The prompt:** what the voice just said.
- **During a sweep:** a progress bar and the time left.
- **During collection:** a map of how much data each part of each camera's image has (`ok` = enough), and what is
  still needed.

---

## 4. Where the results are

On the robot, each session has its own folder: `/data/calib/sessions/<session>/`:
- `report/session.md`: the outcome, the numbers, what was deployed.
- `deploy/rollback.sh`: puts the previous calibration back (the tool's **R** does the same).
- `commands.txt`: every command the tool sent to the robot.

The deployed calibration is `/etc/phantomos/okvis/<robot>_stereo_kb4t_<date>_1280.yaml`. `robot.env` names it.

---

## 5. A new robot

Run `calib`, choose **new robot (setup)**, and type its host name. The tool:
- reads the robot's identity (machine-id);
- finds its head cameras and its live calibration. If the robot has none, it lists the candidates and asks you to
  pick one.
- installs what the robot needs (a Python environment and the motor tool) if it isn't there;
- adds the robot to the tool's list.

It shows **everything it will change**, on the robot and in this repository, **before** asking Y. Run it again
any time: on a robot that is already set up, it changes nothing.

The setup adds files to this repository (`tools/calibration/calibtool/config/robots.yaml`, `profiles.yaml` and a
rig file). Commit and push them, and make a new release, so other laptops know the robot too.

---

## 6. When something goes wrong

| You see | What it means | What to do |
|---|---|---|
| **BLOCKED** … *master* | the motor master isn't running (e.g. after a reboot) | press **S**; the checks run again every 10 s and it continues by itself |
| motor screen: *drives not answering* | the drives have no power | switch the robot's motor power on; when the line is ok, press **Y** |
| motor screen: *<joint> drive faulted during the enable* | that joint's drive refused, so the master left all motors off | press **R** (restart the motor master), then **Y**. If it repeats, the joint needs a hardware look (encoder, cable). |
| **HOLD** | the tool stopped and is waiting | read the reason on the screen; press **G** to continue |
| *Connection lost … reconnecting* | the laptop's link to the robot dropped | wait; it reconnects by itself. The robot holds; press **G** when back. |
| the voice keeps asking for a spot you can't reach | the stand can't get there | **U** (use it where it is) or **C** (skip) |
| *not enough data* at the end | too few parts of the image were checked | **E** for more placements, or **X** to decide anyway |

How to read it: find the message you see in the left column; the right column is what to do.

If it's something else, press **Q** (the robot holds), run `calib` again, and choose *status*.

---

## 7. For developers

- **Code:** `tools/calibration/calibtool/`. Its own README lists the modules. The fit and validation pipeline it
  builds on is in `tools/calibration/scripts/`.
- **Robots and profiles:** `tools/calibration/calibtool/config/robots.yaml` and `profiles.yaml`.
- **Tests:** `python3 -m pytest tools/calibration -n 8`, with the pinned dependencies of
  `tools/calibration/requirements-dev.txt`. Run them on a big machine, not a small laptop. They take about a minute on
  8 cores.
- **CI** (CircleCI, `.circleci/config.yml`): every push runs the tests and builds `calib` (kept as an artifact).
- **Releasing for other laptops:** `tools/calibration/release.sh` builds `calib` from the pushed commit and attaches
  it to a GitHub release; `calib --update` and new installs take the newest.
- **A local build:** `cd tools/calibration && python3 -m calibtool.cli --install` (writes `~/.local/bin/calib`).

Design notes and plans are in `docs/plans/` (the head calibration's decisions D1-D12; the next project, hand-eye
calibration, is `docs/plans/2026-11-11-handeye-plan.md`, a proposal under review). The state of the work and the
handoff between sessions: `tools/calibration/calibtool/HANDOFF.md`.

History: this repository was split out of OKVIS2-X (`tools/calibration`, with its history) on 10 Nov 2026.
