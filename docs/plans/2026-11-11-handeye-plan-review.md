# Review of the hand-eye plan rev 0 (11 Nov 2026)

Two independent reads of `2026-11-11-handeye-plan.md` rev 0, each checked against the code and data the plan cites:
- the calibration agent's own check;
- a second reviewer (a subagent given only the plan, the code and the data locations, not the first read's
  conclusions).

They agree on every point they both checked; the second reviewer found more (findings 3, 4, 6, 7, 8 and 9 are mostly
its). Every new claim that changes the plan was spot-checked a second time against the source; the paths are given
below. Rev 1 of the plan (same file, rev 0 in git history) folds these findings in.

Nothing was run on a robot. On Moonstone only files were read (logs, code, burst timestamps).

## Summary in plain words

- **The phantom14 result is weaker than rev 0 says.** Its URDF corrections are real (the arm-to-camera error dropped
  from 21-36 mm to about 5 mm), but "≤ 3 mm on held-out poses" was not shown:
  - typical error per capture after the fit is about **4.5-5 mm**;
  - the held-out captures repeated arm poses that were also in the fit;
  - the worse held-out captures were left out of the quoted numbers.
- **The plan describes a different fit from the one that worked.** H0 must first port and reproduce the method as it
  ran.
- **The session design would lose what made phantom14 work.**
  - phantom14 moved the neck to 3-4 angles at each arm pose. That is what made the head-mount correction measurable.
  - phantom21 kept the neck fixed and failed.
- **The time budget is 5-15x below everything measured.** Measured costs:

  | Step | Measured |
  |---|---|
  | One burst at 1280 | 7.7 s |
  | One arm move | about 8 s |
  | One pose, end to end (phantom12) | about 19-20 s |
  | Read B on one pose, one Moonstone core | about 80 s |

- **The plan is not ready to start.** A cheap offline go/no-go on the stored phantom14 data should come first (new
  H-1). It sets the gate and the pose design from evidence.

## Findings, most important first

| # | Severity | Plan section | Finding in one line |
|---|---|---|---|
| 1 | blocker | 1, 2, 4, 9.4 | The 3 mm gate is not supported by phantom14's data |
| 2 | major | 4 | The proposed fit is not the method that worked |
| 3 | major | 5 | A fixed-neck session cannot measure the NeckYaw origin |
| 4 | major | 5 | The time budget is not backed by measurements |
| 5 | major | 2, 7 | Where read B runs is open, and porting it is heavy |
| 6 | major | 2, 5 | Read B matches the forearm and wrist only (no hand, no upper arm) |
| 7 | major | 2, 5 | Motion safety: speeds, unlisted joints, sag, arm drop on a fault |
| 8 | major | 2 | phantom21's wave was run and failed; the plan says "not executed" |
| 9 | major | 6 | Deploy: OKVIS is a definite consumer; per-consumer URDFs; post-check |
| 10 | major | 8 | Milestones: an offline go/no-go must come first |
| 11 | minor | 2, 4 | phantom12's 1.5-1.8 mm is agreement with a plate fit; robots differ widely |

How to read it: one row per finding, most important first. "Blocker" means rev 0 cannot go ahead as written; "major"
means a milestone would fail or waste robot time; "minor" means wording or context. Details follow.

### 1. The 3 mm gate is not supported by phantom14's data (blocker)

**What the data are.** Wave 2 (`p14-armwave-1104a-right`, `1104b-left`) has 5 arm poses (p04, p07, p08, p09, p10),
each captured at 3-4 neck angles. The capture name encodes them: `01-p07-10-m12` = arm pose 07, neck pitch 10°, yaw −12°.

**How it was held out.** The held-out captures (`urdfx/twoarm.py`: right 03, 08, 14, 15; left 02, 10, 15, 19) all share
their arm pose with captures in the fit. The same captures were also used to choose among about 35 model variants
(`urdfx/fitvar.py`, `twoarm.py` MODELS M0-M12, X1, X2). So they acted as a selection set, not as a test.

**What rev 0 quotes, and what the log says** (`evalM12.log`, final URDF M12):

| Arm | Rev 0 quotes | Log, all held-out captures | Error on the fit's own clean captures |
|---|---|---|---|
| Right | 2.5 / 2.0 / 3.2 mm | 2.5 / 1.7 / 2.0 / 3.2 mm (the 1.7 capture had a 25.9° bad match) | median 4.8 mm, RMS 6.4 mm, 2 of 13 within 3 mm |
| Left | 2.9 / 3.8 / 6.1 mm | 2.9 / 3.8 / **10.9** / 6.1 mm | median 4.4-4.6 mm, RMS 5.6 mm, 3 of 14 within 3 mm |

How to read it:
- each number is one capture's wrist-point disagreement between the camera (read B) and the corrected arm chain;
- "fit's own captures" are the captures used to fit, so they should score *better* than held-out ones; they score
  worse, so the held-out numbers were lucky.

**The left arm already fails a 3 mm gate** (6.1 mm, and 10.9 mm unless excluded), so "the result passed" is wrong.
The left capture 01 (48 mm) was dropped after the fact.

**Recommendation.**
- Re-run the stored fit with whole arm poses held out (5 folds per arm). This is the new H-1, on Moonstone only.
- Gate on the mean over held-out arm poses and on the paired improvement against the live URDF, not on single
  captures.
- Set the number only after that, against a measured noise floor.

### 2. The proposed fit is not the method that worked (major)

**Rev 0 §4:** one robust point-to-surface fit over URDF parameters, with a prior.

**What ran on phantom14:** two stages.
1. `meshfit.fit_pose` per capture (read B "B0": a free 6-DOF correction of the matched forearm and wrist). It gives
   the wrist offset, the coverage and the rotation of the match. A capture is clean when rotation ≤ 12° and coverage
   ≥ 0.6 (`urdfx/fitrun.py`); there is also a ≤ 15 mm distance-to-median outlier cut.
2. scipy `least_squares` of the wrist-point offsets of **both arms jointly** over 5 parameters (M12): NeckYaw xyz, and
   the right and left ShoulderRoll zero offsets (`urdfx/twoarm.py`, `predict.py`). The ShoulderRoll offsets are
   written into the URDF as the joint origin's rpy (`urdfvar.py`, `post_axis_deg`).
   - `predict.py` holds the match fixed to first order, valid only upstream of the elbow (its docstring).

**Joint zero offsets beyond those have failed before.**
- Shoulder pitch is degenerate: the meshfit docstring says "B2 with it ran off to 1e4 rad".
- B2 on phantom14 wave 1 returned wrist yaw 14.2° and camera 6.75° (`b2_14b.log`).
- OKVIS2-X commit e71b3f1c: "B2 refutes joint zeros".

**Recommendation.**
- H0 ports and reproduces the two-stage method exactly.
- A one-stage fit is a later, separate experiment.
- The candidate parameter list is fixed in advance and small. Model choice is made by cross-validation inside the fit
  data, never on the held-out set.

### 3. A fixed-neck session cannot measure the NeckYaw origin (major)

- **The correction that mattered most.** phantom14's NeckYaw shift was (−8.4, +14.9, −13.1) mm. It is measurable only
  because the neck moved between captures.
- **At a single neck pose** it is indistinguishable from a head-frame translation. Consumers at other neck angles would
  then get wrong geometry.
- **phantom21 shows the failure.** Its wave (`1102a`) kept the neck at 22.7° for every capture; its log reports
  `B-neck ... neck_ok False` (`p21arm/out/p21-armwave-1102a.log`).
- **Rev 0's neck band breaks a limit.** It names 18-28°, but `calibtool/neck.py` clamps pitch at 26°: +26° commanded
  reached +27.7° on phantom15, against a 27.5° limit.

**Recommendation.** Keep wave 2's design: a few arm poses, several neck angles each. Neck moves are cheap (the head
tool measures 1.33 s per pose); arm moves are not.

### 4. The time budget is not backed by measurements (major)

| Item | Rev 0 estimate | Measured | Source |
|---|---|---|---|
| Burst | 0.5 s | **7.73 s** (5 frames at 1280, 1.93 s apart); 0.27 s at phantom21's smaller depth | `stamp_ns` in the p14 and p12 bursts |
| Arm move | about 2.5 s | 8.1 s at ramp 6 (6 s ramp + 2 s hold); 16 s at ramp 14 | `run-sc1026b-left23.log` |
| One pose, end to end | about 4 s | 19-20 s (phantom12); 10-11 s (phantom21) | runner logs |
| Read B per capture | live, in the background | about 80 s on one Moonstone core; 9 fits + the joint fit = 7.4 min with 8 jobs | OKVIS2-X 6c33bc5e |
| Compute vs robot time | fit done 30 s after the last pose | sc1025d: 9 min 26 s of compute vs 6 min 21 s on the robot | 67ea5e96 |

How to read it: one row per cost; the measured column replaces rev 0's guess. Both validated results used the slow
1280 burst, so a shorter burst also needs its accuracy shown.

**Recommendation.** Budget from these numbers, and measure read B on a Thor CPU before promising live quality on
screen.

### 5. Where read B runs is open, and porting it is heavy (major)

- **What read B needs.** `blind_b.py` (OKVIS2-X c5b4c6b0) uses `_burstfusion` (a pybind11 C++ module from
  `okvis_burst_fusion`, OpenMP), `meshfit`, `hand_check`, `frames`, `armfk`, `mesh`, `burst_io`, scipy and cv2. It does
  not use open3d, trimesh or torch.
  - `blind_b.py` is the phantom12 plate-yardstick driver. On phantom14 the same matcher ran from ad-hoc scripts
    (`ab14w2.py`, `abL.py`).
- **Not in any repo:**
  - the arm meshes: Moonstone `~/burst-pc/meshes/cycloidal_arm/`;
  - the URDF-route fit: `urdfx/*.py`;
  - `BURST_URDF`, an uncommitted patch to `frames.py` in Moonstone's unversioned copy `/data/kaustabh/p21arm/code-repo`.
  - Otherwise meshfit is identical across c5b4c6b0, `feat/pointcloud-agent-integration` and that copy.
- **Where it can run:**
  - **Thor** (design D10: images stay on the robot) needs an aarch64 build of `_burstfusion` in the kit or image, and
    runs beside OKVIS;
  - **Moonstone** needs the bursts copied (about 55 MB each, about 24 s on phantom21's WiFi);
  - **the laptop** is out.
- **The safety check isn't tested automatically.** The repo has no MJCF, and mujoco is in neither
  `requirements-dev.txt` nor CI. `test_arm_safety.py` and one `test_arm_session.py` test are skipped in every
  automated run (the Moonstone suite run: 1699 passed, 3 skipped).

### 6. Read B matches the forearm and wrist only (major)

- `hand_check.LINKS` = ElbowPitchLink (the forearm), WristYawLink, WristRollLink. There is no hand mesh ("the hand ...
  unmodelled, never pairs", meshfit docstring) and no upper-arm mesh.
- So rev 0's "an open hand's fingers give read B a distinctive shape" does not hold.
- **The forearm-roll ambiguity was filtered, not solved, in wave 2:** 6 of 18 right captures matched with 14-39° of
  rotation, and captures at 10-12° still counted as clean.
- **The gate quantity is the wrist point** (WristRoll origin). A grasp point further out adds the lever arm times
  any rotation error.

**Recommendation.**
- Drop the hand-shape claims, or plan hand and upper-arm meshes as new work.
- State the gate at the wrist point.

### 7. Motion safety (major; the speed point is close to a blocker)

- **Speed.** Rev 0 proposes 30-40°/s ramps. The 23 Oct EtherCAT drop (7 slaves) happened at about 9-13°/s (55°
  shoulder yaw and 80° wrist roll in 6 s). Faster moves invert that lesson.
- **Unlisted joints.** Rev 0 lists the 7 arm joints plus the neck. `calibtool/neck.py` (its header, lines 4-10) pins
  **every** other joint (arms, spine, legs) at its measured angle. A re-hold of unlisted joints made phantom14's body
  creep, and the 2.1.1 master leaves unlisted joints at zero stiffness. The arm tool must do the same.
- **Sag at kp 100**, from the runner logs:

  | Joint | Measured error (sag) |
  |---|---|
  | Shoulder pitch | −5.8 to −12° |
  | Shoulder roll | −9.7° |
  | Elbow | −3 to −5.5° |

  The runner passed these moves with a 15/30° tolerance.
  - Views and clearance must be computed for the sagged pose.
  - Arrival needs a stated tolerance; "settled" cannot mean "at target".
- **A fault with the arm raised.** Rev 0 does not say what happens on an EtherCAT drop or a master stop with the arm
  raised: the arm falls. Keep poses over clear space, the operator out of the arm's fall zone, and state the recovery.
- **The other arm.** `selfcal_park.py` clears one move only. Park both arms in both orders, checked in MuJoCo.

### 8. phantom21's wave was run and failed; the plan says "planned, not executed" (major)

- **Where its data are.** `results/phantom21-2026-11-02-armwave/` is not in this repo. It is on OKVIS2-X branch
  `feat/calibration-agent-p21-armwave` (ddeaba11), with `p21view`, `p21mj` and `p21render`.
- **It was run** on 2 Nov, 23:43-23:48: 13 bursts, Moonstone `p21arm/bursts-1102a/`.
- **Result:**
  - 5 of 13 captures were excluded for low coverage;
  - plain-chain error 21-36 mm, two captures off by 72-106 mm;
  - translation-only held-out 12.5 / 3.3 / 12.6 mm;
  - fixed neck (finding 3).
- It is a third robot with a failure, and should go into the lessons and the H-1 replay set.

### 9. Deploy is under-specified (major)

- **OKVIS is a definite consumer, not "possibly".** `mk12_stereo_holevisor.yaml:306` builds its TF tree from a
  URDF (`OKVIS_URDF` / `dma:urdf`), and `okvis2x_app_dma_live.cpp:613` uses it for TF and pelvis FK. A NeckYaw change
  moves OKVIS's head-in-body pose (navigation, exploration).
- **URDFs differ per robot and per consumer.** phantom21's URDF is pre-#816 (md5 95fdb1d7). "The live URDF" and "the
  template" must be defined **per consumer**.
- **cuRobo runs in ghost-wbc-manip** and has its own arm model without the shoulder-roll terms. Rev 0 needs:
  - an owner;
  - a permission rule for restarting it;
  - its own generator.
- **Post-check.** A 2-pose check at 3 mm, against about 5 mm of noise per capture, would roll back by chance. Use the
  same statistic as the gate.
- **Missing from rev 0:**
  - a provenance header in the URDF (phantom14's file has one);
  - schema validation of the generated file;
  - a rule for the combined T_BS + URDF state. The URDF is fitted *with* a given T_BS, so a later head calibration that
    changes T_BS makes the hand-eye stale.

### 10. Milestones: an offline go/no-go must come first (major)

- **Add H-1 on stored data, Moonstone only:**
  - leave-arm-pose-out cross-validation on phantom14 wave 2;
  - a fixed-neck subset (pitch 22 only) to show whether NeckYaw is observable;
  - the noise floor from per-capture match scatter;
  - phantom21 `1102a` as a failure case.
- **H1 can't be tested offline as written.** Its "fewer poses reach the same error" has only 5 arm poses to choose
  from offline, so it can only pick subsets.
- **Decide read B's compute host** (Thor or Moonstone) before the live milestone.
- **One repeat is not a noise floor.** Rev 0's "1 repeat" measures repeatability, not the pose-dependent bias of the
  matcher.

### 11. Minor

- **"phantom12 1.5-1.8 mm"** is agreement with the plate fit B1, which itself has 1.75 mm RMS at unseen poses. It is
  not truth, and it was translation-only, camera-side, at neck 28.3° only. That "phantom12" is the robot called that
  before the 9 Nov renames, not today's phantom12 (formerly phantom21).
- **Robots differ widely.** The phantom14 prior error was 34-36 mm (right) and 21-22 mm (left), about 10x phantom12's
  3.4 mm. Any prior in the fit must allow that.
- **No independent truth on phantom14.** Its held-out error is read B against the chain, so a bias of the matcher
  itself is invisible. Examples: the mesh against the real arm covers, or the head depth scale (0.21 % = 1.7 mm at
  0.8 m). An independent yardstick (the old-phantom12 plate data, or a touch test) is the only check of that.

## Verified true

- **The arm scripts exist with tests:** `armchain`, `neckfk`, `armfit*`, `armplan`, `armsafety` and `selfcal*`, with
  `test_arm_*` and `test_selfcal`. The armsafety tests skip without mujoco, and armchain defines only the right arm.
- **The phantom12 runner:** `selfcal_run.py` ran sc1025d in 6 min 21 s of robot time, 16 min 10 s end to end
  (67ea5e96). Translation-only read B gave 1.80 / 1.53 / 1.79 mm against B1.
- **phantom14, 3 Nov, forearm-only views:** match rotation 29-49°, held-out 12.8 / 5.0 / 41.3 mm (`ab14.log`).
- **phantom14 URDF diff** (`phantom_mk1.urdf` → `phantom14_mk1_handeye-20261104.urdf`):

  | Joint origin | Before | After |
  |---|---|---|
  | NeckYaw xyz | −0.015 0 0.3815 | −0.0234 0.0149 0.3684 |
  | RightShoulderRoll rpy | 0.6981 | 0.6668 (−1.80°) |
  | LeftShoulderRoll rpy | −0.6981 | −0.6897 (+0.48°) |

  The base file is the repo's #816 URDF (md5 40879599).
- **No tag-free rotation source on phantom12:** T3, R1 and route 2 (`RECIPE-per-robot.md`).

## Could not check (robot-side or not logged)

- The inspire hand poses and the hand-state readback.
- The consumers' reload paths and cuRobo's arm model.
- The M12 condition number and standard errors (stdout not logged; re-run in H-1).
- Whether the held-out captures were named before the data were seen.
- Read B's runtime on a Thor CPU.

## Evidence paths

- **Moonstone** `/data/kaustabh/p14arm/`:
  - `evalM12.log`, `evalX_R.log`, `ab14w2.log`, `abL.log`, `ab14.log`, `b2_14b.log`;
  - `urdfx/twoarm.py`, `urdfx/predict.py`, `urdfx/fitrun.py`, `urdfx/urdfvar.py`, `urdfx/r_M12_w2.log`,
    `urdfx/r_M12_L.log`;
  - the bursts' `stamp_ns`.
- **Moonstone, other:**
  - `/data/kaustabh/p21arm/out/p21-armwave-1102a.log`;
  - `/data/kaustabh/p21arm/route/cal/results/phantom12-2026-10-26-left/run-sc1026b-left23.log`;
  - `~/burst-pc/meshes/cycloidal_arm/`.
- **OKVIS2-X:**
  - commits c5b4c6b0 (`blind_b.py`), 6c33bc5e, 67ea5e96, e71b3f1c, ddeaba11;
  - `config/foundation_rig/mk12_stereo_holevisor.yaml:306`;
  - `okvis_apps/src/okvis2x_app_dma_live.cpp:613`.
- **This repo:** `tools/calibration/calibtool/neck.py` (MAX_PITCH, joint pinning), `tools/calibration/scripts/selfcal*.py`,
  `tools/calibration/tests/test_arm_safety.py`.
