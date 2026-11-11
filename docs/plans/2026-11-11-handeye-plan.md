# Hand-eye calibration in `calib` — plan rev 0 (11 Nov 2026), for review

Status: **proposal, not started.** It needs a reviewer's pass (the successor session's first job) and the user's
answers to section 9 before any milestone starts. No robot motion comes from this document.

"Hand-eye" here means **camera-to-arm**: after it, a point the head cameras see and the hand's pose computed from
the joint angles agree, so a grasp planned in the camera's view lands where the hand goes.

---

## 1. Goal

For one robot, both arms, run from the same `calib` tool:
- **Tag-free.** No plate, AprilTag or fixture on the robot (the user's rule since 24 Oct: robots get swapped; per-robot
  fixtures don't scale). The free-standing board is used only by the head stereo calibration.
- **Output:** a per-robot URDF whose kinematic corrections make camera and arm agree to **≤ 3 mm at the hand** on
  held-out poses (proposal; section 9). It is **deployed on the robot automatically**: the programs that read the
  URDF are switched to it, with a rollback, and only if it beats the live URDF on the same held-out data.
- **Fast:** ≤ about 5 min of arm motion for both arms, the fit finished seconds after the last pose. Past runs were
  16 min for one arm (phantom12 `selfcal_run.py`), plus hours of hand work around it.
- **The same operator experience** as the head calibration: voice, screen, the operator's Y before motion,
  countdowns, HOLD / abort, progress and quality on screen, report, discard / continue.

---

## 2. What exists (reuse, don't rewrite)

| What | Where | State |
|---|---|---|
| URDF forward kinematics of the arms and neck | `tools/calibration/scripts/armchain.py`, `neckfk.py` | tested |
| Fit models with a plate (M0/M1/M2: plate X, camera ΔT, joint zero offsets δ) | `scripts/armfit.py`, `armfit_multi.py`, step 92 | tested; plate route is **not** the target any more |
| Pose list from a view and information model | `scripts/armplan.py`, step 90 | plate-based; its information and coverage logic carries over |
| MuJoCo clearance and gravity sag | `scripts/armsafety.py` | tested (needs `mujoco`, the MJCF) |
| One-command arm-wave runner, park the other arm | `scripts/selfcal.py`, `selfcal_run.py`, `selfcal_park.py` | ran live on phantom12 (25-26 Oct) |
| phantom21 arm-wave route (view model, MuJoCo variants) | `results/phantom21-2026-11-02-armwave/` | planned, not executed |
| **Tag-free arm matcher "read B"** (arm mesh registered in head depth) | OKVIS2-X `tools/burst_pointcloud/experiments/arm816/blind_b.py` on a point-cloud branch (`git log --all -- '*blind_b.py'`) | **not in this repo**; trans-only passed blind on phantom12 (1.5-1.8 mm) |
| **URDF route fit** (phantom14, 4 Nov: NeckYaw origin + shoulder-roll rpy per arm) | ad-hoc scripts on Moonstone `/data/kaustabh/p14arm/` (`ab14*.py`, `b2_14*.py`, `evalX.py`, `urdfx/`), bursts there too | **not in this repo**; the result passed: held-out R 2.5/2.0/3.2 mm, L 2.9/3.8/6.1 mm |
| Depth bursts (depth + joints) | OKVIS2-X `tools/burst_pointcloud/grab.py` | live on several robots |

How to read it: left, the capability; middle, where its code is; right, how far it is proven. The two rows in bold
are the core of a tag-free method and must be ported here first (milestone H0).

**Lessons that shape this plan:**
- **Forearm-only views fail.** A forearm is nearly a cylinder; its mesh match can rotate 29-49° about its axis
  (phantom14, 3 Nov: held-out 5/13/41 mm). The poses that worked showed **the elbow and upper arm**, varied **wrist
  roll**, with the **hand open**.
- **Rotation needs the right views.** On phantom12, rotation had no tag-free source from the views tried (T3, R1, the
  torso CAD "route 2"); phantom14's URDF route did fit shoulder-roll rotations with the wave-2 poses. Which URDF
  parameters are observable must be decided from the information of the actual poses (section 4).
- **Cables.** phantom12's right arm lost 7 EtherCAT slaves mid-sweep on a move with 55° of shoulder yaw and 80° of
  wrist roll in 6 s at kp 100 (23 Oct). Cap per-move travel of those joints; read the motor diagnostics before every
  move; a drop stops everything.
- **Every arm joint in every command.** `dma_motor_motion` holds an unlisted joint at its *measured* (sagged) pose, so
  partial commands make a sagging arm sink. List all seven arm joints (and the neck).
- **The other arm** must be parked out of the route (the left route passes the right hand at the rest pose).
- **kp only**, never kd.
- **A plate on the hand blocks read B.**
- **The head calibration comes first.** Read B uses head depth, so its scale error goes straight into the arm.

---

## 3. Inputs the session needs

- A **current head stereo calibration**, from the same session or deployed recently. Depth scale error at 0.6-1 m
  must be well under the 3 mm gate.
- `T_BS` (camera in the head) as deployed. The URDF route absorbs the head mount through the NeckYaw origin. Don't
  also fold an arm correction into `T_BS` (phantom14's note).
- The robot's live URDF, its MJCF (for clearance), a joint snapshot (spine and legs as they hang now).
- Motors on (the tool's Y), loco and policies off, robot on the gantry, **space around both arms clear**, the operator
  standing back during the wave.

---

## 4. The model: what is fitted

Per arm, from K captures (depth burst + joint angles), each matched by read B to the arm's links seen in the cloud:
- **Candidate parameters:**
  - the NeckYaw joint origin (translation; the head's place on the torso, shared by both arms);
  - per arm, the shoulder-roll joint rpy (phantom14's result);
  - optionally, small zero offsets of shoulder pitch, shoulder yaw and elbow;
  - optionally, a camera rotation term.
- **Fit:** minimise the read-B residual of every capture's matched links (point-to-surface, robust) over the
  parameters, with a prior that keeps them near the URDF.
- **Choosing the parameter set:** an **identifiability check** on the planned poses (Fisher information of each
  parameter; condition number). Parameters the poses can't separate are fixed at the URDF. M0-M2-style model
  selection on held-out captures, as in `armfit.py`. A reviewer should challenge this choice first.
- **Held out:** 3 captures per arm not used in the fit, plus 1 repeat (noise floor). The gate is the hand's position
  error at those captures.
- **Paired against the live URDF:** the same held-out captures scored with the live URDF. Deploy only if the new one
  is better (as the head calibration decides).

---

## 5. The session (per arm)

1. **Park the other arm** (`selfcal_park.py` logic: a cleared hang pose, no wrist twist).
2. **Open the hand:** `inspire_hand_poses --hand <0 left | 1 right> --pose open --no-loop` (on the robot; checked
   10 Nov on phantom10: poses `open`, `relax`, `fist`, …). Verify that it opened from the hand's state (the readback
   source is to be found in H2), and keep it open for the wave. An open hand's fingers give read B a distinctive
   shape; a fist ends the arm like a cylinder.
3. **Neck to the view band:** pitch about 18-28°, the band where the arm workspace is in view on phantom12/14.
4. **The wave: stop-and-go over K poses.** It is not a continuous motion: stereo depth needs the arm still, and a
   moving arm blurs and lags. For each pose:
   1. Move. Every arm joint is listed, kp as decided. The ramp scales with the largest joint step, about 30-40°/s with
      a minimum of about 1.5 s. Per-move caps on shoulder yaw and wrist roll come from the cable lesson.
   2. Settle: wait until the measured joints are still (as the neck does).
   3. Burst: a few depth frames plus joints.
   4. Go on.
5. **Live quality:** each capture gets read B right away in background workers, and the screen shows per capture:
   - how much of hand, forearm, elbow and upper arm was seen;
   - the match residual;
   - the running fit;
   - "parameters well determined: yes/no".

   A bad capture is retaken once or skipped, never silently used.
6. **Home**, hand back to `relax`, and the next arm.

**Pose planning, per robot, automatically:**
- **Source:** the robot's snapshot (spine and legs as they hang), the live URDF, the MJCF and the live head config.
- **View:** enough of the arm's surface visible in both cameras, with the elbow and upper arm in view, at 0.4-1.2 m.
- **Information:** greedy selection maximising the log-det of the parameter Fisher matrix. This forces wrist-roll and
  elbow variety (`armplan.py`'s idea, with the URDF parameters).
- **Safety:** MuJoCo clearance along every leg against the body as it hangs now, joint limits and gravity sag
  (`armsafety.py`).
- **Order:** the shortest joint-space tour under the per-move caps, with held-out poses interleaved.
- **Pose count:** about 12-16 per arm, against phantom14's 18-19. Fewer if the information saturates. Measure, don't
  assume.

**Time budget (estimate; measure in H3):**
- per pose: about 2.5 s move, 1 s settle, 0.5 s burst ≈ 4 s;
- 14 poses ≈ 1 min per arm, plus park, hand and neck about 30 s;
- **about 3-4 min for both arms**, the fit done about 30 s after the last pose.

---

## 6. Deploying the URDF ("auto update URDF on the robot")

- **The file:** `<robot>_mk1_handeye_<date>.urdf`. Template = the URDF the robot uses now; only the fitted origins
  change; the diff is shown on screen and in the report.
- **Who reads it — inventory first (H4).** Known or likely:
  - the burst point cloud (`BURST_URDF`), as on phantom14;
  - the cuRobo planner (Mk2-locomotion `feat/planner`). Its arm model lacks the shoulder-roll terms: it needs its own
    generator, agreed with its owner;
  - box-pick;
  - possibly OKVIS (neck FK / self-mask).

  For each: where it reads the URDF (env var, file, config), how it reloads, who owns it.
- **One transaction,** like the head calibration's deploy:
  1. a snapshot of every switch point;
  2. write the file;
  3. switch the consumers;
  4. restart or reload them;
  5. verify that each one loaded the new file (sha in its log, or a read-back);
  6. `rollback.sh`;
  7. **R** rolls back.
- **Post-check:** two fresh poses per arm after the deploy. The hand error with the new URDF, through the consumers'
  own path where possible, must be within the gate; otherwise roll back.

---

## 7. Integration in `calib`

- **Menu:** *hand-eye calibration* as a session type, and an offer at the end of a head calibration ("Hand-eye next?
  Y").
- **Reuse:** motors (enable / takeover / disable), the motion gate, HOLD / abort / deadman, countdown and voice,
  progress bar and quality panel, the deploy transaction and rollback, the report, continue / discard.
- **New modules (proposal):**
  - `armmotion.py`, like `neck.py`: all joints listed, caps, arrival and stillness, motor-diagnostics check per move;
  - `hands.py`: open, relax, verify;
  - `armplan2.py`: the per-robot planner;
  - `readb.py`: the port of `blind_b.py`;
  - `urdffit.py`: the model, identifiability, held-out;
  - `urdfdeploy.py`: consumers, transaction, post-check.
- **Permissions:** the operator's Y before any arm motion and before deploy. Hand commands are new for the tool:
  **ask the user** (section 9).

---

## 8. Milestones (each gets its own TDD plan and the user's go)

| # | Milestone | Done when |
|---|---|---|
| H0 | Port read B and the URDF-route fit into this repo; stored data (phantom14 wave-2 + left bursts on Moonstone; phantom12 sessions as offline yardstick only) | replay tests reproduce phantom14's 4 Nov URDF and its held-out numbers within tolerance |
| H1 | Per-robot pose planner (view + information + MuJoCo + tour) | on phantom14's stored data, fewer poses reach the same held-out error; plans for phantom10/11 pass clearance |
| H2 | `armmotion.py` + `hands.py` with fakes; a safety review by a separate reviewer before any robot use | review says safe; dry run records every command |
| H3 | Live capture on one arm of one robot (phantom10 proposed), operator present | captures land with quality on screen; time per pose measured |
| H4 | URDF deploy: consumer inventory, transaction, post-check | deploy and rollback verified on the robot |
| H5 | Both arms, speed tuning, offered after a head calibration | full hand-eye run ≤ ~5 min of motion; the gate held |

How to read it: in order; each row's right column is what must be true before the next row starts.

---

## 9. Questions for the user (before H0)

1. **Robots and arms:** both arms on every robot; first live robot phantom10?
2. **Hands:** may the tool open and relax the hands itself (`inspire_hand_poses … --pose open`)? It's a new kind of
   command for it.
3. **Who must use the new URDF:** burst point cloud, cuRobo, box-pick, OKVIS, others? And who owns each (cuRobo needs
   its own model update)?
4. **The gate:** ≤ 3 mm at the hand on held-out poses, as on phantom14?
5. **Arm motion limits:** kp for arm moves (100 was used), per-move caps for shoulder yaw and wrist roll, minimum ramp.
   Past rule "ramps ≥ 6 s" made it slow; the proposal is travel-scaled ramps with caps.
6. **Space and people:** gantry, a clear radius around the arms, the operator behind a line during the wave?
7. **phantom12's plate data** as an offline yardstick only (allowed on 24 Oct): still fine?
