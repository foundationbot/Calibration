# Hand-eye calibration in `calib` — plan rev 1 (11 Nov 2026), for the user's answers

Status: **proposal, not started.** Rev 0 was reviewed by two independent reads (`2026-11-11-handeye-plan-review.md`);
rev 1 folds the findings in. It needs the user's answers to section 9. The first step is now an offline go/no-go on
stored data (H-1, no robot). No robot motion comes from this document.

"Hand-eye" here means **camera-to-arm**: after it, a point the head cameras see and the arm's pose computed from the
joint angles agree, so a grasp planned in the camera's view lands where the hand goes.

**What changed from rev 0, in short:**
- The 3 mm gate is withdrawn until H-1 measures what the method can reach (phantom14's own data: about 4.5-5 mm
  per capture).
- The fit is the two-stage method that actually ran on phantom14, ported as it ran.
- Each arm pose is captured at several neck angles (that is what made phantom14 work).
- Times come from measurements: about 10-12 min of motion for both arms (estimate), not 3-4 min.
- Motion keeps the proven slow ramps, and pins every joint it doesn't move.
- No hand commands in v1.
- OKVIS is a definite URDF consumer.

---

## 1. Goal

For one robot, both arms, run from the same `calib` tool:
- **Tag-free.** No plate, AprilTag or fixture on the robot (the user's rule since 24 Oct: robots get swapped; per-robot
  fixtures don't scale).
- **Output:** a per-robot URDF whose corrections make camera and arm agree at the **wrist point** (the WristRoll
  origin, which is what read B measures).
- **Deploy rule:** deployed automatically, with a rollback, only when it beats the live URDF on held-out arm poses (the
  paired rule of the head calibration).
- **The absolute bar is set in H-1** from the measured noise floor. On phantom14's data it will be near 5 mm, not 3 mm
  (review finding 1). The user decides whether that is enough for grasping (section 9).
- **The same operator experience** as the head calibration: voice, screen, the operator's Y before motion,
  countdowns, HOLD / abort, progress and quality on screen, report, discard / continue.
- **Speed:** as fast as the measured costs allow (section 5). Past runs: 16 min for one arm (phantom12), plus hours of
  hand work around it.

---

## 2. What exists

| What | Where | State |
|---|---|---|
| URDF forward kinematics of the arms and neck | `tools/calibration/scripts/armchain.py` (right arm only), `neckfk.py` | tested |
| Plate fit models (M0/M1/M2) | `scripts/armfit.py`, `armfit_multi.py`, step 92 | tested; plate route not the target |
| Pose list from a view and information model | `scripts/armplan.py`, step 90 | plate-based; its information logic carries over |
| MuJoCo clearance and gravity sag | `scripts/armsafety.py` | tests **skipped in every automated run** (no mujoco in `requirements-dev.txt` or CI; no MJCF in this repo) |
| Arm-wave runner, park the other arm | `scripts/selfcal.py`, `selfcal_run.py`, `selfcal_park.py` | ran live on (old) phantom12, 25-26 Oct; lists only the 7 arm joints + neck |
| Read B matcher (B0: per capture, the arm mesh matched in head depth) | OKVIS2-X `tools/burst_pointcloud/meshfit.py`, `hand_check.py` (3 links: forearm, wrist yaw, wrist roll), `frames.py`, `armfk.py`, `mesh.py`, `burst_io.py` + `_burstfusion` (pybind11 C++, OpenMP); driver `experiments/arm816/blind_b.py` (c5b4c6b0) | **not in this repo**; about 80 s per capture on one Moonstone core |
| Arm meshes | Moonstone `~/burst-pc/meshes/cycloidal_arm/` | **in no repo** |
| `BURST_URDF` (the matcher reads a chosen URDF) | uncommitted patch to `frames.py` in Moonstone `/data/kaustabh/p21arm/code-repo` (unversioned copy) | **in no repo** |
| URDF-route fit (phantom14, 4 Nov) | Moonstone `/data/kaustabh/p14arm/urdfx/` (`twoarm.py`, `predict.py`, `fitrun.py`, `urdfvar.py`), drivers `ab14w2.py`, `abL.py`, `evalX.py` | **in no repo**; result in section 2.1 |
| phantom21 arm wave (2 Nov) | OKVIS2-X `feat/calibration-agent-p21-armwave` (ddeaba11); bursts Moonstone `p21arm/bursts-1102a/` | **run and failed** (section 2.1) |
| Depth bursts (depth + joints) | OKVIS2-X `tools/burst_pointcloud/grab.py` | live on several robots; 7.7 s per burst at 1280 |

How to read it: left, the capability; middle, where its code is; right, how far it is proven. Bold marks gaps that
H0 must close.

### 2.1 What the three robots showed

| Robot | Date | Views | Result |
|---|---|---|---|
| (old) phantom12 | 25-26 Oct | right arm, neck 28° only, translation only | 1.5-1.8 mm against the plate fit B1 (itself 1.75 mm RMS); rotation had no tag-free source |
| phantom14 wave 1 | 3 Nov | forearm only | match rotation 29-49° wrong; held-out 12.8 / 5.0 / 41.3 mm: **failed** |
| phantom14 wave 2 | 4 Nov | 5 arm poses x 3-4 neck angles, elbow in view, wrist roll varied | URDF fit M12: 34-36 mm (R) and 21-22 mm (L) before; after, about 4.5-5 mm median per capture (several 7-11 mm). Held-out captures shared arm poses with the fit and served for model choice. |
| phantom21 | 2 Nov | right arm, neck fixed at 22.7° | 5 of 13 captures excluded (coverage); held-out 12.5 / 3.3 / 12.6 mm: **failed** |

How to read it:
- one row per arm wave so far, oldest first;
- "(old) phantom12" is the robot called that before the 9 Nov renames;
- numbers are the wrist-point disagreement between camera and arm.

**Lessons that shape this plan:**
- **Views.** Forearm-only views fail: a forearm is nearly a cylinder, and its match can spin about its axis. Poses
  must show the elbow and vary wrist roll.
  - Even then, wave 2 had 6 of 18 right captures matched with 14-39° rotation. The rotation and coverage gates
    (≤ 12°, ≥ 0.6) are part of the method.
- **Neck angles.** Vary them at every arm pose. The head-mount correction (NeckYaw origin) is measurable only that
  way; phantom21's fixed neck failed.
- **What read B matches.** Forearm and wrist links only: no hand, no upper arm. The hand's state doesn't help the
  match.
- **Parameters.** Few, chosen in advance. Joint zero offsets beyond shoulder roll ran away before (B2: shoulder pitch
  to 1e4 rad; wrist yaw 14°, camera 6.75° on phantom14).
- **Cables.** phantom12's right arm lost 7 EtherCAT slaves on a move of 55° shoulder yaw and 80° wrist roll in 6 s
  (about 9-13°/s) at kp 100. Don't go faster; cap the travel per move.
- **Pin every joint the move doesn't drive.** List the other arm, spine and legs too, at their measured angles, as
  `calibtool/neck.py` does. The 2.1.1 master leaves unlisted joints at zero stiffness; a tool re-hold made
  phantom14's body creep.
- **Sag.** At kp 100 the arm sags: shoulder pitch −6 to −12°, roll −10°, elbow −3 to −5.5°. Plan views and clearance on
  the sagged pose; arrival has a tolerance.
- **The other arm** must be parked clear, in both orders.
- **kp only**, never kd.
- **The head calibration comes first.** Read B uses head depth; 0.21 % depth error is 1.7 mm at 0.8 m.
- **T_BS and the URDF go together.** The URDF is fitted with a given T_BS. A later head calibration that changes T_BS
  makes the hand-eye stale.

---

## 3. Inputs the session needs

- **A current head stereo calibration** (same session or recent), and its T_BS as deployed. The URDF route absorbs the
  head mount in the NeckYaw origin; never also fold an arm correction into T_BS.
- **Per consumer:** the URDF it reads now (they differ per robot and per consumer, e.g. phantom21's is pre-#816).
- **The robot's state:** its MJCF for clearance, and a joint snapshot (spine and legs as they hang).
- **At the robot:** motors on (the tool's Y), loco and policies off, the robot on the gantry, space around both arms
  clear, the operator outside the arms' fall zone during the wave.

---

## 4. The model: what is fitted

**Stage 1, per capture (read B, B0, as ported).**
- The forearm and wrist meshes are matched in the head depth. The result is the wrist offset, the coverage and the
  rotation of the match.
- Clean = rotation ≤ 12° and coverage ≥ 0.6, plus phantom14's distance-to-median outlier cut (≤ 15 mm).
- Unclean captures are shown on screen and never used silently.

**Stage 2, both arms jointly.**
- A robust least squares of the clean wrist offsets over a small parameter set.
- **Default = M12+head** (H-1, 11 Nov: best on unseen arm poses, held-out median 3.6 / 5.3 mm, p90 7.1 mm): NeckYaw
  origin xyz (shared), a ShoulderRoll zero offset per arm, and the camera's position in the head (8 parameters).
  phantom14's deployed M12 (5 parameters, without the head term) is the fallback when the neck angles can't separate
  the head term (correlation, standard errors). Results: `tools/calibration/results/handeye-h1-2026-11-11/`.
- **Since 13 Nov the first candidate is M12+headR**: the camera's rotation in the head instead of its position. It won
  on both phantom10 days; phantom14's data still choose M12+head. M12+headRT (both terms) overfits and is opt-in.
  See `tools/calibration/armcal/README.md`, "The models and the camera term".
- **The candidate list is fixed before any data are seen.** Proposal:
  - M12;
  - M9 (one shoulder roll);
  - X1 (a head translation only).
  - No other joint zero offsets (they ran away before).
- **The model is chosen by cross-validation inside the fit data**, leaving one arm pose out at a time; never on the
  held-out set.
- **Identifiability.** The parameter covariance from the Jacobian (condition number, standard errors) is reported. A
  parameter whose standard error exceeds its value is fixed at the URDF and said so.

**Held out and gate.**
- **Held out:** whole arm poses (all their neck angles), named in the plan before the session, at least 2 per arm.
- **Paired:** the same held-out captures scored with the live URDF. Deploy only if the new one is better (paired, as
  the head calibration decides).
- **Absolute bar:** the median wrist error over held-out captures within the bar set in H-1.
- **Noise floor:** the per-capture scatter of B0 over repeated and neck-varied captures of the same arm pose. One
  repeat alone measures repeatability, not the matcher's pose-dependent bias.

**Not measured by this method:** a bias of read B itself (mesh against the real arm covers, depth scale). The only
independent yardstick is the (old) phantom12 plate data; H0 scores the ported matcher against it.

---

## 5. The session (per arm)

1. **Park the other arm** clear (checked in MuJoCo for both orders).
2. **Hands:** not commanded in v1 (read B doesn't use them). Their state is recorded.
3. **For each arm pose** (proposal: 5-6 per arm, chosen per robot):
   1. **Move the arm.**
      - Every joint is listed: this arm at its target, everything else at its measured angle.
      - kp as decided; ramp ≥ 6 s (the proven rule) until H3 measures faster moves as safe.
      - Per-move travel caps on shoulder yaw and wrist roll, below the 23 Oct values.
      - The motor diagnostics are checked before each move.
   2. **Arrival:** the joints are still and within a stated tolerance of the sagged target.
   3. **For each of 3-4 neck angles** (pitch within `neck.py`'s ±26° clamp, yaw about ±24°): a neck move (about
      1.3 s, the head tool's measurement), then a burst (7.7 s at 1280 measured; a shorter burst only if H0 shows the
      same accuracy).
4. **Quality on screen** per capture as soon as read B has run on it: coverage, match rotation, wrist offset, clean
   yes/no, the running fit and "parameters determined: yes/no".
   - A bad capture gets one retake, or it is skipped by the operator (C).
   - Live per-capture quality depends on where read B runs and how fast it is (section 9.3; H0 measures it).
5. **Home**, and the next arm.

**Pose planning, per robot, automatically:**
- **View:** forearm and wrist visible in both cameras with the elbow in view, at 0.4-1.2 m, at every planned neck
  angle, computed on the sagged pose.
- **Information:** greedy selection over arm poses times neck angles, maximising the log-det of the stage-2 Fisher
  matrix. Wrist roll and elbow variety come out of it.
- **Safety:** MuJoCo clearance along every move against the body as it hangs now; joint limits; sag (`armsafety.py`).
- **Order:** the shortest joint-space tour under the caps.

**Time (estimate from measured parts; to be measured in H3):**

| Part | Measured cost | Per arm (6 poses x 3.5 neck angles = 21 captures) |
|---|---|---|
| Arm move + settle | about 8 s at ramp 6 | 6 x 8 s ≈ 50 s |
| Neck move | about 1.3 s | 21 x 1.3 s ≈ 30 s |
| Burst at 1280 | 7.7 s | 21 x 7.7 s ≈ 160 s |
| Park, home, checks | not measured; guess 60 s | 60 s |
| **Total** | | **about 5 min per arm, 10-12 min for both** |

How to read it:
- each row is one cost, with the measured number it comes from;
- a shorter burst, if H0 proves it as accurate, would cut the total to about 2.5 min per arm;
- read B's compute (about 80 s per capture per core on Moonstone) comes on top unless it runs in parallel with the
  wave; H0 measures it on a Thor CPU.

---

## 6. Deploying the URDF

- **The file:** `<robot>_mk1_handeye_<date>.urdf`, written per consumer from the URDF that consumer reads now. Only
  the fitted origins change. It carries a provenance header:
  - the session;
  - the T_BS (config sha) it was fitted with;
  - the parameters and held-out results;
  - the base file's md5.

  It is validated by parsing it and comparing FK against the fit.
- **Consumers (inventory in H2, read-only on the robot):**

  | Consumer | How it reads the URDF | State |
  |---|---|---|
  | OKVIS | `OKVIS_URDF` / `dma:urdf` (`mk12_stereo_holevisor.yaml:306`; `okvis2x_app_dma_live.cpp:613`, TF tree and pelvis FK) | definite; a NeckYaw change moves its head-in-body pose |
  | Burst point cloud | `BURST_URDF` (phantom14) | the patch must be committed first |
  | cuRobo (Mk2-locomotion `feat/planner`, ghost-wbc-manip) | its own arm model, without the shoulder-roll terms | needs its own generator, its owner's agreement, and a permission rule for restarting it |
  | box-pick | to be found | to be found |

  How to read it: one row per program that uses the robot's geometry. Each needs a switch point, a reload path, an
  owner and a load check before the tool may deploy to it.
- **One transaction,** like the head calibration's deploy:
  1. snapshot every switch point;
  2. write the files;
  3. switch the consumers;
  4. reload them (OKVIS through the path the head tool already uses);
  5. verify that each one loaded the new file (sha in its log, or a read-back);
  6. `rollback.sh`; **R** rolls back.
- **Post-check:** fresh captures at two arm poses per arm, each at its neck angles, scored with the same paired
  statistic as the gate. No single-capture threshold: at about 5 mm of noise per capture it would roll back by chance.
- **Coupling with the head calibration:** the head tool's deploy flags "hand-eye stale" when it changes T_BS, and
  offers the hand-eye run.

---

## 7. Integration in `calib`

- **Menu:** *hand-eye calibration* as a session type, and an offer at the end of a head calibration.
- **Reuse:** motors (enable / takeover / disable), the motion gate, HOLD / abort / deadman, countdown and voice,
  progress bar and quality panel, the deploy transaction and rollback, the report, continue / discard.
- **New modules (proposal):**
  - `armmotion.py`, like `neck.py`: every joint listed, caps, arrival with tolerance on the sagged pose, a diagnostics
    check per move, behaviour on a fault;
  - `armplan2.py`: the per-robot planner over arm poses x neck angles;
  - `readb.py`: the port of meshfit and hand_check, with `_burstfusion`;
  - `urdffit.py`: stage 2, cross-validation, identifiability, held-out;
  - `urdfdeploy.py`: consumers, transaction, post-check.
- **Permissions:** the operator's Y before any arm motion and before deploy.
- **On a fault** (EtherCAT drop, master stop, a fault latch) with the arm raised, the arm falls. So:
  - poses stay over clear space;
  - the operator is outside the fall zone;
  - the tool stops, holds what still holds, says what happened, and waits for the operator (no automatic retry).

---

## 8. Milestones (each gets its own TDD plan and the user's go)

| # | Milestone | Where | Done when |
|---|---|---|---|
| H-1 | **DONE 11 Nov: GO** (`results/handeye-h1-2026-11-11/`). **Offline go/no-go** on stored data with the existing Moonstone scripts: leave-one-arm-pose-out cross-validation on phantom14 wave 2 (both arms); the noise floor from B0 scatter; M12's standard errors and condition number; phantom21 `1102a` as a failure case | Moonstone only | a written verdict: what error the method reaches on unseen arm poses, which parameters are shown (standard errors), the proposed bar. **Stop here if unseen-pose error is not clearly below the live URDF's.** |
| H0 | **DONE 11 Nov; robust matcher merged** (`results/handeye-readb2-2026-11-11/`: even mesh, two passes + settled check, ~3 s/capture on one Moonstone core, M12+head CV 2.8 / 4.5 mm median, p90 5.8 / 6.3 mm; 5-frame bursts kept for now) (`tools/calibration/armcal/`, `results/handeye-h0-2026-11-11/`): exact reproduction of all 48 phantom14 captures and H-1's fits; read B 6-7x faster (13 s/capture on one Moonstone core), numpy fusion replaces `_burstfusion`; a 2-frame burst calibrates as well on phantom14. Open: the plate yardstick, mujoco in CI, Thor timing, where the camera-in-head term is deployed, B0's fragility. Original row: Port read B (meshfit, hand_check, `_burstfusion`, meshes, `BURST_URDF`) and the stage-2 fit into this repo (subject to section 9.2), with mujoco + MJCF in the test venv and CI | Moonstone, CI | replay reproduces phantom14's M12 URDF and its per-capture numbers within tolerance; scored against the (old) phantom12 plate yardstick; B0 time per capture measured on a Thor CPU (needs the user's OK) |
| H1 | Per-robot planner (arm poses x neck angles, view on the sagged pose, information, MuJoCo, tour) | Moonstone | on phantom14's data it picks a subset that reaches H-1's error; plans for phantom10/11 pass clearance |
| H2 | `armmotion.py` with fakes; consumer inventory (read-only on robots); a safety review by a separate reviewer | Moonstone; robots read-only | review says safe; a dry run records every command; consumer table filled with owners |
| H3 | Live capture on one arm of one robot (phantom10 proposed), operator present | robot | captures land with quality on screen; time per pose measured; ramp speed decided from it |
| H4 | **Built offline 11 Nov** (`calibtool/handeye_deploy.py`, consumers in `calibtool/config/handeye_consumers.yaml`; live check on the robot still to do). URDF deploy: transaction, post-check, the head tool's "stale" flag | robot | deploy and rollback verified on the robot for each agreed consumer |
| H5 | Both arms, speed tuning, offered after a head calibration | robot | full run within the measured budget; the gate held |

How to read it: in order; each row's right column must be true before the next row starts. H-1 is the only one that
can end the project early, and it needs no robot.

---

## 9. Questions for the user (before H-1)

1. **Robots and arms:** both arms on every robot; first live robot phantom10?
2. **Public repository:** porting read B means putting the arm meshes (robot CAD) and the `_burstfusion` C++ code from
   OKVIS2-X into the public Calibration repo. Is that allowed? If not, they stay in OKVIS2-X, and the kit fetches them
   at install time from a private source.
3. **Where read B runs:**
   - on Thor (images stay on the robot, as D10; needs an aarch64 build of `_burstfusion`; runs beside OKVIS);
   - or on Moonstone (about 55 MB per burst copied off the robot; an exception to D10)?

   H0 measures Thor's speed.
4. **The gate:** phantom14's data point to about 5 mm at the wrist per capture, not 3 mm. What does the grasp need?
   Proposal: deploy when paired-better than the live URDF and the held-out median is within the floor H-1 measures.
5. **Hands:** v1 doesn't command the hands (read B doesn't model them). Agreed?
6. **Consumers:** OKVIS (`OKVIS_URDF`) is definite. Is it in scope for v1, or does v1 deploy to the burst point cloud
   only? Who owns cuRobo's model update, and may the tool restart it?
7. **Arm motion limits:** kp 100; ramp ≥ 6 s until H3 measures faster moves; travel caps on shoulder yaw and wrist
   roll below the 23 Oct values. Agreed?
8. **Space and people:** gantry, a clear radius around both arms, the operator outside the fall zone during the wave?
9. **The (old) phantom12 plate data** as an offline yardstick only (allowed on 24 Oct): still fine?
10. **Go for H-1?** It is offline, Moonstone only (light, niced, 1 BLAS thread per job), no robot, about an hour of
    compute and writing (an estimate).

### 9.1 The user's answers (11 Nov 2026)

| # | Answer | What it changes |
|---|---|---|
| 1 | Both arms on every robot; phantom10 first | as proposed |
| 2 | Porting the meshes and `_burstfusion` into the public repo is allowed | H0 ports them here |
| 3 | Read B runs **on the robot only** | `_burstfusion` needs an aarch64 build in the kit; H0 measures B0 on a Thor CPU; the time budget must include it |
| 4 | As accurate as possible within a reasonable time; **aim for < 6 mm** | the absolute bar proposal is 6 mm at the wrist (held-out median), refined by H-1; the paired better-than-live rule stays |
| 5 | Not against hand commands; asked whether they help | open: answered in chat (one fixed hand pose per arm to repeat wave 2's open-hand conditions; a hand mesh is later work) |
| 6 | Point-cloud tools, and OKVIS too | v1 consumers: burst point cloud (`BURST_URDF`) and OKVIS (`OKVIS_URDF`); cuRobo and box-pick later |
| 7 | OK; **don't overdo kp: some joints start oscillating** | kp ≤ 100 for arm moves (lower where a joint oscillates); `armmotion.py` watches tracking error for oscillation and holds; H2's review checks it |
| 8 | Yes; the arms are clear now | as proposed |
| 9 | Probably fine | the (old) phantom12 plate data stays an offline yardstick |
| 10 | Explain H-1 first; **do not start yet** | H-1 waits for an explicit go |
| — | **The head may move during the wave** (11 Nov, later) | decided: 3-4 neck angles per arm pose; H-1 drops the fixed-neck subset check |
