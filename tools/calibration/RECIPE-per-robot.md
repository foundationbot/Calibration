# Per-robot head calibration: the recipe for a replacement robot (no tags on the robot)

For whoever sets up a new or repaired robot's head cameras for depth and grasping. Three steps, each with a pass/fail
line. Nothing is taped to the robot: step 1 uses a free-standing printed board, step 3 uses the arm itself.
Plan of record: `docs/superpowers/plans/2026-10-25-tagfree-selfcal.md` (stereo-depth-eval branch).

| step | what it gives | robot time | compute | measured on |
|---|---|---|---|---|
| 1. Board calibration | the two cameras' lenses and their relative pose (depth) | ~13 min | ~51 min on Thor | phantom12, Fit C, 22 Oct |
| 2. Drift check | that the cameras have not moved since step 1 | ~1 min (30 s take) | 15 s (Moonstone), 44 s (Thor) | phantom12, 24 Oct |
| 3. Arm-wave self-calibration | where the camera sits relative to the arm (the grasp correction, translation) | 6 min 21 s | 9 min 26 s (Moonstone) | phantom12 sc1025d-right, 25 Oct: 16 min 10 s end to end |

Total robot time is about 20 min; total wall time is over an hour, because step 1's fit is ~51 min; step 3 alone is ~16 min.

## Before anything
- Motors 29/29 enabled (`dma_cmd_client state`), the walker idle (`/desired` not advancing).
- Thor's root disk: free space above kubelet's eviction line + 5 GB (24 Oct: OKVIS was evicted on disk pressure).
- OKVIS running, no `[cam-stall]` in its log (else: dma-video producer scale 0 → 1, RUNBOOK §3).
- Robot writes (recordings, moves, config) only with the operator's yes.

## 1. Board calibration (KB4T lenses + stereo pose)
Follow `calibration.md` §11 ("New robot") with the rig file for the robot, the board measured with a steel rule, and
the neck-grid takes of the Fit C plan (`2026-10-21-phantom12-fit-c.md`).
- **PASS:** step 66 (`66_validate_lens.py`) on the test sweep says PASS: every image cell × distance entry with ≥ 8
  holds within ±0.5 % depth (corner cells ±1.0 %), and scale against distance flat within ±1 % from 0.4 to 2 m.
- **FAIL / NO RESULT:** do not deploy; the report names the cause (record more coverage, or refit).
- Deploy the config (`71_export_lens.py --okvis-kb4t` into the robot's OKVIS config), keep the previous file as the
  rollback, and check the OKVIS log's `KB4T rectification (stereo network)` line at startup.

## 2. Drift check (step 67)
Board on a stand ~0.7-1.0 m away, centred within ~150 × 100 px of the image centre, still; record 30 s; run
`67_drift_check.py` (`calibration.md` §5b).
- **PASS (exit 0, OK):** camera 1 roll within 0.10°, pitch within 0.03°, board depth error within 1 % at 0.4-1.3 m.
- **FAIL (exit 1, DRIFTED):** do not trust depth; redo step 1. **NO RESULT (exit 2):** move the board, record again.
- Run it after any handling of the head, after power-up, and before step 3.

## 3. Arm-wave self-calibration (no tags)
**Status (25 Oct 2026, 20:45): translation validated on phantom12's right arm, blind, plate off; rotation not.**
Session sc1025d-right (one command, 16 min 10 s end to end, measured): the tag-free fit (the point-cloud agent's
trans-only read B) put the hand within 1.5-1.8 mm of the plate calibration at three unseen poses, where the
uncorrected chain is 3.2-3.4 mm off (reviewed). **Head angles (reviewed):** the correction from a 28° session
holds at 23° and 18° within ~2 mm (1.7-2.1 mm vs 3.2-3.4 mm uncorrected). **Left arm:** its own correction differs from
the right's by 3.4 mm, so the right arm's does not serve the left hand; the left one is a candidate without truth.
Not shown: rotation (not fitted; the IMU-gravity check failed its flatness test, 0.17° p-p). Use the correction for
translation only, per arm; the rotation comes from step 1's camera side.

The arm visits ≥ 9 poses in view of the head cameras; a depth burst at each; the point-cloud fit (read B) matches the
arm's mesh to the depth and returns the camera correction.
1. Route: `left_plan.py` / `right_verify.py` (poses in view) → `left_clear.py` / `right_clear.py` (MuJoCo path
   clearance ≥ 2.5 cm against the body at the robot's current pose, `armsnap.py`).
2. Pre-checks and runner: `selfcal.py` (motors 29/29 before every move; abort on a motor drop, failed move or failed
   burst; held-out captures named before data; no recording, bursts only).
3. Fit and report: read B on the bursts (the point-cloud agent's `blind_b.py --trans-only --min-coverage 0.70`; a
   capture enters the fit only if it is in view and its fit covers ≥ 70 % of the arm).
   One command drives 1-3: `selfcal_run.py check|plan|go <session>` (plan clears the route at the robot's pose).
- **PASS:** every capture's fit converged; the held-out captures, predicted from the others, agree within 3 mm at the
  wrist and 0.3° (the yardstick's gate).
- **FAIL:** keep the URDF chain without a correction (phantom12: 3.9 mm RMS at unseen poses) and report it.
- **Open: rotation.** The candidate fit (the point-cloud agent's "trans-only" read B, 25 Oct) corrects translation only
  and keeps the chain's rotation; on phantom12 the chain is 0.19° from the plate, inside the 0.3° gate, but a robot
  whose head mount is rotated more would pass unseen. Read B's free rotation cannot stand in: it scatters ~0.9°
  between sessions (rv1024 0.98°, rl1024 1.19°, 0.92° apart; the point-cloud agent, 25 Oct). The rotation has to come
  from the camera side: the board calibration's camera-IMU step (T_BS, `calibration.md` §6) and the neck URDF. An
  tag-free check of that was tried on 25 Oct (gravity from the NeckPitch IMU against the Torso IMU through the neck
  chain, 65 bursts): NOT usable as is — residual 0.17° p-p across neck bins (needed ≤ 0.05°), tilt 0.34° from B1's.
  Two more tag-free sources failed the same night (all reviewed, rules fixed before data): neck-motion hand-eye (R1)
  cannot see a torso-side rotation (it is absorbed by the board's pose), and fitting the robot's own torso CAD in the
  head depth (route 2, `results/phantom12-2026-10-26-route2/`) is flat about torso x — the axis of phantom12's 0.19° —
  because the chest is only a grazing band at the image bottom (3.05° off B1 at neck 28°; out of view at 18°).
  **So today no tag-free rotation source exists:** the recipe keeps the chain's rotation.

## What phantom12's numbers say (for comparison)
- Plate-measured grasp correction, right arm, #816 URDF: see `results/phantom12-2026-10-25-dt/` (Phase B).
- URDF chain alone: 3.9 mm RMS at unseen poses (23 Oct); with the 23 Oct plate dT: 2.6 mm (1023c + 1023d).
