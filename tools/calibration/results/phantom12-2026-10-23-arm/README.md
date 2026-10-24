# phantom12 camera-to-arm, 23 Oct 2026 (arm session, sweep 1023c): partial, reviewed

The run that produced these files: plan `docs/superpowers/plans/2026-10-22-phantom12-arm-session.md` (stereo-depth-eval
branch), Results of 23 Oct. Sweep 1023c stopped at stop 17 (a cable fault on the right-arm EtherCAT segment). 12 holds
were kept: 8 fit, 1 held out, 2 same-side repeats, 1 opposite-side repeat. The plate was a 2x2 AprilGrid (45 mm tags,
pitch not measured) taped across the back of the hand and the wrist. Wrist roll (15.42°) and gripper yaw (−56.25°)
were frozen for the sweep.

**The result (independently reviewed 23 Oct; the review's corrections are folded in):**

| | camera-to-torso correction dT | error at unseen poses (leave-one-out over the 8 fit holds) |
|---|---|---|
| M0: URDF chain + the live hand-eye, plate place only | none | ~14-15 mm RMS (worst ~23 mm) |
| **M1: + dT (6 dof)** | **~27.5-28.4 mm, 0.27-0.32°; (x, y, z) ≈ (−25, +6, −9) mm in the torso** | **~2.6-3.5 mm RMS, worst ~6 mm** |
| M2: + shoulder roll/yaw and elbow offsets | ~27-29 mm | worse (overfits 8 holds) |

- dT is stable when any one fit hold is dropped (x −25.5…−26.9, y 4.6…6.5, z −7.0…−9.8 mm). It is not a tag-size or
  camera-index artefact.
- The MJCF's shoulder and neck origins explain x and z to within 1.5 mm, but not y: the URDF's NeckPitch y = 0.01 m
  looks right and the MJCF's 0 wrong.
- **Formally VOID under the plan's gates:** the plate-slip check reads 0.43° (still frames: 0.35°) against 0.2°.
  That may reflect M1's own 3-5 mm misfit rather than real slip.
- The board-scale column (+0.3 … +10 %) is NOT plate bending: bending shortens distances, and every hold reads
  longer. It tracks corner motion during a hold (0.3-1.3 % on still holds), so it points to a stereo-disparity
  error on a moving hand. The cause is open.
- One held-out pose only: the leave-one-out numbers above replace the single "held-out" value in the report.

**Use:** for grasping with phantom12's right arm today, apply dT on the camera side:
`T_torso_cam(true) = dT · T_torso_cam(URDF chain, live T_BS)`. dT (4x4) from the still-frames M1 fit:

    [[ 0.999995,  0.002911, -0.001167, -0.025205],
     [-0.002916,  0.999985, -0.004696,  0.006660],
     [ 0.001153,  0.004699,  0.999988, -0.008880],
     [ 0,         0,         0,         1       ]]

Expect ~3 mm RMS (worst ~6 mm) at the plate over this workspace, at neck pitch ~21°. Validity at other neck angles is
untested.

**To make it trustworthy:**
- a rigid plate fixed to the hand (gripper link) only, not across the wrist;
- captures gated on corner stillness (< 0.1 px), with ≥ 4 held-out poses;
- the right-arm cable repaired so the sweep can finish.

Files:
- `report-1023c.json`: step 92 on all hold frames;
- `still_refit.py`: the still-frame refit with leave-one-out;
- `poses-1023c.json`: the plan (step 90);
- `arm-session-1023c.jsonl`: the step-91 log.

## Update, 23 Oct ~17:30: sweep 1023d (head pitch 28.4°) and both sweeps together (`joint_fit.py`, `joint_fit.out`)

1023d stopped at stop 13 (the fifth cable drop). 6 fit holds plus 2 repeats were kept.

| fit | dT (x, y, z) mm, angle | leave-one-out RMS / max |
|---|---|---|
| 1023c alone (pitch 24°) | (−26.5, +5.3, −8.7), 0.27° | 2.6 / 5.7 mm |
| 1023d alone (pitch 28°) | (−25.4, +7.6, −6.2), 0.36° | 1.9 / 3.0 mm |
| both, one dT | (−27.7, +3.9, −7.1), 0.41° | 5.0 / 11.7 mm |
| **URDF with the MJCF shoulder/neck origins (NeckPitch y kept 0.01), no dT** | none | M0 3.2 px (URDF as is: 13.7 px) |
| the same, with dT | (−1…−3, +1…+4, +1…+3), 0.3-0.4° | as above |

- **The URDF's shoulder and neck origins carry ~90 % of the error.** With the MJCF's values (keeping NeckPitch
  y = 0.01) the remaining correction is ≤ ~5 mm / 0.4°.
- A residual ~0.4° differs between head pitch 24° and 28°, so one fixed dT does not cover both head angles to
  3 mm. Pinning that down (a neck-pitch-dependent term) needs more head angles: a working arm.
- **Recommended:** fix the URDF's RightShoulderPitch / NeckYaw / NeckPitch origins to
  `-0.015 -0.1034 0.2823` / `-0.015 0 0.3815` / `0.025 0.01 0.1927`. Then, at a given head angle, apply that
  session's small dT for ~2-3 mm.
