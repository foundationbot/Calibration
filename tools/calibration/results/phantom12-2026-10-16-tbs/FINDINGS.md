# phantom12 T_BS neck takes, 16 Oct 2026 night: findings

Session `okvis2-x-96`. Takes on Thor in `/root/calib-work/takes/tbs-neck-{A,A2,A3}` (record app, 300 s
each, exposure 1500 µs / ISO 1600, neck unpowered and moved by hand). Reports and logs here; analysis
scripts in `analysis/` (run from this checkout; their extracted data, ~6 MB a take, and the intermediate
reruns are in `~/calib/phantom12/results/tbs-2026-10-16/`).

## The takes
| take | what happened | step 75 | step 81 (hand-eye, URDF neck) | step 82 (gyro) |
|---|---|---|---|---|
| A | right camera died at ~20 s (XLink); the whole robot turned ~20° about vertical (torso + pelvis IMUs; camera cross-check corr −0.998, slope −1.07) | RETAKE | spread 160 mm / 7°: unusable | 1318 samples: unusable |
| A2 | robot locked (torso within ±0.7°); motion mostly continuous sweeps, 13 usable holds, yaw ±40° | RETAKE | X vs config 0.61° / 19 mm, spread 9 mm / 0.56° | R_BS 1.69° (few samples) |
| A3 | lock tightened (torso ±0.4°); 16 usable holds; good sweeps | RETAKE | X vs config 0.60° / 32 mm, spread 13 mm / 0.66° | PASS: R_BS 0.567° (gate 1°; retracted below: no error bar, takes scatter 0.6–2.3°); axes 88.79° (an artefact, below) |

A2 and A3 disagree on X by 1.2° / 21 mm (plan: 0.1° / 1 mm). Error bars are clustered by hold.

## What was ruled out
- **Torso motion** after locking: torso yaw ±0.4° in A3 and uncorrelated with the camera's residual
  (corr −0.23).
- **URDF neck geometry** (angle between the axes, axis offset): a fused fit of A2 + A3 with both free
  finds 89.5–91° ± 0.5–0.9° and leaves the residual unchanged (0.63° / 6.5 mm). Validated on synthetic
  data at the real poses, where it recovers 88.80° and the offset exactly. (The yaw axis's direction
  and position and both encoder zeros cannot be seen by a neck-only test.)
- **Lens edge**: dropping holds beyond ±30° yaw does not help (takes still 1.5–2° apart).

## What it is: the encoders do not say where the head is, by ~0.3–0.5° per hold
- Per hold, the head IMU's gravity direction (torso tilt corrected) disagrees with the encoders'
  prediction by 0.20° rms (A2) / 0.40° rms (A3).
- **The camera and the head IMU agree hold by hold** (A3: about Head x corr +0.92, slope 0.88; about
  Head y corr +0.76, slope 0.62; both ~0.3° rms). They are independent sensors; the only shared input
  is the encoder angles. So the encoders (or the joints' output vs their encoders) are off by
  ~0.3–0.5° per hold, varying with how the hand loads the head: holds at nearly the same pose differ
  by ~1° (A3 holds 4, 5, 12).
- No sign flip with the pitch approach side (A3: −0.20 vs −0.24°), so not a clean backlash; more like
  compliance or play loaded by the hand in varying directions.
- It also says the camera calibration (T_SC, intrinsics) and the head IMU are consistent with each
  other.

## Corrections after the reviews (same night)
- **Step 82's "angle between the axes 88.79°" is not evidence against the URDF.** Its per-sweep axis
  averages accept samples carrying up to 5 % of the other joint's rate; 2 % coupling alone gives
  88.85° from an exact 90° neck in simulation (Task 8 review). The camera fit with the angle free does
  not see it either.
- **Step 82's A3 "PASS 0.567°" is not a pass:** the three takes give 2.3°, 1.7° and 0.6°, and the
  report carries no error bar. Read it as R_BS within ~2° of the config. The gravity fits likewise
  differ between A2 and A3 by ~0.9°: per-take systematics from the hand loading.
- **The encoders are output-side** (17-bit output encoders, 19:1 gear; DMA.ethercat
  `dma_elmo_motor.cpp`), so gear backlash cannot explain the per-hold error: the head moves relative to
  the joint outputs (structure, bearings, a roll no joint measures).
- **Gaps in the recorded /joints stream** (100–900 ms in many holds, 2.3–2.6 s overall; the record pod
  was CPU-starved) do not explain the camera–gravity agreement: on gap-free holds only (A3: 1, 2, 5, 7,
  8, 9, 12, 15) it is about Head x corr +0.92, slope 0.96 and about Head y corr +0.90, slope 0.65.

## Rerun with the reviewed checks (16 Oct ~23:05)
Steps 75/81/82 after the Task 5/6/8/9 review fixes, same takes:
- Step 75: torso yaw marked unreliable on both (the robot rocked during the holds: rate scatter
  6.4e-4 / 1.9e-4 rad/s); A3 tilt > 0.1° in 10 holds (max 0.40), rotation during a hold > 0.1° in 11;
  A2 tilt in none, rotation in 4. Holds with a /joints gap while the neck moved dropped (A3: 7, A2: 5).
  Verdict RETAKE (holds, torso) on both, with actions.
- Step 81 on the remaining holds: A3 0.93° / 37.8 mm, A2 0.68° / 23.9 mm (clustered sigma up to 1.0° /
  13 mm on A3).
- Step 82: R_BS 0.51 ± 0.47° (A3), 0.98 ± 0.45° (A2): INCONCLUSIVE. Angle between the axes 88.65 ±
  0.81° over 18 strokes: consistent with 90°, no warning.

## Final rerun with all review fixes (17 Oct ~01:30; superseded by the rerun below)
| take | step 75 | step 81 | step 82 |
|---|---|---|---|
| A | **STOP** robot-or-board-moved (board spread in the torso 153 mm / 6.6°) | INCONCLUSIVE (model does not fit; 19 holds) | INCONCLUSIVE 0.74 ± 0.54° |
| A3 | RETAKE (holds, torso) | INCONCLUSIVE: 1.86° / 55.8 mm ± 2.9° / 24.9 mm, 8 holds | INCONCLUSIVE 0.51 ± 0.47°; axes 89.84 ± 1.58° |
| A2 | RETAKE (holds) | INCONCLUSIVE: 0.66° / 21.6 mm ± 1.1° / 8.7 mm, 8 holds | INCONCLUSIVE 0.98 ± 0.45° |

Step 81 on A3 moved from 0.93° / 38 mm (first rerun) to 1.86° / 56 mm with the stricter image selection
(images in /joints gaps dropped, board poses refined in pixels and needing 20 tags): with 8 holds the
answer swings, and the error bar now says so instead of REPLACE.

## Final rerun with the final code (17 Oct ~03:00; the reports and logs in `takes-2026-10-16/`)
| take | step 75 | step 81 | step 82 |
|---|---|---|---|
| A | **STOP** robot-or-board-moved (board spread 153 mm / 6.6°); also camera 2 stopped at 19 s, 19 usable holds | INCONCLUSIVE (model does not fit; 19 holds) | INCONCLUSIVE 0.74 ± 0.54° |
| A3 | RETAKE (holds, torso shifted, 8 usable holds) | INCONCLUSIVE: 1.86° / 55.8 mm ± 2.9° / 24.9 mm | INCONCLUSIVE 0.51 ± 0.47°; axes 89.84 ± 1.58° |
| A2 | RETAKE (holds, 8 usable holds) | INCONCLUSIVE: 0.66° / 21.6 mm ± 1.1° / 8.7 mm | INCONCLUSIVE 0.98 ± 0.45° |

A3 against A2 (`run-A3-vs-A2-v5.log`): 2.37° / 39.1 mm apart, inside their combined 1-sigma of 3.1° /
26.4 mm: the takes do not contradict each other; they are too noisy to measure anything at the gate.

## What tonight settles and what it does not
- Settled: encoder signs as the URDF; the NeckPitch IMU is the head's; R_BS within ~2° of the config
  (gyro, three takes); the camera calibration and
  the head IMU agree with each other hold by hold. Not settled: camera-in-Head is not settled (0.6–1.9° / 20–56 mm from the config across takes, error bars 1–3° / 9–25 mm on 8–16 holds).
- Not settled: KEEP or REPLACE at 0.3° / 3 mm. Hand-held data cannot reach it: its per-hold encoder
  error (~0.4°) is the floor.

## Next: powered neck (17 Oct)
With the motors holding commanded poses the load is the same at every pose (gravity only), there is no
hand, and holds are truly still. The same two checks decide:
1. gravity residual per hold (head IMU vs encoders; needs no camera, runs in seconds): ≲0.1° means
   the encoders are trustworthy when powered;
2. then the hand-eye fit: KEEP / REPLACE, two takes agreeing.
If the powered gravity residual stays ~0.3°, the neck's encoders limit any FK-based camera-in-torso to
~0.3° (~5 mm at 0.85 m) and grasping needs camera-to-arm calibration or visual feedback instead.
