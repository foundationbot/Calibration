# phantom12 T_BS, powered-neck takes, 17 Oct 2026: findings

Session `okvis2-x-d6`. Takes on Thor in `/root/calib-work/takes/tbs-neck-{P1,P2}` (record app, 420 s
each, exposure auto, the neck held by its motors through DMA.ethercat's `dma_motor_motion`; how:
`powered/README.md`). Reports and logs here; the analysis scripts and their outputs in `analysis/`.

**Superseded 21 Oct:** refitted with Fit A's KB4T lens, the same takes put camera 0 ~14 mm / 0.5° from the config (IMPROVE on both, cameras now 1.4 mm apart): `../phantom12-2026-10-20-handeye/FINDINGS.md`. The old lens pulled X ~5.2 mm along the optical axis. Kept below as the 18 Oct record.

## Answer
**The config's camera-in-Head is very likely wrong, by about 10 mm and 0.4°, but this measurement
cannot certify a replacement.** Both powered takes put camera 0 ~10 mm / 0.4° from where the config
(phantom21's T_BS) says (~16 mm at the gripper at 0.85 m; the gate was 0.3° / 3 mm), and they agree
with each other to 2.1 mm. The error bar is **~3.5 mm (1 sigma)**, half of it a disagreement between
the two cameras' own calibrations that no number of takes removes. So the config is ~3 sigma off, and
the fit is not shown better than the config at 2 sigma: step 81 says INCONCLUSIVE on both takes and
step 83 writes nothing. What would settle it: a check of the stereo calibration (then refit these takes
offline, no new take needed), or a camera-to-arm calibration.

| take | step 75 | step 81 (pixels): X vs config | the fit's own 1 sigma | cameras' systematic | total 1 sigma | verdict |
|---|---|---|---|---|---|---|
| P1 (powered) | GOOD | 0.375° / 9.95 mm (Head x y z +1.4 −9.5 +2.6) | 0.070° / 2.54 mm | 0.080° / 2.58 mm | 0.106° / 3.62 mm | INCONCLUSIVE |
| P2 (powered) | GOOD | 0.410° / 10.31 mm (+2.2 −9.0 +4.5) | 0.065° / 2.26 mm | 0.087° / 2.59 mm | 0.109° / 3.43 mm | INCONCLUSIVE |
| P2 vs P1 | | 0.044° / 2.10 mm apart | combined 0.095° / 3.40 mm | (common to both) | | agree |
| A3 (by hand, 16 Oct) | RETAKE | — | — | — | — | INCONCLUSIVE: 60 % of corners cut, 8 holds |
| A2 (by hand, 16 Oct) | RETAKE | — | — | — | — | INCONCLUSIVE: 25 % of corners cut, 8 holds |

## What the powered takes established
- **The neck held by its motors**: encoders vs the head IMU's gravity, per hold, **0.09° rms** after
  step 75's drops (`analysis/gravity_fit2-P1.txt` prints 0.26° with hold 33, whose 2.2 s /joints gap gives 1.6°)
  (16 Oct by hand: 0.20–0.40°). The powered encoders say where the head is; the hand-held takes'
  problem is gone.
- **A pixel-level hand-eye fit** (`handeye_px.py`, step 81's default) instead of per-image board
  poses. On the same images the pose-level fit gave 14.5 mm, its two cameras 12 mm apart: PnP is
  weakest along the optical axis and the pose fit weighs 1 rad like 1 m. The pixel fit's residual,
  0.77 px, is what the encoders' 0.09° makes at 1.1 m (~0.9 px); 0.4–0.5 % of corners are cut
  (hand-held: 25–60 %, and their survivors' rms fell under the 1.5 px line: only the cut share stops them).
- **The two cameras disagree.** Each camera's own pixel fit, on the even and on the odd holds of
  each take (`analysis/split_half.py`): camera 0 − camera 1 = (−1.4…−1.7, +4.6…+4.9, −1.0…−1.7) mm
  in Head axes, 4.9–5.4 mm, and 0.15–0.20°, the same in all four halves to ~0.5 mm, while each
  camera's own error bar is 3–5 mm. Their calibrations (intrinsics or the stereo pose) disagree by
  that much at 1.1 m, mostly sideways (Head y, along the baseline). Half of it goes into X's error bar.
- **Stereo triangulation does not work here**: a 45 mm baseline at 1.1 m scatters each corner's depth
  by ~20 mm (`analysis/proto-*.txt`, stereo rows).

## Rule changes against 16 Oct (for the user)
1. Verdict **IMPROVE** (new): the config shown outside the gate at 2 sigma, the fit's 2 sigma not
   inside it, and in rotation and in translation the fit either good to the gate or shown better than
   the config (the difference over 4 sigma). Not reached on 17 Oct.
2. Two takes **agree** within 0.1° / 1 mm, or within 2 combined sigma of the fits' own error when
   each is within the gate and they are within the gate of each other (each take's own sigma is over 1 mm).
3. X's error bar includes the cameras' systematic; a take with > 2 % of its corners cut, or one
   camera only, gets no verdict.
4. Step 75 grades holds over their own range and calls recorder gaps a warning (both cameras ran to the
   end; 92 % cover from the CPU-starved record pod).

## Settled and not settled
- Settled: the two takes and both cameras put camera 0 ~10 mm off the config, mostly along Head y
  (−9 mm); the powered encoders are good to ~0.1°; encoder signs as the URDF; the NeckPitch IMU is the
  head's.
- Not settled: that the config is outside the 3 mm gate. The fit is 2.7σ (P1) and 3.0σ (P2) from the
  config; at 2σ, P1 alone does not show it outside the gate (9.95 − 2 × 3.62 = 2.70 mm), P2 barely does
  (3.44 mm) (each report's `verdict_reason`). Nor camera-in-Head to 3 mm. Limits: the cameras' 5 mm disagreement (fixable by
  calibration), each take's own ~2.4 mm (more holds), and what no neck-only test sees (a NeckYaw zero,
  the yaw joint's placeholder position). R_BS by the gyro: step 82 INCONCLUSIVE because the
  angle plus 2σ is over its 1° gate (P1 1.16 ± 0.29°, P2 0.53 ± 0.42°; 16 Oct: within ~2°); the nod, at
  16°/s, was also too slow for its axes-angle test (20°/s).
