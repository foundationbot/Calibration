# H0: read B ported and made faster; how short the burst can be (11 Nov 2026)

Hand-eye plan milestone H0 (`docs/plans/2026-11-11-handeye-plan.md`, section 8). The code is in
`tools/calibration/armcal/` (its README says what was ported from where). No robot was used. All runs were on Moonstone
(AMD Ryzen 9 9950X): one BLAS thread per process, `nice -n 10`, at most 8 processes.

- **Data:** all 48 stored phantom14 captures:
  - right arm wave 1 (`1103a`, 11) and wave 2 (`1104a`, 18);
  - left arm (`1104b`, 19).
- **Baseline:** the per-capture records phantom14's driver stored on 4 Nov (`urdfx/base_w1/base_w2/base_L.json`), with
  the base URDF (`#816`, md5 40879599).
- **Runs:** `bench.py` (one capture per process) and `frames.py`, driven by `queue_speed.sh` and `queue_frames.sh`.
  Logs: `speed.log`, `frames.log`; every run's summary: `h0_summary.json`.

## Result in plain words

1. **The port reproduces phantom14 exactly.** On all 48 captures, from the burst file to the record:
   - the numpy depth fusion plus the matcher give the stored wrist offsets to 3e-13 mm;
   - every clean / not-clean decision is unchanged.
2. **The C++ module is not needed.** Read B used only its per-pixel depth fusion, which is now numpy (0.16-0.34 s
   per burst). The robot needs numpy, scipy and cv2 only.
3. **Read B is 6-7x faster, with identical results.** Per capture on one core:
   - 13.0 s alone (was 78.9 s);
   - 20.9 s with 8 captures in parallel (was 135 s).

   Every change that alters the arithmetic failed the acceptance rule, so none is used.
4. **Read B is fragile on many captures.** This matters for the plan, not just for speed.
   - A change far below the depth noise (thinning the model's sample points to 0.5 mm) moved 32 of 48 captures by
     more than 0.5 mm, the worst by 30 mm, and flipped 11 clean decisions.
   - Even clean captures at about 1° match rotation moved 3-7 mm (left 07 and 08).
   - Each such capture lands in one of several nearby minima. Its exact offset is partly chance.
5. **A shorter burst does not make the calibration worse on this data.** Held-out wrist error with M12+head
   (median, right / left):

   | Frames | Error (mm) |
   |---|---|
   | 5 | 3.6 / 5.3 |
   | 4 | 3.9 / 3.9 |
   | 3 | 3.5 / 3.6 |
   | 2 | 4.0 / 3.1 |
   | 1 | 3.1 / 4.2 |

   Per capture, fewer frames move the offsets by a median of 1-3 mm. That is the same size as the fragility in
   point 4, not a loss of accuracy. **Proposal: 2 frames (frames come 1.93 s apart, so about 1.9 s instead of
   7.7 s), checked live in H3 with the paired held-out rule.** It rests on one robot and one wave.

## Speed (B0 per capture; acceptance: on all 48 captures the wrist offset within 0.5 mm and the match rotation within 1° of the stored record, and no clean / not-clean change)

| Version (each row adds to the one above) | s/capture/core, 8 parallel: mean (max) | Max deviation from stored | Accepted |
|---|---|---|---|
| reference: the algorithm as it ran on phantom14 | 135.1 (287) | 0 | baseline |
| one KD-tree per stage on the E-free model, the data moved by E⁻¹; one z-buffer per stage; coverage by pixel index | 73.0 (149) | 3e-13 mm | yes (exact) |
| z-buffer by `ufunc.minimum.at`, projection shared with the visibility test | 68.4 (146) | 3e-13 mm | yes (exact) |
| KD-tree unbalanced and not compacted (scipy options) | 27.2 (48) | 3e-13 mm | yes (exact) |
| leaf size 64 | 22.9 (39) | 3e-13 mm | yes (exact) |
| link slices instead of boolean masks when placing the model (**FAST**, the default) | **20.9 (37)** | **3e-13 mm** | **yes (exact)** |
| FAST alone on one core (8 wave-2 captures, both arms) | **13.0 (19.9)** | 3e-13 mm | — |
| reference alone on one core (capture right 01) | 78.9 | 0 | — |

How to read it:
- each row is one more change to the matcher, top to bottom;
- the time is the fit only; loading the burst (0.2-0.4 s) and the fusion (0.16-0.34 s for 5 frames) come on top;
- "8 parallel" means 8 captures at once, one per core; with all 8 sharing memory bandwidth, a capture takes about
  1.6x its time alone.

**Rejected** (each tried on top of the exact version, all 48 captures):

| Change | s/capture/core (8 parallel) | Max deviation | Clean flips |
|---|---|---|---|
| model points thinned to one per 0.5 mm voxel (5.2 M -> 0.93 M) | 19.0 (before the tree change) | 30.4 mm, 19.9° | 11 |
| ... to one per 1 mm voxel (0.26 M) | 10.4 (before the tree change) | 47.6 mm, 43.1° | 9 |
| every 2nd depth pixel in u and v | 12.9 | 14.1 mm, 108° | 5 |
| coarse stages capped at 15 steps (not 30) | 18.1 | 109 mm, 127° | 1 |
| last stage settles at 0.1 mm / 0.01° (not 0.01 mm / 0.001°) | 19.2 | 11.2 mm, 2.6° | 1 |
| last stage settles at 0.03 mm / 0.003° | 20.3 | 0.45 mm, 0.22° | 0: passes, but saves 3 % and uses up most of the tolerance |
| nearest neighbours kept between steps (exact) | no gain: 70-90 % had to be re-asked every step | — | — |
| KD-tree asked within 4 mm first (exact) | slower | — | — |

**Where the time goes now** (capture right 01 alone, 13 s):
- KD-tree queries: 9.4 s, about 70 %;
- the rest: placing, z-buffers, visibility, statistics.

Why the data are so heavy:
- the CAD sampler puts ceil(edge/1 mm)² points on every long, thin triangle;
- that gives 5.2 M points for 0.24 m² of surface, about 1.1 M of them visible;
- the data are only about 50 k points.

A cleaner remesh of the CAD (even sampling) would be the large next gain. It changes the model, so it needs a
re-baseline, and given point 4 it can't be held to the 0.5 mm rule.

## Burst length (FAST matcher, numpy fusion of the first N frames; deviations against all 5 frames)

| Frames | Burst span (measured: 1.93 s between frames) | All 48: median / max (mm) | 34 clean at 5: median / max (mm) | Clean flips | Stage 2 M12+head held-out, right: median / p90 | ... left: median / p90 |
|---|---|---|---|---|---|---|
| 5 | 7.7 s | 0 / 0 | 0 / 0 | 0 | 3.6 / 7.1 | 5.3 / 7.1 |
| 4 | 5.8 s | 0.95 / 39.6 | 0.70 / 32.5 | 5 | 3.9 / 5.1 | 3.9 / 6.5 |
| 3 | 3.9 s | 1.99 / 74.8 | 1.16 / 37.4 | 10 | 3.5 / 6.7 | 3.6 / 6.3 |
| 2 | 1.9 s | 1.83 / 75.3 | 1.50 / 15.1 | 7 | 4.0 / 5.9 | 3.1 / 6.4 |
| 1 | one frame | 2.88 / 78.3 | 2.24 / 26.9 | 14 | 3.1 / 7.6 | 4.2 / 7.0 |

How to read it:
- one row per number of depth frames fused;
- the span is the time between the first and the last frame; the capture also waits for its first frame;
- the deviation columns compare each capture's wrist offset with the 5-frame result;
- "stage 2" is leave-one-arm-pose-out on wave 2 of both arms, as in H-1: lower is better.

What changes with fewer frames:
- **the shoulder-roll terms stay put:** right −1.67 to −1.81°, left +0.21 to +0.52°;
- **the NeckYaw and camera-in-head terms swap value among themselves:** for example HEAD.z ranges from −8 to −23 mm.
  H-1 found them 0.93-0.97 correlated, and their sum is what the data determine.

## Time on a Thor CPU: an estimate, not measured

- **The ratio used:** per core, Thor ran the head calibration's step 45 at 1.2x (23 s vs 19 s, fast solver) to 1.5x
  (444 s vs 296 s, old solver) Moonstone's time (`docs/plans/2026-11-09-calibration-terminal-tool.md`, 9-10 Nov).
- **Estimate for read B on a Thor core:**
  - about 16-20 s per capture alone;
  - about 25-32 s with several captures in parallel;
  - plus about 0.5-1 s of loading and fusion.
- **Memory:** about 1.4 GB RSS per matcher process (measured on Moonstone).
- **What it means for the wave:** a capture comes every ~9 s with a 7.7 s burst, or about 3.2 s with a 2-frame burst.
  Keeping up live needs 3-4 matcher processes with 5 frames, about 8-10 with 2. Alternatively, the captures are matched
  during the next arm move and after the wave: 21 captures x 25-32 s / 4 processes ≈ 2-3 min per arm (estimate).
- **To be measured** on a Thor (with the user's OK): `python3 -m armcal.readb` on a stored burst.

## Reproduce

```
# Moonstone, from a worktree synced to ~/calibtool/wt-h0; data in ~/calibtool/data/p14arm (symlinks + copies)
bash queue_speed.sh reference exact model_voxel=0.0005 ...   # one bench.py run per config, 8 processes
bash queue_frames.sh                                         # frames 5/1/2/3/4 + frames.py
```
