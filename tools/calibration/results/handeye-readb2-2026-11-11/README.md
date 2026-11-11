# readb2: read B made stable and faster (11 Nov 2026)

The step after H0 (`results/handeye-h0-2026-11-11/`): H0 ported read B's per-capture matcher (B0) exactly and found it
fragile. This makes it robust and faster. Code: `tools/calibration/armcal/` (`meshfit.ROBUST`, `readb.match_wave`).
No robot was used. All runs were on Moonstone (AMD Ryzen 9 9950X, shared): one BLAS thread per process,
`nice -n 10`, at most 8 processes.

- **Data:** the 48 stored phantom14 captures (right wave 1 `1103a` 11, wave 2 `1104a` 18, left `1104b` 19), fused
  in numpy from their bursts; the base URDF (`#816`).
- **Accuracy yardstick:** stage 2's leave-one-arm-pose-out cross-validation (CV) on wave 2 of both arms, as in H-1:
  the held-out wrist error in mm, lower is better.
- **Stability yardstick:** the same capture under changes that should not matter; how far its wrist offset moves.
- **"FAST"** is the matcher before this work (H0's default: phantom14's numbers, computed faster). **"ROBUST"** is
  the new one.

## Result in plain words

1. **Why B0 was fragile.** In the first, coarse stage it turned the forearm by up to 15° about its own axis. The last
   stage then crept back along that nearly flat direction, about 0.1° per step for hundreds of steps. It stopped
   wherever its steps happened to get small (`diag.py`). Captures that differ only in detail stop in different
   places. The competing answers differ mostly by a roll about the forearm axis: for most pairs, the cosine
   between the rotation that separates them and that axis is 0.9-1.0.
2. **What changed** (all in `meshfit.ROBUST` and `readb.match_wave`):
   - **the model:** the CAD meshes sampled evenly, about one point per mm² (0.24 M points, not 5.2 M);
   - **the coarse stages** (30 and 15 mm gates) fit the translation only;
   - **every step** is scaled by a line search on the stage's cost, so each stage goes down to a minimum of a fixed
     function;
   - **a weak pull** keeps the forearm's roll about its own axis at the encoders' value unless the depth says
     otherwise;
   - **every 2nd pixel** is used until the last stage;
   - **two passes:** each capture is matched from the encoders, then again from the wave's median offset. A capture
     counts as clean only when the two answers land within 3 mm of each other ("settled").
3. **More accurate on the plan's model.** M12+head held-out error, median / p90:
   - right 2.8 / 5.8 mm (FAST: 3.6 / 7.1);
   - left 4.5 / 6.3 mm (FAST: 5.3 / 7.1).

   It is better in every run tried. On the fallback model M12, the right arm is better and the left arm is worse in
   the single base run (5.7 / 12.0 against 5.3 / 11.0), but better on the mean over equivalent runs.
4. **Stable where FAST was fragile.** Moving the model's sample points now moves a capture by 0.25 mm (median; p90
   0.6 mm), not 0.7 mm (p90 3.1 mm).
   - Other frames of the same burst still move a capture by about 1.5 mm (median), with either matcher. That is
     the noise of the fused depth itself.
   - A few captures (3-5 of 32) still jump by 7-17 mm under such a change. The two-pass check catches most jumps,
     not all.
5. **About 5x faster, a third of the memory.**
   - Per capture on one core: 2.9 s for both passes (FAST: 13.7 s).
   - 8 in parallel: 5.2 s (FAST: 23.0 s).
   - Memory: 0.5 GB (FAST: 1.7 GB).
6. **The price is captures.** One third of the captures are not settled and are left out:
   - wave 2: stage 2 uses 13 right / 11 left, against FAST's 14 / 14;
   - the capture plan should allow for that (more neck angles per arm pose).
7. **Default switched.** `readb` now runs ROBUST by default. `--matcher fast` reproduces the stored phantom14 records
   (unchanged, still tested to 3e-13 mm).

## Accuracy: stage 2's cross-validation (wave 2, both arms)

Base runs (all 5 frames, each matcher as it is):

| Model | Arm | FAST: median / p90 (mm) | ROBUST: median / p90 (mm) |
|---|---|---|---|
| M12+head (the plan's default) | right | 3.60 / 7.13 | **2.82 / 5.77** |
| M12+head | left | 5.29 / 7.13 | **4.50 / 6.31** |
| M12 (fallback) | right | 5.30 / 10.35 | **4.77 / 10.20** |
| M12 | left | **5.31 / 10.97** | 5.69 / 11.96 |
| captures used (right / left) | | 14 / 14 | 13 / 11 |

How to read it:
- each cell is the wrist error on arm poses the fit did not see, as in H-1;
- the bold cell is the better of the two;
- "captures used" are those clean (for ROBUST: and settled) and within 15 mm of the arm's median.

**One run's p90 hangs on one or two captures** (11-14 per arm). The same CV over runs that should not differ:

| Model | Arm | FAST: mean of 3 runs | ROBUST: mean of 4 runs |
|---|---|---|---|
| M12+head | right | 3.86 / 7.56 | **2.75 / 5.78** |
| M12+head | left | 4.91 / 6.92 | **4.07 / 5.94** |
| M12 | right | 5.60 / **10.16** | **4.60** / 10.22 |
| M12 | left | 5.33 / 10.21 | **5.06 / 9.67** |

How to read it:
- the cells are median / p90 in mm, averaged over the base run and its perturbed copies with all 5 frames;
- FAST's runs are base, model points jittered and start moved;
- ROBUST's runs are those three plus model points chosen otherwise;
- every run is in `cvens.log`, which lists, for example, FAST's left M12 p90 ranging over 8.4-11.3 mm.

**Per held-out arm pose** (M12+head, `stab.log`):
- left pose L10 stays 12-14 mm off with either matcher;
- FAST kept 1 of its 4 captures, by chance (the others turned more than 12°);
- ROBUST matches 3 of them consistently, within about 3.5 mm of each other, and also keeps one: the others are not
  settled, or beyond the 15 mm outlier cut;
- that pose is a limit of the stage-2 model or of B0's bias there, not matcher noise (as H-1 said).

## Stability: the same capture under changes that should not matter

| Change | FAST: captures, median / p90 / max (mm) | ROBUST: captures, median / p90 / max (mm) |
|---|---|---|
| every model point moved 0.3 mm in its face's plane | 30: 0.66 / 3.14 / 8.8 | 26: **0.25 / 0.59 / 1.1** |
| the model's points chosen otherwise (R2 sequence started elsewhere) | (not applicable) | 29: 0.24 / 0.69 / 11.4 |
| only the first 2 depth frames | 27: **1.40 / 2.58 / 3.9** | 28: 1.66 / 2.63 / 7.0 |
| only the last 2 depth frames | 31: 1.79 / **3.97 / 14.2** | 25: **1.49** / 4.54 / 17.4 |
| only the first 3 depth frames | 27: **0.96** / 2.91 / **9.2** | 26: 1.08 / **2.75** / 16.9 |
| the start moved by 1° and 2 mm | 33: **0.08 / 0.72 / 4.3** | 26: 0.30 / 1.44 / 12.3 |
| worst of these per capture | 21: 2.08 / **5.15 / 8.8** | 16: **2.03** / 5.87 / 16.9 |

How to read it:
- each cell is how far a capture's wrist offset moved from its base run (mm);
- only captures clean in both runs count (for ROBUST: clean and settled), and the count comes first;
- lower is better; bold is the better of the two.

**The same, over every capture clean in the base run** (whether or not the changed run kept it; `stab.log`,
`agree.log`):

| Change | FAST, 34 captures | ROBUST, 32 captures |
|---|---|---|
| model points moved | 0.68 / 3.93 / 8.8 | 0.30 / 1.04 / 9.8 |
| first 2 frames | 1.52 / 5.81 / 29.1 | 1.72 / 3.79 / 7.0 |
| last 2 frames | 1.88 / 5.45 / 50.5 | 2.04 / 8.49 / 19.0 |
| first 3 frames | 1.15 / 6.09 / 25.2 | 1.29 / 10.28 / 20.3 |
| start moved | 0.08 / 0.71 / 4.3 | 0.33 / 2.64 / 19.7 |
| clean decisions flipped (per change) | 1-10 | 7-12 |

How to read it:
- these are median / p90 / max in mm;
- before this work (H0), thinning FAST's model moved 32 of 48 captures by more than 0.5 mm, some clean ones by
  3-7 mm;
- now model changes move a ROBUST capture by about 0.3 mm (median).

What the two tables say:
- **model sampling no longer matters** (the H0 finding is fixed);
- **depth frames matter about as much as before:** about 1.5 mm median with either matcher, which is the noise of
  the depth;
- **FAST is steadier against a small change of its start:**
  - its first stage jumps far in any case, and a 1° change does not survive that;
  - ROBUST keeps the encoders' orientation in its coarse stages, so a 1° change reaches the fine stages;
- **ROBUST's settled/not-settled decision is itself unsure** for 7-12 of the 48 captures.

## Speed (s per capture, the match only)

| Matcher | One process (8 wave-2 captures): mean / max | 8 in parallel (all 48): mean / max |
|---|---|---|
| FAST | 13.71 / 19.39 | 22.99 / 40.25 |
| ROBUST pass 1 (from the encoders) | 1.64 / 2.27 | 2.87 / 6.29 |
| ROBUST pass 2 (from the wave's median offset) | 1.21 / 2.42 | 2.32 / 6.34 |
| **ROBUST, both passes** | **2.85** | **5.19** |

How to read it:
- "one process" is one capture at a time on an otherwise idle core;
- "8 in parallel" is 8 captures at once, one per core, sharing memory bandwidth on a box other users were also
  loading;
- the burst's loading and fusion come on top: 0.17 s for 5 frames alone (fusion 0.15 s); with 8 in parallel the
  fusion takes 0.44 s (5 frames), 0.27 s (3) or 0.22 s (2);
- ROBUST loads each capture once per pass;
- the readb command on 3 captures: 9.6 s wall and 0.51 GB of memory (FAST: 48.3 s and 1.75 GB).

Where ROBUST's time goes:
- about 80 % is KD-tree queries: about 100 per capture, about 10 ms each, over about 20 k points;
- the line search asks 2-4 times per step.

Tried and dropped:
- keeping each point's 8 nearest model points between steps: exact, but slower (most points had to be asked again);
- 1.5 mm model spacing: 10 % faster, but the CV got worse;
- a cheaper first pass (every 4th pixel, looser stops): saves 0.4 s, but 4 more captures fail the settled check
  (28 settled of 48, against 32).

## A capture on the robot, end to end

Measured on Moonstone, one core, 5 frames:

| Step | Time |
|---|---|
| load + fuse | 0.17 s |
| pass 1 | 1.64 s |
| load + fuse again | 0.17 s |
| pass 2 | 1.21 s |
| **total** | **about 3.2 s** |

**On a Thor core: an estimate, not measured** (no robot access).
- H0's ratio, Thor at 1.2-1.5x Moonstone's time per core, gives about 3.8-4.8 s per capture.
- **Live, pass 1 on each capture as it lands:** about 2.2-2.7 s (estimate). That is shorter than the ~9 s between
  5-frame captures, so one process keeps up and the screen shows each capture's quality at once.
- **Pass 2 after the wave**, for all its captures, from the wave's median: 20 captures x about 1.6-2.0 s / 4
  processes, about 8-10 s per arm (estimate).

## Proposed default for the live tool

- **Matcher:** ROBUST, two passes, settled within 3 mm (`readb --matcher robust`, the default), `--jobs 4` for
  pass 2.
- **Frames per burst: keep 5** (7.7 s).

  M12+head, median / p90, right | left:

  | Frames | ROBUST | FAST |
  |---|---|---|
  | 5 | 2.75 / 5.78 \| 4.07 / 5.94 (mean of 4 runs) | 3.86 / 7.56 \| 4.91 / 6.92 |
  | 3 | 2.47 / 6.05 \| 4.30 / 7.92 (one run) | 3.52 / 6.63 \| 3.62 / 6.40 |
  | 2 | 2.67 / 4.11 \| 4.59 / 8.61 (mean of 2) | 3.78 / 5.03 \| 3.50 / 6.56 |

  - With ROBUST, fewer frames cost the left arm's p90. H0's 2-frame proposal was made with FAST.
  - The matcher no longer dominates the time; the burst does.
  - 3 frames (3.9 s) is the candidate to check live in H3, with the paired held-out rule.
- **Capture plan:** expect about a third of captures not to settle. Plan 4 neck angles per arm pose, not 3.

## Open problems

- **Depth noise sets the floor.** The same capture fused from other frames moves 1-2 mm (median), with either matcher.
- **A residue of jumps.** 3-5 settled captures of 32 still jump 7-17 mm under a frame or start change. The settled
  decision flips for 7-12 of 48 captures per change.
- **The settled check is conservative.** Pass 1 starts at the encoders, 20-35 mm away, and sometimes lands far
  off (left 11-pL09: 18 mm from pass 2). A capture whose pass 2 may well be right is then dropped.
- **The roll pull has a cost.** It keeps part of a true roll about the forearm out of the answer. On synthetic data
  that is up to 0.7° of 2°, and the wrist point moves less than 1 mm (`test_meshfit.py`).
- **The clean test's rotation limit** (≤ 12°) seldom fires under the pull. Settledness, coverage and stage 2's 15 mm
  outlier cut do the work.
- **M12's left arm** is worse in the single base run (median 5.7 against 5.3 mm). The mean over equivalent runs is
  better.
- **One robot, one wave per arm.** All tuning (roll pull 0.03, 3 mm settled, strides) was done on this data. The
  numbers are in-sample for those choices. The pull was flat over 0.003-0.1; the rest was chosen for speed or
  stability, not for CV.
- **Thor timing** is still to be measured: `python3 -m armcal.readb BURST.npz... --side right --urdf U --out o.json`.

## Reproduce

```
# Moonstone, from a worktree synced to ~/calibtool/wt-readb2; data in ~/calibtool/data/p14arm
W=~/calibtool/readb2-work bash queue.sh "fast|fast|base jit f2 f2b start f3" \
  "K1|robust|base jit phase f2 f2b start f3" "K2|robust|base jit phase f2 f2b start f3|--pass1 $W/r_K1_VAR.json"
python3 stab.py $W fast K2 > stab.log         # per change: deviations, CV (own / common captures, 2-3 frames)
python3 agree.py $W K1 K2 3 > agree.log       # the settled check (pass 2 within 3 mm of pass 1)
python3 cvens.py $W fast K2:K1 K2 > cvens.log # CV of every run
python3 both.py $W > both.log                 # stability over the captures clean in both runs
```

- `b2.py` runs one config and one change over all 48 captures. The configs are `fast`, `robust`, or either with
  `,key=value` overrides of `FitConfig`.
- `runs.json` holds every run's settings and timing, including the ones explored on the way:
  - `r1`-`r3`, `c1`-`c3`: translation-only stages and multi-start;
  - `p1`-`p4`: roll-pull weights;
  - `A`-`J`: strides, line search and spacing.
- `records_robust_pass1.json` and `records_robust_pass2.json` are ROBUST's per-capture records of the base run.
- `diag.py` shows the creep, step by step.
