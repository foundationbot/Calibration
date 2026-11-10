# calibtool M1 C2: the collection protocol, measured (9 Nov 2026)

Design plan section 13.1, measured on the stored neck-grid takes of phantom10 (ex-phantom11; 6 Nov `n6_*` and 5 Nov
`d*`) and phantom21 (5 Nov). Code: `tools/calibration/calibtool/protocol.py` at commit e5a986c1; every number below
comes from the commands in "Commands" (Moonstone, `~/calibtool/c2-work`).

## Result in plain words

1. **Pose lists.** Up to 1.6 m every one of the 49 grid poses sees the board (>= 6 tags in both cameras) in at least
   half the takes, and the outer ring is what fills the image edges: **keep all 49 poses at 0.5, 0.75, 1.1 and
   1.6 m.** At 2.0 m drop 8 poses (41 left), on the tilted placement drop the 5 that never saw the board (44 left);
   neither drop loses an edge-strip corner. Saving: 8 x 3.13 s = 25 s and 5 x 3.13 s = 16 s. The grid is not where
   the time goes.
2. **Frames per pose.** Keeping 1 pair per pose instead of 3 gives the same fit (rms 0.416 vs 0.417 px; TEST median
   -0.04 % on the common TEST set; rows 0.279 px), but on its own TEST set one entry drops below 8 holds (8 judged
   vs 9). **2 pairs per pose is the smallest with no measurable loss** (9 of 9 entries, median -0.05 %, IQR
   -0.46..+0.34 %, rows 0.280 px). It also halves the fit time (373 s vs 812 s). It saves no robot time with today's
   capture script, whose hold is fixed. Ending the hold once the frames are in would cut each pose from 3.13 s to
   about 2.0 s.
3. **Far limit.** The detector loses tags past about 1.6 m: 12-23 of 35 tags at 1.62-1.67 m, 8-14 at 1.90-1.95 m,
   1 at about 2.5-3 m, where no pose is detected (p10 `d300`: 0 of 49). Corner noise does not grow with distance
   (median per-corner sd 0.09-0.14 px on p10, 0.05-0.06 px on p21). **Ask for the far placement at 1.8 m measured,
   accept 1.7-2.0 m and refuse beyond 2.1 m.**
4. **Targets.** No stored session meets `coverage.Targets` as configured, although both sessions PASSED step 66.
   Thirds >= 5,000 FIT corners: best 4,599 (cam0) and 5,251 (cam1). Rim >= 1,500: best 561. Entries judged with
   >= 10 TEST holds: 5 in each session, against 6 wanted. p21 also has 43 far pairs against 50, and its left edge
   has 93 corners against 535 on the right. The defaults alone judge 1 entry (p10) or 3 entries (p21); the rest
   came from off-axis extras. C3 has to re-set these thresholds (numbers in section 4). **Expected session:
   6 default placements, 281 poses: about 21 min at today's 3.13 s/pose, or 16 min with the shorter hold; each
   extra adds about 3.6 min (2.7 min). Plan on 2-4 extras, 28-35 min (21-26 min).** Every placement time is a 60 s
   placeholder; it has not been measured.

**Missing.** Phantom21's raw takes were still copying. The noise and stillness columns cover only `d050`, `d050b` and
`d110`; `d160` had 35 of 49 pose files when the analysis ran, and `d200`, `dnear`, `dleft` and `dtiltleft` had none.
Phantom21's detection, distance, pose grids and target costs come from its stored corner files, which are complete
(the same 15b rule). The per_hold re-dumps cover phantom10 6 Nov only, as asked.

## Data and definitions

- Raw takes: every frame (5 per pose, 15 fps) of every pose file, detected in both cameras (`protocol detect`). Step
  15b's still-pair rule, re-applied to these detections, gives the same pairs as the recorded `out1280/n6_200.npz`.
  Corners agree to <= 0.0014 px; the robot's and Moonstone's OpenCV builds differ by 4 corners out of 19,456
  (`test_selection_reproduces_a_recorded_corner_file`). The per_hold 3 re-dump with 15b reproduces the recorded p10
  run exactly: 587/572 split, rms 0.417 px over 578 pairs, PASS with the same 9 entry values.
- Detected pose: at least 1 still pair with >= 6 tags in both cameras (15b). Distance: step 16's rough z of the
  pose's middle still pair. Band: the nearest nominal distance (0.5/0.75/1.1/1.6/2.0/3.0 m, edges at the midpoints).
  Bearing: neck yaw minus the board centre's angle off the optical axis (+ = toward the robot's left). Default
  placements = upright, |bearing| <= 25 deg, all 49 poses. That keeps off-axis `n6_litR/nearR/right` (-37..-51 deg)
  out and p21 `d050` (-21 deg) in.
- Corner noise: per corner, the sd of its position over a pose's still frames (corners seen in >= 2), after removing
  each frame's median shift. Reported as the median over corners, plus the "unstable" share of corners with
  sd > 0.5 px. The pooled sd is much larger (0.22-0.44 px). The cause is real: a share of the corners jumps 1-2 px
  between frames of a still head (0-27 % per take, median over its poses; seen on `n6_near` poses 0/24/40). A
  pooled sd would report those jumps as noise.
- Target costs (`protocol.target_costs`): holds replayed in capture order through `coverage.fit_tally`/`Coverage.deficits`
  with step 16's assignment (`split16_assignment`, holdout_every 2). TEST cells use the nominal principal point.
  A target's cost = the poses swept (every pose of earlier takes, detected or not) when it was met for the last
  time. Relative edge/box targets (>= 0.5 x the mirror strip) count as "met" at pose 1 while both strips are empty,
  so only their unmet/late values carry information.

## 1. Pose usefulness per default distance

Measured distance of the placements the operators made as the defaults:

| nominal | measured (take) |
|---|---|
| 0.5 m | p10 `d050` 0.64, `n6_near` 0.63; p21 `d050` 0.69, `d050b` 0.66, `dnear` 0.68; only p21 `dleft` reached 0.42 |
| 0.75 m | p10 `d075` 0.79, `n6_075` 0.83 |
| 1.1 m | p10 `d110` 1.03, `n6_110` 1.08; p21 `d110` 1.03 |
| 1.6 m | p10 `d160` 1.62, `n6_160` 1.39, `n6_200` 1.67 (named 2.0); p21 `d160` 1.62 |
| 2.0 m | p10 `d200` 1.90, `n6_far` 1.91; p21 `d200` 1.95 |

So 5 of 6 "0.5 m" boards stood at 0.63-0.69 m. The 0.5 m band needs the tool's distance check before the sweep.

| band (takes) | poses detected in >= half the takes | proposed list | dropped (pitch, yaw deg) | edge-strip corners kept | time per placement |
|---|---|---|---|---|---|
| 0.5 m (1: p21 `dleft` 0.42 m) | 49 | 49 | none | 657/657 | 49 x 3.13 = 153 s |
| 0.75 m (7) | 49 (all 7/7) | 49 | none | 6,000/6,000 | 153 s |
| 1.1 m (3) | 49 (47 in 3/3) | 49 | none | 2,795/2,795 | 153 s |
| 1.6 m (4) | 49 (min 2/4, row +24) | 49 | none | 2,430/2,430 | 153 s |
| 2.0 m (3, 1.90-1.95 m) | 35 | 41 | (-24,-34) (-24,-23) (-24,-11) (-24,+11) (-16,-34) (0,+34) (+8,+23) (+8,+34): each 1/3, no edge strip | 823/823 | 128 s |
| tilted (3: p10 `n6_tiltL/R`, p21 `dtiltleft`) | 41 | 44 | (-24,-34) (-24,-23) (-24,-11) (-24,+23) (-24,+34): 0/3 | 2,057/2,057 | 138 s |

Rule (`protocol.propose`): keep a pose if it was detected in >= 50 % of the band's takes, or detected at least once
and it filled an edge strip. With the plain 50 % rule, 2.0 m would keep 35 poses but lose 235 of 823 edge-strip
corners, because rows +16/+24 fill the top strip in 1 of 3 takes. What the poses add (per-pose grids in the appendix):
the left/right strips come from the yaw +-34 (and +-23) columns, the bottom strip from pitch <= -8 at 0.75 m, the
top strip from pitch >= 0. The centre poses feed the centre/top TEST cells. No pose at <= 1.6 m is idle.

## 2. Frames (pairs) per pose: phantom10 6 Nov, 11 takes, 15b re-dumped with per_hold 1/2/3/5

Every variant is step 16 (max_fit_z 3.0, holdout_every 2), then step 45 (KB4T, max_z 3.0), then step 66 + summary row
on its own TEST set. The fit is also scored on the common TEST set, the recorded `fit1280/test.npz`. Its holds are the
same in every variant: the holdout is per hold, and every variant keeps the same holds.

| per_hold | pairs (FIT/TEST) | step 45 | own TEST: step 66 | own TEST: median (IQR) %, rows px | common TEST: step 66 | common TEST: median (IQR) %, rows px | fit s |
|---|---|---|---|---|---|---|---|
| 1 | 398 (203/195) | rms 0.416 px over 200 pairs, baseline 44.06 mm | PASS 8 entries (bottom 0.4-0.6 m: 7 holds < 8) | -0.036 (-0.445..+0.304), 0.278 | PASS 9 | -0.036 (-0.441..+0.324), 0.279 | 222 |
| 2 | 783 (397/386) | rms 0.416 px over 389 pairs, 44.05 mm | PASS 9 | -0.046 (-0.463..+0.341), 0.280 | PASS 9 | -0.042 (-0.451..+0.334), 0.280 | 373 |
| 3 (recorded) | 1,159 (587/572) | rms 0.417 px over 578 pairs, 44.06 mm | PASS 9 | -0.042 (-0.451..+0.331), 0.279 | PASS 9 | (same set) | 812 |
| 5 | 1,866 (946/920) | rms 0.417 px over 936 pairs, 44.06 mm | PASS 9 | -0.013 (-0.422..+0.335), 0.279 | PASS 9 | -0.020 (-0.440..+0.319), 0.279 | 1,078 |

Entry values (common TEST, %), per_hold 1 / 2 / 3 / 5: top 0.9-1.3 0.44/0.45/0.45/0.43; left 0.6-0.9
-0.18/-0.19/-0.18/-0.17; left 0.9-1.3 -0.26/-0.25/-0.25/-0.19; centre 0.9-1.3 0.27/0.29/0.29/0.25; right 0.6-0.9
0.18/0.18/0.15/0.18; right 0.9-1.3 -0.29/-0.30/-0.30/-0.31; bottom-left 0.9-1.3 -0.74/-0.75/-0.76/-0.74; bottom
0.4-0.6 -0.25/-0.26/-0.26/-0.26; bottom 0.9-1.3 0.06/0.05/0.04/0.01. Summary bands (common TEST, (lo, hi] m: n,
median %): per_hold 1 0-1.0: 199, -0.12; 1.0-1.3: 184, +0.02; 1.3-1.7: 127, +0.22; 1.7+: 62, -0.46. per_hold 5
-0.11 / +0.04 / +0.24 / -0.48. Every difference is <= 0.07 % on an entry and <= 0.03 % on the median, against limits
of tenths of a percent, so it is not measurable.

Fit seconds: 4 fits in parallel on the shared box. The recorded p10 fit (per_hold 3) took 510 s alone.

**Recommendation: per_hold 2.** Per_hold 1 loses nothing in the fit but judges one entry fewer. A hold at a bin or
cell border keeps only its middle pair, which can fall outside the entry: bottom 0.4-0.6 m had 7 holds instead of 8,
and bottom 0.9-1.3 m 8 instead of 10. Capture 3 frames per pose: 15b's still check compares later frames with the
first detected one, so a single frame is never checked for stillness. 15b then keeps the middle 2.

Coverage counts scale with per_hold. FIT pairs and corners are 0.67 x the per_hold 3 values at 2 and 0.34 x at 1
(pairs 783 and 398 vs 1,159). `coverage.Targets`' pair and corner targets assume 3 pairs per pose and have to be
re-based with it.

**Time per pose** (from the stamps: frames 66.67 ms apart, the 5-frame window 266.7 ms, pose period 3.13 s in all 24
raw takes). Step 08 ramps 1.0 s, waits 0.8 s settle, grabs the pairs, then waits out the motion command's fixed 2.0 s
hold. Capture ends about 2.1 s into the 3.13 s pose, so fewer frames save **0 s** today. With the hold ended at the
last frame, the period is about 1.8 s + n x 66.7 ms + 0.13 s overhead: about 2.2 s at 5 frames and 2.07 s at 3. That
saves about 1.06 s per pose (34 %), or 5.0 min over the 281 default poses. This number is computed from the script's
waits and the measured frame spacing; it has not been measured. No take was captured with less than 1.8 s of
ramp + settle, so whether the settle time could be shorter is not measured. At <= 1.1 m, 96-100 % of the detected
poses had all 5 frames still (p10 `d110` 81 %). The lower shares at 1.6-2.0 m (69-85 %; `n6_160` 96 %) come with partial
detections, where frames can drop below 6 tags. This analysis does not separate that from motion.

## 3. Far limit (7 x 5 board, 1280 x 800)

| measured distance | poses (raw takes) | detected | left tags at the 9 centre poses (of 35), per take | tag px | corner noise L (median sd) | unstable corners |
|---|---|---|---|---|---|---|
| 0.42-0.83 m | p10 0.63-0.83 m takes | 49/49 each | 33-35 | 53-70 (102 at 0.42) | 0.10-0.13 px (p21 0.05-0.06) | 3-15 % |
| 1.03-1.08 m | `d110`, `n6_110`, p21 `d110` | 47-49 | 33-35 | 43-45 | 0.11-0.13 (p21 0.05) | 1-7 % |
| 1.39 m | `n6_160` | 45 | 33 | 34 | 0.095 | 5 % |
| 1.62-1.67 m | `d160`, `n6_200`, p21 `d160` | 41-43 | 12-23 | 28-29 | 0.09-0.14 | 1-4 % |
| 1.90-1.95 m | `d200`, `n6_far`, p21 `d200` | 44, 32, 22 | 14, 14, 8 | 24-26 | 0.10-0.15 | 2-11 % |
| ~2.5-3.2 m | p10 `d300` | 0 | 1 | 18 (8 poses with >= 2 tags) | - | - |

Pooled over the raw takes in distance buckets (appendix, "Detection and corner noise vs distance"): detected 97 %
at 0.50-0.75 m, 81-93 % at 0.75-2.25 m (where a pose had a distance estimate), 0 % at >= 2.5 m; tag size 65.6 ->
24.6 -> 17.7 px. Corner noise has no distance trend: 0.09-0.14 px median sd from 0.5 to 2.25 m.

**Recommendation: ask for 1.8 m measured; accept 1.7-2.0 m (the FIT far band starts at 1.7 m); refuse beyond 2.1 m.**
At 1.67 m the board kept 41/49 poses and 23 tags. At 1.90-1.95 m detection depends on the room: 22-44 of 49 poses,
8-14 tags, about 25 px tags (the "~27 px tags at ~2.15 m, 22/49" of 13.1 is p21 `d200`). At about 2.5 m nothing is
detected. Going farther than 2.0 m needs a board with bigger tags, not more poses.

## 4. Poses per coverage target and session length

Full tables in the appendix ("Poses per coverage target"). The scenarios: the session as recorded; the proposed
lists; defaults + one tilted placement only, which leaves out p10 `n6_litR/nearR/right/tiltR` and p21
`dleft/dtiltleft`.

**phantom10 (539 poses swept, 11 takes, 398 holds):**
- cam0 edges: left met at pose 50 (`n6_075`), top at 96, bottom at 364 (`n6_tiltL`: 39 FIT holds put corners
  there). Right never: 68 corners vs 222 left.
- FIT pairs >= 500 at 434. Far pairs >= 50 at 507, in the last take `n6_far` (23 holds).
- TEST entries with >= 10 holds: left 0.6-0.9 at 79 (`n6_075`); right 0.9-1.3 at 310, centre 0.9-1.3 at 322 and
  top 0.9-1.3 at 337 (all `n6_right`); right 0.6-0.9 at 459 (`n6_litR`). That makes 5 judged; the bottom side is
  never judged (7-9 holds).
- Defaults + `n6_tiltL` only (343 poses): 1 judged entry (left 0.6-0.9), and FIT pairs stay under 500. The
  0.9-1.3 m entries came from the off-axis extras.

**phantom21 (392 poses, 8 takes, 354 holds):**
- cam0 left edge never: 93 vs 535, 18 FIT holds, even with `dleft`. Bottom-left box never. Top-left box at 288
  (`dnear`).
- FIT pairs >= 500 at 383. Far pairs never: 43 of 50 (20 holds).
- Judged: bottom 0.6-0.9 at 65, right 0.6-0.9 at 83, centre 0.6-0.9 at 269, centre 0.4-0.6 at 326, bottom 0.9-1.3
  at 368. That makes 5; the left and top sides are never judged.
- Defaults only (294 poses): 3 judged, and FIT pairs stay under 500.

**Thresholds vs what the passing sessions reached** (FIT side, per_hold 3, as recorded):

| target (`coverage.Targets`) | p10 | p21 | met? |
|---|---|---|---|
| thirds >= 5,000 corners (TL/TR/BL/BR) | cam0 4,599/3,111/3,929/2,991; cam1 5,251/3,443/4,421/3,839 | cam0 1,872/2,997/2,227/4,660; cam1 2,074/2,865/2,881/4,856 | 1 of 16 |
| rim >= 1,500 | cam0 275, cam1 561 | 459, 534 | no |
| far pairs >= 50 | 63 | 43 | p10 only |
| FIT pairs >= 500 | 587 | 512 | yes |
| judged >= 6 entries at 10 TEST holds | 5 | 5 | no |
| each side judged | left, right, top (not bottom) | right, bottom (not left, top) | no |

Step 66 judged 9 and 7 entries itself. It needs 8 holds and uses the fitted principal point, and both sessions passed.
So the 10-hold, 6-entry, thirds and rim thresholds are stricter than any passing session reached. Those are C3's
numbers to set.

What a target costs in poses: a TEST entry needs 10 TEST holds, which is about 20 detected holds in its (cell, bin)
at holdout_every 2. A default placement delivers that for at most 1-3 entries; each of the 0.9-1.3 m edge entries took
an off-axis extra of 49 poses. The far FIT target took a whole 2.0 m take (p10: 23 contributing holds for 63 pairs;
p21: 20 holds, 43 pairs, short).

**Expected session length** (3.13 s/pose measured; each placement 60 s, a **placeholder, not measured**):

| part | poses | at 3.13 s/pose | at about 2.07 s/pose (hold ended at capture, computed) |
|---|---|---|---|
| defaults 0.5 / 0.75 / 1.1 / 1.6 / 1.8 m + tilted: 49+49+49+49+41+44 | 281 | 14.7 min + 6 x 1 min = **20.7 min** | 9.7 + 6 = **15.7 min** |
| each adaptive extra (49 poses + placement) | 49 | 3.6 min | 2.7 min |
| typical session: defaults + 2-4 extras (p21 needed 2, p10 4) | 379-477 | **28-35 min** | **21-26 min** |

## Commands

All on Moonstone, code synced to `~/calibtool/wt-c2` from commit e5a986c1, `OMP_NUM_THREADS=1`, `nice -n 10`,
`CALIBTOOL_DATA=~/calibtool/data`, `PY=~/calibtool/venv/bin/python`, run from `~/calibtool/wt-c2/tools/calibration`.

```bash
# 2. per_hold re-dumps (15b on the raw takes), per ph in 1 2 3 5 and per take t of the 11 n6_* takes:
$PY scripts/15b_dump_neck_take.py ~/calibtool/data/p10/kit/takes/$t ~/calibtool/c2-work/ph$ph/$t.npz \
  --rig config/rig-phantom11.yaml --still-px 1.0 --per-hold $ph --min-tags 6 --procs 6
# 2. each variant: split16 + fit45 + validate66/summary on own and common TEST (4 in parallel)
$PY -m calibtool.protocol variant --rig config/rig-phantom11.yaml --corners ~/calibtool/c2-work/ph$ph \
  --out ~/calibtool/c2-work/var-ph$ph --common-test ~/calibtool/data/p10/kit/fit1280/test.npz \
  n6_075 n6_110 n6_160 n6_200 n6_far n6_litR n6_near n6_nearR n6_right n6_tiltL n6_tiltR
#    -> var-ph$ph/variant.json (+ 45.log, validate-own/, validate-common/)
# 1, 3. every frame detected (R10=~/calibtool/data/p10/kit/takes, R21=~/calibtool/data/p21/kit/takes)
$PY -m calibtool.protocol detect --rig config/rig-phantom11.yaml --out ~/calibtool/c2-work/det/p10 --procs 2 \
  $R10/n6_075 $R10/n6_110 $R10/n6_160 $R10/n6_200 $R10/n6_far $R10/n6_litR $R10/n6_near $R10/n6_nearR $R10/n6_right \
  $R10/n6_tiltL $R10/n6_tiltR $R10/d050 $R10/d075 $R10/d110 $R10/d160 $R10/d200 $R10/d300 $R10/d075floor \
  $R10/d075floordn $R10/d075tilt $R10/d110floor
$PY -m calibtool.protocol detect --rig config/rig-phantom21.yaml --out ~/calibtool/c2-work/det/p21 --procs 2 \
  $R21/d050 $R21/d050b $R21/d110
# 1, 3, 4. the analysis (the appendix below is its stdout, verbatim but for heading levels; JSON in analyse.json)
D=~/calibtool/data W=~/calibtool/c2-work
$PY -m calibtool.protocol analyse --json $W/analyse.json \
  --robot p10:config/rig-phantom11.yaml:$W/det/p10:$D/p10/kit/out1280:$D/p10/kit/takes:p10_1106 \
  --robot p21:config/rig-phantom21.yaml:$W/det/p21:$D/p21/kit/out:$D/p21/kit/takes:p21_1105 \
  --extra p21/dleft p21/dtiltleft p10/n6_tiltR > $W/analyse.md
# variant table: read from var-ph{1,2,3,5}/variant.json (fit.verdict, own/common validate + summary)
```

Tests: `bash tools/calibration/calibtool/tests/run_moonstone.sh <worktree> c2 tools/calibration/calibtool/tests/test_protocol.py -q -n 4`
gives 27 passed (26 synthetic, 1 against the stored data).

## Appendix: `protocol analyse` output

### Takes

| robot | take | source | kind | poses | detected | all frames still % | left tags at centre poses (of 35) | distance m (median) | band | bearing deg | tag px | noise L px / unstable % | noise R px / unstable % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| p10 | d050 | raw | upright | 49 | 49 | 100 | 33 | 0.64 | 0.75 | -4.1 | 70.3 | 0.097 / 8.0 | 0.088 / 5.2 |
| p10 | d075 | raw | upright | 49 | 49 | 96 | 34 | 0.79 | 0.75 | -4.2 | 58.5 | 0.105 / 2.8 | 0.098 / 1.4 |
| p10 | d075floor | raw | floor | 28 | 1 | 0 | 4 | 0.97 | 1.10 | 4.3 | 38.0 | - / - | - / - |
| p10 | d075floordn | raw | floor | 21 | 1 | 0 | 6 | 0.90 | 0.75 | -6.5 | 39.3 | - / - | - / - |
| p10 | d075tilt | raw | tilted | 28 | 28 | 100 | 24 | 1.19 | 1.10 | 1.7 | 41.6 | 0.108 / 0.7 | 0.091 / 0.0 |
| p10 | d110 | raw | upright | 49 | 47 | 81 | 33 | 1.03 | 1.10 | -0.2 | 45.4 | 0.126 / 0.8 | 0.125 / 0.0 |
| p10 | d110floor | raw | floor | 28 | 16 | 56 | 11 | 1.31 | 1.10 | 10.6 | 33.8 | 0.128 / 12.5 | 0.110 / 7.0 |
| p10 | d160 | raw | upright | 49 | 41 | 78 | 13 | 1.62 | 1.60 | 12.1 | 28.9 | 0.139 / 3.9 | 0.132 / 4.2 |
| p10 | d200 | raw | upright | 49 | 44 | 70 | 14 | 1.90 | 2.00 | 10.6 | 24.6 | 0.154 / 10.7 | 0.141 / 10.6 |
| p10 | d300 | raw | upright | 49 | 0 | - | 1 | - | - | - | - | - / - | - / - |
| p10 | n6_075 | raw | upright | 49 | 49 | 100 | 35 | 0.83 | 0.75 | 7.7 | 53.4 | 0.120 / 11.7 | 0.113 / 7.9 |
| p10 | n6_110 | raw | upright | 49 | 49 | 98 | 35 | 1.08 | 1.10 | 0.5 | 42.9 | 0.107 / 7.1 | 0.100 / 4.3 |
| p10 | n6_160 | raw | upright | 49 | 45 | 96 | 33 | 1.39 | 1.60 | 8.5 | 33.8 | 0.095 / 5.0 | 0.089 / 1.4 |
| p10 | n6_200 | raw | upright | 49 | 41 | 85 | 23 | 1.67 | 1.60 | 7.6 | 28.3 | 0.086 / 1.2 | 0.078 / 0.4 |
| p10 | n6_far | raw | upright | 49 | 32 | 69 | 14 | 1.91 | 2.00 | 6.5 | 25.8 | 0.100 / 1.7 | 0.104 / 1.3 |
| p10 | n6_litR | raw | upright | 49 | 28 | 79 | 8 | 0.95 | 1.10 | -50.7 | 44.6 | 0.166 / 3.9 | 0.148 / 3.6 |
| p10 | n6_near | raw | upright | 49 | 49 | 96 | 34 | 0.63 | 0.75 | -6.6 | 65.2 | 0.131 / 15.0 | 0.121 / 14.1 |
| p10 | n6_nearR | raw | upright | 49 | 15 | 40 | 4 | 1.04 | 1.10 | -36.8 | 45.0 | 0.171 / 1.9 | 0.141 / 1.0 |
| p10 | n6_right | raw | upright | 49 | 19 | 47 | 3 | 1.24 | 1.10 | -38.4 | 39.7 | 0.164 / 0.0 | 0.150 / 0.0 |
| p10 | n6_tiltL | raw | tilted | 49 | 37 | 95 | 25 | 1.22 | 1.10 | 30.5 | 34.7 | 0.128 / 25.0 | 0.120 / 26.6 |
| p10 | n6_tiltR | raw | tilted | 49 | 34 | 82 | 17 | 1.30 | 1.10 | -24.7 | 33.9 | 0.157 / 1.7 | 0.143 / 1.2 |
| p21 | d050 | raw | upright | 49 | 49 | 100 | 35 | 0.69 | 0.75 | -21.0 | 61.8 | 0.062 / 10.3 | 0.059 / 9.6 |
| p21 | d050b | raw | upright | 49 | 49 | 100 | 35 | 0.66 | 0.75 | -14.1 | 66.8 | 0.054 / 5.8 | 0.049 / 5.0 |
| p21 | d110 | raw | upright | 49 | 49 | 100 | 35 | 1.03 | 1.10 | -9.1 | 44.1 | 0.052 / 2.9 | 0.048 / 3.6 |
| p21 | d160 | corners | upright | 49 | 43 | - | 12 | 1.62 | 1.60 | -9.5 | 28.1 | - / - | - / - |
| p21 | d200 | corners | upright | 49 | 22 | - | 8 | 1.95 | 2.00 | -15.7 | 24.3 | - / - | - / - |
| p21 | dleft | corners | upright | 49 | 49 | - | 28 | 0.42 | 0.50 | -0.2 | 102.3 | - / - | - / - |
| p21 | dnear | corners | upright | 49 | 49 | - | 35 | 0.68 | 0.75 | 0.8 | 65.5 | - / - | - / - |
| p21 | dtiltleft | corners | tilted | 49 | 44 | - | 22 | 1.22 | 1.10 | 6.5 | 35.7 | - / - | - / - |

### Pose usefulness per default distance (upright, |bearing| <= 25 deg, full 49-pose takes; tilted takes apart)

#### 0.5 m: p21/dleft (0.42 m)

| pitch \ yaw | -34 | -23 | -11 | +0 | +11 | +23 | +34 |
|---|---|---|---|---|---|---|---|
| -24 | 1/1 l LB | 1/1 l LB | 1/1 c B | 1/1 c B | 1/1 c RB | 1/1 b B | 1/1 r RB |
| -16 | 1/1 l LB | 1/1 l LB | 1/1 c LB | 1/1 c B | 1/1 c RB | 1/1 r RB | 1/1 r RB |
| -8 | 1/1 l LT | 1/1 l LT | 1/1 c LT | 1/1 c T | 1/1 c RT | 1/1 c RT | 1/1 r RTB |
| +0 | 1/1 l L | 1/1 l L | 1/1 c LT | 1/1 c T | 1/1 c RT | 1/1 r R | 1/1 r R |
| +8 | 1/1 l LT | 1/1 c LT | 1/1 c LT | 1/1 c T | 1/1 c RT | 1/1 r RT | 1/1 r RT |
| +16 | 1/1 l L | 1/1 l L | 1/1 c T | 1/1 c T | 1/1 c RT | 1/1 r RT | 1/1 r RT |
| +24 | 1/1 l LT | 1/1 t LT | 1/1 t LT | 1/1 t T | 1/1 t RT | 1/1 r RT | 1/1 r RT |

Cell = takes detecting the pose / takes; step-66 cells of its camera-0 board (nominal principal point); edge strips (80 px) it puts corners in, either camera.

Proposed list (detected in >= 50% of the takes, or at least once when it fills an edge strip): 49 poses, dropped grid []; keeps 49/49 detected pose-takes, edge-strip corners 657/657, corner-box corners 68/68, rim corners 90/90, cells 5/5; time 49 x 3.1 s = 152 s (all 49: 152 s).

#### 0.75 m: p10/d050 (0.64 m), p10/d075 (0.79 m), p10/n6_075 (0.83 m), p10/n6_near (0.63 m), p21/d050 (0.69 m), p21/d050b (0.66 m), p21/dnear (0.68 m)

| pitch \ yaw | -34 | -23 | -11 | +0 | +11 | +23 | +34 |
|---|---|---|---|---|---|---|---|
| -24 | 7/7 b/bl/l LB | 7/7 b/bl/l LB | 7/7 b/c B | 7/7 b/c B | 7/7 b/br/r RB | 7/7 b/br/r RB | 7/7 b/br/r RB |
| -16 | 7/7 b/bl/l LB | 7/7 b/bl/l LB | 7/7 b/c/l B | 7/7 b/c/r B | 7/7 b/br/c/r RB | 7/7 b/br/r RB | 7/7 b/br/r RB |
| -8 | 7/7 b/l LB | 7/7 b/c/l LB | 7/7 b/c/l B | 7/7 b/c/r B | 7/7 b/c/r RB | 7/7 br/r RB | 7/7 br/r RB |
| +0 | 7/7 c/l L | 7/7 c/l L | 7/7 c/l | 7/7 c/r | 7/7 c/r RT | 7/7 r RTB | 7/7 br/r RB |
| +8 | 7/7 c/l/tl LT | 7/7 c/l LT | 7/7 c/l/t T | 7/7 c/r/t RT | 7/7 c/r RT | 7/7 r/tr RT | 7/7 r RTB |
| +16 | 7/7 l/t/tl LT | 7/7 c/l/t/tl LT | 7/7 c/l/t T | 7/7 c/r/t RT | 7/7 r/t RT | 7/7 r/t/tr RT | 7/7 r/tr RT |
| +24 | 7/7 l/t/tl LT | 7/7 l/t/tl LT | 7/7 c/t LT | 7/7 r/t RT | 7/7 r/t RT | 7/7 r/t/tr RT | 7/7 r/tr RT |

Cell = takes detecting the pose / takes; step-66 cells of its camera-0 board (nominal principal point); edge strips (80 px) it puts corners in, either camera.

Proposed list (detected in >= 50% of the takes, or at least once when it fills an edge strip): 49 poses, dropped grid []; keeps 343/343 detected pose-takes, edge-strip corners 6000/6000, corner-box corners 832/832, rim corners 782/782, cells 9/9; time 49 x 3.1 s = 152 s (all 49: 152 s).

#### 1.1 m: p10/d110 (1.03 m), p10/n6_110 (1.08 m), p21/d110 (1.03 m)

| pitch \ yaw | -34 | -23 | -11 | +0 | +11 | +23 | +34 |
|---|---|---|---|---|---|---|---|
| -24 | 3/3 l L | 3/3 l | 3/3 c/l | 3/3 c | 3/3 b/r | 3/3 r | 3/3 r R |
| -16 | 3/3 l L | 3/3 c/l | 3/3 c | 3/3 c | 3/3 c/r | 3/3 r | 3/3 r R |
| -8 | 3/3 l L | 3/3 l | 3/3 c | 3/3 c | 3/3 c/r | 3/3 r | 3/3 r R |
| +0 | 3/3 l/tl L | 3/3 l T | 3/3 t T | 3/3 c/t T | 3/3 r/t T | 3/3 r RT | 3/3 r/tr RT |
| +8 | 3/3 tl LT | 3/3 t/tl T | 3/3 t T | 3/3 t T | 3/3 r/t T | 3/3 r/t/tr RT | 3/3 tr RT |
| +16 | 3/3 tl LT | 3/3 t/tl T | 3/3 t T | 3/3 t T | 3/3 t T | 3/3 t/tr RT | 3/3 tr RT |
| +24 | 3/3 tl LT | 3/3 t/tl T | 3/3 t T | 3/3 t T | 2/3 t T | 2/3 tr T | 3/3 tr RT |

Cell = takes detecting the pose / takes; step-66 cells of its camera-0 board (nominal principal point); edge strips (80 px) it puts corners in, either camera.

Proposed list (detected in >= 50% of the takes, or at least once when it fills an edge strip): 49 poses, dropped grid []; keeps 145/145 detected pose-takes, edge-strip corners 2795/2795, corner-box corners 232/232, rim corners 171/171, cells 7/7; time 49 x 3.1 s = 152 s (all 49: 152 s).

#### 1.6 m: p10/d160 (1.62 m), p10/n6_160 (1.39 m), p10/n6_200 (1.67 m), p21/d160 (1.62 m)

| pitch \ yaw | -34 | -23 | -11 | +0 | +11 | +23 | +34 |
|---|---|---|---|---|---|---|---|
| -24 | 2/4 l L | 2/4 l | 3/4 b/l | 3/4 b/c | 3/4 b/c | 3/4 br/r B | 3/4 br/r B |
| -16 | 3/4 l L | 3/4 l | 4/4 c/l | 4/4 b/c/l | 4/4 c/r | 4/4 br/c/r | 4/4 br/r |
| -8 | 4/4 l L | 4/4 c/l/tl | 4/4 c/l | 4/4 c/t | 4/4 c/r/t | 4/4 c/r | 4/4 r/tr |
| +0 | 4/4 l/tl L | 4/4 c/l/tl T | 4/4 c/l/t T | 4/4 c/t T | 4/4 c/r/t T | 4/4 c/r/t T | 4/4 r/tr T |
| +8 | 4/4 l/tl LT | 4/4 l/tl T | 4/4 c/l/t T | 4/4 c/l/t T | 4/4 c/r/t T | 4/4 r/t T | 4/4 r/tr T |
| +16 | 4/4 l/tl LT | 4/4 l/t/tl T | 4/4 l/t T | 3/4 t T | 4/4 t T | 3/4 t T | 3/4 r/tr T |
| +24 | 3/4 l/t/tl LT | 3/4 t/tl T | 2/4 t T | 2/4 t T | 2/4 t T | 2/4 t/tr T | 2/4 t/tr T |

Cell = takes detecting the pose / takes; step-66 cells of its camera-0 board (nominal principal point); edge strips (80 px) it puts corners in, either camera.

Proposed list (detected in >= 50% of the takes, or at least once when it fills an edge strip): 49 poses, dropped grid []; keeps 170/170 detected pose-takes, edge-strip corners 2430/2430, corner-box corners 238/238, rim corners 173/173, cells 8/8; time 49 x 3.1 s = 152 s (all 49: 152 s).

#### 2.0 m: p10/d200 (1.90 m), p10/n6_far (1.91 m), p21/d200 (1.95 m)

| pitch \ yaw | -34 | -23 | -11 | +0 | +11 | +23 | +34 |
|---|---|---|---|---|---|---|---|
| -24 | 1/3 l | 1/3 l | 1/3 l | 2/3 c/l | 1/3 c | 2/3 b/r | 2/3 r |
| -16 | 1/3 l | 2/3 l | 3/3 c/l | 3/3 c/r | 3/3 c/r | 3/3 c/r | 2/3 r |
| -8 | 2/3 l/tl | 2/3 l/tl | 2/3 l | 3/3 c/r/t | 3/3 c/r/t | 2/3 c/r | 2/3 r/tr |
| +0 | 3/3 l/tl | 3/3 c/l/tl T | 3/3 c/l/t T | 2/3 t T | 3/3 c/r/t T | 3/3 r/t T | 1/3 r |
| +8 | 3/3 l/t/tl T | 3/3 t/tl T | 2/3 l/t T | 3/3 r/t T | 2/3 t T | 1/3 t | 1/3 r |
| +16 | 2/3 t/tl | 2/3 t/tl | 2/3 t T | 2/3 t T | 1/3 t T | 1/3 t T | 1/3 t T |
| +24 | 2/3 t/tl T | 2/3 t/tl T | 2/3 t T | 2/3 t T | 1/3 t T | 1/3 t T | 1/3 t T |

Cell = takes detecting the pose / takes; step-66 cells of its camera-0 board (nominal principal point); edge strips (80 px) it puts corners in, either camera.

Proposed list (detected in >= 50% of the takes, or at least once when it fills an edge strip): 41 poses, dropped grid [0, 1, 2, 4, 7, 27, 33, 34]; keeps 90/98 detected pose-takes, edge-strip corners 823/823, corner-box corners 11/11, rim corners 1/1, cells 7/7; time 41 x 3.1 s = 127 s (all 49: 152 s).

#### 3.0: no take

#### tilted: p10/n6_tiltL (1.22 m), p10/n6_tiltR (1.30 m), p21/dtiltleft (1.22 m)

| pitch \ yaw | -34 | -23 | -11 | +0 | +11 | +23 | +34 |
|---|---|---|---|---|---|---|---|
| -24 | 0/3 | 0/3 | 0/3 | 1/3 b B | 1/3 b B | 0/3 | 0/3 |
| -16 | 2/3 b B | 2/3 b B | 1/3 b B | 2/3 b/bl B | 2/3 b B | 2/3 b B | 2/3 b B |
| -8 | 2/3 b/bl B | 2/3 b B | 3/3 b/bl B | 3/3 b/bl B | 3/3 b B | 2/3 b B | 2/3 b B |
| +0 | 3/3 b/bl B | 3/3 b/bl/l B | 3/3 b/bl B | 3/3 b/bl B | 3/3 b/br B | 3/3 b/br B | 2/3 b/r B |
| +8 | 3/3 bl/c B | 3/3 b/bl/l B | 3/3 bl/l/r B | 3/3 bl/c/r B | 3/3 br/c/l B | 3/3 br/c/l B | 3/3 br/c/r B |
| +16 | 3/3 bl/c/l B | 3/3 bl/c/l B | 3/3 bl/c/r B | 3/3 c/l/r | 3/3 c/l/r | 3/3 br/c/l B | 3/3 br/c/r B |
| +24 | 3/3 bl/c/l B | 3/3 bl/c/l B | 3/3 l/r | 3/3 c/l/r | 3/3 c/l/r | 3/3 c/l/r | 3/3 br/c/r |

Cell = takes detecting the pose / takes; step-66 cells of its camera-0 board (nominal principal point); edge strips (80 px) it puts corners in, either camera.

Proposed list (detected in >= 50% of the takes, or at least once when it fills an edge strip): 44 poses, dropped grid [0, 1, 2, 5, 6]; keeps 115/115 detected pose-takes, edge-strip corners 2057/2057, corner-box corners 160/160, rim corners 56/56, cells 6/6; time 44 x 3.1 s = 136 s (all 49: 152 s).


### Detection and corner noise vs distance (raw takes, every pose; distance = rough z of the pose's middle still pair, or of its frame with most left tags)

| distance m | poses | detected | detected % | median left tags | tag px (cam0) | noise L px | unstable L % | pooled sd L px |
|---|---|---|---|---|---|---|---|---|
| 0.50-0.75 | 215 | 209 | 97 | 29 | 65.6 | 0.092 | 9.5 | 0.441 |
| 0.75-1.00 | 198 | 160 | 81 | 25 | 45.8 | 0.118 | 6.4 | 0.330 |
| 1.00-1.25 | 246 | 201 | 82 | 24 | 41.9 | 0.119 | 3.1 | 0.246 |
| 1.25-1.50 | 116 | 106 | 91 | 23 | 33.8 | 0.127 | 2.5 | 0.230 |
| 1.50-1.75 | 80 | 69 | 86 | 15 | 28.5 | 0.120 | 3.8 | 0.224 |
| 1.75-2.00 | 55 | 49 | 89 | 16 | 25.7 | 0.104 | 4.7 | 0.264 |
| 2.00-2.25 | 29 | 27 | 93 | 14 | 24.6 | 0.136 | 10.5 | 0.318 |
| 2.50-2.75 | 3 | 0 | 0 | 2 | 17.7 | - | - | - |
| 2.75-3.00 | 4 | 0 | 0 | 2 | 18.8 | - | - | - |
| 3.00-3.25 | 1 | 0 | 0 | 2 | 17.6 | - | - | - |

### Capture timing (raw takes)

- p10/d050: frame 66.67 ms, 5-frame window 266.7 ms, pose period 3.13 s
- p10/d075: frame 66.67 ms, 5-frame window 266.7 ms, pose period 3.13 s
- p10/d075floor: frame 66.67 ms, 5-frame window 266.7 ms, pose period 3.13 s
- p10/d075floordn: frame 66.67 ms, 5-frame window 266.7 ms, pose period 3.13 s
- p10/d075tilt: frame 66.67 ms, 5-frame window 266.7 ms, pose period 3.13 s
- p10/d110: frame 66.67 ms, 5-frame window 266.7 ms, pose period 3.13 s
- p10/d110floor: frame 66.67 ms, 5-frame window 266.7 ms, pose period 3.13 s
- p10/d160: frame 66.67 ms, 5-frame window 266.7 ms, pose period 3.13 s
- p10/d200: frame 66.67 ms, 5-frame window 266.7 ms, pose period 3.13 s
- p10/d300: frame 66.67 ms, 5-frame window 266.7 ms, pose period 3.13 s
- p10/n6_075: frame 66.67 ms, 5-frame window 266.7 ms, pose period 3.13 s
- p10/n6_110: frame 66.67 ms, 5-frame window 266.7 ms, pose period 3.13 s
- p10/n6_160: frame 66.67 ms, 5-frame window 266.7 ms, pose period 3.13 s
- p10/n6_200: frame 66.67 ms, 5-frame window 266.7 ms, pose period 3.13 s
- p10/n6_far: frame 66.67 ms, 5-frame window 266.7 ms, pose period 3.13 s
- p10/n6_litR: frame 66.67 ms, 5-frame window 266.7 ms, pose period 3.13 s
- p10/n6_near: frame 66.67 ms, 5-frame window 266.7 ms, pose period 3.13 s
- p10/n6_nearR: frame 66.67 ms, 5-frame window 266.7 ms, pose period 3.13 s
- p10/n6_right: frame 66.67 ms, 5-frame window 266.7 ms, pose period 3.13 s
- p10/n6_tiltL: frame 66.67 ms, 5-frame window 266.7 ms, pose period 3.13 s
- p10/n6_tiltR: frame 66.67 ms, 5-frame window 266.7 ms, pose period 3.13 s
- p21/d050: frame 66.67 ms, 5-frame window 266.7 ms, pose period 3.13 s
- p21/d050b: frame 66.67 ms, 5-frame window 266.7 ms, pose period 3.13 s
- p21/d110: frame 66.67 ms, 5-frame window 266.7 ms, pose period 3.13 s

### Poses per coverage target: p10_1106 (split16 holdout_every 2; TEST cells on the nominal principal point; swept poses in capture order, every pose of earlier takes counted)

Scenarios: **as recorded**: 398 holds, 539 poses swept over 11 takes (n6_near, n6_075, n6_110, n6_160, n6_200, n6_tiltR, n6_right, n6_tiltL, n6_nearR, n6_litR, n6_far); **proposed pose lists (defaults + tilted reduced, extras whole)**: 393 holds, 526 poses swept over 11 takes (n6_near, n6_075, n6_110, n6_160, n6_200, n6_tiltR, n6_right, n6_tiltL, n6_nearR, n6_litR, n6_far); **defaults + tilted only, all poses**: 302 holds, 343 poses swept over 7 takes (n6_near, n6_075, n6_110, n6_160, n6_200, n6_tiltL, n6_far); **defaults + tilted only, proposed lists**: 297 holds, 330 poses swept over 7 takes (n6_near, n6_075, n6_110, n6_160, n6_200, n6_tiltL, n6_far)

| target | as recorded: met at swept pose, in take (holds) | proposed pose lists (defaults + tilted reduced, extras whole): met at swept pose, in take (holds) | defaults + tilted only, all poses: met at swept pose, in take (holds) | defaults + tilted only, proposed lists: met at swept pose, in take (holds) |
|---|---|---|---|---|
| edge cam0 left | 50 n6_075 (22) | 50 n6_075 (22) | 50 n6_075 (22) | 50 n6_075 (22) |
| edge cam0 right | unmet (16) | unmet (16) | unmet (10) | unmet (10) |
| edge cam0 top | 96 n6_075 (52) | 96 n6_075 (52) | 96 n6_075 (45) | 96 n6_075 (45) |
| edge cam0 bottom | 364 n6_tiltL (39) | 359 n6_tiltL (39) | 272 n6_tiltL (29) | 267 n6_tiltL (29) |
| box cam0 top-left | 1 n6_near (12) | 1 n6_near (12) | 1 n6_near (12) | 1 n6_near (12) |
| box cam0 top-right | unmet (4) | unmet (4) | unmet (4) | unmet (4) |
| box cam0 bottom-left | 50 n6_075 (11) | 50 n6_075 (11) | 50 n6_075 (11) | 50 n6_075 (11) |
| box cam0 bottom-right | unmet (11) | unmet (11) | unmet (6) | unmet (6) |
| third cam0 top-left | unmet (47) | unmet (47) | unmet (47) | unmet (47) |
| third cam0 top-right | unmet (46) | unmet (46) | unmet (33) | unmet (33) |
| third cam0 bottom-left | unmet (43) | unmet (43) | unmet (43) | unmet (43) |
| third cam0 bottom-right | unmet (49) | unmet (49) | unmet (23) | unmet (23) |
| rim cam0 rim | unmet (36) | unmet (36) | unmet (34) | unmet (34) |
| edge cam1 left | 43 n6_near (26) | 43 n6_near (26) | 43 n6_near (26) | 43 n6_near (26) |
| edge cam1 right | unmet (17) | unmet (17) | unmet (10) | unmet (10) |
| edge cam1 top | 98 n6_075 (57) | 98 n6_075 (57) | 98 n6_075 (47) | 98 n6_075 (47) |
| edge cam1 bottom | 364 n6_tiltL (37) | 359 n6_tiltL (37) | unmet (28) | unmet (28) |
| box cam1 top-left | 1 n6_near (13) | 1 n6_near (13) | 1 n6_near (13) | 1 n6_near (13) |
| box cam1 top-right | unmet (6) | unmet (6) | unmet (3) | unmet (3) |
| box cam1 bottom-left | 50 n6_075 (12) | 50 n6_075 (12) | 50 n6_075 (12) | 50 n6_075 (12) |
| box cam1 bottom-right | 452 n6_litR (13) | 447 n6_litR (13) | unmet (6) | unmet (6) |
| third cam1 top-left | 505 n6_far (54) | 495 n6_far (54) | 309 n6_far (54) | 299 n6_far (54) |
| third cam1 top-right | unmet (53) | unmet (53) | unmet (34) | unmet (34) |
| third cam1 bottom-left | unmet (44) | unmet (44) | unmet (44) | unmet (44) |
| third cam1 bottom-right | unmet (48) | unmet (48) | unmet (22) | unmet (22) |
| rim cam1 rim | unmet (47) | unmet (47) | unmet (37) | unmet (37) |
| pairs pairs | 434 n6_nearR (203) | 429 n6_nearR (200) | unmet (154) | unmet (151) |
| far (1.7, 3.0) | 507 n6_far (23) | 503 n6_far (21) | 311 n6_far (23) | 307 n6_far (21) |
| side left | 79 n6_075 (12) | 79 n6_075 (12) | 79 n6_075 (12) | 79 n6_075 (12) |
| side right | 310 n6_right (25) | 310 n6_right (25) | unmet (9) | unmet (9) |
| side top | 337 n6_right (12) | 337 n6_right (12) | unmet (7) | unmet (7) |
| side bottom | unmet (9) | unmet (9) | unmet (7) | unmet (7) |
| judged judged | unmet (5) | unmet (5) | unmet (1) | unmet (1) |
| far_band (1.3, 1.7) | 169 n6_160 (44) | 169 n6_160 (43) | 169 n6_160 (36) | 169 n6_160 (35) |
| far_band (1.7, 2.3) | 492 n6_far (22) | 486 n6_far (21) | 296 n6_far (22) | 290 n6_far (21) |
| entry ('bottom', (0.4, 0.6)) | unmet (7) | unmet (7) | unmet (7) | unmet (7) |
| entry ('bottom', (0.6, 0.9)) | unmet (5) | unmet (5) | unmet (5) | unmet (5) |
| entry ('bottom', (0.9, 1.3)) | unmet (9) | unmet (9) | unmet (3) | unmet (3) |
| entry ('bottom-left', (0.6, 0.9)) | unmet (2) | unmet (2) | unmet (2) | unmet (2) |
| entry ('bottom-left', (0.9, 1.3)) | unmet (9) | unmet (9) | unmet (9) | unmet (9) |
| entry ('bottom-right', (0.4, 0.6)) | unmet (1) | unmet (1) | unmet (1) | unmet (1) |
| entry ('bottom-right', (0.6, 0.9)) | unmet (5) | unmet (5) | unmet (2) | unmet (2) |
| entry ('bottom-right', (0.9, 1.3)) | unmet (6) | unmet (6) | - | - |
| entry ('centre', (0.6, 0.9)) | unmet (3) | unmet (3) | unmet (3) | unmet (3) |
| entry ('centre', (0.9, 1.3)) | 322 n6_right (13) | 322 n6_right (13) | unmet (7) | unmet (7) |
| entry ('left', (0.6, 0.9)) | 79 n6_075 (12) | 79 n6_075 (12) | 79 n6_075 (12) | 79 n6_075 (12) |
| entry ('left', (0.9, 1.3)) | unmet (8) | unmet (8) | unmet (8) | unmet (8) |
| entry ('right', (0.6, 0.9)) | 459 n6_litR (13) | 454 n6_litR (13) | unmet (9) | unmet (9) |
| entry ('right', (0.9, 1.3)) | 310 n6_right (25) | 310 n6_right (25) | unmet (6) | unmet (6) |
| entry ('top', (0.6, 0.9)) | unmet (6) | unmet (6) | unmet (6) | unmet (6) |
| entry ('top', (0.9, 1.3)) | 337 n6_right (12) | 337 n6_right (12) | unmet (7) | unmet (7) |
| entry ('top-left', (0.6, 0.9)) | unmet (2) | unmet (2) | unmet (2) | unmet (2) |
| entry ('top-left', (0.9, 1.3)) | unmet (7) | unmet (7) | unmet (7) | unmet (7) |
| entry ('top-right', (0.6, 0.9)) | unmet (1) | unmet (1) | unmet (1) | unmet (1) |
| entry ('top-right', (0.9, 1.3)) | unmet (4) | unmet (4) | unmet (3) | unmet (3) |

Final FIT tallies (corners; thirds = image-corner thirds TL/TR/BL/BR) and TEST entries with >= 10 TEST holds:

- **as recorded**: FIT pairs 587, far pairs 63; cam0: thirds 4599/3111/3929/2991, rim 275, edges L/R/T/B 222/68/2666/1660, boxes TL/TR/BL/BR 195/21/168/62; cam1: thirds 5251/3443/4421/3839, rim 561, edges L/R/T/B 450/175/3024/1770, boxes TL/TR/BL/BR 361/69/213/107; judged 5: centre 0.9-1.3 (13), left 0.6-0.9 (12), right 0.6-0.9 (13), right 0.9-1.3 (25), top 0.9-1.3 (12); far-band TEST holds 1.3-1.7: 44, 1.7-2.3: 22
- **proposed pose lists (defaults + tilted reduced, extras whole)**: FIT pairs 578, far pairs 57; cam0: thirds 4599/3111/3929/2991, rim 275, edges L/R/T/B 222/68/2666/1660, boxes TL/TR/BL/BR 195/21/168/62; cam1: thirds 5251/3443/4421/3839, rim 561, edges L/R/T/B 450/175/3024/1770, boxes TL/TR/BL/BR 361/69/213/107; judged 5: centre 0.9-1.3 (13), left 0.6-0.9 (12), right 0.6-0.9 (13), right 0.9-1.3 (25), top 0.9-1.3 (12); far-band TEST holds 1.3-1.7: 43, 1.7-2.3: 21
- **defaults + tilted only, all poses**: FIT pairs 457, far pairs 63; cam0: thirds 4599/2599/3929/1642, rim 257, edges L/R/T/B 222/44/2609/1492, boxes TL/TR/BL/BR 195/21/168/39; cam1: thirds 5251/2576/4421/1593, rim 489, edges L/R/T/B 450/50/2891/1413, boxes TL/TR/BL/BR 361/53/213/65; judged 1: left 0.6-0.9 (12); far-band TEST holds 1.3-1.7: 36, 1.7-2.3: 22
- **defaults + tilted only, proposed lists**: FIT pairs 448, far pairs 57; cam0: thirds 4599/2599/3929/1642, rim 257, edges L/R/T/B 222/44/2609/1492, boxes TL/TR/BL/BR 195/21/168/39; cam1: thirds 5251/2576/4421/1593, rim 489, edges L/R/T/B 450/50/2891/1413, boxes TL/TR/BL/BR 361/53/213/65; judged 1: left 0.6-0.9 (12); far-band TEST holds 1.3-1.7: 35, 1.7-2.3: 21

### Poses per coverage target: p21_1105 (split16 holdout_every 2; TEST cells on the nominal principal point; swept poses in capture order, every pose of earlier takes counted)

Scenarios: **as recorded**: 354 holds, 392 poses swept over 8 takes (d050, d050b, d110, d160, d200, dnear, dleft, dtiltleft); **proposed pose lists (defaults + tilted reduced, extras whole)**: 354 holds, 384 poses swept over 8 takes (d050, d050b, d110, d160, d200, dnear, dleft, dtiltleft); **defaults + tilted only, all poses**: 261 holds, 294 poses swept over 6 takes (d050, d050b, d110, d160, d200, dnear); **defaults + tilted only, proposed lists**: 261 holds, 286 poses swept over 6 takes (d050, d050b, d110, d160, d200, dnear)

| target | as recorded: met at swept pose, in take (holds) | proposed pose lists (defaults + tilted reduced, extras whole): met at swept pose, in take (holds) | defaults + tilted only, all poses: met at swept pose, in take (holds) | defaults + tilted only, proposed lists: met at swept pose, in take (holds) |
|---|---|---|---|---|
| edge cam0 left | unmet (18) | unmet (18) | unmet (8) | unmet (8) |
| edge cam0 right | 1 d050 (46) | 1 d050 (46) | 1 d050 (35) | 1 d050 (35) |
| edge cam0 top | 131 d110 (52) | 131 d110 (52) | 131 d110 (37) | 131 d110 (37) |
| edge cam0 bottom | 1 d050 (49) | 1 d050 (49) | 1 d050 (32) | 1 d050 (32) |
| box cam0 top-left | 288 dnear (12) | 280 dnear (12) | 288 dnear (6) | 280 dnear (6) |
| box cam0 top-right | 1 d050 (14) | 1 d050 (14) | 1 d050 (8) | 1 d050 (8) |
| box cam0 bottom-left | unmet (6) | unmet (6) | unmet (5) | unmet (5) |
| box cam0 bottom-right | 1 d050 (17) | 1 d050 (17) | 1 d050 (16) | 1 d050 (16) |
| third cam0 top-left | unmet (54) | unmet (54) | unmet (32) | unmet (32) |
| third cam0 top-right | unmet (63) | unmet (63) | unmet (45) | unmet (45) |
| third cam0 bottom-left | unmet (45) | unmet (45) | unmet (23) | unmet (23) |
| third cam0 bottom-right | unmet (71) | unmet (71) | unmet (52) | unmet (52) |
| rim cam0 rim | unmet (58) | unmet (58) | unmet (40) | unmet (40) |
| edge cam1 left | unmet (20) | unmet (20) | unmet (8) | unmet (8) |
| edge cam1 right | 1 d050 (44) | 1 d050 (44) | 1 d050 (33) | 1 d050 (33) |
| edge cam1 top | 131 d110 (53) | 131 d110 (53) | 131 d110 (38) | 131 d110 (38) |
| edge cam1 bottom | 1 d050 (50) | 1 d050 (50) | 1 d050 (32) | 1 d050 (32) |
| box cam1 top-left | 92 d050b (14) | 92 d050b (14) | 92 d050b (7) | 92 d050b (7) |
| box cam1 top-right | 1 d050 (18) | 1 d050 (18) | 1 d050 (10) | 1 d050 (10) |
| box cam1 bottom-left | unmet (8) | unmet (8) | unmet (4) | unmet (4) |
| box cam1 bottom-right | 1 d050 (22) | 1 d050 (22) | 1 d050 (17) | 1 d050 (17) |
| third cam1 top-left | unmet (56) | unmet (56) | unmet (34) | unmet (34) |
| third cam1 top-right | unmet (63) | unmet (63) | unmet (45) | unmet (45) |
| third cam1 bottom-left | unmet (47) | unmet (47) | unmet (25) | unmet (25) |
| third cam1 bottom-right | unmet (66) | unmet (66) | unmet (50) | unmet (50) |
| rim cam1 rim | unmet (61) | unmet (61) | unmet (39) | unmet (39) |
| pairs pairs | 383 dtiltleft (180) | 375 dtiltleft (180) | unmet (133) | unmet (133) |
| far (1.7, 3.0) | unmet (20) | unmet (20) | unmet (20) | unmet (20) |
| side left | unmet (9) | unmet (9) | unmet (7) | unmet (7) |
| side right | 83 d050b (21) | 83 d050b (21) | 83 d050b (21) | 83 d050b (21) |
| side top | unmet (8) | unmet (8) | unmet (8) | unmet (8) |
| side bottom | 65 d050b (16) | 65 d050b (16) | 65 d050b (16) | 65 d050b (16) |
| judged judged | unmet (5) | unmet (5) | unmet (3) | unmet (3) |
| far_band (1.3, 1.7) | 180 d160 (21) | 180 d160 (21) | 180 d160 (17) | 180 d160 (17) |
| far_band (1.7, 2.3) | 208 d200 (19) | 204 d200 (19) | 208 d200 (19) | 204 d200 (19) |
| entry ('bottom', (0.4, 0.6)) | unmet (1) | unmet (1) | - | - |
| entry ('bottom', (0.6, 0.9)) | 65 d050b (16) | 65 d050b (16) | 65 d050b (16) | 65 d050b (16) |
| entry ('bottom', (0.9, 1.3)) | 368 dtiltleft (10) | 360 dtiltleft (10) | - | - |
| entry ('bottom-left', (0.6, 0.9)) | unmet (1) | unmet (1) | unmet (1) | unmet (1) |
| entry ('bottom-left', (0.9, 1.3)) | unmet (2) | unmet (2) | - | - |
| entry ('bottom-right', (0.4, 0.6)) | unmet (4) | unmet (4) | unmet (4) | unmet (4) |
| entry ('bottom-right', (0.6, 0.9)) | unmet (5) | unmet (5) | unmet (5) | unmet (5) |
| entry ('centre', (0.4, 0.6)) | 326 dleft (12) | 318 dleft (12) | - | - |
| entry ('centre', (0.6, 0.9)) | 269 dnear (12) | 261 dnear (12) | 269 dnear (12) | 261 dnear (12) |
| entry ('centre', (0.9, 1.3)) | unmet (5) | unmet (5) | unmet (3) | unmet (3) |
| entry ('left', (0.4, 0.6)) | unmet (7) | unmet (7) | - | - |
| entry ('left', (0.6, 0.9)) | unmet (7) | unmet (7) | unmet (7) | unmet (7) |
| entry ('left', (0.9, 1.3)) | unmet (9) | unmet (9) | unmet (4) | unmet (4) |
| entry ('right', (0.4, 0.6)) | unmet (9) | unmet (9) | unmet (4) | unmet (4) |
| entry ('right', (0.6, 0.9)) | 83 d050b (21) | 83 d050b (21) | 83 d050b (21) | 83 d050b (21) |
| entry ('right', (0.9, 1.3)) | unmet (7) | unmet (7) | unmet (5) | unmet (5) |
| entry ('top', (0.4, 0.6)) | unmet (2) | unmet (2) | - | - |
| entry ('top', (0.6, 0.9)) | unmet (8) | unmet (8) | unmet (8) | unmet (8) |
| entry ('top', (0.9, 1.3)) | unmet (7) | unmet (7) | unmet (7) | unmet (7) |
| entry ('top-left', (0.6, 0.9)) | unmet (1) | unmet (1) | unmet (1) | unmet (1) |
| entry ('top-left', (0.9, 1.3)) | unmet (1) | unmet (1) | unmet (1) | unmet (1) |
| entry ('top-right', (0.6, 0.9)) | unmet (1) | unmet (1) | unmet (1) | unmet (1) |
| entry ('top-right', (0.9, 1.3)) | unmet (2) | unmet (2) | unmet (2) | unmet (2) |

Final FIT tallies (corners; thirds = image-corner thirds TL/TR/BL/BR) and TEST entries with >= 10 TEST holds:

- **as recorded**: FIT pairs 512, far pairs 43; cam0: thirds 1872/2997/2227/4660, rim 459, edges L/R/T/B 93/535/1507/1843, boxes TL/TR/BL/BR 51/88/26/287; cam1: thirds 2074/2865/2881/4856, rim 534, edges L/R/T/B 151/555/1547/2202, boxes TL/TR/BL/BR 89/115/78/376; judged 5: bottom 0.6-0.9 (16), bottom 0.9-1.3 (10), centre 0.4-0.6 (12), centre 0.6-0.9 (12), right 0.6-0.9 (21); far-band TEST holds 1.3-1.7: 21, 1.7-2.3: 19
- **proposed pose lists (defaults + tilted reduced, extras whole)**: FIT pairs 512, far pairs 43; cam0: thirds 1872/2997/2227/4660, rim 459, edges L/R/T/B 93/535/1507/1843, boxes TL/TR/BL/BR 51/88/26/287; cam1: thirds 2074/2865/2881/4856, rim 534, edges L/R/T/B 151/555/1547/2202, boxes TL/TR/BL/BR 89/115/78/376; judged 5: bottom 0.6-0.9 (16), bottom 0.9-1.3 (10), centre 0.4-0.6 (12), centre 0.6-0.9 (12), right 0.6-0.9 (21); far-band TEST holds 1.3-1.7: 21, 1.7-2.3: 19
- **defaults + tilted only, all poses**: FIT pairs 383, far pairs 43; cam0: thirds 1512/2653/1415/4065, rim 402, edges L/R/T/B 58/492/1289/1332, boxes TL/TR/BL/BR 40/71/24/285; cam1: thirds 1695/2561/1685/4219, rim 467, edges L/R/T/B 106/490/1332/1452, boxes TL/TR/BL/BR 71/85/62/362; judged 3: bottom 0.6-0.9 (16), centre 0.6-0.9 (12), right 0.6-0.9 (21); far-band TEST holds 1.3-1.7: 17, 1.7-2.3: 19
- **defaults + tilted only, proposed lists**: FIT pairs 383, far pairs 43; cam0: thirds 1512/2653/1415/4065, rim 402, edges L/R/T/B 58/492/1289/1332, boxes TL/TR/BL/BR 40/71/24/285; cam1: thirds 1695/2561/1685/4219, rim 467, edges L/R/T/B 106/490/1332/1452, boxes TL/TR/BL/BR 71/85/62/362; judged 3: bottom 0.6-0.9 (16), centre 0.6-0.9 (12), right 0.6-0.9 (21); far-band TEST holds 1.3-1.7: 17, 1.7-2.3: 19
