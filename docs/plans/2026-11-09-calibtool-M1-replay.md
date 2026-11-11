# calibtool M1 — replay engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:test-driven-development for every task (test first, seen
> failing, then code). Executed with superpowers:subagent-driven-development: Phase A by one agent, Phase B by three
> agents in parallel (separate worktrees, disjoint files), Phase C after merging. Steps use `- [ ]`.

**Goal:** `calibtool` replays a stored neck-grid calibration session as if it were live: it feeds holds in capture
order, assigns FIT/TEST at capture time, tracks both coverage target sets, writes comparative gap sentences, predicts
and plans board placements, scores the new fit and the live config, and decides deploy / keep live — and on the two
stored sessions it reproduces the verdicts reached by hand.

**Architecture:** pure-Python modules in `tools/calibration/calibtool/`, calling the existing pipeline (`16`, `45`,
`66`, `lensscore`, `stereofit`, `lenscells`, `neckfk`, `okviscfg`) through imports, never rewriting them. Stored
sessions are described in one registry (`sessions.py`). Slow replays are pytest tests marked `replay` and need
`CALIBTOOL_DATA`.

**Tech stack:** Python 3.12, NumPy, SciPy, OpenCV (aruco), PyYAML; pytest + xdist. Tests run on Moonstone only:
`tools/calibration/calibtool/tests/run_moonstone.sh <worktree> <name> [pytest args]` (sets one BLAS thread per worker
and `CALIBTOOL_DATA=~/calibtool/data`). Never run a suite on the laptop.

Design: `docs/superpowers/plans/2026-11-09-calibration-terminal-tool.md` (sections 4, 5, 6, 9.2, 13, 13.1).

---

## Facts every task relies on (measured 9 Nov 2026; cite, do not re-derive)

- **Stored data** (Moonstone `~/calibtool/data`): `p10/kit` = phantom10 (ex-phantom11) 6 Nov session; per-take corner
  files `out1280/n6_*.npz` (11 files, sorted; `out1280/rejected/` excluded); raw takes `takes/n6_*/pose_NNN.npz`; recorded
  run `fit1280/final1280.log`, per-entry table `fit1280/cells.txt`. `p21/kit` = phantom21 5 Nov; corner files
  `out/<take>.npz`; recorded run `final/final21.log` + `final/validate-{kb4t,kb4,live}/`; raw takes may still be copying.
  Robot OKVIS configs in `p{10,21}/etc-okvis/`.
- **Corner files = 15b format** (`scripts/15b_dump_neck_take.py:262-300`): per pair `img_hold` (pose index within the
  take), `img_t0`/`img_t1` ns, `img_yaw`/`img_pitch` **rad** (measured neck), `img_ay`/`img_ap` in {-1,0,1}; per corner
  `det_img` (pair index), `det_cam` (0 = left = OKVIS cam0), `det_tag`, `det_k`, `det_u`, `det_v`. Up to 3 pairs per hold.
- **Raw pose files** (`pose_NNN.npz`): `cmd` (2,) deg [pitch, yaw] commanded, `neck` (2,) deg measured, `tl`/`tr` (5,) ns,
  `L`/`R` (5,800,1280) u8. Stored takes have **no** `queues` key.
- **Step 16** (`16_split_distance.py:94-106`): `split(files, board, focal_px, resolution, max_fit_z, holdout_every)`;
  hold ids offset per file by `max(img_hold)+1` **in the given file order**; output drops `img_yaw/img_pitch/img_ay/img_ap`
  and adds `img_z`; TEST = z > max_fit_z plus every N-th near hold in time order. Join split pairs back to takes by
  `img_t0`.
- **Recorded runs used** `16 --max-fit-z 3.0 --holdout-every 2` and `45 --max-z 3.0`. p21 take order (matters):
  `d050 d050b d110 d160 d200 dnear dleft dtiltleft`. p10: sorted `n6_*`.
- **Verdicts to reproduce** (quoted from the logs):
  - p21 new KB4T: "converged: rms 0.367 px over 503 pairs, baseline 43.92 mm"; "66 kb4t PASS | all 7 validated entries";
    entries left 0.9-1.3 +0.11, centre 0.4-0.6 +0.03, centre 0.6-0.9 -0.03, right 0.4-0.6 +0.14, right 0.6-0.9 +0.10,
    bottom 0.6-0.9 -0.32, bottom 0.9-1.3 -0.17 (%). Live (`etc-okvis/mk12_stereo_holevisor-radtan-pre-daksh.yaml`,
    radtan): "66 live FAIL | 2 of 6 validated entries are outside the limits: right 0.6-0.9 m +0.54 %; bottom 0.9-1.3 m
    +3.99 %". Summary: new "+0.04 % (IQR -0.19 .. +0.28) | rows 0.207 px"; live "+0.77 % (IQR +0.07 .. +1.77) | rows 0.820 px".
  - p10 new: "converged: rms 0.417 px over 578 pairs, baseline 44.06 mm"; "PASS: all 9 validated entries". Live
    (`etc-okvis/phantom11_stereo_kb4t_20261105_1280_rscfix.yaml`, KB4T at 1280): "FAIL: 7 of 7". Summary: new "-0.04 %
    (IQR -0.45 .. +0.33) | rows 0.279 px"; live "+0.06 % (IQR -2.49 .. +1.59) | rows 3.967 px".
- **Step 66 / lensscore:** `lensscore.score(model, intr, C, data)`; `cell_dist_table(rec, pp, min_holds, seed=20261020)`
  (bootstrap order-dependent); `verdict(table, gates)`; cells per `lenscells` (cam0 mean corner vs the *scored* principal
  point), `DIST_BINS` [lo,hi) 0.4-0.6/0.6-0.9/0.9-1.3; `report.json` keys `table`, `validated_cells`, `verdict`, ...;
  **exit code 0 even on FAIL: parse the verdict.** Timings: 45 took 274 s (p21) / 510 s (p10); 66 42-62 s new, 122-156 s live.
- **Live config loading:** `run_lens_kb4t.okvis_calibration` infers the model from the coefficient count and silently
  calls radtan "KB4" — never use it for radtan. p21's kit scored radtan through `kit/radtan_shim.py`, which patches
  `lensmodel` process-wide and hard-codes `/root/p21-cal` paths.
- **Coverage (COLLECT-lens-data.md:92-94, kit `cov*.py`):** image thirds (3x3) of cam0 corners, rim = r / half-diagonal
  > 0.8 from (W/2, H/2), far pairs 1.7-3 m; targets cell >= ~5,000, rim >= ~1,500, >= 50 far pairs, >= 6 judged entries.
- **Kinematics:** `neckfk.NeckChain(urdf)`, `T_TB(q_yaw, q_pitch)` rad (Head->Torso); URDF
  `config/foundation_rig/20250206_phantom_mk1.urdf`; no camera link: camera-in-Head X = T_BS . T_SC0 (OKVIS config).
- **Gotchas:** numeric script names need `sys.path.insert(<scripts>)` + `importlib.import_module("16_split_distance")`;
  `rigcfg.load` prints to stdout; 15b uses a fork-only pool with a module global; `summary_row` bands are (lo,hi], step-66
  bins [lo,hi).

## File ownership

| Phase | Agent | Creates (in `tools/calibration/calibtool/`) |
|---|---|---|
| A | A | `paths.py`, `sessions.py`, `session.py`, `pipeline.py`, `tests/conftest.py`, `tests/test_session.py`, `tests/test_reproduce.py` |
| B | B1 | `assign.py`, `tests/test_assign.py` |
| B | B2 | `coverage.py`, `placement.py`, `planner.py`, `tests/test_coverage.py`, `tests/test_placement.py`, `tests/test_planner.py` |
| B | B3 | `livecfg.py`, `scaling.py`, `decide.py`, `tests/test_livecfg.py`, `tests/test_scaling.py`, `tests/test_decide.py` |
| C | lead + C | `replay.py`, `protocol.py`, `tests/test_replay.py`, `results/2026-11-calibtool-M1.md` |

Nobody edits another agent's files or `DEPS.txt` (agents list new dependencies in their report; the lead edits).

## Phase A — data model and reproduction harness (one agent)

### Task A1: paths + session registry
- [ ] Test (`test_session.py`): `sessions.get("p21_1105")` and `("p10_1106")` return a `StoredSession` with `robot`,
  `rig_path` (`config/rig-phantom21.yaml` / `rig-phantom11.yaml`), `take_order` (exact lists above), `corner_files`
  (resolved under `CALIBTOOL_DATA`), `raw_dirs`, `live_config`, `live_model` ("RADTAN"/"KB4T"), `recorded` (the quoted
  numbers above as data). Unknown name -> `KeyError`. Tests needing data are skipped when `CALIBTOOL_DATA` is unset.
- [ ] Implement `paths.py` (`data_root()`, scripts dir on `sys.path`, `import_script(name)` via importlib) and
  `sessions.py`.

### Task A2: Session data model
- [ ] Tests: `Session.load(stored)` keeps take order; global hold id = per-take offset in take order (same rule as 16);
  `pairs` table has take, hold, pose, t0, t1, neck_pitch_rad, neck_yaw_rad, n_cam0, n_cam1; `holds()` yields holds in
  capture order; `to_npz_dict(mask)` round-trips to a dict 16/`stereofit.Data` accept; joining 16's split output back by
  `img_t0` recovers neck angles for every TEST pair (p10 and p21).
- [ ] Implement `session.py`.

### Task A3: pipeline wrappers + reproduction (the M1 baseline)
- [ ] `pipeline.py`: `split16(session, out_dir, max_fit_z=3.0, holdout_every=2)`, `fit45(fit_npz, rig, out_dir, max_z=3.0)`,
  `validate66(calib_json, test_npz, out_dir)` -> parsed `report.json` + verdict text; `summary(calib_json, test_npz, rig)`
  -> depth median/IQR, rows (same math as `run_lens_kb4t.summary_row`). Subprocess or import, but verdicts parsed from
  text/json, never exit codes.
- [ ] `test_reproduce.py` (marker `replay`): for p10 and p21, split16 reproduces the recorded FIT/TEST pair counts
  (p21 512/499, p10 587/572); fit45 reproduces rms within 0.005 px and baseline within 0.01 mm; validate66 new = PASS with
  the recorded entries (per-entry median within 0.02 %); p10 live FAIL 7 of 7. p21 live: score the radtan config with a
  path-patched **temporary** copy of the kit's `radtan_shim.py`/`run66.py` (not committed) to establish the reference
  numbers for B3; record them in the test as constants with their source.
- [ ] Register the `replay` marker in `tests/conftest.py`; default runs exclude it (`-m "not replay"`).

## Phase B — three agents in parallel (worktrees off the Phase A commit)

### B1: capture-time FIT/TEST assignment (`assign.py`)
Contract: `Assigner(rig, pp_nominal, focal_nominal, max_fit_z=3.0, target_test=10, min_placements=2, holdout_every=2)`;
`.observe(hold) -> "FIT" | "TEST"` called once per hold in capture order (hold = pairs + corners + take/placement id);
`.entry_counts()`; `.write_split(session, out_dir)` writes `fit.npz`/`test.npz` in 16's format (incl. `img_z`).
- [ ] Tests (synthetic holds first, then stored sessions):
  1. pairs beyond `max_fit_z` are TEST (as 16);
  2. within each (lenscells cell, DIST_BIN) entry the TEST fraction is 1/holdout_every within +-1 hold;
  3. once >= 2 placements fed an entry, its TEST holds come from >= 2 placements; an entry fed by one placement holds
     back at most ceil(target_test/2) TEST holds until a second placement arrives;
  4. deterministic for the same input order;
  5. output files load in `stereofit.Data` and pass through `pipeline.fit45`/`validate66`;
  6. (`replay`) on p10 and p21 the number of step-66 entries with >= 8 TEST holds is >= that of the 16 split, and the
     new fit still validates PASS.

### B2: coverage, placement model, planner (`coverage.py`, `placement.py`, `planner.py`)
- [ ] `coverage.py` tests: FIT tally per camera = image thirds, rim, far pairs (identical numbers to the kit `cov*.py`
  definition on p10 `out1280` and p21 `out`), plus edge strips (corners within 80 px of each image edge) and 160 px
  corner boxes (the kit's `edge21.py` definition; over all 8 p21 takes cam0 reads left 176 vs right 1,099, bottom-left 50
  vs bottom-right 588, handoff section 000000000); TEST tally per (cell, bin) from an assignment map; far bands per band;
  `deficits(targets)` ordered edges first; `sentences(deficits)` comparative and directional:
  *"Left camera, left edge: 176 corners vs 1,099 on the right edge (6x fewer)."* An image-left deficit asks for the board
  toward the **robot's left = the operator's right** (camera looks forward; image x grows to the robot's right); test
  each edge's wording; no sentence says "robot's right" without "your left".
- [ ] `placement.py` tests: `BoardModel.from_first_pose(hold, rig, lens, X, chain)`; `.predict(neck_pitch, neck_yaw)` ->
  predicted cam0 board centroid + visible-corner fraction. On every stored take: per-pose predicted vs measured centroid
  error reported (median and p90); the test asserts median < 60 px (bound to be tightened in C from the measured value).
  X from the session's OKVIS config (T_BS . T_SC0); chain from the repo URDF.
- [ ] `planner.py` tests: candidates = distance {0.5, 0.75, 1.1, 1.6, 2.0} m x bearing {-40, -20, 0, 20, 40} deg (+ the
  tilted-low variant at 0.75 m), board facing the head; for a candidate, simulate the 49-pose grid (pitch -24..24 step 8,
  yaw -34..34 in 7 steps) with the placement model -> predicted entries / regions; `suggest(deficits, k=3)` greedy by
  weighted deficit reduction (edges weighted up). **Counterfactual (`replay`):** p21 without `dleft` and `dtiltleft`:
  the top suggestion targets the left edge with a bearing toward the robot's left, and a tilted/low placement appears
  for the bottom/bottom-left deficit. Also `pose_subset(candidate, deficit)` returns the poses predicted to fill it.

### B3: live config, scaling, decision (`livecfg.py`, `scaling.py`, `decide.py`)
- [ ] `livecfg.py` tests: `load(okvis_yaml) -> LiveCalib(model, intr, C, size)` reads `distortion_type` explicitly
  (radtan -> "RADTAN", equidistant -> "KB4", kb4t -> "KB4T"); never infers from the coefficient count; `C` =
  inv(T_SC0) @ T_SC1. `radtan_scope()` context manager registers a RADTAN model in `lensmodel` only while active and
  restores it after (test: `lensmodel.MODELS` unchanged after exit, even on exception); RADTAN projection equals
  `cv2.projectPoints` within 1e-9 px. (`replay`) p21 live scored through `lensscore` inside `radtan_scope()` reproduces
  Phase A's reference numbers (per entry within 0.02 %).
- [ ] `scaling.py` tests: `scale_config(text, s)` and `unscale_config(text, s)`: focal x s, principal point
  (c + 0.5) x s - 0.5, `image_dimension` exact ints, `resize_factor` such that `int(dim x rf)` (float32 and float64)
  gives exactly 640x400 for 1280->960, 1280->640, 1280->1280; nothing else in the file changes (byte diff limited to those
  keys); unscale(scale(x)) == x within 1e-9; a 960 config scaled to 1280 scores on 1280 data.
- [ ] `decide.py` tests: per-TEST-hold depth error from `lensscore` records for new and live; median |error| = median of
  absolute values; band table [lo,hi) 0.4-0.6/0.6-0.9/0.9-1.3/1.3-1.7/1.7-2.3; paired new-live difference with a 95 %
  bootstrap over holds clustered by placement (seeded); `verdict(new, live, step66_new, gates)` -> `DEPLOY` /
  `KEEP_LIVE` / `NOT_ENOUGH_DATA` / `LIVE_NOT_SCORABLE_DEPLOY` with reasons. Synthetic cases: identical -> KEEP_LIVE;
  new better everywhere -> DEPLOY; new worse in one band by more than its CI -> KEEP_LIVE; fewer than 6 judged entries ->
  NOT_ENOUGH_DATA; with `require_all_edges=True`, an image edge (left/right/top/bottom) with no judged entry ->
  NOT_ENOUGH_DATA. (`replay`) both stored sessions: new better than live on every paired statistic; verdict DEPLOY with
  `require_all_edges=False`; with it True, report the verdict (p21's 7 judged entries include **no top entry**, so it is
  expected NOT_ENOUGH_DATA: the strictness is decided with the user in C3 from these numbers).

## Phase C — integration, protocol measurements, review

### C1: replay orchestrator (`replay.py`)
- [ ] `run(stored, assignment="capture"|"split16", out_dir)`: feeds holds in capture order through Assigner + coverage
  (snapshot and gap sentences after each placement), planner suggestions after the defaults, fit45, validate66 new +
  live, decide; writes `report.json` + `report.md`. Tests (`replay`): both sessions -> new better than live, DEPLOY with
  `require_all_edges=False`; p21's gap sentences name the left edge and bottom-left, and computed over all 8 takes they
  carry the handoff's numbers (cam0 corners within 80 px of the left/right edge 176 vs 1,099; bottom-left vs bottom-right
  160 px boxes 50 vs 588).

### C2: collection protocol (`protocol.py`, plan 13.1) on p10 raw takes (p21 when copied)
- [ ] Per default distance: which poses detect the board (>= 6 tags both cameras) and what they add; proposed pose list.
- [ ] Frames per pose: re-dump with `per_hold` 1..5 (15b on raw), fit + validate each; TEST median/IQR and judged entries
  vs per_hold; the smallest per_hold with no measurable loss.
- [ ] Far limit: detection rate and corner noise vs distance.
- [ ] Poses per target: poses needed per FIT/TEST target; expected session length.
- [ ] Write `results/2026-11-calibtool-M1.md` with every number and its command.

### C3: thresholds and review
- [ ] Fix the section-6 thresholds of the design plan from C1/C2's numbers, including whether every image edge must have
  a judged entry (p21 5 Nov had no top entry); show them to the user.
- [ ] Independent review subagent of all M1 code and results before reporting to the user.

### C3 decisions (9 Nov, approved by the user)
1. **Edges:** a deploy needs judged entries on the left, right and bottom image edges (a corner cell counts for both,
   `coverage.edges_of_cell`); the top is asked for (deficit + sentence) but blocks neither the deploy nor the stop rule,
   and the verdict says when it is not judged. `Gates.required_edges` (default {left, right, bottom}) replaces
   `require_all_edges`; all four can still be required. Replay: p21 and p10 DEPLOY; with all four p21 is
   NOT_ENOUGH_DATA (no top entry).
2. **Targets:** each is the lower of what p21 5 Nov and p10 6 Nov reached as `replay.Collection` counts it
   (capture-time assignment, labels as `finish()` leaves them), pair/corner counts x 2/3 for per_hold 2:
   `coverage.REACHED_*` and `Targets.for_per_hold`. Per_hold 3 / 2: FIT pairs 476 / 317, each corner third 3,665 /
   2,443, rim 486 / 324, far pairs 79 / 52, bottom edge strip 3,004 / 2,002 (new); hold targets 6 judged entries,
   9 TEST holds on each required side, 21 per far band. The comparative edge/box balance, single entries and the top
   side are asked for, not blocking. Stop rule (`Coverage.stop_met`) checked after every hold
   (`Collection.stop`); replay check: first met at hold 352/354 (p21, `dtiltleft` pose 46) and 387/398 (p10,
   `n6_far` pose 20), i.e. 2 and 28 poses before the end (about 6 s and 1.5 min at 3.13 s/pose); the data up to there
   still PASSes and DEPLOYs (`test_replay_cut_at_the_stop_rule_still_passes`). With step 16's split the rule is not
   met on either session.
3. **Planner:** one suggestion per top deficit (`Planner.suggest`); the counterfactual's strict xfail now passes and
   is a normal test (2nd suggestion: the tilted board, 0.75 m straight ahead, for the bottom edge strip).
4. **Gates:** band_min_placements 3; `allow_deploy_without_live` False by default; deploy needs new better on depth
   and rows (already pinned by `test_new_not_better_on_rows_keeps_live` and
   `test_depth_is_judged_on_the_intervals_upper_end`).
5. **Protocol constants for M2** in `protocol.py`: `PER_HOLD` 2, `CAPTURE_FRAMES` 3, `FAR_ASK_M` 1.8,
   `FAR_ACCEPT_M` (1.7, 2.0), `FAR_REFUSE_BEYOND_M` 2.1, `POSES_UPRIGHT`/`POSES_FAR`/`POSES_TILTED` 49/41/44,
   `HOLD_ENDS_AT_CAPTURE`; the planner's far candidate stands at 1.8 m.
6. **Robot identity:** `calibtool/config/robots.yaml` (P10, P11, P12) and `robots.identify(hostname, machine_id)`.

## Acceptance (M1 done)

Phase A reproduction green; B1-B3 tests green; C1 both sessions DEPLOY with the expected gap sentences; counterfactual
green; C2 report written; `devel_compat.sh` green; full calibration suite green on Moonstone; review addressed.
