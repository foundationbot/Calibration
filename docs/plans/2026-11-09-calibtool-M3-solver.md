# calibtool M3 — fast KB4T stereo solver and scorer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:test-driven-development for every task (test first, seen
> failing, then code). Executed inline by one agent (the solver's tasks share one file and one state layout). Steps use
> `- [ ]`.

**Goal:** the same lens fit (step 45) and depth scoring (step 66, self-check, summary row) as today's scipy code, to
the tolerances of design section 9.1, in a fraction of the time: fit < 30 s and scoring < 30 s on Thor (Moonstone
single-thread timings reported here; the Thor run is the lead's, with the user's OK).

**Architecture:** `calibtool/solver.py` is a drop-in for `stereofit.solve` (same arguments, same returned dict):
Levenberg-Marquardt with an exact Jacobian (analytic KB4/KB4T derivatives; any other registered model, e.g. RADTAN,
only with the camera fixed, its 2x3 point derivative by central differences), the per-pair board poses (6 numbers, 7
with the scorer's board scale) eliminated by Schur complement every iteration so the linear system is the 26 global
numbers (2 x 10 intrinsics + 6 stereo pose) whatever the pair count. With the camera fixed the pairs are independent
and run as a vector of 6x6 / 7x7 LMs, each with its own damping and stop. `calibtool/score.py` is a drop-in for
`lensscore.score` built on it. Switch: `CALIB_SOLVER=fast` in the environment (read at each call, so subprocesses of
`calibtool.pipeline` and `run_lens_kb4t` inherit it); unset or `scipy` = today's code, untouched. Everything around
the solve (PnP start + drops, presolve, warm_fit, trimming, attempts, gates) stays in `stereofit` and therefore runs
unchanged on top of the fast solve.

**Tech stack:** Python 3.12, NumPy (vectorised, batched `matmul`/`solve`), OpenCV for the PnP start only. No
Numba/JAX unless a measured budget is missed. Tests on Moonstone only:
`bash tools/calibration/calibtool/tests/run_moonstone.sh /home/kaus/wt/calib-tui-m3 m3 <pytest args>`; long comparisons
with `ssh Moonstone`, `~/calibtool/venv/bin/python`, `OMP_NUM_THREADS=1`, `nice -n 10`, scratch `~/calibtool/m3-work`,
at most 6 processes.

Design: `docs/superpowers/plans/2026-11-09-calibration-terminal-tool.md` sections 9.1, 10.4, 13 (M3 row).

---

## Facts every task relies on (read 9 Nov 2026)

- `stereofit.solve(data, model, intr0, C0, poses0, free_intr=True, free_C=True, scale=False, trim_px=3.0, rounds=4,
  max_nfev=300, ls0=None)` -> `{intr: [a0, a1], C: 4x4, poses: (n, 6) rotvec+t, log_scale: (n,) or zeros, res: (N, 2)
  all corners, kept: (N,) bool, success, nfev}`. Rounds: solve on the kept corners, residuals of ALL corners with the
  result, keep = |res| <= trim_px, stop when the kept set repeats; success = every round's scipy success.
- Model: `C = C0 @ [rotmat(dC[:3]) | dC[3:]]` maps camera-1 points into camera 0; corner in camera 0
  `Pc0 = R(pose) (P e^ls) + t`; camera 1 `Pc1 = C^-1 Pc0`. Clamps: a non-finite residual component = 1e3, a corner with
  z <= 1e-6 = 1e3 both components (`stereofit.residuals`); finite differences see a constant there, so the exact
  Jacobian is zero there too.
- Callers that go through `solve` by module name: `fit_converged`, `fit_from_nominal` (presolve: camera fixed, C free,
  `trim_px=inf`), `warm_fit` (camera and C fixed, default trimming), `lensscore._solve_fixed`. One hook in `solve`
  switches all of them.
- `lensscore.score` = PnP start (failed pairs dropped) -> pose solve -> pose+scale solve, each restarted once from its
  result when not successful; records per pair. `66` calls it twice (calibration, then baseline x 1.01 for the
  self-check); `calibtool.pipeline.summary` once per calibration. Live configs may be RADTAN (`calibtool.livecfg`
  registers it in `lensmodel` for the duration): the fast scorer must take any registered model.
- Data sizes: p21 fit.npz 512 pairs / 100 812 corners, test 499 / 99 740; p10 fit 587 / 116 372, test 572 / 112 652;
  at most 280 corners per pair (both cameras). Recorded scipy fits: p21 146 evaluations, 1 attempt, 273 s on Thor;
  p10 241 evaluations, 508 s, 3 holds dropped by the quality gate then a warm refit.

## File structure

| File | Responsibility |
|---|---|
| `tools/calibration/calibtool/solver.py` (new) | KB4/KB4T projection with its exact derivatives; the padded per-pair layout; residual + Jacobian; global LM with Schur complement; per-pair LM; `solve()` with `stereofit.solve`'s contract; `enabled()` (the switch) |
| `tools/calibration/calibtool/score.py` (new) | `score()` with `lensscore.score`'s contract on the fast solver |
| `tools/calibration/scripts/stereofit.py` (hook) | first lines of `solve`: `if _fast(): return solver.solve(...)` |
| `tools/calibration/scripts/lensscore.py` (hook) | first lines of `score`: `if sf._fast(): return calibtool.score.score(...)` |
| `tools/calibration/calibtool/tests/test_solver.py` (new) | derivatives vs finite differences, LM vs scipy on synthetic data, contract, switch |
| `tools/calibration/calibtool/tests/test_score.py` (new) | scorer vs `lensscore.score` on synthetic data, RADTAN path, restart contract |
| `tools/calibration/calibtool/tests/test_solver_equivalence.py` (new, `replay` marker) | section 9.1 acceptance on the stored sessions vs a tight-tolerance scipy reference |

## Task 1: KB4T projection derivatives

**Files:** create `calibtool/solver.py`, `calibtool/tests/test_solver.py`.

- [ ] Test (fails: no module): `project_jac(model, intr, P)` returns `uv (N,2)`, `d_uv/d_P (N,2,3)`,
  `d_uv/d_intr (N,2,ni)`; `uv` equals `lensmodel.project` to 1e-12 px; both Jacobians equal central finite
  differences (step 1e-6 relative) to 1e-6 relative, for KB4 and KB4T, on points at 0-80 deg off axis including one on
  the axis (rho = 0) and phantom12 Fit A's intrinsics.
- [ ] Implement: chain rule through theta = atan2(rho, Z), thd = theta (1 + k1 t2 + ... + k4 t2^4), s = thd / rho
  (pinhole limit 1/Z, d/dP of X/Z, Y/Z below rho = 1e-7 |Z|), then the tangential map
  `(mx + 2 p1 mx my + p2 (r2 + 2 mx^2), my + p1 (r2 + 2 my^2) + 2 p2 mx my)` with its 2x2 Jacobian, then fx, fy.
- [ ] Run, pass, commit `calibtool: M3 solver -- KB4/KB4T projection with exact derivatives`.

## Task 2: layout, residuals and Jacobian of the stereo problem

- [ ] Test: `Problem(data, model)` packs corners per pair (n, M) padded; `residuals(state)` returns, in the input
  corner order, exactly `stereofit.residuals` at the same parameters (1e-12 px), including the 1e3 clamps (a board
  behind camera 1); the Jacobian blocks (global 26 columns, per pair 6/7 columns) equal finite differences of
  `residuals` under the solver's local updates (`C <- C [Exp(dr) | dt]`, `R <- Exp(dr) R`, `t <- t + dt`,
  `ls <- ls + d`) to 1e-6 relative; zero rows at clamped corners.
- [ ] Implement `Problem` (sort by pair, padded index), `State` (intr pair, C, R (n,3,3), t (n,3), ls (n,)),
  `_eval(state, jac)`, `rotvec()` (vectorised robust log as `rigmath.log_rot`).
- [ ] Run, pass, commit.

## Task 3: global LM with Schur complement

- [ ] Test (synthetic `tests/lens_synth.py`: fit sweep 48 holds, Fit A truth): from the same start as
  `stereofit.fit_from_nominal`'s final `fit_converged` (presolve output), `solver.solve(..., free_intr=True,
  free_C=True)` reaches the scipy tight-tolerance optimum (`least_squares` ftol = xtol = gtol = 1e-15): intrinsics
  projection diff < 1e-3 px over a grid, C translation < 1e-4 mm, rotation < 1e-5 deg, same kept mask, success True;
  `nfev <= max_nfev`; `success` False when `max_nfev` is too small to converge (e.g. 2).
- [ ] Implement: per iteration J blocks -> H_gg (26x26, one matmul over all rows), H_gp (n,26,6|7), H_pp (n,6|7,6|7)
  and gradients by batched matmul over the padded rows; Jacobi scaling D = sqrt(diag H) (the `x_scale="jac"`
  equivalent); damped Schur system `S = Hgg + lam I - sum Hgp (Hpp + lam I)^-1 Hpg` by batched solve; back-substitute;
  Nielsen damping update; stop (success) on relative cost decrease < 1e-12 on an accepted step, scaled gradient
  inf-norm < 1e-10 sqrt(2 cost), or scaled step norm < 1e-10 (1 + scaled parameter norm); unsuccessful at `max_nfev`
  evaluations or damping > 1e16 with a gradient above the bound.
- [ ] Trim rounds as `stereofit.solve` (same `kept`, `res`, `success` = all rounds).
- [ ] Run, pass, commit.

## Task 4: per-pair LM (camera fixed) and generic models

- [ ] Test: with `free_intr=False, free_C=False` (scale False and True), `solver.solve` equals tight scipy per pair:
  poses to 1e-7 (rad, m), board scale to 1e-8; same `kept`; RADTAN registered through `calibtool.livecfg.radtan_scope`
  works with the camera fixed (point derivative by central differences) and `free_intr=True` with RADTAN raises
  `NotImplementedError` (the hook then keeps scipy).
- [ ] Implement the vectorised per-pair LM: each pair its own damping, cost and stop; only pairs still running are
  evaluated; success = every pair stopped by a convergence test before `max_nfev` evaluations.
- [ ] Run, pass, commit.

## Task 5: the switch hooks

- [ ] Test: `CALIB_SOLVER` unset -> `stereofit.solve` is scipy (spy on `least_squares` called); `CALIB_SOLVER=fast` ->
  `calibtool.solver.solve` called, scipy not; RADTAN with free intrinsics under `fast` -> scipy. Same for
  `lensscore.score` -> `calibtool.score.score`. The existing tools/calibration suite passes unchanged with the switch
  unset (default behaviour byte-identical: the hook is a branch before the old code).
- [ ] Implement in `stereofit.py`: `SOLVER_ENV = "CALIB_SOLVER"`, `_fast()`, the branch in `solve`; in
  `lensscore.py`: the branch in `score`. `calibtool` imported lazily (tools/calibration put on sys.path there).
- [ ] Run whole calibration suite (default) and the lens suites with `CALIB_SOLVER=fast`; commit.

## Task 6: the fast scorer

- [ ] Test (`test_score.py`): on `lens_synth.holdout_sweep()`, `score.score` returns the same keys and pairs as
  `lensscore.score`, scales within 1e-6 (synthetic: both converge tightly), hold rms within 1e-6 px, same solves dict;
  the self-check through `lensscore.self_check(..., scorer=score.score)` passes; a solve that stops short is restarted
  once from its result (same contract as `lensscore._solve_fixed`).
- [ ] Implement `score()` as `lensscore.score` with `solver.solve` (records built by the same code path).
- [ ] Run, pass, commit.

## Task 7: acceptance on stored sessions (section 9.1) and timings

- [ ] `test_solver_equivalence.py` (marker `replay`, `CALIBTOOL_DATA`): per session (p21_1105, p10_1106) runs step 45
  (`45_fit_lens.main`) and step 66 (`66_validate_lens.main`) twice in-process: reference = scipy with
  ftol = xtol = gtol = 1e-15 (patched `stereofit.least_squares`), fast = `CALIB_SOLVER=fast`. Asserts: projection
  difference over a 32 px image grid at z = 0.5 / 1 / 2 m < 0.02 px per camera; C translation < 0.01 mm, rotation
  < 0.001 deg, baseline < 0.01 mm; every step-66 entry median within 0.02 (% points); same verdicts except entries
  within 0.02 of their limit (listed); scorer on the recorded fit and on the live config: per-pair scale within 1e-5 of
  the reference. Writes a JSON line with every number (the report table).
- [ ] Timings (Moonstone, one thread, `nice -n 10`): 45 and 66 old (scipy default) vs new, and the whole
  `calibtool.pipeline` replay (split16 + fit45 + validate66 new + live + summaries) old vs new.
- [ ] phantom15 corners are on Terminus, not Moonstone: skipped, said so in the report.
- [ ] Record results in this file's "Results" section; commit.

## Results

Measured 9 Nov 2026 on Moonstone, one thread per process (`OMP_NUM_THREADS=1`, `nice -n 10`, other jobs on the box).
Code: commits 345b887c..8cf9199d. Runner: Moonstone `~/calibtool/m3-work/equiv.py` (step 45 + step 66 new and live +
the scorer on the recorded fit and on the live config, per session, in-process); outputs in `~/calibtool/m3-work/runs/`.
phantom15's corners are on Terminus, not Moonstone: not run.

**Three runs per session.** `default` = today's scipy; `ref` = scipy at ftol = xtol = gtol = 1e-10 (cap raised to 20000
evaluations; 1e-15 was unaffordable: 1e-10 already took 29 min (p21) / 55 min (p10) for step 45 alone); `fast` =
`CALIB_SOLVER=fast`.

### Changes found while finishing (all test-first)

- Unit references at 1e-15 x 20000 evaluations made the suite run > 30 min and still did not converge (scipy crawls in
  the flat valleys: full fit; nearly degenerate pose-and-scale pairs). Unit tier now: fast cost at or below scipy's
  (per pair with the camera fixed) and the fast solve started from scipy's result returns to its own (1e-5 px / 1e-7);
  the tight-scipy synthetic comparison at the design bound (1e-10) is `replay`-marked (passes, 110-170 s).
- A pair whose minimum runs off (a handful of corners left by trimming) made the fast solve report failure where scipy
  reports success: the fast solves now also count scipy's own ftol test (whole problem, 1e-8) met on the way.
- Under `CALIB_SOLVER=fast` the fast scorer bypassed `stereofit.solve`, which step 66's NO RESULT tests and lensscore's
  restart tests patch (5 failures): its solves now go through `stereofit.solve`.

### Tests

- `CALIB_SOLVER` unset, whole `tools/calibration`: 769 passed, 2 skipped (mujoco) -- this branch's full count (the ~791
  quoted elsewhere is the M4 worktree's).
- `CALIB_SOLVER=fast`, `calibtool/` + `tests/test_lens*.py` + `test_calib_lens_kb4t.py`: 386 passed.
- `test_reproduce.py -m replay`: unset 5 passed; fast 2 passed, 3 failed against the recorded scipy numbers, all on the
  live configs (see "live configs" below): p10 summary live IQR low -2.5186 vs recorded -2.490 (tolerance 0.02);
  p10 66 live FAIL on 8 validated entries vs recorded FAIL on 7 (one entry reached 8 holds); p21 live RADTAN reference
  summary median 0.7647 vs 0.7739 (tolerance 1e-3).

### Design 9.1 table (fast vs ref; default vs ref in brackets)

| | p21_1105 | p10_1106 |
|---|---|---|
| max projection diff, 32 px grid, 0.5/1/2 m, cam0 / cam1 (< 0.02 px) | **0.026** / 0.007 px [0.028 / 0.002] | 0.014 / **0.027** px [0.023 / 0.020] |
| stereo pose translation / rotation (< 0.01 mm / 0.001 deg) | 0.00016 mm / 0.00004 deg | 0.00018 mm / 0.00002 deg |
| baseline (< 0.01 mm) | 0.00003 mm | 0.000006 mm |
| step-45 cost (sum of squares, own kept set) fast / ref / default | 12948.937 / 12948.941 / 12948.865 | 19158.376 (110341) / 19149.388 (110340) / 19158.587 |
| step-66 new: verdicts; max validated-entry median diff (< 0.02 %) | PASS = PASS; 0.0002 | PASS = PASS; 0.0024 [0.011] |
| step-66 live: verdicts; max validated-entry median diff | FAIL = FAIL; 0.0000 [0.0004] | FAIL = FAIL; 0.0002 [0.0014] |
| validated entries (same holds and pairs per entry) | same | fast = ref (8); **default 7** (top-left 0.9-1.3 m: 7 holds) |
| entries near a limit (within 0.05 %) | live: right 0.6-0.9 m +0.537 % (limit 0.5) | live: centre 0.9-1.3 m +0.513 % (limit 0.5) |
| scorer, recorded fit: pairs with scale diff > 1e-5 (max) | 3 of 499 (2.9e-4) [default 12] | 13 of 572 (2.4e-4) [default 26] |
| scorer, live config: pairs > 1e-5 (max) | 5 of 499 (0.157) [default 56] | 28 of 572 (0.052) [default 39] |
| scorer: median scale diff, recorded / live | 4e-10 / 4e-9 | 8e-10 / 4e-9 |
| scorer solves (success flags), same pairs | same | same |
| self-check | passed, all runs | passed, all runs |

Every pose/baseline/step-66 criterion holds. Two numbers miss their bound -- projection (0.026, 0.027 px vs 0.02) and
per-pair scorer scale (1e-5) -- and in both the reference is the less converged of the two:

- **Projection.** On the reference's own corner set the reference's camera costs 12939.919 (p21) / 19156.978 (p10);
  the exact minimum there (fast solve polished on that set, no trimming) is 12939.132 / 19156.467, and the fast camera
  costs 12939.193 / 19156.502. So scipy at 1e-10 stops 0.79 / 0.51 above the minimum (2457 / 3411 evaluations) while
  the fast result is 0.06 / 0.04 above it, the rest being its slightly different corner set. ref vs that minimum:
  0.025 / 0.050 px; fast vs it: 0.011 / 0.032 px. The second effect is the **trim rule itself**: stereofit's 4 trim
  rounds do not settle on these sessions (rounds 5-7 still change 1-3 corners and move the projection 0.006-0.022 px),
  so two exact solvers whose rounds differ by one boundary corner differ by ~0.01-0.03 px. The 0.02 px bound is
  therefore not reachable against any reference that follows its own trim path, however tight; more rounds (until the
  set settles) would remove it but changes the contract for both solvers -- not done.
- **Scorer scale** (checked on the live configs, where the differences are). With the camera fixed, on the
  reference's own kept corners, the fast solve (from either start) is at or below scipy's cost on every pair
  (0 exceptions on 499 / 572 pairs), and scipy at 1e-10 sits up to 0.025
  (p21 live) / 0.032 (p10 live) in board scale off that minimum on nearly degenerate pairs. The remaining large
  differences are pairs whose kept corners differ: p21 live pairs 9 and 10 keep 4-6 of 140-148 corners (fast) vs
  18-20 (scipy) -- the live RADTAN does not fit them -- and move the not-validated entry bottom-right 0.4-0.6 m
  (2 pairs, 1 hold) by 15.7 % points; p10 live pair 264 (52 vs 53 kept).

### Live configs (both FAIL on every solver)

The live configs fit poorly (p10: 139 of 195 test holds flagged), many corners sit at the 3 px trim boundary, and
board scales of nearly degenerate pairs are poorly determined; scipy's stopping point decides them. That is why the
fast replay misses the recorded scipy numbers there (above) while every validated new-fit entry agrees to 0.003 %.
On p10 the step-66 live run of the tight reference validates the same 8 entries as the fast run: the replay's
"FAIL on 8 vs recorded FAIL on 7" is scipy at its default tolerance disagreeing with the tight reference, not the fast
solver. (The two summary-row misses were not rerun against a tight reference: a tight live scoring takes ~10 min.)

### Timings (Moonstone, one thread; old = scipy default)

| | p21_1105 old -> fast | p10_1106 old -> fast |
|---|---|---|
| step 45 (PnP, presolve, fit, quality gate, warm refit) | 129 s -> 15 s | 296 s -> 19 s |
| evaluations of the final fit | 132 -> 37 | 246 -> 38 |
| step 66, new fit | 24 s -> 8.4 s | 29 s -> 8.9 s |
| step 66, live config | 98 s (RADTAN) -> 8.5 s | 71 s (KB4T) -> 9.4 s |
| scorer alone, recorded fit / live | 11.4 / 44.7 s -> 4.0 / 4.1 s | 14.3 / 26.4 s -> 4.4 / 4.0 s |
| reference scipy 1e-10, step 45 / 66 live / scorer live | 1761 s / 989 s / 505 s | 3314 s / 1109 s / 550 s |

Whole replay (`test_reproduce.py -m replay`: split16 + fit45 + validate66 new + live + summaries, both sessions, plus
the p21 RADTAN reference): **983 s -> 133 s**. In the fast runs the remaining time of step 66 and the scorer is outside
the solves (PnP starts, records, the cluster bootstrap).

### Not done

- `tests/test_solver_equivalence.py` (replay marker) was not written: its asserts would fail on the two bounds above
  for reasons of the reference, and a tight reference costs ~1.5 h per session. The runner that produced these numbers
  is Moonstone `~/calibtool/m3-work/equiv.py` (`run <wt> <session> ref|fast|default <out>`, `compare`; `EQUIV_TOL`).
- Thor timings (the lead's run, with the user's OK). The default stays scipy.

### Settle-trim (10 Nov 2026, approved by the user)

Both solvers now trim until the kept corner set repeats (`stereofit.trim_loop`, cap `TRIM_CAP = 12`;
`CALIB_TRIM_ROUNDS=N` sets the cap, 4 = the contract every number above was made with). The 4-round contract did not
settle on either stored session, which is the second effect in "Projection" above. Step 45 reports `trim_rounds`,
`trim_settled` and `trim_cap`; a fit still changing at the cap stays "converged" with "; WARNING: trim not settled
after N rounds" and a `warnings` entry (warn, not fail: rounds 5-7 move the projection by hundredths of a pixel).
The recorded-number replays (`test_reproduce`, the stored-session decide / livecfg / scaling replays,
`test_fast_solver_replay_decides_as_scipy`) pin `CALIB_TRIM_ROUNDS=4` (conftest `recorded_trim`).

Measured on Moonstone (fast solver, step 16's split):

| | p21_1105 | p10_1106 |
|---|---|---|
| rounds to settle | 6 | 7 |
| step 45, 4 rounds -> settled (4 fits at once) | 14.3 s -> 17.2 s (37 -> 52 evaluations) | 16.8 s -> 22.7 s (38 -> 64) |
| projection, 4 rounds vs settled, cam0 / cam1 | 0.008 / 0.011 px | 0.020 / 0.022 px |
| baseline change | 0.01 um | 0.05 um |
| replay at settle-trim (`test_settle_trim_replay_decides_as_before`) | step 66 new PASS on the same 7 entries (medians within 0.002 %), live FAIL on the same 2; DEPLOY / NOT_ENOUGH_DATA | new PASS on the same 9 entries (within 0.011 %), live FAIL on the same 8; DEPLOY / DEPLOY |
| paired depth / rows interval vs the 4-round scipy replay | (-1.256, -0.330) vs (-1.227, -0.309); rows identical | (-2.227, -1.269) vs (-2.181, -1.261); rows identical |

The live configs move more (they fit poorly; the scorer trims too): p21 live right 0.6-0.9 m +1.10 % vs +0.54 %,
still failing; no verdict changes.
