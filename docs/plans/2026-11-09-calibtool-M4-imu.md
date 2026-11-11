# calibtool M4: camera-to-IMU rotation v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `calibtool/imu.py` fits the cam0-to-IMU rotation correction (two angles, about camera x and z) from a neck
sweep over an AprilGrid lying flat on the floor, scores it on held-out poses against the live config, decides deploy /
keep live, and writes the corrected config by the kit's `mkfinal.py` rule; `calibtool/imusweep.py` is the sweep
capture (the kit's `imucam_fast.py`) behind an interface M5's neck driver implements.

**Architecture:** Pure numpy/scipy functions on a `Sweep` (per pose: neck angles, IMU acceleration mean/std, board
normal in camera 0, the config's cam0 rotation). The fit is the kit's `rscfit.py` model reproduced exactly (soft-L1 on
cross products, `f_scale` 0.02, start at 0). Held-out scoring = leave one yaw column out (pooled) plus the two pitch
halves (reported). Config surgery edits only the numbers inside the two `T_SC` blocks, in place. Capture is a loop over
a `SweepRig` protocol (moves, IMU window, frames, foreign-command counter) and a `detect(frame)` callable; it imports no
robot module and is tested with fakes only.

**Tech Stack:** Python 3, numpy, scipy (`least_squares`), cv2 (PnP in `imusweep.board_normal`, and one test reading the
written config with OpenCV FileStorage via `scripts/okviscfg.py`), pytest. Runs only on Moonstone.

Design source: `docs/superpowers/plans/2026-11-09-calibration-terminal-tool.md` sections 9.3 and 13 (M4 row). Kit
scripts: branch `origin/feat/kaustabh-stereo-depth-eval`, folder `calib-kit-20261106/` (`imucam_fast.py`, `rscfit.py`,
`mkfinal.py`, `mk_rscfix.py`).

---

## Facts established before planning (Moonstone, 9 Nov 2026)

- Stored sweep `~/calibtool/data/p10/kit/neckcal/imucam_1280.npz`: 15 poses (pitch rows ~9 / ~17 / ~25 deg, yaw
  columns ~-29 / -15 / 0 / +14 / +29 deg), keys `neck acc acc_std nb n R_SC`, `n == nb`, every `acc_std` < 0.23.
  Re-running the kit's `rscfit.py <npz> nb` gives `+0.320 / +0.112 deg` (formal 1-sigma 0.011 / 0.019), gap median
  0.34 -> 0.12 deg, and a rotation equal to `imucam_1280_Rnew_nb.npy` to 2.8e-13.
- `npz R_SC` == the 5 Nov rscfix config's cam0 rotation exactly == the 6 Nov pre-imu config's cam0 rotation (2e-16).
- `mkfinal.py`'s input is `kit/fit1280/phantom11_stereo_kb4t_20261106_1280.yaml` (the 6 Nov lens, stored), its output
  `..._20261106_1280_imu.yaml` (identical in `kit/fit1280/` and `etc-okvis/`). The 6 Nov lens's stereo pose differs from
  the 5 Nov rscfix's by 0.190 deg, so `apply(rscfix, C)` can match the `_imu` file's **cam0** only; **cam1** is compared
  on `apply(6 Nov pre-imu, C)` (same lens).
- `mkfinal.py` rewrote the bottom rows `0, 0, 0, 1` as `0.0, 0.0, 0.0, 1.0` and re-laid the blocks; `apply` keeps every
  unchanged number's text, so the comparison with the `_imu` file is numeric, not byte-wise.
- Third angle, measured: the 3-angle Jacobian's singular values are 3.87 / 3.85 / 0.51; the weakest direction is the mean
  board normal (up) in camera 0, (0.015, -0.562, -0.827). +1 deg about it moves the median gap 0.121 -> 0.157 deg; +1 deg
  about camera x -> 1.000 deg, about camera z -> 0.573 deg. So "unobservable" is "weakly observable through the 16 deg
  pitch span" (a single pitch row makes it exactly unobservable); v1 keeps the live value.
- No phantom21 board sweep exists in the stored data (its 5 Nov -16 deg fix came from the depth/SGBM floor route,
  `rsc_fix.npy` only), so the M4 row's "re-score phantom21 if its data allows" is not possible.

## Files

- Create `tools/calibration/calibtool/imu.py` — Sweep, fit, held-out, observability, verdict, T_SC read/apply.
- Create `tools/calibration/calibtool/imusweep.py` — SweepRig protocol, serpentine poses, board_normal, capture_sweep,
  to_sweep, save_sweep.
- Create `tools/calibration/calibtool/tests/test_imu.py`, `tools/calibration/calibtool/tests/test_imusweep.py`.
- Nothing else is modified (DEPS.txt already lists `scripts/lensmodel.py` and `scripts/okviscfg.py`).

Test command (all tasks): `bash tools/calibration/calibtool/tests/run_moonstone.sh /home/kaus/wt/calib-tui-m4 m4
tools/calibration/calibtool/tests/test_imu.py tools/calibration/calibtool/tests/test_imusweep.py -q -rfs`
(add `-k <name>` per step). Every test module starts with `pytest.importorskip` of numpy, scipy and yaml.

Commit trailer for every commit (exactly):

```
Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: okvis2-x-65 [988d7f]
```

---

### Task 1: Sweep, rotations, floor-vs-gravity gap

**Files:** Create `calibtool/imu.py`, `calibtool/tests/test_imu.py`

- [ ] **Step 1: failing tests** — the test module header with the synthetic sweep used by every later task:

```python
np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("yaml")
from calibtool import imu

R_HC = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])   # camera (x right, y down, z fwd) in head

def up_cam(pitch_deg, yaw_deg):
    """'up' in the camera frame for a neck pose: head = Rz(yaw) Ry(pitch), pitch > 0 looks down"""
    p, y = np.radians(pitch_deg), np.radians(yaw_deg)
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return R_HC.T @ (Rz @ Ry).T @ np.array([0.0, 0.0, 1.0])

R_CFG = R_HC.T @ imu.axis_rotation([1, 2, 3], 7.0)     # the config's cam0 rotation (any rotation works)

def synthetic(correction=np.eye(3), pitches=(9, 17, 25), yaws=(-30, -15, 0, 15, 30), noise_deg=0.0, seed=3):
    """a sweep whose true cam0 rotation is R_CFG @ correction; normals tilted by noise_deg (1-sigma per axis)"""
    rng = np.random.default_rng(seed)
    R_true = R_CFG @ correction
    neck, acc, nrm = [], [], []
    for k, p in enumerate(pitches):
        for y in (yaws if k % 2 == 0 else yaws[::-1]):
            n = up_cam(p, y)
            acc.append(9.81 * (R_true @ n))
            n = imu.axis_rotation(rng.normal(size=3), noise_deg * abs(rng.normal())) @ n if noise_deg else n
            neck.append((p + rng.normal(0, 0.2), y + rng.normal(0, 0.5)))
            nrm.append(n)
    neck, acc, nrm = map(np.array, (neck, acc, nrm))
    return imu.sweep_from_arrays(neck, acc, np.full_like(acc, 0.1), nrm, R_CFG)
```

Tests: `test_axis_rotation_and_rot_xz` (axis_rotation is orthonormal, turns about its axis; `rot_xz(a, b) ==
Rx(a) @ Rz(b)`), `test_gap_zero_when_config_is_true` (`floor_gap_deg(synthetic(), R_CFG)` < 1e-9),
`test_gap_equals_error_angle` (true rotation `R_CFG @ rot_xz(0.5, 0)`, scored with `R_CFG`: each pose's gap equals
`0.5 * |x_hat x up_cam|` deg within 1e-4, the part of the error the up direction can see),
`test_load_sweep_filters_motion` (an npz with one `acc_std` row 0.5 and one NaN normal keeps the rest; `n_recorded`
counts all rows; `key="n"` reads the `n` array), `test_load_sweep_reads_rscfit_format`.

- [ ] **Step 2:** run, expect `ModuleNotFoundError: calibtool.imu`.
- [ ] **Step 3: implement** in `imu.py`: constants `MAX_ACC_STD = 0.4`, `SOFT_L1_SCALE = 0.02`, `COLUMN_GAP_DEG = 5.0`,
  `ROW_GAP_DEG = 3.0`, `MIN_FIT_POSES = 3`; `axis_rotation(axis, deg)` (Rodrigues), `rot_xz(ax_deg, az_deg)`,
  `rotation_angle_deg(Ra, Rb)`, dataclass `Sweep(neck, acc, acc_std, normal, R_SC, n_recorded)` with `up_imu` and
  `__len__`, `sweep_from_arrays(...)` (keeps `acc_std.max(1) < max_acc_std` and finite normals, as rscfit),
  `load_sweep(path, key="nb", max_acc_std=MAX_ACC_STD)`, `floor_gap_deg(sweep, R, idx=None)` =
  `atan2(|v x n|, v . n)` with `v = R.T @ up_imu`.
- [ ] **Step 4:** run, expect PASS. **Step 5:** commit `calibtool: imu sweep loading and floor-vs-gravity gap (M4)`.

### Task 2: the two-angle fit (rscfit's model)

- [ ] **Step 1: failing tests:** `test_fit_recovers_xz_correction` (noise-free `synthetic(rot_xz(0.32, 0.11))` ->
  `fit(s).angles_deg` within 1e-5 of (0.32, 0.11), `gap_deg.max() < 1e-5`); `test_fit_sigma_shrinks_with_noise`
  (noise 0.02 -> both sigmas < 0.05 and > 0; noise 0.2 -> larger); `test_fit_soft_l1_bounds_one_bad_pose` (bounded
  influence: a 12 deg bad pose moves the angles < 1.5 x as far as a 3 deg one, and < 0.15 deg; a first draft asked a
  3 deg outlier to move them < 0.05 deg, which soft-L1 at f_scale 0.02 does not promise); `test_fit_subset_and_minimum` (`idx` of 2 poses -> ValueError);
  `test_r_new_is_config_times_correction`.
- [ ] **Step 2:** run, expect `AttributeError: fit`.
- [ ] **Step 3: implement** `Fit(angles_deg, sigma_deg, gap_deg, idx)` with properties `correction` and
  `combined_sigma_deg = hypot(sigma)`; `fit(sweep, idx=None)`:

```python
U, N, R = sweep.up_imu[idx], sweep.normal[idx], sweep.R_SC
def f(p):
    return np.cross(U @ (R @ _rx(p[0]) @ _rz(p[1])), N).ravel()
s = least_squares(f, [0.0, 0.0], loss="soft_l1", f_scale=SOFT_L1_SCALE)
cov = np.linalg.pinv(s.jac.T @ s.jac) * np.median(s.fun ** 2)
```

  `R_new(sweep, fit) = sweep.R_SC @ fit.correction`.
- [ ] **Step 4:** PASS. **Step 5:** commit `calibtool: imu two-angle fit, rscfit's soft-L1 model (M4)`.

### Task 3: held-out scoring

- [ ] **Step 1: failing tests:** `test_clusters_columns_and_rows` (3x5 grid -> 5 yaw columns, 3 pitch rows, labels in
  ascending order); `test_heldout_folds_partition_the_sweep` (5 column folds + 2 pitch folds; every pose in exactly one
  column fold's test set; fit and test sets disjoint); `test_heldout_noise_free` (`synthetic(rot_xz(0.5, -0.3))`:
  pooled new median < 1e-4, live median > 0.2); `test_heldout_live_rotation_argument` (scoring with `R_live =` the true
  rotation gives live median < 1e-6); `test_heldout_single_column` (one yaw column -> no column folds, medians NaN).
- [ ] **Step 2:** FAIL. **Step 3: implement** `clusters(values, gap)`, `yaw_columns`, `pitch_rows`, `Fold(name,
  fit_idx, test_idx, angles_deg, gap_new_deg, gap_live_deg)`, `HeldOut(columns, pitch, new_median_deg,
  live_median_deg, new_p90_deg, live_p90_deg, n_poses)`, `heldout(sweep, R_live=None)`; pitch halves = rows with label
  < n_rows // 2 vs the rest (rscfit split at pitch 0.5 deg, which puts every pose of a +9..+25 sweep on one side).
- [ ] **Step 4:** PASS. **Step 5:** commit `calibtool: imu held-out scoring by yaw column and pitch half (M4)`.

### Task 4: the third angle, measured

- [ ] **Step 1: failing tests:** `test_observability_weak_axis_is_up` (3x5 sweep: `|weak_axis . mean up| > 0.99`,
  `ratio < 0.3`); `test_single_pitch_row_is_unobservable` (one row: `ratio < 1e-6`);
  `test_gap_change_about_gravity_is_small` (+1 deg about up changes the median gap far less than +1 deg about camera x).
- [ ] **Step 2:** FAIL. **Step 3: implement** `Observability(up_cam, singular, weak_axis, ratio, weak_sigma_deg,
  gap_change_deg)` and `observability(sweep, R, step_deg=1.0)`: Jacobian of the stacked cross products w.r.t. small
  rotations about camera x, y, z (central differences, 1e-6 rad), SVD; `weak_sigma_deg = sqrt(median r^2) / s_min`;
  `gap_change_deg` = median gap after `R @ axis_rotation(axis, step)` minus before, for "gravity", "camera x",
  "camera z".
- [ ] **Step 4:** PASS. **Step 5:** commit `calibtool: imu third-angle observability measured (M4)`.

### Task 5: T_SC read and apply (mkfinal's rule)

- [ ] **Step 1: failing tests** on a synthetic config in the robot layout (two `- {T_SC:` flow blocks with 4 numbers per
  row, `0, 0, 0, 1` bottom rows, a `T_BS:` block, a comment line `# old T_SC: [9, 9, ...]`):
  `test_t_sc_reads_two_blocks_skipping_comments`; `test_apply_rotates_cam0_keeps_stereo_pose`
  (`R0_new == R0 @ C`, `inv(N0) @ N1 == inv(T0) @ T1` within 1e-12, cam0 translation equal);
  `test_apply_diff_is_only_those_numbers` (outside the two T_SC bodies byte-identical; inside, the separators are
  identical; cam0 translation and both bottom rows keep their text); `test_apply_identity_keeps_cam0_text`;
  `test_apply_refuses_bad_input` (one or three T_SC blocks, a non-rotation correction -> ValueError);
  `test_apply_note_after_yaml_header`; `test_apply_output_reads_in_okviscfg` (cv2 importorskip; `okviscfg.load` of the
  written file gives the new rotations within 1e-9).
- [ ] **Step 2:** FAIL. **Step 3: implement** `_TSC = re.compile(r"^(?![ \t]*#)[^\n]*?\bT_SC:\s*\[(?P<body>[^\]]*)\]",
  re.M)`, `_NUM`, `t_sc(text) -> (T0, T1)`, `cam0_rotation(text)`, `apply(config_text, R_correction, note=None)`:
  `N0 = T0` with rotation `R0 @ C`; `N1 = N0 @ inv(T0) @ T1`; rewrite cam0's 9 rotation numbers and cam1's top 12
  numbers in place (`repr(float(v))`, only where the value changed).
- [ ] **Step 4:** PASS. **Step 5:** commit `calibtool: imu T_SC surgery by the kit's mkfinal rule (M4)`.

### Task 6: verdict

- [ ] **Step 1: failing tests:** `test_verdict_deploys_a_real_error` (`synthetic(rot_xz(0.5, 0.3), noise 0.02)`,
  `R_live = R_CFG` -> deploy, `apply` with `v.correction` yields the true rotation within 0.05 deg);
  `test_verdict_keeps_a_correct_live` (`R_live` = true rotation -> keep live, reason names the combined sigma);
  `test_verdict_gates` (too few poses / one yaw column / pitch span 0 -> keep live with the matching reason);
  `test_verdict_reports_third_angle_and_tbs_note`; `test_correction_is_relative_to_live` (live != sweep R_SC:
  `R_live @ v.correction == v.R_new`).
- [ ] **Step 2:** FAIL. **Step 3: implement** `Gates(k_sigma=1.0, min_poses=9, min_yaw_columns=3,
  min_pitch_span_deg=10.0)` (all PLACEHOLDERS until fixed with the user), `Verdict(deploy, reasons, fit, heldout,
  observability, R_new, correction, change_vs_live_deg)`, `verdict(sweep, R_live=None, gates=Gates())`: deploy iff the
  gates hold and `live_median - new_median > k_sigma * fit.combined_sigma_deg`.
- [ ] **Step 4:** PASS. **Step 5:** commit `calibtool: imu verdict with named placeholder gates (M4)`.

### Task 7: phantom10 6 Nov reproduction (stored data; `data_root` fixture, skipped without CALIBTOOL_DATA)

- [ ] **Step 1: tests:** `test_p10_fit_reproduces_recorded` (angles within 0.01 deg of +0.320 / +0.112; `R_new` vs
  `imucam_1280_Rnew_nb.npy` < 0.01 deg); `test_p10_apply_reproduces_imu_config` (`apply(6 Nov pre-imu, C)` cam0 and cam1
  T_SC within 1e-6 of `..._1280_imu.yaml`); `test_p10_apply_on_rscfix_cam0` (`apply(5 Nov rscfix, C)` cam0 rotation
  within 1e-6 of the `_imu` file's; its cam1 follows the rscfix stereo pose, 0.1-0.3 deg from the `_imu` cam1);
  `test_p10_heldout_and_verdict` (vs rscfix live: deploy, held-out new < live; vs the `_imu` config as live: keep
  live); `test_p10_third_angle` (weak axis . up > 0.99, gravity change < 0.1 deg, camera x change > 0.5 deg).
- [ ] **Step 2:** run; these pass on the implementation of tasks 1-6 (they are acceptance tests, not drivers). If one
  fails, debug the implementation, not the expected numbers.
- [ ] **Step 3:** commit `calibtool: imu reproduces phantom10's 6 Nov rotation and config (M4)`.

### Task 8: capture: serpentine poses and the board normal

**Files:** Create `calibtool/imusweep.py`, `calibtool/tests/test_imusweep.py`

- [ ] **Step 1: failing tests:** `test_serpentine_default` (15 poses, yaw order reverses on the middle row, as
  imucam_fast); `test_board_normal_from_synthetic_view` (a 6x6 fake grid of 8 cm tags at 1.2 m, tilted 30 deg, projected
  with `lensmodel.project("KB4T", intr, P)`; `board_normal` returns the board's z axis flipped toward the camera within
  1e-6, rms < 1e-3 px, 144 corners); `test_board_normal_too_few_tags` (7 tags -> None).
- [ ] **Step 2:** FAIL. **Step 3: implement** constants `DEFAULT_PITCH_DEG = (8, 16, 24)`, `DEFAULT_YAW_DEG = (-30, -15,
  0, 15, 30)`, `FOREIGN_WINDOW_S = 0.2`, `SETTLE_S = 1.5`, `IMU_WINDOW_S = 2.0`, `FRAMES_PER_POSE = 3`, `MIN_TAGS = 8`;
  `serpentine(pitches, yaws)`; `BoardView(normal, corners, rms_px)`; `board_normal(det, board, model, intr,
  min_tags)` = imucam_fast's unproject -> `solvePnP` on normalised coordinates -> `solvePnPRefineLM` -> z axis toward
  the camera.
- [ ] **Step 4:** PASS. **Step 5:** commit `calibtool: imusweep poses and board normal, ported from imucam_fast (M4)`.

### Task 9: capture loop behind SweepRig (fakes only; never run on a robot here)

- [ ] **Step 1: failing tests** with `FakeRig` (neck kinematics as in test_imu, a true `R_SC`, gravity 9.81 plus noise,
  frames are pose tokens, `detect` maps a token to `BoardView(up_cam)`):
  `test_capture_then_fit_recovers_rotation` (`to_sweep` + `imu.fit` recover rot_xz(0.4, -0.2) within 0.02 deg);
  `test_capture_aborts_on_foreign_command` (counter advances before pose 4 -> `SweepAborted`, 3 moves done);
  `test_capture_skips_pose_without_board` (`detect` -> None at one pose: recorded with `view None`, left out of the
  sweep); `test_save_sweep_is_rscfit_format` (npz keys `neck acc acc_std nb n R_SC`; `imu.load_sweep` reads it back);
  `test_capture_uses_injected_sleep` (no real sleeping: total injected sleep = n x (0.2 + 1.5)).
- [ ] **Step 2:** FAIL. **Step 3: implement** `SweepAborted`, `SweepRig` (Protocol: `foreign_seq()`, `move_neck(p,
  y)`, `neck_deg()`, `imu_accel(seconds)`, `frames(k)`), `PoseResult(command, neck, acc_mean, acc_std, view, n_views,
  gap_deg)`, `capture_sweep(rig, poses, detect, R_SC, *, frames_per_pose, settle_s, imu_window_s, foreign_window_s,
  sleep=time.sleep, log=None)`, `to_sweep(results, R_SC)`, `save_sweep(path, results, R_SC)`.
- [ ] **Step 4:** PASS. **Step 5:** commit `calibtool: imusweep capture loop behind the SweepRig interface (M4)`.

### Task 10: numbers and status

- [ ] Run both test files on Moonstone; run an analysis script in `~/calibtool/m4-work` printing the fit, held-out folds,
  verdict reasons and observability on the stored sweep; record the numbers in this plan's "Results" section and commit
  `docs: calibtool M4 plan results (M4)`.

## Interface M5 implements (summary)

```python
class SweepRig(Protocol):
    def foreign_seq(self) -> int: ...          # advances when another source commands the robot (/dev/shm/desired)
    def move_neck(self, pitch_deg: float, yaw_deg: float) -> None: ...   # neck only, body pinned, returns on arrival
    def neck_deg(self) -> tuple[float, float]: ...   # measured (pitch, yaw)
    def imu_accel(self, seconds: float) -> np.ndarray: ...   # (k, 3) IMU slot accel samples gathered for `seconds`
    def frames(self, k: int) -> list: ...        # k fresh cam0 frames (grey)
detect = lambda g: imusweep.board_normal(board.detect(g), board, "KB4T", intr)
results = imusweep.capture_sweep(rig, imusweep.serpentine(), detect, imu.cam0_rotation(live_text))
v = imu.verdict(imusweep.to_sweep(results, R_SC), R_live=imu.cam0_rotation(live_text))
new_text = imu.apply(live_text, v.correction) if v.deploy else None
```

## Results (9 Nov 2026, Moonstone)

Tests: `test_imu.py` 37, `test_imusweep.py` 8, all pass; whole calibration suite 791 passed, 2 skipped (mujoco).

**Reproduction (phantom10 6 Nov sweep, 15 poses, all kept):**
- fit +0.319646 deg about camera x, +0.112190 deg about camera z (formal 1-sigma 0.0109 / 0.0191, combined 0.0220):
  -0.00035 / +0.00019 deg from the recorded +0.320 / +0.112; rotation vs `imucam_1280_Rnew_nb.npy`: max element
  1.9e-12, angle 0.0 deg.
- `apply(6 Nov pre-imu config = mkfinal's input, correction)` vs `..._20261106_1280_imu.yaml`: cam0 rotation 1.9e-12,
  cam1 rotation 1.9e-12, cam0 translation 0 (text kept), cam1 translation 2.6e-14; 6 of 353 lines change (the three
  top rows of each T_SC block).
- `apply(5 Nov rscfix config, correction)`: cam0 rotation 1.9e-12 from the `_imu` file's; its cam1 follows the 5 Nov
  lens's stereo pose and is 0.190 deg from the `_imu` cam1 (that is the lens change, not the fit).

**Held-out (leave one yaw column out, live = rscfix's cam0 rotation):**

| left out | fit angles x / z (deg) | new median | live median |
|---|---|---|---|
| yaw -29 | +0.330 / +0.074 | 0.086 | 0.313 |
| yaw -15 | +0.324 / +0.101 | 0.123 | 0.357 |
| yaw 0 | +0.339 / +0.115 | 0.137 | 0.306 |
| yaw +15 | +0.298 / +0.130 | 0.187 | 0.400 |
| yaw +29 | +0.306 / +0.143 | 0.077 | 0.340 |
| **pooled (15 poses)** | | **0.123** (p90 0.226) | **0.340** (p90 0.454) |
| fit pitch 17+25, score 9 | +0.360 / +0.146 | 0.193 | 0.291 |
| fit pitch 9, score 17+25 | +0.239 / +0.074 | 0.145 | 0.375 |

In-sample median 0.121 deg (the header's 0.12). Verdict vs rscfix: DEPLOY (gain 0.217 > 1 x 0.022). Verdict with the
`_imu` config as live: keep live (gain -0.002).

**Third angle:** 3-angle Jacobian singular values 3.865 / 3.847 / 0.511 (ratio 0.13); weakest axis (0.016, -0.562,
-0.827) = the mean up direction in camera 0 to 1e-4. Median gap response at the fit: +1 deg about gravity +0.036 deg,
about camera x +0.879, about camera z +0.453; +5 deg: +0.56 / +4.88 / +2.62. If fitted, its formal 1-sigma would be
0.08 deg. Weakly observable through the 16 deg pitch span; exactly unobservable for one pitch row (test).

Note on rscfit's parameterisation: camera z is 34 deg from up (up_cam z = -0.83), so the fitted +0.112 deg about camera
z carries ~0.09 deg about gravity; the angle v1 actually keeps from the live config is the one about camera y. A v2
model could fit the two rotations perpendicular to up instead.

phantom21's 5 Nov -16 deg fix: not re-scorable — the stored data hold no board sweep for it (`rsc_fix.npy` only; it came
from the depth/SGBM floor route).
