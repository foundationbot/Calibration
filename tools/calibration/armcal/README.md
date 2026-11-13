# armcal — read B and the stage-2 URDF fit (hand-eye milestone H0)

The camera-to-arm calibration's computing parts, ported into this repo on 11 Nov 2026 (plan:
`docs/plans/2026-11-11-handeye-plan.md`, sections 4 and 8; H-1's verdict: `results/handeye-h1-2026-11-11/`). No robot
code here: capture (arm motion, bursts) is H2/H3.

| Module | What | Ported from |
|---|---|---|
| `burst_io` | the burst `.npz` (depth + RGB frames, sigma, joints, T_BS, body IMUs) | OKVIS2-X `tools/burst_pointcloud/burst_io.py` (c5b4c6b0), without the C++ hand-off |
| `fuse` | the burst's depth frames fused per pixel (Tukey-weighted, sigma-weighted) | `okvis_burst_fusion` `fuseDepth` (C++), as numpy |
| `urdf` | the URDF in memory: chains, origin changes, the corrected file's writer | `armfk.py` + phantom14 `urdfx/urdfvar.py` |
| `frames` | a burst's camera in the Head / Torso / Pelvis, **through the URDF passed in** | `frames.py` (no hard-coded URDF, no `BURST_URDF` patch; box-pick's level frame left out) |
| `mesh` | OBJ loading, area sampling, z-buffer splat, visibility | `mesh.py`, verbatim |
| `meshfit` | B0: one capture's forearm + wrist meshes matched in the depth (point-to-plane ICP); the robust solver (readb2) | `meshfit.py`: `fit_pose`, `view_gate` (B1/B2/B-neck/RGB term left out) |
| `readb` | burst file -> capture record, either arm; CLI | `hand_check.py` (load_pose, sides, motion/view checks) + phantom14 `urdfx/fitrun.py` (the record) |
| `urdffit` | stage 2: models, fit, standard errors, condition, leave-one-arm-pose-out CV, the URDF writer; CLI | phantom14 `urdfx/{predict,twoarm}.py` + H-1's `h1.py` |
| `meshes/` | the CAD meshes of both arms' ElbowPitch, WristYaw and WristRoll links (3.6 MB OBJ) | Moonstone `~/burst-pc/meshes/cycloidal_arm/` (allowed in this public repo, plan 9.1 #2) |

## Use

```
cd tools/calibration
python3 -m armcal.readb BURST.npz... --side right --urdf ROBOT.urdf --out right.json   # per capture (B0)
python3 -m armcal.readb BURST.npz... --side left  --urdf ROBOT.urdf --out left.json
python3 -m armcal.urdffit --right right.json --left left.json --urdf ROBOT.urdf \
    --report fit.json --out-urdf ROBOT_handeye.urdf [--held-out p08,pL09] [--note "session ..."]
```

- `--urdf` is the URDF the robot's consumers read now; the fit corrects it.
- `readb --frames N` fuses only the first N depth frames (a shorter burst); `--overlay DIR` writes the match
  pictures; `--jobs N` matches N captures at once.
- `readb --matcher`:
  - `robust` (the default since readb2): two passes per capture, clean only when they agree within 3 mm;
  - `fast`: the phantom14 algorithm's numbers, as in the stored records;
  - `reference` (= `--reference`): that algorithm exactly as it ran on phantom14, about 80 s per capture on one
    Moonstone core.

## The C++ module `_burstfusion` is not needed

Read B used only `run(...).depth.depth` (the fused depth) from it: `hand_check.load_pose` fused each burst with
`bf.run(burst, bf.Params())`. The fused RGB fed only the overlay picture, and the point cloud and voxels fed nothing.
`fuse.py` is that `fuseDepth` in numpy. It equals the stored C++ fusions of phantom14's bursts to float rounding
(`tests/test_p14_replay.py`), and on the robot it needs numpy only: no aarch64 build and no pybind11.

## Speed and robustness

`meshfit.FitConfig` selects how B0 computes:

| Config | What | Per capture, one Moonstone core |
|---|---|---|
| `REFERENCE` | the algorithm as it ran on phantom14 | 78.9 s |
| `FAST` (= `EXACT`) | the same numbers, to 3e-13 mm on all 48 stored captures | 13.0-13.7 s |
| `ROBUST` | a changed algorithm, run twice per capture by `readb.match_wave` (default) | 2.9 s, both passes |

How to read it:
- `REFERENCE` and `FAST` give the same answers; `FAST` is H0's exact speed-up;
- `ROBUST` gives different answers, judged by stage 2's cross-validation and per-capture stability, not by
  reproduction.

**FAST is fragile** (H0): thinning the model moved most captures. The coarse stage turns the forearm about its own
axis; the last stage creeps back along that near-flat direction and stops wherever its steps get small.

**ROBUST** (`results/handeye-readb2-2026-11-11/`):
- **the model:** the meshes sampled evenly (`mesh.sample_even`: 0.24 M points, not 5.2 M);
- **the coarse stages:** translation only at the 30 and 15 mm gates;
- **every step:** scaled by a line search on the stage's cost;
- **the forearm's roll:** a weak pull keeps it at the encoders' value unless the depth says otherwise;
- **the pixels:** every 2nd pixel until the last stage;
- **two passes:** from the encoders, then from the wave's median offset; a capture is clean only when the two land
  within 3 mm ("settled").

What it gives on phantom14's wave 2, against FAST:
- M12+head held-out error, median / p90: right 2.8 / 5.8 mm (FAST 3.6 / 7.1), left 4.5 / 6.3 mm (FAST 5.3 / 7.1);
- model changes move a capture about 0.3 mm (FAST: 0.7 mm, p90 3-4 mm);
- about a third of the captures fail the settled check and are not used.

## The models and the camera term

`urdffit` fits every candidate, scores each by leaving one arm pose out at a time, and keeps the one with the
smallest held-out wrist error. The candidates (`urdffit.MODELS`):

| Model | Terms | Note |
|---|---|---|
| `M12+headR` | NeckYaw origin, a shoulder-roll zero per arm, the camera's **rotation** in the head (HEADR, deg) | listed first: wins a tie |
| `M12+head` | the same, with the camera's **position** in the head (HEAD, mm) instead | H-1's model on phantom14 |
| `M12` | NeckYaw origin and the shoulder rolls; no camera term | the fallback and the model-only deploy |
| `M9`, `X1` | NeckYaw origin + right roll; the camera's position only | |

How to read it:
- **Why the rotation:** on phantom10 the head calibration's camera rotation was a few degrees off. One degree at the
  wrist's ~0.6 m is ~10 mm. Held-out median / p90, M12+head -> M12+headR:
  - 11 Nov: right 7.0 / 11.5 -> 3.9 / 10.4 mm, left 5.1 / 8.4 -> 3.6 / 8.4 mm (HEADR about -0.3, +3.4, 0.0 deg);
  - 13 Nov (another head calibration): right 9.1 / 12.8 -> 5.7 / 10.5 mm, left 17.1 / 22.3 -> 11.2 / 14.8 mm (HEADR
    about -2.7, +2.1, -1.5 deg).
- **phantom14 still picks the position:** on H-1's data M12+head scores 3.8 mm and M12+headR 5.8 mm, so the choice
  stays M12+head there (`test_p14_replay.py` pins this).
- **Not a candidate unless asked for:** `M12+headRT` (position and rotation, 11 terms). It overfits: on 13 Nov it put
  HEAD.y at +42.6 mm. Ask for it with `--models M12+headRT,...`.
- **Captures more than 15 mm from their arm's median are dropped** before the fit. Without that cut, 11 Nov's right arm
  goes to 11.5 / 52 mm.

The camera term is one transform D = `urdffit.head_transform(HEAD, HEADR)` = [R | t] in the head frame. A point p of
the head goes to R p + t: the camera is turned about the head's axes first, then moved. Synthetic captures with a
known 2-3 deg turn give it back exactly when noise-free. With 2 mm of noise per capture, each axis is within 0.33 deg
(standard errors 0.13-0.25 deg).

## The camera term and the deploy (H4)

The camera term is not a URDF origin: the camera sits in the Head link through OKVIS's T_BS, which the head
calibration owns. `urdffit.write_urdf` writes the URDF terms only and states HEAD / HEADR in the file's header.

The deploy (`calibtool/handeye_deploy.py`, module text = its README) puts D into T_BS, once: T_BS' = D T_BS. For
M12+headR that changes T_BS's rotation and translation (all 12 numbers, full precision). For M12+head it changes the
translation only. Why this order: the fit's prediction is T_TC = T_TB(neck) · D · T_BS · T_SC, so D T_BS is the only
place D can go without changing the prediction. A test rebuilds each capture from the written URDF and config and
gets the fit's offsets back to 1e-7 mm, for a camera moved, turned, and both. The other order (T_BS D) misses by 17-25
mm, and a deploy built that way refuses itself. In short:

- **What it writes** on the robot, next to the old files (never changing them): `<robot>_mk1_handeye_<date>.urdf` (the
  robot's own URDF with the fitted origins and a provenance header) and a copy of the LIVE OKVIS config with T_BS moved
  by D, `urdf:` pointing at the new URDF and a `# handeye: {...}` line; then robot.env names the new config and
  OKVIS restarts by the path the head tool uses (DaemonSet or okvis.sh; never a zero grace period).
- **Its checks** (any one refuses, nothing written): the machine-id names the fit's robot; robot.env, the config and the
  URDF are what the snapshot read; the URDF OKVIS reads is the one the captures were matched with (md5); the live
  T_BS and cameras' T_SC are the ones the captures were taken with (else "hand-eye stale for this camera geometry");
  the live config does not carry this fit already (D never twice); the written files reproduce the fit.
- **Verified** by OKVIS's log: the new config's sha (entrypoint) and `[Urdf] loaded '<new URDF>'`. A crash loop or a
  failed load rolls back by itself.
- **Rolling back:** R in the tool, or `bash <deploy dir>/rollback.sh`: robot.env as it was, OKVIS restarted.
- **Model only** (`--model-only`): T_BS kept, the M12 URDF (fitted without a camera term).
- **Other consumers** (`calibtool/config/handeye_consumers.yaml`): the burst point cloud (Moonstone, hard-coded URDF)
  gets the per-robot URDF published as a file; cuRobo and box-pick are listed as not updated.

Fit folder it reads: `fit.json`, `readb-{right,left}.json` and `geometry.json` (what the captures were taken with:
`python -m calibtool.handeye_deploy geometry BURST.npz... --robot P10 --session S --out geometry.json`).

## Tests

`tests/` runs in the normal suite (`run_moonstone.sh <worktree> <name> tools/calibration -q -n 8`). The unit tests
use synthetic captures made from the real meshes. `test_p14_replay.py` needs `$CALIBTOOL_DATA/p14arm` (Moonstone
`~/calibtool/data/p14arm`: symlinks to phantom14's bursts and C++ fusions in `/data/kaustabh/p14arm`, plus copies of
the stored records and the URDFs) and is skipped without it. It holds FAST to the stored records and ROBUST's two
passes to the readb2 runs.
