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
| `meshfit` | B0: one capture's forearm + wrist meshes matched in the depth (point-to-plane ICP) | `meshfit.py`: `fit_pose`, `view_gate` (B1/B2/B-neck/RGB term left out) |
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
- `readb --frames N` fuses only the first N depth frames (a shorter burst); `--reference` runs the matcher exactly
  as on phantom14 (about 80 s per capture on one Moonstone core); `--overlay DIR` writes the match pictures.

## The C++ module `_burstfusion` is not needed

Read B used only `run(...).depth.depth` (the fused depth) from it: `hand_check.load_pose` fused each burst with
`bf.run(burst, bf.Params())`. The fused RGB fed only the overlay picture, and the point cloud and voxels fed nothing.
`fuse.py` is that `fuseDepth` in numpy. It equals the stored C++ fusions of phantom14's bursts to float rounding
(`tests/test_p14_replay.py`), and on the robot it needs numpy only: no aarch64 build and no pybind11.

## Speed

`meshfit.FitConfig` selects how B0 computes. `REFERENCE` is the algorithm as it ran on phantom14 (78.9 s per capture
alone on one Moonstone core). `FAST` (the default, `= EXACT`) uses only changes that give the same numbers: on all 48
stored phantom14 captures, both arms, the offsets agree to 3e-13 mm and every clean decision is unchanged. It takes
13.0 s per capture alone, or 20.9 s with 8 in parallel. Every approximation tried failed the 0.5 mm / 1° rule:
thinning the model, every 2nd pixel, fewer or looser steps. The measurements, and the finding that B0 is fragile on
many captures, are in `results/handeye-h0-2026-11-11/README.md`.

## The HEAD term (open)

M12+head fits the camera's position in the head. That is not a URDF origin: the camera sits in the Head link through
OKVIS's T_BS, which the head calibration owns. `urdffit.write_urdf` writes the URDF terms only and states the HEAD
term in the file's header. Where it goes on a robot (T_BS's translation, once, or a camera joint) is still open. It must
not be applied twice.

## Tests

`tests/` runs in the normal suite (`run_moonstone.sh <worktree> <name> tools/calibration -q -n 8`). The unit tests
use synthetic captures made from the real meshes. `test_p14_replay.py` needs `$CALIBTOOL_DATA/p14arm` (Moonstone
`~/calibtool/data/p14arm`: symlinks to phantom14's bursts and C++ fusions in `/data/kaustabh/p14arm`, plus copies of
the stored records and the URDFs) and is skipped without it.
