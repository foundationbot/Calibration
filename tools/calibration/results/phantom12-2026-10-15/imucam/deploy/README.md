# phantom12: the 15 Oct calibration deployed into OKVIS (2026-10-16, 16:12 PDT)

phantom12's OKVIS (DaemonSet `okvis2x`, namespace `positronic`) ran a config that was never a
calibration of its cameras. Since 2026-10-16 it runs its own calibration, put into that config by
transplant: the robot's file with only the calibration replaced. Done by session `okvis2-x-8d`
with the user's go.

## What was wrong

`/etc/phantomos/okvis/okvis_stereo_dense_thor.yaml` (2026-08-09, sha256 `95fce9a8`), selected by
`robot.env` (`OKVIS_CALIBRATION`), with `OKVIS_DMA_CAMERAS=1,2`:

| | old file | phantom12's left/right pair (15 Oct) |
|---|---|---|
| baseline | 85.75 mm | **44.83 mm** |
| focal length | 584 / 588 px | 563 / 565 px |
| camera axis in the IMU frame | 25 deg up | 4 deg up |
| `T_BS` | identity: OKVIS's front = IMU +x, 90 deg off the cameras | phantom21's head value |
| IMU | no `imu_name`: slot 0 = **RightFoot** | NeckPitch (slot 2) |

The 85.75 mm baseline belongs to another camera pair, so the 21 deg between the old and new camera
axes does not mean the cameras were re-angled. The robot's other on-disk configs disagree with the
new result in the same way (`mk12_stereo.yaml`: 66 mm, 26 deg; phantom21's: 43 mm, 24 deg): they
share a head design that phantom12's head does not match. That gap is the open `T_BS` question
(`calibration.md` section 7).

## The transplant

`transplant.py okvis_stereo_dense_thor-20260809.yaml ../phantom12_stereo.yaml phantom12_stereo_dense_thor.yaml --check`

- **Replaced:** per camera `T_SC`, `image_dimension`, `focal_length`, `principal_point`,
  `distortion_coefficients`, `distortion_type`, plus `dma_queue` 1 / 2;
  `camera_parameters.image_delay` (+0.02 -> -0.0056 s); `imu_parameters.T_BS`; new
  `dma.imu_name: "NeckPitch"`.
- **Kept, byte for byte:** everything else, i.e. estimator timing, final BA, online calibration off,
  IMU noise, SuperPoint frontend, depth network, mapping and output settings, camera roles.
  Our step-70 file is built on phantom21's template and would have changed those too
  (`realtime_time_limit` 0.035 -> 0.015 s, `do_final_ba` true -> false, online extrinsics on,
  `a_max` 176 -> 78).
- **Checked** with OpenCV FileStorage, the parser OKVIS uses: of the live file's 175 values, 53
  changed (all of them calibration numbers, each equal to `../phantom12_stereo.yaml`'s), 122 are
  identical, and 3 were added (two `dma_queue`, `dma.imu_name`). The cameras come out along the
  Head's forward axis, 30 deg down, baseline 44.83 mm.

Files: `okvis_stereo_dense_thor-20260809.yaml` (the robot's file, as it was),
`phantom12_stereo_dense_thor.yaml` (deployed, sha256 `7dd63554`), `transplant.py`,
`install.sh` (what ran on the robot).

## Installed

`install.sh` refuses unless the live yaml, `robot.env` and the staged file still have the expected
hashes and no walker pod runs. It then copies `robot.env` to `robot.env.bak-20261016-161136`,
installs the new file beside the old one (the old file is not modified) and points
`OKVIS_CALIBRATION` at `/etc/okvis/phantom12_stereo_dense_thor.yaml`. Then
`k0s kubectl -n positronic rollout restart ds/okvis2x` (graceful; never `--grace-period=0`).

The new pod's startup lines:

    [entrypoint] config: phantom12_stereo_dense_thor.yaml explicit path /etc/okvis/phantom12_stereo_dense_thor.yaml (7dd63554ba77)
    [DMASensor] OKVIS cam 0 <- DMA cam 1 (video_1, colour/RGB)
    [DMASensor] OKVIS cam 1 <- DMA cam 2 (video_2, colour/RGB)
    Resolved DMA IMU 'NeckPitch' -> slot 2 via RobotConfigData (/config).
    Using IMU at slot 2 from the DMA (name 'NeckPitch')

After 90 s: no errors, no restarts; SLAM latency p50 149-163 ms (old config 186-189 ms), depth
78 ms (unchanged), no IMU drops, frame drops 1.5 % (old 3.7 %). `head_yaw` reads about +92 deg in
OKVIS's world frame, whose zero is the IMU's +x at start-up: the 90 deg correction.

## Still open

- `T_BS` is phantom21's: heading is right, the Head frame's pitch is not known (cameras 30 deg
  down in the Head frame against phantom21's ~10 deg). Depth and VIO do not use it; Head- and
  Pelvis-frame outputs and grasping do. Being measured (`docs/superpowers/plans/2026-10-16-phantom12-tbs-tests.md`).
- The deployment has no URDF (`dma.urdf` unset, as before), so no neck FK: the pelvis pose falls
  back to the head pose (`haveNeckFk=0`).
- `okvis-follow-me` also sources `robot.env`: it picks up the new file at its next start.
- Whatever wrote `robot.env` on 2026-10-16 09:20 (`scripts/configure-okvis-robot-env.sh`, not on
  the robot, nothing reruns it) would point it back at the old file if run again.

## Rollback (on the robot)

    cp -a /etc/phantomos/okvis/robot.env.bak-20261016-161136 /etc/phantomos/okvis/robot.env
    k0s kubectl -n positronic rollout restart ds/okvis2x
