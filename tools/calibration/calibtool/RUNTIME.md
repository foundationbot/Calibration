# calibtool robot runtime: the `calib` image

The robot side of calibtool runs from a CI-built image (plan section 10.3, decided 9 Nov 2026). It replaces the
hand-made kit venvs that were copied onto each robot. Robots pull the image like any other. **Never build it on a
robot.**

## What is in it

| | |
|---|---|
| Python | Ubuntu 24.04's python3 (3.12), venv `/opt/calib-venv` (first on `PATH`) |
| Packages | numpy 2.4.6, scipy 1.17.1, opencv-python-headless 5.0.0.93 (`cv2.aruco`, AprilTag 36h11), PyYAML 6.0.3, Pillow 12.3.0 |
| DMA | DMA.common's `dma_ipc` binding (+ its `dma_common` Python), built for the `ARG DMA_COMMON_TAG=` pin in `docker/Dockerfile`: the wire format the OKVIS image speaks |
| Code | `/opt/okvis-calib` mirrors the repo: `tools/calibration/` (calibtool, scripts, config, tests), `interface/mcap_reader.py`, `config/foundation_rig/20250206_phantom_mk1.urdf` |
| Not in it | CUDA, TensorRT, torch; textual/rich (the TUI runs on the laptop); mcap and mujoco (offline and arm-only scripts) |

The working directory is `/opt/okvis-calib/tools/calibration`. The entrypoint is `python3 -m calibtool`, and the
default command is `check`.

## Tags

All tags live in `foundationbot/okvis2x`, built by the CircleCI `build-calib` job on every branch push:

| Tag | Meaning |
|---|---|
| `calib-<sha>` | multi-arch: a robot (arm64) and an amd64 host pull the same name |
| `calib-thor-<sha>`, `calib-amd64-<sha>` | one arch per tag |
| `calib-<arch>-k<key>` | content tag; a commit that changes nothing the image holds re-tags it instead of rebuilding |

`<sha>` is the full commit SHA, as for the `okvis2x` images.

## Run it on a robot

```bash
IMG=foundationbot/okvis2x:calib-<sha>
docker pull "$IMG"

# self-check: libraries, cv2.aruco, every DEPS.txt module, dma_ipc, and how many /dev/shm entries the container sees
docker run --rm --network host -v /dev/shm:/dev/shm "$IMG"

# a pipeline script, e.g. the neck-grid check
docker run --rm -it --network host -v /dev/shm:/dev/shm -v /data/calib:/data/calib \
    "$IMG" script 08_neck_grid_direct --help
```

What each flag is for:

- `-v /dev/shm:/dev/shm`: the DMA rings (head cameras, IMU, config and actuals, commands) are POSIX shm segments
  under the host's `/dev/shm`. Without this mount, `check` reports `/dev/shm: 0 entries`, and every reader stays
  pending. `--ipc=host`, as the OKVIS container uses, works too.
- `--network host`: calibd's link to the laptop TUI, and anything the tool calls on the host's network.
- `-v /data/calib:/data/calib`: session folders (plan section 11). Without this mount, session data is lost when the
  container exits.
- Run as root (the default): the ring segments belong to root.
- A host program that a script runs, such as the motor tool that `08_neck_grid_direct --tool` names, is not in the
  image. Bind-mount it read-only (`-v <host path>:<same path>:ro`) and pass that path.

On a host without a docker CLI, the same settings fit a pod: `hostNetwork: true`, a `hostPath` volume for
`/dev/shm` (and one for the session folder), and image args such as `["check"]`.

## Commands (`python -m calibtool <command>`)

| Command | Does |
|---|---|
| `check [--no-dma]` | The runtime self-check. Exits 0 only if every item passes; the `/dev/shm` line is information. `--no-dma` skips `dma_ipc` on a machine without it |
| `script NAME [ARGS...]` | Runs `tools/calibration/scripts/NAME.py ARGS` |
| `MODULE [ARGS...]` | Runs `calibtool.MODULE` as a program, e.g. `livecfg`, `protocol`, and `daemon` (calibd) once it lands |
| `--help` | This list |

`CALIBTOOL_DATA` names the stored-session data root for replays (`-e CALIBTOOL_DATA=/data/calib/<dir>` together with
its mount).

## How it is built and checked

The image is defined in `docker/calib.Dockerfile` and built by the `build-calib` job in `.circleci/config.yml`, one
native runner per arch. Before any tag is published, three checks run on both arches:

1. During the image build, the `check` command and calibtool's fast tests (`pytest calibtool -m "not replay"`) run.
2. The image is pushed to an internal tag, pulled back, and run with the robot's flags (`check`).
3. Only then is it tagged `calib-<arch>-<sha>`. The `calib-manifest` job joins the two arches into `calib-<sha>`.

To change a package or a pin, edit `docker/calib.Dockerfile` and push. For a different wire format, change
`docker/Dockerfile`'s `ARG DMA_COMMON_TAG=` pin. That pin also drives the OKVIS image, so both move together.

The image is about 150-180 MB compressed and roughly 400-470 MB unpacked (estimated from the wheel sizes; arm64 is the
smaller one). The job log prints the measured pull size.
