# syntax=docker/dockerfile:1
# foundationbot/okvis2x:calib-<arch>-<sha> -- the calibration tool's robot runtime (calibtool; plan
# docs/superpowers/plans/2026-11-09-calibration-terminal-tool.md, section 10.3). Python 3.12 + numpy, SciPy,
# OpenCV (cv2.aruco: the AprilTag 36h11 detector), PyYAML, Pillow, DMA.common's dma_ipc binding, and the repo's
# tools/calibration tree. Nothing CUDA, TensorRT or torch: the tool's math is CPU numpy/SciPy.
#
# Built by CI only (.circleci/config.yml, build-calib), per arch: thor (arm64, what the robots pull) and amd64. NEVER
# built on a robot (decided 9 Nov). It replaces the hand-made kit venvs (venv + pydeps folders copied onto each robot):
# robot hosts have no OpenCV and no system pip, and the OKVIS image has only numpy and dma_ipc.
#
# Stages:
#   dma_common -> foundationbot/dma-common, only COPY'd from (no RUN: the foreign-arch userland never runs here). CI
#                 passes it BY DIGEST, resolved from docker/Dockerfile's `ARG DMA_COMMON_TAG=` pin + the arch suffix
#                 (-aarch64 on thor), exactly as build-perception does: the binding speaks the wire format the OKVIS
#                 image is built against, and the robots' producers write. No main-latest fallback (an older DMA.common
#                 lacks rings the tool reads; the check below would fail anyway).
#   runtime    -> the published image (default target).
#   test       -> runtime + pytest: the self-check and calibtool's fast tests, run INSIDE the build (CI builds
#                 `--target test` first; a failure fails the build). Not in the published image's chain.
#
# Python: Ubuntu 24.04's python3 (3.12), the interpreter DMA.common builds dma_ipc for (its image is ubuntu:24.04 and
# ships the binding under python3.12/dist-packages). Packages in a venv on that interpreter, from manylinux wheels only
# (--only-binary: no compiler in the image, and a missing aarch64 wheel fails the build by name instead of compiling).
# Pins = the versions the calibration pipeline was validated on (the Moonstone/Thor calibration venv, 9 Nov 2026);
# Pillow as the perception image pins it. All have cp312 wheels for manylinux_2_28 aarch64 and x86_64 (checked 9 Nov).
#
# Layout: the repo's layout under /opt/okvis-calib, because the package reads files by repo-relative path
# (calibtool.placement: config/foundation_rig/<urdf>; scripts/necktake.py: interface/mcap_reader.py).
#
# By hand (CI does this):
#   docker buildx build -f docker/calib.Dockerfile --target test \
#     --build-arg DMA_COMMON_IMAGE=foundationbot/dma-common:<pin>[-aarch64] .
# Run (tools/calibration/calibtool/RUNTIME.md):
#   docker run --rm --network host -v /dev/shm:/dev/shm foundationbot/okvis2x:calib-thor-<sha> check

ARG BASE_IMAGE=ubuntu:24.04
ARG DMA_COMMON_IMAGE=foundationbot/dma-common:AI-1184-latest-aarch64
FROM ${DMA_COMMON_IMAGE} AS dma_common

# ============================================================================
# runtime: the published image
# ============================================================================
FROM ${BASE_IMAGE} AS runtime
ARG DEBIAN_FRONTEND=noninteractive
RUN set -eu; ok=0; for _ in 1 2 3; do \
      if apt-get update -o Acquire::Retries=3 \
         && apt-get install -y --no-install-recommends -o Acquire::Retries=3 python3 python3-venv; then ok=1; break; fi; \
      sleep 10; \
    done; [ "$ok" = 1 ]; rm -rf /var/lib/apt/lists/*; \
    python3 -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'

ENV CALIB_VENV=/opt/calib-venv
ENV PATH=${CALIB_VENV}/bin:${PATH}
RUN python3 -m venv ${CALIB_VENV} \
 && pip install --no-cache-dir --only-binary=:all: \
        numpy==2.4.6 scipy==1.17.1 opencv-python-headless==5.0.0.93 PyYAML==6.0.3 pillow==12.3.0 \
 && pip check

# dma_ipc into the venv (as docker/perception.Dockerfile does), and dma_common's own Python beside it, as every other
# image that copies the binding keeps it. pybind11 binds numpy at runtime, so the venv's numpy 2 is the one it uses.
COPY --from=dma_common /usr/local/lib/python3.12/dist-packages/dma_ipc*.so ${CALIB_VENV}/lib/python3.12/site-packages/
COPY --from=dma_common /usr/local/lib/python3.12/dist-packages/dma_common ${CALIB_VENV}/lib/python3.12/site-packages/dma_common
# Guard: the binding's shared libraries resolve in THIS image (ubuntu:24.04 is leaner than the bases the other images
# copy it onto). ldd names a missing soname or a GLIBC/GLIBCXX version gap; either fails here, not on a robot.
RUN <<'GUARD'
#!/usr/bin/env bash
set -euo pipefail
so=$(ls "${CALIB_VENV}"/lib/python3.12/site-packages/dma_ipc*.so)
echo "dma_ipc: ${so} ($(du -h "${so}" | cut -f1))"
out=$(ldd "${so}" 2>&1)
echo "${out}"
if echo "${out}" | grep -E 'not found'; then
  echo "ERROR: dma_ipc cannot load in the calib image: the libraries above are missing" >&2
  exit 1
fi
GUARD

# The calibration package, at its repo path. Only what it reads: the tools/calibration tree (calibtool, scripts,
# config, tests) and the two repo files outside it.
WORKDIR /opt/okvis-calib
COPY tools/calibration/ tools/calibration/
COPY interface/mcap_reader.py interface/mcap_reader.py
COPY config/foundation_rig/20250206_phantom_mk1.urdf config/foundation_rig/20250206_phantom_mk1.urdf
RUN find /opt/okvis-calib -name __pycache__ -prune -exec rm -rf {} + \
 && chmod -R a+rX /opt/okvis-calib
WORKDIR /opt/okvis-calib/tools/calibration
ENV PYTHONPATH=/opt/okvis-calib/tools/calibration \
    PYTHONUNBUFFERED=1

# The runtime self-check in the build: libraries, cv2.aruco, every DEPS.txt module, the repo files, and dma_ipc (the
# camera/config/command rings the tool reads; pending readers made before their ring exists).
RUN python3 -c "import numpy, scipy, cv2, yaml, PIL, dma_ipc; cv2.aruco.ArucoDetector; import calibtool; print('imports ok')" \
 && python3 -m calibtool check

# `docker run <image>` = the self-check; `docker run <image> script 08_neck_grid_direct ...`, `... daemon` (calibd,
# once it lands), etc.: see calibtool/__main__.py and RUNTIME.md.
ENTRYPOINT ["python3", "-m", "calibtool"]
CMD ["check"]

# ============================================================================
# test: calibtool's fast tests in the image (not shipped)
# ============================================================================
FROM runtime AS test
# CI's runners are small: keep BLAS from oversubscribing them (calibration-tests sets the same).
ENV OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
RUN pip install --no-cache-dir --only-binary=:all: pytest==9.1.1
# The `replay` tests need stored session data (CALIBTOOL_DATA) and are deselected; the data-backed tests skip.
# 285 passed / 28 skipped in 12 s on 4 workstation cores (9 Nov).
RUN python3 -m pytest calibtool -q -m "not replay" -rfs -p no:cacheprovider --durations=10
