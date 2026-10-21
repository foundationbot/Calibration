#!/usr/bin/env bash
# Step 40 — build the bags and run the two Kalibr fits, inside the Kalibr docker.
#
# Kalibr is ROS1; docker is the sane path. Build the image ONCE:
#     git clone https://github.com/ethz-asl/kalibr.git
#     cd kalibr && docker build -t kalibr -f Dockerfile_ros1_20_04 .
#
# Then run this from the package root with your work dir mounted at /data:
#     xhost +local:root
#     docker run -it --rm -e DISPLAY -e QT_X11_NO_MITSHM=1 \
#       -v /tmp/.X11-unix:/tmp/.X11-unix -v "$PWD:/data" kalibr \
#       bash /data/scripts/40_calibrate.sh stereo
#
#     ... then step 30 on the HOST (it needs the stereo camchain) ...
#
#     docker run ... kalibr bash /data/scripts/40_calibrate.sh imucam
#
# ORDER MATTERS AND IS NOT NEGOTIABLE:
#     40 stereo  ->  30 label (host)  ->  40 imucam
# The labeller needs intrinsics, which only the stereo fit provides; and the
# camera-IMU fit needs the LABELLED tree, or it diverges to |T_SC| = 1.03 m.
set -euo pipefail

MODE="${1:?usage: 40_calibrate.sh stereo|imucam}"
DATA="${DATA:-/data}"
# Target, IMU model and lens model come from the rig config; env vars still win.
RIG="${RIG:-$DATA/config/rig.yaml}"
rig() {   # rig target|imu|models|target_type  -> value from $RIG (paths resolved against it)
  python3 - "$RIG" "$1" <<'PY'
import sys, yaml
from pathlib import Path
p = Path(sys.argv[1]); d = yaml.safe_load(p.read_text())
res = lambda v: str(Path(v) if Path(v).is_absolute() else p.parent / v)
print({"target": lambda: res(d["target"]),
       "imu":    lambda: res(d["imu"]["kalibr_yaml"]),
       "models": lambda: " ".join([d["cameras"]["model"]] * len(d["cameras"]["kalibr"])),
       "target_type": lambda: yaml.safe_load(open(res(d["target"])))["target_type"]}
      [sys.argv[2]]())
PY
}
stage_tree() {   # stage_tree TREE -> temp dir with only TREE's top-level cam* folders and imu*.csv
  # kalibr_bagcreater os.walk()s the whole tree for folders named cam* but keeps only their
  # NAMES, so step 10's TREE/heldout2/cam0 makes it bag TREE/cam0 a second time (phantom12,
  # 2026-10-15: 900 messages per camera for 450 images). Bag a staged view instead.
  local s p
  s="$(mktemp -d)"
  for p in "$1"/cam* "$1"/imu*.csv; do
    [ -e "$p" ] && ln -s "$(cd "$(dirname "$p")" && pwd)/$(basename "$p")" "$s/$(basename "$p")"
  done
  echo "$s"
}
[ -f "$RIG" ] || { echo "ERROR: rig config $RIG not found (set RIG=...)" >&2; exit 1; }
TARGET="${TARGET:-$(rig target)}"
IMU="${IMU:-$(rig imu)}"
MODELS="${MODELS:-$(rig models)}"   # mk13: pinhole-equi (fisheye), NOT pinhole-radtan
echo "rig $RIG: target=$TARGET imu=$IMU models=$MODELS"

source /catkin_ws/devel/setup.bash 2>/dev/null || source ~/kalibr_workspace/devel/setup.bash

case "$MODE" in
stereo)
  TREE="${TREE:-$DATA/tree/stereo}"
  BAG="${BAG:-$DATA/calib_stereo.bag}"
  echo "==> bag from $TREE"
  rosrun kalibr kalibr_bagcreater --folder "$(stage_tree "$TREE")" --output-bag "$BAG"
  echo "==> stereo fit (models: $MODELS)"
  rosrun kalibr kalibr_calibrate_cameras \
    --bag "$BAG" \
    --topics /cam0/image_raw /cam1/image_raw \
    --models $MODELS \
    --target "$TARGET" \
    --dont-show-report
  echo
  echo "SANITY CHECKS on the printed result:"
  echo "  * reprojection sigma well under 0.5 px   (2026-08-22: 0.266 / 0.234)"
  echo "  * baseline |t| from T_cn_cnm1 matches the rig  (2026-08-22: 79.0 mm)"
  echo "  * the two lenses' distortion coefficients have the SAME SIGNS."
  echo "    Sign-flipped between two identical lenses = the coverage failure."
  echo "  * reprojection error alone proves NOTHING about extrinsics — run step 60."
  ;;
imucam)
  TARGET_TYPE="$(rig target_type)"
  if [ "$TARGET_TYPE" = "aprilgrid" ]; then
    TREE="${TREE:-$DATA/tree/imucam}"             # tag ids anchor every corner: no step 30
  else
    TREE="${TREE:-$DATA/tree/imucam_canon}"       # the LABELLED tree from step 30
  fi
  BAG="${BAG:-$DATA/calib_imucam.bag}"
  CAM="${CAM:?set CAM=/data/camchain-calib_stereo.yaml (from the stereo fit)}"
  if [ ! -e "$TREE/imu0.csv" ]; then
    echo "ERROR: $TREE/imu0.csv missing — build the imucam tree with step 10 --mode imucam" >&2
    exit 1
  fi
  [ "$TARGET_TYPE" = "aprilgrid" ] || case "$TREE" in *canon*) ;; *)
    echo "WARNING: $TREE is not a *_canon tree. If you have not run step 30," >&2
    echo "         this fit will diverge (|T_SC| ~ 1 m). Ctrl-C now." >&2
    sleep 5 ;;
  esac
  echo "==> bag from $TREE"
  rosrun kalibr kalibr_bagcreater --folder "$(stage_tree "$TREE")" --output-bag "$BAG"
  echo "==> camera-IMU fit"
  rosrun kalibr kalibr_calibrate_imu_camera \
    --bag "$BAG" --cam "$CAM" --imu "$IMU" --target "$TARGET" \
    --dont-show-report
  echo
  echo "SANITY CHECKS:"
  echo "  * |T_SC| ~ 17 cm per camera.  ~1 m means step 30 was skipped or failed."
  echo "  * reprojection < ~1 px        (2026-08-22: 0.237 / 0.254)"
  echo "  * gyro/accel residual plots white-noise-like"
  echo "  * |timeshift_cam_imu| a few ms on host-clock stamps (phantom21 raw stamps: +1.2 ms)."
  echo "    Tens of ms (phantom13 2026-08-22: -23.6 / -24.6) = stamps rebased before the fit;"
  echo "    step 70 warns, and that image_delay must not be deployed."
  ;;
*) echo "usage: 40_calibrate.sh stereo|imucam" >&2; exit 2 ;;
esac
