#!/usr/bin/env bash
# phantom12 head cameras: fixed exposure for a calibration take, or back to auto.
#   bash /root/cam-exposure.sh fixed [exposure_us] [iso]     (default 1000 1600)
#   bash /root/cam-exposure.sh auto
# The producer reads head_camera.json only at start, so each change restarts it by scale 0 -> 1
# (a rolling restart port-clashes), then the viewer and rtsp-streamer, which otherwise keep
# showing frozen frames after a producer restart.
set -euo pipefail
CFG=/etc/phantom/head_camera.json
AUTO=/root/head_camera.json.auto-noBottom      # left + right on auto, no bottom camera (15 Oct)
K="k0s kubectl -n dma-video"

case "${1:-}" in
  fixed)
    EXP=${2:-1000}; ISO=${3:-1600}
    if ! cmp -s "$CFG" "$AUTO"; then
      echo "ERROR: $CFG is not the known auto config; not touching it. Differences:" >&2
      diff "$AUTO" "$CFG" >&2 || true
      exit 1
    fi
    python3 - "$CFG" "$EXP" "$ISO" <<'PY'
import json, sys
path, exp, iso = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
cfg = json.load(open(path))
for cam in ("left", "right"):
    cfg["cameras"][cam]["exposure"] = exp
    cfg["cameras"][cam]["iso"] = iso
json.dump(cfg, open(path, "w"), indent=2)
print(f"left and right: exposure {exp} us, iso {iso}")
PY
    ;;
  auto)
    cp "$AUTO" "$CFG"
    echo "restored $AUTO"
    ;;
  *)
    echo "usage: cam-exposure.sh fixed [exposure_us] [iso] | auto" >&2
    exit 2
    ;;
esac

restart() {
  $K scale "deploy/$1" --replicas=0
  $K wait --for=delete pod -l "app=$1" --timeout=60s 2>/dev/null || true
  $K scale "deploy/$1" --replicas=1
  $K rollout status "deploy/$1" --timeout=120s
}
restart producer
sleep 10
restart viewer
restart rtsp-streamer
echo "video files in /dev/shm: $(ls /dev/shm | grep -c '^video_') (want 4)"
$K logs deploy/producer --tail=2
