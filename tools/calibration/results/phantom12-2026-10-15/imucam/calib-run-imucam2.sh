#!/usr/bin/env bash
# Refit B and C (edge-filtered) with the loosened IMU noise model, after calib-run-imucam.sh ends.
set -u
R=/data/okvis-recordings; W=/root/calib-work; CAM=calib_stereo3noedge-camchain.yaml
IMU=/data/config/imu-phantom12-neckpitch-loose.yaml
until grep -q RUN-IMUCAM-ALL-DONE /root/calib-run-imucam.log; do sleep 30; done
echo "START $(date +%T)"
fit() {   # fit <tree> <bag name>
  docker run --rm --cpus=2 --cpu-shares=2 --memory=16g -e MPLBACKEND=Agg \
    -e LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1 -v $W:/data -v $R/tree:/data/tree -w /data \
    --entrypoint bash kalibr:1f60227-arm64 -c "source /catkin_ws/devel/setup.bash && TREE=/data/tree/$1 BAG=/data/$2.bag RIG=/data/config/rig-phantom12.yaml IMU=$IMU CAM=/data/$CAM bash /data/scripts/40_calibrate.sh imucam" \
    > $W/step40-$2.log 2>&1 < /dev/null
  echo "FIT-DONE $2 exit $? $(date +%T)"
}
fit imucam-B-noedge calib_imucam-B-loose & fit imucam-C-noedge calib_imucam-C-loose & wait
echo "RUN-IMUCAM2-ALL-DONE $(date +%T)"
