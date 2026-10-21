#!/usr/bin/env bash
# phantom12 camera-IMU calibration, takes of 15 Oct 2026 evening (PDT): A (weak motion), B, C.
# Stereo camchain held fixed: calib_stereo3noedge (equidistant, the recommended stereo fit).
#   1. left/right sync at the take's fixed exposure (1500 us / ISO 1600)
#   2. step 10 --mode imucam, then filter_edges 450 (each camera on its own corners)
#   3. Kalibr camera-IMU fits: B and C filtered (the results), then B unfiltered and A filtered
# Run:  setsid nohup bash /root/calib-run-imucam.sh > /root/calib-run-imucam.log 2>&1 < /dev/null &
set -u
R=/data/okvis-recordings; W=/root/calib-work; T=/root/calib-tools
CAM=calib_stereo3noedge-camchain.yaml
PY="docker run --rm --cpus=2 --cpu-shares=2 --memory=8g -v $W:/w -v $R/tree:/w/tree -w /w calib-py:1"

echo "START $(date +%T)"
for x in B C; do
  docker run --rm --cpus=2 --cpu-shares=2 --memory=8g -e CC=/w/$CAM -v $W:/w -v $R/calib-imucam-$x-euroc:/e:ro \
    calib-py:1 python /w/sync_diag.py 6 /w/sync-calib-imucam-$x.csv 2>&1 | tail -1
  echo "SYNC-DONE $x $(date +%T)"
done

for x in A B C; do
  rm -rf "$R/tree/imucam-$x" "$R/tree/imucam-$x-noedge"
  (cd $T/tools/calibration && CALIB_RIG=$W/config/rig-phantom12.yaml nice -n 15 ionice -c3 python3 \
    scripts/10_build_trees.py --src $R/calib-imucam-$x-euroc --dst $R/tree/imucam-$x --mode imucam --jobs 2 2>&1 | tail -4)
  $PY python /w/filter_edges.py /w/tree/imucam-$x /w/tree/imucam-$x-noedge 450 2>&1 | grep FILTER-EDGES
  echo "TREE-DONE $x $(date +%T)"
done

fit() {   # fit <tree name>: Kalibr camera-IMU fit -> $W/calib_<tree>-{camchain-imucam,results-imucam}.*
  docker run --rm --cpus=2 --cpu-shares=2 --memory=16g -e MPLBACKEND=Agg \
    -e LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1 -v $W:/data -v $R/tree:/data/tree -w /data \
    --entrypoint bash kalibr:1f60227-arm64 -c "source /catkin_ws/devel/setup.bash && TREE=/data/tree/$1 BAG=/data/calib_$1.bag RIG=/data/config/rig-phantom12.yaml CAM=/data/$CAM bash /data/scripts/40_calibrate.sh imucam" \
    > $W/step40-$1.log 2>&1 < /dev/null
  echo "FIT-DONE $1 exit $? $(date +%T)"
}
fit imucam-B-noedge & fit imucam-C-noedge & wait
fit imucam-B & fit imucam-A-noedge & wait
echo "RUN-IMUCAM-ALL-DONE $(date +%T)"
