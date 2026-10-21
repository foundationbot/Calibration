#!/usr/bin/env bash
# Dataset-combination tests, all judged on the same pairs no fit uses (not in tree/stereo3).
R=/data/okvis-recordings; W=/root/calib-work
EV="-v $R/calib-stereo-2-euroc:/e1:ro -v $R/calib-stereo-2b-euroc:/e2:ro -v $R/calib-stereo-3-250-euroc:/e3:ro"
DS="docker run --rm --cpus=2 --cpu-shares=2 --memory=8g -e USED_TREE=/w/tree/stereo3 -v $W:/w -v $R/tree:/w/tree $EV calib-py:1 python /w/depth_scale2.py"
fit() {   # fit TREE NAME
  yes 568 | head -20 | docker run -i --rm --cpus=2 --cpu-shares=2 --memory=16g -e MPLBACKEND=Agg \
    -e LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1 -v $W:/data -v $R/tree:/data/tree -w /data \
    --entrypoint bash kalibr:1f60227-arm64 -c "source /catkin_ws/devel/setup.bash && TREE=/data/tree/$1 BAG=/data/$2.bag RIG=/data/config/rig-phantom12.yaml bash /data/scripts/40_calibrate.sh stereo" \
    > $W/step40-$2.log 2>&1
  echo "FIT $2 exit $? $(date +%T)"
}
echo "1: calibration 2 on the common held-out set $(date +%T)"
$DS calib_stereo2 4 /w/depth_scale2c.csv /e1 /e2 /e3 2>&1 | tail -1
echo "2: take 3 alone $(date +%T)"
fit calib-stereo-3 calib_stereo3only
$DS calib_stereo3only 4 /w/depth_scale3only.csv /e1 /e2 /e3 2>&1 | tail -1
echo "3: all takes without frames closer than 0.6 m $(date +%T)"
rm -rf $R/tree/stereo3nonear
docker run --rm --cpus=2 --cpu-shares=2 --memory=8g -v $W:/w -v $R/tree:/w/tree calib-py:1 python /w/filter_near.py 0.6 2>&1 | tail -1
fit stereo3nonear calib_stereo3nonear
$DS calib_stereo3nonear 4 /w/depth_scale3nonear.csv /e1 /e2 /e3 2>&1 | tail -1
echo "RUN4-ALL-DONE $(date +%T)"
