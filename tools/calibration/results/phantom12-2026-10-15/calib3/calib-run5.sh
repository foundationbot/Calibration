#!/usr/bin/env bash
# Combination tests, fixed: all takes minus close-range frames, and all takes minus edge frames.
R=/data/okvis-recordings; W=/root/calib-work
EV="-v $R/calib-stereo-2-euroc:/e1:ro -v $R/calib-stereo-2b-euroc:/e2:ro -v $R/calib-stereo-3-250-euroc:/e3:ro"
DS="docker run --rm --cpus=2 --cpu-shares=2 --memory=8g -e USED_TREE=/w/tree/stereo3 -v $W:/w -v $R/tree:/w/tree $EV calib-py:1 python /w/depth_scale2.py"
fit() {
  yes 568 | head -20 | docker run -i --rm --cpus=2 --cpu-shares=2 --memory=16g -e MPLBACKEND=Agg \
    -e LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1 -v $W:/data -v $R/tree:/data/tree -w /data \
    --entrypoint bash kalibr:1f60227-arm64 -c "source /catkin_ws/devel/setup.bash && TREE=/data/tree/$1 BAG=/data/$2.bag RIG=/data/config/rig-phantom12.yaml bash /data/scripts/40_calibrate.sh stereo" \
    > $W/step40-$2.log 2>&1
  echo "FIT $2 exit $? $(date +%T)"
}
for spec in "nonear 0.6" "noedge 450"; do set -- $spec
  rm -rf $R/tree/stereo3$1
  docker run --rm --cpus=2 --cpu-shares=2 --memory=8g -v $W:/w -v $R/tree:/w/tree calib-py:1 python /w/filter_near.py $1 $2 2>&1 | tail -1
  fit stereo3$1 calib_stereo3$1
  $DS calib_stereo3$1 4 /w/depth_scale3$1.csv /e1 /e2 /e3 2>&1 | tail -1
done
echo "RUN5-ALL-DONE $(date +%T)"
