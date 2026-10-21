#!/usr/bin/env bash
# Same data as calibration 3 (tree/stereo3), fitted with pinhole-radtan instead of pinhole-equi,
# so the two lens models can be compared on identical views. Waits for calib-run3.sh to finish.
R=/data/okvis-recordings; W=/root/calib-work
until grep -q "RUN3-ALL-DONE" /root/calib-run3.log 2>/dev/null; do sleep 30; done
echo "start radtan $(date +%T)"
PY="docker run --rm --cpus=2 --cpu-shares=2 --memory=8g -v $W:/w -v $R/tree:/w/tree -w /w -e CALIB_RIG=/w/config/rig-phantom12-radtan.yaml calib-py:1"
yes 568 | head -20 | docker run -i --rm --cpus=2 --cpu-shares=2 --memory=16g -e MPLBACKEND=Agg \
  -e LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1 -v $W:/data -v $R/tree:/data/tree -w /data \
  --entrypoint bash kalibr:1f60227-arm64 -c "source /catkin_ws/devel/setup.bash && TREE=/data/tree/stereo3 BAG=/data/calib_stereo3r.bag MODELS='pinhole-radtan pinhole-radtan' RIG=/data/config/rig-phantom12.yaml bash /data/scripts/40_calibrate.sh stereo" \
  > $W/step40-stereo3r.log 2>&1
echo "STEP40R-DONE exit $? $(date +%T)"
$PY bash -c "python scripts/60_validate_stereo.py calib_stereo3r-camchain.yaml --heldout tree/stereo3/heldout2; python scripts/65_scale_check.py calib_stereo3r-camchain.yaml --tree tree/stereo3/heldout2" 2>&1 | grep -v "^rig config"
docker run --rm --cpus=2 --cpu-shares=2 --memory=8g -e USED_TREE=/w/tree/stereo3 -e RIG_FILE=/w/config/rig-phantom12-radtan.yaml -v $W:/w -v $R/tree:/w/tree \
  -v $R/calib-stereo-2-euroc:/e1:ro -v $R/calib-stereo-2b-euroc:/e2:ro -v $R/calib-stereo-3-250-euroc:/e3:ro calib-py:1 \
  python /w/depth_scale2.py calib_stereo3r 4 /w/depth_scale3r.csv /e1 /e2 /e3 2>&1 | tail -1
echo "RUN3R-ALL-DONE $(date +%T)"
