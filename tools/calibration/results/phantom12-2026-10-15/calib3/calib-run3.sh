#!/usr/bin/env bash
# Calibration 3: takes 2 + 2b (15 Oct afternoon) and the near/edges take 3, merged.
R=/data/okvis-recordings; W=/root/calib-work; T=/root/calib-tools
PY="docker run --rm --cpus=2 --cpu-shares=2 --memory=8g -v $W:/w -v $R/tree:/w/tree -w /w -e CALIB_RIG=/w/config/rig-phantom12.yaml calib-py:1"
nice -n 15 ionice -c3 python3 $T/tools/mcap_to_euroc.py $R/calib-stereo-3-250/session.mcap $R/calib-stereo-3-250-euroc
echo "EXPORT-DONE $(date +%T)"
docker run --rm --cpus=2 --cpu-shares=2 --memory=8g -v $W:/w -v $R/calib-stereo-3-250-euroc:/e:ro calib-py:1 \
  python /w/sync_diag.py 3 /w/sync-calib-stereo-3.csv 2>&1 | tail -1
cd $T/tools/calibration
CALIB_RIG=$W/config/rig-phantom12.yaml nice -n 15 ionice -c3 python3 scripts/10_build_trees.py \
  --src $R/calib-stereo-3-250-euroc --dst $R/tree/calib-stereo-3 --mode stereo --jobs 2 2>&1 | grep -E "paired|calib pairs|done:"
rm -rf $R/tree/stereo3; mkdir -p $R/tree/stereo3
for t in calib-stereo-2 calib-stereo-2b calib-stereo-3; do for c in cam0 cam1; do
  mkdir -p $R/tree/stereo3/$c $R/tree/stereo3/heldout2/$c
  cp -l $R/tree/$t/$c/*.png $R/tree/stereo3/$c/
  cp -l $R/tree/$t/heldout2/$c/*.png $R/tree/stereo3/heldout2/$c/
done; done
echo "STEP10-DONE $(date +%T) calib $(ls $R/tree/stereo3/cam0 | wc -l) heldout $(ls $R/tree/stereo3/heldout2/cam0 | wc -l)"
$PY python scripts/20_check_coverage.py tree/stereo3 2>&1 | grep -E "frames detected|grid coverage|radial reach|VERDICT"
echo "STEP20-DONE"
yes 568 | head -20 | docker run -i --rm --cpus=2 --cpu-shares=2 --memory=16g -e MPLBACKEND=Agg \
  -e LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1 -v $W:/data -v $R/tree:/data/tree -w /data \
  --entrypoint bash kalibr:1f60227-arm64 -c "source /catkin_ws/devel/setup.bash && TREE=/data/tree/stereo3 BAG=/data/calib_stereo3.bag RIG=/data/config/rig-phantom12.yaml bash /data/scripts/40_calibrate.sh stereo" \
  > $W/step40-stereo3.log 2>&1
echo "STEP40-DONE exit $? $(date +%T)"
$PY bash -c "python scripts/60_validate_stereo.py calib_stereo3-camchain.yaml --heldout tree/stereo3/heldout2; python scripts/65_scale_check.py calib_stereo3-camchain.yaml --tree tree/stereo3/heldout2" 2>&1 | grep -v "^rig config"
docker run --rm --cpus=2 --cpu-shares=2 --memory=8g -e USED_TREE=/w/tree/stereo3 -v $W:/w -v $R/tree:/w/tree \
  -v $R/calib-stereo-2-euroc:/e1:ro -v $R/calib-stereo-2b-euroc:/e2:ro -v $R/calib-stereo-3-250-euroc:/e3:ro calib-py:1 \
  python /w/depth_scale2.py calib_stereo3 4 /w/depth_scale3.csv /e1 /e2 /e3 2>&1 | tail -1
echo "RUN3-ALL-DONE $(date +%T)"
