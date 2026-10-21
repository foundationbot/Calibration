#!/bin/bash
# Install phantom12's transplanted calibration beside the old one and point robot.env at it.
# Does NOT restart OKVIS. Refuses if anything changed since it was read (sha checks).
set -euo pipefail
D=/etc/phantomos/okvis
NEW=phantom12_stereo_dense_thor.yaml
STAGED=/root/calib-work/deploy/$NEW
TS=$(date +%Y%m%d-%H%M%S)
want_live=95fce9a80d46fc24d8da02ed07aa4ac469904fe5e8f1567dd6cbce827a343527
want_new=7dd63554ba77b42314d0b26fe7aacbba36e7119fd6270748c436faa3a7e32046
[ "$(sha256sum $D/okvis_stereo_dense_thor.yaml | cut -d' ' -f1)" = "$want_live" ] || { echo "ABORT: live yaml changed"; exit 1; }
[ "$(sha256sum $STAGED | cut -d' ' -f1)" = "$want_new" ] || { echo "ABORT: staged file sha mismatch"; exit 1; }
grep -qx 'OKVIS_CALIBRATION=/etc/okvis/okvis_stereo_dense_thor.yaml' $D/robot.env || { echo "ABORT: robot.env changed"; cat $D/robot.env; exit 1; }
grep -qx 'OKVIS_DMA_CAMERAS=1,2' $D/robot.env || { echo "ABORT: robot.env camera pair changed"; exit 1; }
[ ! -e $D/$NEW ] || { echo "ABORT: $D/$NEW already exists"; exit 1; }
if k0s kubectl get pods -A 2>/dev/null | grep -qiE 'loco|walker|ghost'; then echo "ABORT: a walker pod is running"; exit 1; fi
cp -a $D/robot.env $D/robot.env.bak-$TS
install -m 0644 -o root -g root $STAGED $D/$NEW
[ "$(sha256sum $D/$NEW | cut -d' ' -f1)" = "$want_new" ]
tmp=$(mktemp $D/.robot.env.XXXXXX)
awk -v ts="$TS" '
  /^OKVIS_DMA_CAMERAS=/ && !done {
    print "# " ts ": calibration switched to phantom12'"'"'s own (Kalibr 2026-10-15, provenance in the yaml header)."
    print "#   Previous: OKVIS_CALIBRATION=/etc/okvis/okvis_stereo_dense_thor.yaml (kept; robot.env.bak-" ts ")."
    done = 1 }
  /^OKVIS_CALIBRATION=/ { print "OKVIS_CALIBRATION=/etc/okvis/phantom12_stereo_dense_thor.yaml"; next }
  { print }' $D/robot.env > $tmp
chmod 0644 $tmp; mv $tmp $D/robot.env
echo "== installed:"; ls -la --time-style=long-iso $D/$NEW $D/robot.env $D/robot.env.bak-$TS
echo "== robot.env now:"; cat $D/robot.env
echo "== diff vs backup:"; diff $D/robot.env.bak-$TS $D/robot.env || true
echo "INSTALL-OK $TS"
