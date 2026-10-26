#!/bin/bash
set -u; LOG=/root/burst-grab/sc1025h-lift.log; cd /root/burst-grab
# Right-arm self-calibration session sc1025h-lift (selfcal.py, generated). Aborts on motors < 29/29, a failed move or a failed burst.
# captures: 
# held-out: 
# stop: pkill -f sc1025h-lift.sh  (the move in flight finishes first, up to ~16 s; then the arm holds that pose)
exec 9>/root/burst-grab/selfcal.lock; flock -n 9 || { echo "another self-calibration session holds /root/burst-grab/selfcal.lock"; exit 5; }
log(){ echo "$(date +%T.%N | cut -c1-12) $*" | tee -a $LOG; }
chk(){ s=$(timeout 10 dma_cmd_client state 2>&1); echo "$s" | grep -q "Motors enabled:    29/29" || { log "ABORT motors: $(echo $s | tr -s " ")"; exit 2; }; }
mv(){ chk; log "MOVE $1 ramp $2: $3"; out=$(dma_motor_motion "$3" targets position --ramp $2 --hold 2 --no-return --kp 100 --tolerance 30 2>&1); rc=$?; echo "$out" | grep -E "RESULT|FAIL|PASS|error" | tee -a $LOG; [ $rc -eq 0 ] || { log "ABORT rc $rc at $1"; exit 1; }; }
grab(){ sleep 1; log "BURST $1"; PYTHONPATH=pydeps nice python3 repo/tools/burst_pointcloud/grab.py --once 5 --joints all --out /root/burst-grab/sc1025h-lift-$1.npz 2>&1 | tail -2 | tee -a $LOG; [ ${PIPESTATUS[0]} -eq 0 ] || { log "ABORT burst $1"; exit 3; }; }
log "START"; PYTHONPATH=pydeps python3 armsnap.py >> $LOG
mv 01-lift-Right 6 "RightShoulderPitch=-20.00,RightShoulderRoll=-10.00,RightShoulderYaw=10.00,RightElbowPitch=-40.00,RightWristYaw=102.55,RightWristRoll=16.80,RightGripperYaw=-55.91,NeckPitch=28.39,NeckYaw=-0.40"
PYTHONPATH=pydeps python3 armsnap.py >> $LOG; log "DONE"
