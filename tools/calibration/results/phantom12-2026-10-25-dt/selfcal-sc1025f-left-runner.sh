#!/bin/bash
set -u; LOG=/root/burst-grab/sc1025f-left.log; cd /root/burst-grab
# Left-arm self-calibration session sc1025f-left (selfcal.py, generated). Aborts on motors < 29/29, a failed move or a failed burst.
# captures: 03-1023d_0 04-1023d_1 05-1023d_2 06-1023d_3 07-1023d_4 18-1023c_12 19-1023c_13 20-1023c_14 21-1023c_15 22-1023c_16
# held-out: 06-1023d_3 18-1023c_12 22-1023c_16
# stop: pkill -f sc1025f-left.sh  (the move in flight finishes first, up to ~16 s; then the arm holds that pose)
exec 9>/root/burst-grab/selfcal.lock; flock -n 9 || { echo "another self-calibration session holds /root/burst-grab/selfcal.lock"; exit 5; }
log(){ echo "$(date +%T.%N | cut -c1-12) $*" | tee -a $LOG; }
chk(){ s=$(timeout 10 dma_cmd_client state 2>&1); echo "$s" | grep -q "Motors enabled:    29/29" || { log "ABORT motors: $(echo $s | tr -s " ")"; exit 2; }; }
mv(){ chk; log "MOVE $1 ramp $2: $3"; out=$(dma_motor_motion "$3" targets position --ramp $2 --hold 2 --no-return --kp 100 --tolerance 30 2>&1); rc=$?; echo "$out" | grep -E "RESULT|FAIL|PASS|error" | tee -a $LOG; [ $rc -eq 0 ] || { log "ABORT rc $rc at $1"; exit 1; }; }
grab(){ sleep 1; log "BURST $1"; PYTHONPATH=pydeps nice python3 repo/tools/burst_pointcloud/grab.py --once 5 --joints all --out /root/burst-grab/sc1025f-left-$1.npz 2>&1 | tail -2 | tee -a $LOG; [ ${PIPESTATUS[0]} -eq 0 ] || { log "ABORT burst $1"; exit 3; }; }
log "START"; PYTHONPATH=pydeps python3 armsnap.py >> $LOG
mv 01-S1 6 "LeftShoulderPitch=-24.45,LeftShoulderRoll=5.75,LeftShoulderYaw=7.46,LeftElbowPitch=-43.06,LeftWristYaw=25.58,LeftWristRoll=-10.30,LeftGripperYaw=29.40,NeckPitch=28.39,NeckYaw=-0.41"
mv 02-start 14 "LeftShoulderPitch=-24.45,LeftShoulderRoll=5.75,LeftShoulderYaw=-49.34,LeftElbowPitch=-43.06,LeftWristYaw=-131.54,LeftWristRoll=-15.72,LeftGripperYaw=55.98,NeckPitch=28.39,NeckYaw=-0.41"
mv 03-1023d_0 14 "LeftShoulderPitch=-49.20,LeftShoulderRoll=3.48,LeftShoulderYaw=-50.59,LeftElbowPitch=-85.05,LeftWristYaw=-125.50,LeftWristRoll=-15.10,LeftGripperYaw=56.31,NeckPitch=28.39,NeckYaw=-0.41"
grab 03-1023d_0
mv 04-1023d_1 6 "LeftShoulderPitch=-41.28,LeftShoulderRoll=7.93,LeftShoulderYaw=-45.97,LeftElbowPitch=-95.84,LeftWristYaw=-130.84,LeftWristRoll=-15.13,LeftGripperYaw=56.32,NeckPitch=28.39,NeckYaw=-0.41"
grab 04-1023d_1
mv 05-1023d_2 6 "LeftShoulderPitch=-49.20,LeftShoulderRoll=3.48,LeftShoulderYaw=-50.59,LeftElbowPitch=-85.05,LeftWristYaw=-125.50,LeftWristRoll=-15.10,LeftGripperYaw=56.31,NeckPitch=28.39,NeckYaw=-0.41"
grab 05-1023d_2
mv 06-1023d_3 6 "LeftShoulderPitch=-41.28,LeftShoulderRoll=7.93,LeftShoulderYaw=-45.97,LeftElbowPitch=-95.84,LeftWristYaw=-130.84,LeftWristRoll=-15.13,LeftGripperYaw=56.32,NeckPitch=28.39,NeckYaw=-0.41"
grab 06-1023d_3
mv 07-1023d_4 6 "LeftShoulderPitch=-56.00,LeftShoulderRoll=34.08,LeftShoulderYaw=-50.13,LeftElbowPitch=-97.81,LeftWristYaw=-128.37,LeftWristRoll=-15.33,LeftGripperYaw=56.30,NeckPitch=28.39,NeckYaw=-0.41"
grab 07-1023d_4
mv 08-1023d_5 6 "LeftShoulderPitch=-58.85,LeftShoulderRoll=48.93,LeftShoulderYaw=-44.79,LeftElbowPitch=-92.63,LeftWristYaw=-127.69,LeftWristRoll=-15.45,LeftGripperYaw=56.29,NeckPitch=28.39,NeckYaw=-0.41"
mv 09-1023d_6 6 "LeftShoulderPitch=-34.14,LeftShoulderRoll=56.49,LeftShoulderYaw=-17.05,LeftElbowPitch=-93.69,LeftWristYaw=-95.19,LeftWristRoll=-15.10,LeftGripperYaw=56.31,NeckPitch=28.39,NeckYaw=-0.41"
mv 10-1023d_7 6 "LeftShoulderPitch=-5.36,LeftShoulderRoll=22.01,LeftShoulderYaw=-49.01,LeftElbowPitch=-107.95,LeftWristYaw=-91.35,LeftWristRoll=-14.38,LeftGripperYaw=56.55,NeckPitch=28.39,NeckYaw=-0.41"
mv 11-1023d_8 6 "LeftShoulderPitch=-8.55,LeftShoulderRoll=35.46,LeftShoulderYaw=-36.36,LeftElbowPitch=-127.40,LeftWristYaw=-82.90,LeftWristRoll=-14.84,LeftGripperYaw=56.31,NeckPitch=28.39,NeckYaw=-0.41"
mv 12-1023d_9 6 "LeftShoulderPitch=-6.29,LeftShoulderRoll=22.12,LeftShoulderYaw=-49.61,LeftElbowPitch=-103.83,LeftWristYaw=-90.93,LeftWristRoll=-14.84,LeftGripperYaw=56.30,NeckPitch=28.39,NeckYaw=-0.41"
mv 13-1023d_10 6 "LeftShoulderPitch=-8.55,LeftShoulderRoll=35.46,LeftShoulderYaw=-36.36,LeftElbowPitch=-127.40,LeftWristYaw=-82.90,LeftWristRoll=-14.84,LeftGripperYaw=56.31,NeckPitch=28.39,NeckYaw=-0.41"
mv 14-1023d_11 6 "LeftShoulderPitch=-15.56,LeftShoulderRoll=44.73,LeftShoulderYaw=-26.25,LeftElbowPitch=-126.76,LeftWristYaw=-76.49,LeftWristRoll=-14.87,LeftGripperYaw=56.31,NeckPitch=28.39,NeckYaw=-0.41"
mv 15-1023d_12 6 "LeftShoulderPitch=-24.22,LeftShoulderRoll=52.17,LeftShoulderYaw=-8.60,LeftElbowPitch=-120.86,LeftWristYaw=-91.49,LeftWristRoll=-15.05,LeftGripperYaw=56.32,NeckPitch=28.39,NeckYaw=-0.41"
mv 16-rev_1023d_11 6 "LeftShoulderPitch=-15.56,LeftShoulderRoll=44.73,LeftShoulderYaw=-26.25,LeftElbowPitch=-126.76,LeftWristYaw=-76.49,LeftWristRoll=-14.87,LeftGripperYaw=56.31,NeckPitch=28.39,NeckYaw=-0.41"
mv 17-rev_1023d_10 6 "LeftShoulderPitch=-8.55,LeftShoulderRoll=35.46,LeftShoulderYaw=-36.36,LeftElbowPitch=-127.40,LeftWristYaw=-82.90,LeftWristRoll=-14.84,LeftGripperYaw=56.31,NeckPitch=28.39,NeckYaw=-0.41"
mv 18-1023c_12 6 "LeftShoulderPitch=-28.64,LeftShoulderRoll=10.61,LeftShoulderYaw=-43.87,LeftElbowPitch=-100.44,LeftWristYaw=-76.61,LeftWristRoll=-14.82,LeftGripperYaw=56.30,NeckPitch=28.39,NeckYaw=-0.41"
grab 18-1023c_12
mv 19-1023c_13 6 "LeftShoulderPitch=-38.89,LeftShoulderRoll=2.59,LeftShoulderYaw=-39.66,LeftElbowPitch=-101.59,LeftWristYaw=-69.58,LeftWristRoll=-14.83,LeftGripperYaw=56.29,NeckPitch=28.39,NeckYaw=-0.41"
grab 19-1023c_13
mv 20-1023c_14 6 "LeftShoulderPitch=-28.64,LeftShoulderRoll=10.61,LeftShoulderYaw=-43.87,LeftElbowPitch=-100.44,LeftWristYaw=-76.61,LeftWristRoll=-14.82,LeftGripperYaw=56.30,NeckPitch=28.39,NeckYaw=-0.41"
grab 20-1023c_14
mv 21-1023c_15 6 "LeftShoulderPitch=-38.89,LeftShoulderRoll=2.59,LeftShoulderYaw=-39.66,LeftElbowPitch=-101.59,LeftWristYaw=-69.58,LeftWristRoll=-14.83,LeftGripperYaw=56.29,NeckPitch=28.39,NeckYaw=-0.41"
grab 21-1023c_15
mv 22-1023c_16 6 "LeftShoulderPitch=-45.44,LeftShoulderRoll=-0.36,LeftShoulderYaw=-23.50,LeftElbowPitch=-99.02,LeftWristYaw=-84.35,LeftWristRoll=-14.86,LeftGripperYaw=56.30,NeckPitch=28.39,NeckYaw=-0.41"
grab 22-1023c_16
mv 23-start 14 "LeftShoulderPitch=-24.45,LeftShoulderRoll=5.75,LeftShoulderYaw=-49.34,LeftElbowPitch=-43.06,LeftWristYaw=-131.54,LeftWristRoll=-15.72,LeftGripperYaw=55.98,NeckPitch=28.39,NeckYaw=-0.41"
mv 24-S1 14 "LeftShoulderPitch=-24.45,LeftShoulderRoll=5.75,LeftShoulderYaw=7.46,LeftElbowPitch=-43.06,LeftWristYaw=25.58,LeftWristRoll=-10.30,LeftGripperYaw=29.40,NeckPitch=28.39,NeckYaw=-0.41"
mv 25-hang 6 "LeftShoulderPitch=-11.36,LeftShoulderRoll=-0.14,LeftShoulderYaw=7.46,LeftElbowPitch=-8.16,LeftWristYaw=25.58,LeftWristRoll=-10.30,LeftGripperYaw=29.40,NeckPitch=28.39,NeckYaw=-0.41"
PYTHONPATH=pydeps python3 armsnap.py >> $LOG; log "DONE"
