#!/bin/bash
set -u; LOG=/root/burst-grab/sc1025c-right.log; cd /root/burst-grab
# Right-arm self-calibration session sc1025c-right (selfcal.py, generated). Aborts on motors < 29/29, a failed move or a failed burst.
# captures: 03-1023d_0 04-1023d_1 07-1023d_4 08-1023d_5 09-1023d_6 10-1023d_7 11-1023d_8 12-1023d_9 14-1023d_11 15-1023d_12 18-1023c_12 19-1023c_13 22-1023c_16
# held-out: 04-1023d_1 07-1023d_4 19-1023c_13
# stop: pkill -f sc1025c-right.sh  (the move in flight finishes first, up to ~16 s; then the arm holds that pose)
exec 9>/root/burst-grab/selfcal.lock; flock -n 9 || { echo "another self-calibration session holds /root/burst-grab/selfcal.lock"; exit 5; }
log(){ echo "$(date +%T.%N | cut -c1-12) $*" | tee -a $LOG; }
chk(){ s=$(timeout 10 dma_cmd_client state 2>&1); echo "$s" | grep -q "Motors enabled:    29/29" || { log "ABORT motors: $(echo $s | tr -s " ")"; exit 2; }; }
mv(){ chk; log "MOVE $1 ramp $2: $3"; out=$(dma_motor_motion "$3" targets position --ramp $2 --hold 2 --no-return --kp 100 --tolerance 30 2>&1); rc=$?; echo "$out" | grep -E "RESULT|FAIL|PASS|error" | tee -a $LOG; [ $rc -eq 0 ] || { log "ABORT rc $rc at $1"; exit 1; }; }
grab(){ sleep 1; log "BURST $1"; PYTHONPATH=pydeps nice python3 repo/tools/burst_pointcloud/grab.py --once 5 --joints all --out /root/burst-grab/sc1025c-right-$1.npz 2>&1 | tail -2 | tee -a $LOG; [ ${PIPESTATUS[0]} -eq 0 ] || { log "ABORT burst $1"; exit 3; }; }
log "START"; PYTHONPATH=pydeps python3 armsnap.py >> $LOG
mv 01-S1 6 "RightShoulderPitch=-24.45,RightShoulderRoll=-5.75,RightShoulderYaw=7.36,RightElbowPitch=-43.06,RightWristYaw=97.01,RightWristRoll=15.56,RightGripperYaw=-55.93,NeckPitch=28.00,NeckYaw=-0.41"
mv 02-start 14 "RightShoulderPitch=-24.45,RightShoulderRoll=-5.75,RightShoulderYaw=49.34,RightElbowPitch=-43.06,RightWristYaw=131.54,RightWristRoll=15.72,RightGripperYaw=-55.98,NeckPitch=28.00,NeckYaw=-0.41"
mv 03-1023d_0 14 "RightShoulderPitch=-49.20,RightShoulderRoll=-3.48,RightShoulderYaw=50.59,RightElbowPitch=-85.05,RightWristYaw=125.50,RightWristRoll=15.10,RightGripperYaw=-56.31,NeckPitch=28.00,NeckYaw=-0.41"
grab 03-1023d_0
mv 04-1023d_1 6 "RightShoulderPitch=-41.28,RightShoulderRoll=-7.93,RightShoulderYaw=45.97,RightElbowPitch=-95.84,RightWristYaw=130.84,RightWristRoll=15.13,RightGripperYaw=-56.32,NeckPitch=28.00,NeckYaw=-0.41"
grab 04-1023d_1
mv 05-1023d_2 6 "RightShoulderPitch=-49.20,RightShoulderRoll=-3.48,RightShoulderYaw=50.59,RightElbowPitch=-85.05,RightWristYaw=125.50,RightWristRoll=15.10,RightGripperYaw=-56.31,NeckPitch=28.00,NeckYaw=-0.41"
mv 06-1023d_3 6 "RightShoulderPitch=-41.28,RightShoulderRoll=-7.93,RightShoulderYaw=45.97,RightElbowPitch=-95.84,RightWristYaw=130.84,RightWristRoll=15.13,RightGripperYaw=-56.32,NeckPitch=28.00,NeckYaw=-0.41"
mv 07-1023d_4 6 "RightShoulderPitch=-56.00,RightShoulderRoll=-34.08,RightShoulderYaw=50.13,RightElbowPitch=-97.81,RightWristYaw=128.37,RightWristRoll=15.33,RightGripperYaw=-56.30,NeckPitch=28.00,NeckYaw=-0.41"
grab 07-1023d_4
mv 08-1023d_5 6 "RightShoulderPitch=-58.85,RightShoulderRoll=-48.93,RightShoulderYaw=44.79,RightElbowPitch=-92.63,RightWristYaw=127.69,RightWristRoll=15.45,RightGripperYaw=-56.29,NeckPitch=28.00,NeckYaw=-0.41"
grab 08-1023d_5
mv 09-1023d_6 6 "RightShoulderPitch=-34.14,RightShoulderRoll=-56.49,RightShoulderYaw=17.05,RightElbowPitch=-93.69,RightWristYaw=95.19,RightWristRoll=15.10,RightGripperYaw=-56.31,NeckPitch=28.00,NeckYaw=-0.41"
grab 09-1023d_6
mv 10-1023d_7 6 "RightShoulderPitch=-5.36,RightShoulderRoll=-22.01,RightShoulderYaw=49.01,RightElbowPitch=-107.95,RightWristYaw=91.35,RightWristRoll=14.38,RightGripperYaw=-56.55,NeckPitch=28.00,NeckYaw=-0.41"
grab 10-1023d_7
mv 11-1023d_8 6 "RightShoulderPitch=-8.55,RightShoulderRoll=-35.46,RightShoulderYaw=36.36,RightElbowPitch=-127.40,RightWristYaw=82.90,RightWristRoll=14.84,RightGripperYaw=-56.31,NeckPitch=28.00,NeckYaw=-0.41"
grab 11-1023d_8
mv 12-1023d_9 6 "RightShoulderPitch=-6.29,RightShoulderRoll=-22.12,RightShoulderYaw=49.61,RightElbowPitch=-103.83,RightWristYaw=90.93,RightWristRoll=14.84,RightGripperYaw=-56.30,NeckPitch=28.00,NeckYaw=-0.41"
grab 12-1023d_9
mv 13-1023d_10 6 "RightShoulderPitch=-8.55,RightShoulderRoll=-35.46,RightShoulderYaw=36.36,RightElbowPitch=-127.40,RightWristYaw=82.90,RightWristRoll=14.84,RightGripperYaw=-56.31,NeckPitch=28.00,NeckYaw=-0.41"
mv 14-1023d_11 6 "RightShoulderPitch=-15.56,RightShoulderRoll=-44.73,RightShoulderYaw=26.25,RightElbowPitch=-126.76,RightWristYaw=76.49,RightWristRoll=14.87,RightGripperYaw=-56.31,NeckPitch=28.00,NeckYaw=-0.41"
grab 14-1023d_11
mv 15-1023d_12 6 "RightShoulderPitch=-24.22,RightShoulderRoll=-52.17,RightShoulderYaw=8.60,RightElbowPitch=-120.86,RightWristYaw=91.49,RightWristRoll=15.05,RightGripperYaw=-56.32,NeckPitch=28.00,NeckYaw=-0.41"
grab 15-1023d_12
mv 16-rev_1023d_11 6 "RightShoulderPitch=-15.56,RightShoulderRoll=-44.73,RightShoulderYaw=26.25,RightElbowPitch=-126.76,RightWristYaw=76.49,RightWristRoll=14.87,RightGripperYaw=-56.31,NeckPitch=28.00,NeckYaw=-0.41"
mv 17-rev_1023d_10 6 "RightShoulderPitch=-8.55,RightShoulderRoll=-35.46,RightShoulderYaw=36.36,RightElbowPitch=-127.40,RightWristYaw=82.90,RightWristRoll=14.84,RightGripperYaw=-56.31,NeckPitch=28.00,NeckYaw=-0.41"
mv 18-1023c_12 6 "RightShoulderPitch=-28.64,RightShoulderRoll=-10.61,RightShoulderYaw=43.87,RightElbowPitch=-100.44,RightWristYaw=76.61,RightWristRoll=14.82,RightGripperYaw=-56.30,NeckPitch=28.00,NeckYaw=-0.41"
grab 18-1023c_12
mv 19-1023c_13 6 "RightShoulderPitch=-38.89,RightShoulderRoll=-2.59,RightShoulderYaw=39.66,RightElbowPitch=-101.59,RightWristYaw=69.58,RightWristRoll=14.83,RightGripperYaw=-56.29,NeckPitch=28.00,NeckYaw=-0.41"
grab 19-1023c_13
mv 20-1023c_14 6 "RightShoulderPitch=-28.64,RightShoulderRoll=-10.61,RightShoulderYaw=43.87,RightElbowPitch=-100.44,RightWristYaw=76.61,RightWristRoll=14.82,RightGripperYaw=-56.30,NeckPitch=28.00,NeckYaw=-0.41"
mv 21-1023c_15 6 "RightShoulderPitch=-38.89,RightShoulderRoll=-2.59,RightShoulderYaw=39.66,RightElbowPitch=-101.59,RightWristYaw=69.58,RightWristRoll=14.83,RightGripperYaw=-56.29,NeckPitch=28.00,NeckYaw=-0.41"
mv 22-1023c_16 6 "RightShoulderPitch=-45.44,RightShoulderRoll=0.36,RightShoulderYaw=23.50,RightElbowPitch=-99.02,RightWristYaw=84.35,RightWristRoll=14.86,RightGripperYaw=-56.30,NeckPitch=28.00,NeckYaw=-0.41"
grab 22-1023c_16
mv 23-start 14 "RightShoulderPitch=-24.45,RightShoulderRoll=-5.75,RightShoulderYaw=49.34,RightElbowPitch=-43.06,RightWristYaw=131.54,RightWristRoll=15.72,RightGripperYaw=-55.98,NeckPitch=28.00,NeckYaw=-0.41"
mv 24-S1 14 "RightShoulderPitch=-24.45,RightShoulderRoll=-5.75,RightShoulderYaw=7.36,RightElbowPitch=-43.06,RightWristYaw=97.01,RightWristRoll=15.56,RightGripperYaw=-55.93,NeckPitch=28.00,NeckYaw=-0.41"
mv 25-hang 6 "RightShoulderPitch=-19.15,RightShoulderRoll=-0.82,RightShoulderYaw=7.36,RightElbowPitch=-34.25,RightWristYaw=97.01,RightWristRoll=15.56,RightGripperYaw=-55.93,NeckPitch=28.00,NeckYaw=-0.41"
PYTHONPATH=pydeps python3 armsnap.py >> $LOG; log "DONE"
