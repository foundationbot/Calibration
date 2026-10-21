# phantom12 powered-neck takes (17 Oct 2026): how they were done

Why: hand-held neck takes cannot measure camera-in-Head to the gate (16 Oct: a hand loading the
unpowered neck puts the head 0.3–0.5° from where the encoders say, differently every hold). With the
motors holding commanded poses the load is gravity only and the holds are still.

## Motors and the neck, without the walker
- The walker (ghost-wbc) is not needed and was not used: it stays `OFF` and writes nothing to
  `/desired`. Joint targets go straight to the EtherCAT master with DMA.ethercat's own tools,
  installed on phantom12 with the `dma-ethercat` package (2.1.0):
  - motors: `dma_cmd_client enable-motors` / `disable-motors` / `estop` (the robot hanging on its
    gantry: `disable-motors` makes it limp);
  - the neck: `dma_motor_motion "NeckYaw=<deg>,NeckPitch=<deg>" targets position --ramp <s> --hold <s> --no-return`
    ramps the listed joints to absolute targets while every other joint holds its measured pose
    (Kp 30 Nm/rad, Kd 3 on all joints); with `--no-return` dma_main keeps holding the last pose.
    It exits 1 when a hold misses its 1° tolerance (gravity sag ~0.5–0.9° at Kp 30): expected.
- If `enable-motors` latches Elmo fault FF10 on every drive with status bit 4 (voltage enabled)
  clear and the drives reading ~3 V of bus: the motor power is off (on 17 Oct the E-stop was in);
  release it and send `enable-motors` again (the bus then reads ~53 V).
- `+NeckPitch` = head down, `+NeckYaw` = head left; MK1 neck limits yaw ±90°, pitch −28.4…+27.5°.

## The take
- Board fixed ~1.1 m in front. It was seen only with the head level or up (16 Oct data: pitch
  −28…+1°), so the grid is yaw −30, −15, 0, 15, 30 × pitch −24, −18, −12, −6, 0 deg.
- `tbs-grid.sh` (here; on Thor `/root/calib-work/tbs-grid.sh`): the 5 × 5 grid snaked along the yaw
  rows and back (49 holds), 1.5 s ramps, 3 s holds; then (P2) single-joint sweeps for step 82: a nod
  (pitch −24 ↔ 0 at yaw 0, 1.5 s strokes, 16°/s: too slow for step 82's 20°/s axis test) and a turn
  (yaw −30 ↔ +30 at pitch −12, 2 s strokes); then back to level. ~4.5 min.
- Recorder: the agent starts it (`05_record.sh` piped with a 500m CPU request, image
  `thor-2607a8f2`, `CAMERAS=1,2 IMU_INDEX=2 DURATION=420`), confirms both cameras and the IMU, then
  starts the grid under `setsid nohup`. Exposure left on auto (still holds).
- Check: `results/phantom12-2026-10-16-tbs/run-tbs.sh <take> [<earlier take>]` (steps 75 → 81 → 82); on Thor
  it is `/root/calib-work/run-tbs-ref.sh`, the same script with `CALIB_TOOLS` defaulting to `/root/calib-tools-ref`.
