#!/usr/bin/env python3
"""Create an EMPTY /raw_imu_actuals shm ring so okvis2x_app_record can start on a
box with no dma-ethercat service (stereo-only calibration capture — the recorder
hard-exits if it cannot open the IMU queue, but never needs IMU *data* for the
camera streams).

Geometry matches the documented DMA ABI for RawImuFrame exactly:
  ShmHeader 192 B (magic "DMA_QUEU", version 1, slot_size 832, capacity 1024)
  + 1024 slots x 832 B  => 852160 B total, head=0 (no frames ever published).

A reader (dma::ShmRingBuffer attach) validates size/magic/version/capacity and
sees an empty ring. Refuses to touch an existing file. Remove with:
  rm /dev/shm/raw_imu_actuals   (or just reboot — /dev/shm is tmpfs)
"""
import mmap
import os
import struct
import sys

PATH = "/dev/shm/raw_imu_actuals"
MAGIC = 0x444D415F51554555  # "DMA_QUEU"
SLOT, CAP = 832, 1024
SIZE = 192 + SLOT * CAP  # 852160

if os.path.exists(PATH):
    sys.exit(f"{PATH} already exists — leaving it untouched")

fd = os.open(PATH, os.O_CREAT | os.O_RDWR, 0o666)
os.ftruncate(fd, SIZE)
mm = mmap.mmap(fd, SIZE)
mm[:] = b"\x00" * SIZE
struct.pack_into("<QQQQ", mm, 0, MAGIC, 1, SLOT, CAP)
mm.flush()
print(f"created {PATH}: {SIZE} B, empty ring (head=0)")
