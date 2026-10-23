#!/usr/bin/env python3
"""The arm session's corner dump (Task 7's input), run where the recording is (calib-py image, PYTHONPATH with
interface/ and this scripts/ directory): every image of both cameras inside the given time windows -> the palm
plate's AprilGrid corners, plus every /joints sample of the recording. One npz:

    cam, t, tag, k, u, v      one row per corner found (cam 0 = /cam1 = OKVIS camera 0, the left one)
    img_cam, img_t, img_n     one row per image looked at, with its number of corners (0 = plate not found)
    jt, jq                    /joints: stamps (ns) and positions (rad, /config order)

usage: arm_corners.py session.mcap windows.json out.npz [cols rows tag_size spacing]
windows.json: [[t0_ns, t1_ns], ...] in the robot's clock (image stamps = the mcap's publish times), or "-" for every
image (the holds are then found from /joints by the analysis, which needs no clock shared with the laptop)."""
import json
import sys

import cv2
import numpy as np

import armplan
import mcap_reader
import rigcfg

TOPICS = ("/cam1/image", "/cam2/image")


def main(argv):
    path, wpath, out = argv[:3]
    cols, rows, size, spacing = (int(argv[3]), int(argv[4]), float(argv[5]), float(argv[6])) if len(argv) > 3 \
        else (2, 2, 0.045, 0.2)
    grid = rigcfg.AprilGrid(cols, rows, size, spacing)
    win = None if wpath == "-" else np.array(json.loads(open(wpath).read()), np.int64).reshape(-1, 2)
    rows_c, rows_i, jt, jq = [], [], [], []

    def inside(t):
        return win is None or bool(((win[:, 0] <= t) & (t <= win[:, 1])).any())

    def on(ch, log_t, pub_t, data):
        if ch.topic in TOPICS:
            if not inside(pub_t):
                return
            cam = TOPICS.index(ch.topic)
            g = cv2.imdecode(np.frombuffer(bytes(data), np.uint8), cv2.IMREAD_GRAYSCALE)
            det = armplan.detect_plate(grid, g) if g is not None else {}     # the 7x5 board shares ids 0-8
            rows_i.append((cam, pub_t, len(det)))
            for (tid, k), (u, v) in det.items():
                rows_c.append((cam, pub_t, tid, k, u, v))
        elif ch.topic == "/joints":
            m = json.loads(bytes(data))
            jt.append(m["timestamp_ns"])
            jq.append(m["positions"])

    mcap_reader.read_session(path, on_message=on)
    c = np.array(rows_c, float).reshape(-1, 6)
    i = np.array(rows_i, np.int64).reshape(-1, 3)
    np.savez(out, cam=c[:, 0].astype(int), t=c[:, 1].astype(np.int64), tag=c[:, 2].astype(int),
             k=c[:, 3].astype(int), u=c[:, 4], v=c[:, 5], img_cam=i[:, 0], img_t=i[:, 1], img_n=i[:, 2],
             jt=np.array(jt, np.int64), jq=np.array(jq, float))
    print(f"images {len(i)} (plate found in {int((i[:, 2] > 0).sum())}), corners {len(c)}, joints {len(jt)}")


if __name__ == "__main__":
    main(sys.argv[1:])
