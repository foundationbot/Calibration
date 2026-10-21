"""Read-only on the takes: per hold, up to N stereo pairs (a camera 0 and a camera 1 image within 2 ms of
each other, none in a /joints gap), every detected AprilGrid corner of both images and the neck angles at
the stamp, saved as /w/tbs/<take>-corners.npz so mono, stereo and pixel-level fits run offline.
Usage (Thor, calib-py:2, PYTHONPATH=the package's scripts): python - <take> [<pairs per hold>] < dump_corners.py"""
import importlib
import sys
from pathlib import Path

import cv2
import numpy as np

import necktake
import rigcfg

s81 = importlib.import_module("81_tbs_handeye")
take = sys.argv[1]
per = int(sys.argv[2]) if len(sys.argv) > 2 else 8
W = Path("/w/takes")
exp = W / f"{take}-euroc"
rig = rigcfg.load(Path("/w/config/rig-phantom12.yaml"))
j = necktake.read_joints(W / take / "session.mcap")
holds = necktake.find_holds(j)
(t0s, p0), (t1s, p1) = [necktake.read_frames(exp, rig.kalibr[k]["record_dir"]) for k in rig.okvis_order]
k = np.clip(np.searchsorted(t1s, t0s), 1, len(t1s) - 1)
k = np.where(np.abs(t1s[k - 1] - t0s) < np.abs(t1s[k] - t0s), k - 1, k)
paired = np.abs(t1s[k] - t0s) <= 2_000_000

img, det = [], []
for hi, h in enumerate(holds):
    idx = np.flatnonzero((t0s >= h.t0) & (t0s <= h.t1) & paired)
    idx = idx[~s81.in_joint_gap(j, t0s[idx])]
    for i in idx[np.linspace(0, idx.size - 1, min(per, idx.size)).astype(int)] if idx.size else []:
        y, p = necktake.at(j, np.array([t0s[i]]))
        n = len(img)
        img.append((hi, int(t0s[i]), int(t1s[k[i]]), float(y[0]), float(p[0]), h.approach_yaw, h.approach_pitch))
        for cam, path in ((0, p0[i]), (1, p1[k[i]])):
            for (tag, c), (u, v) in rig.board.detect(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)).items():
                det.append((n, cam, tag, c, u, v))

I, D = np.array(img, dtype=object), np.array(det, dtype=object)
np.savez(f"/w/tbs/{take}-corners.npz",
         img_hold=I[:, 0].astype(int), img_t0=I[:, 1].astype(np.int64), img_t1=I[:, 2].astype(np.int64),
         img_yaw=I[:, 3].astype(float), img_pitch=I[:, 4].astype(float),
         img_ay=I[:, 5].astype(int), img_ap=I[:, 6].astype(int),
         det_img=D[:, 0].astype(int), det_cam=D[:, 1].astype(int), det_tag=D[:, 2].astype(int),
         det_k=D[:, 3].astype(int), det_u=D[:, 4].astype(float), det_v=D[:, 5].astype(float))
print(f"{take}: {len(img)} stereo pairs over {len(set(I[:, 0]))} holds, {len(det)} corners "
      f"(camera 0 {int((D[:, 1] == 0).sum())}, camera 1 {int((D[:, 1] == 1).sum())}) -> /w/tbs/{take}-corners.npz")
