#!/usr/bin/env python3
"""AprilGrid coverage for the kalibr takes: have both cameras seen the board everywhere, near and far?

Live, while recording (the robot host has no OpenCV, so it runs inside the OKVIS image):

  docker run --rm --network host -v /dev/shm:/dev/shm -v /data/okvis-perf/tools:/t \\
      --entrypoint python3 okvis2x:on-device /t/calib_coverage.py --cams 2,0 --port 8096

  then open http://phantom21:8096: both cameras with the detected tags drawn, and over each a
  grid of cells that turns green once the board has been seen there in a few frames, plus a
  near / mid / far tick (tag size in the image). Press RESET on the page between take A and
  take B. The cameras are read straight off the DMA.video rings, so SLAM need not run.
  The top line also shows each camera's frame rate and the head IMU (/dev/shm/raw_imu_actuals,
  ~500 Hz), in red "IMU MISSING" when it is silent: the IMU comes from the EtherCAT side, and on
  2026-10-20 two camera-IMU takes recorded zero IMU samples because EtherCAT was down.

Offline, on a recorded take (the EuRoC tree tools/mcap_to_euroc.py writes):

  python3 tools/calibration/calib_coverage.py --euroc /path/to/take [--cams 2,0] [--stride 3]

  prints the same numbers per camera, writes <take>/coverage/cam<N>.jpg, and exits 1 when a
  camera covered fewer than --min-cover of the cells, so a thin take is refused before kalibr.

--cams names DMA camera ids in OKVIS pair order (phantom21: 2,0 = OKVIS_DMA_CAMERAS); offline
they are the cam<N> folders of the tree. Board: config/calibration/target.yaml (36h11 tags).
"""
from __future__ import annotations

import argparse
import glob
import http.server
import json
import mmap
import os
import struct
import sys
import threading
import time

try:  # the board detection needs them; the stream-rate logic (and its stdlib-only test) does not
    import cv2
    import numpy as np
except ImportError:
    cv2 = np = None

GRID_COLS, GRID_ROWS = 8, 5
IMU_SHM = "/dev/shm/raw_imu_actuals"
IMU_STALE_S = 1.0                  # no new IMU sample for this long = the stream is gone
COVERED_FRAMES = 3                 # a cell is covered once a tag corner landed in it this often
NEAR, FAR = 0.058, 0.028           # tag edge / image width: ~1.2 m and ~2.5 m for a 10 cm tag at 1280 px
_QUEUE_MAGIC, _POOL_MAGIC = 0x444D415F51554555, 0x4652414D45504F4F   # "DMA_QUEU", "FRAMEPOO"
_META = "<qqQIIffIHHBBB"            # VideoFrameMeta after the meta slot's u64 sequence
_NV12, _MJPEG = 3, 0


class CamRing:
    """Newest frame of one DMA.video camera ring as 8-bit grey (the NV12 Y plane); None if torn."""

    def __init__(self, cam_id: int):
        self.meta_path = f"/dev/shm/video_meta_cam{cam_id}"
        self.pool_path = f"/dev/shm/video_frames_cam{cam_id}"
        self.meta = self.pool = self.ino = None

    def _open(self) -> None:
        with open(self.meta_path, "rb") as f:
            meta = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
            ino = os.fstat(f.fileno()).st_ino
        with open(self.pool_path, "rb") as f:
            pool = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
        if struct.unpack_from("<Q", meta, 0)[0] != _QUEUE_MAGIC or struct.unpack_from("<QQ", meta, 16) != (128, 64) \
                or struct.unpack_from("<Q", pool, 0)[0] != _POOL_MAGIC:
            raise OSError("camera ring layout mismatch")
        self.meta, self.pool, self.ino = meta, pool, ino
        self.slots, self.max_size, self.stride = struct.unpack_from("<QQQ", pool, 16)

    def latest(self):
        try:
            if self.meta is None or os.stat(self.meta_path).st_ino != self.ino:   # producer restarted
                self._open()
        except (OSError, ValueError, struct.error):
            self.meta = None
            return None
        head = struct.unpack_from("<Q", self.meta, 64)[0]
        if head == 0:
            return None
        off = 192 + ((head - 1) & 63) * 128
        if struct.unpack_from("<Q", self.meta, off)[0] != head:
            return None
        cap_ns, _, _, slot, size, _, _, _, w, h, _, fmt, _ = struct.unpack_from(_META, self.meta, off + 8)
        if slot >= self.slots:
            return None
        hdr = 64 + slot * self.stride
        seq = struct.unpack_from("<Q", self.pool, hdr)[0]
        size = min(size, struct.unpack_from("<I", self.pool, hdr + 12)[0])
        if size == 0 or size > self.max_size:
            return None
        data = np.frombuffer(self.pool[hdr + 64:hdr + 64 + size], np.uint8)   # a copy (seqlock read)
        if struct.unpack_from("<Q", self.pool, hdr)[0] != seq:
            return None
        if fmt == _NV12 and size >= w * h:
            return cap_ns, data[:w * h].reshape(h, w)
        if fmt == _MJPEG:
            img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
            return None if img is None else (cap_ns, img)
        return None


class RingRate:
    """Write rate and silence of one DMA ring, from its head counter (u64 at 0x40, the number of
    slots written; same header for the camera meta rings and the IMU ring)."""

    def __init__(self, path: str, now: float):
        self.path, self.mm, self.ino = path, None, None
        self.head, self.t_adv = None, now            # silent since start until the head moves
        self.win, self.rate = None, 0.0

    def _read(self):
        try:
            if self.mm is None or os.stat(self.path).st_ino != self.ino:
                with open(self.path, "rb") as f:
                    self.mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
                    self.ino = os.fstat(f.fileno()).st_ino
                self.head = self.win = None
            return struct.unpack_from("<Q", self.mm, 64)[0]
        except (OSError, ValueError, struct.error):
            self.mm = None
            return None

    def poll(self, now: float) -> tuple:
        """(rate_hz over the last ~1 s, seconds since the head last moved)."""
        head = self._read()
        if head is None:
            self.rate = 0.0
        else:
            if self.head is not None and head != self.head:
                self.t_adv = now
            self.head = head
            if self.win is None:
                self.win = (now, head)
            elif now - self.win[0] >= 1.0:
                self.rate = (head - self.win[1]) / (now - self.win[0])
                self.win = (now, head)
        return self.rate, now - self.t_adv


def imu_status(rate_hz: float, age_s: float) -> tuple:
    """(ok, one-line text) for the head IMU: missing when it has no rate or went quiet."""
    if rate_hz <= 0 or age_s > IMU_STALE_S:
        return False, f"IMU MISSING — start EtherCAT ({rate_hz:.0f} Hz, last sample {age_s:.1f} s ago)"
    return True, f"IMU OK {rate_hz:.0f} Hz"


def make_detector():
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    p = cv2.aruco.DetectorParameters()
    p.markerBorderBits = 2                  # kalibr's aprilgrid prints a double black border; with the
    p.minMarkerPerimeterRate = 0.015        # default of 1 almost nothing decodes (tools/mk2_xone_stereo_extrinsic.py)
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(d, p)


class Coverage:
    """Per-camera tally: which grid cells a tag corner has landed in, and how big the tags were."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.cells = np.zeros((GRID_ROWS, GRID_COLS), int)
        self.frames = self.frames_with_board = 0
        self.scale = {"near": 0, "mid": 0, "far": 0}
        self.edge_min, self.edge_max = float("inf"), 0.0

    def update(self, corners, w: int, h: int) -> int:
        """corners: the detector's list of (1, 4, 2) arrays. Returns the number of tags."""
        self.frames += 1
        if not corners:
            return 0
        self.frames_with_board += 1
        hit = set()
        edges = []
        for c in corners:
            q = c.reshape(4, 2)
            edges.append(float(np.mean(np.linalg.norm(np.roll(q, -1, 0) - q, axis=1))))
            for x, y in q:
                hit.add((min(GRID_ROWS - 1, max(0, int(y * GRID_ROWS / h))),
                         min(GRID_COLS - 1, max(0, int(x * GRID_COLS / w)))))
        for r, c in hit:
            self.cells[r, c] += 1
        e = float(np.median(edges)) / w
        self.edge_min, self.edge_max = min(self.edge_min, e * w), max(self.edge_max, e * w)
        self.scale["near" if e >= NEAR else "far" if e < FAR else "mid"] += 1
        return len(corners)

    def covered(self) -> float:
        return float((self.cells >= COVERED_FRAMES).mean())

    def stats(self) -> dict:
        return {"frames": self.frames, "frames_with_board": self.frames_with_board,
                "cells_covered": int((self.cells >= COVERED_FRAMES).sum()), "cells": GRID_ROWS * GRID_COLS,
                "covered": round(self.covered(), 3), "scale_frames": dict(self.scale),
                "tag_edge_px": [None if self.edge_min == float("inf") else round(self.edge_min, 1),
                                round(self.edge_max, 1)]}


def annotate(gray, corners, ids, cov: Coverage, label: str, n_tags: int):
    """The frame in colour with the tags, the coverage grid and one status line drawn on it."""
    h, w = gray.shape
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if corners:
        cv2.aruco.drawDetectedMarkers(img, corners, ids)
    shade = img.copy()
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            n = cov.cells[r, c]
            colour = (0, 170, 0) if n >= COVERED_FRAMES else (0, 200, 230) if n > 0 else (0, 0, 200)
            x0, y0 = c * w // GRID_COLS, r * h // GRID_ROWS
            x1, y1 = (c + 1) * w // GRID_COLS, (r + 1) * h // GRID_ROWS
            cv2.rectangle(shade, (x0, y0), (x1, y1), colour, -1)
            cv2.rectangle(img, (x0, y0), (x1, y1), (60, 60, 60), 1)
    cv2.addWeighted(shade, 0.18, img, 0.82, 0, img)
    s = cov.stats()
    ticks = " ".join(f"{k}{'+' if s['scale_frames'][k] >= COVERED_FRAMES else '-'}" for k in ("near", "mid", "far"))
    text = f"{label}  tags {n_tags}/35  cells {s['cells_covered']}/{s['cells']}  {ticks}  edge {s['tag_edge_px'][1]:.0f}px"
    cv2.putText(img, text, (12, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (12, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return img


PAGE = """<!doctype html><title>calib coverage</title>
<body style="margin:0;background:#111;color:#ddd;font:14px monospace">
<div style="padding:8px">AprilGrid coverage &nbsp; <button onclick="fetch('/reset')">RESET</button>
 &nbsp; <span id=imu></span> &nbsp; <span id=s></span></div>
<div style="display:flex;flex-wrap:wrap">%s</div>
<script>
const imgs=[...document.querySelectorAll('img')];
setInterval(()=>{imgs.forEach(i=>i.src=i.dataset.src+'?'+Date.now());
 fetch('/stats.json').then(r=>r.json()).then(j=>{const m=document.getElementById('imu');
 m.textContent=j.imu.text;m.style.cssText=j.imu.ok?'color:#4c4':'color:#fff;background:#c00;padding:2px 6px;font-weight:bold';
 document.getElementById('s').textContent=
 Object.entries(j.cams).map(([k,v])=>k+': '+(j.fps[k]??0).toFixed(1)+' fps, '+v.cells_covered+'/'+v.cells+' cells, '+
 ['near','mid','far'].map(s=>s+(v.scale_frames[s]>=3?'+':'-')).join(' ')).join('  |  ')})},300);
</script>"""


def serve(port: int, labels: list, frames: dict, stats: dict, lock: threading.Lock, reset):
    imgs = "".join(f'<img src="/{i}.jpg" data-src="/{i}.jpg" style="width:50%;min-width:480px">'
                   for i in range(len(labels)))
    page = (PAGE % imgs).encode()

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):   # quiet
            pass

        def do_GET(self):
            path = self.path.split("?", 1)[0]          # the page appends ?<time> to defeat caching
            if path == "/reset":
                reset()
                body, ctype = b"ok", "text/plain"
            elif path == "/stats.json":
                with lock:
                    body = json.dumps(dict(stats)).encode()
                ctype = "application/json"
            elif path.endswith(".jpg"):
                with lock:
                    body = frames.get(path[1:-4], b"")
                ctype = "image/jpeg"
            else:
                body, ctype = page, "text/html"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    srv = http.server.ThreadingHTTPServer(("", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def live(cams: list, port: int, hz: float) -> int:
    det = make_detector()
    rings = [CamRing(c) for c in cams]
    covs = [Coverage() for _ in cams]
    labels = [f"cam{i} (DMA {c})" for i, c in enumerate(cams)]
    now = time.time()
    imu_rate = RingRate(IMU_SHM, now)
    cam_rates = [RingRate(r.meta_path, now) for r in rings]
    frames, lock = {}, threading.Lock()
    stats = {"cams": {}, "fps": {}, "imu": {"ok": False, "text": "IMU ?"}}
    serve(port, labels, frames, stats, lock, lambda: [c.reset() for c in covs])
    print(f"serving http://<robot>:{port}  cams {cams}", flush=True)
    last_print, last_cap = now, [0] * len(cams)        # rates need ~1 s of history: no verdict before
    while True:
        t0 = time.time()
        imu_ok, imu_text = imu_status(*imu_rate.poll(t0))
        with lock:
            if t0 - now >= 1.0:
                stats["imu"] = {"ok": imu_ok, "text": imu_text}
            for l, r in zip(labels, cam_rates):
                stats["fps"][l] = round(r.poll(t0)[0], 1)
        for i, (ring, cov) in enumerate(zip(rings, covs)):
            got = ring.latest()
            if got is None or got[0] == last_cap[i]:
                continue
            last_cap[i], gray = got
            corners, ids, _ = det.detectMarkers(gray)
            n = cov.update(corners, gray.shape[1], gray.shape[0])
            img = annotate(gray, corners, ids, cov, labels[i], n)
            small = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA)
            ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
            with lock:
                if ok:
                    frames[str(i)] = jpg.tobytes()
                stats["cams"][labels[i]] = cov.stats()
        if t0 - last_print >= 2.0:
            last_print = t0
            print(imu_text if imu_ok else f"\033[1;31m{imu_text}\033[0m", flush=True)
            print("  ".join(f"{l}: {stats['fps'][l]:.1f} fps, {c.stats()['cells_covered']}/{GRID_ROWS * GRID_COLS} cells, "
                            + " ".join(f"{k}{'+' if v >= COVERED_FRAMES else '-'}" for k, v in c.stats()["scale_frames"].items())
                            for l, c in zip(labels, covs)), flush=True)
        time.sleep(max(0.0, 1.0 / hz - (time.time() - t0)))


def offline(tree: str, cams: list, stride: int, min_cover: float) -> int:
    det = make_detector()
    out = os.path.join(tree, "coverage")
    os.makedirs(out, exist_ok=True)
    ok = True
    for i, c in enumerate(cams):
        files = sorted(glob.glob(os.path.join(tree, f"cam{c}", "data", "*.png")))[::max(1, stride)]
        if not files:
            print(f"cam{c}: no frames under {tree}/cam{c}/data", file=sys.stderr)
            return 2
        cov, img = Coverage(), None
        for f in files:
            gray = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            corners, ids, _ = det.detectMarkers(gray)
            n = cov.update(corners, gray.shape[1], gray.shape[0])
            img = annotate(gray, corners, ids, cov, f"cam{c}", n)
        cv2.imwrite(os.path.join(out, f"cam{c}.jpg"), img)
        s = cov.stats()
        verdict = "ok" if s["covered"] >= min_cover else "THIN"
        ok &= verdict == "ok"
        print(f"cam{c}: {len(files)} frames, board in {s['frames_with_board']}, cells {s['cells_covered']}/{s['cells']}"
              f" ({s['covered']:.0%}), near/mid/far frames {s['scale_frames']['near']}/{s['scale_frames']['mid']}/"
              f"{s['scale_frames']['far']}, tag edge {s['tag_edge_px'][0]}-{s['tag_edge_px'][1]} px -> {verdict}")
    print(f"pictures in {out}")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cams", default="2,0", help="DMA camera ids in OKVIS pair order (offline: cam<N> folders)")
    ap.add_argument("--euroc", help="score a recorded take instead of the live rings")
    ap.add_argument("--port", type=int, default=8096)
    ap.add_argument("--hz", type=float, default=5.0, help="live detection rate")
    ap.add_argument("--stride", type=int, default=3, help="offline: every n-th frame")
    ap.add_argument("--min-cover", type=float, default=0.9, help="offline: covered-cell fraction to pass")
    a = ap.parse_args(argv)
    cams = [int(x) for x in a.cams.split(",")]
    if a.euroc:
        return offline(a.euroc, cams, a.stride, a.min_cover)
    return live(cams, a.port, a.hz)


if __name__ == "__main__":
    sys.exit(main())
