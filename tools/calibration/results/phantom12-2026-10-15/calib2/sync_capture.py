# Capture cameras B (left, saved as cam1) and D (right, saved as cam2) for N seconds with one of
# them as FSYNC sender, configured exactly as DMA.video's producer does (its own functions), and
# save every frame's luma plane as a lossless 8-bit PNG named by DepthAI's timestamp in ns.
# Writes no configuration: the producer must be stopped while this runs, and restarted after.
import sys, os, time, zlib, struct, threading, queue
sys.path.insert(0, "/app/src")
import numpy as np, depthai as dai
import multi_camera_producer as mcp
master, secs, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
c_mode = sys.argv[4] if len(sys.argv) > 4 else "-"
# "with-c": bottom camera C as in production (FSYNC listener, auto exposure); "with-c-fixed": C on the
# stereo pair's fixed exposure; "with-c-nosync": C on auto exposure but not on the sync pulse.
with_c = c_mode.startswith("with-c")
W, H = 1280, 800
def png_gray(path, y):
    raw = b"".join(b"\x00" + y[r].tobytes() for r in range(y.shape[0]))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", y.shape[1], y.shape[0], 8, 0, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 1)) + chunk(b"IEND", b""))
infos = dai.Device.getAllAvailableDevices()
if not infos: sys.exit("no OAK device found (is the producer still running?)")
dev = dai.Device(infos[0], dai.UsbSpeed.SUPER_PLUS)
pipe = dai.Pipeline(dev); pipe.setXLinkChunkSize(0)
mcp.setup_gpio_bridge(pipe)
q, dirs = {}, {"B": "cam1", "D": "cam2"}
for name in ("B", "D"):
    os.makedirs(f"{out}/{dirs[name]}/data", exist_ok=True)
    mode = dai.CameraControl.FrameSyncMode.OUTPUT if name == master else dai.CameraControl.FrameSyncMode.INPUT
    _, q[name] = mcp.create_camera_node(pipe, mcp.CAMERA_SOCKETS[name], 15, color=True,
                                        isp3a_fps=1, exposure=(3000, 800), fsync_mode=mode)
if with_c:   # as in production: socket C is an FSYNC listener on auto exposure; frames read and discarded
    c_exp = (3000, 800) if c_mode == "with-c-fixed" else "auto"
    c_sync = None if c_mode == "with-c-nosync" else dai.CameraControl.FrameSyncMode.INPUT
    _, q["C"] = mcp.create_camera_node(pipe, mcp.CAMERA_SOCKETS["C"], 15, color=True,
                                       isp3a_fps=(1 if c_exp != "auto" else 0), exposure=c_exp, fsync_mode=c_sync)
work = queue.Queue(maxsize=64); counts = {"B": 0, "D": 0, "dropped": 0}
def writer():
    while True:
        item = work.get()
        if item is None: return
        name, ts, y = item; png_gray(f"{out}/{dirs[name]}/data/{ts}.png", y)
threads = [threading.Thread(target=writer, daemon=True) for _ in range(2)]
for t in threads: t.start()
pipe.start(); t_end = time.time() + secs; last = time.time()
while time.time() < t_end:
    got = False
    for name in ("B", "D"):
        f = q[name].tryGet()
        if f is None: continue
        got = True
        if with_c and "C" in q: q["C"].tryGet()
        ts = int(f.getTimestamp().total_seconds() * 1e9)
        y = np.frombuffer(f.getData(), np.uint8)[:W * H].reshape(H, W).copy()
        try: work.put_nowait((name, ts, y)); counts[name] += 1
        except queue.Full: counts["dropped"] += 1
    if time.time() - last > 5:
        print(f"  t={secs - (t_end - time.time()):5.1f}s  B {counts['B']}  D {counts['D']}  dropped {counts['dropped']}", flush=True); last = time.time()
    if not got: time.sleep(0.002)
pipe.stop()
for _ in threads: work.put(None)
for t in threads: t.join()
print(f"CAPTURE-DONE master={master} c_mode={c_mode} B {counts['B']} D {counts['D']} dropped {counts['dropped']}")
