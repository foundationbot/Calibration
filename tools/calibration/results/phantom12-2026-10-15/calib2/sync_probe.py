# Ask DepthAI itself when each camera exposed. Configures sockets B and D exactly as
# DMA.video's producer does (its own create_camera_node / setup_gpio_bridge), with one
# of them as FSYNC sender, captures a few seconds and compares, per frame pair, every
# timestamp DepthAI reports. Needs the producer stopped (one process per device).
import sys, time, statistics as st
sys.path.insert(0, "/app/src")
import depthai as dai
import multi_camera_producer as mcp
master = sys.argv[1]; secs = float(sys.argv[2]) if len(sys.argv) > 2 else 8
infos = dai.Device.getAllAvailableDevices()
if not infos: sys.exit('no OAK device found (is the producer still running?)')
dev = dai.Device(infos[0], dai.UsbSpeed.SUPER_PLUS)
pipe = dai.Pipeline(dev); pipe.setXLinkChunkSize(0)
mcp.setup_gpio_bridge(pipe)
q = {}
for name in ("B", "D"):
    mode = dai.CameraControl.FrameSyncMode.OUTPUT if name == master else dai.CameraControl.FrameSyncMode.INPUT
    _, q[name] = mcp.create_camera_node(pipe, mcp.CAMERA_SOCKETS[name], 15, color=True,
                                        isp3a_fps=1, exposure=(3000, 800), fsync_mode=mode)
pipe.start()
off = getattr(dai, "CameraExposureOffset", None)
rows = {"B": [], "D": []}; t_end = time.time() + secs
while time.time() < t_end:
    for name in ("B", "D"):
        f = q[name].tryGet()
        if f is None: continue
        r = {"seq": f.getSequenceNum(), "ts": f.getTimestamp().total_seconds(),
             "tsdev": f.getTimestampDevice().total_seconds(), "exp_ms": f.getExposureTime().total_seconds() * 1e3}
        if off is not None:
            for k in ("START", "MIDDLE", "END"):
                try: r["exp_" + k] = f.getTimestampDevice(getattr(off, k)).total_seconds()
                except Exception as e: r["exp_" + k] = None
        rows[name].append(r)
    time.sleep(0.002)
pipe.stop()
print(f"master={master}: frames B {len(rows['B'])}, D {len(rows['D'])}; exposure ms B {st.median(r['exp_ms'] for r in rows['B']):.2f}, D {st.median(r['exp_ms'] for r in rows['D']):.2f}")
for key in ("ts", "tsdev", "exp_START", "exp_MIDDLE", "exp_END"):
    if not rows["B"] or rows["B"][0].get(key) is None: print(f"  {key}: not available"); continue
    d = []
    for rb in rows["B"]:
        rd = min(rows["D"], key=lambda r: abs(r["tsdev"] - rb["tsdev"]))
        if abs(rd["tsdev"] - rb["tsdev"]) < 0.03: d.append((rd[key] - rb[key]) * 1e3)
    print(f"  D minus B, {key:10s}: median {st.median(d):+8.3f} ms  (min {min(d):+.3f}, max {max(d):+.3f}, n {len(d)})")
print("PROBE-DONE")
