#!/usr/bin/env python3
"""The calibration capture on a Phantom robot, as steps that refuse to go on with something missing.

On 2026-10-20 two takes were recorded with the EtherCAT master down: the recorder opened the
cameras, found no IMU ring, logged one line and wrote 13 GB of frames with an empty /imu0. This
script is the procedure that cannot do that again. Run it on the robot host:

  sudo python3 calib_record.py             # steps 1-6, interactive (phantom21: --cams 2,0)
  sudo python3 calib_record.py --cams 3,0  # P10 / phantom11: ego_left, ego_right; the docker recorder
  sudo python3 calib_record.py preflight   # step 1 only
  sudo python3 calib_record.py take A      # one take (Enter starts and stops it; --seconds N instead)
  python3 calib_record.py verify DIR/session.mcap --take B

  1  preflight   dma-ethercat active; /dev/shm/raw_imu_actuals advancing at >= 400 Hz; the head IMU
                 (--imu, default NeckPitch) resolved to its slot in that ring; the two camera rings
                 advancing at >= 10 Hz; a recorder available and nothing recording; >= 40 GB free.
                 Each failure prints the fix. Nothing starts until all pass.
  2  viewer      tools/calibration/calib_coverage.py in the OKVIS image on :8096 (board detections, coverage).
  3  take A      stereo: board fixed, head slowly through the whole frame, near and far, tilted.
  4  verify A    the file itself: frames per camera, IMU samples at 500 Hz, gaps, monotonic stamps.
  5  take B      camera-IMU: board always in view, rotate about x/y/z and translate along them.
  6  verify B    as 4, plus the gyro peak per axis (the excitation kalibr needs).

While a take runs, the recorder's own list of channels still at 0 messages (its record_status.json,
read through nav_cli.record_status()["missing"]) is checked every 2 s from 5 s on: a take missing
/imu0 or a camera is stopped and its directory renamed *-BAD. The counters in the recorder's pod log
back that up: an IMU or camera count still 0 after 15 s does the same. verify renames a failing take
the same way, so a bad file is never handed to kalibr.

The IMU slot is resolved BY NAME (2026-10-27, P10). /raw_imu_actuals carries every IMU on the bus
in hardware order, and the recorder takes one by index, defaulting to 0. On phantom21 slot 0 is
NeckPitch, the head IMU; on P10 the two foot IMUs come first, so slot 0 is RightFoot and a take
with the default index is a foot IMU against head cameras -- a calibration kalibr completes without
a complaint. The names come from dma-ethercat's /dev/shm/dma_names.pb (field 5, `imus`, the same
hardware order). The agent's recorder pod passes no --imu-index, so --backend agent is refused
unless the head IMU is slot 0.

Two recorders (--backend, default auto):
  agent   the okvis-record pod the k0s agent starts (nav-cli record), as on phantom21.
  docker  okvis2x_app_record from the OKVIS image on this host's docker, with --cameras and
          --imu-index passed. For a robot with no OKVIS deployment (P10 has no okvis namespace).
  auto    the agent when it answers and the head IMU is slot 0; docker otherwise.
Each take also gets <take>/calib_meta.json: cameras, IMU name and slot, backend, image.

Deploy beside its helpers (it runs on the host, which has no checkout):
  scp tools/calibration/calib_record.py tools/calibration/calib_coverage.py interface/nav_cli.py \\
      interface/mcap_reader.py root@phantom21:/root/calibtools/
Runbooks: docs/phantom21-calibration-runbook.md, docs/p10-calibration-runbook.md.
"""
from __future__ import annotations

import argparse
import json
import mmap
import os
import struct
import subprocess
import sys
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "..", "interface")]
import mcap_reader  # noqa: E402
import nav_cli      # noqa: E402

IMU_RING = "/dev/shm/raw_imu_actuals"
NAMES_PB = "/dev/shm/dma_names.pb"
RECORD_DIR = "/data/okvis-recordings"
REC_CONTAINER = "calib-record"
# p21 tags the image both ways; P10 holds only the registry name.
IMAGES = ("okvis2x:on-device", "localhost:5443/okvis2x:on-device")
RING_MAGIC = 0x444D415F51554555      # "DMA_QUEU"
VIEWER_PORT = 8096
KUBECTL = ["k0s", "kubectl", "-n", "okvis"]
IMU_HZ, CAM_HZ = 500.0, 15.0
FREE_GB_MIN = 40.0
TAKES = {
    "A": "STEREO take. Board fixed and well lit. Move the head SLOWLY through the whole frame: every corner,\n"
         "   close (board filling most of the image) and far (2-3 m), tilted both ways. Stop when both grids\n"
         "   on the viewer are green and the near tick shows +. 60-120 s.",
    "B": "CAMERA-IMU take. Board always in view at 1-2 m. Rotate about all three axes (nod, shake, roll) and\n"
         "   translate along them (forward/back, side to side, up/down), briskly but without blur. 60-120 s.",
}


def ring_rate(path: str, dt: float = 0.5):
    """Samples per second the ring at `path` advances by; None if it is missing or not a ring."""
    try:
        with open(path, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
    except (OSError, ValueError):
        return None
    try:
        if struct.unpack_from("<Q", mm, 0)[0] != RING_MAGIC:
            return None
        h0 = struct.unpack_from("<Q", mm, 64)[0]
        time.sleep(dt)
        h1 = struct.unpack_from("<Q", mm, 64)[0]
        return (h1 - h0) / dt
    finally:
        mm.close()


def sh(cmd: list, timeout: float = 20.0) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def say(ok, what: str, fix: str = "") -> bool:
    print(f"  [{'ok' if ok else 'FAIL'}] {what}" + ("" if ok or not fix else f"\n         -> {fix}"))
    return bool(ok)


def imu_names(path: str = NAMES_PB) -> list:
    """IMU names in /raw_imu_actuals slot order: field 5 of dma.names.EntityNames (proto3)."""
    try:
        buf = open(path, "rb").read()
    except OSError:
        return []

    def varint(i):
        v = shift = 0
        while True:
            b = buf[i]
            v |= (b & 0x7F) << shift
            i += 1
            if not b & 0x80:
                return v, i
            shift += 7

    names, i = [], 0
    try:
        while i < len(buf):
            key, i = varint(i)
            field, wire = key >> 3, key & 7
            if wire == 0:
                _, i = varint(i)
            elif wire == 1:
                i += 8
            elif wire == 5:
                i += 4
            elif wire == 2:
                n, i = varint(i)
                if field == 5:
                    names.append(buf[i:i + n].decode("utf-8", "replace"))
                i += n
            else:
                return []
    except IndexError:
        return []
    return names


def okvis_image() -> str:
    for img in IMAGES:
        if sh(["docker", "image", "inspect", "--format", "{{.Id}}", img]).strip():
            return img
    return ""


def free_gb(path: str = RECORD_DIR) -> float:
    try:
        sv = os.statvfs(path)
    except OSError:
        return 0.0
    return sv.f_bavail * sv.f_frsize / 1e9


# Filled by preflight: which recorder, which IMU slot, which image. take() reads it.
RIG = {"backend": None, "imu": None, "imu_index": None, "image": ""}


# ---------------------------------------------------------------- 1. preflight
def preflight(cams: list, imu: str = "NeckPitch", backend: str = "auto") -> bool:
    print("1  preflight")
    active = sh(["systemctl", "is-active", "dma-ethercat"]).strip() == "active"
    ok = say(active, "dma-ethercat is active",
             "sudo systemctl start dma-ethercat; the IMU ring appears ~20 s later (RUNBOOK §2)")
    r = ring_rate(IMU_RING)
    # The master only creates this ring with --publish-debug-queues; P10 shipped without it. Read the
    # RUNNING master's argv, not the unit: an edited unit says nothing until the master restarts.
    fix = "no IMU: wait for the EtherCAT master, then check `ls /dev/shm | grep raw_imu`"
    if active and r is None:
        pid = sh(["systemctl", "show", "dma-ethercat", "-p", "MainPID", "--value"]).strip()
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                running = b"--publish-debug-queues" in f.read()
        except OSError:
            running = True
        unit = "--publish-debug-queues" in sh(["systemctl", "show", "dma-ethercat", "-p", "ExecStart", "--no-pager"])
        if not running:
            fix = ("the running dma-ethercat has no --publish-debug-queues, so it never creates this ring; "
                   + ("the unit has it now: restart the master (motors bus-reset; robot on its stand)" if unit else
                      "add it to ExecStart in /etc/systemd/system/dma-ethercat.service.d/debug.conf, "
                      "daemon-reload, restart the master"))
    ok &= say(r is not None and r >= 400, f"IMU ring {IMU_RING} advancing at {0 if r is None else r:.0f} Hz", fix)
    names = imu_names()
    idx = names.index(imu) if imu in names else None
    ok &= say(idx is not None, f"IMU '{imu}' is slot {idx} of {len(names)} in {IMU_RING} ({', '.join(names[:4])}, ...)",
              f"'{imu}' is not among {names or 'the IMUs (no ' + NAMES_PB + ')'}; pass --imu NAME")
    RIG.update(imu=imu, imu_index=idx)
    for c in cams:
        r = ring_rate(f"/dev/shm/video_meta_cam{c}")
        ok &= say(r is not None and r >= 10, f"camera {c} ring advancing at {0 if r is None else r:.1f} Hz",
                  "the DMA.video producer is not publishing this camera")
    st = nav_cli.record_status()
    if backend == "auto":
        backend = "agent" if st.get("ok") and idx == 0 else "docker"
    RIG["backend"] = backend
    print(f"  [info] recorder: {backend}")
    if backend == "agent":
        ok &= say(st.get("ok"), f"k0s agent answers record status ({st.get('err', 'ok')})",
                  "k0s kubectl -n okvis get pods: the nav-agent pod must be Running; or --backend docker")
        ok &= say(idx == 0, f"the agent's recorder takes IMU slot 0, and '{imu}' is slot {idx}",
                  "--backend docker passes --imu-index")
    else:
        RIG["image"] = okvis_image()
        ok &= say(RIG["image"], f"OKVIS image on this host's docker: {RIG['image'] or 'none of ' + ', '.join(IMAGES)}",
                  "pull or load the OKVIS image first")
        running = sh(["docker", "ps", "-q", "--filter", f"name=^{REC_CONTAINER}$"]).strip()
        ok &= say(not running, f"no {REC_CONTAINER} container running", f"docker stop -t 30 {REC_CONTAINER}")
    ok &= say(not (st.get("ok") and st.get("recording")), "the agent is not recording",
              f"a recording is running ({st.get('name')}): `nav-cli record stop` first")
    free = free_gb()
    ok &= say(free >= FREE_GB_MIN, f"{free:.0f} GB free on {RECORD_DIR} (need {FREE_GB_MIN:.0f})", "free space first")
    return ok


# ---------------------------------------------------------------- 2. viewer
def viewer_stats():
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{VIEWER_PORT}/stats.json", timeout=1.5) as r:
            return json.loads(r.read())
    except (OSError, ValueError):
        return None


def viewer_up(cams: list) -> None:
    print("2  coverage viewer")
    if viewer_stats() is None:
        sh(["docker", "rm", "-f", "calib-coverage"])
        sh(["docker", "run", "-d", "--rm", "--network", "host", "--name", "calib-coverage", "-v", "/dev/shm:/dev/shm",
            "-v", f"{HERE}:/t", "--entrypoint", "python3", RIG["image"] or okvis_image(), "/t/calib_coverage.py",
            "--cams", ",".join(map(str, cams)), "--port", str(VIEWER_PORT)], timeout=60)
        for _ in range(30):
            if viewer_stats() is not None:
                break
            time.sleep(1)
    ip = sh(["hostname", "-I"]).split()
    say(viewer_stats() is not None, f"viewer at http://{ip[0] if ip else '<robot>'}:{VIEWER_PORT}",
        "docker logs calib-coverage")


def viewer_line(st) -> str:
    """One line from the viewer's /stats.json: {"cams": {label: coverage}, "fps": {label: Hz}, "imu": {"text"}}."""
    if not st or "cams" not in st:
        return "viewer: -"
    return st["imu"]["text"] + "  " + "  ".join(
        f"{k} {st['fps'].get(k, 0):.0f}fps {v['cells_covered']}/{v['cells']} "
        + "".join(s[0] + ("+" if v["scale_frames"][s] >= 3 else "-") for s in ("near", "mid", "far"))
        for k, v in st["cams"].items())


# ---------------------------------------------------------------- 3/5. a take
def recorder_counts() -> tuple:
    """(counts from the recorder's last progress line, True if it said it has no IMU)."""
    if RIG["backend"] == "docker":
        p = subprocess.run(["docker", "logs", "--tail=40", REC_CONTAINER], capture_output=True, text=True)
        log = p.stdout + p.stderr   # glog writes to stderr
    else:
        log = sh(KUBECTL + ["logs", "okvis-record", "--tail=40"])
    counts = {}
    for line in log.splitlines():
        if "[record-mcap] t=" in line:
            counts = {k: int(v) for k, v in (t.split("=") for t in line.split("[record-mcap] ")[1].split()
                                             if "=" in t and t.split("=")[1].isdigit())}
    return counts, "Could not open IMU shm" in log


def mark_bad(d: str) -> str:
    bad = d.rstrip("/") + "-BAD"
    try:
        os.rename(d, bad)
    except OSError:
        return d
    return bad


def docker_start(cams: list) -> dict:
    name = "rec-" + time.strftime("%Y%m%d-%H%M%S")
    sh(["docker", "rm", "-f", REC_CONTAINER])          # the previous take's stopped container
    # ipc=host: the recorder shm_opens the DMA rings, as the pod's hostIPC does.
    cid = sh(["docker", "run", "-d", "--name", REC_CONTAINER, "--runtime", "nvidia", "--ipc", "host",
              "--network", "host", "-v", "/dev/shm:/dev/shm", "-v", f"{RECORD_DIR}:/recordings", RIG["image"],
              "record", f"/recordings/{name}", "--duration", "0", "--cameras", ",".join(map(str, cams)),
              "--imu-index", str(RIG["imu_index"])], timeout=60).strip()
    if not cid:
        return {"ok": False, "err": f"docker run failed: `docker logs {REC_CONTAINER}`"}
    return {"ok": True, "name": name, "dir": os.path.join(RECORD_DIR, name)}


def docker_missing(d: str) -> list:
    try:
        with open(os.path.join(d, "record_status.json")) as f:
            return json.load(f).get("missing") or []
    except (OSError, ValueError):
        return []


def write_meta(d: str, cams: list) -> None:
    meta = {"cams": cams, "imu": RIG["imu"], "imu_index": RIG["imu_index"], "backend": RIG["backend"],
            "image": RIG["image"], "host": os.uname().nodename, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    for _ in range(10):   # the recorder creates the directory a moment after it starts
        if os.path.isdir(d):
            with open(os.path.join(d, "calib_meta.json"), "w") as f:
                json.dump(meta, f, indent=1)
            return
        time.sleep(0.5)
    print(f"   [warn] {d} did not appear; calib_meta.json not written")


def take(letter: str, cams: list, seconds: float | None) -> bool:
    print(f"{3 if letter == 'A' else 5}  take {letter}\n   {TAKES[letter]}")
    if seconds is None:
        input("   Press Enter to START recording... ")
        print("   starting the recorder (up to 90 s)... do NOT press Enter again: the next Enter STOPS the take", flush=True)
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{VIEWER_PORT}/reset", timeout=1.5).read()
    except OSError:
        pass
    docker = RIG["backend"] == "docker"
    st = docker_start(cams) if docker else nav_cli.record_start()
    if not st.get("ok"):
        print(f"   FAIL could not start: {st.get('err')}")
        return False
    d, t0 = st["dir"], time.time()
    write_meta(d, cams)
    print(f"   recording {st['name']}  (Enter stops it)" if seconds is None else f"   recording {st['name']} for {seconds:.0f} s")
    stop = threading.Event()
    if seconds is None:
        threading.Thread(target=lambda: (sys.stdin.readline(), stop.set()), daemon=True).start()
    # The recorder before this fix also listed /okvis_depth/image whenever SLAM was down; only these count.
    need = {"/imu0"} | {f"/cam{c}/image" for c in cams}
    bad = ""
    try:
        while not stop.is_set():
            time.sleep(2)
            el = time.time() - t0
            missing = [m for m in (docker_missing(d) if docker else nav_cli.record_status().get("missing") or [])
                       if m in need]
            counts, no_imu = recorder_counts()
            mb = os.path.getsize(os.path.join(d, "session.mcap")) / 1e6 if os.path.exists(os.path.join(d, "session.mcap")) else 0
            left = f"{60 - el:3.0f} s to the 60 s minimum" if el < 60 else "long enough: Enter stops"
            print(f"   t={el:5.0f}s  [{left}]  {mb:7.0f} MB  missing={','.join(missing) or 'none'}  "
                  + " ".join(f"{k}={v}" for k, v in counts.items()) + "  " + viewer_line(viewer_stats()))
            if el >= 5 and missing:   # the recorder's 3 s grace, plus margin
                bad = f"the recorder reports {', '.join(missing)} at 0 messages"
            elif no_imu:
                bad = "the recorder has no IMU ring (it logged 'Could not open IMU shm')"
            elif el > 15 and counts and (counts.get("imu", 0) == 0 or any(counts.get(f"cam{c}", 0) == 0 for c in cams)):
                bad = f"after 15 s the recorder counts are {counts}: a stream is dead"
            if bad or (seconds is not None and el >= seconds):
                break
    except KeyboardInterrupt:
        print("   interrupted; stopping the recording")
    if docker:   # SIGTERM; the recorder closes the mcap (footer + index) on it
        sh(["docker", "stop", "-t", "30", REC_CONTAINER], timeout=45)
    else:
        nav_cli.record_stop()
    time.sleep(2)
    if bad:
        print(f"   FAIL {bad}\n   -> renamed to {mark_bad(d)}")
        return False
    return verify(os.path.join(d, "session.mcap"), letter, cams)


# ---------------------------------------------------------------- 4/6. verify the file
def verify(mcap: str, letter: str, cams: list) -> bool:
    print(f"{4 if letter == 'A' else 6}  verify {mcap}")
    try:
        with open(os.path.join(os.path.dirname(mcap), "calib_meta.json")) as f:
            meta = json.load(f)
        print(f"  [info] cams {meta['cams']}, IMU {meta['imu']} (slot {meta['imu_index']}), {meta['backend']} recorder")
    except (OSError, ValueError, KeyError):
        print("  [info] no calib_meta.json: which IMU slot this take holds is not recorded")
    stamps = {}
    gyro_peak = [0.0, 0.0, 0.0]
    imu_gap_max = [0.0]
    last_imu = [None]

    def on_msg(ch, log_ns, pub_ns, data):
        stamps.setdefault(ch.topic, [log_ns, log_ns, 0])
        s = stamps[ch.topic]
        s[1] = max(s[1], log_ns)
        s[2] += 1
        if ch.topic == "/imu0":
            m = json.loads(bytes(data))
            for i, k in enumerate(("wx", "wy", "wz")):
                gyro_peak[i] = max(gyro_peak[i], abs(float(m[k])))
            t = int(m["timestamp_ns"])
            if last_imu[0] is not None:
                imu_gap_max[0] = max(imu_gap_max[0], (t - last_imu[0]) * 1e-9)
            last_imu[0] = t

    try:
        sess = mcap_reader.read_session(mcap, on_msg)
    except (OSError, ValueError) as e:
        print(f"   FAIL unreadable: {e}")
        mark_bad(os.path.dirname(mcap))
        return False
    ok = say(not sess.truncated, "file complete (not truncated)", "the recorder was killed hard; redo the take")
    # Only the channels kalibr reads set the duration. /joints keeps arriving for ~10 s while the
    # recorder shuts down (P10, 2026-10-28), and timing every rate against that span read a clean
    # 500 Hz IMU as 443 Hz and failed the take.
    need = {"/imu0"} | {f"/cam{c}/image" for c in cams}
    durs = [(stamps[t][1] - stamps[t][0]) * 1e-9 for t in stamps if t in need and stamps[t][2] > 1]
    dur = max(durs) if durs else 0.0
    ok &= say(dur >= 60, f"duration {dur:.0f} s (>= 60)", "too short for kalibr")
    for c in cams:
        n = stamps.get(f"/cam{c}/image", [0, 0, 0])[2]
        hz = n / dur if dur else 0
        ok &= say(hz >= 10, f"cam{c}: {n} frames, {hz:.1f} Hz (>= 10)", "frames missing: check the video producer")
    # The IMU is what take B is for; take A (stereo) only needs the cameras, so there it is reported.
    n_imu = stamps.get("/imu0", [0, 0, 0])[2]
    hz = n_imu / dur if dur else 0
    peak = " ".join(f"{a}={p:.2f}" for a, p in zip("xyz", gyro_peak))
    if letter == "B":
        ok &= say(hz >= 450, f"IMU: {n_imu} samples, {hz:.0f} Hz (>= 450)",
                  "no or thin IMU: dma-ethercat was not publishing; redo the take after preflight passes")
        if n_imu > 1:
            ok &= say(imu_gap_max[0] <= 0.02, f"IMU max gap {imu_gap_max[0] * 1e3:.1f} ms (<= 20)", "samples dropped")
            ok &= say(min(gyro_peak) >= 0.3, f"gyro peak rad/s: {peak} (each >= 0.3)", "an axis was never rotated; redo B")
            if 0.3 <= min(gyro_peak) < 0.8:
                print("  [warn] gyro peak below 0.8 rad/s on some axis: kalibr's IMU-camera translation gets weak")
    else:
        print(f"  [info] IMU: {n_imu} samples, {hz:.0f} Hz, max gap {imu_gap_max[0] * 1e3:.1f} ms, gyro peak rad/s {peak}"
              " (not required for the stereo take)")
    if not ok:
        print(f"   FAIL -> renamed to {mark_bad(os.path.dirname(mcap))}")
    else:
        print(f"   PASS {os.path.dirname(mcap)}  ({os.path.getsize(mcap) / 1e9:.1f} GB)")
    return ok


# ---------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", nargs="?", default="all", choices=["all", "preflight", "take", "verify"])
    ap.add_argument("arg", nargs="?", help="take: A|B; verify: path to session.mcap")
    ap.add_argument("--take", choices=["A", "B"], default="A", help="verify: which bar to apply")
    ap.add_argument("--cams", default="2,0", help="DMA camera ids, OKVIS pair order")
    ap.add_argument("--seconds", type=float, help="take: stop after N s instead of on Enter")
    ap.add_argument("--imu", default="NeckPitch", help="the IMU the cameras ride on, by its dma-ethercat name")
    ap.add_argument("--backend", choices=["auto", "agent", "docker"], default="auto", help="which recorder")
    a = ap.parse_args(argv)
    cams = [int(x) for x in a.cams.split(",")]
    if a.step == "verify":
        if not a.arg:
            ap.error("verify needs the session.mcap path")
        return 0 if verify(a.arg, a.take, cams) else 1
    if a.step == "preflight":
        return 0 if preflight(cams, a.imu, a.backend) else 2
    if a.step == "take":
        if a.arg not in TAKES:
            ap.error("take needs A or B")
        if not preflight(cams, a.imu, a.backend):
            return 2
        viewer_up(cams)
        return 0 if take(a.arg, cams, a.seconds) else 1
    if not preflight(cams, a.imu, a.backend):
        print("\nfix the FAIL lines above and run again; nothing was recorded")
        return 2
    viewer_up(cams)
    results = {}
    for letter in ("A", "B"):
        results[letter] = take(letter, cams, a.seconds)
        if not results[letter]:
            print(f"\ntake {letter} failed; fix the cause and run `take {letter}` again")
            return 1
    print("\nboth takes PASS. Next, on the dev box (docs/phantom21-calibration-runbook.md §2):\n"
          "  python3 tools/mcap_to_euroc.py <take A dir>/session.mcap A/ ; same for B")
    return 0


if __name__ == "__main__":
    sys.exit(main())
