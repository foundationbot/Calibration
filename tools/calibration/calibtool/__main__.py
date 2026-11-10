"""`python -m calibtool <command>`: the calib image's entry point (tools/calibration/calibtool/RUNTIME.md).

    check [--no-dma]       the runtime self-check: Python, numpy/scipy/OpenCV (cv2.aruco's AprilTag 36h11 detector),
                           PyYAML, every DEPS.txt module importable, the repo files the package reads at their repo
                           paths, DMA.common's dma_ipc binding (readers made before their ring exists must not block),
                           and what /dev/shm shows. Exit 0 = every check passed; the /dev/shm line is information only.
    script NAME [ARGS...]  run pipeline script tools/calibration/scripts/NAME.py with ARGS, as `python NAME.py ARGS`
    MODULE [ARGS...]       run calibtool.MODULE as a program (`python -m calibtool.MODULE ARGS`), e.g. livecfg,
                           protocol, and calibd's `daemon` once it lands (plan section 3)

Stdlib only at import: devel's pytest-only CI job collects tools/, and this file must not need numpy to load."""
import importlib.util
import os
import platform
import runpy
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent              # tools/calibration/calibtool
CAL = PKG.parent                                    # tools/calibration
REPO = CAL.parents[1]                               # repo root (the image keeps the repo layout under it)

# dma_ipc names the robot-side code reads: the camera rings (08_neck_grid_direct.Cams), the config/actuals/command
# queues (08, armsnap). A DMA.common without them cannot run calibd, so the check fails rather than the robot.
DMA_NAMES = ("VideoMetaQueue", "VideoFramePool", "video_meta_queue_name", "video_frame_pool_name", "SensorQueue",
             "ConfigQueue", "CommandDataQueue")
# DMA.common's PENDING readers: made before their producer, they neither wait nor raise (as docker/perception.Dockerfile
# checks the same classes).
DMA_PENDING = ("VideoMetaQueue", "VideoFramePool")
# Repo files outside tools/calibration that the package reads by repo-relative path.
REPO_FILES = ("config/foundation_rig/20250206_phantom_mk1.urdf",   # calibtool.placement.URDF (neck chain)
              "interface/mcap_reader.py")                           # scripts/necktake.py, 15_dump_corners.py


def _deps():
    lines = (PKG / "DEPS.txt").read_text().splitlines()
    return [ln.split("#")[0].strip() for ln in lines if ln.split("#")[0].strip()]


def check(argv):
    import argparse
    ap = argparse.ArgumentParser(prog="python -m calibtool check")
    ap.add_argument("--no-dma", action="store_true", help="skip dma_ipc (a laptop or workstation has none)")
    a = ap.parse_args(argv)
    failed = []

    def item(name, fn):
        try:
            print(f"ok    {name}: {fn()}")
        except Exception as e:  # noqa: BLE001  every failure is reported; the exit code says it
            failed.append(name)
            print(f"FAIL  {name}: {type(e).__name__}: {e}")

    def py():
        assert sys.version_info[:2] == (3, 12), \
            f"Python {platform.python_version()}: the robots' dma_ipc is built for 3.12"
        return f"{platform.python_version()} {platform.machine()} ({sys.executable})"

    def libs():
        import cv2
        import numpy
        import scipy
        import yaml
        return (f"numpy {numpy.__version__} | scipy {scipy.__version__} | opencv {cv2.__version__} | "
                f"pyyaml {yaml.__version__}")

    def aruco():
        import cv2
        import numpy as np
        det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
                                      cv2.aruco.DetectorParameters())
        _, ids, _ = det.detectMarkers(np.full((64, 64), 128, np.uint8))
        assert ids is None, ids
        return "ArucoDetector(DICT_APRILTAG_36h11) runs"

    def deps():
        sys.path.insert(0, str(CAL))
        from calibtool import paths  # puts scripts/ on sys.path
        mods = [Path(d).stem for d in _deps() if d.startswith("scripts/")]
        bad = []
        for m in mods:
            try:
                paths.import_script(m)
            except Exception as e:                   # noqa: BLE001
                bad.append(f"{m} ({type(e).__name__}: {e})")
        assert not bad, "; ".join(bad)
        return f"{len(mods)} DEPS.txt modules import"

    def repo_files():
        missing = [f for f in REPO_FILES if not (REPO / f).is_file()]
        assert not missing, f"not under {REPO}: {missing}"
        return f"{len(REPO_FILES)} under {REPO}"

    def dma():
        import dma_ipc
        absent = [n for n in DMA_NAMES if not hasattr(dma_ipc, n)]
        assert not absent, f"dma_ipc has no {absent}: DMA.common too old for the calibration tool"
        for n in DMA_PENDING:
            r = getattr(dma_ipc, n)("/calibtool_check_no_such_ring", wait=False)
            assert not r.attached and r.attach_state.startswith("pending"), (n, r.attach_state)
        return dma_ipc.__file__

    item("python", py)
    item("numeric libs", libs)
    item("cv2.aruco", aruco)
    item("DEPS.txt", deps)
    item("repo files", repo_files)
    if a.no_dma:
        print("skip  dma_ipc (--no-dma)")
    else:
        item("dma_ipc", dma)
    try:
        shm = sorted(os.listdir("/dev/shm"))
    except OSError as e:
        shm = []
        print(f"info  /dev/shm: unreadable ({e})")
    else:
        print(f"info  /dev/shm: {len(shm)} entries" + ("" if shm else " -- on a robot, run with -v /dev/shm:/dev/shm"))
    print("check: FAILED " + ", ".join(failed) if failed else "check: all passed")
    return 1 if failed else 0


def main(argv):
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        return 0 if argv else 2
    cmd, rest = argv[0], argv[1:]
    if cmd == "check":
        return check(rest)
    sys.path.insert(0, str(CAL))
    if cmd == "script":
        if not rest:
            print("usage: python -m calibtool script NAME [ARGS...]", file=sys.stderr)
            return 2
        path = CAL / "scripts" / f"{rest[0].removesuffix('.py')}.py"
        if not path.is_file():
            print(f"no pipeline script {path}", file=sys.stderr)
            return 2
        sys.argv = [str(path), *rest[1:]]
        sys.path.insert(0, str(path.parent))
        runpy.run_path(str(path), run_name="__main__")
        return 0
    if importlib.util.find_spec(f"calibtool.{cmd}") is None:
        print(f"unknown command {cmd!r}: no module calibtool.{cmd} (python -m calibtool --help)", file=sys.stderr)
        return 2
    sys.argv = [f"calibtool.{cmd}", *rest]
    runpy.run_module(f"calibtool.{cmd}", run_name="__main__", alter_sys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
