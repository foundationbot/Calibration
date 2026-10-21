#!/usr/bin/env python3
"""Step 10 — turn a `record.sh` run into Kalibr folder trees.

This is the step `docs/kalibr-calibration.md` skips (it jumps 1 -> 3). Packaged
from the script that actually produced the 2026-08-22 calibration, which ran on
the robot as `/root/terrain_aware/calibration/build_trees.py`.

INPUT   `record.sh` output, EuRoC/ASL layout:
            <run>/cam1/data/<capture_ns>.png   <run>/cam1/timestamps_full.csv
            <run>/cam2/data/<capture_ns>.png   <run>/cam2/timestamps_full.csv
            <run>/imu0/data.csv
OUTPUT  a flat tree `kalibr_bagcreater` accepts:
            <tree>/cam0/<ts>.png  <tree>/cam1/<ts>.png  [<tree>/imu0.csv]

CAMERA MAPPING — GET THIS WRONG AND EVERYTHING DOWNSTREAM IS SILENTLY MIRRORED
-----------------------------------------------------------------------------
Read from config/rig.yaml `cameras.kalibr` (kalibr order). For the mk13 head:

    record dir cam1  ->  kalibr cam0   (DMA video queue 1 = physical RIGHT eye)
    record dir cam2  ->  kalibr cam1   (DMA video queue 2 = physical LEFT  eye)

Kalibr itself does not care, but step 70 does: it writes OKVIS cam0 from KALIBR
cam1, the 2026-08-16 stereo-role swap. Chain it through and OKVIS cam0 ends up
the physical LEFT eye on queue 2, which is what `mk13_stereo.yaml` declares.

Swapping this is the same class of fault as swapping `socket` in
head_camera.json: nothing errors, the de-roll inverts instead of cancelling,
stereo triangulates behind the camera and landmark initialisation collapses
minutes into a run. The mapping is recorded in <tree>/SOURCE.json and step 70
reads it back rather than assuming. Verify by disparity sign, never by config
(see the README, "Camera identity").

CAMERA CLOCK — STAMPS ARE USED AS RECORDED (no rebase), unless they are device time
----------------------------------------------------------------------------------
DMA.video's producer converts the DepthAI device time to the HOST clock before
publishing, so the capture stamp in `timestamps_full.csv` is already on the
IMU's clock (CLOCK_REALTIME). publish - capture is then pure delivery latency,
29-30 ms on phantom12 (measured 2026-10-14). The record app documents the same
and warns not to rebase; live OKVIS stopped rebasing on 2026-09-16.

This step used to add `min(publish_ns - capture_ns)` to every camera stamp in
imucam mode. On host-clock stamps that shifts every frame ~24-29 ms late: Kalibr
reports the shift back as `timeshift_cam_imu` (phantom13 2026-08-22: -23.6 /
-24.6 ms), step 70 bakes it into `image_delay: 0.0241`, and current OKVIS, which
no longer rebases, then reads every camera ~24 ms early.

`clock_offset_ns` decides per run:
  host clock    every camera's median(publish - capture) is in [0, 1 s)
                -> offset 0, stamps used as recorded. Expect Kalibr's
                timeshift within a few ms (phantom21 on raw stamps: +1.2 ms).
  device clock  anything else -> the old estimator, ONE COMMON offset
                min(publish - capture) for both cameras (common, not per-camera,
                so inter-camera sync is untouched).

stereo mode never shifts stamps: no IMU is involved, and both cameras share one
clock so pairs stay tight (0.03 ms measured).

`stereo` mode also reserves HELD-OUT pairs that never enter the fit — step 60
needs them, and a calibration validated on its own training frames is worthless.

Frames are written as single-channel greyscale PNG. Kalibr's chessboard detector
works on intensity and converts internally with the same BT.601 weights, so this
costs no corner accuracy and roughly halves the transfer off the robot.

Usage
-----
    # stereo set: 450 calibration pairs + 25 held-out
    python3 10_build_trees.py --src /data/okvis-recordings/calib-stereo \
                              --dst tree/stereo --mode stereo

    # camera-IMU set: every frame, on the IMU clock (see CAMERA CLOCK above)
    python3 10_build_trees.py --src /data/okvis-recordings/calib-imucam \
                              --dst tree/imucam --mode imucam
    #   --crop-from/--crop-to trim a static lead-in that adds no observability
"""
from __future__ import annotations

import argparse, bisect, csv, json, os, sys
from multiprocessing import Pool
from pathlib import Path

import rigcfg


def read_full(src: Path, cam: str) -> list[tuple[int, int]]:
    """(capture_ns, publish_ns) rows from timestamps_full.csv."""
    p = src / cam / "timestamps_full.csv"
    if not p.is_file():
        sys.exit(f"ERROR: {p} not found. It holds both clocks (capture, publish) that\n"
                 f"       decide whether camera stamps need a rebase. tools/mcap_to_euroc.py\n"
                 f"       writes it next to cam*/data/ when exporting a session.mcap.")
    rows = []
    with p.open() as f:
        for r in csv.reader(f):
            if r and r[0].lstrip("#").strip().isdigit():
                rows.append((int(r[0]), int(r[1])))
    if not rows:
        sys.exit(f"ERROR: no rows parsed from {p}")
    return rows


def clock_offset_ns(full: dict[str, list[tuple[int, int]]]) -> int:
    """Offset to ADD to camera capture stamps to put them on the IMU (host) clock.

    0 when the stamps already are host time (every camera's median
    publish - capture in [0, 1 s): pure delivery latency). Otherwise the common
    min(publish - capture) over both cameras, the old device-clock estimator.
    """
    delays = {c: sorted(p - d for d, p in rows) for c, rows in full.items()}
    if all(0 <= v[len(v) // 2] < 1_000_000_000 for v in delays.values()):
        return 0
    return min(v[0] for v in delays.values())


def convert(job):
    src, dst, grey = job
    if os.path.exists(dst):
        return 1
    try:
        if grey:
            from PIL import Image
            Image.open(src).convert("L").save(dst, optimize=True)
        else:
            os.link(src, dst)
        return 1
    except OSError:
        try:
            import shutil; shutil.copy2(src, dst); return 1
        except Exception as e:                      # noqa: BLE001
            print("  FAIL", src, e); return 0
    except Exception as e:                          # noqa: BLE001
        print("  FAIL", src, e); return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="a record.sh run dir")
    ap.add_argument("--dst", type=Path, required=True, help="kalibr tree to write")
    ap.add_argument("--mode", choices=("stereo", "imucam"), required=True)
    ap.add_argument("--target", type=int, default=None,
                    help="stereo: calibration pairs (default: rig capture.calib_pairs)")
    ap.add_argument("--heldout", type=int, default=None,
                    help="stereo: reserved pairs (default: rig capture.heldout_pairs)")
    ap.add_argument("--crop-from", type=float, default=0.0, help="imucam: seconds")
    ap.add_argument("--crop-to", type=float, default=1e9, help="imucam: seconds")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--link", action="store_true",
                    help="hard-link the original PNGs instead of converting to grey")
    rigcfg.add_arg(ap)
    args = ap.parse_args()

    rig = rigcfg.load(args.rig)
    CAM_MAP = rig.cam_map()                       # [(record dir, kalibr name)]
    PAIR_TOL_NS = int(float(rig.capture["pair_tol_ms"]) * 1e6)
    if args.target is None:
        args.target = int(rig.capture["calib_pairs"])
    if args.heldout is None:
        args.heldout = int(rig.capture["heldout_pairs"])

    src, dst, grey = args.src, args.dst, not args.link
    if not src.is_dir():
        sys.exit(f"ERROR: {src} is not a directory")

    full = {c: read_full(src, c) for c, _ in CAM_MAP}
    off = {c: min(p - d for d, p in rows) for c, rows in full.items()}
    print(f"min(publish-capture): "
          + "  ".join(f"{c}={v/1e6:.3f} ms" for c, v in off.items())
          + f"  (diff {abs(list(off.values())[0]-list(off.values())[1])/1e6:.3f} ms)")
    common = clock_offset_ns(full)
    if common == 0:
        print("camera stamps are already on the host clock (that gap is delivery latency): "
              "used as recorded, no rebase")
    else:
        print(f"camera stamps look like a DEVICE clock: imucam frames rebased by the common "
              f"min(publish-capture) = {common/1e6:.3f} ms")

    jobs: list[tuple[str, str, bool]] = []
    note = "; ".join(f"kalibr {kal} <- record {rec} <- DMA queue {c['dma_queue']} = "
                     f"physical {c['side'].upper()}"
                     for (rec, kal), c in zip(CAM_MAP, rig.kalibr))
    meta: dict = {"source_run": str(src.resolve()), "mode": args.mode,
                  "rig": str(rig.path),
                  "cam_map": {k: v for k, v in CAM_MAP},
                  "cameras": rig.kalibr,
                  "note": note + ". Step 70 checks this against the rig config. "
                          "Do not hand-edit.",
                  "clock_offset_ns": common,
                  "camera_clock": "host (used as recorded)" if common == 0
                                  else "device (rebased by common min(publish-capture))"}

    if args.mode == "stereo":
        t1 = [d for d, _ in full[CAM_MAP[0][0]]]
        t2 = [d for d, _ in full[CAM_MAP[1][0]]]
        pairs, gaps = [], []
        for a in t1:
            i = bisect.bisect_left(t2, a)
            cands = [t2[j] for j in (i - 1, i) if 0 <= j < len(t2)]
            if not cands:
                continue
            b = min(cands, key=lambda x: abs(x - a))
            if abs(b - a) <= PAIR_TOL_NS:
                pairs.append((a, b)); gaps.append(abs(b - a))
        if not pairs:
            sys.exit(f"ERROR: no stereo pairs within {PAIR_TOL_NS/1e6:g} ms — "
                     "the two cameras are not synced")
        print(f"paired {len(pairs)}/{len(t1)} frames; stamp gap max {max(gaps)/1e6:.3f} ms, "
              f"mean {sum(gaps)/len(gaps)/1e6:.3f} ms")
        hstep = max(1, len(pairs) // args.heldout)
        hidx = set(sorted(set(range(0, len(pairs), hstep)))[:args.heldout])
        rest = [p for i, p in enumerate(pairs) if i not in hidx]
        stride = max(1, len(rest) // args.target)
        calib = rest[::stride][:args.target]
        held = [pairs[i] for i in sorted(hidx)]
        print(f"calib pairs={len(calib)} (stride {stride})   heldout pairs={len(held)}")
        if len(calib) < 100:
            print("  WARNING: under 100 calibration pairs — expect a loose fit")
        for tag, sel in (("", calib), ("heldout2/", held)):
            for (rec, kal), which in zip(CAM_MAP, (0, 1)):
                d = dst / f"{tag}{kal}"; d.mkdir(parents=True, exist_ok=True)
                for pr in sel:
                    ts = pr[which]
                    jobs.append((str(src / rec / "data" / f"{ts}.png"),
                                 str(d / f"{ts}.png"), grey))
        meta["frames"] = {"calib_pairs": len(calib), "heldout_pairs": len(held)}
    else:
        imu_csv = src / "imu0" / "data.csv"
        if not imu_csv.is_file():
            sys.exit(f"ERROR: {imu_csv} not found — camera-IMU needs the IMU stream")
        imu = [r for r in csv.reader(imu_csv.open())
               if r and r[0].lstrip("#").strip().isdigit()]
        if not imu:
            sys.exit(f"ERROR: no IMU samples in {imu_csv}")
        base = int(imu[0][0])
        lo = base + int(args.crop_from * 1e9)
        hi = base + int(args.crop_to * 1e9) if args.crop_to < 1e8 else int(imu[-1][0])
        kept = [r for r in imu if lo <= int(r[0]) <= hi]
        dst.mkdir(parents=True, exist_ok=True)
        with (dst / "imu0.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "omega_x", "omega_y", "omega_z",
                        "alpha_x", "alpha_y", "alpha_z"])
            w.writerows(kept)
        span = (int(kept[-1][0]) - int(kept[0][0])) / 1e9
        print(f"imu rows={len(kept)}/{len(imu)}  span={span:.1f}s  "
              f"({len(kept)/span:.0f} Hz)")
        meta["frames"] = {}
        for rec, kal in CAM_MAP:
            d = dst / kal; d.mkdir(parents=True, exist_ok=True)
            n = 0
            for dev, _ in full[rec]:
                host = dev + common
                if host < lo or host > hi:
                    continue
                p = src / rec / "data" / f"{dev}.png"
                if not p.exists():
                    continue
                jobs.append((str(p), str(d / f"{host}.png"), grey)); n += 1
            how = "stamps as recorded" if common == 0 else f"rebased +{common/1e6:.3f} ms"
            print(f"  {rec}->{kal}: {n} frames ({how})")
            meta["frames"][kal] = n
        if min(meta["frames"].values()) == 0:
            sys.exit("ERROR: a camera contributed 0 frames after the rebase+crop — "
                     "check --crop-from/--crop-to against the IMU span above")

    print(f"converting {len(jobs)} images with {args.jobs} procs ...")
    with Pool(args.jobs) as p:
        ok = sum(p.map(convert, jobs, chunksize=16))
    print(f"done: {ok}/{len(jobs)} written -> {dst}")
    (dst / "SOURCE.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"mapping recorded in {dst / 'SOURCE.json'}")
    print("NEXT: step 20 (coverage) before you run any fit")


if __name__ == "__main__":
    main()
