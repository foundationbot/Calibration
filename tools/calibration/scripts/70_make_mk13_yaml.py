#!/usr/bin/env python3
"""Step 70 — write the new mk13_stereo.yaml from the camera-IMU camchain.

Edits ONLY the camera calibration, IN PLACE on top of the current rig file, so
the git diff shows nothing but the recalibrated numbers.

The OKVIS block order comes from config/rig.yaml `cameras.okvis_order`. mk13:

    OKVIS cam0 (dma_queue 2, physical LEFT)  <- kalibr cam1
    OKVIS cam1 (dma_queue 1, physical RIGHT) <- kalibr cam0

That is the 2026-08-16 stereo-role swap. The script REFUSES to run if the
template's dma_queue values are not the queues the rig config says feed each
OKVIS block, or if <tree>/SOURCE.json (with --tree) records a different
record->kalibr mapping than the rig config. distortion_type is rewritten from
the camchain's lens model.

    T_SC        = inv(T_cam_imu)
    image_delay = -mean(timeshift_cam_imu)
    focal / principal / distortion straight from the camchain

===========================================================================
 T_BS AND imu_parameters ARE DELIBERATELY NOT TOUCHED
===========================================================================
T_BS (Head_T_NeckPitchIMU) is the IMU-in-Head leg. It is NOT observable from a
camera-IMU calibration -- nothing in the capture references the Head link -- so
this script leaves it, and imu_parameters, byte-identical. When the cameras move
but the IMU does not, that is exactly right: OKVIS forms camera-in-Head as
T_BC = T_BS @ T_SC, so the new T_SC propagates through the unchanged T_BS with
no URDF edit at all.

If the template has NO T_BS block, this script INJECTS the fallback named by
rig.yaml `imu.T_BS_default_file` (mk13: config/neckpitch_imu.yaml) and says so
loudly. That fallback is ONE robot's IMU-in-Head transform. Do not expect to find T_BS
in a URDF: checked 2026-09-03, no URDF in the workspace declares a NeckPitch IMU
link (every variant declares 16 IMU links and NeckPitch is not among them, while
the live roster has 17). A claim that T_BS was "URDF-validated" is how a 31 deg
yaw error survived two recalibrations. See config/neckpitch_imu.yaml.

Re-derive T_BS only if the IMU BOARD itself moves, and then validate with step
80 checks 2-4 -- never by reprojection error, which is blind to extrinsics.

Usage
-----
    python3 70_make_mk13_yaml.py TEMPLATE CAMCHAIN OUT [--tree tree/imucam_canon]
                                 [--quality-file notes.txt] [--rig config/rig.yaml]
"""
from __future__ import annotations

import argparse, json, re, sys
from pathlib import Path

import numpy as np
import yaml

import rigcfg


TIMESHIFT_SUSPECT_S = 0.010


def image_delay_from_timeshifts(tshift: list[float]) -> tuple[float, str | None]:
    """OKVIS image_delay from Kalibr's per-camera timeshift_cam_imu, and a warning
    when the shift is large enough to be a stamping artefact rather than hardware.

    Kalibr: t_imu = t_cam + shift.  OKVIS: t_cam_correct = t_cam - image_delay,
    so image_delay = -shift. On host-clock capture stamps the true offset is a few
    ms (phantom21: +1.2 ms). Tens of ms is the signature of camera stamps rebased
    by min(publish - capture) before the fit (phantom13 2026-08-22: -23.6 / -24.6
    ms), which current OKVIS, no longer rebasing live, would apply as a real error.
    """
    delay = -float(np.mean(tshift))
    if max(abs(float(t)) for t in tshift) > TIMESHIFT_SUSPECT_S:
        return delay, (f"|timeshift_cam_imu| {', '.join(f'{t*1e3:+.1f}' for t in tshift)} ms "
                       f"exceeds {TIMESHIFT_SUSPECT_S*1e3:.0f} ms. That size usually means the "
                       "camera stamps were rebased by min(publish - capture) before the fit; "
                       "live OKVIS uses capture stamps as recorded. Rebuild the imucam tree "
                       "with step 10 (no rebase on host-clock stamps) and refit before "
                       "deploying this image_delay.")
    return delay, None


def fmt_tbs(tok: list[str]) -> str:
    rows = [", ".join(tok[4 * i:4 * i + 4]) for i in range(4)]
    return ("    T_BS:\n        [" + ",\n         ".join(rows) + "]\n")


def fmt_tsc(T: np.ndarray) -> str:
    r = [", ".join(f"{v:.16g}" for v in T[i]) for i in range(3)]
    return ("T_SC:\n        [" + r[0] + ",\n          " + r[1] + ",\n          "
            + r[2] + ",\n          0, 0, 0, 1]")


def replace_tsc(text: str, T_SC) -> str:
    """the template's two T_SC blocks replaced, in block order (step 71 reuses this)"""
    ms = list(re.compile(r"T_SC:\s*\[[^\]]+\]", re.S).finditer(text))
    if len(ms) != 2:
        sys.exit(f"ERROR: expected 2 T_SC blocks in the template, found {len(ms)}")
    out, last = [], 0
    for i, m in enumerate(ms):
        out.append(text[last:m.start()]); out.append(fmt_tsc(T_SC[i])); last = m.end()
    return "".join(out) + text[last:]


def replace_distortion_type(text: str, name: str) -> str:
    """both blocks' distortion_type set to the OKVIS name (step 71 reuses this)"""
    dt = list(re.finditer(r"distortion_type: \w+", text))
    if len(dt) != 2:
        sys.exit(f"ERROR: expected 2 distortion_type sites in the template, found {len(dt)}")
    for m in reversed(dt):
        text = text[:m.start()] + f"distortion_type: {name}" + text[m.end():]
    return text


def repl_seq(text: str, key: str, values, fmt: str = "{:.16g}") -> str:
    p = re.compile(key + r": \[[^\]]+\]")
    ms = list(p.finditer(text))
    if len(ms) != len(values):
        sys.exit(f"ERROR: {key}: expected {len(values)} sites in the template, "
                 f"found {len(ms)}")
    for m, v in zip(reversed(ms), reversed(values)):
        text = (text[:m.start()] + f"{key}: ["
                + ", ".join(fmt.format(x) for x in v) + "]" + text[m.end():])
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("template", type=Path, help="current mk13_stereo.yaml")
    ap.add_argument("camchain", type=Path, help="camchain-imucam yaml from step 40")
    ap.add_argument("out", type=Path)
    ap.add_argument("--tree", type=Path, help="tree dir; its SOURCE.json is checked")
    ap.add_argument("--quality-file", type=Path,
                    help="file of '# ...' comment lines to splice into the header")
    rigcfg.add_arg(ap)
    args = ap.parse_args()

    rig = rigcfg.load(args.rig)
    ORDER = rig.okvis_from_kalibr()          # OKVIS cam i <- kalibr ORDER[i]

    if args.tree:
        sj = args.tree / "SOURCE.json"
        if sj.is_file():
            m = json.loads(sj.read_text()).get("cam_map", {})
            want = dict(rig.cam_map())
            if m and m != want:
                sys.exit(f"ERROR: {sj} records record->kalibr {m}, but {rig.path}\n"
                         f"       says {want}. The tree was built for a different "
                         "camera mapping;\n       OKVIS would be fed the wrong eye. "
                         "Rebuild it with step 10.")
            print(f"  camera mapping confirmed from {sj}")
        else:
            print(f"  WARNING: {sj} not found; mapping unverified")

    CC = yaml.safe_load(args.camchain.read_text())
    for k in ORDER:
        if k not in CC:
            sys.exit(f"ERROR: {args.camchain} has no '{k}'")
        if "T_cam_imu" not in CC[k]:
            sys.exit(f"ERROR: {args.camchain} '{k}' has no T_cam_imu — this is a "
                     "stereo-only camchain.\n       Run the camera-IMU fit (step 40 "
                     "imucam) and pass ITS camchain.")
    text = args.template.read_text()

    # the template's queues must be the ones the rig config routes to each block
    q = [int(v) for v in re.findall(r"dma_queue: (\d+)", text)]
    if q != rig.okvis_queues():
        sys.exit(f"ERROR: template dma_queue values {q} != {rig.okvis_queues()} that "
                 f"{rig.path.name} routes to OKVIS cam0/cam1\n       (okvis_order "
                 f"{rig.okvis_order} over cameras.kalibr). A block would get another "
                 "camera's T_SC. Fix the template or the rig config.")
    lens = rig.camchain_lens(CC, ORDER[0]); rig.camchain_lens(CC, ORDER[1])

    T_SC = [np.linalg.inv(np.array(CC[k]["T_cam_imu"])) for k in ORDER]
    foc = [CC[k]["intrinsics"][:2] for k in ORDER]
    prin = [CC[k]["intrinsics"][2:] for k in ORDER]
    dist = [CC[k]["distortion_coeffs"] for k in ORDER]
    tshift = [CC[k]["timeshift_cam_imu"] for k in ORDER]
    delay, tshift_warning = image_delay_from_timeshifts(tshift)
    if tshift_warning:
        print(f"  ⚠ WARNING: {tshift_warning}")

    lo, hi = (float(v) for v in rig.gates["t_sc_range_m"])
    for i, T in enumerate(T_SC):
        n = np.linalg.norm(T[:3, 3])
        if not lo < n < hi:
            print(f"  ⚠ WARNING: |T_SC| for OKVIS cam{i} is {n:.3f} m, outside the "
                  f"plausible {lo*100:g}-{hi*100:g} cm band.\n    ~1 m is the signature of the "
                  f"checkerboard-flip divergence — did step 30 run?")

    text = replace_tsc(text, T_SC)
    text = repl_seq(text, "focal_length", foc)
    text = repl_seq(text, "principal_point", prin)
    text = repl_seq(text, "distortion_coefficients", dist)
    text = replace_distortion_type(text, rigcfg.OKVIS_NAME[lens.kind])
    text, cnt = re.subn(r"image_delay: -?[0-9.]+", f"image_delay: {delay:.4f}", text, count=1)
    if cnt != 1:
        sys.exit("ERROR: image_delay not found in the template")

    # --- T_BS: preserve, or inject the documented default --------------------
    if re.search(r"^\s*T_BS:", text, re.M):
        print("  T_BS: present in the template — PRESERVED unchanged (correct: the "
              "IMU-in-Head\n        leg is not observable from a camera-IMU fit)")
    else:
        if not rig.imu.get("T_BS_default_file"):
            sys.exit(f"ERROR: the template {args.template} has no T_BS and {rig.path.name} "
                     "names no fallback (imu.T_BS_default_file: null).\n       Use a template "
                     f"that carries {rig.name}'s own IMU-in-Head T_BS; another robot's would "
                     "be silently wrong.")
        anchor = re.search(r"^(\s*)s_a: \[[^\]]*\]\s*.*$", text, re.M)
        src = rig.resolve(rig.imu["T_BS_default_file"])
        if anchor:
            tok = rig.t_bs_default_tokens()
            text = text[:anchor.end()] + "\n" + fmt_tbs(tok) + text[anchor.end():]
            print(f"  ⚠ T_BS: ABSENT from the template — INJECTED the fallback from\n"
                  f"        {src}\n"
                  f"        ({' '.join(tok[:3])} / {tok[3]}, {tok[7]}, {tok[11]}).\n"
                  f"        It is the {rig.imu['name']} IMU-in-Head transform of rig "
                  f"'{rig.name}' — on any other robot\n        it is WRONG. Validate with step 80.")
        else:
            print(f"  ⚠ T_BS ABSENT and no imu_parameters anchor found — add it BY HAND\n"
                  f"        from {src} before deploying.")

    qual = args.quality_file.read_text() if args.quality_file else ""
    swap = " (stereo-role swap)" if rig.okvis_order != [0, 1] else ""
    cams = "".join(f"# OKVIS cam{i} = physical {s.upper()} = dma_queue {q} <- kalibr {k}"
                   f"{swap if i == 0 else ''}.\n"
                   for i, (k, q, s) in enumerate(zip(ORDER, rig.okvis_queues(),
                                                     rig.okvis_sides())))
    if rig.board.kind == "aprilgrid":
        board = str(rig.board)
        filt = ""
    else:
        board = rig.charuco.get("description") or f"{rig.board} checkerboard"
        filt = ("# The cam-IMU input was filtered to one board orientation (step 30); without\n"
                "# it the solve diverges to |T_SC| = 1.03 m.\n")
    hdr = ("%YAML:1.0\n"
           f"# {rig.okvis_yaml_name} — {rig.name}. RECALIBRATED by calib-phantom13-pkg.\n"
           f"# Kalibr stereo + camera-IMU, {board}, {rig.model}.\n"
           + cams +
           f"# IMU = {rig.imu['name']} on /raw_imu_actuals — verify the index every session\n"
           f"# (deployment/display_imu.py); it is {rig.imu['index']} in "
           f"{rig.path.name} => OKVIS_DMA_IMU_INDEX={rig.imu['index']}.\n"
           "# imu_parameters + T_BS are NOT products of this pipeline — see\n"
           f"# {rig.imu['T_BS_default_file']} for the fallback default.\n"
           + filt + qual)
    text = re.sub(r"^%YAML:1\.0\n(#.*\n)*", hdr, text, count=1)

    args.out.write_text(text)
    print(f"\nwrote {args.out}")
    print(f"  image_delay = {delay:.4f} s  (timeshifts {tshift[0]:+.6f} / {tshift[1]:+.6f})")
    for i in range(2):
        print(f"  OKVIS cam{i} <- kalibr {ORDER[i]}: f={foc[i][0]:.3f},{foc[i][1]:.3f} "
              f"c={prin[i][0]:.3f},{prin[i][1]:.3f} |t_SC|={np.linalg.norm(T_SC[i][:3,3]):.4f} m")
    b = np.linalg.inv(np.array(CC["cam1"]["T_cn_cnm1"]))[:3, 3]
    print(f"  stereo baseline from T_cn_cnm1: {np.linalg.norm(b)*1000:.2f} mm")
    print("\nNEXT: step 80 (validate against physics), then diff against the template "
          "before deploying.")


if __name__ == "__main__":
    main()
