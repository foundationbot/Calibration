#!/usr/bin/env python3
"""Step 75 — quick check of a neck take, run at the robot before the board and the bracing come
down. Judges: enough held poses over the neck's range, each joint reached from both sides; the
encoders' signs agree with what the cameras see; the joint and IMU streams; the recorded IMU is the
head's (step 82's fit explains it); every camera ran the whole take and saw the board in the holds;
the sweeps step 82 needs for the joint axes; and, with --dmarec, a torso that stayed put. The held
poses are graded over their own range (on 17 Oct the board was seen only with the head level or up),
and frames missing only in gaps inside the take (the record pod falling behind) are a warning, not a
stopped camera.

Each finding is STOP (a retake cannot fix it: a reversed encoder, the wrong IMU, a robot that turned
or tipped), RETAKE (the take is short of something) or WARN (steps 81/82 still run, with less), and
says what to do. The verdict is the worst finding (exit 2 / 1 / 0). Holds step 81 should leave out
(a moved torso, a /joints gap while the neck moved) go into --qc-out's drop_holds, which run-tbs.sh
passes on. Thresholds are judgement calls, not package gates."""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import necktake  # noqa: E402

s82 = importlib.import_module("82_tbs_gyro_axes")

MIN_HOLDS, MIN_CELLS, MIN_YAW_SIDES, MIN_PITCH_SIDES = 20, 8, 4, 2
MIN_SWEEP_S = 3.0
MIN_YAW_SPAN_DEG, MIN_PITCH_SPAN_DEG = 15.0, 10.0   # each joint's range over the holds (a 3 x 3 grid over that range)
MIN_PX_PER_RAD = 200.0                   # board motion in the image per radian of neck (measured ~600)
TILT_DEG, YAW_DEG, TURN_DEG, MAX_MOVED = 0.1, 0.1, 0.1, 0.2
TORSO_SPLIT_DEG = 0.05                   # torso_motion's group split: half the flags (equal lets a 2x slide pass)
ROBOT_MOVED_DEG = 1.0                    # any hold this far off: the robot itself turned or tipped
BOARD_SPREAD_DEG, BOARD_SPREAD_MM = 2.0, 40.0   # the board's spread in the torso frame beyond these: something moved
YAW_UNRELIABLE = 1e-4                    # rad/s: holds' rate scatter above this, the torso's yaw is the estimate's
HEAD_IMU_RESIDUAL = 0.15                 # rad/s: step 82's fit on the wrong IMU leaves ~|omega|
JOINT_GAP_MS, GAP_MOVE_DEG = 50.0, 0.05  # a /joints gap inside a hold across which the neck moved: drop the hold
MIN_CAM_COVER, MIN_BOARD = 0.95, 0.8


def grade_holds(holds) -> dict:
    """The held poses: how many; how they fill a 3 x 3 grid over their own yaw and pitch range (the
    board, not the neck, limits that range: on 17 Oct the board was seen only with the head level or
    up); each joint's range [deg]; from which side each joint was reached."""
    y = np.array([h.yaw for h in holds], float)
    p = np.array([h.pitch for h in holds], float)
    thirds = lambda v: np.linspace(v.min(), v.max(), 4)[1:3] if v.size else np.zeros(2)  # noqa: E731
    cells = {(int(np.digitize(a, thirds(y))), int(np.digitize(b, thirds(p)))) for a, b in zip(y, p)}
    yaw_span = float(np.degrees(np.ptp(y))) if y.size else 0.0
    pitch_span = float(np.degrees(np.ptp(p))) if p.size else 0.0
    sides = {"yaw+": sum(h.approach_yaw > 0 for h in holds), "yaw-": sum(h.approach_yaw < 0 for h in holds),
             "pitch+": sum(h.approach_pitch > 0 for h in holds), "pitch-": sum(h.approach_pitch < 0 for h in holds)}
    ok = (len(holds) >= MIN_HOLDS and len(cells) >= MIN_CELLS and yaw_span >= MIN_YAW_SPAN_DEG
          and pitch_span >= MIN_PITCH_SPAN_DEG and min(sides["yaw+"], sides["yaw-"]) >= MIN_YAW_SIDES
          and min(sides["pitch+"], sides["pitch-"]) >= MIN_PITCH_SIDES)
    return {"n": len(holds), "cells": len(cells), "yaw_span_deg": yaw_span, "pitch_span_deg": pitch_span,
            "sides": sides, "ok": ok}


def sweep_seconds(j) -> dict:
    """Seconds of the sweeps step 82 measures the joint axes from (its own sweep_masks)."""
    qdy, qdp = necktake.rates_at(j, j.t)
    dt = np.median(np.diff(j.t)) / 1e9
    pm, ym = s82.sweep_masks(qdy, qdp)
    return {"pitch_only_s": float(pm.sum() * dt), "yaw_only_s": float(ym.sum() * dt)}


def stream(t_ns) -> dict:
    t = np.asarray(t_ns, np.int64)
    if len(t) < 2:
        return {"rate_hz": 0.0, "max_gap_ms": float("inf")}
    return {"rate_hz": float((len(t) - 1) / ((t[-1] - t[0]) / 1e9)), "max_gap_ms": float(np.max(np.diff(t)) / 1e6)}


def camera_coverage(t_cam, t0_ns, t1_ns) -> dict:
    """How much of the take [t0, t1] a camera's frames cover: their share of the span, the longest
    gap (the take's ends included) and, if the frames end over a second early, when it stopped [s]."""
    t = np.sort(np.asarray(t_cam, np.int64))
    span = (t1_ns - t0_ns) / 1e9
    if not len(t):
        return {"cover": 0.0, "max_gap_s": span, "stopped_at_s": 0.0}
    edges = np.r_[t0_ns, t, t1_ns]
    gaps = np.diff(edges) / 1e9
    cover = 1.0 - float(np.sum(gaps[gaps > 1.0])) / span
    stopped = (t1_ns - t[-1]) / 1e9 > 1.0
    return {"cover": cover, "max_gap_s": float(gaps.max()), "stopped_at_s": float((t[-1] - t0_ns) / 1e9) if stopped else None}


def camera_finding(rd: str, cov: dict, empty: int, seen: int, with_frames: int) -> dict | None:
    """A camera whose frames end over a second early (it stopped): RETAKE; one that lost the board in
    many holds: RETAKE; frames missing only in gaps inside the take (the record pod falling behind; 17
    Oct: 92 % cover): WARN, its holds have fewer images (and step 75's usable-holds count says if too
    few remain)."""
    if cov["stopped_at_s"] is not None:
        return {"level": "RETAKE", "what": f"camera-{rd}",
                "action": f"camera {rd} stopped at {cov['stopped_at_s']:.0f} s: restart the camera producer (dma-video) "
                          "and check both cameras stream before retaking"}
    if seen < MIN_BOARD * max(with_frames, 1):
        return {"level": "RETAKE", "what": f"board-{rd}",
                "action": f"camera {rd} lost the board in many holds: keep the whole board in view (smaller neck "
                          "angles or the board farther away)"}
    if cov["cover"] < MIN_CAM_COVER:
        return {"level": "WARN", "what": f"camera-gaps-{rd}",
                "action": f"camera {rd}'s frames cover {cov['cover']:.0%} of the take (longest gap {cov['max_gap_s']:.1f} "
                          f"s, {empty} holds without frames): the recorder fell behind (check the record pod's CPU); "
                          "those holds have fewer images; nothing to do"}
    return None


def gap_holds(j, holds) -> list:
    """Holds with a /joints gap over JOINT_GAP_MS inside across which the neck moved more than
    GAP_MOVE_DEG: their images get angles interpolated across the gap. -> [(hold, gap_ms, moved_deg)]."""
    out = []
    for g in np.flatnonzero(np.diff(j.t) > JOINT_GAP_MS * 1e6):
        moved = float(np.degrees(np.hypot(j.yaw[g + 1] - j.yaw[g], j.pitch[g + 1] - j.pitch[g])))
        for k, h in enumerate(holds):
            if j.t[g + 1] > h.t0 and j.t[g] < h.t1 and moved > GAP_MOVE_DEG:
                out.append((k, float((j.t[g + 1] - j.t[g]) / 1e6), moved))
    return out


def encoder_signs(yaw, pitch, u, v) -> dict:
    """Board centre in a camera's image against the neck angles, over the holds:
    u = a0 + a_yaw * yaw + a_p * pitch, v = b0 + b_y * yaw + b_pitch * pitch. The cameras' image x
    points to the robot's right and image y down, so turning the head left (+NeckYaw about +z) moves
    the board right (a_yaw > 0) and tilting it down (+NeckPitch about +y) moves the board up
    (b_pitch < 0). The wrong sign beyond MIN_PX_PER_RAD: that encoder counts against its URDF axis;
    below it: too little motion to tell."""
    A = np.stack([np.ones(len(yaw)), yaw, pitch], 1)
    a = np.linalg.lstsq(A, np.asarray(u, float), rcond=None)[0]
    b = np.linalg.lstsq(A, np.asarray(v, float), rcond=None)[0]
    state = lambda x: "ok" if x > MIN_PX_PER_RAD else ("reversed" if x < -MIN_PX_PER_RAD else "too little motion")  # noqa: E731
    return {"yaw_px_per_rad": float(a[1]), "pitch_px_per_rad": float(b[2]),
            "yaw": state(a[1]), "pitch": state(-b[2]),
            "yaw_ok": bool(a[1] > MIN_PX_PER_RAD), "pitch_ok": bool(b[2] < -MIN_PX_PER_RAD)}


def moved_holds(motion) -> dict:
    """Holds where the torso stood tilted or turned against most holds, or turned during the hold
    (dmarec_imu.torso_motion). Yaw counts, in the flags and in the largest offset, only where it is
    beyond 3 of its sigma: a robot rocking during the holds makes the bias estimate's error large and
    its sigma says so, while a real turn of degrees still shows. -> flagged holds, per-component
    counts and maxima, whether the yaw is precise (rate scatter), the largest offset, and the
    reference group's share of the holds."""
    ms = [(k, m) for k, m in enumerate(motion) if m is not None]
    scatter = max((m.get("rate_scatter_rad_s", 0.0) for _, m in ms), default=0.0)
    flags = {"tilt": [], "yaw": [], "turn": []}
    real_yaw = []
    for k, m in ms:
        if m["tilt_deg"] > TILT_DEG:
            flags["tilt"].append(k)
        if abs(m["yaw_deg"]) > 3 * m.get("yaw_sigma_deg", 0.0):
            real_yaw.append(abs(m["yaw_deg"]))
            if abs(m["yaw_deg"]) > YAW_DEG:
                flags["yaw"].append(k)
        if m["turn_deg"] > TURN_DEG:
            flags["turn"].append(k)
    mx = {c: max((abs(m[f"{c}_deg"]) for _, m in ms), default=0.0) for c in ("tilt", "yaw", "turn")}
    return {"flagged": sorted(set().union(*flags.values())), "offset_flagged": sorted(set(flags["tilt"]) | set(flags["yaw"])),
            "counts": {c: len(v) for c, v in flags.items()},
            "max_deg": mx, "yaw_reliable": scatter <= YAW_UNRELIABLE, "rate_scatter_rad_s": scatter,
            "max_offset_deg": max([mx["tilt"]] + real_yaw),
            "reference_share": sum(m.get("in_reference", True) for _, m in ms) / max(len(ms), 1)}


def torso_finding(mv, n_holds) -> tuple:
    """-> (finding or None, holds to drop). A hold beyond ROBOT_MOVED_DEG: STOP, the robot moved. A
    reference group under half the holds with more than MAX_MOVED of them standing off (tilt or yaw;
    rotation during a hold is no offset): RETAKE, the torso shifted (a step or slide biases X by up
    to half its size and does not average down). Otherwise the flagged holds are dropped: wander,
    which averages down."""
    if mv["max_offset_deg"] > ROBOT_MOVED_DEG:
        return ({"level": "STOP", "what": "robot-moved",
                 "action": f"the robot turned or tipped by up to {mv['max_offset_deg']:.1f} deg: lock it against turning "
                           "and tilting (both sides, shoulder height), then retake"}, set())
    if mv["reference_share"] < 0.5 and len(mv["offset_flagged"]) > MAX_MOVED * max(n_holds, 1):
        return ({"level": "RETAKE", "what": "torso",
                 "action": f"the torso shifted during the take (only {mv['reference_share']:.0%} of the holds stood where "
                           "most did): brace it harder and move the head with fingertip force"}, set())
    return None, set(mv["flagged"])


def board_spread_finding(spread) -> dict | None:
    """The board in the torso frame through the encoders and the config's camera pose, over the holds
    (step 81's spread before its fit): it stays put unless the robot or the board moved (or the config
    is degrees off). Gyro-free, so a rocking robot cannot hide a turn from it. Take A (the robot
    turned ~20 deg): 6.99 deg / 161 mm; braced takes: 0.56-0.68 deg / 9-16 mm."""
    if spread["rot_rms_deg"] > BOARD_SPREAD_DEG or spread["trans_rms_mm"] > BOARD_SPREAD_MM:
        return {"level": "STOP", "what": "robot-or-board-moved",
                "action": f"the board moved {spread['trans_rms_mm']:.0f} mm / {spread['rot_rms_deg']:.1f} deg rms in the "
                          "torso frame across the holds: lock the robot and the board stand, then retake"}
    return None


def usable_holds_finding(n_usable, need) -> dict | None:
    """Step 81 gives no verdict below `need` holds with a usable board pose (20 tags, 1 px): say so at
    the robot, not after leaving it."""
    if n_usable < need:
        return {"level": "RETAKE", "what": "usable-holds",
                "action": f"only {n_usable} holds have a board pose step 81 can use (it needs {need}): keep the whole "
                          "board in view and the head still in each hold, and hold more poses"}
    return None


def qc_report(verdict, findings, drop) -> dict:
    """The --qc-out contract run-tbs.sh reads: verdict word, findings, drop_holds (ints, sorted)."""
    return {"verdict": verdict.split()[0], "findings": findings, "drop_holds": sorted(int(k) for k in drop)}


def decide(findings) -> tuple:
    """-> (verdict line, exit code): the worst finding wins."""
    stop = [f["what"] for f in findings if f["level"] == "STOP"]
    retake = [f["what"] for f in findings if f["level"] == "RETAKE"]
    if stop:
        return "STOP (" + ", ".join(stop) + ")", 2
    if retake:
        return "RETAKE (" + ", ".join(retake) + ")", 1
    return "GOOD", 0


def main() -> int:
    import argparse
    import cv2
    import dmarec_imu
    import neckfk
    import okviscfg
    import rigcfg
    import rigmath as rm
    s81 = importlib.import_module("81_tbs_handeye")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--session", type=Path, required=True)
    ap.add_argument("--export", type=Path, required=True)
    ap.add_argument("--urdf", type=Path, required=True)
    ap.add_argument("--config", type=Path, help="OKVIS config: with it, the board's spread in the torso frame is checked")
    ap.add_argument("--dmarec", type=Path, help="dma-recorder folder (torso motion)")
    ap.add_argument("--dmarec-copy", type=Path, help="copy the take's recorder segments here")
    ap.add_argument("--torso-index", type=int, default=9)
    ap.add_argument("--qc-out", type=Path, help="write the verdict, findings and drop_holds here (JSON)")
    rigcfg.add_arg(ap)
    a = ap.parse_args()
    rig, chain = rigcfg.load(a.rig), neckfk.NeckChain(a.urdf)
    j = necktake.read_joints(a.session)
    holds = necktake.find_holds(j)
    findings, drop = [], set()

    def find(level, what, action):
        findings.append({"level": level, "what": what, "action": action})

    g = grade_holds(holds)
    short = len(necktake.find_holds(j, min_s=0.3)) - len(holds)
    print(f"holds: {g['n']} (want >= {MIN_HOLDS}), cells {g['cells']}/9 of a grid over their range (want >= "
          f"{MIN_CELLS}), range yaw {g['yaw_span_deg']:.0f} deg / pitch {g['pitch_span_deg']:.0f} deg (want >= "
          f"{MIN_YAW_SPAN_DEG:.0f} / {MIN_PITCH_SPAN_DEG:.0f}); yaw reached "
          f"from left/right {g['sides']['yaw+']}/{g['sides']['yaw-']} (want >= {MIN_YAW_SIDES} each), pitch from "
          f"below/above {g['sides']['pitch+']}/{g['sides']['pitch-']} (want >= {MIN_PITCH_SIDES}); {short} stops too "
          f"short to count; yaw {np.degrees(j.yaw.min()):+.0f}..{np.degrees(j.yaw.max()):+.0f}, pitch "
          f"{np.degrees(j.pitch.min()):+.0f}..{np.degrees(j.pitch.max()):+.0f} deg")
    if holds:
        dr = np.degrees([h.drift_rad for h in holds])
        print(f"drift within holds (sag plus tremor): median {np.median(dr):.2f} deg, max {dr.max():.2f} deg (hold "
              f"{int(np.argmax(dr))})")
    if not g["ok"]:
        find("RETAKE", "holds", f"hold each pose still for ~2.5 s ({short} stops were too short), cover the "
                                "grid and reach each pose from both sides (the protocol's snake there and back)")
    sw = sweep_seconds(j)
    print(f"sweeps step 82 can use: pitch-only {sw['pitch_only_s']:.1f} s, yaw-only {sw['yaw_only_s']:.1f} s (want >= "
          f"{MIN_SWEEP_S} s each; > {s82.SWEEP_RATE} rad/s = {np.degrees(s82.SWEEP_RATE):.0f} deg/s, the other joint "
          f"under {s82.PURITY:.0%} of it)")
    if min(sw.values()) < MIN_SWEEP_S:
        find("WARN", "sweeps", "step 82 will not measure the joint axes (its R_BS still fits): nod and turn one joint at "
                               "a time, faster than ~20 deg/s; a short separate take will do")
    js = stream(j.t)
    gh = gap_holds(j, holds)
    print(f"joints: {js['rate_hz']:.0f} Hz, longest gap {js['max_gap_ms']:.0f} ms; holds with a gap while the neck moved: "
          + (", ".join(f"{k} ({ms:.0f} ms, {mv:.2f} deg)" for k, ms, mv in gh) or "none"))
    if js["max_gap_ms"] > JOINT_GAP_MS:
        drop |= {k for k, _, _ in gh}
        find("WARN", "joints", "the recorder fell behind (gaps in /joints; check the record pod's CPU): those holds are "
                               "dropped; nothing to do unless too few holds remain")
    t_imu, w_imu, _ = necktake.read_imu(a.export)
    ims = stream(t_imu)
    print(f"IMU: {ims['rate_hz']:.1f} Hz, longest gap {ims['max_gap_ms']:.1f} ms")
    if ims["rate_hz"] < 490 or ims["max_gap_ms"] > 10:
        find("WARN", "imu", "gaps in the head IMU stream: step 82 has fewer samples; nothing to do")
    try:
        res = s82.fit(chain, j, t_imu, w_imu)["residual_rad_s"]
    except ValueError as e:
        res = float("inf")
        print(f"IMU check: {e}")
    print(f"IMU is the head's: step 82's fit leaves {res:.3f} rad/s (want < {HEAD_IMU_RESIDUAL}; the wrong IMU leaves ~0.5)")
    if res >= HEAD_IMU_RESIDUAL:
        find("STOP", "not-the-head-imu", "record with IMU_INDEX = the NeckPitch IMU's index (python3 display_imu.py)")
    centre = (rig.resolution[0] / 2.0, rig.resolution[1] / 2.0)
    cfg = okviscfg.load(a.config) if a.config else None
    lens = rigcfg.Lens(cfg.lens) if cfg else None
    board_T, posed = [], []                                 # board in the torso (camera 0, one image per hold); its holds
    signs = {}
    for cam, k in enumerate(rig.okvis_order):
        rd = rig.kalibr[k]["record_dir"]
        t, paths = necktake.read_frames(a.export, rd)
        cov = camera_coverage(t, int(j.t[0]), int(j.t[-1]))
        seen, empty, signs[cam] = 0, 0, []
        for hk, h in enumerate(holds):
            idx = np.flatnonzero((t >= h.t0) & (t <= h.t1))
            if not idx.size:
                empty += 1
                continue
            mid = idx[idx.size // 2]
            det = rig.board.detect(cv2.imread(str(paths[mid]), cv2.IMREAD_GRAYSCALE))
            tags = {tid for (tid, kk), (u, v) in det.items() if np.hypot(u - centre[0], v - centre[1]) <= 450}
            seen += len(tags) >= 8
            if len(det) >= 16:
                uv = np.mean(np.array(list(det.values())), axis=0)
                signs[cam].append((h.yaw, h.pitch, uv[0], uv[1]))
            if cam == 0 and cfg is not None:
                T_CW = s81.pose_from_corners(det, rig.board, lens, cfg.K[0], cfg.D[0], centre)
                if T_CW is not None:
                    qy, qp = necktake.at(j, np.array([t[mid]]))
                    board_T.append(chain.T_TB(float(qy[0]), float(qp[0])) @ cfg.T_BS @ cfg.T_SC[0] @ T_CW)
                    posed.append(hk)
        with_frames = len(holds) - empty
        print(f"camera {rd}: {stream(t)['rate_hz']:.1f} fps, covers {cov['cover']:.0%} of the take"
              + (f", STOPPED at {cov['stopped_at_s']:.0f} s" if cov["stopped_at_s"] is not None else "")
              + f"; holds without frames {empty}, board (>= 8 tags inside 450 px) in {seen}/{with_frames} with frames")
        fd = camera_finding(rd, cov, empty, seen, with_frames)
        if fd:
            findings.append(fd)
    if len(board_T) >= 6:
        sp = rm.spread(board_T)
        print(f"board in the torso frame through the encoders and the config (camera 0, {len(board_T)} holds): spread "
              f"{sp['trans_rms_mm']:.1f} mm / {sp['rot_rms_deg']:.2f} deg rms (beyond {BOARD_SPREAD_MM:.0f} mm or "
              f"{BOARD_SPREAD_DEG} deg: the robot or the board moved)")
        f = board_spread_finding(sp)
        if f:
            findings.append(f)
    use = 0 if len(signs.get(0, [])) >= 6 else 1
    if len(signs.get(use, [])) >= 6:
        s = encoder_signs(*np.array(signs[use]).T)
        word = lambda st: "as the URDF" if st == "ok" else st.upper()  # noqa: E731
        print(f"encoder signs (camera {use}, board in the image per rad): NeckYaw {s['yaw_px_per_rad']:+.0f} px "
              f"({word(s['yaw'])}), NeckPitch {s['pitch_px_per_rad']:+.0f} px ({word(s['pitch'])})")
        for jn in ("yaw", "pitch"):
            if s[jn] == "reversed":
                find("STOP", f"encoder-sign-{jn}", f"Neck{jn.capitalize()}'s encoder counts against its URDF axis: fix "
                                                   "the URDF axis or the driver sign before any fit")
            elif s[jn] != "ok":
                find("RETAKE", f"encoder-sign-{jn}", f"move Neck{jn.capitalize()} over a larger range (> ~10 deg)")
    else:
        print("encoder signs: too few holds with the board in view")
        find("RETAKE", "encoder-sign", "keep the board in view of a camera in at least 6 holds")
    if a.dmarec:
        closed, still_open, unfinished = dmarec_imu.segments_covering(a.dmarec, int(j.t[0]), int(j.t[-1]))
        if still_open:
            print(f"torso: {len(still_open)} recorder segment(s) still being written")
            find("RETAKE", "torso-wait", "wait a minute and rerun step 75 (not a new take)")
        if unfinished:
            print(f"torso: {len(unfinished)} segment(s) never finished (a recorder restart?): read as far as they go")
        readable = closed + unfinished
        if a.dmarec_copy and readable:
            dmarec_imu.copy_segments(readable, a.dmarec_copy)
            print(f"torso: {len(readable)} segment(s) copied to {a.dmarec_copy}")
        tt, wt, at_, errors = dmarec_imu.read_raw_imu(readable, a.torso_index, int(j.t[0]), int(j.t[-1]))
        ends = {str(p) for p in unfinished}
        for p, e in errors:
            if p in ends and e.startswith("EndOfFile"):   # an idle recorder can leave one complete but open
                print(f"torso: read the unfinished segment {p} to its end")
            else:
                print(f"torso: could not read {p}: {e}")
        cover = len(tt) / 500.0 / ((j.t[-1] - j.t[0]) / 1e9)
        if len(tt) > 1:
            ts = stream(tt)
            print(f"torso IMU stream: {ts['rate_hz']:.1f} Hz, longest gap {ts['max_gap_ms']:.1f} ms")
        if cover < 0.95:
            print(f"torso: the recorder covers {cover:.0%} of the take")
            find("RETAKE", "torso-missing", "the dma-recorder must run during the take (k0s kubectl get pods -A | grep "
                                            "dma-recorder)")
        else:
            mv = moved_holds(dmarec_imu.torso_motion(tt, wt, at_, holds, step_deg=TORSO_SPLIT_DEG))
            c, mx = mv["counts"], mv["max_deg"]
            print(f"torso (IMU {a.torso_index}): tilt > {TILT_DEG} in {c['tilt']} holds (max {mx['tilt']:.2f} deg); "
                  f"yaw > max({YAW_DEG}, 3 sigma) in {c['yaw']} (max {mx['yaw']:.2f}"
                  + ("" if mv["yaw_reliable"] else f"; the robot rocked during the holds, rate scatter "
                                                   f"{mv['rate_scatter_rad_s']:.1e} rad/s, so yaw's sigma is large")
                  + f"); rotation during a hold > {TURN_DEG} in {c['turn']} (max {mx['turn']:.2f}); reference group "
                  f"{mv['reference_share']:.0%} of the holds")
            f, d = torso_finding(mv, len(holds))
            if f:
                findings.append(f)
            drop |= d
    left = [h for k, h in enumerate(holds) if k not in drop]
    if drop:
        g2 = grade_holds(left)
        print(f"after dropping {len(drop)} holds: {g2['n']} left, {g2['cells']}/9 cells, sides {g2['sides']}")
        if g["ok"] and not g2["ok"]:
            find("RETAKE", "holds-after-drops", "too few usable holds remain once the moved and gapped ones are left out: "
                                                "retake (and fix what dropped them)")
    if cfg is not None:
        n_ok = len([k for k in posed if k not in drop])
        print(f"holds with a board pose step 81 can use (camera 0, >= {s81.MIN_TAGS} tags, <= {s81.MAX_RMS_PX} px), after "
              f"the drops: {n_ok} (it needs {s81.MIN_VERDICT_HOLDS} for a verdict)")
        f = usable_holds_finding(n_ok, s81.MIN_VERDICT_HOLDS)
        if f:
            findings.append(f)
    findings.sort(key=lambda f: ["STOP", "RETAKE", "WARN"].index(f["level"]))
    for f in findings:
        print(f"  {f['level']:6s} {f['what']}: {f['action']}")
    verdict, code = decide(findings)
    print(f"step 81: DROP_HOLDS={','.join(map(str, sorted(drop)))}" if drop else "step 81: no holds to drop")
    if a.qc_out:
        a.qc_out.parent.mkdir(parents=True, exist_ok=True)
        a.qc_out.write_text(json.dumps(qc_report(verdict, findings, drop), indent=1))
    print("QC VERDICT: " + verdict)
    return code


if __name__ == "__main__":
    sys.exit(main())
