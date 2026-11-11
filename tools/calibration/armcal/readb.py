"""Read B per capture, either arm: a burst (depth frames + joints + T_BS) -> the fused depth -> the arm's forearm and
wrist meshes matched in it (B0, meshfit.fit_pose) -> one capture record.

(Ported from OKVIS2-X tools/burst_pointcloud/hand_check.py, c5b4c6b0 -- load_pose, the side handling, the motion and
view checks -- and the record of the phantom14 driver urdfx/fitrun.py that produced the stored per-capture results
base_w2.json / base_L.json. The depth fusion is armcal/fuse.py, not the C++ module.)

    python3 -m armcal.readb BURST.npz... --side right --urdf ROBOT.urdf --out captures.json [--frames N]
                            [--matcher robust|fast|reference] [--jobs N]

The matcher (results/handeye-readb2-2026-11-11/): "robust" (the default) runs B0 twice per capture -- from the
encoders, then from the wave's median offset -- and calls a capture clean only when the two land within SETTLED_MM;
"fast" computes the phantom14 algorithm's numbers (the stored records), "reference" is that algorithm as it ran.

The record (JSON-able; the stage-2 fit reads it): name, off (wrist offset, mm, Torso frame: where the depth puts the
wrist point minus where the encoders + URDF put it), norm, rot (the match's rotation, deg), cov (coverage), ok
(converged, enough inliers, conditioned), clean, E, T_TC, T_BC, q, yaw, pitch, wrist, rms, inl, steps, cond,
motion_deg / motion_joint (how far a joint moved inside the burst), view (the view gate at the encoders), timings.

clean = ok and rot <= CLEAN_ROT_DEG and cov >= CLEAN_COVERAGE (the hand-eye plan, section 4), and for the robust
matcher settled (its two passes agree); the arm-median outlier cut is the stage-2 selection's (urdffit.select).
Motion above MAX_MOTION_DEG and a failed view gate are reported, not folded into clean: the stored phantom14
decisions did not use them."""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from armcal import MESHES, burst_io, fuse, meshfit
from armcal.frames import Frames
from armcal.urdf import Urdf

LINKS = ("RightElbowPitchLink", "RightWristYawLink", "RightWristRollLink")
ARM = ("RightShoulderPitch", "RightShoulderRoll", "RightShoulderYaw", "RightElbowPitch", "RightWristYaw",
       "RightWristRoll", "RightGripperYaw")
MAX_MOTION_DEG = 0.2
CLEAN_ROT_DEG = 12.0
CLEAN_COVERAGE = 0.6
SPACING = 0.001                    # m, mesh sampling
MATCHERS = {"robust": meshfit.ROBUST, "fast": meshfit.FAST, "reference": meshfit.REFERENCE}
DEFAULT_MATCHER = "robust"         # readb2 (11 Nov 2026); "fast" reproduces the stored phantom14 records
PASS2_GATES = (0.015, 0.008, 0.005)  # the robust matcher's pass 2 (from the wave's median offset)
SETTLED_MM = 3.0                     # the robust matcher: pass 2 within this of pass 1, or the capture is not clean


def sided(names, side: str) -> tuple:
    """the right arm's link or joint names on that side ("Right" / "Left")"""
    side = side.capitalize()
    if side not in meshfit.SIDES:
        raise ValueError(f"side must be right or left, got {side!r}")
    return tuple(side + n[len("Right"):] for n in names)


def arm_model(urdf, side: str, mesh_dir=MESHES, spacing: float | None = None,
              cfg: meshfit.FitConfig = meshfit.FAST, **kw) -> meshfit.ArmModel:
    """the side's fitted links, sampled as cfg says (sampler, spacing unless given, model_voxel); kw: ArmModel's
    phase / jitter / seed"""
    return meshfit.ArmModel.load(Urdf.of(urdf), mesh_dir, sided(LINKS, side), cfg.spacing if spacing is None else
                                 spacing, cfg.model_voxel, sampler=cfg.sampler, **kw)


def capture_name(path, side: str) -> str:
    """the stored records' name: the burst file's stem after '<side>-' (p14-armwave-1104a-right-01-p07-10-m12 ->
    01-p07-10-m12), else the whole stem"""
    stem = Path(path).stem
    key = side.lower() + "-"
    return stem.split(key, 1)[1] if key in stem else stem


@dataclass
class Capture:
    name: str
    side: str
    q: dict                 # the arm's joints (rad), median over the frames used
    angles: dict            # every joint
    T_TC: np.ndarray        # the camera in the Torso
    T_BC: np.ndarray        # the camera in the Head (T_BS · T_SC)
    depth: np.ndarray       # (H,W) fused, m
    K: tuple
    frames: int             # depth frames fused
    motion_deg: float
    motion_joint: str
    rgb: np.ndarray | None = None
    fuse_s: float = 0.0


def load_capture(path, urdf, side: str, frames_used: int = 0, depth=None, with_rgb: bool = False) -> Capture:
    """a burst file -> its Capture. frames_used: fuse (and take the joints over) only the first N frames, as a
    shorter burst would give; depth: an already fused (H,W) depth to use instead (e.g. a stored fusion)"""
    b = burst_io.load(path)
    fr = Frames(b, urdf)
    a = fr.angles(frames_used)
    arm = sided(ARM, side)
    missing = [n for n in arm if n not in a]
    if missing:
        raise ValueError(f"{path}: the burst has no joint(s) {', '.join(missing)} (capture with --joints all)")
    moved = fr.neck_motion_deg(frames_used)
    worst = max(arm + ("NeckYaw", "NeckPitch"), key=lambda n: moved.get(n, 0.0))
    t0 = time.perf_counter()
    if depth is None:
        f = fuse.fuse_depth(b.depth, b.sigma, frames_used)
        depth, n = f.depth.astype(np.float64), f.frames
    else:
        depth, n = np.asarray(depth, np.float64), (frames_used or b.frames)
    fuse_s = time.perf_counter() - t0
    return Capture(name=capture_name(path, side), side=side.capitalize(), q={k: a[k] for k in arm}, angles=a,
                   T_TC=fr.T_FC("torso", frames_used), T_BC=fr.T_BC(), depth=depth,
                   K=tuple(float(k) for k in np.asarray(b.K).reshape(-1)), frames=n,
                   motion_deg=float(moved.get(worst, 0.0)), motion_joint=worst,
                   rgb=fuse.fuse_rgb(b.rgb, frames_used) if with_rgb else None, fuse_s=fuse_s)


def match(arm: meshfit.ArmModel, cap: Capture, cfg: meshfit.FitConfig = meshfit.FAST, view: bool = True,
          start_off=None) -> tuple:
    """(record, PoseFit) of one capture. start_off (mm, Torso frame; the robust matcher's pass 2): start at the
    model moved by this offset and run only PASS2_GATES"""
    H, W = cap.depth.shape
    t0 = time.perf_counter()
    data = meshfit.unproject(cap.depth, cap.K, cfg.data_stride)
    kw = {} if start_off is None else dict(E0=consensus_start(cap.T_TC, start_off), gates=PASS2_GATES)
    f = meshfit.fit_pose(arm, cap.q, cap.T_TC, data, cap.K, W, H, cfg, **kw)
    fit_s = time.perf_counter() - t0
    off = 1e3 * f.wrist_offset_T
    rec = dict(name=cap.name, side=cap.side, off=off.tolist(), norm=float(np.linalg.norm(off)), rot=f.rotation_deg,
               cov=f.coverage, ok=f.ok, E=np.asarray(f.delta_c).tolist(), T_TC=np.asarray(cap.T_TC).tolist(),
               T_BC=np.asarray(cap.T_BC).tolist(), q=cap.q, yaw=cap.angles["NeckYaw"], pitch=cap.angles["NeckPitch"],
               wrist=arm.wrist_point_T(cap.q).tolist(), rms=f.rms_mm, cond=f.cond, steps=f.steps,
               inl={k: v["inliers"] for k, v in f.per_link.items()}, frames=cap.frames,
               motion_deg=cap.motion_deg, motion_joint=cap.motion_joint, fit_s=round(fit_s, 3),
               fuse_s=round(cap.fuse_s, 3))
    if f.score is not None:            # the robust matcher: its score, the starts it compared, the margin
        rec.update(score=f.score, margin=f.margin, starts=f.candidates)
    if start_off is not None:
        rec["start_off"] = np.asarray(start_off, float).tolist()
    rec["clean"] = is_clean(rec)
    if view:
        rec["view"] = meshfit.view_gate(arm, cap.q, cap.T_TC, cap.K, W, H)
    return rec, f


def wave_median(recs, min_clean: int = 3) -> np.ndarray:
    """(3,) mm: the median wrist offset of one arm's captures (the clean ones when there are at least min_clean)"""
    use = [r for r in recs if r["clean"]]
    use = use if len(use) >= min_clean else list(recs)
    return np.median(np.array([r["off"] for r in use], float), axis=0)


def consensus_start(T_TC, off_mm) -> np.ndarray:
    """the camera-frame correction that moves the model by off_mm (Torso frame): pass 2's start at the arm's median
    offset, so that every capture starts where the wave as a whole puts the arm"""
    E = np.eye(4)
    E[:3, 3] = np.linalg.inv(np.asarray(T_TC, float))[:3, :3] @ (1e-3 * np.asarray(off_mm, float))
    return E


def is_clean(rec: dict) -> bool:
    return bool(rec["ok"] and rec["rot"] <= CLEAN_ROT_DEG and rec["cov"] >= CLEAN_COVERAGE)


def overlay(cap: Capture, arm: meshfit.ArmModel, fit: meshfit.PoseFit):
    """BGR picture: the model's outline at the encoders (red) and fitted (green), inliers coloured by residual
    (blue -5 mm .. red +5 mm); needs load_capture(..., with_rgb=True)"""
    import cv2
    from armcal import mesh
    H, W = cap.depth.shape
    img = np.ascontiguousarray(cap.rgb[..., ::-1]).copy()
    T_CT = np.linalg.inv(cap.T_TC)
    for E, colour in ((np.eye(4), (0, 0, 255)), (fit.delta_c, (0, 220, 0))):
        P, _, _ = arm.place(cap.q, T_CT, E)
        sil = (mesh.render(P, np.eye(4), cap.K, W, H)[0] > 0).astype(np.uint8)
        cs, _ = cv2.findContours(sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, cs, -1, colour, 1)
    d = fit.data[fit.inlier]
    if len(d):
        fx, fy, cx, cy = cap.K
        u = np.clip(np.rint(fx * d[:, 0] / d[:, 2] + cx).astype(int), 0, W - 1)
        v = np.clip(np.rint(fy * d[:, 1] / d[:, 2] + cy).astype(int), 0, H - 1)
        t = np.clip((fit.residual_mm[fit.inlier] + 5.0) / 10.0, 0, 1)
        img[v, u] = np.column_stack([255 * (1 - t), np.zeros_like(t), 255 * t]).astype(np.uint8)
    return img


_W = {}


def _work(job):
    """one capture in a worker: (record, overlay picture or None)"""
    path, start_off, pic = job
    a = _W["args"]
    cap = load_capture(path, _W["urdf"], a["side"], a["frames"], with_rgb=pic)
    rec, f = match(_W["arm"], cap, _W["cfg"], start_off=start_off)
    return rec, (overlay(cap, _W["arm"], f) if pic else None)


def match_wave(paths, urdf, side: str, cfg: meshfit.FitConfig = meshfit.FAST, frames: int = 0,
               consensus: bool | None = None, jobs: int = 1, overlays: bool = False, arm=None, log=print) -> list:
    """[(record, overlay or None)] of one arm's captures. consensus (default: on for the robust matcher): pass 1 as
    match() (from the encoders), then pass 2 starts every capture at the wave's median offset (wave_median of pass 1;
    the record keeps pass 1's in rec["pass1"]). A capture is "settled" when its two passes land within SETTLED_MM of
    each other: its answer does not hang on where the matcher started. clean then also needs settled. jobs > 1:
    captures in parallel (fork)"""
    consensus = cfg.robust if consensus is None else consensus
    _W.update(urdf=Urdf.of(urdf), cfg=cfg, arm=arm or arm_model(urdf, side, cfg=cfg),
              args=dict(side=side, frames=frames))

    def run(job_list):
        if jobs > 1 and len(job_list) > 1:
            from multiprocessing import get_context
            with get_context("fork").Pool(min(jobs, len(job_list))) as pool:
                return pool.map(_work, job_list, chunksize=1)
        return [_work(j) for j in job_list]

    out = run([(p, None, overlays and not consensus) for p in paths])
    if consensus and out:
        med = wave_median([r for r, _ in out])
        log(f"pass 2 from the wave's median offset {np.round(med, 1).tolist()} mm")
        out2 = run([(p, med.tolist(), overlays) for p in paths])
        for (r1, _), (r2, _) in zip(out, out2):
            r2["pass1"] = {k: r1[k] for k in ("off", "rot", "cov", "clean", "fit_s")}
            r2["agree_mm"] = float(np.linalg.norm(np.subtract(r2["off"], r1["off"])))
            r2["settled"] = r2["agree_mm"] <= SETTLED_MM
            r2["clean"] = bool(r2["clean"] and r2["settled"])
        out = out2
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("bursts", nargs="+")
    ap.add_argument("--side", choices=("right", "left"), required=True)
    ap.add_argument("--urdf", required=True, help="the robot's URDF (the one its consumers read now)")
    ap.add_argument("--out", required=True, help="the records, JSON")
    ap.add_argument("--meshes", default=str(MESHES))
    ap.add_argument("--frames", type=int, default=0, help="fuse only the first N depth frames (0 = all)")
    ap.add_argument("--matcher", choices=tuple(MATCHERS), default=DEFAULT_MATCHER,
                    help="robust: the even model and the robust solver (readb2, 11 Nov 2026); fast: the phantom14 "
                         "algorithm's numbers, computed faster (what stored records were made with); reference: "
                         "the phantom14 algorithm exactly as it ran (slow)")
    ap.add_argument("--reference", action="store_true", help="= --matcher reference")
    ap.add_argument("--overlay", help="a directory for overlay-<name>.png")
    ap.add_argument("--no-consensus", action="store_true", help="the robust matcher without its pass 2")
    ap.add_argument("--jobs", type=int, default=1, help="captures matched in parallel (one core each)")
    a = ap.parse_args(argv)
    urdf = Urdf.load(a.urdf)
    cfg = MATCHERS["reference" if a.reference else a.matcher]
    arm = arm_model(urdf, a.side, a.meshes, cfg=cfg)
    out = match_wave(a.bursts, urdf, a.side, cfg, a.frames, consensus=False if a.no_consensus else None,
                     jobs=a.jobs, overlays=bool(a.overlay), arm=arm)
    recs = []
    for rec, img in out:
        recs.append(rec)
        if img is not None:
            import cv2
            Path(a.overlay).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(Path(a.overlay) / f"overlay-{rec['name']}.png"), img)
        print(f"{rec['name']}: wrist {np.round(rec['off'], 1).tolist()} mm |{rec['norm']:.1f}|, rot {rec['rot']:.1f} "
              f"deg, coverage {rec['cov']:.2f}, {'CLEAN' if rec['clean'] else 'NOT CLEAN'}, motion "
              f"{rec['motion_deg']:.2f} deg ({rec['motion_joint']}), {rec['fit_s']:.1f} s", flush=True)
    Path(a.out).write_text(json.dumps(dict(urdf=str(a.urdf), urdf_md5=urdf.md5, side=a.side.capitalize(),
                                           config=repr(cfg), res=recs), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
