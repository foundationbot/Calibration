"""readb.py: a burst file -> fused depth -> B0 -> the capture record the stage-2 fit reads, either arm."""
import json

import numpy as np
import pytest

import synth
from armcal import burst_io, meshfit, readb

pytest.importorskip("cv2")


def write_burst(path, side, E=np.eye(4), n=3, move_deg=0.0):
    q, yaw, pitch, T_TC = synth.pose(side)
    frames = [synth.depth_image(side, E, seed=k) for k in range(n)]
    stamps = np.arange(n, dtype=np.int64) * 1_933_000_000 + 10**12
    names = list(q) + ["NeckYaw", "NeckPitch", "SpineYaw"]
    t = np.linspace(stamps[0] - 1e8, stamps[-1] + 1e8, 40)
    cols = [t] + [np.full(40, q[n_]) for n_ in q] + [np.full(40, yaw), np.full(40, pitch), np.zeros(40)]
    cols[1] = cols[1] + np.linspace(0, np.radians(move_deg), 40)      # the shoulder pitch drifts
    T_SC = np.eye(4)
    b = burst_io.Burst(depth=np.array(frames, np.float32), rgb=np.zeros((n, synth.H, synth.W, 3), np.uint8),
                       K=np.array(synth.K), T_SC=T_SC, stamp_ns=stamps, seq=np.arange(n),
                       sigma=np.full((n, synth.H, synth.W), 0.002, np.float32),
                       info={"joint_names": names, "okvis_config_sha256": "abc"},
                       joints=np.column_stack(cols), T_BS=synth.T_BC)
    return burst_io.save(path, b)


def test_sided_and_names():
    assert readb.sided(readb.LINKS, "left") == ("LeftElbowPitchLink", "LeftWristYawLink", "LeftWristRollLink")
    with pytest.raises(ValueError):
        readb.sided(readb.LINKS, "middle")
    assert readb.capture_name("p14-armwave-1104b-left-02-pL07-22-12.npz", "Left") == "02-pL07-22-12"
    assert readb.capture_name("x.npz", "Right") == "x"


@pytest.mark.parametrize("side", ["Right", "Left"])
def test_a_burst_gives_the_record(tmp_path, side):
    E = synth.rigid(1.5, (0, 0, 1), (3.0, 2.0, -1.0))
    p = write_burst(tmp_path / f"t-{side.lower()}-01-p01-10-0.npz", side, E)
    cap = readb.load_capture(p, synth.URDF, side)
    assert cap.name == "01-p01-10-0" and cap.frames == 3 and cap.depth.shape == (synth.H, synth.W)
    rec, f = readb.match(synth.arm(side), cap, meshfit.FAST)
    want = 1e3 * meshfit.wrist_offset(synth.arm(side), cap.q, cap.T_TC, E)
    assert np.linalg.norm(np.subtract(rec["off"], want)) < 0.7
    assert rec["clean"] and rec["view"]["ok"]
    assert rec["motion_deg"] < 1e-9
    json.dumps(rec)                                          # the record is JSON as it is


def test_motion_inside_the_burst_is_reported(tmp_path):
    p = write_burst(tmp_path / "m.npz", "Right", move_deg=1.0)
    cap = readb.load_capture(p, synth.URDF, "Right")
    assert cap.motion_joint == "RightShoulderPitch" and cap.motion_deg > 0.5


def test_fewer_frames_fuse_fewer(tmp_path):
    p = write_burst(tmp_path / "f.npz", "Right")
    assert readb.load_capture(p, synth.URDF, "Right", frames_used=1).frames == 1


def test_clean_is_ok_rotation_and_coverage():
    r = dict(ok=True, rot=11.9, cov=0.6)
    assert readb.is_clean(r)
    assert not readb.is_clean(dict(r, rot=12.1))
    assert not readb.is_clean(dict(r, cov=0.59))
    assert not readb.is_clean(dict(r, ok=False))


def test_consensus_start_moves_the_wrist_by_the_offset():
    q, _, _, T_TC = synth.pose("Left")
    off = np.array([12.0, -9.0, 15.0])
    E = readb.consensus_start(T_TC, off)
    got = 1e3 * meshfit.wrist_offset(synth.arm("Left"), q, T_TC, E)
    assert np.allclose(got, off, atol=1e-9)


def test_wave_median_prefers_the_clean_captures():
    recs = [dict(off=[1.0, 0, 0], clean=True), dict(off=[2.0, 0, 0], clean=True), dict(off=[3.0, 0, 0], clean=True),
            dict(off=[90.0, 0, 0], clean=False), dict(off=[95.0, 0, 0], clean=False)]
    assert np.allclose(readb.wave_median(recs), [2, 0, 0])
    assert np.allclose(readb.wave_median(recs[2:]), [90, 0, 0])      # under 3 clean: all of them


def test_the_robust_wave_runs_two_passes(tmp_path):
    E = synth.rigid(1.0, (0, 1, 0), (6.0, -4.0, 3.0))
    paths = [write_burst(tmp_path / f"t-right-0{k}-p0{k}-10-0.npz", "Right", E) for k in (1, 2, 3)]
    out = readb.match_wave(paths, synth.URDF, "Right", meshfit.ROBUST, log=lambda *_: None)
    assert len(out) == 3
    cap = readb.load_capture(paths[0], synth.URDF, "Right")
    arm = readb.arm_model(synth.URDF, "Right", cfg=meshfit.ROBUST)
    want = 1e3 * meshfit.wrist_offset(arm, cap.q, cap.T_TC, E)
    for rec, img in out:
        assert img is None and rec["clean"] and rec["settled"] and "pass1" in rec and len(rec["start_off"]) == 3
        assert np.linalg.norm(np.subtract(rec["off"], want)) < 0.7
    json.dumps([r for r, _ in out])


def test_the_fast_wave_is_one_pass(tmp_path):
    p = write_burst(tmp_path / "t-right-01-p01-10-0.npz", "Right")
    (rec, _), = readb.match_wave([p], synth.URDF, "Right", meshfit.FAST, arm=synth.arm("Right"))
    assert "pass1" not in rec and "start_off" not in rec
