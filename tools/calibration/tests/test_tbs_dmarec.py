"""T_BS steps: picking dma-recorder segments, reading them, and the torso's motion."""
import os
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
import dmarec_imu  # noqa: E402
import necktake  # noqa: E402
import rigmath as rm  # noqa: E402

T0 = 1_792_000_000_000_000_000
LSB = 0.0005326401442289352            # phantom12's gyro step [rad/s] (16 Oct, indices 2 and 9)


def _segment(path, end_s, closed=True):
    path.write_bytes(dmarec_imu.MCAP_MAGIC + b"\x00" * 64 + (dmarec_imu.MCAP_MAGIC if closed else b"\x01" * 8))
    os.utime(path, (end_s, end_s))


def test_segments_are_picked_by_their_minute_and_sorted_by_state(tmp_path):
    t0 = 1_792_100_000  # s
    for k in range(5):  # one-minute segments, each file's mtime = the end of its minute
        _segment(tmp_path / f"recording_x_{k}.mcap", t0 + 60 * (k + 1), closed=k not in (1, 3))
    closed, still_open, unfinished = dmarec_imu.segments_covering(
        tmp_path, int((t0 + 90) * 1e9), int((t0 + 200) * 1e9), now_s=t0 + 250)
    assert [g.name for g in closed] == ["recording_x_2.mcap"]
    assert [g.name for g in still_open] == ["recording_x_3.mcap"]    # written 10 s ago, no footer yet
    assert [g.name for g in unfinished] == ["recording_x_1.mcap"]    # no footer 130 s on: never finished


def test_copies_keep_the_mtime(tmp_path):
    src = tmp_path / "recording_a_1.mcap"
    _segment(src, 1_792_100_060)
    (out,) = dmarec_imu.copy_segments([src], tmp_path / "copy")
    assert out.read_bytes() == src.read_bytes() and out.stat().st_mtime == 1_792_100_060


def _raw_imu_segment(path, frames):
    """A /raw_imu segment in the recorder's format (dma.recording.RawImuFrame), written with
    mcap-protobuf-support. frames: [(stamp_ns, [(gyro_z, accel_z) per IMU])]."""
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
    from mcap_protobuf.writer import Writer
    fdp = descriptor_pb2.FileDescriptorProto(name="raw_imu_test.proto", package="dma.recording", syntax="proto3")
    imu = fdp.message_type.add(name="Imu")
    for i, n in enumerate(["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"], 1):
        imu.field.add(name=n, number=i, type=descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT,
                      label=descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL)
    fr = fdp.message_type.add(name="RawImuFrame")
    fr.field.add(name="timestamp_ns", number=1, type=descriptor_pb2.FieldDescriptorProto.TYPE_UINT64,
                 label=descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL)
    fr.field.add(name="imus", number=2, type=descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
                 label=descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED, type_name=".dma.recording.Imu")
    pool = descriptor_pool.DescriptorPool()
    pool.Add(fdp)
    Frame = message_factory.GetMessageClass(pool.FindMessageTypeByName("dma.recording.RawImuFrame"))
    with open(path, "wb") as f, Writer(f) as w:
        for stamp, imus in frames:
            m = Frame(timestamp_ns=stamp)
            for gz, az in imus:
                m.imus.add(gyro_z=gz, accel_z=az)
            w.write_message(topic="/raw_imu", message=m, log_time=stamp, publish_time=stamp)


def test_read_picks_the_imu_sorts_windows_and_survives_bad_segments(tmp_path):
    pytest.importorskip("mcap")
    pytest.importorskip("mcap_protobuf")
    one, two = tmp_path / "recording_t_1.mcap", tmp_path / "recording_t_2.mcap"
    frames = [(T0 + k * 2_000_000, [(0.01 * i + k, 9.81) for i in range(10)]) for k in (3, 0, 1, 2, 4)]
    frames.append((T0 + 5 * 2_000_000, [(0.0, 9.81)] * 4))           # a frame without IMU 9
    _raw_imu_segment(one, frames)
    _raw_imu_segment(two, [(T0 + k * 2_000_000, [(0.01 * i + k, 9.81) for i in range(10)]) for k in (6, 7, 9)])
    t, w, a, errors = dmarec_imu.read_raw_imu([two, one], 9, T0 + 2_000_000, T0 + 14_000_000)
    assert list(t - T0) == [2_000_000, 4_000_000, 6_000_000, 8_000_000, 12_000_000, 14_000_000]
    assert np.allclose(w[:, 2], [1.09, 2.09, 3.09, 4.09, 6.09, 7.09], atol=1e-5) and np.allclose(a[:, 2], 9.81, atol=1e-5)
    assert errors == [(str(one), "1 frames without IMU 9")]
    unfinished = tmp_path / "recording_t_3.mcap"                     # the footer's magic never written
    unfinished.write_bytes(two.read_bytes()[:-len(dmarec_imu.MCAP_MAGIC)])
    t, w, a, errors = dmarec_imu.read_raw_imu([unfinished], 9)
    assert list(t - T0) == [12_000_000, 14_000_000, 18_000_000] and len(errors) == 1
    bad = tmp_path / "recording_bad_1.mcap"
    bad.write_bytes(dmarec_imu.MCAP_MAGIC + b"\x00" * 64 + dmarec_imu.MCAP_MAGIC)
    t, w, a, errors = dmarec_imu.read_raw_imu([bad], 9)
    assert len(t) == 0 and w.shape == (0, 3) and len(errors) == 1 and errors[0][0] == str(bad)


MOUNT = rm.rot_axis([0.3, -0.5, 0.8], 2.1)            # the torso IMU, any way round


def _torso(turn_after=None, creep_in=None, slow=None, seed=0, n_holds=12, move=(0.1, 0.0, 0.2),
           creep_moves=None, rock=0.0):
    """500 Hz torso IMU, mounted MOUNT, over n_holds holds of 3 s with 2 s moves between them;
    gyro quantised to LSB with a sub-LSB bias drifting 1e-4 rad/s a minute (2.5 x the wander
    measured on phantom12's torso IMU over 45 s), a 1 s gap in the move after hold 8.
    turn_after=k: during the move after hold k the torso turns `move` deg about its x, y, z
    (default: 0.1 deg of tilt about x and 0.2 deg about the vertical). creep_in=k: it turns
    0.2 deg about the vertical during hold k. slow=(t0, t1, deg): it turns deg about the vertical
    evenly from t0 to t1 [s]. creep_moves=(k0, k1, deg): it turns deg about the vertical in each
    move after holds k0..k1-1. rock=r: during each hold the robot turns about the vertical at a
    random rate (1-sigma r [rad/s]) and back during the following move, as it rocked on its gantry
    on 16 Oct."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, 5 * n_holds, 1 / 500)
    holds = [necktake.Hold(T0 + int(5 * k * 1e9), T0 + int((5 * k + 3) * 1e9), 0.0, 0.0) for k in range(n_holds)]
    w_T = np.zeros((len(t), 3))                                   # torso body rate, torso axes
    if turn_after is not None:
        m = (t > 5 * turn_after + 3) & (t < 5 * turn_after + 5)
        w_T[m] = np.radians(move) / 2.0
    if creep_in is not None:
        m = (t >= 5 * creep_in) & (t < 5 * creep_in + 3)
        w_T[m, 2] += np.radians(0.2) / 3.0
    if slow is not None:
        m = (t >= slow[0]) & (t < slow[1])
        w_T[m, 2] += np.radians(slow[2]) / (slow[1] - slow[0])
    if creep_moves is not None:
        for k in range(creep_moves[0], creep_moves[1]):
            m = (t > 5 * k + 3) & (t < 5 * k + 5)
            w_T[m, 2] += np.radians(creep_moves[2]) / 2.0
    if rock:
        for k, r in enumerate(rock * np.random.default_rng(seed + 1000).standard_normal(n_holds)):
            w_T[(t >= 5 * k) & (t < 5 * k + 3), 2] += r
            w_T[(t > 5 * k + 3) & (t < 5 * k + 5), 2] -= r * 3 / 2
    tilt = np.cumsum(w_T[:, 0]) / 500
    a_T = np.stack([np.zeros_like(t), 9.81 * np.sin(tilt), 9.81 * np.cos(tilt)], 1)
    bias = LSB * np.array([0.4, -0.3, 0.45]) + np.outer(t / 60.0, [1e-4, -0.6e-4, 0.8e-4])
    g = np.round((w_T @ MOUNT.T + bias + 0.0008 * rng.standard_normal(w_T.shape)) / LSB) * LSB
    a = a_T @ MOUNT.T + 0.01 * rng.standard_normal(a_T.shape)
    keep = ~((t > 5 * 8 + 3.5) & (t < 5 * 8 + 4.5))
    return T0 + (t[keep] * 1e9).astype(np.int64), g[keep], a[keep], holds


def test_a_still_torso_reads_still_despite_a_quantised_drifting_gyro():
    m = dmarec_imu.torso_motion(*_torso())
    assert max(abs(h["yaw_deg"]) for h in m) < 0.05 and max(h["tilt_deg"] for h in m) < 0.01
    assert max(h["turn_deg"] for h in m) < 0.02


def test_a_long_still_take_reads_still_because_the_bias_is_local():
    """180 s: the bias drifts 3e-4 rad/s over the take; one median over all holds would read
    0.24-0.29 deg at its ends, the 9-hold window stays near 0.05."""
    m = dmarec_imu.torso_motion(*_torso(n_holds=36))
    assert max(abs(h["yaw_deg"]) for h in m) < 0.08 and max(h["tilt_deg"] for h in m) < 0.01
    assert max(h["turn_deg"] for h in m) < 0.03


def test_a_torso_that_turned_between_holds_is_caught_against_the_majority():
    """It moved after hold 3: holds 4..11 are the majority, so holds 0..3 are the ones off."""
    m = dmarec_imu.torso_motion(*_torso(turn_after=3))
    assert [h["in_reference"] for h in m] == [False] * 4 + [True] * 8
    for h in m[:4]:
        assert abs(abs(h["yaw_deg"]) - 0.2) < 0.03 and abs(h["tilt_deg"] - 0.1) < 0.01
    for h in m[4:]:
        assert abs(h["yaw_deg"]) < 0.05 and h["tilt_deg"] < 0.01


def test_a_step_in_the_middle_puts_one_whole_side_off():
    """6 holds either side of the step: one side is the reference, the other reads the full step."""
    m = dmarec_imu.torso_motion(*_torso(turn_after=5))
    off = [k for k, h in enumerate(m) if abs(h["yaw_deg"]) > 0.1 or h["tilt_deg"] > 0.05]
    assert off in (list(range(6)), list(range(6, 12)))
    for k in off:
        assert 0.17 < abs(m[k]["yaw_deg"]) < 0.25 and abs(m[k]["tilt_deg"] - 0.1) < 0.01


def test_the_yaw_says_how_far_to_trust_it():
    """Still torso: the holds' mean rates scatter at the gyro's level and the yaw's error bar stays
    small near the reference; a robot rocking during the holds (5e-4 rad/s, as on 16 Oct) scatters
    10x more and the error bar says a tenth of a degree of yaw means nothing there."""
    still = dmarec_imu.torso_motion(*_torso())
    assert still[0]["rate_scatter_rad_s"] < 1e-4 and max(h["yaw_sigma_deg"] for h in still) < 0.05
    rock = dmarec_imu.torso_motion(*_torso(rock=5e-4))
    assert 2.5e-4 < rock[0]["rate_scatter_rad_s"] < 1e-3 and max(h["yaw_sigma_deg"] for h in rock) > 0.3
    s = [h["yaw_sigma_deg"] for h in rock]                          # grows away from the reference
    assert min(s) < 0.3 * max(s) and max(s) in (s[0], s[-1])
    # and the rocking does not split the holds into small groups: most stay in the reference
    assert sum(h["in_reference"] for h in rock) >= 9
    stepped = dmarec_imu.torso_motion(*_torso(rock=5e-4, turn_after=5, move=(0.3, 0.0, 0.0)))   # a real tilt step
    assert sum(h["in_reference"] for h in stepped) == 6


def test_a_tilt_only_step_in_the_middle_puts_one_whole_side_off():
    """0.1 deg of tilt and no turn after hold 5: holds split on tilt as well as on yaw (on yaw
    alone the 12 holds stay one group and its median direction sits halfway, ~0.05 deg from each)."""
    m = dmarec_imu.torso_motion(*_torso(turn_after=5, move=(0.1, 0.0, 0.0)))
    off = [k for k, h in enumerate(m) if h["tilt_deg"] > 0.05]
    assert off in (list(range(6)), list(range(6, 12)))
    assert all(abs(m[k]["tilt_deg"] - 0.1) < 0.01 for k in off)
    assert max(h["tilt_deg"] for k, h in enumerate(m) if k not in off) < 0.01
    assert max(abs(h["yaw_deg"]) for h in m) < 0.05


def test_a_creep_in_the_majority_does_not_shrink_the_step_before_it():
    """0.15 deg about the vertical after hold 3, then the torso creeps back 0.015 deg in each move
    after holds 4..10 (0.105 deg in all, each move below the 0.05 deg split). Chained across its
    boundary, holds 0..3 read the step itself, 0.15 (their distance to hold 4); against the
    majority's median, which the creep moved by half its total, ~0.1; the larger counts. Within
    the majority each hold keeps its place: the creep shows across it (0.0975 deg, half a move lost
    in the gap; the bias estimate's noise moves it by up to ~0.03, 0.065-0.111 over 40 seeds)."""
    m = dmarec_imu.torso_motion(*_torso(turn_after=3, move=(0.0, 0.0, 0.15), creep_moves=(4, 11, -0.015)))
    assert all(abs(abs(h["yaw_deg"]) - 0.15) < 0.03 for h in m[:4])
    assert max(abs(h["yaw_deg"]) for h in m[4:]) < 0.09
    assert abs(m[4]["yaw_deg"] - m[11]["yaw_deg"]) > 0.05


def test_a_slide_over_several_moves_is_caught_though_only_one_move_splits():
    """The torso slides 0.03 deg about the vertical in each move after holds 4..7 and 0.06 more in
    the one after hold 6 (0.18 deg in all; only that move is above the 0.05 deg split): holds 8..11
    sit 0.18 deg off holds 0..4. Chained through the one boundary they read 0.08-0.10 (the rest
    looks like drift inside the groups); against the majority's median they read all of it."""
    m = dmarec_imu.torso_motion(*_torso(turn_after=6, move=(0.0, 0.0, 0.06), creep_moves=(4, 8, 0.03)))
    assert all(abs(h["yaw_deg"]) > 0.13 for h in m[8:])
    assert max(abs(h["yaw_deg"]) for h in m[:5]) < 0.05


def test_a_turn_inside_a_hold_is_caught_and_carried_into_later_holds():
    """0.2 deg during hold 8 of 12: holds 0..7 are the majority, holds 9..11 sit 0.2 deg round."""
    m = dmarec_imu.torso_motion(*_torso(creep_in=8))
    assert abs(m[8]["turn_deg"] - 0.2) < 0.03
    assert max(h["turn_deg"] for k, h in enumerate(m) if k != 8) < 0.02
    assert max(abs(h["yaw_deg"]) for h in m[:8]) < 0.05
    assert all(abs(abs(h["yaw_deg"]) - 0.2) < 0.03 for h in m[9:])


def test_a_slow_turn_over_three_holds_is_not_taken_for_drift():
    """0.3 deg spread over holds 4..6 and the moves between them (15 s): a 5-hold bias window
    absorbs it as drift (the holds before and after read -0.03..0.04 deg apart over 100 seeds); with
    9 they end up 0.22-0.29 deg apart (the window still takes up to a quarter of so slow a turn),
    whichever side is the reference, and the three holds during it show their turn (>= 0.044)."""
    m = dmarec_imu.torso_motion(*_torso(slow=(20.0, 35.0, 0.3)))
    before, after = [h["yaw_deg"] for h in m[:4]], [h["yaw_deg"] for h in m[7:]]
    assert 0.15 < np.mean(after) - np.mean(before) < 0.33
    assert max(map(abs, before)) < 0.05 or max(map(abs, after)) < 0.05
    assert all(m[k]["turn_deg"] > 0.04 for k in (4, 5, 6))
