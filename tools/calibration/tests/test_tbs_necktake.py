"""T_BS steps: joints, interpolation, joint rates and held poses of a neck take."""
import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
import necktake  # noqa: E402

NAMES = ["LeftShoulderPitch", "NeckPitch", "NeckYaw", "RightKnee"]
T0 = 1_792_000_000_000_000_000


def _messages(t_s, yaw, pitch):
    return [{"timestamp_ns": int(t * 1e9) + T0, "num_joints": 4,
             "positions": [0.0, p, y, 0.0]} for t, y, p in zip(t_s, yaw, pitch)]


def _joints(knots_t, yaw_knots, pitch_knots, tremor_deg=0.0, hz=8.0):
    t = np.arange(0, knots_t[-1], 1 / 467)
    tr = np.radians(tremor_deg) * np.sin(2 * np.pi * hz * t)
    y, p = np.interp(t, knots_t, yaw_knots) + tr, np.interp(t, knots_t, pitch_knots) - tr
    return necktake.joints_from_messages(_messages(t, y, p), NAMES)


def _profile():
    """2 s hold, 1 s move, 2 s hold, 0.5 s move, 0.5 s hold (too short), 0.5 s move. 467 Hz."""
    t = np.arange(0, 6.5, 1 / 467)
    knots_t = [0, 2, 3, 5, 5.5, 6, 6.5]
    yaw = np.interp(t, knots_t, [0.1, 0.1, -0.3, -0.3, 0.2, 0.2, 0.0])
    pitch = np.interp(t, knots_t, [-0.2, -0.2, 0.3, 0.3, 0.0, 0.0, 0.1])
    return t, yaw, pitch


def _rel(j, t_ns):
    return (t_ns - int(j.t[0])) / 1e9


def test_joints_are_picked_by_name_and_sorted_by_time():
    t, y, p = _profile()
    j = necktake.joints_from_messages(_messages(t, y, p)[::-1], NAMES)
    assert np.all(np.diff(j.t) > 0)
    assert np.allclose(j.yaw[:10], 0.1) and np.allclose(j.pitch[:10], -0.2)


def test_interpolation_and_rates():
    t, y, p = _profile()
    j = necktake.joints_from_messages(_messages(t, y, p), NAMES)
    qy, qp = necktake.at(j, j.t[0] + np.array([int(2.5e9)]))
    assert abs(qy[0] - (-0.1)) < 1e-6 and abs(qp[0] - 0.05) < 1e-6
    qdy, qdp = necktake.rates_at(j, j.t[0] + np.array([int(2.5e9)]))
    assert abs(qdy[0] - (-0.4)) < 1e-2 and abs(qdp[0] - 0.5) < 1e-2


def test_rates_are_central_differences_over_the_given_window():
    """On q = A sin(w t), a central difference over +-h gives A w cos(w t) sin(w h) / (w h)."""
    t = np.arange(0, 4, 1 / 467)
    A, w, h = 0.3, 2 * np.pi * 1.5, 0.05
    j = necktake.joints_from_messages(_messages(t, A * np.sin(w * t), 0 * t), NAMES)
    ts = np.array([0.7, 1.3, 2.1])
    qdy, qdp = necktake.rates_at(j, j.t[0] + (ts * 1e9).astype(np.int64), half_window_s=h)
    assert np.allclose(qdy, A * w * np.cos(w * ts) * np.sin(w * h) / (w * h), atol=2e-4)
    assert np.allclose(qdp, 0.0)


@pytest.mark.parametrize("tremor_deg", [0.0, 0.03, 0.1])
def test_only_long_enough_holds_are_found_even_with_tremor(tremor_deg):
    """8 Hz: unlike 10 Hz, it does not cancel exactly in the +-150 ms window, so a wrong window or
    threshold shows. The hold boundaries are pinned: 0.2 s trimmed off each end of the still run."""
    j = _joints([0, 2, 3, 5, 5.5, 6, 6.5], [0.1, 0.1, -0.3, -0.3, 0.2, 0.2, 0.0],
                [-0.2, -0.2, 0.3, 0.3, 0.0, 0.0, 0.1], tremor_deg)
    holds = necktake.find_holds(j)
    assert len(holds) == 2
    assert abs(holds[0].yaw - 0.1) < 2e-3 and abs(holds[0].pitch + 0.2) < 2e-3
    assert abs(holds[1].yaw + 0.3) < 2e-3 and abs(holds[1].pitch - 0.3) < 2e-3
    assert abs(_rel(j, holds[0].t0) - 0.2) < 0.015 and abs(_rel(j, holds[0].t1) - 1.667) < 0.015
    assert abs(_rel(j, holds[1].t0) - 3.333) < 0.015
    assert holds[0].drift_rad <= 2 * np.radians(tremor_deg) + 1e-9


def test_holds_record_the_side_they_were_reached_from():
    t, y, p = _profile()
    j = necktake.joints_from_messages(_messages(t, y, p), NAMES)
    h = necktake.find_holds(j)
    assert (h[0].approach_yaw, h[0].approach_pitch) == (0, 0)     # the take starts there
    assert (h[1].approach_yaw, h[1].approach_pitch) == (-1, 1)    # yaw came down, pitch went up


@pytest.mark.parametrize("over, tremor_deg", [(0.05, 0.0), (0.03, 0.0), (0.05, 0.1)])
def test_an_overshoot_counts_as_the_last_motion(over, tremor_deg):
    """The hand goes past the pose and comes back: the joint arrived moving down, not up. With
    tremor, a still joint must still read 0 (the side threshold is above the tremor)."""
    j = _joints([0, 2, 2.6, 2.9, 5], [0.0, 0.0, 0.2 + over, 0.2, 0.2], [0.0, 0.0, 0.0, 0.0, 0.0], tremor_deg)
    h = necktake.find_holds(j)
    assert len(h) == 2 and h[1].approach_yaw == -1 and h[1].approach_pitch == 0


def test_a_slow_correction_still_names_its_side():
    """Overshoot, then a slow 0.8 s correction back (just above the hold threshold): the last
    0.02 rad of motion lies more than 0.3 s before the hold, still within the 1 s looked back."""
    j = _joints([0, 2, 2.4, 3.2, 5.5], [0.0, 0.0, 0.225, 0.2, 0.2], [0.0] * 5)
    h = necktake.find_holds(j)
    assert len(h) == 2 and h[1].approach_yaw == -1


def test_a_pitch_only_move_splits_holds_and_a_hold_may_end_the_take():
    j = _joints([0, 2, 2.7, 5, 5.7, 8], [0.2, 0.2, 0.2, 0.2, 0.2, 0.2], [0.0, 0.0, 0.15, 0.15, -0.15, -0.15])
    h = necktake.find_holds(j)
    assert len(h) == 3 and [x.approach_pitch for x in h] == [0, 1, -1] and h[1].approach_yaw == 0
    assert abs(_rel(j, h[2].t1) - (_rel(j, j.t[-1]) - 0.2)) < 0.005   # trimmed at the take's end


def test_the_minimum_duration_counts_after_trimming():
    """A 1.5 s still stretch reads as ~1.23 s still (the +-150 ms speed window eats the edges of the
    moves) and ~0.83 s once 0.2 s is trimmed off each end: not a hold."""
    j = _joints([0, 2, 2.5, 4.0, 4.5, 6.8], [0.0, 0.0, 0.2, 0.2, 0.4, 0.4], [0.0] * 6)
    assert [round(h.yaw, 2) for h in necktake.find_holds(j)] == [0.0, 0.4]


@pytest.mark.parametrize("joint", ["yaw", "pitch"])
def test_a_slow_drift_is_reported(joint):
    sag = [0.0, 0.0, 0.3, 0.32, 0.32]                                             # 0.008 rad/s
    j = _joints([0, 2, 2.5, 5, 5.5], *((sag, [0.0] * 5) if joint == "yaw" else ([0.0] * 5, sag)))
    h = necktake.find_holds(j)
    assert len(h) == 2 and h[0].drift_rad < 1e-9 and 0.015 < h[1].drift_rad < 0.025


def test_read_joints_from_a_session_and_its_failures(tmp_path):
    mcap_writer = pytest.importorskip("mcap.writer")
    t, y, p = _profile()

    def session(name, msgs, meta):
        path = tmp_path / name
        with open(path, "wb") as f:
            wr = mcap_writer.Writer(f, compression=mcap_writer.CompressionType.NONE)
            wr.start()
            sid = wr.register_schema(name="okvis.JointState", encoding="jsonschema", data=b"{}")
            ch = wr.register_channel(topic="/joints", message_encoding="json", schema_id=sid)
            for m in msgs:
                wr.add_message(ch, log_time=m["timestamp_ns"], publish_time=m["timestamp_ns"], data=json.dumps(m).encode())
            if meta:
                wr.add_metadata("robot", {"joint_names": ",".join(NAMES)})
            wr.finish()
        return path

    msgs = _messages(t[:200], y[:200], p[:200])
    j = necktake.read_joints(session("ok.mcap", msgs, True))
    assert len(j.t) == 200 and np.allclose(j.yaw, 0.1) and np.allclose(j.pitch, -0.2)
    with pytest.raises(ValueError, match="no /joints"):
        necktake.read_joints(session("empty.mcap", [], True))
    nameless = session("nameless.mcap", msgs, False)
    with pytest.raises(ValueError, match="no joint names"):
        necktake.read_joints(nameless)
    assert np.allclose(necktake.read_joints(nameless, names=NAMES).yaw, 0.1)
