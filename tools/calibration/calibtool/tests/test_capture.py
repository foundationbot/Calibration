"""calibtool.capture: the head-camera reader. Pairs are matched by capture stamp, every camera has frame-health
counters (reads, None reads = pool desync, sequence gaps, repeated frames = freeze, fps), the capture loop never
blocks on a slow consumer, and a camera whose reads keep coming back None gets its own consumer re-attached."""
import threading
import time
import types

import pytest

np = pytest.importorskip("numpy")

from calibtool import capture as cp  # noqa: E402

MS = 1_000_000


def fr(cam, seq, t, val=0):
    return cp.Frame(cam, seq, t, t + 20 * MS, np.full((4, 6), val, np.uint8))


class FakeSource:
    """scripted reads per camera: a Frame, cp.NONE_READ (meta came, pool read None) or None (nothing new)"""

    def __init__(self, script):
        self.script = {c: list(v) for c, v in script.items()}
        self.reattached = []

    def read(self, cam, timeout):
        s = self.script.get(cam)
        return s.pop(0) if s else None

    def reattach(self, cam):
        self.reattached.append(cam)


# ------------------------------------------------------------------ PairMatcher

def test_pairs_match_by_capture_stamp_within_tolerance():
    m = cp.PairMatcher(3, 0, tol_ns=500_000)
    assert m.push(fr(3, 1, 1000 * MS)) == []
    out = m.push(fr(0, 1, 1000 * MS + 200_000))
    assert len(out) == 1
    p = out[0]
    assert (p.t0, p.t1) == (1000 * MS, 1000 * MS + 200_000)
    assert p.dt_ns == 200_000                   # right - left
    assert p.left.cam == 3 and p.right.cam == 0


def test_stamps_further_apart_than_tolerance_do_not_pair():
    m = cp.PairMatcher(3, 0, tol_ns=500_000)
    m.push(fr(3, 1, 1000 * MS))
    assert m.push(fr(0, 1, 1000 * MS + 600_000)) == []


def test_unmatched_frames_age_out():
    m = cp.PairMatcher(3, 0, tol_ns=500_000, keep=3)
    for i in range(6):
        m.push(fr(3, i, (1000 + 40 * i) * MS))
    assert len(m.pending(3)) == 3
    assert m.dropped[3] == 3


def test_right_before_left_pairs_too():
    m = cp.PairMatcher(3, 0)
    m.push(fr(0, 1, 1000 * MS - 100_000))
    out = m.push(fr(3, 1, 1000 * MS))
    assert len(out) == 1 and out[0].dt_ns == -100_000


# ------------------------------------------------------------------ FrameCounters

def test_counters_sequence_gap_and_repeat():
    c = cp.FrameCounters()
    c.frame(fr(3, 10, 1000 * MS), now=1.0)
    c.frame(fr(3, 13, 1120 * MS), now=1.1)      # 11, 12 missing
    c.frame(fr(3, 13, 1120 * MS), now=1.2)      # same frame again: a repeat (freeze)
    assert c.frames == 3
    assert c.seq_gaps == 2
    assert c.repeats == 1
    assert c.last_seq == 13


def test_counters_none_run_resets_on_a_good_read():
    c = cp.FrameCounters()
    c.none_read(now=1.0)
    c.none_read(now=1.1)
    assert (c.none_reads, c.none_run) == (2, 2)
    c.frame(fr(3, 1, 1000 * MS), now=1.2)
    assert (c.none_reads, c.none_run) == (2, 0)


def test_counters_fps_from_capture_stamps():
    c = cp.FrameCounters()
    for i in range(26):
        c.frame(fr(3, i, 1000 * MS + i * 40 * MS), now=1.0 + i * 0.04)
    assert c.fps() == pytest.approx(25.0, rel=1e-6)


def test_counters_as_dict_is_json_ready():
    import json
    c = cp.FrameCounters()
    c.frame(fr(3, 1, 1000 * MS), now=1.0)
    json.dumps(c.as_dict(now=2.0))
    assert c.as_dict(now=2.0)["since_frame_s"] == pytest.approx(1.0)


def test_pair_stats_sync():
    s = cp.PairStats()
    for dt in (10_000, -20_000, 14_000):
        s.add(types.SimpleNamespace(dt_ns=dt))
    assert s.pairs == 3
    assert s.median_abs_dt_ns() == 14_000
    assert s.max_abs_dt_ns() == 20_000


# ------------------------------------------------------------------ Capture

def _pairs_script(n, t0=1000 * MS, period=40 * MS):
    left = [fr(3, i, t0 + i * period, i) for i in range(n)]
    right = [fr(0, i, t0 + i * period + 14_000, i) for i in range(n)]
    return {3: left, 0: right}


def test_step_reads_both_cameras_and_pairs():
    cap = cp.Capture(FakeSource(_pairs_script(3)), 3, 0, clock=lambda: 1.0)
    for _ in range(3):
        cap.step()
    assert cap.counters[3].frames == 3 and cap.counters[0].frames == 3
    assert cap.pair_stats.pairs == 3
    assert cap.latest().left.seq == 2


def test_reattach_after_a_run_of_none_reads():
    src = FakeSource({3: [cp.NONE_READ] * cp.REATTACH_AFTER + [cp.NONE_READ] * 3, 0: []})
    cap = cp.Capture(src, 3, 0, clock=lambda: 1.0)
    for _ in range(cp.REATTACH_AFTER + 3):
        cap.step()
    assert src.reattached == [3]                # once, not on every None after it
    assert cap.counters[3].reattaches == 1
    assert cap.counters[3].none_reads == cp.REATTACH_AFTER + 3


def test_a_full_subscriber_never_blocks_the_loop():
    cap = cp.Capture(FakeSource(_pairs_script(5)), 3, 0, clock=lambda: 1.0)
    q = cap.subscribe(maxsize=1)
    t = time.monotonic()
    for _ in range(5):
        cap.step()
    assert time.monotonic() - t < 1.0
    assert q.qsize() == 1
    assert cap.sub_dropped == 4


def test_grab_returns_only_new_pairs():
    script = _pairs_script(10)
    src = FakeSource({c: v[:3] for c, v in script.items()})
    cap = cp.Capture(src, 3, 0)
    for _ in range(3):
        cap.step()                              # 3 old pairs
    src.script = {3: [], 0: []}
    cap.start(poll_s=0.001)
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("got", cap.grab(2, timeout=2.0)))
    try:
        t.start()
        time.sleep(0.05)                        # grab is waiting; now the new frames arrive
        src.script = {c: v[3:] for c, v in script.items()}
        t.join(3.0)
    finally:
        cap.stop()
    assert [p.left.seq for p in out["got"]] == [3, 4]


def test_grab_times_out_with_what_it_has():
    cap = cp.Capture(FakeSource({3: [], 0: []}), 3, 0)
    cap.start(poll_s=0.001)
    try:
        assert cap.grab(2, timeout=0.1) == []
    finally:
        cap.stop()


def test_thread_stops_cleanly():
    cap = cp.Capture(FakeSource({3: [], 0: []}), 3, 0)
    cap.start(poll_s=0.001)
    cap.stop()
    assert not cap.running


def test_snapshot_is_json_ready():
    import json
    cap = cp.Capture(FakeSource(_pairs_script(2)), 3, 0, clock=lambda: 1.0)
    cap.step()
    cap.step()
    s = cap.snapshot()
    json.dumps(s)
    assert s["pairs"] == 2 and s["cams"]["3"]["frames"] == 2
    assert s["sync_median_ms"] == pytest.approx(0.014)


# ------------------------------------------------------------------ DmaCameras over a fake dma_ipc

class _Meta:
    def __init__(self, cam, seq, t, w=6, h=4):
        self.camera_id, self.frame_sequence, self.capture_timestamp_ns = cam, seq, t
        self.publish_timestamp_ns, self.width, self.height = t + 1, w, h


def _fake_dma(reads):
    made = {"queues": 0, "pools": 0}

    class Consumer:
        def __init__(self, cam):
            self.cam = cam

        def next(self, timeout):
            r = reads[self.cam]
            return r.pop(0)[0] if r else None

    class Q:
        def __init__(self, name, create=False, wait=True):
            made["queues"] += 1
            self.cam = int(name[-1])

        def make_consumer(self, max_lag=8):
            return Consumer(self.cam)

    class Pool:
        def __init__(self, name, create=False):
            made["pools"] += 1
            self.cam = int(name[-1])
            self.last = {}

        def read(self, m):
            return m._buf

    mod = types.SimpleNamespace(VideoMetaQueue=Q, VideoFramePool=Pool,
                                video_meta_queue_name=lambda c: f"/video_meta_cam{c}",
                                video_frame_pool_name=lambda c: f"/video_frames_cam{c}")
    return mod, made


def test_dma_cameras_slices_the_y_plane_of_nv12():
    w, h = 6, 4
    m = _Meta(3, 7, 1000)
    y = np.arange(w * h, dtype=np.uint8)
    m._buf = memoryview(np.r_[y, np.full(w * h // 2, 255, np.uint8)].tobytes())
    mod, made = _fake_dma({3: [(m,)], 0: []})
    cams = cp.DmaCameras(mod, 3, 0)
    f = cams.read(3, 0.01)
    assert isinstance(f, cp.Frame)
    assert f.y.shape == (h, w) and f.y[1, 0] == 6 and f.y.max() < 255
    assert (f.cam, f.seq, f.t_ns) == (3, 7, 1000)
    assert cams.read(0, 0.01) is None


def test_dma_cameras_none_read_and_reattach():
    m = _Meta(0, 1, 1000)
    m._buf = None
    mod, made = _fake_dma({3: [], 0: [(m,)]})
    cams = cp.DmaCameras(mod, 3, 0)
    assert cams.read(0, 0.01) is cp.NONE_READ
    before = dict(made)
    cams.reattach(0)
    assert made["queues"] == before["queues"] + 1 and made["pools"] == before["pools"] + 1


def test_latest_age_is_the_time_since_the_newest_pair():
    """review 2: latest() never expires, so a frozen camera kept the board 'seen'; the age says how old it is"""
    now = [10.0]
    cap = cp.Capture(FakeSource(_pairs_script(2)), 3, 0, clock=lambda: now[0])
    assert cap.latest_age() is None
    cap.step()
    now[0] = 12.5
    assert cap.latest_age() == pytest.approx(2.5)
    cap.step()
    assert cap.latest_age() == pytest.approx(0.0)
