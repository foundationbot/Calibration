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


# ------------------------------------------------------------------ re-attach: retried, after a stall, on a new segment

class Tick:
    def __init__(self, t=0.0, dt=0.04):
        self.t, self.dt = t, dt

    def __call__(self):
        return self.t


def _long_script(n, t0=1000 * MS, period=40 * MS):
    return {3: [fr(3, i, t0 + i * period, i % 256) for i in range(n)],
            0: [fr(0, i, t0 + i * period + 14_000, i % 256) for i in range(n)]}


def _steps(cap, clk, n):
    for _ in range(n):
        clk.t += clk.dt
        cap.step()


def test_reattach_is_retried_with_backoff_while_reads_stay_empty():
    """review 5: one re-attach at exactly 25 None reads, never again"""
    clk = Tick()
    src = FakeSource({3: [cp.NONE_READ] * 2000, 0: _long_script(2000)[0]})
    src.at = []
    src.reattach = lambda cam: (src.reattached.append(cam), src.at.append(clk.t))
    cap = cp.Capture(src, 3, 0, clock=clk)
    _steps(cap, clk, int(20 / clk.dt))           # 20 s of empty reads on queue 3
    assert src.reattached and set(src.reattached) == {3}
    gaps = [b - a for a, b in zip(src.at, src.at[1:])]
    assert len(src.at) >= 4 and gaps == sorted(gaps) and max(gaps) <= cp.REATTACH_MAX_S + 0.1, src.at
    assert cap.counters[3].reattaches == len(src.at)


def test_reattach_when_frames_stop_without_empty_reads():
    """review 5: after the producer recreates its segments the old consumer reads 'nothing new' (None), never
    NONE_READ, so the run-of-empty-reads rule never fired"""
    clk = Tick()
    sc = _pairs_script(10)
    src = FakeSource(sc)
    cap = cp.Capture(src, 3, 0, clock=clk)
    _steps(cap, clk, 10)
    assert src.reattached == []
    _steps(cap, clk, int((cp.STALL_REATTACH_S + 0.2) / clk.dt))
    assert sorted(set(src.reattached)) == [0, 3]
    src.script = _pairs_script(20, t0=5000 * MS)  # frames again: no more re-attaching
    n = len(src.reattached)
    _steps(cap, clk, 20)
    assert len(src.reattached) == n


def test_reattach_at_once_when_the_shm_segment_is_recreated():
    clk = Tick()
    sc = _long_script(200)
    src = FakeSource(sc)
    flag = {"changed": False}
    src.changed = lambda cam: flag["changed"] and cam == 3
    src.reattach = lambda cam: (src.reattached.append(cam), flag.update(changed=False))
    cap = cp.Capture(src, 3, 0, clock=clk)
    _steps(cap, clk, 10)
    flag["changed"] = True
    _steps(cap, clk, int(1.2 / clk.dt))
    assert src.reattached == [3]


def _shm_dma(tmp_path, block=None):
    """a fake dma_ipc whose segments are files in tmp_path; VideoMetaQueue blocks on `block` (an Event) when given"""
    made = {"queues": 0, "waits": []}

    class Consumer:
        def next(self, timeout):
            return None

    class Q:
        def __init__(self, name, create=False, wait=True):
            made["waits"].append(wait)
            if block is not None and made["queues"] >= 2:
                block.wait(10)
            made["queues"] += 1

        def make_consumer(self, max_lag=8):
            return Consumer()

    class Pool:
        def __init__(self, name, create=False):
            pass
    for c in (3, 0):
        for n in (f"video_meta_cam{c}", f"video_frames_cam{c}"):
            (tmp_path / n).write_bytes(b"x")
    mod = types.SimpleNamespace(VideoMetaQueue=Q, VideoFramePool=Pool,
                                video_meta_queue_name=lambda c: f"/video_meta_cam{c}",
                                video_frame_pool_name=lambda c: f"/video_frames_cam{c}")
    return mod, made


def test_dma_cameras_notice_a_recreated_segment(tmp_path):
    mod, made = _shm_dma(tmp_path)
    cams = cp.DmaCameras(mod, 3, 0, shm_dir=tmp_path)
    assert not cams.changed(3) and not cams.changed(0)
    seg = tmp_path / "video_meta_cam3"
    seg.unlink()
    (tmp_path / "keep").write_bytes(b"y")       # the inode is not reused for the new file
    seg.write_bytes(b"new")
    assert cams.changed(3) and not cams.changed(0)
    assert cams.reattach(3) is True
    assert not cams.changed(3)
    assert made["waits"][-1] is False           # a re-attach never waits for the producer


def test_a_hanging_reattach_does_not_block_the_reader(tmp_path):
    """review 5: _attach ran with wait=True inside the capture thread"""
    release = threading.Event()
    mod, made = _shm_dma(tmp_path, block=release)
    cams = cp.DmaCameras(mod, 3, 0, shm_dir=tmp_path, attach_timeout=0.2)
    t0 = time.monotonic()
    assert cams.reattach(3) is False
    assert time.monotonic() - t0 < 1.0
    assert cams.reattach(3) is False            # the first attempt still hangs: no second helper
    assert made["queues"] == 2
    release.set()
    for _ in range(100):
        if made["queues"] == 3:
            break
        time.sleep(0.01)
    time.sleep(0.05)
    old = cams.q[3]
    cams.read(3, 0.0)                           # the reader adopts the late attach
    assert cams.q[3] is not old


# ------------------------------------------------------------------ 13 Nov: pool sequence offset (phantom10)

def _offset_dma(entries, offset, slots=8, torn_at=(), stale_ns=None):
    """a pool whose slot sequence = meta frame_sequence + offset (phantom10 13 Nov after a producer restart without the
    camera: meta 101665 vs slot 49629). read(check_sequence=True) refuses every frame, as the real binding did"""
    import time as _t

    class Consumer:
        def __init__(self):
            self.i = 0

        def next(self, timeout):
            if self.i >= len(entries):
                return None
            m = entries[self.i]
            self.i += 1
            return m

    class Q:
        def __init__(self, name, create=False, wait=True):
            pass

        def make_consumer(self, max_lag=8):
            return Consumer2()

    POOL = []

    class Pool:
        def __init__(self, name, create=False):
            self.reads, self.bump, self.cur = 0, {}, {}

        def slot_sequence(self, slot):
            m = self.cur[slot]
            return m.frame_sequence + offset + (slots if getattr(m, "_stale", False) else 0) + self.bump.get(slot, 0)

        def read(self, m, check_sequence=True):
            self.reads += 1
            if check_sequence:
                return None
            if m.frame_sequence in torn_at:
                self.bump[m.pool_slot_index] = slots          # the producer rewrote the slot during the copy
            return m._buf

    class Consumer2(Consumer):
        def next(self, timeout):
            m = Consumer.next(self, timeout)
            if m is not None:
                POOL[0].cur[m.pool_slot_index] = m
            return m

    def _mk(seq, slot, over=False):
        e = _Meta(3, seq, _t.time_ns() if stale_ns is None else _t.time_ns() - stale_ns)
        e.publish_timestamp_ns = e.capture_timestamp_ns
        e.pool_slot_index, e._over = slot, over
        e._buf = memoryview(np.zeros(36, np.uint8).tobytes())
        return e
    def make_pool(name, create=False):
        p = Pool(name, create)
        POOL[:] = [p]
        return p
    mod = types.SimpleNamespace(VideoMetaQueue=Q, VideoFramePool=make_pool,
                                video_meta_queue_name=lambda c: f"/video_meta_cam{c}",
                                video_frame_pool_name=lambda c: f"/video_frames_cam{c}")
    return mod, _mk


def _entries(n, start=101665):
    return [(start + i, i % 8) for i in range(n)]


def test_a_constant_pool_sequence_offset_is_learned_and_frames_read():
    import time as _t
    mod, mk = _offset_dma([], 49629 - 101665)
    ents = [mk(s, sl) for s, sl in _entries(6)]
    mod2, _ = _offset_dma(ents, 49629 - 101665)
    cams = cp.DmaCameras(mod2, 3, 3)
    got = [cams.read(3, 0.01) for _ in range(6)]
    assert got[:cp.SEQ_LEARN_N - 1] == [cp.NONE_READ] * (cp.SEQ_LEARN_N - 1)       # learning: not trusted yet
    assert all(isinstance(f, cp.Frame) for f in got[cp.SEQ_LEARN_N - 1:])
    assert cams.seq_offset[3] == 49629 - 101665


def test_a_recycled_slot_is_still_refused_after_learning():
    ents = []
    mod, mk = _offset_dma(ents, -52036)
    ents += [mk(s, sl) for s, sl in _entries(4)]
    stale = mk(101669, 4)
    stale._stale = True                              # the slot holds a newer frame (one pool cycle on)
    ents.append(stale)
    cams = cp.DmaCameras(mod, 3, 3)
    got = [cams.read(3, 0.01) for _ in range(5)]
    assert isinstance(got[3], cp.Frame) and got[4] is cp.NONE_READ


def test_a_lagging_reader_never_learns_an_offset_from_old_frames():
    ents = []
    mod, mk = _offset_dma(ents, -52036, stale_ns=int(0.5e9))     # every entry 0.5 s old
    ents += [mk(s, sl) for s, sl in _entries(6)]
    cams = cp.DmaCameras(mod, 3, 3)
    assert all(cams.read(3, 0.01) is cp.NONE_READ for _ in range(6))
    assert cams.seq_offset.get(3) is None


def test_a_frame_rewritten_during_the_copy_is_refused():
    ents = []
    mod, mk = _offset_dma(ents, -52036, torn_at=(101668,))
    ents += [mk(s, sl) for s, sl in _entries(4)]
    cams = cp.DmaCameras(mod, 3, 3)
    got = [cams.read(3, 0.01) for _ in range(4)]
    assert isinstance(got[2], cp.Frame) and got[3] is cp.NONE_READ
