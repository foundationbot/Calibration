"""The head-camera reader (design plan sections 3, 8, 10.1): one thread that is never blocked.

A camera source yields, per poll of one camera, a Frame (capture stamp, sequence, the Y plane), NONE_READ (a meta
arrived but the frame pool returned None: the pool's sequence desynced, e.g. after a producer restart) or None (nothing
new). Capture.step() polls both cameras once, updates the per-camera FrameCounters, matches left/right frames by
capture stamp (PairMatcher, tolerance 0.5 ms: the section-8 sync limit) and hands each pair to:
  latest()      the newest pair (one slot: a reader that is late sees the newest, never a backlog); latest_age()
                how long ago it came (a frozen camera keeps the slot, its age grows)
  subscribe()   bounded queues filled with put_nowait: a full queue drops the pair (counted in sub_dropped), so a slow
                detector never stalls the reader
  grab(n)       the next n pairs after the call (a hold's capture)
A camera whose reads come back None REATTACH_AFTER times in a row gets its own consumer re-attached (source.reattach,
counted); the health watch decides about a producer restart if that does not help.

DmaCameras is the source over dma_ipc on the robot (step 08's Cams, per-read instead of per-batch): NV12 frames, Y
plane = the first w*h bytes. dma_ipc is passed in (imported by the caller on the robot only)."""
import collections
import queue
import statistics
import threading
import time
from dataclasses import dataclass

import numpy as np

TOL_NS = 500_000                # left/right capture stamps further apart than this do not pair (section 8: 0.5 ms)
KEEP = 6                        # unmatched frames kept per camera
REATTACH_AFTER = 25             # None reads in a row (1 s at 25 fps) before the camera's consumer is re-attached
FPS_WINDOW = 50
SYNC_WINDOW = 100


class _NoneRead:
    def __repr__(self):
        return "NONE_READ"


NONE_READ = _NoneRead()


@dataclass
class Frame:
    cam: int
    seq: int
    t_ns: int                   # capture stamp
    t_pub_ns: int               # publish stamp
    y: np.ndarray               # (h, w) uint8


@dataclass
class Pair:
    left: Frame
    right: Frame

    @property
    def t0(self) -> int:
        return self.left.t_ns

    @property
    def t1(self) -> int:
        return self.right.t_ns

    @property
    def dt_ns(self) -> int:
        """right - left capture stamp"""
        return self.right.t_ns - self.left.t_ns


class PairMatcher:
    """left/right frames matched by capture stamp: a new frame pairs with the closest pending frame of the other camera
    within tol_ns; unmatched frames older than the newest `keep` of their camera are dropped (counted)"""

    def __init__(self, left, right, tol_ns=TOL_NS, keep=KEEP):
        self.left, self.right, self.tol, self.keep = left, right, int(tol_ns), int(keep)
        self._pend = {left: collections.OrderedDict(), right: collections.OrderedDict()}
        self.dropped = {left: 0, right: 0}

    def pending(self, cam) -> list:
        return list(self._pend[cam].values())

    def push(self, f: Frame) -> list:
        other = self.right if f.cam == self.left else self.left
        cand = self._pend[other]
        best = min(cand, key=lambda t: abs(t - f.t_ns), default=None)
        if best is not None and abs(best - f.t_ns) <= self.tol:
            o = cand.pop(best)
            for t in [t for t in cand if t < best]:     # older frames of the other camera can no longer pair
                cand.pop(t)
                self.dropped[other] += 1
            return [Pair(f, o) if f.cam == self.left else Pair(o, f)]
        mine = self._pend[f.cam]
        mine[f.t_ns] = f
        while len(mine) > self.keep:
            mine.popitem(last=False)
            self.dropped[f.cam] += 1
        return []


class FrameCounters:
    """one camera's frame health: frames read, None reads (all / in a row), frames missing by sequence, repeated
    frames (same sequence or stamp as the previous: a freeze), re-attaches, fps from capture stamps"""

    def __init__(self):
        self.frames = self.none_reads = self.none_run = self.seq_gaps = self.repeats = self.reattaches = 0
        self.last_seq = self.last_t_ns = None
        self.last_frame_wall = None
        self._ts = collections.deque(maxlen=FPS_WINDOW)

    def frame(self, f: Frame, now):
        self.frames += 1
        self.none_run = 0
        if self.last_seq is not None and (f.seq == self.last_seq or f.t_ns == self.last_t_ns):
            self.repeats += 1
        elif self.last_seq is not None and f.seq > self.last_seq + 1:
            self.seq_gaps += f.seq - self.last_seq - 1
        if not self._ts or f.t_ns != self._ts[-1]:
            self._ts.append(f.t_ns)
        self.last_seq, self.last_t_ns, self.last_frame_wall = f.seq, f.t_ns, now

    def none_read(self, now):
        self.none_reads += 1
        self.none_run += 1

    def fps(self) -> float:
        if len(self._ts) < 2:
            return 0.0
        span = (self._ts[-1] - self._ts[0]) / 1e9
        return (len(self._ts) - 1) / span if span > 0 else 0.0

    def as_dict(self, now) -> dict:
        return {"frames": self.frames, "none_reads": self.none_reads, "none_run": self.none_run,
                "seq_gaps": self.seq_gaps, "repeats": self.repeats, "reattaches": self.reattaches,
                "last_seq": self.last_seq, "last_t_ns": self.last_t_ns, "fps": round(self.fps(), 2),
                "since_frame_s": None if self.last_frame_wall is None else round(now - self.last_frame_wall, 3)}


class PairStats:
    def __init__(self):
        self.pairs = 0
        self._dt = collections.deque(maxlen=SYNC_WINDOW)

    def add(self, p):
        self.pairs += 1
        self._dt.append(abs(int(p.dt_ns)))

    def median_abs_dt_ns(self):
        return statistics.median(self._dt) if self._dt else None

    def max_abs_dt_ns(self):
        return max(self._dt) if self._dt else None


class Capture:
    """the reader loop over a camera source (see the module text). clock: wall-clock seconds for the counters"""

    def __init__(self, source, left, right, clock=time.monotonic, tol_ns=TOL_NS, read_timeout=0.02):
        self.source, self.left, self.right, self.clock = source, int(left), int(right), clock
        self.read_timeout = read_timeout
        self.matcher = PairMatcher(self.left, self.right, tol_ns)
        self.counters = {self.left: FrameCounters(), self.right: FrameCounters()}
        self.pair_stats = PairStats()
        self.sub_dropped = 0
        self._latest = None
        self._latest_at = None          # clock() when the newest pair was delivered
        self._subs = []
        self._grabs = []                # [(n, list, Event)]
        self._lock = threading.Lock()
        self._thread = None
        self._stop = threading.Event()
        self.error = None

    # -------------------------------------------------------------- one poll

    def step(self):
        for cam in (self.left, self.right):
            r = self.source.read(cam, self.read_timeout)
            now = self.clock()
            c = self.counters[cam]
            if r is None:
                continue
            if r is NONE_READ:
                c.none_read(now)
                if c.none_run == REATTACH_AFTER:
                    self.source.reattach(cam)
                    c.reattaches += 1
                continue
            c.frame(r, now)
            for p in self.matcher.push(r):
                self._deliver(p)

    def _deliver(self, p: Pair):
        self.pair_stats.add(p)
        with self._lock:
            self._latest = p
            self._latest_at = self.clock()
            for q in self._subs:
                try:
                    q.put_nowait(p)
                except queue.Full:
                    self.sub_dropped += 1
            done = []
            for g in self._grabs:
                n, got, ev = g
                got.append(p)
                if len(got) >= n:
                    ev.set()
                    done.append(g)
            for g in done:
                self._grabs.remove(g)

    # -------------------------------------------------------------- consumers

    def latest(self):
        with self._lock:
            return self._latest

    def latest_age(self):
        """seconds since the newest pair was delivered (None: no pair yet); a frozen or dropped camera keeps latest()
        but its age grows"""
        with self._lock:
            return None if self._latest_at is None else self.clock() - self._latest_at

    def subscribe(self, maxsize=2) -> queue.Queue:
        q = queue.Queue(maxsize=maxsize)
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q):
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def grab(self, n, timeout=6.0) -> list:
        """the next n pairs after this call (fewer when timeout runs out); needs the thread running"""
        got, ev = [], threading.Event()
        g = (int(n), got, ev)
        with self._lock:
            self._grabs.append(g)
        ev.wait(timeout)
        with self._lock:
            if g in self._grabs:
                self._grabs.remove(g)
            return list(got[:n])

    # -------------------------------------------------------------- the thread

    def start(self, poll_s=0.0):
        if self._thread is not None:
            return
        self._stop.clear()

        def run():
            try:
                while not self._stop.is_set():
                    self.step()
                    if poll_s:
                        time.sleep(poll_s)
            except Exception as e:      # noqa: BLE001  (the daemon reports it; the reader must not die silently)
                self.error = f"{type(e).__name__}: {e}"
        self._thread = threading.Thread(target=run, name="capture", daemon=True)
        self._thread.start()

    def stop(self, timeout=2.0):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
        self._thread = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def snapshot(self) -> dict:
        now = self.clock()
        med, mx = self.pair_stats.median_abs_dt_ns(), self.pair_stats.max_abs_dt_ns()
        return {"cams": {str(c): k.as_dict(now) for c, k in self.counters.items()},
                "pairs": self.pair_stats.pairs,
                "unpaired_dropped": {str(c): n for c, n in self.matcher.dropped.items()},
                "sub_dropped": self.sub_dropped,
                "sync_median_ms": None if med is None else round(med / 1e6, 4),
                "sync_max_ms": None if mx is None else round(mx / 1e6, 4),
                "error": self.error, "running": self.running}


class DmaCameras:
    """the camera source over dma_ipc (module d): one read-only consumer + frame pool per camera"""

    def __init__(self, d, left, right, max_lag=8):
        self.d, self.max_lag = d, max_lag
        self.cams = (int(left), int(right))
        self.q, self.p = {}, {}
        for c in self.cams:
            self._attach(c)

    def _attach(self, c):
        d = self.d
        self.q[c] = d.VideoMetaQueue(d.video_meta_queue_name(c), create=False, wait=True).make_consumer(
            max_lag=self.max_lag)
        self.p[c] = d.VideoFramePool(d.video_frame_pool_name(c), create=False)

    def reattach(self, c):
        self._attach(int(c))

    def read(self, c, timeout):
        m = self.q[c].next(timeout)
        if m is None:
            return None
        buf = self.p[c].read(m)
        if buf is None:
            return NONE_READ
        w, h = int(m.width), int(m.height)
        y = np.frombuffer(bytes(buf), np.uint8)[: w * h].reshape(h, w).copy()
        return Frame(int(getattr(m, "camera_id", c)), int(m.frame_sequence), int(m.capture_timestamp_ns),
                     int(getattr(m, "publish_timestamp_ns", 0)), y)
