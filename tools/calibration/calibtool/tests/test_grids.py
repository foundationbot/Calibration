"""calibtool.grids: one placement's neck grid -> captured pairs -> detection -> CaptureHolds -> replay.Collection, with
the early stop (darkness vs absence, a bumped board), the end-of-placement snapshot and the session's collector.

Fakes only: a fake neck grid (calls on_pose per pose with a PoseDone-like object), fake capture pairs whose images
carry a key the fake detector maps to scripted corners, a fake Collection; one test replays a stored session's holds
through GridRun (needs CALIBTOOL_DATA)."""
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")

from calibtool import grids, neck as nk  # noqa: E402


def det(n_tags=10, du=0.0, dv=0.0, u0=500.0, v0=300.0):
    return {(t, k): (u0 + 40 * (t % 5) + 10 * k + du, v0 + 40 * (t // 5) + 10 * (k // 2) + dv)
            for t in range(n_tags) for k in range(4)}


class Img:
    """a camera frame: a key for the fake detector and a grey image for the brightness"""

    def __init__(self, key, level=120):
        self.key = key
        self.y = np.full((800, 1280), level, np.uint8)


class FakePair:
    def __init__(self, t, key, level=120):
        self.left = SimpleNamespace(y=Img(("L",) + key, level).y, key=("L",) + key, t_ns=t)
        self.right = SimpleNamespace(y=Img(("R",) + key, level).y, key=("R",) + key, t_ns=t + 10_000)
        self.t0, self.t1, self.dt_ns = t, t + 10_000, 10_000


class World:
    """per pose index: (found, level, du): what the cameras see; predictions per pose"""

    def __init__(self, n, found=None, level=None, shift=None, frames=3):
        self.found = found or [True] * n
        self.level = level or [120] * n
        self.shift = shift or [0.0] * n
        self.pose = -1
        self.t = 10**12
        self.dets = {}
        self.grabs = []

    def grab(self, n, timeout=None):
        self.grabs.append(n)
        out = []
        i = self.pose
        for j in range(n):
            self.t += 66_000_000
            key = (i, j)
            d = det(10 if self.found[i] else 2, du=self.shift[i])
            self.dets[("L",) + key] = d
            self.dets[("R",) + key] = det(10 if self.found[i] else 2, du=self.shift[i] - 30)
            out.append(FakePair(self.t, key, self.level[i]))
        self._last = out
        return out

    def detect(self, yl, yr):
        p = next(p for p in self._last if p.left.y is yl)
        return self.dets[p.left.key], self.dets[p.right.key]


class FakeNeckGrid:
    """neck.run_grid's contract: should_stop before each pose, on_pose after it"""

    def __init__(self, world, sag=0.5):
        self.world, self.sag, self.commands = world, sag, []

    def __call__(self, poses, on_pose, should_stop):
        out = nk.GridOutcome()
        for i, (p, y) in enumerate(poses):
            why = should_stop()
            if why:
                out.stopped = why
                return out
            self.commands.append((p, y))
            self.world.pose = len(self.commands) - 1
            d = nk.PoseDone((p, y), (p, y), False, (p + self.sag, y), True, {}, 1.0, 0.0)
            out.done.append(d)
            on_pose(d)
        return out


class FakeCollection:
    def __init__(self, stop_after=None):
        self.added, self.ended, self.stop, self.stop_after = [], [], None, stop_after

    def add(self, h):
        self.added.append(h)
        if self.stop_after is not None and len(self.added) >= self.stop_after:
            self.stop = SimpleNamespace(hold=h.hold)
        return "FIT" if len(self.added) % 2 else "TEST"

    def end_placement(self, name=None):
        self.ended.append(name)
        return SimpleNamespace(sentences=[f"gap after {name}"], as_dict=lambda: {"placement": name})


POSES = [(-8.0, -11.0), (-8.0, 0.0), (-8.0, 11.0), (0.0, 11.0), (0.0, 0.0), (0.0, -11.0), (8.0, -11.0), (8.0, 0.0),
         (8.0, 11.0), (16.0, 0.0)]


def make(world=None, col=None, n=len(POSES), predict=None, **kw):
    w = world or World(n)
    c = col or FakeCollection()
    g = FakeNeckGrid(w)
    run = grids.GridRun(g, w.grab, w.detect, c, "d110", POSES[:n], predict=predict, hold_ids=iter(range(100, 200)),
                        **kw)
    return run, w, c, g


# ---------------------------------------------------------------- C1: one pose -> one CaptureHold


def test_each_pose_gives_one_hold_of_the_middle_two_pairs():
    run, w, c, g = make(n=3)
    res = run.run()
    assert w.grabs == [3, 3, 3]
    assert [h.hold for h in c.added] == [100, 101, 102]
    h = c.added[1]
    assert h.placement == "d110" and len(h.data["img_t0"]) == 2
    assert np.allclose(h.data["img_pitch"], np.radians(-8.0 + 0.5)) and np.allclose(h.data["img_yaw"], 0.0)
    assert h.data["img_ay"][0] == 1 and h.data["img_ap"][0] == 0         # yaw -11 -> 0: +, pitch unchanged
    assert h.pose == grids.grid_index(-8.0, 0.0)
    assert set(np.unique(h.data["det_cam"])) == {0, 1}
    assert res.holds == [100, 101, 102] and [r.label for r in res.records] == ["FIT", "TEST", "FIT"]
    assert res.records[0].found and res.records[0].tags_l == 10


def test_a_pose_without_the_board_gives_no_hold():
    w = World(3, found=[True, False, True])
    run, w, c, g = make(world=w, n=3)
    res = run.run()
    assert [h.hold for h in c.added] == [100, 101]
    assert not res.records[1].found and res.records[1].hold is None and res.records[1].tags_l == 2


def test_detection_futures_are_resolved():
    from concurrent.futures import Future
    w = World(2)

    def fut_detect(yl, yr):
        f = Future()
        f.set_result(w.detect(yl, yr))
        return f
    c = FakeCollection()
    res = grids.GridRun(FakeNeckGrid(w), w.grab, fut_detect, c, "x", POSES[:2], hold_ids=iter(range(5))).run()
    assert len(c.added) == 2 and res.stopped is None


# ---------------------------------------------------------------- C2: early stop, darkness vs absence, bump


def visible_all(p, y):
    return grids.Visible(True, (400, 250, 900, 500), (600.0, 360.0))


def test_poses_predicted_out_of_view_are_not_judged():
    w = World(10, found=[False] * 10)
    run, w, c, g = make(world=w, predict=lambda p, y: grids.Visible(False, None, None))
    res = run.run()
    assert res.early_stop is None and len(g.commands) == 10
    assert all(r.predicted is False for r in res.records)


def test_absent_board_stops_early_and_goes_to_redo():
    w = World(10, found=[False] * 10, level=[150] * 10)
    run, w, c, g = make(world=w, predict=visible_all)
    res = run.run()
    assert res.early_stop is not None and res.early_stop.reason == "absent"
    assert len(g.commands) < 10 and res.stopped == "early_stop"
    assert res.redo is not None and res.redo.reason == "absent" and res.redo.poses == POSES
    assert "can't find the board" in res.early_stop.sentence


def test_dark_board_is_told_apart():
    w = World(10, found=[False] * 10, level=[12] * 10)
    run, w, c, g = make(world=w, predict=visible_all)
    res = run.run()
    assert res.early_stop.reason == "dark"
    assert "exposure" in res.early_stop.sentence and "gain" in res.early_stop.sentence


def test_a_few_misses_do_not_stop():
    found = [True, False, True, True, False, True, True, True, True, True]
    w = World(10, found=found)
    run, w, c, g = make(world=w, predict=visible_all)
    res = run.run()
    assert res.early_stop is None and len(g.commands) == 10


def test_bumped_board_pauses_the_grid():
    shift = [0, 0, 0, 0, 150, 150, 150, 150, 150, 150]
    w = World(10, shift=shift)
    centre = tuple(np.mean(list(det().values()), axis=0))          # where an unbumped board is seen

    def pred(p, y):
        return grids.Visible(True, (400, 250, 900, 500), centre)
    run, w, c, g = make(world=w, predict=pred)
    res = run.run()
    assert res.early_stop is not None and res.early_stop.reason == "bumped"
    assert "moved" in res.early_stop.sentence and len(g.commands) < 10


def test_early_stop_rule_unit():
    r = grids.EarlyStopRule(min_judged=4, miss_frac=0.5, dark_level=40)
    rec = lambda found, b: SimpleNamespace(predicted=True, found=found, brightness=b, dark=b < 40)  # noqa: E731
    assert r.update(SimpleNamespace(predicted=None, found=False, brightness=None, dark=None)) is None
    assert r.update(rec(True, 100)) is None
    assert r.update(rec(False, 100)) is None
    assert r.update(rec(False, 100)) is None
    es = r.update(rec(False, 10))                 # 3 of 4 missed, 2 of them bright
    assert es is not None and es.reason == "absent" and es.misses == 3 and es.poses_judged == 4


# ---------------------------------------------------------------- C3: end of placement, collector


def test_end_of_placement_snapshot_and_sentences():
    run, w, c, g = make(n=3)
    res = run.run()
    assert c.ended == ["d110"] and res.sentences == ["gap after d110"] and res.redo is None


@pytest.mark.parametrize("why", ["abort", "hold"])
def test_abort_or_hold_stops_between_poses(why):
    k = {"n": 0}

    def stop():
        k["n"] += 1
        return why if k["n"] >= 3 else None
    run, w, c, g = make(should_stop=stop)
    res = run.run()
    assert res.stopped == why and len(g.commands) == 2 and len(c.added) == 2
    assert c.ended == ["d110"]
    assert (res.redo is not None) == (why == "hold")
    if why == "hold":
        assert res.redo.poses == POSES[2:]


def test_abort_before_any_pose_no_snapshot():
    run, w, c, g = make(should_stop=lambda: "abort")
    res = run.run()
    assert res.stopped == "abort" and c.ended == [] and res.snapshot is None


def test_stop_rule_ends_the_grid():
    run, w, c, g = make(col=FakeCollection(stop_after=4), stop_when_rule_met=True)
    res = run.run()
    assert res.stopped == "stop_rule" and len(g.commands) <= 6 and res.redo is None


def test_events_per_pose():
    ev = []
    run, w, c, g = make(n=2, on_event=lambda kind, **kw: ev.append((kind, kw)))
    run.run()
    kinds = [k for k, _ in ev]
    assert kinds.count("pose") == 2 and kinds[-1] == "grid_end"


class T:
    def __init__(self, name, distance_m=1.1, tilted=False):
        self.name, self.distance_m, self.tilted = name, distance_m, tilted


def test_collector_defaults_extras_unreachable_redo():
    col = FakeCollection()
    log = []
    outcomes = {"d050": "ok", "d075": "absent", "d110": "ok", "x1": "ok", "d075#redo": "ok"}

    def make_grid(t, poses=None):
        key = t.name + ("#redo" if poses is not None else "")
        log.append(key)

        class G:
            def run(self_):
                col.added.append(SimpleNamespace(hold=len(col.added)))
                r = outcomes[key]
                redo = grids.RedoItem(t, "absent", [(0.0, 0.0)]) if r == "absent" else None
                return grids.GridResult(t.name, [], [], "early_stop" if redo else None, None, None, [], redo)
        return G()
    extras = iter([T("x1"), None])
    res = grids.Collector(col, make_grid, await_placement=lambda t: t.name != "d160",
                          plan_next=lambda c: next(extras)).run([T("d050"), T("d075"), T("d110"), T("d160")])
    assert log == ["d050", "d075", "d110", "x1", "d075#redo"]
    assert [t.name for t in res.unreachable] == ["d160"] and res.redo == [] and res.stopped is None


def test_collector_stops_when_rule_met_and_on_hold():
    col = FakeCollection()

    def make_grid(t, poses=None):
        class G:
            def run(self_):
                if t.name == "b":
                    col.stop = SimpleNamespace(hold=1)
                return grids.GridResult(t.name, [], [], None, None, None, [], None)
        return G()
    res = grids.Collector(col, make_grid, await_placement=lambda t: True).run([T("a"), T("b"), T("c")])
    assert res.stop_met and [r.placement for r in res.results] == ["a", "b"]

    def held(t, poses=None):
        class G:
            def run(self_):
                return grids.GridResult(t.name, [], [], "hold", None, None, [],
                                        grids.RedoItem(t, "hold", [(8.0, 0.0)]))
        return G()
    res = grids.Collector(FakeCollection(), held, await_placement=lambda t: True).run([T("a"), T("b")])
    assert res.stopped == "hold" and [p.name for p in res.pending] == ["b"] and res.redo[0].poses == [(8.0, 0.0)]


# ---------------------------------------------------------------- stored-session holds through GridRun


def test_stored_holds_through_gridrun_equal_replay_feed(data_root):
    pytest.importorskip("scipy")
    pytest.importorskip("cv2")
    pytest.importorskip("yaml")
    from calibtool import assign, replay as rp, sessions
    st = sessions.get("p21_1105")
    s = rp.load_session(st)
    holds = [h for h in assign.capture_holds(s) if h.placement == st.take_order[0]][:12]
    assert holds
    ref = rp.Collection.for_session(s, st.rig_path)
    rp.feed(ref, holds)

    class Replayer:
        def __init__(self):
            self.i = -1

        def grid(self, poses, on_pose, should_stop):
            out = nk.GridOutcome()
            for i, h in enumerate(holds):
                self.i = i
                pd = (float(np.degrees(np.nanmean(h.data["img_pitch"]))),
                      float(np.degrees(np.nanmean(h.data["img_yaw"]))))
                d = nk.PoseDone(pd, pd, False, pd, True, {}, 1.0, 0.0)
                out.done.append(d)
                on_pose(d)
            return out

        def grab(self, n, timeout=None):
            d = holds[self.i].data
            out = []
            for j in range(len(d["img_t0"])):
                p = SimpleNamespace(left=SimpleNamespace(y=(self.i, j, 0)), right=SimpleNamespace(y=(self.i, j, 1)),
                                    t0=int(d["img_t0"][j]), t1=int(d["img_t1"][j]),
                                    dt_ns=int(d["img_t1"][j]) - int(d["img_t0"][j]))
                out.append(p)
            return out

        def detect(self, yl, yr):
            i, j, _ = yl
            d = holds[i].data
            out = []
            for cam in (0, 1):
                m = (d["det_img"] == j) & (d["det_cam"] == cam)
                out.append({(int(t), int(k)): (float(u), float(v)) for t, k, u, v in
                            zip(d["det_tag"][m], d["det_k"][m], d["det_u"][m], d["det_v"][m])})
            return tuple(out)
    r = Replayer()
    col = rp.Collection.for_session(s, st.rig_path)
    res = grids.GridRun(r.grid, r.grab, r.detect, col, st.take_order[0], [(0.0, 0.0)] * len(holds),
                        hold_ids=iter([h.hold for h in holds]), per_hold=3, frames=3, still_px=1e9).run()
    assert res.holds == [h.hold for h in holds]
    assert rp.tallies_dict(col.tallies()) == rp.tallies_dict(ref.tallies())
    assert col.labels() == ref.labels()
