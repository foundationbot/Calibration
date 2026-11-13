"""flow: the head search for a board it cannot see (12 Nov operator request) and the quiet guide during a camera
recovery; stubs only."""
from types import SimpleNamespace

import pytest

from calibtool import flow


def test_board_move_text_directions():
    assert flow.board_move_text(0.0, 12.0, 1.0) == "Found it. Move the board about 20 centimetres to your left."
    assert flow.board_move_text(0.0, -12.0, 1.0).endswith("to your right.")
    t = flow.board_move_text(10.0, 0.0, 1.0)
    assert t == "Found it. Move the board higher by about 20 centimetres."
    assert "lower by" in flow.board_move_text(-10.0, 12.0, 0.6) and "left" in flow.board_move_text(-10.0, 12.0, 0.6)
    assert flow.board_move_text(0.0, 0.0, 1.0) == "Found it."


class FakeNeck:
    def __init__(self):
        self.fast, self.ramp_s, self.settle_s, self.went = True, 1.0, 0.8, []

    def goto(self, p, y):
        self.went.append((p, y, self.fast, self.ramp_s))
        return SimpleNamespace(command=(p, y))


class Clock:
    t = 0.0

    def mono(self):
        return self.t

    def sleep(self, s):
        self.t += s


def stub(seen_at=None):
    c, n, said, ev, homes = Clock(), FakeNeck(), [], [], []

    def board_obs():
        if seen_at is not None and n.went and n.went[-1][:2] == seen_at:
            return True, SimpleNamespace(z=1.0)
        return True, None
    s = SimpleNamespace(mono=c.mono, sleep=c.sleep, _arm=lambda: n, _home=lambda neck: homes.append(1),
                        _between_poses=lambda: None, _event=lambda k, **kw: ev.append(k),
                        _say=lambda kind, text: said.append((kind, text)), robot=SimpleNamespace(board_obs=board_obs))
    return s, n, said, ev, homes


def test_search_finds_the_board_says_where_and_goes_home_slowly():
    s, n, said, ev, homes = stub(seen_at=(0.0, -12.0))
    flow.FullFlow._search_board(s, "not seen")
    assert said[0] == ("search", flow.SEARCH_TEXT)                 # said before anything moved
    assert [w[:2] for w in n.went] == [(0.0, 12.0), (0.0, -12.0)]   # stops at the pose that found it
    assert all(not fast and ramp == flow.SEARCH_RAMP_S for _, _, fast, ramp in n.went)
    assert (n.fast, n.ramp_s, n.settle_s) == (True, 1.0, 0.8)       # the neck's own timing restored
    assert homes == [1] and said[-1] == ("correction", "Found it. Move the board about 20 centimetres to your right.")
    assert ev == ["board_search", "board_found"]


def test_search_that_finds_nothing_says_so_after_every_pose_and_goes_home():
    s, n, said, ev, homes = stub()
    flow.FullFlow._search_board(s, "key U")
    assert len(n.went) == len(flow.SEARCH_POSES) and homes == [1]
    assert said[-1] == ("lost", flow.SEARCH_NONE_TEXT) and ev[-1] == "board_not_found"
    assert all(abs(p) <= 10 and abs(y) <= 12 for p, y in flow.SEARCH_POSES)


def test_camera_out_reads_recovery_and_stall():
    m = SimpleNamespace(state={})
    s = SimpleNamespace(m=m)
    assert not flow.FullFlow._camera_out(s)
    m.state = {"camera_recovery": {"active": True}}
    assert flow.FullFlow._camera_out(s)
    m.state = {"camera_stalled_s": 3.0}
    assert flow.FullFlow._camera_out(s)
