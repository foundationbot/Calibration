"""flow.FullFlow.keepalive: the motionless re-send only while waiting with the motors on (12 Nov: the master's 30 s
stale-command fallback moved phantom10's legs by itself)."""
from types import SimpleNamespace

import pytest

from calibtool import flow


class Neck:
    def __init__(self, answer="sent", armed=True):
        self.armed, self.answer, self.calls = armed, answer, 0

    def keepalive(self):
        self.calls += 1
        return self.answer


def stub(phase, neck=None, **fl):
    ev = []
    s = SimpleNamespace(neck=neck, fl=dict(fl), m=SimpleNamespace(phase=phase),
                        _event=lambda kind, **kw: ev.append((kind, kw)))
    return s, ev


@pytest.mark.parametrize("phase", ["PLACE", "FIT", "DEPLOY_SIZE", "DONE", "IMU_PLACE"])
def test_sent_while_waiting(phase):
    n = Neck()
    s, _ = stub(phase, n)
    flow.FullFlow.keepalive(s)
    assert n.calls == 1


@pytest.mark.parametrize("phase", ["HOLD", "ABORTED", "STOPPED_LOUD", "GRID", "IMU_SWEEP", "MOTORS", "PREFLIGHT"])
def test_never_in_hold_abort_or_motion(phase):
    n = Neck()
    s, _ = stub(phase, n)
    flow.FullFlow.keepalive(s)
    assert n.calls == 0


def test_never_after_the_motors_were_disabled_or_before_arming():
    n = Neck()
    s, _ = stub("DONE", n, motors_disabled=True)
    flow.FullFlow.keepalive(s)
    s2, _ = stub("PLACE", Neck(armed=False))
    flow.FullFlow.keepalive(s2)
    s3, _ = stub("PLACE", None)
    flow.FullFlow.keepalive(s3)
    assert n.calls == 0 and s2.neck.calls == 0


def test_a_foreign_writer_is_logged_once():
    n = Neck(answer="foreign")
    s, ev = stub("PLACE", n)
    flow.FullFlow.keepalive(s)
    flow.FullFlow.keepalive(s)
    assert [k for k, _ in ev] == ["keepalive_skipped"]
