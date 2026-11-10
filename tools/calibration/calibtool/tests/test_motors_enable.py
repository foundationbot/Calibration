"""calibtool.motors_enable: the motor enable step after the operator's Y, and disable at the end (design plan 7, D5).

Fakes only: a probe returning a state dict and a fake runner standing in for dma_cmd_client / journalctl. The key
guarantee: no command runs unless every check passed and Y was given."""
from pathlib import Path

import pytest

pytest.importorskip("numpy")

from calibtool import motors, motors_enable as me  # noqa: E402


def master(active=True, motor_state=0, estop=False, error=None):
    return motors.MasterStatus(active, "2.1.1", 6065.3, motor_state, estop, error)


def good_state(**kw):
    s = {"autonomy": None, "pods": ["dma-video/producer-abc", "kube-system/coredns"], "desired_still": True,
         "master": master(), "bus_v": 53.1, "head_power": True, "slaves_op": 50, "fault_latches": []}
    s.update(kw)
    return s


BASE = {"slaves_op": 50}


class Run:
    """dma_cmd_client: enable/disable switch motor_state; ping reports it; journalctl prints the enabled line"""

    def __init__(self, enabled_line="dma-ethercat: 50 of 50 motors enabled", ping_after_enable=1):
        self.calls, self.motor_state = [], 0
        self.enabled_line, self.ping_after_enable = enabled_line, ping_after_enable

    def __call__(self, argv, timeout=5):
        self.calls.append(list(argv))
        if argv[:2] == ["dma_cmd_client", "enable-motors"]:
            self.motor_state = self.ping_after_enable
            return 0, "ok"
        if argv[:2] == ["dma_cmd_client", "disable-motors"]:
            self.motor_state = 0
            return 0, "ok"
        if argv[:2] == ["dma_cmd_client", "ping"]:
            return 0, f"PONG  uptime=6070.0s  motor_state={self.motor_state}"
        if argv[0] == "journalctl":
            return 0, ("boot\n" + self.enabled_line + "\n") if self.motor_state else "boot\n"
        return 1, "unknown"


def make(state=None, run=None, **kw):
    r = run or Run()
    st = state if state is not None else good_state()
    t = [100.0]
    en = me.Enabler(r, probe=lambda: st, baseline=BASE, clock=lambda: t[0],
                    sleep=lambda s: t.__setitem__(0, t[0] + s), **kw)
    return en, r


def mutating(calls):
    return [c for c in calls if c[:2] in (["dma_cmd_client", "enable-motors"], ["dma_cmd_client", "disable-motors"])]


# ---------------------------------------------------------------- B1: checks


def test_all_good_every_check_ok():
    cs = me.pre_enable_checks(good_state(), BASE, gantry=True)
    assert cs and all(c.status in ("ok", "info") for c in cs)
    assert me.passed(cs)


BAD = {
    "gantry": ({}, False),
    "autonomy": ({"autonomy": "autonomy: ARMED (goal active)"}, True),
    "policy_pods": ({"pods": ["positronic/ghost-wbc-teleop-1"]}, True),
    "desired_still": ({"desired_still": False}, True),
    "master": ({"master": master(active=False, error="dma-ethercat is inactive")}, True),
    "estop": ({"master": master(estop=True)}, True),
    "estop_unknown": ({"master": master(estop=None)}, True),
    "motors_off": ({"master": master(motor_state=1)}, True),
    "bus_v": ({"bus_v": 3.1}, True),
    "head_power": ({"head_power": False}, True),
    "slaves_op": ({"slaves_op": 46}, True),
    "fault_latches": ({"fault_latches": ["NeckPitch 0x7300"]}, True),
}


@pytest.mark.parametrize("name", sorted(BAD))
def test_each_bad_input_fails_its_check(name):
    over, gantry = BAD[name]
    cs = me.pre_enable_checks(good_state(**over), BASE, gantry=gantry)
    failed = [c for c in cs if c.status == "fail"]
    assert len(failed) == 1, [c.as_dict() for c in failed]
    assert failed[0].name == name.replace("estop_unknown", "estop")
    assert failed[0].detail and not me.passed(cs)


def test_unavailable_optional_probes_are_info_not_blocking():
    st = good_state(bus_v=None, head_power=None, slaves_op=None, fault_latches=None)
    cs = me.pre_enable_checks(st, BASE, gantry=True)
    by = {c.name: c for c in cs}
    for k in ("bus_v", "head_power", "slaves_op", "fault_latches"):
        assert by[k].status == "info" and "not available" in by[k].detail
    assert me.passed(cs)


def test_slaves_without_baseline_is_info():
    cs = me.pre_enable_checks(good_state(), {}, gantry=True)
    assert {c.name: c for c in cs}["slaves_op"].status == "info"


def test_probe_error_is_a_failed_check():
    def boom():
        raise RuntimeError("k0s refused")
    en = me.Enabler(Run(), probe=boom, baseline=BASE)
    cs = en.checks(gantry=True)
    assert not me.passed(cs) and "k0s refused" in cs[0].detail


# ---------------------------------------------------------------- B2: enable / disable


@pytest.mark.parametrize("key", ["", "y ", "N", "G", None])
def test_without_Y_nothing_runs(key):
    en, r = make()
    res = en.enable(key, gantry=True)
    assert not res.ok and res.refused and not mutating(r.calls)


@pytest.mark.parametrize("name", sorted(BAD))
def test_any_failed_check_refuses_even_with_Y(name):
    over, gantry = BAD[name]
    en, r = make(state=good_state(**over))
    res = en.enable("Y", gantry=gantry)
    assert not res.ok and res.refused and not mutating(r.calls)
    assert any(c["status"] == "fail" for c in res.as_dict()["checks"])


def test_enable_runs_once_and_verifies():
    en, r = make()
    res = en.enable("Y", gantry=True)
    assert res.ok and (res.n, res.of) == (50, 50)
    assert mutating(r.calls) == [["dma_cmd_client", "enable-motors"]]
    assert en.commands == [["dma_cmd_client", "enable-motors"]]


def test_enable_partial_is_not_ok_and_not_auto_disabled():
    en, r = make(run=Run(enabled_line="dma-ethercat: 48 of 50 motors enabled"))
    res = en.enable("Y", gantry=True)
    assert not res.ok and not res.refused and (res.n, res.of) == (48, 50)
    assert "48 of 50" in res.detail
    assert mutating(r.calls) == [["dma_cmd_client", "enable-motors"]]


def test_enable_without_journal_line_times_out_not_ok():
    en, r = make(run=Run(enabled_line="something else"))
    res = en.enable("Y", gantry=True)
    assert not res.ok and "journal" in res.detail


def test_motor_state_stays_zero_not_ok():
    en, r = make(run=Run(ping_after_enable=0))
    res = en.enable("Y", gantry=True)
    assert not res.ok and "motor_state" in res.detail


def test_parse_enabled():
    assert me.parse_enabled("x\nfoo: 50 of 50 motors enabled\n") == (50, 50)
    assert me.parse_enabled("12 of 50 motors enabled\n50 of 50 motors enabled") == (50, 50)
    assert me.parse_enabled("nothing") is None


def test_disable_needs_confirmation():
    en, r = make()
    res = en.disable(confirmed=False)
    assert not res.ok and res.refused and me.DISABLE_WARNING in res.detail and not mutating(r.calls)
    assert "head and arms drop" in me.DISABLE_WARNING


def test_disable_runs_and_verifies():
    en, r = make()
    en.enable("Y", gantry=True)
    res = en.disable(confirmed=True)
    assert res.ok and mutating(r.calls)[-1] == ["dma_cmd_client", "disable-motors"]


def test_dry_run_records_and_runs_nothing():
    en, r = make(dry_run=True)
    res = en.enable("Y", gantry=True)
    assert res.dry_run and en.commands == [["dma_cmd_client", "enable-motors"]]
    assert not mutating(r.calls)
    en.disable(confirmed=True)
    assert en.commands[-1] == ["dma_cmd_client", "disable-motors"] and not mutating(r.calls)


def test_only_motors_enable_names_enable_motors():
    pkg = Path(me.__file__).parent
    hits = []
    for f in pkg.glob("*.py"):
        code = "\n".join(ln.split("#")[0] for ln in f.read_text().splitlines())
        if "enable-motors" in code or "disable-motors" in code:
            hits.append(f.name)
    assert hits == ["motors_enable.py"]


def test_motors_already_on_are_taken_over_on_y_only_when_allowed_and_all_else_passes():
    """10 Nov live session: the operator had enabled the motors (to hold the head level) and the tool refused with
    'someone else enabled them'. With adopt_on the operator's Y takes them over: no enable command is sent, the
    state is verified (ping motor_state + 'N of N motors enabled' in the journal); every other check still blocks."""
    r = Run(); r.motor_state = 1
    en, _ = make(state=good_state(master=master(motor_state=2)), run=r, adopt_on=True)
    res = en.enable("Y", gantry=True)
    assert res.ok and not mutating(r.calls) and "taken over" in res.detail and (res.n, res.of) == (50, 50)
    r2 = Run(); r2.motor_state = 1
    en2, _ = make(state=good_state(master=master(motor_state=2)), run=r2)             # default: refuse as before
    assert en2.enable("Y", gantry=True).refused
    r3 = Run(); r3.motor_state = 1
    en3, _ = make(state=good_state(master=master(motor_state=2), desired_still=False), run=r3, adopt_on=True)
    assert en3.enable("Y", gantry=True).refused and not mutating(r3.calls)
    r4 = Run(enabled_line="dma-ethercat: 48 of 50 motors enabled"); r4.motor_state = 1
    en4, _ = make(state=good_state(master=master(motor_state=2)), run=r4, adopt_on=True)
    res4 = en4.enable("Y", gantry=True)
    assert not res4.ok and not mutating(r4.calls)
