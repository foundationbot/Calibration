"""The motor enable step (design plan section 7, decision D5): the tool enables the motors after an on-screen check and
the operator's Y, and disables them at the end with a warning. motors.py stays read-only; this is the only calibtool
module that may send enable-motors / disable-motors (tests/test_motors_enable.py scans the package for it).

Before enable (pre_enable_checks), each a preflight.Check with gate "motion":
  gantry          the operator confirmed the robot hangs on the gantry (a flag the TUI sets)
  autonomy        nav-cli status not armed (None: no nav-cli on this robot, nothing to arm)
  policy_pods     no locomotion/policy pod running (preflight.policy_pods)
  desired_still   /dev/shm/desired still over 1 s (nothing commanding)
  master          dma-ethercat active, ping answers
  estop           e-stop reads false (unknown counts as not clear)
  motors_off      the motors are off now (motors on already: someone else enabled them)
  bus_v           48 V bus charged (>= BUS_MIN_V)                    \
  head_power      head power switch on                                | optional: None = not available on this
  slaves_op       EtherCAT slaves in OP >= the preflight baseline     | master; reported "info", not blocking
  fault_latches   no per-joint fault latch (e.g. phantom11's 0x7300) /
Any failed check refuses the enable; no command runs unless every check passed AND the key is exactly "Y".

After enable: `dma_cmd_client ping` motor_state != 0 and the master's journal saying "N of N motors enabled"
(`dma_cmd_client state` prints zeros on the 2.1.x master). Fewer than all enabled is reported with the numbers and
NOT undone automatically (disabling drops the head and arms: the operator decides). Disable needs the operator's
confirmation after DISABLE_WARNING and is verified by motor_state 0.

run(argv, timeout) -> (rc, text) is injected (motors.run_cmd on the robot); dry_run=True records the commands it
would send and runs none of the two mutating ones."""
import re
import time
from dataclasses import dataclass, field

from calibtool import motors, preflight

ENABLE_ARGV = ["dma_cmd_client", "enable-motors"]
DISABLE_ARGV = ["dma_cmd_client", "disable-motors"]
BUS_MIN_V = 44.0                 # the bus reads ~53 V charged, ~3 V with the motor power off (FF10 latched)
VERIFY_S = 5.0
POLL_S = 0.5
DISABLE_WARNING = ("Disabling the motors makes the robot limp: the head and arms drop. Only with the operator next to "
                   "the robot, holding the head if it is not on the gantry. Press D again to disable.")
_ENABLED = re.compile(r"(\d+)\s+of\s+(\d+)\s+motors\s+enabled", re.I)
OPTIONAL = ("bus_v", "head_power", "slaves_op", "fault_latches")


def parse_enabled(text):
    """(n, of) from the last "N of M motors enabled" line, or None"""
    m = _ENABLED.findall(text or "")
    return None if not m else (int(m[-1][0]), int(m[-1][1]))


def passed(checks) -> bool:
    return bool(checks) and not any(c.status == "fail" for c in checks)


def pre_enable_checks(s: dict, baseline: dict, gantry: bool) -> list:
    out = []

    def add(name, ok, detail_ok, detail_bad):
        out.append(preflight.Check(name, "ok" if ok else "fail", detail_ok if ok else detail_bad, "motion"))

    add("gantry", gantry is True, "the operator confirmed the robot hangs on the gantry",
        "the operator has not confirmed the robot hangs on the gantry")
    a = s.get("autonomy")
    armed = a is not None and re.search(r"\barmed\b", a, re.I) and not re.search(r"\b(dis|un|not\s+)armed\b", a, re.I)
    add("autonomy", not armed, "nav-cli not installed: nothing to arm" if a is None else f"not armed: {a.strip()[:80]}",
        f"autonomy is armed ({(a or '').strip()[:80]}): disarm it first")
    pol = preflight.policy_pods(s.get("pods") or [])
    add("policy_pods", not pol, "no locomotion/policy pod running",
        "running: " + ", ".join(pol) + " (stop them first: the robot is reserved for the session)")
    add("desired_still", s.get("desired_still") is True, "/dev/shm/desired still for 1 s: nothing commanding",
        "/dev/shm/desired is advancing: something is commanding the motors")
    m = s.get("master")
    ok_m = m is not None and m.active and not m.error
    add("master", ok_m, f"dma-ethercat {getattr(m, 'version', '?')} active, ping answers",
        f"master not ready: {getattr(m, 'error', None) or 'no status'}")
    est = getattr(m, "estop", None)
    add("estop", est is False, "e-stop not engaged",
        "e-stop engaged: release it (with it in, the drives latch FF10 and the bus reads ~3 V)" if est else
        f"e-stop state unknown ({est}): not enabling")
    on = getattr(m, "motors_on", None)
    add("motors_off", on is False, "motors are off", f"motors are already on (motor_state "
        f"{getattr(m, 'motor_state', None)}): someone else enabled them; find out who before going on" if on else
        "motor state unknown (ping gave none)")
    na = "not available on this master"
    v = s.get("bus_v")
    if v is None:
        out.append(preflight.Check("bus_v", "info", f"48 V bus: {na}", "motion"))
    else:
        add("bus_v", v >= BUS_MIN_V, f"48 V bus at {v:.1f} V", f"48 V bus at {v:.1f} V (< {BUS_MIN_V:g} V): the motor "
            "power is off (e-stop in, or the power switch)")
    hp = s.get("head_power")
    if hp is None:
        out.append(preflight.Check("head_power", "info", f"head power switch: {na}", "motion"))
    else:
        add("head_power", hp is True, "head power switch on", "head power switch off: switch it on first")
    so, base = s.get("slaves_op"), (baseline or {}).get("slaves_op")
    if so is None or base is None:
        out.append(preflight.Check("slaves_op", "info", f"EtherCAT slaves in OP: {na if so is None else so} "
                                   f"(baseline {'none' if base is None else base})", "motion"))
    else:
        add("slaves_op", so >= base, f"{so} EtherCAT slaves in OP (preflight baseline {base})",
            f"{so} EtherCAT slaves in OP, {base} at preflight: {base - so} left OP")
    fl = s.get("fault_latches")
    if fl is None:
        out.append(preflight.Check("fault_latches", "info", f"per-joint fault latches: {na}", "motion"))
    else:
        add("fault_latches", not fl, "no per-joint fault latch", "fault latched: " + ", ".join(map(str, fl)))
    return out


@dataclass
class EnableResult:
    ok: bool
    refused: bool
    detail: str
    checks: list = field(default_factory=list)
    n: object = None
    of: object = None
    dry_run: bool = False

    def as_dict(self) -> dict:
        return {"ok": self.ok, "refused": self.refused, "detail": self.detail, "n": self.n, "of": self.of,
                "dry_run": self.dry_run, "checks": [c.as_dict() for c in self.checks]}


@dataclass
class DisableResult:
    ok: bool
    refused: bool
    detail: str
    dry_run: bool = False

    def as_dict(self) -> dict:
        return {"ok": self.ok, "refused": self.refused, "detail": self.detail, "dry_run": self.dry_run}


class Enabler:
    """probe() -> the state dict pre_enable_checks reads (EnableProbe on the robot); baseline: preflight's
    (slaves_op); wall: the wall clock for the journal's --since"""

    def __init__(self, run, probe, baseline=None, clock=time.monotonic, sleep=time.sleep, wall=time.time,
                 dry_run=False, verify_s=VERIFY_S, adopt_on=False):
        # adopt_on (10 Nov live session): motors the operator already enabled are taken over on their Y, with no
        # enable command, when every other check passes and all motors are verified on; off = refuse as before
        self.adopt_on = adopt_on
        self.run, self.probe, self.baseline = run, probe, dict(baseline or {})
        self.clock, self.sleep, self.wall, self.dry_run, self.verify_s = clock, sleep, wall, dry_run, verify_s
        self.commands = []

    def checks(self, gantry) -> list:
        try:
            s = self.probe()
        except Exception as e:  # noqa: BLE001  (a broken probe is a failed check, never an enable)
            return [preflight.Check("probe", "fail", f"could not check: {type(e).__name__}: {e}", "motion")]
        return pre_enable_checks(s, self.baseline, gantry)

    def _ping_state(self):
        try:
            rc, out = self.run(["dma_cmd_client", "ping"], timeout=5)
        except Exception:  # noqa: BLE001
            return None
        p = motors.parse_ping(out) if rc == 0 else None
        return None if p is None else p["motor_state"]

    def enable(self, key, gantry) -> EnableResult:
        if key != "Y":
            return EnableResult(False, True, f"enable needs the operator's Y (got {key!r}); nothing sent")
        cs = self.checks(gantry)
        failed = [c for c in cs if c.status == "fail"]
        if self.adopt_on and failed and all(c.name == "motors_off" for c in failed) and "already on" in failed[0].detail:
            return self._adopt(cs)
        if not passed(cs):
            bad = "; ".join(f"{c.name}: {c.detail}" for c in cs if c.status == "fail")
            return EnableResult(False, True, f"not enabling: {bad}", cs)
        self.commands.append(list(ENABLE_ARGV))
        if self.dry_run:
            return EnableResult(False, False, "dry run: would send " + " ".join(ENABLE_ARGV), cs, dry_run=True)
        since = int(self.wall()) - 1
        rc, out = self.run(list(ENABLE_ARGV), timeout=15)
        t0, state, cnt = self.clock(), None, None
        while True:
            state = self._ping_state()
            try:
                jrc, jout = self.run(["journalctl", "-u", motors.SERVICE, "--since", f"@{since}", "-o", "cat",
                                      "--no-pager"], timeout=10)
            except Exception:  # noqa: BLE001
                jrc, jout = 1, ""
            cnt = parse_enabled(jout) if jrc == 0 else None
            if cnt is not None and cnt[0] != cnt[1]:
                return EnableResult(False, False, f"only {cnt[0]} of {cnt[1]} motors enabled: STOP. Not disabled "
                                    "automatically (that drops the head and arms); the operator decides", cs, *cnt)
            if state and cnt is not None:
                return EnableResult(True, False, f"{cnt[0]} of {cnt[1]} motors enabled (motor_state {state})", cs,
                                    *cnt)
            if self.clock() - t0 >= self.verify_s:
                break
            self.sleep(POLL_S)
        why = []
        if not state:
            why.append(f"ping motor_state {state} after enable-motors (rc {rc}: {(out or '').strip()[:80]})")
        if cnt is None:
            why.append(f"no 'N of N motors enabled' line in the {motors.SERVICE} journal")
        return EnableResult(False, False, f"enable not verified within {self.verify_s:g} s: " + "; ".join(why), cs,
                            *(cnt or (None, None)))

    def _adopt(self, cs) -> EnableResult:
        """motors already on: verify (ping + the journal's last 'N of N motors enabled') and take them over; nothing
        is sent. A partial or unverified state is not taken over"""
        state = self._ping_state()
        try:
            jrc, jout = self.run(["journalctl", "-u", motors.SERVICE, "--since", "-2h", "-o", "cat", "--no-pager"],
                                 timeout=10)
        except Exception:  # noqa: BLE001
            jrc, jout = 1, ""
        cnt = parse_enabled(jout) if jrc == 0 else None
        if not state or cnt is None:
            return EnableResult(False, True, f"motors on, but not verified (motor_state {state}, journal "
                                f"{'no N of N line' if cnt is None else cnt}): not taking them over", cs)
        if cnt[0] != cnt[1]:
            return EnableResult(False, True, f"motors on, but only {cnt[0]} of {cnt[1]} enabled: not taking them "
                                "over; the operator decides", cs, *cnt)
        return EnableResult(True, False, f"motors already on ({cnt[0]} of {cnt[1]}, motor_state {state}): taken over "
                            "on the operator's Y, no enable sent", cs, *cnt)

    def disable(self, confirmed) -> DisableResult:
        if confirmed is not True:
            return DisableResult(False, True, DISABLE_WARNING)
        self.commands.append(list(DISABLE_ARGV))
        if self.dry_run:
            return DisableResult(False, False, "dry run: would send " + " ".join(DISABLE_ARGV), True)
        rc, out = self.run(list(DISABLE_ARGV), timeout=15)
        t0 = self.clock()
        while True:
            st = self._ping_state()
            if st == 0:
                return DisableResult(True, False, "motors disabled (motor_state 0)")
            if self.clock() - t0 >= self.verify_s:
                return DisableResult(False, False, f"disable not verified: motor_state {st} (rc {rc}: "
                                     f"{(out or '').strip()[:80]})")
            self.sleep(POLL_S)


class EnableProbe:
    """the robot's state for pre_enable_checks: autonomy and pods from preflight.RobotProbes' callables, the master
    (motors.master), desired_still (motors.desired_still on /dev/shm/desired). The optional facts (bus voltage, head
    power, slaves in OP, fault latches) come from callables passed in; None = not available on this master"""

    def __init__(self, run, autonomy, pods, shm="/dev/shm", bus_v=None, head_power=None, slaves_op=None,
                 fault_latches=None):
        from pathlib import Path
        self.run, self.autonomy, self.pods, self.shm = run, autonomy, pods, Path(shm)
        self.opt = {"bus_v": bus_v, "head_power": head_power, "slaves_op": slaves_op, "fault_latches": fault_latches}

    def __call__(self) -> dict:
        s = {"autonomy": self.autonomy(), "pods": self.pods(),
             "desired_still": motors.desired_still(lambda: motors.ring_head(self.shm / "desired")),
             "master": motors.master(self.run)}
        for k, fn in self.opt.items():
            s[k] = None if fn is None else fn()
        return s
