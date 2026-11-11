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
RESTART_ARGV = ["systemctl", "restart", "dma-ethercat"]
RESTART_WAIT_S = 90.0
# 10 Nov, phantom10: the LeftShoulderPitch drive faulted 19 ms into every enable (a standing encoder error), and the
# master, enabling all motors at once, left all of them off; the screen said "no 'N of N motors enabled' line"
_FAULT = re.compile(r"\[(\w+)\] FAULT latched")
_ENC_ACTIVE = re.compile(r"\[ACTIVE\] Motor \d+ \((\w+)\): ([^()\[]*?error)[^\n]*\bSET\b")


def parse_faults(text) -> list:
    """the joints whose drive latched a fault (journal "[Joint] FAULT latched"), in order, once each"""
    out = []
    for j in _FAULT.findall(text or ""):
        if j not in out:
            out.append(j)
    return out


def parse_active_errors(text) -> dict:
    """{joint: error} from the master's "[ACTIVE] Motor N (Joint): <...> error ... SET" lines (still set)"""
    return {j: e.strip() for j, e in _ENC_ACTIVE.findall(text or "")}
POLL_S = 0.5
DISABLE_WARNING = ("Disabling the motors makes the robot limp: the head and arms drop. Only with the operator next to "
                   "the robot, holding the head if it is not on the gantry. Press D again to disable.")
_ENABLED = re.compile(r"(?:(\d+) of (\d+) motors enabled|\benabled (\d+)/(\d+))")   # 2.1.0 line | 2.1.1 roll-call
_ALL_ENABLED = re.compile(r"All (\d+) motors enabled")                                  # 2.1.1 MotorEnableManager
OPTIONAL = ("bus_v", "head_power", "slaves_op", "fault_latches")


def parse_enabled(text):
    """(n, of) from the last "N of M motors enabled" line, or None"""
    m = _ENABLED.findall(text or "")
    if not m:
        al = _ALL_ENABLED.findall(text or "")
        return (int(al[-1]), int(al[-1])) if al else None
    a, b, c, d = m[-1]
    cnt = (int(a), int(b)) if a else (int(c), int(d))
    al = _ALL_ENABLED.findall(text or "")
    if cnt[0] != cnt[1] and al and int(al[-1]) == cnt[1]:
        # 10 Nov, phantom10 (2.1.1): the roll-call "enabled 0/29" is logged at the request, "All 29 motors enabled"
        # 20 ms later; a partial count before an "All N" line is the request's, not the result
        last_all = (text or "").rfind(f"All {al[-1]} motors enabled")
        last_cnt = max((text or "").rfind(f"{cnt[0]} of {cnt[1]} motors enabled"),
                       (text or "").rfind(f"enabled {cnt[0]}/{cnt[1]}"))
        if last_all > last_cnt:
            return (cnt[1], cnt[1])
    return cnt


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
    dr = s.get("drives")
    degraded = bool(dr) and dr.get("answering") is not None and dr["answering"] < dr["expected"]
    if dr is None:
        out.append(preflight.Check("drives", "info", "EtherCAT drive replies: not readable from the master's journal",
                                   "motion"))
    else:
        # 10 Nov, phantom10: wkc 21 of 144 for hours (the drives' power off); the enable could not work and the screen
        # said "someone else enabled them"
        add("drives", not degraded, "all motor drives answering on EtherCAT",
            f"only {dr.get('answering')} of {dr.get('expected')} EtherCAT replies: most motor drives are not "
            "answering. The motor power is probably off (power switch, breaker, e-stop box): switch it on, wait until "
            "this line is ok, then Y")
    errs = s.get("joint_errors") or {}
    if errs:
        out.append(preflight.Check("joint_errors", "warn", "; ".join(f"{j}: {e}" for j, e in errs.items()) +
                                   " (active now: that drive will likely fault on enable, and the master then leaves "
                                   "all motors off)", "motion"))
    on = getattr(m, "motors_on", None)
    st = getattr(m, "motor_state", None)
    if st == 1:
        # an enable that did not complete (seen right after enable-motors with the drives unpowered)
        add("motors_off", not degraded, "the last enable did not complete (motor_state 1): Y enables again",
            "the last enable did not complete (motor_state 1): the drives are not answering (see 'drives')")
    else:
        add("motors_off", on is False, "motors are off", f"motors are already on (motor_state {st}): someone else "
            "enabled them; find out who before going on" if on else "motor state unknown (ping gave none)")
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
            self._last_state = None
            return [preflight.Check("probe", "fail", f"could not check: {type(e).__name__}: {e}", "motion")]
        self._last_state = s
        return pre_enable_checks(s, self.baseline, gantry)

    def _ping_state(self):
        """motor_state from a fresh ping (motors.ping: a single ping returns the previous request's reply)"""
        try:
            p = motors.ping(self.run, self.sleep)
        except Exception:  # noqa: BLE001
            return None
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
            faults = parse_faults(jout) if jrc == 0 else []
            if faults and not (cnt is not None and cnt[0] == cnt[1]):
                return EnableResult(False, False, f"{', '.join(faults)} drive faulted during the enable, so the master "
                                    "left all motors off (it enables them together). A fault that comes back at once "
                                    "is standing (encoder, cable, power of that joint): it needs hardware attention. "
                                    "R restarts the motor master; then Y", cs)
            if cnt is not None and cnt[0] != cnt[1] and self.clock() - t0 >= self.verify_s:
                # 10 Nov: the first count after the request is the request's ("enabled 0/29"); judged at the end
                return EnableResult(False, False, f"only {cnt[0]} of {cnt[1]} motors enabled: STOP. Not disabled "
                                    "automatically (that drops the head and arms); the operator decides", cs, *cnt)
            if cnt is not None and cnt[0] != cnt[1]:
                self.sleep(POLL_S)
                continue
            if state and cnt is not None:
                return EnableResult(True, False, f"{cnt[0]} of {cnt[1]} motors enabled (motor_state {state})", cs,
                                    *cnt)
            if self.clock() - t0 >= self.verify_s:
                break
            self.sleep(POLL_S)
        # 10 Nov, phantom10: the motors were already on (enabled by this session before a calibd restart; a ping from
        # inside calibd had read motor_state 0): a second ENABLE_MOTORS logs no new count. The master's newest count
        # in the last 15 min plus any ping reading "on" decide then
        try:
            jrc, jout = self.run(["journalctl", "-u", motors.SERVICE, "--since", "-15min", "-o", "cat",
                                  "--no-pager"], timeout=10)
        except Exception:  # noqa: BLE001
            jrc, jout = 1, ""
        recent = parse_enabled(jout) if jrc == 0 else None
        if recent is not None and recent[0] == recent[1] and (state == 2 or self._ping_state() == 2):
            return EnableResult(True, False, f"{recent[0]} of {recent[1]} motors enabled (they were already on; "
                                "the master's log and ping agree)", cs, *recent)
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
        # the motor state the checks just read (10 Nov: a separate ping from inside calibd answered motor_state 0 while
        # the master reported 2 to the probe and to the shell); a second ping only if the probe had none
        m = (getattr(self, "_last_state", None) or {}).get("master")
        state = getattr(m, "motor_state", None) if getattr(m, "motors_on", None) else None
        state = state or self._ping_state()
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

    def restart_master(self) -> EnableResult:
        """R on the motor screen (approved by the operator on 10 Nov): systemctl restart dma-ethercat, only with no
        motor enabled (motor_state 2 = on: a restart would drop the head and arms); then wait for the ping and for the
        drives to answer (no [WKC] degraded line for 5 s)"""
        m = motors.master(self.run, self.sleep)
        if getattr(m, "motor_state", None) == 2:
            return EnableResult(False, True, "the motors are on: restarting the motor master would drop the head and "
                                "arms; not restarting")
        self.commands.append(list(RESTART_ARGV))
        if self.dry_run:
            return EnableResult(False, False, "dry run: would send " + " ".join(RESTART_ARGV), dry_run=True)
        rc, out = self.run(list(RESTART_ARGV), timeout=60)
        if rc != 0:
            return EnableResult(False, False, f"motor master restart failed (rc {rc}): {(out or '').strip()[:200]}")
        t0 = self.clock()
        while self.clock() - t0 < RESTART_WAIT_S:
            self.sleep(POLL_S)
            if self._ping_state() is None:
                continue
            try:
                jrc, jout = self.run(["journalctl", "-u", motors.SERVICE, "--since", "-5s", "-o", "cat",
                                      "--no-pager"], timeout=10)
            except Exception:  # noqa: BLE001
                jrc, jout = 1, ""
            d = parse_drives(jout) if jrc == 0 else None
            if d is not None and d.get("answering") is None and self.clock() - t0 >= 10.0:
                return EnableResult(False, False, "motor master restarted; all drives answering. Y enables the "
                                    "motors")
        return EnableResult(False, False, f"motor master restarted, but the drives were not all answering within "
                            f"{RESTART_WAIT_S:g} s: see the drives line")

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


_WKC_DEGRADED = re.compile(r"\[WKC\] degraded: actual=(\d+) expected=(\d+)")


def parse_drives(journal_text):
    """the master's journal of the last seconds -> {"answering", "expected"}: from its newest "[WKC] degraded:
    actual=A expected=B" (logged every second while drives do not answer), else {"answering": None} (none degraded)"""
    for ln in reversed((journal_text or "").splitlines()):
        m = _WKC_DEGRADED.search(ln)
        if m:
            return {"answering": int(m.group(1)), "expected": int(m.group(2))}
    return {"answering": None, "expected": None}


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
             "master": motors.master(self.run), "drives": self._drives(), "joint_errors": self._joint_errors()}
        for k, fn in self.opt.items():
            s[k] = None if fn is None else fn()
        return s

    def _joint_errors(self):
        """joints with an error the master reports as still SET in the last 2 minutes ({} when none or unreadable)"""
        try:
            rc, out = self.run(["journalctl", "-u", motors.SERVICE, "--since", "-2min", "-o", "cat", "--no-pager"],
                               timeout=10)
        except Exception:  # noqa: BLE001
            return {}
        return parse_active_errors(out) if rc == 0 else {}

    def _drives(self):
        """EtherCAT replies from the master's journal of the last 5 s (None: not readable)"""
        try:
            rc, out = self.run(["journalctl", "-u", motors.SERVICE, "--since", "-5s", "-o", "cat", "--no-pager"],
                               timeout=10)
        except Exception:  # noqa: BLE001  (not readable: an info line, never a pass)
            return None
        return parse_drives(out) if rc == 0 else None
