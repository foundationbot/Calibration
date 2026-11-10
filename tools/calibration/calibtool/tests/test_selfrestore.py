"""calibtool.selfrestore: the stale-lease self-restore (design plan section 4) and its systemd timer. Fakes only: the
producer is sessionctl.DryKubectl (the no-overlap cycle simulated), OKVIS a fake remote robot (the node label), the
clock injected. What it must never do: act while a calibd holds the lock, while the lease or a TUI is fresh, without
the saved spec, over a deploy; scale a producer up while an old pod remains."""
import importlib.util
import json
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from calibtool import __main__ as cli_main, daemon, producer as pr, selfrestore as srs, sessionctl as sc  # noqa: E402

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("_t_srs_sessionctl", HERE / "test_sessionctl.py")
TS = importlib.util.module_from_spec(spec)
spec.loader.exec_module(TS)

SID = "20261110-010203-P10"
H = 3600.0
NOW = 1.79e9


def live_yaml(width=1280, height=800, fps=15, replicas=1):
    d = yaml.safe_load(TS.PRODUCER_YAML)
    d["spec"]["replicas"] = replicas
    c = d["spec"]["template"]["spec"]["containers"][0]
    c["args"] = pr.set_args(c["args"], width=width, height=height, fps=fps)
    return yaml.safe_dump(d, sort_keys=False)


class World:
    """root with one session left by a dead calibd: producer at 1280 under a lease, OKVIS stopped by its node label"""

    def __init__(self, tmp_path, lease_age_h=3.0, tui_age_h=None, okvis_stopped=True, flow=None, spec=True,
                 kube=None, okvis_fail=False):
        self.root = tmp_path / "root"
        self.sdir = self.root / "sessions" / SID
        self.sdir.mkdir(parents=True)
        self.spec_path = self.sdir / "producer.saved.yaml"
        if spec:
            self.spec_path.write_text(TS.PRODUCER_YAML)
        self.lease = pr.Lease(self.root / "producer.lease")
        self.lease.acquire(SID, self.spec_path, NOW - lease_age_h * H)
        fl = {"stage": "collect", "producer": {"changed": True, "size": [1280, 800, 15]},
              "okvis": {"restart": TS.DS, "node": "phantom10", "label_value": "true", "stopped": okvis_stopped}}
        fl.update(flow or {})
        st = {"session": SID, "phase": "HOLD", "status": "running", "flow": fl, "updated": NOW - lease_age_h * H}
        if tui_age_h is not None:
            st["last_hb_t"] = NOW - tui_age_h * H
        store = daemon.Store(self.sdir)
        store.write_state(st)
        store.event("session_start", t=NOW - 5 * H)
        self.kube = kube or sc.DryKubectl(live_yaml())
        self.robot = TS.FakeRemoteRobot(label="false")
        if okvis_fail:
            def failing(argv, input=None, timeout=None):
                self.robot.calls.append(list(argv))
                return 1, "the API server is down"
            self.shell = failing
        else:
            self.shell = self.robot
        self.lines = []
        self.clock = TS.Clock()
        self.clock.t = NOW

    def run(self, hours=2.0, dry_run=False):
        r = srs.Restore(self.root, hours, clock=self.clock, kube_runner=self.kube, shell=self.shell,
                        log=self.lines.append, dry_run=dry_run,
                        cycle_kw={"sleep": self.clock.sleep, "clock": self.clock})
        return r.run()

    def state(self):
        return json.loads((self.sdir / "state.json").read_text())

    def events(self, kind=None):
        ev = daemon.Store(self.sdir).events()
        return [e for e in ev if kind is None or e["kind"] == kind]

    def changes(self):
        return self.kube.changes + sum(1 for a in self.robot.calls if "label" in a)


def test_a_stale_lease_with_no_calibd_and_no_tui_restores_the_producer_then_okvis(tmp_path):
    w = World(tmp_path)
    assert w.run() == 0
    assert w.kube.args == yaml.safe_load(TS.PRODUCER_YAML)["spec"]["template"]["spec"]["containers"][0]["args"]
    assert w.kube.replicas == 1 and w.kube.max_pods <= 1
    assert w.robot.label == "true"                         # the node label back to the value it had
    assert w.lease.read() is None
    fl = w.state()["flow"]
    assert fl["producer"]["changed"] is False and fl["okvis"]["stopped"] is False
    (ev,) = w.events("stale_restore")
    assert ev["session"] == SID and ev["lease_age_h"] == 3.0 and any("OKVIS started" in x for x in ev["done"])
    txt = (w.sdir / "commands.txt").read_text()
    assert "[producer] k0s kubectl replace" in txt and "has-okvis=true" in txt
    assert txt.index("kubectl scale") < txt.index("has-okvis=true")     # cameras back before OKVIS


def test_it_is_idempotent(tmp_path):
    w = World(tmp_path)
    assert w.run() == 0
    n = w.changes()
    assert w.run() == 0 and w.changes() == n and "no lease" in w.lines[-1]


@pytest.mark.parametrize("lease_h,tui_h", [(1.5, None), (3.0, 1.0)])
def test_a_fresh_lease_or_a_recent_tui_does_nothing(tmp_path, lease_h, tui_h):
    w = World(tmp_path, lease_age_h=lease_h, tui_age_h=tui_h)
    assert w.run() == 0
    assert w.changes() == 0 and w.lease.read() is not None and not w.events("stale_restore")
    assert "nothing done" in w.lines[-1]


def test_a_tui_event_counts_as_a_heartbeat(tmp_path):
    w = World(tmp_path)
    daemon.Store(w.sdir).event("link_lost", t=NOW - 0.5 * H)
    assert w.run() == 0 and w.changes() == 0 and "TUI" in w.lines[-1]


def test_a_calibd_holding_the_lock_does_nothing(tmp_path):
    w = World(tmp_path)
    lock = daemon.RobotLock(w.root / "calibd.lock")
    assert lock.acquire(SID)
    try:
        assert w.run() == 0
    finally:
        lock.release()
    assert w.changes() == 0 and w.lease.read() is not None and "calibd holds" in w.lines[-1]


def test_the_lock_is_held_during_the_restore_and_released_after(tmp_path):
    w = World(tmp_path)
    seen = []
    inner = w.kube

    def kube(argv, input=None):
        if argv[0] == "replace":
            other = daemon.RobotLock(w.root / "calibd.lock")
            seen.append(other.acquire("calibd"))
            other.release()
        return inner(argv, input) if input is not None else inner(argv)
    w.kube = kube
    assert w.run() == 0 and seen == [False]
    other = daemon.RobotLock(w.root / "calibd.lock")
    assert other.acquire("calibd")
    other.release()


def test_no_saved_spec_refuses_loudly_and_sends_nothing(tmp_path):
    w = World(tmp_path, spec=False)
    assert w.run() == 1
    assert w.changes() == 0 and w.lease.read() is not None
    assert "REFUSED" in w.lines[-1] and "no saved producer spec" in w.lines[-1]
    assert len(w.events("stale_restore_refused")) == 1
    assert w.run() == 1 and len(w.events("stale_restore_refused")) == 1      # logged once, said every run


def test_a_missing_session_folder_refuses(tmp_path):
    w = World(tmp_path)
    w.lease.release()
    w.lease.acquire("20261109-000000-P10", w.spec_path, NOW - 3 * H)
    assert w.run() == 1 and w.changes() == 0 and "does not exist" in w.lines[-1]


@pytest.mark.parametrize("flow,word", [({"deploy_started": True}, "interrupted deploy"),
                                       ({"deploy_started": True, "deploy_done": True, "outcome": "deployed"},
                                        "not accepted")])
def test_a_session_that_started_a_deploy_is_refused(tmp_path, flow, word):
    w = World(tmp_path, flow=flow)
    assert w.run() == 1
    assert w.changes() == 0 and word in w.lines[-1] and "rollback.sh" in w.lines[-1]


def test_a_rolled_back_deploy_is_restored_like_any_session(tmp_path):
    w = World(tmp_path, flow={"deploy_started": True, "deploy_done": True, "rolled_back": True})
    assert w.run() == 0 and w.lease.read() is None


def test_a_failed_producer_restore_is_loud_keeps_the_lease_and_leaves_okvis_stopped(tmp_path):
    w = World(tmp_path, kube=sc.DryKubectl(live_yaml(), fail=("replace",)))
    assert w.run() == 1
    assert "FAILED" in w.lines[-1] and "NOT restored" in w.lines[-1] and "producer" in w.lines[-1]
    assert w.lease.read() is not None and w.robot.label == "false"
    assert not any("label" in a for a in w.robot.calls)
    (ev,) = w.events("stale_restore_failed")
    assert ev["error"].startswith("producer:")
    assert w.kube.replicas == 1             # the cycle scaled back up once no pod was left (change not applied)


class PodStays(sc.DryKubectl):
    """the old producer pod never goes away"""

    def __call__(self, argv, input=None):
        if list(argv[:2]) == ["get", "pods"]:
            return 0, json.dumps({"items": [{"metadata": {"name": "producer-old", "uid": "u0"},
                                             "status": {"phase": "Running",
                                                        "containerStatuses": [{"ready": True}]}}]})
        return super().__call__(argv, input)


def test_never_scales_up_while_an_old_producer_pod_remains(tmp_path):
    w = World(tmp_path, kube=PodStays(live_yaml()))
    assert w.run() == 1
    scales = [a for a in w.kube.recorder.commands() if "scale" in a] if w.kube.recorder else []
    assert w.kube.replicas == 0
    assert "PRODUCER LEFT AT 0 REPLICAS" in w.lines[-1]
    assert not scales or all(a[-1] == "--replicas=0" for a in scales)
    assert w.lease.read() is not None and w.robot.label == "false"


def test_an_okvis_failure_keeps_the_lease_and_the_next_run_finishes_without_a_second_producer_cycle(tmp_path):
    w = World(tmp_path, okvis_fail=True)
    assert w.run() == 1 and "OKVIS" in w.lines[-1] and w.lease.read() is not None
    n = w.kube.changes
    w.shell = w.robot
    assert w.run() == 0
    assert w.kube.changes == n                        # the producer was already as saved: not restarted again
    assert w.robot.label == "true" and w.lease.read() is None
    assert any("already as saved" in x for x in w.events("stale_restore")[-1]["done"])


def test_okvis_not_recorded_stopped_is_left_alone(tmp_path):
    w = World(tmp_path, okvis_stopped=False)
    assert w.run() == 0 and not w.robot.calls and w.lease.read() is None


def test_dry_run_says_what_it_would_do_and_sends_nothing(tmp_path):
    w = World(tmp_path)
    assert w.run(dry_run=True) == 0
    assert w.changes() == 0 and w.lease.read() is not None and not (w.sdir / "commands.txt").exists()
    assert "DRY RUN" in w.lines[-1] and "dma-video/producer" in w.lines[-1] and "OKVIS started" in w.lines[-1]


def test_the_cli_entry_with_no_lease_does_nothing(tmp_path, capsys):
    assert cli_main.main(["restore", "--if-stale-hours", "2", "--root", str(tmp_path)]) == 0
    assert "no lease" in capsys.readouterr().out


def test_calibd_records_the_tui_heartbeat_time_in_its_state(tmp_path):
    c = TS.Clock()
    m = daemon.Machine(daemon.Store(tmp_path / "s"), c, session=SID, robot="P10", mono=c)
    c.t += 10
    m.heartbeat()
    m.save()
    assert m.store.load_state()["last_hb_t"] == c.t
    assert srs.last_tui_t(tmp_path / "s") == c.t


# ---------------------------------------------------------------- the systemd units


class Systemctl:
    def __init__(self, rc=0):
        self.calls, self.rc = [], rc

    def __call__(self, argv):
        self.calls.append(list(argv))
        return type("R", (), {"returncode": self.rc})()


def test_the_units_run_restore_every_10_minutes_from_this_kit(tmp_path):
    u = srs.units("/data/calib", 2, python="/root/p11-cal/venv/bin/python", kit="/data/calib/kit/tools/calibration",
                  pythonpath="/root/p11-cal/pydeps")
    svc, tim = u["calibtool-restore.service"], u["calibtool-restore.timer"]
    assert ("ExecStart=/root/p11-cal/venv/bin/python -m calibtool restore --if-stale-hours 2 --root /data/calib"
            in svc)
    assert "WorkingDirectory=/data/calib/kit/tools/calibration" in svc
    assert "Environment=PYTHONPATH=/data/calib/kit/tools/calibration:/root/p11-cal/pydeps" in svc
    assert "Type=oneshot" in svc and "ConditionPathExists=/data/calib/producer.lease" in svc
    assert "OnCalendar=*:0/10" in tim and "Unit=calibtool-restore.service" in tim and "WantedBy=timers.target" in tim


def test_install_dry_run_prints_and_writes_nothing(tmp_path):
    out, sysctl = [], Systemctl()
    assert srs.install(tmp_path / "root", unit_dir=tmp_path, dry_run=True, run=sysctl, log=out.append) == 0
    assert not list(tmp_path.glob("calibtool-restore.*")) and not sysctl.calls
    text = "\n".join(out)
    assert "would write" in text and "systemctl enable --now calibtool-restore.timer" in text
    assert "--if-stale-hours 2 --root" in text


def test_install_writes_both_units_and_enables_the_timer_and_uninstall_undoes_it(tmp_path):
    sysctl = Systemctl()
    assert srs.install(tmp_path / "root", 3, unit_dir=tmp_path, run=sysctl, log=lambda s: None) == 0
    assert (tmp_path / "calibtool-restore.service").read_text().count("--if-stale-hours 3 ") == 1
    assert (tmp_path / "calibtool-restore.timer").is_file()
    assert sysctl.calls == [["systemctl", "daemon-reload"], ["systemctl", "enable", "--now", "calibtool-restore.timer"]]
    sysctl.calls.clear()
    assert srs.uninstall(unit_dir=tmp_path, run=sysctl, log=lambda s: None) == 0
    assert not list(tmp_path.glob("calibtool-restore.*"))
    assert sysctl.calls == [["systemctl", "disable", "--now", "calibtool-restore.timer"],
                            ["systemctl", "daemon-reload"]]


def test_a_failing_systemctl_is_an_error(tmp_path):
    with pytest.raises(RuntimeError, match="daemon-reload"):
        srs.install(tmp_path / "root", unit_dir=tmp_path, run=Systemctl(rc=1), log=lambda s: None)
