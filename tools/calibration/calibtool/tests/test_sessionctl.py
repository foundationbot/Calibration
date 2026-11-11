"""calibtool.sessionctl: what a full session changes on the robot besides the neck, motors and the deploy, each with
its way back: the command log every robot action lands in (the dry-run command list), the simulated producer kubectl a
dry run uses, OKVIS stopped for the collection and started again through the restart path the snapshot detected, and
the producer at the collection size under the lease with its snapshot restored. Fakes only."""
import json

import pytest

yaml = pytest.importorskip("yaml")

from calibtool import deploy as dp, producer as pr, sessionctl as sc  # noqa: E402

PRODUCER_YAML = """
apiVersion: apps/v1
kind: Deployment
metadata: {name: producer, namespace: dma-video, generation: 7, resourceVersion: "9"}
spec:
  replicas: 1
  strategy: {type: Recreate}
  selector: {matchLabels: {app: producer}}
  template:
    metadata: {labels: {app: producer}}
    spec:
      containers:
      - name: producer
        image: foundationbot/dma-video:x
        args: ["--width", "960", "--height", "600", "--fps", "25", "--camera-config", "/opt/configs/cameras.json"]
status: {replicas: 1}
"""


class Clock:
    def __init__(self):
        self.t = 1.79e9

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


# ---------------------------------------------------------------- the command log

def test_recorder_keeps_the_order_and_writes_one_line_per_command(tmp_path):
    r = sc.Recorder(dry_run=True)
    r.add("producer", ["k0s", "kubectl", "scale", "deploy", "producer", "--replicas=0"])
    r.add("deploy", ["sh", "-c", "cat > x", "sh", "/etc/x"], input="abc")
    r.note("lease taken by session S")
    p = r.write(tmp_path / "commands.txt")
    lines = p.read_text().splitlines()
    assert lines[0].startswith("# DRY RUN")
    body = [x for x in lines if not x.startswith("#")]
    assert body[0] == "[producer] k0s kubectl scale deploy producer --replicas=0"
    assert body[1].startswith("[deploy] sh -c 'cat > x' sh /etc/x  # <<< 3 bytes on stdin")
    assert r.sources() == ["producer", "deploy"]


def test_remote_hands_every_mutating_command_to_the_log():
    seen = []
    rem = dp.Remote(lambda argv, input=None, timeout=None: (0, "x"), dry_run=True, log=lambda a, i: seen.append(a))
    rem.read("/etc/a")
    rem.write("/etc/b", "t")
    rem.call(["k0s", "kubectl", "rollout", "restart", "ds/okvis2x"], mutating=True)
    assert [a[0] for a in seen] == ["sh", "k0s"]


# ---------------------------------------------------------------- the simulated producer kubectl

def test_dry_kubectl_runs_the_no_overlap_cycle_and_records_only_changes():
    r = sc.Recorder(dry_run=True)
    k = sc.DryKubectl(PRODUCER_YAML, r)
    kube = pr.Kube(k, "dma-video", "producer", read_only=False)
    spec = pr.Spec.from_yaml(kube.get_yaml())
    c = Clock()
    pr.resize(kube, spec, 1280, 800, 15, sleep=c.sleep, clock=c)
    assert pr.Spec.from_yaml(kube.get_yaml()).size() == (1280, 800, 15)
    assert k.max_pods == 1                                               # never two pods at once
    verbs = [ln.split()[3] for ln in r.lines()]
    assert verbs == ["scale", "patch", "scale"]
    pr.restore(kube, spec, sleep=c.sleep, clock=c)
    assert pr.Spec.from_yaml(kube.get_yaml()).size() == (960, 600, 25)


def test_dry_kubectl_can_fail_a_verb():
    k = sc.DryKubectl(PRODUCER_YAML, sc.Recorder(), fail={"patch"})
    kube = pr.Kube(k, "dma-video", "producer", read_only=False)
    c = Clock()
    with pytest.raises(pr.ProducerError):
        pr.resize(kube, pr.Spec.from_yaml(PRODUCER_YAML), 1280, 800, 15, sleep=c.sleep, clock=c)
    assert k.replicas == 1                                               # the failed cycle scaled it back


# ---------------------------------------------------------------- OKVIS stopped for the collection

class FakeRemoteRobot:
    def __init__(self, node="phantom10", label="true", okvis_running=True):
        self.calls, self.node, self.label, self.okvis_running = [], node, label, okvis_running

    def __call__(self, argv, input=None, timeout=None):
        self.calls.append(list(argv))
        a = list(argv)
        if "pods" in a and "-l" in a and not any("jsonpath" in x for x in a):
            items = [{"metadata": {"name": "okvis2x", "namespace": "okvis"}, "status": {"phase": "Running"}}]
            return 0, json.dumps({"items": items if self.okvis_running else []})
        if "jsonpath={.items[0].spec.nodeName}" in a:
            return 0, self.node
        if "get" in a and "node" in a:
            return 0, self.label
        if "label" in a:
            self.label = a[-2].split("=")[1]
            return 0, "node/x labeled"
        return 0, ""


DS = {"kind": "daemonset", "argv": dp.kubectl("-n", dp.OKVIS_NS, "rollout", "restart", "ds/okvis2x"), "detail": ""}
SH = {"kind": "okvis_sh", "argv": ["env", "OKVIS_LAUNCH_ORIGIN=calibtool", "bash", dp.OKVIS_SH, "restart"],
      "detail": ""}


def test_okvis_on_the_daemonset_is_stopped_by_its_node_label_and_started_as_it_was():
    robot = FakeRemoteRobot()
    r = sc.Recorder()
    ok = sc.OkvisCtl(dp.Remote(robot, log=r.hook("okvis")), DS)
    ok.stop()
    assert robot.label == "false" and ok.stopped
    assert any("foundation.bot/has-okvis=false" in " ".join(a) for a in robot.calls)
    assert not any("--grace-period" in " ".join(a) for a in robot.calls)
    ok.start()
    assert robot.label == "true" and not ok.stopped
    assert ok.start_argv()[-2] == "foundation.bot/has-okvis=true"
    assert r.sources() == ["okvis"]


def test_okvis_sh_robots_stop_and_start_through_the_script():
    robot = FakeRemoteRobot()
    ok = sc.OkvisCtl(dp.Remote(robot), SH)
    ok.stop()
    assert robot.calls[-1][-2:] == [dp.OKVIS_SH, "stop"]
    ok.start()
    assert robot.calls[-1][-2:] == [dp.OKVIS_SH, "start"]


def test_an_okvis_that_is_not_running_is_neither_stopped_nor_started_afterwards():
    """10 Nov, phantom10: OKVIS was not running before the session; okvis.sh stop ran anyway and the self-restore then
    started OKVIS"""
    robot = FakeRemoteRobot(okvis_running=False)
    ok = sc.OkvisCtl(dp.Remote(robot), SH)
    ok.stop()
    ok.start()
    assert not any(dp.OKVIS_SH in " ".join(a) for a in robot.calls) and not ok.stopped


def test_no_okvis_on_the_robot_is_left_alone():
    robot = FakeRemoteRobot()
    ok = sc.OkvisCtl(dp.Remote(robot), {"kind": "none", "argv": [], "detail": "none"})
    ok.stop()
    ok.start()
    assert robot.calls == [] and not ok.stopped


def test_okvis_state_survives_a_restart():
    robot = FakeRemoteRobot()
    ok = sc.OkvisCtl(dp.Remote(robot), DS)
    ok.stop()
    again = sc.OkvisCtl.from_dict(dp.Remote(robot), ok.as_dict())
    assert again.stopped and again.node == "phantom10"
    again.start()
    assert robot.label == "true"


# ---------------------------------------------------------------- the producer under the lease

def producer_ctl(tmp_path, fail=(), lease=None):
    r = sc.Recorder(dry_run=True)
    k = sc.DryKubectl(PRODUCER_YAML, r, fail=set(fail))
    kube = pr.Kube(k, "dma-video", "producer", read_only=False)
    c = Clock()
    lease = lease or pr.Lease(tmp_path / "producer.lease")
    return sc.ProducerCtl(kube, pr.Spec.from_yaml(PRODUCER_YAML), lease, "S1", tmp_path / "producer.yaml", clock=c,
                          sleep=c.sleep), k, lease


def test_producer_to_the_collection_size_takes_the_lease_first(tmp_path):
    p, k, lease = producer_ctl(tmp_path)
    p.to_size(1280, 800, 15)
    assert lease.read()["session"] == "S1"
    assert p.changed and p.size == (1280, 800, 15)
    assert (tmp_path / "producer.yaml").read_text() == PRODUCER_YAML
    p.restore()
    assert lease.read() is None and not p.changed
    assert k.args[k.args.index("--width") + 1] == "960"


def test_a_live_lease_of_another_session_refuses_before_any_change(tmp_path):
    lease = pr.Lease(tmp_path / "producer.lease")
    lease.acquire("OTHER", "x", 1.79e9)
    p, k, _ = producer_ctl(tmp_path, lease=lease)
    with pytest.raises(pr.LeaseHeld):
        p.to_size(1280, 800, 15)
    assert k.changes == 0


def test_a_producer_already_at_the_size_is_not_restarted(tmp_path):
    p, k, _ = producer_ctl(tmp_path)
    p.to_size(960, 600, 25)
    assert k.changes == 0 and not p.changed


def test_the_producer_state_survives_a_restart(tmp_path):
    p, k, lease = producer_ctl(tmp_path)
    p.to_size(1280, 800, 15)
    d = json.loads(json.dumps(p.as_dict()))
    p2 = sc.ProducerCtl.from_dict(d, p.kube, lease, "S1", tmp_path / "producer.yaml", clock=p.clock, sleep=p.sleep)
    assert p2.changed and p2.spec.size() == (960, 600, 25)
    p2.restore()
    assert k.args[k.args.index("--width") + 1] == "960"


def test_an_attached_recorder_writes_each_command_as_it_happens(tmp_path):
    r = sc.Recorder(dry_run=True)
    r.add("motors", ["dma_cmd_client", "enable-motors"])
    r.attach(tmp_path / "commands.txt")
    r.add("neck", ["dma_motor_motion", "NeckPitch=1,NeckYaw=2"])
    lines = (tmp_path / "commands.txt").read_text().splitlines()
    assert lines[0].startswith("# DRY RUN") and lines[1:] == ["[motors] dma_cmd_client enable-motors",
                                                             "[neck] dma_motor_motion NeckPitch=1,NeckYaw=2"]
    again = sc.Recorder(dry_run=True)                # a restarted calibd appends after what the crashed one wrote
    again.attach(tmp_path / "commands.txt")
    again.note("calibd restarted")
    lines = (tmp_path / "commands.txt").read_text().splitlines()
    assert len([x for x in lines if x.startswith("# DRY RUN")]) == 1 and lines[-1] == "# calibd restarted"
