"""calibtool.producer: the camera producer's spec snapshot and restore, the no-overlap restart/resize (never a new
producer pod while the old one still exists: two producers on one camera board crash it), the lease file that makes
an abandoned session restore itself, and the read-only guard that keeps every mutating kubectl call from running in
M2. Tested against a FAKE kubectl only; on the robot M2 runs read-only calls."""
import json

import pytest

yaml = pytest.importorskip("yaml")

from calibtool import producer as pr  # noqa: E402

DEPLOY_YAML = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: producer
  namespace: dma-video
  generation: 1276
  resourceVersion: "123456"
  uid: 9b1e
  creationTimestamp: "2026-07-28T00:00:00Z"
  managedFields: [{manager: kubectl}]
  annotations: {deployment.kubernetes.io/revision: "41", note: keep}
spec:
  replicas: 1
  strategy: {type: Recreate}
  selector: {matchLabels: {app: producer}}
  template:
    metadata: {labels: {app: producer}}
    spec:
      containers:
      - name: producer
        image: foundationbot/dma-video:alex-depthai-3.11
        args: ["--width", "1280", "--height", "800", "--fps", "25", "--camera-config", "/opt/configs/cameras.json",
               "--master", "C"]
status: {replicas: 1, readyReplicas: 1}
"""


class FakeKubectl:
    """a kubectl stand-in: records argv; scale changes the wanted replicas; an old pod disappears `down_polls` pod
    listings after a scale to 0; a new pod is Running `up_polls` listings after a scale up"""

    def __init__(self, down_polls=2, up_polls=2, stuck=False):
        self.calls = []
        self.yaml = DEPLOY_YAML
        self.want = 1
        self.pods = [{"name": "producer-a", "uid": "u1", "running": True}]
        self.down_polls, self.up_polls, self.stuck = down_polls, up_polls, stuck
        self._count = 0
        self.max_pods_seen = 1

    def __call__(self, argv, input=None):
        self.calls.append((list(argv), input))
        a = list(argv)
        if a[:2] == ["get", "deploy"]:
            return 0, self.yaml
        if a[:2] == ["get", "pods"]:
            self._tick()
            items = [{"metadata": {"name": p["name"], "uid": p["uid"]},
                      "status": {"phase": "Running" if p["running"] else "Pending",
                                 "containerStatuses": [{"ready": p["running"]}]}} for p in self.pods]
            return 0, json.dumps({"items": items})
        if a[0] == "logs":
            return 0, "line 1\nline 2\n"
        if a[0] == "scale":
            self.want = int(a[-1].split("=")[1])
            self._count = 0
            return 0, "scaled"
        if a[0] in ("patch", "replace"):
            if a[0] == "replace":
                self.want = int(yaml.safe_load(input)["spec"]["replicas"])
            return 0, "ok"
        return 1, f"unknown {a}"

    def _tick(self):
        self._count += 1
        if self.want == 0 and self.pods and not self.stuck and self._count >= self.down_polls:
            self.pods = []
        if self.want > 0 and not self.pods:
            self.pods = [{"name": "producer-b", "uid": "u2", "running": False}]
            self._count = 0
        elif self.want > 0 and self.pods and self._count >= self.up_polls:
            self.pods[0]["running"] = True
        self.max_pods_seen = max(self.max_pods_seen, len(self.pods))

    def verbs(self):
        return [c[0][0] if c[0][0] != "get" else "get " + c[0][1] for c in self.calls]


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def kube(fake, read_only=False):
    return pr.Kube(fake, "dma-video", "producer", read_only=read_only)


# ------------------------------------------------------------------ read-only guard

def test_read_only_runs_reads_and_refuses_writes():
    f = FakeKubectl()
    k = kube(f, read_only=True)
    assert "producer" in k.get_yaml()
    assert len(k.pods()) == 1
    assert k.logs(5) == ["line 1", "line 2"]
    for call in (lambda: k.scale(0), lambda: k.patch_args(["--x"]), lambda: k.replace("x: 1")):
        with pytest.raises(pr.ReadOnlyRefused):
            call()
    assert not any(c[0][0] in ("scale", "patch", "replace") for c in f.calls)


def test_kubectl_argv_names_namespace_and_deployment():
    f = FakeKubectl()
    k = kube(f)
    k.get_yaml()
    k.scale(0)
    assert f.calls[0][0] == ["get", "deploy", "producer", "-n", "dma-video", "-o", "yaml"]
    assert f.calls[1][0] == ["scale", "deploy", "producer", "-n", "dma-video", "--replicas=0"]


def test_real_runner_prefixes_k0s_kubectl_and_kubeconfig():
    seen = {}

    def fake_run(argv, input=None, capture_output=None, text=None, timeout=None, env=None):
        seen.update(argv=argv, env=env)

        class R:
            returncode, stdout, stderr = 0, "ok", ""
        return R()
    rc, out = pr.kubectl_runner(run=fake_run)(["get", "pods"])
    assert seen["argv"][:2] == ["k0s", "kubectl"] and seen["argv"][2:] == ["get", "pods"]
    assert seen["env"]["KUBECONFIG"] == "/var/lib/k0s/pki/admin.conf"
    assert (rc, out) == (0, "ok")


# ------------------------------------------------------------------ Spec

def test_spec_from_yaml():
    s = pr.Spec.from_yaml(DEPLOY_YAML)
    assert s.args[:2] == ["--width", "1280"] and s.image.endswith("alex-depthai-3.11")
    assert (s.replicas, s.generation, s.strategy) == (1, 1276, "Recreate")
    assert s.size() == (1280, 800, 25)


def test_clean_yaml_drops_server_fields_only():
    c = yaml.safe_load(pr.Spec.from_yaml(DEPLOY_YAML).clean_yaml())
    assert "status" not in c
    for k in ("resourceVersion", "uid", "creationTimestamp", "generation", "managedFields"):
        assert k not in c["metadata"]
    assert c["metadata"]["annotations"] == {"note": "keep"}
    assert c["spec"]["template"]["spec"]["containers"][0]["args"][1] == "1280"


def test_set_args_changes_only_the_named_values():
    args = pr.Spec.from_yaml(DEPLOY_YAML).args
    new = pr.set_args(args, width=640, height=400, fps=15)
    assert new == ["--width", "640", "--height", "400", "--fps", "15", "--camera-config", "/opt/configs/cameras.json",
                   "--master", "C"]
    assert pr.set_args(["--master", "C"], fps=15) == ["--master", "C", "--fps", "15"]


# ------------------------------------------------------------------ no-overlap restart / resize / restore

def test_restart_waits_for_the_old_pod_before_scaling_up():
    f, c = FakeKubectl(down_polls=3), Clock()
    pr.restart(kube(f), sleep=c.sleep, clock=c)
    v = f.verbs()
    i0, i1 = v.index("scale"), len(v) - 1 - v[::-1].index("scale")
    assert f.calls[i0][0][-1] == "--replicas=0" and f.calls[i1][0][-1] == "--replicas=1"
    between = v[i0 + 1:i1]
    assert between.count("get pods") >= 3
    assert f.max_pods_seen == 1                       # never two producers
    assert f.pods and f.pods[0]["running"]


def test_a_pod_that_never_goes_stops_the_restart_without_scaling_up():
    f, c = FakeKubectl(stuck=True), Clock()
    with pytest.raises(pr.ProducerError, match="did not go away"):
        pr.restart(kube(f), sleep=c.sleep, clock=c, down_timeout=20)
    assert [x[0][-1] for x in f.calls if x[0][0] == "scale"] == ["--replicas=0"]


def test_resize_patches_args_in_the_no_pod_window():
    f, c = FakeKubectl(), Clock()
    spec = pr.Spec.from_yaml(DEPLOY_YAML)
    pr.resize(kube(f), spec, 640, 400, 15, sleep=c.sleep, clock=c)
    v = f.verbs()
    ip = v.index("patch")
    assert v.index("scale") < ip < len(v) - 1 - v[::-1].index("scale")
    body = json.loads(f.calls[ip][0][f.calls[ip][0].index("-p") + 1])
    assert body[0]["path"] == "/spec/template/spec/containers/0/args"
    assert body[0]["value"][:6] == ["--width", "640", "--height", "400", "--fps", "15"]
    assert f.max_pods_seen == 1


def test_restore_replaces_the_cleaned_saved_spec_then_scales_back(tmp_path):
    f, c = FakeKubectl(), Clock()
    spec = pr.Spec.from_yaml(DEPLOY_YAML)
    lease = pr.Lease(tmp_path / "lease.json")
    lease.acquire("S1", "spec.yaml", now=0.0)
    pr.restore(kube(f), spec, lease=lease, sleep=c.sleep, clock=c)
    v = f.verbs()
    ir = v.index("replace")
    body = yaml.safe_load(f.calls[ir][1])
    assert body["spec"]["replicas"] == 0 and "status" not in body and "resourceVersion" not in body["metadata"]
    assert f.calls[-1][0][0] == "get" or v[-1] == "get pods"
    assert [x[0][-1] for x in f.calls if x[0][0] == "scale"] == ["--replicas=0", "--replicas=1"]
    assert f.max_pods_seen == 1
    assert lease.read() is None                       # released after a good restore


def test_a_failed_restore_keeps_the_lease(tmp_path):
    f, c = FakeKubectl(stuck=True), Clock()
    lease = pr.Lease(tmp_path / "lease.json")
    lease.acquire("S1", "spec.yaml", now=0.0)
    with pytest.raises(pr.ProducerError):
        pr.restore(kube(f), pr.Spec.from_yaml(DEPLOY_YAML), lease=lease, sleep=c.sleep, clock=c, down_timeout=10)
    assert lease.read()["session"] == "S1"


def test_restart_is_refused_read_only():
    f, c = FakeKubectl(), Clock()
    with pytest.raises(pr.ReadOnlyRefused):
        pr.restart(kube(f, read_only=True), sleep=c.sleep, clock=c)
    assert f.verbs() == []


# ------------------------------------------------------------------ Lease

def test_lease_acquire_heartbeat_stale_release(tmp_path):
    lz = pr.Lease(tmp_path / "lease.json")
    assert lz.read() is None
    lz.acquire("S1", "/x/spec.yaml", now=100.0)
    d = lz.read()
    assert (d["session"], d["spec_path"], d["created"], d["heartbeat"]) == ("S1", "/x/spec.yaml", 100.0, 100.0)
    lz.heartbeat(now=200.0)
    assert lz.read()["heartbeat"] == 200.0
    assert not lz.stale(now=200.0 + 7199)
    assert lz.stale(now=200.0 + 7201)
    lz.release()
    assert lz.read() is None


def test_lease_of_another_live_session_is_refused(tmp_path):
    lz = pr.Lease(tmp_path / "lease.json")
    lz.acquire("S1", "a", now=0.0)
    with pytest.raises(pr.LeaseHeld, match="S1"):
        pr.Lease(tmp_path / "lease.json").acquire("S2", "b", now=10.0)
    pr.Lease(tmp_path / "lease.json").acquire("S1", "a", now=10.0)          # the same session takes it back


def test_stale_lease_can_be_taken_over_and_says_so(tmp_path):
    lz = pr.Lease(tmp_path / "lease.json")
    lz.acquire("S1", "a", now=0.0)
    prev = pr.Lease(tmp_path / "lease.json").acquire("S2", "b", now=8000.0)
    assert prev["session"] == "S1"
    assert lz.read()["session"] == "S2"
