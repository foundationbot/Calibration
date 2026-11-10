"""The camera producer (design plan sections 4, 7, 8, D4): its spec snapshot and restore, the no-overlap restart and
resize, the lease file.

No overlap: a second producer pod on the same camera board crashes it, so every restart is scale 0 -> wait until no
producer pod exists -> settle -> (change) -> scale back -> wait until Running; a pod that does not go away stops the
restart before any scale-up (ProducerError). The kit's prodrestart.sh did the same by hand.

Kube runs kubectl through an injected runner (argv, input) -> (rc, text); kubectl_runner() is the robot's
(`k0s kubectl`, KUBECONFIG=/var/lib/k0s/pki/admin.conf). read_only=True (M2) refuses every mutating call
(ReadOnlyRefused) before the runner is reached.

Spec: the deployment as saved in preflight (the "previous default" and the rollback snapshot); clean_yaml() drops the
server-owned fields so `kubectl replace` applies it as saved. Lease: a JSON file naming the session that changed the
producer, the saved spec and a heartbeat; a lease whose heartbeat is older than LEASE_MAX_AGE_S (2 h, design section 4)
is stale: the robot restores itself, or a new session takes it over and restores first."""
import copy
import json
import os
import subprocess
import time
from pathlib import Path

KUBECONFIG = "/var/lib/k0s/pki/admin.conf"
LEASE_MAX_AGE_S = 7200.0
DOWN_TIMEOUT_S = 60.0
UP_TIMEOUT_S = 90.0
SETTLE_S = 3.0
POLL_S = 1.0
SERVER_FIELDS = ("resourceVersion", "uid", "creationTimestamp", "generation", "managedFields", "selfLink")
SERVER_ANNOTATIONS = ("deployment.kubernetes.io/revision", "kubectl.kubernetes.io/last-applied-configuration")


class ReadOnlyRefused(RuntimeError):
    """a mutating kubectl call in read-only mode"""


class ProducerError(RuntimeError):
    pass


class LeaseHeld(RuntimeError):
    pass


def kubectl_runner(run=subprocess.run, kubeconfig=KUBECONFIG, timeout=30):
    """the robot's runner: k0s kubectl with the admin kubeconfig -> (rc, stdout + stderr)"""
    def _run(argv, input=None):
        env = dict(os.environ, KUBECONFIG=kubeconfig)
        try:
            r = run(["k0s", "kubectl", *argv], input=input, capture_output=True, text=True, timeout=timeout, env=env)
        except (OSError, subprocess.TimeoutExpired) as e:
            return 127, f"{type(e).__name__}: {e}"
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    return _run


class Kube:
    def __init__(self, run, ns, deploy, read_only=True):
        self.run, self.ns, self.deploy, self.read_only = run, ns, deploy, read_only

    def _call(self, argv, input=None, mutating=False):
        if mutating and self.read_only:
            raise ReadOnlyRefused(f"read-only: refused kubectl {' '.join(argv[:3])}")
        rc, out = self.run(argv, input=input) if input is not None else self.run(argv)
        if rc != 0:
            raise ProducerError(f"kubectl {' '.join(argv[:3])} failed (rc {rc}): {out.strip()[-300:]}")
        return out

    def get_yaml(self) -> str:
        return self._call(["get", "deploy", self.deploy, "-n", self.ns, "-o", "yaml"])

    def pods(self) -> list:
        """[{name, uid, phase, ready}] of the deployment's pods"""
        out = self._call(["get", "pods", "-n", self.ns, "-l", f"app={self.deploy}", "-o", "json"])
        items = json.loads(out).get("items") or []
        res = []
        for it in items:
            st = it.get("status") or {}
            cs = st.get("containerStatuses") or []
            res.append({"name": it["metadata"]["name"], "uid": it["metadata"].get("uid"), "phase": st.get("phase"),
                        "ready": bool(cs) and all(c.get("ready") for c in cs),
                        "restarts": sum(int(c.get("restartCount") or 0) for c in cs),
                        "deleting": bool(it["metadata"].get("deletionTimestamp"))})
        return res

    def logs(self, tail=50) -> list:
        return self._call(["logs", f"deploy/{self.deploy}", "-n", self.ns, f"--tail={int(tail)}"]).splitlines()

    def scale(self, n):
        return self._call(["scale", "deploy", self.deploy, "-n", self.ns, f"--replicas={int(n)}"], mutating=True)

    def patch_args(self, args):
        body = [{"op": "replace", "path": "/spec/template/spec/containers/0/args", "value": list(args)}]
        return self._call(["patch", "deploy", self.deploy, "-n", self.ns, "--type", "json", "-p", json.dumps(body)],
                          mutating=True)

    def replace(self, yaml_text):
        return self._call(["replace", "-n", self.ns, "-f", "-"], input=yaml_text, mutating=True)


class Spec:
    """the producer deployment as saved"""

    def __init__(self, doc: dict, text: str):
        self.doc, self.text = doc, text
        c = doc["spec"]["template"]["spec"]["containers"][0]
        self.args = [str(a) for a in c.get("args") or []]
        self.image = c.get("image")
        self.replicas = int(doc["spec"].get("replicas", 1))
        self.generation = int(doc["metadata"].get("generation", 0))
        self.strategy = (doc["spec"].get("strategy") or {}).get("type")

    @classmethod
    def from_yaml(cls, text):
        import yaml
        return cls(yaml.safe_load(text), text)

    def size(self):
        """(width, height, fps) from the args (None where absent)"""
        def val(k):
            return int(self.args[self.args.index(k) + 1]) if k in self.args else None
        return val("--width"), val("--height"), val("--fps")

    def clean_yaml(self, replicas=None) -> str:
        import yaml
        d = copy.deepcopy(self.doc)
        d.pop("status", None)
        md = d.get("metadata") or {}
        for k in SERVER_FIELDS:
            md.pop(k, None)
        ann = md.get("annotations")
        if ann:
            for k in SERVER_ANNOTATIONS:
                ann.pop(k, None)
            if not ann:
                md.pop("annotations")
        if replicas is not None:
            d["spec"]["replicas"] = int(replicas)
        return yaml.safe_dump(d, sort_keys=False)


def set_args(args, **kv) -> list:
    """args with `--<key> <value>` replaced (appended when absent); keys with _ become -"""
    out = list(args)
    for k, v in kv.items():
        flag = "--" + k.replace("_", "-")
        if flag in out:
            out[out.index(flag) + 1] = str(v)
        else:
            out += [flag, str(v)]
    return out


def _wait(cond, timeout, sleep, clock, poll=POLL_S):
    t0 = clock()
    while True:
        if cond():
            return True
        if clock() - t0 >= timeout:
            return False
        sleep(poll)


def _cycle(kube: Kube, replicas, between=None, sleep=time.sleep, clock=time.monotonic, down_timeout=DOWN_TIMEOUT_S,
           up_timeout=UP_TIMEOUT_S, settle=SETTLE_S):
    kube.scale(0)
    if not _wait(lambda: not kube.pods(), down_timeout, sleep, clock):
        raise ProducerError(f"the producer pod did not go away within {down_timeout:.0f} s: not scaling up (two "
                            "producers on one camera board crash it)")
    sleep(settle)
    if between is not None:
        between()
    if replicas > 0:
        kube.scale(replicas)
        if not _wait(lambda: any(p["ready"] and p["phase"] == "Running" for p in kube.pods()), up_timeout, sleep,
                     clock):
            raise ProducerError(f"the producer is not Running within {up_timeout:.0f} s after the restart")


def restart(kube: Kube, replicas=1, **kw):
    """no-overlap restart (module text)"""
    _cycle(kube, replicas, **kw)


def resize(kube: Kube, spec: Spec, width, height, fps, **kw):
    """the producer at width x height @ fps: its args patched in the no-pod window, all other args as saved"""
    new = set_args(spec.args, width=width, height=height, fps=fps)
    _cycle(kube, spec.replicas, between=lambda: kube.patch_args(new), **kw)


def restore(kube: Kube, spec: Spec, lease=None, **kw):
    """the saved spec back as saved (replaced with 0 replicas in the no-pod window, then scaled to the saved count);
    the lease is released only when that worked"""
    _cycle(kube, spec.replicas, between=lambda: kube.replace(spec.clean_yaml(replicas=0)), **kw)
    if lease is not None:
        lease.release()


class Lease:
    def __init__(self, path):
        self.path = Path(path)

    def read(self):
        try:
            return json.loads(self.path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def _write(self, d):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(d))
        os.replace(tmp, self.path)

    def stale(self, now, max_age_s=LEASE_MAX_AGE_S) -> bool:
        d = self.read()
        return d is not None and now - float(d["heartbeat"]) > max_age_s

    def acquire(self, session, spec_path, now):
        """take the lease for session -> the previous lease (a stale one taken over, or None). LeaseHeld while
        another session's lease is live"""
        prev = self.read()
        if prev is not None and prev.get("session") != session and not self.stale(now):
            raise LeaseHeld(f"the producer lease is held by session {prev.get('session')} (heartbeat "
                            f"{now - float(prev['heartbeat']):.0f} s ago)")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write({"session": session, "pid": os.getpid(), "spec_path": str(spec_path), "created": float(now),
                     "heartbeat": float(now)})
        return prev if prev is not None and prev.get("session") != session else None

    def heartbeat(self, now):
        d = self.read()
        if d is not None:
            d["heartbeat"] = float(now)
            self._write(d)

    def release(self):
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
