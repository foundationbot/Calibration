"""What a full session changes on the robot besides the neck, the motors and the deploy, each with its way back (design
plan sections 4 and 7; M5 wiring notes: "OKVIS stopped for collection", "the producer's writable Kube").

Recorder     the session's command log: every robot command the session sends (or, in a dry run, would send), in
             order, one line each with its source ([producer], [okvis], [motors], [neck], [deploy]); written to the
             session folder as commands.txt (the dry-run command list). deploy.Remote(log=recorder.hook(source)) feeds
             it; the other sources add their own lines.
DryKubectl   a kubectl runner for producer.Kube that simulates the producer deployment (pods go when scaled to 0 and
             come back Running when scaled up; patch/replace change its spec) and records only the changes: a dry run
             walks the real no-overlap cycle without touching the robot (also the tests' fake kubectl).
OkvisCtl     OKVIS stopped for the whole collection (decided 9 Nov: its config would not match 1280) and started again
             through the path the snapshot detected (deploy.restart_path): on the DaemonSet robots the node label
             foundation.bot/has-okvis (RUNBOOK 7: "taking OKVIS off a robot is the node label", the pod keeps its 60 s
             grace; never --grace-period=0) set to false and back to the value it had; on the older robots okvis.sh
             stop / start; no OKVIS on the robot: nothing. start_argv() is the deploy's restart while OKVIS is stopped
             (the pod comes up on the new robot.env).
ProducerCtl  the producer at the collection (or deploy) size under the lease (producer.Lease: the lease is taken before
             the first change, a live lease of another session refuses), through producer.resize's no-overlap cycle,
             and restore() = the preflight spec back as saved (producer.restore), the lease released only when that
             worked. A producer already at the asked size is not restarted.
OkvisCtl and ProducerCtl round-trip through as_dict / from_dict: a restarted calibd knows what is still changed."""
import json
import shlex
import time
from pathlib import Path

from calibtool import deploy as dp, producer as pr

OKVIS_NODE_LABEL = "foundation.bot/has-okvis"


class Recorder:
    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        self.entries = []           # (source, argv or None, input or None, note or None)
        self.path = None

    def attach(self, path):
        """write every entry to path as it is added (a crash loses nothing); an existing file is appended to"""
        self.path = Path(path)
        new = not self.path.is_file()
        with open(self.path, "a") as f:
            if new:
                f.write(self._head() + "\n")
            for ln in self._lines(self.entries):
                f.write(ln + "\n")
        return self.path

    def _append(self, entry):
        self.entries.append(entry)
        if self.path is not None:
            with open(self.path, "a") as f:
                f.write(self._lines([entry])[0] + "\n")
                f.flush()

    def add(self, source, argv, input=None):
        self._append((str(source), [str(a) for a in argv], input, None))

    def note(self, text, source="session"):
        self._append((str(source), None, None, str(text)))

    def hook(self, source):
        """a log callable for deploy.Remote(log=...)"""
        return lambda argv, input=None: self.add(source, argv, input)

    def sources(self) -> list:
        out = []
        for s, a, _, _ in self.entries:
            if a is not None and s not in out:
                out.append(s)
        return out

    def commands(self, source=None) -> list:
        return [a for s, a, _, _ in self.entries if a is not None and (source is None or s == source)]

    def lines(self) -> list:
        return self._lines(self.entries)

    @staticmethod
    def _lines(entries) -> list:
        out = []
        for s, a, inp, note in entries:
            if a is None:
                out.append(f"# {note}")
                continue
            ln = f"[{s}] {shlex.join(a)}"
            if inp is not None:
                ln += f"  # <<< {len(inp.encode())} bytes on stdin (sha256 {dp.sha12(inp)})"
            out.append(ln)
        return out

    def _head(self) -> str:
        return ("# DRY RUN: nothing below was sent to the robot; these are the commands the session would send, in "
                "order" if self.dry_run else "# the commands this session sent to the robot, in order")

    def text(self) -> str:
        return "\n".join([self._head()] + self.lines()) + "\n"

    def write(self, path):
        p = Path(path)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(self.text())
        tmp.replace(p)
        return p


# ---------------------------------------------------------------- the simulated producer


class DryKubectl:
    """producer.Kube's runner over a simulated producer deployment (module text). fail: verbs that fail (tests)"""

    MUTATING = ("scale", "patch", "replace")

    def __init__(self, yaml_text, recorder=None, fail=(), source="producer"):
        import yaml
        self.doc = yaml.safe_load(yaml_text)
        self.recorder, self.fail, self.source = recorder, set(fail), source
        self.replicas = int(self.doc["spec"].get("replicas", 1))
        self.pods = self.replicas          # pods existing now (they follow the replicas at the next listing)
        self.max_pods = self.pods
        self.changes = 0

    @property
    def args(self) -> list:
        return [str(a) for a in self.doc["spec"]["template"]["spec"]["containers"][0].get("args") or []]

    def __call__(self, argv, input=None):
        import yaml
        a = [str(x) for x in argv]
        verb = a[0] if a else ""
        if verb in self.fail:
            return 1, f"dry kubectl: {verb} failed (test)"
        if verb in self.MUTATING:
            self.changes += 1
            if self.recorder is not None:
                self.recorder.add(self.source, ["k0s", "kubectl", *a], input)
        if a[:2] == ["get", "deploy"]:
            d = dict(self.doc)
            d["spec"] = dict(d["spec"], replicas=self.replicas)
            return 0, yaml.safe_dump(d, sort_keys=False)
        if a[:2] == ["get", "pods"]:
            self.pods = self.replicas
            self.max_pods = max(self.max_pods, self.pods)
            items = [{"metadata": {"name": f"producer-{i}", "uid": f"u{i}"},
                      "status": {"phase": "Running", "containerStatuses": [{"ready": True, "restartCount": 0}]}}
                     for i in range(self.pods)]
            return 0, json.dumps({"items": items})
        if verb in ("logs", "get", "describe"):
            return 0, ""
        if verb == "scale":
            n = int(next(x for x in a if x.startswith("--replicas=")).split("=")[1])
            if n > 0 and self.pods > 0 and self.replicas == 0:
                return 1, "dry kubectl: scale up while a pod exists (would overlap)"
            self.replicas = n
            return 0, f"deployment.apps/{a[2]} scaled"
        if verb == "patch":
            body = json.loads(a[a.index("-p") + 1])
            for op in body:
                if op["path"].endswith("/args"):
                    self.doc["spec"]["template"]["spec"]["containers"][0]["args"] = list(op["value"])
            return 0, "patched"
        if verb == "replace":
            d = yaml.safe_load(input)
            self.doc = d
            self.replicas = int(d["spec"].get("replicas", 1))
            return 0, "replaced"
        return 1, f"dry kubectl: unknown {a[:3]}"


# ---------------------------------------------------------------- OKVIS


class OkvisCtl:
    """remote: deploy.Remote (its log is the session's Recorder); restart: deploy.DeploySnapshot.restart"""

    def __init__(self, remote: dp.Remote, restart: dict, node=None, label_value=None, stopped=False,
                 origin="calibtool"):
        self.remote, self.restart = remote, dict(restart or {})
        self.kind = self.restart.get("kind") or "none"
        self.node, self.label_value, self.stopped, self.origin = node, label_value, bool(stopped), origin

    def _call(self, argv, what):
        rc, out = self.remote.call(argv, mutating=True, timeout=120)
        if rc != 0:
            raise dp.RemoteError(f"OKVIS {what} failed (rc {rc}): {(out or '').strip()[:200]}")

    def _find_node(self):
        rc, out = self.remote.call(dp.kubectl("-n", dp.OKVIS_NS, "get", "pods", "-l", dp.OKVIS_LABEL, "-o",
                                              "jsonpath={.items[0].spec.nodeName}"))
        if rc == 0 and out.strip():
            return out.strip()
        rc, out = self.remote.call(["hostname"])
        return out.strip() if rc == 0 and out.strip() else None

    def _label_argv(self, value):
        return dp.kubectl("label", "node", self.node, f"{OKVIS_NODE_LABEL}={value}", "--overwrite")

    def stop(self):
        if self.kind == "none" or self.stopped:
            return
        if self.kind == "daemonset":
            self.node = self.node or self._find_node()
            if not self.node:
                raise dp.RemoteError("OKVIS stop: the node of the OKVIS pod is not known")
            rc, out = self.remote.call(dp.kubectl("get", "node", self.node, "-o",
                                                  "jsonpath={.metadata.labels.foundation\\.bot/has-okvis}"))
            self.label_value = out.strip() if rc == 0 and out.strip() else "true"
            self._call(self._label_argv("false"), "stop (node label)")
        else:
            self._call(["env", f"OKVIS_LAUNCH_ORIGIN={self.origin}", "bash", dp.OKVIS_SH, "stop"], "stop")
        self.stopped = True

    def start_argv(self) -> list:
        """the command that brings a stopped OKVIS back (the deploy's restart while it is stopped)"""
        if self.kind == "daemonset":
            return self._label_argv(self.label_value or "true")
        if self.kind == "okvis_sh":
            return ["env", f"OKVIS_LAUNCH_ORIGIN={self.origin}", "bash", dp.OKVIS_SH, "start"]
        return []

    def start(self):
        if self.kind == "none" or not self.stopped:
            return
        self._call(self.start_argv(), "start")
        self.stopped = False

    def as_dict(self) -> dict:
        return {"restart": self.restart, "node": self.node, "label_value": self.label_value, "stopped": self.stopped}

    @classmethod
    def from_dict(cls, remote, d):
        return cls(remote, d.get("restart") or {}, d.get("node"), d.get("label_value"), d.get("stopped", False))


# ---------------------------------------------------------------- the producer


class ProducerCtl:
    """kube: producer.Kube(read_only=False); spec: the preflight producer.Spec (the snapshot); lease: producer.Lease;
    spec_path: where the saved spec is kept for a stale-lease restore"""

    def __init__(self, kube, spec, lease, session, spec_path, clock=time.time, sleep=time.sleep, changed=False,
                 size=None, **cycle_kw):
        self.kube, self.spec, self.lease, self.session = kube, spec, lease, session
        self.spec_path = Path(spec_path)
        self.clock, self.sleep, self.cycle_kw = clock, sleep, cycle_kw
        self.changed = bool(changed)
        self.size = tuple(size) if size else tuple(spec.size())

    def _kw(self):
        return {"sleep": self.sleep, "clock": self.clock, **self.cycle_kw}

    def to_size(self, width, height, fps) -> bool:
        """-> True when the producer was restarted"""
        want = (int(width), int(height), int(fps))
        if tuple(self.size) == want:
            return False
        self.spec_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.spec_path.is_file():
            self.spec_path.write_text(self.spec.text)
        self.lease.acquire(self.session, self.spec_path, self.clock())
        self.changed = True             # from here on the restore is owed, whatever the cycle did
        self._own()
        try:
            pr.resize(self.kube, self.spec, *want, **self._kw())
        finally:
            self._own()
        self.size = want
        return True

    def _own(self):
        """the tool's own producer restart, before and after: health takes the next minute's camera USB drop and the
        generation change as the tool's own (10 Nov, phantom11: "someone else changed the producer", "head camera
        not on the USB bus" at every own restart)"""
        cb = getattr(self, "on_change", None)
        if cb is not None:
            cb()

    def heartbeat(self):
        if self.changed:
            self.lease.heartbeat(self.clock())

    def restore(self):
        if not self.changed:
            return False
        self._own()
        try:
            pr.restore(self.kube, self.spec, lease=self.lease, **self._kw())
        finally:
            self._own()
        self.changed = False
        self.size = tuple(self.spec.size())
        return True

    def as_dict(self) -> dict:
        return {"changed": self.changed, "size": list(self.size)}

    @classmethod
    def from_dict(cls, d, kube, lease, session, spec_path, **kw):
        spec = pr.Spec.from_yaml(Path(spec_path).read_text())
        return cls(kube, spec, lease, session, spec_path, changed=d.get("changed", False), size=d.get("size"), **kw)
