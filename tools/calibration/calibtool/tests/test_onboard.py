"""calibtool.onboard: a new robot set up for calibtool with one confirmation (10 Nov 2026: phantom11 / P11 was done by
hand). Every robot command goes through an injected runner; here a fake network of robots (no ssh). The repo files are
copies in tmp_path; the real robots.yaml / profiles.yaml / rig files are only read."""
import hashlib
import re
import shutil
from pathlib import Path

import pytest

pytest.importorskip("yaml")
import yaml  # noqa: E402

from calibtool import onboard as ob, paths, profile, robots  # noqa: E402

FIXED = ob.FIXED_MOTOR_TOOL_MD5
OLD_TOOL = "0" * 32
P10_MID = "60776b088362" + "0123456789abcdef0123"
P11_MID = "dfaae203c0d0" + "aaaabbbbccccdddd0000"
NEW_MID = "abcdef012345" + "99998888777766665555"
P10_IP, P11_IP, NEW_IP = "100.64.0.10", "100.64.0.11", "100.64.0.13"

HEAD_JSON = """{"boards": {"0": {"mxid": "1944301091064E2E00"}},
 "cameras": {"ego_right": {"queue_id": 0, "board": 0, "socket": "C"},
             "ego_left": {"queue_id": 3, "board": 0, "socket": "B"},
             "right_wrist": {"queue_id": 1, "board": 0}}}"""


def okvis_cfg(f0=(565.0, 564.8), f1=(565.2, 564.6), t0=(-0.0536, 0.1374, 0.0255), t1=(-0.0093, 0.1371, 0.0246),
              size=(1280, 800)):
    def cam(f, t, q):
        return (f"    - {{T_SC:\n        [1.0, 0.0, 0.0, {t[0]},\n          0.0, 1.0, 0.0, {t[1]},  # a comment\n"
                f"          0.0, 0.0, 1.0, {t[2]},\n          0.0, 0.0, 0.0, 1.0],\n"
                f"        image_dimension: [{size[0]}, {size[1]}],\n"
                f"        distortion_coefficients: [0.1, 0.2, 0.3, 0.4, 0.0, 0.0],\n        distortion_type: kb4t,\n"
                f"        focal_length: [{f[0]}, {f[1]}],\n        principal_point: [640.0, 400.0],\n"
                f"        dma_queue: {q}}} # cam\n\n")
    return ("%YAML:1.0\n# header: focal_length: [1.0, 1.0] in a comment must not count\ncameras:\n" + cam(f0, t0, 3)
            + cam(f1, t1, 0) + "imu_parameters:\n    T_BS: [1.0, 0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, "
            "0.0, 0.0, 0.0, 0.0, 1.0]\n")


STANDIN = "/etc/phantomos/okvis/phantom07_stereo_dense_thor-STANDIN-1280x800-15fps.yaml"
OLDER = "/etc/phantomos/okvis/phantom07_stereo_old.yaml"
ENV_COMMENTED = "OKVIS_DMA_CAMERAS=3,0\n# use image config\n#OKVIS_CALIBRATION=/etc/okvis/phantom07_stereo_old.yaml\n"
PODS = ("dma-video   producer-7f9c   1/1 Running 0 2d\n"
        "positronic  okvis2x-abcde   1/1 Running 3 1d\n"
        "phantom     ghost-wbc-walker-55d   1/1 Running 0 3h\n")


class FakeRobot:
    def __init__(self, hostname, mid, files=None, dirs=(), pythons=(), dma=(), tools=None, mtimes=None, pods=PODS,
                 ethercat="active", ping="PONG uptime=6065.3s motor_state=0"):
        self.hostname, self.mid = hostname, mid
        self.files = dict(files or {})
        self.dirs = set(dirs)
        self.pythons = set(pythons)         # interpreters importing numpy, scipy, cv2, yaml
        self.dma = set(dma)                 # folders holding dma_ipc
        self.tools = dict(tools or {})      # motor tool path -> md5
        self.mtimes = list(mtimes or [])    # *.yaml in /etc/phantomos/okvis, newest first
        self.pods, self.ethercat, self.ping = pods, ethercat, ping
        self.writes = []

    def handle(self, cmd, net):
        m = re.fullmatch(r"cat (\S+)", cmd)
        if m:
            p = m.group(1).strip("'")
            if p == "/etc/machine-id":
                return 0, self.mid + "\n"
            return (0, self.files[p]) if p in self.files else (1, f"cat: {p}: No such file or directory")
        if cmd == "hostname":
            return 0, self.hostname + "\n"
        if cmd.startswith("ls -1t /etc/phantomos/okvis/"):
            return (0, "".join(p + "\n" for p in self.mtimes)) if self.mtimes else (2, "ls: no match")
        if cmd.startswith("ls -1d /root/*/venv/bin/python"):
            hits = sorted(p for p in self.pythons | {p for p in self.files if p.endswith("/venv/bin/python")}
                          if re.fullmatch(r"/root/[^/]+/venv/bin/python", p))
            return (0, "".join(h + "\n" for h in hits)) if hits else (2, "")
        m = re.match(r"ls -1d (\S+)/dma_ipc\*", cmd)
        if m:
            pat = m.group(1)
            hits = sorted(d for d in self.dma if re.fullmatch(pat.replace("*", "[^/]+"), d))
            return (0, "".join(f"{h}/dma_ipc.cpython-312-aarch64-linux-gnu.so\n" for h in hits)) if hits else (2, "")
        m = re.fullmatch(r"(?:PYTHONPATH=(\S+) )?(\S+) -c '(import [^;]+); cv2\.aruco\.ArucoDetector'", cmd)
        if m:
            pp, py, imp = m.groups()
            ok = py in self.pythons and ("dma_ipc" not in imp or pp in self.dma)
            return (0, "") if ok else (1, "ModuleNotFoundError: No module named 'dma_ipc'" if py in self.pythons
                                       else f"sh: {py}: not found")
        if cmd.startswith("md5sum "):
            out = "".join(f"{md5}  {p}\n" for p, md5 in sorted(self.tools.items()))
            return (0 if out else 1), out
        m = re.fullmatch(r"test -[de] (\S+)", cmd)
        if m:
            p = m.group(1)
            return (0 if p in self.dirs or p in self.files or p in self.tools or p in self.dma else 1), ""
        if cmd.startswith("mkdir -p "):
            self.writes.append(cmd)
            self.dirs.update(cmd.split()[2:])
            return 0, ""
        m = re.fullmatch(r"cp -a (\S+)/dma_ipc\* (\S+)/", cmd)
        if m:
            self.writes.append(cmd)
            if m.group(1) not in self.dma:
                return 1, "cp: no such file"
            self.dma.add(m.group(2))
            return 0, ""
        if "get deployment producer" in cmd:
            return 0, "deployment.apps/producer\n"
        if "get pods -A --no-headers" in cmd:
            return 0, self.pods
        if cmd == "systemctl is-active dma-ethercat":
            return (0 if self.ethercat == "active" else 3), self.ethercat + "\n"
        if cmd == "dma_cmd_client ping":
            return 0, self.ping + "\n"
        m = re.fullmatch(r"ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 root@(\S+) "
                         r"cat /etc/machine-id", cmd)
        if m:
            return 0, net.robots[net.seen_from(self, m.group(1))].mid + "\n"
        m = re.fullmatch(r"rsync -a -e 'ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new' (.+) root@(\S+):"
                         r"/root/calib-kit/", cmd)
        if m:
            self.writes.append(cmd)
            dst = net.robots[net.seen_from(self, m.group(2))]
            for src in m.group(1).split():
                name = src.rstrip("/").split("/")[-1]
                if src in self.tools:
                    dst.tools[f"/root/calib-kit/{name}"] = self.tools[src]
                elif any(p.startswith(src + "/") for p in self.pythons):
                    dst.pythons.add(f"/root/calib-kit/{name}/bin/python")
                else:
                    return 23, f"rsync: {src}: no such file"
            dst.writes.append(f"(from {self.hostname}) {cmd}")
            return 0, ""
        raise AssertionError(f"{self.hostname}: unexpected command {cmd!r}")


class FakeNet:
    """robots by IP; resolve() is the laptop's name lookup; a robot's own lookup of an IP is the IP unless rerouted"""

    def __init__(self, robots_by_ip, names):
        self.robots, self.names = robots_by_ip, names
        self.reroute = {}                   # (robot hostname, ip) -> ip that robot actually reaches
        self.calls = []

    def seen_from(self, robot, ip):
        return self.reroute.get((robot.hostname, ip), ip)

    def resolve(self, host):
        return self.names[host]

    def run(self, host, cmd, timeout=None, agent=False):
        user, ip = host.split("@")
        assert user == "root" and re.fullmatch(r"[\d.]+", ip), f"commands go to the resolved IP, not {host}"
        self.calls.append((ip, cmd, agent))
        return self.robots[ip].handle(cmd, self)


def p10_robot():
    return FakeRobot("phantom10", P10_MID, dirs={"/data/calib", "/root/p11-cal/pydeps"},
                     pythons={"/root/p11-cal/venv/bin/python"}, dma={"/root/p11-cal/pydeps"},
                     tools={"/root/p11-cal/dma_motor_motion": FIXED, "/usr/bin/dma_motor_motion": OLD_TOOL})


def p11_like(hostname="phantom13", mid=NEW_MID, env=ENV_COMMENTED):
    """a robot as phantom11 was on 10 Nov: no kit, its own dma_ipc in /root/burst-grab/pydeps, the old motor tool in
    /usr/bin, robot.env without OKVIS_CALIBRATION, two OKVIS configs (the stand-in newest)"""
    files = {"/etc/phantom/head_camera.json": HEAD_JSON, STANDIN: okvis_cfg(), OLDER: okvis_cfg(f0=(500, 500))}
    if env is not None:
        files["/etc/phantomos/okvis/robot.env"] = env
    return FakeRobot(hostname, mid, files=files, dirs={"/root/burst-grab/pydeps"}, dma={"/root/burst-grab/pydeps"},
                     tools={"/usr/bin/dma_motor_motion": OLD_TOOL}, mtimes=[STANDIN, OLDER])


ROBOTS_YAML = """# the registry (a comment that must survive)
robots:
  P10:
    thor: phantom10
    old_names: [phantom11]
    machine_id_prefix: "60776b088362"
    rig: rig-phantom11.yaml
  P12:
    thor: phantom12             # rename pending
    old_names: [phantom21]
    machine_id_prefix: "9851aca02a7d"
    rig: rig-phantom21.yaml
# trailing comment that must survive
"""
PROFILES_YAML = """# calibd runtime per robot (a comment that must survive)
profiles:
  P10:
    python: /root/p11-cal/venv/bin/python
    pydeps: /root/p11-cal/pydeps
    root: /data/calib          # a comment that must survive
    left: 3
    right: 0
    producer: {namespace: dma-video, deployment: producer}
    motor_tool: /root/p11-cal/dma_motor_motion
    stand: {raise_max_m: 0.30, lower_max_m: 0.40, auto_skip: false, steps: true}
    placement_timeout_s: null
"""


def repo(tmp_path, robots_text=ROBOTS_YAML, profiles_text=PROFILES_YAML):
    cfg = tmp_path / "tools/calibration/calibtool/config"
    cfg.mkdir(parents=True)
    (cfg / "robots.yaml").write_text(robots_text)
    (cfg / "profiles.yaml").write_text(profiles_text)
    rigs = tmp_path / "tools/calibration/config"
    rigs.mkdir(parents=True)
    shutil.copy(paths.CONFIG / "rig-phantom11.yaml", rigs / "rig-phantom11.yaml")
    return tmp_path


def net_for(target_ip, target, **names):
    n = FakeNet({P10_IP: p10_robot(), target_ip: target}, {"phantom10": P10_IP, **names})
    return n


class Asker:
    def __init__(self, *answers):
        self.answers, self.prompts = list(answers), []

    def __call__(self, prompt=""):
        self.prompts.append(prompt)
        if not self.answers:
            raise AssertionError(f"unexpected question: {prompt!r}")
        return self.answers.pop(0)


def go(net, host, root, ask, **kw):
    lines = []
    res = ob.run(host, run_cmd=net.run, resolve=net.resolve, ask=ask, out=lines.append, repo_root=root,
                 today="10 Nov 2026", **kw)
    return res, "\n".join(lines)


def files_of(root):
    return {p.relative_to(root).as_posix(): p.read_text() for p in sorted(root.rglob("*")) if p.is_file()}


# ---------------------------------------------------------------- pure helpers

def test_nominal_from_the_live_okvis_config():
    f, b, size = ob.nominal_from_okvis(okvis_cfg())
    assert size == (1280, 800)
    assert f == pytest.approx((565.0 + 564.8 + 565.2 + 564.6) / 4)
    assert b == pytest.approx(((0.0536 - 0.0093) ** 2 + 0.0003 ** 2 + 0.0009 ** 2) ** 0.5)


def test_nominal_focal_is_brought_to_the_rig_width():
    f, _, size = ob.nominal_from_okvis(okvis_cfg(f0=(423.75, 423.75), f1=(423.75, 423.75), size=(960, 600)), 1280)
    assert size == (960, 600) and f == pytest.approx(565.0)


def test_baseline_is_the_stereo_translation_whatever_the_rotation():
    """|inv(T_SC0) T_SC1 translation| = |t1 - t0| for any rotation of camera 0"""
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(3)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    R0 = q * np.sign(np.linalg.det(q))
    T0, T1 = np.eye(4), np.eye(4)
    T0[:3, :3], T0[:3, 3] = R0, [0.1, 0.2, 0.3]
    T1[:3, 3] = [0.14, 0.21, 0.29]
    want = np.linalg.norm((np.linalg.inv(T0) @ T1)[:3, 3])
    text = okvis_cfg().replace("[1.0, 0.0, 0.0, -0.0536,\n          0.0, 1.0, 0.0, 0.1374,  # a comment\n"
                               "          0.0, 0.0, 1.0, 0.0255,", "[" + ", ".join(
                                   repr(float(x)) for x in T0[:3].ravel()) + ",", 1)
    text = text.replace("-0.0093,", "0.14,").replace("0.1371,", "0.21,").replace("0.0246,", "0.29,")
    assert ob.nominal_from_okvis(text)[1] == pytest.approx(want)


@pytest.mark.parametrize("env,want", [
    ("OKVIS_CALIBRATION=/etc/okvis/a.yaml\n", "/etc/phantomos/okvis/a.yaml"),
    ('X=1\nOKVIS_CALIBRATION="/etc/okvis/b.yaml"  # live\n', "/etc/phantomos/okvis/b.yaml"),
    ("export OKVIS_CALIBRATION=/etc/okvis/c.yaml\n", "/etc/phantomos/okvis/c.yaml"),
    (ENV_COMMENTED, None),
    ("  # OKVIS_CALIBRATION=/etc/okvis/a.yaml\n", None),
    ("OKVIS_CALIBRATION=\n", None),
    (None, None),
])
def test_robot_env_calibration(env, want):
    assert ob.env_calibration(env) == want


def test_head_cameras():
    c = ob.head_cameras(HEAD_JSON)
    assert (c.left, c.right, c.left_socket, c.right_socket, c.mxid) == (3, 0, "B", "C", "1944301091064E2E00")
    with pytest.raises(ValueError, match="ego_left"):
        ob.head_cameras('{"cameras": {"ego_right": {"queue_id": 0}}}')


def test_insert_keeps_every_comment_and_lands_inside_the_block():
    new = ob.insert_entry(ROBOTS_YAML, "robots", "  P13:\n    thor: phantom13\n")
    assert new.startswith("# the registry (a comment that must survive)\n")
    assert "    thor: phantom12             # rename pending\n" in new
    assert new.rstrip().endswith("# trailing comment that must survive")
    assert yaml.safe_load(new)["robots"]["P13"] == {"thor": "phantom13"}
    two = "a:\n  x: 1\n# about b\nb:\n  y: 2\n"
    got = ob.insert_entry(two, "a", "  z: 3\n")
    assert got == "a:\n  x: 1\n  z: 3\n# about b\nb:\n  y: 2\n"
    with pytest.raises(ValueError):
        ob.insert_entry("other:\n  x: 1\n", "robots", "  P13: {}\n")


def test_policy_patterns_match_preflight():
    pf = pytest.importorskip("calibtool.preflight")
    assert ob.POLICY_PATTERNS == pf.POLICY_PATTERNS


# ---------------------------------------------------------------- onboarding a robot like phantom11 on 10 Nov

def test_onboards_a_robot_like_phantom11(tmp_path):
    root = repo(tmp_path)
    tgt = p11_like()
    net = net_for(NEW_IP, tgt, phantom13=NEW_IP)
    ask = Asker("1", "Y")
    res, text = go(net, "phantom13", root, ask, old_names=["phantom09"])
    assert res.status == "done", text
    # the live config: robot.env names none, the operator picked the newest
    assert "newest first" in ask.prompts[0] or "newest first" in text
    assert STANDIN.split("/")[-1] in text
    # robot side: the kit copied from P10 by IP, dma_ipc from its own pydeps, /data/calib made
    assert any("rsync" in w and f"root@{NEW_IP}:/root/calib-kit/" in w for w in net.robots[P10_IP].writes)
    assert "/root/calib-kit/venv/bin/python" in tgt.pythons
    assert tgt.tools["/root/calib-kit/dma_motor_motion"] == FIXED
    assert "/root/calib-kit/pydeps" in tgt.dma and "/data/calib" in tgt.dirs
    # the rsync ran with the agent forwarded, on the reference robot, after it read the target's machine-id
    calls = net.calls
    i_rsync = next(i for i, c in enumerate(calls) if "rsync" in c[1])
    assert calls[i_rsync][0] == P10_IP and calls[i_rsync][2] is True
    assert any(c[0] == P10_IP and f"root@{NEW_IP} cat /etc/machine-id" in c[1] for c in calls[:i_rsync])
    # every robot write is fenced by a machine-id read on the target before and after it
    for i, c in enumerate(calls):
        if c[1].startswith(("mkdir", "cp -a", "rsync")):
            assert calls[i - 1][1] in ("cat /etc/machine-id",) or "cat /etc/machine-id" in calls[i - 1][1]
            assert calls[i + 1] == (NEW_IP, "cat /etc/machine-id", False)
    # the kit import test
    assert (NEW_IP, "PYTHONPATH=/root/calib-kit/pydeps /root/calib-kit/venv/bin/python -c 'import numpy, scipy, "
            "cv2, yaml, dma_ipc; cv2.aruco.ArucoDetector'", False) in calls
    # repo: registry, profile and rig file
    reg = robots.load(root / "tools/calibration/calibtool/config/robots.yaml")
    r = reg["P13"]
    assert (r.thor, r.old_names, r.machine_id_prefix, r.rig) == ("phantom13", ("phantom09",), NEW_MID[:12],
                                                                  "rig-p13.yaml")
    prof = profile.load("P13", root / "tools/calibration/calibtool/config/profiles.yaml", reg)
    assert (prof.python, prof.pydeps, prof.root, prof.left, prof.right) == (
        "/root/calib-kit/venv/bin/python", "/root/calib-kit/pydeps", "/data/calib", 3, 0)
    assert (prof.producer_ns, prof.producer_deploy, prof.motor_tool) == ("dma-video", "producer",
                                                                          "/root/calib-kit/dma_motor_motion")
    assert (prof.stand_raise_m, prof.stand_lower_m, prof.stand_auto_skip, prof.stand_steps) == (0.30, 0.40, False, True)
    assert prof.placement_timeout_s is None and prof.live_config == STANDIN and prof.neck_kp == 60.0
    rig = yaml.safe_load((root / "tools/calibration/config/rig-p13.yaml").read_text())
    assert rig["rig_name"] == "p13"
    assert rig["okvis_yaml_name"] == "phantom13_p13_stereo_kb4t-1280x800.yaml"
    assert rig["cameras"]["nominal_focal_px"] == 565.0 and rig["cameras"]["nominal_baseline_m"] == 0.0443
    assert rig["cameras"]["kalibr"] == [{"record_dir": "cam3", "dma_queue": 3, "side": "left"},
                                        {"record_dir": "cam0", "dma_queue": 0, "side": "right"}]
    assert rig["robot"]["host"] == "root@phantom13"
    tmpl = yaml.safe_load((paths.CONFIG / "rig-phantom11.yaml").read_text())
    for k in ("gates", "capture", "imu", "target"):
        assert rig[k] == tmpl[k]
    rig_text = (root / "tools/calibration/config/rig-p13.yaml").read_text()
    assert rig_text.startswith("# rig-p13.yaml -- P13") and "Margaret" not in rig_text
    # comments kept in both YAML files
    for f in ("robots.yaml", "profiles.yaml"):
        t = (root / "tools/calibration/calibtool/config" / f).read_text()
        assert "a comment that must survive" in t
    # the checks at the end and the next step
    assert "ghost-wbc-walker-55d" in text and "operator" in text
    assert "dma-ethercat active" in text and "PONG" in text and "okvis2x-abcde" in text
    assert text.rstrip().splitlines()[-1] == "Next: start a calibration: calib, choose phantom13"


def test_the_summary_lists_every_change_before_asking(tmp_path):
    root = repo(tmp_path)
    net = net_for(NEW_IP, p11_like(), phantom13=NEW_IP)
    lines = []

    def ask(prompt=""):
        lines.append("<ASK>")
        return "n" if "Y" in prompt else "1"
    res = ob.run("phantom13", run_cmd=net.run, resolve=net.resolve, ask=ask, out=lines.append, repo_root=root,
                 today="10 Nov 2026")
    text = "\n".join(lines)
    summary = text[:text.rindex("<ASK>")]
    for want in ("/root/calib-kit/pydeps", "/data/calib", "phantom10", "/root/p11-cal/venv",
                 "/root/p11-cal/dma_motor_motion", "/root/burst-grab/pydeps", "rig-p13.yaml", "robots.yaml",
                 "profiles.yaml", NEW_MID[:12], NEW_IP):
        assert want in summary, want
    assert res.status == "declined"
    assert not net.robots[NEW_IP].writes and not net.robots[P10_IP].writes
    assert files_of(root) == files_of(repo(tmp_path / "fresh"))


def test_dry_run_changes_nothing_and_does_not_ask_for_y(tmp_path):
    root = repo(tmp_path)
    before = files_of(root)
    net = net_for(NEW_IP, p11_like(), phantom13=NEW_IP)
    ask = Asker("1")                             # the live config pick only (read-only)
    res, text = go(net, "phantom13", root, ask, dry_run=True)
    assert res.status == "dry-run" and "nothing changed" in text.lower()
    assert files_of(root) == before
    assert not net.robots[NEW_IP].writes and not net.robots[P10_IP].writes
    assert not any(c[1].startswith(("mkdir", "cp ", "rsync")) or "rsync" in c[1] for c in net.calls)


@pytest.mark.parametrize("answer", ["", "n", "yes please", "y "])
def test_only_y_applies(tmp_path, answer):
    root = repo(tmp_path)
    net = net_for(NEW_IP, p11_like(), phantom13=NEW_IP)
    res, _ = go(net, "phantom13", root, Asker("1", answer))
    if answer.strip() == "y":
        assert res.status == "done"
    else:
        assert res.status == "declined" and not net.robots[NEW_IP].writes


def test_robot_env_names_the_live_config_no_question_no_live_config_in_the_profile(tmp_path):
    root = repo(tmp_path)
    tgt = p11_like(env="OKVIS_CALIBRATION=/etc/okvis/phantom07_stereo_old.yaml\n")
    net = net_for(NEW_IP, tgt, phantom13=NEW_IP)
    res, text = go(net, "phantom13", root, Asker("Y"))
    assert res.status == "done", text
    raw = yaml.safe_load((root / "tools/calibration/calibtool/config/profiles.yaml").read_text())["profiles"]["P13"]
    assert "live_config" not in raw
    rig = yaml.safe_load((root / "tools/calibration/config/rig-p13.yaml").read_text())
    assert rig["cameras"]["nominal_focal_px"] == pytest.approx(round((500 + 500 + 565.2 + 564.6) / 4))


def test_a_bad_pick_is_asked_again_then_stops(tmp_path):
    root = repo(tmp_path)
    net = net_for(NEW_IP, p11_like(), phantom13=NEW_IP)
    res, _ = go(net, "phantom13", root, Asker("9", "x", ""))
    assert res.status == "stopped" and not net.robots[NEW_IP].writes
    res, _ = go(net, "phantom13", root, Asker("0", "2", "n"))
    assert res.status == "declined" and res.live_config == OLDER


def test_a_robot_with_a_usable_python_keeps_it(tmp_path):
    root = repo(tmp_path)
    tgt = p11_like()
    tgt.pythons.add("/root/burst-grab/venv/bin/python")
    tgt.tools["/root/burst-grab/dma_motor_motion"] = FIXED
    net = net_for(NEW_IP, tgt, phantom13=NEW_IP)
    res, text = go(net, "phantom13", root, Asker("1", "Y"))
    assert res.status == "done", text
    assert not net.robots[P10_IP].writes                     # nothing copied robot to robot
    assert not any(c[0] == P10_IP for c in net.calls)
    prof = profile.load("P13", root / "tools/calibration/calibtool/config/profiles.yaml",
                        robots.load(root / "tools/calibration/calibtool/config/robots.yaml"))
    assert prof.python == "/root/burst-grab/venv/bin/python" and prof.motor_tool == "/root/burst-grab/dma_motor_motion"
    assert prof.pydeps == "/root/calib-kit/pydeps"


# ---------------------------------------------------------------- identity: the 10 Nov name trap

def test_a_name_that_resolves_to_another_robot_is_refused(tmp_path):
    """10 Nov: "phantom11" resolved to phantom10 itself (an old /etc/hosts name); nothing may be written there"""
    root = repo(tmp_path)
    before = files_of(root)
    net = FakeNet({P10_IP: p10_robot()}, {"phantom10": P10_IP, "phantom11": P10_IP})
    res, text = go(net, "phantom11", root, Asker())
    assert res.status == "refused"
    assert "phantom10" in text and "P10" in text and P10_IP in text
    assert files_of(root) == before and not net.robots[P10_IP].writes


def test_a_name_that_resolves_to_an_unknown_robot_with_another_hostname_is_refused(tmp_path):
    root = repo(tmp_path)
    net = net_for(NEW_IP, p11_like(hostname="phantom14"), phantom13=NEW_IP)
    res, text = go(net, "phantom13", root, Asker())
    assert res.status == "refused" and "phantom14" in text and not net.robots[NEW_IP].writes


def test_the_reference_robot_must_reach_the_same_machine(tmp_path):
    """the copy runs on the reference robot: it must see the target's machine-id at the IP, else no copy"""
    root = repo(tmp_path)
    tgt = p11_like()
    other = FakeRobot("phantom99", "0" * 32)
    net = FakeNet({P10_IP: p10_robot(), NEW_IP: tgt, "100.64.0.99": other}, {"phantom10": P10_IP,
                                                                           "phantom13": NEW_IP})
    net.reroute[("phantom10", NEW_IP)] = "100.64.0.99"
    res, text = go(net, "phantom13", root, Asker("1", "Y"))
    assert res.status == "failed" and "machine-id" in text
    assert not any("rsync" in c[1] for c in net.calls)
    assert not (root / "tools/calibration/config/rig-p13.yaml").exists()
    assert "P13" not in (root / "tools/calibration/calibtool/config/robots.yaml").read_text()


def test_the_reference_robot_is_checked_by_its_machine_id(tmp_path):
    root = repo(tmp_path)
    net = net_for(NEW_IP, p11_like(), phantom13=NEW_IP)
    net.robots[P10_IP].mid = "1" * 32
    res, text = go(net, "phantom13", root, Asker("1", "Y"))
    assert res.status == "failed" and "P10" in text
    assert not any("rsync" in c[1] for c in net.calls)


def test_a_machine_id_change_between_probe_and_write_stops(tmp_path):
    root = repo(tmp_path)
    tgt = p11_like()
    net = net_for(NEW_IP, tgt, phantom13=NEW_IP)

    def ask(prompt=""):
        if "Y" in prompt:
            tgt.mid = "2" * 32               # another machine answers at that IP now
            return "Y"
        return "1"
    lines = []
    res = ob.run("phantom13", run_cmd=net.run, resolve=net.resolve, ask=ask, out=lines.append, repo_root=root,
                 today="10 Nov 2026")
    assert res.status == "failed" and "machine-id" in "\n".join(lines)
    assert not tgt.writes and "P13" not in (root / "tools/calibration/calibtool/config/robots.yaml").read_text()


def test_a_bot_name_already_taken_by_another_machine_is_refused(tmp_path):
    root = repo(tmp_path)
    net = net_for(NEW_IP, p11_like(hostname="phantom13"), phantom13=NEW_IP)
    res, text = go(net, "phantom13", root, Asker(), bot="P10")
    assert res.status == "refused" and "P10" in text


def test_a_host_name_without_a_number_needs_a_bot_name(tmp_path):
    root = repo(tmp_path)
    net = net_for(NEW_IP, p11_like(hostname="thor-lab"), **{"thor-lab": NEW_IP})
    res, text = go(net, "thor-lab", root, Asker())
    assert res.status == "refused" and "--bot" in text
    res, text = go(net, "thor-lab", root, Asker("1", "Y"), bot="P13")
    assert res.status == "done", text


# ---------------------------------------------------------------- idempotent: an onboarded robot

def p11_onboarded():
    t = p11_like(hostname="phantom11", mid=P11_MID)
    t.pythons.add("/root/calib-kit/venv/bin/python")
    t.dma.add("/root/calib-kit/pydeps")
    t.tools["/root/calib-kit/dma_motor_motion"] = FIXED
    t.dirs.add("/data/calib")
    t.files["/etc/phantomos/okvis/phantom07_stereo_dense_thor-STANDIN-p12fitc-refitA-yawA2-1280x800-15fps.yaml"] = \
        okvis_cfg()
    return t


def real_repo(tmp_path):
    """this branch's robots.yaml / profiles.yaml / rig files (P11 onboarded by hand on 10 Nov), as copies"""
    root = tmp_path
    (root / "tools/calibration/calibtool").mkdir(parents=True)
    shutil.copytree(paths.PKG / "config", root / "tools/calibration/calibtool/config")
    shutil.copytree(paths.CONFIG, root / "tools/calibration/config")
    return root


def test_rerun_on_the_hand_onboarded_p11_changes_nothing(tmp_path):
    root = real_repo(tmp_path)
    before = files_of(root)
    tgt = p11_onboarded()
    net = net_for(P11_IP, tgt, phantom11=P11_IP)
    res, text = go(net, "phantom11", root, Asker())         # no question at all
    assert res.status == "nothing-to-do", text
    assert "already" in text and "P11" in text
    assert files_of(root) == before and not tgt.writes and not net.robots[P10_IP].writes
    assert text.rstrip().splitlines()[-1] == "Next: start a calibration: calib, choose phantom11"


def test_rerun_after_onboarding_changes_nothing(tmp_path):
    root = repo(tmp_path)
    tgt = p11_like()
    net = net_for(NEW_IP, tgt, phantom13=NEW_IP)
    assert go(net, "phantom13", root, Asker("1", "Y"))[0].status == "done"
    before, nw = files_of(root), (len(tgt.writes), len(net.robots[P10_IP].writes))
    res, text = go(net, "phantom13", root, Asker())
    assert res.status == "nothing-to-do", text
    assert files_of(root) == before and (len(tgt.writes), len(net.robots[P10_IP].writes)) == nw


def test_a_registered_robot_whose_kit_went_missing_gets_only_the_kit_back(tmp_path):
    root = repo(tmp_path)
    tgt = p11_like()
    net = net_for(NEW_IP, tgt, phantom13=NEW_IP)
    assert go(net, "phantom13", root, Asker("1", "Y"))[0].status == "done"
    before = files_of(root)
    tgt.pythons.discard("/root/calib-kit/venv/bin/python")
    tgt.dirs.discard("/data/calib")
    res, text = go(net, "phantom13", root, Asker("Y"))
    assert res.status == "done", text
    assert files_of(root) == before                          # no repo change
    assert "/root/calib-kit/venv/bin/python" in tgt.pythons and "/data/calib" in tgt.dirs


def test_no_dma_ipc_on_the_robot_blocks(tmp_path):
    root = repo(tmp_path)
    tgt = p11_like()
    tgt.dma.clear()
    net = net_for(NEW_IP, tgt, phantom13=NEW_IP)
    res, text = go(net, "phantom13", root, Asker("1"))
    assert res.status == "blocked" and "dma_ipc" in text and not tgt.writes


def test_no_okvis_config_on_the_robot_blocks(tmp_path):
    root = repo(tmp_path)
    tgt = p11_like()
    tgt.mtimes = []
    net = net_for(NEW_IP, tgt, phantom13=NEW_IP)
    res, text = go(net, "phantom13", root, Asker())
    assert res.status == "blocked" and "OKVIS" in text


def test_a_failed_kit_import_stops_before_the_repo_files(tmp_path):
    root = repo(tmp_path)
    before = files_of(root)
    tgt = p11_like()
    net = net_for(NEW_IP, tgt, phantom13=NEW_IP)
    orig = tgt.handle

    def broken(cmd, n):
        if cmd.startswith("cp -a"):
            tgt.writes.append(cmd)
            return 0, ""                     # "copied", but dma_ipc does not import
        return orig(cmd, n)
    tgt.handle = broken
    res, text = go(net, "phantom13", root, Asker("1", "Y"))
    assert res.status == "failed" and "dma_ipc" in text
    assert files_of(root) == before


# ---------------------------------------------------------------- the end checks: read-only

def test_the_end_checks_never_touch_loco(tmp_path):
    root = repo(tmp_path)
    net = net_for(NEW_IP, p11_like(), phantom13=NEW_IP)
    go(net, "phantom13", root, Asker("1", "Y"))
    for _, cmd, _ in net.calls:
        assert not re.search(r"\b(delete|scale|rollout|restart|stop|kill)\b", cmd), cmd


def test_end_checks_report_master_down_and_no_okvis(tmp_path):
    root = repo(tmp_path)
    tgt = p11_like()
    tgt.pods, tgt.ethercat, tgt.ping = "dma-video producer-1 1/1 Running 0 1d\n", "inactive", "error: no master"
    net = net_for(NEW_IP, tgt, phantom13=NEW_IP)
    res, text = go(net, "phantom13", root, Asker("1", "Y"))
    assert res.status == "done"
    assert "inactive" in text and "no master" in text and "no okvis2x pod" in text and "no loco" in text


# ---------------------------------------------------------------- entry point

def test_main_dry_run(tmp_path, monkeypatch, capsys):
    root = repo(tmp_path)
    net = net_for(NEW_IP, p11_like(), phantom13=NEW_IP)
    monkeypatch.setattr("builtins.input", Asker("1"))
    rc = ob.main(["phantom13", "--dry-run", "--old-name", "phantom09"], run_cmd=net.run, resolve=net.resolve,
                 repo_root=root)
    assert rc == 0 and "nothing changed" in capsys.readouterr().out.lower()
    assert not net.robots[NEW_IP].writes


def test_the_default_runner_is_ssh_with_batch_mode(monkeypatch):
    seen = {}

    class R:
        returncode, stdout, stderr = 0, "ok\n", ""

    def fake_run(argv, **kw):
        seen["argv"] = argv
        return R()
    monkeypatch.setattr(ob.subprocess, "run", fake_run)
    assert ob.ssh_run("root@1.2.3.4", "hostname", agent=True) == (0, "ok\n")
    a = seen["argv"]
    assert a[0] == "ssh" and "BatchMode=yes" in a and "-A" in a and a[-2:] == ["root@1.2.3.4", "hostname"]
    ob.ssh_run("root@1.2.3.4", "hostname")
    assert "-A" not in seen["argv"]


def test_fixed_motor_tool_md5():
    assert FIXED == "208231a29c88cc2385971a0b73e02925"
    assert hashlib.md5(b"").hexdigest() != FIXED


def test_the_machine_id_is_read_through_ssh_warnings():
    """10 Nov, first real dry run: ssh's 'Permanently added ... to the list of known hosts' came with the output"""
    o = ob.Onboard.__new__(ob.Onboard)
    o.ip = "100.68.204.20"
    o.sh = lambda cmd, ip=None, **kw: (0, "dfaae203c0d04ae39e8dadef9efae3c5\nWarning: Permanently added "
                                          "'100.68.204.20' (ED25519) to the list of known hosts.\n")
    assert o.machine_id() == "dfaae203c0d04ae39e8dadef9efae3c5"
