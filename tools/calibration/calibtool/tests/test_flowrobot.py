"""calibtool.flowrobot: the full session's robot wiring (flow.Robot from calibd's pieces). Built against fakes only:
a dry run picks the recording backend for every mutating part (Enabler(dry_run), neck.DryRun, sessionctl.DryKubectl,
deploy.Remote(dry_run) in the flow) and keeps the reads real; a real run picks the robot's. The holds -> Session
step the fit needs keeps every pair's left stamp (the FIT/TEST labels are keyed by stamp)."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
yaml = pytest.importorskip("yaml")

from calibtool import assign, detect, flowrobot as fr, neck as nk, producer as pr, sessionctl as sc  # noqa: E402

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("_t_fr_sessionctl", HERE / "test_sessionctl.py")
TS = importlib.util.module_from_spec(spec)
spec.loader.exec_module(TS)


class Prof:
    bot, motor_tool, producer_ns, producer_deploy, root = "P10", "/root/p11-cal/dma_motor_motion", "dma-video", \
        "producer", "/data/calib"
    stand_raise_m, stand_lower_m = 0.3, 0.4


class Shm:
    def desired_head(self):
        return 7

    def joints(self):
        return ["NeckPitch", "NeckYaw", "LeftShoulderPitch"], [1.0, -2.0, 30.0]


def build(tmp_path, dry_run):
    calls = []

    def kubectl(argv, input=None):
        calls.append(list(argv))
        return 0, TS.PRODUCER_YAML if argv[:2] == ["get", "deploy"] else ""
    return fr.build(Prof(), session="S1", dirs=SimpleNamespace(root=tmp_path, deploy=tmp_path / "deploy"),
                    capture=SimpleNamespace(grab=lambda n: []), detect_pair=lambda yl, yr: None,
                    producer_yaml=TS.PRODUCER_YAML, probes={"autonomy": lambda: None, "pods": lambda: []},
                    baseline={}, obs_box=SimpleNamespace(take=lambda: (False, None)), guide_kw={},
                    make_collection=lambda: None, fit_decide=None, dry_run=dry_run,
                    run=lambda argv, input=None, timeout=None: (0, ""), motor_run=lambda argv, timeout=5: (0, ""),
                    kube_runner=kubectl, shm_robot=Shm(),
                    tool_info={"path": Prof.motor_tool, "has_targets": True, "md5": "x"}), calls


def test_a_dry_run_records_every_change_and_keeps_the_reads(tmp_path):
    r, calls = build(tmp_path, True)
    assert r.dry_run and r.recorder.dry_run and r.enabler.dry_run
    assert isinstance(r.kube.run, sc.DryKubectl) and r.kube.read_only is False
    assert calls and calls[0][:2] == ["get", "deploy"]             # the producer spec read from the robot
    n = r.make_neck()
    assert isinstance(n.run.__self__, nk.DryRun)
    assert n.robot.joints()[1][:2] == [1.0, -2.0]                  # the dry neck starts where the real one is
    r.kube.scale(0)
    assert not any(c[0] == "scale" for c in calls)                 # nothing mutating reached the robot's kubectl


def test_a_real_run_uses_the_robot(tmp_path):
    r, calls = build(tmp_path, False)
    assert not r.dry_run and not r.enabler.dry_run
    assert r.kube.run is not None and not isinstance(r.kube.run, sc.DryKubectl) and r.kube.read_only is False
    n = r.make_neck()
    assert n.tool == Prof.motor_tool and n.robot is not None and not isinstance(getattr(n.run, "__self__", None),
                                                                                 nk.DryRun)
    assert r.lease.path == Path(Prof.root) / "producer.lease"           # one lease per robot, not per session


def _hold(h, placement, t0, n=2):
    d = detect.hold_dict([(t0 + i * 66_000_000, t0 + i * 66_000_000 + 10_000,
                           {(t, k): (100.0 + t, 200.0 + k) for t in range(8) for k in range(4)},
                           {(t, k): (90.0 + t, 200.0 + k) for t in range(8) for k in range(4)}) for i in range(n)], h,
                         neck_rad=(0.1, -0.2), approach=(1, 0))
    return assign.CaptureHold(h, placement, t0, d, pose=h)


def test_holds_become_a_session_with_their_stamps(tmp_path):
    holds = [_hold(0, "0.75 m", 10**12), _hold(1, "0.75 m", 2 * 10**12), _hold(2, "1.6 m", 3 * 10**12)]
    s = fr.holds_session(holds)
    assert s.take_order == ["0.75 m", "1.6 m"]
    assert s.n_pairs == 6
    want = sorted(int(t) for h in holds for t in h.data["img_t0"])
    assert sorted(int(t) for t in s.pairs["t0"]) == want


def test_post_check_pairs_become_one_hold_with_the_board_in_both_cameras():
    def pair(t, dt):
        return SimpleNamespace(t0=t, t1=t + dt, dt_ns=dt, left=SimpleNamespace(y=("L", t)),
                               right=SimpleNamespace(y=("R", t)))
    seen = {t: (t != 3) for t in range(5)}

    def detect_pair(yl, yr):
        n = 10 if seen[yl[1]] else 2
        d = {(k, j): (1.0 * k, 2.0 * j) for k in range(n) for j in range(4)}
        return d, d
    holds, sync = fr.pairs_to_holds([pair(t, 200_000) for t in range(5)], detect_pair)
    assert len(holds) == 1 and len(holds[0].data["img_t0"]) == 4 and sync == pytest.approx(0.2)
    assert fr.pairs_to_holds([pair(3, 1)], detect_pair)[0] == []
