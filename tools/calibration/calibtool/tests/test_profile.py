"""calibtool.profile: the per-robot runtime calibd needs (host, python, pydeps, the session root on the robot, the head
camera queues, the producer deployment, the motor tool), from config/profiles.yaml keyed by bot name."""
import pytest

pytest.importorskip("yaml")

from calibtool import paths, profile, robots  # noqa: E402


def test_p10_profile():
    p = profile.load("P10")
    assert p.bot == "P10"
    assert p.host == "root@phantom10"
    assert p.python == "/root/p11-cal/venv/bin/python"
    assert p.pydeps == "/root/p11-cal/pydeps"
    assert p.root == "/tmp/calibtool-m2"
    assert (p.left, p.right) == (3, 0)
    assert (p.producer_ns, p.producer_deploy) == ("dma-video", "producer")
    assert p.motor_tool == "/root/p11-cal/dma_motor_motion"
    assert p.rig_path == robots.load()["P10"].rig_path
    assert p.rig_path.is_file()
    # the operator's chair-shaped board stand (9 Nov trial): up ~30 cm, down ~40 cm from its usual height
    assert (p.stand_raise_m, p.stand_lower_m) == (0.30, 0.40)
    assert p.placement_timeout_s == 180.0


def test_unknown_bot():
    with pytest.raises(KeyError):
        profile.load("P99")


def test_for_host_uses_the_robot_registry():
    assert profile.for_host("phantom10").bot == "P10"
    assert profile.for_host("root@phantom10").bot == "P10"
    with pytest.raises(KeyError):
        profile.for_host("nosuchhost")


def test_profiles_file_lives_in_the_package():
    assert profile.PROFILES_YAML == paths.PKG / "config" / "profiles.yaml"
