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
    # 10 Nov: "Skipping this one." came after 3 minutes while the operator was still trying; on phantom10 only the
    # operator's C skips a placement
    assert p.placement_timeout_s is None


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


@pytest.mark.parametrize("line,want", [("", 180.0), ("    placement_timeout_s: 90\n", 90.0),
                                       ("    placement_timeout_s: null\n", None),
                                       ("    placement_timeout_s: 0\n", None)])
def test_the_placement_time_limit_is_a_profile_setting_null_or_0_is_none(tmp_path, line, want):
    y = tmp_path / "profiles.yaml"
    y.write_text("profiles:\n  P10:\n    python: /x/python\n    root: /tmp/r\n    left: 3\n    right: 0\n" + line)
    assert profile.load("P10", y).placement_timeout_s == want


def test_p11_has_a_profile_with_its_live_config_and_kit():
    """10 Nov 2026: P11 (phantom11) set up for calibtool; robot.env names no calibration, so the profile does"""
    p = profile.load("P11")
    assert p.host == "root@phantom11" and (p.left, p.right) == (3, 0) and p.root == "/data/calib"
    assert p.python == "/root/calib-kit/venv/bin/python" and p.motor_tool == "/root/calib-kit/dma_motor_motion"
    assert p.live_config.endswith("refitA-yawA2-1280x800-15fps.yaml") and p.stand_steps
    assert p.machine_id_prefix == "dfaae203c0d0" and p.rig_path.name == "rig-p11.yaml"
    assert profile.load("P10").live_config is None
    assert p.neck_kp == 60.0 and profile.load("P10").neck_kp is None
