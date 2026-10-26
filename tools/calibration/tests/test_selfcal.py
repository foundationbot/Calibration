"""Phase D of docs/superpowers/plans/2026-10-25-tagfree-selfcal.md (stereo-depth-eval branch): the one-command arm-wave
session's runner (generated shell for phantom12) and its pre-checks (selfcal.py)."""
import re
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
import selfcal  # noqa: E402

NECK = {"NeckYaw": -0.41, "NeckPitch": 28.4}
MOVES = [{"name": "S1", "deg": [-24.45, 5.75, 7.53, -43.06, 25.35, -11.2, 29.3], "burst": False, "max_travel_deg": 37.3,
          "path_min_clear_m": 0.04},
         {"name": "start", "deg": [-24.45, 5.75, -49.34, -43.06, -131.5, -15.7, 56.0], "burst": False,
          "max_travel_deg": 57.0, "path_min_clear_m": 0.03},
         {"name": "1023d/1", "deg": [-41.28, 7.93, -45.97, -95.84, -130.84, -15.13, 56.32], "burst": True,
          "max_travel_deg": 10.8, "path_min_clear_m": 0.037},
         {"name": "1023d/4", "deg": [-56.0, 34.08, -50.13, -97.81, -128.37, -15.33, 56.3], "burst": True,
          "max_travel_deg": 30.0, "path_min_clear_m": 0.03}]


def _run():
    return selfcal.runner(MOVES, NECK, "sc-test", "Left", held=("04-1023d_4",))


def test_every_move_lists_all_seven_arm_joints_and_the_neck_and_checks_motors_first():
    sh = _run()
    moves = [l for l in sh.splitlines() if l.startswith("mv ")]
    assert len(moves) == 4
    for l in moves:
        assert all(f"Left{j}=" in l for j in selfcal.JOINTS) and "NeckPitch=28.40" in l and "NeckYaw=-0.41" in l
    assert re.search(r"mv\(\)\{ chk;", sh)                                 # motors 29/29 before every move
    assert "--kp 100" in sh and "--no-return" in sh


def test_it_aborts_on_a_motor_drop_a_failed_move_or_a_failed_burst():
    sh = _run()
    assert 'grep -q "Motors enabled:    29/29"' in sh and "exit 2" in sh
    assert "ABORT rc" in sh and "exit 1" in sh
    assert "ABORT burst" in sh and "exit 3" in sh
    assert sh.splitlines()[1].startswith("set -u")


def test_bursts_follow_their_moves_and_the_held_out_are_named_before_data():
    sh = _run().splitlines()
    grabs = [l.split()[1] for l in sh if l.startswith("grab ")]
    assert grabs == ["03-1023d_1", "04-1023d_4"]                          # <move number>-<name, / as _>
    assert any(l.startswith("# held-out:") and "04-1023d_4" in l for l in sh)


def test_long_moves_ramp_slower():
    sh = [l for l in _run().splitlines() if l.startswith("mv ")]
    assert [int(l.split()[2]) for l in sh] == [6, 14, 6, 6]


def test_a_move_without_clearance_is_refused():
    bad = [dict(MOVES[0], path_min_clear_m=0.01)]
    with pytest.raises(ValueError, match="clearance"):
        selfcal.runner(bad, NECK, "sc-test", "Left")


SNAP = {"desired": {"advancing": False}, "disk_free_gb": 210.0}
STATE = "Motors enabled:    29/29\n"


def test_prechecks_pass_on_a_ready_robot():
    assert selfcal.precheck(SNAP, STATE, okvis_log="depth ok\n", evict_free_gb=151.0, okvis_ready=True) == []


@pytest.mark.parametrize("snap,state,log,ready,why", [
    (SNAP, "Motors enabled:    28/29\n", "", True, "motors"),
    (dict(SNAP, desired={"advancing": True}), STATE, "", True, "walker"),
    (dict(SNAP, disk_free_gb=152.0), STATE, "", True, "disk"),
    (SNAP, STATE, "W [cam-stall] camera 2 no frames\n", True, "cam-stall"),
    (SNAP, STATE, "", False, "okvis"),                  # 24 Oct 23:59: evicted pod back, container not running
])
def test_prechecks_name_what_fails(snap, state, log, ready, why):
    fails = selfcal.precheck(snap, state, okvis_log=log, evict_free_gb=151.0, okvis_ready=ready)
    assert len(fails) == 1 and why in fails[0]


def test_a_name_with_a_space_or_slash_never_shifts_the_shell_arguments():
    # review 25 Oct: "rev 1023d/11" made `mv 16-rev 1023d_11 6 "..."`, so dma_motor_motion got "6" as its targets
    import shlex
    rev = dict(MOVES[2], name="rev 1023d/11", burst=True)
    sh = selfcal.runner(MOVES + [rev], NECK, "sc-test", "Left", held=("05-rev_1023d_11",)).splitlines()
    for l in sh:
        if l.startswith("mv "):
            w = shlex.split(l)
            assert len(w) == 4 and w[2].isdigit() and "=" in w[3], l          # mv NAME RAMP "TARGETS"
        if l.startswith("grab "):
            assert len(shlex.split(l)) == 2, l
    assert "grab 05-rev_1023d_11" in sh


def test_one_session_at_a_time_and_the_stop_command_is_written_down():
    sh = _run()
    assert "flock -n 9" in sh and "exit 5" in sh                              # a second runner refuses to start
    assert any(l.startswith("# stop:") and "pkill" in l for l in sh.splitlines())


def test_go_refuses_when_the_robot_moved_since_the_plan():
    planned = {"q_deg": {"LeftShoulderPitch": -11.4, "LeftElbowPitch": -6.5, "SpineYaw": -16.7, "NeckPitch": 28.2}}
    same = {"q_deg": dict(planned["q_deg"], LeftShoulderPitch=-10.0)}           # 1.4 deg: fine
    moved = {"q_deg": dict(planned["q_deg"], LeftElbowPitch=-30.0)}             # the arm moved
    turned = {"q_deg": dict(planned["q_deg"], SpineYaw=0.0)}                    # the body turned
    assert selfcal.pose_changes(planned, same, "Left") == []
    assert [n for n, _ in selfcal.pose_changes(planned, moved, "Left")] == ["LeftElbowPitch"]
    assert [n for n, _ in selfcal.pose_changes(planned, turned, "Left")] == ["SpineYaw"]


def test_prechecks_refuse_a_running_or_stale_session():
    fails = selfcal.precheck(SNAP, STATE, okvis_log="", evict_free_gb=151.0, okvis_ready=True,
                             running="12345 dma_motor_motion ...", stale=["sc-test.log"])
    assert any("running" in f for f in fails) and any("stale" in f for f in fails)
