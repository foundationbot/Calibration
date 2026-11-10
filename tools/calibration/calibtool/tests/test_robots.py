"""calibtool.robots: which robot calibtool is talking to. config/robots.yaml maps each bot name to its Thor's current
host name, the host names it had, its machine-id prefix and its rig file; identify() matches on the machine-id prefix
first and refuses a host name that does not belong to the robot the machine-id names (host names were reused across
robots on 9 Nov 2026: phantom11 was P10's name and is now P11's)."""
import pytest

pytest.importorskip("yaml")

from calibtool import paths, robots  # noqa: E402

P10_MID = "60776b088362" + "0123456789abcdef0123"
P12_MID = "9851aca02a7d" + "fedcba9876543210fedc"
OTHER_MID = "deadbeef0000" + "11112222333344445555"


def test_registry_entries():
    r = robots.load()
    assert set(r) == {"P10", "P11", "P12"}
    assert (r["P10"].thor, r["P10"].old_names, r["P10"].machine_id_prefix, r["P10"].rig) == \
        ("phantom10", ("phantom11",), "60776b088362", "rig-phantom11.yaml")
    assert (r["P12"].thor, r["P12"].old_names, r["P12"].machine_id_prefix, r["P12"].rig) == \
        ("phantom12", ("phantom21",), "9851aca02a7d", "rig-phantom21.yaml")
    assert (r["P11"].thor, r["P11"].old_names, r["P11"].machine_id_prefix, r["P11"].rig) == \
        ("phantom11", ("phantom07",), None, None)
    for x in r.values():
        assert x.rig_path is None or x.rig_path == paths.CONFIG / x.rig and x.rig_path.is_file()
    assert robots.ROBOTS_YAML == paths.PKG / "config" / "robots.yaml"


@pytest.mark.parametrize("host,mid,bot", [
    ("phantom10", P10_MID, "P10"),
    ("phantom11", P10_MID, "P10"),            # P10 before its rename: the machine-id decides, not the name
    ("phantom12", P12_MID, "P12"),
    ("phantom21", P12_MID, "P12"),            # rename pending: still reachable as phantom21
    ("PHANTOM10.local", P10_MID.upper() + "\n", "P10"),
])
def test_machine_id_first(host, mid, bot):
    i = robots.identify(host, mid)
    assert i.robot.bot == bot and i.by == "machine-id" and i.machine_id_verified


@pytest.mark.parametrize("host,mid", [("phantom12", P10_MID), ("phantom10", P12_MID), ("phantom07", P12_MID)])
def test_hostname_of_another_robot_is_refused(host, mid):
    with pytest.raises(robots.IdentityError, match="machine-id"):
        robots.identify(host, mid)


def test_known_robot_with_another_machine_id_is_refused():
    with pytest.raises(robots.IdentityError, match="P10"):
        robots.identify("phantom10", OTHER_MID)


def test_robot_without_a_known_machine_id_by_host_name():
    """P11's machine-id is not known: its host name identifies it, marked unverified; phantom11 with an unknown
    machine-id is P11 (P10, whose old name it is, has a known machine-id that does not match)"""
    i = robots.identify("phantom11", OTHER_MID)
    assert i.robot.bot == "P11" and i.by == "hostname" and not i.machine_id_verified
    assert robots.identify("phantom07", None).robot.bot == "P11"


def test_without_a_machine_id_a_name_two_robots_had_is_refused():
    with pytest.raises(robots.IdentityError, match="P10.*P11|P11.*P10"):
        robots.identify("phantom11", None)
    with pytest.raises(robots.IdentityError, match="machine-id"):
        robots.identify("phantom10", "")


def test_unknown_robot():
    with pytest.raises(robots.IdentityError, match="phantom99"):
        robots.identify("phantom99", OTHER_MID)


def test_a_bad_registry_is_refused(tmp_path):
    p = tmp_path / "robots.yaml"
    p.write_text("robots:\n  A: {thor: x, old_names: [], machine_id_prefix: '1234', rig: null}\n"
                 "  B: {thor: y, old_names: [x], machine_id_prefix: null, rig: null}\n")
    with pytest.raises(ValueError, match="machine_id_prefix"):
        robots.load(p)
    p.write_text("robots:\n  A: {thor: x, old_names: [], machine_id_prefix: 60776b088362, rig: null}\n"
                 "  B: {thor: x, old_names: [], machine_id_prefix: null, rig: null}\n")
    with pytest.raises(ValueError, match="x"):
        robots.load(p)
