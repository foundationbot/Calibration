"""Which robot calibtool is talking to: config/robots.yaml maps each bot name (P10, P11, P12, ...) to its Thor's host
name now, the host names it had before, the prefix of its /etc/machine-id and its rig file.

identify(hostname, machine_id): the machine-id prefix first. A known machine-id names the robot, and the host name must
be one of that robot's names (now or before), else IdentityError: host names move between robots (phantom11 was P10's
until 9 Nov 2026 and is now P11's), the machine-id does not. Only a robot whose machine-id is not known yet is matched
by host name alone, and the result says so (machine_id_verified False); a host name that two robots had cannot be
told apart without the machine-id."""
from dataclasses import dataclass
from pathlib import Path

from calibtool import paths

ROBOTS_YAML = paths.PKG / "config" / "robots.yaml"
PREFIX_LEN = 12


class IdentityError(ValueError):
    """the host name and machine-id do not identify one robot of the registry"""


@dataclass(frozen=True)
class Robot:
    bot: str
    thor: str
    old_names: tuple
    machine_id_prefix: object       # str (PREFIX_LEN lowercase hex) or None: not known
    rig: object                     # rig file name in tools/calibration/config, or None

    @property
    def names(self) -> tuple:
        return (self.thor,) + self.old_names

    @property
    def rig_path(self):
        return paths.CONFIG / self.rig if self.rig else None


@dataclass(frozen=True)
class Identity:
    robot: Robot
    by: str                         # "machine-id" | "hostname"
    hostname: str
    machine_id_verified: bool


def _host(h) -> str:
    return str(h or "").strip().lower().split(".")[0]


def _mid(m) -> str:
    return str(m or "").strip().lower()


def load(path=ROBOTS_YAML) -> dict:
    """{bot: Robot}; ValueError for a malformed registry (a machine-id prefix that is not PREFIX_LEN hex digits, two
    robots with the same current host name or the same machine-id prefix)"""
    import yaml
    raw = yaml.safe_load(Path(path).read_text()) or {}
    out = {}
    for bot, e in (raw.get("robots") or {}).items():
        mid = e.get("machine_id_prefix")
        if mid is not None:
            mid = _mid(mid)
            if len(mid) != PREFIX_LEN or any(c not in "0123456789abcdef" for c in mid):
                raise ValueError(f"{path}: {bot}: machine_id_prefix {e.get('machine_id_prefix')!r} is not "
                                 f"{PREFIX_LEN} hex digits")
        out[str(bot)] = Robot(str(bot), _host(e["thor"]), tuple(_host(n) for n in e.get("old_names") or ()), mid,
                              e.get("rig"))
    for key in ("thor", "machine_id_prefix"):
        seen = {}
        for r in out.values():
            v = getattr(r, key)
            if v is not None and v in seen:
                raise ValueError(f"{path}: {seen[v]} and {r.bot} share {key} {v}")
            seen[v] = r.bot
    return out


def identify(hostname, machine_id, robots=None) -> Identity:
    """the robot at this host name and machine-id (module text); IdentityError when they do not name one robot"""
    robots = load() if robots is None else robots
    host, mid = _host(hostname), _mid(machine_id)
    by_mid = [r for r in robots.values() if r.machine_id_prefix and mid.startswith(r.machine_id_prefix)]
    if by_mid:
        r = by_mid[0]
        if host not in r.names:
            raise IdentityError(f"machine-id {mid[:PREFIX_LEN]} is {r.bot} ({', '.join(r.names)}), but the host "
                                f"calls itself {host!r}: refusing a host name that does not belong to that robot")
        return Identity(r, "machine-id", host, True)
    named = [r for r in robots.values() if host in r.names]
    if not named:
        raise IdentityError(f"no robot named {host!r} in {ROBOTS_YAML.name} (machine-id "
                            f"{mid[:PREFIX_LEN] or 'not given'})")
    unknown = [r for r in named if r.machine_id_prefix is None]
    if mid and unknown:
        named = unknown                 # robots with a known machine-id were ruled out by it
    if len(named) > 1:
        raise IdentityError(f"host name {host!r} belongs to {' and '.join(r.bot for r in named)}: the machine-id "
                            f"tells them apart, and none was given")
    r = named[0]
    if r.machine_id_prefix is not None:
        raise IdentityError(f"host name {host!r} is {r.bot}, whose machine-id starts {r.machine_id_prefix}, but this "
                            f"machine-id is {mid[:PREFIX_LEN] or 'not given'}: refusing")
    return Identity(r, "hostname", host, False)
