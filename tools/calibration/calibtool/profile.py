"""The per-robot runtime calibd needs, from config/profiles.yaml (keyed by the bot names of config/robots.yaml): ssh
host, robot-side python and dma_ipc folder, the session root on the robot, the head camera queues, the producer
deployment, the motor tool, and the placement trial's board stand and time limit (null / 0: none). load(bot) /
for_host(host name or user@host)."""
from dataclasses import dataclass
from pathlib import Path

from calibtool import paths, robots

PROFILES_YAML = paths.PKG / "config" / "profiles.yaml"


@dataclass(frozen=True)
class Profile:
    bot: str
    thor: str
    python: str
    pydeps: str
    root: str
    left: int
    right: int
    producer_ns: str
    producer_deploy: str
    motor_tool: str
    rig_path: object                # Path or None
    machine_id_prefix: object
    stand_raise_m: float = 0.30     # the operator's board stand: up / down this far from the placement's start height
    stand_lower_m: float = 0.40
    placement_timeout_s: object = 180.0     # seconds, or None: no limit (only the operator's C skips a placement)
    stand_auto_skip: bool = True   # False: the operator decides reachability (C), the guide never skips itself

    @property
    def host(self) -> str:
        return f"root@{self.thor}"


def load(bot, path=PROFILES_YAML, registry=None) -> Profile:
    """the profile of bot (KeyError when the bot has no profile or is not in robots.yaml)"""
    import yaml
    raw = (yaml.safe_load(Path(path).read_text()) or {}).get("profiles") or {}
    if bot not in raw:
        raise KeyError(f"no calibd profile for {bot!r} in {Path(path).name} (have: {', '.join(sorted(raw))})")
    reg = robots.load() if registry is None else registry
    r = reg[bot]
    e = raw[bot]
    prod = e.get("producer") or {}
    stand = e.get("stand") or {}
    return Profile(bot, r.thor, str(e["python"]), str(e.get("pydeps") or ""), str(e["root"]), int(e["left"]),
                   int(e["right"]), str(prod.get("namespace", "dma-video")), str(prod.get("deployment", "producer")),
                   str(e.get("motor_tool") or ""), r.rig_path, r.machine_id_prefix,
                   float(stand.get("raise_max_m", 0.30)), float(stand.get("lower_max_m", 0.40)),
                   place_limit(e.get("placement_timeout_s", 180.0)), bool(stand.get("auto_skip", True)))


def place_limit(v):
    """the profile's placement_timeout_s: seconds, or None for null / 0 (no limit)"""
    return None if v is None or float(v) <= 0 else float(v)


def for_host(host, path=PROFILES_YAML) -> Profile:
    """the profile of the robot whose current host name is host ("phantom10" or "root@phantom10")"""
    h = str(host).split("@")[-1].split(".")[0].lower()
    reg = robots.load()
    for r in reg.values():
        if h == r.thor:             # current names only: an old name may belong to another robot now
            return load(r.bot, path, reg)
    raise KeyError(f"no robot whose host name is {h!r}")
