"""phantom12's four right-arm plate sessions as frames for armfit (plan docs/superpowers/plans/2026-10-25-tagfree-selfcal.md
on the stereo-depth-eval branch): each loaded exactly as its own plate script did, plus which tag-free capture (burst)
was taken in which hold.

  1023c / 1023d  holds from /joints matched to the step-91 log (joint_fit.load); kinds fit / held-out / repeat;
                 a stop's 'burst' names its capture
  rv1024         each capture's burst window (+10.5 s), static frames (rv_plate.py); held-out d0-p18, d1-p23, d4-p28
  rl1024         holds from /joints split by neck pitch, static frames (rl_plate.py); held-out = the groups of
                 k04-d1-p28, k07-d4-p23, k19-c13-p28 (mid-burst = BURST + 4 s, within 3 s)
"""
from __future__ import annotations

import importlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import armchain  # noqa: E402
import armfit as af  # noqa: E402
import rigcfg  # noqa: E402

LA = ZoneInfo("America/Los_Angeles")
RV_HELD = ("d0-p18", "d1-p23", "d4-p28")
RL_HELD = ("k04-d1-p28", "k07-d4-p23", "k19-c13-p28")


@dataclass
class Session:
    name: str
    obs: list            # armfit.Frame
    fit: set             # hold ids the plate fits use
    unseen: list         # hold ids held out (held-out + repeats)
    X0: np.ndarray       # the plan's plate start
    captures: dict       # capture name (as its burst file, without .npz) -> hold id


def _grid(plan):
    g = plan["grid"]
    return rigcfg.AprilGrid(g["cols"], g["rows"], g["tag_size"], g["tag_spacing"])


def bursts(logf, day="2026-10-24") -> dict:
    """{name: stamp ns} of a runner log's 'HH:MM:SS.fff BURST name' lines (robot local time)."""
    out = {}
    for line in open(logf):
        m = re.match(r"(\d\d:\d\d:\d\d\.\d+) BURST (\S+)", line)
        if m:
            t = datetime.strptime(f"{day} {m.group(1)[:12]}", "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=LA)
            out[m.group(2)] = int(t.timestamp() * 1e9)
    return out


def static(obs, deg=0.3):
    """The frames within deg (arm + neck) of their hold's median joints."""
    med = {h: np.median([np.r_[x.q_arm, x.q_neck] for x in obs if x.hold == h], axis=0) for h in {x.hold for x in obs}}
    return [x for x in obs if np.degrees(np.abs(np.r_[x.q_arm, x.q_neck] - med[x.hold])).max() <= deg]


def load_cd(task2, tag) -> Session:
    s92 = importlib.import_module("92_arm_fit")
    plan = json.load(open(f"{task2}/poses-{tag}.json")); names = plan["names"]
    d = dict(np.load(f"{task2}/sweep_{tag}_corners.npz")); ia = [names.index(n) for n in armchain.RIGHT_ARM]
    holds = af.holds_from_joints(d["jt"], d["jq"][:, ia])
    hq = np.array([d["jq"][(d["jt"] >= a) & (d["jt"] <= b)][:, ia].mean(0) for a, b in holds])
    m = af.match_holds(hq, s92.real_stops([json.loads(x) for x in open(f"{task2}/arm-session-{tag}.jsonl")]))
    keep = sorted(m)
    obs = af.frames_from_dump(d, [holds[h] for h in keep], names, armchain.RIGHT_ARM, _grid(plan))
    for f in obs:
        f.hold = keep[f.hold]
    fit = {h for h in keep if m[h]["kind"] == "fit"}
    caps = {Path(m[h]["burst"]).stem: h for h in keep if m[h].get("burst")}
    return Session(tag, obs, fit, sorted(set(keep) - fit), np.array(plan["X0"]), caps)


def load_rv(task2, rv) -> Session:
    plan = json.load(open(f"{task2}/poses-1023d.json"))
    b = bursts(f"{rv}/rv-log.txt"); caps = list(b)
    assert len(caps) == 9, caps
    win = [(b[c], b[c] + 10_500_000_000) for c in caps]
    obs = static(af.frames_from_dump(dict(np.load(f"{rv}/rv1024_corners.npz")), win, plan["names"],
                                     armchain.RIGHT_ARM, _grid(plan)))
    held = {i for i, c in enumerate(caps) if c in RV_HELD}
    assert len(held) == 3
    return Session("rv1024", obs, set(range(len(caps))) - held, sorted(held), np.array(plan["X0"]),
                   {"rv1024-" + c: i for i, c in enumerate(caps)})


def load_rl(task2, rl) -> Session:
    plan = json.load(open(f"{task2}/poses-1023d.json")); names = plan["names"]
    b = bursts(f"{rl}/rl-log.txt")
    assert len(b) == 9, b
    d = dict(np.load(f"{rl}/rl1024_corners.npz")); ia = [names.index(n) for n in armchain.RIGHT_ARM]
    holds = af.holds_from_joints(d["jt"], d["jq"][:, ia])
    obs = af.frames_from_dump(d, holds, names, armchain.RIGHT_ARM, _grid(plan))
    for x in obs:
        x.hold = x.hold * 100 + int(round(np.degrees(x.q_neck[1])))
    obs = static(obs)
    caps = {}
    for n, t in b.items():
        hs = {x.hold for x in obs if abs(x.t - (t + 4_000_000_000)) < 3_000_000_000}
        assert len(hs) == 1, (n, hs)
        caps["rl1024-" + n] = hs.pop()
    held = {caps["rl1024-" + n] for n in RL_HELD}
    return Session("rl1024", obs, {x.hold for x in obs} - held, sorted(held), np.array(plan["X0"]), caps)


def load_all(task2, rv, rl) -> list:
    return [load_cd(task2, "1023c"), load_cd(task2, "1023d"), load_rv(task2, rv), load_rl(task2, rl)]
