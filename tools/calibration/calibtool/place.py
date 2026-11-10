"""Spoken placement guidance, pause-then-speak (design plan 4.1), and the placement-trial mode (M2, neck at home).

PlacementGuide(target).update(t, obs) is fed the board as the left camera sees it (detect.BoardObs, or None) about
twice a second. It never speaks while the board moves: once the board's centroid has stayed within still_px for
still_s (~1 s) it says ONE thing, the most important first:
  board cut by the image edge  -> move it toward the middle ("to YOUR left (the robot's right)", "higher", "lower")
  wrong distance band          -> "about N centimetres further back / closer" (the far placement: "too far")
  off the target bearing       -> sideways (only for targets with a bearing: adaptive extras)
  otherwise                    -> "Good. Step back." (the target is done)
then stays quiet until the board has moved and come to rest again (or repeats the same correction after repeat_s if
nobody moved it). A board not seen for lost_s gets one "I can't see the board" prompt. Directions always name the
operator's side first: the operator faces the robot, so the robot's left is YOUR right. The word "hold" is not used in
speech (inside the tool it is a still neck pose).

Default placements (design section 4 step 1): 0.5, 0.75, 1.1, 1.6 m, the far one (asked at protocol.FAR_ASK_M,
accepted in FAR_ACCEPT_M, too far beyond FAR_REFUSE_BEYOND_M) and the tilted one at 0.75 m (leaning back on the chair).
Tolerance is loose for defaults: a capture counts toward the band it measured (protocol.band_of), so any distance in
the target's nominal band is accepted.

Trial(targets, log_path) runs the timed trials (an existing log is read back: a restarted calibd continues with the
next placement): an intro per placement, the guide until "Good", key C = "can't put it
there" (logged unreachable), a timeout (failed); trial.jsonl gets one line per placement (seconds from the intro to the
board coming to rest, result, prompts spoken); summary() = success rate and seconds against the 60 s budget (design
10.4)."""
import itertools
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

from calibtool import protocol

STILL_S = 1.0
STILL_PX = 6.0
LOST_S = 6.0
REPEAT_S = 20.0
TIMEOUT_S = 180.0
BUDGET_S = 60.0                 # design 10.4: target < 1 min per placement
YOUR_LEFT = "to YOUR left (the robot's right)"
YOUR_RIGHT = "to YOUR right (the robot's left)"


@dataclass(frozen=True)
class Target:
    name: str
    distance_m: float
    accept: tuple                       # (lo, hi) m accepted
    tilted: bool = False
    bearing_deg: object = None          # + toward the robot's left; None: straight ahead, not checked
    tol_deg: object = None
    refuse_beyond_m: object = None      # beyond this: "too far" (the far placement)


def _band(d):
    """the nominal band of protocol.band_of around d: (lo, hi)"""
    nom = list(protocol.NOMINAL)
    i = nom.index(d)
    lo = (nom[i - 1] + d) / 2 if i > 0 else 0.0
    hi = (nom[i + 1] + d) / 2 if i + 1 < len(nom) else math.inf
    return (lo, hi)


DEFAULTS = (Target("0.5 m", 0.5, _band(0.5)), Target("0.75 m", 0.75, _band(0.75)), Target("1.1 m", 1.1, _band(1.1)),
            Target("1.6 m", 1.6, _band(1.6)),
            Target("far", protocol.FAR_ASK_M, tuple(protocol.FAR_ACCEPT_M),
                   refuse_beyond_m=protocol.FAR_REFUSE_BEYOND_M),
            Target("tilted", 0.75, _band(0.75), tilted=True))


def target(name) -> Target:
    for t in DEFAULTS:
        if t.name == name:
            return t
    raise KeyError(name)


def _dist_words(m) -> str:
    if m < 1.0:
        return f"{int(round(m * 100))} centimetres"
    return f"{m:g} metres"


def intro(t: Target, i, n) -> str:
    if t.tilted:
        where = (f"about {_dist_words(t.distance_m)} in front of the head, leaning back against the chair's "
                 "backrest so it faces up toward the head")
    elif t.bearing_deg:
        side = YOUR_RIGHT if t.bearing_deg > 0 else YOUR_LEFT
        where = (f"about {_dist_words(t.distance_m)} from the head, {abs(t.bearing_deg):.0f} degrees {side}, "
                 "facing the head")
    else:
        where = f"on its stand about {_dist_words(t.distance_m)} in front of the head, facing it"
    return f"Placement {i} of {n}: put the board {where}."


@dataclass
class Say:
    id: int
    t: float
    kind: str                   # intro | correction | good | lost | end
    text: str

    def as_dict(self):
        return {"id": self.id, "t": self.t, "kind": self.kind, "text": self.text}


def _cm(m) -> int:
    return max(10, int(math.floor(abs(m) * 10 + 0.5 + 1e-9)) * 10)     # to 10 cm, half up


class PlacementGuide:
    def __init__(self, target: Target, focal, size=(1280, 800), pp=None, still_s=STILL_S, still_px=STILL_PX,
                 lost_s=LOST_S, repeat_s=REPEAT_S, ids=None):
        self.target, self.f, self.size = target, float(focal), tuple(size)
        self.pp = pp or ((size[0] - 1) / 2.0, (size[1] - 1) / 2.0)
        self.still_s, self.still_px, self.lost_s, self.repeat_s = still_s, still_px, lost_s, repeat_s
        self.ids = ids or itertools.count(1)
        self.done = False
        self.rest_t = None              # when the board came to rest for the accepted check
        self.says = []
        self._start = None              # (t, centroid) of the current still stretch
        self._spoken_at = None          # t of the last say in this still stretch
        self._last_seen = None
        self._lost_since = None
        self._lost_said = False
        self.last_obs = None

    def _say(self, t, kind, text):
        s = Say(next(self.ids), float(t), kind, text)
        self.says.append(s)
        self._spoken_at = t
        return s

    def update(self, t, obs):
        if self.done:
            return None
        self.last_obs = obs
        if obs is None:
            self._start = None
            self._spoken_at = None
            if self._lost_since is None:
                self._lost_since = t
            if not self._lost_said and t - self._lost_since >= self.lost_s:
                self._lost_said = True
                return self._say(t, "lost", "I can't see the board. Turn it to face the head, and keep it within "
                                            "about a metre of straight ahead.")
            return None
        self._lost_since, self._lost_said = None, False
        c = obs.centroid
        if self._start is None or math.hypot(c[0] - self._start[1][0], c[1] - self._start[1][1]) > self.still_px:
            self._start = (t, (float(c[0]), float(c[1])))
            self._spoken_at = None
            return None
        if t - self._start[0] < self.still_s:
            return None
        if self._spoken_at is not None and t - self._spoken_at < self.repeat_s:
            return None
        kind, text = self.judge(obs)
        if kind == "good":
            self.done = True
            self.rest_t = self._start[0]
        return self._say(t, kind, text)

    def bearing_deg(self, obs) -> float:
        """+ toward the robot's left (image left), neck at home"""
        return -math.degrees(math.atan2(obs.centroid[0] - self.pp[0], self.f))

    def judge(self, obs):
        """(kind, text) for a board at rest: the most important correction, or good"""
        t = self.target
        if obs.cut:
            side = obs.cut[0]
            z = obs.z
            span = {"left": obs.lo[0], "right": self.size[0] - 1 - obs.hi[0], "top": obs.lo[1],
                    "bottom": self.size[1] - 1 - obs.hi[1]}[side]
            move = _cm(max(0.1, (60.0 - span) * z / self.f + 0.05) if span < 60 else 0.1)
            words = {"left": YOUR_LEFT, "right": YOUR_RIGHT, "top": "lower", "bottom": "higher"}[side]
            return "correction", f"The board is cut off by the camera's edge: move it about {move} centimetres {words}."
        lo, hi = t.accept
        z = obs.z
        if t.refuse_beyond_m is not None and z > t.refuse_beyond_m:
            return "correction", (f"That is too far for the tags to be read: move the board about "
                                  f"{_cm(z - t.distance_m)} centimetres closer.")
        if z < lo:
            return "correction", f"Move the board about {_cm(t.distance_m - z)} centimetres further back."
        if z >= hi:
            return "correction", f"Move the board about {_cm(z - t.distance_m)} centimetres closer."
        if t.bearing_deg is not None and t.tol_deg is not None:
            b = self.bearing_deg(obs)
            if abs(b - t.bearing_deg) > t.tol_deg:
                cm = _cm(z * (math.tan(math.radians(t.bearing_deg)) - math.tan(math.radians(b))))
                side = YOUR_RIGHT if t.bearing_deg > b else YOUR_LEFT
                return "correction", f"Move the board about {cm} centimetres {side}."
        return "good", "Good. Step back."


class Trial:
    """the timed placement trials (module text)"""

    def __init__(self, targets, log_path, timeout_s=TIMEOUT_S, focal=568.0, size=(1280, 800), budget_s=BUDGET_S,
                 **guide_kw):
        self.targets = list(targets)
        self.log_path = Path(log_path)
        self.timeout_s, self.focal, self.size, self.budget_s = timeout_s, focal, size, budget_s
        self.guide_kw = guide_kw
        self.ids = itertools.count(1)
        self.i = -1
        self.guide = None
        self.t_intro = None
        self.records = []
        self.finished = False
        self.says = []
        if self.log_path.is_file():         # a restarted calibd: continue after the placements already logged
            for ln in self.log_path.read_text().splitlines():
                try:
                    self.records.append(json.loads(ln))
                except json.JSONDecodeError:
                    continue
            self.i = min(len(self.records), len(self.targets)) - 1

    def _next(self, t):
        self.i += 1
        if self.i >= len(self.targets):
            self.finished = True
            self.guide = None
            s = Say(next(self.ids), float(t), "end", "That was the last placement. Thank you.")
            self.says.append(s)
            return [s]
        tg = self.targets[self.i]
        self.guide = PlacementGuide(tg, self.focal, self.size, ids=self.ids, **self.guide_kw)
        self.t_intro = t
        s = Say(next(self.ids), float(t), "intro", intro(tg, self.i + 1, len(self.targets)))
        self.says.append(s)
        return [s]

    def start(self, t) -> Say:
        return self._next(t)[0]

    def _record(self, t, result, rest_t=None):
        g = self.guide
        rec = {"target": self.targets[self.i].name, "distance_m": self.targets[self.i].distance_m, "result": result,
               "seconds": round((rest_t if rest_t is not None else t) - self.t_intro, 2),
               "prompts": len(g.says) if g else 0, "t_intro": self.t_intro, "t_end": t,
               "z_m": None if g is None or g.last_obs is None else round(float(g.last_obs.z), 3)}
        self.records.append(rec)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def update(self, t, obs) -> list:
        if self.finished or self.guide is None:
            return []
        out = []
        s = self.guide.update(t, obs)
        if s is not None:
            self.says.append(s)
            out.append(s)
        if self.guide.done:
            self._record(t, "placed", self.guide.rest_t)
            out += self._next(t)
        elif t - self.t_intro >= self.timeout_s:
            self._record(t, "timeout")
            out += self._next(t)
        return out

    def key(self, t, k) -> list:
        if self.finished or self.guide is None:
            return []
        if str(k).upper() == "C":
            self._record(t, "unreachable")
            return self._next(t)
        return []

    def summary(self) -> dict:
        placed = [r["seconds"] for r in self.records if r["result"] == "placed"]
        n = len(self.records)
        return {"placements": n, "placed": len(placed), "success_rate": len(placed) / n if n else None,
                "mean_s": statistics.mean(placed) if placed else None,
                "median_s": statistics.median(placed) if placed else None, "budget_s": self.budget_s,
                "over_budget": sum(s > self.budget_s for s in placed),
                "results": {r["target"]: r["result"] for r in self.records}}

    def state(self) -> dict:
        cur = None if self.finished or self.i < 0 else self.targets[self.i]
        g = self.guide
        o = None if g is None else g.last_obs
        return {"index": self.i + 1, "of": len(self.targets), "target": None if cur is None else cur.name,
                "finished": self.finished, "since_intro_s": None,
                "board": None if o is None else {"z_m": round(float(o.z), 3), "u": round(float(o.centroid[0]), 1),
                                                 "v": round(float(o.centroid[1]), 1), "tags": int(o.n_tags),
                                                 "cut": list(o.cut)},
                "records": self.records, "summary": self.summary()}
