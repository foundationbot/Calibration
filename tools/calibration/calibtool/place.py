"""Spoken placement guidance, pause-then-speak (design plan 4.1), and the placement-trial mode (M2, neck at home).

PlacementGuide(target).update(t, obs) is fed the board as the left camera sees it (detect.BoardObs, or None) several
times a second. It never speaks while the board moves: once the board has stayed put for still_s (0.4 s; moved = the
centroid beyond still_px, the depth by more than 3 %, or the height by more than 5 cm) it says ONE thing, the most
important first:
  board cut by the image edge  -> move it toward the middle ("to your left", "Raise the board. At least N centimetres")
  few tags read (oblique)      -> "Tilt the board back. Face it to the robot's head."
  wrong distance band          -> with gravity ('up' in the camera, gravity.py) the board's position splits into its
                                  height relative to the camera and its distance along the floor:
                                  reachable at this height -> "Move the board N centimetres closer. Keep the same
                                  height." (or "further back"); reachable only higher / lower within the stand's range
                                  -> "Raise the board. At least N centimetres." (or Lower); not within the stand's
                                  range -> "This one is out of reach. Skipping." and the placement ends as unreachable
                                  (it is final: only after two separate rests of 1 s judged so, or one of 3 s)
                                  without gravity -> the camera distance alone, as before ("About N centimetres")
  off the target bearing       -> sideways (only for targets with a bearing: adaptive extras)
  otherwise                    -> "Good. That one is done. Step back."
After the operator moved the board and it rests again: the same problem, smaller -> "Not yet. N more centimetres
closer." (what is left); the same problem, not smaller -> "Not yet. " + the instruction; another problem -> "Good. " +
the instruction. Unchanged and unmoved, an instruction is repeated at most once, after repeat_s. A board not seen for
lost_s gets one "I cannot see the board" prompt. Directions name the operator's side ("to your left"); the word "hold"
is not used in speech (inside the tool it is a still neck pose).

The distance model (9 Nov trial, phantom10): the head camera is ~1.5 m up and looks steeply down, so a board standing
at the robot's feet is already ~1.1 m from the camera, and the near placements need the board high near the head. The
operator's chair-shaped stand moves in coarse steps: from the height the board had at the start of the placement (h0,
recorded at the first rest that gets an instruction; a silent rest may be a pause while carrying) it can go up raise_max_m or down lower_max_m (Stand, from the robot profile). A height dh
(m above the camera) can serve a target of depth D when the board, moved along the floor at that height in its
current direction, reaches depth D (exactly: depth is linear in the floor distance) with
  - |dh| < D sin(STEEPEST_DEG): not nearly straight below or above the head, where the robot's body is (the plain
    rule "|dh| >= D cannot reach distance D", with a margin), and
  - its centre within MAX_OFF_AXIS_DEG of the optical axis (the image is +-40 deg tall; the board spans +-10-16 deg).
D is the target's own depth, else another depth of its band (BAND_MARGIN_M inside the edges): on phantom10 (camera
65.9 deg down) 1.6 m deep that close to the axis is near the floor, but the band's 1.4 m is not. The board's position
is the PnP centre (BoardObs.p); acceptance stays the depth band (protocol.band_of) as before.

Default placements (design section 4 step 1): 0.5, 0.75, 1.1, 1.6 m, the far one (asked at protocol.FAR_ASK_M,
accepted in FAR_ACCEPT_M, too far beyond FAR_REFUSE_BEYOND_M) and the tilted one at 0.75 m (leaning back on the chair).
Tolerance is loose for defaults: a capture counts toward the band it measured (protocol.band_of), so any distance in
the target's nominal band is accepted.

Trial(targets, log_path) runs the timed trials (an existing log is read back: a restarted calibd continues with the
next placement): an intro per placement, the guide until "Good" (placed) or "out of reach" (unreachable), key C =
"can't put it there" (unreachable), and after timeout_s "Skipping this one." (timeout); trial.jsonl gets one line per
placement (seconds from the intro to the board coming to rest, result, prompts spoken); summary() = success rate and
seconds against the 60 s budget (design 10.4); state() names the CURRENT placement (9 Nov: a stale "placement 2 of 6"
misled everyone)."""
import itertools
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from calibtool import protocol

STILL_S = 0.4            # 9 Nov trial: 1.0 s, then 0.6 s, plus slow detection felt laggy
STILL_PX = 6.0
UNREACHABLE_STILL_S = 1.0  # "out of reach" ends the placement, so it needs rests of at least this long ...
UNREACHABLE_RESTS = 2      # ... judged out of reach on this many separate rests (review 9: one pause while carrying
UNREACHABLE_LONG_S = 3.0   # the board low ended a placement for good), or one rest this long
FEW_TAGS = 20          # of 35: fewer readable tags with the board in view = seen too obliquely (9 Nov trial: 8)
FEW_TAGS_STILL_PX = 25.0   # few tags make the centroid jitter; stillness is judged more loosely then
LOST_S = 6.0
MOVE_Z_FRAC = 0.03       # a distance change above this counts as the board moving
MOVE_H_M = 0.05          # so does a height change above this (a board raised or lowered on its stand)
SHRINK_M = 0.03          # the same correction counts as smaller when its amount fell by more than this
REPEAT_S = 30.0          # the same instruction is repeated at most once, after this long unchanged
GAP_S = 3.0              # after a prompt has been spoken, the operator gets this long to act before the next one
WORD_S, SENTENCE_S = 0.55, 0.45   # estimated speaking time per word / extra per sentence at the slow rate (-95)


def speech_s(text) -> float:
    """estimated seconds to speak `text` at the TUI's slow rate with its pauses (10 Nov trial: 'Not yet' came right
    after the first sentence)"""
    words = text.split()
    return WORD_S * len(words) + SENTENCE_S * sum(1 for w in words if w[-1:] in ".!?")
STEEPEST_DEG = 75.0      # a board further below / above the camera than this, seen from the camera: under the head
MAX_OFF_AXIS_DEG = 30.0  # the board's centre at most this far from the optical axis (image: +-40 deg tall)
HEIGHT_STEP_M = 0.01     # the stand's range is searched in these steps
BAND_STEP_M = 0.05       # depths of the target's band tried when its own is out of reach (phantom10: 1.6 m is)
BAND_MARGIN_M = 0.05     # ... this far inside the band's edges
DONE_TEXT = "Good. That one is done. Step back."
UNREACHABLE_TEXT = "This one is out of reach. Skipping."
NOPLACE_TEXT = "I cannot find a good spot. Press C to skip this one."
AUTO_OFF_RANGE_M = 1.5
SKIP_TEXT = "Skipping this one."
KEEP_HEIGHT = "Keep the same height."
TIMEOUT_S = 180.0
BUDGET_S = 60.0                 # design 10.4: target < 1 min per placement
YOUR_LEFT = "to YOUR left (the robot's right)"
YOUR_RIGHT = "to YOUR right (the robot's left)"
# the direction words of "Not yet. N more centimetres <dir>." per kind of correction
MORE = {"closer": "closer", "back": "further back", "far": "closer", "near": "further back", "raise": "higher",
        "lower": "lower", "cut_left": "to your left", "cut_right": "to your right", "cut_top": "lower",
        "cut_bottom": "higher", "side_left": "to your left", "side_right": "to your right"}
# the rounding each kind's amount is spoken with: "At least" amounts round up (the stand moves in coarse steps)
AT_LEAST = {"raise", "lower", "cut_top", "cut_bottom"}


@dataclass(frozen=True)
class Stand:
    """the operator's board stand: from the board's height at the start of a placement it goes up / down this far"""
    raise_max_m: float = 0.30
    lower_max_m: float = 0.40
    # 10 Nov trial: judging "out of reach" from a guessed range ended a placement the operator could reach. With
    # auto_skip off the guide searches every height (AUTO_OFF_RANGE_M around the board), never ends a placement by
    # itself, and only the operator's C skips
    auto_skip: bool = True


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
    # spoken: short plain sentences, one idea each (9 Nov trial: long sentences were not understood)
    if t.tilted:
        where = f"Board {_dist_words(t.distance_m)} in front of the robot. Lean it back on the chair."
    elif t.bearing_deg:
        side = "to your right" if t.bearing_deg > 0 else "to your left"
        where = f"Board {_dist_words(t.distance_m)} from the robot. Off {side}."
    elif t.distance_m < 1.0:
        where = f"Board {_dist_words(t.distance_m)} from the robot's head. Keep it high, near the head."
    else:
        where = f"Board {_dist_words(t.distance_m)} in front of the robot."
    return f"Placement {i} of {n}. {where}"


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


def _cm_up(m) -> int:
    return max(10, int(math.ceil(abs(m) * 10 - 1e-6)) * 10)            # to 10 cm, up: a minimum


def _amount(key, m) -> int:
    return _cm_up(m) if key in AT_LEAST else _cm(m)


def split(p, up):
    """(dh, dxy, h_dir): the height of camera point p above the camera, its distance along the floor, and the unit
    floor direction toward it (camera coordinates)"""
    p, up = np.asarray(p, float), np.asarray(up, float)
    dh = float(p @ up)
    h = p - dh * up
    dxy = float(np.linalg.norm(h))
    if dxy < 1e-6:                      # straight below / above: the floor direction of the optical axis
        h = np.array([0.0, 0.0, 1.0]) - up[2] * up
        n = float(np.linalg.norm(h))
        h = h / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
    else:
        h = h / dxy
    return dh, dxy, h


def floor_distance_for(depth, dh, up, h_dir):
    """the floor distance at which a board at height dh, in floor direction h_dir, is at `depth`; None when no
    usable one: nearly straight below / above the head, behind the camera, or too far off the optical axis"""
    if abs(dh) >= depth * math.sin(math.radians(STEEPEST_DEG)):
        return None
    hz = float(h_dir[2])
    if hz <= 1e-6:
        return None
    x = (depth - dh * float(up[2])) / hz           # depth = dh up_z + x h_z
    if x <= 0:
        return None
    if depth / math.hypot(dh, x) < math.cos(math.radians(MAX_OFF_AXIS_DEG)):
        return None
    return x


class PlacementGuide:
    def __init__(self, target: Target, focal, size=(1280, 800), pp=None, still_s=STILL_S, still_px=STILL_PX,
                 lost_s=LOST_S, repeat_s=REPEAT_S, ids=None, up=None, stand=None):
        self.target, self.f, self.size = target, float(focal), tuple(size)
        self.pp = pp or ((size[0] - 1) / 2.0, (size[1] - 1) / 2.0)
        self.still_s, self.still_px, self.lost_s, self.repeat_s = still_s, still_px, lost_s, repeat_s
        self.ids = ids or itertools.count(1)
        self.up = up                    # 'up' in camera 0: a vector, a callable giving one (or None), or None
        self.stand = stand or Stand()
        self.done = False
        self.result = None              # placed | unreachable, once done
        self.rest_t = None              # when the board came to rest for the accepted check
        self.h0 = None                  # the board's height (m above the camera) at the first rest that got an
                                        # instruction in this placement (the stand's base)
        self._unreach_n = 0             # separate rests judged out of reach, since the last reachable judgement
        self._unreach_rest = None       # the start of the last rest counted there
        self.geom = None                # (dh, dxy) of the last board seen, with gravity
        self.says = []
        self._start = None              # (t, centroid, z, dh) of the current still stretch
        self._spoken_at = None          # t of the last say in this still stretch
        self._lost_since = None
        self._lost_said = False
        self.last_obs = None
        self._last_key = None           # the last correction spoken (kind of problem), for feedback
        self._last_err = None
        self._repeats = 0

    def _say(self, t, kind, text):
        s = Say(next(self.ids), float(t), kind, text)
        self.says.append(s)
        self._spoken_at = t
        self._quiet_until = float(t) + speech_s(text) + GAP_S
        return s

    def _up(self):
        u = self.up() if callable(self.up) else self.up
        return None if u is None else np.asarray(u, float)

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
                return self._say(t, "lost", "I cannot see the board. Turn it to face the robot.")
            return None
        self._lost_since, self._lost_said = None, False
        up = self._up()
        sp = split(obs.p, up) if up is not None and getattr(obs, "p", None) is not None else None
        self.geom = None if sp is None else (sp[0], sp[1])
        c = obs.centroid
        tol = self.still_px if obs.n_tags >= FEW_TAGS else max(self.still_px, FEW_TAGS_STILL_PX)
        moved = self._start is None or math.hypot(c[0] - self._start[1][0], c[1] - self._start[1][1]) > tol or \
            abs(float(obs.z) - self._start[2]) > MOVE_Z_FRAC * self._start[2] or \
            (sp is not None and self._start[3] is not None and abs(sp[0] - self._start[3]) > MOVE_H_M)
        if moved:
            # a board moved straight toward or away from the camera keeps its centroid: distance and height count too
            self._start = (t, (float(c[0]), float(c[1])), float(obs.z), None if sp is None else sp[0])
            self._spoken_at = None
            return None
        if t - self._start[0] < self.still_s:
            return None
        if t < getattr(self, "_quiet_until", float("-inf")):
            return None                 # the last prompt is still being spoken / the operator is still acting on it
        if self._spoken_at is not None:
            # already spoke in this still stretch: repeat the same instruction at most once, after repeat_s
            if t - self._spoken_at < self.repeat_s or self._repeats >= 1:
                return None
            self._repeats += 1
        else:
            self._repeats = 0
        kind, text, key, err = self._judge(obs, up, sp)
        if kind == "good":
            self.done, self.result = True, "placed"
            self.rest_t = self._start[0]
            return self._say(t, kind, DONE_TEXT)
        if kind == "unreachable":
            # final, so a pause while carrying the board must not end the placement: two separate rests of a second
            # or one of three (review 9)
            rest = t - self._start[0]
            if rest < UNREACHABLE_STILL_S:
                return None
            if self._unreach_rest != self._start[0]:
                self._unreach_rest = self._start[0]
                self._unreach_n += 1
            if self._unreach_n < UNREACHABLE_RESTS and rest < UNREACHABLE_LONG_S:
                return None
            self.done, self.result = True, "unreachable"
            self.rest_t = self._start[0]
            return self._say(t, kind, UNREACHABLE_TEXT)
        self._unreach_n = 0
        if self.h0 is None and sp is not None:
            self.h0 = sp[0]                 # the stand's base: the first rest the operator gets an instruction at
        if self._spoken_at is None and self._last_key is not None:
            # the board was moved since the last instruction: tell the operator how that went
            if key != self._last_key:
                text = "Good. " + text
            elif key in MORE and err is not None and self._last_err is not None and err < self._last_err - SHRINK_M:
                text = f"Not yet. {_amount(key, err)} more centimetres {MORE[key]}."
            else:
                text = "Not yet. " + text
        self._last_key, self._last_err = key, err
        return self._say(t, kind, text)

    def interrupted(self):
        """the trial was interrupted (HOLD, a stalled camera): the board may have moved meanwhile, so the rest, the
        lost timer and the feedback on the last instruction start over (h0, the stand's base, is kept)"""
        self._start = self._spoken_at = self._lost_since = None
        self._lost_said = False
        self._last_key = self._last_err = None
        self._repeats = 0

    def bearing_deg(self, obs) -> float:
        """+ toward the robot's left (image left), neck at home"""
        return -math.degrees(math.atan2(obs.centroid[0] - self.pp[0], self.f))

    def judge(self, obs):
        """(kind, text) for a board at rest: the most important correction, unreachable, or good"""
        up = self._up()
        sp = split(obs.p, up) if up is not None and getattr(obs, "p", None) is not None else None
        kind, text, _, _ = self._judge(obs, up, sp)
        return {"good": (kind, DONE_TEXT), "unreachable": (kind, UNREACHABLE_TEXT)}.get(kind, (kind, text))

    def depths(self):
        """the depths to aim for, best first: the target's own, then the rest of its band (a capture counts toward
        the band it measured) in BAND_STEP_M steps, BAND_MARGIN_M inside its edges"""
        t = self.target
        D = t.distance_m
        lo, hi = t.accept
        if t.refuse_beyond_m is not None:
            hi = min(hi, t.refuse_beyond_m)
        hi = min(hi, D + 1.0)
        out = [D]
        for k in range(1, 40):
            for d in (D - k * BAND_STEP_M, D + k * BAND_STEP_M):
                if lo + BAND_MARGIN_M - 1e-9 <= d <= hi - BAND_MARGIN_M + 1e-9:
                    out.append(d)
        return out

    def _distance(self, obs, up, sp):
        """the distance correction with gravity: (kind, text, key, err). At this height if any depth of the band can
        be reached there (the target's own preferred), else the smallest height change within the stand's range"""
        dh, dxy, h = sp
        depths = self.depths()
        for d in depths:
            x = floor_distance_for(d, dh, up, h)
            if x is not None:
                move = dxy - x
                key, word = ("closer", "closer") if move > 0 else ("back", "further back")
                return "correction", f"Move the board {_cm(move)} centimetres {word}. {KEEP_HEIGHT}", key, abs(move)
        h0 = dh if self.h0 is None else self.h0
        if self.stand.auto_skip:
            lo, hi = h0 - self.stand.lower_max_m, h0 + self.stand.raise_max_m
        else:
            lo, hi = dh - AUTO_OFF_RANGE_M, dh + AUTO_OFF_RANGE_M
        n = int(math.floor((hi - lo) / HEIGHT_STEP_M + 1e-9))
        ok = [lo + k * HEIGHT_STEP_M for k in range(n + 1)]
        ok = [v for v in ok if any(floor_distance_for(d, v, up, h) is not None for d in depths)]
        if not ok:
            if not self.stand.auto_skip:
                return "correction", NOPLACE_TEXT, "noplace", None
            return "unreachable", UNREACHABLE_TEXT, "unreachable", None
        best = min(ok, key=lambda v: abs(v - dh))
        if best > dh:
            return "correction", f"Raise the board. At least {_cm_up(best - dh)} centimetres.", "raise", best - dh
        return "correction", f"Lower the board. At least {_cm_up(dh - best)} centimetres.", "lower", dh - best

    def _judge(self, obs, up=None, sp=None):
        """(kind, text, key, err): key names the problem (for feedback), err its size in metres (smaller = closer to
        done)"""
        t = self.target
        if obs.cut:
            side = obs.cut[0]
            z = obs.z
            span = {"left": obs.lo[0], "right": self.size[0] - 1 - obs.hi[0], "top": obs.lo[1],
                    "bottom": self.size[1] - 1 - obs.hi[1]}[side]
            m = max(0.1, (60.0 - span) * z / self.f + 0.05) if span < 60 else 0.1
            first = {"left": "Move the board to your left.", "right": "Move the board to your right.",
                     "top": "Lower the board.", "bottom": "Raise the board."}[side]
            # height on the chair-shaped stand moves only in coarse steps (operator, 9 Nov: "lower maybe 40 cm or raise
            # by 30 cm"), so up/down says the minimum; sideways stays approximate
            amount = "At least" if side in ("top", "bottom") else "About"
            return "correction", f"{first} {amount} {_cm(m)} centimetres.", f"cut_{side}", float(m)
        if obs.n_tags < FEW_TAGS and not t.tilted:
            return "correction", "Tilt the board back. Face it to the robot's head.", "tilt", None
        lo, hi = t.accept
        z = obs.z
        far = t.refuse_beyond_m is not None and z > t.refuse_beyond_m
        if far or z < lo or z >= hi:
            if sp is not None:
                return self._distance(obs, up, sp)
            # no gravity: the camera distance alone (the trial's first version)
            if far:
                return ("correction", f"Too far. Bring the board closer. About {_cm(z - t.distance_m)} centimetres.",
                        "far", z - t.distance_m)
            if z < lo:
                return ("correction", f"Move the board back. About {_cm(t.distance_m - z)} centimetres.", "near",
                        t.distance_m - z)
            if t.distance_m < 1.0 and obs.centroid[1] > self.pp[1] + 0.1 * self.size[1]:
                # 9 Nov trial: the camera is ~1.5 m up looking down; a board low in the image is far below the head, so
                # a near placement needs it raised toward the head, not moved along the floor ("it is at the feet")
                return ("correction", f"Raise the board, up toward the robot's head. At least {_cm(z - t.distance_m)} "
                        "centimetres.", "raise_head", z - t.distance_m)
            return ("correction", f"Bring the board closer. About {_cm(z - t.distance_m)} centimetres.", "far",
                    z - t.distance_m)
        if t.bearing_deg is not None and t.tol_deg is not None:
            b = self.bearing_deg(obs)
            if abs(b - t.bearing_deg) > t.tol_deg:
                m = z * (math.tan(math.radians(t.bearing_deg)) - math.tan(math.radians(b)))
                side = "to your right" if t.bearing_deg > b else "to your left"
                return ("correction", f"Move the board {side}. About {_cm(m)} centimetres.",
                        "side_right" if t.bearing_deg > b else "side_left", abs(m))
        return "good", DONE_TEXT, "good", 0.0


class Trial:
    """the timed placement trials (module text). gravity: an object with up() (and status()) or a callable giving
    'up' in camera 0, or None; stand: the operator's Stand"""

    def __init__(self, targets, log_path, timeout_s=TIMEOUT_S, focal=568.0, size=(1280, 800), budget_s=BUDGET_S,
                 gravity=None, stand=None, **guide_kw):
        self.targets = list(targets)
        self.log_path = Path(log_path)
        self.timeout_s, self.focal, self.size, self.budget_s = timeout_s, focal, size, budget_s
        self.gravity, self.stand = gravity, stand or Stand()
        self.guide_kw = guide_kw
        self.ids = itertools.count(1)
        self.i = -1
        self.guide = None
        self.t_intro = None
        self.paused_s = 0.0             # the current placement's time in HOLD (or with the camera stalled)
        self.t_last = None
        self.records = []
        self.finished = False
        self.started = False
        self.says = []
        if self.log_path.is_file():         # a restarted calibd: continue after the placements already logged
            for ln in self.log_path.read_text().splitlines():
                try:
                    self.records.append(json.loads(ln))
                except json.JSONDecodeError:
                    continue
            self.i = min(len(self.records), len(self.targets)) - 1

    def use_ids(self, ids):
        """the prompt-id counter to number from (calibd's: ids stay unique across a restart, so a TUI that skips the
        ids it has spoken keeps speaking); call before start()"""
        self.ids = ids

    def _up(self):
        g = self.gravity
        if g is None:
            return None
        return g.up() if hasattr(g, "up") else g()

    def gravity_status(self) -> dict:
        g = self.gravity
        if g is not None and hasattr(g, "status"):
            return g.status()
        u = self._up()
        if u is None:
            return {"ok": False, "up": None, "pitch_down_deg": None,
                    "note": "no IMU reading: placement guidance uses the camera distance only"}
        u = np.asarray(u, float)
        return {"ok": True, "up": [round(float(v), 4) for v in u],
                "pitch_down_deg": round(float(-np.degrees(np.arcsin(np.clip(u[2], -1, 1)))), 1), "note": ""}

    def _next(self, t):
        self.i += 1
        if self.i >= len(self.targets):
            self.finished = True
            self.guide = None
            s = Say(next(self.ids), float(t), "end", "That was the last placement. Thank you.")
            self.says.append(s)
            return [s]
        tg = self.targets[self.i]
        self.guide = PlacementGuide(tg, self.focal, self.size, ids=self.ids, up=self._up, stand=self.stand,
                                    **self.guide_kw)
        self.t_intro = t
        self.paused_s = 0.0
        s = Say(next(self.ids), float(t), "intro", intro(tg, self.i + 1, len(self.targets)))
        self.says.append(s)
        return [s]

    def resume(self, t, paused_s, say_intro=True) -> list:
        """after an interruption of paused_s (HOLD, a stalled camera): the placement's clock is shifted by it (a 200 s
        HOLD is not a timeout), the guide starts its rest over, and the current intro is said again"""
        if self.finished or self.guide is None:
            return []
        self.t_intro += float(paused_s)
        self.paused_s += float(paused_s)
        self.t_last = t
        self.guide.interrupted()
        if not say_intro:
            return []
        s = Say(next(self.ids), float(t), "intro", intro(self.targets[self.i], self.i + 1, len(self.targets)))
        self.says.append(s)
        return [s]

    def start(self, t) -> Say:
        self.started = True
        self.t_last = t
        return self._next(t)[0]

    def _record(self, t, result, rest_t=None):
        g = self.guide
        geom = None if g is None else g.geom
        rec = {"target": self.targets[self.i].name, "distance_m": self.targets[self.i].distance_m, "result": result,
               "seconds": round((rest_t if rest_t is not None else t) - self.t_intro, 2),
               "prompts": len(g.says) if g else 0, "t_intro": self.t_intro, "t_end": t,
               "paused_s": round(self.paused_s, 2),
               "z_m": None if g is None or g.last_obs is None else round(float(g.last_obs.z), 3),
               "dh_m": None if geom is None else round(geom[0], 3),
               "dxy_m": None if geom is None else round(geom[1], 3)}
        self.records.append(rec)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def update(self, t, obs) -> list:
        if self.finished or self.guide is None:
            return []
        self.t_last = t
        out = []
        s = self.guide.update(t, obs)
        if s is not None:
            self.says.append(s)
            out.append(s)
        if self.guide.done:
            self._record(t, self.guide.result or "placed", self.guide.rest_t)
            out += self._next(t)
        else:
            out += self.tick(t)
        return out

    def tick(self, t) -> list:
        """the placement's clock without a new detection (calibd calls it every step: a slow or stuck detector must
        not stop the timeout)"""
        if self.finished or self.guide is None or t - self.t_intro < self.timeout_s:
            return []
        self.t_last = t
        # 9 Nov trial: placements ran out silently; say it before moving on
        s = Say(next(self.ids), float(t), "skip", SKIP_TEXT)
        self.says.append(s)
        self._record(t, "timeout")
        return [s] + self._next(t)

    def key(self, t, k) -> list:
        if self.finished or self.guide is None:
            return []
        self.t_last = t
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

    def state(self, t=None) -> dict:
        """the CURRENT placement: before start the one that comes next (a restarted calibd continues after the
        logged ones), then the one being placed"""
        n = len(self.targets)
        live = self.guide is not None and not self.finished
        idx = n if self.finished else self.i + 1 if live else self.i + 2
        cur = self.targets[idx - 1] if 1 <= idx <= n and not self.finished else None
        t = self.t_last if t is None else t
        g = self.guide
        o = None if g is None else g.last_obs
        geom = None if g is None else g.geom
        board = None
        if o is not None:
            board = {"z_m": round(float(o.z), 3), "u": round(float(o.centroid[0]), 1),
                     "v": round(float(o.centroid[1]), 1), "tags": int(o.n_tags), "cut": list(o.cut),
                     "dh_m": None if geom is None else round(geom[0], 3),
                     "dxy_m": None if geom is None else round(geom[1], 3)}
        return {"started": self.started, "index": idx, "of": n,
                "target": None if cur is None else cur.name, "finished": self.finished,
                "since_intro_s": None if not live or t is None or self.t_intro is None else round(t - self.t_intro, 1),
                "timeout_s": self.timeout_s, "board": board, "gravity": self.gravity_status(),
                "stand": {"raise_max_m": self.stand.raise_max_m, "lower_max_m": self.stand.lower_max_m,
                          "h0_m": None if g is None or g.h0 is None else round(g.h0, 3)},
                "records": self.records, "summary": self.summary()}
