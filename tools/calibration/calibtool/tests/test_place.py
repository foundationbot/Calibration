"""calibtool.place: spoken placement guidance, pause-then-speak (design 4.1), and the placement-trial mode (neck at
home, nothing moves): the tool watches the board continuously but speaks only once it has been still ~1 s, says one
thing at a time ("Good. Step back." or one correction), names directions from the operator's side, and never says
"hold"."""
import json
import math
import re

import pytest

np = pytest.importorskip("numpy")

from calibtool import detect, place, protocol  # noqa: E402

F = 568.0
W, H = 1280, 800


def obs(z=1.1, u=640.0, v=400.0, cut=(), n_tags=35):
    c = np.array([u, v])
    return detect.BoardObs(c, z, n_tags, 4 * n_tags, bool(cut), tuple(cut), c - 150, c + 150)


def guide(target=None, **kw):
    return place.PlacementGuide(target or place.target("1.1 m"), focal=F, size=(W, H), **kw)


def feed(g, seq, t0=0.0, dt=0.5):
    """seq of observations at 2 Hz -> the Says"""
    out = []
    for i, o in enumerate(seq):
        s = g.update(t0 + i * dt, o)
        if s is not None:
            out.append(s)
    return out


def moving(n, z=1.1):
    return [obs(z=z, u=300 + 60 * i) for i in range(n)]


# ------------------------------------------------------------------ pause-then-speak

def test_silent_while_carried_then_one_say_after_a_second_still():
    g = guide()
    assert feed(g, moving(8)) == []
    says = feed(g, [obs()] * 4, t0=10.0)
    assert len(says) == 1
    assert says[0].kind == "good" and says[0].text == "Good. That one is done. Step back."
    assert g.done


def test_still_needs_0_4_s():
    """9 Nov trial: movement and rest were noticed too slowly (1.0 s, then 0.6 s)"""
    g = guide()
    assert g.update(0.0, obs()) is None
    assert g.update(0.3, obs()) is None
    assert g.update(0.4, obs()) is not None


def test_one_correction_then_quiet_until_it_moves_and_stops_again():
    g = guide()
    s = feed(g, [obs(z=0.8)] * 6)
    assert len(s) == 1 and s[0].kind == "correction"
    assert feed(g, [obs(z=0.8)] * 10, t0=3.0) == []             # same spot: no nagging
    assert feed(g, moving(4, z=1.0), t0=8.0) == []                # being moved: silent
    s = feed(g, [obs(z=1.12)] * 4, t0=10.0)
    assert [x.kind for x in s] == ["good"]


def test_a_correction_is_repeated_once_after_a_long_still():
    g = guide(repeat_s=20.0)
    assert len(feed(g, [obs(z=0.8)] * 4)) == 1
    s = feed(g, [obs(z=0.8)] * 44, t0=2.0)
    assert len(s) == 1 and s[0].kind == "correction"


# ------------------------------------------------------------------ what it says

def test_out_of_view_comes_before_distance():
    g = guide()
    s = feed(g, [obs(z=0.8, u=1150, cut=("right",))] * 4)
    assert len(s) == 1
    assert s[0].text == "Move the board to your right. About 20 centimetres."
    assert "back" not in s[0].text


@pytest.mark.parametrize("z,first,cm", [(0.8, "Move the board back.", 30), (1.45, "Bring the board closer.", 40),
                                         (0.9, "Move the board back.", 20)])
def test_distance_wording(z, first, cm):
    """spoken: one short instruction, then the amount (9 Nov trial: long sentences were not understood)"""
    s = feed(guide(), [obs(z=z)] * 4)
    assert s[0].text == f"{first} About {cm} centimetres."


def test_loose_tolerance_for_defaults_counts_the_band_it_measured():
    for z in (0.95, 1.0, 1.2, 1.3):
        s = feed(guide(), [obs(z=z)] * 4)
        assert (s[0].kind == "good") == (protocol.band_of(z)[0] == 1.1), z


def test_far_placement_limits():
    far = place.target("far")
    assert far.accept == protocol.FAR_ACCEPT_M
    assert feed(guide(far), [obs(z=1.9)] * 4)[0].kind == "good"
    s = feed(guide(far), [obs(z=2.15)] * 4)[0]
    assert s.kind == "correction" and s.text.startswith("Too far. Bring the board closer.")
    s = feed(guide(far), [obs(z=1.5)] * 4)[0]
    assert s.text.startswith("Move the board back.")


def test_cut_sides_name_the_operators_side():
    s = feed(guide(), [obs(u=100, cut=("left",))] * 4)[0].text
    assert s.startswith("Move the board to your left.")      # image left = robot's left: move toward the robot's right
    s = feed(guide(), [obs(v=760, cut=("bottom",))] * 4)[0].text
    assert s.startswith("Raise the board. At least ")      # the stand's height moves only in coarse steps
    s = feed(guide(), [obs(v=40, cut=("top",))] * 4)[0].text
    assert s.startswith("Lower the board. At least ")


def test_bearing_target_corrects_sideways():
    t = place.Target("left edge", 1.1, (1.0, 1.2), bearing_deg=20.0, tol_deg=5.0)
    s = feed(guide(t), [obs(u=640)] * 4)[0]              # straight ahead: needs to go 20 deg to the robot's left
    assert s.kind == "correction" and s.text.startswith("Move the board to your right.")
    u = 640 - F * np.tan(np.radians(20))                 # + bearing = robot's left = image left
    assert feed(guide(t), [obs(u=u)] * 4)[0].kind == "good"


def test_a_board_seen_at_a_steep_angle_is_asked_to_tilt_back():
    """9 Nov trial: the head looks steeply down; a vertical board in full view read 8 of 35 tags and the tool fell
    silent. Few tags with the board in view -> tilt it back to face the head (before distance corrections)."""
    s = feed(guide(), [obs(z=1.19, n_tags=8)] * 4)
    assert len(s) == 1 and s[0].kind == "correction"
    assert s[0].text == "Tilt the board back. Face it to the robot's head."


def test_few_tags_jitter_still_counts_as_resting():
    """with few tags the centroid jitters by more than still_px; it must still be judged"""
    seq = [obs(z=1.19, n_tags=8, u=640 + (9 if i % 2 else -9)) for i in range(6)]
    assert len(feed(guide(), seq)) == 1


def test_feedback_good_when_the_problem_changes_remaining_amount_when_it_shrinks_not_yet_otherwise():
    """9 Nov trial: the operator wanted positive feedback, and the same sentence repeated felt like parroting; after
    a partial move the tool says what is left ("Not yet. 40 more centimetres closer.")"""
    g = guide()                                                     # target 1.1 m
    s1 = feed(g, [obs(z=1.1, v=40, cut=("top",))] * 4)[0].text       # cut at the top
    assert s1.startswith("Lower the board.")
    s2 = feed(g, [obs(z=1.6)] * 4, t0=10.0)[0].text                  # moved: no longer cut, but too far
    assert s2.startswith("Good. Bring the board closer.")
    s3 = feed(g, [obs(z=1.45)] * 4, t0=20.0)[0].text                 # moved closer, still too far: what is left
    assert s3 == "Not yet. 40 more centimetres closer."
    s4 = feed(g, [obs(z=1.5, u=700)] * 4, t0=30.0)[0].text           # moved, but not closer
    assert s4.startswith("Not yet. Bring the board closer.")
    s5 = feed(g, [obs(z=1.1, u=640)] * 4, t0=40.0)[0]
    assert s5.kind == "good" and s5.text == "Good. That one is done. Step back."


def test_the_same_instruction_is_repeated_at_most_once_while_nothing_changes():
    g = guide()
    says = feed(g, [obs(z=1.6)] * 200)                               # 100 s, board never moves
    assert [x.text for x in says] == ["Bring the board closer. About 50 centimetres."] * 2


def test_a_near_placement_with_the_board_low_asks_to_raise_it_toward_the_head():
    """9 Nov trial: 'bring the board closer by 60 cm' while it stood at the robot's feet: the camera is ~1.5 m up,
    so a near placement needs the board raised toward the head"""
    t = place.target("0.75 m")
    s = feed(guide(t), [obs(z=1.15, v=600)] * 4)[0].text
    assert s.startswith("Raise the board, up toward the robot's head.")
    s = feed(guide(t), [obs(z=1.15, v=380)] * 4)[0].text            # board at the image centre: closer is right
    assert s.startswith("Bring the board closer.")
    assert "high, near the head" in place.intro(t, 2, 6)


def test_no_new_prompt_until_the_last_one_was_spoken_and_the_operator_had_time():
    """10 Nov trial: 'Not yet' came right after the first sentence, before the operator could act. Nothing is said
    until the last prompt's estimated speaking time plus GAP_S has passed, even if the board moved and rested."""
    g = guide()                                                     # target 1.1 m
    first = feed(g, [obs(z=1.6)] * 3)                               # says the correction at t=1.0
    assert len(first) == 1
    t1 = first[0].t
    quick = feed(g, [obs(z=1.55, u=660)] * 4, t0=t1 + 0.5)          # moved a little and rested at once
    assert quick == []
    later = feed(g, [obs(z=1.55, u=660)] * 4, t0=t1 + place.speech_s(first[0].text) + place.GAP_S + 0.1)
    assert len(later) == 1 and later[0].text.startswith("Not yet.")


def test_lost_board_prompt_once():
    g = guide(lost_s=6.0)
    says = feed(g, [None] * 30)
    assert [x.kind for x in says] == ["lost"]
    assert says[0].text == "I cannot see the board. Turn it to face the robot."


def test_never_robots_right_without_your_left_and_never_hold():
    texts = []
    for o in (obs(u=100, cut=("left",)), obs(u=1180, cut=("right",)), obs(z=0.7), obs(z=1.6), obs(v=40, cut=("top",)),
              obs(v=760, cut=("bottom",))):
        texts += [s.text for s in feed(guide(), [o] * 4)]
    t = place.Target("x", 1.1, (1.0, 1.2), bearing_deg=-20.0, tol_deg=5.0)
    texts += [s.text for s in feed(guide(t), [obs()] * 4)]
    texts += [place.intro(tg, i, 6) for i, tg in enumerate(place.DEFAULTS, 1)]
    for s in texts:
        assert "hold" not in s.lower(), s
        if "robot's right" in s:
            assert "YOUR left" in s, s
        if "robot's left" in s:
            assert "YOUR right" in s, s


def test_default_placements():
    names = [t.name for t in place.DEFAULTS]
    assert names == ["0.5 m", "0.75 m", "1.1 m", "1.6 m", "far", "tilted"]
    assert place.target("tilted").tilted and place.target("tilted").distance_m == 0.75
    assert place.target("far").distance_m == protocol.FAR_ASK_M


# ------------------------------------------------------------------ with gravity: height and floor distance
# 9 Nov trial (phantom10): the head camera is ~1.5 m up and looks steeply down. A board at the robot's feet is already
# ~1.1 m from the camera; the near placements need it high near the head, which the chair-shaped stand cannot reach
# (it goes up ~30 cm or down ~40 cm from its usual height, in coarse steps).

PITCH = 50.0                                        # camera pitched 50 deg below the horizon
STAND = place.Stand(raise_max_m=0.30, lower_max_m=0.40)


def up_cam(pitch=PITCH):
    """'up' in camera coordinates (x right, y down, z forward)"""
    a = np.radians(pitch)
    return np.array([0.0, -np.cos(a), -np.sin(a)])


def at(dh, dxy, pitch=PITCH, lateral=0.0, n_tags=35, cut=()):
    """the board centre dh above the camera (m, - below) and dxy in front of it along the floor, as the camera sees
    it: 3-D centre, depth, and the centroid in the nominal equidistant camera"""
    a = np.radians(pitch)
    p = np.array([lateral, -dh * np.cos(a) - dxy * np.sin(a), dxy * np.cos(a) - dh * np.sin(a)])
    rho = float(np.hypot(p[0], p[1]))
    k = F * np.arctan2(rho, p[2]) / rho if rho > 0 else 0.0
    c = np.array([(W - 1) / 2 + k * p[0], (H - 1) / 2 + k * p[1]])
    return detect.BoardObs(c, float(p[2]), n_tags, 4 * n_tags, bool(cut), tuple(cut), c - 150, c + 150, p)


def gguide(name, pitch=PITCH, stand=None, **kw):
    return guide(place.target(name), up=up_cam(pitch), stand=stand or STAND, **kw)


AT_FEET = (-1.2, 0.45)          # a board standing on the floor at the feet: centre 0.3 m up, 45 cm in front


def test_a_board_at_the_feet_is_already_in_the_1_1_m_band():
    o = at(*AT_FEET)
    assert protocol.band_of(o.z)[0] == 1.1
    s = feed(gguide("1.1 m"), [o] * 4)
    assert [x.text for x in s] == ["Good. That one is done. Step back."]


@pytest.mark.parametrize("name", ["0.75 m", "0.5 m", "tilted"])
def test_a_placement_the_stand_cannot_reach_is_skipped_at_once(name):
    """at the feet the board is 1.2 m below the camera; raised by the stand's 30 cm it is still 0.9 m below, too low
    for any board in the 0.75 m band (up to 0.925 m deep): say so after 3 s of rest and finish, no 180 s wait"""
    g = gguide(name)
    s = feed(g, [at(*AT_FEET)] * 8)
    assert [(x.kind, x.text) for x in s] == [("unreachable", "This one is out of reach. Skipping.")]
    assert g.done and g.result == "unreachable"


def test_a_short_pause_while_carrying_does_not_skip_a_placement():
    """the skip is final, so it waits for a full second of rest, not the 0.4 s a correction waits for"""
    g = gguide("0.75 m")
    assert feed(g, [at(*AT_FEET)] * 4, dt=0.2) == []               # 0.6 s at rest, then carried on
    assert feed(g, [at(-0.9, 0.9 + 0.1 * i) for i in range(6)], t0=1.0, dt=0.2) == []
    assert not g.done


def test_one_rest_while_carrying_low_neither_skips_nor_sets_the_stand_height():
    """review 9: a 1 s pause while carrying the board low ended the placement as unreachable for good, and its
    height became the stand's base (h0) for the rest of the placement"""
    g = gguide("0.75 m")
    assert feed(g, [at(*AT_FEET)] * 4) == []                       # 1.5 s at rest at the feet, then carried on
    assert not g.done and g.h0 is None
    assert feed(g, [at(-1.0, 0.9 + 0.1 * i) for i in range(6)], t0=2.0) == []
    s = feed(g, [at(-0.9, 0.9)] * 4, t0=10.0)                      # on the stand: 10 cm too low, within reach
    assert [(x.kind, x.text) for x in s] == [("correction", "Raise the board. At least 10 centimetres.")]
    assert g.h0 == pytest.approx(-0.9, abs=1e-6)


def test_unreachable_needs_two_separate_rests_or_three_seconds():
    g = gguide("0.75 m")
    assert feed(g, [at(*AT_FEET)] * 4) == []
    assert feed(g, [at(-1.2, 0.45 + 0.1 * i) for i in range(4)], t0=2.0) == []      # moved, then at rest again
    s = feed(g, [at(*AT_FEET)] * 4, t0=10.0)
    assert [x.kind for x in s] == ["unreachable"] and g.done
    g = gguide("0.75 m")
    s = feed(g, [at(*AT_FEET)] * 8, dt=0.5)                        # one rest: said after 3 s, not 1 s
    assert [x.kind for x in s] == ["unreachable"] and s[0].t == pytest.approx(place.UNREACHABLE_LONG_S)


def test_a_board_too_low_but_within_the_stands_reach_is_raised_first():
    g = gguide("0.75 m")
    s1 = feed(g, [at(-0.9, 0.9)] * 4)[0]
    assert (s1.kind, s1.text) == ("correction", "Raise the board. At least 10 centimetres.")
    s2 = feed(g, [at(-0.6, 0.9)] * 4, t0=10.0)[0].text            # raised 30 cm (one stand step)
    m = re.fullmatch(r"Good\. Move the board (\d+) centimetres (closer|further back)\. Keep the same height\.", s2)
    assert m, s2
    n = int(m.group(1)) / 100.0 * (1 if m.group(2) == "closer" else -1)
    s3 = feed(g, [at(-0.6, 0.9 - n)] * 4, t0=20.0)
    assert [x.kind for x in s3] == ["good"]


def test_a_far_board_too_high_to_be_seen_at_that_distance_is_lowered():
    """the camera looks down 50 deg: 1.8 m away, a board only 0.6 m below the camera would sit ~34 deg above the
    optical axis, at the image's top edge; a little lower (and at 1.75 m, still in the far band) it is within 30 deg"""
    s = feed(gguide("far"), [at(-0.6, 0.9)] * 4)[0]
    assert (s.kind, s.text) == ("correction", "Lower the board. At least 10 centimetres.")


@pytest.mark.parametrize("name,dh,dxy", [("1.6 m", -1.1, 0.45), ("1.1 m", -0.6, 2.0), ("far", -0.9, 1.2),
                                          ("0.75 m", -0.4, 1.7), ("1.1 m", -0.7, 0.3)])
def test_a_horizontal_correction_keeps_the_height_and_lands_in_the_band(name, dh, dxy):
    g = gguide(name)
    s = feed(g, [at(dh, dxy)] * 4)[0].text
    m = re.fullmatch(r"Move the board (\d+) centimetres (closer|further back)\. Keep the same height\.", s)
    assert m, s
    n = int(m.group(1)) / 100.0 * (1 if m.group(2) == "closer" else -1)
    s2 = feed(g, [at(dh, dxy - n)] * 4, t0=10.0)
    assert [x.kind for x in s2] == ["good"], [x.text for x in s2]


def test_partial_progress_says_what_is_left():
    """9 Nov trial: after 'move 90 cm closer' and a 60 cm move, say 'Not yet. 30 more centimetres closer.'"""
    a = np.radians(PITCH)
    x = (0.75 - 0.4 * np.sin(a)) / np.cos(a)                       # 0.4 m below the camera: depth 0.75 m here
    g = gguide("0.75 m")
    s1 = feed(g, [at(-0.4, x + 0.9)] * 4)[0].text
    assert s1 == "Move the board 90 centimetres closer. Keep the same height."
    s2 = feed(g, [at(-0.4, x + 0.3)] * 4, t0=10.0)[0].text         # moved 60 cm closer
    assert s2 == "Not yet. 30 more centimetres closer."
    s3 = feed(g, [at(-0.4, x + 0.4)] * 4, t0=20.0)[0].text         # moved, but back the wrong way
    assert s3 == "Not yet. Move the board 40 centimetres closer. Keep the same height."
    s4 = feed(g, [at(-0.4, x)] * 4, t0=30.0)[0]
    assert s4.kind == "good" and s4.text == "Good. That one is done. Step back."


def test_partial_progress_wording_for_every_direction():
    g = guide()
    feed(g, [obs(u=100, cut=("left",))] * 4)                        # cut on the left: move it to your left
    s = feed(g, [obs(u=40, cut=("left",))] * 4, t0=10.0)            # it went the wrong way: not smaller
    assert s[0].text.startswith("Not yet. Move the board to your left.")
    g = guide()
    feed(g, [obs(u=20, cut=("left",))] * 4)
    s = feed(g, [obs(u=60, cut=("left",))] * 4, t0=10.0)            # smaller
    assert re.fullmatch(r"Not yet\. \d+ more centimetres to your left\.", s[0].text), s[0].text
    g = gguide("0.75 m")
    feed(g, [at(-1.0, 0.6)] * 4)                                    # too low: raise
    s = feed(g, [at(-0.85, 0.6)] * 4, t0=10.0)
    assert re.fullmatch(r"Not yet\. \d+ more centimetres higher\.", s[0].text), s[0].text
    g = guide()
    feed(g, [obs(v=40, cut=("top",))] * 4)
    s = feed(g, [obs(v=4, cut=("top",))] * 4, t0=10.0)
    assert s[0].text.startswith("Not yet. Lower the board.")


def follow(g, dh, dxy, pitch=PITCH, steps=6):
    """an operator who does exactly what the guide says (the stand's height in whole 10 cm steps): the texts"""
    texts, t = [], 0.0
    for _ in range(steps):
        s = feed(g, [at(dh, dxy, pitch)] * 8, t0=t)            # 3.5 s at rest: long enough for "out of reach"
        t += 10.0
        if not s:
            break
        texts.append(s[0].text)
        if s[0].kind in ("good", "unreachable"):
            break
        m = re.search(r"(\d+) (?:more )?centimetres", s[0].text)
        n = int(m.group(1)) / 100.0
        if "closer" in s[0].text:
            dxy -= n
        elif "further back" in s[0].text:
            dxy += n
        elif "Raise" in s[0].text or "higher" in s[0].text:
            dh += n
        elif "Lower" in s[0].text or "lower" in s[0].text:
            dh -= n
    return texts, dh


@pytest.mark.parametrize("pitch", [50.0, 65.9])
def test_following_the_guide_ends_good_or_out_of_reach_never_beyond_the_stand(pitch):
    """from any resting place in front of the robot, for every default placement: the guide says at once that it is
    out of reach, or an operator following it (height only within the stand's range) gets 'Good' in a few steps"""
    for name in ("0.5 m", "0.75 m", "1.1 m", "1.6 m", "far"):
        for dh0 in (-1.2, -1.0, -0.8, -0.6, -0.4):
            for dxy0 in (0.3, 0.6, 0.9, 1.2, 1.6):
                if at(dh0, dxy0, pitch).z <= 0.2:
                    continue
                texts, dh = follow(gguide(name, pitch), dh0, dxy0, pitch)
                where = (name, dh0, dxy0, texts)
                assert texts[-1] in (place.DONE_TEXT, place.UNREACHABLE_TEXT), where
                if texts[-1] == place.UNREACHABLE_TEXT:
                    assert len(texts) == 1, where
                assert dh0 - STAND.lower_max_m - 1e-9 <= dh <= dh0 + STAND.raise_max_m + 1e-9, where


def test_a_band_depth_short_of_the_nominal_counts_when_the_nominal_is_out_of_reach():
    """phantom10 (camera 65.9 deg down, 9 Nov live): 1.6 m deep within 30 deg of the axis is near the floor, out of
    the stand's reach; the 1.6 m band starts at 1.35 m, which the trial reached (1.43 m) -> guide there, not skip"""
    texts, _ = follow(gguide("1.6 m", 65.9), -0.6, 0.72, pitch=65.9)
    assert texts[-1] == place.DONE_TEXT, texts


def test_a_height_change_counts_as_movement():
    """a board raised or lowered on the stand: centroid and distance can stay put, the height does not"""
    o1 = at(-0.6, 1.0)
    o2 = at(-0.6, 1.0)
    o2.p = o1.p + 0.06 * up_cam()                                   # 6 cm higher, same centroid and depth
    g = gguide("1.1 m")
    assert g.update(0.0, o1) is None
    assert g.update(0.3, o2) is None                                # moved: the stillness starts again
    assert g.update(0.45, o2) is None                               # 0.45 s after the first look, 0.15 s still
    assert g.update(0.75, o2) is not None


def test_without_gravity_the_guide_falls_back_to_the_camera_distance():
    g = guide(place.target("1.6 m"), up=lambda: None, stand=STAND)
    s = feed(g, [at(*AT_FEET)] * 4)[0].text
    assert s.startswith("Move the board back. About ")


# ------------------------------------------------------------------ the trial

class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_trial_logs_seconds_and_success(tmp_path):
    log = tmp_path / "trial.jsonl"
    tr = place.Trial([place.target("0.5 m"), place.target("1.1 m"), place.target("far")], log, timeout_s=180.0,
                     focal=F, size=(W, H))
    first = tr.start(0.0)
    assert first.kind == "intro" and "50 centimetres" in first.text
    t = 0.0
    for o in moving(20, z=0.9):                        # 10 s carrying
        t += 0.5
        tr.update(t, o)
    says = []
    for _ in range(3):
        t += 0.5
        s = tr.update(t, obs(z=0.52))
        says += s
    assert [s.kind for s in says] == ["good", "intro"]
    rec = [json.loads(x) for x in log.read_text().splitlines()]
    assert rec[0]["target"] == "0.5 m" and rec[0]["result"] == "placed"
    assert rec[0]["seconds"] == pytest.approx(10.5)        # from the intro to the board at rest
    t += 1.0
    assert tr.key(t, "C")[0].kind == "intro"              # can't put it there -> next
    t += 200.0
    says = tr.update(t, None)                           # far: no board for 200 s -> failed by timeout
    assert says[-1].kind == "end" and tr.finished
    rec = [json.loads(x) for x in log.read_text().splitlines()]
    assert [r["result"] for r in rec] == ["placed", "unreachable", "timeout"]
    s = tr.summary()
    assert (s["placements"], s["placed"], s["success_rate"]) == (3, 1, pytest.approx(1 / 3))
    assert s["median_s"] == pytest.approx(10.5) and s["over_budget"] == 0 and s["budget_s"] == 60.0


def test_trial_state_is_json_ready(tmp_path):
    tr = place.Trial(place.DEFAULTS, tmp_path / "t.jsonl", focal=F, size=(W, H))
    tr.start(0.0)
    json.dumps(tr.state())
    assert tr.state()["index"] == 1 and tr.state()["of"] == 6


def test_a_trial_resumes_after_the_logged_placements(tmp_path):
    """a calibd restarted under the same session (crash, reboot) continues with the next placement, not the first"""
    log = tmp_path / "trial.jsonl"
    targets = [place.target("0.5 m"), place.target("1.1 m"), place.target("far")]
    tr = place.Trial(targets, log, focal=F, size=(W, H))
    tr.start(0.0)
    tr.key(5.0, "C")
    tr2 = place.Trial(targets, log, focal=F, size=(W, H))
    s = tr2.start(100.0)
    assert s.kind == "intro" and s.text.startswith("Placement 2 of 3")
    assert [r["result"] for r in tr2.records] == ["unreachable"]
    assert tr2.state()["index"] == 2


def test_an_unreachable_placement_is_logged_and_the_trial_moves_on_at_once(tmp_path):
    log = tmp_path / "trial.jsonl"
    tr = place.Trial([place.target("0.75 m"), place.target("1.1 m")], log, focal=F, size=(W, H),
                     gravity=lambda: up_cam(), stand=STAND)
    tr.start(0.0)
    says = []
    t = 0.0
    for _ in range(25):
        t += 0.15
        says += tr.update(t, at(*AT_FEET))
    assert [s.text for s in says][:2] == ["This one is out of reach. Skipping.",
                                          "Placement 2 of 2. Board 1.1 metres in front of the robot."]
    rec = [json.loads(x) for x in log.read_text().splitlines()]
    assert rec[0]["result"] == "unreachable" and rec[0]["seconds"] < 2.0


def test_a_timeout_says_it_is_skipping(tmp_path):
    tr = place.Trial([place.target("1.6 m"), place.target("1.1 m")], tmp_path / "t.jsonl", timeout_s=180.0, focal=F,
                     size=(W, H))
    tr.start(0.0)
    assert tr.update(100.0, None) == []
    assert [s.kind for s in tr.update(179.0, None)] == ["lost"]
    says = tr.update(180.5, None)
    assert [(s.kind, s.text) for s in says] == [("skip", "Skipping this one."),
                                                ("intro", "Placement 2 of 2. Board 1.1 metres in front of the robot.")]
    assert tr.records[-1]["result"] == "timeout"


def test_the_trial_state_names_the_current_placement(tmp_path):
    """9 Nov trial: a stale 'placement 2 of 6' line misled the operator and the lead"""
    log = tmp_path / "trial.jsonl"
    targets = [place.target("0.5 m"), place.target("1.1 m"), place.target("far")]
    tr = place.Trial(targets, log, focal=F, size=(W, H))
    st = tr.state()
    assert (st["started"], st["index"], st["target"]) == (False, 1, "0.5 m")
    tr.start(0.0)
    tr.update(180.5, None)                                          # timeout: placement 2 now
    st = tr.state(181.0)
    assert (st["started"], st["index"], st["of"], st["target"]) == (True, 2, 3, "1.1 m")
    assert st["since_intro_s"] == pytest.approx(0.5) and st["timeout_s"] == 180.0
    tr2 = place.Trial(targets, log, focal=F, size=(W, H))            # restarted calibd, not started yet
    st = tr2.state()
    assert (st["started"], st["index"], st["target"]) == (False, 2, "1.1 m")


def test_the_trial_state_carries_gravity_and_the_boards_height(tmp_path):
    class G:
        def up(self):
            return up_cam()

        def status(self):
            return {"ok": True, "pitch_down_deg": 50.0, "note": "fake"}
    tr = place.Trial([place.target("1.1 m"), place.target("far")], tmp_path / "t.jsonl", focal=F, size=(W, H),
                     gravity=G(), stand=STAND)
    tr.start(0.0)
    tr.update(0.15, at(-0.6, 2.0))
    st = tr.state(0.15)
    json.dumps(st)
    assert st["gravity"]["pitch_down_deg"] == 50.0
    assert st["board"]["dh_m"] == pytest.approx(-0.6, abs=1e-3) and st["board"]["dxy_m"] == pytest.approx(2.0, abs=1e-3)


def test_without_auto_skip_the_operator_decides_reachability():
    """10 Nov trial: 'This one is out of reach. Skipping.' was judged from where the board first rested plus a guessed
    stand range, and ended a placement the operator could reach. With auto_skip off (phantom10's profile) the guide
    searches all heights, says what is needed, never ends a placement by itself, and only C (the operator) skips."""
    stand = place.Stand(auto_skip=False)
    g = gguide("0.75 m", stand=stand)
    s = feed(g, [at(*AT_FEET)] * 12)                                # 6 s at rest at the robot's feet
    assert s and all(x.kind == "correction" for x in s) and not g.done
    assert s[0].text.startswith("Raise the board. At least ")      # beyond the old +30 cm window, still asked for
    assert "out of reach" not in " ".join(x.text for x in s).lower()


def test_profile_reads_the_stands_auto_skip():
    from calibtool import profile
    assert profile.load("P10").stand_auto_skip is False


@pytest.mark.parametrize("limit", [None, 0, 0.0])
def test_a_trial_without_a_time_limit_never_skips_by_itself(tmp_path, limit):
    """10 Nov (phantom10): 'Skipping this one.' came without warning after 3 minutes while the operator was still
    trying. With no limit (the profile's null / 0) only the operator's C ends a placement"""
    tr = place.Trial([place.target("0.5 m"), place.target("1.1 m")], tmp_path / "t.jsonl", timeout_s=limit, focal=F,
                     size=(W, H))
    tr.start(0.0)
    says = []
    for k in range(1, 2001):
        says += tr.update(k * 0.5, None) if k % 2 else tr.tick(k * 0.5)
    assert "skip" not in [s.kind for s in says] and tr.records == [] and not tr.finished
    assert tr.state(1000.0)["timeout_s"] is None
    assert tr.key(1000.0, "C")[0].text.startswith("Placement 2 of 2.")
    assert [r["result"] for r in tr.records] == ["unreachable"]



# ------------------------------------------------------------------ near placements: close enough (10 Nov, phantom10)
# The 0.5 m placement bounced between "Lower" and "Raise": a near board raised gets cut by the image's top edge. After
# two corrections that did not bring the board more than ~3 cm nearer, a board within 15 cm of a near target is used
# where it is.

QUIET = {"status_s": math.inf}         # rests 20 s apart with no frames between: no "Still waiting" in these tests


def rest(g, o, t0):
    """the board at rest for 1.5 s from t0 (rests 20 s apart: the last prompt has been spoken and acted on)"""
    return feed(g, [o] * 4, t0=t0)


def test_a_near_board_that_stops_improving_is_close_enough():
    g = guide(place.target("0.5 m"), **QUIET)
    assert rest(g, obs(z=0.66, v=40, cut=("top",)), 0.0)[0].text.startswith("Lower the board.")
    assert rest(g, obs(z=0.70, v=600), 20.0)[0].text.startswith("Good. Raise the board")
    s = rest(g, obs(z=0.64, v=40, cut=("top",)), 40.0)
    assert [(x.kind, x.text) for x in s] == [("good", "Close enough. Using 64 centimetres.")]
    assert g.done and g.result == "placed_near" and g.near_z == pytest.approx(0.64)


def test_a_near_board_still_getting_closer_is_guided_on():
    g = guide(place.target("0.5 m"), **QUIET)
    rest(g, obs(z=0.80, v=600), 0.0)
    rest(g, obs(z=0.72, v=600), 20.0)
    s = rest(g, obs(z=0.64, v=40, cut=("top",)), 40.0)                 # 16 cm better than two corrections ago
    assert s[0].kind == "correction" and not g.done


def test_close_enough_needs_two_corrections_first():
    g = guide(place.target("0.5 m"), **QUIET)
    rest(g, obs(z=0.64, v=40, cut=("top",)), 0.0)
    s = rest(g, obs(z=0.64, v=40, cut=("top",), u=700), 20.0)          # one correction so far
    assert s[0].kind == "correction" and not g.done


@pytest.mark.parametrize("name,z", [("0.5 m", 0.70), ("1.1 m", 1.40)])
def test_close_enough_only_within_15_cm_of_a_near_target(name, z):
    g = guide(place.target(name), **QUIET)
    for i in range(4):
        cut = ("top",) if i % 2 == 0 else ()
        s = rest(g, obs(z=z, v=40 if cut else 600, cut=cut, u=640 + 20 * i), 20.0 * i)
        assert s and s[0].kind == "correction", (i, [x.text for x in s])
    assert not g.done


def test_close_enough_with_gravity_and_the_operator_deciding_reachability():
    """phantom10's profile (auto_skip off): raised, the board is cut at the top; lowered, it is too far"""
    g = gguide("0.5 m", stand=place.Stand(auto_skip=False), **QUIET)
    o1 = at(-0.25, 0.62, cut=("top",))
    o1.lo[1] = 2.0
    o2 = at(-0.45, 0.55)
    o3 = at(-0.27, 0.62, cut=("top",))
    o3.lo[1] = 2.0
    assert all(abs(o.z - 0.5) <= 0.15 for o in (o1, o3)) and o2.z >= 0.625
    assert rest(g, o1, 0.0)[0].kind == "correction"
    assert rest(g, o2, 20.0)[0].kind == "correction"
    s = rest(g, o3, 40.0)
    assert s[0].kind == "good" and s[0].text == f"Close enough. Using {int(round(o3.z * 100))} centimetres."
    assert g.result == "placed_near"


def test_the_trial_logs_a_close_enough_placement_with_its_distance(tmp_path):
    tr = place.Trial([place.target("0.5 m"), place.target("1.1 m")], tmp_path / "t.jsonl", focal=F, size=(W, H),
                     **QUIET)
    tr.start(0.0)
    says = []
    for t0, o in ((1.0, obs(z=0.66, v=40, cut=("top",))), (21.0, obs(z=0.70, v=600)),
                  (41.0, obs(z=0.64, v=40, cut=("top",)))):
        for i in range(4):
            says += tr.update(t0 + 0.5 * i, o)
    assert "Close enough. Using 64 centimetres." in [x.text for x in says]
    assert tr.records[0]["result"] == "placed_near" and tr.records[0]["z_m"] == pytest.approx(0.64)
    assert tr.summary()["placed"] == 1



# ------------------------------------------------------------------ more communication (10 Nov, phantom10)
# A board not seen only got "I cannot see the board. Turn it to face the robot." after 6 s. The guide now says why
# (from detect.board_miss), once per reason until it changes; "Keep the board still." when the board keeps appearing
# and vanishing; and "Still waiting. ..." when nothing has been said for STATUS_S.

def miss(reason, cut=(), n_tags=3, tag_px=30.0, bright=120.0):
    return detect.Miss(reason, n_tags, tag_px, bright, tuple(cut))


def feed_miss(g, m, seconds, t0=0.0, dt=0.5):
    out = []
    for i in range(int(round(seconds / dt))):
        s = g.update(t0 + i * dt, None, miss=m)
        if s is not None:
            out.append(s)
    return out


@pytest.mark.parametrize("m,text", [
    (miss("dark", n_tags=0, bright=20.0), "It is too dark for me to see the board."),
    (miss("far", tag_px=9.0), "The board is too far for me to read. Bring it closer."),
    (miss("partial", cut=("left",)), "I can see only part of the board. Move the board to your left."),
    (miss("partial", cut=("right",)), "I can see only part of the board. Move the board to your right."),
    (miss("partial", cut=("top",)), "I can see only part of the board. Lower the board."),
    (miss("partial", cut=("bottom",)), "I can see only part of the board. Raise the board."),
    (miss("few"), "I can read only a few tags. Turn the board to face the robot."),
])
def test_a_board_not_seen_gets_its_reason_said_once(m, text):
    says = feed_miss(guide(), m, 12.0)
    assert [x.text for x in says] == [text]
    assert says[0].t == pytest.approx(place.MISS_S)


def test_nothing_found_in_a_bright_image_waits_lost_s_as_before():
    says = feed_miss(guide(lost_s=6.0), miss("none", n_tags=0), 12.0)
    assert [(x.t, x.text) for x in says] == [(6.0, "I cannot see the board. Turn it to face the robot.")]


def test_a_new_reason_is_said_and_an_old_one_again_once_it_changed():
    g = guide()
    a = feed_miss(g, miss("dark", n_tags=0, bright=20.0), 12.0)
    b = feed_miss(g, miss("far", tag_px=9.0), 12.0, t0=12.0)
    c = feed_miss(g, miss("dark", n_tags=0, bright=20.0), 12.0, t0=24.0)
    assert [x.text for x in a + b + c] == ["It is too dark for me to see the board.",
                                         "The board is too far for me to read. Bring it closer.",
                                         "It is too dark for me to see the board."]


def test_a_reason_waits_until_the_last_prompt_was_spoken():
    g = guide()
    s1 = feed(g, [obs(z=1.6)] * 3)
    assert len(s1) == 1
    s2 = feed_miss(g, miss("far", tag_px=9.0), 10.0, t0=1.5)
    assert len(s2) == 1 and s2[0].t >= s1[0].t + place.speech_s(s1[0].text) + place.GAP_S


def test_a_board_that_keeps_appearing_and_vanishing_is_asked_to_keep_still():
    """tags found in one step and not the next while the operator moves: blur"""
    g = guide()
    says = []
    for i in range(24):                                             # 12 s, seen every other step
        o = obs(z=1.3, u=400 + 20 * i) if i % 2 == 0 else None
        s = g.update(i * 0.5, o, miss=None if o else miss("none", n_tags=0))
        if s is not None:
            says.append(s)
    assert [x.text for x in says] == ["Keep the board still."]
    assert says[0].t > place.UNSETTLED_S


def test_keep_still_only_after_it_does_not_settle_for_unsettled_s():
    g = guide()
    says = []
    for i in range(7):                                              # 3 s of flicker, then at rest in the band
        o = obs(z=1.3, u=400 + 20 * i) if i % 2 == 0 else None
        s = g.update(i * 0.5, o, miss=None if o else miss("none", n_tags=0))
        says += [s] if s else []
    says += feed(g, [obs(z=1.1)] * 4, t0=3.5)
    assert [x.text for x in says] == [place.DONE_TEXT]


def test_a_board_carried_steadily_is_not_asked_to_keep_still():
    assert "Keep the board still." not in [x.text for x in feed(guide(), moving(20))]


def test_a_status_after_status_s_of_silence_while_the_board_is_carried():
    g = guide(place.target("0.5 m"))
    says = feed(g, [obs(z=1.1, u=300 + 30 * (i % 20)) for i in range(40)])          # 20 s, always moving
    assert [(x.kind, x.text) for x in says] == [
        ("status", "Still waiting. Board 110 centimetres away; I need about 50.")]
    assert says[0].t == pytest.approx(place.STATUS_S)


def test_a_status_repeats_while_the_board_is_not_seen_with_the_reason():
    says = feed_miss(guide(lost_s=6.0), miss("none", n_tags=0), 40.0)
    assert [(x.t, x.text) for x in says] == [
        (6.0, "I cannot see the board. Turn it to face the robot."),
        (21.0, "Still waiting. I cannot see the board. Turn it to face the robot."),
        (36.0, "Still waiting. I cannot see the board. Turn it to face the robot.")]
    says = feed_miss(guide(), miss("far", tag_px=9.0), 20.0)
    assert [x.text for x in says][-1] == "Still waiting. The board is too far for me to read. Bring it closer."


def test_a_status_never_comes_into_a_still_board_with_its_instruction_standing():
    """the repeat rule governs a board at rest after an instruction: one repeat after repeat_s, no nagging"""
    says = feed(guide(), [obs(z=1.6)] * 200)
    assert [x.kind for x in says] == ["correction", "correction"]


def test_a_status_does_not_take_the_rest_of_its_still_stretch():
    """a status said while the board settles must not count as the rest's instruction"""
    g = guide()
    s1 = feed(g, [obs(z=1.6, u=300 + 30 * (i % 20)) for i in range(40)])             # 20 s carried: the status
    assert [x.kind for x in s1] == ["status"]
    says = feed(g, [obs(z=1.6, u=600)] * 30, t0=20.0)
    assert [x.kind for x in says] == ["correction"]


def test_new_sentences_never_say_hold_or_robots_right_alone():
    texts = list(place.MISS_TEXT.values()) + list(place.CUT_FIRST.values()) + [place.STILL_TEXT, place.near_text(0.63)]
    texts += [x.text for x in feed_miss(guide(), miss("none", n_tags=0), 40.0)]
    for s in texts:
        assert "hold" not in s.lower(), s
        if "robot's right" in s:
            assert "YOUR left" in s, s
        if "robot's left" in s:
            assert "YOUR right" in s, s


def test_the_trial_passes_the_reason_to_the_guide(tmp_path):
    tr = place.Trial([place.target("0.5 m")], tmp_path / "t.jsonl", focal=F, size=(W, H))
    tr.start(0.0)
    says = []
    for i in range(1, 12):
        says += tr.update(i * 0.5, None, miss=miss("dark", n_tags=0, bright=20.0))
    assert "It is too dark for me to see the board." in [x.text for x in says]
