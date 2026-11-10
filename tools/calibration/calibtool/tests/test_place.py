"""calibtool.place: spoken placement guidance, pause-then-speak (design 4.1), and the placement-trial mode (neck at
home, nothing moves): the tool watches the board continuously but speaks only once it has been still ~1 s, says one
thing at a time ("Good. Step back." or one correction), names directions from the operator's side, and never says
"hold"."""
import json

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
    assert says[0].kind == "good" and says[0].text == "Good. Step back."
    assert g.done


def test_still_needs_a_full_second():
    g = guide()
    assert g.update(0.0, obs()) is None
    assert g.update(0.5, obs()) is None
    assert g.update(1.0, obs()) is not None


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
    for _ in range(4):
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
