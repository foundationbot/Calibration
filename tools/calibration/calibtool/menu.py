"""`calib` with no arguments: the operator's menu (10 Nov 2026: "i want tool to be launched and tool to ask which robot
it should use and i can say phantom10"; the scripted path needed `--full --session <id> --replace --voice piper`
remembered, and a crashed calibd left a held producer lease and a session behind).

Plain numbered prompts (input()), nothing to remember:
  1. Which robot? The robots with a profile (config/profiles.yaml), by Thor host name and bot name; a number, the host
     name or the bot name; Enter = the one used last time. "new robot (setup)" runs calibtool.onboard.run(host) when
     that module exists.
  2. One ssh call looks at the robot (PROBE, read-only, stdlib, needs no kit there): every calibd answering on its
     session socket (session, phase, kit -- the same liveness rule as `calibd status`), the producer lease, the
     unfinished sessions (state.json phase not DONE/ABORTED/STOPPED_LOUD).
  3. Only what makes sense is offered:
     - a live calibd: continue it (attached as it is when it runs this laptop's kit; restarted with this kit when it
       runs other code, which is what --session ID --replace did), stop it, status; never a second session beside it;
     - no live calibd: continue an unfinished session (calibd restarted under its id, it comes back in HOLD), start a
       new calibration, the placement trial, status. When a dead session still holds the producer lease, a new
       session first runs the robot's own self-restore for it (`python -m calibtool restore --if-stale-hours 0`:
       producer as saved, OKVIS back, the lease released), shows its output, and starts nothing if it failed.
  4. The voice: asked once per start ("1) Piper voice 2) text on screen"; text without Piper), remembered in
     ~/.config/calibtool/laptop.json (with the robot) and offered as the default next time; the menu works without
     that file.
The kit is always copied (cli.bootstrap) before calibd is started or restarted. Stdlib only."""
import importlib
import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from calibtool import handeyeflow, cli, daemon, profile as prof_mod, robots

ENDED = frozenset(daemon.TERMINAL) | {"DISCARDED"}     # DONE, ABORTED, STOPPED_LOUD, discarded: nothing to continue
DISCARD_MARK = "calibtool-menu-discard"
# 10 Nov: "when i delete old sessions they are not deleted": a session can be discarded from the menu. Its folder is
# kept (data retention, the report, rollback.sh); its state.json phase becomes DISCARDED, so it is no longer offered
DISCARD = r'''# calibtool-menu-discard
import json, os, sys, time
p = os.path.join(sys.argv[1], "sessions", sys.argv[2], "state.json")
with open(p) as f:
    st = json.load(f)
st["discarded_from"], st["phase"], st["discarded_t"] = st.get("phase"), "DISCARDED", time.time()
with open(p + ".tmp", "w") as f:
    json.dump(st, f)
os.replace(p + ".tmp", p)
print("discarded " + sys.argv[2])
'''
SHOW_UNFINISHED = 3                 # the newest unfinished sessions offered to continue
# 10 Nov: espeak is not offered ("why is espeak even an option?"); a PC without Piper shows the prompts as text
VOICES = (("piper", "Piper voice (recommended)"), ("text", "text on screen, no voice"))
PROBE_MARK = "calibtool-menu-probe"

# Read-only look at <root> (argv[1]) on the robot, stdlib only (no kit needed): the lease, the lock's last holder, the
# unfinished sessions (and any the lease, lock or `current` names), each with its calibd's first state line when its
# socket answers (= live, as calibd status decides). Prints one JSON line.
PROBE = r'''# calibtool-menu-probe
import glob, json, os, socket, sys, time
root = sys.argv[1]
def rd(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None
def first_line(sock):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(1.5)
    try:
        s.connect(sock)
        line = s.makefile("rb").readline()
        return json.loads(line) if line else None
    except Exception:
        return None
    finally:
        s.close()
lease = rd(os.path.join(root, "producer.lease"))
lock = rd(os.path.join(root, "calibd.lock"))
try:
    with open(os.path.join(root, "current")) as f:
        current = f.read().strip() or None
except Exception:
    current = None
named = set(x for x in ((lease or {}).get("session"), (lock or {}).get("session"), current) if x)
found = []
for sj in glob.glob(os.path.join(root, "sessions", "*", "state.json")):
    st = rd(sj) or {}
    found.append((float(st.get("created") or 0), os.path.basename(os.path.dirname(sj)), st))
found.sort(reverse=True)
out, n = [], 0
for created, sid, st in found:
    ended = st.get("phase") in ("DONE", "ABORTED", "STOPPED_LOUD", "DISCARDED")
    if sid not in named:
        if ended or n >= 6:
            continue
        n += 1
    line = first_line(os.path.join(root, "sessions", sid, "calibd.sock"))
    out.append({"id": sid, "phase": (line or {}).get("phase") or st.get("phase"), "mode": st.get("mode"),
                "created": created or None, "updated": st.get("updated"), "held_from": st.get("held_from"),
                "live": line is not None, "pid": (line or {}).get("pid"), "kit": (line or {}).get("kit")})
print(json.dumps({"now": time.time(), "lease": lease, "lock": lock, "current": current, "sessions": out}))
'''


# ---------------------------------------------------------------- the laptop's memory


def config_path() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "calibtool" / "laptop.json"


def load_cfg(path) -> dict:
    try:
        d = json.loads(Path(path).read_text())
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def save_cfg(path, d):
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, indent=1))
        os.replace(tmp, path)
    except OSError:
        pass                                # a laptop without a writable config just asks again next time


# ---------------------------------------------------------------- looking at the robot


@dataclass
class Look:
    sessions: list = field(default_factory=list)
    lease: object = None
    now: float = 0.0
    error: object = None

    @property
    def live(self) -> list:
        return [s for s in self.sessions if s.get("live")]

    @property
    def unfinished(self) -> list:
        return [s for s in self.sessions if not s.get("live") and s.get("phase") not in ENDED][:SHOW_UNFINISHED]

    @property
    def dead_lease(self):
        """the lease when the session holding it has no live calibd (the robot was left changed)"""
        if not self.lease or not self.lease.get("session"):
            return None
        return None if any(s["id"] == self.lease["session"] for s in self.live) else self.lease


def probe_line(p) -> str:
    return (f"P={shlex.quote(p.python)}; [ -x \"$P\" ] || P=python3; \"$P\" -c {shlex.quote(PROBE)} "
            f"{shlex.quote(p.root)}")


def look(p, run=subprocess.run) -> Look:
    """one ssh call, read-only"""
    try:
        r = run(cli.ssh_argv(p, probe_line(p)), capture_output=True, text=True, timeout=45)
    except (OSError, subprocess.TimeoutExpired) as e:
        return Look(error=f"{type(e).__name__}: {e}")
    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "").strip().splitlines()
        return Look(error=(tail[-1] if tail else f"ssh exited {r.returncode}"))
    try:
        d = cli._json_out(r)
    except ValueError:
        return Look(error=f"the robot's answer is not JSON: {(r.stdout or '').strip()[-200:]}")
    return Look(list(d.get("sessions") or []), d.get("lease"), float(d.get("now") or time.time()))


def started(s, today=None) -> str:
    t = s.get("created")
    if not t:
        return "?"
    lt = time.localtime(float(t))
    if time.strftime("%Y%m%d", lt) == (today or time.strftime("%Y%m%d")):
        return time.strftime("%H:%M", lt)
    return time.strftime("%d %b %H:%M", lt)


def _ago(s) -> str:
    s = max(0.0, float(s))
    return f"{s:.0f} s" if s < 120 else f"{s / 60:.0f} min" if s < 7200 else f"{s / 3600:.1f} h"


def describe(p, lk, kit) -> list:
    L = []
    if not lk.live:
        L.append("  calibd: not running")
    for s in lk.live:
        same = "the same kit as this laptop" if s.get("kit") == kit else \
            f"OTHER code (kit {s.get('kit')}; this laptop has {kit})"
        L.append(f"  calibd: running session {s['id']}, phase {s.get('phase')}, {same}")
    dl = lk.dead_lease
    if dl:
        hb = dl.get("heartbeat")
        age = f", last heartbeat {_ago(lk.now - float(hb))} ago" if hb else ""
        L.append(f"  producer lease: held by session {dl['session']}, whose calibd is not running{age}:")
        L.append("    the cameras may still be at the calibration size and OKVIS stopped")
    elif lk.lease:
        L.append(f"  producer lease: held by the running session {lk.lease.get('session')} (normal)")
    if lk.unfinished:
        L.append("  unfinished sessions: " + "; ".join(f"{s['id']} (phase {s.get('phase')}, started {started(s)})"
                                                      for s in lk.unfinished))
    return L


# ---------------------------------------------------------------- the menu


@dataclass
class Option:
    text: str
    kind: str                           # continue | new | trial | status | stop | discard | retry | quit
    session: object = None


def options(lk, kit, handeye=False) -> list:
    if lk.error:
        return [Option("try again", "retry"), Option("quit", "quit")]
    o = []
    for s in lk.live:
        how = "" if s.get("kit") == kit else (": it runs other code; it is restarted with this laptop's kit and comes "
                                              "back in HOLD")
        o.append(Option(f"continue session {s['id']} (phase {s.get('phase')}, started {started(s)}){how}",
                        "continue", s))
    if not lk.live:
        for s in lk.unfinished:
            o.append(Option(f"continue session {s['id']} (phase {s.get('phase')}, started {started(s)}): calibd is "
                            "not running; it is restarted with this laptop's kit and comes back in HOLD",
                            "continue", s))
        dl = lk.dead_lease
        first = (f" (first restores the robot from session {dl['session']}: producer as saved, OKVIS back)"
                 if dl else "")
        o.append(Option(f"start a new calibration{first}", "new"))
        o.append(Option(f"placement trial, nothing moves{first}", "trial"))
        if handeye:
            o.append(Option("hand-eye calibration: both arms, the neck and the hands move (the URDF is fitted, not "
                            "deployed)", "handeye"))
    o.append(Option("status only", "status"))
    for s in lk.live:
        o.append(Option(f"stop the running session {s['id']}", "stop", s))
    if not lk.live:
        for s in lk.unfinished:
            o.append(Option(f"discard session {s['id']} (the robot put back first if it changed it; its data is "
                            "kept; no longer offered)", "discard", s))
    o.append(Option("quit", "quit"))
    return o


class Menu:
    def __init__(self, input_fn=input, out=print, run=subprocess.run, cfg_path=None, kit=None, profiles=None,
                 attach=None, sync=None, import_module=importlib.import_module, voice=None):
        self.input, self.out, self.run = input_fn, out, run
        self.cfg_path = Path(cfg_path) if cfg_path is not None else config_path()
        self.cfg = load_cfg(self.cfg_path)
        self.kit = kit or daemon.kit_hash()
        self.profiles = profiles if profiles is not None else all_profiles()
        self.attach = attach or (lambda p, sid, voice: cli.attach_tui(p, sid, cli.make_speaker(voice)))
        self.sync = sync or (lambda p: cli.bootstrap(p, run=self.run))
        self.import_module = import_module
        self.voice = voice                  # given on the command line: not asked

    # ------------------------------------------------ prompts

    def ask(self, prompt):
        """the answer, stripped; None for q / EOF / Ctrl-C"""
        try:
            a = self.input(prompt)
        except (EOFError, KeyboardInterrupt):
            self.out("")
            return None
        a = (a or "").strip()
        return None if a.lower() in ("q", "quit", "exit") else a

    def choose(self, title, opts, default=None):
        """an Option by its number (Enter = default, an index); None = leave"""
        self.out(title)
        for i, o in enumerate(opts, 1):
            self.out(f"  {i}) {o.text}")
        hint = f"Enter = {default + 1}" if default is not None else f"1-{len(opts)}"
        while True:
            a = self.ask(f"Choice [{hint}]: ")
            if a is None:
                return None
            if a == "" and default is not None:
                return opts[default]
            if a.isdigit() and 1 <= int(a) <= len(opts):
                return opts[int(a) - 1]
            self.out(f"  type a number from 1 to {len(opts)} (q = quit)")

    # ------------------------------------------------ which robot

    def pick_robot(self):
        """("robot", Profile) | ("setup", None) | None"""
        last = self.cfg.get("robot")
        self.out("Which robot?")
        for i, p in enumerate(self.profiles, 1):
            self.out(f"  {i}) {p.thor} ({p.bot})")
        self.out(f"  {len(self.profiles) + 1}) new robot (setup)")
        default = next((p for p in self.profiles if p.thor == last), None)
        hint = f"number, host or bot name; Enter = {default.thor}, used last time" if default else \
            "number, host or bot name"
        while True:
            a = self.ask(f"Robot [{hint}]: ")
            if a is None:
                return None
            if a == "" and default is not None:
                return "robot", default
            p = self.match(a)
            if p is not None:
                return "robot", p
            if a == str(len(self.profiles) + 1) or a.lower() in ("new", "new robot", "setup", "new robot (setup)"):
                return "setup", None
            self.out(f"  {a!r} is not in the list: type its number, its host name (phantom10) or its bot name (P10); "
                     "a robot not listed needs the setup first")

    def match(self, a):
        a = a.strip().lower().split("@")[-1]
        if a.isdigit() and 1 <= int(a) <= len(self.profiles):
            return self.profiles[int(a) - 1]
        return next((p for p in self.profiles if a in (p.thor.lower(), p.bot.lower())), None)

    def setup(self) -> int:
        host = self.ask("Host name of the new robot's Thor (e.g. phantom13): ")
        if not host:
            return 0
        try:
            onboard = self.import_module("calibtool.onboard")
        except ImportError:
            self.out("setup is not available yet: the robot needs a profile in config/profiles.yaml first")
            return 1
        r = onboard.run(host, ask=self.input, out=self.out)
        if isinstance(r, int) or r is None:
            return int(r or 0)
        st = getattr(r, "status", None)
        self.out(f"setup: {st}")
        return 0 if st in ("done", "nothing-to-do", "dry-run", "declined") else 1

    def pick_voice(self) -> str:
        """Piper or text on screen; the menu's text says the prompt is on screen either way"""
        if self.voice:
            return self.voice
        keys = [k for k, _ in VOICES]
        last = self.cfg.get("voice") if self.cfg.get("voice") in keys else "piper"
        if not piper_available():
            self.out("  Piper is not installed on this PC: the prompts are shown as text on screen "
                     "(`calib --install-voice` installs it).")
            return "text"
        d = keys.index(last)
        line = "  ".join(f"{i}) {t}" for i, (_, t) in enumerate(VOICES, 1))
        while True:
            a = self.ask(f"Voice: {line}  [Enter = {d + 1}]: ")
            if a is None or a == "":
                v = last
                break
            if a.isdigit() and 1 <= int(a) <= len(VOICES):
                v = keys[int(a) - 1]
                break
            if a.lower() in keys:
                v = a.lower()
                break
            self.out(f"  type 1 or 2")
        self.cfg["voice"] = v
        save_cfg(self.cfg_path, self.cfg)
        return v

    # ------------------------------------------------ the run

    def main(self, robot=None) -> int:
        if robot is not None:
            p = self.match(robot)
            if p is None:
                try:
                    p = cli.robot_profile(robot)
                except KeyError as e:
                    self.out(f"calib: {e}")
                    return 2
        else:
            picked = self.pick_robot()
            if picked is None:
                return 0
            if picked[0] == "setup":
                return self.setup()
            p = picked[1]
        self.cfg["robot"] = p.thor
        save_cfg(self.cfg_path, self.cfg)
        while True:
            self.out(f"Looking at {p.thor} ({p.bot}) ...")
            lk = look(p, self.run)
            if lk.error:
                self.out(f"  Could not look at {p.thor}: {lk.error}")
            else:
                for line in describe(p, lk, self.kit):
                    self.out(line)
            o = self.choose("What now?", options(lk, self.kit, handeye=handeyeflow.available(p)))
            if o is None or o.kind == "quit":
                return 0
            rc = self.act(p, lk, o)
            if rc is not None:
                return rc

    def act(self, p, lk, o):
        """-> an exit code (the TUI ran), or None: back to the menu"""
        if o.kind == "retry":
            return None
        if o.kind == "status":
            return self.status(p, lk)
        if o.kind == "stop":
            a = self.ask(f"Stop session {o.session['id']}? calibd stops; the session can be continued later "
                         "[y/N]: ")
            if (a or "").lower() in ("y", "yes"):
                r = cli.stop(p, o.session["id"], run=self.run)
                self.out(f"  stopped session {o.session['id']}" if r.get("stopped") else
                         f"  could not stop it: {r.get('error') or r}")
            return None
        if o.kind == "continue":
            return self.cont(p, o.session)
        if o.kind == "discard":
            return self.discard(p, lk, o.session)
        if o.kind == "handeye":
            return handeyeflow.run_session(p, run=self.run, out=self.out, ask=self.ask)
        return self.new(p, lk, trial=o.kind == "trial")

    def discard(self, p, lk, s):
        """the robot put back first when this session still holds the producer (its own restore), then the session
        marked DISCARDED (folder kept); the motors are not touched"""
        sid = s["id"]
        if (self.ask(f"Discard session {sid}? It is no longer offered; its folder stays on the robot [y/N]: ")
                or "").lower() not in ("y", "yes"):
            return None
        dl = lk.dead_lease
        if dl and dl.get("session") == sid and not self._restore(p, sid):
            return None
        line = f"python3 - {shlex.quote(p.root)} {shlex.quote(sid)} <<'EOF'\n{DISCARD}EOF"
        try:
            r = self.run(cli.ssh_argv(p, line), capture_output=True, text=True, timeout=60)
            ok, text = r.returncode == 0, (r.stdout or "") + (r.stderr or "")
        except (OSError, subprocess.TimeoutExpired) as e:
            ok, text = False, f"{type(e).__name__}: {e}"
        self.out(f"  discarded session {sid} (its folder stays on the robot; the motors are as they were)" if ok else
                 f"  could not discard {sid}: {text.strip()[-200:]}")
        return None

    def _restore(self, p, sid) -> bool:
        self.out(f"Session {sid} still holds the camera producer: running the robot's own restore for it ...")
        try:
            r = self.run(cli.ssh_argv(p, cli.restore_line(p, 0)), capture_output=True, text=True, timeout=600)
            rc, text = r.returncode, (r.stdout or "") + (r.stderr or "")
        except (OSError, subprocess.TimeoutExpired) as e:
            rc, text = 1, f"{type(e).__name__}: {e}"
        for line in text.strip().splitlines():
            self.out(f"  | {line}")
        if rc != 0:
            self.out("The restore did not finish (its output is above): nothing else was done.")
        return rc == 0

    def status(self, p, lk):
        s = (lk.live or lk.unfinished or [None])[0]
        if s is None:
            self.out("  no running or unfinished session")
            return None
        st = cli.status(p, s["id"], run=self.run) or {}
        self.out(f"  session {s['id']}: phase {st.get('phase', s.get('phase'))}, mode {st.get('mode')}, "
                 f"{'calibd running' if st.get('live') else 'calibd not running'}, kit {st.get('kit')}")
        if st.get("held_from"):
            self.out(f"  holding since {st['held_from']} ({st.get('hold_reason') or 'the link was lost'})")
        for b in st.get("banner") or ():
            self.out(f"  NOT NORMAL: {b}")
        return None

    def _start(self, fn):
        try:
            return fn()
        except RuntimeError as e:
            self.out(f"  calib: {e}")
            return None

    def cont(self, p, s):
        voice = self.pick_voice()
        sid = s["id"]
        if s.get("live") and s.get("kit") == self.kit:
            self.out(f"Attaching to session {sid} ...")
        else:
            why = (f"calibd of session {sid} runs other code (kit {s.get('kit')}): stopping it, copying this "
                   "laptop's kit and starting it again" if s.get("live") else
                   f"calibd of session {sid} is not running: copying the kit and starting it again")
            self.out(f"{why}. It comes back in HOLD where the robot could move: press G when ready.")
            sid = self._start(lambda: cli.start_or_attach(p, sid, run=self.run, kit=self.kit,
                                                          replace=bool(s.get("live")), sync=lambda: self.sync(p)))
            if sid is None:
                return None
        return self.attach(p, sid, voice)

    def new(self, p, lk, trial=False):
        voice = self.pick_voice()
        self.out("Copying the kit to the robot ...")
        self.sync(p)
        dl = lk.dead_lease
        if dl:
            self.out(f"Session {dl['session']} stopped with the robot changed and still holds the producer lease: "
                     "running the robot's own restore for it ...")
            try:
                r = self.run(cli.ssh_argv(p, cli.restore_line(p, 0)), capture_output=True, text=True, timeout=600)
                rc, text = r.returncode, (r.stdout or "") + (r.stderr or "")
            except (OSError, subprocess.TimeoutExpired) as e:
                rc, text = 1, f"{type(e).__name__}: {e}"
            for line in text.strip().splitlines():
                self.out(f"  | {line}")
            if rc != 0:
                self.out("The restore did not finish (its output is above): nothing was started. Fix the robot or "
                         "ask for help; the session folder has rollback.sh and commands.txt.")
                return None
            if self.ask("Robot restored. Press Enter to start (q = back to the menu): ") is None:
                return None
        what = "placement trial" if trial else "calibration"
        self.out(f"Starting a new {what} ...")
        sid = self._start(lambda: cli.start_or_attach(p, None, trial=trial, run=self.run, kit=self.kit,
                                                      mode=None if trial else "full"))
        if sid is None:
            return None
        return self.attach(p, sid, voice)


def all_profiles() -> list:
    """every robot with a calibd profile, by host name"""
    import yaml
    raw = (yaml.safe_load(prof_mod.PROFILES_YAML.read_text()) or {}).get("profiles") or {}
    reg = robots.load()
    out = []
    for bot in raw:
        try:
            out.append(prof_mod.load(bot, registry=reg))
        except (KeyError, ValueError):
            continue
    return sorted(out, key=lambda p: p.thor)


def main(robot=None, voice=None, **kw) -> int:
    return Menu(voice=voice, **kw).main(robot)


def piper_available() -> bool:
    try:
        from calibtool import tui
        return tui.PiperSay().available()
    except Exception:  # noqa: BLE001
        return False
