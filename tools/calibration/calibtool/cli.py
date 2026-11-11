"""`calib`: the laptop entry point (design plan sections 3, 10.3).

calib                    the menu (calibtool.menu): which robot, then only what makes sense for it (continue a session,
                         a new calibration, the placement trial, status, stop), the voice asked once and remembered
calib phantom10          the same menu for that robot
calib --install [--install-dir DIR]   builds the single-file executable (calibtool.exe) into DIR/calib (~/.local/bin)
calib --version          the kit hash this laptop runs (the same from the worktree and from the executable)
calib phantom10 [--trial | --full [--dry-run] | --watch] [--session ID] [--replace] [--headless N]
    [--voice espeak|piper|text] [--no-speech] [--stop] [--status] [--say TEXT] [--no-sync]   the scripted path:
  - asks calibd's status first (live = its socket answers, with the kit hash it runs): a live session running this
    kit is attached as it is; one running other code, or another session, is refused unless --replace stops it;
  - copies the kit (tools/calibration/{calibtool,scripts,config}, code only) to <root>/kit on the robot (rsync; M2: the
    hand-made venv runs it; the CI-built calib image replaces this, 10.3);
  - starts calibd there (niced, at most 2 BLAS threads, the profile's python + dma_ipc folder) or, with --session ID
    of a live session, re-attaches without starting anything;
  - runs the TUI over `ssh robot python -m calibtool.daemon attach` (state lines in, heartbeat and keys out). Leaving
    the TUI (Q) leaves calibd running: it holds after 5 s without a heartbeat and waits for G. A dropped link does
    not leave (10 Nov: every network blip exited the screen): the screen says so and re-attaches with backoff (1, 2,
    4, 8, then every 10 s) until it is back or Q; keys pressed meanwhile are not sent (not queued);
  - --stop asks calibd to stop; --status prints its state; --say TEXT speaks TEXT once on the laptop (a speech test).
ssh, rsync and the speaker are injected for the tests."""
import argparse
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

from calibtool import daemon, paths, profile as prof_mod, tui

KIT_DIRS = daemon.KIT_DIRS


def kit_dir(p) -> str:
    return f"{p.root}/kit/tools/calibration"


def kit_python(p) -> str:
    """the shell line's start that runs the robot's kit python (niced, at most 2 BLAS threads)"""
    py = ":".join(x for x in (kit_dir(p), p.pydeps) if x)
    return (f"cd {shlex.quote(kit_dir(p))} && PYTHONPATH={py} OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 "
            f"MKL_NUM_THREADS=2 nice -n 10 {p.python}")


def remote(p, cmd, *args) -> str:
    """the shell line that runs calibd's CLI on the robot"""
    tail = " ".join(shlex.quote(a) for a in args)
    return (f"{kit_python(p)} -m calibtool.daemon {cmd} --root {p.root} --profile {p.bot}"
            + (f" {tail}" if tail else ""))


def restore_line(p, hours=0) -> str:
    """the robot's own stale-lease self-restore (calibtool.selfrestore) for <root>; hours 0 = whatever the lease's
    age (only when no calibd is alive: it refuses itself while a calibd holds the lock)"""
    return f"{kit_python(p)} -m calibtool restore --if-stale-hours {hours:g} --root {p.root}"


def ssh_argv(p, line) -> list:
    return ["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=2", "-o", "ServerAliveCountMax=3",
            "-o", "ConnectTimeout=10", p.host, line]


URDF_REL = "config/foundation_rig/20250206_phantom_mk1.urdf"     # the neck chain's robot model (placement.URDF)


def bootstrap(p, run=subprocess.run):
    urdf_dir = f"{p.root}/kit/{Path(URDF_REL).parent}"
    run(ssh_argv(p, f"mkdir -p {kit_dir(p)} {urdf_dir}"), check=True, timeout=60)
    src = [str(paths.CAL / d) for d in KIT_DIRS]
    run(["rsync", "-a", "--delete", "--exclude=__pycache__", "--exclude=tests", "--exclude=results",
         "--exclude=*.npz", *src, f"{p.host}:{kit_dir(p)}/"], check=True, timeout=300)
    # 10 Nov: the planner's neck chain reads the URDF from the repo's config/ (the calib image carries it too)
    run(["rsync", "-a", str(paths.CAL.parents[1] / URDF_REL), f"{p.host}:{urdf_dir}/"], check=True, timeout=120)


def _json_out(r):
    out = (r.stdout or "").strip().splitlines()
    return json.loads(out[-1]) if out else {}


def start(p, session=None, trial=False, run=subprocess.run, mode=None, dry_run=False) -> dict:
    """mode "full": the whole calibration session (M5: motors, grids, deploy); dry_run: its robot changes recorded
    in <session>/commands.txt, none sent"""
    args = (["--session", session] if session else []) + (["--trial"] if trial else []) + (
        ["--session-mode", "full"] if mode == "full" else []) + (["--dry-run"] if dry_run else [])
    r = run(ssh_argv(p, remote(p, "start", *args)), capture_output=True, text=True, timeout=90)
    return _json_out(r)


def status(p, session=None, run=subprocess.run) -> dict:
    r = run(ssh_argv(p, remote(p, "status", *(["--session", session] if session else []))), capture_output=True,
            text=True, timeout=30)
    return _json_out(r) or {}


def start_or_attach(p, session=None, trial=False, run=subprocess.run, kit=None, replace=False, sync=None, mode=None,
                    dry_run=False) -> str:
    """the session to attach to: a live one running this kit as it is; a dead one restarted under its id (it resumes in
    HOLD); else a new one. A live calibd running other code (the kit changed since it started), or another session
    running when a new one is asked for, is refused unless replace (then it is stopped first). Liveness = the session
    socket answers (calibd status). sync() copies the kit, only once a start is decided (never under a running
    daemon). Review 6: start used to report success while the old daemon kept running the old code."""
    kit = kit or daemon.kit_hash()
    st = status(p, session, run=run) or {}
    if st.get("live"):
        running = st.get("session") or session
        if session is None and not replace:
            raise RuntimeError(f"calibd is already running session {running}: `calib {p.thor} --session {running}` "
                               f"re-attaches; --replace stops it and starts a new session")
        if session is not None and st.get("kit") != kit and not replace:
            raise RuntimeError(f"calibd of session {running} runs other code (kit {st.get('kit')}, this laptop has "
                               f"{kit}): the kit changed since it started. --replace stops it and starts it again with "
                               "this code (it resumes in HOLD); or attach with the matching code")
        if not replace:
            return session
        r = stop(p, running, run=run)
        if not r.get("stopped"):
            raise RuntimeError(f"could not stop calibd of session {running}: {r}")
    if sync is not None:
        sync()
    res = start(p, session=session, trial=trial, run=run, mode=mode, dry_run=dry_run)
    if res.get("error") or not res.get("session"):
        raise RuntimeError(f"calibd did not start: {res}")
    return res["session"]


def stop(p, session=None, run=subprocess.run) -> dict:
    r = run(ssh_argv(p, remote(p, "stop", *(["--session", session] if session else []))), capture_output=True,
            text=True, timeout=60)
    return _json_out(r)


class Link:
    """the attach link: ssh running `calibd attach`, state lines on its stdout, our lines on its stdin. quiet: ssh's
    own errors not printed (a re-attach attempt: "No route to host" over the screen helps nobody)"""

    def __init__(self, p, session, popen=subprocess.Popen, quiet=False):
        kw = {"stderr": subprocess.DEVNULL} if quiet else {}
        self.proc = popen(ssh_argv(p, remote(p, "attach", "--session", session)), stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, text=True, bufsize=1, **kw)
        self._lock = threading.Lock()

    def lines(self):
        try:
            for line in self.proc.stdout:
                yield line
        except (OSError, ValueError):         # closed under us (Q while re-attaching)
            return

    def send(self, s):
        with self._lock:
            try:
                self.proc.stdin.write(s + "\n")
                self.proc.stdin.flush()
            except (OSError, ValueError):
                pass

    def close(self):
        try:
            self.proc.stdin.close()
        except (OSError, AttributeError):
            pass
        try:
            self.proc.terminate()
        except OSError:
            pass


def _keys(t: tui.Tui, quit_ev):
    """single keys from the terminal (cbreak), forwarded; Q leaves"""
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while not quit_ev.is_set():
            ch = sys.stdin.read(1)
            if not ch:
                break
            if ch.upper() == "Q":
                quit_ev.set()
                break
            if ch.strip():
                t.key(ch)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


BACKOFF_S = (1, 2, 4, 8, 10)     # re-attach waits (s); the last one repeats until the link is back or Q


class _Down:
    """a link that never came up (ssh could not even start)"""

    def lines(self):
        return iter(())

    def send(self, s):
        pass

    def close(self):
        pass


def run_tui(link, speaker, headless=0, out=None, interactive=False, reconnect=None, robot="the robot", wait=None,
            backoff=BACKOFF_S) -> int:
    """the TUI loop over a link; headless N: take N states then leave, drawing each changed one (no 2-a-second limit).
    A screen is drawn only when its text changed (10 Nov: the whole screen was re-printed for every state line), at
    most tui.MIN_DRAW_S apart; the last one is drawn when leaving. reconnect() -> a new link: when the link drops the
    TUI stays (10 Nov: every network blip exited the screen), says so (NOT NORMAL, attempt N) and re-attaches after
    backoff[0], [1], ... (the last repeated) until state lines come again ("Connection back") or Q; heartbeats and keys
    are not sent meanwhile (keys are not queued). Without reconnect a dropped link leaves, as before. wait(s) -> True
    when leaving (tests); default: Q during the wait. -> states taken"""
    term = not headless and out is None and sys.stdout.isatty()
    clear = "" if headless or out is not None else "\033[2J\033[H"
    out = out or (lambda s: (sys.stdout.write(clear + s + "\n"), sys.stdout.flush()))
    cur = {"link": link}
    t = tui.Tui(lambda s: cur["link"].send(s), speaker, out=out, ansi=term,
                min_draw_s=0.0 if headless else tui.MIN_DRAW_S)
    quit_ev = threading.Event()
    wait = wait or quit_ev.wait

    def beat():
        while not quit_ev.is_set():
            t.heartbeat()
            quit_ev.wait(0.25)

    def closer():                       # Q while a re-attach hangs in ssh: close it so the loop sees the end
        quit_ev.wait()
        try:
            cur["link"].close()
        except Exception:  # noqa: BLE001
            pass
    t.heartbeat()
    threading.Thread(target=beat, daemon=True).start()
    if interactive:
        threading.Thread(target=_keys, args=(t, quit_ev), daemon=True).start()
        threading.Thread(target=closer, daemon=True).start()
    n, attempt = 0, 0
    try:
        while True:
            leave = False
            for line in cur["link"].lines():
                if attempt:
                    attempt = 0
                    t.link_back()
                t.step(line)
                n += 1
                if (headless and n >= headless) or quit_ev.is_set():
                    leave = True
                    break
            if leave or quit_ev.is_set() or reconnect is None:
                break
            attempt += 1
            t.link_lost(robot, attempt)
            if wait(backoff[min(attempt, len(backoff)) - 1]) or quit_ev.is_set():
                break
            try:
                cur["link"].close()
            except Exception:  # noqa: BLE001
                pass
            try:
                cur["link"] = reconnect()
            except OSError:
                cur["link"] = _Down()
    finally:
        quit_ev.set()
        t.flush()
        cur["link"].close()
    return n


def make_speaker(voice, no_speech=False):
    return tui.Silent() if no_speech else tui.make_speaker(voice)


def attach_tui(p, sid, speaker, headless=0, popen=subprocess.Popen) -> int:
    """the TUI on session sid of robot p, re-attaching by itself when the link drops (not headless)"""
    sp = speaker if isinstance(speaker, (tui.Silent, tui.TextPrompts)) else tui.Background(speaker)
    n = run_tui(Link(p, sid, popen=popen), sp, headless=headless,
                interactive=not headless and sys.stdin.isatty(), robot=p.thor,
                reconnect=None if headless else (lambda: Link(p, sid, popen=popen, quiet=True)))
    print(f"left the TUI after {n} states; calibd keeps running (session {sid}): `calib` (or calib {p.thor} "
          f"--session {sid}) re-attaches, calib {p.thor} --stop stops it", file=sys.stderr)
    time.sleep(0.2)
    return 0


def robot_profile(name):
    """the profile of a robot named by bot name (P10) or host name (phantom10)"""
    return prof_mod.load(name.upper()) if name.upper().startswith("P") and name[1:].isdigit() else \
        prof_mod.for_host(name)


SCRIPTED = ("trial", "full", "watch", "session", "headless", "stop", "status", "say", "replace", "dry_run")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="calib", description=__doc__.split("\n")[0])
    ap.add_argument("robot", nargs="?", default=None,
                    help="host name (phantom10) or bot name (P10); none: the menu asks. Alone: the menu for it")
    ap.add_argument("--install", action="store_true",
                    help="build the single-file executable `calib` (the kit inside) and install it")
    ap.add_argument("--install-dir", default="~/.local/bin", help="where --install puts calib")
    ap.add_argument("--version", action="store_true", help="print the kit hash this laptop runs and where the kit is")
    ap.add_argument("--watch", action="store_true", help="a read-only watch session (what a bare robot name did "
                                                         "before the menu)")
    ap.add_argument("--session", default=None, help="re-attach to (or restart) this session")
    ap.add_argument("--trial", action="store_true", help="placement-trial mode (neck at home, nothing moves)")
    ap.add_argument("--full", action="store_true",
                    help="the whole calibration session (motors, neck grids, deploy): M5")
    ap.add_argument("--dry-run", action="store_true",
                    help="with --full: every robot change recorded in the session's commands.txt, none sent")
    ap.add_argument("--headless", type=int, default=0, help="render N states and leave (no keys)")
    ap.add_argument("--no-speech", action="store_true")
    ap.add_argument("--voice", choices=["espeak", "piper", "text"], default=None,
                    help="piper = local neural voice in ~/.local/share/calib-voice (falls back to espeak if missing); "
                         "text = no voice: the prompts in a panel at the top of the screen. Default espeak; the "
                         "menu asks once and remembers")
    ap.add_argument("--no-sync", action="store_true", help="do not copy the kit to the robot")
    ap.add_argument("--replace", action="store_true",
                    help="stop a calibd already running (other code, or another session) and start this one")
    ap.add_argument("--stop", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--say", default=None, help="speak this once on the laptop and leave (speech test)")
    a = ap.parse_args(argv)
    if a.version:
        print(f"calib kit {daemon.kit_hash()} ({paths.CAL})")
        return 0
    if a.install:
        from calibtool import exe
        return exe.install(a.install_dir)
    if sum(map(bool, (a.full, a.trial, a.watch))) > 1:
        ap.error("--full, --trial and --watch are different sessions: pick one")
    if a.dry_run and not a.full:
        ap.error("--dry-run is for a --full session")
    if not any(getattr(a, k) for k in SCRIPTED):
        from calibtool import menu
        return menu.main(robot=a.robot, voice=a.voice)
    if a.robot is None and not a.say:
        ap.error("a robot is needed with these options (or run `calib` alone for the menu)")
    speaker = make_speaker(a.voice or "espeak", a.no_speech)
    if a.say:
        if isinstance(speaker, tui.TextPrompts):
            print(a.say)
        else:
            speaker.say(a.say)
        return 0
    p = robot_profile(a.robot)
    if a.stop:
        print(json.dumps(stop(p, a.session)))
        return 0
    if a.status:
        print(json.dumps(status(p, a.session), indent=1))
        return 0
    try:
        sid = start_or_attach(p, a.session, a.trial, replace=a.replace, sync=None if a.no_sync else
                              (lambda: bootstrap(p)), mode="full" if a.full else None, dry_run=a.dry_run)
    except RuntimeError as e:
        print(f"calib: {e}", file=sys.stderr)
        return 2
    return attach_tui(p, sid, speaker, headless=a.headless)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    sys.exit(main())
