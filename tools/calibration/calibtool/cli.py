"""`calib <robot>`: the laptop entry point (design plan sections 3, 10.3).

calib phantom10 [--trial] [--session ID] [--headless N] [--no-speech] [--stop] [--status] [--say TEXT] [--no-sync]
  - copies the kit (tools/calibration/{calibtool,scripts,config}, code only) to <root>/kit on the robot (rsync; M2: the
    hand-made venv runs it; the CI-built calib image replaces this, 10.3);
  - starts calibd there (niced, at most 2 BLAS threads, the profile's python + dma_ipc folder) or, with --session ID
    of a live session, re-attaches without starting anything;
  - runs the TUI over `ssh robot python -m calibtool.daemon attach` (state lines in, heartbeat and keys out). Leaving
    the TUI (Q, or the link dropping) leaves calibd running: it holds after 5 s without a heartbeat and waits for G;
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

from calibtool import paths, profile as prof_mod, tui

KIT_DIRS = ("calibtool", "scripts", "config")


def kit_dir(p) -> str:
    return f"{p.root}/kit/tools/calibration"


def remote(p, cmd, *args) -> str:
    """the shell line that runs calibd's CLI on the robot"""
    py = ":".join(x for x in (kit_dir(p), p.pydeps) if x)
    tail = " ".join(shlex.quote(a) for a in args)
    return (f"cd {shlex.quote(kit_dir(p))} && PYTHONPATH={py} OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 "
            f"MKL_NUM_THREADS=2 nice -n 10 {p.python} -m calibtool.daemon {cmd} --root {p.root} --profile {p.bot}"
            + (f" {tail}" if tail else ""))


def ssh_argv(p, line) -> list:
    return ["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=2", "-o", "ServerAliveCountMax=3",
            "-o", "ConnectTimeout=10", p.host, line]


def bootstrap(p, run=subprocess.run):
    run(ssh_argv(p, f"mkdir -p {kit_dir(p)}"), check=True, timeout=60)
    src = [str(paths.CAL / d) for d in KIT_DIRS]
    run(["rsync", "-a", "--delete", "--exclude=__pycache__", "--exclude=tests", "--exclude=results",
         "--exclude=*.npz", *src, f"{p.host}:{kit_dir(p)}/"], check=True, timeout=300)


def _json_out(r):
    out = (r.stdout or "").strip().splitlines()
    return json.loads(out[-1]) if out else {}


def start(p, session=None, trial=False, run=subprocess.run) -> dict:
    args = (["--session", session] if session else []) + (["--trial"] if trial else [])
    r = run(ssh_argv(p, remote(p, "start", *args)), capture_output=True, text=True, timeout=90)
    return _json_out(r)


def status(p, session=None, run=subprocess.run) -> dict:
    r = run(ssh_argv(p, remote(p, "status", *(["--session", session] if session else []))), capture_output=True,
            text=True, timeout=30)
    return _json_out(r) or {}


def _alive(p, pid, run=subprocess.run) -> bool:
    r = run(ssh_argv(p, f"kill -0 {int(pid)} 2>/dev/null && echo alive || echo dead"), capture_output=True, text=True,
            timeout=30)
    return "alive" in (r.stdout or "")


def start_or_attach(p, session=None, trial=False, run=subprocess.run, alive=_alive) -> str:
    """the session to attach to: a live one as it is, a dead one restarted under its id (it resumes in HOLD), or a
    new one"""
    if session:
        st = status(p, session, run=run)
        if st and st.get("pid") and alive(p, st["pid"], run):
            return session
    res = start(p, session=session, trial=trial, run=run)
    if res.get("error"):
        raise RuntimeError(f"calibd did not start: {res}")
    return res["session"]


def stop(p, session=None, run=subprocess.run) -> dict:
    r = run(ssh_argv(p, remote(p, "stop", *(["--session", session] if session else []))), capture_output=True,
            text=True, timeout=60)
    return _json_out(r)


class Link:
    """the attach link: ssh running `calibd attach`, state lines on its stdout, our lines on its stdin"""

    def __init__(self, p, session, popen=subprocess.Popen):
        self.proc = popen(ssh_argv(p, remote(p, "attach", "--session", session)), stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, text=True, bufsize=1)
        self._lock = threading.Lock()

    def lines(self):
        for line in self.proc.stdout:
            yield line

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
        except OSError:
            pass
        self.proc.terminate()


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


def run_tui(link, speaker, headless=0, out=None, interactive=False) -> int:
    """the TUI loop over a link; headless N: render N states then leave. -> states rendered"""
    clear = "" if headless or out is not None else "\033[2J\033[H"
    out = out or (lambda s: (sys.stdout.write(clear + s + "\n"), sys.stdout.flush()))
    t = tui.Tui(link.send, speaker, out=out)
    quit_ev = threading.Event()

    def beat():
        while not quit_ev.is_set():
            t.heartbeat()
            quit_ev.wait(0.25)
    t.heartbeat()
    threading.Thread(target=beat, daemon=True).start()
    if interactive:
        threading.Thread(target=_keys, args=(t, quit_ev), daemon=True).start()
    n = 0
    try:
        for line in link.lines():
            t.step(line)
            n += 1
            if (headless and n >= headless) or quit_ev.is_set():
                break
    finally:
        quit_ev.set()
        link.close()
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(prog="calib", description=__doc__.split("\n")[0])
    ap.add_argument("robot", help="host name (phantom10) or bot name (P10)")
    ap.add_argument("--session", default=None, help="re-attach to (or restart) this session")
    ap.add_argument("--trial", action="store_true", help="placement-trial mode (neck at home, nothing moves)")
    ap.add_argument("--headless", type=int, default=0, help="render N states and leave (no keys)")
    ap.add_argument("--no-speech", action="store_true")
    ap.add_argument("--no-sync", action="store_true", help="do not copy the kit to the robot")
    ap.add_argument("--stop", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--say", default=None, help="speak this once on the laptop and leave (speech test)")
    a = ap.parse_args(argv)
    speaker = tui.Silent() if a.no_speech else tui.SpdSay()
    if a.say:
        speaker.say(a.say)
        return 0
    p = prof_mod.load(a.robot) if a.robot.upper().startswith("P") and a.robot[1:].isdigit() else \
        prof_mod.for_host(a.robot)
    if a.stop:
        print(json.dumps(stop(p, a.session)))
        return 0
    if a.status:
        print(json.dumps(status(p, a.session), indent=1))
        return 0
    if not a.no_sync:
        bootstrap(p)
    sid = start_or_attach(p, a.session, a.trial)
    sp = speaker if isinstance(speaker, tui.Silent) else tui.Background(speaker)
    n = run_tui(Link(p, sid), sp, headless=a.headless, interactive=not a.headless and sys.stdin.isatty())
    print(f"left the TUI after {n} states; calibd keeps running (session {sid}): calib {a.robot} --session {sid} "
          f"re-attaches, --stop stops it", file=sys.stderr)
    time.sleep(0.2)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    sys.exit(main())
