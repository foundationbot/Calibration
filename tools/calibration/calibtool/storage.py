"""Session folders, raw-data retention, protected paths and disk checks (design section 11).

<root>/sessions/<session id>/{raw,corners,fit,report,deploy}, plus the session's state.json and events.jsonl (daemon).
Raw (GB) is kept for the session behind the live deployment and the last KEEP_COMPLETED completed-deploy sessions;
aborted and read-only sessions do not count toward those; a running session is never touched; anything older than the
kept completed sessions goes (retention). Protected paths (rollback scripts' and the tool's references, the stored
M1-M3 datasets) are never deleted (is_protected, delete_raw). Disk: preflight refuses below PREFLIGHT_MIN_GB free on
the filesystem the session writes to; capture stops below CAPTURE_MIN_GB."""
import fnmatch
import re
import shutil
import time
from pathlib import Path

PREFLIGHT_MIN_GB = 20.0
CAPTURE_MIN_GB = 5.0
KEEP_COMPLETED = 2
SUBDIRS = ("raw", "corners", "fit", "report", "deploy")
PROTECTED = ("/root/p21-cal/dma_motor_motion", "/root/p21-cal/backup/dma-pre-210-*", "/root/p07-stack",
             "/root/p09-*-backup*", "/root/p11-cal/takes", "/root/p21-cal/takes", "/root/p11-cal/backup",
             "/root/p21-cal/backup")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class Protected(RuntimeError):
    pass


def new_session_id(bot, now=None) -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.localtime(time.time() if now is None else now)) + f"-{bot}"


class SessionDirs:
    def __init__(self, root, sid):
        if not _ID.match(str(sid)) or str(sid) in (".", ".."):
            raise ValueError(f"session id {sid!r} is not a plain name")
        self.sid = str(sid)
        self.root = Path(root) / "sessions" / self.sid
        for k in SUBDIRS:
            setattr(self, k, self.root / k)

    def create(self):
        for k in SUBDIRS:
            getattr(self, k).mkdir(parents=True, exist_ok=True)
        return self

    def exists(self) -> bool:
        return self.root.is_dir()


def disk_free_gb(path) -> float:
    p = Path(path)
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free / 1e9


def disk_check(path, min_gb=PREFLIGHT_MIN_GB):
    """(enough, free GB) on the filesystem holding path"""
    free = disk_free_gb(path)
    return free >= min_gb, free


def is_protected(path, refs=()) -> bool:
    """path is (inside) a protected path or a reference given by refs"""
    p = str(Path(path))
    for pat in tuple(PROTECTED) + tuple(str(Path(r)) for r in refs):
        if fnmatch.fnmatch(p, pat) or fnmatch.fnmatch(p, pat.rstrip("/") + "/*"):
            return True
    return False


def retention(sessions, live_id=None, keep_completed=KEEP_COMPLETED) -> list:
    """session ids (sorted, oldest first) whose raw data goes. sessions: [{id, status, deployed}], ids sortable by
    time (new_session_id)"""
    ss = sorted(sessions, key=lambda s: s["id"])
    done = [s["id"] for s in ss if s.get("status") == "completed" and s.get("deployed")]
    if len(done) < keep_completed:
        return []
    cutoff = done[-keep_completed]
    return [s["id"] for s in ss if s["id"] < cutoff and s["id"] != live_id and s.get("status") != "running"]


def delete_raw(dirs: SessionDirs, refs=()):
    """remove the session's raw folder (Protected when it or the session is referenced)"""
    for p in (dirs.raw, dirs.root):
        if is_protected(p, refs):
            raise Protected(f"{p} is protected: not deleting {dirs.raw}")
    shutil.rmtree(dirs.raw, ignore_errors=True)


def list_sessions(root) -> list:
    """[{id, status, deployed}] of the sessions under root (status from their state.json; 'unknown' without one)"""
    import json
    out = []
    base = Path(root) / "sessions"
    for d in sorted(base.iterdir()) if base.is_dir() else ():
        try:
            st = json.loads((d / "state.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            st = {}
        out.append({"id": d.name, "status": st.get("status", "unknown"), "deployed": bool(st.get("deployed"))})
    return out
