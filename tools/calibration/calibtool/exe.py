"""`calib --install`: the laptop's single-file executable (10 Nov 2026: the operator asked for "some kind of executable"
instead of `cd tools/calibration && python3 -m calibtool.cli ...`).

build(out) writes a zipapp (as `python3 -m zipapp` makes: `#!/usr/bin/env python3`, then a zip with __main__.py at its
root) holding the whole kit at its repo layout under kit/:
  - kit/tools/calibration/{calibtool,scripts,config}: exactly the files daemon.kit_hash() hashes (no tests, results,
    caches, *.npz, dot files), which are also what bootstrap copies to the robot;
  - kit/config/foundation_rig/20250206_phantom_mk1.urdf: the neck chain's robot model bootstrap copies too;
  - calibtool-zipapp.json: {"kit": the kit hash, "payload": a hash of every file above}.
The executable's __main__ extracts kit/ once to ~/.cache/calibtool/kit-<payload> (into a temporary folder renamed into
place when complete: two runs at once cannot leave half a kit) and runs calibtool.cli from there, so paths.CAL is the
extracted tools/calibration and bootstrap, kit_hash and the profiles work unchanged: the kit hash shown and compared
with the robot's calibd is the same as from the worktree. install(dir) builds it to dir/calib (chmod 755) and says how
to run it. Stdlib only."""
import hashlib
import json
import os
import time
import zipfile
from pathlib import Path

from calibtool import daemon, paths

URDF_REL = "config/foundation_rig/20250206_phantom_mk1.urdf"    # cli.URDF_REL (bootstrap copies it to the robot)
MANIFEST = "calibtool-zipapp.json"
SHEBANG = b"#!/usr/bin/env python3\n"

# the executable's entry point: stdlib only, no calibtool import before the kit is on sys.path
MAIN = '''"""calib: the calibration tool's single-file executable (calibtool.exe). Extracts its kit once to
~/.cache/calibtool/kit-<payload> and runs calibtool.cli from there."""
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path


def _kit_dir(zpath):
    with zipfile.ZipFile(zpath) as z:
        man = json.loads(z.read("%(manifest)s"))
        base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "calibtool"
        d = base / ("kit-" + man["payload"])
        if (d / ".complete").is_file():
            return d
        base.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=d.name + ".", dir=base))
        for info in z.infolist():
            name = info.filename
            if not name.startswith("kit/") or name.endswith("/"):
                continue
            rel = Path(name[4:])
            if rel.is_absolute() or ".." in rel.parts:
                continue
            dst = tmp / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(dst, "wb") as f:
                shutil.copyfileobj(src, f)
            mode = (info.external_attr >> 16) & 0o777
            if mode:
                os.chmod(dst, mode)
        (tmp / ".complete").write_text(man["kit"])
        try:
            os.rename(tmp, d)
        except OSError:                     # another run extracted it first
            shutil.rmtree(tmp, ignore_errors=True)
        return d


def main():
    d = _kit_dir(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, str(d / "tools" / "calibration"))
    from calibtool import cli
    return cli.main(sys.argv[1:])


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    sys.exit(main())
''' % {"manifest": MANIFEST}


def kit_files(cal=None) -> list:
    """[(name in the zip, file)]: the files daemon.kit_hash() hashes (its filter, file for file), then the URDF"""
    cal = Path(cal) if cal is not None else paths.CAL
    out = []
    for d in daemon.KIT_DIRS:
        for f in sorted((cal / d).rglob("*")):
            rel = f.relative_to(cal)
            if not f.is_file() or f.suffix == ".npz" or any(x in daemon.KIT_SKIP or x.startswith(".")
                                                            for x in rel.parts):
                continue
            out.append((f"kit/tools/calibration/{rel.as_posix()}", f))
    urdf = cal.parents[1] / URDF_REL
    if not urdf.is_file():
        raise FileNotFoundError(f"{urdf}: bootstrap copies it to the robot (the neck chain's model)")
    out.append((f"kit/{URDF_REL}", urdf))
    return out


def build(out, cal=None) -> dict:
    """the executable at out (written beside it, then renamed into place) -> its manifest"""
    cal = Path(cal) if cal is not None else paths.CAL
    files = kit_files(cal)
    h = hashlib.sha256()
    for name, f in files:
        h.update(name.encode() + b"\0" + f.read_bytes() + b"\0")
    man = {"kit": daemon.kit_hash(cal), "payload": h.hexdigest()[:16], "built": time.strftime("%Y-%m-%d %H:%M:%S")}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(SHEBANG)
        with zipfile.ZipFile(fh, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("__main__.py", MAIN)
            z.writestr(MANIFEST, json.dumps(man))
            for name, f in files:
                z.write(f, name)
    os.chmod(tmp, 0o755)
    os.replace(tmp, out)
    return man


def install(dest_dir="~/.local/bin", log=print) -> int:
    dest = Path(os.path.expanduser(str(dest_dir)))
    target = dest / "calib"
    man = build(target)
    log(f"installed {target} (kit {man['kit']}, {target.stat().st_size // 1024} kB)")
    on_path = str(dest.resolve()) in {str(Path(x).expanduser().resolve()) for x in
                                      os.environ.get("PATH", "").split(os.pathsep) if x}
    if not on_path:
        log(f"{dest} is not on your PATH: run it as {target}, or add it: "
            f"echo 'export PATH=\"{dest}:$PATH\"' >> ~/.bashrc")
    log("Run it:")
    log("  calib                      asks which robot, then offers what makes sense for it")
    log("  calib phantom10            the same, for phantom10")
    log("  calib phantom10 --full     the scripted path (every flag as before)")
    log("After pulling new code, run `python3 -m calibtool.cli --install` again (in tools/calibration).")
    return 0
