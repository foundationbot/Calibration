"""`calib --install`: the laptop's single-file executable (10 Nov 2026: the operator asked for "some kind of executable"
instead of `cd tools/calibration && python3 -m calibtool.cli ...`).

build(out) writes a zipapp (as `python3 -m zipapp` makes: `#!/usr/bin/env python3`, then a zip with __main__.py at its
root) holding the whole kit at its repo layout under kit/:
  - kit/tools/calibration/{calibtool,scripts,config}: exactly the files daemon.kit_hash() hashes (no tests, results,
    caches, *.npz, dot files), which are also what bootstrap copies to the robot;
  - kit/config/foundation_rig/20250206_phantom_mk1.urdf: the neck chain's robot model bootstrap copies too;
  - deps/yaml: the pure-Python PyYAML of the build's environment (its .py files and licence; no compiled _yaml: PyYAML
    falls back to pure Python without it), so another PC needs nothing but python3 (10 Nov: the laptop side loads
    the profiles with yaml);
  - calibtool-zipapp.json: {"kit": the kit hash, "payload": a hash of every file above, "pyyaml": its version}.
The executable's __main__ extracts kit/ once to ~/.cache/calibtool/kit-<payload> (deps/ to its pydeps/; into a
temporary folder renamed into place when complete: two runs at once cannot leave half a kit) and runs calibtool.cli
from there, the kit and then pydeps first on sys.path (the bundled yaml before any installed one), so paths.CAL is the
extracted tools/calibration and bootstrap, kit_hash and the profiles work unchanged: the kit hash shown and compared
with the robot's calibd is the same as from the worktree. install(dir) builds it to dir/calib (chmod 755) and says how
to run it. update() replaces an installed calib with the latest GitHub release's (`calib --update`). Stdlib only."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
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


PREFIXES = (("kit/", ""), ("deps/", "pydeps/"))     # zip folder -> folder in the extracted kit


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
            pre = [(a, b) for a, b in PREFIXES if name.startswith(a)]
            if not pre or name.endswith("/"):
                continue
            rel = Path(pre[0][1] + name[len(pre[0][0]):])
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


def _setup(zpath):
    """the kit extracted, its code and then the bundled dependencies (yaml) first on sys.path -> the kit folder"""
    d = _kit_dir(zpath)
    sys.path[:0] = [str(d / "tools" / "calibration"), str(d / "pydeps")]
    return d


def main():
    _setup(os.path.dirname(os.path.abspath(__file__)))
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


def dep_files(yaml_dir=None) -> tuple:
    """([(name in the zip, file)], PyYAML version): the pure-Python yaml package of this environment (every .py, no
    compiled extension) and its licence"""
    if yaml_dir is None:
        try:
            import yaml
        except ImportError:
            raise RuntimeError("PyYAML is needed to build calib (it is bundled into the file): pip install PyYAML")
        yaml_dir, version = Path(yaml.__file__).parent, getattr(yaml, "__version__", "?")
    else:
        version = "?"
    yaml_dir = Path(yaml_dir)
    out = [(f"deps/yaml/{f.name}", f) for f in sorted(yaml_dir.glob("*.py"))]
    if not any(n.endswith("/__init__.py") for n, _ in out):
        raise RuntimeError(f"{yaml_dir}: not a yaml package (no __init__.py)")
    for lic in sorted(yaml_dir.parent.glob("[Pp]y[Yy][Aa][Mm][Ll]-*.dist-info/**/LICENSE*")):
        out.append(("deps/yaml/LICENSE", lic))
        break
    return out, version


def build(out, cal=None, yaml_dir=None) -> dict:
    """the executable at out (written beside it, then renamed into place) -> its manifest"""
    cal = Path(cal) if cal is not None else paths.CAL
    deps, yaml_version = dep_files(yaml_dir)
    files = kit_files(cal) + deps
    h = hashlib.sha256()
    for name, f in files:
        h.update(name.encode() + b"\0" + f.read_bytes() + b"\0")
    man = {"kit": daemon.kit_hash(cal), "payload": h.hexdigest()[:16], "built": time.strftime("%Y-%m-%d %H:%M:%S"),
           "pyyaml": yaml_version}
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
    log("After pulling new code, run `python3 -m calibtool.cli --install` again (in tools/calibration); "
        "`calib --update` fetches the latest release instead.")
    return 0


# ------------------------------------------------------------------ calib --update: the latest release's calib

REPO = "foundationbot/Calibration"
ASSET = "calib"
LATEST_URL = f"https://github.com/{REPO}/releases/latest/download/{ASSET}"


def manifest(path):
    """the manifest of a calib file, or None when path is not one"""
    try:
        with zipfile.ZipFile(path) as z:
            return json.loads(z.read(MANIFEST))
    except (OSError, KeyError, ValueError, zipfile.BadZipFile):
        return None


def update_target(argv0=None) -> Path:
    """the calib file to replace: the one running (sys.argv[0] when it is a calib file), else ~/.local/bin/calib"""
    a = sys.argv[0] if argv0 is None else argv0
    if a and os.path.isfile(a) and manifest(a) is not None:
        return Path(a).resolve()
    return Path(os.path.expanduser("~/.local/bin")) / "calib"


def fetch_url(url, dest, timeout=120):
    """url -> the file dest (urllib, redirects followed)"""
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "calibtool"})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def _describe(man):
    return f"kit {man.get('kit')} (built {man.get('built', '?')})" if man else "none"


def update(target=None, run=subprocess.run, fetch=fetch_url, which=shutil.which, python=None, log=print) -> int:
    """replace target (update_target()) with the latest release's calib of foundationbot/Calibration: `gh release
    download` when gh is installed (a private repo needs it, logged in), else (or when gh fails) the public download
    URL. Downloaded next to target, chmod 755, run once (`--version`) and only then renamed over it. -> exit code"""
    target = Path(target) if target is not None else update_target()
    python = python or sys.executable or "python3"
    old = manifest(target) if target.is_file() else None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        log(f"calib --update: cannot create {target.parent}: {e}")
        return 2
    tmp = target.with_name(f".{target.name}.update-{os.getpid()}")
    errors = []
    try:
        got = False
        if which("gh"):
            r = run(["gh", "release", "download", "--repo", REPO, "--pattern", ASSET, "--output", str(tmp),
                     "--clobber"], capture_output=True, text=True, timeout=300)
            got = r.returncode == 0 and tmp.is_file()
            if not got:
                errors.append(f"gh release download: {(r.stderr or r.stdout or '').strip() or 'failed'}")
        if not got:
            try:
                fetch(LATEST_URL, tmp)
                got = tmp.is_file()
            except Exception as e:  # noqa: BLE001  (HTTPError, URLError, timeouts, OSError)
                code = getattr(e, "code", None)
                errors.append(f"{LATEST_URL}: {e}")
                if code == 404 and not which("gh"):
                    errors.append("the release is not public (or has no calib file): install gh and run "
                                  "`gh auth login`, then `calib --update` again; or ask for the calib file")
                elif not which("gh"):
                    errors.append("no network to GitHub? Or install gh and run `gh auth login`; or ask for the "
                                  "calib file")
        if not got:
            log("calib --update: could not download the latest calib:\n  " + "\n  ".join(errors))
            return 2
        new = manifest(tmp)
        if new is None:
            log(f"calib --update: the downloaded file is not a calib executable ({tmp.stat().st_size} bytes); "
                f"left {target} as it is")
            return 2
        if old is not None and old.get("payload") == new.get("payload"):
            log(f"calib is up to date: {target}, {_describe(old)}")
            return 0
        os.chmod(tmp, 0o755)
        r = run([python, str(tmp), "--version"], capture_output=True, text=True, timeout=120)
        if r.returncode != 0 or "calib kit " not in (r.stdout or ""):
            log(f"calib --update: the new calib does not run (`--version` exit {r.returncode}): "
                f"{(r.stderr or r.stdout or '').strip()[-500:]}; left {target} as it is")
            return 2
        os.replace(tmp, target)
        log(f"updated {target}")
        log(f"  old: {_describe(old)}")
        log(f"  new: {_describe(new)}")
        return 0
    except OSError as e:
        log(f"calib --update: {e}; left {target} as it is")
        return 2
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass
