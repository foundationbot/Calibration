"""calibtool.exe: the single-file calib carries PyYAML (another PC needs only python3), and `calib --update` replaces
it with the latest release's. The downloads are fakes (gh runner, URL fetcher): no test touches the network."""
import json
import os
import subprocess
import sys
import urllib.error
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("yaml")

from calibtool import cli, exe  # noqa: E402


# ------------------------------------------------------------------ PyYAML inside the file


def test_calib_runs_with_no_installed_yaml(tmp_path):
    """python3 -I -S: no site-packages at all (no system PyYAML); --version and a profile load still work, with the
    yaml bundled in the file (pure Python: no compiled extension inside)"""
    calib = tmp_path / "bin" / "calib"
    man = exe.build(calib)
    names = zipfile.ZipFile(calib).namelist()
    assert "deps/yaml/__init__.py" in names and "deps/yaml/loader.py" in names
    assert not [n for n in names if n.startswith("deps/") and n.endswith((".so", ".pyd", ".pyc"))]
    assert json.loads(zipfile.ZipFile(calib).read(exe.MANIFEST))["pyyaml"] == man["pyyaml"] != "?"
    env = dict(os.environ, XDG_CACHE_HOME=str(tmp_path / "cache"), HOME=str(tmp_path))
    iso = [sys.executable, "-I", "-S"]
    r = subprocess.run(iso + ["-c", "import yaml"], capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode != 0 and "yaml" in r.stderr                 # the isolation hides any installed yaml
    r = subprocess.run(iso + [str(calib), "--version"], capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0, r.stderr
    assert f"calib kit {man['kit']}" in r.stdout
    probe = ("import runpy, sys\n"
             "g = runpy.run_path(sys.argv[1], run_name='calib_probe')\n"
             "d = g['_setup'](sys.argv[1])\n"
             "from calibtool import profile\n"
             "import yaml\n"
             "p = profile.load('P10')\n"
             "print(p.host, p.bot, yaml.__file__, d)\n")
    r = subprocess.run(iso + ["-c", probe, str(calib)], capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0, r.stderr
    host, bot, yaml_file, kit = r.stdout.split()
    assert bot == "P10" and host.endswith("phantom10")
    assert yaml_file.startswith(str(tmp_path / "cache" / "calibtool")) and yaml_file.startswith(kit)


def test_build_refuses_a_folder_that_is_not_yaml(tmp_path):
    with pytest.raises(RuntimeError, match="not a yaml package"):
        exe.build(tmp_path / "calib", yaml_dir=tmp_path)


# ------------------------------------------------------------------ calib --update


def fake_calib(path, kit, payload, rc=0):
    """a tiny calib-shaped file: shebang, a zip with __main__.py printing its --version line, the manifest"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(exe.SHEBANG)
        with zipfile.ZipFile(fh, "w") as z:
            z.writestr("__main__.py", f"import sys\nprint('calib kit {kit} (/x)')\nsys.exit({rc})\n")
            z.writestr(exe.MANIFEST, json.dumps({"kit": kit, "payload": payload, "built": "2026-11-10 12:00:00"}))
    os.chmod(path, 0o755)
    return path


class Runner:
    """gh writes the new calib to its --output; everything else (the --version check) runs for real"""

    def __init__(self, make=None, gh_rc=0, gh_err=""):
        self.make, self.gh_rc, self.gh_err, self.calls = make, gh_rc, gh_err, []

    def __call__(self, argv, **kw):
        self.calls.append(argv)
        if argv[0] == "gh":
            if self.gh_rc == 0:
                self.make(Path(argv[argv.index("--output") + 1]))
            return subprocess.CompletedProcess(argv, self.gh_rc, "", self.gh_err)
        return subprocess.run(argv, **kw)


def no_fetch(url, dest):
    pytest.fail(f"fetched {url}")


def leftovers(d):
    return [p.name for p in d.iterdir() if p.name != "calib"]


def test_update_with_gh_replaces_the_file_and_says_both_versions(tmp_path):
    target = fake_calib(tmp_path / "bin" / "calib", "oldkit111111", "p-old")
    run = Runner(make=lambda out: fake_calib(out, "newkit222222", "p-new"))
    said = []
    assert exe.update(target, run=run, fetch=no_fetch, which=lambda n: "/usr/bin/gh", log=said.append) == 0
    gh = run.calls[0]
    assert gh[:3] == ["gh", "release", "download"] and gh[gh.index("--repo") + 1] == "foundationbot/Calibration"
    assert gh[gh.index("--pattern") + 1] == "calib" and "--clobber" in gh
    assert os.path.dirname(gh[gh.index("--output") + 1]) == str(target.parent)       # beside the target
    assert run.calls[1][-1] == "--version"
    assert exe.manifest(target)["kit"] == "newkit222222" and os.stat(target).st_mode & 0o777 == 0o755
    text = "\n".join(said)
    assert "oldkit111111" in text and "newkit222222" in text
    assert leftovers(target.parent) == []


def test_update_without_gh_downloads_the_public_release(tmp_path):
    target = fake_calib(tmp_path / "bin" / "calib", "oldkit111111", "p-old")
    urls = []

    def fetch(url, dest):
        urls.append(url)
        fake_calib(dest, "newkit222222", "p-new")
    assert exe.update(target, run=Runner(), fetch=fetch, which=lambda n: None, log=lambda s: None) == 0
    assert urls == ["https://github.com/foundationbot/Calibration/releases/latest/download/calib"]
    assert exe.manifest(target)["kit"] == "newkit222222"


def test_update_falls_back_to_the_url_when_gh_fails(tmp_path):
    target = fake_calib(tmp_path / "bin" / "calib", "oldkit111111", "p-old")
    got = []
    assert exe.update(target, run=Runner(gh_rc=1, gh_err="HTTP 401: Bad credentials"),
                      fetch=lambda u, d: got.append(u) or fake_calib(d, "newkit222222", "p-new"),
                      which=lambda n: "/usr/bin/gh", log=lambda s: None) == 0
    assert got and exe.manifest(target)["kit"] == "newkit222222"


def test_update_private_repo_without_gh_says_what_to_do(tmp_path):
    target = fake_calib(tmp_path / "bin" / "calib", "oldkit111111", "p-old")
    before = target.read_bytes()

    def fetch(url, dest):
        raise urllib.error.HTTPError(url, 404, "Not Found", None, None)
    said = []
    assert exe.update(target, run=Runner(), fetch=fetch, which=lambda n: None, log=said.append) == 2
    text = "\n".join(said)
    assert "install gh" in text and "gh auth login" in text and "ask for the calib file" in text
    assert target.read_bytes() == before and leftovers(target.parent) == []


def test_update_keeps_the_old_file_when_the_download_is_not_calib(tmp_path):
    target = fake_calib(tmp_path / "bin" / "calib", "oldkit111111", "p-old")
    before = target.read_bytes()
    said = []
    assert exe.update(target, run=Runner(), fetch=lambda u, d: d.write_text("<html>sign in</html>"),
                      which=lambda n: None, log=said.append) == 2
    assert "not a calib executable" in "\n".join(said)
    assert target.read_bytes() == before and leftovers(target.parent) == []


def test_update_keeps_the_old_file_when_the_new_one_does_not_run(tmp_path):
    target = fake_calib(tmp_path / "bin" / "calib", "oldkit111111", "p-old")
    before = target.read_bytes()
    said = []
    assert exe.update(target, run=Runner(make=lambda out: fake_calib(out, "newkit222222", "p-new", rc=3)),
                      fetch=no_fetch, which=lambda n: "/usr/bin/gh", log=said.append) == 2
    assert "does not run" in "\n".join(said)
    assert target.read_bytes() == before and leftovers(target.parent) == []


def test_update_says_up_to_date_and_leaves_the_file(tmp_path):
    target = fake_calib(tmp_path / "bin" / "calib", "samekit33333", "p-same")
    before = target.read_bytes()
    said = []
    assert exe.update(target, run=Runner(make=lambda out: fake_calib(out, "samekit33333", "p-same")),
                      fetch=no_fetch, which=lambda n: "/usr/bin/gh", log=said.append) == 0
    assert "up to date" in "\n".join(said) and "samekit33333" in "\n".join(said)
    assert target.read_bytes() == before and leftovers(target.parent) == []


def test_update_installs_when_there_was_no_calib_yet(tmp_path):
    target = tmp_path / "newbin" / "calib"
    said = []
    assert exe.update(target, run=Runner(), fetch=lambda u, d: fake_calib(d, "newkit222222", "p-new"),
                      which=lambda n: None, log=said.append) == 0
    assert exe.manifest(target)["kit"] == "newkit222222" and "old: none" in "\n".join(said)


def test_update_target_is_the_running_calib_else_the_installed_one(tmp_path, monkeypatch):
    calib = fake_calib(tmp_path / "somewhere" / "calib", "k", "p")
    link = tmp_path / "link-to-calib"
    link.symlink_to(calib)
    assert exe.update_target(str(calib)) == calib.resolve()
    assert exe.update_target(str(link)) == calib.resolve()             # the file itself, not the link
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert exe.update_target(str(tmp_path / "pytest")) == tmp_path / "home" / ".local" / "bin" / "calib"
    assert exe.update_target("-c") == tmp_path / "home" / ".local" / "bin" / "calib"


def test_cli_update_and_install_voice_flags(monkeypatch):
    from calibtool import menu, voice
    monkeypatch.setattr(menu, "main", lambda **kw: pytest.fail("menu opened"))
    monkeypatch.setattr(exe, "update", lambda: 7)
    monkeypatch.setattr(voice, "install", lambda: 5)
    assert cli.main(["--update"]) == 7
    assert cli.main(["--install-voice"]) == 5
