"""calibtool.voice: `calib --install-voice` downloads what is missing of the Piper voice, checks it speaks, and says
how to test it. The downloader, the runner and `which` are fakes: no test touches the network or runs piper."""
import io
import os
import tarfile

from calibtool import tui, voice


def piper_tarball(bad=None) -> bytes:
    """a tarball shaped like the piper release: piper/piper, a library and its symlink, espeak-ng data"""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        def add(name, data=b"x", mode=0o644):
            ti = tarfile.TarInfo(name)
            ti.size, ti.mode = len(data), mode
            tf.addfile(ti, io.BytesIO(data))
        d = tarfile.TarInfo("piper")
        d.type, d.mode = tarfile.DIRTYPE, 0o755
        tf.addfile(d)
        add("piper/piper", b"#!/bin/sh\n", 0o755)
        add("piper/libpiper_phonemize.so.1.2.0")
        ln = tarfile.TarInfo("piper/libpiper_phonemize.so.1")
        ln.type, ln.linkname = tarfile.SYMTYPE, "libpiper_phonemize.so.1.2.0"
        tf.addfile(ln)
        add("piper/espeak-ng-data/phontab")
        if bad == "dotdot":
            add("piper/../../escaped")
        if bad == "link":
            ln = tarfile.TarInfo("piper/evil")
            ln.type, ln.linkname = tarfile.SYMTYPE, "/etc/passwd"
            tf.addfile(ln)
    return buf.getvalue()


class Fetch:
    def __init__(self, tarball=None, fail=None):
        self.tarball, self.fail, self.urls = tarball or piper_tarball(), fail, []

    def __call__(self, url, dest):
        self.urls.append(url)
        if self.fail and self.fail in url:
            dest.write_bytes(b"half")                # a broken download leaves a partial file behind
            raise OSError("connection reset")
        dest.write_bytes(self.tarball if url.endswith(".tar.gz") else
                         b"{}" if url.endswith(".json") else b"ONNX" * 100)


class Run:
    def __init__(self, rc=0):
        self.rc, self.calls = rc, []

    def __call__(self, argv, **kw):
        self.calls.append((argv, kw))
        if self.rc == 0:
            with open(argv[argv.index("--output_file") + 1], "wb") as f:
                f.write(b"RIFF" + b"\0" * 200)
        return type("R", (), {"returncode": self.rc, "stdout": b"", "stderr": b"piper: model broken"})()


def aplay_only(name):
    return "/usr/bin/aplay" if name == "aplay" else None


def test_install_voice_downloads_extracts_and_checks(tmp_path):
    f, r, said = Fetch(), Run(), []
    assert voice.install(tmp_path, fetch=f, run=r, machine="x86_64", system="Linux", which=aplay_only,
                         log=said.append) == 0
    assert f.urls == [
        "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz",
        "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
        "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/"
        "en_US-lessac-medium.onnx.json"]
    binary = tmp_path / "piper" / "piper"
    assert binary.is_file() and os.access(binary, os.X_OK)
    assert (tmp_path / "piper" / "libpiper_phonemize.so.1").is_symlink()
    assert (tmp_path / "piper" / "espeak-ng-data" / "phontab").is_file()
    assert (tmp_path / "en_US-lessac-medium.onnx").is_file() and (tmp_path / "en_US-lessac-medium.onnx.json").is_file()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["en_US-lessac-medium.onnx", "en_US-lessac-medium.onnx.json",
                                                         "piper"]              # no .part or temp folders left
    (argv, kw), = r.calls
    assert argv[0] == str(binary) and str(tmp_path / "en_US-lessac-medium.onnx") in argv and "--output_file" in argv
    assert kw["input"].startswith(b"Voice test")
    text = "\n".join(said)
    assert "aplay" in text and 'calib --voice piper --say "Voice test"' in text
    # exactly where tui.make_speaker looks
    assert tui.PiperSay(binary=binary, model=tmp_path / "en_US-lessac-medium.onnx").available()


def test_install_voice_skips_what_is_there_and_picks_the_arch(tmp_path):
    (tmp_path / "en_US-lessac-medium.onnx").write_bytes(b"ONNX")
    f = Fetch()
    assert voice.install(tmp_path, fetch=f, run=Run(), machine="aarch64", system="Linux",
                         which=lambda n: "/usr/bin/pw-play" if n == "pw-play" else None, log=lambda s: None) == 0
    assert [u.rsplit("/", 1)[1] for u in f.urls] == ["piper_linux_aarch64.tar.gz", "en_US-lessac-medium.onnx.json"]
    f2 = Fetch()
    assert voice.install(tmp_path, fetch=f2, run=Run(), machine="aarch64", system="Linux",
                         which=lambda n: "/usr/bin/pw-play", log=lambda s: None) == 0
    assert f2.urls == []                                                        # the second run downloads nothing


def test_install_voice_refuses_other_systems(tmp_path):
    for system, machine in (("Darwin", "arm64"), ("Linux", "armv7l"), ("Windows", "AMD64")):
        f, said = Fetch(), []
        assert voice.install(tmp_path, fetch=f, run=Run(), machine=machine, system=system, which=aplay_only,
                             log=said.append) == 2
        assert f.urls == [] and "x86_64 or aarch64" in said[0]


def test_install_voice_refuses_a_tarball_that_escapes(tmp_path):
    for bad in ("dotdot", "link"):
        dest = tmp_path / bad / "voice"
        said = []
        assert voice.install(dest, fetch=Fetch(piper_tarball(bad)), run=Run(), machine="x86_64", system="Linux",
                             which=aplay_only, log=said.append) == 2
        assert "unsafe" in "\n".join(said)
        assert not (dest / "piper").exists() and not (tmp_path / bad / "escaped").exists()
        assert list(dest.iterdir()) == []


def test_install_voice_keeps_no_partial_download(tmp_path):
    said = []
    assert voice.install(tmp_path, fetch=Fetch(fail=".onnx.json"), run=Run(), machine="x86_64", system="Linux",
                         which=aplay_only, log=said.append) == 2
    assert "connection reset" in "\n".join(said)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["en_US-lessac-medium.onnx", "piper"]
    f = Fetch()                                                                 # the rerun fetches only what is missing
    assert voice.install(tmp_path, fetch=f, run=Run(), machine="x86_64", system="Linux", which=aplay_only,
                         log=lambda s: None) == 0
    assert [u.rsplit("/", 1)[1] for u in f.urls] == ["en_US-lessac-medium.onnx.json"]


def test_install_voice_reports_a_piper_that_does_not_speak(tmp_path):
    said = []
    assert voice.install(tmp_path, fetch=Fetch(), run=Run(rc=1), machine="x86_64", system="Linux",
                         which=aplay_only, log=said.append) == 2
    assert "did not speak" in said[-1] and "model broken" in said[-1]


def test_install_voice_says_a_player_is_needed(tmp_path):
    said = []
    assert voice.install(tmp_path, fetch=Fetch(), run=Run(), machine="x86_64", system="Linux",
                         which=lambda n: None, log=said.append) == 1
    text = "\n".join(said)
    assert "pw-play" in text and "aplay" in text and "no player" in text


def test_piper_plays_with_aplay_when_pw_play_is_missing():
    assert tui.default_player(which=lambda n: f"/usr/bin/{n}") == "/usr/bin/pw-play"
    assert tui.default_player(which=aplay_only) == "/usr/bin/aplay"
    assert tui.default_player(which=lambda n: None) == "/usr/bin/pw-play"
