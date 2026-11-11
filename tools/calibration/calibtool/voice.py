"""`calib --install-voice`: the Piper voice (tui.PiperSay) downloaded to ~/.local/share/calib-voice on a Linux PC
(x86_64 or aarch64), so another laptop gets the voice with one command (10 Nov: it was copied by hand):
  - the piper binary: the rhasspy/piper release tarball for this machine, extracted so the binary is piper/piper
    (with the libraries and espeak-ng data beside it);
  - the voice en_US-lessac-medium.onnx and its .onnx.json (rhasspy/piper-voices v1.0.0, Hugging Face).
What is already there is kept (nothing downloaded again); downloads land in a .part file renamed into place. Then
piper speaks a short sentence into a temporary wav (the check) and the player is looked for (pw-play, else aplay).
The downloader, the runner and the machine are injected (tests use fakes; never the network)."""
import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

from calibtool import tui

PIPER_TAG = "2023.11.14-2"
PIPER_URL = "https://github.com/rhasspy/piper/releases/download/" + PIPER_TAG + "/piper_linux_{arch}.tar.gz"
VOICE = "en_US-lessac-medium.onnx"
VOICE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/" + VOICE
ARCH = {"x86_64": "x86_64", "amd64": "x86_64", "aarch64": "aarch64", "arm64": "aarch64"}
PLAYERS = ("pw-play", "aplay")
TEST_SENTENCE = "Voice test. The calibration voice works."


def fetch_url(url, dest, timeout=300):
    """url -> the file dest (urllib, redirects followed)"""
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "calibtool"})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def _download(fetch, url, dest: Path, log):
    """url -> dest through dest.part (a broken download never looks installed)"""
    part = dest.with_name(dest.name + ".part")
    log(f"downloading {url}")
    try:
        fetch(url, part)
        if not part.is_file() or part.stat().st_size == 0:
            raise OSError(f"{url}: nothing downloaded")
        os.replace(part, dest)
    finally:
        try:
            part.unlink()
        except OSError:
            pass
    log(f"  -> {dest} ({dest.stat().st_size // 1024} kB)")


def _safe_members(tf: tarfile.TarFile, root: Path):
    """the tarball's members, refused when one would land outside root (absolute, .., or a link pointing out)"""
    root = root.resolve()
    for m in tf.getmembers():
        dst = (root / m.name).resolve()
        if os.path.isabs(m.name) or not str(dst).startswith(str(root) + os.sep) and dst != root:
            raise OSError(f"unsafe path in the piper tarball: {m.name}")
        if m.issym() or m.islnk():
            base = dst.parent if m.issym() else root
            tgt = (base / m.linkname).resolve()
            if os.path.isabs(m.linkname) or not str(tgt).startswith(str(root) + os.sep):
                raise OSError(f"unsafe link in the piper tarball: {m.name} -> {m.linkname}")
        if m.isdev():
            raise OSError(f"device file in the piper tarball: {m.name}")
    return tf.getmembers()


def extract_piper(tarball, dest: Path):
    """the tarball's piper/ folder -> dest/piper (extracted beside it, then renamed into place)"""
    tmp = Path(tempfile.mkdtemp(prefix=".piper-", dir=dest))
    try:
        with tarfile.open(tarball) as tf:
            members = _safe_members(tf, tmp)
            kw = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
            tf.extractall(tmp, members=members, **kw)
        if not (tmp / "piper" / "piper").is_file():
            raise OSError("the piper tarball has no piper/piper")
        old = dest / "piper"
        if old.exists() or old.is_symlink():
            shutil.rmtree(old) if old.is_dir() and not old.is_symlink() else old.unlink()
        os.replace(tmp / "piper", old)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def install(dest=None, fetch=fetch_url, run=subprocess.run, machine=None, system=None, which=shutil.which,
            log=print) -> int:
    """download what is missing of the Piper voice into dest (tui.PIPER_DIR), check it speaks -> exit code"""
    dest = Path(dest) if dest is not None else tui.PIPER_DIR
    system = system or platform.system()
    machine = (machine or platform.machine()).lower()
    if system != "Linux" or machine not in ARCH:
        log(f"calib --install-voice: Piper is set up here for Linux x86_64 or aarch64 only (this is {system} "
            f"{machine}); `calib --voice text` shows the prompts on screen instead")
        return 2
    binary, model = dest / "piper" / "piper", dest / VOICE
    meta = dest / (VOICE + ".json")
    try:
        dest.mkdir(parents=True, exist_ok=True)
        if binary.is_file():
            log(f"piper: already there ({binary})")
        else:
            url = PIPER_URL.format(arch=ARCH[machine])
            with tempfile.TemporaryDirectory(prefix=".piper-dl-", dir=dest) as td:
                tarball = Path(td) / Path(url).name
                _download(fetch, url, tarball, log)
                extract_piper(tarball, dest)
            log(f"piper: installed {binary}")
        for f, url in ((model, VOICE_URL), (meta, VOICE_URL + ".json")):
            if f.is_file() and f.stat().st_size > 0:
                log(f"voice: already there ({f})")
            else:
                _download(fetch, url, f, log)
    except Exception as e:  # noqa: BLE001  (HTTPError, URLError, timeouts, OSError, tarfile errors)
        log(f"calib --install-voice: {e}")
        log("  nothing half-installed is kept; run `calib --install-voice` again (what is already there stays)")
        return 2
    os.chmod(binary, os.stat(binary).st_mode | 0o755)
    # the check: piper speaks a sentence into a temporary wav
    with tempfile.TemporaryDirectory(prefix="calib-voice-check-") as td:
        wav = Path(td) / "check.wav"
        try:
            r = run([str(binary), "--model", str(model), "--output_file", str(wav)],
                    input=(TEST_SENTENCE + "\n").encode(), capture_output=True, timeout=120)
            ok = r.returncode == 0 and wav.is_file() and wav.stat().st_size > 44
            why = (r.stderr or b"").decode(errors="replace").strip()[-400:] if not ok else ""
        except (OSError, subprocess.SubprocessError) as e:
            ok, why = False, str(e)
    if not ok:
        log(f"calib --install-voice: the files are in {dest}, but piper did not speak the test sentence: "
            f"{why or 'no wav written'}")
        return 2
    log(f"voice installed and checked: {dest}")
    player = next((p for p in PLAYERS if which(p)), None)
    if player:
        log(f"player: {player}")
    else:
        log("no player found: the voice needs pw-play (PipeWire) or aplay (alsa-utils), e.g. "
            "`sudo apt install pipewire-bin` or `sudo apt install alsa-utils`")
    log('test it: calib --voice piper --say "Voice test"   (the menu offers the Piper voice from now on)')
    return 0 if player else 1
