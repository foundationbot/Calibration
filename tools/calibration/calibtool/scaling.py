"""An OKVIS config for another camera-producer size (design plan 6.3): the lens is fitted at the sensor's 1280x800, the
robot may run its producer at 960x600 or 640x400, and VIO always runs at 640x400.

The rule, for a resize by s (scaled size = native size x s):
  focal_length      f x s
  principal_point   (c + 0.5) x s - 0.5: the half-pixel-centre convention of cv::resize, which OKVIS's own resize
                    of the SLAM intrinsics uses (okvis_common/src/ViParametersReader.cpp, frontend.resize_factor).
                    phantom10's 960 configs (kit cal960.py) used c x s: 0.125 px lower at s = 0.75.
  image_dimension   the scaled size, whole numbers (a factor that gives a fractional size is refused)
  resize_factor     (frontend_parameters) such that OKVIS's int(dim x resize_factor) (ThreadedSlam::addImages, a
                    double) gives exactly the VIO size, in float64 and in float32, and within RF_REL_TOL of the true
                    ratio (OKVIS scales the SLAM intrinsics by it too): the shortest decimal at or above the ratio
                    that does. A value already in the file that satisfies this is kept as written.
Distortion terms (radtan k1 k2 p1 p2, KB4/KB4T k1..k4 p1 p2) act on normalised coordinates: unchanged. Nothing else
in the file changes: only the numbers of those keys, on lines where the key is not inside a comment, are rewritten;
a trailing comment stays as it was. Stdlib only."""
import math
import re
import struct

VIO_SIZE = (640, 400)
RF_REL_TOL = 1e-7               # resize_factor vs the exact ratio: f x 1e-7 = 6e-5 px at f = 600 px

_LIST = re.compile(r"\b(?P<key>image_dimension|focal_length|principal_point)(?P<pre>\s*:\s*\[\s*)(?P<a>[^,\]\s]+)"
                   r"(?P<sep>\s*,\s*)(?P<b>[^,\]\s]+)(?P<post>\s*\])")
_RF = re.compile(r"^(?P<pre>\s*resize_factor\s*:\s*)(?P<v>[-+0-9.eE]+)")


def scaled_pp(c, s):
    return (c + 0.5) * s - 0.5


def unscaled_pp(c, s):
    return (c + 0.5) / s - 0.5


def _f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def _gives(rf, size, vio):
    """int(dim x rf) == vio for both sides, rf read as a double and as a float32"""
    return all(int(d * r) == v for r in (rf, _f32(rf)) for d, v in zip(size, vio))


def resize_factor_text(size, vio=VIO_SIZE):
    """the resize_factor to write for images of `size` and VIO at `vio`: "1.0" when they are equal"""
    size, vio = tuple(size), tuple(vio)
    if size == vio:
        return "1.0"
    ratio = max(v / d for v, d in zip(vio, size))
    if ratio > 1:
        raise ValueError(f"VIO {vio} is larger than the images {size}: OKVIS only shrinks (resize_factor <= 1)")
    for n in range(1, 17):
        t = f"{math.ceil(ratio * 10 ** n - 1e-9) / 10 ** n:.{n}f}".rstrip("0")
        rf = float(t)
        if abs(rf - ratio) <= RF_REL_TOL * ratio and _gives(rf, size, vio):
            return t
    raise ValueError(f"no resize_factor turns {size} into exactly {vio} (the sides need different factors)")


def _code(line):
    """(the part of a line before any comment, the rest)"""
    i = line.find("#")
    return (line, "") if i < 0 else (line[:i], line[i:])


def _whole(x, what):
    r = round(x)
    if abs(x - r) > 1e-6:
        raise ValueError(f"image_dimension: {what} gives {x:g}, not a whole number of pixels")
    return int(r)


def config_size(text):
    """(W, H) of the config's camera blocks (all of them must agree)"""
    sizes = set()
    for line in text.splitlines():
        for m in _LIST.finditer(_code(line)[0]):
            if m["key"] == "image_dimension":
                sizes.add((int(float(m["a"])), int(float(m["b"]))))
    if len(sizes) != 1:
        raise ValueError(f"image_dimension: {sorted(sizes) or 'none'}; want one size for every camera")
    return sizes.pop()


def factor(scaled_size, native_size):
    """s with native x s == scaled on both sides"""
    s = scaled_size[0] / native_size[0]
    if abs(native_size[1] * s - scaled_size[1]) > 1e-9:
        raise ValueError(f"{tuple(native_size)} -> {tuple(scaled_size)} is not one factor for both sides")
    return s


def _rewrite(text, dim, focal, pp, vio):
    size = config_size(text)
    new_size = tuple(_whole(dim(d), f"{d} px") for d in size)
    counts = {"image_dimension": 0, "focal_length": 0, "principal_point": 0}
    fn = {"focal_length": focal, "principal_point": pp}

    def sub(m):
        k = m["key"]
        counts[k] += 1
        if k == "image_dimension":
            a, b = new_size
        else:
            a, b = (repr(float(fn[k](float(x)))) for x in (m["a"], m["b"]))
        return f"{k}{m['pre']}{a}{m['sep']}{b}{m['post']}"

    lines, rf_lines = text.split("\n"), []
    for i, line in enumerate(lines):
        code, rest = _code(line)
        code = _LIST.sub(sub, code)
        if _RF.match(code):
            rf_lines.append(i)
        lines[i] = code + rest
    if not (counts["image_dimension"] == counts["focal_length"] == counts["principal_point"] >= 1):
        raise ValueError(f"camera keys found {counts}: want image_dimension, focal_length and principal_point once "
                         "per camera")
    if len(rf_lines) > 1:
        raise ValueError(f"resize_factor appears {len(rf_lines)} times outside comments")
    if vio is not None:
        if rf_lines:
            i = rf_lines[0]
            code, rest = _code(lines[i])
            m = _RF.match(code)
            have = float(m["v"])
            ratio = max(v / d for v, d in zip(vio, new_size))
            if not (abs(have - ratio) <= RF_REL_TOL * ratio and _gives(have, new_size, vio)):
                code = m["pre"] + resize_factor_text(new_size, vio) + code[m.end():]
            lines[i] = code + rest
        elif new_size != tuple(vio):
            raise ValueError(f"no resize_factor in frontend_parameters: images {new_size} would run VIO at that "
                             f"size, not {tuple(vio)}")
    return "\n".join(lines)


def scale_config(text, s, vio=VIO_SIZE):
    """the config for images s x its size (s = 0.75: 1280x800 -> 960x600). vio=None leaves resize_factor alone"""
    return _rewrite(text, lambda d: d * s, lambda f: f * s, lambda c: scaled_pp(c, s), vio)


def unscale_config(text, s, vio=VIO_SIZE):
    """scale_config's inverse: a config written for s x the native size back to the native size"""
    return _rewrite(text, lambda d: d / s, lambda f: f / s, lambda c: unscaled_pp(c, s), vio)
