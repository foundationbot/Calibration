#!/usr/bin/env python3
"""Put phantom12's calibration into the OKVIS config the robot already runs, changing nothing else.

    transplant.py <robot's live yaml> <our step-70 yaml> <out yaml> [--check]

Replaced, from our yaml: per camera T_SC, image_dimension, focal_length, principal_point,
distortion_coefficients, distortion_type, plus dma_queue 1 / 2; camera_parameters.image_delay;
imu_parameters.T_BS; and a new dma.imu_name (NeckPitch). Every other line of the live file is
kept as it is: estimator, frontend, depth, mapping and output tuning belong to whoever runs OKVIS
there, not to the calibration. Text surgery rather than a YAML round trip, so comments and
formatting survive and the diff shows only the calibration.

--check parses all three files with OpenCV FileStorage (the parser OKVIS uses) and prints every
leaf that differs between live and out, and any changed value that is not our calibration's.
Used on 2026-10-16 to make deploy/phantom12_stereo_dense_thor.yaml (README.md beside this file).
"""
import re
import sys

CAL_KEYS = ("image_dimension", "distortion_coefficients", "distortion_type", "focal_length",
            "principal_point")
OLD_SLAM_COMMENT = "#none, okvis, okvis-depth, okvis-virtual"
HEADER = """%YAML:1.0
# phantom12_stereo_dense_thor.yaml -- phantom12's okvis_stereo_dense_thor.yaml (sha256 95fce9a8,
# 2026-08-09) with ONLY its calibration replaced, 2026-10-16. All tuning is the 9 Aug file's.
# Replaced: per camera T_SC, image_dimension, focal_length, principal_point, distortion
#   (equidistant), plus dma_queue 1 / 2; camera_parameters.image_delay; imu_parameters.T_BS;
#   added dma.imu_name NeckPitch.
# Source: tools/calibration/results/phantom12-2026-10-15/imucam/phantom12_stereo.yaml on the
#   stereo-depth-eval work branch (Kalibr, 15 Oct): stereo baseline 44.83 mm, held-out row error
#   0.204 px, scale within +-1 % from 0.4 to 2.1 m; camera-IMU take B, reprojection 0.29 / 0.26 px,
#   repeatable to 4.4 mm / 0.08 deg; image_delay -0.0056 s.
# T_BS is phantom21's head-design value, UNVERIFIED on phantom12: heading is consistent (cameras
#   0.2 deg off the Head's forward axis in azimuth), but with it the cameras look 30 deg down in
#   the Head frame (phantom21 ~10 deg). Depth and VIO do not depend on it; Head/Pelvis-frame
#   outputs do. The previous file: okvis_stereo_dense_thor.yaml beside this one, unchanged.
"""


def cam_blocks(text):
    return list(re.finditer(r"- \{T_SC:.*?\}", text, flags=re.S))


def field_line(block, key):
    m = re.search(rf"^\s*{key}:.*$", block, flags=re.M)
    assert m, key
    return m.group(0).strip()


def tsc_text(block):
    m = re.search(r"T_SC:\s*\n(\s*\[.*?\]),", block, flags=re.S)
    assert m
    return m.group(1)


def transplant(live: str, ours: str) -> str:
    ob, lb = cam_blocks(ours), cam_blocks(live)
    assert len(ob) == 2 and len(lb) == 2, "expects two cameras in both files"
    out, pos = [], 0
    for i, (l, o) in enumerate(zip(lb, ob)):
        new = l.group(0).replace(tsc_text(l.group(0)), tsc_text(o.group(0)), 1)
        for k in CAL_KEYS:
            new = new.replace(field_line(new, k), field_line(o.group(0), k), 1)
        q = i + 1                                    # OKVIS cam0 = queue 1 (left), cam1 = queue 2
        side = "left, socket B" if q == 1 else "right, socket D"
        new = new.replace("slam_use: okvis}", f"slam_use: okvis, @@C\n        dma_queue: {q}}} "
                          f"# OKVIS cam{i} = DMA queue {q} ({side}); OKVIS_DMA_CAMERAS=1,2 agrees @@D", 1)
        out += [live[pos:l.start()], new]
        pos = l.end()
    text = "".join(out + [live[pos:]])
    # the live file's comment sat after the closing brace: move it back onto the slam_use line
    text, n = re.subn(rf"@@C\n(.*?) @@D {OLD_SLAM_COMMENT}", rf"{OLD_SLAM_COMMENT}\n\1", text)
    assert n == 2, n
    text, n = re.subn(r"(\n\s*image_delay:\s*)0\.02(\s*#)", r"\g<1>-0.0056\2", text)
    assert n == 1, "image_delay"
    m_live = re.search(r"T_BS:\s*\n(\s*\[.*?\])", text, flags=re.S)
    m_ours = re.search(r"T_BS:\s*\n(\s*\[.*?\])", ours, flags=re.S)
    assert m_live and m_ours
    text = text.replace(m_live.group(1), m_ours.group(1), 1)
    text = text.replace(
        "    # transform Body-Sensor (IMU)\n",
        "    # transform Body-Sensor (IMU). Body = the URDF Head link. phantom21's head-design value (CAD\n"
        "    # Head->camera combined with phantom21's T_SC); UNVERIFIED on phantom12 (see header).\n"
        "    # It replaces an identity that made OKVIS take the IMU's +x as the robot's front.\n", 1)
    assert "UNVERIFIED on phantom12 (see header)" in text
    assert not re.search(r"^dma:", text, flags=re.M), "live file already has a dma section"
    text = (text.rstrip("\n") + "\n\n# DMA transport. imu_name picks the head IMU by its RobotConfigData"
            " name; without it OKVIS\n# used slot 0, which on phantom12 is RightFoot.\ndma:\n"
            "    imu_name: \"NeckPitch\"\n")
    assert text.startswith("%YAML:1.0\n")
    return HEADER + text[len("%YAML:1.0\n"):]


def check(live_path, ours_path, out_path) -> int:
    import cv2

    def leaves(node, path):
        if node.isMap():
            return {k2: v for k in node.keys() for k2, v in leaves(node.getNode(k), f"{path}.{k}").items()}
        if node.isSeq():
            return {k2: v for i in range(node.size()) for k2, v in leaves(node.at(i), f"{path}[{i}]").items()}
        if node.isString():
            return {path: node.string()}
        return {path: node.real() if (node.isReal() or node.isInt()) else None}

    def load(p):
        fs = cv2.FileStorage(str(p), cv2.FILE_STORAGE_READ)
        assert fs.isOpened(), p
        r = fs.root()
        return {k2: v for k in r.keys() for k2, v in leaves(r.getNode(k), k).items()}

    L, O, T = load(live_path), load(ours_path), load(out_path)
    changed = sorted(k for k in set(L) | set(T) if L.get(k) != T.get(k))
    not_ours = [k for k in changed if T.get(k) != O.get(k)]
    print(f"{len(changed)} leaves differ from the live file:")
    for k in changed:
        print(f"  {k}")
    print("changed values that are not our calibration's:", not_ours or "none")
    return 1 if not_ours else 0


def main() -> int:
    args = [a for a in sys.argv[1:] if a != "--check"]
    if len(args) != 3:
        print(__doc__)
        return 2
    live_path, ours_path, out_path = args
    with open(live_path) as f:
        live = f.read()
    with open(ours_path) as f:
        ours = f.read()
    with open(out_path, "w") as f:
        f.write(transplant(live, ours))
    print(f"wrote {out_path}")
    return check(live_path, ours_path, out_path) if "--check" in sys.argv else 0


if __name__ == "__main__":
    sys.exit(main())
