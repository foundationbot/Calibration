"""Step 40 must hand kalibr_bagcreater each image exactly once.

kalibr_bagcreater (ethz-asl/kalibr 1f60227) finds camera folders with os.walk over the
WHOLE tree but keeps only their NAMES, then reads `<tree>/<name>`. Step 10 puts the
held-out pairs in `<tree>/heldout2/cam0`, so on phantom12 (2026-10-15) the bag got the
top-level cam0 twice: 900 messages per camera for 450 images. Step 40 therefore bags a
staged copy of the tree holding only its top-level cam* folders and imu*.csv files.
"""
import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "40_calibrate.sh"


# --- verbatim from kalibr_bagcreater (aslam_offline_calibration/kalibr/python) ---------
def getImageFilesFromDir(dir):
    image_files = list()
    timestamps = list()
    if os.path.exists(dir):
        for path, names, files in os.walk(dir):
            for f in files:
                if os.path.splitext(f)[1] in ['.bmp', '.png', '.jpg']:
                    image_files.append(os.path.join(path, f))
                    timestamps.append(os.path.splitext(f)[0])
    sort_list = sorted(zip(timestamps, image_files))
    return [file[1] for file in sort_list]


def getCamFoldersFromDir(dir):
    cam_folders = list()
    if os.path.exists(dir):
        for path, folders, files in os.walk(dir):
            for folder in folders:
                if folder[0:3] == "cam":
                    cam_folders.append(folder)
    return cam_folders


def getImuCsvFiles(dir):
    imu_files = list()
    if os.path.exists(dir):
        for path, folders, files in os.walk(dir):
            for file in files:
                if file[0:3] == 'imu' and os.path.splitext(file)[1] == ".csv":
                    imu_files.append(os.path.join(path, file))
    return imu_files
# ----------------------------------------------------------------------------------------


def bagged(tree):
    """{topic: [image basenames]} and the imu files, as kalibr_bagcreater would write them."""
    topics = {}
    for name in getCamFoldersFromDir(tree):
        topics.setdefault(name, []).extend(
            os.path.basename(f) for f in getImageFilesFromDir(os.path.join(tree, name)))
    return topics, [os.path.basename(f) for f in getImuCsvFiles(tree)]


def make_tree(root: Path) -> Path:
    tree = root / "stereo"
    for cam in ("cam0", "cam1"):
        (tree / cam).mkdir(parents=True)
        (tree / "heldout2" / cam).mkdir(parents=True)
        for ts in (100, 200, 300):
            (tree / cam / f"{ts}.png").write_bytes(b"")
        (tree / "heldout2" / cam / "150.png").write_bytes(b"")
    (tree / "imu0.csv").write_text("t,wx,wy,wz,ax,ay,az\n")
    (tree / "SOURCE.json").write_text("{}")
    return tree


def stage(tree: Path) -> Path:
    """Run step 40's stage_tree() alone and return the staged directory."""
    out = subprocess.run(
        ["bash", "-c", 'eval "$(sed -n "/^stage_tree()/,/^}/p" "$0")"; stage_tree "$1"',
         str(SCRIPT), str(tree)],
        check=True, capture_output=True, text=True)
    return Path(out.stdout.strip())


def test_the_raw_tree_is_bagged_twice(tmp_path):
    # Pins the Kalibr behaviour this staging exists for.
    topics, _ = bagged(make_tree(tmp_path))
    assert topics["cam0"] == ["100.png", "200.png", "300.png"] * 2


def test_the_staged_tree_bags_each_image_once_and_no_heldout(tmp_path):
    staged = stage(make_tree(tmp_path))
    topics, imu = bagged(str(staged))
    assert topics == {"cam0": ["100.png", "200.png", "300.png"],
                      "cam1": ["100.png", "200.png", "300.png"]}
    assert imu == ["imu0.csv"]
