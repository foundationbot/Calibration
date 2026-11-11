"""armcal: the hand-eye (camera-to-arm) calibration's computing parts: read B (each capture's forearm and wrist meshes
matched in the head depth, B0) and the stage-2 fit of a few URDF origins over the clean captures of both arms. See README.md (this folder) and
docs/plans/2026-11-11-handeye-plan.md.

Ported on 11 Nov 2026 from OKVIS2-X tools/burst_pointcloud (c5b4c6b0: meshfit, hand_check, frames, armfk, mesh,
burst_io; _burstfusion's depth fusion as numpy) and the phantom14 stage-2 scripts (Moonstone p14arm/urdfx). Needs
numpy, scipy and cv2 only. (Not named handeye: scripts/handeye.py, the head calibration's T_BS hand-eye, is on
sys.path in the same test run.)"""
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent              # tools/calibration/armcal
CAL = PKG.parent                                   # tools/calibration
SCRIPTS = CAL / "scripts"                          # neckfk, rigmath (shared with the head calibration)
MESHES = PKG / "meshes"                            # the arms' CAD meshes (forearm and wrist links, both sides)
REPO_URDF = CAL.parents[1] / "config" / "foundation_rig" / "20250206_phantom_mk1.urdf"   # #816, = phantom14's base

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
