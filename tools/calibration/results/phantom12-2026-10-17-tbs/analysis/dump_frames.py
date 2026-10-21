"""Read-only on the takes: step 81's per-image observations (hold, cam, neck angles, board pose) saved
as an npz, so fits can be rerun without the image pass. Writes /w/tbs/<take>-frames.npz."""
import importlib, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import necktake, okviscfg, rigcfg
s81 = importlib.import_module("81_tbs_handeye")
take = sys.argv[1]; W = Path("/w/takes")
rig = rigcfg.load(Path("/w/config/rig-phantom12.yaml")); cfg = okviscfg.load(Path("/w/phantom12_stereo.yaml"))
j = necktake.read_joints(W / take / "session.mcap"); holds = necktake.find_holds(j)
fr, n_gap = s81._frames(SimpleNamespace(export=W / f"{take}-euroc", per_hold=5), rig, cfg, j, holds, set())
np.savez(f"/w/tbs/{take}-frames.npz", hold=np.array([f["hold"] for f in fr]), cam=np.array([f["cam"] for f in fr]),
         yaw=np.array([f["yaw"] for f in fr]), pitch=np.array([f["pitch"] for f in fr]),
         T_CW=np.array([f["T_CW"] for f in fr]), ay=np.array([f["approach_yaw"] for f in fr]),
         ap=np.array([f["approach_pitch"] for f in fr]))
print(f"{take}: {len(fr)} images over {len({f['hold'] for f in fr})} holds -> /w/tbs/{take}-frames.npz")
