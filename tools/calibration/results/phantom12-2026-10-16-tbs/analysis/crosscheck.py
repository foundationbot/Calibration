"""Read-only: per hold, the board's heading in the torso frame (camera + neck encoders + config X)
against the torso IMU's integrated yaw. If the body turned, they move together (opposite signs)."""
import importlib, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import dmarec_imu, necktake, neckfk, okviscfg, rigcfg, rigmath as rm
s81 = importlib.import_module("81_tbs_handeye")
take = sys.argv[1]; W = Path("/w/takes")
rig = rigcfg.load(Path("/w/config/rig-phantom12.yaml")); cfg = okviscfg.load(Path("/w/phantom12_stereo.yaml"))
chain = neckfk.NeckChain(Path("/t/config/foundation_rig/20250206_phantom_mk1.urdf"))
j = necktake.read_joints(W / take / "session.mcap"); holds = necktake.find_holds(j)
fr = s81._frames(SimpleNamespace(export=W / f"{take}-euroc", per_hold=3), rig, cfg, j, holds, set())
obs, meta = s81.observations(fr, chain, cfg)
X = cfg.T_BS @ cfg.T_SC[0]
Zs = s81.board_in_torso(obs, X)
segs = sorted((W / f"{take}-dmarec").glob("recording_*.mcap"))
t, g, a, _ = dmarec_imu.read_raw_imu(segs, 9, int(j.t[0]), int(j.t[-1]))
up = np.median(a, 0); up /= np.linalg.norm(up)
dt = np.minimum(np.diff(t, prepend=t[0]) / 1e9, 0.006)
yaw = np.degrees(np.cumsum(((g - np.median(g, 0)) @ up) * dt))
rows = []
for h in sorted({m["hold"] for m in meta}):
    Zh = [Z for Z, m in zip(Zs, meta) if m["hold"] == h]
    head = np.degrees(np.median([np.arctan2(Z[1, 0], Z[0, 0]) for Z in Zh]))     # board x axis heading in torso xy
    pos = 1e3 * np.median([Z[:3, 3] for Z in Zh], 0)
    hm = (t >= holds[h].t0) & (t <= holds[h].t1)
    rows.append((h, head, yaw[hm].mean(), *pos))
r = np.array(rows)
r[:, 1] -= np.median(r[:, 1]); r[:, 2] -= np.median(r[:, 2])
print("hold | board heading in torso [deg] | torso IMU yaw [deg] | board position in torso [mm]")
for row in r:
    print(f"  {int(row[0]):2d}   {row[1]:+7.2f}   {row[2]:+7.2f}   {row[3]:7.0f} {row[4]:7.0f} {row[5]:7.0f}")
c = np.corrcoef(r[:, 1], r[:, 2])[0, 1]; k = np.polyfit(r[:, 2], r[:, 1], 1)[0]
print(f"correlation {c:+.3f}, slope {k:+.3f} (a turning body: about -1)")
