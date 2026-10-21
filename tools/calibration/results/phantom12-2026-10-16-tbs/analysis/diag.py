"""Per-image vs per-hold: split the fused (URDF neck) fit's residuals into within-hold scatter and
hold means; correlate hold means with neck angles and approach sides."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import fusedfit as ff
import neckfk, okviscfg
cfg, chain = okviscfg.load(Path(sys.argv[1])), neckfk.NeckChain(Path(sys.argv[2]))
takes = ff.load(sys.argv[3:])
m = ff.Model(cfg, chain, takes, False)
p, res, cov, cond = ff.fit(m)
k0 = 0
for k, t in enumerate(takes):
    n = len(t["hold"]); r = res[k0:k0 + n]; k0 += n
    rot = np.degrees(r[:, :3]); tr = 1e3 * r[:, 3:]
    within_r, within_t, rows = [], [], []
    for h in np.unique(t["hold"]):
        s = t["hold"] == h
        for c in (0, 1):
            sc = s & (t["cam"] == c)
            if sc.sum() >= 2:
                within_r.append(rot[sc].std(0)); within_t.append(tr[sc].std(0))
        rows.append((h, np.degrees(t["yaw"][s].mean()), np.degrees(t["pitch"][s].mean()), t["ay"][s][0], t["ap"][s][0],
                     *rot[s].mean(0), *tr[s].mean(0)))
    rows = np.array(rows)
    print(f"take {k}: within-hold scatter (same camera) rot {np.mean(within_r, 0).round(3)} deg, trans {np.mean(within_t, 0).round(2)} mm")
    print(f"          hold-mean spread               rot {rows[:, 5:8].std(0).round(3)} deg, trans {rows[:, 8:11].std(0).round(2)} mm")
    print("   hold  yaw  pitch  ay ap | mean residual rot x y z [deg] (camera axes) | trans [mm]")
    for row in rows:
        print(f"   {int(row[0]):3d} {row[1]:+5.0f} {row[2]:+5.0f}  {int(row[3]):+d} {int(row[4]):+d} | {row[5]:+.2f} {row[6]:+.2f} {row[7]:+.2f} | {row[8]:+5.1f} {row[9]:+5.1f} {row[10]:+5.1f}")
