"""attribution.py restricted to holds whose recorded /joints stream has no gap > max_gap_ms inside."""
import sys, io, contextlib
import numpy as np
import attribution as at
take, max_gap = sys.argv[1], float(sys.argv[2])
s = dict(np.load(f"{take}-signals.npz"))
clean = set()
for k, (t0, t1, *_r) in enumerate(s["holds"]):
    m = (s["jt"] >= t0) & (s["jt"] <= t1)
    if m.sum() > 1 and np.diff(s["jt"][m]).max() / 1e6 <= max_gap:
        clean.add(k)
buf = io.StringIO()
sys.argv = ["attribution.py", take]
with contextlib.redirect_stdout(buf):
    at.main()
rows = [l.split() for l in buf.getvalue().splitlines() if l.strip()[:1].isdigit() or (l.strip().split("|")[0].strip().isdigit())]
data = []
for l in buf.getvalue().splitlines():
    parts = l.replace("|", " ").split()
    if len(parts) == 5 and parts[0].isdigit():
        data.append([int(parts[0])] + [float(v) for v in parts[1:]])
d = np.array([r for r in data if r[0] in clean])
print(f"{take}: {len(d)} gap-free holds (<= {max_gap:g} ms): {sorted(int(h) for h in d[:, 0])}")
for i, nm in ((1, "Head x"), (2, "Head y")):
    c, g = d[:, i] - d[:, i].mean(), d[:, i + 2] - d[:, i + 2].mean()
    print(f"   about {nm}: correlation {np.corrcoef(c, g)[0, 1]:+.2f}, slope {np.polyfit(c, g, 1)[0]:+.2f}; "
          f"rms camera {np.sqrt(np.mean(c ** 2)):.2f}, gravity {np.sqrt(np.mean(g ** 2)):.2f} deg")
