"""Per hold: the longest gap in the recorded /joints stream inside the hold window, the joint sample
rate there, and the drift; then the camera-vs-gravity attribution on gap-free holds only."""
import sys
import numpy as np
take = sys.argv[1]
s = dict(np.load(f"{take}-signals.npz"))
jt = s["jt"]
print(f"{take}: /joints {len(jt)} samples, overall {(len(jt) - 1) / ((jt[-1] - jt[0]) / 1e9):.0f} Hz; gaps > 50 ms at (s, ms):",
      [(round((jt[k] - jt[0]) / 1e9, 1), round((jt[k + 1] - jt[k]) / 1e6)) for k in np.flatnonzero(np.diff(jt) > 50e6)])
for k, (t0, t1, y, p, ay, ap, drift) in enumerate(s["holds"]):
    m = (jt >= t0) & (jt <= t1)
    g = np.diff(jt[m]).max() / 1e6 if m.sum() > 1 else float("nan")
    print(f"  hold {k:2d}: {(t0 - jt[0]) / 1e9:6.1f}-{(t1 - jt[0]) / 1e9:6.1f} s, joints {m.sum() / ((t1 - t0) / 1e9):4.0f} Hz, "
          f"longest gap {g:7.1f} ms, drift {np.degrees(drift):.2f} deg")
