# Where do board tags 0, 6, 28, 34 sit in one still left frame of take B? Saves a small JPEG to look at.
import os, sys, cv2, numpy as np
sys.path.insert(0, "/w/scripts"); import rigcfg
b = rigcfg.load("/w/config/rig-phantom12.yaml").board
d = "/e/cam1/data"; f = sorted(os.listdir(d))[int(sys.argv[1])]
img = cv2.imread(f"{d}/{f}"); g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
det = b.detect(g)
for tid in (0, 6, 28, 34):
    pts = [det[(tid, k)] for k in range(4) if (tid, k) in det]
    print(f"tag {tid:2d}: " + (f"centre ({np.mean([p[0] for p in pts]):.0f}, {np.mean([p[1] for p in pts]):.0f}) px" if pts else "not seen"))
for tid in (0, 6, 28, 34):
    pts = [det[(tid, k)] for k in range(4) if (tid, k) in det]
    if pts:
        c = tuple(int(v) for v in np.mean(pts, axis=0)); cv2.putText(img, str(tid), c, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
cv2.imwrite("/w/orient_probe.jpg", cv2.resize(img, (640, 400)), [cv2.IMWRITE_JPEG_QUALITY, 80])
print("frame", f, "size", img.shape)
