"""H-1b: extra candidate terms on top of M12, judged only by leave-one-arm-pose-out cross-validation."""
import sys, json
import numpy as np
from multiprocessing import get_context
import h1
Z = h1.P_zero
M12 = h1.MODELS["M12"]
h1.MODELS.clear()
h1.MODELS.update({
    "M12": M12,
    "M12+yaw0": M12 + [("Ryaw0", Z("RightShoulderYaw")), ("Lyaw0", Z("LeftShoulderYaw"))],
    "M12+elbow0": M12 + [("Relb0", Z("RightElbowPitch")), ("Lelb0", Z("LeftElbowPitch"))],
    "M12+pitch0": M12 + [("Rpit0", Z("RightShoulderPitch")), ("Lpit0", Z("LeftShoulderPitch"))],
    "M12+head": M12 + h1.HEAD,
})
if __name__ == "__main__":
    h1.main()
