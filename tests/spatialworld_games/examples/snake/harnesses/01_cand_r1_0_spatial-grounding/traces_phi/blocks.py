import sys, json, math
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path("/home/azureuser/workspace/AutoSpatial")))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from phi_probe import phi, Z, META
p, cols, raw = phi(sys.argv[1])
C = Z["Z"]; off=0; per={}
for s,c in zip(META["blocks"], cols):
    per[s["name"]]=(off,off+len(c)); off+=len(c)
d = np.linalg.norm(C-p[None,:],axis=1)
print(f"{'cell':>5} {'total':>8} " + " ".join(f"{n:>8}" for n in per))
for j in np.argsort(d):
    row = " ".join(f"{np.linalg.norm(C[j][a:b]-p[a:b]):8.4f}" for a,b in per.values())
    print(f"{j+1:>5} {d[j]:8.4f} {row}")
