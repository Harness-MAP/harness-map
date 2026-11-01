import sys, json, math
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/azureuser/workspace/AutoSpatial")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from phi_probe import phi, Z, META
BOOT = Path("/home/azureuser/workspace/AutoSpatial/spatial_world/game_baselines/runs_adaptive/snake/bootstrap")
ANCH = Path("/home/azureuser/workspace/AutoSpatial/spatial_world/game_baselines/runs_adaptive/snake__k32_claude_pv/staged_anchor")
dirs = [ANCH] + [d for d in sorted(BOOT.glob("cand_*")) if (d/"system_prompt.txt").exists()]
C = Z["Z"]
off=0; per={}
for s in META["blocks"]:
    per[s["name"]]=(off,off+ (s["dim"])); off+=s["dim"]
rows=[]
for d in dirs:
    p,_c,_r = phi(d)
    dd = np.linalg.norm(C-p[None,:],axis=1)
    j=int(np.argmin(dd))
    rows.append((j+1, float(dd[j]), d.name, p))
from collections import Counter
cnt = Counter(r[0] for r in rows)
print("occupancy:", dict(sorted(cnt.items())))
print("\n-- members of cell 21 --")
for cell, dist, nm, p in rows:
    if cell==21: print(f"  {nm}  d={dist:.4f}")
print("\n-- members of cell 12 --")
for cell, dist, nm, p in rows:
    if cell==12: print(f"  {nm}  d={dist:.4f}")
np.save("/tmp/corpus_phi.npy", np.stack([r[3] for r in rows]))
Path("/tmp/corpus_names.json").write_text(json.dumps([[r[0],r[1],r[2]] for r in rows]))
