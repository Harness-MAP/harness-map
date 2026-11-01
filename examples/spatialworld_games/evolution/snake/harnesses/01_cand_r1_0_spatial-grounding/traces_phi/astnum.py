import sys, json, math
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/azureuser/workspace/AutoSpatial")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness_map.qd import descriptors as D
CVT = Path("/home/azureuser/workspace/AutoSpatial/spatial_world/game_baselines/runs_adaptive/snake__k32_claude_pv/cvt_v3_k32")
Z = np.load(CVT/"cvt_blocks.npz"); META = json.loads((CVT/"blocks_meta.json").read_text())
C = Z["Z"]
offs={}; off=0
for s in META["blocks"]: offs[s["name"]]=(off,off+s["dim"]); off+=s["dim"]
tgt = int(sys.argv[1]) - 1
for d in sys.argv[2:]:
    fa, fn = D.f_AST(d), D.f_num(d)
    print(f"\n### {Path(d).name}")
    for bi, (nm, raw) in enumerate([("ast", fa), ("num", fn)], start=1):
        spec = META["blocks"][bi]
        keys = spec["keys"]; a,b = offs[nm]
        x = np.array([raw[k] for k in keys], float)
        mu = Z[f"mu_{bi}"]; sd = np.where(Z[f"sd_{bi}"]<1e-9,1.0,Z[f"sd_{bi}"])
        zx = (x-mu)/sd/math.sqrt(spec["dim"])
        cz = C[tgt][a:b]
        print(f"  -- {nm} (dim {spec['dim']}) total d={np.linalg.norm(zx-cz):.4f}")
        for k, xv, muv, sdv, zv, cv in zip(keys, x, mu, sd, zx, cz):
            tv = muv + cv*sdv*math.sqrt(spec["dim"])
            print(f"     {k:<16} raw={xv:9.4f} centroid_raw={tv:9.4f} |dz|={abs(zv-cv):.4f}  (sd={sdv:.4f})")
