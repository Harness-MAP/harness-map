"""phi_probe.py — OFFLINE descriptor probe (lives under traces_* so phi SKIPS it).

Replays harness_map/qd/descriptors.py + partition.make_phi_blocks against the FROZEN
cvt_v3_k32 partition, so a candidate's rung-0 cell can be read BEFORE the outer loop
places it. Usage:  python traces_phi/phi_probe.py <dir> [<dir> ...]
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/azureuser/workspace/AutoSpatial")
sys.path.insert(0, str(REPO))
CVT = REPO / "spatial_world/game_baselines/runs_adaptive/snake__k32_claude_pv/cvt_v3_k32"

from harness_map.qd import descriptors as D   # noqa: E402

Z = np.load(CVT / "cvt_blocks.npz")
META = json.loads((CVT / "blocks_meta.json").read_text())


def phi(cd):
    bpd = D.desc_blocks(cd, D.DEFAULT_ROLE_RULES, CVT, META["descriptor"])
    cols, raw = [], {}
    for i, spec in enumerate(META["blocks"]):
        b = bpd[i][1]
        if spec["kind"] == "dict":
            x = np.array([b.get(k, 0.0) for k in spec["keys"]], float)
        else:
            x = np.asarray(b, float)
        raw[spec["name"]] = b
        mu = Z[f"mu_{i}"]
        sd = np.where(Z[f"sd_{i}"] < 1e-9, 1.0, Z[f"sd_{i}"])
        cols.append((x - mu) / sd / math.sqrt(max(spec["dim"], 1)))
    return np.hstack(cols), cols, raw


def report(cd, want=21):
    p, cols, raw = phi(cd)
    C = Z["Z"]
    d = np.linalg.norm(C - p[None, :], axis=1)
    order = np.argsort(d)
    cell = int(order[0]) + 1
    names = [s["name"] for s in META["blocks"]]
    off = 0
    per = {}
    for s, c in zip(META["blocks"], cols):
        per[s["name"]] = (off, off + len(c))
        off += len(c)
    tw = want - 1
    sub = {nm: float(np.linalg.norm(C[tw][a:b] - p[a:b])) for nm, (a, b) in per.items()}
    print(f"\n=== {Path(cd).name} ===")
    print(f"  CELL = {cell}   d={d[order[0]]:.4f}   runner-up {order[1]+1} d={d[order[1]]:.4f}")
    print(f"  d(cell{want}) = {d[tw]:.4f}   margin(runner-up - cell{want}) = "
          f"{min(d[j] for j in range(len(C)) if j != tw) - d[tw]:+.4f}")
    print("  per-block distance to cell%d: " % want
          + "  ".join(f"{k}={v:.4f}" for k, v in sub.items()))
    print("  top5 cells: " + ", ".join(f"{j+1}:{d[j]:.4f}" for j in order[:5]))
    print("  ast:", {k: round(v, 4) for k, v in sorted(raw["ast"].items()) if v})
    print("  num:", {k: round(v, 4) for k, v in raw["num"].items()})
    print("  wl :", {k: round(v, 4) for k, v in sorted(raw["wl"].items())})
    return cell, d


if __name__ == "__main__":
    for a in sys.argv[1:]:
        report(a)
