"""loc3.py — OFFLINE descriptor localizer + cell-3 founder finder over the FROZEN cvt_v3_k32
partition.  Lives under traces_phi3/ so phi SKIPS it (DEFAULT_SKIP_PREFIXES = (..., "traces")).
Usage:
    python traces_phi3/loc3.py founders          # which corpus points found each cell
    python traces_phi3/loc3.py loc <dir> ...     # where does <dir> land
"""
import json, math, sys
from pathlib import Path
import numpy as np

REPO = Path("/home/azureuser/workspace/AutoSpatial")
sys.path.insert(0, str(REPO))
from harness_map.qd import descriptors as D   # noqa: E402

RUNS = REPO / "spatial_world/game_baselines/runs_adaptive"
CVT = RUNS / "snake__k32_claude_pv/cvt_v3_k32"
STAGED = RUNS / "snake__k32_claude_pv/staged_anchor"
BOOT = RUNS / "snake/bootstrap"

Z = np.load(CVT / "cvt_blocks.npz")
META = json.loads((CVT / "blocks_meta.json").read_text())
ZC = Z["Z"]

OFFS, _o = [], 0
for _s in META["blocks"]:
    OFFS.append((_s["name"], _o, _o + _s["dim"])); _o += _s["dim"]


def phi(cd):
    bpd = D.desc_blocks(cd, D.DEFAULT_ROLE_RULES, CVT, META["descriptor"])
    cols, raw = [], {}
    for i, spec in enumerate(META["blocks"]):
        b = bpd[i][1]
        raw[spec["name"]] = b
        x = (np.array([b.get(k, 0.0) for k in spec["keys"]], float)
             if spec["kind"] == "dict" else np.asarray(b, float))
        mu = Z[f"mu_{i}"]; sd = np.where(Z[f"sd_{i}"] < 1e-9, 1.0, Z[f"sd_{i}"])
        cols.append((x - mu) / sd / math.sqrt(max(spec["dim"], 1)))
    return np.hstack(cols), raw


def report(cd, want=3):
    p, raw = phi(cd)
    d = np.linalg.norm(ZC - p[None, :], axis=1)
    order = np.argsort(d)
    others = min(d[j] for j in range(len(ZC)) if j != want - 1)
    per = {nm: float(np.linalg.norm(ZC[want - 1][a:b] - p[a:b])) for nm, a, b in OFFS}
    print(f"\n=== {Path(cd).name} ===")
    print(f"  CELL={int(order[0])+1}  d={d[order[0]]:.4f}   "
          f"d(c{want})={d[want-1]:.4f}  margin={others - d[want-1]:+.4f}")
    print("  top6: " + ", ".join(f"c{int(j)+1}:{d[j]:.4f}" for j in order[:6]))
    print(f"  per-block to c{want}: " + "  ".join(f"{k}={v:.4f}" for k, v in per.items()))
    print("  ast:", {k: round(v, 4) for k, v in sorted(raw["ast"].items()) if abs(v) > 1e-9})
    print("  num:", {k: round(v, 4) for k, v in raw["num"].items()})
    print("  wl0:", {k: round(v, 4) for k, v in sorted(raw["wl"].items()) if k.startswith("0:")})
    return int(order[0]) + 1, d


def founders():
    dirs = [STAGED] + [d for d in sorted(BOOT.glob("cand_*")) if (d / "system_prompt.txt").exists()]
    print(f"corpus candidates on disk: {len(dirs)}")
    P, keep = [], []
    for d in dirs:
        try:
            v, _ = phi(d)
        except Exception as e:
            print("  skip", d.name, repr(e)); continue
        P.append(v); keep.append(d)
    P = np.stack(P)
    lab = np.argmin(((P[:, None, :] - ZC[None]) ** 2).sum(-1), axis=1)
    out = {}
    for d, j in zip(keep, lab):
        out.setdefault(int(j) + 1, []).append(d)
    for cid in sorted(out):
        mem = out[cid]
        mean = np.mean(np.stack([phi(m)[0] for m in mem]), 0)
        fix = float(np.linalg.norm(mean - ZC[cid - 1]))
        print(f"cell {cid:>2}: n={len(mem):>2}  |mean-centroid|={fix:.5f}  "
              + ", ".join(m.name for m in mem))
    print("\nEMPTY-of-corpus cells:", [c for c in range(1, len(ZC) + 1) if c not in out])
    return out, keep, P


if __name__ == "__main__":
    if sys.argv[1] == "founders":
        founders()
    else:
        for a in sys.argv[2:]:
            report(a)
