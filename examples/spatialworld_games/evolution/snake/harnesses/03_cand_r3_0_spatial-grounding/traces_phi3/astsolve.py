"""astsolve.py — price an ast/num profile against cell 3's centroid slice, offline.

FIXED comps (agent.py + tools/action.py + tools/frame.py, untouched): per-comp cf = [1,3,4,0,3,0]
with types If 3, For 1, comprehension 4, BoolOp 2, IfExp 1 and literals [0, 0].
A candidate adds 8 components in tools/vlm.py; this scores a chosen (per-comp cf, type totals,
literal) profile without writing a line of code.
"""
import json, math, sys, itertools
from pathlib import Path
import numpy as np

CVT = Path("/home/azureuser/workspace/AutoSpatial/spatial_world/game_baselines/runs_adaptive/snake__k32_claude_pv/cvt_v3_k32")
Z = np.load(CVT / "cvt_blocks.npz"); META = json.loads((CVT / "blocks_meta.json").read_text())
OFF, o = {}, 0
for s in META["blocks"]:
    OFF[s["name"]] = (o, o + s["dim"], s["keys"]); o += s["dim"]
IDX = {s["name"]: i for i, s in enumerate(META["blocks"])}
TGT = Z["Z"][2]

CF_TYPES = ["If", "For", "AsyncFor", "While", "AsyncWith", "With", "Try",
            "comprehension", "BoolOp", "IfExp", "Break", "Continue"]
FIX_CF = [1, 3, 4, 0, 3, 0]
FIX_T = {"If": 3, "For": 1, "comprehension": 4, "BoolOp": 2, "IfExp": 1}
FIX_LIT = [0.0, 0.0]


def ast_vec(percomp, types):
    per = FIX_CF + list(percomp)
    n = len(per); mu = sum(per) / n
    tot = max(sum(types.values()) + sum(FIX_T.values()), 1)
    allt = dict(FIX_T)
    for k, v in types.items():
        allt[k] = allt.get(k, 0) + v
    d = {"cf_mean": mu, "cf_max": float(max(per)),
         "cf_std": (sum((x - mu) ** 2 for x in per) / n) ** 0.5}
    for t in CF_TYPES + ["While"]:
        d[f"p_{t}"] = allt.get(t, 0) / tot
    return d


def num_vec(lits):
    v = FIX_LIT + [abs(float(x)) for x in lits]
    logs = [math.log1p(x) for x in v]
    n = max(len(logs), 1); mu = sum(logs) / n
    return {"num_per_comp": len(v) / 14.0, "logmag_mean": mu,
            "logmag_std": (sum((x - mu) ** 2 for x in logs) / n) ** 0.5}


def block_d(name, raw):
    a, b, keys = OFF[name]
    i = IDX[name]
    x = np.array([raw.get(k, 0.0) for k in keys], float)
    mu = Z[f"mu_{i}"]; sd = np.where(Z[f"sd_{i}"] < 1e-9, 1.0, Z[f"sd_{i}"])
    p = (x - mu) / sd / math.sqrt(len(keys))
    return float(np.linalg.norm(p - TGT[a:b]))


def price(percomp, types, lits, tag=""):
    da = block_d("ast", ast_vec(percomp, types))
    dn = block_d("num", num_vec(lits))
    print(f"  {tag:<46} ast={da:.4f} num={dn:.4f} sum_sq={(da*da+dn*dn)**0.5:.4f}")
    return da, dn


if __name__ == "__main__":
    # the founder itself (sanity: both must be 0.0000)
    price([0, 1, 3, 2, 1, 2, 1, 9], {"If": 8, "BoolOp": 8, "IfExp": 2, "Try": 1},
          [1, 1, 1, 1], "FOUNDER (expect 0.0000 / 0.0000)")
    # my natural draft
    price([2, 2, 4, 2, 1, 3, 3, 9], {"If": 10, "BoolOp": 4, "IfExp": 6, "Try": 3},
          [], "draft A: If10 B4 X6 T3, 0 lits")
    price([2, 2, 4, 2, 2, 3, 3, 9], {"If": 8, "BoolOp": 8, "IfExp": 2, "Try": 1},
          [], "draft B: founder types, 0 lits")
    price([2, 2, 4, 2, 2, 3, 3, 9], {"If": 8, "BoolOp": 8, "IfExp": 2, "Try": 1},
          [1, 1, 1, 1], "draft C: founder types + 4x |1|")
    for ifs in (8, 9, 10, 11):
        for bo in (5, 6, 7, 8):
            for xp in (2, 3, 4):
                for tr in (1, 2):
                    tot = ifs + bo + xp + tr
                    pc = [2, 2, 3, 2, 2, 3, 3, tot - 17] if tot - 17 >= 2 else None
                    if pc is None or sum(pc) != tot:
                        continue
                    price(pc, {"If": ifs, "BoolOp": bo, "IfExp": xp, "Try": tr}, [],
                          f"If{ifs} B{bo} X{xp} T{tr} (vlm cf={tot})")
