"""ast3.py — per-component control-flow + literal census (what f_AST / f_num actually see)."""
import ast, sys, math
from pathlib import Path
sys.path.insert(0, "/home/azureuser/workspace/AutoSpatial")
from harness_map.qd import descriptors as D


def census(cd):
    print(f"\n### {Path(cd).name}")
    tot = {t: 0 for t in D._CF_TYPES}
    nums, ncomp, per = [], 0, []
    for name, seg in D._component_srcs(cd):
        ncomp += 1
        t = ast.parse(seg)
        loc, lits = {}, []
        c = 0
        for nd in ast.walk(t):
            nm = type(nd).__name__
            if nm in tot:
                tot[nm] += 1; loc[nm] = loc.get(nm, 0) + 1; c += 1
            if isinstance(nd, ast.Constant) and isinstance(nd.value, (int, float)) \
                    and not isinstance(nd.value, bool):
                lits.append(nd.value); nums.append(abs(float(nd.value)))
        per.append(c)
        print(f"  {name:<28} cf={c:<3} {loc}  lits={lits}")
    logs = [math.log1p(x) for x in nums]
    n = max(len(logs), 1); mu = sum(logs) / n
    print(f"  TOTAL cf={sum(per)} over {ncomp} comps  cf_mean={sum(per)/ncomp:.4f} cf_max={max(per)}")
    print("  cf types:", {k: v for k, v in tot.items() if v})
    print(f"  nums={len(nums)} per_comp={len(nums)/ncomp:.4f} logmag_mean={mu:.4f} "
          f"logmag_std={(sum((x-mu)**2 for x in logs)/n)**0.5:.4f}")


if __name__ == "__main__":
    for a in sys.argv[1:]:
        census(a)
