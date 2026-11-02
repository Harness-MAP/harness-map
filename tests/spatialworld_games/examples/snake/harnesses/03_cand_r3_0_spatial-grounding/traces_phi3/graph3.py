"""graph3.py — print the labelled component graph phi actually builds, and diff two dirs' WL."""
import sys, hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/azureuser/workspace/AutoSpatial")
from harness_map.qd import descriptors as D


def show(cd):
    cd = Path(cd)
    rl = D.roles(cd)
    N, E = D.component_graph(cd)
    adj = {n: set() for n in N}
    for a, b in E:
        if a in adj and b in adj:
            adj[a].add(b); adj[b].add(a)
    lab = {n: rl.get(n, "logic") for n in N}
    print(f"\n### {cd.name}  nodes={len(N)} edges={len(E)}")
    from collections import Counter
    print("  role counts:", dict(sorted(Counter(lab.values()).items())))
    sigs = Counter()
    for n in sorted(N):
        s = lab[n] + "|" + ",".join(sorted(lab[m] for m in adj[n]))
        sigs[s] += 1
        print(f"   {n:<34} {lab[n]:<11} -> {sorted(adj[n])}")
    print("  h1 signature multiset:")
    for s, c in sorted(sigs.items()):
        print(f"    x{c}  {s}")
    return D.wl_features(cd), sigs


if __name__ == "__main__":
    outs = [show(a) for a in sys.argv[1:]]
    if len(outs) == 2:
        (w1, s1), (w2, s2) = outs
        keys = sorted(set(w1) | set(w2))
        v1 = np.array([w1.get(k, 0.0) for k in keys]); v2 = np.array([w2.get(k, 0.0) for k in keys])
        print(f"\nRAW wl L2 = {np.linalg.norm(v1-v2):.6f}")
        print("sig multiset diff (a-only):", sorted(set(s1) - set(s2)))
        print("sig multiset diff (b-only):", sorted(set(s2) - set(s1)))
