"""CVT partition of φ space: k-means, the fitted and frozen φ, and the centroid ladder."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from harness_map.qd import descriptors as D


def kmeans(X, k, seed=0, iters=100):
    """k-means (kmeans++ seeding); (centroids, labels), fewer than `k` on few distinct points."""
    rng = np.random.default_rng(seed)
    n = len(X)
    cent = [X[rng.integers(n)]]
    for _ in range(k - 1):
        d2 = np.min([np.sum((X - c) ** 2, axis=1) for c in cent], axis=0)
        tot = float(d2.sum())
        if not np.isfinite(tot) or tot <= 0.0:          # points coincide: fewer than k centroids
            break
        p = d2 / tot
        cent.append(X[rng.choice(n, p=p)])
    k = len(cent)
    C = np.stack(cent)
    for _ in range(iters):
        assign = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
        newC = np.stack([X[assign == j].mean(0) if (assign == j).any() else C[j] for j in range(k)])
        if np.allclose(newC, C):
            break
        C = newC
    return C, np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), axis=1)


def fit_cvt_blocks(harness_dirs, cvt_dir, rules, k0=2):
    """Fit the CVT on a corpus and save it to `cvt_dir`; returns (n, joint_dim, k, block_split)."""
    cvt_dir = Path(cvt_dir); cvt_dir.mkdir(parents=True, exist_ok=True)
    bpd = [D.desc_blocks(cd, rules, cvt_dir) for cd in harness_dirs]
    names = [nm for nm, _ in bpd[0]]
    meta, arrays, cols = {"names": names, "blocks": []}, {}, []
    for i, nm in enumerate(names):
        M, keys = D._block_matrix(bpd, i)
        mu, sd = M.mean(0), M.std(0); sd = np.where(sd < 1e-9, 1.0, sd)
        Z = (M - mu) / sd / math.sqrt(max(M.shape[1], 1))
        arrays[f"mu_{i}"] = mu; arrays[f"sd_{i}"] = sd
        meta["blocks"].append({"name": nm, "kind": "dict" if keys else "vec",
                               "keys": keys, "dim": int(M.shape[1])})
        cols.append(Z)
    J = np.hstack(cols)
    block_split = int(cols[0].shape[1])
    Zc, _lab = kmeans(J, min(k0, len(J)), seed=0)
    arrays["Z"] = Zc; arrays["block_split"] = np.array([block_split])
    np.savez(cvt_dir / "cvt_blocks.npz", **arrays)
    meta["block_split"] = block_split
    (cvt_dir / "blocks_meta.json").write_text(json.dumps(meta))
    return len(J), J.shape[1], len(Zc), block_split


def make_phi_blocks(cvt_dir, rules):
    """Load a fitted CVT from `cvt_dir`; returns (phi_fn, centroids, block_split)."""
    cvt_dir = Path(cvt_dir)
    z = np.load(cvt_dir / "cvt_blocks.npz")
    meta = json.loads((cvt_dir / "blocks_meta.json").read_text())
    block_split = int(meta["block_split"])

    def phi(node):
        bpd = D.desc_blocks(node.genome.source_dir, rules, cvt_dir)
        cols = []
        for i, spec in enumerate(meta["blocks"]):
            b = bpd[i][1]
            if spec["kind"] == "dict":
                x = np.array([b.get(k, 0.0) for k in spec["keys"]], float)
            else:
                x = np.asarray(b, float)
            mu = z[f"mu_{i}"]; sd = np.where(z[f"sd_{i}"] < 1e-9, 1.0, z[f"sd_{i}"])
            cols.append((x - mu) / sd / math.sqrt(max(spec["dim"], 1)))
        return np.hstack(cols)
    return phi, z["Z"], block_split


class Ladder(list):
    """Centroid rungs, coarse to fine, with an optional parent map `parents[l][j]` (None: flat)."""

    parents = None


def ladder_cell_index(ladder, p, level):
    """Rung-`level` cell of φ point `p`, descending the ladder through nearest children."""
    p = np.asarray(p, float)
    parents = getattr(ladder, "parents", None)
    if parents is None:
        raise ValueError("ladder_cell_index needs a ladder carrying a parent->children map "
                         "(build_centroid_ladder's nested default); a flat ladder is placed by "
                         "global nearest centroid instead")
    Z0 = np.asarray(ladder[0], float)
    j = int(np.argmin(((p[None, :] - Z0) ** 2).sum(-1)))
    for lvl in range(1, int(level) + 1):
        Z = np.asarray(ladder[lvl], float)
        kids = [i for i, par in enumerate(parents[lvl]) if par == j]
        if not kids:                                   # coarse cell was empty when built
            j = int(np.argmin(((p[None, :] - Z) ** 2).sum(-1)))
            continue
        j = min(kids, key=lambda i: float(((p - Z[i]) ** 2).sum()))
    return j


def ladder_nesting(ladder, X):
    """Per transition, whether flat placement of `X` nests each fine cell in one coarse cell."""
    X = np.asarray(X, float)
    reports = []
    if len(X) == 0:
        return reports
    assign = [np.argmin(((X[:, None, :] - Z[None, :, :]) ** 2).sum(-1), axis=1) for Z in ladder]
    for l in range(len(ladder) - 1):
        parents_of = {}
        for i in range(len(X)):
            parents_of.setdefault(int(assign[l + 1][i]), set()).add(int(assign[l][i]))
        viol = sorted((fine, sorted(ps)) for fine, ps in parents_of.items() if len(ps) > 1)
        reports.append({"level": l + 1, "k_from": len(ladder[l]), "k_to": len(ladder[l + 1]),
                        "n_points": int(len(X)), "n_fine_occupied": len(parents_of),
                        "violations": viol, "nested": not viol})
    return reports


def build_centroid_ladder(phi_fn, corpus_nodes, ks=None, *, k0=2, levels=3, branching=2,
                          nested=True, seed=0, log=None):
    """Coarse-to-fine centroid ladder of the hier mode: k-means at k0, then nested splits (§9.2)."""
    _log = log or (lambda *a, **k: None)
    J = np.stack([np.asarray(phi_fn(n), float) for n in corpus_nodes])
    if ks is not None:
        ks = [int(k) for k in ks]
        if not nested:
            return Ladder(kmeans(J, min(k, len(J)), seed=seed)[0] for k in ks)   # no parent map
        k0, levels = ks[0], len(ks)
    if not nested:
        raise ValueError("build_centroid_ladder(nested=False) needs an explicit ks=[k0, k1, …] "
                         "(the independent-CVT ladder has no branching rule to derive them from)")
    Z0 = kmeans(J, min(int(k0), len(J)), seed=seed)[0]
    ladder, parts = [Z0], [[np.arange(len(J))[np.argmin(((J[:, None, :] - Z0[None, :, :]) ** 2)
                                                        .sum(-1), axis=1) == j]
                            for j in range(len(Z0))]]
    par = [[None] * len(Z0)]
    for _lvl in range(1, int(levels)):
        Zs, groups, pof = [], [], []
        for ci, idx in enumerate(parts[-1]):
            if len(idx) == 0:
                continue
            sub = J[idx]
            Zsub, lab = kmeans(sub, min(int(branching), len(sub)), seed=seed)
            for j in range(len(Zsub)):
                sel = idx[np.asarray(lab) == j]
                if len(sel) == 0:
                    continue
                Zs.append(Zsub[j]); groups.append(sel); pof.append(ci)
        ladder.append(np.stack(Zs)); parts.append(groups); par.append(pof)
    out = Ladder(ladder)
    out.parents = par
    if ks is not None and [len(z) for z in out] != ks:
        _log(f"[hier] WARNING: nested ladder realized k={[len(z) for z in out]}, not the "
             f"requested ks={ks} — some cell held fewer than branching={branching} corpus points")
    rep = ladder_nesting(out, J)
    _log(f"[hier] ladder: k={[len(z) for z in out]} over {len(J)} corpus points "
         f"(nested, branching={branching}); parent map carried, so `place` assigns HIERARCHICALLY "
         f"and nesting holds for every point by construction. For reference, flat "
         f"global-nearest placement would agree with the hierarchy at: "
         f"{[(r['level'], r['nested']) for r in rep]}")
    for r in rep:
        if not r["nested"]:
            _log(f"[hier] NOTE: rung {r['level']} would NOT be nested under FLAT placement — "
                 f"{len(r['violations'])} fine cell(s) would draw corpus points from more than one "
                 f"coarse cell {r['violations'][:3]}. This ladder is not placed flatly (see "
                 f"`ladder_cell_index`), so it is a measure of how much the two rules diverge on "
                 f"this corpus, not a defect: well-separated clusters make them agree, overlapping "
                 f"ones do not.")
    return out
