"""The CVT-MAP-Elites archive (`Frontier`): cells, placement, elites, resolution and breeding."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from harness_map import main as MH
from harness_map.main import BatchContractError, Node, _anchor_nodes, _evaluate_all, _propose_ex, _rank
# Imported as a module: coverage imports this module too, so its names must resolve at call time.
from harness_map.qd import coverage as _cov
from harness_map.qd.partition import Ladder, kmeans


@dataclass
class Frontier:
    """Archive state: cells {cid, z, mem, depth, parent, children, pre_elite, born, split_failed}."""
    ctx: object
    phi_fn: object = None
    block_split: Optional[int] = None                 # width of φ's structure block (steering)
    # params
    sigma: float = 0.0
    k_split: int = 2
    depth_cap: int = 2
    # >= 1: right after a split the best descendant is the pre-split elite, so a split is judged
    # only after at least one breeding turn in its subtree.
    patience: int = 1
    steer: bool = True
    split_target: str = "variance"
    # archive state
    cells: list = field(default_factory=list)         # active leaves of the tree
    open_splits: list = field(default_factory=list)
    splits: list = field(default_factory=list)
    retreats: list = field(default_factory=list)
    trace: list = field(default_factory=list)
    steer_log: list = field(default_factory=list)
    disagree_log: list = field(default_factory=list)
    _cid: int = 0                                      # monotone cell-id counter
    allnodes: list = field(default_factory=list)       # every evaluated Node, in creation order
    _render: object = None                             # transient per-round context for admit's log
    tag: str = "frontier"                              # log prefix ("frontier" | "hier" | "fixed")
    ladder: object = None                              # hier only: the Ladder `place` resolves in
    level: int = 0                                     # hier only: the live rung index
    _phi_cache: dict = field(default_factory=dict)     # node.idx -> φ (see phi_of)
    anchors: list = field(default_factory=list)        # anchor nodes; [0] is the root (node 0)
    pre_seed: list = field(default_factory=list)       # caller-supplied seed_genomes, materialized
    n_pre_seeded: int = 0                              # ...of which this many evaluated valid
    steer_ok: bool = True                              # adapter can receive a directive (see seed)


def mk_cell(fr, z, depth=0, parent=None, pre_elite=None, born=0):
    """Create a cell (a tree node) with centroid `z`."""
    fr._cid += 1
    return {"cid": fr._cid, "z": np.asarray(z, float), "mem": [], "depth": depth,
            "parent": parent, "children": [], "pre_elite": pre_elite, "born": born,
            # score shape recorded when a split of this cell is retreated (see `score_shape`)
            "split_failed": None}


def phi_of(fr, node):
    """φ of `node`, memoized on `node.idx` (re-placement reuses the φ computed at admission)."""
    key = getattr(node, "idx", None)
    if key is None:
        return np.asarray(fr.phi_fn(node), float)
    p = fr._phi_cache.get(key)
    if p is None:
        p = np.asarray(fr.phi_fn(node), float)
        fr._phi_cache[key] = p
    return p


def elite(fr, c):
    """The best member of cell `c` under the run's ranking, or None if the cell is empty."""
    return _rank(c["mem"], fr.ctx.cfg)[0] if c["mem"] else None


def score_shape(fr, c):
    """The set of 2σ-wide fitness buckets occupied by the members of `c` (see `splittable`)."""
    b = 2.0 * fr.sigma
    if b <= 0.0:                    # sigma=0: every distinct fitness is its own bucket
        return frozenset(n.fit.primary for n in c["mem"])
    return frozenset(math.floor(n.fit.primary / b) for n in c["mem"])


def cell_range(c):                                 # max - min member fitness (the 2σ trigger)
    vs = [n.fit.primary for n in c["mem"]]
    return (max(vs) - min(vs)) if len(vs) > 1 else 0.0


def place(fr, node):
    """Add `node` to its cell (`coverage.locate_cell`); returns (cell, became_elite)."""
    p = phi_of(fr, node)
    c = _cov.locate_cell(fr, p)
    before = elite(fr, c)
    c["mem"].append(node)
    return c, (before is None or elite(fr, c) is node)


def split_cell(fr, c, it):
    """Split `c` by k-means over its members' φ; returns the children, or None if not separable."""
    X = np.array([phi_of(fr, n) for n in c["mem"]])
    Z, lab = kmeans(X, min(fr.k_split, len(X)), seed=0)
    pre = elite(fr, c).fit.primary
    kids = []
    for j in range(len(Z)):
        idx = [i for i, l in enumerate(lab) if int(l) == j]
        if not idx:
            continue
        k = mk_cell(fr, Z[j], depth=c["depth"] + 1, parent=c, pre_elite=pre, born=it)
        k["mem"] = [c["mem"][i] for i in idx]
        kids.append(k)
    if len(kids) < 2:
        return None
    c["children"] = kids
    fr.cells.remove(c); fr.cells.extend(kids)           # parent leaves the active set
    fr.open_splits.append({"parent": c, "pre_elite": pre, "born": it, "breeds": 0})
    return kids


def is_desc(x, p):
    """True if cell `x` is `p` or one of its descendants."""
    cur = x
    while cur is not None:
        if cur is p:
            return True
        cur = cur["parent"]
    return False


def subtree(fr, p):
    """Every descendant of `p`, intermediate and leaf cells alike."""
    out = []
    for k in p["children"]:
        out.append(k); out.extend(subtree(fr, k))
    return out


def splittable(fr, c):
    """Whether `c` is active, under `depth_cap` and changed in score shape since its last retreat."""
    if c is None or c not in fr.cells or c["depth"] >= fr.depth_cap:
        return False
    failed = c.get("split_failed")
    if failed is not None and score_shape(fr, c) == failed:
        return False
    return True


def refine(fr, it):
    """Zoom in: split at most one cell per generation, if its fitness range exceeds 2σ."""
    occ = [c for c in fr.cells if c["mem"]]
    if not occ:
        return
    cellA = max(occ, key=lambda c: elite(fr, c).fit.primary)     # best-harness cell
    cellB = max(occ, key=lambda c: cell_range(c))            # max-variance cell
    agree = cellA is cellB
    fr.disagree_log.append({"round": it, "best_cell": cellA["cid"], "var_cell": cellB["cid"],
                            "agree": agree, "best_range": round(cell_range(cellA), 3),
                            "var_range": round(cell_range(cellB), 3)})
    target = {"best": cellA, "variance": cellB, "agree": (cellA if agree else None)}.get(fr.split_target)
    thr = 2.0 * fr.sigma
    if splittable(fr, target) and cell_range(target) > thr:
        kids = split_cell(fr, target, it)
        if kids:
            fr.splits.append({"round": it, "cell": target["cid"], "depth": target["depth"],
                              "range": round(cell_range(target), 3), "thr_2sigma": round(thr, 3),
                              "strategy": fr.split_target, "children": [k["cid"] for k in kids]})
            fr.ctx.log(f"[frontier] round {it}: ZOOM-IN [{fr.split_target}] cell {target['cid']} "
                       f"(d{target['depth']}, range={cell_range(target):.3f}>2σ={thr:.3f}, "
                       f"{len(target['mem'])} mem) -> {[k['cid'] for k in kids]} "
                       f"| A=c{cellA['cid']} B=c{cellB['cid']} agree={agree}")


def retreat(fr, it):
    """Zoom out: merge back a split that, after `patience` turns, did not beat its parent by 2σ."""
    thr = 2.0 * fr.sigma
    for sp in sorted(fr.open_splits, key=lambda s: -s["born"]):
        if sp["breeds"] < fr.patience:
            continue
        par, pre = sp["parent"], sp["pre_elite"]
        leaves = [x for x in fr.cells if x is not par and is_desc(x, par)]
        if not leaves:
            fr.open_splits.remove(sp); continue
        best_desc = max((elite(fr, d).fit.primary for d in leaves if elite(fr, d)), default=None)
        if best_desc is not None and pre is not None and best_desc > pre + thr:
            fr.open_splits.remove(sp); continue
        full = subtree(fr, par)
        n_before = len(par["mem"])
        par["mem"] = [n for d in leaves for n in d["mem"]]
        assert len(par["mem"]) >= n_before, "retreat must not lose harnesses"
        par["children"] = []
        par["split_failed"] = score_shape(fr, par)
        for d in full:
            if d in fr.cells:
                fr.cells.remove(d)
        if par not in fr.cells:
            fr.cells.append(par)                         # parent becomes an active leaf again
        for s2 in [s for s in fr.open_splits if s is sp or is_desc(s["parent"], par)]:
            fr.open_splits.remove(s2)
        fr.retreats.append({"round": it, "parent": par["cid"], "merged_subtree": [d["cid"] for d in full],
                            "merged_members": len(par["mem"]), "pre_elite": pre, "best_desc": best_desc,
                            "split_failed_shape": sorted(par["split_failed"])})
        fr.ctx.log(f"[frontier] round {it}: ZOOM-OUT merge subtree {[d['cid'] for d in full]} "
                   f"({len(par['mem'])} harnesses) back into cell {par['cid']} "
                   f"(best {best_desc} <= {pre}+2σ) — active level up; will not re-split "
                   f"until the score distribution changes "
                   f"({len(par['split_failed'])} distinct 2σ levels recorded)")


def hier_ladder(centroid_levels):
    """Validate a coarse-to-fine list of centroid sets; returns a `Ladder` (parent map kept)."""
    if centroid_levels is None or len(centroid_levels) == 0:
        raise ValueError("mapelites_hier requires centroid_levels — a coarse->fine LIST of "
                         "centroid sets (see main.build_centroid_ladder)")
    out = []
    for lvl, Z in enumerate(centroid_levels):
        A = np.asarray(Z, float)
        if A.ndim != 2 or A.shape[0] < 1 or A.shape[1] < 1:
            raise ValueError(f"centroid_levels[{lvl}] must be a 2-D (k, d) centroid array, got "
                             f"shape {A.shape} — a ladder is a LIST of centroid sets, not one "
                             f"centroid set (did you pass make_phi_blocks' Z directly?)")
        out.append(A)
    d = out[0].shape[1]
    bad = [l for l, A in enumerate(out) if A.shape[1] != d]
    if bad:
        raise ValueError(f"every rung must share phi's dimension {d}; rungs {bad} differ — all "
                         f"rungs must come from ONE fit of ONE corpus + descriptor")
    # Rung 0 is the coverage denominator k0 and the rung live during initialization, so warn (not
    # raise) when rung sizes decrease: coverage would be measured against a finer partition.
    ks = [A.shape[0] for A in out]
    if any(b < a for a, b in zip(ks, ks[1:])):
        import warnings as _warnings
        _warnings.warn(
            f"centroid_levels is not coarse->fine: rung sizes {ks} DECREASE somewhere. A ladder is "
            f"validated and consumed coarse-first — rung 0 is the coverage denominator K0 and the "
            f"rung the archive is live at during initialization — so a decreasing ladder measures "
            f"coverage against the FINEST partition while claiming the coarsest. "
            f"build_centroid_ladder always emits coarse->fine; a hand-built ladder must too.",
            RuntimeWarning, stacklevel=2)
    out = Ladder(out)
    par = getattr(centroid_levels, "parents", None)
    if par is not None:
        if len(par) != len(out) or any(len(par[l]) != len(out[l]) for l in range(len(out))):
            raise ValueError(f"ladder parent map has shape {[len(p) for p in par]} for rungs "
                             f"{[len(z) for z in out]} — one parent entry per centroid per rung")
        wrong = [(l, j) for l in range(1, len(out)) for j, q in enumerate(par[l])
                 if q is None or not (0 <= int(q) < len(out[l - 1]))]
        if wrong:
            raise ValueError(f"ladder parent map points outside the coarser rung at {wrong[:3]} — "
                             f"parents[l][j] must index rung l-1")
        out.parents = [list(p) for p in par]
    return out


def hier_program(level_start, rounds, L):
    """Normalize the resolution-change program; returns (program, caller_supplied)."""
    if level_start is None:
        return [1 + (lvl * rounds) // L for lvl in range(L)], False
    p = [int(x) for x in level_start]
    if len(p) != L:
        raise ValueError(f"level_start has {len(p)} entries for {L} centroid levels — the "
                         f"resolution-change program has exactly one start generation per rung")
    bad = [lvl for lvl in range(1, L) if p[lvl] < 1]
    if bad:
        raise ValueError(f"level_start{bad} < 1 — generation indices are 1-based")
    return p, True


def hier_target_level(program, it, L):
    """The finest rung whose programmed start generation has arrived (skipping rungs if needed)."""
    return max([0] + [lvl for lvl in range(1, L) if it >= program[lvl]])


def advance(fr, it, ladder, program, active):
    """Scheduled resolution change (§9.2): switch to the finest due rung, re-place every member."""
    L = len(ladder)
    tgt = hier_target_level(program, it, L)
    if tgt <= active:
        return active, None
    old = fr.cells
    members = sorted([n for c in old for n in c["mem"]], key=lambda n: n.idx)
    occ_before = sum(1 for c in old if c["mem"])
    fr.cells = [mk_cell(fr, z, depth=tgt) for z in ladder[tgt]]   # fresh cids, depth = rung
    fr.level = tgt                     # read by `place`: fr.cells[i] is rung tgt's centroid i
    for n in members:
        place(fr, n)
    assert sum(len(c["mem"]) for c in fr.cells) == len(members), "advance must not lose harnesses"
    occ_after = sum(1 for c in fr.cells if c["mem"])
    row = {"round": it, "from": active, "to": tgt, "skipped": list(range(active + 1, tgt)),
           "k_from": len(old), "k_to": len(fr.cells),
           "retired_cells": [c["cid"] for c in old], "new_cells": [c["cid"] for c in fr.cells],
           "n_members": len(members), "occupied_before": occ_before, "occupied_after": occ_after}
    fr.ctx.log(f"[hier] round {it}: ADVANCE L{active} -> L{tgt} (k {len(old)} -> {len(fr.cells)}, "
               f"program level_start[{tgt}]={program[tgt]}); re-placed {len(members)} harnesses, "
               f"occupied {occ_before} -> {occ_after}")
    if row["skipped"]:
        fr.ctx.log(f"[hier] round {it}: SKIP levels {row['skipped']} (programmed starts "
                   f"{[program[s] for s in row['skipped']]} all fell at or before this generation)"
                   f" — re-placed L{active} -> L{tgt} directly, intermediate rungs never built")
    if occ_after < occ_before:
        if getattr(fr.ladder, "parents", None) is None:
            fr.ctx.log(f"[hier] round {it}: NOTE occupancy fell {occ_before} -> {occ_after} across "
                       f"the advance — members that sat in different coarse cells now share one "
                       f"fine cell. This ladder carries NO parent->children map (an "
                       f"independent-CVT ladder from build_centroid_ladder(nested=False), or a "
                       f"hand-built list of centroid sets), so `place` assigns by GLOBAL nearest "
                       f"centroid and the rungs are independent partitions rather than a "
                       f"refinement hierarchy. Under a ladder from build_centroid_ladder's NESTED "
                       f"default, placement is HIERARCHICAL (coarse cell first, then the nearest "
                       f"child within it) and occupancy cannot fall THIS way: distinct coarse "
                       f"cells have disjoint descendant sets, so each occupied coarse cell "
                       f"contributes at least one occupied fine cell.")
        else:
            fr.ctx.log(f"[hier] round {it}: NOTE occupancy fell {occ_before} -> {occ_after} across "
                       f"the advance even though this ladder DOES carry a parent->children map. "
                       f"The only way that happens is `ladder_cell_index`'s documented degenerate "
                       f"fallback: a coarse cell with NO CHILDREN at rung {tgt}, for which the "
                       f"hierarchical rule cannot be applied and placement falls back to global "
                       f"nearest over the whole rung — so a member can leave its coarse ancestor's "
                       f"descendant set and the subset property is not stated for it. "
                       f"build_centroid_ladder's nested build does not produce a childless coarse "
                       f"cell; a HAND-BUILT parent map can. Check the map you supplied.")
    return tgt, row


def steer(fr, cell):
    """Steer the proposer toward a better or wider cell; (directive, target_cid) or (None, None)."""
    if not fr.steer:
        return None, None
    here = elite(fr, cell)
    others = [c for c in fr.cells if c["mem"] and c is not cell]
    if here is None or not others:
        return None, None
    better = [c for c in others if elite(fr, c).fit.primary > here.fit.primary]
    if better:                                           # exploit: nearest better cell
        tgt = min(better, key=lambda c: float(((c["z"] - cell["z"]) ** 2).sum()))
        sense = "toward"
    else:                                                # explore: widest-range cell
        tgt = max(others, key=lambda c: (cell_range(c), len(c["mem"])))
        sense = "toward the diverse"
    axis = "structure/wiring"
    B = fr.block_split
    if B is not None and 0 < B < len(cell["z"]):
        ds = float(((tgt["z"][:B] - cell["z"][:B]) ** 2).sum()) ** 0.5
        dt = float(((tgt["z"][B:] - cell["z"][B:]) ** 2).sum()) ** 0.5
        axis = "structure/wiring (component graph)" if ds >= dt else "token-level implementation"
    ex = [str(n.genome.source_dir) for n in _rank(tgt["mem"], fr.ctx.cfg)[:2]]
    directive = (f"SEARCH-SPACE STEER — move this harness {sense} region exemplified by "
                 f"{ex}; the productive difference there is in its {axis}. Vary that aspect "
                 f"(a qualitative direction, not an exact target); keep the public entry intact.")
    return directive, tgt["cid"]


# Alias: the drivers have a parameter named `steer`.
_steer = steer


def frontier_cover_view(fr, *, k0, random_init, parent_select, seed, land_fn):
    """The `coverage._CoverView` through which coverage-gated initialization sees `fr`."""
    def _target():
        # ceil(m*k0) clamped to [0, k0]: the same target `coverage_init` gates on
        return min(k0, max(0, math.ceil(random_init * k0)))

    def _exemplars():
        return [str(elite(fr, c).genome.source_dir)
                for c in sorted((c for c in fr.cells if c["mem"]), key=lambda c: c["cid"])[:3]]

    def _select(p, b, claimed):
        # No split is open during the coverage phase, so `select_cells`' breed credit is a no-op.
        cells = select_cells(fr, p, b, rule=parent_select, rng=gen_rng(seed, -p))
        if not cells:
            return None, None
        occ_n = sum(1 for c in fr.cells if c["mem"])
        ex = _exemplars()
        # One aim per slot, each at a different unclaimed empty cell (`cover_aim`); the move is
        # measured from the slot's own parent cell, as in `steer`.
        aims = _cov.cover_aim(fr, set(claimed or ()), len(cells), srcs=cells)
        dirs = [_cov.coverage_directive(occupied=occ_n, k0=k0, target=_target(), exemplars=ex, aim=a)
                for a in aims]
        return [elite(fr, c) for c in cells], {
            "cells": cells, "directives": dirs,
            # (directive, target) pairs for `_land`; the target is None on the novelty fallback.
            "steers": [(d, (a["target"] if a else None)) for d, a in zip(dirs, aims)],
            "prev": {id(c): bool(c["mem"]) for c in fr.cells}}

    def _land(meta, units, p):
        import time as _time
        t0 = _time.time()
        _t, imp, dis, rej = land_fn(meta["cells"], meta["steers"], units, -p, 1, meta["prev"])
        return batch_gen_row(-p, [c["cid"] for c in meta["cells"]], units, improvements=imp,
                             discoveries=dis, rejected=rej, archive=len(fr.cells),
                             occupied=sum(1 for c in fr.cells if c["mem"]), total=len(fr.cells),
                             elites=[elite(fr, c) for c in fr.cells if c["mem"]],
                             wall_s=_time.time() - t0)

    return _cov._CoverView(
        k0=k0, tag=fr.tag,
        occupied=lambda: {c["cid"] for c in fr.cells if c["mem"]},
        key_of_genome=lambda g: _cov.locate_cell(fr, _cov.phi_of_genome(fr, g))["cid"],
        select=_select, land=_land,
        empty_cells=lambda: [c["cid"] for c in fr.cells if not c["mem"]])


def snap(fr):
    """JSON-serializable view of the active cells (stored in trace rows)."""
    return [{"cid": c["cid"], "depth": c["depth"],
             "parent": (c["parent"]["cid"] if c["parent"] else None),
             "z": [float(x) for x in c["z"]],
             "elite": (round(elite(fr, c).fit.primary, 3) if c["mem"] else None),
             "n": len(c["mem"]),
             "members": [str(n.genome.source_dir) for n in c["mem"]]} for c in fr.cells]


def _frontier_occ(fr):
    occ = [c for c in fr.cells if c["mem"]]
    occ.sort(key=lambda c: c["cid"])
    return occ


def select_cells(fr, it, b, *, rule="roundrobin", rng=None, log=None, tally=None,
                 tag="batch"):
    """Select `b` cells from a cid-ordered snapshot, crediting open splits with breeding turns."""
    occ = _frontier_occ(fr)
    cells = select_batch(occ, it, b, rule=rule, rng=rng, log=log, tally=tally, tag=tag)
    for cell in cells:
        for sp in fr.open_splits:
            if is_desc(cell, sp["parent"]):
                sp["breeds"] += 1
    return cells


def admit(fr, children_nodes):
    """Place and log each evaluated child; returns the touched cells in first-touch order."""
    r = fr._render or {}
    touched = []
    for child in children_nodes:
        cprime, adm = place(fr, child)
        if cprime not in touched:
            touched.append(cprime)
        fr.ctx.log(f"[{fr.tag}] {_cov.phase_label(r.get('round'), child.depth)}: "
                   f"cell {cprime['cid']} (d{cprime['depth']}) <- "
                   f"{child.label} primary={child.fit.primary:.3f} {'[new elite]' if adm else ''} "
                   f"(parent={r['parent'].label if r.get('parent') else '?'}@cell "
                   f"{r['cell']['cid'] if r.get('cell') else '?'}, steer->{r.get('tgt')})")
    return touched


def best(fr):
    """The archive's best node: max primary fitness over every evaluated harness."""
    return max(fr.allnodes, key=lambda n: n.fit.primary, default=None)


def seed(ctx, *, phi_fn, init_centroids, block_split, seed_genomes, sigma, k_split,
         depth_cap, patience, steer, split_target, tag="frontier"):
    """Build the Frontier: one cell per centroid; evaluate and place anchors and `seed_genomes`."""
    B = int(block_split) if block_split else None
    fr = Frontier(ctx=ctx, phi_fn=phi_fn, block_split=B, sigma=sigma, k_split=k_split,
                  depth_cap=depth_cap, patience=patience, steer=steer,
                  split_target=split_target, tag=tag)
    fr.cells = [mk_cell(fr, z) for z in np.asarray(init_centroids, float)]
    anchors = _anchor_nodes(ctx)
    root = anchors[0]
    fr.anchors = list(anchors)
    fr.allnodes = list(anchors)
    for a in anchors:
        place(fr, a)
    # Materialize once: seed_genomes may be a generator, and its length is reported later.
    fr.pre_seed = list(seed_genomes or [])
    fr.n_pre_seeded = 0
    for g, f in zip(fr.pre_seed, _evaluate_all(ctx.adapter, fr.pre_seed, 0) if fr.pre_seed else []):
        ctx.spend(1)
        if f is not None and getattr(f, "valid", False):
            n = Node(ctx.next_idx(), g, f, depth=0, parents=(root.idx,))
            fr.allnodes.append(n); place(fr, n)
            fr.n_pre_seeded += 1
    if len(anchors) > 1:
        ctx.log(f"[{tag}] {len(anchors)} ANCHORS from adapter.anchor(): "
                f"{[a.label for a in anchors]} — each evaluated as itself at adapter scalar 0 and "
                f"placed as an initial individual; {root.label} is THE root (node id 0)")
    # If `propose` accepts none of `directive=`, `feedback=` or `**kwargs`, every directive is
    # dropped, so warn. An adapter without `propose` is assumed to accept directives.
    _proposes = hasattr(ctx.adapter, "propose")
    fr.steer_ok = MH.directive_supported(ctx.adapter) if _proposes else True
    if steer and _proposes and not fr.steer_ok:
        ctx.log(f"[{tag}] WARNING: steer=True but {type(ctx.adapter).__name__}.propose accepts "
                f"neither `directive=` nor `feedback=` nor `**kwargs`, so EVERY directive this run "
                f"computes will be discarded before it reaches the proposer. The run will be "
                f"UNSTEERED. steer_log rows carry delivered=false. Add `directive=None` to the "
                f"adapter's propose signature, or run with steer=False so the ablation is honest.")
    ctx.log(f"[{tag}] seeded: {len(fr.cells)} cells, occupied {sum(1 for c in fr.cells if c['mem'])} | "
            f"k0={len(init_centroids)} k_split={k_split} 2σ={2 * sigma:.3f} "
            f"depth_cap={depth_cap} steer={steer}")
    fr.trace.append({"round": 0, "parent_cell": None, "n_cells": len(fr.cells),
                     "occupied": sum(1 for c in fr.cells if c["mem"]),
                     "best": round(max((n.fit.primary for n in fr.allnodes), default=0.0), 3),
                     "cells": snap(fr)})
    return fr



_seed = seed


# ── breeding one generation ───────────────────────────────────────────────────────────────────
@dataclass
class BredUnit:
    """One parent's slot in a generation: its pre-allocated rid, proposals, and results."""
    parent: object                                    # Node
    slot: int                                         # 0..b-1: submission order (= admission order)
    rid: int                                          # proposal id; the adapter's iter_n
    directive: Optional[str] = None
    cands: list = field(default_factory=list)         # list[Genome], proposer order
    fits: list = field(default_factory=list)          # list[Fitness|None], aligned with cands
    children: list = field(default_factory=list)      # list[Node] for valid pairs, proposer order
    propose_error: Optional[str] = None
    eval_errors: int = 0
    # ── budget and gate accounting ─────────────────────────────────────────────────────────────
    proposed: int = 0                                 # genomes propose() returned, before gating/C
    truncated: int = 0                                # children dropped by children_per_parent (C)
    evaluated: int = 0                                # candidates handed to evaluate/evaluate_batch
    gate_rejects: list = field(default_factory=list)  # pre-eval gate rejections, with the reason
    propose_attempts: int = 1                         # proposals made (>1 = re-proposed)
    screen_error: Optional[str] = None                # the screen raised, so all were evaluated


def gen_rng(seed, gen):
    """RNG for one generation, seeded by (seed, gen) only, so a resumed run reproduces its draws."""
    import random as _random
    return _random.Random(int(seed) * 1_000_003 + int(gen))


def select_batch(snapshot, gen, batch_size, *, rule="roundrobin", rng=None,
                 log=None, tally=None, tag="batch"):
    """Select `batch_size` parents from `snapshot`: 'roundrobin' positions or 'uniform' draws."""
    n = len(snapshot)
    if n == 0:
        return []
    b = max(1, int(batch_size))
    if rule == "roundrobin":
        start = (gen - 1) * b
        picked = [snapshot[(start + j) % n] for j in range(b)]
    elif rule == "uniform":
        if rng is None:
            raise ValueError("select_batch(rule='uniform') requires a seeded rng (see gen_rng)")
        picked = [snapshot[rng.randrange(n)] for _ in range(b)]
    else:
        raise ValueError(f"unknown parent_select {rule!r}; use 'roundrobin' or 'uniform'")
    if n < b:                    # under-full generation
        _note_under_full(picked, n=n, b=b, gen=gen, rule=rule, log=log, tally=tally, tag=tag)
    return picked


def _note_under_full(picked, *, n, b, gen, rule, log=None, tally=None, tag="batch"):
    turns = {}
    for x in picked:
        turns[id(x)] = turns.get(id(x), 0) + 1
    max_dup, distinct = max(turns.values()), len(turns)
    floor = min(turns.values()) if distinct >= n else 0   # undrawn occupied cells take 0 turns
    extra = sum(1 for v in turns.values() if v > floor)   # == b mod n under 'roundrobin'
    if log is not None:
        log(f"[{tag}] generation {gen}: UNDER-FULL SELECTION — the archive has {n} occupied "
            f"cell(s), fewer than requested={b}, so the {rule} rule fills all {b} slots by "
            f"RE-SELECTING: {distinct} distinct cell(s), {extra} of them taking an extra turn "
            f"(>{floor}), {max_dup} breeding events for the most-selected. Each duplicate is an "
            f"INDEPENDENT breeding event with its own rid, artifact dir and propose() call — "
            f"deliberate and reference-faithful (pymap_elites draws parents WITH REPLACEMENT from "
            f"list(archive.keys())): P stays {b}, so the budget "
            f"identity N = M + G*P*C still holds.")
    if tally is not None:
        tally["generations"] = tally.get("generations", 0) + 1
        tally["max_duplication"] = max(tally.get("max_duplication", 1), max_dup)
        tally["min_occupied"] = min(tally.get("min_occupied", n), n)
        tally.setdefault("gens", []).append(
            {"gen": gen, "occupied": n, "requested": b, "distinct": distinct,
             "extra_turns": extra, "duplication": max_dup})


def effective_workers(adapter, n_workers, *, log=print, stage="all"):
    """Worker threads for `stage`: 1 unless the adapter is `PARALLEL_SAFE` (env overrides apply)."""
    w = max(1, int(n_workers or 1))
    if w == 1:
        return 1
    if not getattr(adapter, "PARALLEL_SAFE", False):
        log(f"[batch] {type(adapter).__name__} does not declare PARALLEL_SAFE — forcing n_workers=1")
        return 1
    cap = int(os.environ.get("HARNESS_MAP_MAX_WORKERS", "8"))
    env = os.environ.get(f"HARNESS_MAP_{stage.upper()}_WORKERS")
    if env:
        w = max(1, int(env))
    if w > cap:
        log(f"[batch] clamping n_workers {w} -> {cap} (HARNESS_MAP_MAX_WORKERS)")
        w = cap
    return w


def breed_batch(ctx, parents, *, directives=None, n_workers=1, gen=0,
                tag="batch", n_children=None, screen=None):
    """Propose and evaluate each parent's children; returns one `BredUnit` per slot, in order."""
    import concurrent.futures as _futures
    b = len(parents)
    if b == 0:
        return []
    directives = list(directives) if directives is not None else [None] * b
    if len(directives) != b:
        raise ValueError(f"breed_batch: {len(directives)} directives for {b} parents")
    # (1) Pre-allocate one rid per slot, in slot order, before dispatch (it names the slot's
    # artifact directory); every slot consumes one even if it proposes nothing.
    rid0 = ctx._round
    ctx._round += b
    units = [BredUnit(parent=parents[j], slot=j, rid=rid0 + j, directive=directives[j])
             for j in range(b)]
    w_propose = effective_workers(ctx.adapter, n_workers, log=ctx.log, stage="propose")
    w_eval = effective_workers(ctx.adapter, n_workers, log=ctx.log, stage="eval")
    isolate = (b > 1 or max(w_propose, w_eval) > 1)
    # Opt-in: the barrier breaks adapters that keep per-slot state between propose and evaluate.
    barrier = bool(getattr(ctx.adapter, "BATCH_BARRIER_SAFE",
                           getattr(ctx.adapter, "PARALLEL_SAFE", False)))
    if screen is not None and not barrier:
        # A screen must see every proposal before any evaluation, so it forces the barrier.
        if b > 1:
            ctx.log(f"[{tag}] WARNING: a screen is active at b={b} and "
                    f"{type(ctx.adapter).__name__} declares neither BATCH_BARRIER_SAFE nor "
                    f"PARALLEL_SAFE — forcing the WIDE BARRIER, which corrupts an adapter that "
                    f"stashes per-slot state in propose(). Use random_init_batch=1 (the default), "
                    f"where the two schedules are the same program.")
        barrier = True
    C = None if n_children is None else max(1, int(n_children))

    # (2) Propose
    def _do_propose(u):
        # Prefix the diagnosis (what is wrong) to the steering directive (where to move).
        # introspect=True: a TypeError inside propose propagates instead of an unsteered retry.
        d = MH.diagnose_of(ctx.adapter, u.parent.genome, u.parent.fit)
        directive = (f"{d}\n\n{u.directive}" if (d and u.directive) else (d or u.directive))
        return _propose_ex(ctx.adapter, u.parent.genome, u.rid, directive,
                           n_children=C, introspect=True)

    _MAX_PROPOSE = max(1, int((ctx.cfg.get("propose", {}) or {}).get("max_attempts", 3)))

    def _take(u, getter):
        want = C if C is not None else 1
        for attempt in range(1, _MAX_PROPOSE + 1):
            u.propose_attempts = attempt
            try:
                got = list(getter() or [])
            except Exception as e:                      # noqa: BLE001 — degrade this slot only
                # Environment errors (missing binary, permissions) would recur on every slot, so
                # they propagate even when slots are isolated; other errors degrade this slot only.
                if not isolate or isinstance(e, (FileNotFoundError, PermissionError, NotADirectoryError)):
                    raise
                u.propose_error = repr(e)
                ctx.log(f"[{tag}] gen {gen} slot {u.slot}: propose failed: {e}")
                break
            u.proposed += len(got)
            for g in got:
                reason = MH._pre_eval_gate(ctx.adapter, g)
                if reason is None:
                    u.cands.append(g)
                else:
                    u.gate_rejects.append({"attempt": attempt, "reason": reason,
                                           "label": getattr(g, "label", "?")})
                    ctx.log(f"[{tag}] gen {gen} slot {u.slot}: GATE REJECTED "
                            f"{getattr(g, 'label', '?')} — {reason}")
            if len(u.cands) >= want or not got or attempt == _MAX_PROPOSE:
                break
            ctx.log(f"[{tag}] gen {gen} slot {u.slot}: {len(u.cands)}/{want} candidates cleared the "
                    f"pre-eval gate — RE-PROPOSING (attempt {attempt + 1}/{_MAX_PROPOSE})")
        if u.gate_rejects and len(u.cands) < want:
            ctx.log(f"[{tag}] gen {gen} slot {u.slot}: still {len(u.cands)}/{want} after "
                    f"{u.propose_attempts} attempts and {len(u.gate_rejects)} gate rejection(s); "
                    f"breeding what cleared")
        if C is not None and len(u.cands) > C:          # enforce C: keep the first C candidates
            u.truncated = len(u.cands) - C
            ctx.log(f"[{tag}] gen {gen} slot {u.slot}: children_per_parent={C} — keeping the first "
                    f"{C} of {len(u.cands)} proposed, dropping {u.truncated} UNEVALUATED "
                    f"({[getattr(g, 'label', '?') for g in u.cands[C:]]})")
            u.cands = u.cands[:C]

    def _propose_units(us):
        if w_propose == 1:
            for u in us:
                _take(u, lambda u=u: _do_propose(u))
        else:
            with _futures.ThreadPoolExecutor(max_workers=w_propose) as ex:
                futs = [ex.submit(_do_propose, u) for u in us]            # submitted in slot order
            for u, fut in zip(us, futs):
                _take(u, fut.result)                                      # collected in slot order

    # (3) Evaluate, in slot order: `evaluate_batch` on the main thread one slot at a time (the
    # adapter owns its concurrency), else per-genome `evaluate`, threaded when w_eval > 1.
    batch_hook = getattr(ctx.adapter, "evaluate_batch", None)

    def _eval_one(u, g):
        f = ctx.adapter.evaluate(g, u.rid)
        # Per-genome evaluations are checkpointed here (`_evaluate_all` covers the batch path).
        from harness_map.qd import modes as _modes
        _modes.record([g], [f])
        return f

    _logged_hook_note = []

    def _evaluate_units(us):
        for u in us:                                    # budget meter: handed to evaluation
            u.evaluated += len(u.cands)
        if batch_hook is not None:
            if w_eval > 1 and not _logged_hook_note:
                _logged_hook_note.append(True)          # once per generation, not once per slot
                ctx.log(f"[batch] {type(ctx.adapter).__name__} owns evaluate_batch — n_workers is "
                        f"ignored for EVALUATION (the adapter's own fan-out drives concurrency)")
            for u in us:
                if not u.cands:
                    continue
                try:
                    # check_contract=True: a mis-paired fitness would become a persistent elite
                    fits = _evaluate_all(ctx.adapter, u.cands, u.rid, check_contract=True)
                except BatchContractError:
                    # An adapter bug, not a transient failure: swallowing it would drop or mis-pair
                    # candidates.
                    raise
                except Exception as e:                  # noqa: BLE001
                    if not isolate:
                        raise
                    u.fits = [None] * len(u.cands)
                    u.eval_errors += len(u.cands)       # per candidate, as in the other branches
                    ctx.log(f"[{tag}] gen {gen} slot {u.slot}: evaluate_batch failed for all "
                            f"{len(u.cands)} candidate(s): {e}")
                else:
                    u.fits = fits
        elif w_eval > 1:
            flat = [(u, i, g) for u in us for i, g in enumerate(u.cands)]   # slot-then-candidate
            for u in us:
                u.fits = [None] * len(u.cands)
            with _futures.ThreadPoolExecutor(max_workers=w_eval) as ex:
                futs = [ex.submit(_eval_one, u, g) for u, _, g in flat]
            for (u, i, g), fut in zip(flat, futs):                        # scatter back by position
                try:
                    u.fits[i] = fut.result()
                except Exception as e:                  # noqa: BLE001 — drop this candidate only
                    u.fits[i] = None
                    u.eval_errors += 1
                    ctx.log(f"[{tag}] gen {gen} slot {u.slot} cand {i}: evaluate failed: {e}")
        else:
            for u in us:
                fits = []
                for i, g in enumerate(u.cands):
                    try:
                        fits.append(_eval_one(u, g))
                    except Exception as e:              # noqa: BLE001
                        if not isolate:
                            raise
                        fits.append(None)
                        u.eval_errors += 1
                        ctx.log(f"[{tag}] gen {gen} slot {u.slot} cand {i}: evaluate failed: {e}")
                u.fits = fits

    # (3b) Optional screen (coverage-gated init): main thread, once, between propose and evaluate.
    _subset_note = []

    def _eval_slot_per_genome(u):
        u.evaluated += len(u.cands)
        fits = []
        for i, g in enumerate(u.cands):
            try:
                fits.append(_eval_one(u, g))
            except Exception as e:                      # noqa: BLE001
                if not isolate:
                    raise
                fits.append(None)
                u.eval_errors += 1
                ctx.log(f"[{tag}] gen {gen} slot {u.slot} cand {i}: evaluate failed: {e}")
        u.fits = fits

    def _run_screen(us):
        allof = {u.slot: set(range(len(u.cands))) for u in us}
        try:
            got = screen(us) or {}
        except Exception as e:                          # noqa: BLE001 — a screen never kills a run
            ctx.log(f"[{tag}] gen {gen}: screen raised ({e!r}) — evaluating EVERY candidate "
                    f"unscreened (fail-open)")
            for u in us:
                u.screen_error = repr(e)
            return allof
        keep = {}
        for u in us:
            k = got.get(u.slot)
            if not isinstance(k, (set, frozenset)) or any(
                    not isinstance(i, int) or not (0 <= i < len(u.cands)) for i in k):
                # Fail open for this slot and log it, so a malformed result is not mistaken for a
                # screen that rejected nothing. An empty set is valid and means "evaluate none".
                ctx.log(f"[{tag}] gen {gen} slot {u.slot}: screen returned {k!r}, not a set of "
                        f"valid candidate indices in 0..{len(u.cands) - 1} — evaluating this slot "
                        f"unscreened (fail-open)")
                keep[u.slot] = allof[u.slot]
            else:
                keep[u.slot] = set(k)
        return keep

    def _screened_evaluate(us):
        keep = _run_screen(us)
        subset_ok = bool(getattr(ctx.adapter, "EVAL_SUBSET_SAFE", False))
        restore = []
        for u in us:
            idx = sorted(keep[u.slot])
            restore.append((u, u.cands, idx))
            u.cands = [u.cands[i] for i in idx]         # shrunk for dispatch only; restored below
        partial = [t for t in restore if len(t[2]) != len(t[1])]
        solo = [t for t in partial if batch_hook is not None and not subset_ok]
        if solo and not _subset_note:
            _subset_note.append(True)
            ctx.log(f"[{tag}] {type(ctx.adapter).__name__} owns evaluate_batch but does not declare "
                    f"EVAL_SUBSET_SAFE — screened slots fall back to per-genome adapter.evaluate() "
                    f"so screened-out candidates are not evaluated anyway")
        solo_ids = {id(u) for u, _, _ in solo}
        for u, _, _ in solo:
            _eval_slot_per_genome(u)
        rest = [u for u in us if id(u) not in solo_ids]
        if rest:
            _evaluate_units(rest)
        for u, orig, idx in restore:                    # scatter back to full length, by position
            fits = [None] * len(orig)
            for j, i in enumerate(idx):
                if j < len(u.fits):
                    fits[i] = u.fits[j]
            u.cands, u.fits = orig, fits

    if screen is not None:
        _propose_units(units)                           # a screen forces the wide barrier
        _screened_evaluate(units)
    elif barrier:
        _propose_units(units)                           # wide barrier (opt-in)
        _evaluate_units(units)
    else:
        for u in units:                                 # per-slot schedule, in slot order
            _propose_units([u])
            _evaluate_units([u])

    # (4) Post-join, main thread, slot order: budget and node ids (admission is the caller's)
    for u in units:
        ctx.spend(u.evaluated)                          # charge the run's budget N
        pairs = [(g, f) for g, f in zip(u.cands, u.fits)
                 if f is not None and getattr(f, "valid", False)]
        u.children = [Node(ctx.next_idx(), g, f) for g, f in pairs]
    return units


def _admit_verdict(cells, child, cfg):
    """(cell, became_elite) for a child already placed by `admit`, recomputed from member order."""
    for c in cells:
        for i, n in enumerate(c["mem"]):
            if n is child:
                return c, (_rank(c["mem"][:i + 1], cfg)[0] is child)
    return None, False


def batch_gen_row(gen, parents, units, *, improvements, discoveries, rejected,
                  archive, occupied, total, elites, wall_s):
    """One row of the per-generation telemetry out[...]["batch"]["gens"], shared by every mode."""
    prim = [e.fit.primary for e in elites if e is not None]
    return {"gen": gen, "parents": list(parents),
            "n_cands": sum(len(u.cands) for u in units),
            "n_valid": sum(len(u.children) for u in units),
            "improvements": improvements, "discoveries": discoveries, "rejected": rejected,
            "archive": archive, "occupied": occupied,
            "coverage": (round(occupied / total, 3) if total else None),
            "qd_score": round(sum(prim), 3),
            "max_fitness": (round(max(prim), 3) if prim else None),
            "wall_s": round(wall_s, 2)}


def batch_out(ctx, *, batch_size, n_workers, parent_select, seed, gens,
              propose_errors, eval_errors, under_full=None):
    """The out[...]["batch"] report (emitted unless P = 1, one worker and round-robin selection)."""
    G = sum(1 for r in gens if r["gen"] > 0)
    _quiet = lambda *a, **k: None                        # noqa: E731 — telemetry must not re-log
    return {"batch_size": batch_size, "n_workers": n_workers,
            "effective_workers": {
                "propose": effective_workers(ctx.adapter, n_workers, log=_quiet, stage="propose"),
                "eval": effective_workers(ctx.adapter, n_workers, log=_quiet, stage="eval")},
            "barrier": bool(getattr(ctx.adapter, "BATCH_BARRIER_SAFE",
                                    getattr(ctx.adapter, "PARALLEL_SAFE", False))),
            "parallel_safe": bool(getattr(ctx.adapter, "PARALLEL_SAFE", False)),
            "parent_select": parent_select, "seed": seed,
            # Counted from the rows: under `max_evals` the last generation may have fewer parents.
            "generations": G, "proposals": sum(len(r["parents"]) for r in gens),
            "evals": sum(r["n_valid"] for r in gens), "gens": gens,
            **({"under_full": under_full} if under_full else {}),
            "totals": {"improvements": sum(r["improvements"] for r in gens),
                       "discoveries": sum(r["discoveries"] for r in gens),
                       "rejected": sum(r["rejected"] for r in gens),
                       "propose_errors": propose_errors, "eval_errors": eval_errors}}


def _batch_validate(parent_select, batch_size):
    if parent_select not in ("roundrobin", "uniform"):
        raise ValueError(f"parent_select must be 'roundrobin' or 'uniform', got {parent_select!r}")
    return max(1, int(batch_size))
