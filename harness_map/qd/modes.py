"""The three CVT-MAP-Elites drivers, the budget accounting N = M + G·P·C, and `run_frontier`."""
from __future__ import annotations

import dataclasses
import json
import os
import threading
import time
from pathlib import Path

import numpy as np

from harness_map import main as MH
from harness_map.main import Ctx, EpisodeResult, Fitness, Genome, Node, SearchMode, _out
from harness_map.qd import archive as _arch
from harness_map.qd import coverage as _cov
from harness_map.qd import partition as _part


# ── evaluation budget and run reports ───────────────────────────────────────────────────────
def audit_out(ctx, nodes, *, steer=None, steer_ok=None):
    """The out[...]["audit"] report: declared audit (`AUDIT_PROVIDED`) vs verdicts observed."""
    a = ctx.adapter
    taint = sum(1 for n in nodes if getattr(n.fit, "taint", "MISSING") != "MISSING")
    judge = sum(1 for n in nodes if getattr(n.fit, "judge", "MISSING") != "MISSING")
    fails = sum(1 for n in nodes if getattr(n.fit, "taint", "") == "FAIL"
                or getattr(n.fit, "judge", "") == "FAIL")
    declared = bool(getattr(a, "AUDIT_PROVIDED", False))
    out = {"declared": declared, "n_evaluated": len(nodes),
           "taint_populated": taint, "judge_populated": judge, "fail_verdicts": fails,
           "diagnose": hasattr(a, "diagnose")}
    if steer is not None:
        out["steer"] = bool(steer)
        out["steer_delivered"] = bool(steer_ok)
    if declared and not (taint or judge):
        ctx.log(f"[audit] WARNING: {type(a).__name__} declares AUDIT_PROVIDED=True but not one of "
                f"{len(nodes)} evaluated genomes carries a taint or judge verdict. Either the audit "
                f"never ran or its verdict never reached Fitness — this run is UNAUDITED.")
    return out


def _budget_slots(ctx, max_evals, batch_size, children_per_parent):
    """Parent slots that fit the remaining budget, and whether P was trimmed (C never is)."""
    if max_evals is None:
        return batch_size, False
    room = int(max_evals) - int(ctx._evals)
    slots = max(0, room // max(1, int(children_per_parent)))
    return min(batch_size, slots), (slots < batch_size)


def _check_M_against_N(ctx, tag, max_evals, m_evals):
    if max_evals is not None and m_evals > int(max_evals):
        ctx.log(f"[{tag}] BUDGET EXCEEDED BY M: initialization spent {m_evals} evaluations against "
                f"max_evals={max_evals}. M (anchors + seed_genomes pre-seed + the coverage phase) "
                f"is not optional and is charged before the generation loop can trim anything, so "
                f"N CANNOT BIND M — the run is already over budget by {m_evals - int(max_evals)} "
                f"and no generation will run. budget.bound_by will read "
                f"'max_evals_exceeded_by_M', NOT 'max_evals'.")


def _init_out(*, anchors, anchor_evals, pre_seed_offered, pre_seed_placed, pre_seed_evals,
              cover, k0, occupied):
    """The out[...]["init"] report: M by source, and the occupancy at the end of initialization."""
    proposer = (cover or {}).get("evaluations_valid", 0)
    return {"M": anchor_evals + pre_seed_evals + (cover or {}).get("evaluations", 0),
            "M_note": ("EVALUATIONS spent on initial individuals: anchors + pre-seed offered + "
                       "every candidate the coverage phase evaluated (including invalid ones — "
                       "they consumed the budget)"),
            "anchors": anchors, "anchor_evals": anchor_evals,
            "pre_seeded_offered": pre_seed_offered, "pre_seeded_placed": pre_seed_placed,
            "pre_seeded_evals": pre_seed_evals,
            "proposer_generated_placed": proposer,
            "proposer_generated_evals": (cover or {}).get("evaluations", 0),
            "k0": k0, "occupied_after_init": occupied,
            "coverage_after_init": (round(occupied / k0, 4) if k0 else None),
            "occupancy_note": ("occupied_after_init is the archive at the END OF INITIALIZATION "
                               "(anchors + pre-seed + coverage phase), NOT at the end of the run — "
                               "for that read final_occupied / final_cells"),
            "coverage": cover}


def budget_out(ctx, *, max_evals, rounds, batch_size, children_per_parent, m_evals,
               generations_ran, trims, bound_by):
    """The out[...]["budget"] report: planned vs realized N and the binding stop (`bound_by`)."""
    predicted = m_evals + rounds * batch_size * children_per_parent
    C = max(1, int(children_per_parent))
    over = residue = unspent = None
    if max_evals is not None:
        N = int(max_evals)
        over = max(0, ctx._evals - N)
        residue = max(0, N - m_evals) % C
        unspent = max(0, N - ctx._evals)              # measured shortfall, not the arithmetic one
        if m_evals > N:
            bound_by = "max_evals_exceeded_by_M"
        elif (bound_by == "rounds" and generations_ran >= rounds and ctx._evals >= N):
            bound_by = "rounds+max_evals"
    return {"paradigm": "N = M + G*P*C",
            "N_budget": max_evals, "N_realized": ctx._evals,
            "M": m_evals, "G_planned": rounds, "G_ran": generations_ran,
            "P": batch_size, "C": children_per_parent,
            "B": batch_size * children_per_parent,
            "N_predicted": predicted,
            "N_predicted_note": ("M + G_planned*P*C, with M the realized initial-individual count "
                                 "(random_init decides it, so it is not knowable a priori)"),
            "on_prediction": (ctx._evals == predicted),
            "N_overspend": over,
            "N_unreachable_residue": residue,
            "N_unspent": unspent,
            "N_exactness_note": ("max_evals cannot bind M (N_overspend > 0 means it was exceeded "
                                 "before generation 1); the loop otherwise lands on the largest "
                                 "M + k*C <= N, whose arithmetic shortfall is "
                                 "N_unreachable_residue = (N - M) mod C. N_unspent = N_budget - "
                                 "N_realized is the measured shortfall and is the one to read: a "
                                 "slot that delivers fewer than C children leaves evaluations "
                                 "unspent that the arithmetic residue cannot see"),
            "bound_by": bound_by,
            "trimmed_generations": trims}


def run_frontier(adapter, cvt_dir, rules, *, corpus=None, k0=2, seed_genomes=None,
                 rounds=12, sigma=0.0, k_split=2, depth_cap=2, patience=1,
                 steer=True, refit=False, log=print,
                 batch_size=1, n_workers=1, parent_select="roundrobin", seed=0,
                 children_per_parent=2, max_evals=None,
                 random_init=0.5, random_init_batch=1, random_init_max_proposals=0,
                 random_init_max_localizations=0, random_init_stall=0, random_init_safety=0,
                 random_init_nonuniformity=0,
                 random_init_strict=True, random_init_screen=True):
    """Fit or load the frozen CVT in `cvt_dir`, then run `mapelites_adaptive` over `adapter`."""
    cvt_dir = Path(cvt_dir)
    if refit or not (cvt_dir / "cvt_blocks.npz").exists():
        if not corpus:
            raise ValueError("run_frontier needs a corpus to fit the block CVT (or a pre-fit cvt_dir)")
        n, dim, k, bs = _part.fit_cvt_blocks(corpus, cvt_dir, rules, k0=k0)
        log(f"[frontier] fit CVT: n={n} joint_dim={dim} k0={k} block_split(structure)={bs}")
    phi, Z0, block_split = _part.make_phi_blocks(cvt_dir, rules)
    return MH.run(adapter, adapter.cfg, mode="mapelites_adaptive", log=log,
                  phi_fn=phi, init_centroids=Z0, block_split=block_split, steer=steer,
                  seed_genomes=seed_genomes or [], rounds=rounds, sigma=sigma,
                  k_split=k_split, depth_cap=depth_cap, patience=patience,
                  batch_size=batch_size, n_workers=n_workers, parent_select=parent_select,
                  seed=seed,
                  children_per_parent=children_per_parent, max_evals=max_evals,
                  random_init=random_init, random_init_batch=random_init_batch,
                  random_init_max_proposals=random_init_max_proposals,
                  random_init_max_localizations=random_init_max_localizations,
                  random_init_stall=random_init_stall, random_init_safety=random_init_safety,
                  random_init_nonuniformity=random_init_nonuniformity,
                  random_init_strict=random_init_strict,
                  random_init_screen=random_init_screen)


# ── run persistence: snapshot, per-evaluation ledger, resume ─────────────────────────────────
def _write_partial_frontier(fr, ctx, *, it, k0, k_split, sigma, depth_cap, patience, steer, split_target,
                            m_evals, occ_after_init, cover_tel, K0, max_evals, rounds, batch_size, C,
                            trims):
    """Write the rolling resume snapshot to $HARNESS_MAP_CKPT_DIR/frontier_partial.json, if set."""
    import json as _json
    import os as _os
    d = _os.environ.get("HARNESS_MAP_CKPT_DIR")
    if not d:
        return
    try:
        out = {"frontier": {
            "k0": k0, "k_split": k_split, "sigma": sigma, "two_sigma": 2 * sigma, "depth_cap": depth_cap,
            "patience": patience, "steer": steer, "split_target": split_target, "block_split": fr.block_split,
            "n_splits": len(fr.splits), "n_retreats": len(fr.retreats),
            "disagree_log": fr.disagree_log, "splits": fr.splits, "retreats": fr.retreats,
            "steer_log": fr.steer_log, "trace": fr.trace,
            "anchors": [str(n.genome.source_dir) for n in fr.anchors],
            "nodes": [{"idx": n.idx, "source_dir": str(n.genome.source_dir), "label": n.genome.label,
                       "depth": n.depth, "parents": list(n.parents)} for n in fr.allnodes],
            "open_splits": [{"parent": s["parent"]["cid"], "pre_elite": s["pre_elite"],
                             "born": s["born"], "breeds": s["breeds"]} for s in fr.open_splits],
            "final_cells": len(fr.cells), "final_occupied": sum(1 for c in fr.cells if c["mem"]),
            "init": _init_out(anchors=len(fr.anchors), anchor_evals=len(fr.anchors),
                                      pre_seed_offered=len(fr.pre_seed), pre_seed_placed=fr.n_pre_seeded,
                                      pre_seed_evals=len(fr.pre_seed), cover=cover_tel, k0=K0,
                                      occupied=occ_after_init),
            "budget": budget_out(ctx, max_evals=max_evals, rounds=rounds, batch_size=batch_size,
                                         children_per_parent=C, m_evals=m_evals, generations_ran=it,
                                         trims=trims, bound_by="partial"),
            "partial": True, "last_round": it}}
        Path(d).mkdir(parents=True, exist_ok=True)
        p = Path(d) / "frontier_partial.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(_json.dumps(out, indent=1, default=str))
        tmp.replace(p)
    except Exception as e:                                # noqa: BLE001 — a snapshot must never fail a round
        try:
            ctx.log(f"[frontier] WARN: partial snapshot not written: {e!r}")
        except Exception:                                 # noqa: BLE001
            pass


_LOCK = threading.Lock()
_STATE: dict = {"k": 0, "best": None, "dir": None}


def _target() -> Path | None:
    d = os.environ.get("HARNESS_MAP_CKPT_DIR")
    return Path(d) if d else None


def _per_instance(fit) -> dict:
    try:
        v = getattr(fit, "vector", None) or {}
        return {str(k): round(float(x), 4) for k, x in v.items()}
    except Exception:
        return {}


def _plain(x):
    """JSON-ready copy of a Fitness: dataclasses, numpy values and Paths converted."""
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return {f.name: _plain(getattr(x, f.name)) for f in dataclasses.fields(x)}
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    return x


def _fitness_from_row(row: dict) -> Fitness:
    """The Fitness a ledger row recorded (unknown fields are ignored)."""
    d = dict(row["fitness"])
    ep = {f.name for f in dataclasses.fields(EpisodeResult)}
    d["rows"] = [EpisodeResult(**{k: v for k, v in r.items() if k in ep}) for r in d.get("rows") or []]
    keep = {f.name for f in dataclasses.fields(Fitness)}
    return Fitness(**{k: v for k, v in d.items() if k in keep})


def seed_from_disk() -> tuple[int, float | None]:
    """On resume, continue the existing ledger's k and best_so_far; returns (k, best)."""
    d = _target()
    if d is None or not (d / "evals.jsonl").exists():
        return 0, None
    rows = [json.loads(l) for l in (d / "evals.jsonl").read_text().splitlines() if l.strip()]
    best = max((r["primary"] for r in rows if r.get("valid") and r.get("primary") is not None),
               default=None)
    with _LOCK:
        _STATE.update(k=len(rows), best=best, dir=str(d))
    return len(rows), best


def record(cands, fits) -> None:
    """Append one row per (genome, fitness) pair (None = failed evaluation); never raises."""
    d = _target()
    if d is None:
        return
    try:
        with _LOCK:
            if _STATE["dir"] != str(d):                 # first record into this dir: reset k
                _STATE.update(k=0, best=None, dir=str(d))
                d.mkdir(parents=True, exist_ok=True)
            lines = []
            for g, f in zip(cands, fits):
                _STATE["k"] += 1
                prim = float(getattr(f, "primary")) if f is not None and getattr(f, "primary", None) is not None else None
                valid = bool(getattr(f, "valid", False)) if f is not None else False
                if valid and prim is not None and (_STATE["best"] is None or prim > _STATE["best"]):
                    _STATE["best"] = prim
                lines.append(json.dumps({
                    "k": _STATE["k"], "label": getattr(g, "label", "?"),
                    "source_dir": str(getattr(g, "source_dir", "")),
                    "primary": prim, "valid": valid, "best_so_far": _STATE["best"],
                    "per_instance": _per_instance(f), "ts": round(time.time(), 1),
                    "fitness": _plain(f) if f is not None else None,    # what resume rebuilds from
                }, default=str))
            with open(d / "evals.jsonl", "a") as fh:    # append = crash-safe, ordered
                fh.write("\n".join(lines) + "\n")
            tmp = d / ".ckpt_latest.tmp"                 # atomic rolling snapshot
            tmp.write_text(json.dumps({"k": _STATE["k"], "best_so_far": _STATE["best"],
                                       "ckpt_dir": str(d)}, indent=2))
            tmp.rename(d / "ckpt_latest.json")
    except Exception as e:                               # noqa: BLE001 — never fail an evaluation
        try:
            # A full disk also truncates score files and caches, so it gets its own warning.
            import errno as _errno
            if getattr(e, "errno", None) == _errno.ENOSPC:
                print("[checkpoint] *** DISK FULL (ENOSPC) *** — checkpoints, caches and score files "
                      "are being truncated. Free space NOW; treat any file written since as suspect.",
                      flush=True)
            print(f"[checkpoint] WARN: {e!r}", flush=True)
        except Exception:
            pass


# ── resume from a snapshot ──────────────────────────────────────────────────────────────────────
def _ledger_rows(work_root: Path) -> list[dict]:
    """Rows of the run's `evals.jsonl`: in $HARNESS_MAP_CKPT_DIR, else <work_root>/../checkpoints."""
    for d in (os.environ.get("HARNESS_MAP_CKPT_DIR"), work_root.parent / "checkpoints"):
        p = Path(d) / "evals.jsonl" if d else None
        if p is not None and p.exists():
            return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return []


def restore_frontier(ctx, R: dict, *, phi_fn, block_split, sigma, k_split, depth_cap, patience,
                     steer, split_target, tag="frontier", log=print):
    """Rebuild the Frontier from snapshot block `R` without re-evaluating; (frontier, last round)."""
    work_root = Path(ctx.adapter.work_root)
    trace = R["trace"]
    if not trace:
        raise ValueError("RESULT.json has no frontier.trace — cannot restore")

    fr = _arch.Frontier(ctx=ctx, phi_fn=phi_fn, block_split=block_split, sigma=sigma,
                        k_split=k_split, depth_cap=depth_cap, patience=patience, steer=steer,
                        split_target=split_target, tag=tag)

    # ── 1. every cell that ever existed (split parents included), from its first snapshot
    cells: dict[int, dict] = {}
    for snap in trace:
        for c in snap["cells"]:
            if c["cid"] not in cells:
                cells[c["cid"]] = {"cid": c["cid"], "z": np.asarray(c["z"], float), "mem": [],
                                   "depth": c["depth"], "parent": None, "children": [],
                                   "pre_elite": None, "born": snap["round"], "split_failed": set()}
    for snap in trace:                                    # parent links (by cid -> object)
        for c in snap["cells"]:
            cell = cells[c["cid"]]
            if c["parent"] is not None and cell["parent"] is None:
                par = cells[c["parent"]]
                cell["parent"] = par
                if cell not in par["children"]:
                    par["children"].append(cell)
    for r in R.get("retreats", []):                       # per the retreat log: failed shapes, and
        if r["parent"] in cells:                          # a merged subtree leaves the tree for good
            par = cells[r["parent"]]
            par["split_failed"] = set(tuple(s) if isinstance(s, list) else s
                                      for s in r.get("split_failed_shape", []))
            merged = set(r.get("merged_subtree", []))
            par["children"] = [k for k in par["children"] if k["cid"] not in merged]

    last = trace[-1]
    fr.cells = [cells[c["cid"]] for c in last["cells"]]  # active leaves = last snapshot
    fr._cid = max(cells)

    # ── 2. nodes: lineage from the snapshot, each Fitness from the ledger (no re-evaluation)
    if "nodes" not in R or "open_splits" not in R:
        raise ValueError("snapshot has no nodes/open_splits blocks: written by an older version")
    rows = _ledger_rows(work_root)
    row_of = {r["source_dir"]: r for r in rows if r.get("source_dir") and r.get("fitness")}
    node_by_dir: dict[str, Node] = {}
    for rec in sorted(R["nodes"], key=lambda r: r["idx"]):
        sd = rec["source_dir"]
        row = row_of.get(sd)
        if row is None:
            raise FileNotFoundError(f"no ledger row with a fitness for {sd}; resuming needs the "
                                    f"run's evals.jsonl (HARNESS_MAP_CKPT_DIR)")
        node_by_dir[sd] = Node(idx=rec["idx"], genome=Genome(source_dir=Path(sd), label=rec["label"]),
                               fit=_fitness_from_row(row), depth=rec["depth"],
                               parents=tuple(rec["parents"]))
    for c in last["cells"]:
        missing = [sd for sd in c["members"] if sd not in node_by_dir]
        if missing:
            raise ValueError(f"snapshot cell {c['cid']} lists members absent from its nodes: {missing}")
        cells[c["cid"]]["mem"] = [node_by_dir[sd] for sd in c["members"]]
    fr.allnodes = list(node_by_dir.values())
    ctx._next = max((n.idx for n in fr.allnodes), default=-1) + 1
    anchors = [node_by_dir[sd] for sd in R.get("anchors", []) if sd in node_by_dir]
    fr.anchors = anchors or fr.allnodes[:1]              # the anchor may have left every cell
    fr.pre_seed, fr.n_pre_seeded = [], 0

    # ── 3. logs, verbatim
    fr.splits = list(R.get("splits", []))
    fr.retreats = list(R.get("retreats", []))
    fr.trace = list(trace)
    fr.steer_log = list(R.get("steer_log", []))
    fr.disagree_log = list(R.get("disagree_log", []))

    # ── 4. open splits, exactly as the snapshot recorded them (the trace rounds elites)
    fr.open_splits = [{"parent": cells[s["parent"]], "pre_elite": s["pre_elite"], "born": s["born"],
                       "breeds": s["breeds"]} for s in R["open_splits"]]

    log(f"[resume] frontier restored: {len(fr.cells)} active cells (of {len(cells)} ever), "
        f"{len(fr.allnodes)} member nodes, {len(fr.open_splits)} open split(s), "
        f"{len(fr.splits)} splits / {len(fr.retreats)} retreats replayed, last round {last['round']}")
    return fr, last["round"]


def verify_restored(fr, R: dict, atol: float = 1e-9) -> list[str]:
    """Compare `archive.snap(fr)` with the snapshot's last trace row; returns the mismatches."""
    want = R["trace"][-1]["cells"]
    got = _arch.snap(fr)
    errs = []
    if len(got) != len(want):
        errs.append(f"cell count {len(got)} != {len(want)}")
    for g, w in zip(sorted(got, key=lambda c: c["cid"]), sorted(want, key=lambda c: c["cid"])):
        for k in ("cid", "depth", "parent", "n"):
            if g[k] != w[k]:
                errs.append(f"cell {w['cid']}: {k} {g[k]} != {w[k]}")
        if (g["elite"] is None) != (w["elite"] is None) or (
                g["elite"] is not None and abs(g["elite"] - w["elite"]) > 1e-6):
            errs.append(f"cell {w['cid']}: elite {g['elite']} != {w['elite']}")
        if sorted(g["members"]) != sorted(w["members"]):
            errs.append(f"cell {w['cid']}: members differ")
        if np.max(np.abs(np.asarray(g["z"]) - np.asarray(w["z"]))) > atol:
            errs.append(f"cell {w['cid']}: centroid z differs")
    return errs


def seed_ctx_counters(ctx, ledger_rows: list[dict], n_evals_fallback: int = 0) -> None:
    """Continue the run's counters after a resume: proposal rid, evaluation count, node index."""
    import re
    rids = []
    work_root = Path(ctx.adapter.work_root)
    if work_root.exists():                                # proposal directories name their rid
        for pat in (r"^prop_r0*(\d+)$", r"^cand_r0*(\d+)_"):
            for d in work_root.iterdir():
                m = re.match(pat, d.name)
                if m:
                    rids.append(int(m.group(1)))
    ctx._round = (max(rids) + 1) if rids else 1
    ctx._evals = len(ledger_rows) if ledger_rows else int(n_evals_fallback)
    try:                                                  # continue the ledger's counters
        seed_from_disk()
    except Exception:                                     # noqa: BLE001 — counters are cosmetic
        pass

def _mapelites_adaptive(ctx: Ctx, *, rounds=12, seed_genomes=None, phi_fn=None, init_centroids=None,
                        sigma=0, k_split=2, depth_cap=2, patience=1,
                        block_split=None, steer=True, split_target="variance",
                        batch_size=1, n_workers=1, parent_select="roundrobin", seed=0,
                        children_per_parent=2, max_evals=None,
                        random_init=0.5, random_init_batch=1, random_init_max_proposals=0,
                        random_init_max_localizations=0, random_init_max_evals=0,
                        random_init_stall=0,
                        random_init_safety=0,
                        random_init_nonuniformity=0,
                        random_init_strict=True, random_init_screen=True,
                        resume_from=None):
    """Adaptive-resolution CVT-MAP-Elites (the main method): breed, place, refine or retreat."""
    if phi_fn is None or init_centroids is None:
        raise ValueError("mapelites_adaptive requires phi_fn and init_centroids")
    import time as _time
    batch_size = _arch._batch_validate(parent_select, batch_size)
    C = max(1, int(children_per_parent))
    max_evals = None if max_evals is None else max(0, int(max_evals))
    (random_init, ri_batch, ri_max_prop, ri_max_loc,
     ri_stall, ri_max_evals, ri_safety, ri_nonunif) = _cov._coverage_validate(
        random_init, random_init_batch, random_init_max_proposals,
        random_init_max_localizations, random_init_stall, random_init_max_evals,
        random_init_safety, random_init_nonuniformity)
    _batched = (batch_size > 1 or n_workers > 1 or parent_select != "roundrobin")
    # ── Resume from a snapshot, or seed a new archive ─────────────────────────────────────────────
    _resume_round = 0
    if resume_from:
        import json as _json
        _R = _json.loads(Path(resume_from).read_text())["frontier"]
        fr, _resume_round = restore_frontier(
            ctx, _R, phi_fn=phi_fn, block_split=block_split, sigma=sigma, k_split=k_split,
            depth_cap=depth_cap, patience=patience, steer=steer, split_target=split_target,
            log=ctx.log)
        _errs = verify_restored(fr, _R)
        if _errs:
            raise RuntimeError(f"resume: restored archive does not match the snapshot: {_errs[:3]}")
        seed_ctx_counters(ctx, _ledger_rows(Path(ctx.adapter.work_root)),
                                n_evals_fallback=int((_R.get("budget") or {}).get("N_realized", 0)))
        ctx.log(f"[frontier] RESUMED from {resume_from}: archive verified field-for-field; "
                f"continuing at round {_resume_round + 1} of {rounds}, ctx._round={ctx._round} "
                f"ctx._evals={ctx._evals}")
        random_init = 0.0                                # coverage-init already happened
    else:
        fr = _arch._seed(ctx, phi_fn=phi_fn, init_centroids=init_centroids, block_split=block_split,
                   seed_genomes=seed_genomes, sigma=sigma, k_split=k_split, depth_cap=depth_cap,
                   patience=patience, steer=steer, split_target=split_target)
    K0 = int(_R["k0"]) if resume_from else len(fr.cells)   # == len(init_centroids); no split yet
    gens, n_perr, n_eerr = [], 0, 0

    def _land(cells_sel, steers, units, it, depth, prev):
        """Admit every parent's children; return (touched, improved, discovered, rejected)."""
        touched, imp, dis, rej = [], 0, 0, 0
        for cell, (directive, tgt_cid), unit in zip(cells_sel, steers, units):
            parent, round_fits = unit.parent, []
            for child in unit.children:
                child.depth = depth
                child.parents = (parent.idx,)
                fr.allnodes.append(child)
                round_fits.append(round(child.fit.primary, 3))
            fr._render = {"round": it, "parent": parent, "cell": cell, "tgt": tgt_cid}
            u_touched = _arch.admit(fr, unit.children)
            for c in u_touched:
                if c not in touched:
                    touched.append(c)
            fr.steer_log.append({"round": it, "parent_cell": cell["cid"], "steer_target": tgt_cid,
                                 "directive": (directive[:160] if directive else None),
                                 "delivered": bool(directive) and fr.steer_ok,
                                 "n_cands": len(unit.cands), "n_evaluated": len(round_fits),
                                 "child_fits": round_fits,
                                 "touched_cells": [c["cid"] for c in u_touched]})
            if prev is not None:                         # discovery = cell was empty before the gen
                for child in unit.children:
                    c, adm = _arch._admit_verdict(fr.cells, child, ctx.cfg)
                    if not adm:
                        rej += 1
                    elif prev.get(id(c)):
                        imp += 1
                    else:
                        dis += 1
        return touched, imp, dis, rej

    # ── Coverage-gated initialization ─────────────────────────────────────────────────────────────
    cover_tel = None
    if random_init > 0:
        view = _arch.frontier_cover_view(fr, k0=K0, random_init=random_init,
                                   parent_select=parent_select, seed=seed, land_fn=_land)
        cover_tel = _cov.coverage_init(
            ctx, view, random_init=random_init, batch=ri_batch,
            screen_on=random_init_screen, strict=random_init_strict,
            max_proposals=ri_max_prop, max_localizations=ri_max_loc, stall=ri_stall,
            max_evals=max_evals, phase_max_evals=ri_max_evals, safety=ri_safety,
            nonuniformity=ri_nonunif,
            n_workers=n_workers)
        n_perr += cover_tel["propose_errors"]; n_eerr += cover_tel["eval_errors"]
        if not cover_tel["covered"] and random_init_strict:
            raise _cov._coverage_raise(cover_tel, "mapelites_adaptive")
    m_evals = ctx._evals
    occ_after_init = sum(1 for c in fr.cells if c["mem"])
    if resume_from:                                      # M and init telemetry of the original run
        _I = _R.get("init") or {}
        m_evals = int(_I.get("M", m_evals))
        occ_after_init = int(_I.get("occupied_after_init", occ_after_init))
        cover_tel = _I.get("coverage")
    _check_M_against_N(ctx, fr.tag, max_evals, m_evals)
    g_ran, trims, bound_by = _resume_round, [], "rounds"
    if resume_from:                                      # carry the prior run's trims forward
        trims = list((_R.get("budget") or {}).get("trimmed_generations", []))
    # Generations that asked for more parents than there were occupied cells (see `select_batch`).
    under_full = {}
    it = _resume_round                                  # 0 fresh; last completed round on resume
    while it < rounds:
        it += 1
        if ctx.should_stop and ctx.should_stop():
            bound_by = "should_stop"
            break
        # A coverage-init founder may already score 1.0, so check saturation before breeding.
        if os.environ.get("ADAPTIVE_SAT_BEFORE_ROUND", "1") != "0":
            _bv0 = max((n.fit.primary for n in fr.allnodes), default=0.0)
            if _bv0 >= 1.0:
                ctx.log(f"[frontier] SATURATED before round {it} (best={_bv0:.3f}) — early-stopping; "
                        f"{rounds - it + 1} of {rounds} round(s) skipped.")
                bound_by = "saturated_preround"
                break
        # Trim P (never C) at the generation boundary so the run lands exactly on max_evals.
        b_it, trimmed = _budget_slots(ctx, max_evals, batch_size, C)
        if b_it <= 0:
            bound_by = "max_evals"
            ctx.log(f"[frontier] max_evals={max_evals} reached at {ctx._evals} evaluations — "
                    f"stopping before generation {it} of {rounds} (no full child fits)")
            break
        if trimmed:
            trims.append({"gen": it, "P": batch_size, "P_trimmed": b_it})
            ctx.log(f"[frontier] generation {it}: TRIMMING P {batch_size} -> {b_it} so the run "
                    f"lands exactly on max_evals={max_evals} ({ctx._evals} spent)")
        t0 = _time.time()
        prev = {id(c): bool(c["mem"]) for c in fr.cells} if _batched else None
        # (1) Select P cells from one snapshot; each slot credits its open splits a breeding turn.
        cells_sel = _arch.select_cells(fr, it, b_it, rule=parent_select, rng=_arch.gen_rng(seed, it),
                                 log=ctx.log, tally=under_full, tag="frontier")
        cell = cells_sel[0]                              # the trace records the first parent cell
        parents = [_arch.elite(fr, c) for c in cells_sel]
        # (2) Steering directives for every parent, all computed before any admission.
        steers = [_arch._steer(fr, c) for c in cells_sel]      # alias: `steer` is a parameter
        # (3) Propose C children per parent and evaluate all of them behind a barrier.
        units = _arch.breed_batch(ctx, parents, directives=[d for d, _ in steers], n_workers=n_workers,
                            gen=it, tag="frontier", n_children=C)
        g_ran = it
        # (4) Admit each child to its nearest-centroid cell; non-elites stay on as members.
        touched, imp, dis, rej = _land(cells_sel, steers, units, it, it, prev)
        # (5) Resolution control: split at most one cell (2σ gate), then retreat unpaid splits.
        _arch.refine(fr, it)
        _arch.retreat(fr, it)
        if _batched:
            n_perr += sum(1 for u in units if u.propose_error)
            n_eerr += sum(u.eval_errors for u in units)
            gens.append(_arch.batch_gen_row(it, [c["cid"] for c in cells_sel], units, improvements=imp,
                                      discoveries=dis, rejected=rej, archive=len(fr.cells),
                                      occupied=sum(1 for c in fr.cells if c["mem"]),
                                      total=len(fr.cells),
                                      elites=[_arch.elite(fr, c) for c in fr.cells if c["mem"]],
                                      wall_s=_time.time() - t0))
        best_val = max((n.fit.primary for n in fr.allnodes), default=0.0)
        fr.trace.append({"round": it, "parent_cell": cell["cid"],
                         "touched_cells": [c["cid"] for c in touched], "n_cells": len(fr.cells),
                         "occupied": sum(1 for c in fr.cells if c["mem"]),
                         "depths": sorted(c["depth"] for c in fr.cells if c["mem"]),
                         "best": round(best_val, 3), "cells": _arch.snap(fr)})
        ctx.log(f"[frontier] round {it}: frontier={len(fr.cells)} cells (occupied "
                f"{sum(1 for c in fr.cells if c['mem'])}, depths {sorted(c['depth'] for c in fr.cells if c['mem'])}) "
                f"best={best_val:.3f}")
        # Rolling snapshot (if HARNESS_MAP_CKPT_DIR is set); pass it as `resume_from` to continue.
        _write_partial_frontier(fr, ctx, it=it, k0=len(init_centroids), k_split=k_split, sigma=sigma,
                                depth_cap=depth_cap, patience=patience, steer=steer,
                                split_target=split_target, m_evals=m_evals, occ_after_init=occ_after_init,
                                cover_tel=cover_tel, K0=K0, max_evals=max_evals, rounds=rounds,
                                batch_size=batch_size, C=C, trims=trims)
        # Stop at saturation: primary lies in [0, 1], so a best of 1.0 cannot improve.
        if best_val >= 1.0:
            ctx.log(f"[frontier] round {it}: SATURATED (best={best_val:.3f}) — early-stopping this arm; "
                    f"{rounds - it} of {rounds} rounds skipped.")
            break

    out = _out(fr.allnodes, ctx.cfg, ctx)
    n_dis = sum(1 for d in fr.disagree_log if not d["agree"])
    out["frontier"] = {
        "k0": len(init_centroids), "k_split": k_split, "sigma": sigma,
        "two_sigma": 2 * sigma, "depth_cap": depth_cap, "patience": patience,
        "steer": steer, "split_target": split_target, "block_split": fr.block_split,
        "n_splits": len(fr.splits), "n_retreats": len(fr.retreats),
        "disagreement_rate": (round(n_dis / len(fr.disagree_log), 3) if fr.disagree_log else None),
        "n_rounds_logged": len(fr.disagree_log), "disagree_log": fr.disagree_log,
        "splits": fr.splits, "retreats": fr.retreats, "final_cells": len(fr.cells),
        "final_occupied": sum(1 for c in fr.cells if c["mem"]),
        "final_depths": sorted(c["depth"] for c in fr.cells if c["mem"]),
        "mixed_resolution": len(set(c["depth"] for c in fr.cells if c["mem"])) > 1,
        "final_cell_table": [{"cid": c["cid"], "depth": c["depth"], "n_mem": len(c["mem"]),
                              "elite": (round(_arch.elite(fr, c).fit.primary, 3) if c["mem"] else None)}
                             for c in sorted(fr.cells, key=lambda x: x["cid"])],
        "steer_log": fr.steer_log, "trace": fr.trace,
        # N = M + G·P·C: the initialization and budget reports
        "init": _init_out(anchors=len(fr.anchors), anchor_evals=len(fr.anchors),
                          pre_seed_offered=len(fr.pre_seed),
                          pre_seed_placed=fr.n_pre_seeded,
                          pre_seed_evals=len(fr.pre_seed),
                          cover=cover_tel, k0=K0,
                          occupied=occ_after_init),
        "budget": budget_out(ctx, max_evals=max_evals, rounds=rounds, batch_size=batch_size,
                             children_per_parent=C, m_evals=m_evals, generations_ran=g_ran,
                             trims=trims, bound_by=bound_by)}
    out["frontier"]["audit"] = audit_out(ctx, fr.allnodes, steer=steer, steer_ok=fr.steer_ok)
    if _batched:                                         # per-generation batch telemetry
        out["frontier"]["batch"] = _arch.batch_out(ctx, batch_size=batch_size, n_workers=n_workers,
                                             parent_select=parent_select, seed=seed,
                                             gens=gens, propose_errors=n_perr,
                                             eval_errors=n_eerr, under_full=under_full)
    return out


def _mapelites_hier(ctx: Ctx, *, rounds=12, seed_genomes=None, phi_fn=None,
                    centroid_levels=None, level_start=None, block_split=None,
                    steer=True, resume_from=None,
                    batch_size=1, n_workers=1, parent_select="roundrobin", seed=0,
                    children_per_parent=2, max_evals=None,
                    random_init=0.5, random_init_batch=1, random_init_max_proposals=0,
                    random_init_max_localizations=0, random_init_max_evals=0,
                    random_init_stall=0,
                    random_init_safety=0,
                    random_init_nonuniformity=0,
                    random_init_strict=True, random_init_screen=True):
    """CVT-MAP-Elites with scheduled coarse-to-fine resolution (Mouret & Clune 2015, §9.2)."""
    if phi_fn is None:
        raise ValueError("mapelites_hier requires phi_fn (see main.make_phi_blocks)")
    ladder = _arch.hier_ladder(centroid_levels)
    import time as _time
    batch_size = _arch._batch_validate(parent_select, batch_size)
    C = max(1, int(children_per_parent))
    max_evals = None if max_evals is None else max(0, int(max_evals))
    (random_init, ri_batch, ri_max_prop, ri_max_loc,
     ri_stall, ri_max_evals, ri_safety, ri_nonunif) = _cov._coverage_validate(
        random_init, random_init_batch, random_init_max_proposals,
        random_init_max_localizations, random_init_stall, random_init_max_evals,
        random_init_safety, random_init_nonuniformity)
    _batched = (batch_size > 1 or n_workers > 1 or parent_select != "roundrobin")
    L = len(ladder)
    program, _prog_given = _arch.hier_program(level_start, rounds, L)
    # ── Resume from a snapshot, or seed a new archive ─────────────────────────────────────────────
    # sigma, k_split, patience and split_target only populate the Frontier (hier never splits);
    # depth_cap=L-1 because a cell's depth is its rung.
    _resume_round = 0
    if resume_from:
        import json as _json
        _R = _json.loads(Path(resume_from).read_text())["frontier"]
        fr, _resume_round = restore_frontier(
            ctx, _R, phi_fn=phi_fn, block_split=block_split, sigma=0.05, k_split=2,
            depth_cap=L - 1, patience=1, steer=steer, split_target="variance", tag="hier",
            log=ctx.log)
        _errs = verify_restored(fr, _R)
        if _errs:
            raise RuntimeError(f"resume: restored archive does not match the snapshot: {_errs[:3]}")
        seed_ctx_counters(ctx, _ledger_rows(Path(ctx.adapter.work_root)),
                                n_evals_fallback=int((_R.get("budget") or {}).get("N_realized", 0)))
        ctx.log(f"[hier] RESUMED from {resume_from}: archive verified field-for-field; "
                f"continuing at round {_resume_round + 1} of {rounds}, ctx._round={ctx._round} "
                f"ctx._evals={ctx._evals}")
        random_init = 0.0                                # coverage-init already happened
    else:
        fr = _arch._seed(ctx, phi_fn=phi_fn, init_centroids=ladder[0], block_split=block_split,
                   seed_genomes=seed_genomes, sigma=0.05, k_split=2, depth_cap=L - 1,
                   patience=1, steer=steer, split_target="variance", tag="hier")
    # K0 is the coarsest rung, the live partition throughout initialization.
    K0 = int(_R["k0"]) if resume_from else len(ladder[0])
    # Attach the ladder for hierarchical placement (used only if it carries a parent map). Doing
    # this after `seed` is safe: on rung 0 both placement rules are global nearest centroid.
    fr.ladder, fr.level = ladder, 0
    fr.trace[0]["level"] = 0                             # every trace row carries `level`
    if getattr(ladder, "parents", None) is not None:
        ctx.log("[hier] placement: HIERARCHICAL — the ladder carries a parent->children map, so "
                "`place` resolves the coarse cell first and then the nearest child within it. "
                "Every occupied fine cell's members are a subset of exactly ONE coarse cell's "
                "members (for the live population, not just the fitted corpus), an advance can "
                "only SUBDIVIDE competition, and occupancy cannot fall — EXCEPT for a member "
                "resolving into a coarse cell with no children at the target rung, "
                "`ladder_cell_index`'s documented degenerate fallback, which "
                "build_centroid_ladder's nested build cannot produce but a hand-built parent map "
                "can.")
    ctx.log(f"[hier] ladder: {L} levels k={[len(z) for z in ladder]} | program level_start={program} "
            f"({'caller-supplied' if _prog_given else f'default even split of {rounds} rounds'}) | "
            f"Mouret & Clune 2015 arXiv:1504.04909 §9.2: UNCONDITIONAL, catch-up, monotone, single "
            f"live map. No spread gate, no backtrack, no split/retreat.")
    for lvl in range(1, L):
        if len(ladder[lvl]) <= len(ladder[lvl - 1]):
            ctx.log(f"[hier] WARNING: rung {lvl} has k={len(ladder[lvl])} <= rung {lvl - 1}'s "
                    f"k={len(ladder[lvl - 1])} — the ladder contract is coarse->FINE (not raising: "
                    f"a caller may deliberately repeat a rung)")
    _late = [lvl for lvl in range(1, L) if program[lvl] > rounds]
    if _late:
        ctx.log(f"[hier] WARNING: levels {_late} start at generations "
                f"{[program[l] for l in _late]}, past the {rounds}-generation budget — they will "
                f"never be reached (reported as unreached_levels, not skipped_levels)")
    active = 0
    activated_at = {0: 0}                                # level -> first generation run at it
    advances, skipped_levels = [], set()
    gens, n_perr, n_eerr = [], 0, 0

    def _land(cells_sel, steers, units, it, depth, prev):
        """Same as `_mapelites_adaptive`'s `_land`."""
        touched, imp, dis, rej = [], 0, 0, 0
        for cell, (directive, tgt_cid), unit in zip(cells_sel, steers, units):
            parent, round_fits = unit.parent, []
            for child in unit.children:
                child.depth = depth
                child.parents = (parent.idx,)
                fr.allnodes.append(child)
                round_fits.append(round(child.fit.primary, 3))
            fr._render = {"round": it, "parent": parent, "cell": cell, "tgt": tgt_cid}
            u_touched = _arch.admit(fr, unit.children)
            for c in u_touched:
                if c not in touched:
                    touched.append(c)
            fr.steer_log.append({"round": it, "parent_cell": cell["cid"], "steer_target": tgt_cid,
                                 "directive": (directive[:160] if directive else None),
                                 "delivered": bool(directive) and fr.steer_ok,
                                 "n_cands": len(unit.cands), "n_evaluated": len(round_fits),
                                 "child_fits": round_fits,
                                 "touched_cells": [c["cid"] for c in u_touched]})
            if prev is not None:                         # discovery = cell was empty before the gen
                for child in unit.children:
                    c, adm = _arch._admit_verdict(fr.cells, child, ctx.cfg)
                    if not adm:
                        rej += 1
                    elif prev.get(id(c)):
                        imp += 1
                    else:
                        dis += 1
        return touched, imp, dis, rej

    # ── Coverage-gated initialization, on rung 0 ──────────────────────────────────────────────────
    cover_tel = None
    if random_init > 0:
        view = _arch.frontier_cover_view(fr, k0=K0, random_init=random_init,
                                   parent_select=parent_select, seed=seed, land_fn=_land)
        cover_tel = _cov.coverage_init(
            ctx, view, random_init=random_init, batch=ri_batch,
            screen_on=random_init_screen, strict=random_init_strict,
            max_proposals=ri_max_prop, max_localizations=ri_max_loc, stall=ri_stall,
            max_evals=max_evals, phase_max_evals=ri_max_evals, safety=ri_safety,
            nonuniformity=ri_nonunif,
            n_workers=n_workers)
        cover_tel["level"] = 0
        cover_tel["level_note"] = ("coverage is denominated in rung 0 (the COARSEST), the rung live "
                                   "during the phase; `advance` re-places the whole archive onto a "
                                   "different partition, so this FRACTION does not carry to a "
                                   "finer rung")
        n_perr += cover_tel["propose_errors"]; n_eerr += cover_tel["eval_errors"]
        if not cover_tel["covered"] and random_init_strict:
            raise _cov._coverage_raise(cover_tel, "mapelites_hier")
    m_evals = ctx._evals
    occ_after_init = sum(1 for c in fr.cells if c["mem"])
    _check_M_against_N(ctx, fr.tag, max_evals, m_evals)
    g_ran, trims, bound_by = 0, [], "rounds"
    # Generations that asked for more parents than there were occupied cells (see `select_batch`).
    under_full = {}
    it = _resume_round                                  # 0 fresh; last completed round on resume
    while it < rounds:
        it += 1
        if ctx.should_stop and ctx.should_stop():
            bound_by = "should_stop"
            break
        if os.environ.get("ADAPTIVE_SAT_BEFORE_ROUND", "1") != "0":
            _bv0 = max((nd.fit.primary for nd in fr.allnodes), default=0.0)
            if _bv0 >= 1.0:
                ctx.log(f"[hier] SATURATED before round {it} (best={_bv0:.3f}) — early-stopping; "
                        f"{rounds - it + 1} of {rounds} round(s) skipped.")
                bound_by = "saturated_preround"
                break
        b_it, trimmed = _budget_slots(ctx, max_evals, batch_size, C)
        if b_it <= 0:
            bound_by = "max_evals"
            ctx.log(f"[hier] max_evals={max_evals} reached at {ctx._evals} evaluations — stopping "
                    f"before generation {it} of {rounds} (no full child fits)")
            break
        if trimmed:
            trims.append({"gen": it, "P": batch_size, "P_trimmed": b_it})
            ctx.log(f"[hier] generation {it}: TRIMMING P {batch_size} -> {b_it} so the run lands "
                    f"exactly on max_evals={max_evals} ({ctx._evals} spent)")
        t0 = _time.time()
        # (0) Advance if due, before selection, so generation level_start[l] itself runs at rung l.
        active, adv = _arch.advance(fr, it, ladder, program, active)
        if adv is not None:
            advances.append(adv)
            activated_at[active] = it
            skipped_levels.update(adv["skipped"])
        prev = {id(c): bool(c["mem"]) for c in fr.cells} if _batched else None
        # (1)-(4) as in `_mapelites_adaptive`, on the live map: select, steer, breed, admit.
        cells_sel = _arch.select_cells(fr, it, b_it, rule=parent_select, rng=_arch.gen_rng(seed, it),
                                 log=ctx.log, tally=under_full, tag="hier")
        cell = cells_sel[0]                              # the trace records the first parent cell
        parents = [_arch.elite(fr, c) for c in cells_sel]
        steers = [_arch._steer(fr, c) for c in cells_sel]      # alias: `steer` is a parameter
        units = _arch.breed_batch(ctx, parents, directives=[d for d, _ in steers], n_workers=n_workers,
                            gen=it, tag="hier", n_children=C)
        g_ran = it
        touched, imp, dis, rej = _land(cells_sel, steers, units, it, it, prev)
        # (5) No refine or retreat: §9.2 has no data-driven split and never coarsens.
        if _batched:
            n_perr += sum(1 for u in units if u.propose_error)
            n_eerr += sum(u.eval_errors for u in units)
            gens.append(_arch.batch_gen_row(it, [c["cid"] for c in cells_sel], units, improvements=imp,
                                      discoveries=dis, rejected=rej, archive=len(fr.cells),
                                      occupied=sum(1 for c in fr.cells if c["mem"]),
                                      total=len(fr.cells),
                                      elites=[_arch.elite(fr, c) for c in fr.cells if c["mem"]],
                                      wall_s=_time.time() - t0))
        best_val = max((n.fit.primary for n in fr.allnodes), default=0.0)
        fr.trace.append({"round": it, "level": active, "parent_cell": cell["cid"],
                         "touched_cells": [c["cid"] for c in touched], "n_cells": len(fr.cells),
                         "occupied": sum(1 for c in fr.cells if c["mem"]),
                         "depths": sorted(c["depth"] for c in fr.cells if c["mem"]),
                         "best": round(best_val, 3), "cells": _arch.snap(fr)})
        ctx.log(f"[hier] round {it}: L{active} archive={len(fr.cells)} cells (occupied "
                f"{sum(1 for c in fr.cells if c['mem'])}) best={best_val:.3f}")
        _write_partial_frontier(fr, ctx, it=it, k0=len(centroid_levels[0]), k_split=0, sigma=0.0,
                                depth_cap=len(centroid_levels) - 1, patience=0, steer=steer,
                                split_target="hier", m_evals=m_evals, occ_after_init=occ_after_init,
                                cover_tel=cover_tel, K0=K0, max_evals=max_evals, rounds=rounds,
                                batch_size=batch_size, C=C, trims=trims)
        if best_val >= 1.0:
            ctx.log(f"[hier] round {it}: SATURATED (best={best_val:.3f}) — early-stopping this "
                    f"arm; {rounds - it} of {rounds} rounds skipped.")
            break

    out = _out(fr.allnodes, ctx.cfg, ctx)
    unreached = sorted(lvl for lvl in range(1, L)
                       if lvl not in activated_at and lvl not in skipped_levels)
    out["hier"] = {
        "levels": L,
        "k_per_level": [len(z) for z in ladder],
        "program": list(program),            # level_start: the resolution-change program
        "program_source": ("caller-supplied level_start" if _prog_given
                           else f"default even split of {rounds} rounds across {L} levels"),
        "activated_at": dict(activated_at),  # level -> first generation run at it (0 -> 0)
        "advances": advances,
        "skipped_levels": sorted(skipped_levels),     # rungs a catch-up jumped over, never built
        "unreached_levels": unreached,                # rungs whose start fell past the budget
        "final_level": active,
        "steer": steer, "block_split": fr.block_split,
        "open_splits": len(fr.open_splits),  # always 0: hier never splits
        "final_cells": len(fr.cells),
        "final_occupied": sum(1 for c in fr.cells if c["mem"]),
        "final_depths": sorted(c["depth"] for c in fr.cells if c["mem"]),
        # Always False here; computed rather than hard-coded so that a regression would show.
        "mixed_resolution": len(set(c["depth"] for c in fr.cells if c["mem"])) > 1,
        "final_cell_table": [{"cid": c["cid"], "depth": c["depth"], "n_mem": len(c["mem"]),
                              "elite": (round(_arch.elite(fr, c).fit.primary, 3) if c["mem"] else None)}
                             for c in sorted(fr.cells, key=lambda x: x["cid"])],
        "steer_log": fr.steer_log, "trace": fr.trace,
        "citation": "Mouret & Clune 2015, arXiv:1504.04909 §9.2 (params §9.3)",
        # N = M + G·P·C: the initialization and budget reports
        "init": _init_out(anchors=len(fr.anchors), anchor_evals=len(fr.anchors),
                          pre_seed_offered=len(fr.pre_seed),
                          pre_seed_placed=fr.n_pre_seeded,
                          pre_seed_evals=len(fr.pre_seed),
                          cover=cover_tel, k0=K0,
                          occupied=occ_after_init),
        "budget": budget_out(ctx, max_evals=max_evals, rounds=rounds, batch_size=batch_size,
                             children_per_parent=C, m_evals=m_evals, generations_ran=g_ran,
                             trims=trims, bound_by=bound_by),
    }
    out["hier"]["audit"] = audit_out(ctx, fr.allnodes, steer=steer, steer_ok=fr.steer_ok)
    if _batched:                                         # per-generation batch telemetry
        out["hier"]["batch"] = _arch.batch_out(ctx, batch_size=batch_size, n_workers=n_workers,
                                         parent_select=parent_select, seed=seed,
                                         gens=gens, propose_errors=n_perr,
                                         eval_errors=n_eerr, under_full=under_full)
    return out


def _mapelites_fixed(ctx: Ctx, *, rounds=12, seed_genomes=None, phi_fn=None, init_centroids=None,
                     block_split=None, steer=True, resume_from=None,
                     batch_size=1, n_workers=1, parent_select="roundrobin", seed=0,
                     children_per_parent=2, max_evals=None,
                     random_init=0.5, random_init_batch=1, random_init_max_proposals=0,
                     random_init_max_localizations=0, random_init_max_evals=0,
                     random_init_stall=0,
                     random_init_safety=0,
                     random_init_nonuniformity=0,
                     random_init_strict=True, random_init_screen=True):
    """Fixed-resolution CVT-MAP-Elites (Vassiliades, Chatzilygeroudis & Mouret 2018)."""
    if phi_fn is None or init_centroids is None:
        raise ValueError("mapelites_fixed requires phi_fn and init_centroids")
    import time as _time
    batch_size = _arch._batch_validate(parent_select, batch_size)
    C = max(1, int(children_per_parent))
    max_evals = None if max_evals is None else max(0, int(max_evals))
    (random_init, ri_batch, ri_max_prop, ri_max_loc,
     ri_stall, ri_max_evals, ri_safety, ri_nonunif) = _cov._coverage_validate(
        random_init, random_init_batch, random_init_max_proposals,
        random_init_max_localizations, random_init_stall, random_init_max_evals,
        random_init_safety, random_init_nonuniformity)
    _batched = (batch_size > 1 or n_workers > 1 or parent_select != "roundrobin")
    # ── Resume from a snapshot, or seed a new archive ─────────────────────────────────────────────
    # sigma, k_split, patience and split_target only populate the Frontier; depth_cap=0 forbids
    # every split.
    _resume_round = 0
    if resume_from:
        import json as _json
        _R = _json.loads(Path(resume_from).read_text())["frontier"]
        fr, _resume_round = restore_frontier(
            ctx, _R, phi_fn=phi_fn, block_split=block_split, sigma=0.05, k_split=2,
            depth_cap=0, patience=1, steer=steer, split_target="variance", tag="fixed",
            log=ctx.log)
        _errs = verify_restored(fr, _R)
        if _errs:
            raise RuntimeError(f"resume: restored archive does not match the snapshot: {_errs[:3]}")
        seed_ctx_counters(ctx, _ledger_rows(Path(ctx.adapter.work_root)),
                                n_evals_fallback=int((_R.get("budget") or {}).get("N_realized", 0)))
        ctx.log(f"[fixed] RESUMED from {resume_from}: archive verified field-for-field; "
                f"continuing at round {_resume_round + 1} of {rounds}, ctx._round={ctx._round} "
                f"ctx._evals={ctx._evals}")
        random_init = 0.0                                # coverage-init already happened
    else:
        fr = _arch._seed(ctx, phi_fn=phi_fn, init_centroids=init_centroids, block_split=block_split,
                   seed_genomes=seed_genomes, sigma=0.05, k_split=2, depth_cap=0,
                   patience=1, steer=steer, split_target="variance", tag="fixed")
    K0 = int(_R["k0"]) if resume_from else len(fr.cells)   # == len(init_centroids), throughout
    gens, n_perr, n_eerr = [], 0, 0

    def _land(cells_sel, steers, units, it, depth, prev):
        """Same as `_mapelites_adaptive`'s `_land`."""
        touched, imp, dis, rej = [], 0, 0, 0
        for cell, (directive, tgt_cid), unit in zip(cells_sel, steers, units):
            parent, round_fits = unit.parent, []
            for child in unit.children:
                child.depth = depth
                child.parents = (parent.idx,)
                fr.allnodes.append(child)
                round_fits.append(round(child.fit.primary, 3))
            fr._render = {"round": it, "parent": parent, "cell": cell, "tgt": tgt_cid}
            u_touched = _arch.admit(fr, unit.children)
            for c in u_touched:
                if c not in touched:
                    touched.append(c)
            fr.steer_log.append({"round": it, "parent_cell": cell["cid"], "steer_target": tgt_cid,
                                 "directive": (directive[:160] if directive else None),
                                 "delivered": bool(directive) and fr.steer_ok,
                                 "n_cands": len(unit.cands), "n_evaluated": len(round_fits),
                                 "child_fits": round_fits,
                                 "touched_cells": [c["cid"] for c in u_touched]})
            if prev is not None:                         # discovery = cell was empty before the gen
                for child in unit.children:
                    c, adm = _arch._admit_verdict(fr.cells, child, ctx.cfg)
                    if not adm:
                        rej += 1
                    elif prev.get(id(c)):
                        imp += 1
                    else:
                        dis += 1
        return touched, imp, dis, rej

    # ── Coverage-gated initialization ─────────────────────────────────────────────────────────────
    cover_tel = None
    if random_init > 0:
        view = _arch.frontier_cover_view(fr, k0=K0, random_init=random_init,
                                   parent_select=parent_select, seed=seed, land_fn=_land)
        cover_tel = _cov.coverage_init(
            ctx, view, random_init=random_init, batch=ri_batch,
            screen_on=random_init_screen, strict=random_init_strict,
            max_proposals=ri_max_prop, max_localizations=ri_max_loc, stall=ri_stall,
            max_evals=max_evals, phase_max_evals=ri_max_evals, safety=ri_safety,
            nonuniformity=ri_nonunif,
            n_workers=n_workers)
        n_perr += cover_tel["propose_errors"]; n_eerr += cover_tel["eval_errors"]
        if not cover_tel["covered"] and random_init_strict:
            raise _cov._coverage_raise(cover_tel, "mapelites_fixed")
    m_evals = ctx._evals
    occ_after_init = sum(1 for c in fr.cells if c["mem"])
    _check_M_against_N(ctx, fr.tag, max_evals, m_evals)
    g_ran, trims, bound_by = 0, [], "rounds"
    # Generations that asked for more parents than there were occupied cells (see `select_batch`).
    under_full = {}
    it = _resume_round                                  # 0 fresh; last completed round on resume
    while it < rounds:
        it += 1
        if ctx.should_stop and ctx.should_stop():
            bound_by = "should_stop"
            break
        if os.environ.get("ADAPTIVE_SAT_BEFORE_ROUND", "1") != "0":
            _bv0 = max((nd.fit.primary for nd in fr.allnodes), default=0.0)
            if _bv0 >= 1.0:
                ctx.log(f"[fixed] SATURATED before round {it} (best={_bv0:.3f}) — early-stopping; "
                        f"{rounds - it + 1} of {rounds} round(s) skipped.")
                bound_by = "saturated_preround"
                break
        b_it, trimmed = _budget_slots(ctx, max_evals, batch_size, C)
        if b_it <= 0:
            bound_by = "max_evals"
            ctx.log(f"[fixed] max_evals={max_evals} reached at {ctx._evals} evaluations — "
                    f"stopping before generation {it} of {rounds} (no full child fits)")
            break
        if trimmed:
            trims.append({"gen": it, "P": batch_size, "P_trimmed": b_it})
            ctx.log(f"[fixed] generation {it}: TRIMMING P {batch_size} -> {b_it} so the run "
                    f"lands exactly on max_evals={max_evals} ({ctx._evals} spent)")
        t0 = _time.time()
        prev = {id(c): bool(c["mem"]) for c in fr.cells} if _batched else None
        # (1)-(4) as in `_mapelites_adaptive`: select, steer, breed, admit. No resolution control.
        cells_sel = _arch.select_cells(fr, it, b_it, rule=parent_select, rng=_arch.gen_rng(seed, it),
                                 log=ctx.log, tally=under_full, tag="fixed")
        cell = cells_sel[0]                              # the trace records the first parent cell
        parents = [_arch.elite(fr, c) for c in cells_sel]
        steers = [_arch._steer(fr, c) for c in cells_sel]      # alias: `steer` is a parameter
        units = _arch.breed_batch(ctx, parents, directives=[d for d, _ in steers], n_workers=n_workers,
                            gen=it, tag="fixed", n_children=C)
        g_ran = it
        touched, imp, dis, rej = _land(cells_sel, steers, units, it, it, prev)
        if _batched:
            n_perr += sum(1 for u in units if u.propose_error)
            n_eerr += sum(u.eval_errors for u in units)
            gens.append(_arch.batch_gen_row(it, [c["cid"] for c in cells_sel], units, improvements=imp,
                                      discoveries=dis, rejected=rej, archive=len(fr.cells),
                                      occupied=sum(1 for c in fr.cells if c["mem"]),
                                      total=len(fr.cells),
                                      elites=[_arch.elite(fr, c) for c in fr.cells if c["mem"]],
                                      wall_s=_time.time() - t0))
        best_val = max((n.fit.primary for n in fr.allnodes), default=0.0)
        fr.trace.append({"round": it, "parent_cell": cell["cid"],
                         "touched_cells": [c["cid"] for c in touched], "n_cells": len(fr.cells),
                         "occupied": sum(1 for c in fr.cells if c["mem"]),
                         "depths": sorted(c["depth"] for c in fr.cells if c["mem"]),
                         "best": round(best_val, 3), "cells": _arch.snap(fr)})
        # The adaptive driver's round line with `fr.tag`, so logs match after substituting the tag.
        ctx.log(f"[{fr.tag}] round {it}: {fr.tag}={len(fr.cells)} cells (occupied "
                f"{sum(1 for c in fr.cells if c['mem'])}, depths {sorted(c['depth'] for c in fr.cells if c['mem'])}) "
                f"best={best_val:.3f}")
        _write_partial_frontier(fr, ctx, it=it, k0=len(init_centroids), k_split=0, sigma=0.0,
                                depth_cap=0, patience=0, steer=steer, split_target="fixed",
                                m_evals=m_evals, occ_after_init=occ_after_init, cover_tel=cover_tel,
                                K0=K0, max_evals=max_evals, rounds=rounds, batch_size=batch_size,
                                C=C, trims=trims)
        if best_val >= 1.0:
            ctx.log(f"[{fr.tag}] round {it}: SATURATED (best={best_val:.3f}) — early-stopping this "
                    f"arm; {rounds - it} of {rounds} rounds skipped.")
            break

    # The fixed-resolution invariant holds by construction; checking it guards against regressions.
    if len(fr.cells) != K0 or fr.splits or fr.retreats or fr.open_splits or any(
            c["depth"] for c in fr.cells):
        raise RuntimeError(
            f"mapelites_fixed violated its fixed-resolution invariant: cells={len(fr.cells)} "
            f"(k0={K0}), splits={len(fr.splits)}, retreats={len(fr.retreats)}, "
            f"open_splits={len(fr.open_splits)}, depths={sorted(set(c['depth'] for c in fr.cells))}")
    out = _out(fr.allnodes, ctx.cfg, ctx)
    out["fixed"] = {
        "k0": len(init_centroids),
        "resolution": "fixed",
        "final_cells": len(fr.cells),        # == k0 (checked above)
        "final_occupied": sum(1 for c in fr.cells if c["mem"]),
        "final_depths": sorted(c["depth"] for c in fr.cells if c["mem"]),
        # Always False here; computed rather than hard-coded so that a regression would show.
        "mixed_resolution": len(set(c["depth"] for c in fr.cells if c["mem"])) > 1,
        "n_splits": 0, "n_retreats": 0,
        "steer": steer, "block_split": fr.block_split,
        "final_cell_table": [{"cid": c["cid"], "depth": c["depth"], "n_mem": len(c["mem"]),
                              "elite": (round(_arch.elite(fr, c).fit.primary, 3) if c["mem"] else None)}
                             for c in sorted(fr.cells, key=lambda x: x["cid"])],
        "steer_log": fr.steer_log, "trace": fr.trace,
        "citation": ("Vassiliades, Chatzilygeroudis & Mouret 2018, IEEE TEVC 22(4):623-630 "
                     "(arXiv:1610.05729) — centroids k-means-fitted to a harness CORPUS here, not "
                     "to uniform samples of the feature space (see the README)"),
        # N = M + G·P·C: the initialization and budget reports
        "init": _init_out(anchors=len(fr.anchors), anchor_evals=len(fr.anchors),
                          pre_seed_offered=len(fr.pre_seed),
                          pre_seed_placed=fr.n_pre_seeded,
                          pre_seed_evals=len(fr.pre_seed),
                          cover=cover_tel, k0=K0,
                          occupied=occ_after_init),
        "budget": budget_out(ctx, max_evals=max_evals, rounds=rounds, batch_size=batch_size,
                             children_per_parent=C, m_evals=m_evals, generations_ran=g_ran,
                             trims=trims, bound_by=bound_by),
    }
    out["fixed"]["audit"] = audit_out(ctx, fr.allnodes, steer=steer, steer_ok=fr.steer_ok)
    if _batched:                                         # per-generation batch telemetry
        out["fixed"]["batch"] = _arch.batch_out(ctx, batch_size=batch_size, n_workers=n_workers,
                                          parent_select=parent_select, seed=seed,
                                          gens=gens, propose_errors=n_perr,
                                          eval_errors=n_eerr, under_full=under_full)
    return out


class MapElitesFixed(SearchMode):
    """The `mapelites_fixed` search mode."""

    name = "mapelites_fixed"

    def run(self, ctx, **knobs):
        return _mapelites_fixed(ctx, **knobs)


class MapElitesHier(SearchMode):
    """The `mapelites_hier` search mode."""

    name = "mapelites_hier"

    def run(self, ctx, **knobs):
        return _mapelites_hier(ctx, **knobs)


class MapElitesAdaptive(SearchMode):
    """The `mapelites_adaptive` search mode (the main method)."""

    name = "mapelites_adaptive"

    def run(self, ctx, **knobs):
        return _mapelites_adaptive(ctx, **knobs)
