"""Baseline: Meta-Harness, running upstream's frontier and history functions over this adapter."""
from __future__ import annotations

import json

from pathlib import Path

from . import common as C
from .upstream import load_meta_harness, pins


PROMPT = """\
Run iteration {it} of the harness evolution loop.
Start from `{anchor}` as the parent.

## Eval split: {n_tasks} task(s), evaluated in full for every candidate you propose

## Frontier — best result per task, and the best agent overall
{frontier}

## History — every candidate evaluated so far
{history}

The two files above are also on disk, in upstream's own format:
- `{summary_path}` — past results
- `{frontier_path}` — the frontier

Propose {k} candidate harness(es). For each, state the hypothesis it tests and what you changed.
Do not repeat a change the history above already shows to be neutral or harmful.
"""


def run(adapter, *, out_dir, ledger: C.Ledger, iterations: int = 6, k: int = 2,
        parent_mode: str = "anchor", history_window: int = 20, holdout: bool = True, log=print,
        **_) -> dict:
    """Run the meta-harness loop with upstream's frontier/history functions over this adapter."""
    MH = load_meta_harness()
    logs = Path(out_dir)
    logs.mkdir(parents=True, exist_ok=True)
    # upstream's functions write through these module globals; point them at this run
    MH.LOGS_DIR = logs
    MH.FRONTIER_VAL = logs / "frontier_val.json"
    MH.EVOLUTION_SUMMARY = logs / "evolution_summary.jsonl"
    MH.PENDING_EVAL = logs / "pending_eval.json"

    rl = C.RunLog(out_dir, "meta-harness", {
        "engine": "upstream meta_harness.update_frontier / update_evolution_summary",
        "iterations": iterations, "k": k, "parent_mode": parent_mode,
        "history_window": history_window, "upstream": pins()["meta-harness"],
        "replaced": ["harbor_run + parse_job_results (Harbor / Terminal-Bench 2)",
                     "propose_claude (TB2-specific proposer skill)",
                     "compute_pass_rates (binarises trials at reward>0)"]})
    ids = C.task_ids(adapter, "evolve")
    pool: list[C.Cand] = []
    next_idx = 0

    def evaluate_and_record(genome, it, parent, role) -> C.Cand:
        """Evaluate, then run upstream's two state updates on the result."""
        nonlocal next_idx
        fit, _ = C.evaluate_on(adapter, genome, rid=it, ledger=ledger)
        cand = C.Cand(idx=next_idx, genome=genome, fit=fit, scores=C.per_instance(fit),
                      parent=parent, gen=it, meta={"role": role})
        next_idx += 1
        pool.append(cand)
        rl.log(cand, ledger)

        if cand.valid:                      # invalid = missing data, not a score of 0 on every task
            results = {cand.label: (dict(cand.scores), float(cand.primary))}
            MH.update_frontier(results)                                   # UPSTREAM
            MH.update_evolution_summary(                                  # UPSTREAM
                it,
                [{"name": cand.label, "import_path": str(genome.source_dir),
                  "hypothesis": "", "changes": ""}],
                results)
        return cand

    # ── Phase 0: the baseline agent, evaluated, seeding the frontier ─────────
    anchor = adapter.anchor()
    log(f"[meta-harness] upstream loop functions: {MH.__file__}")
    log(f"[meta-harness] phase 0: evaluating anchor {anchor.label or anchor.source_dir}")
    root = evaluate_and_record(anchor, 0, None, "anchor")
    log(f"[meta-harness] anchor primary={root.primary:.4f} valid={root.valid} "
        f"tasks={len(root.scores)}")

    # ── Phase 1..N ───────────────────────────────────────────────────────────
    try:
        for i in range(iterations):
            ledger.check()
            it = MH.count_iterations() + 1                                # UPSTREAM
            frontier = _frontier(MH)
            best = frontier.get("_best", {})
            parent = (C.pick_best(pool) or root) if parent_mode == "best" else root
            log(f"[meta-harness] iteration {it}  frontier={best.get('agent', 'none')} "
                f"@ {best.get('avg_pass_rate', 0):.4f}  parent={parent.label}")

            if MH.PENDING_EVAL.exists():                  # upstream clears this before proposing
                MH.PENDING_EVAL.unlink()
            directive = PROMPT.format(
                it=it, anchor=parent.label, n_tasks=len(ids), k=k,
                frontier=_render_frontier(frontier), history=_render_history(pool, history_window),
                summary_path=MH.EVOLUTION_SUMMARY, frontier_path=MH.FRONTIER_VAL)
            cands = C.propose(adapter, parent.genome, rid=it, k=k, directive=directive)
            ledger.proposals += 1
            if not cands:
                log(f"[meta-harness] iteration {it}: proposer returned no candidates")
                continue

            valid = [g for g in cands if C.validate(adapter, g)]
            log(f"[meta-harness] iteration {it}: proposed {len(cands)}, {len(valid)} validated")
            for g in valid:
                ledger.check()
                c = evaluate_and_record(g, it, parent.idx, "candidate")
                delta = c.primary - float(best.get("avg_pass_rate", 0) or 0)
                log(f"[meta-harness]   {c.label}: primary={c.primary:.4f} ({delta:+.4f}) "
                    f"valid={c.valid}")
            # A perfect 1.0 cannot be improved: stop, as every arm does.
            _b = C.pick_best(pool)
            if _b is not None and (_b.primary or 0) >= 1.0:
                log(f"[meta-harness] SATURATED at iteration {it} (best={_b.primary:.3f}) "
                    f"— stopping early.")
                break
    except C.BudgetExhausted as e:
        log(f"[meta-harness] budget exhausted: {e}")

    best = C.pick_best(pool)
    return C.finalize(adapter, rl, ledger, pool, best,
                      extra={"frontier": _frontier(MH), "anchor_primary": root.primary},
                      holdout=holdout, logf=log)


def _frontier(MH) -> dict:
    return json.loads(MH.FRONTIER_VAL.read_text()) if MH.FRONTIER_VAL.exists() else {}


def _render_frontier(frontier: dict) -> str:
    if not frontier:
        return "  (empty — nothing evaluated yet)"
    best = frontier.get("_best", {})
    lines = [f"  BEST OVERALL: {best.get('agent', 'none')} @ "
             f"{float(best.get('avg_pass_rate', 0) or 0):.4f}"]
    lines += [f"  {t:<44} {float(d.get('pass_rate', 0)):.3f}  ({d.get('best_agent')})"
              for t, d in sorted(frontier.items()) if t != "_best"]
    return "\n".join(lines)


def _render_history(pool: list[C.Cand], window: int) -> str:
    if not pool:
        return "  (empty)"
    rows, lines = pool[-window:], []
    for c in rows:
        lines.append(f"  it={c.gen:<3} {c.label:<28} primary={c.primary:.4f} "
                     f"{'ok' if c.valid else 'FAILED'}")
    if len(pool) > window:
        lines.insert(0, f"  ... {len(pool) - window} earlier candidate(s) elided ...")
    return "\n".join(lines)
