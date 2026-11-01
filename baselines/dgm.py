"""Baseline: Darwin Gödel Machine, upstream's selection and archive code over this adapter."""
from __future__ import annotations

import json
import logging
import random

from pathlib import Path

from . import common as C
from .upstream import load_dgm, pins


PROMPT_INSTANCE = """\
Improve the harness `{parent}`.

## The failure this attempt is aimed at
Instance `{entry}` scored {score:.3f}. It is one of {n_unresolved} unresolved instance(s).

## How the parent scored across the split
{scores}

## Diagnostics reported by the evaluator
{dense}

Diagnose why `{entry}` failed and make the smallest change to the harness that plausibly fixes that
class of failure. Do not regress the instances that already score well.
"""

PROMPT_MODE = {
    "solve_empty_patches": """\
Improve the harness `{parent}`.

## The failure this attempt is aimed at
{n_empty} of {n_total} instance(s) scored ZERO — on those the harness produced no usable output at
all, rather than a wrong one: {empty}

## How the parent scored across the split
{scores}

## Diagnostics reported by the evaluator
{dense}

Find why the harness terminates without producing anything on those instances and fix that path.
An output that is wrong but well-formed is strictly better than no output.
""",
    "solve_stochasticity": """\
Improve the harness `{parent}`.

## The failure this attempt is aimed at
Run-to-run variance: the harness does not reliably reproduce its own results across repeats.

## How the parent scored across the split
{scores}

## Diagnostics reported by the evaluator
{dense}

Find the sources of nondeterminism the harness itself controls — unseeded sampling, unordered
iteration, retry paths that change behaviour, decisions taken on a single model call that could be
checked — and make its behaviour more repeatable without lowering its ceiling.
""",
    "solve_contextlength": """\
Improve the harness `{parent}`.

## The failure this attempt is aimed at
Context-length blowup: the evaluator's logs show the harness sending inputs the backbone rejects as
too long, and the harness does not recover when that happens.

## How the parent scored across the split
{scores}

## Diagnostics reported by the evaluator
{dense}

Reduce what the harness keeps and re-sends — summarise or window its history, elide large payloads
from all but the most recent message — and make the over-length path recoverable rather than fatal.
""",
}


def run(adapter, *, out_dir, ledger: C.Ledger, generations: int = 6, selfimprove_size: int = 2,
        k: int = 1, method: str = "score_child_prop", update_archive: str = "keep_all",
        noise_leeway: float = 0.1, resolve_thresh: float = 1.0, small_n: int = 3,
        test_more_threshold: float = 0.4, shallow_eval: bool = False, run_baseline=None,
        seed: int = 0, holdout: bool = True, log=print, **_) -> dict:
    """Run DGM's loop with upstream's selection code over this repo's testbed."""
    D = load_dgm()
    rl = C.RunLog(out_dir, "dgm", {
        "engine": "upstream DGM_outer.choose_selfimproves / filter_compiled / update_archive",
        "generations": generations, "selfimprove_size": selfimprove_size, "k": k, "method": method,
        "update_archive": update_archive, "noise_leeway": noise_leeway,
        "resolve_thresh": resolve_thresh, "small_n": small_n,
        "test_more_threshold": test_more_threshold, "shallow_eval": shallow_eval,
        "run_baseline": run_baseline, "seed": seed, "upstream": pins()["dgm"],
        "replaced": ["self_improve_step.self_improve (Docker + SWE-bench harness)"],
        "derived": ["resolved/unresolved/empty from per-instance scores",
                    "directive text (upstream's prompts describe DGM's own coding_agent.py)"]})

    dgm_out = Path(out_dir) / "output_dgm"          # upstream's on-disk layout lives here
    dgm_out.mkdir(parents=True, exist_ok=True)
    logger = _logger(Path(out_dir) / "dgm_outer.log")
    random.seed(seed)                                # upstream samples from the `random` module

    ids = C.task_ids(adapter, "evolve")
    small = ids[:max(1, min(small_n, len(ids)))] if ids else []
    pool: list[C.Cand] = []
    by_commit: dict[str, C.Cand] = {}

    def commit_of(cand: C.Cand) -> str:
        return "initial" if cand.idx == 0 else cand.label

    # ── the initial version ──────────────────────────────────────────────────
    anchor = adapter.anchor()
    log(f"[dgm] upstream selection: {D.__file__}")
    log(f"[dgm] evaluating the initial version {anchor.label or anchor.source_dir}")
    fit, _ = C.evaluate_on(adapter, anchor, rid=0, ledger=ledger)
    root = C.Cand(idx=0, genome=anchor, fit=fit, scores=C.per_instance(fit), gen=0,
                  meta={"role": "initial", "stage": "full"})
    pool.append(root)
    rl.log(root, ledger)
    if not root.valid:
        log("[dgm] initial version produced no valid score — no archive to evolve")
        return C.finalize(adapter, rl, ledger, pool, None, holdout=holdout, logf=log)
    by_commit["initial"] = root
    _write_node(dgm_out, "initial", None, root, resolve_thresh)
    archive = ["initial"]
    log(f"[dgm] initial primary={root.primary:.4f} "
        f"resolved={len(_resolved(root, resolve_thresh))} "
        f"unresolved={len(_unresolved(root, resolve_thresh))} empty={len(_empty(root))}")

    try:
        for gen in range(generations):
            ledger.check()
            entries = _choose(D, str(dgm_out), archive, selfimprove_size, method, run_baseline,
                              by_commit, resolve_thresh, log)
            log(f"[dgm] generation {gen}: archive={len(archive)} attempts={entries}")
            if not entries:
                continue

            new_commits: list[str] = []
            for j, (parent_commit, entry) in enumerate(entries):
                ledger.check()
                parent = by_commit.get(parent_commit)
                if parent is None:
                    continue
                # One rid per attempt: the proposer keys a candidate directory on its rid, so two
                # attempts of one generation must not share one.
                rid = (gen + 1) * 10 + j
                if run_baseline == "no_selfimprove":
                    children = [parent.genome]       # upstream's ablation: re-evaluate unchanged
                else:
                    directive = _directive(parent, entry, resolve_thresh)
                    children = C.propose(adapter, parent.genome, rid=rid, k=k, directive=directive)
                    ledger.proposals += 1
                if not children:
                    log(f"[dgm]   {parent_commit}/{entry}: proposer returned nothing")
                    continue

                for child in children:
                    ledger.check()
                    if not C.validate(adapter, child):
                        log(f"[dgm]   {child.label or child.source_dir}: did not compile")
                        continue
                    cand = _staged_eval(adapter, child, rid, gen + 1, parent, entry, small,
                                        test_more_threshold, shallow_eval, ledger, pool, rl, log)
                    commit = commit_of(cand)
                    by_commit[commit] = cand
                    _write_node(dgm_out, commit, parent_commit, cand, resolve_thresh)
                    new_commits.append(commit)

            # ── upstream: which children may enter, and how the archive grows ──
            compiled = D.filter_compiled(new_commits, str(dgm_out),
                                         num_swe_issues=[len(small) or 1], logger=logger)
            archive = D.update_archive(str(dgm_out), archive, compiled, method=update_archive,
                                       noise_leeway=noise_leeway)
            rl.append("dgm_metadata.jsonl", {"generation": gen, "selfimprove_entries": entries,
                                             "children": new_commits,
                                             "children_compiled": compiled, "archive": archive})
            log(f"[dgm]   archive now {len(archive)}: "
                f"{[f'{c}@{by_commit[c].primary:.3f}' for c in archive if c in by_commit]}")
            # A perfect 1.0 cannot be improved: stop, as every arm does.
            _b = C.pick_best(pool)
            if _b is not None and (_b.primary or 0) >= 1.0:
                log(f"[dgm] SATURATED at generation {gen} (best={_b.primary:.3f}) — stopping early.")
                break
    except C.BudgetExhausted as e:
        log(f"[dgm] budget exhausted: {e}")

    best = C.pick_best([by_commit[c] for c in archive if c in by_commit]) or C.pick_best(pool)
    # A champion screened only on the small subset is re-scored on the full split (not charged),
    # so its primary is comparable to the anchor's.
    if best is not None and (best.meta or {}).get("stage") != "full":
        log(f"[dgm] re-scoring champion {best.label} on the full split "
            f"(was stage={(best.meta or {}).get('stage')}) for a comparable primary")
        full, _ = C.evaluate_on(adapter, best.genome, rid=ledger.metric_calls, ledger=None)
        if full is not None:
            best = C.Cand(idx=best.idx, genome=best.genome, fit=full, scores=C.per_instance(full),
                          gen=getattr(best, "gen", 0),
                          meta={**(best.meta or {}), "stage": "full", "rescored_full": True})
        best = C.pick_best([best, root]) or best      # anchor is full-scored; keep the better on full
    return C.finalize(adapter, rl, ledger, pool, best,
                      extra={"archive": archive, "archive_size": len(archive),
                             "initial_primary": root.primary,
                             "dgm_output_dir": str(dgm_out)},
                      holdout=holdout, logf=log)


# ── upstream's chooser, with its one reachable crash handled ─────────────────
def _choose(D, dgm_out, archive, size, method, run_baseline, by_commit, thresh, log):
    """Upstream `choose_selfimproves`, retried over eligible parents on its empty-list crash."""
    try:
        return [tuple(e) for e in D.choose_selfimproves(dgm_out, archive, size, method=method,
                                                        run_baseline=run_baseline, polyglot=False)]
    except IndexError:
        eligible = [c for c in archive
                    if c in by_commit and _unresolved(by_commit[c], thresh)]
        log(f"[dgm]   upstream choose_selfimproves hit its empty-unresolved case "
            f"(DGM_outer.py:143); retrying over {len(eligible)}/{len(archive)} parent(s) that have "
            f"unresolved instances")
        if not eligible:
            return []
        try:
            return [tuple(e) for e in D.choose_selfimproves(dgm_out, eligible, size, method=method,
                                                            run_baseline=run_baseline,
                                                            polyglot=False)]
        except IndexError:
            return []


# ── the on-disk layout upstream reads ───────────────────────────────────────
def _write_node(dgm_out: Path, commit: str, parent_commit, cand: C.Cand, thresh: float) -> None:
    """Write `metadata.json` and per-instance prediction logs in upstream's layout."""
    node = dgm_out / commit
    (node / "predictions" / "run0").mkdir(parents=True, exist_ok=True)
    resolved, unresolved, empty = (_resolved(cand, thresh), _unresolved(cand, thresh),
                                   _empty(cand))
    (node / "metadata.json").write_text(json.dumps({
        "run_id": commit,
        "parent_commit": parent_commit,
        "overall_performance": {
            "accuracy_score": float(cand.primary),
            "total_resolved_ids": resolved,
            "total_unresolved_ids": unresolved,
            "total_emptypatch_ids": empty,
            "total_submitted_instances": len(cand.scores),
        },
        "stage": cand.meta.get("stage"), "entry": cand.meta.get("entry"),
        "source_dir": str(cand.genome.source_dir),
    }, indent=2))
    dense = C._jsonable(dict(getattr(cand.fit, "dense", None) or {}))
    for inst, score in cand.scores.items():
        pred = node / "predictions" / "run0" / f"{inst}.md"
        pred.parent.mkdir(parents=True, exist_ok=True)   # inst ids may contain '/' (e.g. "maze/level_01")
        pred.write_text(
            f"# {commit} on {inst}\n\nscore: {score}\n\ndiagnostics: {json.dumps(dense)}\n"
            + (cand.meta.get("log") or ""))


def _staged_eval(adapter, child, rid, gen, parent, entry, small, threshold, shallow_only,
                 ledger, pool, rl, log) -> C.Cand:
    """Screen on the small subset, then evaluate the full split only above `threshold`."""
    idx = len(pool)
    fit, _ = C.evaluate_on(adapter, child, rid=rid, ids=(small or None), ledger=ledger)
    stage, scores = "small", C.per_instance(fit)
    primary = float(fit.primary) if fit is not None else 0.0
    log(f"[dgm]   {child.label}: small({len(small)}) primary={primary:.4f}")

    if not shallow_only and fit is not None and getattr(fit, "valid", False) and primary > threshold:
        ledger.check()
        full, _ = C.evaluate_on(adapter, child, rid=rid, ledger=ledger)
        if full is not None:
            fit, stage, scores = full, "full", C.per_instance(full)
            log(f"[dgm]   {child.label}: escalated -> full primary={float(fit.primary):.4f}")
    elif not shallow_only:
        log(f"[dgm]   {child.label}: below escalation threshold {threshold} — not escalated")

    cand = C.Cand(idx=idx, genome=child, fit=fit, scores=scores, parent=parent.idx, gen=gen,
                  meta={"entry": entry, "stage": stage})
    pool.append(cand)
    rl.log(cand, ledger)
    return cand


# ── the resolved / unresolved / empty split ─────────────────────────────────
def _resolved(c: C.Cand, thresh: float) -> list[str]:
    return sorted([k for k, v in c.scores.items() if v >= thresh])


def _unresolved(c: C.Cand, thresh: float) -> list[str]:
    return sorted([k for k, v in c.scores.items() if 0.0 < v < thresh])


def _empty(c: C.Cand) -> list[str]:
    return sorted([k for k, v in c.scores.items() if v <= 0.0])


def _directive(parent: C.Cand, entry: str, thresh: float) -> str:
    scores = C.render_scores(parent)
    dense = "\n".join(f"  {k:<24} {v}"
                      for k, v in sorted((dict(getattr(parent.fit, "dense", None) or {})).items())
                      if isinstance(v, (int, float, str, bool))) or "  (none reported)"
    if entry in PROMPT_MODE:
        empty = _empty(parent)
        return PROMPT_MODE[entry].format(parent=parent.label, scores=scores, dense=dense,
                                         n_empty=len(empty), n_total=len(parent.scores),
                                         empty=", ".join(empty[:12]) or "(none)")
    return PROMPT_INSTANCE.format(parent=parent.label, entry=entry,
                                  score=parent.scores.get(entry, 0.0),
                                  n_unresolved=len(_unresolved(parent, thresh)),
                                  scores=scores, dense=dense)


def _logger(path: Path):
    """A real logging.Logger, because upstream's `is_compiled_self_improve` calls `logger.info`."""
    lg = logging.getLogger(f"dgm.{path.parent.name}")
    if not lg.handlers:
        path.parent.mkdir(parents=True, exist_ok=True)
        h = logging.FileHandler(path)
        h.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        lg.addHandler(h)
        lg.setLevel(logging.INFO)
        lg.propagate = False
    return lg
