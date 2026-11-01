#!/usr/bin/env python
"""Run one baseline or ablation arm on a benchmark described by a config under examples/.

    python baselines/run_baseline.py --config examples/balrog/config.json --anchor <ANCHOR_DIR> \\
        --baseline meta-harness --budget 50

`--budget` caps candidate evaluations, the N of the method's own runs. Outputs land in
runs/<testbed>/<split>/<baseline>/: history.jsonl, summary.json and the upstream's own
artifacts. See readme.md.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

import run_benchmark as RB                                               # noqa: E402
from harness_map import main as MH                                       # noqa: E402
from baselines import BASELINES, Ledger, run_baseline                    # noqa: E402
from baselines.upstream import pins                                      # noqa: E402

UPSTREAM_OF = {"meta-harness": "meta-harness", "gepa": "gepa", "dgm": "dgm",
               "openevolve": "openevolve", "shinka": "shinkaevolve",
               "openevolve_cvt": "openevolve", "open_harness": "openevolve"}
BUNDLE_SUFFIXES = [".py", ".md"]                                   # the evolvable surface of a harness
BUNDLE_EXCLUDE = ["system_prompt.txt", "NOTES.md", "memory.md"]    # inert or regenerated files


def log(*m):
    print(*m, flush=True)


def arm_kwargs(a, cfg: dict, adapter, budget: int | None) -> dict:
    """The knobs each arm takes beyond the common ones (upstream's defaults where it has one)."""
    bench = cfg["benchmark"]
    eval_timeout = int(bench.get("timeout_s") or 7200)
    loop = a.iterations if a.iterations_set else (10 * budget if budget else 24)   # the budget binds
    if a.baseline == "meta-harness":
        return dict(iterations=loop, parent_mode=a.meta_harness_parent,
                    history_window=a.meta_harness_history, **({"k": a.k} if a.k_set else {}))
    if a.baseline == "dgm":
        return dict(generations=loop, selfimprove_size=a.dgm_selfimprove_size, method=a.dgm_method,
                    update_archive=a.dgm_archive, noise_leeway=a.dgm_noise_leeway,
                    resolve_thresh=a.dgm_resolve_thresh, small_n=a.dgm_small_n,
                    test_more_threshold=a.dgm_threshold, shallow_eval=a.dgm_shallow,
                    run_baseline=a.dgm_run_baseline or None, seed=a.seed,
                    **({"k": a.k} if a.k_set else {}))
    if a.baseline == "gepa":                 # no loop count: upstream's engine stops on the budget
        return dict(minibatch_size=a.gepa_minibatch, front=a.gepa_front, seed=a.seed,
                    perfect_score=a.gepa_perfect, skip_perfect_score=not a.gepa_no_skip_perfect,
                    use_merge=a.gepa_merge, proposer=a.gepa_proposer, proposer_model=a.gepa_model,
                    reflection_lm=a.gepa_reflection_lm, sandbox=not a.gepa_no_sandbox,
                    max_workers=a.gepa_workers, stop_at_score=a.gepa_stop_at_score)
    if a.baseline == "openevolve":           # one upstream iteration = one candidate evaluation
        return dict(iterations=a.iterations if a.iterations_set else 0, num_islands=a.oe_islands,
                    archive_size=a.oe_archive, feature_bins=a.oe_bins,
                    feature_dimensions=[s.strip() for s in a.oe_features.split(",") if s.strip()],
                    diff_based=not a.oe_full_rewrites, model=a.oe_model, temperature=a.oe_temperature,
                    target_score=1.0, eval_timeout_s=eval_timeout)
    if a.baseline == "shinka":
        return dict(generations=a.iterations if a.iterations_set else 0,
                    num_islands=a.shinka_islands, archive_size=a.shinka_archive,
                    models=a.shinka_models, llm_selection=a.shinka_selection,
                    embedding_model=a.shinka_embedding, max_novelty_attempts=a.shinka_novelty_attempts,
                    max_parallel=a.shinka_parallel, eval_timeout_s=eval_timeout)
    # the two descriptor-swap ablations share the corpus the method's runs fit their CVT on
    k0 = a.abl_k0 or (cfg.get("search") or {}).get("k0", 4)
    if a.baseline == "openevolve_cvt":
        return dict(iterations=a.iterations if a.iterations_set else 0, num_islands=a.oe_islands,
                    archive_size=a.oe_archive, diff_based=not a.oe_full_rewrites, model=a.oe_model,
                    temperature=a.oe_temperature, eval_timeout_s=eval_timeout, k0=k0,
                    cvt_dir=a.abl_cvt_dir or None, corpus_dirs=_corpus(a, cfg, adapter, k0),
                    rules=MH.role_rules(cfg))
    return dict(mode=a.abl_mode, k0=k0, rounds=a.iterations if a.iterations_set else 20,
                batch_size=a.abl_batch or None, random_init=a.abl_m, seed=a.seed,
                corpus_dirs=_corpus(a, cfg, adapter, k0), allow_collapse=a.abl_allow_collapse,
                desc_dir=a.abl_desc_dir or None)


def _corpus(a, cfg: dict, adapter, k0: int) -> list:
    """Directories the ablations fit their partition on: `--corpus`, else the proposer corpus."""
    anchor_dir = adapter.anchor().source_dir
    if a.corpus:
        return [anchor_dir] + sorted(d for d in a.corpus.iterdir() if MH.entry_file(d))
    root = Path("runs") / (cfg.get("testbed") or a.config.parent.name)
    return RB.proposer_corpus(adapter, anchor_dir, root / "corpus", k=max(5, k0 + 3))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", type=Path, required=True, help="benchmark config (JSON)")
    ap.add_argument("--anchor", type=Path, required=True, help="anchor harness directory")
    ap.add_argument("--baseline", required=True, choices=list(BASELINES))
    ap.add_argument("--budget", type=int, default=None,
                    help="cap on candidate evaluations (default: search.budget of the config; 0 = none)")
    ap.add_argument("--metric-budget", type=int, default=0,
                    help="cap on per-instance evaluations (GEPA's max_metric_calls); 0 = none")
    ap.add_argument("--wallclock", type=float, default=0.0, help="seconds; 0 = none")
    ap.add_argument("--iterations", type=int, default=None,
                    help="loop count (meta-harness iterations, dgm generations, openevolve/shinka "
                         "iterations, open_harness rounds); default lets --budget bind")
    ap.add_argument("--k", type=int, default=None,
                    help="children per proposal for meta-harness/dgm "
                         "(default: each arm's own: meta-harness 2, dgm 1)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="output directory (default: runs/<testbed>/<split>/<baseline>)")
    ap.add_argument("--tag", default="", help="suffix of the default run directory")
    ap.add_argument("--search-split", default=None, help="override benchmark.search_split")
    ap.add_argument("--holdout-split", default=None, help="override benchmark.holdout_split")
    ap.add_argument("--no-holdout", action="store_true", help="skip the winner's held-out evaluation")
    ap.add_argument("--corpus", type=Path, default=None,
                    help="ablations: directory of harness variants to fit the partition on "
                         "(default: the proposer corpus run_benchmark.py caches)")
    # meta-harness
    ap.add_argument("--meta-harness-parent", default="anchor", choices=["anchor", "best"],
                    help="what the prompt names as the parent; upstream is 'anchor'")
    ap.add_argument("--meta-harness-history", type=int, default=20,
                    help="past candidates shown to the proposer")
    # gepa
    ap.add_argument("--gepa-minibatch", type=int, default=3)
    ap.add_argument("--gepa-front", default="instance", choices=["instance", "objective", "hybrid"])
    ap.add_argument("--gepa-perfect", type=float, default=1.0)
    ap.add_argument("--gepa-no-skip-perfect", action="store_true")
    ap.add_argument("--gepa-merge", action="store_true", help="enable upstream's merge proposer")
    ap.add_argument("--gepa-proposer", default="claude-code", choices=["claude-code", "reflection-lm"],
                    help="upstream's ClaudeCodeAgentProposer (needs the claude CLI + bwrap) or a reflection LM")
    ap.add_argument("--gepa-model", default="opus", help="model of the Claude Code proposer")
    ap.add_argument("--gepa-reflection-lm", default=os.environ.get("GEPA_REFLECTION_LM", ""),
                    help="LiteLLM model id, or openai-sdk[:model] for an OpenAI-compatible endpoint")
    ap.add_argument("--gepa-no-sandbox", action="store_true")
    ap.add_argument("--gepa-workers", type=int, default=1)
    ap.add_argument("--gepa-stop-at-score", type=float, default=None)
    # dgm
    ap.add_argument("--dgm-method", default="score_child_prop",
                    choices=["score_child_prop", "score_prop", "best", "random"])
    ap.add_argument("--dgm-archive", default="keep_all", choices=["keep_all", "keep_better"])
    ap.add_argument("--dgm-selfimprove-size", type=int, default=2)
    ap.add_argument("--dgm-noise-leeway", type=float, default=0.1)
    ap.add_argument("--dgm-resolve-thresh", type=float, default=1.0)
    ap.add_argument("--dgm-small-n", type=int, default=3)
    ap.add_argument("--dgm-threshold", type=float, default=0.4)
    ap.add_argument("--dgm-shallow", action="store_true")
    ap.add_argument("--dgm-run-baseline", default="", choices=["", "no_selfimprove", "no_darwin"])
    # openevolve
    ap.add_argument("--oe-islands", type=int, default=2)
    ap.add_argument("--oe-archive", type=int, default=20)
    ap.add_argument("--oe-features", default="complexity,diversity")
    ap.add_argument("--oe-bins", type=int, default=10)
    ap.add_argument("--oe-full-rewrites", action="store_true", help="whole-file rewrites instead of diffs")
    ap.add_argument("--oe-model", default="", help="model id (default: $OPENEVOLVE_MODEL or $OPENAI_META_MODEL)")
    ap.add_argument("--oe-temperature", type=float, default=0.7)
    # shinka
    ap.add_argument("--shinka-islands", type=int, default=2)
    ap.add_argument("--shinka-archive", type=int, default=20)
    ap.add_argument("--shinka-models", default="",
                    help="comma-separated upstream model ids (default: local/$OPENAI_META_MODEL@$OPENAI_BASE_URL, else headless/claude)")
    ap.add_argument("--shinka-selection", default="ucb")
    ap.add_argument("--shinka-embedding", default="", help="embedding model for novelty rejection; empty = off")
    ap.add_argument("--shinka-novelty-attempts", type=int, default=3)
    ap.add_argument("--shinka-parallel", type=int, default=1)
    # ablations
    ap.add_argument("--abl-k0", type=int, default=None, help="cells of the ablation partition (default: search.k0)")
    ap.add_argument("--abl-mode", default="adaptive", choices=["adaptive", "fixed", "hier"],
                    help="open_harness: which of our resolution modes runs")
    ap.add_argument("--abl-batch", type=int, default=0, help="open_harness: P; 0 = log2(k0)")
    ap.add_argument("--abl-m", type=float, default=0.5, help="coverage fraction m")
    ap.add_argument("--abl-cvt-dir", default="", help="openevolve_cvt: reuse a frozen CVT (a run's cvt_k<k0>/)")
    ap.add_argument("--abl-desc-dir", default="", help="open_harness: reuse a frozen descriptor (a run's oe_descriptor/)")
    ap.add_argument("--abl-allow-collapse", action="store_true",
                    help="open_harness: run even if k-means collapses k0 (reports k0_realized)")
    a = ap.parse_args(argv)
    a.iterations_set = a.iterations is not None
    a.k_set = a.k is not None

    cfg = MH.load(a.config)
    bench = cfg["benchmark"]
    budget = a.budget if a.budget is not None else (cfg.get("search") or {}).get("budget")
    search_split = a.search_split or bench["search_split"]
    holdout_split = a.holdout_split or bench.get("holdout_split")
    root = Path("runs") / (cfg.get("testbed") or a.config.parent.name)
    run_dir = a.run_dir or root / search_split / f"{a.baseline}{a.tag}"
    adapter = RB.CommandAdapter(cfg, a.anchor, run_dir, search_split=search_split,
                                holdout_split=holdout_split)
    b = cfg.get("baselines") or {}
    adapter.bundle_suffixes = set(b.get("bundle_suffixes") or BUNDLE_SUFFIXES)
    adapter.bundle_exclude = set(b.get("bundle_exclude") or BUNDLE_EXCLUDE)

    ledger = Ledger(max_cand_evals=budget or None, max_metric_calls=a.metric_budget or None,
                    max_wallclock_s=a.wallclock or None,
                    override_file=Path(run_dir) / "BUDGET_OVERRIDE")     # write a new cap to extend a live run
    pin = pins()[UPSTREAM_OF[a.baseline]]
    log(f"[run_baseline] baseline={a.baseline} config={a.config} out={run_dir}")
    log(f"[run_baseline] upstream {pin['url']} @ {pin['commit'][:12]}"
        + ("" if pin["at_pin"] else f" (NOT the pinned {pin['pinned'][:12]})"))
    log(f"[run_baseline] search={len(adapter.evolve)} holdout={len(adapter.evalset)} "
        f"budget=cand:{budget or '-'} metric:{a.metric_budget or '-'} seed={a.seed}")
    if a.baseline == "gepa":
        n_val = len(adapter.evolve) or 1
        have = a.metric_budget or (budget * n_val if budget else 0)
        if have and have < 15 * n_val:
            log(f"[run_baseline] WARNING: the budget buys ~{have // n_val} proposal(s) over "
                f"{n_val} instances; GEPA's skill advises >= 15-20 (metric budget >= {15 * n_val})")

    t0 = time.time()
    summary = run_baseline(a.baseline, adapter, out_dir=run_dir, ledger=ledger,
                           holdout=not a.no_holdout, log=log,
                           **arm_kwargs(a, cfg, adapter, budget))
    best, hold = summary.get("best") or {}, summary.get("holdout") or {}
    log(f"[run_baseline] {a.baseline}: best={best.get('label', 'none')} "
        f"primary={best.get('primary', 0):.4f} holdout={hold.get('primary', float('nan')):.4f} "
        f"({summary['budget']['cand_evals']} cand evals / {summary['budget']['metric_calls']} "
        f"metric calls / {time.time() - t0:.0f}s) -> {summary['stop_reason']}")
    log(f"[run_baseline] artifacts: {run_dir}")


if __name__ == "__main__":
    main()
