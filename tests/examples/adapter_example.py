#!/usr/bin/env python
"""A custom adapter for Harness-MAP: the adapter protocol, as a runnable skeleton.

`run_benchmark.py` covers the common case, an evaluator that is a shell command writing a score file,
through its CommandAdapter. Subclass AgenticAdapter yourself for in-process evaluation, your own
batching or a different judge policy. The search (harness_map.main.run) calls these members:

    starting_harness(cfg=None) -> Genome        required    the harness the search starts from (its
                                                            source_dir and a label); `anchor` is the
                                                            older name of this hook and still works
    evaluate(genome, iter_n=0) -> Fitness       required    run the frozen base model with the harness on the
                                                            search split; primary in [0, 1], 1.0 = solved
    evaluate_batch(genomes, iter_n=0) -> list   recommended evaluate several genomes (you control the
                                                            concurrency); one Fitness or None per genome.
                                                            Candidates evaluated through it are reviewed by
                                                            the LLM judge and diagnosed; with evaluate()
                                                            alone, nothing is judged.
    evaluate_holdout(genome) -> Fitness         optional    score the final best harness on the held-out split
    privileged_keys() -> set[str]               override    names a harness must never read; the static
                                                            pre-eval gate scans the source for them
    pre_eval_gate(genome) -> str | None         optional    replace the default static gate; return a
                                                            rejection reason or None
    trace_dir_for(genome) -> Path | None        optional    where evaluate() wrote traces, so the judge and
                                                            the diagnostician can read frames and reasoning
    proposer_env(parent) -> dict                optional    extra environment variables for the skill agents
    propose / judge / diagnose                  inherited   drive the proposer, judge and diagnostician
                                                            skills; if you replace propose, keep the
                                                            signature propose(parent, iter_n, directive=None,
                                                            n_children=None) -> list[Genome]
    meta_agent, work_root                       set these   the backbone config; where candidates are written
    AUDIT_PROVIDED = True                       flag        the adapter produces taint / judge verdicts
    EVAL_SUBSET_SAFE = True                     flag        coverage initialization may pass any subset of
                                                            the proposed genomes to evaluate_batch

Fitness fields the search reads: primary (maximized), valid (False drops the candidate), taint / judge
(anti-cheat verdicts; a FAIL loses to any clean candidate), rows (per-episode results), and optionally
vector, dense, noise, cost.

    python adapter_example.py --config config.json --start harnesses/my_task --corpus corpus/ --budget 50

`--corpus` is a directory of harness directories, variants of the starting harness including its code,
tools and skills (not only its prompt), on which the CVT is fit once and then frozen. Use the same corpus
and the same frozen phi for every mode you compare. The budget N gives the number of generations,
G = floor((N - ceil(m * k0)) / (P * C)) with P = ceil(log2 k0). mapelites_fixed takes the same
arguments minus the adaptive-only ones (sigma, k_split, depth_cap, patience, split_target);
mapelites_hier takes a coarse-to-fine ladder (centroid_levels) instead of init_centroids.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from harness_map import main as MH
from harness_map.main import Fitness, Genome
from harness_map.meta_agent.agent import AgenticAdapter


class MyAdapter(AgenticAdapter):
    """Evaluates harnesses in-process; propose / judge / diagnose are inherited from AgenticAdapter."""

    def __init__(self, cfg: dict, start_dir: Path, work_root: Path):
        self.cfg, self.meta_agent = cfg, cfg["meta_agent"]
        self.start_dir, self.work_root = Path(start_dir), Path(work_root)

    def starting_harness(self, cfg=None) -> Genome:
        return Genome(source_dir=self.start_dir, label="start")

    def evaluate(self, genome: Genome, iter_n: int = 0) -> Fitness:
        # Run the frozen base model with genome.source_dir on the search split, score the outcome from
        # the true final state, and return Fitness(primary=success_rate, rows=per_episode, valid=True).
        raise NotImplementedError("evaluate(): run your benchmark on genome.source_dir and return a Fitness")

    def evaluate_batch(self, genomes, iter_n: int = 0):
        return [self.evaluate(g, iter_n) for g in genomes]

    def privileged_keys(self):
        return set(MH.privileged_keys(self.cfg))


def generations(budget: int, k0: int, children: int, m: float = 0.5) -> int:
    """G from the budget N = ceil(m * k0) + G * P * C, with P = ceil(log2 k0)."""
    P = max(1, math.ceil(math.log2(k0))) if k0 > 1 else 1
    return max(1, (budget - math.ceil(m * k0)) // (P * children))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="JSON config merged over harness_map.main.DEFAULTS")
    ap.add_argument("--start", "--anchor", dest="start", required=True, help="the starting harness directory")
    ap.add_argument("--corpus", required=True, help="directory of harness directories the CVT is fit on")
    ap.add_argument("--mode", default="mapelites_adaptive",
                    choices=["mapelites_adaptive", "mapelites_fixed", "mapelites_hier"])
    ap.add_argument("--k0", type=int, default=8, help="number of CVT cells")
    ap.add_argument("--budget", type=int, default=50, help="N, the number of harness evaluations")
    ap.add_argument("--children", type=int, default=2, help="C, children proposed per parent")
    ap.add_argument("--m", type=float, default=0.5, help="fraction of the k0 cells initialization must fill")
    ap.add_argument("--work-root", default="runs/my_task")
    a = ap.parse_args(argv)

    cfg = MH.load(a.config)
    rules = MH.role_rules(cfg)
    corpus_dirs = sorted(p for p in Path(a.corpus).iterdir() if p.is_dir())
    work_root = Path(a.work_root)
    adapter = MyAdapter(cfg, Path(a.start), work_root / "candidates")
    rounds = generations(a.budget, a.k0, a.children, a.m)
    P = max(1, math.ceil(math.log2(a.k0))) if a.k0 > 1 else 1

    cvt_dir = str(work_root / f"cvt_k{a.k0}")
    MH.fit_cvt_blocks(corpus_dirs, cvt_dir, rules, k0=a.k0)            # fit once, then frozen
    phi_fn, init_centroids, block_split = MH.make_phi_blocks(cvt_dir, rules)
    common = dict(phi_fn=phi_fn, block_split=block_split, rounds=rounds,
                  batch_size=P, children_per_parent=a.children, max_evals=a.budget)
    if a.mode == "mapelites_hier":
        nodes = [MH.Node(idx=i, genome=Genome(source_dir=Path(d), label=f"corpus_{i}"), fit=None)
                 for i, d in enumerate(corpus_dirs)]
        ladder = MH.build_centroid_ladder(phi_fn, nodes, k0=a.k0, levels=3, branching=2)
        out = MH.run(adapter, cfg, mode=a.mode, centroid_levels=ladder, **common)
    else:
        out = MH.run(adapter, cfg, mode=a.mode, init_centroids=init_centroids, random_init=a.m, **common)
    print("best:", out["best"]["fit"].primary)


if __name__ == "__main__":
    main()
