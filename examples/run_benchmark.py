#!/usr/bin/env python
"""Run Harness-MAP on a benchmark described by a config file.

    python examples/run_benchmark.py --config examples/balrog/config.json --anchor <ANCHOR_DIR>

`benchmark.evaluate` in the config is a shell command that runs one harness on one split and
writes a score file; see the benchmark READMEs under examples/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness_map import main as MH                                      # noqa: E402
from harness_map.main import Fitness, Genome                      # noqa: E402
from harness_map.meta_agent.agent import AgenticAdapter, propose_candidates  # noqa: E402

MODES = ("adaptive", "fixed", "hier")
STAGE_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "diagnosis*.md", "judge*.json",
                                      "read_audit.json", "propose.log")


def log(msg: str) -> None:
    print(msg, flush=True)


# ── budget ─────────────────────────────────────────────────────────────────────────────────────
def parents_per_round(k0: int) -> int:
    """P, the parents bred per generation: ceil(log2 k0), at least 1."""
    return max(1, math.ceil(math.log2(max(2, int(k0)))))


def default_children(k0: int) -> int:
    """C used in the paper: two children per parent for k0 <= 4, one for larger archives."""
    return 2 if k0 <= 4 else 1


def budget(k0: int, rounds: int, children: int, m: float = 0.5) -> dict:
    """The terms of the evaluation budget N = M + G*P*C, with M = ceil(m*k0)."""
    P = parents_per_round(k0)
    M = math.ceil(m * k0)
    return {"k0": k0, "M": M, "G": rounds, "P": P, "C": children, "N": M + rounds * P * children}


# ── adapter ────────────────────────────────────────────────────────────────────────────────────
class CommandAdapter(AgenticAdapter):
    """Runs the `benchmark.evaluate` command on a harness and reads the score file it writes."""

    AUDIT_PROVIDED = True

    def __init__(self, cfg: dict, anchor_dir, run_dir, *, search_split: str, holdout_split=None):
        self.cfg = cfg
        self.bench = cfg["benchmark"]
        self.meta_agent = cfg.get("meta_agent")
        # Absolute: the judge and diagnostician run with the candidate directory as cwd and
        # receive its path as OUT_DIR, so a relative run directory would not resolve from there.
        self.run_dir = Path(run_dir).resolve()
        self.work_root = self.run_dir / "candidates"
        self.work_root.mkdir(parents=True, exist_ok=True)
        self.anchor_dir = Path(anchor_dir)
        self.search_split, self.holdout_split = search_split, holdout_split
        splits = cfg.get("splits") or {}
        self.evolve = list(splits.get(search_split) or [])            # task records of the two
        self.evalset = list(splits.get(holdout_split) or []) if holdout_split else []   # splits
        self._last_out = {}

    def anchor(self, cfg=None) -> Genome:
        dst = self.run_dir / "anchor"
        if not dst.exists():
            shutil.copytree(self.anchor_dir, dst, ignore=STAGE_IGNORE)
        return Genome(source_dir=dst, label="anchor")

    def evaluate(self, genome: Genome, iter_n: int = 0) -> Fitness:
        return self._run(genome, self.search_split, f"r{iter_n:03d}")

    def evaluate_holdout(self, genome: Genome):
        return self._run(genome, self.holdout_split, "holdout") if self.holdout_split else None

    @staticmethod
    def task_id(record: dict) -> str:
        """The id of a split record: its `task_id` (or `id`), else its values joined by "/"."""
        return str(record.get("task_id") or record.get("id")
                   or "/".join(str(record[k]) for k in sorted(record)))

    def evaluate_subset(self, genome: Genome, task_ids, rid: int = 0):
        """The search split restricted to `task_ids` (the baselines' minibatch hook)."""
        want = {str(t) for t in task_ids}
        tasks = [t for t in self.evolve if self.task_id(t) in want]
        return self._run(genome, self.search_split, f"r{rid:03d}_sub{len(tasks)}", tasks) if tasks else None

    def privileged_keys(self) -> set:
        return set(MH.privileged_keys(self.cfg))

    def trace_dir_for(self, genome):
        return self._last_out.get(genome.label)

    def _run(self, genome: Genome, split: str, tag: str, tasks=None) -> Fitness:
        tasks = self.cfg["splits"][split] if tasks is None else tasks
        out = self.run_dir / "evals" / f"{genome.label}_{tag}"
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        (out / "tasks.json").write_text(json.dumps(tasks, indent=1))
        cmd = os.path.expandvars(self.bench["evaluate"]).format(
            harness=Path(genome.source_dir).resolve(), out=out.resolve(), split=split,
            tasks=(out / "tasks.json").resolve())
        score_file = out / self.cfg["score_file"]
        for _ in range(2):                    # one retry: a crash before scoring is usually infra
            try:
                with open(out / "eval.log", "ab") as fh:
                    subprocess.run(cmd, shell=True, stdout=fh, stderr=subprocess.STDOUT,
                                   timeout=self.bench.get("timeout_s"))
            except subprocess.TimeoutExpired:
                return Fitness(primary=0.0, valid=False, extra={"error": "evaluation timeout"})
            if score_file.exists():
                break
        else:
            return Fitness(primary=0.0, valid=False, extra={"error": f"no {score_file.name}"})
        self._last_out[genome.label] = out
        fit = MH.fitness_from_dir(out, self.cfg)
        expected = sum(int(t.get("episodes", 1)) for t in tasks)
        if fit.n < expected:                  # a partial run must not count as a full evaluation
            fit.valid = False
            fit.extra["truncated"] = f"{fit.n}/{expected} episodes"
        return fit


class JudgedCommandAdapter(CommandAdapter):
    """CommandAdapter whose evaluations go through `evaluate_batch`, so the LLM judge runs."""

    def evaluate_batch(self, genomes, iter_n: int = 0):
        return [self.evaluate(g, iter_n) for g in genomes]


# ── corpus and search ──────────────────────────────────────────────────────────────────────────
def content_hash(d: Path) -> str:
    h = hashlib.sha1()
    d = Path(d)
    for f in sorted(d.rglob("*")):
        if f.is_file() and f.suffix in (".py", ".md", ".txt") and "__pycache__" not in f.parts:
            h.update(f.relative_to(d).as_posix().encode() + b"\0" + f.read_bytes())
    return h.hexdigest()


def proposer_corpus(adapter, anchor_dir, out_dir, k: int, *, max_rounds: int = 6) -> list:
    """The anchor and `k` distinct, never-evaluated meta-agent variants of it, to fit the CVT on."""
    anchor_dir, out_dir = Path(anchor_dir), Path(out_dir)
    anchor = Genome(source_dir=anchor_dir, label="anchor")

    def distinct():
        seen, keep = {content_hash(anchor_dir)}, []
        for d in sorted(out_dir.glob("cand_r*")):
            if (d / "harness.py").exists():
                s = content_hash(d)
                if s not in seen:
                    seen.add(s)
                    keep.append(d)
        return keep

    have = distinct() if out_dir.exists() else []
    for _ in range(max_rounds):
        if len(have) >= k:
            break
        used = [int(d.name.split("_")[1][1:]) for d in out_dir.glob("cand_r*")]
        need = k - len(have)
        log(f"[corpus] proposing {need + need // 2 + 1} variants of the anchor ({len(have)}/{k} so far)")
        propose_candidates(anchor, work_root=out_dir, rid=max(used, default=-1) + 1,
                           n_children=need + need // 2 + 1, meta_agent=adapter._ma(),
                           env=adapter.proposer_env(anchor))
        before, have = len(have), distinct()
        if len(have) == before:
            break
    if len(have) < k:
        log(f"[corpus] only {len(have)} distinct variants (wanted {k})")
    return [anchor_dir] + have[:k]


def run_search(adapter, cfg: dict, *, corpus: list, run_dir, mode: str = "adaptive", k0: int = 4,
               rounds: int = 12, children: int = 2, m: float = 0.5, sigma: float = 0.0,
               max_evals: int | None = None, strict_coverage: bool = True,
               hier_levels: int = 3, hier_branching: int = 2, resume: bool = False) -> dict:
    """Fit (or reuse) the k0-cell CVT on `corpus`, run one search, and write run_dir/result.json."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    rules = MH.role_rules(cfg)

    cvt_dir = run_dir / f"cvt_k{k0}"
    if not (cvt_dir / "cvt_blocks.npz").exists():
        n, dim, k, _ = MH.fit_cvt_blocks([Path(d) for d in corpus], cvt_dir, rules, k0=k0)
        log(f"[search] CVT fit on {n} harnesses: {k} cells, {dim} dimensions")
        if k != k0:
            raise SystemExit(f"the corpus separates only {k} of k0={k0} cells; "
                             f"build a larger or more varied corpus")
    phi, Z0, block_split = MH.make_phi_blocks(cvt_dir, rules)

    b = budget(k0, rounds, children, m)
    knobs = {"phi_fn": phi, "block_split": block_split, "seed_genomes": [], "steer": True,
             "rounds": rounds, "batch_size": b["P"], "children_per_parent": children,
             "random_init": m, "random_init_strict": strict_coverage, "max_evals": max_evals}
    if mode == "hier":
        nodes = [MH.Node(idx=i, genome=Genome(source_dir=Path(d), label=f"corpus_{i}"), fit=None)
                 for i, d in enumerate(corpus)]
        knobs["centroid_levels"] = MH.build_centroid_ladder(
            phi, nodes, k0=k0, levels=hier_levels, branching=hier_branching, log=log)
    else:
        knobs["init_centroids"] = Z0
    if mode == "adaptive":
        knobs["sigma"] = sigma
    if resume:                              # continue from the rolling snapshot; nothing is re-evaluated
        snap = run_dir / "checkpoints" / "frontier_partial.json"
        if not snap.exists():
            raise SystemExit(f"--resume: no snapshot at {snap}")
        knobs["resume_from"] = str(snap)

    os.environ.setdefault("HARNESS_MAP_CKPT_DIR", str(run_dir / "checkpoints"))
    log(f"[search] mode={mode} k0={k0} budget N = {b['M']} + {b['G']}*{b['P']}*{b['C']} = {b['N']}"
        + (f" (capped at {max_evals})" if max_evals else ""))
    t0 = time.time()
    out = MH.run(adapter, cfg, mode=f"mapelites_{mode}", log=log, **knobs)
    _save(run_dir / "result.json", out, mode=mode, budget=b, sigma=sigma, wall_s=time.time() - t0)
    return out


def _fit_row(fit) -> dict:
    if fit is None:
        return {}
    return {"primary": fit.primary, "valid": fit.valid, "n": fit.n, "taint": fit.taint,
            "judge": fit.judge, "vector": fit.vector, "dense": fit.dense}


def _save(path: Path, out: dict, **meta) -> None:
    best, hold = out.get("best"), out.get("holdout")
    rec = {**meta,
           "best": {"label": best["label"], **_fit_row(best["fit"])} if best else None,
           "holdout": _fit_row(hold) if hold else None,
           "history": [{"iter": r["iter"], "label": r["label"], "depth": r["depth"],
                        "parents": list(r["parents"]), "source_dir": str(r["genome"].source_dir),
                        **_fit_row(r["fit"])} for r in out.get("history", [])]}
    for key in ("frontier", "fixed", "hier"):
        if key in out:
            rec[key] = out[key]
    path.write_text(json.dumps(rec, indent=1, default=str))
    if best:
        log(f"[search] best {best['label']}: primary={best['fit'].primary:.3f}"
            + (f", holdout={hold.primary:.3f}" if hold else "") + f" -> {path}")


# ── command line ───────────────────────────────────────────────────────────────────────────────
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", type=Path, required=True, help="benchmark config (JSON)")
    ap.add_argument("--anchor", type=Path, required=True, help="anchor harness directory")
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="output directory (default: runs/<testbed>/<mode>_k<k0>)")
    ap.add_argument("--corpus", type=Path, default=None,
                    help="directory of harness variants to fit the CVT on (default: meta-agent "
                         "variants of the anchor, cached in runs/<testbed>/corpus)")
    ap.add_argument("--search-split", default=None, help="override benchmark.search_split")
    ap.add_argument("--holdout-split", default=None, help="override benchmark.holdout_split")
    ap.add_argument("--mode", choices=MODES, default=None,
                    help="resolution control: adaptive (the method), fixed or hier (ablations)")
    ap.add_argument("--k0", type=int, default=None, help="initial number of CVT cells")
    ap.add_argument("--rounds", type=int, default=None, help="G, number of generations")
    ap.add_argument("--budget", type=int, default=None,
                    help="N; derives G = floor((N - M) / (P*C)) instead of --rounds")
    ap.add_argument("--children", type=int, default=None,
                    help="C, children per parent (default: 2 if k0 <= 4, else 1)")
    ap.add_argument("--m", type=float, default=None, help="coverage target as a fraction of k0")
    ap.add_argument("--sigma", type=float, default=None, help="adaptive only: noise floor")
    ap.add_argument("--lenient-coverage", action="store_true",
                    help="continue when coverage-gated initialization falls short of its target")
    ap.add_argument("--resume", action="store_true",
                    help="continue the run in --run-dir from checkpoints/frontier_partial.json")
    a = ap.parse_args(argv)

    cfg = MH.load(a.config)
    s = cfg.get("search") or {}
    mode = a.mode or s.get("mode", "adaptive")
    k0 = a.k0 or s.get("k0", 4)
    C = a.children or s.get("children") or default_children(k0)
    m = a.m if a.m is not None else s.get("m", 0.5)
    sigma = a.sigma if a.sigma is not None else s.get("sigma", 0.0)
    n_budget = a.budget if a.budget is not None else s.get("budget")
    if a.rounds is not None:                # explicit G; the budget, if given, stays a cap
        rounds, n_budget = a.rounds, a.budget
    elif n_budget:
        b = budget(k0, 0, C, m)
        rounds = max(1, (n_budget - b["M"]) // (b["P"] * C))
    else:
        rounds = s.get("rounds", 12)
    strict = s.get("strict_coverage", True) and not a.lenient_coverage

    bench = cfg["benchmark"]
    search_split = a.search_split or bench["search_split"]
    holdout_split = a.holdout_split or bench.get("holdout_split")
    root = Path("runs") / (cfg.get("testbed") or a.config.parent.name)
    run_dir = a.run_dir or root / search_split / f"{mode}_k{k0}"
    cls = JudgedCommandAdapter if bench.get("judge") else CommandAdapter
    adapter = cls(cfg, a.anchor, run_dir, search_split=search_split, holdout_split=holdout_split)

    if a.corpus:
        corpus = [adapter.anchor().source_dir] + sorted(
            d for d in a.corpus.iterdir() if (d / "harness.py").exists())
    else:
        corpus = proposer_corpus(adapter, adapter.anchor().source_dir, root / "corpus",
                                 k=max(5, k0 + 3))
    run_search(adapter, cfg, corpus=corpus, run_dir=run_dir, mode=mode, k0=k0, rounds=rounds,
               children=C, m=m, sigma=sigma, max_evals=n_budget, strict_coverage=strict,
               resume=a.resume)


if __name__ == "__main__":
    main()
