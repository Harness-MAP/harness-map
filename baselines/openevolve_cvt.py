"""Ablation: upstream OpenEvolve's search, with our frozen CVT cell as the archive key."""
from __future__ import annotations

import hashlib
import json
import os
import threading

from collections import Counter
from pathlib import Path

import numpy as np

from . import common as C
from .upstream import load_openevolve, pins
# Reused from the vanilla arm: everything that is not the descriptor, so a fix lands on both.
from .openevolve import EVALUATE_PY, SYSTEM_MESSAGE, _EvalServer, _metrics


FEATURE_DIM = "cvt_cell"          # the single archive axis; deliberately NOT an evaluator metric


def _nearest_cell(Z0, p) -> int:
    """Index of the nearest frozen centroid to `p` (first minimum on ties)."""
    return int(np.argmin(((np.asarray(Z0, float) - np.asarray(p, float)) ** 2).sum(axis=1)))


def _load_or_fit_cvt(*, cvt_dir: Path, corpus_dirs, rules, k0: int, log):
    """Reuse the CVT in `cvt_dir` or fit it on `corpus_dirs`; (phi, Z0, block_split, meta)."""
    from harness_map.qd import partition as P

    cvt_dir = Path(cvt_dir)
    cvt_dir.mkdir(parents=True, exist_ok=True)
    frozen = cvt_dir / "cvt_blocks.npz"
    n_corpus, fitted_here = None, False
    if frozen.exists():
        log(f"[openevolve-cvt] reusing the frozen partition at {cvt_dir} (no refit)")
    else:
        corpus = [Path(d) for d in (corpus_dirs or [])]
        if len(corpus) < 2:
            # one directory gives one centroid, a hill-climb reported as a CVT arm: an error
            raise ValueError(
                f"no frozen CVT at {frozen} and corpus_dirs={len(corpus)} harness dirs. Pass "
                f"cvt_dir=<the ours arm's cvt_k{k0}> to share its partition, or pass "
                f"corpus_dirs=<bootstrap_corpus(...)> (the same corpus the ours arm fitted on).")
        log(f"[openevolve-cvt] fitting CVT k0={k0} on {len(corpus)} corpus harnesses ...")
        fitted_here = True
        n_corpus, dim, k, bs = P.fit_cvt_blocks(corpus, cvt_dir, rules, k0=k0)
        if k != k0:
            # k-means caps k at the number of distinct points; K0 is every coverage denominator
            log(f"[openevolve-cvt] WARNING: kmeans collapsed to {k} < k0={k0} (corpus not distinct "
                f"enough); the archive will have {k} cells, not {k0}.")
        log(f"[openevolve-cvt] CVT n={n_corpus} joint_dim={dim} k0={k} block_split(structure)={bs}")

    phi, Z0, block_split = P.make_phi_blocks(cvt_dir, rules)
    if not fitted_here and len(Z0) != k0:
        # a shared partition is authoritative over the caller's k0
        log(f"[openevolve-cvt] NOTE: the frozen partition has {len(Z0)} cells, not the requested "
            f"k0={k0}; using {len(Z0)} (the shared partition wins).")
    meta_path = cvt_dir / "blocks_meta.json"
    bmeta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return phi, np.asarray(Z0, float), int(block_split), {
        "cvt_dir": str(cvt_dir), "k0_requested": int(k0), "k0_realized": int(len(Z0)),
        "joint_dim": int(np.asarray(Z0).shape[1]) if len(Z0) else 0,
        "block_split": int(block_split), "fitted_here": fitted_here,
        "corpus_size_at_fit": n_corpus,
        "corpus_dirs": [str(d) for d in (corpus_dirs or [])][:64],
        "blocks": bmeta.get("blocks", [])}


def _cvt_database_class(base_cls, cell_of, k0: int, sink: list):
    """A `ProgramDatabase` subclass whose feature coordinate is our frozen CVT cell."""

    class CvtProgramDatabase(base_cls):
        """Upstream's database with `_calculate_feature_coords` replaced, nothing else."""

        def __init__(self, config):
            super().__init__(config)
            # after construction: upstream resolves feature_bins from archive_size, and both the
            # coverage denominator and the inspiration clamp must see K0
            self.feature_bins = int(k0)
            self.feature_bins_per_dim = {FEATURE_DIM: int(k0)}
            sink.append(self)

        def _calculate_feature_coords(self, program):
            # the entire ablation: one nominal coordinate, the candidate's frozen CVT cell
            return [cell_of(program.code)]

    CvtProgramDatabase.__name__ = "CvtProgramDatabase"
    return CvtProgramDatabase


def run(adapter, *, out_dir, ledger: C.Ledger, k0: int = 8, cvt_dir=None, corpus_dirs=None,
        rules=None, iterations: int = 0, num_islands: int = 2,
        archive_size: int = 20, diff_based: bool = True, model: str = "", api_base: str = "",
        api_key: str = "", temperature: float = 0.7, max_tokens: int = 16000, target_score=1.0,
        eval_timeout_s: int = 7200, holdout: bool = True, log=print, **_) -> dict:
    """Run upstream openevolve with our CVT as the archive key, and report the winner."""
    load_openevolve()
    import openevolve.controller as oe_controller
    from openevolve.api import run_evolution
    from openevolve.config import Config, LLMModelConfig
    from harness_map.main import Genome
    from harness_map.qd import descriptors as D
    from harness_map.qd.coverage import _PhiProbe        # the same shim our coverage screen uses

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    anchor = adapter.anchor()
    anchor_dir = Path(anchor.source_dir)
    rules = D.DEFAULT_ROLE_RULES if rules is None else rules

    # (1) the partition, before anything else: a failure here must cost zero proposer tokens.
    phi, Z0, block_split, cvt_pins = _load_or_fit_cvt(
        cvt_dir=Path(cvt_dir) if cvt_dir else out / f"cvt_k{k0}",
        corpus_dirs=corpus_dirs, rules=rules, k0=k0, log=log)
    K0 = int(len(Z0))                       # AUTHORITATIVE: a shared/collapsed partition may differ
    SENTINEL = K0                           # outside [0, K0), so the inspiration clamp cannot reach it

    seed_path = out / "initial.py"
    seed_path.write_text(C.bundle_harness(anchor_dir, suffixes=getattr(adapter, "bundle_suffixes", None),
                                          exclude=getattr(adapter, "bundle_exclude", None)))

    n_iters = iterations or (ledger.max_cand_evals * 2 if ledger.max_cand_evals else 0) or 12
    model = model or os.environ.get("OPENEVOLVE_MODEL") or os.environ.get("OPENAI_META_MODEL", "")
    api_base = api_base or os.environ.get("OPENAI_BASE_URL", "")
    api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not model:
        raise ValueError(
            "OpenEvolve needs an LLM. Set OPENAI_META_MODEL (+ OPENAI_BASE_URL / OPENAI_API_KEY) "
            "or pass --openevolve-model; upstream drives an OpenAI-compatible endpoint.")

    rl = C.RunLog(out_dir, "openevolve_cvt", {
        "engine": "upstream openevolve.api.run_evolution",
        "ablation": "openevolve-cvt: upstream search, frozen CVT archive key",
        "iterations": n_iters, "num_islands": num_islands, "archive_size": archive_size,
        "feature_dimensions": [FEATURE_DIM], "feature_bins": K0,
        "diff_based_evolution": diff_based, "model": model, "api_base": api_base,
        "upstream": pins()["openevolve"],
        "candidate": "whole-package bundle, one EVOLVE-BLOCK region per source file",
        "cvt": cvt_pins,
        # WHAT WAS PATCHED AND WHY, banked next to the numbers it produced. A reader of this run
        # directory should never have to diff the vendor clone to find out.
        "runtime_patch": {
            "target": "openevolve.controller.ProgramDatabase",
            "replacement": "CvtProgramDatabase(ProgramDatabase), overriding _calculate_feature_coords",
            "why": ("the archive key is the ablated variable; controller.py:16 binds the class BY "
                    "VALUE so openevolve.database.ProgramDatabase is the wrong name to rebind, and "
                    "routing the cell through a custom METRIC would push a nominal k-means label "
                    "through _update_feature_stats + _scale_feature_value (running min/max) and "
                    "split one niche across several keys"),
            "scope": "set immediately before run_evolution, restored in a finally",
            "vendor_files_edited": [],
            "dead_upstream_code_in_this_arm": ["_update_feature_stats", "_scale_feature_value",
                                               "_calculate_complexity_bin", "_calculate_diversity_bin"],
            "known_semantic_change": ("_sample_inspirations perturbs the coordinate by randint(-2,2) "
                                      "(database.py:1647-1650), which assumes an ordinal axis; on "
                                      "nominal CVT labels at small K0 this is close to a uniform "
                                      "random cell draw. LEFT AS UPSTREAM WROTE IT and declared.")}})

    pool: list[C.Cand] = []
    by_hash: dict[str, C.Cand] = {}
    lock = threading.Lock()
    cell_cache: dict[str, int] = {}                 # program-code hash -> frozen CVT cell
    probe_dirs: dict[str, Path] = {}                # unpaid materialisations, for cells only
    cell_lock = threading.Lock()                    # never held while `lock` is held

    def cell_of(code: str) -> int:
        """The frozen CVT cell of a program, memoised on the code hash."""
        h = hashlib.sha1(code.encode()).hexdigest()[:12]
        with cell_lock:
            hit = cell_cache.get(h)
        if hit is not None:
            return hit
        with lock:
            known = by_hash.get(h)
        d = Path(known.genome.source_dir) if known is not None else None
        if d is None or not d.is_dir():
            # refused or failed candidates were never unbundled; materialise (unpaid) for the cell
            with cell_lock:
                d = probe_dirs.get(h)
            if d is None:
                try:
                    d = C.unbundle_harness(code, anchor_dir, out / "cvt_probes" / f"probe_{h}")
                except Exception as e:
                    log(f"[openevolve-cvt] could not materialise {h} for localization: {e!r} "
                        f"-> sentinel cell {SENTINEL}")
                    with cell_lock:
                        cell_cache[h] = SENTINEL
                    return SENTINEL
                with cell_lock:
                    probe_dirs[h] = d
        try:
            p = np.asarray(phi(_PhiProbe(Genome(source_dir=d, label=f"cand_{h}"))), float)
            cell = _nearest_cell(Z0, p)
        except Exception as e:                       # a descriptor failure must not take the run down
            # do not memoise a failure: the descriptor cache is read from two threads and a read
            # landing mid-write fails transiently
            log(f"[openevolve-cvt] phi failed for {h}: {e!r} -> sentinel cell {SENTINEL} "
                f"(NOT cached; will retry on next localization)")
            return SENTINEL
        with cell_lock:
            cell_cache[h] = cell
        return cell

    def handle(program_path) -> dict:
        """Score one candidate -> upstream's metrics dict, recording its CVT cell too."""
        text = Path(program_path).read_text()
        h = hashlib.sha1(text.encode()).hexdigest()[:12]
        with lock:
            known = by_hash.get(h)
        if known is not None and known.fit is not None:
            return _metrics(known)                       # upstream re-scores clones; don't re-pay

        # budget stop as in openevolve: refuse to pay past the cap
        if ledger.max_cand_evals and ledger.cand_evals >= ledger.max_cand_evals:
            return {"combined_score": 0.0, "budget_exhausted": True}

        d = C.unbundle_harness(text, anchor_dir, out / "candidates" / f"cand_{h}")
        genome = Genome(source_dir=d, label=f"cand_{h}")
        if not C.validate(adapter, genome):
            ledger.spend(0)
            cand = C.Cand(idx=len(pool), genome=genome, fit=None,
                          meta={"role": "invalid", "hash": h})
            with lock:
                pool.append(cand); by_hash[h] = cand
            cand.meta["cell"] = cell_of(text)            # after by_hash, so the dir is found
            rl.log(cand, ledger)
            log(f"[openevolve-cvt] {genome.label}: does not compile -> 0.0")
            return {"combined_score": 0.0, "error": "candidate does not compile"}

        fit, _n = C.evaluate_on(adapter, genome, rid=ledger.cand_evals, ledger=ledger)
        cand = C.Cand(idx=len(pool), genome=genome, fit=fit, scores=C.per_instance(fit),
                      meta={"role": "candidate", "hash": h})
        with lock:
            pool.append(cand); by_hash[h] = cand
        cand.meta["cell"] = cell_of(text)                # after by_hash, so the dir is found
        rl.log(cand, ledger)
        log(f"[openevolve-cvt] {genome.label}: combined_score={cand.primary:.4f} "
            f"cell={cand.meta['cell']}/{K0} "
            f"({ledger.cand_evals}/{ledger.max_cand_evals or '-'} cand evals)")
        # Do not raise on saturation here: the RPC thread would report the saturating candidate
        # as 0.0; upstream's own target_score is the stop that works.
        if (cand.primary or 0) >= 1.0:
            log(f"[openevolve-cvt] SATURATED at {ledger.cand_evals} evals "
                f"(best={cand.primary:.3f}) -> upstream target_score should stop the run.")
        return _metrics(cand)

    eval_py = out / "evaluate.py"
    eval_py.write_text(EVALUATE_PY.format(timeout=int(eval_timeout_s)))
    server = _EvalServer(out / "rpc", handle, log=log)
    server.start()

    # ── upstream config: identical to the vanilla arm except the two feature knobs ────────────
    cfg = Config()
    cfg.language = "python"
    cfg.max_iterations = n_iters
    cfg.checkpoint_interval = int(os.environ.get("OE_CHECKPOINT_INTERVAL", "10"))
    cfg.diff_based_evolution = diff_based
    cfg.max_code_length = max(8 * len(seed_path.read_text()), 100_000)
    sys_msg = SYSTEM_MESSAGE + ("\n\n" + C.DIFF_DISCIPLINE if diff_based else "")
    cfg.prompt.system_message = sys_msg
    _eff = os.environ.get("OPENAI_REASONING_EFFORT", "high")
    cfg.llm.models = [LLMModelConfig(name=model, api_base=api_base or None,
                                     api_key=api_key or None, temperature=temperature,
                                     max_tokens=max_tokens, system_message=sys_msg,
                                     reasoning_effort=(_eff if _eff and _eff.lower() != "none" else None))]
    # upstream's 60 s LLM timeout is too short for a CLI-backed endpoint editing a bundle
    cfg.llm.timeout = int(os.environ.get("OPENEVOLVE_LLM_TIMEOUT", "1800"))
    cfg.database.num_islands = num_islands
    cfg.database.archive_size = archive_size
    # one nominal dimension; its name must not be an evaluator metric, or upstream's metric
    # branch would take it before the override
    cfg.database.feature_dimensions = [FEATURE_DIM]
    cfg.database.feature_bins = K0                  # re-forced post-construction in the subclass
    cfg.evaluator.cascade_evaluation = False
    cfg.evaluator.parallel_evaluations = 1
    cfg.evaluator.timeout = int(eval_timeout_s)

    log(f"[openevolve-cvt] upstream engine: iterations={n_iters} islands={num_islands} "
        f"archive={archive_size} features=[{FEATURE_DIM}] K0={K0} diff={diff_based}")
    log(f"[openevolve-cvt] partition: {cvt_pins['cvt_dir']} "
        f"k0 requested={cvt_pins['k0_requested']} realized={K0} block_split={block_split}")
    log(f"[openevolve-cvt] model={model} @ {api_base or 'default endpoint'}")

    db_sink: list = []
    result = None
    _orig_db = oe_controller.ProgramDatabase
    # THE PATCH. Scoped to this call and undone in the finally below, so nothing about openevolve
    # stays modified for any other arm running later in the same process.
    oe_controller.ProgramDatabase = _cvt_database_class(_orig_db, cell_of, K0, db_sink)
    try:
        result = run_evolution(initial_program=str(seed_path), evaluator=str(eval_py), config=cfg,
                               iterations=n_iters, output_dir=str(out / "openevolve_run"),
                               cleanup=False, target_score=target_score)
    except C.BudgetExhausted as e:
        log(f"[openevolve-cvt] budget exhausted: {e}")
    except Exception as e:                              # a failed run must still report what it spent
        log(f"[openevolve-cvt] run_evolution raised: {e!r}")
        rl.write("openevolve_error.json", {"error": repr(e)})
    finally:
        oe_controller.ProgramDatabase = _orig_db        # ALWAYS, including on the saturation path
        server.stop()
        server.join(timeout=10)

    # ── occupancy: a CVT arm whose candidates all share one cell is a hill-climb; report it ──
    evaluated = Counter(int(c.meta.get("cell", SENTINEL)) for c in pool if c.fit is not None)
    localized = Counter(int(v) for v in cell_cache.values())
    archive_cells: list = []
    try:
        db = db_sink[0] if db_sink else None
        if db is not None:
            archive_cells = [sorted(m.keys()) for m in getattr(db, "island_feature_maps", [])]
    except Exception as e:
        log(f"[openevolve-cvt] could not read the upstream archive map: {e!r}")

    covered = sum(1 for c in range(K0) if evaluated.get(c, 0) > 0)
    hist = " ".join(f"{c}:{evaluated.get(c, 0)}" for c in range(K0))
    log(f"[openevolve-cvt] cell occupancy over EVALUATED candidates (K0={K0}): {hist}"
        + (f" sentinel:{evaluated.get(SENTINEL, 0)}" if evaluated.get(SENTINEL) else ""))
    log(f"[openevolve-cvt] cells covered {covered}/{K0} "
        f"({100.0 * covered / max(K0, 1):.0f}%); localized(incl. unpaid/refused) "
        f"{sum(1 for c in range(K0) if localized.get(c, 0) > 0)}/{K0}")
    # one evaluated candidate says nothing about the partition
    n_eval_cells = sum(evaluated.values())
    if K0 > 1 and covered <= 1 and n_eval_cells >= 2:
        log(f"[openevolve-cvt] WARNING: the partition COLLAPSED ({n_eval_cells} evaluated candidates in "
            f"{covered} cell). Treat this arm's archive as degenerate: upstream's search ran, but "
            f"MAP-Elites had one niche to compete in. Check the corpus and k0 before quoting it.")
    occupancy = {"k0": K0, "covered_cells": covered,
                 "evaluated_per_cell": {str(c): int(evaluated.get(c, 0)) for c in range(K0)},
                 "localized_per_cell": {str(c): int(localized.get(c, 0)) for c in range(K0)},
                 "sentinel_evaluated": int(evaluated.get(SENTINEL, 0)),
                 "sentinel_localized": int(localized.get(SENTINEL, 0)),
                 "upstream_archive_keys_per_island": archive_cells}
    rl.write("cvt_occupancy.json", occupancy)

    best = None
    if result is not None and getattr(result, "best_code", None):
        h = hashlib.sha1(result.best_code.encode()).hexdigest()[:12]
        best = by_hash.get(h)
        rl.write("openevolve_result.json", {
            "best_score": getattr(result, "best_score", None),
            "metrics": C._jsonable(getattr(result, "metrics", {}) or {}),
            "output_dir": getattr(result, "output_dir", None),
            "best_hash": h, "best_cell": cell_cache.get(h),
            "matched_pool_entry": best is not None})
        (out / "best_program.py").write_text(result.best_code)
    best = best if (best is not None and best.valid) else C.pick_best(pool)

    _n_bad = sum(1 for c in pool if not c.valid)
    if pool and not pool[0].valid:
        log("[openevolve-cvt] DISQUALIFYING: the SEED evaluation did not produce a valid score, so "
            "every candidate after it was compared against a false baseline. Treat this run as "
            "void and re-run; do not quote its numbers.")
    elif _n_bad:
        log(f"[openevolve-cvt] {_n_bad} evaluation(s) failed and are recorded as invalid, not as "
            f"0.0 scores. They are excluded from selection.")

    return C.finalize(adapter, rl, ledger, pool, best,
                      extra={"upstream_output_dir": str(out / "openevolve_run"),
                             "upstream_best_score": getattr(result, "best_score", None)
                             if result else None,
                             "iterations_configured": n_iters,
                             "ablation": "openevolve-cvt",
                             "cvt": cvt_pins, "occupancy": occupancy,
                             # a failed seed evaluation voids the run: every candidate after it
                             # was optimised against a false baseline
                             "n_invalid_evals": sum(1 for c in pool if not c.valid),
                             "seed_valid": (pool[0].valid if pool else None),
                             "best_cell": (best.meta.get("cell") if best is not None else None)},
                      holdout=holdout, logf=log)