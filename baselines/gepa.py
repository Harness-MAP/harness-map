"""Baseline: GEPA via upstream `optimize_anything`, given a seed candidate, evaluator and config."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading

from pathlib import Path

from . import common as C
from .upstream import load_gepa, pins

class _OpenAISDKReflectionLM:
    """A GEPA reflection LM on an OpenAI-compatible endpoint, selected by `openai-sdk[:model]`."""

    def __init__(self, model: str | None = None):
        from openai import OpenAI                         # reads OPENAI_BASE_URL + OPENAI_API_KEY
        self._client = OpenAI()
        self.model = model or os.environ.get("OPENAI_META_MODEL", "")
        if not self.model:
            raise ValueError("reflection-lm openai-sdk: set OPENAI_META_MODEL or pass openai-sdk:<model>")
        self.total_cost = 0.0                             # GEPA cost hook (no per-call pricing here)

    def __call__(self, prompt) -> str:
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        _kw = dict(model=self.model, messages=messages,
                   max_completion_tokens=int(os.environ.get("GEPA_REFLECTION_MAX_TOKENS", "8192")))
        # reasoning effort, as the other arms' backbones run it
        _eff = os.environ.get("OPENAI_REASONING_EFFORT", "high")
        if _eff and _eff.lower() != "none":
            _kw["reasoning_effort"] = _eff
        r = self._client.chat.completions.create(**_kw)
        return r.choices[0].message.content or ""


OBJECTIVE = ("Improve the agent harness so it scores higher on the benchmark's evaluation "
             "instances. The candidate is the harness source code, one component per file.")

BACKGROUND = """\
Each component of the candidate is one source file of an agent harness, keyed by its path relative
to the harness root. The harness is executed by a fixed runner against benchmark instances; it calls
a fixed backbone model and its score on each instance is returned by the evaluator.

Rules that hold no matter what you change:
- Keep the harness's entry point and its module layout importable — the runner imports it as it
  stands. A file that does not compile scores 0 on every instance.
- You may add, restructure or rewrite logic in any component: perception, memory, planning, action
  selection, retries, how much context is sent per call, how many model calls a step costs.
- You may not change the benchmark, the runner, or the scorer — they are outside the candidate.
- The evaluator's feedback for each instance is the real signal: read the per-instance scores and
  diagnostics and target the instances that are actually failing.
"""

TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".cfg", ".toml"}
MAX_COMPONENT_BYTES = 256_000


# ── candidate <-> harness directory ──────────────────────────────────────────
def components_of(harness_dir: Path, suffixes=None, exclude=None) -> dict[str, str]:
    """The harness as a GEPA candidate: {relative path: file text}, on the evolvable surface."""
    allow = {s.lower() for s in suffixes} if suffixes else TEXT_SUFFIXES
    drop = {e for e in (exclude or ())}
    out: dict[str, str] = {}
    for p in sorted(harness_dir.rglob("*")):
        if not p.is_file() or "__pycache__" in p.parts:
            continue
        if p.suffix.lower() not in allow or p.stat().st_size > MAX_COMPONENT_BYTES:
            continue
        if str(p.relative_to(harness_dir)) in drop or p.name in drop:
            continue
        try:
            out[str(p.relative_to(harness_dir))] = p.read_text()
        except UnicodeDecodeError:
            continue
    return out


class Materializer:
    """Candidate dict -> content-addressed harness directory, so a repeat is not charged again."""

    def __init__(self, base_dir: Path, out_root: Path):
        self.base = Path(base_dir)
        self.root = Path(out_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._dirs: dict[str, Path] = {}
        self._lock = threading.Lock()             # --gepa-workers>1 calls materialize concurrently

    @staticmethod
    def key(candidate: dict) -> str:
        blob = json.dumps({k: v for k, v in sorted(candidate.items())}, sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()[:12]

    def materialize(self, candidate: dict) -> tuple[Path, bool]:
        """Returns (harness_dir, is_new); locked, as GEPA's workers call it concurrently."""
        h = self.key(candidate)
        with self._lock:
            if h in self._dirs:
                return self._dirs[h], False
            d = self.root / f"cand_{h}"
            if not d.exists():
                # copy the anchor first so non-component assets survive, then overlay the components
                shutil.copytree(self.base, d,
                                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
                for rel, text in candidate.items():
                    f = d / rel
                    f.parent.mkdir(parents=True, exist_ok=True)
                    f.write_text(text)
            self._dirs[h] = d
            return d, True


# ── the run ───────────────────────────────────────────────────────────────────
def run(adapter, *, out_dir, ledger: C.Ledger, minibatch_size: int = 3, front: str = "instance",
        seed: int = 0, perfect_score: float = 1.0, skip_perfect_score: bool = True,
        use_merge: bool = False, proposer: str = "claude-code", proposer_model: str = "opus",
        reflection_lm: str = "", sandbox: bool = True, max_workers: int = 1,
        stop_at_score=None, holdout: bool = True, log=print, **_) -> dict:
    """Run upstream `optimize_anything`; the winner is re-scored on the full split and reported."""
    load_gepa()
    from gepa.gepa_launcher import EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything
    from harness_map.main import Genome

    out = Path(out_dir)
    run_dir = out / "gepa_run"                    # upstream's workspace: iterations/, pareto/, state
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset = list(getattr(adapter, "evolve", None) or [])
    if not dataset:
        raise ValueError("the testbed adapter exposes no `evolve` split for GEPA to optimise over")

    anchor = adapter.anchor()
    anchor_dir = Path(anchor.source_dir)
    seed_candidate = components_of(anchor_dir, suffixes=getattr(adapter, "bundle_suffixes", None),
                                   exclude=getattr(adapter, "bundle_exclude", None))
    if not seed_candidate:
        raise ValueError(f"no text components found in the anchor harness {anchor_dir}")
    mat = Materializer(anchor_dir, out / "candidates")

    rl = C.RunLog(out_dir, "gepa", {
        "engine": "upstream gepa.gepa_launcher.optimize_anything (optimize-anything API)",
        "mode": "generalization (dataset + valset)", "proposer": proposer,
        "proposer_model": proposer_model if proposer == "claude-code" else reflection_lm,
        "sandbox": sandbox, "minibatch_size": minibatch_size, "frontier_type": front,
        "seed": seed, "use_merge": use_merge, "components": sorted(seed_candidate),
        "upstream": pins()["gepa"],
        "skill": "vendor/gepa/.claude/skills/gepa-optimize-anything"})

    pool: list[C.Cand] = []
    by_hash: dict[str, C.Cand] = {}
    state_lock = threading.Lock()                  # guards pool/by_hash under --gepa-workers>1

    def evaluator(candidate, example=None, **kw):
        """Upstream's `(score, info)` contract, one (candidate, example) pair per call."""
        d, is_new = mat.materialize(dict(candidate))
        h = mat.key(dict(candidate))
        genome = Genome(source_dir=d, label=f"cand_{h}")
        tid = C._task_id(example) if example is not None else None

        if is_new and not C.validate(adapter, genome):
            with state_lock:
                ledger.spend(0, new_candidate=True)
                by_hash.setdefault(h, C.Cand(idx=len(pool), genome=genome, fit=None, meta={"role": "invalid"}))
                if by_hash[h] not in pool:
                    pool.append(by_hash[h])
            return 0.0, {"score": 0.0, "error": "candidate does not compile — every instance scores 0"}

        fit, _ = C.evaluate_on(adapter, genome, rid=ledger.metric_calls,   # OUTSIDE the lock (parallel)
                               ids=([tid] if tid else None), ledger=None)

        scores = C.per_instance(fit)
        score = float(scores.get(tid, fit.primary if fit is not None else 0.0)) if tid \
            else float(fit.primary if fit is not None else 0.0)

        with state_lock:
            ledger.spend(1 if tid else max(len(dataset), 1), new_candidate=is_new)
            cand = by_hash.get(h)
            if cand is None:
                cand = C.Cand(idx=len(pool), genome=genome, fit=fit, scores={}, meta={"role": "candidate"})
                pool.append(cand)
                by_hash[h] = cand
            cand.scores.update(scores)             # accumulates as GEPA walks the val instances
            if cand.fit is None or (fit is not None and len(scores) >= len(C.per_instance(cand.fit))):
                cand.fit = fit
        return score, {"score": round(score, 6), "instance": tid,
                       "verdict": _verdict(score, perfect_score),
                       "diagnostics": C._jsonable(dict(getattr(fit, "dense", None) or {}))}

    # ── the proposer: upstream's Claude Code agent, or a reflection LM ────────
    reflection = dict(reflection_minibatch_size=minibatch_size,
                      skip_perfect_score=skip_perfect_score, perfect_score=perfect_score)
    if proposer == "claude-code":
        from gepa.oa.proposers import ClaudeCodeAgentProposer
        reflection["custom_candidate_proposer"] = ClaudeCodeAgentProposer(
            model=proposer_model, run_dir=str(run_dir), objective=OBJECTIVE,
            background=BACKGROUND + _bench_background(adapter), sandbox=sandbox)
    elif callable(reflection_lm):
        # GEPA's LM protocol is just `__call__(prompt) -> str`, so a caller may pass any callable
        # (a self-hosted client, a test double) instead of a model-id string.
        reflection["reflection_lm"] = reflection_lm
    elif str(reflection_lm).startswith("openai-sdk"):   # OpenAI SDK route, no litellm
        model = reflection_lm.split(":", 1)[1] if ":" in reflection_lm else None
        reflection["reflection_lm"] = _OpenAISDKReflectionLM(model)
    else:
        reflection["reflection_lm"] = reflection_lm or "openai/gpt-5.1"

    # Convergence stop: once the parent is perfect the engine skips mutation every iteration and
    # evaluates nothing, so the budget never fills; stop after GEPA_NOPROG_CAP flat iterations.
    # keyed on cand_evals, which stays flat while the engine skips (total_num_evals does not)
    _noprog = {"last_cand": -1, "stale": 0}
    _NOPROG_CAP = int(os.environ.get("GEPA_NOPROG_CAP", "60"))
    def _stop_cb(state):
        if ledger.exhausted():
            return True
        cur = ledger.cand_evals
        if cur != _noprog["last_cand"]:
            _noprog["last_cand"] = cur
            _noprog["stale"] = 0
        else:
            _noprog["stale"] += 1
            if _noprog["stale"] >= _NOPROG_CAP:
                log(f"[gepa] no-progress stop: {_noprog['stale']} iterations with cand_evals flat at "
                    f"{cur} (converged, proposer stopped producing new candidates); finalizing")
                return True
        return False

    cfg = GEPAConfig(
        engine=EngineConfig(
            run_dir=str(run_dir), seed=seed,
            max_metric_calls=ledger.max_metric_calls,
            stop_at_score=stop_at_score,
            frontier_type=front,
            candidate_selection_strategy="pareto",
            acceptance_criterion="strict_improvement",
            # the Claude Code proposer reads this tree
            write_agent_state=True,
            parallel=max_workers > 1, max_workers=max_workers,
            raise_on_exception=False,          # a crashed candidate scores 0 instead of killing the run
            display_progress_bar=False),
        reflection=ReflectionConfig(**reflection),
        stop_callbacks=[_stop_cb])

    log(f"[gepa] optimize_anything: {run_dir}")
    log(f"[gepa] mode=generalization dataset={len(dataset)} valset={len(dataset)} "
        f"components={len(seed_candidate)} minibatch={minibatch_size} frontier={front}")
    log(f"[gepa] proposer={proposer}"
        + (f" model={proposer_model} sandbox={sandbox}" if proposer == "claude-code" else ""))

    result = None
    try:
        result = optimize_anything(
            seed_candidate=seed_candidate,
            evaluator=evaluator,
            dataset=dataset,
            valset=dataset,                    # selection split == the search split, as on every arm
            objective=OBJECTIVE,
            background=BACKGROUND + _bench_background(adapter),
            config=cfg)
    except C.BudgetExhausted as e:
        log(f"[gepa] budget exhausted: {e}")
    except Exception as e:                     # a failed run must still report what it spent
        log(f"[gepa] optimize_anything raised: {e!r}")
        rl.write("gepa_error.json", {"error": repr(e)})

    best = None
    if result is not None and getattr(result, "best_candidate", None):
        d, _ = mat.materialize(dict(result.best_candidate))
        best = by_hash.get(mat.key(dict(result.best_candidate)))
        if best is None:
            best = C.Cand(idx=len(pool), genome=Genome(source_dir=d, label=f"best_{mat.key(dict(result.best_candidate))}"),
                          fit=None)
        rl.write("gepa_result.json", {
            "best_idx": getattr(result, "best_idx", None),
            "val_aggregate_scores": list(getattr(result, "val_aggregate_scores", []) or []),
            "parents": C._jsonable(getattr(result, "parents", []) or []),
            "total_metric_calls": getattr(result, "total_metric_calls", None),
            "num_full_val_evals": getattr(result, "num_full_val_evals", None),
            "per_val_instance_best_candidates":
                C._jsonable(getattr(result, "per_val_instance_best_candidates", {}) or {}),
            "n_candidates": len(getattr(result, "candidates", []) or []),
            "best_components": sorted(dict(result.best_candidate))})
        rl.write("best_candidate.json", C._jsonable(dict(result.best_candidate)))
    best = best if (best is not None and best.valid) else C.pick_best(pool)

    # GEPA scores per example, so the champion is re-scored on the full split (not charged) to be
    # comparable with the anchor and the other arms.
    if best is not None and best.genome is not None and best.valid:
        log(f"[gepa] re-scoring champion {best.label} on the full split for a comparable primary")
        full, _ = C.evaluate_on(adapter, best.genome, rid=ledger.metric_calls, ledger=None)
        if full is not None:
            best = C.Cand(idx=best.idx, genome=best.genome, fit=full, scores=C.per_instance(full),
                          meta={**(best.meta or {}), "rescored_full": True})

    for c in pool:
        rl.log(c, ledger)
    return C.finalize(adapter, rl, ledger, pool, best,
                      extra={"upstream_run_dir": str(run_dir),
                             "upstream_total_metric_calls":
                                 getattr(result, "total_metric_calls", None) if result else None,
                             "upstream_n_candidates":
                                 len(getattr(result, "candidates", []) or []) if result else 0,
                             "proposer": proposer},
                      holdout=holdout, logf=log)


def _verdict(score: float, perfect: float) -> str:
    if score >= perfect:
        return "solved"
    return "no progress at all on this instance" if score <= 0 else "partial credit only"


def _bench_background(adapter) -> str:
    n = len(getattr(adapter, "evolve", None) or [])
    return (f"\nThis run optimises over {n} evaluation instance(s). Their ids appear in the "
            f"per-instance feedback.\n")
