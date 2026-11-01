"""Baseline: ShinkaEvolve, upstream's `ShinkaEvolveRunner` on a one-file bundle of the harness."""
from __future__ import annotations

import hashlib
import json
import os
import queue
import sys
import threading

from pathlib import Path

from . import common as C
from .upstream import load_shinka, pins


EVALUATE_PY = '''\
"""Generated ShinkaEvolve evaluator — upstream's contract, backed by the arm's live adapter.

Upstream calls this as `evaluate.py --program_path <candidate> --results_dir <dir>` and reads back
`metrics.json` + `correct.json`. The scoring itself needs the testbed adapter, which lives in the
arm process, so this hands the candidate over a file-based request/response pair and waits.
"""
import argparse
import json
import time
import uuid
from pathlib import Path

RPC = Path(__file__).resolve().parent / "rpc"
TIMEOUT_S = {timeout}
POLL_S = 0.25


def main(program_path: str, results_dir: str) -> None:
    rid = uuid.uuid4().hex[:12]
    req_dir, resp_dir = RPC / "requests", RPC / "responses"
    req_dir.mkdir(parents=True, exist_ok=True)
    resp_dir.mkdir(parents=True, exist_ok=True)

    tmp = req_dir / ("." + rid + ".tmp")            # write-then-rename: the server never sees a
    tmp.write_text(json.dumps({{"program_path": str(program_path)}}))   # half-written request
    tmp.rename(req_dir / (rid + ".json"))

    resp, deadline = resp_dir / (rid + ".json"), time.time() + TIMEOUT_S
    payload = None
    while time.time() < deadline:
        if resp.exists():
            try:
                payload = json.loads(resp.read_text())
                break
            except json.JSONDecodeError:
                time.sleep(POLL_S)                  # still being written; try again
                continue
        time.sleep(POLL_S)
    if payload is None:
        payload = {{"metrics": {{"combined_score": 0.0}}, "correct": False,
                   "error": "timed out waiting for the evaluation server"}}

    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(payload["metrics"], indent=2))
    (out / "correct.json").write_text(json.dumps(
        {{"correct": bool(payload.get("correct")), "error": payload.get("error", "")}}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--program_path", required=True)
    ap.add_argument("--results_dir", required=True)
    a = ap.parse_args()
    main(a.program_path, a.results_dir)
'''


class _EvalServer(threading.Thread):
    """Serves the generated evaluate.py's requests from the process that owns the adapter."""

    daemon = True

    def __init__(self, rpc_dir: Path, handler, log=print):
        super().__init__(name="shinka-eval-server")
        self.req = rpc_dir / "requests"
        self.resp = rpc_dir / "responses"
        self.req.mkdir(parents=True, exist_ok=True)
        self.resp.mkdir(parents=True, exist_ok=True)
        self.handler = handler
        self.log = log
        # NOT `_stop`: threading.Thread has a private `_stop()` method that `join()` calls, and
        # shadowing it with an Event makes every join() raise TypeError.
        self._stopping = threading.Event()
        self.errors: queue.Queue = queue.Queue()

    def stop(self):
        self._stopping.set()

    def run(self):
        seen: set[str] = set()
        while not self._stopping.is_set():
            for f in sorted(self.req.glob("*.json")):
                if f.name in seen:
                    continue
                seen.add(f.name)
                try:
                    req = json.loads(f.read_text())
                    metrics, correct, err = self.handler(req["program_path"])
                except Exception as e:                       # never take the runner down
                    metrics, correct, err = {"combined_score": 0.0}, False, repr(e)
                    self.errors.put(e)
                    self.log(f"[shinka] evaluation failed: {e!r}")
                payload = {"metrics": metrics, "correct": correct, "error": err}
                tmp = self.resp / ("." + f.stem + ".tmp")
                tmp.write_text(json.dumps(payload))
                tmp.rename(self.resp / f.name)               # atomic for the waiting client
            self._stopping.wait(0.2)


def run(adapter, *, out_dir, ledger: C.Ledger, generations: int = 0, num_islands: int = 2,
        archive_size: int = 20, models: str = "", patch_types=None, patch_type_probs=None,
        embedding_model: str = "", max_novelty_attempts: int = 3, llm_selection: str = "ucb",
        max_parallel: int = 1, eval_timeout_s: int = 7200, holdout: bool = True, log=print,
        **_) -> dict:
    """Hand the bundled anchor to upstream `ShinkaEvolveRunner` and report the winner."""
    load_shinka()
    from shinka.core import EvolutionConfig, ShinkaEvolveRunner
    from shinka.database import DatabaseConfig
    from shinka.launch import LocalJobConfig
    from harness_map.main import Genome

    out = Path(out_dir)
    task = out / "shinka_task"
    task.mkdir(parents=True, exist_ok=True)
    anchor = adapter.anchor()
    anchor_dir = Path(anchor.source_dir)
    (task / "initial.py").write_text(
        C.bundle_harness(anchor_dir, suffixes=getattr(adapter, "bundle_suffixes", None),
                         exclude=getattr(adapter, "bundle_exclude", None)))
    (task / "evaluate.py").write_text(EVALUATE_PY.format(timeout=eval_timeout_s))

    n_gen = generations or (ledger.max_cand_evals * 10 if ledger.max_cand_evals else 0) or 12  # *10 so the CAND-EVAL budget binds, not the generation count
    model_list = [m.strip() for m in (models or _default_models()).split(",") if m.strip()]
    # Patch operators: upstream's native diff / full / cross mix unless SHINKA_PATCH_TYPES
    # overrides it (e.g. "full" for whole rewrites).
    if not patch_types:
        env_pt = os.environ.get("SHINKA_PATCH_TYPES", "").strip()
        if env_pt:
            patch_types = [t.strip() for t in env_pt.split(",") if t.strip()]
    # max_patch_attempts > 1: upstream re-queries the model with the parser's error on a failed diff
    max_patch_attempts = int(os.environ.get("SHINKA_MAX_PATCH_ATTEMPTS", "3"))
    # task_sys_msg is prepended to upstream's diff format message: the diff discipline goes here
    task_sys_msg = (
        "You are improving an agent-harness bundle so a frozen agent scores higher on a benchmark. "
        "The program is the whole harness package folded into one file (one EVOLVE-BLOCK region per "
        "source file); it is split back into a runnable package and scored per instance, and "
        "combined_score (the mean over instances) is what you maximize.\n\n" + C.DIFF_DISCIPLINE)

    from shinka.defaults import default_patch_types, default_patch_type_probs
    eff_patch_types = list(patch_types or default_patch_types())          # native = diff/full/cross
    eff_patch_type_probs = (list(patch_type_probs) if patch_type_probs
                            else (default_patch_type_probs() if not patch_types else None))
    rl = C.RunLog(out_dir, "shinka", {
        "engine": "upstream shinka.core.ShinkaEvolveRunner",
        "generations": n_gen, "num_islands": num_islands, "archive_size": archive_size,
        "llm_models": model_list, "llm_dynamic_selection": llm_selection,
        "patch_types": eff_patch_types,
        "patch_type_probs": eff_patch_type_probs,
        "max_patch_attempts": max_patch_attempts,
        "embedding_model": embedding_model or None,
        "max_novelty_attempts": max_novelty_attempts,
        "upstream": pins()["shinkaevolve"],
        "candidate": "whole-package bundle, one EVOLVE-BLOCK region per source file",
        "evaluator": "upstream subprocess contract (metrics.json/correct.json), served over a "
                     "file RPC by the arm process that owns the adapter"})

    pool: list[C.Cand] = []
    by_hash: dict[str, C.Cand] = {}

    lock = threading.Lock()

    def handle(program_path: str):
        """Score one candidate; returns upstream's (metrics, correct, error) triple."""
        text = Path(program_path).read_text()
        h = hashlib.sha1(text.encode()).hexdigest()[:12]
        with lock:
            known = by_hash.get(h)
        if known is not None and known.fit is not None:
            return _metrics(known), True, ""

        # Budget stop: upstream has no budget hook and an exception here is swallowed by the RPC
        # thread, so past the cap new candidates are refused a paid evaluation (renewable cap).
        cap = ledger._cap() if hasattr(ledger, "_cap") else ledger.max_cand_evals
        if cap and ledger.cand_evals >= cap:
            return {"combined_score": 0.0, "budget_exhausted": True}, False, "budget exhausted"

        d = C.unbundle_harness(text, anchor_dir, out / "candidates" / f"cand_{h}")
        genome = Genome(source_dir=d, label=f"cand_{h}")
        if not C.validate(adapter, genome):
            ledger.spend(0)
            cand = C.Cand(idx=len(pool), genome=genome, fit=None, meta={"role": "invalid"})
            with lock:
                pool.append(cand); by_hash[h] = cand
            rl.log(cand, ledger)
            log(f"[shinka] {genome.label}: does not compile -> 0.0")
            return {"combined_score": 0.0}, False, "candidate does not compile"

        fit, _ = C.evaluate_on(adapter, genome, rid=ledger.cand_evals, ledger=ledger)
        cand = C.Cand(idx=len(pool), genome=genome, fit=fit, scores=C.per_instance(fit),
                      meta={"role": "candidate", "hash": h})
        with lock:
            pool.append(cand); by_hash[h] = cand
        rl.log(cand, ledger)
        log(f"[shinka] {genome.label}: combined_score={cand.primary:.4f} "
            f"({ledger.cand_evals}/{ledger.max_cand_evals or '-'} cand evals)")
        # A perfect 1.0 cannot be improved: stop, as every arm does.
        if (cand.primary or 0) >= 1.0:
            log(f"[shinka] SATURATED at {ledger.cand_evals} evals "
                f"(best={cand.primary:.3f}) — stopping early.")
            raise C.BudgetExhausted(f"saturated at 1.0 after {ledger.cand_evals} cand evals")
        return _metrics(cand), bool(cand.valid), ""

    server = _EvalServer(task / "rpc", handle, log=log)
    server.start()

    job_config = LocalJobConfig(eval_program_path=str(task / "evaluate.py"),
                                python_executable=sys.executable)   # this env, not the sidecar
    # upstream rewrites db_path to <results_dir>/programs.sqlite
    db_config = DatabaseConfig(db_path=str(out / "evolution_db.sqlite"),
                               num_islands=num_islands, archive_size=archive_size)
    evo_kwargs = dict(num_generations=n_gen, job_type="local", language="python",
                      llm_models=model_list, llm_dynamic_selection=llm_selection,
                      init_program_path=str(task / "initial.py"),
                      results_dir=str(out / "shinka_results"),
                      max_novelty_attempts=max_novelty_attempts,
                      max_patch_attempts=max_patch_attempts, task_sys_msg=task_sys_msg)
    if patch_types:
        evo_kwargs["patch_types"] = list(patch_types)
        # patch_type_probs must match patch_types in length (upstream's default is 3-long)
        if patch_type_probs:
            evo_kwargs["patch_type_probs"] = list(patch_type_probs)
        else:
            evo_kwargs["patch_type_probs"] = [1.0 / len(patch_types)] * len(patch_types)
    evo_kwargs["embedding_model"] = embedding_model or None    # None = novelty rejection off
    # reasoning effort, as the other arms' backbones run it (read from llm_kwargs upstream)
    _eff = os.environ.get("OPENAI_REASONING_EFFORT", "high")
    evo_kwargs["llm_kwargs"] = {"temperatures": [0.0, 0.5, 1.0], "max_tokens": 16384,
                                "reasoning_efforts": [_eff if _eff.lower() != "none" else "disabled"]}
    evo_config = EvolutionConfig(**evo_kwargs)

    log(f"[shinka] upstream runner: generations={n_gen} islands={num_islands} "
        f"archive={archive_size} selection={llm_selection} novelty={embedding_model or 'off'}")
    log(f"[shinka] models={model_list}")

    try:
        ShinkaEvolveRunner(evo_config=evo_config, job_config=job_config, db_config=db_config,
                           max_evaluation_jobs=max_parallel, max_proposal_jobs=max_parallel,
                           max_db_workers=max_parallel, verbose=True).run()
    except C.BudgetExhausted as e:
        log(f"[shinka] budget exhausted: {e}")
    except Exception as e:                            # a failed run must still report what it spent
        log(f"[shinka] ShinkaEvolveRunner raised: {e!r}")
        rl.write("shinka_error.json", {"error": repr(e)})
    finally:
        server.stop()
        server.join(timeout=10)

    best = C.pick_best(pool)
    return C.finalize(adapter, rl, ledger, pool, best,
                      extra={"upstream_results_dir": str(out / "shinka_results"),
                             "upstream_db": str(out / "evolution_db.sqlite"),
                             "generations_configured": n_gen,
                             "task_dir": str(task)},
                      holdout=holdout, logf=log)


def _default_models() -> str:
    """Mutation model from the environment: an OpenAI-compatible endpoint, else headless claude."""
    base, model = os.environ.get("OPENAI_BASE_URL", ""), os.environ.get("OPENAI_META_MODEL", "")
    if base and model:
        # ?api_key_env names the env var ShinkaEvolve reads the bearer key from; without it the
        # provider sends no key. The parser strips the query param before the request.
        return f"local/{model}@{base.rstrip('/')}?api_key_env=OPENAI_API_KEY"
    return "headless/claude"


def _metrics(cand: C.Cand) -> dict:
    """upstream ranks on `combined_score`; `public` is what its prompts show the mutation model."""
    return {"combined_score": round(cand.primary, 6),
            "public": {"combined_score": round(cand.primary, 6),
                       **{f"inst_{k}": round(v, 6) for k, v in cand.scores.items()}},
            "private": {}}
