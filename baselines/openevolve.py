"""Baseline: OpenEvolve, running upstream `run_evolution` on a one-file bundle of the harness."""
from __future__ import annotations

import hashlib
import json
import os
import queue
import threading

from pathlib import Path

from . import common as C
from .upstream import load_openevolve, pins


# OpenEvolve runs its evaluator in a process pool, which cannot carry the live adapter, so it
# gets this self-contained evaluator file that reaches the adapter over a file RPC served by
# the arm process (the same design as the shinka arm).
EVALUATE_PY = '''\
"""Generated OpenEvolve evaluator — upstream calls module.evaluate(program_path) -> metrics dict.
The scoring needs the testbed adapter, which lives in the arm process, so this hands the candidate
over a file request/response pair and returns the metrics the server computes."""
import json
import time
import uuid
from pathlib import Path

RPC = Path(__file__).resolve().parent / "rpc"
STALE_S = {timeout}        # give up only after the server's heartbeat is this stale
GRACE_S = 120              # never declare death before the server has had a chance to start
POLL_S = 0.25


def evaluate(program_path):
    t0 = time.time()
    rid = uuid.uuid4().hex[:12]
    req_dir, resp_dir = RPC / "requests", RPC / "responses"
    req_dir.mkdir(parents=True, exist_ok=True)
    resp_dir.mkdir(parents=True, exist_ok=True)
    tmp = req_dir / ("." + rid + ".tmp")                # write-then-rename: the server never sees a
    tmp.write_text(json.dumps({{"program_path": str(program_path)}}))   # half-written request
    tmp.rename(req_dir / (rid + ".json"))
    # Liveness, not a deadline: a slow evaluation under load must not be read as a dead server,
    # so wait while the server's heartbeat is fresh and give up only once it goes stale.
    hb_dir = RPC / "heartbeat"
    resp = resp_dir / (rid + ".json")
    while True:
        if resp.exists():
            try:
                return json.loads(resp.read_text())["metrics"]
            except (json.JSONDecodeError, KeyError):
                time.sleep(POLL_S)                       # still being written; try again
                continue
        hb = hb_dir / (rid + ".hb")
        alive = hb.exists() and (time.time() - hb.stat().st_mtime) < STALE_S
        if not alive and time.time() - t0 > GRACE_S:
            # invalid=True is the load-bearing flag: the arm must NOT record this as a real 0.0.
            return {{"combined_score": 0.0, "invalid": True,
                    "error": "evaluation server heartbeat went stale"}}
        time.sleep(POLL_S)
'''


class _EvalServer(threading.Thread):
    """Serves the generated evaluate.py's requests from the process that owns the adapter."""

    daemon = True

    def __init__(self, rpc_dir: Path, handler, log=print):
        super().__init__(name="openevolve-eval-server")
        self.req = rpc_dir / "requests"
        self.resp = rpc_dir / "responses"
        self.hb = rpc_dir / "heartbeat"
        for d in (self.req, self.resp, self.hb):
            d.mkdir(parents=True, exist_ok=True)
        self.handler = handler
        self.log = log
        self._stopping = threading.Event()               # not `_stop`: Thread.join() calls _stop()
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
                # beat on a side thread while the handler blocks: the client tells slow from dead
                beat_stop = threading.Event()
                hb = self.hb / (f.stem + ".hb")
                hb.parent.mkdir(parents=True, exist_ok=True)

                def _beat(path=hb, stop=beat_stop):
                    while not stop.is_set():
                        try:
                            path.touch()
                        except OSError:
                            pass
                        stop.wait(5.0)

                beater = threading.Thread(target=_beat, daemon=True,
                                          name=f"eval-heartbeat-{f.stem}")
                beater.start()
                try:
                    req = json.loads(f.read_text())
                    metrics = self.handler(req["program_path"])
                except Exception as e:                       # never take the runner down
                    # invalid=True so the arm records a FAILED evaluation rather than a real 0.0.
                    metrics = {"combined_score": 0.0, "invalid": True, "error": repr(e)}
                    self.errors.put(e)
                    self.log(f"[openevolve] evaluation failed: {e!r}")
                finally:
                    beat_stop.set()
                    beater.join(timeout=10)
                    hb.unlink(missing_ok=True)
                tmp = self.resp / ("." + f.stem + ".tmp")
                tmp.write_text(json.dumps({"metrics": metrics}))
                tmp.rename(self.resp / f.name)               # atomic for the waiting client
            self._stopping.wait(0.2)


SYSTEM_MESSAGE = """\
You are improving an agent harness so it scores higher on a benchmark.

The program you are editing is a BUNDLE: the whole harness package folded into one file, one
`# ===== FILE: <path> =====` marker per source file, each wrapped in EVOLVE-BLOCK markers. It is
split back into a runnable package before it is scored, so:
- keep every `# ===== FILE: ... =====` marker and the EVOLVE-BLOCK markers exactly as they are,
- change anything you like inside a region; a region you do not want to touch, leave alone,
- code that does not compile scores 0 on every instance.

The harness is imported and run by a fixed runner against benchmark instances, calls a fixed
backbone model, and is scored per instance. You may not change the benchmark, the runner or the
scorer. `combined_score` is the mean score over the instances — that is what you are maximising.
"""


def run(adapter, *, out_dir, ledger: C.Ledger, iterations: int = 0, num_islands: int = 2,
        archive_size: int = 20, feature_dimensions=None, feature_bins: int = 10,
        diff_based: bool = True, model: str = "", api_base: str = "", api_key: str = "",
        temperature: float = 0.7, max_tokens: int = 16000, target_score=None,
        eval_timeout_s: int = 7200, holdout: bool = True, log=print, **_) -> dict:
    """Hand the bundled anchor to upstream `run_evolution` and report the winner."""
    load_openevolve()
    from openevolve.api import run_evolution
    from openevolve.config import Config, LLMModelConfig
    from harness_map.main import Genome

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    anchor = adapter.anchor()
    anchor_dir = Path(anchor.source_dir)
    seed_path = out / "initial.py"
    seed_path.write_text(C.bundle_harness(anchor_dir, suffixes=getattr(adapter, "bundle_suffixes", None),
                                          exclude=getattr(adapter, "bundle_exclude", None)))

    # 2x headroom: clones and invalid programs cost iterations but no evaluation; `handle`
    # refuses to pay past the cap, so the budget is reached but not exceeded.
    n_iters = iterations or (ledger.max_cand_evals * 2 if ledger.max_cand_evals else 0) or 12
    model = model or os.environ.get("OPENEVOLVE_MODEL") or os.environ.get("OPENAI_META_MODEL", "")
    api_base = api_base or os.environ.get("OPENAI_BASE_URL", "")
    api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not model:
        raise ValueError(
            "OpenEvolve needs an LLM. Set OPENAI_META_MODEL (+ OPENAI_BASE_URL / OPENAI_API_KEY) "
            "or pass --openevolve-model; upstream drives an OpenAI-compatible endpoint.")

    rl = C.RunLog(out_dir, "openevolve", {
        "engine": "upstream openevolve.api.run_evolution",
        "iterations": n_iters, "num_islands": num_islands, "archive_size": archive_size,
        "feature_dimensions": feature_dimensions or ["complexity", "diversity"],
        "feature_bins": feature_bins, "diff_based_evolution": diff_based,
        "model": model, "api_base": api_base, "upstream": pins()["openevolve"],
        "candidate": "whole-package bundle, one EVOLVE-BLOCK region per source file"})

    pool: list[C.Cand] = []
    by_hash: dict[str, C.Cand] = {}
    lock = threading.Lock()

    def handle(program_path) -> dict:
        """Score one candidate -> upstream's metrics dict (in this process, via the RPC server)."""
        text = Path(program_path).read_text()
        h = hashlib.sha1(text.encode()).hexdigest()[:12]
        with lock:
            known = by_hash.get(h)
        if known is not None:                            # upstream re-scores clones; don't re-pay
            if known.fit is not None:
                return _metrics(known)
            return {"combined_score": 0.0, "error": "candidate does not compile"}   # known-invalid clone

        # Budget stop: upstream has no budget hook and an exception here is swallowed by the RPC
        # thread, so past the cap new candidates are refused a paid evaluation (renewable cap).
        cap = ledger._cap() if hasattr(ledger, "_cap") else ledger.max_cand_evals
        if cap and ledger.cand_evals >= cap:
            return {"combined_score": 0.0, "budget_exhausted": True}

        d = C.unbundle_harness(text, anchor_dir, out / "candidates" / f"cand_{h}")
        genome = Genome(source_dir=d, label=f"cand_{h}")
        if not C.validate(adapter, genome):
            ledger.spend(0)
            cand = C.Cand(idx=len(pool), genome=genome, fit=None, meta={"role": "invalid"})
            with lock:
                pool.append(cand); by_hash[h] = cand
            rl.log(cand, ledger)
            log(f"[openevolve] {genome.label}: does not compile -> 0.0")
            return {"combined_score": 0.0, "error": "candidate does not compile"}

        fit, _ = C.evaluate_on(adapter, genome, rid=ledger.cand_evals, ledger=ledger)
        cand = C.Cand(idx=len(pool), genome=genome, fit=fit, scores=C.per_instance(fit),
                      meta={"role": "candidate", "hash": h})
        with lock:
            pool.append(cand); by_hash[h] = cand
        rl.log(cand, ledger)
        log(f"[openevolve] {genome.label}: combined_score={cand.primary:.4f} "
            f"({ledger.cand_evals}/{ledger.max_cand_evals or '-'} cand evals)")
        # A perfect 1.0 cannot be improved: stop, as every arm does. Raising here unwinds the
        # engine call into the arm's own except.
        if (cand.primary or 0) >= 1.0:
            log(f"[openevolve] SATURATED at {ledger.cand_evals} evals "
                f"(best={cand.primary:.3f}) — stopping early.")
            raise C.BudgetExhausted(f"saturated at 1.0 after {ledger.cand_evals} cand evals")
        return _metrics(cand)

    # OpenEvolve runs THIS evaluator file in its subprocess pool; it RPC-bridges to `handle` above.
    eval_py = out / "evaluate.py"
    eval_py.write_text(EVALUATE_PY.format(timeout=int(eval_timeout_s)))
    server = _EvalServer(out / "rpc", handle, log=log)
    server.start()

    cfg = Config()
    cfg.language = "python"
    cfg.max_iterations = n_iters
    # upstream's default of 100 would write no checkpoint at these budgets (OE_CHECKPOINT_INTERVAL)
    cfg.checkpoint_interval = int(os.environ.get("OE_CHECKPOINT_INTERVAL", "10"))
    cfg.diff_based_evolution = diff_based
    # upstream's 10,000-char program cap is sized for kernels; a bundle is several times that
    cfg.max_code_length = max(8 * len(seed_path.read_text()), 100_000)
    # in diff mode, hold the model to diffs upstream's exact-match parser can apply
    sys_msg = SYSTEM_MESSAGE + ("\n\n" + C.DIFF_DISCIPLINE if diff_based else "")
    # the prompt sampler reads cfg.prompt.system_message, not the model config's
    cfg.prompt.system_message = sys_msg
    # reasoning effort, as the other arms' backbones run it
    _eff = os.environ.get("OPENAI_REASONING_EFFORT", "high")
    cfg.llm.models = [LLMModelConfig(name=model, api_base=api_base or None,
                                     api_key=api_key or None, temperature=temperature,
                                     max_tokens=max_tokens, system_message=sys_msg,
                                     reasoning_effort=(_eff if _eff and _eff.lower() != "none" else None))]
    # upstream's 60 s LLM timeout is too short for a CLI-backed endpoint editing a bundle
    cfg.llm.timeout = int(os.environ.get("OPENEVOLVE_LLM_TIMEOUT", "1800"))
    cfg.database.num_islands = num_islands
    cfg.database.archive_size = archive_size
    cfg.database.feature_dimensions = list(feature_dimensions or ["complexity", "diversity"])
    cfg.database.feature_bins = feature_bins
    # one evaluation per candidate: the cascade would spend a variable number
    cfg.evaluator.cascade_evaluation = False
    cfg.evaluator.parallel_evaluations = 1
    # Upstream caps one evaluation at 300 s, sized for numeric kernels; a harness evaluation plays
    # episodes and takes far longer, so the bound follows the benchmark's own timeout.
    cfg.evaluator.timeout = int(eval_timeout_s)

    log(f"[openevolve] upstream engine: iterations={n_iters} islands={num_islands} "
        f"archive={archive_size} features={cfg.database.feature_dimensions} "
        f"bins={feature_bins} diff={diff_based}")
    log(f"[openevolve] model={model} @ {api_base or 'default endpoint'}")

    result = None
    try:
        result = run_evolution(initial_program=str(seed_path), evaluator=str(eval_py), config=cfg,
                               iterations=n_iters, output_dir=str(out / "openevolve_run"),
                               cleanup=False, target_score=target_score)
    except C.BudgetExhausted as e:
        log(f"[openevolve] budget exhausted: {e}")
    except Exception as e:                              # a failed run must still report what it spent
        log(f"[openevolve] run_evolution raised: {e!r}")
        rl.write("openevolve_error.json", {"error": repr(e)})
    finally:
        server.stop()
        server.join(timeout=10)

    best = None
    if result is not None and getattr(result, "best_code", None):
        h = hashlib.sha1(result.best_code.encode()).hexdigest()[:12]
        best = by_hash.get(h)
        rl.write("openevolve_result.json", {
            "best_score": getattr(result, "best_score", None),
            "metrics": C._jsonable(getattr(result, "metrics", {}) or {}),
            "output_dir": getattr(result, "output_dir", None),
            "best_hash": h, "matched_pool_entry": best is not None})
        (out / "best_program.py").write_text(result.best_code)
    best = best if (best is not None and best.valid) else C.pick_best(pool)

    return C.finalize(adapter, rl, ledger, pool, best,
                      extra={"upstream_output_dir": str(out / "openevolve_run"),
                             "upstream_best_score": getattr(result, "best_score", None)
                             if result else None,
                             "iterations_configured": n_iters},
                      holdout=holdout, logf=log)


def _metrics(cand: C.Cand) -> dict:
    m = {"combined_score": round(cand.primary, 6)}
    m.update({f"inst_{k}": round(v, 6) for k, v in cand.scores.items()})
    return m
