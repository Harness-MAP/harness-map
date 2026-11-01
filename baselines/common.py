"""Shared substrate of the arms: budget ledger, candidate record, evaluation plumbing, bundles."""
from __future__ import annotations

import json
import os
import py_compile
import re
import shutil
import threading
import time

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from harness_map.main import Fitness, Genome


# ── budget ────────────────────────────────────────────────────────────────────
class BudgetExhausted(Exception):
    """Raised inside an arm's loop when a budget cap is hit, so it unwinds to its summary."""


@dataclass
class Ledger:
    """The budget shared by every arm: candidate and per-instance evaluations, with caps."""
    max_cand_evals: Optional[int] = None
    max_metric_calls: Optional[int] = None
    max_wallclock_s: Optional[float] = None
    # An integer in this file replaces max_cand_evals on every check, so a live run can be
    # extended by writing a new cap (set by run_baseline.py; None leaves the cap fixed).
    override_file: Optional[Path] = None
    cand_evals: int = 0
    metric_calls: int = 0
    proposals: int = 0
    t0: float = field(default_factory=time.time)
    # GEPA's engine may call spend() from several threads.
    _lock: threading.Lock = field(default_factory=threading.Lock, compare=False, repr=False)

    def spend(self, n_instances: int, new_candidate: bool = True) -> None:
        """Record one evaluation; `new_candidate=False` charges metric calls only."""
        with self._lock:
            if new_candidate:
                self.cand_evals += 1
            self.metric_calls += max(int(n_instances), 0)

    def elapsed(self) -> float:
        return time.time() - self.t0

    def _cap(self) -> Optional[int]:
        """max_cand_evals, unless BUDGET_OVERRIDE names a new one (renewable budget — see field)."""
        f = self.override_file
        if f is not None:
            try:
                p = Path(f)
                if p.exists():
                    v = int(p.read_text().strip())
                    if v != self.max_cand_evals:
                        print(f"[ledger] BUDGET_OVERRIDE: max_cand_evals {self.max_cand_evals} -> {v}",
                              flush=True)
                        self.max_cand_evals = v
            except Exception:                            # noqa: BLE001 — a bad file must not stop an arm
                pass
        return self.max_cand_evals

    def exhausted(self) -> bool:
        cap = self._cap()
        return ((cap is not None and self.cand_evals >= cap)
                or (self.max_metric_calls is not None and self.metric_calls >= self.max_metric_calls)
                or (self.max_wallclock_s is not None and self.elapsed() >= self.max_wallclock_s))

    def check(self) -> None:
        if self.exhausted():
            raise BudgetExhausted(self.reason())

    def reason(self) -> str:
        if self.max_cand_evals is not None and self.cand_evals >= self.max_cand_evals:
            return f"cand_evals {self.cand_evals}/{self.max_cand_evals}"
        if self.max_metric_calls is not None and self.metric_calls >= self.max_metric_calls:
            return f"metric_calls {self.metric_calls}/{self.max_metric_calls}"
        if self.max_wallclock_s is not None and self.elapsed() >= self.max_wallclock_s:
            return f"wallclock {self.elapsed():.0f}s/{self.max_wallclock_s:.0f}s"
        return "not exhausted"

    def as_dict(self) -> dict:
        return {"cand_evals": self.cand_evals, "metric_calls": self.metric_calls,
                "proposals": self.proposals, "wallclock_s": round(self.elapsed(), 1),
                "caps": {"cand_evals": self.max_cand_evals, "metric_calls": self.max_metric_calls,
                         "wallclock_s": self.max_wallclock_s}}


# ── the unit of search ────────────────────────────────────────────────────────
@dataclass
class Cand:
    """One evaluated genome with its per-instance scores, parent and generation."""
    idx: int
    genome: Genome
    fit: Optional[Fitness]
    scores: dict = field(default_factory=dict)
    parent: Optional[int] = None
    gen: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.genome.label or f"cand{self.idx}"

    @property
    def primary(self) -> float:
        return float(self.fit.primary) if self.fit is not None else 0.0

    @property
    def valid(self) -> bool:
        return self.fit is not None and bool(getattr(self.fit, "valid", False))

    def row(self) -> dict:
        return {"idx": self.idx, "label": self.label, "parent": self.parent, "gen": self.gen,
                "primary": round(self.primary, 6), "valid": self.valid,
                "source_dir": str(self.genome.source_dir),
                "scores": {k: round(v, 6) for k, v in self.scores.items()},
                "dense": _jsonable(dict(self.fit.dense) if self.fit is not None else {}),
                "meta": _jsonable(self.meta)}


# ── per-instance scores ───────────────────────────────────────────────────────
def per_instance(fit: Optional[Fitness], front: str = "instance") -> dict[str, float]:
    """Per-instance scores of a Fitness: episode rows, else the rubric vector, else a pooled key."""
    if fit is None:
        return {}
    rows: dict[str, float] = {}
    if getattr(fit, "rows", None):
        acc: dict[str, list[float]] = {}
        for r in fit.rows:
            acc.setdefault(str(r.task_id), []).append(float(r.primary))
        rows = {k: sum(v) / len(v) for k, v in acc.items()}
    vec = {str(k): float(v) for k, v in (getattr(fit, "vector", None) or {}).items()
           if isinstance(v, (int, float))}

    if front == "hybrid":
        out = {f"inst::{k}": v for k, v in rows.items()}
        out.update({f"obj::{k}": v for k, v in vec.items()})
        return out or {"__pooled__": float(fit.primary)}
    order = (vec, rows) if front == "objective" else (rows, vec)
    for cand in order:
        if cand:
            return dict(cand)
    return {"__pooled__": float(fit.primary)}


def failing(scores: dict[str, float], thresh: float = 0.0) -> list[str]:
    """Instance ids scoring at or below `thresh` (DGM's unresolved set, generalized)."""
    return sorted([k for k, v in scores.items() if v <= thresh])


# ── adapter plumbing (every call degrades gracefully) ─────────────────────────
def task_ids(adapter, split: str = "evolve") -> list[str]:
    """The adapter's split as a list of ids, whatever shape its task records have."""
    return [_task_id(t) for t in (getattr(adapter, split, None) or [])]


def _task_id(item) -> str:
    """The id of a task record: its `task_id`/`id` field, else its values joined by "/"."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for k in ("task_id", "instance_id", "game_id", "id", "name"):
            if k in item:
                return str(item[k])
        return "/".join(str(item[k]) for k in sorted(item))
    return str(item)


def evaluate_on(adapter, genome: Genome, rid: int, ids=None, ledger: Optional[Ledger] = None):
    """Evaluate `genome`, restricted to instance `ids` if given; returns (Fitness, n charged)."""
    if ids is None:
        fit = adapter.evaluate(genome, rid)
        n = _n_instances(fit, adapter, "evolve")
    else:
        ids = list(ids)
        sub = getattr(adapter, "evaluate_subset", None)
        fit = sub(genome, ids, rid) if sub is not None else None
        if fit is not None:
            n = _n_instances(fit, adapter, None, default=len(ids))
        else:
            fit = adapter.evaluate(genome, rid)          # no subset hook: pay for the whole split
            n = _n_instances(fit, adapter, "evolve")
    if ledger is not None:
        ledger.spend(n)
    return fit, n


def _n_instances(fit, adapter, split, default: int = 1) -> int:
    if fit is None:
        return default
    if getattr(fit, "n", 0):
        return int(fit.n)
    if getattr(fit, "rows", None):
        return len(fit.rows)
    if split is not None:
        return max(len(getattr(adapter, split, None) or []), 1)
    return default


def propose(adapter, parent: Genome, rid: int, k: int = 1, directive: Optional[str] = None):
    """Up to `k` children of `parent` from the adapter's proposer, conditioned on `directive`."""
    from harness_map import main as MH
    out = MH._propose_ex(adapter, parent, rid, directive or None, n_children=k, introspect=True)
    return list(out or [])[:k]


def validate(adapter, genome: Genome) -> bool:
    """Whether a candidate may be evaluated: adapter `validate`, else static gate + byte-compile."""
    hook = getattr(adapter, "validate", None)
    if hook is not None:
        try:
            return bool(hook(genome))
        except Exception:
            return False
    from harness_map import main as MH
    if MH._pre_eval_gate(adapter, genome):              # the static privileged-key gate, every arm
        return False
    d = Path(genome.source_dir)
    if not d.is_dir():
        return False
    # A candidate must keep the root harness.py its parent had: without it the evaluator
    # fails silently to 0 instead of being caught here.
    par = Path(genome.parent_dir) if getattr(genome, "parent_dir", None) else None
    if par and (par / "harness.py").exists() and not (d / "harness.py").exists():
        return False
    pys = sorted(d.rglob("*.py"))
    if not pys:
        return False
    for p in pys:
        try:
            py_compile.compile(str(p), doraise=True, cfile=str(p) + "c")
        except Exception:
            return False
        finally:
            Path(str(p) + "c").unlink(missing_ok=True)
    return True


# ── run artifacts ─────────────────────────────────────────────────────────────
class RunLog:
    """Append-only run record: `history.jsonl`, `run_meta.json`, and the arm's own artifacts."""

    def __init__(self, out_dir, method: str, meta: Optional[dict] = None):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.method = method
        self.history = self.dir / "history.jsonl"
        # _jsonable: knobs may hold live objects (a proposer, an LM callable).
        (self.dir / "run_meta.json").write_text(json.dumps(_jsonable(
            {"method": method, "started": time.strftime("%Y-%m-%dT%H:%M:%S"), **(meta or {})}),
            indent=2))

    def log(self, cand: Cand, ledger: Ledger, **extra) -> None:
        row = {**cand.row(), "method": self.method, "budget": ledger.as_dict(), **_jsonable(extra)}
        with self.history.open("a") as f:
            f.write(json.dumps(row) + "\n")

    def write(self, name: str, obj) -> None:
        # atomic: write a sibling temp then os.replace, so a kill mid-write never leaves a
        # truncated summary.json behind.
        p = self.dir / name
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(_jsonable(obj), indent=2))
        os.replace(tmp, p)

    def append(self, name: str, obj) -> None:
        with (self.dir / name).open("a") as f:
            f.write(json.dumps(_jsonable(obj)) + "\n")


def finalize(adapter, log: RunLog, ledger: Ledger, pool: list[Cand], best: Optional[Cand],
             extra: Optional[dict] = None, holdout: bool = True, logf=print) -> dict:
    """Common tail of every arm: score the winner held out, write `summary.json`, return it."""
    summary = {"method": log.method, "budget": ledger.as_dict(),
               "n_evaluated": len(pool), "n_valid": sum(1 for c in pool if c.valid),
               "best": best.row() if best is not None else None,
               "stop_reason": ledger.reason() if ledger.exhausted() else "iterations exhausted",
               **(extra or {})}
    if best is not None and holdout:
        hook = getattr(adapter, "evaluate_holdout", None)
        if hook is not None:
            try:
                hf = hook(best.genome)
            except Exception as e:                        # a holdout failure must not lose the run
                hf, summary["holdout_error"] = None, repr(e)
            if hf is not None:
                summary["holdout"] = {"primary": round(float(hf.primary), 6),
                                      "valid": bool(getattr(hf, "valid", False)),
                                      "dense": _jsonable(dict(hf.dense)),
                                      "scores": {k: round(v, 6)
                                                 for k, v in per_instance(hf).items()}}
                logf(f"[{log.method}] holdout(best={best.label}) primary={hf.primary:.4f}")
    log.write("summary.json", summary)
    return summary


def pick_best(pool: list[Cand]) -> Optional[Cand]:
    """Highest primary among valid candidates; ties go to the earlier index."""
    valid = [c for c in pool if c.valid]
    return max(valid, key=lambda c: (c.primary, -c.idx)) if valid else None


def render_scores(cand: Cand, limit: int = 24) -> str:
    """Per-instance scores as a compact block for a proposer directive."""
    items = sorted(cand.scores.items(), key=lambda kv: kv[1])[:limit]
    return "\n".join(f"  {k:<44} {v:.3f}" for k, v in items) or "  (no per-instance scores)"


# ── single-file candidate bundles (OpenEvolve / ShinkaEvolve) ───────────────────────────────
# Both frameworks evolve one program file with EVOLVE-BLOCK regions. A harness is a package and
# its logic lives across files, so the whole package is bundled into one file, one region per
# source file, and split back out before it runs: every file is evolvable and the framework
# still sees the single-file candidate it expects.
BUNDLE_SUFFIXES = {".py"}
_FILE_MARK = "# ===== FILE: {rel} ====="
_FILE_RE = re.compile(r"^# ===== FILE: (.+?) =====\s*$", re.M)
BUNDLE_PREAMBLE = '''\
"""Harness bundle — one EVOLVE-BLOCK region per source file of the harness package.

Each region below is the full text of one file. Keep the `# ===== FILE: <path> =====` markers and
the EVOLVE-BLOCK markers exactly as they are: they are how this bundle is split back into a runnable
package before it is scored. You may change anything INSIDE a region, add code, or restructure
freely; a region whose file must not change can simply be left alone.
"""
'''

# Both frameworks apply SEARCH/REPLACE diffs by exact line match and silently apply nothing on
# a miss, so this discipline is added to their task/system message (no vendor template edit).
DIFF_DISCIPLINE = """\
OUTPUT FORMAT — you MUST use SEARCH/REPLACE diff blocks and nothing else:

<<<<<<< SEARCH
(a few lines copied VERBATIM from the current program)
=======
(the replacement lines)
>>>>>>> REPLACE

Hard rules (a block that breaks any of these applies NOTHING and wastes the step):
- Emit ONLY SEARCH/REPLACE blocks. No prose, no explanation, no ```python fences around them.
- The SEARCH body must be 3-8 CONSECUTIVE lines copied character-for-character from the current
  program, INCLUDING their exact indentation. Do not paraphrase, reformat, or re-wrap them.
- Never put a blank line inside a SEARCH body — pick a run of consecutive NON-BLANK lines.
- Keep each marker line exact and on its own line: seven '<' then ' SEARCH', seven '=', seven '>'
  then ' REPLACE'.
- Make ONE small, targeted change. Do NOT rewrite a whole file or region.
- Never alter a `# ===== FILE: ... =====` line or an EVOLVE-BLOCK marker.
- You may emit several blocks for several small edits; each SEARCH must match the program exactly.
"""


def bundle_harness(harness_dir, suffixes=None, exclude=None) -> str:
    """A harness package as one program file, one EVOLVE-BLOCK region per source file."""
    d = Path(harness_dir)
    parts = [BUNDLE_PREAMBLE]
    exclude = {e for e in (exclude or ())}
    if suffixes:
        suffixes = {s.lower() for s in suffixes}
    else:
        suffixes = BUNDLE_SUFFIXES
        if not any(q.is_file() and q.suffix.lower() in BUNDLE_SUFFIXES and "__pycache__" not in q.parts
                   for q in d.rglob("*")):
            suffixes = BUNDLE_SUFFIXES | {".txt", ".md"}
    for p in sorted(d.rglob("*")):
        if not p.is_file() or "__pycache__" in p.parts or p.suffix.lower() not in suffixes:
            continue
        if str(p.relative_to(d)) in exclude or p.name in exclude:   # drop inert/redundant files
            continue
        try:
            text = p.read_text()
        except UnicodeDecodeError:
            continue
        parts.append(f"\n{_FILE_MARK.format(rel=p.relative_to(d))}\n"
                     f"# EVOLVE-BLOCK-START\n{text.rstrip()}\n# EVOLVE-BLOCK-END\n")
    return "".join(parts)


def unbundle_harness(text: str, base_dir, out_dir) -> Path:
    """The bundle back into a runnable harness directory, over a copy of the anchor's files."""
    out = Path(out_dir)
    if not out.exists():
        shutil.copytree(base_dir, out, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    marks = list(_FILE_RE.finditer(text))
    if not marks:
        (out / "harness.py").write_text(text)
        return out
    for i, m in enumerate(marks):
        rel = m.group(1).strip()
        body = text[m.end():(marks[i + 1].start() if i + 1 < len(marks) else len(text))]
        body = "\n".join(ln for ln in body.splitlines()
                         if ln.strip() not in ("# EVOLVE-BLOCK-START", "# EVOLVE-BLOCK-END"))
        f = out / rel
        if not str(f.resolve()).startswith(str(out.resolve())):
            continue                                   # a '../' path in a marker never escapes out/
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(body.strip("\n") + "\n")
    return out


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_jsonable(v) for v in o]
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    return str(o)
