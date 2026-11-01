"""Search core: types, config, evaluation funnel, ranking, adapter protocol, `run`; re-exports qd."""
from __future__ import annotations

import inspect
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Protocol


# ── core types ──────────────────────────────────────────────────────────────────────────────────
@dataclass
class Genome:
    """A harness variant: a directory of code and text files."""
    source_dir: Path
    parent_dir: Optional[Path] = None
    label: str = ""
    parents: list = field(default_factory=list)     # parent source_dirs; defaults to [parent_dir]

    def __post_init__(self):
        if not self.parents and self.parent_dir is not None:
            self.parents = [self.parent_dir]


@dataclass
class EpisodeResult:
    """One episode, keyed by `task_id` (typically a seed shared by all candidates)."""
    task_id: str
    primary: float
    dense: dict[str, float] = field(default_factory=dict)          # extras from cfg episode.dense
    cost: float = 0.0


@dataclass
class Fitness:
    """Aggregated fitness of a genome over a task set; selection compares only these fields."""
    primary: float
    vector: dict[str, float] = field(default_factory=dict)         # per-task scores (rubric_labels)
    dense: dict[str, float] = field(default_factory=dict)          # secondary scores (tie-breakers)
    rows: list[EpisodeResult] = field(default_factory=list)
    noise: dict[str, float] = field(default_factory=dict)          # standard error over episodes
    cost: float = 0.0
    taint: str = "MISSING"                          # PASS | WARN | FAIL (runtime read audit)
    judge: str = "MISSING"                          # OK | WARN | FAIL (LLM judge)
    extra: dict = field(default_factory=dict)       # notes, e.g. why an evaluation is invalid
    valid: bool = True                              # False: no real score, or a rejected cheat
    n: int = 0


# ── benchmark config ────────────────────────────────────────────────────────────────────────────
DEFAULTS = {
    "testbed": None,
    # Lexicographic [field, "desc"|"asc"] pairs. "taint" and "judge" sort by verdict rank, so
    # listing them puts an audited-clean candidate above a FAIL at equal score.
    "selection": {"greedy_key": [["primary", "desc"]]},
    # Fitness: which key of the adapter's score file feeds each Fitness field.
    "fitness": {"primary": "score", "vector": None, "dense": {}, "dense_from_diag": {},
                "cost": None, "n": None, "valid": "auto"},
    "rubric_labels": [],
    "noise_fields": [],
    # Anti-cheat: verdict files, verdict key, and `privileged_keys`, the observation fields a
    # harness must never read.
    "audit": {"taint_file": "read_audit.json", "judge_file": "judge_verdict.json",
              "verdict_key": "verdict", "privileged_keys": []},
    "roles": {"vocabulary": "default", "extra": []},
}


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def load(src=None, **overrides) -> dict:
    """Deep-merge a config (dict, JSON path or None) and then `overrides` over `DEFAULTS`."""
    if isinstance(src, (str, Path)):
        src = json.loads(Path(src).read_text())
    return _merge(_merge(DEFAULTS, src or {}), overrides)


def role_rules(cfg: dict | None = None) -> list:
    """Role rules for φ's structure block: `roles.extra` pairs first, then `DEFAULT_ROLE_RULES`."""
    c = (cfg or {}).get("roles") or {}
    name = c.get("vocabulary", "default")
    if name != "default":
        raise ValueError(f"unknown role vocabulary {name!r}: only 'default' is built in; add "
                         f"benchmark-specific keywords with roles.extra")
    from harness_map.qd import descriptors as _D          # lazy: only this function needs it
    extra = [(r, tuple(k)) for r, k in (c.get("extra") or [])]
    return extra + list(_D.DEFAULT_ROLE_RULES)


def is_valid(cfg: dict, *, vector: dict, rows: list, primary_present: bool) -> bool:
    """Whether an evaluation produced a real score, by the rule `cfg['fitness']['valid']`."""
    rule = ((cfg.get("fitness") or {}).get("valid")) or "auto"
    if rule == "auto":
        return bool(vector) if (cfg.get("rubric_labels") or []) else bool(rows or primary_present)
    if rule == "vector":
        return bool(vector)
    if rule == "rows":
        return bool(rows)
    if rule == "always":
        return True
    raise ValueError(f"fitness.valid must be auto|vector|rows|always, got {rule!r}")


def audit_from_dir(out_dir, cfg: dict) -> tuple:
    """(taint, verdict, judge document) of a candidate directory; an absent verdict is "MISSING"."""
    a = _merge(DEFAULTS["audit"], cfg.get("audit") or {})
    # top-level taint_file / judge_file keys take precedence
    tf = cfg.get("taint_file", a["taint_file"])
    jf = cfg.get("judge_file", a["judge_file"])
    out = Path(out_dir)

    def _read(name):
        p = out / name
        try:
            return json.loads(p.read_text()) if p.exists() else {}
        except Exception:                                # noqa: BLE001 — a corrupt audit is MISSING
            return {}

    jd = _read(jf)
    return (_read(tf).get(a["verdict_key"], "MISSING"), jd.get(a["verdict_key"], "MISSING"), jd)


def privileged_keys(cfg: dict) -> list:
    """Observation fields a harness must never read (`audit.privileged_keys` or top-level key)."""
    a = _merge(DEFAULTS["audit"], cfg.get("audit") or {})
    return list(a.get("privileged_keys") or cfg.get("privileged_keys") or [])


# ── anti-cheat ranking ──────────────────────────────────────────────────────────────────────────
TAINT_RANK = {"PASS": 0, "OK": 0, "WARN": 1, "FAIL": 2, "MISSING": 3}
JUDGE_RANK = {"OK": 0, "PASS": 0, "WARN": 1, "FAIL": 2, "MISSING": 3}


def _get(fit: Fitness, path: str):
    if path == "primary":
        return fit.primary
    if path == "cost":
        return fit.cost
    if path == "taint":
        return TAINT_RANK.get(fit.taint, 99)
    if path == "judge":
        return JUDGE_RANK.get(fit.judge, 99)
    head, _, key = path.partition(".")
    d = getattr(fit, head, None)
    if isinstance(d, dict):
        return d.get(key, 0)
    return d if d is not None else 0


def _sort_key(fit: Fitness, spec):
    return tuple((-_get(fit, f) if dirn == "desc" else _get(fit, f)) for f, dirn in spec)


def _clean(fits):
    """Prefer valid candidates without a FAIL verdict, so a cheat never outranks a clean rival."""
    clean = [f for f in fits if f.valid and f.taint != "FAIL" and f.judge != "FAIL"]
    return clean or [f for f in fits if f.valid] or list(fits)


# ── evaluation funnel ───────────────────────────────────────────────────────────────────────────
class BatchContractError(ValueError):
    """`evaluate_batch` broke its contract (one Fitness-or-None per genome, in order)."""


def _check_batch_fits(adapter, cands, fits):
    who = f"{type(adapter).__name__}.evaluate_batch"
    if fits is None:
        raise BatchContractError(
            f"{who} returned None for {len(cands)} candidate(s) — it must return a "
            f"list of exactly one Fitness-or-None per candidate, in candidate order")
    try:
        fits = list(fits)
    except TypeError as e:
        raise BatchContractError(
            f"{who} returned a non-iterable {type(fits).__name__} — it must return a "
            f"list of one Fitness-or-None per candidate, in candidate order") from e
    if len(fits) != len(cands):
        raise BatchContractError(
            f"{who} returned {len(fits)} fitness/fitnesses for {len(cands)} candidate(s) "
            f"{[getattr(g, 'label', '?') for g in cands]} — the results are zipped POSITIONALLY "
            f"against the genomes, so a wrong length silently drops or mis-pairs candidates. "
            f"Return exactly one Fitness-or-None per candidate, in candidate order.")
    bad = [i for i, f in enumerate(fits)
           if f is not None and not hasattr(f, "valid")]
    if bad:
        raise BatchContractError(
            f"{who} returned entries at position(s) {bad[:5]} that are neither None nor a Fitness "
            f"(got {[type(fits[i]).__name__ for i in bad[:5]]}) — every entry must be a Fitness or "
            f"None, positionally aligned to the candidate at the same index.")
    return fits


def _evaluate_all(adapter, cands, it, *, check_contract=False):
    """Static gate, `evaluate_batch` (or `evaluate`), judge, diagnosis; one result per candidate."""
    reasons = [_pre_eval_gate(adapter, g) for g in cands]
    from pathlib import Path as _P
    if any((_P(getattr(g, "source_dir", ".")) / "harness.py").exists() for g in cands):
        _rej = [f"{getattr(g, 'label', '?')}: {r}" for g, r in zip(cands, reasons) if r]
        print(f"[gate] pre-eval GATE: {sum(1 for r in reasons if r is None)}/{len(cands)} pass"
              + (" | REJECT " + "; ".join(_rej) if _rej else ""))
    survivors = [g for g, r in zip(cands, reasons) if r is None]
    batch = getattr(adapter, "evaluate_batch", None)
    if survivors and batch is not None:
        sfits = batch(survivors, it)
        if check_contract:
            sfits = _check_batch_fits(adapter, survivors, sfits)
    else:
        sfits = [adapter.evaluate(g, it) for g in survivors]
    _sit = iter(sfits)
    # A short, unchecked evaluate_batch result leaves the trailing candidates failed (None).
    fits = [(_rejected_fitness(r) if r is not None else next(_sit, None)) for r in reasons]
    _post_eval_gates(adapter, cands, fits, it)     # judge + diagnose
    # Per-evaluation ledger; a no-op unless HARNESS_MAP_CKPT_DIR is set.
    from harness_map.qd import modes as _modes
    _modes.record(cands, fits)
    return fits


def _rejected_fitness(reason: str):
    return Fitness(primary=0.0, valid=False, taint="FAIL", extra={"gate_reject": reason})


def _pre_eval_gate(adapter, genome):
    own = getattr(adapter, "pre_eval_gate", None)
    if own is not None:
        try:
            return own(genome)
        except Exception as e:                       # a gate error must not stop the loop -> pass
            print(f"[gate] pre_eval_gate({getattr(genome, 'label', '?')}) error: {type(e).__name__}: {e}")
            return None
    return _default_pre_eval_gate(adapter, genome)


def _default_pre_eval_gate(adapter, genome):
    """Reject, without running it, harness code with a syntax error or a privileged-key access."""
    import ast
    from pathlib import Path as _Path
    src_dir = _Path(getattr(genome, "source_dir", "."))
    entry = src_dir / "harness.py"
    if not entry.exists():
        return None                                  # no harness.py: nothing to gate
    py_files = [entry] + sorted(src_dir.glob("tools/*.py"))
    trees = {}
    for f in py_files:
        try:
            trees[f] = ast.parse(f.read_text())
        except SyntaxError as e:
            return f"syntax-error {f.name}:{e.lineno}: {e.msg}"
    _own_priv = getattr(adapter, "privileged_keys", None)
    if callable(_own_priv):
        priv = set(_own_priv() or [])
    else:
        priv = set(privileged_keys(getattr(adapter, "cfg", None) or {}))
    # Only attribute and subscript accesses count, not mentions in strings or comments.
    if priv:
        for f, tree in trees.items():
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and node.attr in priv:
                    return f"privileged-access {f.name}:{node.lineno}: .{node.attr} (oracle)"
                if isinstance(node, ast.Subscript):
                    sl = node.slice
                    key = sl.value if isinstance(sl, ast.Constant) else None
                    if key in priv:
                        return f"privileged-access {f.name}:{getattr(node, 'lineno', '?')}: ['{key}'] (oracle)"
    return None


def _post_eval_gates(adapter, cands, fits, it):
    judge = getattr(adapter, "judge", None)
    diagnose = getattr(adapter, "diagnose", None)
    if not (judge or diagnose):
        return
    for g, f in zip(cands, fits):
        if f is None:
            continue
        if judge is not None and getattr(f, "valid", True):
            try:
                judge(g, f, it)                     # mutates f.judge (+ f.valid=False on FAIL)
            except Exception as e:                  # a gate failure never stops the loop
                print(f"[gate] judge({getattr(g, 'label', '?')}) error: {type(e).__name__}: {e}")
        if diagnose is not None and getattr(f, "valid", True):
            try:
                diagnose(g, it, fitness=f)          # skipped if the judge rejected it
            except Exception as e:
                print(f"[gate] diagnose({getattr(g, 'label', '?')}) error: {type(e).__name__}: {e}")


# ── score files -> Fitness ──────────────────────────────────────────────────────────────────────
def _load(p: Path) -> dict:
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return {}


def _se(xs) -> float:
    xs = [x for x in xs if x is not None]
    return (statistics.stdev(xs) / (len(xs) ** 0.5)) if len(xs) > 1 else 0.0


def _row_field(r: EpisodeResult, f: str) -> float:
    return r.primary if f == "primary" else r.dense.get(f, 0.0)


def episode_results(score: dict, cfg: dict) -> list[EpisodeResult]:
    """Per-episode rows of a score dict, read through the field names in `cfg['episode']`."""
    em = cfg["episode"]
    rows: list[EpisodeResult] = []
    for e in score.get(em.get("list_key", "per_episode"), []):
        rows.append(EpisodeResult(
            task_id=str(e.get(em["task_id"], "")),
            primary=float(e.get(em["primary"], 0) or 0),
            dense={k: float(e.get(src, 0) or 0) for k, src in em.get("dense", {}).items()},
            cost=float(e.get(em.get("cost", "__none__"), 0) or 0),
        ))
    return rows


def fitness_from_dir(out_dir, cfg: dict) -> Fitness:
    """Build a `Fitness` from an evaluation directory; file and field names come from `cfg`."""
    out = Path(out_dir)
    score = _load(out / cfg["score_file"])
    diag = _load(out / cfg.get("diag_file", "diagnostics_summary.json"))
    fm = cfg["fitness"]
    labels = cfg["rubric_labels"]

    rows = episode_results(score, cfg)
    vfield = fm.get("vector")                      # absent for tasks with a single scalar score
    vec_src = (score.get(vfield) or []) if vfield else []
    vector = {labels[i]: float(vec_src[i]) for i in range(min(len(vec_src), len(labels)))}

    dense = {k: float(score.get(src, 0) or 0) for k, src in fm.get("dense", {}).items()}
    dense.update({k: float(diag.get(src, 0) or 0) for k, src in fm.get("dense_from_diag", {}).items()})

    noise = {f: _se([_row_field(r, f) for r in rows]) for f in cfg.get("noise_fields", [])}

    # Anti-cheat verdicts are only recorded here; selection (`_clean`) acts on them.
    _taint, _judge, _ = audit_from_dir(out, cfg)

    return Fitness(
        primary=float(score.get(fm["primary"], 0) or 0),
        vector=vector,
        dense=dense,
        rows=rows,
        noise=noise,
        cost=float(score.get(fm.get("cost", "__none__"), 0) or 0),
        taint=_taint,
        judge=_judge,
        valid=is_valid(cfg, vector=vector, rows=rows,
                               primary_present=fm.get("primary") in score),
        n=int(score.get(fm.get("n", "__none__"), len(rows)) or len(rows)),
    )


class TaskAdapter(Protocol):
    def anchor(self) -> Genome:
        """The seed harness, or a list of seed harnesses, evaluated at round 0."""

    def propose(self, parent: Genome, iter_n: int) -> list[Genome]:
        """Candidate genomes derived from `parent`; may take `directive=` and `n_children=`."""

    def evaluate(self, genome: Genome, iter_n: int) -> Fitness:
        """Run the harness on the task set and return its Fitness."""


# ── run state and adapter calls ─────────────────────────────────────────────────────────────────
@dataclass
class Node:
    """An evaluated genome in the archive; `depth` is 0 for anchors and seeds, > 0 when bred."""
    idx: int
    genome: Genome
    fit: Fitness
    depth: int = 0
    parents: tuple = ()                              # parent node ids

    @property
    def label(self) -> str:
        return self.genome.label or f"node{self.idx}"


@dataclass
class Ctx:
    """State of one mode run: adapter, config, stop / log callbacks and counters."""
    adapter: object
    cfg: dict
    should_stop: Optional[Callable[[], bool]] = None
    log: Callable = print
    _next: int = 1                                    # node id counter
    _round: int = 1                                   # iter_n for the adapter; unique per proposal
    _evals: int = 0                                   # evaluation budget meter (the run's N)

    def spend(self, n: int) -> int:
        """Charge `n` evaluations to the budget and return the total; holdout runs are free."""
        self._evals += int(n)
        return self._evals

    def next_idx(self) -> int:
        i = self._next; self._next += 1
        return i


def _directive_kwarg(fn):
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return "_probe"
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return "directive"
    for kw in ("directive", "feedback"):
        p = params.get(kw)
        if p is not None and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                        inspect.Parameter.KEYWORD_ONLY):
            return kw
    return None


def directive_supported(adapter) -> bool:
    """Whether `adapter.propose` can receive a directive; if not, every directive is dropped."""
    fn = getattr(adapter, "propose", None)
    if fn is None:
        return False
    return _directive_kwarg(fn) is not None


def conformance(adapter, cfg=None) -> dict:
    """Report how an adapter meets the `TaskAdapter` contract: {ok, problems, notes, surface}."""
    probs, notes = [], []
    surface = {}
    for req in ("anchor", "propose", "evaluate"):
        if not callable(getattr(adapter, req, None)):
            probs.append(f"missing REQUIRED hook {req}()")
    surface["directive"] = directive_supported(adapter)
    if not surface["directive"] and callable(getattr(adapter, "propose", None)):
        probs.append("propose() accepts neither directive= nor feedback= nor **kwargs, so every "
                     "steer and coverage directive is discarded before it reaches the proposer")
    gate = callable(getattr(adapter, "pre_eval_gate", None))
    judge = callable(getattr(adapter, "judge", None))
    declared = bool(getattr(adapter, "AUDIT_PROVIDED", False))
    surface.update(pre_eval_gate=gate, judge=judge, audit_declared=declared,
                   diagnose=callable(getattr(adapter, "diagnose", None)))
    if declared and not (gate or judge):
        probs.append("AUDIT_PROVIDED=True but neither pre_eval_gate() nor judge() exists, so "
                     "nothing can ever populate Fitness.taint or Fitness.judge")
    if not declared and not (gate or judge):
        notes.append("no anti-cheat: this benchmark declares none and implements none, so taint "
                     "and judge stay MISSING and _clean can demote nothing")
    c = cfg if cfg is not None else getattr(adapter, "cfg", None)
    surface["cfg"] = bool(c)
    if not c:
        notes.append("no cfg reaches this adapter, so the benchmark's own selection key, fitness "
                     "field names and privileged keys cannot be read")
    else:
        keys = [k for k, _ in (c.get("selection", {}) or {}).get("greedy_key", [])]
        surface["ranks_audit"] = ("taint" in keys or "judge" in keys)
        if (gate or judge) and not surface["ranks_audit"]:
            notes.append("an audit runs but selection.greedy_key ranks on neither taint nor "
                         "judge, so a verdict is recorded and never acted on")
    return {"ok": not probs, "problems": probs, "notes": notes, "surface": surface}


def call_diagnose(adapter, genome, fit, it=0):
    """Text of the adapter's optional `diagnose(genome, fit[, it])`, or None (also on error)."""
    fn = getattr(adapter, "diagnose", None)
    if fn is None:
        return None
    try:
        n = len([q for q in inspect.signature(fn).parameters.values()
                 if q.kind in (q.POSITIONAL_ONLY, q.POSITIONAL_OR_KEYWORD)])
    except (TypeError, ValueError):
        n = 2
    try:
        d = fn(genome, fit, it) if n >= 3 else fn(genome, fit)
    except Exception:                                   # noqa: BLE001 — advisory, never fatal
        return None
    return (d.strip() or None) if isinstance(d, str) else None


def diagnose_of(adapter, genome, fit):
    """The adapter's diagnosis of a parent, which breeding prefixes to the steering directive."""
    return call_diagnose(adapter, genome, fit)


_CHILDREN_KWARGS = ("n_children", "children_per_parent", "n_cands", "children")


def _children_kwarg(fn):
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return None                                     # the caller keeps the first C
    for kw in _CHILDREN_KWARGS:
        p = params.get(kw)
        if p is not None and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                        inspect.Parameter.KEYWORD_ONLY):
            return kw
    return None


def _propose_ex(adapter, parent, rid, directive=None, n_children=None,
                introspect=False):
    """Call `adapter.propose` with `directive` and `n_children` only through keywords it accepts."""
    kws = {}
    if n_children is not None:
        ckw = _children_kwarg(adapter.propose)
        if ckw:
            kws[ckw] = int(n_children)
    if directive:
        kw = _directive_kwarg(adapter.propose) if introspect else "_probe"
        if kw == "_probe":                              # try the usual directive keywords
            for k in ("directive", "feedback"):
                try:
                    return adapter.propose(parent, rid, **{k: directive}, **kws)
                except TypeError:
                    continue
        elif kw:
            return adapter.propose(parent, rid, **{kw: directive}, **kws)
    return adapter.propose(parent, rid, **kws)


def _rank(nodes, cfg):
    """`nodes` after anti-cheat demotion (`_clean`), best first by `selection.greedy_key`."""
    if not nodes:
        return []
    keep = set(id(f) for f in _clean([n.fit for n in nodes]))
    clean = [n for n in nodes if id(n.fit) in keep] or list(nodes)
    return sorted(clean, key=lambda n: _sort_key(n.fit, cfg["selection"]["greedy_key"]))


def _out(nodes, cfg, ctx):
    if not nodes:
        return {"history": [], "best": None}
    best = _rank(nodes, cfg)[0]
    history = [{"iter": n.idx, "genome": n.genome, "fit": n.fit, "label": n.label,
                "depth": n.depth, "parents": n.parents}
               for n in nodes]
    out = {"history": history, "best": {"iter": best.idx, "genome": best.genome,
                                        "fit": best.fit, "label": best.label}}
    hold = getattr(ctx.adapter, "evaluate_holdout", None)
    if hold is not None and best.depth > 0:
        hf = hold(best.genome)
        if hf is not None:
            out["holdout"] = hf
            ctx.log(f"[{cfg.get('_mode', '?')}] holdout({best.label}): primary={hf.primary:.2f}")
    return out


def _anchor_nodes(ctx: Ctx) -> list:
    got = ctx.adapter.anchor()
    genomes = list(got) if isinstance(got, (list, tuple)) else [got]
    if not genomes:
        raise ValueError(f"{type(ctx.adapter).__name__}.anchor() returned an EMPTY list — a "
                         f"MAP-Elites run needs at least one anchor individual to seed the archive")
    fits = _evaluate_all(ctx.adapter, genomes, 0)      # gated, judged, diagnosed and ledgered too
    nodes = []
    for i, (g, f) in enumerate(zip(genomes, fits)):
        ctx.spend(1)
        nodes.append(Node(0 if i == 0 else ctx.next_idx(), g, f, depth=0))
    return nodes


# ── search modes ────────────────────────────────────────────────────────────────────────────────
class SearchMode:
    """A search mode: its `name` and `run(ctx, **knobs)`; instances are callable."""
    name: str = ""

    def run(self, ctx, **knobs) -> dict:
        raise NotImplementedError(f"{type(self).__name__}.run must delegate to its free function")

    def __call__(self, ctx, **knobs) -> dict:
        return self.run(ctx, **knobs)


# `MODES` and the qd names (fit_cvt_blocks, run_frontier, ...) are looked up on first access
# (PEP 562) instead of being imported here: the qd modules import this module, so a top-level
# import would be circular.
_MODES_CACHE = None
_QD_NAMES = {
    "Frontier": "archive", "CoverageError": "coverage",
    "DEFAULT_ROLE_RULES": "descriptors", "EMB_MODEL": "descriptors", "desc_blocks": "descriptors",
    "kmeans": "partition", "fit_cvt_blocks": "partition", "make_phi_blocks": "partition",
    "build_centroid_ladder": "partition",
    "run_frontier": "modes", "MapElitesFixed": "modes", "MapElitesHier": "modes",
    "MapElitesAdaptive": "modes",
}


def _modes() -> dict:
    global _MODES_CACHE
    if _MODES_CACHE is None:
        from harness_map.qd.modes import MapElitesAdaptive, MapElitesFixed, MapElitesHier
        classes = (MapElitesFixed, MapElitesHier, MapElitesAdaptive)
        _MODES_CACHE = {cls.name: cls() for cls in classes}
    return _MODES_CACHE


def __getattr__(name):
    if name == "MODES":
        return _modes()
    if name in _QD_NAMES:
        import importlib
        return getattr(importlib.import_module(f"harness_map.qd.{_QD_NAMES[name]}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted([*globals(), "MODES", *_QD_NAMES])


# ── run ─────────────────────────────────────────────────────────────────────────────────────────
def run(adapter, cfg: dict, *, mode: str = "mapelites_adaptive", should_stop=None, log=print,
        **knobs) -> dict:
    """Run one mode on `adapter`; returns {history, best, holdout?} and the mode's report block."""
    if mode not in _modes():
        raise ValueError(f"unknown mode {mode!r}; choose one of {sorted(_modes())}")
    cfg = {**cfg, "_mode": mode}
    ctx = Ctx(adapter=adapter, cfg=cfg, should_stop=should_stop, log=log)
    log(f"[meta-harness] mode={mode}")
    return _modes()[mode](ctx, **knobs)
