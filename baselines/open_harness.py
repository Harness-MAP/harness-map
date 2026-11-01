"""Ablation: our search modes over OpenEvolve's (complexity, diversity) descriptor, frozen."""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import random
import threading

from pathlib import Path

import numpy as np

from . import common as C
from .upstream import load_openevolve, pins


# Upstream's defaults: config.database.feature_dimensions and diversity_reference_size.
FEATURE_DIMENSIONS = ("complexity", "diversity")
_TELEMETRY_KEY = {"adaptive": "frontier", "fixed": "fixed", "hier": "hier"}

# These would duplicate keywords the driver supplies explicitly.
# explicitly, raising TypeError inside the mode call.
_RESERVED_KNOBS = frozenset({"phi_fn", "block_split", "seed_genomes",
                             "init_centroids", "centroid_levels"})

DIVERSITY_REFERENCE_SIZE = 20

_MODE_KEYS = {"adaptive": "mapelites_adaptive", "fixed": "mapelites_fixed",
              "hier": "mapelites_hier"}


def _zscore(R):
    """`fit_cvt_blocks`' transform: z-score (constant columns only centred), then 1/sqrt(dim)."""
    R = np.asarray(R, float)
    mu, sd = R.mean(0), R.std(0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return (R - mu) / sd / math.sqrt(max(R.shape[1], 1)), mu, sd


# ── upstream's distance function, called rather than copied ──────────────────────────────────
def _code_distance():
    """Upstream `_fast_code_diversity` as a plain two-argument function (it never reads `self`)."""
    load_openevolve()
    from openevolve.database import ProgramDatabase
    fn = ProgramDatabase._fast_code_diversity
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):                         # no source available: trust the pin
        src = ""
    if "self." in src:
        raise RuntimeError(
            "openevolve's _fast_code_diversity now reads `self`; it can no longer be called "
            "unbound. Re-check vendor/openevolve/openevolve/database.py:2095 and either bind a "
            "real ProgramDatabase or update this arm, but do not silently change the descriptor.")
    return lambda a, b: float(fn(None, a, b))


def _bundler(adapter):
    """(Path) -> the harness's single-file bundle on the adapter's declared surface."""
    suffixes = getattr(adapter, "bundle_suffixes", None)
    exclude = getattr(adapter, "bundle_exclude", None)

    def bundle(d) -> str:
        d = Path(d)
        if not d.is_dir():
            # a missing tree would bundle to the preamble alone and land in a wrong cell
            raise ValueError(f"open-harness descriptor: no such harness dir {d}")
        return C.bundle_harness(d, suffixes=suffixes, exclude=exclude)
    return bundle


# ── the descriptor ───────────────────────────────────────────────────────────────────────────
class FrozenOpenEvolveDescriptor:
    """OpenEvolve's (complexity, diversity) as a pure phi, reference set and scale frozen at fit."""

    def __init__(self, corpus_dirs, *, bundle, ref_size: int = DIVERSITY_REFERENCE_SIZE,
                 seed: int = 0, self_exclusion: bool = True, log=print):
        self.bundle = bundle
        self.dist = _code_distance()
        self.log = log
        self.dim = len(FEATURE_DIMENSIONS)
        self.seed = int(seed)
        self.self_exclusion = bool(self_exclusion)
        self.corpus_dirs = [Path(d) for d in corpus_dirs]
        if not self.corpus_dirs:
            raise ValueError("open-harness needs a bootstrap corpus to freeze the descriptor on")
        self.corpus_code = [self.bundle(d) for d in self.corpus_dirs]
        self.ref_size = int(ref_size)
        self.refs = self._select_refs(self.corpus_code, self.ref_size, self.seed)
        self.mu = None
        self.sd = None
        self.raw_corpus = None                           # (n, 2) raw feature matrix, set by fit()
        self._cache: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()                    # n_workers>1 calls phi from threads

    # -- upstream's reference-set rule, made deterministic ----------------------------------
    def _select_refs(self, codes, ref_size, seed) -> list[str]:
        """Upstream's reference-set rule over the corpus, with a seeded first pick."""
        if len(codes) <= ref_size:
            return list(codes)
        rng = random.Random(seed)
        remaining = list(codes)
        selected = [remaining.pop(rng.randrange(len(remaining)))]
        while len(selected) < ref_size and remaining:
            best_i, best_min = -1, -1.0
            for i, cand in enumerate(remaining):
                m = min(self.dist(cand, s) for s in selected)
                if m > best_min:
                    best_min, best_i = m, i
            if best_i < 0:
                break
            selected.append(remaining.pop(best_i))
        return selected

    # -- the two features -------------------------------------------------------------------
    def raw_of_code(self, code: str) -> np.ndarray:
        """[complexity, diversity] as upstream computes them, before binning."""
        comp = float(len(code))
        scores = [self.dist(code, r) for r in self.refs
                  if not (self.self_exclusion and r == code)]
        div = (sum(scores) / len(scores)) if scores else 0.0
        return np.array([comp, div], dtype=float)

    def standardize(self, raw) -> np.ndarray:
        """Corpus z-score then 1/sqrt(dim), with the frozen constants."""
        if self.mu is None:
            raise RuntimeError("descriptor not fitted: call fit() before using phi")
        return (np.asarray(raw, float) - self.mu) / self.sd / math.sqrt(self.dim)

    # -- the phi our modes are handed --------------------------------------------------------
    def __call__(self, node) -> np.ndarray:
        """phi(node) -> (2,), reading only `node.genome.source_dir`; memoised on the bundle hash."""
        code = self.bundle(Path(node.genome.source_dir))
        key = hashlib.sha1(code.encode()).hexdigest()
        with self._lock:
            hit = self._cache.get(key)
        if hit is not None:
            return hit
        p = self.standardize(self.raw_of_code(code))
        with self._lock:
            self._cache[key] = p
        return p

    # -- fit ---------------------------------------------------------------------------------
    def fit(self, k0: int, seed: int = 0):
        """Freeze mu/sd on the corpus and k-means up to k0 centroids; returns (Z0, labels, J)."""
        from harness_map.qd import partition as _part
        R = np.stack([self.raw_of_code(c) for c in self.corpus_code])
        J, mu, sd = _zscore(R)                   # the same z + 1/sqrt(dim) fit_cvt_blocks applies
        self.mu, self.sd = mu, sd
        self.raw_corpus = R
        Z0, lab = _part.kmeans(J, min(int(k0), len(J)), seed=seed)
        return Z0, lab, J

    # -- reuse --------------------------------------------------------------------------------
    @classmethod
    def load(cls, desc_dir, *, bundle, log=print):
        """Rehydrate a descriptor and partition an earlier run froze, without refitting."""
        d = Path(desc_dir)
        meta = json.loads((d / "descriptor.json").read_text())
        z = np.load(d / "partition.npz", allow_pickle=True)
        self = cls.__new__(cls)
        self.bundle = bundle
        self.dist = _code_distance()
        self.log = log
        self.dim = len(FEATURE_DIMENSIONS)
        self.seed = int(meta.get("seed", 0) or 0)
        # The saved string is the sentence `meta()` wrote, not a bool; "dropped" is the only
        # other value it can hold, so anything else means the upstream rule was in force.
        self.self_exclusion = (str(meta.get("self_exclusion", "")) != "dropped")
        self.corpus_dirs = [Path(x) for x in (meta.get("corpus_dirs") or [])]
        self.corpus_code = []          # deliberately empty: refs and mu/sd are what phi needs
        self.ref_size = int(meta.get("ref_size", DIVERSITY_REFERENCE_SIZE))
        self.refs = [(d / "refs" / f.name).read_text()
                     for f in sorted((d / "refs").glob("ref_*.txt"))]
        self.mu = np.asarray(meta["mu"], float)
        self.sd = np.asarray(meta["sd"], float)
        # the raw corpus matrix, inverted from the saved J (z = (raw - mu) / sd / sqrt(dim))
        self.raw_corpus = (np.asarray(z["J"], float) * math.sqrt(self.dim)
                           * np.where(self.sd < 1e-9, 1.0, self.sd) + self.mu)
        self._cache = {}
        self._lock = threading.Lock()
        self.reused_from = str(d)
        if len(self.refs) != int(meta.get("n_refs", len(self.refs))):
            raise ValueError(f"{d}: descriptor.json says {meta.get('n_refs')} reference programs "
                             f"but {len(self.refs)} ref_*.txt files are present")
        return self, z["Z0"], z["labels"], z["J"]

    # -- provenance ---------------------------------------------------------------------------
    def meta(self) -> dict:
        return {
            "descriptor": "openevolve complexity+diversity, FROZEN",
            "features": list(FEATURE_DIMENSIONS),
            "complexity": "len(bundle_harness(source_dir)); upstream database.py:873",
            "diversity": "mean _fast_code_diversity(code, ref) over the frozen reference set "
                         "; upstream database.py:2095/2119, called unbound from the clone",
            "standardization": "corpus z-score then 1/sqrt(2) (as fit_cvt_blocks), "
                               "constants frozen",
            "ref_size": self.ref_size, "n_refs": len(self.refs),
            "ref_sha1": [hashlib.sha1(r.encode()).hexdigest()[:12] for r in self.refs],
            "ref_chars": [len(r) for r in self.refs],
            "self_exclusion": ("upstream exact-string rule, against the frozen list"
                               if self.self_exclusion else "dropped"),
            "corpus_dirs": [str(d) for d in self.corpus_dirs],
            "corpus_order_matters": "kmeans++ draws its first centroid by index; this order IS "
                                    "part of the frozen partition",
            "mu": (self.mu.tolist() if self.mu is not None else None),
            "sd": (self.sd.tolist() if self.sd is not None else None),
            "seed": self.seed,
            # Set only by `load`: this partition was NOT fitted by the run that used it.
            "reused_from": getattr(self, "reused_from", None),
        }

    def save(self, out_dir) -> Path:
        """Persist the frozen reference set and constants under `out_dir`."""
        d = Path(out_dir)
        (d / "refs").mkdir(parents=True, exist_ok=True)
        for i, r in enumerate(self.refs):
            (d / "refs" / f"ref_{i:02d}.txt").write_text(r)
        (d / "descriptor.json").write_text(json.dumps(self.meta(), indent=2))
        return d


def _occupancy(labels, k: int) -> list[int]:
    h = [0] * k
    for j in labels:
        h[int(j)] += 1
    return h


def _log_geometry(desc: FrozenOpenEvolveDescriptor, J, labels, k, log):
    """Pre-flight of the corpus partition: occupancy, feature spread, collinearity."""
    hist = _occupancy(labels, k)
    log(f"[open-harness] corpus cell occupancy (k={k}, n={len(J)}): "
        + " ".join(f"c{i}={n}" for i, n in enumerate(hist))
        + f"  | empty={sum(1 for n in hist if n == 0)}")
    R = desc.raw_corpus
    log(f"[open-harness] raw features: complexity {R[:, 0].min():.0f}..{R[:, 0].max():.0f} chars "
        f"(sd {R[:, 0].std():.1f}) | diversity {R[:, 1].min():.2f}..{R[:, 1].max():.2f} "
        f"(sd {R[:, 1].std():.2f})")
    n_distinct = len({tuple(np.round(p, 9)) for p in J})
    corr = None
    if R[:, 0].std() > 1e-9 and R[:, 1].std() > 1e-9:
        corr = float(np.corrcoef(R[:, 0], R[:, 1])[0, 1])
    log(f"[open-harness] {n_distinct}/{len(J)} distinct phi points; "
        f"corr(complexity, diversity) = {('%.3f' % corr) if corr is not None else 'undefined'}")
    if corr is not None and abs(corr) > 0.95:
        log("[open-harness] WARNING: the two axes are near-collinear, so upstream's '2-D' "
            "descriptor is effectively ONE length axis here (see the module docstring: char_diff "
            "saturates between Python files). Expect cells to collapse and coverage-init to fail "
            "at higher k0; that is a finding about the descriptor, not a bug in the run.")
    if n_distinct < k:
        log(f"[open-harness] WARNING: only {n_distinct} distinct corpus points for k={k}: "
            f"kmeans caps itself (partition.py:53-54) and the realized k will be smaller.")
    return {"histogram": hist, "n_empty": sum(1 for n in hist if n == 0),
            "n_distinct_phi": n_distinct, "corr_complexity_diversity": corr,
            "complexity_range": [float(R[:, 0].min()), float(R[:, 0].max())],
            "diversity_range": [float(R[:, 1].min()), float(R[:, 1].max())]}


# ── the metered adapter ──────────────────────────────────────────────────────────────────────
class _MeteredAdapter:
    """The adapter, with every evaluation charged to the ledger and logged to the run."""

    evaluate_holdout = None

    def __init__(self, inner, *, ledger: C.Ledger, pool: list, rl: C.RunLog, log=print):
        self._inner = inner
        self._ledger = ledger
        self._pool = pool
        self._rl = rl
        self._log = log
        self._lock = threading.Lock()

    def __getattr__(self, name):
        inner = object.__getattribute__(self, "_inner")
        if name == "evaluate_batch":
            hook = getattr(inner, "evaluate_batch", None)
            if hook is None:
                raise AttributeError(name)               # so getattr(..., None) stays honest
            return self._evaluate_batch
        return getattr(inner, name)

    def _record(self, genome, fit, rid) -> C.Cand:
        n = C._n_instances(fit, self._inner, "evolve")
        self._ledger.spend(n)
        with self._lock:
            cand = C.Cand(idx=len(self._pool), genome=genome, fit=fit,
                          scores=C.per_instance(fit), gen=int(rid or 0),
                          meta={"role": "candidate", "rid": rid})
            self._pool.append(cand)
        self._rl.log(cand, self._ledger)
        self._log(f"[open-harness] {cand.label}: primary={cand.primary:.4f} valid={cand.valid} "
                  f"({self._ledger.cand_evals}/{self._ledger.max_cand_evals or '-'} cand evals)")
        return cand

    def evaluate(self, genome, rid):
        fit = self._inner.evaluate(genome, rid)
        self._record(genome, fit, rid)
        return fit

    def _evaluate_batch(self, cands, rid):
        fits = list(self._inner.evaluate_batch(cands, rid) or [])
        # a short result still cost the dropped genomes an evaluation slot
        for i, g in enumerate(cands):
            self._record(g, fits[i] if i < len(fits) else None, rid)
        return fits


# ── the arm ──────────────────────────────────────────────────────────────────────────────────
def run(adapter, *, out_dir, ledger: C.Ledger, mode: str = "adaptive", k0: int = 4,
        corpus_dirs=None, rounds: int = 12, batch_size: int = 1, random_init: float = 0.5,
        holdout: bool = True, log=print,
        children_per_parent: int = 2, steer: bool = True, seed: int = 0, n_workers: int = 1,
        parent_select: str = "roundrobin", max_evals=None, seed_genomes=None,
        hier_levels: int = 3, hier_branching: int = 2, level_start=None,
        ref_size: int = DIVERSITY_REFERENCE_SIZE, self_exclusion: bool = True, desc_dir=None,
        extra_knobs=None, allow_collapse=False, **_) -> dict:
    """Run one of our three modes over upstream's frozen descriptor, on the shared corpus."""
    from harness_map import main as MH
    from harness_map.main import Node
    from harness_map.main import Genome
    from harness_map.qd import coverage as _cov
    from harness_map.qd import modes as _MODES
    from harness_map.qd import partition as _part

    mode = str(mode).replace("mapelites_", "")
    if mode not in _MODE_KEYS:
        raise ValueError(f"unknown mode {mode!r}; choose one of {sorted(_MODE_KEYS)}")
    mode_key = _MODE_KEYS[mode]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not corpus_dirs:
        raise ValueError(
            "open-harness needs `corpus_dirs`: the bootstrap corpus of harness variants the "
            "descriptor is frozen on and the CVT is fitted to. Pass the SAME corpus the ours arms "
            "used (the proposer corpus of examples/run_benchmark.py) so the comparison isolates "
            "the descriptor; a single-anchor corpus collapses k-means to one cell.")

    anchor = adapter.anchor()
    anchor = anchor[0] if isinstance(anchor, (list, tuple)) else anchor
    anchor_dir = Path(anchor.source_dir)
    # The anchor belongs to the corpus; de-duplicate keeping order (k-means++ draws by index).
    dirs, seen = [], set()
    for d in [Path(x) for x in corpus_dirs] + [anchor_dir]:
        r = str(Path(d).resolve())
        if r not in seen:
            seen.add(r)
            dirs.append(Path(d))

    # (1) freeze the descriptor and fit the partition (no evaluations); `desc_dir` reuses one
    if desc_dir:
        desc, Z0, labels, J = FrozenOpenEvolveDescriptor.load(
            desc_dir, bundle=_bundler(adapter), log=log)
        log(f"[open-harness] descriptor REUSED from {desc_dir}: {len(desc.refs)} reference "
            f"programs, k={len(Z0)} centroids, corpus of record n={len(desc.corpus_dirs)} "
            f"(this run did NOT refit; declared deviation)")
    else:
        desc = FrozenOpenEvolveDescriptor(dirs, bundle=_bundler(adapter), ref_size=ref_size,
                                          seed=seed, self_exclusion=self_exclusion, log=log)
        log(f"[open-harness] descriptor frozen: {len(desc.refs)} reference programs "
            f"(upstream default size {DIVERSITY_REFERENCE_SIZE}), corpus n={len(dirs)}")
        Z0, labels, J = desc.fit(k0, seed=seed)
    geom_tel = _log_geometry(desc, J, labels, len(Z0), log)
    k_realized = len(Z0)
    if k_realized != k0:
        log(f"[open-harness] WARNING: kmeans collapsed k0 {k0} -> {k_realized}; the archive, the "
            f"coverage denominator and N are all denominated in the REALIZED k.")
        # A collapse to one cell is a hill-climb reported like an archive run: refuse by default.
        if not allow_collapse:
            raise ValueError(
                f"the OpenEvolve descriptor collapsed k0={k0} to {k_realized} distinct centroids on "
                f"this corpus ({len(dirs)} harnesses, {geom_tel.get('n_distinct_phi')} distinct phi "
                f"points, corr={geom_tel.get('corr_complexity_diversity')}). Lower k0, widen the "
                f"corpus, or pass allow_collapse=True to run it anyway and report k0_realized.")
    desc.save(out / "oe_descriptor")
    np.savez(out / "oe_descriptor" / "partition.npz", Z0=Z0, J=J, labels=np.asarray(labels),
             mu=desc.mu, sd=desc.sd)

    # (2) the geometry: a flat centroid set, or for hier a ladder whose rung 0 is Z0
    ladder = None
    if mode == "hier":
        nodes = [Node(idx=i, genome=Genome(source_dir=Path(d), label=f"corpus_{i}"), fit=None)
                 for i, d in enumerate(dirs)]
        # the same seed as Z0, so rung 0 is the partition saved above
        ladder = _part.build_centroid_ladder(desc, nodes, k0=k_realized, levels=hier_levels,
                                             branching=hier_branching, seed=seed, log=log)
        # the ladder is never persisted by the method; freeze it for reproducibility
        np.savez(out / "oe_descriptor" / "ladder.npz",
                 **{f"rung_{l}": np.asarray(z) for l, z in enumerate(ladder)})
        (out / "oe_descriptor" / "ladder_parents.json").write_text(
            json.dumps(C._jsonable(getattr(ladder, "parents", None))))
        log(f"[open-harness] hier ladder REALIZED rungs = {[len(z) for z in ladder]} "
            f"(requested levels={hier_levels} branching={hier_branching}). A 2-D, possibly "
            f"near-rank-1 corpus cannot always be subdivided that far; the realized sizes are the "
            f"ones to report.")
    _geom = {"centroid_levels": ladder} if mode == "hier" else {"init_centroids": Z0}

    rl = C.RunLog(out_dir, f"open_harness_{mode}", {
        "engine": f"harness_map MH.run(mode={mode_key})",
        "ablation": "open-harness: OUR search, OPENEVOLVE's behavioural descriptor",
        "upstream": pins()["openevolve"],
        "descriptor": desc.meta(), "geometry": geom_tel,
        "k0_requested": k0, "k0_realized": k_realized,
        "rounds": rounds, "batch_size": batch_size, "random_init": random_init,
        "children_per_parent": children_per_parent, "steer": steer,
        "block_split": None, "coverage_safety": _cov.COVERAGE_SAFETY_DEFAULT,
        "hier": ({"levels": hier_levels, "branching": hier_branching,
                  "rungs": [len(z) for z in ladder]} if ladder is not None else None)})

    # (3) the knobs, filtered to what this mode's signature declares. P defaults to log2(k0)
    # (realized), the value the budget identity N = M + G*P*C is stated against.
    if batch_size is None:
        batch_size = max(1, int(math.log2(max(k_realized, 2))))
        log(f"[open-harness] batch_size P not given -> log2(k0_realized={k_realized}) = {batch_size} "
            f"(the sweep-recipe value; N = M + G*P*C is stated against it)")
    wanted = {"rounds": rounds, "batch_size": batch_size, "n_workers": n_workers,
              "parent_select": parent_select, "seed": seed,
              "children_per_parent": children_per_parent, "random_init": random_init,
              "steer": steer, "level_start": level_start,
              # The mode's own in-loop evaluation bound, in the same unit the ledger caps on, so
              # the search stops at N instead of being unwound by an exception.
              "max_evals": (max_evals if max_evals is not None else ledger.max_cand_evals),
              **{k: v for k, v in dict(extra_knobs or {}).items() if k not in _RESERVED_KNOBS}}
    impl = {"mapelites_adaptive": _MODES._mapelites_adaptive,
            "mapelites_fixed": _MODES._mapelites_fixed,
            "mapelites_hier": _MODES._mapelites_hier}[mode_key]
    accepted = set(inspect.signature(MH._modes()[mode_key].run).parameters)
    if "knobs" in accepted:                              # run(ctx, **knobs) -> ask the impl
        accepted = set(inspect.signature(impl).parameters)
    knobs = {k: v for k, v in wanted.items() if k in accepted and v is not None}
    # Report the two reasons separately. Conflating them hid a batch_size=None fallthrough as
    # "the mode does not take batch_size", which is false and changes N.
    unsupported = sorted(k for k in wanted if k not in accepted)
    unset = sorted(k for k in wanted if k in accepted and wanted[k] is None)
    if unsupported:
        log(f"[open-harness] mode {mode} does not accept {unsupported}; dropped")
    if unset:
        log(f"[open-harness] mode {mode} accepts {unset} but no value was given; the mode default "
            f"applies (check this is what you intend for N)")

    pool: list[C.Cand] = []
    metered = _MeteredAdapter(adapter, ledger=ledger, pool=pool, rl=rl, log=log)
    log(f"[open-harness] mode={mode_key} k0={k_realized} rounds={rounds} "
        f"batch_size={batch_size} m={random_init} C={children_per_parent} steer={steer} "
        f"budget={ledger.max_cand_evals or '-'} cand evals")
    log("[open-harness] block_split=None: upstream's descriptor has no structure/token split, so "
        "cover_aim returns no aim and the coverage directive keeps its NOVELTY framing (declared "
        "deviation 6: the proposer prompt differs from the ours arms).")

    # the adapter's cfg over the package defaults (the modes read selection/fitness keys)
    cfg = MH.load(getattr(adapter, "cfg", None))

    result, cov_err, run_err = None, None, None
    try:
        result = MH.run(metered, cfg, mode=mode_key, log=log,
                        phi_fn=desc, block_split=None, seed_genomes=list(seed_genomes or []),
                        should_stop=(lambda: ledger.exhausted()), **_geom, **knobs)
    except _cov.CoverageError as e:
        # A coverage shortfall is this arm's headline result: the descriptor could not separate
        # ceil(m*k0) niches on this corpus. Record it and report what was spent.
        cov_err = repr(e)
        log(f"[open-harness] CoverageError at k0={k_realized}: {e}")
        log("[open-harness] the descriptor could not cover the requested fraction of K0. Report "
            "this as a property of upstream's descriptor; retry at a smaller k0 or with "
            "COVERAGE_SAFETY raised before concluding anything about the search.")
    except C.BudgetExhausted as e:
        log(f"[open-harness] budget exhausted: {e}")
    except Exception as e:                               # a failed run must still report its spend
        run_err = repr(e)
        log(f"[open-harness] MH.run raised: {e!r}")
        rl.write("open_harness_error.json", {"error": run_err})

    # each mode reports under its own key: frontier / fixed / hier
    fr = ((result or {}).get(_TELEMETRY_KEY[mode]) or {})
    collapse_tel = {"k0_requested": k0, "k0_realized": k_realized,
                    "k0_collapsed": k_realized != k0, "allow_collapse": bool(allow_collapse)}
    if fr:
        rl.write("frontier.json", fr)                    # the full telemetry, out of summary.json

    # (4) the winner: the mode's own ranking, matched back to the metered pool; pick_best otherwise
    best, selected_by = None, "pick_best"
    mb = (result or {}).get("best") or None
    if mb is not None:
        src = str(Path(mb["genome"].source_dir).resolve())
        for c in pool:
            if str(Path(c.genome.source_dir).resolve()) == src:
                best, selected_by = c, "mode_rank"
                break
    if best is None or not best.valid:
        best, selected_by = (C.pick_best(pool) or best), "pick_best"

    extra = {
        "ablation": "open-harness",
        "mode": mode, "mode_key": mode_key,
        "descriptor": desc.meta(), "geometry": geom_tel,
        **collapse_tel,
        "ladder_rungs": ([len(z) for z in ladder] if ladder is not None else None),
        "phi_artifact": str(out / "oe_descriptor"),
        "block_split": None,
        "selected_by": selected_by,
        "coverage_error": cov_err, "run_error": run_err,
        "coverage_safety": _cov.COVERAGE_SAFETY_DEFAULT,
        # The compact slice of the mode's telemetry a results table joins on; frontier.json has all.
        "frontier": {k: fr.get(k) for k in ("final_cells", "final_occupied", "final_depths",
                                            "mixed_resolution", "n_splits", "n_retreats",
                                            "init", "budget", "audit")} if fr else None,
        "proposals_note": "ledger.proposals is 0 by design (deviation 9): propose is passed "
                          "through unwrapped so the modes introspect the real adapter's signature; "
                          "proposal counts live in frontier.budget / frontier.init.",
        "deviations": [
            "diversity reference set, its size and the self-exclusion rule frozen at fit time",
            "both axes normalised by frozen corpus z-score + 1/sqrt(2), not upstream's running "
            "min/max onto a feature_bins grid",
            "upstream's <2-programs diversity warm-up branch removed",
            "reference selection seeded (upstream uses a bare random.randint)",
            "complexity measured on the single-file bundle, the same text the openevolve arm bins",
            "block_split=None: cover_aim gives no aim and the coverage directive falls back to "
            "NOVELTY framing, so the proposer prompt differs from the ours arms",
            "the CVT is fitted on the same bootstrap corpus as the ours arms (never evaluated)",
            "holdout scored once, by common.finalize, with the mode's own holdout suppressed",
            "proposals are not metered (signature introspection fidelity)",
        ],
    }
    return C.finalize(adapter, rl, ledger, pool, best, extra=extra, holdout=holdout, logf=log)