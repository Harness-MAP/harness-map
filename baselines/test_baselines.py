#!/usr/bin/env python3
"""Mock-adapter tests for the arms: no GPU, no model, no benchmark.

    python baselines/test_baselines.py
"""
from __future__ import annotations

import json
import random
import shutil
import sys
import tempfile

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baselines import common as C                                                # noqa: E402
from baselines import dgm, gepa, meta_harness, upstream              # noqa: E402
from baselines import openevolve, shinka_arm                             # noqa: E402
from harness_map.main import EpisodeResult, Fitness, Genome                # noqa: E402


N_BITS = 8
N_TASKS = 6


class MockAdapter:
    """A genome is a bit vector on disk; an instance scores the fraction of its needed bits set."""

    def __init__(self, root: Path, seed: int = 0):
        self.root = root
        self.rng = random.Random(seed)
        self.evolve = [{"instance_id": f"t{i}"} for i in range(N_TASKS)]
        self.evalset = [{"instance_id": f"h{i}"} for i in range(3)]
        r = random.Random(1234)
        self.needs = {f"t{i}": sorted(r.sample(range(N_BITS), 3)) for i in range(N_TASKS)}
        self.needs.update({f"h{i}": sorted(r.sample(range(N_BITS), 3)) for i in range(3)})
        self.n_eval, self.n_subset, self.n_propose = 0, 0, 0
        self.directives: list[str] = []
        self._seed = self._write("seed_000", [0] * N_BITS)

    def anchor(self):
        return Genome(source_dir=self._seed, label="seed_000")

    def propose(self, parent, rid=0, directive=None, n_children=None, **kw):
        k = int(n_children or 1)
        self.n_propose += 1
        if directive:
            self.directives.append(directive)
        bits = self._read(Path(parent.source_dir))
        out = []
        for j in range(k):
            child = list(bits)
            off = [i for i, b in enumerate(child) if not b]
            if off:
                child[self.rng.choice(off)] = 1
            name = f"c{rid}_{j}_{self.n_propose}"
            out.append(Genome(source_dir=self._write(name, child),
                              parent_dir=Path(parent.source_dir), label=name))
        return out

    def evaluate(self, genome, rid=0):
        self.n_eval += 1
        return self._fit(genome, [t["instance_id"] for t in self.evolve])

    def evaluate_subset(self, genome, task_ids, rid=0):
        self.n_subset += 1
        want = [t["instance_id"] for t in self.evolve if t["instance_id"] in set(task_ids)]
        return self._fit(genome, want) if want else None

    def evaluate_holdout(self, genome):
        return self._fit(genome, [t["instance_id"] for t in self.evalset])

    def _fit(self, genome, ids):
        bits = self._read(Path(genome.source_dir))
        rows = [EpisodeResult(task_id=i,
                              primary=sum(bits[b] for b in self.needs[i]) / len(self.needs[i]))
                for i in ids]
        mean = sum(r.primary for r in rows) / len(rows)
        return Fitness(primary=round(mean, 6), rows=rows, valid=True, n=len(rows),
                       dense={"n_bits_on": sum(bits)})

    def _write(self, name, bits) -> Path:
        """ONE text file, so the harness is a single GEPA component a reflection LM can rewrite."""
        d = self.root / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "orchestrator.py").write_text("BITS = %r\n" % (bits,))
        return d

    @staticmethod
    def _read(d: Path):
        import ast, re
        m = re.search(r"BITS\s*=\s*(\[[^\]]*\])", (d / "orchestrator.py").read_text())
        bits = ast.literal_eval(m.group(1)) if m else [0] * N_BITS
        bits = [int(bool(b)) for b in bits][:N_BITS]
        return bits + [0] * (N_BITS - len(bits))


def _fake_reflection_lm(prompt: str) -> str:
    """A stand-in proposer LM: reads the candidate out of the prompt and turns one more bit on."""
    import ast
    import re
    found = re.findall(r"BITS\s*=\s*(\[[^\]]*\])", prompt)
    bits = [0] * N_BITS
    for raw in found:                                    # last match = the most recent candidate
        try:
            b = [int(bool(x)) for x in ast.literal_eval(raw)][:N_BITS]
            bits = b + [0] * (N_BITS - len(b))
        except Exception:
            continue
    off = [i for i, b in enumerate(bits) if not b]
    if off:
        bits[off[0]] = 1
    return f"Turning on one more capability bit.\n\n```\nBITS = {bits!r}\n```\n"


FAILS = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def quiet(*a, **k):
    pass


# ── 1. the vendored upstreams load, and are the real thing ───────────────────
def test_upstream():
    print("\nupstream: the vendored repositories load")
    g = upstream.load_gepa()
    check(str(upstream.VENDOR) in g.__file__, f"gepa loaded from vendor ({g.__file__})")
    check(callable(getattr(g, "optimize", None)), "gepa.optimize is the real entry point")

    d = upstream.load_dgm()
    check(str(upstream.VENDOR) in d.__file__, "DGM_outer loaded from vendor")
    for fn in ("choose_selfimproves", "filter_compiled", "update_archive", "get_original_score",
               "any_exceeding_context_length"):
        check(callable(getattr(d, fn, None)), f"upstream DGM_outer.{fn} is available")
    try:
        d.self_improve()
        check(False, "the stubbed Docker self-improve step raises")
    except NotImplementedError:
        check(True, "the stubbed Docker self-improve step raises instead of no-oping")

    m = upstream.load_meta_harness()
    check(str(upstream.VENDOR) in m.__file__, "meta_harness loaded from vendor")
    for fn in ("update_frontier", "update_evolution_summary", "count_iterations"):
        check(callable(getattr(m, fn, None)), f"upstream meta_harness.{fn} is available")

    check(not any(str(upstream.VENDOR / "dgm") in p or "terminal_bench_2" in p for p in sys.path),
          "dgm / tb2 paths do not leak into sys.path (they shadow this repo's own modules)")
    p = upstream.pins()
    check(all(len(v["commit"]) == 40 for v in p.values()),
          f"every clone reports a pinned commit ({', '.join(v['commit'][:7] for v in p.values())})")


# ── 2. upstream GEPA's selection behaves as the docs claim ───────────────────
def test_gepa_upstream_selection():
    print("\ngepa: upstream Pareto selection (vendor/gepa/src/gepa/gepa_utils.py)")
    upstream.load_gepa()
    from gepa.gepa_utils import remove_dominated_programs, select_program_candidate_from_pareto_front

    surv = remove_dominated_programs({"a": {0}, "b": {0, 1}}, scores=[0.9, 0.5])
    check(surv == {"a": {0}, "b": {0}}, f"a nowhere-uniquely-best program is dominated ({surv})")

    front = {"a": {0}, "b": {1}, "c": {0}}
    check(remove_dominated_programs(front, scores=[0.9, 0.5]) == front,
          "a uniquely-best specialist survives even with the lower aggregate score")
    picks = [select_program_candidate_from_pareto_front(front, [0.9, 0.5], random.Random(s))
             for s in range(400)]
    share0 = picks.count(0) / len(picks)
    check(0.60 < share0 < 0.73,
          f"sampled proportional to instances-best-on (2/3 expected, got {share0:.2f})")
    check(picks.count(1) > 0, "the low-aggregate specialist is still bred from")


# ── 3. our on-disk layout satisfies upstream DGM's file-driven readers ───────
def test_dgm_layout_contract():
    print("\ndgm: the metadata layout upstream reads")
    D = upstream.load_dgm()
    root = Path(tempfile.mkdtemp())
    try:
        out = root / "output_dgm"
        g = Genome(source_dir=root, label="x")
        init = C.Cand(idx=0, genome=g, fit=Fitness(primary=0.5, valid=True),
                      scores={"a": 1.0, "b": 0.4, "c": 0.0})
        kid = C.Cand(idx=1, genome=g, fit=Fitness(primary=0.9, valid=True),
                     scores={"a": 1.0, "b": 0.9, "c": 0.8}, parent=0)
        dgm._write_node(out, "initial", None, init, 1.0)
        dgm._write_node(out, "kid1", "initial", kid, 1.0)

        md = json.loads((out / "initial" / "metadata.json").read_text())
        check(md["overall_performance"]["total_resolved_ids"] == ["a"]
              and md["overall_performance"]["total_unresolved_ids"] == ["b"]
              and md["overall_performance"]["total_emptypatch_ids"] == ["c"],
              "resolved / unresolved / empty written in upstream's schema")

        check(D.get_original_score(str(out)) == 0.5,
              "upstream get_original_score reads our metadata")            # UPSTREAM
        check(D.any_exceeding_context_length(str(out), "initial", ["a", "b"]) is False,
              "upstream any_exceeding_context_length runs against our prediction logs")  # UPSTREAM

        import logging
        lg = logging.getLogger("t"); lg.addHandler(logging.NullHandler())
        check(D.filter_compiled(["kid1"], str(out), num_swe_issues=[3], logger=lg) == ["kid1"],
              "upstream filter_compiled admits a scored child")            # UPSTREAM
        check(D.update_archive(str(out), ["initial"], ["kid1"], method="keep_all") ==
              ["initial", "kid1"], "upstream update_archive keep_all")     # UPSTREAM

        random.seed(0)
        ents = D.choose_selfimproves(str(out), ["initial", "kid1"], 2,   # UPSTREAM
                                     method="score_child_prop", polyglot=False)
        check(all(e[0] in ("initial", "kid1") for e in ents) and ents,
              f"upstream choose_selfimproves returns (parent, entry) pairs from our layout ({ents})")
        unres = {"initial": set(dgm._unresolved(init, 1.0)),
                 "kid1": set(dgm._unresolved(kid, 1.0))}
        check(all(e[1].startswith("solve_") or e[1] in unres[e[0]] for e in ents),
              f"each entry is one of THAT parent's unresolved instances, or a systematic mode "
              f"({ents})")

        # the upstream crash our arm has to survive: a parent with NO unresolved instances
        solved = C.Cand(idx=2, genome=g, fit=Fitness(primary=1.0, valid=True),
                        scores={"a": 1.0, "b": 1.0}, parent=0)
        dgm._write_node(out, "solved", "initial", solved, 1.0)
        raised = False
        try:
            for s in range(40):
                random.seed(s)
                D.choose_selfimproves(str(out), ["solved"], 1, method="score_child_prop",
                                      polyglot=False)
        except IndexError:
            raised = True
        check(raised, "upstream's empty-unresolved case does crash (DGM_outer.py:143)")
        got = dgm._choose(D, str(out), ["solved"], 1, "score_child_prop", None,
                              {"solved": solved}, 1.0, quiet)
        check(got == [], "our arm survives it and returns no attempt for that parent")
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ── 4. common substrate ──────────────────────────────────────────────────────
def test_common():
    print("\ncommon: per-instance scores, budget, validation")
    f = Fitness(primary=0.5, rows=[EpisodeResult(task_id="a", primary=0.2),
                                   EpisodeResult(task_id="a", primary=0.4),
                                   EpisodeResult(task_id="b", primary=1.0)],
                vector={"v1": 0.7}, valid=True)
    pi = C.per_instance(f)
    check(set(pi) == {"a", "b"} and abs(pi["a"] - 0.3) < 1e-9 and pi["b"] == 1.0,
          "rows win, repeats of an id are averaged")
    check(C.per_instance(f, "objective") == {"v1": 0.7}, "objective front reads the vector")
    check(C.per_instance(Fitness(primary=0.4, vector={"v": 0.9})) == {"v": 0.9},
          "no rows -> the vector is the per-instance axis")
    check(C.per_instance(Fitness(primary=0.4)) == {"__pooled__": 0.4},
          "no structure at all -> a single pooled key, not a crash")

    led = C.Ledger(max_cand_evals=2)
    led.spend(5); check(not led.exhausted(), "ledger not exhausted below the cap")
    led.spend(5); check(led.exhausted() and led.metric_calls == 10,
                        "ledger caps on candidate evaluations and totals metric calls")

    root = Path(tempfile.mkdtemp())
    try:
        ok = root / "ok"; ok.mkdir(); (ok / "h.py").write_text("x = 1\n")
        bad = root / "bad"; bad.mkdir(); (bad / "h.py").write_text("def broken(:\n")
        a = MockAdapter(root / "mock")
        check(C.validate(a, Genome(source_dir=ok)), "a compiling harness validates")
        check(not C.validate(a, Genome(source_dir=bad)), "a syntax error fails validation")
        check(not list(ok.glob("*.pyc")), "validation leaves no .pyc behind")
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ── 5. the three arms, end to end ────────────────────────────────────────────
def run_arm(name, root, budget=8):
    ad = MockAdapter(root / f"work_{name}", seed=7)
    led = C.Ledger(max_cand_evals=budget)
    out = root / f"out_{name}"
    if name == "meta-harness":
        s = meta_harness.run(ad, out_dir=out, ledger=led, iterations=20, k=1, log=quiet)
    elif name == "gepa":
        s = gepa.run(ad, out_dir=out, ledger=led, minibatch_size=3, log=quiet,
                         proposer="reflection-lm", reflection_lm=_fake_reflection_lm)
    else:
        s = dgm.run(ad, out_dir=out, ledger=led, generations=20, selfimprove_size=1, k=1,
                        small_n=3, resolve_thresh=1.0, log=quiet)
    return ad, led, s, out


def test_arms():
    root = Path(tempfile.mkdtemp())
    try:
        for name in ("meta-harness", "gepa", "dgm"):
            print(f"\n{name}: end to end on the mock testbed (upstream driving)")
            ad, led, s, out = run_arm(name, root, budget=8)

            # gepa's loop is upstream's: its stop callback may overshoot by one iteration
            slack = 3 if name == "gepa" else 0
            check(led.cand_evals <= 8 + slack,
                  f"budget respected ({led.cand_evals} candidate evals, cap 8 +{slack} slack)")
            check(s["best"] is not None, "a best candidate was selected")
            check(s["best"]["primary"] >= s.get("anchor_primary", s.get("initial_primary", 0.0)),
                  f"best >= anchor ({s['best']['primary']:.3f})")
            check("holdout" in s, "the winner was scored out-of-sample")
            check((out / "summary.json").exists() and (out / "history.jsonl").exists(),
                  "summary.json + history.jsonl written")
            # gepa mutates through upstream's own proposer, so the adapter's is never called
            if name == "gepa":
                check(ad.n_propose == 0,
                      "gepa mutates via upstream's own proposer, not the testbed's")
            else:
                check(ad.n_propose > 0 and all(d.strip() for d in ad.directives),
                      f"the testbed proposer was driven with a non-empty directive "
                      f"({ad.n_propose} calls)")

            want = {"meta-harness": ["frontier_val.json", "evolution_summary.jsonl"],
                    "gepa": ["gepa_result.json"],
                    "dgm": ["dgm_metadata.jsonl", "output_dgm/initial/metadata.json"]}[name]
            check(all((out / w).exists() for w in want),
                  f"upstream-produced artifacts written ({', '.join(want)})")

            if name == "gepa":
                res = json.loads((out / "gepa_result.json").read_text())
                check(res.get("total_metric_calls") and res.get("n_candidates", 0) >= 2,
                      f"upstream GEPAResult captured ({res.get('n_candidates')} candidates, "
                      f"{res.get('total_metric_calls')} metric calls)")
                rd = out / "gepa_run"
                check((rd / "gepa_state.json").exists() and (rd / "iterations").is_dir()
                      and (rd / "pareto").is_dir(),
                      "upstream wrote its own run_dir state (gepa_state.json, iterations/, pareto/)")
                check(res.get("per_val_instance_best_candidates"),
                      "upstream tracked a per-val-instance Pareto frontier")
            if name == "meta-harness":
                fr = json.loads((out / "frontier_val.json").read_text())
                check("_best" in fr and any(k != "_best" for k in fr),
                      "upstream update_frontier wrote per-task entries and a _best")
                n = sum(1 for _ in (out / "evolution_summary.jsonl").open())
                check(n == s["n_evaluated"],
                      f"upstream update_evolution_summary wrote one row per candidate ({n})")
            if name == "dgm":
                arch = s.get("archive") or []
                check("initial" in arch, f"upstream update_archive kept the initial node ({arch})")
                check(ad.n_subset > 0, f"staged screening used the subset hook ({ad.n_subset})")
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ── 6. the two evolution frameworks ──────────────────────────────────────────
def test_bundle_roundtrip():
    """Both frameworks evolve ONE file with EVOLVE-BLOCK regions; a harness is a package."""
    print("\ncommon: whole-package bundle <-> runnable harness")
    root = Path(tempfile.mkdtemp())
    try:
        src = root / "h"; (src / "modules").mkdir(parents=True)
        (src / "orchestrator.py").write_text("import modules.a\nX = 1\n")
        (src / "modules" / "__init__.py").write_text("")
        (src / "modules" / "a.py").write_text("def go():\n    return 42\n")
        (src / "notes.md").write_text("not bundled, but must survive\n")

        b = C.bundle_harness(src)
        files = C._FILE_RE.findall(b)
        check(set(files) == {"orchestrator.py", "modules/__init__.py", "modules/a.py"},
              f"every .py file becomes its own region ({files})")
        check(b.count("# EVOLVE-BLOCK-START") == len(files)
              and b.count("# EVOLVE-BLOCK-END") == len(files),
              "one EVOLVE-BLOCK pair per region (the marker both frameworks read)")

        out = C.unbundle_harness(b, src, root / "out")
        check((out / "modules" / "a.py").read_text().strip() == "def go():\n    return 42",
              "a nested module round-trips byte-for-byte")
        check((out / "notes.md").exists(),
              "non-bundled assets survive (copied from the anchor, not dropped)")

        # an edit inside a region is what lands
        edited = b.replace("return 42", "return 43")
        out2 = C.unbundle_harness(edited, src, root / "out2")
        check("return 43" in (out2 / "modules" / "a.py").read_text(),
              "an edit inside a region lands in the right file")

        # a mangled bundle must fail loudly, not silently run the anchor
        out3 = C.unbundle_harness("total garbage, no markers", src, root / "out3")
        check((out3 / "orchestrator.py").read_text() == "total garbage, no markers",
              "a bundle with no markers is written as-is so it fails validation, not silently "
              "reverting to the anchor")

        # traversal guard
        C.unbundle_harness("# ===== FILE: ../escape.py =====\nX=1\n", src, root / "out4")
        check(not (root / "escape.py").exists(), "a '../' path in a marker cannot escape the dir")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_openevolve_glue():
    """The evaluator-file contract upstream calls from its process pool, over the RPC server."""
    print("\nopenevolve: the evaluator contract upstream calls")
    upstream.load_openevolve()
    from openevolve.config import Config, LLMModelConfig
    cfg = Config()
    check(cfg.database.feature_bins == 10 and cfg.database.num_islands >= 1,
          "upstream MAP-Elites config constructs (bins + islands)")
    check(hasattr(LLMModelConfig(), "api_base"),
          "upstream LLMModelConfig takes an OpenAI-compatible api_base")

    root = Path(tempfile.mkdtemp())
    try:
        ad = MockAdapter(root / "w", seed=3)
        led = C.Ledger()
        anchor_dir = Path(ad.anchor().source_dir)
        task = root / "task"; task.mkdir(parents=True)
        eval_py = task / "evaluate.py"
        eval_py.write_text(openevolve.EVALUATE_PY.format(timeout=60))

        from harness_map.main import Genome
        def handle(program_path):
            text = Path(program_path).read_text()
            d = C.unbundle_harness(text, anchor_dir, root / "cand" / str(len(list((root/'cand').glob('*')) if (root/'cand').exists() else [])))
            g = Genome(source_dir=d, label="cand")
            fit, _ = C.evaluate_on(ad, g, rid=0, ledger=led)
            c = C.Cand(idx=0, genome=g, fit=fit, scores=C.per_instance(fit))
            return openevolve._metrics(c)

        server = openevolve._EvalServer(task / "rpc", handle, log=quiet)
        server.start()
        try:
            # import it exactly as upstream's worker does
            import importlib.util
            spec = importlib.util.spec_from_file_location("oe_eval", eval_py)
            mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
            check(callable(getattr(mod, "evaluate", None)),
                  "the generated file exposes upstream's `evaluate(program_path)` entry point")

            seed_prog = task / "initial.py"
            seed_prog.write_text(C.bundle_harness(anchor_dir))
            m0 = mod.evaluate(str(seed_prog))
            better = task / "child.py"
            better.write_text(C.bundle_harness(anchor_dir).replace(
                "BITS = [0, 0, 0, 0, 0, 0, 0, 0]", "BITS = [1, 1, 1, 0, 0, 0, 0, 0]"))
            m1 = mod.evaluate(str(better))

            check("combined_score" in m0,
                  "the evaluator returns upstream's `combined_score` ranking key")
            check(any(k.startswith("inst_") for k in m0),
                  "per-instance scores ride along in the metrics dict")
            check(m1["combined_score"] > m0["combined_score"],
                  f"an improved bundle scores higher ({m0['combined_score']:.3f} -> "
                  f"{m1['combined_score']:.3f})")
            check(led.cand_evals == 2,
                  f"one candidate evaluation charged per program ({led.cand_evals})")
        finally:
            server.stop(); server.join(timeout=10)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_shinka_rpc():
    """Upstream's `evaluate.py` subprocess contract, executed for real over the file RPC."""
    print("\nshinka: upstream's subprocess evaluator contract, over the file RPC")
    upstream.load_shinka()
    from shinka.launch import LocalJobConfig
    import dataclasses
    check(any(f.name == "python_executable" for f in dataclasses.fields(LocalJobConfig)),
          "upstream LocalJobConfig lets us pin the interpreter that runs evaluate.py")

    root = Path(tempfile.mkdtemp())
    try:
        ad = MockAdapter(root / "w", seed=5)
        led = C.Ledger()
        task = root / "task"; task.mkdir(parents=True)
        (task / "evaluate.py").write_text(shinka_arm.EVALUATE_PY.format(timeout=60))
        anchor = ad.anchor()
        prog = task / "initial.py"
        prog.write_text(C.bundle_harness(Path(anchor.source_dir)))

        pool = []
        def handle(program_path):
            text = Path(program_path).read_text()
            d = C.unbundle_harness(text, Path(anchor.source_dir), root / "cand")
            from harness_map.main import Genome
            g = Genome(source_dir=d, label="cand")
            fit, _ = C.evaluate_on(ad, g, rid=0, ledger=led)
            c = C.Cand(idx=0, genome=g, fit=fit, scores=C.per_instance(fit))
            pool.append(c)
            return shinka_arm._metrics(c), True, ""

        server = shinka_arm._EvalServer(task / "rpc", handle, log=quiet)
        server.start()
        try:
            import subprocess
            res = task / "results"
            r = subprocess.run([sys.executable, str(task / "evaluate.py"),
                                "--program_path", str(prog), "--results_dir", str(res)],
                               capture_output=True, text=True, timeout=120)
            check(r.returncode == 0, f"the generated evaluate.py exits cleanly ({r.returncode})")
            m = json.loads((res / "metrics.json").read_text())
            c = json.loads((res / "correct.json").read_text())
            check("combined_score" in m and "public" in m,
                  "metrics.json carries upstream's combined_score + public block")
            check(c.get("correct") is True and "error" in c,
                  "correct.json carries upstream's correct/error contract")
            check(led.cand_evals == 1 and len(pool) == 1,
                  "the subprocess reached the live adapter and charged one evaluation")
        finally:
            server.stop(); server.join(timeout=10)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_no_subset_hook_charges_full():
    """An adapter without the minibatch hook must be CHARGED for the full split it actually ran."""
    print("\ncommon: a missing evaluate_subset hook is charged in full")
    root = Path(tempfile.mkdtemp())
    try:
        ad = MockAdapter(root / "w")
        saved = MockAdapter.evaluate_subset
        del MockAdapter.evaluate_subset
        try:
            led = C.Ledger()
            fit, n = C.evaluate_on(ad, ad.anchor(), rid=0, ids=["t0"], ledger=led)
            check(n == N_TASKS and led.metric_calls == N_TASKS,
                  f"full split charged when the hook is absent (n={n})")
            check(len(fit.rows) == N_TASKS, "the full Fitness is returned for the caller to filter")
        finally:
            MockAdapter.evaluate_subset = saved
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    test_upstream()
    test_gepa_upstream_selection()
    test_dgm_layout_contract()
    test_common()
    test_arms()
    test_bundle_roundtrip()
    test_openevolve_glue()
    test_shinka_rpc()
    test_no_subset_hook_charges_full()
    print(f"\n{'FAILED: ' + str(len(FAILS)) if FAILS else 'all checks passed'}")
    for f in FAILS:
        print(f"  - {f}")
    sys.exit(1 if FAILS else 0)
