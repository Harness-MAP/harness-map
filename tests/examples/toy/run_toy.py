#!/usr/bin/env python
"""The adaptive mode end to end on a toy task, in about ten seconds and without any model.

    python tests/examples/toy/run_toy.py

A harness is a directory holding one number whose score has two humps (near 0.2 and 0.8); the
script checks that the archive keeps both humps instead of collapsing onto one.
"""
from __future__ import annotations

import math
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import harness_map.main as MH                                            # noqa: E402
from harness_map.main import Fitness, Genome                       # noqa: E402

WORK = Path(tempfile.mkdtemp(prefix="hmap_toy_"))
RNG = random.Random(7)


def score_of(t: float) -> float:
    return round(max(0.0, 1.0 - 8.0 * (t - 0.2) ** 2) * 0.9,
                 4) if t < 0.5 else round(max(0.0, 1.0 - 8.0 * (t - 0.8) ** 2), 4)


def write_harness(t: float, label: str) -> Genome:
    d = WORK / label
    d.mkdir(parents=True, exist_ok=True)
    (d / "system_prompt.txt").write_text(f"temperature = {t:.4f}\n")
    return Genome(source_dir=d, label=label)


def temp_of(genome_or_node) -> float:
    g = getattr(genome_or_node, "genome", genome_or_node)
    txt = (Path(g.source_dir) / "system_prompt.txt").read_text()
    return float(txt.split("=")[1])


class ToyAdapter:
    """The three methods an adapter must provide, plus its config."""

    def __init__(self):
        self.cfg = MH.load()
        self.n_evals = 0

    def anchor(self, cfg=None) -> Genome:
        return write_harness(0.50, "anchor")          # starts in the valley between the humps

    def propose(self, parent, iter_n=0, directive=None, n_children=None, **kw):
        # A real adapter calls the meta-agent here; this one jitters the number.
        out = []
        for i in range(int(n_children or 1)):
            t = min(1.0, max(0.0, temp_of(parent) + RNG.uniform(-0.25, 0.25)))
            out.append(write_harness(t, f"cand_r{iter_n:02d}_{i}_{RNG.randrange(1 << 20):05x}"))
        return out

    def evaluate(self, genome: Genome, iter_n: int = 0) -> Fitness:
        self.n_evals += 1
        s = score_of(temp_of(genome))
        return Fitness(primary=s, rows=[{"task": "toy", "score": s}], vector=[s],
                       valid=True, taint="PASS")


def main() -> int:
    adapter = ToyAdapter()
    k0 = 4
    knobs = dict(
        mode="mapelites_adaptive",
        rounds=12,                     # G, generations
        batch_size=2,                  # P, parents per generation
        children_per_parent=1,         # C, children per parent
        random_init=0.5,               # m, fraction of the k0 cells to cover before breeding
        sigma=0.05,                    # noise floor: split when a cell's range exceeds 2*sigma
        depth_cap=2,
        k_split=2,
        patience=1,
        steer=True,
        # A real benchmark fits phi and the centroids from a corpus (see tests/examples/run_benchmark.py).
        phi_fn=lambda node: [temp_of(node), 0.0],
        init_centroids=[[i / (k0 - 1), 0.0] for i in range(k0)],
        block_split=1,
    )
    n = (math.ceil(knobs["random_init"] * k0)
         + knobs["rounds"] * knobs["batch_size"] * knobs["children_per_parent"])
    print(f"{knobs['mode']}: k0={k0} G={knobs['rounds']} P={knobs['batch_size']} "
          f"C={knobs['children_per_parent']} -> about {n} evaluations\n")

    rep = MH.conformance(adapter)
    print(f"adapter conforms: {rep['ok']}" + (f" ({rep['problems']})" if rep["problems"] else ""))

    out = MH.run(adapter, adapter.cfg, log=print, seed_genomes=[], **knobs)

    fr = out.get("frontier") or {}
    best = out["best"]
    print("\n" + "=" * 78)
    print(f"evaluations      {adapter.n_evals}")
    print(f"best score       {best['fit'].primary:.4f} at temperature {temp_of(best['genome']):.3f}")
    print(f"final cells      {fr.get('final_cells')} (occupied {fr.get('final_occupied')}, "
          f"depths {fr.get('final_depths')})")
    print(f"splits/reverts   {fr.get('n_splits')} / {fr.get('n_retreats')}")

    scored = [(temp_of(r["genome"]), r["fit"].primary) for r in out["history"]
              if r.get("genome") is not None and r.get("fit") is not None]
    low = max((s for t, s in scored if t < 0.5), default=0.0)
    high = max((s for t, s in scored if t >= 0.5), default=0.0)
    kept_both = low > 0.5 and high > 0.5
    print(f"best near 0.2    {low:.4f}")
    print(f"best near 0.8    {high:.4f}")
    print(f"archive holds both humps: {kept_both}")
    return 0 if kept_both else 1


if __name__ == "__main__":
    raise SystemExit(main())
