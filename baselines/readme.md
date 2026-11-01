# Baselines and ablations

The five published harness/program-optimization methods Harness-MAP is compared against in the
paper, plus the two descriptor-swap ablations, run as search arms over the same benchmark adapter
as `examples/run_benchmark.py`. **Each arm runs the original upstream implementation**, cloned at
a pinned commit; the arm replaces only that method's own testbed with ours, so on a benchmark all
arms share the anchor harness, the meta-agent, the evaluator, the splits and the evaluation budget.

## The baselines and where they come from

| arm | method (paper) | upstream repository | commit used |
|---|---|---|---|
| `meta-harness` | Meta-Harness: End-to-End Optimization of Model Harnesses (Lee et al., 2026) | https://github.com/stanford-iris-lab/meta-harness | `44b9942` (2026-07-11) |
| `gepa` | GEPA: Reflective Prompt Evolution Can Outperform RL, via `optimize_anything` | https://github.com/gepa-ai/gepa | `d26029f` (2026-08-15) |
| `dgm` | Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents | https://github.com/jennyzzt/dgm | `a565fd2` (2025-08-13) |
| `openevolve` | OpenEvolve, the open-source AlphaEvolve (MAP-Elites) | https://github.com/algorithmicsuperintelligence/openevolve | `411fb59` (2026-07-18) |
| `shinka` | ShinkaEvolve: Towards Open-Ended and Sample-Efficient Program Evolution | https://github.com/SakanaAI/shinkaevolve | `2bf8cfe` (2026-08-16) |

The two ablations reuse the OpenEvolve clone:

| arm | search loop | niche descriptor |
|---|---|---|
| `openevolve_cvt` | upstream OpenEvolve (islands, diff patching, inspiration sampling) | **ours**: the frozen CVT cell of `harness_map.qd` |
| `open_harness` | **ours**: coverage-gated initialization, resolution control, steering | upstream's `(complexity, diversity)` |

Read as a pair, they split the Harness-MAP-vs-OpenEvolve gap into the archive key and everything
else.

### Fetching the upstream code

```bash
bash baselines/clone.sh          # clones the five repositories into baselines/vendor/ at the
                                 # pinned commits, installs dacite + flask (OpenEvolve) and builds
                                 # ShinkaEvolve's dependency sidecar in baselines/.venv_shinka/
python baselines/upstream.py     # prints each clone's pinned and checked-out commit
```

Set `HARNESS_MAP_VENDOR` (and `SHINKA_VENV`) to keep the clones elsewhere. Nothing under
`vendor/` is edited: `upstream.py` is the only place that knows how to import each repository and
documents the two DGM modules it stubs (the Docker/SWE-bench self-improve step, which *is* the
testbed being replaced). ShinkaEvolve's dependencies (hydra-core, instructor, moviepy, boto3, …)
pin packages such as pydantic, so they live in a sidecar virtualenv that `upstream.py` appends to
`sys.path`; packages you already have keep precedence.

## Running an arm

Set up a benchmark as in `examples/<benchmark>/README.md` (anchor harness, evaluator command,
config), then:

```bash
python baselines/run_baseline.py --config examples/balrog/config.json --anchor <ANCHOR_DIR> \
    --baseline meta-harness --budget 50
```

`--baseline` is one of `meta-harness gepa dgm openevolve shinka openevolve_cvt open_harness`.
`--budget` caps **candidate evaluations**, the unit N of the method's own runs (default:
`search.budget` of the config, 50 for BALROG and the 3D games, 16 for navigation).
`--metric-budget` additionally caps per-instance evaluations, GEPA's own currency. The run
directory (default `runs/<testbed>/<split>/<baseline>`) holds `history.jsonl` (one row per
evaluated candidate, the same schema for every arm), `summary.json` (the winner, its held-out
score, what was spent and why the run stopped) and the artifacts the upstream writes itself.
Writing a new integer to `<run dir>/BUDGET_OVERRIDE` extends a live run without restarting it.

### The settings of the paper's tables

Every arm was run at the same N as the Harness-MAP arm it is compared against and reported on the
same split (the held-out split where the benchmark has one, see the examples' READMEs).

| arm | proposer / mutation model | flags beyond `--budget` |
|---|---|---|
| `meta-harness` | the shared meta-agent (`meta_agent.backbone`) | `--k 2` (two candidates per iteration, the method's own default) |
| `dgm` | the shared meta-agent | none (2 self-improve attempts per generation, 1 child each) |
| `gepa` | upstream's Claude Code proposer (`--gepa-proposer claude-code --gepa-model opus`) | `--gepa-workers 1` |
| `shinka` | `--shinka-models headless/claude` (Claude through ShinkaEvolve's headless backend; needs `node`/`npx`) | native patch mix (diff/full/cross) |
| `openevolve` | an OpenAI-compatible endpoint (`OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_META_MODEL` or `--oe-model`); OpenEvolve has no Claude path, so the paper's runs used a hosted model or Claude behind an OpenAI-compatible shim that is not part of this release | `--oe-islands 1` when the endpoint serializes requests |

`meta-harness` and `dgm` propose through the benchmark's own meta-agent, exactly as Harness-MAP does, so
against it they isolate the search algorithm. `gepa`, `shinka` and `openevolve` propose through
their own operators because those operators are part of what the method publishes; their rows
answer "how does the method as published do here". The OpenEvolve backbone asymmetry is a disclosed
one, not an oversight.

The ablations take the cell count and the corpus the Harness-MAP run used:

```bash
# their loop, our key: reuse the frozen CVT of a Harness-MAP run
python baselines/run_baseline.py --config examples/balrog/config.json --anchor <ANCHOR_DIR> \
    --baseline openevolve_cvt --abl-k0 4 --abl-cvt-dir runs/balrog-minihack/search/adaptive_k4/cvt_k4
# our loop, their key: the same proposer corpus (cached per benchmark by both drivers)
python baselines/run_baseline.py --config examples/balrog/config.json --anchor <ANCHOR_DIR> \
    --baseline open_harness --abl-mode adaptive --abl-k0 4
```

Without `--abl-cvt-dir` / `--corpus`, both fit their partition on the proposer-made corpus that
`examples/run_benchmark.py` caches under `runs/<testbed>/corpus`, the corpus the Harness-MAP runs
fit theirs on. `open_harness` refuses a partition that k-means collapsed below `k0`
(`--abl-allow-collapse` runs it anyway and reports `k0_realized`), and `--abl-desc-dir` reuses the
descriptor an earlier run froze, since refitting a length-dominated descriptor is not stable
across corpus draws.

### What the adapter provides

The arms use the `CommandAdapter` of `examples/run_benchmark.py`: `anchor()`, `propose()` (the
shared meta-agent), `evaluate()`, `evaluate_subset(genome, task_ids)` (a minibatch, used by GEPA
and DGM; the search split restricted to the given task ids), `evaluate_holdout()`, and the task
records of the two splits as `evolve` / `evalset`. Per-instance scores come from the score file's
per-episode rows (`cfg["episode"]`): GEPA's Pareto front and DGM's failing-instance draw are keyed
by them, falling back to the per-task vector, then to a single pooled score. For the single-file
methods (OpenEvolve, ShinkaEvolve, GEPA's components), the harness package is bundled into one
program with one `EVOLVE-BLOCK` region per file; the evolvable surface is every `.py` and `.md`
except `system_prompt.txt`, `NOTES.md` and `memory.md` (override with a `baselines` block in the
config: `{"bundle_suffixes": [...], "bundle_exclude": [...]}`).

## What we add to every baseline: anti-cheat and a budget-matched protocol

None of the upstream methods ships either, and a comparison is only as good as both, so the two
are applied to every arm identically, outside the upstream code:

**Anti-cheat.**
- Every candidate of every arm passes the same static privileged-key gate as Harness-MAP's own
  candidates before it is evaluated (`common.validate`): code that reads a forbidden observation
  field (`audit.privileged_keys` in the config) or does not compile is rejected and never scored.
  This is a uniformity of the release; the paper's baseline runs relied on the evaluators' runtime
  guards instead.
- Information hiding at run time is the evaluator's (`examples/<benchmark>/README.md`): a harness
  only ever sees the sanctioned observation, whichever arm proposed it.
- A failed evaluation (crash, timeout, fewer episodes than the split) is recorded as invalid, not
  as a score of 0, and is excluded from selection; `summary.json` carries `n_valid`, so a run whose
  seed evaluation failed is visible (dgm stops outright, openevolve_cvt marks the run void).
- The LLM judge is part of Harness-MAP's own selection funnel and is not applied inside the
  baselines' loops; a baseline's winner can be judged afterwards with the same skill
  (`harness_map.meta_agent.agent.run_judge`), as the paper did post hoc.

**Budget matching.**
- One currency: candidate evaluations (`Ledger.cand_evals`). `--budget N` is the same N as the
  Harness-MAP run's N = M + G·P·C (default `search.budget`: 50 for BALROG and the 3D games, 16 for
  navigation). Every candidate handed to the evaluator is charged, valid or not; a minibatch
  (`evaluate_subset`) is charged by its instance count in the second currency, metric calls, which
  is also recorded (`--metric-budget` caps it, GEPA's own `max_metric_calls`).
- How each arm is held to N, since none of the upstream loops takes a budget: meta-harness and dgm check
  the ledger inside their loop and stop exactly at the cap; gepa through a stop callback the engine
  tests once per iteration, so it can overshoot by one iteration's evaluations (`--metric-budget`
  is the strict form); openevolve, openevolve_cvt and shinka by refusing to evaluate new
  candidates past the cap; open_harness through the mode's own `max_evals`. The loop counts
  default high (10 × N for meta-harness and dgm, derived from the cap for openevolve and shinka) so that
  the budget, not a loop count, is what stops a run.
- Every arm stops once a candidate scores a perfect 1.0, so evaluations-to-best are comparable.
- The reported number is the winner's held-out score (where the benchmark has a held-out split),
  evaluated once, after the search and outside the budget; a winner scored only on subsets during
  the search is re-scored on the full search split first. The search-split maximum flatters
  whichever arm evaluated the most candidates and is not the comparison.
- The ablations' partition corpus is never evaluated (N is unchanged) but costs proposer tokens,
  exactly as the Harness-MAP runs' corpus does; both fit on the same corpus.
- `summary.json` records what was actually spent (`budget.cand_evals`, `budget.metric_calls`) and
  why the run stopped; report those, not the cap.

## What each arm replaces, and the divergences

| arm | upstream code that runs the search | what is replaced |
|---|---|---|
| `gepa` | all of it — `optimize_anything`: reflective mutation, the Pareto frontier over instances, minibatch sampling, acceptance, state, budget, and its own Claude Code proposer | nothing; we supply the seed candidate (one component per file), the evaluator and a config |
| `dgm` | `choose_selfimproves`, `filter_compiled`, `update_archive`, `get_original_score`, `any_exceeding_context_length` | `self_improve_step.self_improve` (a Docker container running DGM's coding agent on SWE-bench) |
| `meta-harness` | `update_frontier`, `update_evolution_summary`, `count_iterations` | `harbor_run` / `parse_job_results` (Harbor on Terminal-Bench 2) and the TB2-specific proposer skill |
| `openevolve` | all of it — `run_evolution`: the MAP-Elites archive, islands, prompt sampling, diff patching, checkpointing | nothing; we supply the initial program and an `evaluate(program_path)` file |
| `shinka` | all of it — `ShinkaEvolveRunner`: island database, bandit model selection, patch-type mixture, novelty rejection | nothing in the search; its `evaluate.py` subprocess reaches the live adapter over a file RPC |

Three rules hold for every arm, so that evaluations-to-best are comparable: a run stops once a
candidate scores a perfect 1.0 (GEPA's `skip_perfect_score`, OpenEvolve's `target_score`, an added
check in meta-harness, dgm and shinka); a winner that was only ever scored on a subset or per instance
(dgm, gepa) is re-scored on the full search split before it is reported, without being charged;
and the proposer backbones the frameworks own (GEPA's reflection LM, OpenEvolve, ShinkaEvolve) run
at `OPENAI_REASONING_EFFORT` (default `high`), the effort the shared meta-agent runs at.

Divergences, all recorded in each run's `run_meta.json`:

- **gepa** — driven through `gepa.gepa_launcher.optimize_anything` (the signature that takes a
  dict candidate) in generalization mode (`dataset` = `valset` = the search split), with Pareto
  candidate selection, strict-improvement acceptance and `write_agent_state` so the Claude Code
  proposer reads the run tree; candidates are content-addressed into `candidates/cand_<hash>/` so
  one full evaluation per candidate is charged however many (candidate, instance) calls the engine
  makes; non-text files are copied, not exposed as components; a crashed candidate scores 0
  instead of ending the run; the run stops on the budget or when the proposer stops producing new
  candidates (`GEPA_NOPROG_CAP` flat iterations, default 60).
- **dgm** — resolved / unresolved / empty instances are derived from per-instance scores (there is
  no binary resolve flag outside SWE-bench; an instance is resolved at `--dgm-resolve-thresh`,
  default 1.0); the diagnosis prompts are written here because upstream's describe DGM's own
  `coding_agent.py`; two evaluation rungs instead of three (a child is screened on the first
  `--dgm-small-n` = 3 instances and evaluated in full only above `--dgm-threshold` = 0.4); an
  upstream crash on a parent with no unresolved instances is caught and the call retried over
  eligible parents; selection, archive rule and noise leeway keep upstream's defaults.
- **meta-harness** — `compute_pass_rates` is bypassed (it binarizes trials at reward > 0); the prompt is
  rewritten for this anchor while keeping upstream's structure; its smoke test folds into the
  compile check.
- **openevolve / shinka** — the whole package is the program (one region per file, see above);
  duplicate programs are served from a cache; the budget is enforced by refusing to evaluate past
  the cap, because neither library exposes a budget hook; in diff mode both receive the same
  SEARCH/REPLACE discipline in their system/task message, since their diff parsers silently apply
  nothing on an inexact match. OpenEvolve: cascade evaluation off (one evaluation per candidate),
  `max_code_length` sized from the seed (upstream's 10,000-character cap would discard every
  bundle), LLM timeout 1800 s, a checkpoint every 10 iterations instead of upstream's 100, so the
  archive is on disk at these budgets.
  ShinkaEvolve: upstream's native diff/full/cross patch mix (`SHINKA_PATCH_TYPES` overrides it),
  up to `SHINKA_MAX_PATCH_ATTEMPTS` = 3 retries of a failed diff with the parser's error, and
  novelty rejection off unless `--shinka-embedding` names an embedding model, since it calls an
  embedding provider.
- **openevolve_cvt** — `ProgramDatabase._calculate_feature_coords` is overridden (by a subclass
  bound for the duration of the call) to return the frozen CVT cell; the archive is 1-D with k0
  nominal cells, so upstream's inspiration sampling, which perturbs an ordinal coordinate, is close
  to a uniform cell draw and is left as written; the proposer prompt is unchanged.
- **open_harness** — upstream's `(complexity, diversity)` is frozen at fit time (reference set,
  z-score constants, self-exclusion rule) because our archive requires a pure descriptor; there is
  no structure/token split, so the coverage directive keeps its untargeted framing and the proposer
  prompt differs from the Harness-MAP arms; the held-out split is scored once, by the arm.

## Tests

```bash
python baselines/test_baselines.py      # ~1 min, no GPU, no model
```

Checks that each clone loads and behaves as described (upstream's dominance filter, its Pareto
sampling proportions, its stubs raising), and the integration contracts: our on-disk layout
satisfies DGM's file-driven readers, the seed candidate and evaluator satisfy `optimize_anything`
end to end (with a fake reflection LM), meta-harness's frontier functions accept our per-instance
scores, the bundle round-trips a multi-file harness byte for byte, and the OpenEvolve and
ShinkaEvolve evaluator contracts reach the live adapter over the file RPC.
