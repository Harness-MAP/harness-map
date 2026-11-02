# BALROG-MiniHack

Harness-MAP evolves a text-only MiniHack agent for [BALROG](https://github.com/balrog-ai/BALROG),
starting from BALROG's published naive agent. The search set is the eight official MiniHack tasks
with five episodes each (40 episodes per evaluation); the evaluation budget is N = 50.

## Setup

1. **BALROG**, in its own environment (it needs `numpy<2` and NLE):
   ```bash
   git clone https://github.com/balrog-ai/BALROG.git && cd BALROG && git checkout b7afe79
   conda create -n balrog python=3.10 -y && conda activate balrog
   pip install -e . && balrog-post-install
   ```
2. **Base model**: a vLLM server with `Qwen3-VL-30B-A3B-Instruct` (text-only requests are
   enough here):
   ```bash
   vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct --served-model-name Qwen3-VL-30B-A3B-Instruct \
       --port 8000 --max-model-len 65536 --max-num-seqs 8
   ```
3. **Meta-agent**: the Claude Code CLI (`claude`) on `PATH`, or set `meta_agent.backbone` in
   `config.json` to `cli:codex` or `chat-loop` (see the top-level README).
4. **Starting harness**: BALROG's `balrog/agents/naive.py` placed in the harness layout described
   in `harness_map/meta_agent/skills/proposer/SKILL.md`, with `agent.py` exporting
   `build_agent(client_factory, prompt_builder)` that returns a BALROG agent. The paper's starting harness
   reproduces the naive agent unchanged.
5. **Evaluator** (`BALROG_EVALUATOR` in `run.sh`): a script run under the BALROG interpreter as
   `evaluator.py --harness DIR --tasks TASKS.json --out OUT`. It should inject the harness's
   `build_agent` as the agent factory of BALROG's `EvaluatorManager` (which takes a factory as
   an argument, so BALROG itself is not modified), seed each (task, episode index) pair
   deterministically from the split's `seed_base` so every candidate plays the same levels, and
   write `OUT/score.json` with the fields in `config.json`:
   `success` (fraction of episodes whose final status is TASK_SUCCESSFUL; the primary metric),
   `progression`, `survival_rate`, `step_efficiency`, `cost_per_episode`, `n_episodes`,
   `per_task_vector` (one success rate per task in the order of `rubric_labels`), and
   `per_episode` rows with `episode`, `success`, `died`, `steps`, `progression` and
   `total_tokens`. A crashed episode counts as a failure.

## Run

Fill in the placeholders in `run.sh`, then:

```bash
MODE=adaptive K0=4 bash tests/examples/balrog/run.sh          # the paper's main setting
MODE=fixed    K0=4 bash tests/examples/balrog/run.sh          # resolution ablations
MODE=hier     K0=4 bash tests/examples/balrog/run.sh
```

`config.json` fixes the budget at N = 50 and derives the number of generations from k0 with
P = ⌈log2 k0⌉ and C = 2 (k0 ≤ 4) or 1 (k0 ≥ 8): k0 = 2, 4, 8, 16, 32 give G = 24, 12, 15, 10, 6.
Evaluations go through `evaluate_batch`, so the LLM judge reviews every candidate.

The CVT in the paper's BALROG runs was fit on a fixed corpus of 48 structural variants of the
naive starting harness (the starting harness plus copies with the `tools/` modules inlined into `agent.py` or kept
separate, and the text priors filled with placeholder text or left empty), never evaluated. By
default `run_benchmark.py` fits it on meta-agent variants instead; to reproduce the paper's
partition, build that corpus as a directory of harness directories and pass `--corpus DIR`.

Outputs land in `runs/balrog-minihack/search/<mode>_k<k0>/`: `result.json` (best harness, the
full history and the archive report), `candidates/` (every proposed harness with its
`NOTES.md`, `diagnosis.md` and `judge.json`), `evals/` (score files and traces) and
`checkpoints/evals.jsonl` (one line per evaluation). The two held-out splits in `config.json`,
`holdout_levels` (the same tasks with unseen level seeds) and `holdout_tasks` (eight tasks outside
the official set), are for scoring the final harness after the search; `holdout_levels` is run
automatically at the end when the best harness is an evolved one (never on the starting harness).

## The paper's best run

[`evolution/`](evolution/README.md) holds the 41 harnesses of the paper's best BALROG run
(adaptive, k0 = 16), from the naive starting harness to the harness that reached success 0.900, with the
proposer's notes, the diagnostician's reports, the judge's verdicts, and videos of four episodes
played by the four harnesses on the path to the best (engine replays and the text frames).
