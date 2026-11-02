# SimWorld-Robotics navigation (SimWorld-MMNav)

Harness-MAP evolves a multi-module navigation harness for the single-agent SimWorld-MMNav task of
[SimWorld-Robotics](https://github.com/SCAI-JHU/SimWorld-Robotics): a quadruped robot follows
multimodal instructions through a procedurally generated city rendered by Unreal Engine 5. The
search set is 50 tasks (maps 20_0 to 20_24, two tasks each) and the held-out set 10 tasks (maps
20_25 to 20_29), for the hard and the easy setting. The primary metric is `SubtaskSR`, the mean
fraction of completed subtasks (`SR`, full-task success, is reported as a dense field). The
evaluation budget is N = 16.

## Setup

1. **SimWorld-Robotics** (gym package, baselines and test data):
   ```bash
   git clone https://github.com/SCAI-JHU/SimWorld-Robotics.git && cd SimWorld-Robotics
   conda create -n simworld python=3.10 -y && conda activate simworld
   pip install -e simworld_gym && pip install -r baseline/requirements.txt
   # test data: dataset jise/simworld-20k on Hugging Face, file test_settings/single_test.tar.gz,
   # extracted into simworld_gym/envs/setting/single_agent_world/
   ```
   The gym reads task data under its own `envs/setting/single_agent_world/`;
   `tests/simworld_nav/tasks/settings/` holds one hard task (map 20_10, dist_21_0_1) in that layout,
   enough to replay the task example. The single-agent baseline (`baseline/single`) targets the OpenAI Responses API; the paper's
   runs added a `vllm` backend that converts the agent's `input_text` / `input_image` parts into
   chat-completion requests, a `base_url` argument, and an output-token budget of 8192. The nav
   starting harness imports `prompt_template` and `utils` from that directory, found through
   `SIMWORLD_BASELINE_DIR`.
2. **Simulator**: the SimWorld UE binary (Hugging Face dataset `SimWorld-AI/SimWorld`, file
   `Base20260201/Linux.zip`), the `simworld` Python package
   (`pip install -e` of https://github.com/SimWorld-AI/SimWorld), an `unrealcv.ini` with
   `Port=9000` beside the binary, and a launch on the empty map:
   ```bash
   ./SimWorld.sh /Game/Maps/empty -RenderOffScreen -unattended   # UnrealCV listens on :9000 after ~30 s
   ```
   The simulator serves one client at a time; never probe its port while an evaluation runs.
3. **Base model**: a vLLM server with `Qwen3-VL-30B-A3B-Instruct` (no reasoning parser), e.g.
   `vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct --served-model-name Qwen3-VL-30B-A3B-Instruct --port 8000 --max-model-len 65536 --limit-mm-per-prompt '{"image":32}'`.
   The starting harness's spatial tools use the simulator's depth observation; the paper's runs optionally
   replaced it with a metric-depth estimate from a Depth Anything 3 service.
4. **Meta-agent**: the Claude Code CLI (`claude`) on `PATH`, or another backbone (top-level README).
5. **Starting harness**: the multi-module navigation harness (`agent.py` plus `modules/`:
   perceiver, spatial memory, planner, prompt assembler, active perception, completion checker)
   in the layout of `harness_map/meta_agent/skills/proposer/SKILL.md`; the paper's starting harness runs
   the ReAct baseline agent through these modules unchanged.
6. **Evaluator**: `evaluate.py` (`NAV_EVALUATOR` in `run.sh`), run under the `simworld` environment as
   `evaluate.py --harness DIR --tasks TASKS.json --out OUT`. For each task record (`map`,
   `task`) it resets the gym environment on that map and task over UnrealCV
   (`UE_PORT`), runs the harness's episode entry point with the frozen model, and records
   per-subtask success from the environment's own checker. `OUT/score.json` holds
   `SubtaskSR`, `SR`, `distance_progress_mean`, `cost_per_episode`, `n_episodes`,
   `per_position_SR` (success rate per subtask position, in the order of `rubric_labels`) and
   `per_episode` rows with `episode`, `success`, `subtask_sr`,
   `distance_progress`, `final_dist_to_target` and `total_tokens`. Every simulator call is
   bounded (`UNREALCV_RPC_TIMEOUT_S`) and a failed reset is retried once the simulator's port
   answers again; the runner marks an evaluation with fewer episodes than the split as invalid.

## Run

Fill in the placeholders in `run.sh`, then:

```bash
SETTING=hard MODE=adaptive K0=4 bash tests/simworld_nav/tasks/run.sh      # the paper's main setting
SETTING=hard MODE=adaptive K0=2 bash tests/simworld_nav/tasks/run.sh
SETTING=easy MODE=adaptive K0=4 bash tests/simworld_nav/tasks/run.sh
```

`config.json` fixes the budget at N = 16 with C = 1; with P = ⌈log2 k0⌉, k0 = 2, 4, 8 give
G = 15, 7, 4 generations. Coverage-gated initialization is lenient here (`strict_coverage`
false): navigation edits form few structural clusters, so a run continues when fewer than
⌈m·k0⌉ cells can be covered. The nav adapter evaluates through `evaluate` only, so the LLM judge
is not run. An evaluation takes several hours (50 episodes of up to ~12 minutes each), so a
search runs for days; the per-evaluation ledger `checkpoints/evals.jsonl` is written as it goes.

Outputs land in `runs/simworld-mmnav/search_<setting>/<mode>_k<k0>/` (`result.json`,
`candidates/`, `evals/`, `checkpoints/evals.jsonl`); the best harness is scored on
`val_<setting>` at the end.

## The paper's best run

[`../examples/`](../examples/README.md) holds the 16 harnesses of the paper's best hard-setting run
(adaptive, k0 = 4), from the starting harness to the harness that reached SubtaskSR 0.555 on the search
tasks and 0.567 held out, with each harness's score file, the proposer's notes and the
diagnostician's reports, and videos of one hard task re-run with four harnesses of the lineage.
