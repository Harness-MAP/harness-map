# The paper's benchmarks, hands on

Each benchmark directory holds what runs it (`tasks/`: the setup README, `config.json`, `run.sh`,
the evaluator and the task splits) and the results of the paper's best run (`examples/`: every
harness the run evolved, with the proposer's notes and the diagnostician's reports, and videos of
the task examples played by the starting harness and by harnesses of the lineage).
[`run_benchmark.py`](run_benchmark.py) is the search driver the run scripts call and
[`adapter_example.py`](adapter_example.py) the skeleton of a custom adapter, with the adapter
protocol documented.

| benchmark | `tasks/`: setup, evaluator and task examples | `examples/`: the evolved harnesses and their videos |
|---|---|---|
| SpatialWorld 3D games | [`spatialworld_games/tasks/`](spatialworld_games/tasks/README.md); one level per game: maze level 6, block3d level 12, maze3d_pro level 5, rubik level 6, snake run 20 | [`spatialworld_games/examples/`](spatialworld_games/examples/README.md): 4 to 13 harnesses per game under `<game>/harnesses/`, videos under `<game>/videos/` |
| BALROG-MiniHack | [`balrog/tasks/`](balrog/tasks/README.md); four episodes: MazeWalk-15x15 run 00, Corridor-R3 run 04, Boxoban-Medium run 00, Quest-Easy run 00 | [`balrog/examples/`](balrog/examples/README.md): 41 harnesses under `harnesses/`, videos under `videos/` |
| SimWorld-MMNav, hard | [`simworld_nav/tasks/`](simworld_nav/tasks/README.md); map 20_10, task dist_21_0_1 | [`simworld_nav/examples/`](simworld_nav/examples/README.md): 16 harnesses under `harnesses/`, videos under `videos/` |

## Layout

```
tests/
├── run_benchmark.py              the search driver
├── adapter_example.py            a custom adapter skeleton
└── <benchmark>/                  balrog, spatialworld_games, simworld_nav
    ├── tasks/                    what runs the benchmark
    │   ├── README.md             installing the benchmark and the base model
    │   ├── config.json           the search configuration
    │   ├── run.sh                the paper's run
    │   ├── evaluate.py           the evaluator
    │   ├── task_spec/            the task splits and the task examples
    │   ├── episode.py            games only: one episode
    │   ├── spatialworld.patch    games only: the pure-visual variant
    │   └── settings/             navigation only: the task example's directory
    └── examples/                 the paper's best run
        ├── README.md             the lineage and the videos
        ├── lineage.json          every harness, its score and its parent
        ├── harnesses/            every harness as evaluated
        └── videos/               the task examples played by the lineage
```

`lineage.json` lists the harnesses of a run with their scores, parents, cell depth and the size of
the diff against the parent; `NOTES.md` is the proposer's own account of a change and
`diagnosis.md` the diagnostician's report on the harness's evaluation.

## Benchmark settings

| | SpatialWorld 3D games | BALROG-MiniHack | SimWorld-MMNav |
|---|---|---|---|
| task splits | [`spatialworld_games/task_spec/`](spatialworld_games/tasks/task_spec/): `<game>_evolve.json` (the search set of each game: maze 20, block3d 20, maze3d_pro 25, rubik 20 levels, snake 20 seeds) and `snake_eval.json` (snake's five held-out seeds; the other games search every level and have no held-out split) | [`balrog/task_spec/`](balrog/tasks/task_spec/): `evolve.json` (the official 8 tasks × 5 episodes, seed base 0), `eval_levels.json` (the same tasks on unseen levels, seed base 1, seeds listed), `eval_tasks.json` (8 unseen tasks), and [`SPLITS.md`](balrog/tasks/task_spec/SPLITS.md) on how the seeds are fixed | [`simworld_nav/task_spec/`](simworld_nav/tasks/task_spec/): `search_val_test_split.json` (the paper's hard run used `search_hard_50`, maps 20_0–20_24, and `val_hard_10`, maps 20_25–20_29; `search_easy_50` and `val_easy_10` for the easy setting) and `api_docs.md` (the harness-facing API of the environment) |
| task definitions | the level files stay in SpatialWorld (`data/game/...`) | the tasks are MiniHack's; the seeds pin the levels | [`simworld_nav/settings/hard/map_road_20_10/task_dist_21_0_1/`](simworld_nav/tasks/settings/hard/map_road_20_10/task_dist_21_0_1/): the task example's real directory from the SimWorld-MMNav test settings (`task_config.json`, `roads.json`, `progen_world.json`, `map.png`, the two robot views, `traffic.json`). The gym reads task data under its own `envs/setting/single_agent_world/`; copy this directory (or extract the full test settings) there |

## Evaluators

The three `evaluate.py` scripts share one interface and drive the benchmark's own runner with the
frozen base model; `run_benchmark.py` calls them during a search (through each benchmark's `run.sh`)
and they run a single task example just as well:

```bash
python tests/<benchmark>/tasks/evaluate.py        --harness DIR --tasks TASKS.json         --out OUT
python tests/balrog/tasks/evaluate.py             --harness DIR --split evolve             --out OUT
python tests/spatialworld_games/tasks/evaluate.py --harness DIR --split maze3d_pro_evolve  --out OUT
python tests/simworld_nav/tasks/evaluate.py       --harness DIR --subset search_hard_50    --out OUT
```

`--harness` is a harness directory (`agent.py`, or the older `harness.py`); `--tasks` a JSON list of
task records (one of the task-example files here, or a split file), or `--split` / `--subset` the
name of a split in `task_spec/`; `OUT/score.json` receives the score the search reads (the fields
`config.json` names under `fitness` and `episode`) plus per-episode rows. Each evaluator runs under
its benchmark's interpreter and is configured through environment variables:

| | SpatialWorld 3D games | BALROG-MiniHack | SimWorld-MMNav |
|---|---|---|---|
| script | [`spatialworld_games/evaluate.py`](spatialworld_games/tasks/evaluate.py), one [`episode.py`](spatialworld_games/tasks/episode.py) subprocess per episode | [`balrog/evaluate.py`](balrog/tasks/evaluate.py) | [`simworld_nav/evaluate.py`](simworld_nav/tasks/evaluate.py) |
| what it reuses | SpatialWorld's per-game `*OpenAIEvaluator` classes; the harness's `build_agent(evaluator)` replaces the per-step decision | BALROG's `EvaluatorManager` with the harness's `build_agent(client_factory, prompt_builder)` as the agent factory; one runtime patch fixes the seed per (task, episode) | the SimWorld-Robotics gym over UnrealCV; the harness's `run_episode(env, task_spec, vlm_clients, trace_dir)` plays the episode and the environment's own checker scores the subtasks |
| benchmark checkout | `SPATIALWORLD_REPO` (8e10e24 with `spatialworld.patch`) | `BALROG_REPO` (b7afe79) | `SIMWORLD_ROBOTICS_REPO` (the harnesses import helpers from its `baseline/single`; `SIMWORLD_BASELINE_DIR` to point elsewhere); the task data under the gym package's `envs/setting/single_agent_world/` |
| base model | `GAME_MODEL`, `GAME_API_BASE` (comma-separated servers spread the episodes), `GAME_API_KEY` | `BALROG_MODEL`, `BALROG_API_BASE` | `NAV_MODEL`, `NAV_API_BASE` |
| parallelism and limits | `GAME_EVAL_WORKERS`, `GAME_EPISODE_TIMEOUT_S` (1800), `GAME_MAX_STEPS`, `GAME_MAX_TOKENS`, `SPATIALWORLD_MAX_IMAGES` (32), `HISTORY_WINDOW` | `BALROG_WORKERS` (8), `--episodes`, `--step-cap` (100) | one episode at a time on one simulator: `UE_PORT` (9000), `--setting hard\|easy` (`NAV_SETTING`), `VLM_CALL_TIMEOUT_S` (180), `VLM_CALL_RETRIES` (3), `MAX_VLM_CALLS_PER_STEP` (30), `UNREALCV_RPC_TIMEOUT_S` (15, the longest wait for a simulator reply) |
| anti-cheat | `ANTICHEAT_GUARD` (default 1): an episode is voided when the harness reads engine state other than the frame, reads the level files, or never calls the base model | fixed seeds (`seed_base`, `BALROG_SEED_BASE`); the LLM judge runs in the search | a read audit of the environment fields the harness touched (`read_audit.json`: PASS, WARN for undeclared fields, FAIL for the answer key); the static gate scans for `config.json`'s `privileged_keys` |
| writes | `score.json` (`score`, `n_episodes`, `per_episode`, `voided`), `episodes/<task>/` | `score.json` (`success`, `progression`, `survival_rate`, `step_efficiency`, `per_task_vector`, `per_episode`, `seeds`), BALROG's per-episode files | `score.json` (`SubtaskSR`, `SR`, `per_position_SR`, `distance_progress_mean`, `cost_per_episode`, `per_episode`), `traces/ep_<map>_<task>/` (`actions.jsonl`, `env_step_log.jsonl`, `vlm_trajectory.jsonl` with the frames under `frames/`, `llm_calls.jsonl`, `observations.jsonl`, `verdict.json`, `gt.json`), `read_audit.json` |

All defaults point at `Qwen3-VL-30B-A3B-Instruct` on `http://127.0.0.1:8000/v1` and at checkouts
under `~/workspace/`. The full variable list is in each script's docstring.

## Task examples

One record per shipped video in each `task_spec/`, cut from the split it sits beside, so that any
harness of the paper's runs plays the task its video shows ([how to run one](#run-a-task-example)). What every
shown harness did on these tasks is in the video table of each pack's README
([games](spatialworld_games/examples/README.md#videos),
[BALROG](balrog/examples/README.md#videos),
[navigation](simworld_nav/examples/README.md#videos)).

| benchmark | record | the task |
|---|---|---|
| SpatialWorld, maze | [`spatialworld_games/task_spec/maze_level_06.json`](spatialworld_games/tasks/task_spec/maze_level_06.json) | level 6: a 7×7 first-person maze (`data/game/maze3d/Level_06.txt`); reach the exit within 100 steps |
| SpatialWorld, block3d | [`spatialworld_games/task_spec/block3d_level_12.json`](spatialworld_games/tasks/task_spec/block3d_level_12.json) | level 12: a five-block target configuration (`LEVEL_DESIGNS["12"]` in `data/Block3D/level_data.py`); 50-step cap |
| SpatialWorld, maze3d_pro | [`spatialworld_games/task_spec/maze3d_pro_level_05.json`](spatialworld_games/tasks/task_spec/maze3d_pro_level_05.json) | level 5: two 7×7 floors with the exit on the other floor (`data/game/maze3d_pro/Level_05.txt`); 100-step cap |
| SpatialWorld, rubik | [`spatialworld_games/task_spec/rubik_level_06.json`](spatialworld_games/tasks/task_spec/rubik_level_06.json) | level 6: a cube scrambled by two quarter turns, U' then R (`INITIAL_CONFIGS["6"]` in `data/rubik_cube/scramble_data.py`); solve within 100 steps |
| SpatialWorld, snake | [`spatialworld_games/task_spec/snake_run_020.json`](spatialworld_games/tasks/task_spec/snake_run_020.json) | run 20: snake with seed 20 (the seed fixes the food placement); score = min(food eaten / 4, 1), 100-step cap |
| BALROG-MiniHack | [`balrog/task_spec/MiniHack-MazeWalk-15x15-v0_run00.json`](balrog/tasks/task_spec/MiniHack-MazeWalk-15x15-v0_run00.json) | MazeWalk-15x15, run 00: reach the staircase through a 15×15 maze; engine seed 360437341 |
| BALROG-MiniHack | [`balrog/task_spec/MiniHack-Corridor-R3-v0_run04.json`](balrog/tasks/task_spec/MiniHack-Corridor-R3-v0_run04.json) | Corridor-R3, run 04: reach the staircase through three rooms linked by unlit corridors; engine seed 3971796377 |
| BALROG-MiniHack | [`balrog/task_spec/MiniHack-Boxoban-Medium-v0_run00.json`](balrog/tasks/task_spec/MiniHack-Boxoban-Medium-v0_run00.json) | Boxoban-Medium, run 00: a Sokoban level, push every boulder onto a fountain; engine seed 291150209 |
| BALROG-MiniHack | [`balrog/task_spec/MiniHack-Quest-Easy-v0_run00.json`](balrog/tasks/task_spec/MiniHack-Quest-Easy-v0_run00.json) | Quest-Easy, run 00: cross a river of lava with the items in the inventory and reach the staircase; engine seed 255275049 |
| SimWorld-MMNav, hard | [`simworld_nav/task_spec/map_20_10_task_dist_21_0_1.json`](simworld_nav/tasks/task_spec/map_20_10_task_dist_21_0_1.json) | map 20_10, task dist_21_0_1: a Spot quadruped spawns at x = −102 m, y = −17 m facing yaw 90° and must stop within 20 m of the destination at x = 525 m, y = −17 m, 628 m away, within 1336 steps; two subtasks are scored, ORIENTATION ("Turn to face west…") and FINISH ("Proceed forward until reaching the high gray concrete building…"); five vehicles and five pedestrians |

**SpatialWorld 3D games.** `{"task_id": "<game>/level_NN", "game": "<game>", "mode": "level",
"level": NN}`, or `"mode": "run"` with a `"seed"` for snake, the record of the level exactly as it
appears in `task_spec/<game>_evolve.json`. The levels are defined in
[SpatialWorld](https://github.com/Hongcheng-Gao/SpatialWorld) at commit 8e10e24 and are not
redistributed; the paper's runs use the pure-visual variant (see
[`spatialworld_games/README.md`](spatialworld_games/tasks/README.md)).

**BALROG-MiniHack.** The task's record from `task_spec/evolve.json` (`task_id`, `task`, `episodes`,
`family`, `official_balrog`) with `episodes` cut to the shown run. The evaluator seeds episode *i*
of a task deterministically from the seed base (0 for the search split, `BALROG_SEED_BASE` to change
it), so `episodes: 1` plays run 00 and Corridor-R3's run 04 needs `episodes: 5` (its fifth episode).
The engine seeds above are the ones BALROG logged for these episodes in the paper's run; they follow
from the same formula, which also reproduces the 40 seeds listed in `task_spec/eval_levels.json`,
and with them [MiniHack](https://github.com/samvelyan/minihack) reproduces the exact level directly.
The task descriptions are BALROG's (`balrog/config/config.yaml` at b7afe79).

**SimWorld-MMNav.** `{"map": "20_10", "task": "task_dist_21_0_1"}`, the item of `search_hard_50` in
`task_spec/search_val_test_split.json`. The task's real directory is shipped under
[`simworld_nav/settings/hard/map_road_20_10/task_dist_21_0_1/`](simworld_nav/tasks/settings/hard/map_road_20_10/task_dist_21_0_1/)
(`task_config.json`, `roads.json`, `progen_world.json`, `map.png`, the two robot views and
`traffic.json`); copied to `<simworld_gym>/envs/setting/single_agent_world/hard/map_road_20_10/`,
where the gym reads task data, it runs this example without the dataset. The other tasks are in the
SimWorld-MMNav test settings (Hugging Face dataset `jise/simworld-20k`, file
`test_settings/single_test.tar.gz`), extracted to the same place.

## Run a task example

Install the benchmark and serve the base model as its `tasks/README.md` describes, set the
evaluator's environment variables (the table above), then point the evaluator at a harness
directory and a task-example record:

```bash
export SPATIALWORLD_REPO=/path/to/SpatialWorld GAME_API_BASE=http://127.0.0.1:8000/v1
"$GAME_PYTHON" tests/spatialworld_games/tasks/evaluate.py \
    --harness tests/spatialworld_games/examples/maze3d_pro/harnesses/12_cand_r6_1_action-planning \
    --tasks   tests/spatialworld_games/tasks/task_spec/maze3d_pro_level_05.json \
    --out     /tmp/maze3d_pro_level_05
```

`OUT/score.json` holds the score and the per-episode rows; the video table of the benchmark's
`examples/README.md` gives the outcome every shown harness obtained on the same task. The base
model samples at every step, so a replay is a fresh draw of the same harness on the same task
rather than a frame-for-frame copy of the video; the BALROG episodes are fixed-seed, so the level
itself is identical. The same evaluators run the searches: each benchmark's `run.sh` passes them
to `run_benchmark.py`.
