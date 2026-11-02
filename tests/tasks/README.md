# Benchmark settings, evaluators and task examples

For each benchmark of the paper, this directory holds the benchmark settings the paper's runs used,
copied verbatim from the experiment workspace (the task splits and the search configuration), the
evaluator the runs used, and one task record per video of the evolution packs, cut from the real
split. The harnesses, the videos and the outcomes on these tasks are under
[`../examples/<benchmark>/evolution/`](../README.md). [`adapter_example.py`](adapter_example.py) is
the skeleton of a custom adapter, with the adapter protocol documented.

## Benchmark settings

| | SpatialWorld 3D games | BALROG-MiniHack | SimWorld-MMNav |
|---|---|---|---|
| task splits | [`spatialworld_games/task_spec/`](spatialworld_games/task_spec/): `<game>_evolve.json` (the search set of each game: maze 20, block3d 20, maze3d_pro 25, rubik 20 levels, snake 20 seeds) and `snake_eval.json` (snake's five held-out seeds; the other games search every level and have no held-out split) | [`balrog/task_spec/`](balrog/task_spec/): `evolve.json` (the official 8 tasks × 5 episodes, seed base 0), `eval_levels.json` (the same tasks on unseen levels, seed base 1, seeds listed), `eval_tasks.json` (8 unseen tasks), and [`SPLITS.md`](balrog/task_spec/SPLITS.md) on how the seeds are fixed | [`simworld_nav/task_spec/`](simworld_nav/task_spec/): `search_val_test_split.json` (the paper's hard run used `search_hard_50`, maps 20_0–20_24, and `val_hard_10`, maps 20_25–20_29; `search_easy_50` and `val_easy_10` for the easy setting) and `api_docs.md` (the harness-facing API of the environment) |
| search configuration | [`spatialworld_games/meta_config.json`](spatialworld_games/meta_config.json) | [`balrog/meta_config.json`](balrog/meta_config.json) | [`simworld_nav/meta_config.json`](simworld_nav/meta_config.json) |
| task definitions | the level files stay in SpatialWorld (`data/game/...`, see below) | the tasks are MiniHack's; the seeds pin the levels | [`simworld_nav/settings/hard/map_road_20_10/task_dist_21_0_1/`](simworld_nav/settings/hard/map_road_20_10/task_dist_21_0_1/): the task example's real directory from the SimWorld-MMNav test settings (`task_config.json`, `roads.json`, `progen_world.json`, `map.png`, the two robot views, `traffic.json`); point `SIMWORLD_SETTINGS_DIR` at `tests/tasks/simworld_nav/settings` to run it without the full dataset |

The `meta_config.json` files are the configurations the paper's adapters read (fitness mapping,
selection key, anti-cheat vocabulary, meta-agent backbone, benchmark defaults); the `config.json` of
each benchmark under `../examples/` carries the same settings in the form `run_benchmark.py` takes.
The CVT partition is fit at run time, on the starting harness and variants of it that
`run_benchmark.py` proposes (or a corpus passed with `--corpus DIR`).

## Evaluators

The three `evaluate.py` scripts share one interface and drive the benchmark's own runner with the
frozen base model; `run_benchmark.py` calls them during a search (through each benchmark's `run.sh`)
and they run a single task example just as well:

```bash
python tests/tasks/<benchmark>/evaluate.py --harness DIR --tasks TASKS.json --out OUT
python tests/tasks/balrog/evaluate.py            --harness DIR --split evolve          --out OUT
python tests/tasks/spatialworld_games/evaluate.py --harness DIR --split maze3d_pro_evolve --out OUT
python tests/tasks/simworld_nav/evaluate.py       --harness DIR --subset search_hard_50 --out OUT
```

`--harness` is a harness directory (`agent.py`, or the older `harness.py`); `--tasks` a JSON list of
task records (one of the task-example files here, or a split of the benchmark's `config.json`), or
`--split` / `--subset` the name of a split in the benchmark's `task_spec/`; `OUT/score.json` receives
the score the search reads (the fields `config.json` names under `fitness` and `episode`) plus
per-episode rows. Each evaluator runs under its benchmark's interpreter and is configured through
environment variables:

| | SpatialWorld 3D games | BALROG-MiniHack | SimWorld-MMNav |
|---|---|---|---|
| script | [`spatialworld_games/evaluate.py`](spatialworld_games/evaluate.py), one [`episode.py`](spatialworld_games/episode.py) subprocess per episode | [`balrog/evaluate.py`](balrog/evaluate.py) | [`simworld_nav/evaluate.py`](simworld_nav/evaluate.py) (+ `unrealcv_patch.py`) |
| what it reuses | SpatialWorld's per-game `*OpenAIEvaluator` classes; the harness's `build_agent(evaluator)` replaces the per-step decision | BALROG's `EvaluatorManager` with the harness's `build_agent(client_factory, prompt_builder)` as the agent factory; one runtime patch fixes the seed per (task, episode) | the SimWorld-Robotics gym over UnrealCV; the harness's `run_episode(env, task_spec, vlm_clients, trace_dir)` |
| benchmark checkout | `SPATIALWORLD_REPO` (8e10e24 with `spatialworld.patch`) | `BALROG_REPO` (b7afe79) | `SIMWORLD_ROBOTICS_REPO`; `SIMWORLD_SETTINGS_DIR` (the extracted test settings), `SIMWORLD_BASELINE_DIR` |
| base model | `GAME_MODEL`, `GAME_API_BASE` (comma-separated servers spread the episodes), `GAME_API_KEY` | `BALROG_MODEL`, `BALROG_API_BASE` | `NAV_MODEL`, `NAV_API_BASE`, or `EVAL_ALLOWED_MODELS` (a JSON pool file) |
| parallelism and limits | `GAME_EVAL_WORKERS`, `GAME_EPISODE_TIMEOUT_S` (1800), `GAME_MAX_STEPS`, `GAME_MAX_TOKENS`, `SPATIALWORLD_MAX_IMAGES` (32), `HISTORY_WINDOW` | `BALROG_WORKERS` (8), `--episodes`, `--step-cap` (100) | `UE_PORT` (9000), `--setting hard|easy` (`NAV_SETTING`), `EVAL_RESET_TIMEOUT_S`, `EVAL_STALL_LIMIT_S`, `VLM_CALL_TIMEOUT_S`, `MAX_VLM_CALLS_PER_STEP`, `SIM_LAUNCHER` + `EVAL_SIM_RESTART_EVERY` |
| anti-cheat | `ANTICHEAT_GUARD` (default 1): an episode is voided when the harness reads engine state other than the frame, reads the level files, or never calls the base model | fixed seeds (`seed_base`, `BALROG_SEED_BASE`); the LLM judge runs in the search | a read audit of the environment info the harness touched (`read_audit.json`) |
| writes | `score.json` (`score`, `n_episodes`, `per_episode`, `voided`), `episodes/<task>/` | `score.json` (`success`, `progression`, `survival_rate`, `step_efficiency`, `per_task_vector`, `per_episode`, `seeds`), BALROG's per-episode files | `score.json` (`SubtaskSR`, `SR`, `per_position_SR`, `distance_progress_mean`, `per_episode`), `traces/ep_<map>_<task>/`, `source/` |

All defaults point at `Qwen3-VL-30B-A3B-Instruct` on `http://127.0.0.1:8000/v1` and at checkouts
under `~/workspace/`. The full variable list is in each script's docstring.

## Task examples

One record per shipped video, cut from the real split, so that any harness of the evolution packs
runs on the task its video shows ([how to run one](../README.md#run-a-task-example)). What every
shown harness did on these tasks is in the video table of each pack's README
([games](../examples/spatialworld_games/evolution/README.md#videos),
[BALROG](../examples/balrog/evolution/README.md#videos),
[navigation](../examples/simworld_nav/evolution/README.md#videos)).

| benchmark | record | the task |
|---|---|---|
| SpatialWorld, maze | [`spatialworld_games/maze_level_06.json`](spatialworld_games/maze_level_06.json) | level 6: a 7×7 first-person maze (`data/game/maze3d/Level_06.txt`); reach the exit within 100 steps |
| SpatialWorld, block3d | [`spatialworld_games/block3d_level_12.json`](spatialworld_games/block3d_level_12.json) | level 12: a five-block target configuration (`LEVEL_DESIGNS["12"]` in `data/Block3D/level_data.py`); 50-step cap |
| SpatialWorld, maze3d_pro | [`spatialworld_games/maze3d_pro_level_05.json`](spatialworld_games/maze3d_pro_level_05.json) | level 5: two 7×7 floors with the exit on the other floor (`data/game/maze3d_pro/Level_05.txt`); 100-step cap |
| SpatialWorld, rubik | [`spatialworld_games/rubik_level_06.json`](spatialworld_games/rubik_level_06.json) | level 6: a cube scrambled by two quarter turns, U' then R (`INITIAL_CONFIGS["6"]` in `data/rubik_cube/scramble_data.py`); solve within 100 steps |
| SpatialWorld, snake | [`spatialworld_games/snake_run_020.json`](spatialworld_games/snake_run_020.json) | run 20: snake with seed 20 (the seed fixes the food placement); score = min(food eaten / 4, 1), 100-step cap |
| BALROG-MiniHack | [`balrog/MiniHack-MazeWalk-15x15-v0_run00.json`](balrog/MiniHack-MazeWalk-15x15-v0_run00.json) | MazeWalk-15x15, run 00: reach the staircase through a 15×15 maze; engine seed 360437341 |
| BALROG-MiniHack | [`balrog/MiniHack-Corridor-R3-v0_run04.json`](balrog/MiniHack-Corridor-R3-v0_run04.json) | Corridor-R3, run 04: reach the staircase through three rooms linked by unlit corridors; engine seed 3971796377 |
| BALROG-MiniHack | [`balrog/MiniHack-Boxoban-Medium-v0_run00.json`](balrog/MiniHack-Boxoban-Medium-v0_run00.json) | Boxoban-Medium, run 00: a Sokoban level, push every boulder onto a fountain; engine seed 291150209 |
| BALROG-MiniHack | [`balrog/MiniHack-Quest-Easy-v0_run00.json`](balrog/MiniHack-Quest-Easy-v0_run00.json) | Quest-Easy, run 00: cross a river of lava with the items in the inventory and reach the staircase; engine seed 255275049 |
| SimWorld-MMNav, hard | [`simworld_nav/map_20_10_task_dist_21_0_1.json`](simworld_nav/map_20_10_task_dist_21_0_1.json) | map 20_10, task dist_21_0_1: a Spot quadruped spawns at x = −102 m, y = −17 m facing yaw 90° and must stop within 20 m of the destination at x = 525 m, y = −17 m, 628 m away, within 1336 steps; two subtasks are scored, ORIENTATION ("Turn to face west…") and FINISH ("Proceed forward until reaching the high gray concrete building…"); five vehicles and five pedestrians |

**SpatialWorld 3D games.** `{"task_id": "<game>/level_NN", "game": "<game>", "mode": "level",
"level": NN}`, or `"mode": "run"` with a `"seed"` for snake, the record of the level exactly as it
appears in `task_spec/<game>_evolve.json`. The levels are defined in
[SpatialWorld](https://github.com/Hongcheng-Gao/SpatialWorld) at commit 8e10e24 and are not
redistributed; the paper's runs use the pure-visual variant (see
[`../examples/spatialworld_games/README.md`](../examples/spatialworld_games/README.md)).

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
[`simworld_nav/settings/hard/map_road_20_10/task_dist_21_0_1/`](simworld_nav/settings/hard/map_road_20_10/task_dist_21_0_1/)
(`task_config.json`, `roads.json`, `progen_world.json`, `map.png`, the two robot views and
`traffic.json`), so `SIMWORLD_SETTINGS_DIR=tests/tasks/simworld_nav/settings` runs this example
without the dataset; the other tasks are in the SimWorld-MMNav test settings (Hugging Face dataset
`jise/simworld-20k`, file `test_settings/single_test.tar.gz`, extracted to the same layout).
