# SpatialWorld 3D games: the evolution behind the paper's best harnesses

One directory per game holds every harness the chosen run evaluated, from the starting harness to the first
harness that solved every level of the game, in evaluation order, plus videos of one level played
by four of those harnesses. All runs are `mapelites_adaptive` on the pure-visual variant of the
games, with the Claude Code CLI as meta-agent and `Qwen3-VL-30B-A3B-Instruct` as the base model.

**Which run is shown.** Each game was run at five archive sizes (k0 = 2, 4, 8, 16 and 32).
Every run of maze, block3d, rubik and snake reached a score of 1.0 (every level solved), and
three of the five maze3d_pro runs did; the table gives the evaluation at which 1.0 was first
reached. The run shown per game (in bold) is one that reached 1.0 after evaluating at least
four harnesses, so that the videos can show the progression from the starting harness to the best, and
whose episodes can still be shown (the engine records every episode but recordings older than a
cut-off were deleted; rubik's episodes are replayed instead, see below). The quickest runs of
maze, snake and block3d solved every level with their first proposal, at the second evaluation,
and have nothing to show in between.

| game | k0 = 2 | k0 = 4 | k0 = 8 | k0 = 16 | k0 = 32 |
|---|---|---|---|---|---|
| maze | 3 | 3 | 2 | 5 | **4** |
| block3d | 2 | 2 | 5 | 2 | **4** |
| rubik | 3 | **4** | 5 | 9 | 12 |
| snake | 2 | 3 | 11 | 3 | **4** |
| maze3d_pro | **13** | — (0.96) | — (0.92) | 35 | 17 |

Maze3d_pro is the game with a real search: the starting harness solves 2 of 25 levels, and twelve
proposals follow before `cand_r6_1_action-planning` solves all 25.

**The starting harnesses** are SpatialWorld's shipped per-step agent in the harness layout: `agent.py`
composes the system prompt from `AGENT.md`, `skills/*.md` and `rules.md` (the shipped prompt is
kept verbatim as `system_prompt.txt`) and forwards every step to the engine's own model call
through `tools/vlm.py`; `tools/action.py`, `tools/frame.py` and `memory.md` are present but
unused by the starting harness, so that the proposer has a real multi-file surface to edit.

## What is in a harness directory

| file | written by | content |
|---|---|---|
| `agent.py`, `tools/*.py`, `skills/*.md`, `AGENT.md`, `rules.md`, `memory.md`, `system_prompt.txt`, `system_prompt.md` | the proposer | the harness as evaluated |
| `NOTES.md` | the proposer | its own account of the change: module map, axis, what changed and why |
| `diagnosis.md` | the diagnostician | the report on this harness's evaluation, read by the proposer when the harness is bred from |

`lineage.json` in each game directory lists the harnesses with their scores, parents, cell
depth and the size of the diff against the parent.

## Videos

Each game directory's `videos/` holds one level played by four harnesses of the lineage, the
starting harness, two in between and the best, and a `_progression_grid` video that tiles the four
(starting harness top left, best bottom right). The two harnesses in between are the best's ancestors
when it has at least two; otherwise the harnesses that set a new best score before it, and
among more than two candidates the two whose scores lie nearest to one third and two thirds of
the way from the starting harness's score to the best's (so maze3d_pro shows evaluations 1, 5, 8 and 13).
Maze, block3d, rubik and snake evaluated exactly four harnesses before reaching 1.0, so all four
are shown. The level is one on which the starting harness fails and the best harness succeeds, chosen
among the levels for which every shown harness's recording could be matched unambiguously to
its evaluation.

For maze, block3d, snake and maze3d_pro the videos are the engine's own recordings of the
evaluation, every frame kept and re-encoded at two steps per second under a label band (the
frame itself is untouched). No recording of any rubik run survives, so its episodes were
replayed in the engine with
[`../../render/replay_rubik_episode.py`](../../render/replay_rubik_episode.py): the engine is
started on the same level and the face and view turns the harness executed, as the evaluation's
session log lists them, are replayed through the same engine calls; the replay must execute
exactly the logged number of turns and end in the logged outcome. A replay also shows the cube
before the first turn, so its frame count is the step count plus one. The grids were made with
[`../../render/grid_videos.py`](../../render/grid_videos.py).

The engine draws the frames; the harness only decides. A frame shows the agent's current view
of the game, so what changes between the videos is the sequence of decisions. The maze frames
include the engine's bar above the view with the position, the heading and a compass, which the
pure-visual patch leaves in place (see [`../README.md`](../README.md)); the model saw that bar.

### maze (k0 = 32): 4 harnesses, 1.0 first reached at evaluation 4

| # | harness | parent | cell depth | score | levels solved | Δ vs parent | diff vs parent (files, +/−) |
|---|---|---|---|---|---|---|---|
| 0 | [`00_anchor_maze`](maze/harnesses/00_anchor_maze) *(on the path)* | — | 0 | 0.25 | 5/20 | — | — |
| 1 | [`01_cand_r1_0_spatial-grounding`](maze/harnesses/01_cand_r1_0_spatial-grounding) | anchor_maze | 1 | 0.40 | 8/20 | +0.150 | 4, +385/−13 |
| 2 | [`02_cand_r2_0_spatial-grounding`](maze/harnesses/02_cand_r2_0_spatial-grounding) | anchor_maze | 1 | 0.35 | 7/20 | +0.100 | 10, +287/−48 |
| 3 | [`03_cand_r4_0_spatial-grounding`](maze/harnesses/03_cand_r4_0_spatial-grounding) **(best)** | anchor_maze | 1 | 1.00 | 20/20 | +0.750 | 14, +566/−60 |

| video | harness | outcome |
|---|---|---|
| [`level_6__00_anchor_maze.mp4`](maze/videos/level_6__00_anchor_maze.mp4) | `anchor_maze` (eval 1) | score 0.00 in 100 steps |
| [`level_6__01_cand_r1_0_spatial-grounding.mp4`](maze/videos/level_6__01_cand_r1_0_spatial-grounding.mp4) | `cand_r1_0_spatial-grounding` (eval 2) | solved in 48 steps |
| [`level_6__02_cand_r2_0_spatial-grounding.mp4`](maze/videos/level_6__02_cand_r2_0_spatial-grounding.mp4) | `cand_r2_0_spatial-grounding` (eval 3) | solved in 20 steps |
| [`level_6__03_cand_r4_0_spatial-grounding.mp4`](maze/videos/level_6__03_cand_r4_0_spatial-grounding.mp4) | `cand_r4_0_spatial-grounding` (eval 4) | solved in 10 steps |
| [`level_6__progression_grid.mp4`](maze/videos/level_6__progression_grid.mp4) | the four above, 2×2 (starting harness top left, best bottom right) | — |

### block3d (k0 = 32): 4 harnesses, 1.0 first reached at evaluation 4

| # | harness | parent | cell depth | score | levels solved | Δ vs parent | diff vs parent (files, +/−) |
|---|---|---|---|---|---|---|---|
| 0 | [`00_anchor_block3d`](block3d/harnesses/00_anchor_block3d) *(on the path)* | — | 0 | 0.05 | 1/20 | — | — |
| 1 | [`01_cand_r1_0_spatial-grounding`](block3d/harnesses/01_cand_r1_0_spatial-grounding) | anchor_block3d | 1 | 0.00 | 0/20 | -0.050 | 14, +939/−77 |
| 2 | [`02_cand_r3_0_spatial-grounding`](block3d/harnesses/02_cand_r3_0_spatial-grounding) | anchor_block3d | 1 | 0.85 | 17/20 | +0.800 | 16, +1396/−110 |
| 3 | [`03_cand_r4_0_spatial-grounding`](block3d/harnesses/03_cand_r4_0_spatial-grounding) **(best)** | anchor_block3d | 1 | 1.00 | 20/20 | +0.950 | 18, +1435/−74 |

| video | harness | outcome |
|---|---|---|
| [`level_12__00_anchor_block3d.mp4`](block3d/videos/level_12__00_anchor_block3d.mp4) | `anchor_block3d` (eval 1) | score 0.00 in 50 steps |
| [`level_12__01_cand_r1_0_spatial-grounding.mp4`](block3d/videos/level_12__01_cand_r1_0_spatial-grounding.mp4) | `cand_r1_0_spatial-grounding` (eval 2) | score 0.00 in 50 steps |
| [`level_12__02_cand_r3_0_spatial-grounding.mp4`](block3d/videos/level_12__02_cand_r3_0_spatial-grounding.mp4) | `cand_r3_0_spatial-grounding` (eval 3) | solved in 15 steps |
| [`level_12__03_cand_r4_0_spatial-grounding.mp4`](block3d/videos/level_12__03_cand_r4_0_spatial-grounding.mp4) | `cand_r4_0_spatial-grounding` (eval 4) | solved in 15 steps |
| [`level_12__progression_grid.mp4`](block3d/videos/level_12__progression_grid.mp4) | the four above, 2×2 (starting harness top left, best bottom right) | — |

### rubik (k0 = 4): 4 harnesses, 1.0 first reached at evaluation 4

| # | harness | parent | cell depth | score | levels solved | Δ vs parent | diff vs parent (files, +/−) |
|---|---|---|---|---|---|---|---|
| 0 | [`00_anchor_rubik`](rubik/harnesses/00_anchor_rubik) *(on the path)* | — | 0 | 0.05 | 1/20 | — | — |
| 1 | [`01_cand_r1_0_spatial-grounding`](rubik/harnesses/01_cand_r1_0_spatial-grounding) *(on the path)* | anchor_rubik | 1 | 0.05 | 1/20 | +0.000 | 8, +534/−55 |
| 2 | [`02_cand_r2_0_spatial-grounding`](rubik/harnesses/02_cand_r2_0_spatial-grounding) | cand_r1_0_spatial-grounding | 1 | 0.45 | 9/20 | +0.400 | 8, +1122/−462 |
| 3 | [`03_cand_r2_1_action-planning`](rubik/harnesses/03_cand_r2_1_action-planning) **(best)** | cand_r1_0_spatial-grounding | 1 | 1.00 | 20/20 | +0.950 | 28, +3315/−519 |

| video | harness | outcome |
|---|---|---|
| [`level_6__00_anchor_rubik.mp4`](rubik/videos/level_6__00_anchor_rubik.mp4) | `anchor_rubik` (eval 1) | score 0.00 in 100 steps |
| [`level_6__01_cand_r1_0_spatial-grounding.mp4`](rubik/videos/level_6__01_cand_r1_0_spatial-grounding.mp4) | `cand_r1_0_spatial-grounding` (eval 2) | score 0.00 in 100 steps |
| [`level_6__02_cand_r2_0_spatial-grounding.mp4`](rubik/videos/level_6__02_cand_r2_0_spatial-grounding.mp4) | `cand_r2_0_spatial-grounding` (eval 3) | solved in 28 steps |
| [`level_6__03_cand_r2_1_action-planning.mp4`](rubik/videos/level_6__03_cand_r2_1_action-planning.mp4) | `cand_r2_1_action-planning` (eval 4) | solved in 4 steps |
| [`level_6__progression_grid.mp4`](rubik/videos/level_6__progression_grid.mp4) | the four above, 2×2 (starting harness top left, best bottom right) | — |

### snake (k0 = 32): 4 harnesses, 1.0 first reached at evaluation 4

| # | harness | parent | cell depth | score | levels solved | Δ vs parent | diff vs parent (files, +/−) |
|---|---|---|---|---|---|---|---|
| 0 | [`00_anchor_snake`](snake/harnesses/00_anchor_snake) *(on the path)* | — | 0 | 0.00 | 0/20 | — | — |
| 1 | [`01_cand_r1_0_spatial-grounding`](snake/harnesses/01_cand_r1_0_spatial-grounding) | anchor_snake | 1 | 0.44 | 4/20 | +0.438 | 13, +614/−40 |
| 2 | [`02_cand_r2_0_spatial-grounding`](snake/harnesses/02_cand_r2_0_spatial-grounding) | anchor_snake | 1 | 0.11 | 0/20 | +0.113 | 6, +457/−77 |
| 3 | [`03_cand_r3_0_spatial-grounding`](snake/harnesses/03_cand_r3_0_spatial-grounding) **(best)** | anchor_snake | 1 | 1.00 | 20/20 | +1.000 | 10, +1175/−13 |

| video | harness | outcome |
|---|---|---|
| [`run_20__00_anchor_snake.mp4`](snake/videos/run_20__00_anchor_snake.mp4) | `anchor_snake` (eval 1) | score 0.00 in 3 steps |
| [`run_20__01_cand_r1_0_spatial-grounding.mp4`](snake/videos/run_20__01_cand_r1_0_spatial-grounding.mp4) | `cand_r1_0_spatial-grounding` (eval 2) | score 0.25 in 100 steps |
| [`run_20__02_cand_r2_0_spatial-grounding.mp4`](snake/videos/run_20__02_cand_r2_0_spatial-grounding.mp4) | `cand_r2_0_spatial-grounding` (eval 3) | score 0.25 in 100 steps |
| [`run_20__03_cand_r3_0_spatial-grounding.mp4`](snake/videos/run_20__03_cand_r3_0_spatial-grounding.mp4) | `cand_r3_0_spatial-grounding` (eval 4) | solved in 16 steps |
| [`run_20__progression_grid.mp4`](snake/videos/run_20__progression_grid.mp4) | the four above, 2×2 (starting harness top left, best bottom right) | — |

### maze3d_pro (k0 = 2): 13 harnesses, 1.0 first reached at evaluation 13

| # | harness | parent | cell depth | score | levels solved | Δ vs parent | diff vs parent (files, +/−) |
|---|---|---|---|---|---|---|---|
| 0 | [`00_anchor_maze3d_pro`](maze3d_pro/harnesses/00_anchor_maze3d_pro) *(on the path)* | — | 0 | 0.08 | 2/25 | — | — |
| 1 | [`01_cand_r1_0_spatial-grounding`](maze3d_pro/harnesses/01_cand_r1_0_spatial-grounding) | anchor_maze3d_pro | 1 | 0.32 | 8/25 | +0.240 | 7, +597/−62 |
| 2 | [`02_cand_r1_1_action-planning`](maze3d_pro/harnesses/02_cand_r1_1_action-planning) | anchor_maze3d_pro | 1 | 0.28 | 7/25 | +0.200 | 5, +496/−34 |
| 3 | [`03_cand_r2_0_spatial-grounding`](maze3d_pro/harnesses/03_cand_r2_0_spatial-grounding) | cand_r1_1_action-planning | 2 | 0.28 | 7/25 | +0.000 | 7, +423/−114 |
| 4 | [`04_cand_r2_1_action-planning`](maze3d_pro/harnesses/04_cand_r2_1_action-planning) | cand_r1_1_action-planning | 2 | 0.36 | 9/25 | +0.080 | 7, +613/−161 |
| 5 | [`05_cand_r3_0_spatial-grounding`](maze3d_pro/harnesses/05_cand_r3_0_spatial-grounding) | cand_r2_1_action-planning | 3 | 0.36 | 9/25 | +0.000 | 6, +620/−324 |
| 6 | [`06_cand_r3_1_action-planning`](maze3d_pro/harnesses/06_cand_r3_1_action-planning) | cand_r2_1_action-planning | 3 | 0.28 | 7/25 | -0.080 | 7, +785/−520 |
| 7 | [`07_cand_r4_0_spatial-grounding`](maze3d_pro/harnesses/07_cand_r4_0_spatial-grounding) | anchor_maze3d_pro | 4 | 0.76 | 19/25 | +0.680 | 9, +1092/−75 |
| 8 | [`08_cand_r4_1_action-planning`](maze3d_pro/harnesses/08_cand_r4_1_action-planning) | anchor_maze3d_pro | 4 | 0.88 | 22/25 | +0.800 | 8, +1178/−83 |
| 9 | [`09_cand_r5_0_spatial-grounding`](maze3d_pro/harnesses/09_cand_r5_0_spatial-grounding) | anchor_maze3d_pro | 5 | 0.96 | 24/25 | +0.880 | 11, +1184/−86 |
| 10 | [`10_cand_r5_1_action-planning`](maze3d_pro/harnesses/10_cand_r5_1_action-planning) | anchor_maze3d_pro | 5 | 0.68 | 17/25 | +0.600 | 9, +1287/−92 |
| 11 | [`11_cand_r6_0_spatial-grounding`](maze3d_pro/harnesses/11_cand_r6_0_spatial-grounding) | anchor_maze3d_pro | 6 | 0.96 | 24/25 | +0.880 | 11, +1539/−86 |
| 12 | [`12_cand_r6_1_action-planning`](maze3d_pro/harnesses/12_cand_r6_1_action-planning) **(best)** | anchor_maze3d_pro | 6 | 1.00 | 25/25 | +0.920 | 11, +1316/−81 |

| video | harness | outcome |
|---|---|---|
| [`level_5__00_anchor_maze3d_pro.mp4`](maze3d_pro/videos/level_5__00_anchor_maze3d_pro.mp4) | `anchor_maze3d_pro` (eval 1) | score 0.00 in 100 steps |
| [`level_5__04_cand_r2_1_action-planning.mp4`](maze3d_pro/videos/level_5__04_cand_r2_1_action-planning.mp4) | `cand_r2_1_action-planning` (eval 5) | score 0.00 in 100 steps |
| [`level_5__07_cand_r4_0_spatial-grounding.mp4`](maze3d_pro/videos/level_5__07_cand_r4_0_spatial-grounding.mp4) | `cand_r4_0_spatial-grounding` (eval 8) | solved in 56 steps |
| [`level_5__12_cand_r6_1_action-planning.mp4`](maze3d_pro/videos/level_5__12_cand_r6_1_action-planning.mp4) | `cand_r6_1_action-planning` (eval 13) | solved in 18 steps |
| [`level_5__progression_grid.mp4`](maze3d_pro/videos/level_5__progression_grid.mp4) | the four above, 2×2 (starting harness top left, best bottom right) | — |

