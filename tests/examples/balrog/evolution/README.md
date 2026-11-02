# BALROG-MiniHack: the evolution behind the paper's best harness

This directory holds every harness the paper's best BALROG run evaluated, from the starting harness to the
first harness that reached the run's best score, in the order they were evaluated, together with
videos of the episodes they played.

**The run.** `mapelites_adaptive` with k0 = 16, C = 1, P = 4 and G = 10 (M = 8 founders from
coverage-gated initialization, 48 evaluations in total), on the eight official MiniHack tasks with
five fixed-seed episodes each. The meta-agent was the Claude Code CLI (the proposer transcripts
record `claude-opus-5-5`), the base model `Qwen3-VL-30B-A3B-Instruct`. The best harness,
`cand_r68_0_spatial-grounding`, solves 36 of the 40 search episodes (success 0.900) and was the
41st evaluation; the seven harnesses evaluated after it did not improve on it and are not
included. Its ancestry is the starting harness → `cand_r35_0` (0.150) → `cand_r41_0` (0.375) →
`cand_r68_0` (0.900): the first step moved the decision from a per-turn model call to an
in-episode value learner that asks the model only to break ties, the second put a per-level
chart with a BFS router in front of the learner, and the third rewired that learner onto an
ordered stage table with `tools/` owners for the Quest and Sokoban tasks and a frontier attractor
in the router (edits the proposer was steered toward from two sibling cells, `cand_r64_0` and
`cand_r58_0`).

**The starting harness** (`harnesses/00_anchor_naive`) is BALROG's published naive agent, reproduced in the
harness layout: `agent.py` appends the same instruction and extracts the action with the same
filter as `balrog/agents/naive.py`; the structural split into `tools/` modules adds no behaviour
(see the docstring in its `agent.py`). The text priors `AGENT.md`, `rules.md`, `memory.md` and
`system_prompt.txt` start empty.

## What is in a harness directory

| file | written by | content |
|---|---|---|
| `agent.py`, `tools/*.py`, `skills/*.md`, `AGENT.md`, `rules.md`, `memory.md`, `system_prompt.txt` | the proposer | the harness as evaluated (the starting harness's are the starting point) |
| `NOTES.md` | the proposer | its own account of the change: module map, the axis it worked on, what changed and why |
| `DESIGN.txt` | the proposer | a design note, present when the proposer chose to write one |
| `diagnosis.md` | the diagnostician | the report on this harness's evaluation, which the proposer reads when the harness is bred from |
| `judge.json` | the LLM judge | the verdict on the harness (BALROG is the benchmark where the judge ran) |

`lineage.json` lists the same harnesses with their scores, parents, cell depth and the size of
the diff against the parent. The proposer's transcripts (`propose.log`) and the read audits are
not included.

## Harnesses in evaluation order

Cell depth is the resolution depth of the archive cell the harness landed in. Δ is the change in
success against the parent; the diff column counts changed files and inserted/deleted lines of
the harness files (notes and reports excluded). Per-task success is over the five episodes of
each task.

| # | harness | parent | cell depth | success | per-task success | Δ vs parent | diff vs parent (files, +/−) |
|---|---|---|---|---|---|---|---|
| 0 | [`00_anchor_naive`](harnesses/00_anchor_naive) *(on the path)* | — | 0 | 0.100 | Boxoban-Hard 0.0, Boxoban-Medium 0.0, MazeWalk-9x9 0.6, MazeWalk-15x15 0.2, Corridor-R3 0.0, CorridorBattle-Dark 0.0, Quest-Easy 0.0, Quest-Medium 0.0 | — | — |
| 1 | [`01_cand_r1_0_spatial-grounding`](harnesses/01_cand_r1_0_spatial-grounding) | anchor_naive | 1 | 0.350 | Boxoban-Hard 0.0, Boxoban-Medium 0.0, MazeWalk-9x9 1.0, MazeWalk-15x15 1.0, Corridor-R3 0.8, CorridorBattle-Dark 0.0, Quest-Easy 0.0, Quest-Medium 0.0 | +0.250 | 7, +490/−39 |
| 2 | [`02_cand_r18_0_spatial-grounding`](harnesses/02_cand_r18_0_spatial-grounding) | anchor_naive | 1 | 0.175 | Boxoban-Hard 0.0, Boxoban-Medium 0.0, MazeWalk-9x9 0.8, MazeWalk-15x15 0.4, Corridor-R3 0.2, CorridorBattle-Dark 0.0, Quest-Easy 0.0, Quest-Medium 0.0 | +0.075 | 5, +301/−29 |
| 3 | [`03_cand_r26_0_spatial-grounding`](harnesses/03_cand_r26_0_spatial-grounding) | anchor_naive | 1 | 0.325 | Boxoban-Hard 0.0, Boxoban-Medium 0.0, MazeWalk-9x9 1.0, MazeWalk-15x15 1.0, Corridor-R3 0.4, CorridorBattle-Dark 0.2, Quest-Easy 0.0, Quest-Medium 0.0 | +0.225 | 3, +264/−98 |
| 4 | [`04_cand_r27_0_spatial-grounding`](harnesses/04_cand_r27_0_spatial-grounding) | anchor_naive | 1 | 0.325 | Boxoban-Hard 0.0, Boxoban-Medium 0.0, MazeWalk-9x9 1.0, MazeWalk-15x15 1.0, Corridor-R3 0.4, CorridorBattle-Dark 0.2, Quest-Easy 0.0, Quest-Medium 0.0 | +0.225 | 6, +200/−95 |
| 5 | [`05_cand_r29_0_spatial-grounding`](harnesses/05_cand_r29_0_spatial-grounding) | anchor_naive | 1 | 0.150 | Boxoban-Hard 0.0, Boxoban-Medium 0.0, MazeWalk-9x9 0.8, MazeWalk-15x15 0.4, Corridor-R3 0.0, CorridorBattle-Dark 0.0, Quest-Easy 0.0, Quest-Medium 0.0 | +0.050 | 5, +134/−48 |
| 6 | [`06_cand_r33_0_spatial-grounding`](harnesses/06_cand_r33_0_spatial-grounding) | anchor_naive | 1 | 0.175 | Boxoban-Hard 0.0, Boxoban-Medium 0.0, MazeWalk-9x9 1.0, MazeWalk-15x15 0.4, Corridor-R3 0.0, CorridorBattle-Dark 0.0, Quest-Easy 0.0, Quest-Medium 0.0 | +0.075 | 8, +221/−93 |
| 7 | [`07_cand_r35_0_spatial-grounding`](harnesses/07_cand_r35_0_spatial-grounding) *(on the path)* | anchor_naive | 1 | 0.150 | Boxoban-Hard 0.0, Boxoban-Medium 0.0, MazeWalk-9x9 1.0, MazeWalk-15x15 0.0, Corridor-R3 0.2, CorridorBattle-Dark 0.0, Quest-Easy 0.0, Quest-Medium 0.0 | +0.050 | 4, +266/−98 |
| 8 | [`08_cand_r36_0_spatial-grounding`](harnesses/08_cand_r36_0_spatial-grounding) | cand_r26_0_spatial-grounding | 1 | 0.375 | Boxoban-Hard 0.0, Boxoban-Medium 0.0, MazeWalk-9x9 1.0, MazeWalk-15x15 1.0, Corridor-R3 0.8, CorridorBattle-Dark 0.2, Quest-Easy 0.0, Quest-Medium 0.0 | +0.050 | 6, +458/−232 |
| 9 | [`09_cand_r37_0_spatial-grounding`](harnesses/09_cand_r37_0_spatial-grounding) | cand_r29_0_spatial-grounding | 1 | 0.325 | Boxoban-Hard 0.0, Boxoban-Medium 0.0, MazeWalk-9x9 1.0, MazeWalk-15x15 1.0, Corridor-R3 0.4, CorridorBattle-Dark 0.2, Quest-Easy 0.0, Quest-Medium 0.0 | +0.175 | 7, +508/−104 |
| 10 | [`10_cand_r38_0_spatial-grounding`](harnesses/10_cand_r38_0_spatial-grounding) | cand_r1_0_spatial-grounding | 1 | 0.325 | Boxoban-Hard 0.0, Boxoban-Medium 0.0, MazeWalk-9x9 1.0, MazeWalk-15x15 1.0, Corridor-R3 0.6, CorridorBattle-Dark 0.0, Quest-Easy 0.0, Quest-Medium 0.0 | -0.025 | 2, +40/−2 |
| 11 | [`11_cand_r39_0_spatial-grounding`](harnesses/11_cand_r39_0_spatial-grounding) | anchor_naive | 1 | 0.125 | Boxoban-Hard 0.0, Boxoban-Medium 0.0, MazeWalk-9x9 0.8, MazeWalk-15x15 0.2, Corridor-R3 0.0, CorridorBattle-Dark 0.0, Quest-Easy 0.0, Quest-Medium 0.0 | +0.025 | 6, +233/−36 |
| 12 | [`12_cand_r40_0_spatial-grounding`](harnesses/12_cand_r40_0_spatial-grounding) | cand_r33_0_spatial-grounding | 2 | 0.250 | Boxoban-Hard 0.0, Boxoban-Medium 0.0, MazeWalk-9x9 1.0, MazeWalk-15x15 1.0, Corridor-R3 0.0, CorridorBattle-Dark 0.0, Quest-Easy 0.0, Quest-Medium 0.0 | +0.075 | 2, +116/−3 |
| 13 | [`13_cand_r41_0_spatial-grounding`](harnesses/13_cand_r41_0_spatial-grounding) *(on the path)* | cand_r35_0_spatial-grounding | 2 | 0.375 | Boxoban-Hard 0.0, Boxoban-Medium 0.0, MazeWalk-9x9 1.0, MazeWalk-15x15 1.0, Corridor-R3 0.8, CorridorBattle-Dark 0.2, Quest-Easy 0.0, Quest-Medium 0.0 | +0.225 | 1, +102/−11 |
| 14 | [`14_cand_r42_0_spatial-grounding`](harnesses/14_cand_r42_0_spatial-grounding) | cand_r18_0_spatial-grounding | 2 | 0.300 | Boxoban-Hard 0.0, Boxoban-Medium 0.0, MazeWalk-9x9 1.0, MazeWalk-15x15 1.0, Corridor-R3 0.4, CorridorBattle-Dark 0.0, Quest-Easy 0.0, Quest-Medium 0.0 | +0.125 | 5, +167/−6 |
| 15 | [`15_cand_r43_0_spatial-grounding`](harnesses/15_cand_r43_0_spatial-grounding) | cand_r1_0_spatial-grounding | 2 | 0.325 | Boxoban-Hard 0.0, Boxoban-Medium 0.0, MazeWalk-9x9 1.0, MazeWalk-15x15 1.0, Corridor-R3 0.6, CorridorBattle-Dark 0.0, Quest-Easy 0.0, Quest-Medium 0.0 | -0.025 | 3, +91/−18 |
| 16 | [`16_cand_r44_0_spatial-grounding`](harnesses/16_cand_r44_0_spatial-grounding) | cand_r26_0_spatial-grounding | 3 | 0.375 | Boxoban-Hard 0.0, Boxoban-Medium 0.0, MazeWalk-9x9 1.0, MazeWalk-15x15 1.0, Corridor-R3 0.8, CorridorBattle-Dark 0.2, Quest-Easy 0.0, Quest-Medium 0.0 | +0.050 | 2, +129/−12 |
| 17 | [`17_cand_r45_0_spatial-grounding`](harnesses/17_cand_r45_0_spatial-grounding) | cand_r29_0_spatial-grounding | 3 | 0.300 | Boxoban-Hard 0.0, Boxoban-Medium 0.0, MazeWalk-9x9 1.0, MazeWalk-15x15 0.8, Corridor-R3 0.6, CorridorBattle-Dark 0.0, Quest-Easy 0.0, Quest-Medium 0.0 | +0.150 | 5, +309/−86 |
| 18 | [`18_cand_r46_0_spatial-grounding`](harnesses/18_cand_r46_0_spatial-grounding) | cand_r41_0_spatial-grounding | 3 | 0.375 | Boxoban-Hard 0.0, Boxoban-Medium 0.0, MazeWalk-9x9 1.0, MazeWalk-15x15 1.0, Corridor-R3 0.8, CorridorBattle-Dark 0.2, Quest-Easy 0.0, Quest-Medium 0.0 | +0.000 | 4, +470/−214 |
| 19 | [`19_cand_r47_0_spatial-grounding`](harnesses/19_cand_r47_0_spatial-grounding) | anchor_naive | 3 | 0.200 | Boxoban-Hard 0.0, Boxoban-Medium 0.0, MazeWalk-9x9 1.0, MazeWalk-15x15 0.4, Corridor-R3 0.2, CorridorBattle-Dark 0.0, Quest-Easy 0.0, Quest-Medium 0.0 | +0.100 | 6, +234/−49 |
| 20 | [`20_cand_r48_0_spatial-grounding`](harnesses/20_cand_r48_0_spatial-grounding) | cand_r27_0_spatial-grounding | 4 | 0.375 | Boxoban-Hard 0.0, Boxoban-Medium 0.0, MazeWalk-9x9 1.0, MazeWalk-15x15 1.0, Corridor-R3 0.8, CorridorBattle-Dark 0.2, Quest-Easy 0.0, Quest-Medium 0.0 | +0.050 | 3, +241/−101 |
| 21 | [`21_cand_r49_0_spatial-grounding`](harnesses/21_cand_r49_0_spatial-grounding) | cand_r33_0_spatial-grounding | 4 | 0.325 | Boxoban-Hard 0.0, Boxoban-Medium 0.0, MazeWalk-9x9 1.0, MazeWalk-15x15 1.0, Corridor-R3 0.4, CorridorBattle-Dark 0.2, Quest-Easy 0.0, Quest-Medium 0.0 | +0.150 | 3, +168/−2 |
| 22 | [`22_cand_r50_0_spatial-grounding`](harnesses/22_cand_r50_0_spatial-grounding) | cand_r35_0_spatial-grounding | 4 | 0.325 | Boxoban-Hard 0.0, Boxoban-Medium 0.0, MazeWalk-9x9 1.0, MazeWalk-15x15 1.0, Corridor-R3 0.4, CorridorBattle-Dark 0.2, Quest-Easy 0.0, Quest-Medium 0.0 | +0.175 | 1, +81/−4 |
| 23 | [`23_cand_r51_0_spatial-grounding`](harnesses/23_cand_r51_0_spatial-grounding) | cand_r47_0_spatial-grounding | 4 | 0.375 | Boxoban-Hard 0.0, Boxoban-Medium 0.0, MazeWalk-9x9 1.0, MazeWalk-15x15 1.0, Corridor-R3 0.8, CorridorBattle-Dark 0.2, Quest-Easy 0.0, Quest-Medium 0.0 | +0.175 | 2, +279/−33 |
| 24 | [`24_cand_r52_0_spatial-grounding`](harnesses/24_cand_r52_0_spatial-grounding) | cand_r48_0_spatial-grounding | 5 | 0.600 | Boxoban-Hard 0.0, Boxoban-Medium 0.0, MazeWalk-9x9 1.0, MazeWalk-15x15 1.0, Corridor-R3 0.8, CorridorBattle-Dark 0.2, Quest-Easy 0.8, Quest-Medium 1.0 | +0.225 | 3, +344/−6 |
| 25 | [`25_cand_r53_0_spatial-grounding`](harnesses/25_cand_r53_0_spatial-grounding) | anchor_naive | 5 | 0.225 | Boxoban-Hard 0.0, Boxoban-Medium 0.0, MazeWalk-9x9 1.0, MazeWalk-15x15 0.6, Corridor-R3 0.2, CorridorBattle-Dark 0.0, Quest-Easy 0.0, Quest-Medium 0.0 | +0.125 | 5, +176/−49 |
| 26 | [`26_cand_r54_0_spatial-grounding`](harnesses/26_cand_r54_0_spatial-grounding) | cand_r26_0_spatial-grounding | 5 | 0.375 | Boxoban-Hard 0.0, Boxoban-Medium 0.0, MazeWalk-9x9 1.0, MazeWalk-15x15 1.0, Corridor-R3 0.8, CorridorBattle-Dark 0.2, Quest-Easy 0.0, Quest-Medium 0.0 | +0.050 | 3, +361/−193 |
| 27 | [`27_cand_r55_0_spatial-grounding`](harnesses/27_cand_r55_0_spatial-grounding) | cand_r29_0_spatial-grounding | 5 | 0.275 | Boxoban-Hard 0.0, Boxoban-Medium 0.0, MazeWalk-9x9 1.0, MazeWalk-15x15 0.8, Corridor-R3 0.4, CorridorBattle-Dark 0.0, Quest-Easy 0.0, Quest-Medium 0.0 | +0.125 | 8, +397/−98 |
| 28 | [`28_cand_r56_0_spatial-grounding`](harnesses/28_cand_r56_0_spatial-grounding) | cand_r26_0_spatial-grounding | 6 | 0.375 | Boxoban-Hard 0.0, Boxoban-Medium 0.0, MazeWalk-9x9 1.0, MazeWalk-15x15 1.0, Corridor-R3 0.8, CorridorBattle-Dark 0.2, Quest-Easy 0.0, Quest-Medium 0.0 | +0.050 | 1, +83/−33 |
| 29 | [`29_cand_r57_0_spatial-grounding`](harnesses/29_cand_r57_0_spatial-grounding) | cand_r29_0_spatial-grounding | 6 | 0.300 | Boxoban-Hard 0.0, Boxoban-Medium 0.0, MazeWalk-9x9 1.0, MazeWalk-15x15 1.0, Corridor-R3 0.2, CorridorBattle-Dark 0.2, Quest-Easy 0.0, Quest-Medium 0.0 | +0.150 | 8, +395/−106 |
| 30 | [`30_cand_r58_0_spatial-grounding`](harnesses/30_cand_r58_0_spatial-grounding) | cand_r41_0_spatial-grounding | 6 | 0.625 | Boxoban-Hard 0.0, Boxoban-Medium 0.0, MazeWalk-9x9 1.0, MazeWalk-15x15 1.0, Corridor-R3 0.8, CorridorBattle-Dark 0.2, Quest-Easy 1.0, Quest-Medium 1.0 | +0.250 | 2, +460/−42 |
| 31 | [`31_cand_r59_0_spatial-grounding`](harnesses/31_cand_r59_0_spatial-grounding) | cand_r27_0_spatial-grounding | 6 | 0.600 | Boxoban-Hard 0.0, Boxoban-Medium 0.0, MazeWalk-9x9 1.0, MazeWalk-15x15 1.0, Corridor-R3 0.8, CorridorBattle-Dark 0.2, Quest-Easy 0.8, Quest-Medium 1.0 | +0.275 | 4, +617/−101 |
| 32 | [`32_cand_r60_0_spatial-grounding`](harnesses/32_cand_r60_0_spatial-grounding) | cand_r33_0_spatial-grounding | 7 | 0.350 | Boxoban-Hard 0.0, Boxoban-Medium 0.0, MazeWalk-9x9 1.0, MazeWalk-15x15 1.0, Corridor-R3 0.6, CorridorBattle-Dark 0.2, Quest-Easy 0.0, Quest-Medium 0.0 | +0.175 | 3, +141/−6 |
| 33 | [`33_cand_r61_0_spatial-grounding`](harnesses/33_cand_r61_0_spatial-grounding) | cand_r35_0_spatial-grounding | 7 | 0.350 | Boxoban-Hard 0.0, Boxoban-Medium 0.0, MazeWalk-9x9 1.0, MazeWalk-15x15 1.0, Corridor-R3 0.4, CorridorBattle-Dark 0.4, Quest-Easy 0.0, Quest-Medium 0.0 | +0.200 | 1, +80/−2 |
| 34 | [`34_cand_r62_0_spatial-grounding`](harnesses/34_cand_r62_0_spatial-grounding) | cand_r47_0_spatial-grounding | 7 | 0.225 | Boxoban-Hard 0.0, Boxoban-Medium 0.0, MazeWalk-9x9 1.0, MazeWalk-15x15 0.8, Corridor-R3 0.0, CorridorBattle-Dark 0.0, Quest-Easy 0.0, Quest-Medium 0.0 | +0.025 | 4, +145/−24 |
| 35 | [`35_cand_r63_0_spatial-grounding`](harnesses/35_cand_r63_0_spatial-grounding) | cand_r53_0_spatial-grounding | 7 | 0.325 | Boxoban-Hard 0.0, Boxoban-Medium 0.0, MazeWalk-9x9 1.0, MazeWalk-15x15 0.8, Corridor-R3 0.8, CorridorBattle-Dark 0.0, Quest-Easy 0.0, Quest-Medium 0.0 | +0.100 | 2, +111/−5 |
| 36 | [`36_cand_r64_0_spatial-grounding`](harnesses/36_cand_r64_0_spatial-grounding) | cand_r58_0_spatial-grounding | 8 | 0.875 | Boxoban-Hard 1.0, Boxoban-Medium 1.0, MazeWalk-9x9 1.0, MazeWalk-15x15 1.0, Corridor-R3 0.8, CorridorBattle-Dark 0.2, Quest-Easy 1.0, Quest-Medium 1.0 | +0.250 | 2, +243/−8 |
| 37 | [`37_cand_r65_0_spatial-grounding`](harnesses/37_cand_r65_0_spatial-grounding) | cand_r54_0_spatial-grounding | 8 | 0.625 | Boxoban-Hard 0.0, Boxoban-Medium 0.0, MazeWalk-9x9 1.0, MazeWalk-15x15 1.0, Corridor-R3 0.8, CorridorBattle-Dark 0.2, Quest-Easy 1.0, Quest-Medium 1.0 | +0.250 | 4, +359/−2 |
| 38 | [`38_cand_r66_0_spatial-grounding`](harnesses/38_cand_r66_0_spatial-grounding) | cand_r26_0_spatial-grounding | 8 | 0.375 | Boxoban-Hard 0.0, Boxoban-Medium 0.0, MazeWalk-9x9 1.0, MazeWalk-15x15 1.0, Corridor-R3 0.8, CorridorBattle-Dark 0.2, Quest-Easy 0.0, Quest-Medium 0.0 | +0.050 | 4, +484/−197 |
| 39 | [`39_cand_r67_0_spatial-grounding`](harnesses/39_cand_r67_0_spatial-grounding) | cand_r29_0_spatial-grounding | 8 | 0.275 | Boxoban-Hard 0.0, Boxoban-Medium 0.0, MazeWalk-9x9 1.0, MazeWalk-15x15 1.0, Corridor-R3 0.2, CorridorBattle-Dark 0.0, Quest-Easy 0.0, Quest-Medium 0.0 | +0.125 | 9, +435/−112 |
| 40 | [`40_cand_r68_0_spatial-grounding`](harnesses/40_cand_r68_0_spatial-grounding) **(best)** | cand_r41_0_spatial-grounding | 9 | 0.900 | Boxoban-Hard 1.0, Boxoban-Medium 1.0, MazeWalk-9x9 1.0, MazeWalk-15x15 1.0, Corridor-R3 1.0, CorridorBattle-Dark 0.2, Quest-Easy 1.0, Quest-Medium 1.0 | +0.525 | 3, +711/−45 |

## Videos

`videos/` shows the four harnesses on the path to the best (the starting harness, `cand_r35_0`,
`cand_r41_0` and `cand_r68_0`) on four episodes, one per task family the path unlocked in turn:
MazeWalk-15x15 run 00, Corridor-R3 run 04 (first solved by `cand_r35_0`), Boxoban-Medium run 00
and Quest-Easy run 00 (first solved by `cand_r68_0`). Each episode has one video per harness and
a `_progression_grid` video that tiles the four (starting harness top left, best bottom right).

The videos show the game as MiniHack renders it: each episode was replayed in the engine from
the seed and the action sequence BALROG logged (the agent is text-only, so the tiles are a
rendering of the same state it read), with the action taken after each step, the agent's
reasoning tag, the game message and the running return beside the map and the game's status
lines under it. Every replay was checked step by step against the observation text the agent
saw; a video is written only if the two match for the whole episode. The four replays of an
episode share one framing of the map (the union of the regions they uncovered), so that the
panels of the grid line up. `videos_text/` holds the same episodes under the same file names as
the text frames the agent actually received. Both run at two steps per second and were produced
with [`../../render/replay_balrog_episode.py`](../../render/replay_balrog_episode.py),
[`../../render/render_balrog_episode.py`](../../render/render_balrog_episode.py) and
[`../../render/grid_videos.py`](../../render/grid_videos.py) from BALROG's per-episode
`*_run_NN.json` and `*_run_NN.csv.gz` files; any other harness or episode of the run can be
rendered the same way from the run's logs.

| video | harness | outcome |
|---|---|---|
| [`MiniHack-MazeWalk-15x15-v0_run00__00_anchor_naive.mp4`](videos/MiniHack-MazeWalk-15x15-v0_run00__00_anchor_naive.mp4) | `anchor_naive` (eval 1) | failed in 100 steps |
| [`MiniHack-MazeWalk-15x15-v0_run00__07_cand_r35_0_spatial-grounding.mp4`](videos/MiniHack-MazeWalk-15x15-v0_run00__07_cand_r35_0_spatial-grounding.mp4) | `cand_r35_0_spatial-grounding` (eval 8) | failed in 100 steps |
| [`MiniHack-MazeWalk-15x15-v0_run00__13_cand_r41_0_spatial-grounding.mp4`](videos/MiniHack-MazeWalk-15x15-v0_run00__13_cand_r41_0_spatial-grounding.mp4) | `cand_r41_0_spatial-grounding` (eval 14) | solved in 70 steps |
| [`MiniHack-MazeWalk-15x15-v0_run00__40_cand_r68_0_spatial-grounding.mp4`](videos/MiniHack-MazeWalk-15x15-v0_run00__40_cand_r68_0_spatial-grounding.mp4) | `cand_r68_0_spatial-grounding` (eval 41) | solved in 70 steps |
| [`MiniHack-MazeWalk-15x15-v0_run00__progression_grid.mp4`](videos/MiniHack-MazeWalk-15x15-v0_run00__progression_grid.mp4) | the four above, 2×2 (starting harness top left, best bottom right) | — |
| [`MiniHack-Corridor-R3-v0_run04__00_anchor_naive.mp4`](videos/MiniHack-Corridor-R3-v0_run04__00_anchor_naive.mp4) | `anchor_naive` (eval 1) | failed in 100 steps |
| [`MiniHack-Corridor-R3-v0_run04__07_cand_r35_0_spatial-grounding.mp4`](videos/MiniHack-Corridor-R3-v0_run04__07_cand_r35_0_spatial-grounding.mp4) | `cand_r35_0_spatial-grounding` (eval 8) | solved in 67 steps |
| [`MiniHack-Corridor-R3-v0_run04__13_cand_r41_0_spatial-grounding.mp4`](videos/MiniHack-Corridor-R3-v0_run04__13_cand_r41_0_spatial-grounding.mp4) | `cand_r41_0_spatial-grounding` (eval 14) | solved in 41 steps |
| [`MiniHack-Corridor-R3-v0_run04__40_cand_r68_0_spatial-grounding.mp4`](videos/MiniHack-Corridor-R3-v0_run04__40_cand_r68_0_spatial-grounding.mp4) | `cand_r68_0_spatial-grounding` (eval 41) | solved in 41 steps |
| [`MiniHack-Corridor-R3-v0_run04__progression_grid.mp4`](videos/MiniHack-Corridor-R3-v0_run04__progression_grid.mp4) | the four above, 2×2 (starting harness top left, best bottom right) | — |
| [`MiniHack-Boxoban-Medium-v0_run00__00_anchor_naive.mp4`](videos/MiniHack-Boxoban-Medium-v0_run00__00_anchor_naive.mp4) | `anchor_naive` (eval 1) | failed in 100 steps |
| [`MiniHack-Boxoban-Medium-v0_run00__07_cand_r35_0_spatial-grounding.mp4`](videos/MiniHack-Boxoban-Medium-v0_run00__07_cand_r35_0_spatial-grounding.mp4) | `cand_r35_0_spatial-grounding` (eval 8) | failed in 100 steps |
| [`MiniHack-Boxoban-Medium-v0_run00__13_cand_r41_0_spatial-grounding.mp4`](videos/MiniHack-Boxoban-Medium-v0_run00__13_cand_r41_0_spatial-grounding.mp4) | `cand_r41_0_spatial-grounding` (eval 14) | failed in 100 steps |
| [`MiniHack-Boxoban-Medium-v0_run00__40_cand_r68_0_spatial-grounding.mp4`](videos/MiniHack-Boxoban-Medium-v0_run00__40_cand_r68_0_spatial-grounding.mp4) | `cand_r68_0_spatial-grounding` (eval 41) | solved in 38 steps |
| [`MiniHack-Boxoban-Medium-v0_run00__progression_grid.mp4`](videos/MiniHack-Boxoban-Medium-v0_run00__progression_grid.mp4) | the four above, 2×2 (starting harness top left, best bottom right) | — |
| [`MiniHack-Quest-Easy-v0_run00__00_anchor_naive.mp4`](videos/MiniHack-Quest-Easy-v0_run00__00_anchor_naive.mp4) | `anchor_naive` (eval 1) | failed in 100 steps |
| [`MiniHack-Quest-Easy-v0_run00__07_cand_r35_0_spatial-grounding.mp4`](videos/MiniHack-Quest-Easy-v0_run00__07_cand_r35_0_spatial-grounding.mp4) | `cand_r35_0_spatial-grounding` (eval 8) | failed in 100 steps |
| [`MiniHack-Quest-Easy-v0_run00__13_cand_r41_0_spatial-grounding.mp4`](videos/MiniHack-Quest-Easy-v0_run00__13_cand_r41_0_spatial-grounding.mp4) | `cand_r41_0_spatial-grounding` (eval 14) | failed in 100 steps |
| [`MiniHack-Quest-Easy-v0_run00__40_cand_r68_0_spatial-grounding.mp4`](videos/MiniHack-Quest-Easy-v0_run00__40_cand_r68_0_spatial-grounding.mp4) | `cand_r68_0_spatial-grounding` (eval 41) | solved in 48 steps |
| [`MiniHack-Quest-Easy-v0_run00__progression_grid.mp4`](videos/MiniHack-Quest-Easy-v0_run00__progression_grid.mp4) | the four above, 2×2 (starting harness top left, best bottom right) | — |
