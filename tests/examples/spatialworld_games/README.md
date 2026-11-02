# SpatialWorld 3D games

Harness-MAP evolves a vision-language game agent for the five 3D games of
[SpatialWorld](https://github.com/Hongcheng-Gao/SpatialWorld): maze (20 levels), block3d (20),
maze3d_pro (25), rubik (20) and snake (20 seeds, plus 5 held-out seeds). Each game is searched
separately with a budget of N = 50 evaluations; an evaluation plays every level of the game.

The paper evaluates a **pure-visual variant** of the games: `spatialworld.patch` removes the
on-screen text that would let an agent read its exact position or the legal moves (position and
cursor readouts, axis labels, stair tips, move-affordance letters) from block3d, maze3d_pro and
snake; maze, whose engine draws a position-and-heading bar above the view, and rubik are run as
shipped. Scores are therefore not comparable with the official SpatialWorld leaderboard, which
renders that text.

## Setup

1. **SpatialWorld** at commit 8e10e24 with the patch, and its game environment:
   ```bash
   git clone https://github.com/Hongcheng-Gao/SpatialWorld.git && cd SpatialWorld && git checkout 8e10e24
   git apply /path/to/harness-map/examples/spatialworld_games/spatialworld.patch
   ln -s game/Block3D data/Block3D && ln -s game/maze3d data/maze3d && ln -s game/maze3d_pro data/maze3d_pro \
       && ln -s game/rubik_cube data/rubik_cube         # the game runners resolve levels through data/
   cd envs/game && uv sync                             # pygame, PyOpenGL, openai, ...
   ```
   The patch also adds a client-side image budget (`SPATIALWORLD_MAX_IMAGES`) and an output-token
   setting (`GAME_MAX_TOKENS`) to the game runner's model client.
2. **Base model**: a vLLM server with `Qwen3-VL-30B-A3B-Instruct`. The agent sends up to 30
   frames per request, so allow enough image slots and context:
   ```bash
   vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct --served-model-name Qwen3-VL-30B-A3B-Instruct \
       --port 8000 --max-model-len 65536 --max-num-seqs 8 --limit-mm-per-prompt '{"image":32}'
   ```
3. **Meta-agent**: the Claude Code CLI (`claude`) on `PATH`, or another backbone (top-level README).
   The paper's runs additionally prevented the meta-agent from reading the game engine's source,
   so that it could only learn from the rendered frames and the harness contract; the simplest way
   is to run the search in a container that mounts only the run directory and the anchor.
4. **Anchor harnesses**, one per game, in the layout of
   `harness_map/meta_agent/skills/proposer/SKILL.md`. `agent.py` exports
   `build_agent(evaluator)` returning an async per-step function `agent(frame_data) -> Action`;
   the paper's anchors compose the system prompt from `AGENT.md`, `skills/*.md` and `rules.md`
   and delegate each step to the frozen model through `evaluator.baseline_step`, reproducing
   SpatialWorld's shipped agent.
5. **Evaluator** (`GAME_EVALUATOR` in `run.sh`): a script run under the game environment as
   `evaluator.py --harness DIR --tasks TASKS.json --out OUT`. For each task record (`game`,
   `level` or `seed`) it should run one episode through SpatialWorld's own per-game evaluator
   class (`scripts/game/<game>_openai_evaluation.py`), in a fresh process because pygame cannot
   be re-initialized, with the harness's `build_agent(evaluator)` installed as the per-step model
   function. Scores per episode: `success` / `won` / `cube_solved` (1 or 0) for maze, block3d,
   maze3d_pro and rubik, and `min(food eaten / 4, 1)` for snake. The paper's runs void an
   episode (score 0) when the harness reads engine state other than the rendered frame
   (`evaluator.game.*` apart from `.screen`), reads the level files under `data/`, or never calls
   the model. Write `OUT/score.json` with `score` (mean over episodes), `n_episodes` and
   `per_episode` rows with `task_id` and `score`.

## Run

Fill in the placeholders in `run.sh`, then:

```bash
GAME=maze       MODE=adaptive K0=4 bash tests/examples/spatialworld_games/run.sh
GAME=maze3d_pro MODE=adaptive K0=8 bash tests/examples/spatialworld_games/run.sh
GAME=snake      MODE=fixed    K0=4 bash tests/examples/spatialworld_games/run.sh   # also scores the held-out seeds
```

`config.json` fixes the budget at N = 50; with P = ⌈log2 k0⌉ and C = 2 (k0 ≤ 4) or 1 (k0 ≥ 8),
k0 = 2, 4, 8, 16, 32 give G = 24, 12, 15, 10, 6 generations. The games adapter evaluates through
`evaluate` only, so the LLM judge is not run; the static pre-evaluation gate and the runtime
guard above are the anti-cheat checks.

Outputs land in `runs/spatialworld-3dgames/<game>/<mode>_k<k0>/` (`result.json`, `candidates/`,
`evals/`, `checkpoints/evals.jsonl`).

## The paper's best runs

[`evolution/`](evolution/README.md) holds, for each game, every harness of the paper's best run
from the anchor to the first harness that solved every level, with the proposer's notes and the
diagnostician's reports, and videos of one level played by four of them (the anchor, two in
between and the best; the engine's own recordings, or engine replays for rubik).
