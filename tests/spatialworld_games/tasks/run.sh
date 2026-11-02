#!/usr/bin/env bash
# Harness-MAP on one SpatialWorld 3D game with the paper's settings (config.json). Replace the
# placeholder paths first; README.md describes the setup and what the evaluator must write.
set -euo pipefail
cd "$(dirname "$0")/../../.."

export SPATIALWORLD_REPO=/path/to/SpatialWorld             # at 8e10e24 with spatialworld.patch applied
export GAME_PYTHON=/path/to/spatialworld-game-env/bin/python
export GAME_EVALUATOR=tests/spatialworld_games/tasks/evaluate.py           # runs a harness on a split, writes score.json
export GAME_API_BASE=http://127.0.0.1:8000/v1              # vLLM server with the frozen base model
export GAME_MODEL=Qwen3-VL-30B-A3B-Instruct

GAME=${GAME:-maze}                                         # maze | block3d | maze3d_pro | rubik | snake
STARTING_HARNESS=/path/to/starting_harnesses/$GAME          # the game's starting harness
MODE=${MODE:-adaptive}                                     # adaptive | fixed | hier
K0=${K0:-4}                                                # the paper sweeps 2 4 8 16 32

HOLDOUT=()
if [ "$GAME" = snake ]; then HOLDOUT=(--holdout-split snake_holdout); fi   # only snake has one
python tests/run_benchmark.py --config tests/spatialworld_games/tasks/config.json \
    --starting-harness "$STARTING_HARNESS" --search-split "$GAME" ${HOLDOUT[@]+"${HOLDOUT[@]}"} --mode "$MODE" --k0 "$K0"
