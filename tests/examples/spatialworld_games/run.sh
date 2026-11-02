#!/usr/bin/env bash
# Harness-MAP on one SpatialWorld 3D game with the paper's settings (config.json). Replace the
# placeholder paths first; README.md describes the setup and what the evaluator must write.
set -euo pipefail
cd "$(dirname "$0")/../../.."

export SPATIALWORLD_DIR=/path/to/SpatialWorld              # at 8e10e24 with spatialworld.patch applied
export GAME_PYTHON=/path/to/spatialworld-game-env/bin/python
export GAME_EVALUATOR=/path/to/game_evaluator.py           # runs a harness on a split, writes score.json
export POLICY_BASE_URL=http://127.0.0.1:8000/v1            # vLLM server with the frozen policy model
export POLICY_MODEL=Qwen3-VL-30B-A3B-Instruct

GAME=${GAME:-maze}                                         # maze | block3d | maze3d_pro | rubik | snake
ANCHOR=/path/to/anchors/$GAME                              # the game's anchor harness
MODE=${MODE:-adaptive}                                     # adaptive | fixed | hier
K0=${K0:-4}                                                # the paper sweeps 2 4 8 16 32

HOLDOUT=()
if [ "$GAME" = snake ]; then HOLDOUT=(--holdout-split snake_holdout); fi   # only snake has one
python tests/examples/run_benchmark.py --config tests/examples/spatialworld_games/config.json \
    --anchor "$ANCHOR" --search-split "$GAME" ${HOLDOUT[@]+"${HOLDOUT[@]}"} --mode "$MODE" --k0 "$K0"
