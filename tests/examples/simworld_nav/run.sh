#!/usr/bin/env bash
# Harness-MAP on SimWorld-MMNav with the paper's settings (config.json). Replace the placeholder
# paths first; README.md describes the setup and what the evaluator must write.
set -euo pipefail
cd "$(dirname "$0")/../../.."

export SIMWORLD_ROBOTICS_DIR=/path/to/SimWorld-Robotics    # gym + baselines, with the test data
export SIMWORLD_PYTHON=/path/to/simworld-env/bin/python
export NAV_EVALUATOR=/path/to/nav_evaluator.py             # runs a harness on a split, writes score.json
export UNREALCV_PORT=9000                                  # port of the running SimWorld simulator
export POLICY_BASE_URL=http://127.0.0.1:8000/v1            # vLLM server with the frozen policy model
export POLICY_MODEL=Qwen3-VL-30B-A3B-Instruct
ANCHOR=/path/to/nav_anchor_harness

SETTING=${SETTING:-hard}                                   # hard | easy
MODE=${MODE:-adaptive}                                     # adaptive | fixed | hier
K0=${K0:-4}                                                # the paper uses 2 4 8

python tests/examples/run_benchmark.py --config tests/examples/simworld_nav/config.json \
    --anchor "$ANCHOR" --search-split "search_$SETTING" --holdout-split "val_$SETTING" \
    --mode "$MODE" --k0 "$K0"
