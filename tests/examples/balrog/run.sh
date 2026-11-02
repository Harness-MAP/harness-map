#!/usr/bin/env bash
# Harness-MAP on BALROG-MiniHack with the paper's settings (config.json). Replace the placeholder
# paths first; README.md describes the setup and what the evaluator must write.
set -euo pipefail
cd "$(dirname "$0")/../../.."

export BALROG_PYTHON=/path/to/balrog-env/bin/python       # interpreter with BALROG, NLE and MiniHack
export BALROG_EVALUATOR=tests/tasks/balrog/evaluate.py      # runs a harness on a split, writes score.json
export POLICY_BASE_URL=http://127.0.0.1:8000/v1           # vLLM server with the frozen base model
export POLICY_MODEL=Qwen3-VL-30B-A3B-Instruct
ANCHOR=/path/to/anchor_harness                             # BALROG's naive agent in the harness layout

MODE=${MODE:-adaptive}                                     # adaptive | fixed | hier
K0=${K0:-4}                                                # the paper sweeps 2 4 8 16 32

python tests/examples/run_benchmark.py --config tests/examples/balrog/config.json \
    --anchor "$ANCHOR" --mode "$MODE" --k0 "$K0"
