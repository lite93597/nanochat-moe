#!/usr/bin/env bash
# Prepared-data 12-layer MoE pretraining. This is not the historical R5 recovery run.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DEPTH=12
export MOE_NUM_EXPERTS=8
export MOE_TOP_K=2
export MOE_HIDDEN_MULT=2.0
export MOE_EVERY=2
export WINDOW_PATTERN=L
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"
export TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-524288}"
export NUM_ITERATIONS="${NUM_ITERATIONS:-1680}"
export MODEL_TAG="${MODEL_TAG:-moe12_run}"
export SKIP_SETUP="${SKIP_SETUP:-1}"
export SKIP_DATA_PREP="${SKIP_DATA_PREP:-1}"
export STOP_AFTER_BASE="${STOP_AFTER_BASE:-1}"
exec bash "$REPO_DIR/bundle/moe/runs/speedrun.sh"
