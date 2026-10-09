#!/usr/bin/env bash
# Prepared-data Dense baseline through the compatible public runner. Original source stays in bundle/dense.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DEPTH=12
export MOE_NUM_EXPERTS=0
export WINDOW_PATTERN=L
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-32}"
export TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-524288}"
export NUM_ITERATIONS="${NUM_ITERATIONS:-1680}"
export MODEL_TAG="${MODEL_TAG:-dense12_run}"
export SKIP_SETUP="${SKIP_SETUP:-1}"
export SKIP_DATA_PREP="${SKIP_DATA_PREP:-1}"
export STOP_AFTER_BASE="${STOP_AFTER_BASE:-1}"
exec bash "$REPO_DIR/bundle/moe/runs/speedrun.sh"
