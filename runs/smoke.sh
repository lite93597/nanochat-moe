#!/usr/bin/env bash
# Offline CPU correctness checks. Requires an installed CPU-compatible environment.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR/bundle/moe"
export NANOCHAT_DTYPE=float32
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
"${PYTHON:-python}" -m pytest -q \
  tests/test_moe.py tests/test_moe_training_metrics.py \
  tests/test_moe_repair_diagnostics.py tests/test_attention_fallback.py \
  tests/test_infer_bench.py tests/test_public_repair_tools.py
