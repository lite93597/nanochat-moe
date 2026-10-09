#!/usr/bin/env bash
# Offline small-model correctness and train/generate smoke test. Downloads nothing.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export NANOCHAT_DTYPE=float32
"${PYTHON:-python}" -m pytest -q tests/test_moe.py
