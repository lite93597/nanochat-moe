#!/usr/bin/env bash
# Full upstream CORE/BPB/sample and the five chat tasks; separate from training.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR/bundle/moe"
: "${MODEL_TAG:?Set MODEL_TAG to the checkpoint directory name}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-8}"
SOURCE="${SOURCE:-sft}"
REPORT_DIR="${REPORT_DIR:-$REPO_DIR/outputs/evaluation/$MODEL_TAG}"
mkdir -p "$REPORT_DIR"
BASE_STEP_ARGS=()
SFT_STEP_ARGS=()
if [[ -n "${BASE_STEP:-}" ]]; then BASE_STEP_ARGS=(--step="$BASE_STEP"); fi
if [[ -n "${SFT_STEP:-}" ]]; then SFT_STEP_ARGS=(--step="$SFT_STEP"); fi
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_eval -- \
  --model-tag="$MODEL_TAG" --device-batch-size="$DEVICE_BATCH_SIZE" \
  --max-per-task=-1 --moe-routing-output="$REPORT_DIR/base_routes.json" "${BASE_STEP_ARGS[@]}"
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.chat_eval -- \
  -i "$SOURCE" -g "$MODEL_TAG" --moe-routing-output="$REPORT_DIR/chat_routes.json" "${SFT_STEP_ARGS[@]}"
"${PYTHON:-python}" -m scripts.infer_bench --model-tag="$MODEL_TAG" --source="$SOURCE" \
  --prompt-tokens=256 --decode-tokens=64 --batch-sizes=1,8 \
  "${SFT_STEP_ARGS[@]}" | tee "$REPORT_DIR/infer_bench.log"
tail -n 1 "$REPORT_DIR/infer_bench.log" > "$REPORT_DIR/infer_bench.json"
