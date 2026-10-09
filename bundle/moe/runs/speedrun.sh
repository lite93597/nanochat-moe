#!/usr/bin/env bash
# Nanochat MoE end-to-end run: dataset -> tokenizer -> pretrain -> eval -> SFT -> eval.
# Run from any directory with: bash /path/to/nanochat-moe/runs/speedrun.sh
# The hardware requirements and runtime depend on the chosen model and GPUs.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat-moe}"
export NANOCHAT_DATA_DIR="${NANOCHAT_DATA_DIR:-$HOME/.cache/nanochat-moe-data}"
mkdir -p "$NANOCHAT_BASE_DIR" "$NANOCHAT_DATA_DIR"

# Defaults are a starting point, not a guarantee that any particular GPU fits.
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
DEPTH="${DEPTH:-12}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
WINDOW_PATTERN="${WINDOW_PATTERN:-L}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-4}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:--1}"
TARGET_PARAM_DATA_RATIO="${TARGET_PARAM_DATA_RATIO:-8}"
NUM_ITERATIONS="${NUM_ITERATIONS:--1}"
RESUME_FROM_STEP="${RESUME_FROM_STEP:--1}"
MOE_NUM_EXPERTS="${MOE_NUM_EXPERTS:-8}"
MOE_TOP_K="${MOE_TOP_K:-2}"
MOE_HIDDEN_MULT="${MOE_HIDDEN_MULT:-2.0}"
MOE_EVERY="${MOE_EVERY:-2}"
MOE_AUX_LOSS_COEF="${MOE_AUX_LOSS_COEF:-0.01}"
ROUTER_LR="${ROUTER_LR:-}"
MODEL_TAG="${MODEL_TAG:-d${DEPTH}_s${MAX_SEQ_LEN}_moe_e${MOE_NUM_EXPERTS}k${MOE_TOP_K}h${MOE_HIDDEN_MULT}every${MOE_EVERY}}"
WANDB_RUN="${WANDB_RUN:-dummy}"
TOKENIZER_SHARDS="${TOKENIZER_SHARDS:-8}"
DATA_SHARDS="${DATA_SHARDS:-170}"
BASE_EVAL_EVERY="${BASE_EVAL_EVERY:-250}"
BASE_CORE_METRIC_EVERY="${BASE_CORE_METRIC_EVERY:-2000}"
BASE_SAMPLE_EVERY="${BASE_SAMPLE_EVERY:-2000}"

if (( MOE_NUM_EXPERTS != 0 && (MOE_NUM_EXPERTS < 2 || MOE_TOP_K < 1 || MOE_TOP_K > MOE_NUM_EXPERTS || MOE_EVERY < 1) )); then
    echo "Invalid MoE configuration: require experts = 0 (Dense), or experts >= 2, 1 <= top_k <= experts, every >= 1" >&2
    exit 2
fi
if (( DATA_SHARDS < TOKENIZER_SHARDS )); then
    echo "DATA_SHARDS must be at least TOKENIZER_SHARDS" >&2
    exit 2
fi
if [[ "$RESUME_FROM_STEP" == "-1" ]] && compgen -G "$NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG/model_*.pt" >/dev/null; then
    echo "Model tag $MODEL_TAG already has a base checkpoint. Choose a new MODEL_TAG or set RESUME_FROM_STEP explicitly." >&2
    exit 2
fi
ROUTER_LR_ARGS=()
if [[ -n "$ROUTER_LR" ]]; then
    ROUTER_LR_ARGS=(--router-lr="$ROUTER_LR")
fi

# Set SKIP_SETUP=1 to use the current Python environment (for example, Conda).
if [[ "${SKIP_SETUP:-0}" != "1" ]]; then
    command -v uv >/dev/null 2>&1 || { echo "Install uv first, or set SKIP_SETUP=1" >&2; exit 2; }
    uv sync --extra gpu
    source .venv/bin/activate
fi

echo "MoE model tag: $MODEL_TAG"
echo "Checkpoints/tokenizer: $NANOCHAT_BASE_DIR"
echo "Dataset: $NANOCHAT_DATA_DIR"

# Set SKIP_DATA_PREP=1 when the required shards and trained tokenizer already exist.
if [[ "${SKIP_DATA_PREP:-0}" != "1" ]]; then
    python -m nanochat.dataset -n "$TOKENIZER_SHARDS"
    python -m nanochat.dataset -n "$DATA_SHARDS" &
    DATASET_DOWNLOAD_PID=$!
    python -m scripts.tok_train
    python -m scripts.tok_eval
    wait "$DATASET_DOWNLOAD_PID"
else
    for tokenizer_file in tokenizer.pkl token_bytes.pt; do
        if [[ ! -f "$NANOCHAT_BASE_DIR/tokenizer/$tokenizer_file" ]]; then
            echo "Missing prepared tokenizer: $NANOCHAT_BASE_DIR/tokenizer/$tokenizer_file" >&2
            echo "Prepare data/tokenizer explicitly, or set SKIP_DATA_PREP=0 for the download pipeline." >&2
            exit 2
        fi
    done
    shopt -s nullglob
    DATA_FILES=("$NANOCHAT_DATA_DIR"/base_data_climbmix/shard_*.parquet)
    shopt -u nullglob
    if (( ${#DATA_FILES[@]} < 2 )); then
        echo "Need at least two prepared ClimbMix shards (training and held-out validation) in NANOCHAT_DATA_DIR." >&2
        exit 2
    fi
fi

# Expert weights are replicated on every GPU; nanochat uses custom gradient
# collectives and sharded optimizer state, without expert parallelism.
# FP8 is intentionally omitted until its interaction with sparse experts is validated.
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_train -- \
    --depth="$DEPTH" \
    --max-seq-len="$MAX_SEQ_LEN" \
    --window-pattern="$WINDOW_PATTERN" \
    --router-diagnostics \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --total-batch-size="$TOTAL_BATCH_SIZE" \
    --target-param-data-ratio="$TARGET_PARAM_DATA_RATIO" \
    --num-iterations="$NUM_ITERATIONS" \
    --resume-from-step="$RESUME_FROM_STEP" \
    --moe-num-experts="$MOE_NUM_EXPERTS" \
    --moe-top-k="$MOE_TOP_K" \
    --moe-hidden-mult="$MOE_HIDDEN_MULT" \
    --moe-every="$MOE_EVERY" \
    --moe-aux-loss-coef="$MOE_AUX_LOSS_COEF" \
    --eval-every="$BASE_EVAL_EVERY" \
    --core-metric-every="$BASE_CORE_METRIC_EVERY" \
    --sample-every="$BASE_SAMPLE_EVERY" \
    --model-tag="$MODEL_TAG" \
    --run="$WANDB_RUN" \
    "${ROUTER_LR_ARGS[@]}"

# For a small cloud capacity check, stop before the long evaluation/SFT stages.
if [[ "${STOP_AFTER_BASE:-0}" == "1" ]]; then
    echo "Pretraining complete; STOP_AFTER_BASE=1 requested. Model tag: $MODEL_TAG"
    exit 0
fi

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_eval -- \
    --model-tag="$MODEL_TAG" --device-batch-size="$DEVICE_BATCH_SIZE"

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.chat_sft_recovery_r1 -- \
    --model-tag="$MODEL_TAG" --run="$WANDB_RUN" --router-diagnostics

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.chat_eval -- \
    -i sft --model-tag="$MODEL_TAG"

# RL is an optional next stage, as in the upstream project.
if [[ "${RUN_RL:-0}" == "1" ]]; then
    torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.chat_rl -- \
        --model-tag="$MODEL_TAG" --run="$WANDB_RUN"
    torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.chat_eval -- \
        -i rl --model-tag="$MODEL_TAG"
fi

echo "Run complete. Chat: python -m scripts.chat_cli -i sft -g $MODEL_TAG -p 'Hello'"
