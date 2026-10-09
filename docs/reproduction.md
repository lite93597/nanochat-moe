# Run the source workflow

[Back to README](../README.md) · [Experiments](experiments.md) · [Weights](weights.md)

This guide lets you test the implementation and train **your own** Dense/MoE models. Reproducing the historical R5 run requires its source recovery checkpoint, optimizer shards, and calibration inputs, which are not included in the repository.

## 1. Environment

Use Python 3.10+ and [uv](https://docs.astral.sh/uv/). Commands below begin at the repository root unless a `cd` changes that explicitly. GPU runs use Linux/Bash and a compatible CUDA driver for the PyTorch CUDA 12.8 wheel.

```bash
git clone https://github.com/lite93597/nanochat-moe.git nanochat-moe
cd nanochat-moe/bundle/moe
uv sync --extra gpu --group dev
source .venv/bin/activate
export OMP_NUM_THREADS=1
export NANOCHAT_DTYPE=bfloat16
export NANOCHAT_BASE_DIR=/persistent/nanochat-moe/models
export NANOCHAT_DATA_DIR=/persistent/nanochat-moe/data
```

The two directories should be explicit, writable, persistent locations with enough space for data, checkpoints, and optimizer shards. Replace `/persistent` with your storage mount. For CPU tests, use `--extra cpu` and `NANOCHAT_DTYPE=float32` instead.

The measured node had eight RTX PRO 6000 Blackwell Server Edition GPUs. Memory/throughput on other GPUs is a new measurement, not an upstream speedrun guarantee. Every GPU stores the complete model and all experts.

## 2. Offline small-model checks

From `bundle/moe`, with the matching environment:

```bash
NANOCHAT_DTYPE=float32 python -m pytest -q \
  tests/test_moe.py tests/test_moe_training_metrics.py \
  tests/test_moe_repair_diagnostics.py tests/test_infer_bench.py
```

The selected tests download nothing. They check small-model math, gradient/unused-expert behavior, reporting, diagnostics, and benchmark helpers. They do not establish full-scale GPU convergence. `runs/smoke.sh` at the repository root is also an offline test wrapper using your active Python.

## 3. Prepare data and tokenizer before the long GPU run

Still from `bundle/moe`:

```bash
# Downloads training shards plus the fixed validation shard.
python -m nanochat.dataset -n 8
python -m scripts.tok_train --max-chars 2000000000 --vocab-size 32768
python -m scripts.tok_eval

# Additional training data; select a count suitable for your experiment.
python -m nanochat.dataset -n 170
```

These commands require network access, time, and disk space. They use the project's ClimbMix dataset reader. Eight initial shards are a tokenizer-preparation setting; `-n 170` is a workflow default, not a guarantee of precisely the original consumed documents.

SFT/evaluation task loaders also download Hub datasets if the cache is absent. Prepare them before renting GPUs if possible. A simple SFT-cache preparation, from `bundle/moe`, is:

```bash
python - <<'PY'
from tasks.smoltalk import SmolTalk
from tasks.mmlu import MMLU
from tasks.gsm8k import GSM8K
for split in ("train", "test"):
    SmolTalk(split=split)
for split in ("auxiliary_train", "test"):
    MMLU(subset="all", split=split)
for split in ("train", "test"):
    GSM8K(subset="main", split=split)
PY
```

The inherited SFT validation mixture uses some benchmark test data. Cache preparation does not train on it, but new strict experiments should change the validation split before monitoring; see [the limitation](experiments.md#validationtest-separation). CORE/ARC/HumanEval assets may still need separate preparation or first-use downloads.

## 4. A short capacity check

Return to the repository root; the activated environment stays on `PATH`:

```bash
cd ../..
MODEL_TAG=moe12_capacity_check NUM_ITERATIONS=20 \
  BASE_EVAL_EVERY=-1 BASE_CORE_METRIC_EVERY=-1 BASE_SAMPLE_EVERY=-1 \
  bash runs/train_moe12.sh
```

The wrapper defaults to prepared data (`SKIP_DATA_PREP=1`), your existing environment (`SKIP_SETUP=1`), and pretraining only (`STOP_AFTER_BASE=1`). It does not silently provision an environment or download the corpus. Inspect finite loss, route use, available memory, and steady update time. Use a new tag for the full run; this pilot used a different scheduler horizon.

If memory is insufficient, lower `DEVICE_BATCH_SIZE` while retaining a divisible `TOTAL_BATCH_SIZE`; the trainer changes accumulation accordingly. A different model/context/expert configuration needs a new tag and a fresh capacity check.

## 5. Twelve-layer pretraining

From the repository root:

```bash
MODEL_TAG=moe12_experiment bash runs/train_moe12.sh
MODEL_TAG=dense12_experiment bash runs/train_dense12.sh
```

Run these sequentially on one node. Defaults:

| Setting | Dense12 wrapper | MoE12 wrapper |
|---|---:|---:|
| Depth / width | 12 / 768 | 12 / 768 |
| Attention window | Full (`L`) | Full (`L`) |
| Per-rank microbatch | 32 | 16 |
| Ranks | 8 | 8 |
| Global scheduled positions/update | 524288 | 524288 |
| Updates | 1680 | 1680 |
| Experts / Top-k | 0 / — | 8 / 2 |

The Dense wrapper selects the compatible zero-expert path in the public runner; the original reference snapshot remains in `bundle/dense` for comparison. This is a source workflow, not a bitwise replay of the historical run.

Existing tags with checkpoints are protected from accidental new-run overwrite. Resume explicitly with the matching architecture/tag and complete checkpoint files. The loader restores shard/row-group/epoch state, not every prefetched document or RNG/packing buffer.

## 6. Full-epoch SFT

From the repository root, choose a completed Base tag, then enter `bundle/moe`:

```bash
export MODEL_TAG=moe12_experiment
cd bundle/moe
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft_recovery_r1 -- \
  --model-tag="$MODEL_TAG" --run=dummy --num-iterations=-1 \
  --router-diagnostics --chatcore-every=-1 --eval-every=-1 --eval-tokens=524288
cd ../..
```

Repeat with `MODEL_TAG=dense12_experiment` for the baseline. The active SFT entry contains the accepted masked-token counting and JSON/save synchronization corrections. It loads Base optimizer state by default and runs the original full mixed epoch; step counts can differ with packing/input order.

`--eval-every=-1` disables periodic validation, but the entry still performs endpoint checks; the inherited validation/test overlap is therefore still relevant. Do not pass the fixed-20-layer guard to a 12-layer model. Training diagnostic collection is not itself a guarantee of healthy routing.

For an explicitly requested end-to-end run, `STOP_AFTER_BASE=0` on the training wrapper continues into Base evaluation, new-entry SFT, and chat evaluation. `SKIP_DATA_PREP=0` enables dataset/tokenizer preparation. These opt-ins can consume substantial time; the staged workflow makes their costs and outputs easier to inspect.

## 7. Evaluate, inspect routes, and chat

From the repository root, after Base and SFT exist:

```bash
MODEL_TAG=moe12_experiment bash runs/evaluate.sh
```

The wrapper runs full Base CORE/BPB/sample, the five SFT chat tasks, and a single-device inference benchmark. It produces route files and benchmark output under `outputs/evaluation/<tag>` by default; task scores are printed and also follow the upstream report paths. Missing cached assets may trigger downloads.

The recorded 31-stage experiment also included a separate inference functionality check in addition to these tasks.

For one chat with route output:

```bash
cd bundle/moe
python -m scripts.chat_cli -i sft -g moe12_experiment \
  -p "Explain the difference between weather and climate." --max-tokens 128 \
  --moe-routing-output ./chat_routes.json
```

Read `assignment_share` as count/(2N), `token_hit_rate` as count/N, and `gate_share` as selected normalized weight/N. Compare adequate sample sizes; a short prefill's zero-hit expert does not establish collapse. [Architecture](architecture.md) explains all denominators and phases.

## 8. Twenty-layer source experiments and historical R5

A fresh 20-layer experiment can use the underlying `bundle/moe/runs/speedrun.sh` with explicit depth, batch, horizon, tag, and router learning rate. It starts a new model; it does not replay R5 automatically.

The reported MoE20-R5 has width1280, microbatch8 on eight ranks, accumulation4, 6641 scheduled pretraining updates, then full-epoch SFT. It includes an intervention on the actual 3600-step checkpoint and validation-based acceptance. Reproducing that recovery requires its original checkpoint, optimizer shards, calibration inputs, and validation records, which are not distributed here.

[`research/historical_r5`](../research/historical_r5/) preserves the relevant repair/calibration algorithms for review. [`tools/upcycle_experts.py`](../tools/upcycle_experts.py) copies donor weights into a new **model-only** state dict; [`tools/calibrate_router_bias.py`](../tools/calibrate_router_bias.py) fits bias from saved TRAIN logits. Neither creates a resumable checkpoint or passes a quality gate. Preserve these boundaries when designing a new recovery experiment.

## Troubleshooting checklist

| Symptom | First checks |
|---|---|
| Missing tokenizer/data | Confirm both `NANOCHAT_*_DIR` variables and completed preparation |
| CUDA OOM | Lower microbatch, inspect full expert residency/activation/optimizer buffers |
| NaN or infinite values | Save evidence and inspect CE, router probabilities, gradients, precision |
| Loss decreases but routes concentrate | Inspect every layer, CV/min share/zero experts and sustained windows |
| Distributed hang | Check rank participation, missing expert gradients, identical collective order |
| Resume mismatch | Check architecture, step/meta, tokenizer, all optimizer shards, source identity |
| Slow evaluation | Check full question counts, generation length, task/cache downloads, batch scope |

Do not treat a partial checkpoint, a one-window route ratio, or an offline smoke pass as a completed full training result.
