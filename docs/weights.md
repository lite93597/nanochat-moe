# Weights, tokenizer, and checkpoint scope

[Back to README](../README.md) · [Reproduction](reproduction.md)

## Current availability

This GitHub release includes **source code and public aggregate experiment results**. It does **not** include a publicly hosted pretrained checkpoint, a Git LFS weight download, or an external model-hub link.

The final MoE20-R5 checkpoints correspond to these training endpoints:

| Artifact | Endpoint | Parameters |
|---|---:|---:|
| Base checkpoint | Pretrain step 6641 | 1,289,852,146 |
| SFT checkpoint | Full mixed epoch, step 932 | 1,289,852,146 |
| Tokenizer / metadata | Vocabulary 32768, context 2048 | Required to load either model |

The checkpoint export is approximately **8.524 GB** and contains the two final model files, tokenizer, metadata, and supporting calibration/provenance records. Validation covers archive/file checksums and model tensor/configuration consistency. The export is not currently available to download from this repository.

Until an external model host is published, use the [source training workflow](reproduction.md). Do not expect `git clone` or the first chat command to provide pretrained weights.

## Why weights are separate from Git

Large model binaries and datasets do not belong in normal Git history. A future weight release should use a suitable model host and include an immutable revision, model card, configuration, tokenizer, license/provenance terms, SHA256 checksums, and a verified load example. This document will contain the actual links only when such a release exists.

## Layout for models you train

The loader uses the explicit environment variable `NANOCHAT_BASE_DIR`:

```text
NANOCHAT_BASE_DIR/
├── tokenizer/
├── base_checkpoints/
│   └── YOUR_MODEL_TAG/
│       ├── model_XXXXXX.pt
│       └── meta_XXXXXX.json
└── chatsft_checkpoints/
    └── YOUR_MODEL_TAG/
        ├── model_XXXXXX.pt
        └── meta_XXXXXX.json
```

Training checkpoints may also contain optimizer shards. For actual resumption, keep all required model, metadata, and per-rank optimizer files together and validate that they agree. A model file alone is enough neither for equivalent optimizer resumption nor for exact historical R5 recovery.

The released final export is not a complete archive of all historical source/recovery optimizer states. Do not claim that it can resume every intermediate repair attempt. Data-loader state is also coarse: it records shard/row-group/epoch rather than all random/prefetch/packing buffers.

## Loading your own checkpoint

From `bundle/moe`, with dependencies installed and the correct tokenizer/metadata:

```bash
export NANOCHAT_BASE_DIR=/persistent/nanochat-moe/models
python -m scripts.chat_cli -i sft -g YOUR_MODEL_TAG -p "Hello" --max-tokens 128
python -m scripts.infer_bench -i sft -g YOUR_MODEL_TAG \
  --prompt-tokens 256 --decode-tokens 64 --batch-sizes 1,8
```

Only load trusted PyTorch checkpoint files. Preserve the tokenizer and full MoE configuration, including the R5 router bias/hold fields when applicable; reconstructing a superficially similar model does not establish identity with the trained model.

## Terms and intended use

The MIT license covers repository code. It does not relicense the training corpora or automatically specify the terms of a future hosted weight release. These small experimental models have limited factual/reasoning ability; the published scores and benchmark scope are documented in [experiments](experiments.md).
