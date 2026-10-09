# Experiments and what the scores mean

[Back to README](../README.md) · [Architecture](architecture.md) · [Reproduction](reproduction.md)

## Model and training context

All reported models use a 32768-token BPE vocabulary, context 2048, head dimension 128, and full causal attention. Dense12 and MoE12 share depth/width; MoE20 increases both.

| Configuration | Dense12 | MoE12 | MoE20-R5 |
|---|---:|---:|---:|
| Layers | 12 | 12 | 20 |
| Width | 768 | 768 | 1280 |
| Query / KV heads | 6 / 6 | 6 / 6 | 10 / 10 |
| MoE layers | 0 | 6 | 10 |
| Experts / selected per position | — | 8 / 2 | 8 / 2 |
| Per-expert hidden width | — | 1536 | 2560 |
| Total parameters | 286,261,730 | 371,233,250 | 1,289,852,146 |
| Active-parameter accounting | 286,261,730 | 286,298,594 | 896,636,146 |
| Pretrain updates | 1680 | 1680 | 6641 |
| Configured positions per update | 524288 | 524288 | 524288 |
| Configured pretrain positions | 880,803,840 | 880,803,840 | 3,481,796,608 |
| SFT updates | 931 | 932 | 932 |

The global update uses eight ranks; per-device batch/accumulation are 32/1, 16/2, and 8/4 respectively. Position budgets are schedule accounting, not unique corpus-token counts. Interrupted/failed trials and coarse data-loader resumption are not evidence of bitwise-identical data order.

### Data

- Pretraining: `karpathy/climbmix-400b-shuffle`, with fixed-source preparation in the original run. The dataset name does not mean this project trained on all 400 billion tokens.
- Tokenizer: RustBPE training with a 32768 vocabulary; the recorded run read about 2.0 billion characters from about 758,523 documents.
- SFT mixture: SmolTalk train + MMLU `auxiliary_train` repeated 3× + GSM8K train repeated 4×; the constructed mixture contains 789,759 conversations, shuffled with seed 42.
- SFT is full-parameter training from the Base checkpoint and optimizer state. It supervises assistant targets, with user/tool-result/padding targets masked out. Long conversations are truncated by the configured 2048-token rendering limit before packing.

## Evaluation stages

The final MoE20 run completed 31 stages:

| Stage family | Scope |
|---|---|
| CORE | 22 full tasks, 91,037 questions total |
| Chat | ARC-Easy, ARC-Challenge, MMLU, GSM8K, HumanEval |
| Other | Base BPB, Base samples, inference function check, inference benchmark |

The five chat task counts are 2376 / 1172 / 14042 / 1319 / 164. “31 stages” is not 31 independent capability datasets, and retained RL modules do not mean RL was run.

### CORE

For task accuracy $a_i$ and random baseline $r_i$:

$$s_i=\frac{a_i-r_i}{1-r_i},\qquad CORE=\frac1{22}\sum_i s_i.$$

CORE is an unweighted mean over chance-normalized tasks, not overall raw accuracy or a question-weighted mean. The upstream task metadata uses varying few-shot counts; the entire suite is not zero-shot. Multiple-choice/schematic tasks rank continuation likelihood, while relevant LM tasks evaluate exact token continuation matches. This is not a single free-generation grading protocol.

Full-task examples are distributed by rank and correctness is aggregated; the final reported counts are not a 500-example preview. Some Dense historical tasks retain accepted aggregate correctness recovery rather than per-question predictions; no paired confidence interval or invented prediction file is supplied.

### BPB

$$BPB=\frac{\text{total negative log likelihood in nats}}{\ln(2)\cdot\text{valid UTF-8 bytes}}.$$

Lower is better on the same text/protocol. Validation excludes the MoE auxiliary loss. Training-window BPB, full evaluation BPB, and masked SFT validation BPB have different sample/masking scopes; they should not be interchanged.

### Chat tasks

- ARC and MMLU choose among candidate answer letters from logits at the answer position. They are not unrestricted explanation-generation accuracy.
- GSM8K generates a response and extracts/compares the final numeric answer; format failures can be incorrect.
- HumanEval generates code and checks the supplied unit tests. The recorded single-candidate setting is not pass@10 or pass@100.
- Completion prompts omit the final reference assistant answer before generation. References are used for grading.

## Results

| Metric | Dense12 | MoE12 | MoE20-R5 |
|---|---:|---:|---:|
| CORE ↑ | 0.127071 | 0.128288 | 0.224274 |
| Base validation BPB ↓ | 0.869251 | 0.853670 | 0.746360 |
| SFT ARC-Easy | 36.32% | 37.71% | 54.34% |
| SFT ARC-Challenge | 31.40% | 31.91% | 44.11% |
| SFT MMLU | 31.53% | 31.69% | 34.50% |
| SFT GSM8K | 0.53% | 0.61% | 4.25% |
| SFT HumanEval | 9.15% | 7.93% | 13.41% |

Final MoE20 chat correctness counts: 1291/2376, 517/1172, 4844/14042, 56/1319, and 22/164 respectively.

The public [results directory](../results/) contains tables and aggregate data. Missing fields stay missing; a zero and an unavailable measurement are not substituted for each other.

## Inference workload

Measured on one RTX PRO 6000 Blackwell Server Edition, prompt 256, requested generation 64, temperature 0:

| Metric | Batch 1 | Batch 8 |
|---|---:|---:|
| Time to first token | 19.9 ms | 20.2 ms |
| Median decode step time | 17.25 ms | 17.54 ms |
| Aggregate decode throughput | 58.0 tok/s | 454.3 tok/s |

Batch 8 uses eight continuations of the same prompt, after one prefill plus KV replication. Its throughput is aggregate, not 454.3 tok/s per request. The first generated token comes from the last prefill logits; subsequent decode calls supply the remaining tokens. Timings include warmup and CUDA synchronization around measurement. These measurements are not an independent-request serving/QPS benchmark.

Dense inference used different hardware, so a three-model speed ranking is unsupported. This release does not infer MFU/MBU from an unverified sparse roofline or label all operations as a custom FlashAttention kernel.

## Limitations that affect interpretation

### Validation/test separation

The inherited SFT validation candidate mixture includes SmolTalk test, the first 5200 MMLU test examples, and the first 420 GSM8K test examples. MMLU/GSM8K therefore overlap final benchmark test sets. Actual monitoring used a smaller token budget and no-gradient evaluation; the final checkpoint was the completed epoch endpoint rather than a selected best-test checkpoint. Even so, the test set was **not entirely untouched**.

A strict follow-up should construct validation from training data, freeze model/repair decisions, then evaluate locked test data. The project also does not establish a complete benchmark-contamination audit of the pretraining corpus.

### Comparison design

- MoE12 vs. Dense12 has the same depth/width and scheduled positions, but model capacity and runtime details differ. Scores are mixed; HumanEval decreases in MoE12.
- MoE20 has greater depth/width, about 3.953× the scheduled pretraining positions, and the R5 intervention. Its score increase cannot isolate the effect of MoE, depth, width, or one repair step.
- There are no repeated-seed significance estimates or factorial repair ablations.
- Healthy final routing means experts are used in the observed diagnostics/task aggregates. It does not imply uniform routing on every input, named-domain expert specialization, or complete per-update diagnostic coverage.
- Results are mostly English benchmarks. They do not establish Chinese/domain capability, production readiness, or strong mathematical reasoning; GSM8K remains 4.25%.

These constraints are part of the experiment record and should accompany any reused plot or score.
