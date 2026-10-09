# Results / 实验结果

This directory contains aggregate results and routing ratios from the Dense12, MoE12, and MoE20-R5 experiments. All CSV files are UTF-8 and can be downloaded directly.

本目录提供三个模型的实验结果、配置与专家路由统计。**预训练/SFT 权重尚未公开托管**。

## Quality / 能力结果

| Metric | Dense12 | MoE12 | MoE20-R5 |
|---|---:|---:|---:|
| CORE, centered unweighted mean | 0.127071 | 0.128288 | 0.224274 |
| Base validation BPB ↓ | 0.869251 | 0.853670 | 0.746360 |
| SFT ARC-Easy | 36.32% | 37.71% | 54.34% |
| SFT ARC-Challenge | 31.40% | 31.91% | 44.11% |
| SFT MMLU | 31.53% | 31.69% | 34.50% |
| SFT GSM8K | 0.53% | 0.61% | 4.25% |
| SFT HumanEval | 9.15% | 7.93% | 13.41% |

Each model completed all 22 CORE tasks: **91,037 questions**. The five chat tasks contain 2,376 / 1,172 / 14,042 / 1,319 / 164 examples, respectively. Chat accuracy is `correct / examples`; HumanEval is the single-candidate test pass fraction, not pass@10 or pass@100. CORE uses the upstream task-specific protocols, including different few-shot settings; it is not an all-zero-shot suite.

CORE is `(accuracy - random_baseline) / (1 - random_baseline)`, averaged equally across the 22 tasks. A CORE value of 0.224274 does **not** mean 22.4274% overall raw accuracy. BPB is negative log-likelihood divided by `ln(2) * valid UTF-8 bytes` and is measured at the base checkpoint. SFT monitoring BPB and formal base BPB use different targets/windows and should not be substituted for one another.

## Downloadable tables / 可下载表格

| File | Rows | Contents |
|---|---:|---|
| [model_context.csv](model_context.csv) | 3 | Architecture, training positions, intervention and measured inference hardware |
| [summary_metrics.csv](summary_metrics.csv) | 24 | CORE, base BPB and the five SFT task scores with units |
| [core_tasks.csv](core_tasks.csv) | 66 | Counts, raw accuracy, centered task score and evidence type |
| [chat_tasks.csv](chat_tasks.csv) | 15 | Full test counts and SFT accuracy |
| [bpb.csv](bpb.csv) | 6 | Train/validation base BPB |
| [inference_benchmark.csv](inference_benchmark.csv) | 5 | Single-GPU throughput, latency, VRAM and hardware |
| [routing_expert_ratios.csv](routing_expert_ratios.csv) | 15,872 | Every MoE layer, task/report object, phase and expert |
| [routing_layer_balance.csv](routing_layer_balance.csv) | 1,984 | Valid forward positions, load CV, minimum share and zero hits |
| [short_prefill_zero_hits.csv](short_prefill_zero_hits.csv) | 2 | Explicit measured zero hits on short nonempty prefills |

[summary.json](summary.json) is the compact machine-readable experiment summary. [moe20_model_spec.json](moe20_model_spec.json) records the final architecture and routing-control semantics. [checkpoint_identity.json](checkpoint_identity.json) records the evaluated checkpoint identities. [artifact_manifest.json](artifact_manifest.json) contains sizes and SHA-256 fingerprints of the tables and figures; it detects file changes, but is not a substitute for rerunning the experiments.

For 19 Dense CORE tasks, the integer correct count was uniquely recovered from the rounded aggregate accuracy and the known full denominator. Three tasks retained full predictions. `core_tasks.csv` records this distinction; recovered counts do not provide per-question predictions.

## Expert ratios / 路由指标

Let `N` be the number of included forward positions, `K=2`, and `c_e` the selection count for expert `e`.

- `assignment_share = c_e / (2N)`: sums to **1** over experts.
- `token_hit_rate = c_e / N`: sums to **2** over experts.
- `top1_share`: first-choice count divided by `N`; sums to 1 where recorded.
- `gate_share`: selected mixture weights summed and divided by `N`; frequency and mixture contribution differ.
- `mean_router_probability`: full softmax probability averaged over included positions.
- `load_cv`: population standard deviation of the eight selection counts divided by their mean.

Counts are aggregated before normalization. `world_size` and `aggregation` distinguish global eight-rank reports, single-rank inference, and repeated sampling reported from rank 0. There are 30 routing files and 31 report objects; not every report uses all eight ranks. `N` is not the number of unique corpus tokens or newly generated tokens, and must not be summed across the ten MoE layers as a training-token budget.

Training SFT statistics count supervised target positions. Evaluation statistics count valid non-padding forward positions. Their denominators have different meanings. `total`, `prefill`, `decode`, and `other` remain separate.

Empty `N=0` phases keep zero counts but leave undefined ratios, CV and zero-hit indicators blank. Blank cells mean **not measured / undefined / unavailable**, never a fabricated zero. Nonempty zero hits are retained. In MoE20-R5, the sample prefill has `N=60` and expert 4 is unused; the short inference prefill has `N=14` and experts 4/5 are unused. These short observations do not establish persistent collapse.

## Inference protocol / 推理口径

Measurements use one GPU, a 256-token prompt, 64 requested generated tokens, temperature 0, warmup and CUDA synchronization. The first generated token comes from prefill, leaving 63 timed decode steps. Batch 8 repeats the same prompt across eight continuation branches after cloning its prefill KV cache. Its throughput is aggregate tokens/second, not per-user throughput or an eight-distinct-prompt service benchmark.

MoE12 and MoE20 use RTX PRO 6000 Blackwell Server Edition; Dense12 uses RTX 4060 Laptop GPU. The table describes observed deployments and does not establish a three-model causal architecture speedup. MFU/MBU are left blank when the required sparse roofline assumptions are unavailable.

## Interpretation and limits / 结论边界

1. Dense12 and MoE12 have 12 layers, width 768, and 880,803,840 configured sequence positions. MoE20-R5 has 20 layers, width 1280, and 3,481,796,608 positions (about 3.953 times as many). It also includes a joint collapse-recovery intervention. Its improvement cannot be attributed solely to MoE, depth, or one repair action.
2. The recorded training positions are configuration counts, not unique corpus tokens. SFT finishes one epoch of the prepared mixture; the mixture itself repeats some datasets. The complete SFT supervised-token accumulation is unavailable, so it remains blank.
3. SFT monitoring follows the upstream validation pool, which includes parts of MMLU/GSM8K test sets also used by the final benchmarks. Those monitoring batches do not receive gradient updates, and no best checkpoint was selected or early stopping performed on those scores. Nevertheless, validation and final test are **not strictly independent**. A future controlled study should split validation from training data and lock the test set.
4. There is no complete pretraining/benchmark contamination audit, no repeated-seed confidence interval and no significance claim. Benchmarks are mainly English. The low absolute GSM8K score remains visible.
5. Eight GPUs use model replicas with distributed optimizer updates, **not expert parallelism**. Expert IDs are not aligned semantically between models. Task-dependent load imbalance and short zero hits are disclosed rather than replaced with an ideal uniform distribution.
6. The R5 recovery combines donor expert copying, targeted Muon-state reset, training-window router-intercept calibration, probability feedback and holding the last-layer router matrix after real optimizer updates. There is no separate ablation proving the contribution of each operation. The repair chart shows distinct measured diagnostic windows, not a continuous curve or controlled causal experiment.

## Data and licensing

Code licensing does not relicense third-party training datasets or benchmark text. The repository does not redistribute raw ClimbMix, SmolTalk, MMLU, GSM8K, ARC or HumanEval data. Obtain those datasets from their original publishers and follow their licenses/terms. The public CSVs contain aggregate measurements and routing counters rather than dataset examples, prompts or generated personal content.
