# Architecture and implementation

[Back to README](../README.md) · [Experiments](experiments.md) · [Repair](routing-repair.md)

## What is inherited?

The Dense reference is a snapshot of [NanoChat](https://github.com/karpathy/nanochat). Its causal Transformer, RoPE, QK normalization, RMSNorm, Value Embedding, residual machinery, ReLU² FFN, tokenizer, Muon/AdamW design, task interfaces, and KV-cache engine remain the foundation.

The reported runs use full causal attention (`window_pattern=L`), context 2048, vocabulary 32768, and head dimension 128. Query and KV head counts are equal: these configurations use MHA, even though the code can express other configurations. Precision is explicit rather than automatic AMP throughout the entire network.

## Dense to MoE

The original FFN is

$$F(x)=W_2[\operatorname{ReLU}(W_1x)]^2,\quad D\to4D\to D.$$

Every second block replaces it with eight independently parameterized experts:

$$F_e(x)=W_{2,e}[\operatorname{ReLU}(W_{1,e}x)]^2,\quad D\to2D\to D.$$

The router computes FP32 logits and probabilities:

$$p(x)=\operatorname{softmax}(W_rx).$$

For selected experts $S(x)=\operatorname{TopK}(p,2)$,

$$g_e(x)=\frac{p_e(x)}{\sum_{j\in S(x)}p_j(x)},\qquad y(x)=\sum_{e\in S(x)}g_e(x)F_e(x).$$

For example, selected probabilities 0.4 and 0.2 become gates 2/3 and 1/3. Top-k indices are discrete; selected gates and expert computations still provide differentiable task paths. The auxiliary softmax term adds router gradients beyond the selected gate path.

In the recovered MoE20-R5 checkpoint, only the last MoE layer instead uses `softmax(Wx + b)`. The bias affects expert selection, normalized gates, and auxiliary probabilities. Evaluation freezes the bias; it does not fit routing to a benchmark during inference. See the [repair recipe](routing-repair.md).

### Dispatch

`[B,T,D]` is flattened to `[B*T,D]`. For each expert, the implementation gathers selected rows, runs its FFN, applies gates, and accumulates outputs into the original row positions with `index_add_`.

There is no capacity limit or token dropping. The reference path does not implement grouped GEMM, custom fused expert kernels, shared experts, expert parallelism, or cross-GPU token All-to-All. This makes the computation easy to inspect; it also leaves room for performance work.

## Parameter capacity versus computation

Ignoring biases:

| FFN component | Matrix parameters |
|---|---:|
| Dense, hidden width 4D | $8D^2$ |
| One expert, hidden width 2D | $4D^2$ |
| All eight experts | $32D^2$ |
| Two selected experts | $8D^2$ |

At the same depth/width, a replaced FFN stores about four times the original matrix capacity while keeping active expert matrix work approximately equal. Router, gather/scatter, dtype casts, and memory access still cost time. All expert weights remain resident on each GPU.

The model-level active-parameter count subtracts inactive expert matrices from total parameters. It includes shared embedding tables, even though one token looks up only particular rows. It is an accounting convention, not a count of weights multiplied per token or a wall-clock/FLOP measurement. Upstream Value Embedding tables are a substantial part of the shared parameters.

## Training objective

For valid training positions $N$, eight experts $E$, and two choices $K$, let $c_e$ be selection count and $P_e$ mean softmax probability:

$$f_e=\frac{c_e}{NK},\qquad P_e=\frac1N\sum_i p_{i,e},\qquad L_{aux}=E\sum_e f_eP_e.$$

The main training objective is

$$L=L_{CE}+0.01\cdot\frac1M\sum_{\ell=1}^{M}L_{aux,\ell}.$$

Under uniform assignment/probability, this auxiliary term is **1**, not 0. Selection frequencies are detached/discrete; softmax probabilities carry the auxiliary gradient. Scalar training loss includes the auxiliary term; validation/BPB and per-token loss use cross-entropy only.

SFT masks `targets == -1` for user text, tool outputs, and padding. Valid supervised positions determine training auxiliary/feedback diagnostics; these positions are not the same as all prompt tokens. The input still passes through the model even when it has no direct token-level CE target.

The CE implementation takes local valid-token means within microbatches, then averages accumulated/rank contributions. With unequal SFT valid-token counts, this is not exact global token-weighted CE. Global valid-token normalization is a possible follow-up, not an implemented claim.

## Precision and optimization

- Compute uses BF16 on the reported GPU node; selected sensitive calculations, including routing, use FP32.
- Embedding tables use compute dtype storage; major linear weights retain FP32 storage and are cast for forward computation.
- Attention/Dense/expert matrix groups use Muon; embeddings, output projection, scalar parameters, and router groups use AdamW. The router has an independent learning-rate group.
- Each GPU has the full model. Custom gradient collectives and sharded optimizer updates/state provide data parallelism; there is no `DistributedDataParallel` wrapper.
- A locally unused expert receives a zero-gradient contribution so all ranks enter the same collectives. After global reduction, a globally unused expert matrix does not receive a stale-momentum or decay-only update.
- Sparse expert groups use sequential synchronization/update/gather to limit temporary buffers. This does not imply the entire optimizer fits into a single fixed-size buffer.

Eight GPUs therefore increase data throughput and shard optimizer work; they do not divide the experts into one expert per GPU.

## Routing metrics: use the right denominator

The reporter aggregates raw counts/mass across ranks before normalizing. $N$ is a valid **forward-position count**, not unique corpus tokens. It can include repeated contexts or candidates; summing it across layers does not produce a training-token count.

| Metric | Definition | Sum over experts |
|---|---|---:|
| Assignment share | $c_e/(NK)$ | 1 |
| Token hit rate | $c_e/N$ | K = 2 |
| Top-1 share | first-choice count / N | 1 |
| Gate share | selected normalized gate mass / N | 1 |
| Mean router probability | softmax probability mass / N | 1 |
| Load CV | population std(selection counts) / mean(counts) | — |

Reports separate `prefill`, `decode`, and other forward passes. Padding and completed generation rows are excluded through valid-position masks. Training masks count supervised SFT positions; evaluation prompt masks count nonpadding prompt positions. For zero valid positions, ratios are undefined and should be interpreted as missing, not evidence of uniform/healthy routing.

An expert unused in a 14-token prompt is different evidence from an expert unused over sustained training windows. Load uniformity is not semantic specialization; this project does not assign experts to named domains.

## Inference

The engine prefills the prompt, stores per-layer K/V, then generates new tokens using that cache. KV caching avoids recomputing old K/V, but each new query still attends to historical cache entries. Experts route the current forward positions in both phases.

The batch-8 microbenchmark expands one prompt into eight continuations after one prefill. It measures aggregate decode throughput for this workload, not independent-request concurrency or continuous batching.

## Code reading order

1. [`gpt.py`](../bundle/moe/nanochat/gpt.py): expert/router math, parameter accounting, model config.
2. [`optim.py`](../bundle/moe/nanochat/optim.py): parameter groups and cross-rank updates.
3. [`moe_training_metrics.py`](../bundle/moe/nanochat/moe_training_metrics.py): whole-update diagnostics and guard interfaces.
4. [`moe_report.py`](../bundle/moe/nanochat/moe_report.py): normalized reports and phase statistics.
5. [`engine.py`](../bundle/moe/nanochat/engine.py): generation, masks, prefill/decode, KV cache.
