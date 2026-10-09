# Contributing

Thanks for improving NanoChat MoE. Small changes with clear measurements are especially useful.

## Good starting points

- Strict training/validation/test separation for SFT.
- Matched-budget, multi-seed Dense/MoE experiments.
- Simpler routing stabilization with independent quality gates.
- Grouped expert kernels, load-aware dispatch, or expert parallelism.
- Independent-request serving benchmarks and memory measurements.
- Documentation examples that make routing, gradients, and denominator conventions easier to understand.

## Before changing behavior

1. Explain the problem and expected observable result in an issue or pull request.
2. Keep an unmodified baseline and use a new model tag for a new configuration.
3. Add a focused test for a concrete computation or regression risk.
4. Run the small offline test suite from `bundle/moe`:

```bash
uv sync --extra cpu --group dev
NANOCHAT_DTYPE=float32 uv run --extra cpu --group dev pytest -q \
  tests/test_moe.py tests/test_moe_training_metrics.py \
  tests/test_moe_repair_diagnostics.py tests/test_infer_bench.py
```

GPU/distributed changes also need an actual short test on the affected hardware. CPU tests do not establish multi-rank convergence or GPU throughput.

## Reporting an experiment

Include commit/configuration, hardware, precision, tokenizer/data versions, batch and accumulation, scheduled updates/positions, evaluation protocol and full counts, routing denominators, runtime/memory scope, and all failures or missing fields that affect interpretation.

Keep Base and SFT results separate. CORE is a chance-centered task mean; BPB requires matched text/masks. Report aggregate versus per-sequence throughput correctly. Preserve the validation/test overlap note for the existing results, and do not label MoE20's joint configuration change as a causal MoE-only result.

## Repository boundaries

Keep training data, model binaries, caches, credentials, account/server information, and local operational logs out of Git. Public result tables should contain aggregate experimental measurements, not private machine paths or authentication details. Historical source-bound R5 checks should remain explicit; a portable new intervention needs its own source identity and acceptance criteria.

Respect upstream NanoChat's MIT notices and third-party dataset terms. Credit the upstream functionality you reuse.
