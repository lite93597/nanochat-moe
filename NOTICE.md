# Attribution and distribution

NanoChat MoE is an independent extension of [Andrej Karpathy's nanochat](https://github.com/karpathy/nanochat). It retains the original MIT license and copyright notice. The Dense baseline was prepared from upstream commit `92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`.

The Transformer framework, tokenizer and task interfaces, generation engine, and Muon/AdamW optimizer originate in nanochat. This project adds sparse expert FFNs, routing diagnostics, sparse optimizer handling, experiment tooling, and the documented routing recovery experiments. This repository is not an official nanochat release.

The repository distributes source code, aggregate experiment measurements, and figures. It does not redistribute the ClimbMix, SmolTalk, MMLU, GSM8K, ARC, CORE or HumanEval datasets, downloaded Python/CUDA runtimes, or trained checkpoint binaries. Obtain datasets through their original sources and follow their respective terms and licenses. The code's MIT license does not replace dataset or model distribution terms.

Historical R5 research code is preserved separately from portable examples. The recorded experiment used checkpoint-specific provenance and optimizer shards; the portable model-only upcycling example is not an exact reproduction of that recovery.
