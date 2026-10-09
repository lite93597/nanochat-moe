"""
Inference benchmark: measure latency, throughput, memory and bandwidth
utilization of a trained checkpoint, sweeping over the decode batch size.

This is an "eval" in its own right: intelligence metrics (CORE etc.) say
nothing about what a model costs to run. Architecture choices (GQA, sliding
windows, ...) show up here, so improvements can be judged on both axes.

Background: inference has two very different regimes.
- Prefill processes the whole prompt in parallel: big matmuls, compute-bound.
- Decode generates one token at a time. Dense models can use an approximate
  weight-plus-KV bandwidth model. MoE traffic also depends on the selected
  experts, casting, dispatch and cache reuse; this benchmark does not estimate it.
Sweeping the batch size traces out the latency <-> throughput tradeoff curve
between these regimes. MBU (model bandwidth utilization) is the decode
counterpart of training MFU: achieved bytes/sec over the peak bandwidth of
the GPU. It measures how far the implementation is from the physical ceiling.

Output: a human-readable card and table, and then the very last line of stdout
is a single compact JSON document with all of the same data, so that scripts
can consume the benchmark without parsing the pretty formatting:

    result = json.loads(subprocess.run([...], capture_output=True, text=True).stdout.splitlines()[-1])

Examples:

    # benchmark a base model checkpoint on one GPU
    python -m scripts.infer_bench -i base -g d12

    # benchmark the SFT model, custom sweep
    python -m scripts.infer_bench -i sft --batch-sizes 1,4,16,64 --decode-tokens 512

    # machine-readable: grab the last line
    python -m scripts.infer_bench -i base -g d12 | tail -1 | jq .sweep
"""

import argparse
import json
import math
import time
import torch

from nanochat.common import compute_init, compute_cleanup, autodetect_device_type, get_peak_bandwidth, get_peak_flops
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine

# -----------------------------------------------------------------------------
# Measurement
# (the architecture-side cost accounting - FLOPs, KV cache bytes - lives on the
# GPT model itself: estimate_decode_flops, estimate_prefill_flops, kv_bytes_per_token, kv_read_bytes)

def weight_bytes(model):
    """Resident parameter storage, including every MoE expert."""
    return sum(p.numel() * p.element_size() for p in model.parameters())


def model_cost_card(model, peak_bw, total_vram, context_len):
    """Separate resident storage, active compute estimates and unsupported MoE traffic."""
    counts = model.num_scaling_params()
    moe_enabled = model.config.moe_num_experts > 0
    stored_bytes = weight_bytes(model)
    kv_store = model.kv_bytes_per_token()
    kv_read = model.kv_read_bytes(context_len)
    ceiling = None
    if not moe_enabled and math.isfinite(peak_bw) and peak_bw > 0:
        ceiling = round(peak_bw / (stored_bytes + kv_read), 1)
    return {
        "num_total_params": counts["total"],
        "num_active_params": counts["active_total"],
        "num_total_matmul_params": model.num_matmul_params(),
        "num_active_matmul_params": counts["active_matmul_params"],
        "weight_bytes": stored_bytes,
        "weight_bytes_scope": "resident parameter storage; not measured per-step memory traffic",
        "kv_bytes_per_token": kv_store,
        "kv_read_bytes_per_step": kv_read,
        "ceiling_bs1_tok_per_sec": ceiling,
        "max_full_context_rows": int((total_vram - stored_bytes) / (kv_store * model.config.sequence_len)),
        "max_full_context_rows_scope": "storage-only upper bound, excluding temporary copies, activations and allocator overhead",
        "memory_roofline_supported": not moe_enabled,
        "memory_roofline_note": ("MoE expert union, casting, dispatch and cache reuse are not measured; ceiling and MBU are unavailable"
                                 if moe_enabled else "approximate dense weight-plus-KV model"),
        "flops_estimate_scope": "active matmuls including router matmul, plus attention; excludes softmax/top-k routing, dispatch, casting and other elementwise operations",
        "active_params_scope": "shared parameters plus top-k experts per token; not an observed speedup",
        "moe": {
            "enabled": moe_enabled,
            "num_experts": model.config.moe_num_experts,
            "top_k": model.config.moe_top_k if moe_enabled else 0,
            "layers": [i for i, block in enumerate(model.transformer.h) if hasattr(block.mlp, "experts")],
            "expert_params": counts["moe_experts"],
            "active_expert_params_per_token": counts["active_experts"],
            "router_params": counts["moe_router"],
        },
    }


def utilization(work_per_second, peak):
    """Unknown hardware peaks cannot produce a meaningful utilization percentage."""
    return 100 * work_per_second / peak if math.isfinite(peak) and peak > 0 else None


def percent_text(value, precision=1):
    return f"{value:.{precision}f}" if value is not None else "N/A"

def bench_generate(engine, prompt_tokens, batch_size, decode_tokens, temperature):
    """Run one timed generation. Returns dict of measurements."""
    device = engine.model.get_device()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    generator = engine.generate(prompt_tokens, num_samples=batch_size,
                                max_tokens=decode_tokens, temperature=temperature)
    # The first next() runs the batch=1 prefill, the KV cache replication to
    # batch_size rows, and samples the first token: that is the TTFT.
    t_start = time.perf_counter()
    next(generator)
    torch.cuda.synchronize(device)
    ttft = time.perf_counter() - t_start

    # Every subsequent next() is one decode step for all rows.
    step_times = []
    while True:
        t0 = time.perf_counter()
        try:
            next(generator)
        except StopIteration:
            break
        torch.cuda.synchronize(device)
        step_times.append(time.perf_counter() - t0)

    peak_vram = torch.cuda.max_memory_allocated(device)
    return dict(ttft=ttft, step_times=step_times, peak_vram=peak_vram)

def build_prompt(tokenizer, num_tokens):
    """A natural-language prompt of exactly num_tokens tokens (random ids would
    do for speed, but a real prompt keeps argmax decoding from degenerating)."""
    paragraph = ("The history of science is the study of the development of science, "
                 "including both the natural and social sciences. Science is a body of "
                 "empirical, theoretical, and practical knowledge about the natural world. ")
    text = paragraph * (num_tokens // 10) # more than enough tokens
    tokens = tokenizer.encode(text, prepend="<|bos|>")
    assert len(tokens) >= num_tokens, "prompt text too short, increase the repetition"
    return tokens[:num_tokens]

# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Inference benchmark")
    parser.add_argument("-i", "--source", type=str, default="base", help="Checkpoint source: base|mid|sft")
    parser.add_argument("-g", "--model-tag", type=str, default=None, help="Model tag to load")
    parser.add_argument("-s", "--step", type=int, default=None, help="Step to load (default = last)")
    parser.add_argument("--prompt-tokens", type=int, default=2048, help="Prompt length for prefill")
    parser.add_argument("--decode-tokens", type=int, default=256, help="Tokens to generate per row")
    parser.add_argument("--batch-sizes", type=str, default="1,8,32,128", help="Comma-separated decode batch sizes")
    parser.add_argument("-t", "--temperature", type=float, default=0.0)
    args = parser.parse_args()

    device_type = autodetect_device_type()
    assert device_type == "cuda", "infer_bench currently assumes a CUDA GPU (for timing and VRAM measurement)"
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    assert ddp_world_size == 1, "infer_bench is a single GPU benchmark, run without torchrun"

    model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
    config = model.config
    engine = Engine(model, tokenizer)

    # Clamp the prompt so prompt + decode fits in the training context
    max_prompt = config.sequence_len - args.decode_tokens
    prompt_len = min(args.prompt_tokens, max_prompt)
    if prompt_len < args.prompt_tokens:
        print(f"note: clamping prompt to {prompt_len} tokens so prompt+decode fits sequence_len={config.sequence_len}")
    prompt_tokens = build_prompt(tokenizer, prompt_len)

    # ------------------------------------------------------------------------
    # Static card: inference cost implied by the architecture, before measuring
    device_name = torch.cuda.get_device_name(device)
    peak_bw = get_peak_bandwidth(device_name)
    peak_flops = get_peak_flops(device_name)
    total_vram = torch.cuda.get_device_properties(device).total_memory
    w_bytes = weight_bytes(model)
    num_params = sum(p.numel() for p in model.parameters())
    context_mid = prompt_len + args.decode_tokens // 2 # representative decode context
    card = model_cost_card(model, peak_bw, total_vram, context_mid)
    kv_store, kv_read = card["kv_bytes_per_token"], card["kv_read_bytes_per_step"]
    ceiling_bs1, max_rows = card["ceiling_bs1_tok_per_sec"], card["max_full_context_rows"]

    print("=" * 100)
    print(f"Model: {args.source} {meta.get('model_tag', '')} (step {meta['step']}) | "
          f"depth {config.n_layer}, dim {config.n_embd}, heads {config.n_head}, kv heads {config.n_kv_head} (GQA)")
    bandwidth_text = f"{peak_bw/1e12:.2f} TB/s" if math.isfinite(peak_bw) else "N/A"
    compute_text = f"{peak_flops/1e12:.0f} TFLOPS" if math.isfinite(peak_flops) else "N/A"
    print(f"GPU: {device_name} | peak bandwidth {bandwidth_text} | peak compute {compute_text} | VRAM {total_vram/2**30:.0f} GiB")
    if not (math.isfinite(peak_bw) and math.isfinite(peak_flops)):
        print("Unavailable hardware peak values are reported as null; corresponding utilization is N/A.")
    print("-" * 100)
    dtype_counts = {}
    for p in model.parameters():
        dtype_name = str(p.dtype).replace("torch.", "")
        dtype_counts[dtype_name] = dtype_counts.get(dtype_name, 0) + p.numel()
    param_dtypes = ", ".join(f"{n:,} {dtype_name}" for dtype_name, n in sorted(dtype_counts.items()))
    print(f"Parameters: {num_params:,} ({param_dtypes}) | weight bytes as stored: {w_bytes/2**20:.0f} MiB")
    if card["moe"]["enabled"]:
        print(f"MoE: {len(card['moe']['layers'])} layers, {card['moe']['num_experts']} experts, top-{card['moe']['top_k']} | "
              f"active parameters/token: {card['num_active_params']:,} | active matmul parameters/token: {card['num_active_matmul_params']:,}")
    print(f"FLOPs estimate: {card['flops_estimate_scope']}")
    print(f"KV cache: {kv_store:,} bytes/token stored | {kv_read:,} bytes read/step at context {context_mid} "
          f"(window pattern {config.window_pattern})")
    ceiling_text = f"{ceiling_bs1:,.0f} tok/s" if ceiling_bs1 is not None else "N/A"
    print(f"Theoretical decode ceiling at batch 1: {ceiling_text} | "
          f"max ~{max_rows:,} full-context rows in VRAM")
    print(f"Bandwidth estimate: {card['memory_roofline_note']}")
    print(f"KV rows estimate: {card['max_full_context_rows_scope']}")
    print("=" * 100)

    # Everything printed above also goes into the final JSON line for scripts
    payload = {
        "source": args.source,
        "step": meta["step"],
        "model_config": meta["model_config"],
        "gpu": device_name,
        # None (not Infinity) for unknown GPUs, so the last line stays valid JSON
        "peak_bandwidth_bytes_per_sec": peak_bw if peak_bw != float("inf") else None,
        "total_vram_bytes": total_vram,
        "num_params": num_params,
        "param_dtypes": dtype_counts,
        **card,
        "context_mid": context_mid,
        "peak_flops_per_sec": peak_flops if peak_flops != float("inf") else None,
        "decode_flops_per_token": model.estimate_decode_flops(context_mid),
        "prompt_tokens": prompt_len,
        "decode_tokens": args.decode_tokens,
        "temperature": args.temperature,
        "sweep": [],
    }

    # ------------------------------------------------------------------------
    # Prefill measurement: batch 1, a single decode step, so TTFT ~= prefill time.
    # Prefill is compute-bound, so MFU (not MBU) is its distance from the roofline.
    bench_generate(engine, prompt_tokens, 1, 2, args.temperature) # warmup
    prefill_result = bench_generate(engine, prompt_tokens, 1, 2, args.temperature)
    prefill_time = prefill_result["ttft"]
    prefill_mfu = utilization(model.estimate_prefill_flops(prompt_len) / prefill_time, peak_flops)
    prefill_tok_per_sec = prompt_len / prefill_time
    mfu_text = percent_text(prefill_mfu) + ("%" if prefill_mfu is not None else "")
    print(f"Prefill (batch 1, {prompt_len} tokens): {prefill_tok_per_sec:,.0f} tok/s | MFU {mfu_text}")
    payload["prefill"] = {
        "tok_per_sec": round(prefill_tok_per_sec, 1),
        "mfu_percent": round(prefill_mfu, 2) if prefill_mfu is not None else None,
        "time_sec": round(prefill_time, 6),
    }

    # ------------------------------------------------------------------------
    # Measured sweep. MoE MBU is unavailable; MFU uses active compute estimates.
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    header = f"{'batch':>6} {'TTFT ms':>9} {'TPOT ms':>9} {'tok/s':>10} {'MBU %':>7} {'MFU %':>7} {'VRAM GiB':>9} {'steps':>6}"
    print(header)
    print("-" * len(header))
    for batch_size in batch_sizes:
        # warmup (cublas autotune, allocator warm, attention kernels)
        bench_generate(engine, prompt_tokens, batch_size, 8, args.temperature)
        # timed run
        result = bench_generate(engine, prompt_tokens, batch_size, args.decode_tokens, args.temperature)
        step_times = result["step_times"]
        num_steps = len(step_times)
        if num_steps == 0:
            print(f"{batch_size:>6}  all rows terminated during warmup?! skipping")
            continue
        tpot = sorted(step_times)[num_steps // 2] # median decode step time
        tok_per_sec = batch_size * num_steps / sum(step_times)
        # MBU: bytes each decode step must move, over what the GPU can move
        bytes_per_step = w_bytes + batch_size * kv_read
        mbu = utilization(bytes_per_step / tpot, peak_bw) if card["memory_roofline_supported"] else None
        # MFU: FLOPs each decode step must do, over what the GPU can do
        flops_per_step = batch_size * model.estimate_decode_flops(context_mid)
        mfu = utilization(flops_per_step / tpot, peak_flops)
        vram_gib = result["peak_vram"] / 2**30
        note = "" if num_steps == args.decode_tokens - 1 else f" (early stop @ {num_steps})"
        print(f"{batch_size:>6} {result['ttft']*1e3:>9.1f} {tpot*1e3:>9.2f} {tok_per_sec:>10,.0f} "
              f"{percent_text(mbu):>7} {percent_text(mfu, 2):>7} {vram_gib:>9.2f} {num_steps:>6}{note}")
        payload["sweep"].append({
            "batch_size": batch_size,
            "ttft_sec": round(result["ttft"], 6), # microsecond resolution, plenty for wall clock
            "tpot_sec": round(tpot, 6),
            "tok_per_sec": round(tok_per_sec, 1),
            "mbu_percent": round(mbu, 2) if mbu is not None else None,
            "mfu_percent": round(mfu, 4) if mfu is not None else None,

            "peak_vram_bytes": result["peak_vram"],
            "decode_steps": num_steps,
        })

    # The last line of stdout is the machine-readable version of the whole run
    print("-" * len(header))
    print(json.dumps(payload))

    compute_cleanup()

if __name__ == "__main__":
    main()
