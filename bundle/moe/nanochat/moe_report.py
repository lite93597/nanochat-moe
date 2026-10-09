"""Human-readable and JSON summaries of MoE routing during evaluation."""

import json
import math
import os

import torch.distributed as dist

from nanochat.common import print0


def _summarize_counters(counters, top_k):
    tokens = int(counters["token_count"])
    counts = [int(value) for value in counters["expert_counts"]]
    top1 = [int(value) for value in counters["top1_counts"]]
    gates = [float(value) for value in counters["gate_weight_sums"]]
    probs = [float(value) for value in counters["router_prob_sums"]]
    assignments = sum(counts)
    assert assignments == tokens * top_k, "MoE routing counts do not match valid tokens and top-k"
    assert sum(top1) == tokens, "MoE top-1 counts do not match valid tokens"
    experts = [
        {
            "expert": index,
            "selections": count,
            "assignment_share": count / assignments if assignments else 0.0,
            "token_hit_rate": count / tokens if tokens else 0.0,
            "top1_count": top1[index],
            "top1_share": top1[index] / tokens if tokens else 0.0,
            "gate_share": gates[index] / tokens if tokens else 0.0,
            "mean_router_probability": probs[index] / tokens if tokens else 0.0,
        }
        for index, count in enumerate(counts)
    ]
    mean = assignments / len(counts) if counts else 0.0
    load_cv = math.sqrt(sum((count - mean) ** 2 for count in counts) / len(counts)) / mean if mean else 0.0
    return {
        "token_count": tokens,
        "assignment_count": assignments,
        "top_k": top_k,
        "load_cv": load_cv,
        "unused_experts": [index for index, count in enumerate(counts) if count == 0],
        "experts": experts,
    }


def collect_moe_routing_report(model, label, distributed=False):
    """Return a per-layer report; all ranks must call when distributed=True.

    A selection share divides expert hits by all top-k slots and sums to one.
    A token hit rate divides by valid tokens and sums to top-k.
    """
    if getattr(model.config, "moe_num_experts", 0) == 0:
        return None
    raw = model.get_moe_cumulative_stats(as_python=True, distributed=distributed)
    if not raw:
        raise RuntimeError("MoE routing collection was not started before evaluation")
    layers = []
    for layer_index, counters in sorted(raw.items(), key=lambda item: int(item[0])):
        layer = _summarize_counters(counters, model.config.moe_top_k)
        layer["layer"] = int(layer_index)
        layer["phases"] = {
            phase: _summarize_counters(phase_counters, model.config.moe_top_k)
            for phase, phase_counters in counters["phases"].items()
        }
        layers.append(layer)
    return {"label": label, "num_experts": model.config.moe_num_experts, "top_k": model.config.moe_top_k, "layers": layers}


def print_moe_routing_report(report):
    """Print the per-expert selection and top-1 shares on rank zero."""
    if report is None:
        return
    print0(f"MoE routing [{report['label']}]: expert share of top-{report['top_k']} selections / top-1 share")
    for layer in report["layers"]:
        shares = " ".join(
            f"E{expert['expert']} {expert['assignment_share']:.1%}/{expert['top1_share']:.1%}"
            for expert in layer["experts"]
        )
        print0(
            f"  layer {layer['layer']}: {layer['token_count']} valid tokens, "
            f"load CV {layer['load_cv']:.3f}, unused {len(layer['unused_experts'])} | {shares}"
        )
        phase_sizes = ", ".join(
            f"{name} {phase['token_count']}" for name, phase in layer["phases"].items()
            if phase["token_count"]
        )
        if phase_sizes:
            print0(f"    phases: {phase_sizes}")


def save_moe_routing_reports(path, reports, metadata=None):
    """Write machine-readable evaluation routing metrics from rank zero."""
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    payload = {"schema_version": 1, "metadata": {**(metadata or {}), "world_size": world_size}, "reports": reports}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print0(f"MoE routing JSON: {path}")


def add_last_forward_router_metrics(log_data, router_stats, prefix='train/moe'):
    """Add sampled expert usage to a training log without enabling accumulation."""
    for layer, stats in router_stats.items():
        counts = stats['expert_counts'].detach()
        total = counts.sum().clamp_min(1)
        shares = (counts.float() / total).tolist()
        probabilities = stats['mean_router_probs'].detach().tolist()
        log_data[f'{prefix}/layer_{layer}/unused_experts'] = int((counts == 0).sum().item())
        for expert, (share, probability) in enumerate(zip(shares, probabilities)):
            log_data[f'{prefix}/layer_{layer}/expert_{expert}_assignment_share'] = share
            log_data[f'{prefix}/layer_{layer}/expert_{expert}_mean_router_probability'] = probability
