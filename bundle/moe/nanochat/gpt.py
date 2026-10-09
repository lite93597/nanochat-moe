"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration
- optional sparse top-k Mixture-of-Experts feed-forward layers
"""

from functools import partial
from dataclasses import dataclass
import hashlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW

# Our custom Flash Attention module that automatically uses FA3 when compatible and SDPA fallback otherwise
from nanochat.flash_attention import flash_attn

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (quarter context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "SSSL"
    # Set moe_num_experts=0 to retain the original dense architecture/checkpoints.
    moe_num_experts: int = 0
    moe_top_k: int = 2
    moe_hidden_mult: float = 2.0
    moe_every: int = 2
    moe_aux_loss_coef: float = 0.01
    moe_load_bias_rate: float = 0.0
    moe_load_bias_layer: int = -1
    moe_load_bias_mode: str = "selection_probability_sign"
    moe_load_bias_max_step: float = 0.05
    moe_router_hold_layer: int = -1


def moe_load_bias_recipe(config):
    intercept = config.moe_load_bias_mode == "router_logit_probability_proportional"
    recipe = {"moe_load_bias_rate": config.moe_load_bias_rate,
            "moe_load_bias_layer": config.moe_load_bias_layer,
            "moe_load_bias_mode": config.moe_load_bias_mode,
            "moe_load_bias_max_step": config.moe_load_bias_max_step,
            "selection": "router_logits_plus_bias_softmax" if intercept else "router_probs_plus_bias",
            "gate": "biased_router_probs_renormalized" if intercept else "unbiased_router_probs_renormalized",
            "feedback": ("global_optimizer_step_probability_proportional_clipped_centered" if intercept
                         else "global_optimizer_step_sign_centered"),
            "aux": "biased_router_probs_switch_0.01" if intercept else "unbiased_router_probs_switch_0.01",
            "eval": "frozen"}
    if config.moe_router_hold_layer >= 0:
        recipe['moe_router_hold_layer'] = config.moe_router_hold_layer
    return recipe


def norm(x):
    return F.rms_norm(x, (x.size(-1),)) # note that this will run in bf16, seems ok

class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.
    Replaces autocast: master weights stay fp32 for optimizer precision,
    but matmuls run in the activation dtype (typically bf16 from embeddings)."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
    # note: this rotates by -theta, the transpose of the textbook convention. Functionally
    # equivalent (only the relative q/k rotation matters), kept for checkpoint compatibility.
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 12
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
            v = v + gate.unsqueeze(-1) * ve

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k) # QK norm
        q = q * 1.2  # sharper attention (split scale between Q and K), TODO think through better
        k = k * 1.2

        # Flash Attention (FA3 or SDPA fallback)
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        if kv_cache is None:
            # Training: causal attention with optional sliding window
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            # Inference: use flash_attn_with_kvcache which handles cache management
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            # Advance position after last layer processes
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        # Re-assemble the heads and project back to residual stream
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


MOE_STATS_PHASES = ("prefill", "decode", "other")
MOE_STATS_COUNTERS = ("token_count", "expert_counts", "top1_counts", "gate_weight_sums", "router_prob_sums")


def _new_moe_counters(num_experts, device):
    return {
        "token_count": torch.zeros((), dtype=torch.long, device=device),
        "expert_counts": torch.zeros(num_experts, dtype=torch.long, device=device),
        "top1_counts": torch.zeros(num_experts, dtype=torch.long, device=device),
        "gate_weight_sums": torch.zeros(num_experts, dtype=torch.float64, device=device),
        "router_prob_sums": torch.zeros(num_experts, dtype=torch.float64, device=device),
    }


def _summarize_moe_counters(counters, top_k, as_python=False):
    """Convert raw counters to useful rates, with finite values for zero tokens."""
    token_count = counters["token_count"]
    expert_counts = counters["expert_counts"]
    top1_counts = counters["top1_counts"]
    gate_sums = counters["gate_weight_sums"]
    router_sums = counters["router_prob_sums"]
    safe_tokens = token_count.clamp_min(1)
    assignment_share = expert_counts.double() / (safe_tokens * top_k)
    top1_share = top1_counts.double() / safe_tokens
    gate_share = gate_sums / safe_tokens
    mean_router_probs = router_sums / safe_tokens
    counts_f = expert_counts.double()
    load_cv = counts_f.std(unbiased=False) / counts_f.mean().clamp_min(1e-12)
    unused_experts = torch.where(expert_counts == 0)[0]
    if as_python:
        return {
            "token_count": int(token_count.item()),
            "expert_counts": expert_counts.tolist(),
            "top1_counts": top1_counts.tolist(),
            "gate_weight_sums": gate_sums.tolist(),
            "router_prob_sums": router_sums.tolist(),
            "mean_router_probs": mean_router_probs.tolist(),
            "assignment_share": assignment_share.tolist(),
            "top1_share": top1_share.tolist(),
            "gate_share": gate_share.tolist(),
            "load_cv": float(load_cv.item()),
            "unused_experts": unused_experts.tolist(),
        }
    return {
        **counters,
        "mean_router_probs": mean_router_probs,
        "assignment_share": assignment_share,
        "top1_share": top1_share,
        "gate_share": gate_share,
        "load_cv": load_cv,
        "unused_experts": unused_experts,
    }


def all_reduce_moe_stats_counters(layer_counters, group=None):
    """Return globally summed counters; all ranks must call this in layer order.

    This is separate from forward so ordinary inference never introduces NCCL
    collectives. The caller should start tracking on every rank before reducing.
    """
    reduced = {
        layer_idx: {phase: {name: value.detach().clone() for name, value in counters.items()}
                    for phase, counters in phases.items()}
        for layer_idx, phases in layer_counters.items()
    }
    if not reduced or not (dist.is_available() and dist.is_initialized()):
        return reduced
    backend = str(dist.get_backend(group)).lower()
    reduce_device = torch.device("cuda", torch.cuda.current_device()) if "nccl" in backend else torch.device("cpu")
    # Pack all layers and phases into two tensors so evaluation uses only two
    # collectives, rather than one tiny collective per counter.
    for names, dtype in (
        (("token_count", "expert_counts", "top1_counts"), torch.long),
        (("gate_weight_sums", "router_prob_sums"), torch.float64),
    ):
        layout = [
            (layer_idx, phase, name, reduced[layer_idx][phase][name].shape)
            for layer_idx in sorted(reduced)
            for phase in ("total", *MOE_STATS_PHASES)
            for name in names
        ]
        packed = torch.cat([
            reduced[layer_idx][phase][name].reshape(-1).to(device=reduce_device, dtype=dtype)
            for layer_idx, phase, name, _ in layout
        ])
        dist.all_reduce(packed, op=dist.ReduceOp.SUM, group=group)
        offset = 0
        for layer_idx, phase, name, shape in layout:
            size = reduced[layer_idx][phase][name].numel()
            reduced[layer_idx][phase][name] = packed[offset:offset + size].reshape(shape).clone()
            offset += size
    return reduced


class MoE(nn.Module):
    """Token-wise sparse top-k feed-forward layer.

    Each selected expert sees only its assigned tokens. The router and balancing
    loss use fp32, while expert matmuls use the activation compute dtype.
    """
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.num_experts = config.moe_num_experts
        self.top_k = config.moe_top_k
        self.load_bias_rate = config.moe_load_bias_rate if layer_idx == config.moe_load_bias_layer else 0.0
        self.load_bias_layer = layer_idx
        self.load_bias_mode = config.moe_load_bias_mode
        self.load_bias_max_step = config.moe_load_bias_max_step
        self._load_feedback_active = False
        if self.load_bias_rate > 0:
            self.register_buffer("selection_bias", torch.zeros(self.num_experts, dtype=torch.float32))
            self.register_buffer("_load_counts", torch.zeros(self.num_experts, dtype=torch.long), persistent=False)
            self.register_buffer("_load_tokens", torch.zeros((), dtype=torch.long), persistent=False)
            self.register_buffer("_load_invalid", torch.zeros((), dtype=torch.long), persistent=False)
            if self.load_bias_mode == "router_logit_probability_proportional":
                self.register_buffer("_load_prob_sums", torch.zeros(self.num_experts, dtype=torch.float64), persistent=False)
        hidden_dim = max(1, round(config.n_embd * config.moe_hidden_mult))
        self.router = Linear(config.n_embd, self.num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.ModuleDict({
                "c_fc": Linear(config.n_embd, hidden_dim, bias=False),
                "c_proj": Linear(hidden_dim, config.n_embd, bias=False),
            }) for _ in range(self.num_experts)
        ])
        self.last_router_stats = None
        self._moe_stats_enabled = False
        self._cumulative_stats = None
        self._router_diagnostics_enabled = False
        self.last_router_diagnostics = None

    @torch.no_grad()
    def reset_load_bias_feedback(self):
        if self.load_bias_rate <= 0:
            return
        self._load_counts.zero_()
        self._load_tokens.zero_()
        self._load_invalid.zero_()
        if self.load_bias_mode == "router_logit_probability_proportional":
            self._load_prob_sums.zero_()
        self._load_feedback_active = self.training

    @torch.no_grad()
    def update_load_bias_feedback(self):
        if self.load_bias_rate <= 0 or not self.training or not self._load_feedback_active:
            return None
        distributed = dist.is_available() and dist.is_initialized()
        world = dist.get_world_size() if distributed else 1
        before = self.selection_bias.detach().clone()
        invalid = self._load_invalid + (~torch.isfinite(before).all()).long()
        # One packed collective covers all valid tokens/microbatches, finite
        # flags, and rank consistency of the persistent controller state.
        packet = torch.cat((self._load_counts.double(), self._load_tokens.double().reshape(1),
                            invalid.double().reshape(1), before.double(), before.double().square()))
        intercept = self.load_bias_mode == "router_logit_probability_proportional"
        if intercept:
            packet = torch.cat((packet, self._load_prob_sums))
        if distributed:
            dist.all_reduce(packet, op=dist.ReduceOp.SUM)
        values = packet.cpu().tolist()
        e = self.num_experts
        counts, tokens, invalid = values[:e], values[e], values[e + 1]
        mean = torch.tensor(values[e + 2:e + 2 + e], dtype=torch.float64) / world
        second = torch.tensor(values[e + 2 + e:e + 2 + 2 * e], dtype=torch.float64) / world
        variance = (second - mean.square()).abs().max().item()
        if (invalid or not all(math.isfinite(value) for value in values)
                or variance > 1e-12 or (before.double().cpu() - mean).abs().max().item() > 1e-7
                or tokens < 0 or tokens != int(tokens)
                or any(value < 0 or value != int(value) for value in counts)
                or sum(counts) != tokens * self.top_k):
            raise RuntimeError("Invalid or rank-inconsistent complete-update load-bias feedback")
        target = tokens * self.top_k / e
        probability_sums = values[e + 2 + 2 * e:] if intercept else None
        if intercept and (any(value < 0 for value in probability_sums)
                          or abs(sum(probability_sums) - tokens) > max(1e-5, tokens * 1e-6)):
            raise RuntimeError("Invalid global router-probability sums for complete-update feedback")
        if tokens:
            if intercept:
                mean_probs = packet[-e:] / tokens
                delta = (self.load_bias_rate * (1 / e - mean_probs)).clamp(
                    -self.load_bias_max_step, self.load_bias_max_step).to(torch.float32)
            else:
                delta = torch.sign(target - packet[:e]).to(torch.float32) * self.load_bias_rate
            self.selection_bias.add_(delta)
            self.selection_bias.sub_(self.selection_bias.mean())
        self._load_feedback_active = False
        after = self.selection_bias.detach().cpu().tolist()
        if not all(math.isfinite(value) for value in after):
            raise RuntimeError("Nonfinite persistent selection bias after feedback")
        report = {"layer": self.load_bias_layer, "rate": self.load_bias_rate,
            "world_size": world, "global_valid_tokens": int(tokens),
            "global_expert_counts": [int(value) for value in counts], "target_count": target,
            "bias_before": before.cpu().tolist(), "bias_after": after,
            "updated": tokens > 0, "all_ranks_bias_consistent": True,
            "aggregation": "all ranks and all gradient accumulation microbatches in this optimizer step",
            "mask": "targets != -1", "selection": ("router_logits_plus_bias_softmax" if intercept else "router_probs_plus_bias"),
            "gate": ("biased_router_probs_renormalized" if intercept else "unbiased_router_probs_renormalized"), "eval": "frozen"}
        if intercept:
            report.update(mode=self.load_bias_mode, max_step=self.load_bias_max_step,
                global_probability_sums=probability_sums,
                global_mean_router_probs=[value / tokens if tokens else 0. for value in probability_sums],
                target_probability=1 / e,
                feedback="global_optimizer_step_probability_proportional_clipped_centered")
        return report

    def reset_stats(self):
        device = self.router.weight.device
        self._cumulative_stats = {
            phase: _new_moe_counters(self.num_experts, device)
            for phase in ("total", *MOE_STATS_PHASES)
        }

    @torch.no_grad()
    def _record_stats(self, top_indices, top_probs, router_probs, valid_mask, phase):
        if not self._moe_stats_enabled:
            return
        if self._cumulative_stats is None:
            self.reset_stats()
        if self._cumulative_stats["total"]["expert_counts"].device != top_indices.device:
            self._cumulative_stats = {
                key: {name: value.to(top_indices.device) for name, value in counters.items()}
                for key, counters in self._cumulative_stats.items()
            }
        if valid_mask is not None:
            valid = valid_mask.reshape(-1)
            selected_indices = top_indices[valid]
            selected_probs = top_probs[valid]
            selected_router_probs = router_probs[valid]
        else:
            selected_indices = top_indices
            selected_probs = top_probs
            selected_router_probs = router_probs
        n_tokens = selected_indices.shape[0]
        counts = torch.bincount(selected_indices.reshape(-1), minlength=self.num_experts)
        top1 = torch.bincount(selected_indices[:, 0], minlength=self.num_experts)
        gate_sums = torch.bincount(
            selected_indices.reshape(-1), weights=selected_probs.double().reshape(-1),
            minlength=self.num_experts,
        )
        router_sums = selected_router_probs.double().sum(dim=0)
        for key in ("total", phase):
            counters = self._cumulative_stats[key]
            counters["token_count"].add_(n_tokens)
            counters["expert_counts"].add_(counts)
            counters["top1_counts"].add_(top1)
            counters["gate_weight_sums"].add_(gate_sums)
            counters["router_prob_sums"].add_(router_sums)

    def forward(self, x, valid_token_mask=None, stats_token_mask=None, stats_phase="other"):
        original_shape = x.shape
        tokens = x.reshape(-1, original_shape[-1])
        if tokens.shape[0] == 0:
            zero = self.router.weight.sum() * 0.0
            if self._moe_stats_enabled:
                self._record_stats(
                    torch.empty((0, self.top_k), dtype=torch.long, device=x.device),
                    torch.empty((0, self.top_k), dtype=torch.float32, device=x.device),
                    torch.empty((0, self.num_experts), dtype=torch.float32, device=x.device),
                    None, stats_phase,
                )
            self.last_router_stats = {
                "expert_counts": torch.zeros(self.num_experts, dtype=torch.long, device=x.device),
                "mean_router_probs": torch.zeros(self.num_experts, dtype=torch.float32, device=x.device),
                "aux_loss": zero.detach(),
                "token_count": torch.zeros((), dtype=torch.long, device=x.device),
            }
            return torch.zeros_like(x), zero

        router_logits = self.router(tokens.float())
        intercept = self.load_bias_rate > 0 and self.load_bias_mode == "router_logit_probability_proportional"
        if intercept:
            router_logits = router_logits + self.selection_bias
        router_probs = F.softmax(router_logits, dim=-1)
        if self.load_bias_rate > 0:
            selection_scores = router_probs if intercept else router_probs + self.selection_bias
            _, top_indices = torch.topk(selection_scores, self.top_k, dim=-1)
            # Normalize the selected router logits stably. R2 keeps bias only
            # in selection; R3 explicitly uses the calibrated router intercept
            # in selection, fusion weights, and the original balancing loss.
            top_probs = F.softmax(router_logits.gather(-1, top_indices).float(), dim=-1)
            if self.training and self._load_feedback_active:
                with torch.no_grad():
                    selected = top_indices if valid_token_mask is None else top_indices[valid_token_mask.reshape(-1)]
                    self._load_counts.add_(torch.bincount(selected.reshape(-1), minlength=self.num_experts))
                    self._load_tokens.add_(selected.shape[0])
                    self._load_invalid.add_((~torch.isfinite(router_probs).all()).long())
                    if intercept:
                        valid_probs = router_probs if valid_token_mask is None else router_probs[valid_token_mask.reshape(-1)]
                        self._load_prob_sums.add_(valid_probs.double().sum(0))
        else:
            top_probs, top_indices = torch.topk(router_probs, self.top_k, dim=-1)
            top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)

        result = torch.zeros_like(tokens)
        for expert_idx, expert in enumerate(self.experts):
            token_idx, choice_idx = torch.where(top_indices == expert_idx)
            selected = tokens.index_select(0, token_idx)
            hidden = F.relu(expert["c_fc"](selected)).square()
            expert_output = expert["c_proj"](hidden)
            weight = top_probs[token_idx, choice_idx].to(expert_output.dtype).unsqueeze(-1)
            result.index_add_(0, token_idx, expert_output * weight)

        # Switch-style balancing: uniform routing gives 1.0. Assignment fractions
        # are nondifferentiable; mean router probabilities carry the gradient.
        def routing_marginals(mask):
            if mask is None:
                counts = torch.bincount(top_indices.reshape(-1), minlength=self.num_experts)
                count = torch.tensor(tokens.shape[0], dtype=torch.long, device=x.device)
                mean_probs = router_probs.mean(dim=0)
            else:
                valid = mask.reshape(-1)
                counts = torch.bincount(top_indices[valid].reshape(-1), minlength=self.num_experts)
                count = valid.sum()
                mean_probs = (router_probs * valid.unsqueeze(-1)).sum(dim=0) / count.clamp_min(1)
            return counts, count, mean_probs

        aux_counts, aux_count, aux_mean_probs = routing_marginals(valid_token_mask)
        aux_fraction = aux_counts.float() / (aux_count.clamp_min(1) * self.top_k)
        aux_loss = self.num_experts * torch.sum(aux_mean_probs * aux_fraction)
        stats_mask = valid_token_mask if stats_token_mask is None else stats_token_mask
        if self._router_diagnostics_enabled:
            valid = stats_mask.reshape(-1) if stats_mask is not None else None
            logits = router_logits.detach() if valid is None else router_logits.detach()[valid]
            probs = router_probs.detach() if valid is None else router_probs.detach()[valid]
            self.last_router_diagnostics = {
                "logit_min": logits.amin() if logits.numel() else logits.new_zeros(()),
                "logit_max": logits.amax() if logits.numel() else logits.new_zeros(()),
                "prob_min": probs.amin() if probs.numel() else probs.new_zeros(()),
                "prob_max": probs.amax() if probs.numel() else probs.new_zeros(()),
                "logits_finite": torch.isfinite(logits).all(),
                "probabilities_finite": torch.isfinite(probs).all(),
            }
        if stats_mask is valid_token_mask:
            stats_counts, stats_count, stats_mean_probs = aux_counts, aux_count, aux_mean_probs
        else:
            stats_counts, stats_count, stats_mean_probs = routing_marginals(stats_mask)
        self.last_router_stats = {
            "expert_counts": stats_counts.detach(),
            "mean_router_probs": stats_mean_probs.detach(),
            "aux_loss": aux_loss.detach(),
            "token_count": stats_count.detach(),
        }
        if self._moe_stats_enabled:
            self._record_stats(top_indices, top_probs, router_probs, stats_mask, stats_phase)
        return result.reshape(original_shape), aux_loss


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        use_moe = config.moe_num_experts > 0 and (
            (layer_idx + 1) % config.moe_every == 0 or
            (config.n_layer < config.moe_every and layer_idx == config.n_layer - 1)
        )
        self.mlp = MoE(config, layer_idx) if use_moe else MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache, valid_token_mask=None,
                stats_token_mask=None, stats_phase="other"):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        if isinstance(self.mlp, MoE):
            mlp_out, aux_loss = self.mlp(norm(x), valid_token_mask, stats_token_mask, stats_phase)
        else:
            mlp_out, aux_loss = self.mlp(norm(x)), None
        x = x + mlp_out
        return x, aux_loss


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        if config.moe_num_experts < 0:
            raise ValueError("moe_num_experts must be nonnegative")
        if config.moe_num_experts > 0:
            if not (1 <= config.moe_top_k <= config.moe_num_experts):
                raise ValueError("moe_top_k must be between 1 and moe_num_experts")
            if config.moe_hidden_mult <= 0 or config.moe_every <= 0 or config.moe_aux_loss_coef < 0:
                raise ValueError("moe_hidden_mult and moe_every must be positive; moe_aux_loss_coef must be nonnegative")
        if not math.isfinite(config.moe_load_bias_rate) or config.moe_load_bias_rate < 0:
            raise ValueError("moe_load_bias_rate must be finite and nonnegative")
        if config.moe_load_bias_mode not in ("selection_probability_sign", "router_logit_probability_proportional"):
            raise ValueError("Unknown persistent router-bias mode")
        if not math.isfinite(config.moe_load_bias_max_step) or config.moe_load_bias_max_step <= 0:
            raise ValueError("moe_load_bias_max_step must be finite and positive")
        if type(config.moe_router_hold_layer) is not int or config.moe_router_hold_layer not in (-1, 19):
            raise ValueError("Router holding is restricted to the explicit last MoE20 layer19")
        if config.moe_router_hold_layer == 19 and not (config.n_layer == 20 and
                config.moe_load_bias_layer == 19 and config.moe_load_bias_rate == .5 and
                config.moe_load_bias_mode == 'router_logit_probability_proportional' and
                config.moe_load_bias_max_step == .05):
            raise ValueError("Held router requires the explicit R3 calibrated intercept/controller")
        if config.moe_load_bias_rate > 0 and not (config.moe_num_experts > 0 and
                0 <= config.moe_load_bias_layer < config.n_layer and
                ((config.moe_load_bias_layer + 1) % config.moe_every == 0 or config.n_layer < config.moe_every)):
            raise ValueError("Enabled load bias must identify an actual MoE layer")
        self.config = config
        self.last_moe_aux_loss = None
        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        self.window_sizes = self._compute_window_sizes(config)
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        # Separate parameters so they can have different optimizer treatment
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # fake init, real init in init_weights()
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))     # fake init, real init in init_weights()
        # Smear: mix previous token's embedding into current token (cheap bigram-like info)
        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        # Backout: subtract cached mid-layer residual before final norm to remove low-level features
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
        # Value embeddings (ResFormer-style): alternating layers, last layer always included
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """

        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s) # weights use Uniform to avoid outliers
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight) # projections are zero
            if isinstance(block.mlp, MoE):
                torch.nn.init.normal_(block.mlp.router.weight, mean=0.0, std=0.01)
                if block.mlp.load_bias_rate > 0:
                    block.mlp.selection_bias.zero_()
                    block.mlp.reset_load_bias_feedback()
                for expert in block.mlp.experts:
                    torch.nn.init.uniform_(expert["c_fc"].weight, -s * 0.4, s * 0.4)
                    torch.nn.init.zeros_(expert["c_proj"].weight)
            else:
                torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)  # 0.4x init scale for c_fc
                torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # Per-layer scalars
        # Per-layer resid init: stronger residual at early layers, weaker at deep layers
        n_layer = self.config.n_layer
        for i in range(n_layer):
            self.resid_lambdas.data[i] = 1.15 - (0.10 * i / max(n_layer - 1, 1))
        # Decaying x0 init: earlier layers get more input embedding blending
        for i in range(n_layer):
            self.x0_lambdas.data[i] = 0.20 - (0.15 * i / max(n_layer - 1, 1))

        # Smear/backout scalars and smear gate must be explicitly initialized 
        torch.nn.init.zeros_(self.smear_lambda)
        torch.nn.init.constant_(self.backout_lambda, 0.2)
        torch.nn.init.uniform_(self.smear_gate.weight, 0.0, 0.02)

        # Value embeddings (init like c_v: uniform with same std)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # Gate weights init with small positive values so gates start slightly above neutral
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to COMPUTE_DTYPE: optimizer can tolerate reduced-precision
        # embeddings and it saves memory. Exception: fp16 requires fp32 embeddings
        # because GradScaler cannot unscale fp16 gradients.
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)
            for ve in self.value_embeds.values():
                ve.to(dtype=COMPUTE_DTYPE)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def _compute_window_sizes(self, config):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (quarter context)
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        # Map characters to window sizes
        long_window = config.sequence_len
        short_window = -(-long_window // 4 // 128) * 128  # ceil to FA3 tile size (2048 -> 768)
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
        With sliding windows, effective_seq_len varies per layer (capped by window size).
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        """
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # Sum attention FLOPs per layer, accounting for sliding window
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]  # (left, right) tuple, we use left
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * self.num_active_matmul_params() + attn_flops
        return num_flops_per_token

    def num_matmul_params(self):
        """
        The number of parameters that participate in matmuls with the token stream,
        i.e. contribute 2 FLOPs/param to the forward pass. Counted structurally: every
        matmul in this model goes through the Linear class, while non-matmul params
        (embeddings = lookups, per-layer scalars) are nn.Embedding or raw Parameters.
        """
        matmul_params = sum(m.weight.numel() for m in self.modules() if isinstance(m, Linear))
        return matmul_params

    def reset_moe_load_bias_feedback(self):
        for block in self.transformer.h:
            if isinstance(block.mlp, MoE):
                block.mlp.reset_load_bias_feedback()

    def update_moe_load_bias_feedback(self):
        reports = [block.mlp.update_load_bias_feedback() for block in self.transformer.h
                   if isinstance(block.mlp, MoE)]
        return [report for report in reports if report is not None]

    @staticmethod
    def _router_tensor_sha256(tensor):
        return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

    @torch.no_grad()
    def capture_moe_router_hold(self, optimizer, diagnostics=False):
        """Keep the W snapshot; leave real gradients and optimizer mapping intact."""
        if self.config.moe_router_hold_layer < 0:
            return None
        parameter = self.transformer.h[19].mlp.router.weight
        groups = [group for group in optimizer.param_groups if any(p is parameter for p in group['params'])]
        if len(groups) != 1 or groups[0].get('kind') != 'adamw':
            raise RuntimeError("Held W is not in its original unique AdamW group")
        if parameter.dtype != torch.float32 or not parameter.requires_grad or parameter.grad is None:
            raise RuntimeError("Held W must retain FP32 master weights and real autograd gradients")
        state = optimizer.state[parameter]
        if state and set(state) != {'step', 'exp_avg', 'exp_avg_sq'}:
            raise RuntimeError("Unexpected original AdamW state for held W")
        if type(state.get('step', 0)) is not int:
            raise RuntimeError("Original held-router AdamW clock must be an actual integer")
        snapshot = {'weight': parameter.detach().clone(), 'step': int(state.get('step', 0)),
                    'group_initial_lr': groups[0]['initial_lr'], 'group_lr': groups[0]['lr']}
        if diagnostics:
            snapshot['weight_sha256'] = self._router_tensor_sha256(parameter)
            snapshot['moments_before_sha256'] = {key: self._router_tensor_sha256(state[key]) if state else None
                                                 for key in ('exp_avg', 'exp_avg_sq')}
        return snapshot

    @torch.no_grad()
    def restore_moe_router_hold(self, optimizer, snapshot, optimizer_updated=True, diagnostics=False):
        """Restore only W after the unchanged optimizer completes its gathers.

        AdamW moments use the actual gradient and its clock advances normally.
        This does not reset moments, remove a parameter, or alter a group/LR.
        """
        if snapshot is None:
            return None
        parameter = self.transformer.h[19].mlp.router.weight
        parameter.copy_(snapshot['weight'])
        state = optimizer.state[parameter]
        if type(state.get('step', 0)) is not int:
            raise RuntimeError("Held-router AdamW clock changed its original integer representation")
        step_after = int(state.get('step', 0))
        expected_step = snapshot['step'] + int(optimizer_updated)
        if step_after != expected_step or not torch.equal(parameter, snapshot['weight']):
            raise RuntimeError("Held W restore or original AdamW clock advancement failed")
        report = {'layer': 19, 'parameter_name': 'transformer.h.19.mlp.router.weight',
            'policy': 'restore_after_real_optimizer_step',
            'optimizer_state_policy': 'real_gradients_moments_and_step_advance',
            'optimizer_updated': bool(optimizer_updated), 'optimizer_kind': 'adamw',
            'step_before': snapshot['step'], 'step_after': step_after,
            'weight_exactly_equal': True, 'weight_max_delta': 0.,
            'group_initial_lr': snapshot['group_initial_lr'], 'group_lr': snapshot['group_lr']}
        if diagnostics:
            distributed = dist.is_available() and dist.is_initialized()
            world = dist.get_world_size() if distributed else 1
            local = {'rank': dist.get_rank() if distributed else 0,
                'weight_before_sha256': snapshot['weight_sha256'],
                'weight_after_sha256': self._router_tensor_sha256(parameter),
                'weight_exactly_equal': True, 'weight_max_delta': 0.,
                'step_before': snapshot['step'], 'step_after': step_after,
                'moments_before_sha256': snapshot['moments_before_sha256'],
                'moments_after_sha256': {key: self._router_tensor_sha256(state[key]) if state else None
                                        for key in ('exp_avg', 'exp_avg_sq')},
                'moments_finite': all(torch.isfinite(state[key]).all().item() for key in
                                      ('exp_avg', 'exp_avg_sq')) if state else True,
                'local_state_shape': list(state['exp_avg'].shape) if state else None}
            ranks = [None] * world
            if distributed:
                dist.all_gather_object(ranks, local)
            else:
                ranks[0] = local
            before_hash = ranks[0]['weight_before_sha256']
            if any(row['rank'] != rank or not row['moments_finite'] or
                   row['weight_before_sha256'] != before_hash or row['weight_after_sha256'] != before_hash or
                   row['step_before'] != snapshot['step'] or row['step_after'] != expected_step
                   for rank, row in enumerate(ranks)):
                raise RuntimeError("Held W or original AdamW clocks disagree across ranks")
            report.update(world_size=world, all_ranks_weight_equal=True,
                          all_ranks_state_steps_consistent=True, rank_proofs=ranks)
        return report

    def num_active_matmul_params(self):
        """Matmul parameters used by one token (expected active MoE compute)."""
        active_params = self.num_matmul_params()
        for block in self.transformer.h:
            if isinstance(block.mlp, MoE):
                per_expert = sum(p.numel() for p in block.mlp.experts[0].parameters())
                active_params -= (block.mlp.num_experts - block.mlp.top_k) * per_expert
        return active_params

    def estimate_decode_flops(self, context_len):
        """
        Forward FLOPs to decode one token at a given context length during inference:
        2 FLOPs per matmul param, plus attention over min(context, window) per layer.
        """
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        attn_flops = sum(4 * h * q * min(context_len, window) for window, _ in self.window_sizes)
        decode_flops = 2 * self.num_active_matmul_params() + attn_flops
        return decode_flops

    def estimate_prefill_flops(self, num_tokens):
        """Forward FLOPs to prefill a prompt: causal, so token t attends to min(t, window)."""
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        attn_flops = 0
        for window, _ in self.window_sizes:
            w = min(window, num_tokens)
            attended_tokens = w * (w + 1) // 2 + (num_tokens - w) * w # ramp up to w, then flat
            attn_flops += 4 * h * q * attended_tokens
        prefill_flops = 2 * self.num_active_matmul_params() * num_tokens + attn_flops
        return prefill_flops

    def kv_bytes_per_token(self):
        """Bytes to *store* one token of KV cache during inference, per row (all layers)."""
        head_dim = self.config.n_embd // self.config.n_head
        kv_dtype_bytes = COMPUTE_DTYPE.itemsize # the KV cache is kept in the compute dtype
        return self.config.n_layer * 2 * self.config.n_kv_head * head_dim * kv_dtype_bytes

    def kv_read_bytes(self, context_len):
        """Bytes of KV cache *read* by one decode step at a given context length, per row.
        Sliding window layers only attend to (and read) the last `window` tokens."""
        head_dim = self.config.n_embd // self.config.n_head
        kv_dtype_bytes = COMPUTE_DTYPE.itemsize
        total = 0
        for window, _ in self.window_sizes:
            total += 2 * self.config.n_kv_head * head_dim * kv_dtype_bytes * min(context_len, window)
        return total

    def num_scaling_params(self):
        """
        Return detailed parameter counts for scaling law analysis.
        Different papers use different conventions:
        - Kaplan et al. excluded embedding parameters
        - Chinchilla included all parameters
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper)

        Returns a dict with counts for each parameter group, so downstream analysis
        can experiment with which combination gives the cleanest scaling laws.
        """
        # Count each group separately (mirrors the grouping in setup_optimizers)
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel() + self.smear_gate.weight.numel() + self.smear_lambda.numel() + self.backout_lambda.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        moe_experts = sum(p.numel() for block in self.transformer.h if isinstance(block.mlp, MoE)
                          for p in block.mlp.experts.parameters())
        moe_router = sum(p.numel() for block in self.transformer.h if isinstance(block.mlp, MoE)
                         for p in block.mlp.router.parameters())
        active_experts = sum(sum(p.numel() for p in block.mlp.experts[0].parameters()) * block.mlp.top_k
                             for block in self.transformer.h if isinstance(block.mlp, MoE))
        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'scalars': scalars,
            'total': total,
            'moe_experts': moe_experts,
            'moe_router': moe_router,
            'active_experts': active_experts,
            'active_total': total - moe_experts + active_experts,
            'active_matmul_params': self.num_active_matmul_params(),
        }

    def get_moe_stats(self):
        """Detached per-layer statistics from the most recent forward only."""
        return {
            i: block.mlp.last_router_stats
            for i, block in enumerate(self.transformer.h)
            if isinstance(block.mlp, MoE) and block.mlp.last_router_stats is not None
        }

    def start_moe_stats(self, reset=True):
        """Enable cumulative routing counters (call before a task or generation)."""
        for block in self.transformer.h:
            if isinstance(block.mlp, MoE):
                if reset or block.mlp._cumulative_stats is None:
                    block.mlp.reset_stats()
                block.mlp._moe_stats_enabled = True

    def reset_moe_stats(self):
        """Clear cumulative counters while retaining their enabled state."""
        for block in self.transformer.h:
            if isinstance(block.mlp, MoE):
                block.mlp.reset_stats()

    def stop_moe_stats(self):
        """Stop accumulation without discarding the existing snapshot."""
        for block in self.transformer.h:
            if isinstance(block.mlp, MoE):
                block.mlp._moe_stats_enabled = False

    def set_moe_router_diagnostics(self, enabled):
        """Opt-in scalar diagnostics; ordinary training/evaluation stays unchanged."""
        for block in self.transformer.h:
            if isinstance(block.mlp, MoE):
                block.mlp._router_diagnostics_enabled = bool(enabled)
                block.mlp.last_router_diagnostics = None

    def get_moe_router_diagnostics(self):
        return {i: block.mlp.last_router_diagnostics for i, block in enumerate(self.transformer.h)
                if isinstance(block.mlp, MoE) and block.mlp.last_router_diagnostics is not None}

    def get_moe_cumulative_stats(self, as_python=False, distributed=False):
        """Return per-layer total and phase counters since the last reset.

        If distributed=True, all ranks must call this method together. The
        reduction happens only here; forward never performs metric collectives.
        """
        raw = {
            i: block.mlp._cumulative_stats
            for i, block in enumerate(self.transformer.h)
            if isinstance(block.mlp, MoE) and block.mlp._cumulative_stats is not None
        }
        if not raw:
            return {}
        if distributed:
            raw = all_reduce_moe_stats_counters(raw)
        else:
            raw = {
                i: {phase: {name: value.detach().clone() for name, value in counters.items()}
                    for phase, counters in phases.items()}
                for i, phases in raw.items()
            }
        result = {}
        for i, phases in raw.items():
            block = self.transformer.h[i]
            total = _summarize_moe_counters(phases["total"], block.mlp.top_k, as_python)
            total["phases"] = {
                phase: _summarize_moe_counters(phases[phase], block.mlp.top_k, as_python)
                for phase in MOE_STATS_PHASES
            }
            result[i] = total
        return result

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5, router_lr=None):
        model_dim = self.config.n_embd

        # Separate out all parameters into groups
        router_params = [block.mlp.router.weight for block in self.transformer.h if isinstance(block.mlp, MoE)]
        expert_ids = {id(p) for block in self.transformer.h if isinstance(block.mlp, MoE)
                      for p in block.mlp.experts.parameters()}
        router_ids = {id(p) for p in router_params}
        matrix_params = [p for p in self.transformer.h.parameters() if id(p) not in router_ids]
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        smear_params = [self.smear_gate.weight, self.smear_lambda, self.backout_lambda]
        assert len(list(self.parameters())) == len(matrix_params) + len(router_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params) + len(smear_params)

        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Build param_groups with all required fields explicit
        param_groups = [
            # AdamW groups (embeddings, lm_head, scalars)
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),  # higher beta1 for x0
            dict(kind='adamw', params=smear_params, lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        if router_params:
            effective_router_lr = unembedding_lr if router_lr is None else router_lr
            param_groups.append(dict(kind='adamw', params=router_params,
                                     lr=effective_router_lr * dmodel_lr_scale,
                                     betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0))
        # Muon groups: cap same-shape stacks so many experts do not create a
        # multi-gigabyte gradient/parameter/momentum buffer in a single step.
        _, _, _, world_size = get_dist_info()
        max_group_bytes = 128 * 1024 * 1024
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            bytes_per_param = group_params[0].numel() * group_params[0].element_size()
            # Keep dense grouping exactly as before so older optimizer states
            # can still be loaded; only MoE models need bounded stacks.
            group_size = (max(world_size, max(1, max_group_bytes // bytes_per_param))
                          if expert_ids else len(group_params))
            for start in range(0, len(group_params), group_size):
                param_groups.append(dict(
                    kind='muon', params=group_params[start:start + group_size], lr=matrix_lr,
                    momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
                    streaming=any(id(p) in expert_ids for p in group_params[start:start + group_size]),
                ))

        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean',
                routing_mask=None, routing_phase=None):
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == COMPUTE_DTYPE, f"Rotary embeddings must be in {COMPUTE_DTYPE}, got {self.cos.dtype}"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # Embed the tokens
        x = self.transformer.wte(idx) # embed current token
        x = x.to(COMPUTE_DTYPE) # ensure activations are in compute dtype (no-op usually, but active for fp16 code path)
        x = norm(x)

        # Smear: mix previous token's embedding into current position (cheap bigram info)
        if kv_cache is None:
            # Training / naive generate: full sequence available, use fast slice
            assert T > 1, "Training forward pass should have T > 1"
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        else:
            # KV cache inference: read prev embedding from cache, store current for next step
            x_pre_smear = kv_cache.prev_embedding
            kv_cache.prev_embedding = x[:, -1:, :]
            if T > 1:
                # Prefill: apply smear to positions 1+, same as training
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
                x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
            elif x_pre_smear is not None:
                # Decode: single token, use cached prev embedding
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, :, :24]))
                x = x + gate * x_pre_smear

        # Forward the trunk of the Transformer
        x0 = x  # save initial normalized embedding for x0 residual
        n_layer = self.config.n_layer
        backout_layer = n_layer // 2  # cache at halfway point
        x_backout = None
        moe_aux_losses = []
        valid_token_mask = targets != -1 if targets is not None and self.config.moe_num_experts > 0 else None
        stats_token_mask = None
        if routing_mask is not None and self.config.moe_num_experts > 0:
            stats_token_mask = torch.as_tensor(routing_mask, dtype=torch.bool, device=idx.device)
            if stats_token_mask.shape == (B,):
                stats_token_mask = stats_token_mask[:, None].expand(B, T)
            if stats_token_mask.shape != (B, T):
                raise ValueError(f"routing_mask must have shape {(B, T)} or {(B,)}, got {tuple(stats_token_mask.shape)}")
            if valid_token_mask is not None:
                stats_token_mask = stats_token_mask & valid_token_mask
        stats_phase = routing_phase or "other"
        if stats_phase not in MOE_STATS_PHASES:
            raise ValueError(f"routing_phase must be one of {MOE_STATS_PHASES}")
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
            x, aux_loss = block(x, ve, cos_sin, self.window_sizes[i], kv_cache,
                                valid_token_mask, stats_token_mask, stats_phase)
            if aux_loss is not None:
                moe_aux_losses.append(aux_loss)
            if i == backout_layer:
                x_backout = x
        self.last_moe_aux_loss = torch.stack(moe_aux_losses).mean() if moe_aux_losses else None
        # Subtract mid-layer residual to remove low-level features before logit projection
        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]
        logits = self.lm_head(x) # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
        logits = logits[..., :self.config.vocab_size] # slice to remove padding
        logits = logits.float() # switch to fp32 for logit softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap) # squash the logits

        if targets is not None:
            # training: given the targets, compute and return the loss
            # TODO experiment with chunked cross-entropy?
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            if self.training and loss_reduction == 'mean' and self.last_moe_aux_loss is not None:
                loss = loss + self.config.moe_aux_loss_coef * self.last_moe_aux_loss
            return loss
        else:
            # inference: just return the logits directly
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
