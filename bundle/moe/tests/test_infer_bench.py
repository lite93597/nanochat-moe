"""Architecture accounting checks; no GPU, model download, or compilation required."""
import json

import pytest
import torch

from nanochat.gpt import GPT, GPTConfig
from scripts.infer_bench import model_cost_card, percent_text, utilization, weight_bytes


def tiny_model(experts):
    config = GPTConfig(sequence_len=128, vocab_size=64, n_layer=2, n_head=1,
                       n_kv_head=1, n_embd=64, window_pattern="L",
                       moe_num_experts=experts, moe_top_k=2, moe_every=2)
    with torch.device("meta"):
        return GPT(config)


def test_moe_has_active_compute_and_resident_storage_but_no_dense_bandwidth_claim():
    model = tiny_model(4)
    card = model_cost_card(model, peak_bw=1e12, total_vram=8 * 1024**3, context_len=64)
    counts = model.num_scaling_params()
    assert card["num_total_params"] == counts["total"]
    assert card["num_active_params"] == counts["active_total"] < counts["total"]
    assert card["num_active_matmul_params"] < card["num_total_matmul_params"]
    assert card["moe"]["layers"] == [1]
    assert card["moe"]["active_expert_params_per_token"] == counts["moe_experts"] // 2
    assert card["moe"]["router_params"] > 0
    assert card["weight_bytes"] == weight_bytes(model)
    assert card["max_full_context_rows"] > 0
    assert card["ceiling_bs1_tok_per_sec"] is None
    assert card["memory_roofline_supported"] is False
    assert "not measured per-step" in card["weight_bytes_scope"]
    assert "excludes" in card["flops_estimate_scope"]
    json.dumps(card, allow_nan=False)


def test_dense_retains_approximate_memory_roofline():
    model = tiny_model(0)
    card = model_cost_card(model, peak_bw=1e12, total_vram=8 * 1024**3, context_len=64)
    assert card["num_active_params"] == card["num_total_params"]
    assert card["num_active_matmul_params"] == card["num_total_matmul_params"]
    assert card["memory_roofline_supported"] is True
    assert card["ceiling_bs1_tok_per_sec"] == round(1e12 / (card["weight_bytes"] + card["kv_read_bytes_per_step"]), 1)
    assert card["moe"]["enabled"] is False


@pytest.mark.parametrize("peak", [float("inf"), float("nan"), 0])
def test_unknown_hardware_does_not_report_zero_utilization(peak):
    assert utilization(123, peak) is None
    assert percent_text(utilization(123, peak)) == "N/A"


def test_known_hardware_utilization_and_unknown_ceiling():
    assert utilization(25, 100) == 25
    assert percent_text(12.34, precision=2) == "12.34"
    card = model_cost_card(tiny_model(0), peak_bw=float("inf"), total_vram=8 * 1024**3, context_len=64)
    assert card["ceiling_bs1_tok_per_sec"] is None
    json.dumps(card, allow_nan=False)
