"""Offline CPU checks for the sparse MoE model and its checkpoint/inference path.

Run: python -m pytest -q tests/test_moe.py
No dataset, tokenizer files, GPU, or network are required.
"""

from dataclasses import asdict
import json

import pytest
import torch

from nanochat.checkpoint_manager import build_model, load_model_from_dir, save_checkpoint
from nanochat.common import COMPUTE_DTYPE, get_base_dir, get_data_dir
from nanochat.core_eval import forward_model
from nanochat.engine import Engine, KVCache
from nanochat.gpt import GPT, GPTConfig
from nanochat.moe_report import _summarize_counters, add_last_forward_router_metrics, collect_moe_routing_report, save_moe_routing_reports


def tiny_config(**overrides):
    kwargs = dict(
        sequence_len=12,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
        window_pattern="L",
        moe_num_experts=4,
        moe_top_k=2,
        moe_hidden_mult=2.0,
        moe_every=2,
        moe_aux_loss_coef=0.01,
    )
    kwargs.update(overrides)
    return GPTConfig(**kwargs)


def tiny_model(**overrides):
    torch.manual_seed(7)
    model = GPT(tiny_config(**overrides))
    model.init_weights()
    return model


def moe_layers(model):
    return [(i, block.mlp) for i, block in enumerate(model.transformer.h)
            if hasattr(block.mlp, "router")]


def test_alternating_moe_and_dense_layers():
    model = tiny_model(n_layer=4)
    assert len(moe_layers(model)) == 2
    assert all(len(moe.experts) == 4 for _, moe in moe_layers(model))
    assert len([b for b in model.transformer.h if not hasattr(b.mlp, "router")]) == 2


def test_top_k_dispatch_and_unused_expert_gradients():
    model = tiny_model()
    _, moe = moe_layers(model)[0]
    # Every token chooses experts 0 and 1; 2 and 3 receive no token data.
    with torch.no_grad():
        moe.router.weight[0].fill_(2.0)
        moe.router.weight[1].fill_(1.0)
        moe.router.weight[2:].fill_(-2.0)
        for expert in moe.experts[:2]:
            expert.c_fc.weight.fill_(0.02)
            expert.c_proj.weight.fill_(0.03)
    calls = [0] * 4
    handles = []
    for expert_id, expert in enumerate(moe.experts):
        def count_forward(_module, inputs, _output, expert_id=expert_id):
            calls[expert_id] += inputs[0].shape[0]
        handles.append(expert.c_fc.register_forward_hook(count_forward))
    try:
        x = torch.ones(2, 3, model.config.n_embd)
        result = moe(x)
        y = result[0] if isinstance(result, tuple) else result
        assert y.shape == x.shape
        assert calls == [6, 6, 0, 0]
        y.sum().backward()
        assert moe.experts[0].c_proj.weight.grad is not None
        assert moe.experts[1].c_proj.weight.grad is not None
        for expert in moe.experts[2:]:
            assert all(p.grad is None or torch.count_nonzero(p.grad) == 0
                       for p in expert.parameters())
    finally:
        for handle in handles:
            handle.remove()


def test_zero_token_dispatch_is_finite():
    model = tiny_model()
    _, moe = moe_layers(model)[0]
    x = torch.empty(0, 0, model.config.n_embd, requires_grad=True)
    result = moe(x)
    y, aux = result if isinstance(result, tuple) else (result, None)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    if aux is not None:
        assert aux.ndim == 0 and torch.isfinite(aux)


def test_train_mean_includes_aux_but_none_and_eval_are_pure_ce():
    model = tiny_model(moe_aux_loss_coef=0.2)
    idx = torch.randint(0, 32, (2, 6))
    targets = torch.randint(0, 32, (2, 6))
    model.train()
    per_token_ce = model(idx, targets, loss_reduction="none")
    mean_train_loss = model(idx, targets, loss_reduction="mean")
    aux = model.last_moe_aux_loss
    assert per_token_ce.shape == (idx.numel(),)
    assert aux is not None and torch.isfinite(aux)
    torch.testing.assert_close(mean_train_loss, per_token_ce.mean() + 0.2 * aux)
    stats = model.get_moe_stats()
    assert len(stats) == 1
    layer_stats = next(iter(stats.values()))
    assert layer_stats["expert_counts"].sum().item() == idx.numel() * 2
    assert layer_stats["mean_router_probs"].shape == (4,)
    assert torch.isfinite(layer_stats["aux_loss"])
    model.eval()
    pure_eval_loss = model(idx, targets, loss_reduction="mean")
    eval_per_token_ce = model(idx, targets, loss_reduction="none")
    torch.testing.assert_close(pure_eval_loss, eval_per_token_ce.mean())


def test_masked_tokens_do_not_count_toward_router_balance():
    model = tiny_model(moe_aux_loss_coef=0.2)
    idx = torch.randint(0, 32, (2, 6))
    targets = torch.randint(0, 32, (2, 6))
    targets[:, :2] = -1
    model.train()
    per_token_ce = model(idx, targets, loss_reduction="none")
    mean_train_loss = model(idx, targets)
    stats = next(iter(model.get_moe_stats().values()))
    valid_count = (targets != -1).sum().item()
    assert stats["expert_counts"].sum().item() == valid_count * model.config.moe_top_k
    torch.testing.assert_close(
        mean_train_loss,
        per_token_ce.sum() / valid_count + model.config.moe_aux_loss_coef * model.last_moe_aux_loss,
    )


def test_checkpoint_rebuild_preserves_moe_and_logits(tmp_path, monkeypatch):
    model = tiny_model()
    model.eval()
    idx = torch.randint(0, 32, (1, 5))
    with torch.no_grad():
        expected_logits = model(idx)
    metadata = {"model_config": asdict(model.config)}
    save_checkpoint(tmp_path, 3, model.state_dict(), None, metadata)

    class FakeTokenizer:
        def get_vocab_size(self):
            return 32

    monkeypatch.setattr("nanochat.checkpoint_manager.get_tokenizer", FakeTokenizer)
    loaded, tokenizer, loaded_metadata = build_model(tmp_path, 3, torch.device("cpu"), "eval")
    assert tokenizer.get_vocab_size() == 32
    assert loaded.config.moe_num_experts == 4
    assert asdict(loaded.config) == loaded_metadata["model_config"]
    with torch.no_grad():
        torch.testing.assert_close(loaded(idx), expected_logits)


def test_legacy_rl_checkpoint_gets_step_from_filename(tmp_path, monkeypatch):
    model = tiny_model()
    checkpoint_dir = tmp_path / 'legacy'
    save_checkpoint(checkpoint_dir, 3, model.state_dict(), None, {'model_config': asdict(model.config)})

    class FakeTokenizer:
        def get_vocab_size(self):
            return 32

    monkeypatch.setattr('nanochat.checkpoint_manager.get_tokenizer', FakeTokenizer)
    _, _, metadata = load_model_from_dir(tmp_path, torch.device('cpu'), 'eval', model_tag='legacy')
    assert metadata['step'] == 3
    assert metadata['model_tag'] == 'legacy'


def test_optimizer_groups_include_each_router_and_expert_once():
    model = tiny_model(n_layer=4)
    optimizer = model.setup_optimizer(router_lr=0.003)
    grouped_ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert len(grouped_ids) == len(set(grouped_ids)) == len(list(model.parameters()))
    for _, moe in moe_layers(model):
        router_groups = [g for g in optimizer.param_groups
                         if any(p is moe.router.weight for p in g["params"])]
        assert len(router_groups) == 1 and router_groups[0]["kind"] == "adamw"
        for expert_param in moe.experts.parameters():
            expert_groups = [g for g in optimizer.param_groups
                             if any(p is expert_param for p in g["params"])]
            assert len(expert_groups) == 1
            assert expert_groups[0]["kind"] == "muon"
            assert expert_groups[0]["streaming"]


def test_moe_optimizer_step_with_unused_experts(monkeypatch):
    # Exercise the real mixed Muon/AdamW optimizer without compiling its fused
    # kernels, which is unnecessary for a two-token CPU correctness check.
    import nanochat.optim as optim_module

    monkeypatch.setattr(optim_module, "adamw_step_fused",
                        getattr(optim_module.adamw_step_fused, "__wrapped__", optim_module.adamw_step_fused))
    monkeypatch.setattr(optim_module, "muon_step_fused",
                        getattr(optim_module.muon_step_fused, "__wrapped__", optim_module.muon_step_fused))
    model = tiny_model(n_layer=1, moe_every=1, moe_num_experts=8, moe_top_k=1)
    optimizer = model.setup_optimizer()
    idx = torch.randint(0, 32, (1, 2))
    targets = torch.randint(0, 32, (1, 2))
    loss = model(idx, targets)
    loss.backward()
    stats = next(iter(model.get_moe_stats().values()))
    assert torch.count_nonzero(stats["expert_counts"] == 0) >= 6
    before = model.lm_head.weight.detach().clone()
    optimizer.step()
    assert torch.isfinite(model.lm_head.weight).all()
    assert not torch.equal(before, model.lm_head.weight.detach())


def test_tiny_train_step_and_generation():
    model = tiny_model()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    idx = torch.randint(0, 32, (2, 6))
    targets = torch.randint(0, 32, (2, 6))
    before = model.lm_head.weight.detach().clone()
    loss = model(idx, targets)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    assert not torch.equal(before, model.lm_head.weight.detach())
    model.eval()
    generated = list(model.generate([1, 2, 3], max_tokens=3, temperature=0))
    assert len(generated) == 3
    assert all(0 <= token < model.config.vocab_size for token in generated)


def test_kv_cache_decode_matches_full_sequence():
    model = tiny_model()
    model.eval()
    idx = torch.randint(0, 32, (1, 5))
    cache = KVCache(batch_size=1, num_heads=model.config.n_kv_head,
                    seq_len=12, head_dim=model.config.n_embd // model.config.n_head,
                    num_layers=model.config.n_layer, device="cpu", dtype=COMPUTE_DTYPE)
    with torch.no_grad():
        full_last = model(idx)[:, -1]
        model(idx[:, :-1], kv_cache=cache)
        cached_last = model(idx[:, -1:], kv_cache=cache)[:, -1]
    tolerance = 1e-5 if COMPUTE_DTYPE == torch.float32 else 2e-2
    torch.testing.assert_close(cached_last, full_last, atol=tolerance, rtol=tolerance)


def test_data_and_checkpoint_dir_overrides(tmp_path, monkeypatch):
    model_root = tmp_path / "models"
    data_root = tmp_path / "datasets"
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(model_root))
    monkeypatch.setenv("NANOCHAT_DATA_DIR", str(data_root))
    assert get_base_dir() == str(model_root)
    assert get_data_dir() == str(data_root)
    assert model_root.is_dir() and data_root.is_dir()


def _cumulative_layer(model):
    stats = model.get_moe_cumulative_stats()
    assert len(stats) == 1
    return next(iter(stats.values()))


def _check_routing_denominators(stats, top_k, num_experts):
    """One token makes top_k assignments but only one top-1 choice."""
    tokens = int(torch.as_tensor(stats["token_count"]).item())
    counts = torch.as_tensor(stats["expert_counts"])
    top1 = torch.as_tensor(stats["top1_counts"])
    assert counts.shape == top1.shape == (num_experts,)
    assert int(counts.sum()) == tokens * top_k
    assert int(top1.sum()) == tokens
    for name in ("assignment_share", "top1_share", "gate_share", "mean_router_probs"):
        values = torch.as_tensor(stats[name])
        assert values.shape == (num_experts,)
        assert torch.isfinite(values).all()
        expected_sum = 1.0 if tokens else 0.0
        torch.testing.assert_close(values.sum().float(), torch.tensor(expected_sum), atol=1e-5, rtol=1e-5)
    gate_sum = torch.as_tensor(stats["gate_weight_sums"])
    prob_sum = torch.as_tensor(stats["router_prob_sums"])
    torch.testing.assert_close(gate_sum.sum().float(), torch.tensor(float(tokens)), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(prob_sum.sum().float(), torch.tensor(float(tokens)), atol=1e-5, rtol=1e-5)
    assert torch.isfinite(torch.as_tensor(stats["load_cv"]))
    assert list(stats["unused_experts"]) == torch.nonzero(counts == 0).flatten().tolist()


def test_cumulative_stats_are_opt_in_and_accumulate_across_forwards():
    model = tiny_model()
    model.eval()
    idx = torch.randint(0, 32, (2, 5))
    with torch.no_grad():
        model(idx)
    assert model.get_moe_cumulative_stats() == {}  # default: no collection

    model.start_moe_stats(reset=True)
    _check_routing_denominators(_cumulative_layer(model), top_k=2, num_experts=4)
    assert int(torch.as_tensor(_cumulative_layer(model)["token_count"])) == 0
    with torch.no_grad():
        model(idx)
    one = _cumulative_layer(model)
    first_counts = torch.as_tensor(one["expert_counts"]).clone()
    assert int(torch.as_tensor(one["token_count"])) == idx.numel()
    _check_routing_denominators(one, top_k=2, num_experts=4)

    with torch.no_grad():
        model(idx)
    two = _cumulative_layer(model)
    assert int(torch.as_tensor(two["token_count"])) == 2 * idx.numel()
    torch.testing.assert_close(torch.as_tensor(two["expert_counts"]), 2 * first_counts)
    _check_routing_denominators(two, top_k=2, num_experts=4)
    # The public Python representation must be ready for JSON reports.
    json.dumps(model.get_moe_cumulative_stats(as_python=True))


def test_cumulative_stats_reset_and_stop_lifecycle():
    model = tiny_model()
    model.eval()
    idx = torch.randint(0, 32, (1, 4))
    model.start_moe_stats(reset=True)
    with torch.no_grad():
        model(idx)
    first = _cumulative_layer(model)
    first_counts = torch.as_tensor(first["expert_counts"]).clone()
    model.reset_moe_stats()  # reset keeps collection enabled
    zero = _cumulative_layer(model)
    assert int(torch.as_tensor(zero["token_count"])) == 0
    _check_routing_denominators(zero, top_k=2, num_experts=4)
    with torch.no_grad():
        model(idx)
    torch.testing.assert_close(torch.as_tensor(_cumulative_layer(model)["expert_counts"]), first_counts)
    model.stop_moe_stats()
    with torch.no_grad():
        model(idx)
    after_stop = _cumulative_layer(model)
    assert int(torch.as_tensor(after_stop["token_count"])) == idx.numel()
    torch.testing.assert_close(torch.as_tensor(after_stop["expert_counts"]), first_counts)


def test_cumulative_stats_masks_padding_and_handles_all_masked():
    model = tiny_model()
    idx = torch.randint(0, 32, (2, 6))
    targets = torch.randint(0, 32, (2, 6))
    targets[:, :2] = -1
    routing_mask = torch.ones_like(idx, dtype=torch.bool)
    routing_mask[0, 3:] = False
    valid = (targets != -1) & routing_mask
    model.start_moe_stats(reset=True)
    model(idx, targets, routing_mask=routing_mask)
    stats = _cumulative_layer(model)
    assert int(torch.as_tensor(stats["token_count"])) == int(valid.sum())
    _check_routing_denominators(stats, top_k=2, num_experts=4)
    model.reset_moe_stats()
    model(idx, targets, routing_mask=torch.zeros_like(routing_mask))
    zero = _cumulative_layer(model)
    assert int(torch.as_tensor(zero["token_count"])) == 0
    _check_routing_denominators(zero, top_k=2, num_experts=4)


def test_cumulative_stats_count_cached_prefill_and_decode_once():
    model = tiny_model()
    model.eval()
    idx = torch.randint(0, 32, (1, 5))
    cache = KVCache(batch_size=1, num_heads=model.config.n_kv_head,
                    seq_len=12, head_dim=model.config.n_embd // model.config.n_head,
                    num_layers=model.config.n_layer, device="cpu", dtype=COMPUTE_DTYPE)
    model.start_moe_stats(reset=True)
    with torch.no_grad():
        model(idx[:, :4], kv_cache=cache, routing_phase="prefill")
        model(idx[:, 4:], kv_cache=cache, routing_phase="decode")
    stats = _cumulative_layer(model)
    assert int(torch.as_tensor(stats["token_count"])) == 5
    assert int(torch.as_tensor(stats["phases"]["prefill"]["token_count"])) == 4
    assert int(torch.as_tensor(stats["phases"]["decode"]["token_count"])) == 1
    assert int(torch.as_tensor(stats["phases"]["other"]["token_count"])) == 0
    for phase in ("prefill", "decode", "other"):
        _check_routing_denominators(stats["phases"][phase], top_k=2, num_experts=4)
    _check_routing_denominators(stats, top_k=2, num_experts=4)


def test_dense_model_has_no_cumulative_moe_stats():
    model = tiny_model(moe_num_experts=0)
    assert model.get_moe_cumulative_stats() == {}
    model.start_moe_stats(reset=True)
    with torch.no_grad():
        model(torch.randint(0, 32, (1, 4)))
    assert model.get_moe_cumulative_stats() == {}
    model.reset_moe_stats()
    model.stop_moe_stats()
    assert model.get_moe_cumulative_stats(as_python=True) == {}


def test_routing_report_top_k_denominators_and_zero_tokens():
    counters = {
        "token_count": 5,
        "expert_counts": [5, 3, 2, 0],
        "top1_counts": [3, 2, 0, 0],
        "gate_weight_sums": [2.5, 1.5, 1.0, 0.0],
        "router_prob_sums": [2.0, 1.5, 1.0, 0.5],
    }
    report = _summarize_counters(counters, top_k=2)
    experts = report["experts"]
    assert report["assignment_count"] == 10
    assert report["unused_experts"] == [3]
    assert sum(e["assignment_share"] for e in experts) == pytest.approx(1.0)
    assert sum(e["token_hit_rate"] for e in experts) == pytest.approx(2.0)
    assert sum(e["top1_share"] for e in experts) == pytest.approx(1.0)
    assert sum(e["gate_share"] for e in experts) == pytest.approx(1.0)
    assert experts[0]["assignment_share"] == 0.5
    assert experts[0]["token_hit_rate"] == 1.0

    empty = {key: ([0] * 4 if isinstance(value, list) else 0)
             for key, value in counters.items()}
    zero = _summarize_counters(empty, top_k=2)
    assert zero["token_count"] == zero["assignment_count"] == 0
    assert zero["unused_experts"] == [0, 1, 2, 3]
    assert all(e["assignment_share"] == e["token_hit_rate"] == e["top1_share"] == 0
               for e in zero["experts"])


def test_routing_report_includes_phase_metrics_and_json(tmp_path):
    model = tiny_model()
    model.eval()
    model.start_moe_stats(reset=True)
    with torch.no_grad():
        model(torch.randint(0, 32, (1, 4)), routing_phase="prefill")
        model(torch.randint(0, 32, (1, 2)), routing_phase="decode")
    report = collect_moe_routing_report(model, label="test")
    assert report["label"] == "test" and report["top_k"] == 2
    layer = report["layers"][0]
    assert layer["token_count"] == 6
    assert layer["phases"]["prefill"]["token_count"] == 4
    assert layer["phases"]["decode"]["token_count"] == 2
    assert layer["phases"]["other"]["token_count"] == 0
    assert sum(e["token_hit_rate"] for e in layer["experts"]) == pytest.approx(2.0)
    assert sum(e["assignment_share"] for e in layer["experts"]) == pytest.approx(1.0)
    for phase in ("prefill", "decode"):
        phase_experts = layer["phases"][phase]["experts"]
        assert sum(e["assignment_share"] for e in phase_experts) == pytest.approx(1.0)
        assert sum(e["token_hit_rate"] for e in phase_experts) == pytest.approx(2.0)
        assert sum(e["top1_share"] for e in phase_experts) == pytest.approx(1.0)
    path = tmp_path / "routing.json"
    save_moe_routing_reports(path, [report], metadata={"source": "tiny"})
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["metadata"]["source"] == "tiny"
    assert payload["reports"][0]["layers"][0]["phases"]["decode"]["token_count"] == 2
    assert "gate_share" in payload["reports"][0]["layers"][0]["experts"][0]


def test_routing_report_dense_model_returns_none():
    model = tiny_model(moe_num_experts=0)
    model.start_moe_stats(reset=True)
    with torch.no_grad():
        model(torch.randint(0, 32, (1, 4)))
    assert collect_moe_routing_report(model, label="dense") is None


def test_training_router_log_includes_each_expert_share():
    model = tiny_model()
    with torch.no_grad():
        model(torch.randint(0, 32, (1, 4)))
    log_data = {}
    add_last_forward_router_metrics(log_data, model.get_moe_stats())
    shares = [log_data[f'train/moe/layer_1/expert_{index}_assignment_share'] for index in range(4)]
    assert sum(shares) == pytest.approx(1.0)
    assert 'train/moe/layer_1/unused_experts' in log_data
    assert 'train/moe/layer_1/expert_0_mean_router_probability' in log_data


def test_engine_tags_prefill_and_decode_routing(monkeypatch):
    model = tiny_model()
    model.eval()
    model.start_moe_stats(reset=True)

    class TinyTokenizer:
        def encode_special(self, token):
            return {"<|python_start|>": 24, "<|python_end|>": 25,
                    "<|output_start|>": 26, "<|output_end|>": 27,
                    "<|assistant_end|>": 28}[token]

        def get_bos_token_id(self):
            return 29

    def choose_regular_token(logits, rng, temperature=1.0, top_k=None):
        return torch.full((logits.size(0), 1), 10, device=logits.device, dtype=torch.long)

    monkeypatch.setattr("nanochat.engine.sample_next_token", choose_regular_token)
    generated = list(Engine(model, TinyTokenizer()).generate(
        [1, 2, 3], num_samples=2, max_tokens=3, temperature=0,
    ))
    assert len(generated) == 3
    stats = _cumulative_layer(model)
    # Prefill runs once for the three prompt tokens; decode runs twice for two rows.
    assert int(torch.as_tensor(stats["phases"]["prefill"]["token_count"])) == 3
    assert int(torch.as_tensor(stats["phases"]["decode"]["token_count"])) == 4
    assert int(torch.as_tensor(stats["token_count"])) == 7
    _check_routing_denominators(stats, top_k=2, num_experts=4)


def test_core_forward_excludes_right_padding_from_routing_stats():
    model = tiny_model()
    model.eval()
    ids = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 7, 8]])
    valid = torch.tensor([[True, True, True, False, False], [True] * 5])
    model.start_moe_stats(reset=True)
    losses, predictions = forward_model(model, ids, routing_mask=valid)
    assert losses.shape == predictions.shape == ids.shape
    stats = _cumulative_layer(model)
    assert int(torch.as_tensor(stats["token_count"])) == 8
    _check_routing_denominators(stats, top_k=2, num_experts=4)


def test_cumulative_stats_single_expert_and_invalid_inputs():
    model = tiny_model(moe_num_experts=1, moe_top_k=1)
    idx = torch.randint(0, 32, (1, 3))
    model.start_moe_stats(reset=True)
    with torch.no_grad():
        model(idx)
    stats = _cumulative_layer(model)
    assert int(torch.as_tensor(stats["token_count"])) == 3
    _check_routing_denominators(stats, top_k=1, num_experts=1)
    assert list(stats["unused_experts"]) == []
    with pytest.raises(ValueError, match="routing_mask"):
        model(idx, routing_mask=torch.ones(2, 2, dtype=torch.bool))
    with pytest.raises(ValueError, match="routing_phase"):
        model(idx, routing_phase="unknown")
