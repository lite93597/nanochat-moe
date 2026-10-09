import json

import pytest
import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.moe_training_metrics import MoETrainingMetrics


def tiny_model(experts=4):
    torch.manual_seed(17)
    model = GPT(GPTConfig(sequence_len=12, vocab_size=32, n_layer=2,
                          n_head=2, n_kv_head=2, n_embd=32, window_pattern='L',
                          moe_num_experts=experts, moe_top_k=2, moe_every=2,
                          moe_hidden_mult=2, moe_aux_loss_coef=0.01))
    model.init_weights()
    return model.train()


def observe_two_batches(metrics, model):
    expected = []
    for valid in (12, 3):
        x = torch.randint(0, 32, (1, 12))
        y = torch.randint(0, 32, (1, 12))
        y[:, valid:] = -1
        loss = model(x, y)
        auxiliary = float(model.last_moe_aux_loss.detach())
        expected.append((float(loss.detach()) - 0.01 * auxiliary, auxiliary, valid, float(loss.detach())))
        metrics.observe(loss, y)
    return expected


def test_token_weighting_all_microbatches_and_json_api(tmp_path):
    model = tiny_model()
    path = tmp_path / 'metrics.jsonl'
    metrics = MoETrainingMetrics(model, path, 'sft', 10, retain_final=True)
    metrics.begin(1)
    expected = observe_two_batches(metrics, model)
    result = metrics.finish()
    saved = json.loads(path.read_text())
    assert saved == result
    assert saved['valid_loss_tokens'] == 15
    assert saved['rank_microbatches'] == 2
    assert saved['ce_loss_token_weighted'] == pytest.approx(sum(ce * n for ce, _, n, _ in expected) / 15)
    assert saved['aux_loss_token_weighted'] == pytest.approx(sum(aux * n for _, aux, n, _ in expected) / 15)
    assert saved['optimizer_objective_rank_microbatch_mean'] == pytest.approx(sum(loss for _, _, _, loss in expected) / 2)
    assert saved['routing']['layers'][0]['token_count'] == 15
    assert saved['routing']['layers'][0]['assignment_count'] == 30
    assert sum(e['assignment_share'] for e in saved['routing']['layers'][0]['experts']) == pytest.approx(1)
    assert saved['metadata']['scope'] == 'supervised_loss_tokens'
    assert metrics.finish(force=True) is None  # final iteration must not duplicate


def test_deferred_full_epoch_final_step_keeps_actual_routes(tmp_path):
    model = tiny_model()
    path = tmp_path / 'metrics.jsonl'
    metrics = MoETrainingMetrics(model, path, 'sft', 10, retain_final=True)
    metrics.begin(2)
    assert not model.transformer.h[1].mlp._moe_stats_enabled
    observe_two_batches(metrics, model)
    assert metrics.finish(elapsed_seconds=1.25) is None
    assert not path.exists()
    result = metrics.finish(force=True)
    assert result['optimizer_step'] == 2
    assert result['valid_loss_tokens'] == 15
    assert result['elapsed_seconds'] == 1.25
    assert result['metadata']['routing_detail'] == 'selection_counts_and_router_probabilities'
    assert result['routing']['layers'][0]['assignment_count'] == 30
    assert sum(e['mean_router_probability'] for e in result['routing']['layers'][0]['experts']) == pytest.approx(1)
    assert 'top1_share' not in result['routing']['layers'][0]['experts'][0]


def test_distributed_packet_uses_three_collectives_at_log_only(tmp_path, monkeypatch):
    import torch.distributed as dist
    model = tiny_model()
    metrics = MoETrainingMetrics(model, tmp_path / 'm.jsonl', 'pretrain', 100)
    calls = []
    monkeypatch.setattr(dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(dist, 'get_rank', lambda: 0)
    monkeypatch.setattr(dist, 'get_world_size', lambda: 2)
    monkeypatch.setattr(dist, 'get_backend', lambda group=None: 'gloo')
    def two_equal_ranks(packet, op=None, group=None):
        calls.append((packet.dtype, packet.numel()))
        packet.mul_(2)
        if packet.numel() == 5:  # rank 1 has a different CE/objective
            packet[0].add_(15)
            packet[3].add_(2)
    monkeypatch.setattr(dist, 'all_reduce', two_equal_ranks)
    metrics.begin(1)
    expected = observe_two_batches(metrics, model)
    assert not calls
    result = metrics.finish()
    assert len(calls) == 3
    assert result['valid_loss_tokens'] == 30
    assert result['rank_microbatches'] == 4
    assert result['ce_loss_token_weighted'] == pytest.approx(sum(ce * n for ce, _, n, _ in expected) / 15 + 0.5)
    assert result['optimizer_objective_rank_microbatch_mean'] == pytest.approx(sum(loss for _, _, _, loss in expected) / 2 + 0.5)
    assert result['routing']['layers'][0]['token_count'] == 30
    metrics.begin(2)
    observe_two_batches(metrics, model)
    assert metrics.finish() is None
    assert len(calls) == 3


def test_dense_disabled_and_unlogged_pretrain_has_no_retention(tmp_path):
    dense = tiny_model(experts=0)
    metrics = MoETrainingMetrics(dense, tmp_path / 'dense.jsonl', 'pretrain', 100)
    metrics.begin(1)
    metrics.observe(torch.tensor(1.0), torch.zeros(1, 12, dtype=torch.long))
    assert metrics.finish() is None
    sparse = tiny_model()
    metrics = MoETrainingMetrics(sparse, tmp_path / 'sparse.jsonl', 'pretrain', 100)
    metrics.begin(2)
    observe_two_batches(metrics, sparse)
    assert metrics.packet is None
    assert metrics.finish() is None
    assert not list(tmp_path.iterdir())
