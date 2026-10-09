"""Exercise the opt-in diagnostic producer with real CPU forwards/backwards."""
import json

import pytest
import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.moe_training_metrics import MoETrainingMetrics, require_router_initial_lr


def model_and_optimizer():
    torch.manual_seed(47)
    model = GPT(GPTConfig(sequence_len=8, vocab_size=32, n_layer=2, n_head=2,
        n_kv_head=2, n_embd=32, window_pattern="L", moe_num_experts=8, moe_top_k=2,
        moe_every=2, moe_hidden_mult=1.0)).to("cpu")
    model.init_weights()
    routers = [block.mlp.router.weight for block in model.transformer.h if hasattr(block.mlp, "router")]
    router_ids = {id(parameter) for parameter in routers}
    other = [parameter for parameter in model.parameters() if id(parameter) not in router_ids]
    optimizer = torch.optim.SGD([{"params": other, "initial_lr": .1},
                                {"params": routers, "initial_lr": .0006196773353931868}], lr=.1)
    return model, optimizer, routers


def forward_backward(metrics, model):
    for _ in range(2):
        inputs = torch.randint(0, 32, (2, 8))
        targets = torch.randint(0, 32, (2, 8))
        loss = model(inputs, targets)
        metrics.observe(loss, targets)
        (loss / 2).backward()


def test_diagnostics_cover_whole_accumulation_with_actual_gradient_evidence(tmp_path):
    model, optimizer, _ = model_and_optimizer()
    metrics = MoETrainingMetrics(model, tmp_path / "diagnostics.jsonl", "pretrain", 100, diagnostics=True)
    metrics.begin(3210, force=True)
    forward_backward(metrics, model)
    metrics.capture_optimizer_diagnostics(optimizer)
    record = metrics.finish()
    assert record["routing"]["layers"][0]["token_count"] == 32
    assert record["routing"]["layers"][0]["assignment_count"] == 64
    diagnostic = record["router_diagnostics"]
    assert diagnostic["all_weights_finite"] is True
    assert diagnostic["all_gradients_finite"] is True
    assert diagnostic["router_initial_lrs"] == [.0006196773353931868]
    layer = diagnostic["layers"][0]
    assert layer["layer"] == 1 and layer["logit_min"] <= layer["logit_max"]
    assert 0 <= layer["prob_min"] <= layer["prob_max"] <= 1
    assert layer["router_gradient_norm"] > 0
    assert json.loads((tmp_path / "diagnostics.jsonl").read_text()) == record
    assert model.transformer.h[1].mlp._router_diagnostics_enabled is False


def test_nonfinite_router_gradient_is_reported_instead_of_claimed_finite(tmp_path):
    model, optimizer, routers = model_and_optimizer()
    metrics = MoETrainingMetrics(model, tmp_path / "bad.jsonl", "pretrain", 100, diagnostics=True)
    metrics.begin(3210, force=True)
    forward_backward(metrics, model)
    routers[0].grad[0, 0] = float("nan")
    metrics.capture_optimizer_diagnostics(optimizer)
    record = metrics.finish()
    diagnostic = record["router_diagnostics"]
    assert diagnostic["all_gradients_finite"] is False
    assert diagnostic["layers"][0]["gradients_finite"] is False
    assert diagnostic["layers"][0]["router_gradient_norm"] is None
    assert metrics.guard_stop_event["decision"] == "stop"
    assert record["invalid_nonfinite_evidence"]["finite_flags_false"]


def test_resume_lr_check_uses_the_loaded_optimizer_group(tmp_path):
    model, optimizer, _ = model_and_optimizer()
    expected = .0006196773353931868
    assert require_router_initial_lr(optimizer, expected, model) == [expected]
    optimizer.param_groups[1]["initial_lr"] = expected * 10
    with pytest.raises(ValueError, match="Loaded router initial_lr"):
        require_router_initial_lr(optimizer, expected, model)


def test_weights_are_checked_again_after_the_optimizer_update(tmp_path):
    model, optimizer, routers = model_and_optimizer()
    metrics = MoETrainingMetrics(model, tmp_path / "post_update.jsonl", "pretrain", 100, diagnostics=True)
    metrics.begin(3210, force=True)
    forward_backward(metrics, model)
    metrics.capture_optimizer_diagnostics(optimizer)
    with torch.no_grad():
        routers[0][0, 0] = float("inf")
    assert metrics.finish()["router_diagnostics"]["all_weights_finite"] is False


@pytest.mark.parametrize("stage,retain_final", [("pretrain", False), ("sft", True)])
def test_nonfinite_loss_outside_snapshot_still_latches_stop_and_writes_valid_json(tmp_path, stage, retain_final):
    model, _, _ = model_and_optimizer()
    path = tmp_path / "unscheduled_fault.jsonl"
    metrics = MoETrainingMetrics(model, path, stage, 100, diagnostics=True, retain_final=retain_final)
    metrics.begin(3211)
    inputs, targets = torch.randint(0, 32, (2, 8)), torch.randint(0, 32, (2, 8))
    loss = model(inputs, targets) * float("nan")
    metrics.observe(loss, targets)
    (loss / 2).backward()
    record = metrics.finish()
    assert record["optimizer_step"] == 3211
    assert record["guard_stop_event"]["forward_objective_finite"] is False
    assert metrics.guard_stop_event["decision"] == "stop"
    if retain_final:
        assert record["metadata"]["routing_detail"] == "selection_counts_and_router_probabilities"
    else:
        assert "routing_unavailable" in record
    assert "NaN" not in path.read_text() and json.loads(path.read_text()) == record
    assert metrics.finish() is None
