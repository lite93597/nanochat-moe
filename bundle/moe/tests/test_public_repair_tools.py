"""Small, offline checks for public repair adapters; no historical receipts."""
import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_upcycle_preserves_input_and_independent_experts():
    module = load_file("public_upcycle", ROOT / "tools/upcycle_experts.py")
    state = {"shared": torch.ones(2)}
    for expert in range(4):
        for matrix in ("c_fc", "c_proj"):
            state[f"transformer.h.1.mlp.experts.{expert}.{matrix}.weight"] = torch.full((2, 3), float(expert))
    copied, changed = module.copy_experts(state, 1, 2, 4)
    assert len(changed) == 6
    assert copied["shared"] is state["shared"]
    target = "transformer.h.1.mlp.experts.0.c_fc.weight"
    donor = "transformer.h.1.mlp.experts.2.c_fc.weight"
    assert torch.equal(copied[target], state[donor])
    copied[target].fill_(7)
    assert torch.all(state[target] == 0)
    assert torch.all(copied[donor] == 2)


def test_upcycle_rejects_missing_experts():
    module = load_file("public_upcycle_missing", ROOT / "tools/upcycle_experts.py")
    with pytest.raises(ValueError, match="missing"):
        module.copy_experts({}, 1, 0, 8)


def test_bias_solver_corrects_skewed_router_probabilities():
    module = load_file("public_calibration_solver", ROOT / "research/historical_r5/cloud/calibrate_moe20_router.py")
    torch.manual_seed(3)
    logits = torch.randn(128, 8) * .3 + torch.arange(8) * .8
    bias, receipt = module.fit_intercept(logits, torch.zeros(8))
    mean = (logits.double() + bias.double()).softmax(-1).mean(0)
    assert receipt["converged"]
    assert (mean - 1 / 8).abs().max() < 1e-6
    assert abs(bias.mean().item()) < 1e-6

