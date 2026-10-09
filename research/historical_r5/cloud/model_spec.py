"""Exact architecture and completed checkpoint contract for the reference MoE20 run."""
import json
import hashlib
from pathlib import Path


MOE20_TAG = "moe_d20_e8_k2"
MOE20_ARCHITECTURE = {
    "n_layer": 20, "n_embd": 1280, "n_head": 10, "n_kv_head": 10,
    "sequence_len": 2048, "vocab_size": 32768, "window_pattern": "L",
    "moe_num_experts": 8, "moe_top_k": 2, "moe_every": 2, "moe_hidden_mult": 2.0,
}
MOE20_BASE_STEP = 6641
MOE20_R2_ROUTING_CONTROL = {
    'kind': 'last_layer_selection_bias', 'layer': 19, 'rate': 0.005,
    'score': 'router_probs_plus_bias', 'gate': 'unbiased_router_probs_renormalized',
    'feedback': 'global_optimizer_step_sign_centered', 'eval': 'frozen',
}
MOE20_R2_BIAS_KEY = 'transformer.h.19.mlp.selection_bias'
MOE20_R3_ROUTING_CONTROL = {
    'kind': 'last_layer_router_intercept', 'layer': 19, 'rate': 0.5, 'max_step': 0.05,
    'score': 'router_logits_plus_bias_softmax', 'gate': 'biased_router_probs_renormalized',
    'feedback': 'global_optimizer_step_probability_proportional_clipped_centered',
    'aux': 'biased_router_probs_switch_0.01', 'eval': 'frozen',
    'calibration': 'training_window_softmax_intercept_newton_float64',
}
MOE20_R4_ROUTING_CONTROL = {
    **MOE20_R3_ROUTING_CONTROL, 'kind': 'last_layer_router_intercept_held_router_weight',
    'hold_layer': 19, 'hold_policy': 'restore_after_real_optimizer_step',
    'optimizer_state_policy': 'real_gradients_moments_and_step_advance',
}
MOE20_HELD_ROUTER_KEY = 'transformer.h.19.mlp.router.weight'


def validate_routing_control(value):
    for recipe in (MOE20_R2_ROUTING_CONTROL, MOE20_R3_ROUTING_CONTROL, MOE20_R4_ROUTING_CONTROL):
        if (isinstance(value, dict) and set(value) == set(recipe)
                and all(type(value[key]) is type(expected) and value[key] == expected for key, expected in recipe.items())):
            return dict(value)
    raise ValueError('Only the explicit supported R2, R3 or R4 routing-control recipe is permitted')


def routing_control_from_config(config):
    rate, layer = config.get('moe_load_bias_rate', 0), config.get('moe_load_bias_layer', -1)
    mode = config.get('moe_load_bias_mode', 'selection_probability_sign')
    max_step = config.get('moe_load_bias_max_step', 0.05)
    hold_layer = config.get('moe_router_hold_layer', -1)
    if type(hold_layer) is not int or hold_layer not in (-1, 19):
        raise ValueError('Checkpoint contains an unsupported router-weight hold layer')
    if type(max_step) is not float or max_step != .05:
        raise ValueError('Checkpoint contains an unsupported router-intercept maximum step')
    if rate == 0 and layer == -1 and type(rate) in (int, float) and type(layer) is int:
        if mode != 'selection_probability_sign' or hold_layer != -1:
            raise ValueError('Disabled routing control cannot declare R3 semantics')
        return None
    if type(rate) is float and rate == 0.005 and type(layer) is int and layer == 19 and mode == 'selection_probability_sign' and hold_layer == -1:
        return dict(MOE20_R2_ROUTING_CONTROL)
    if type(rate) is float and rate == 0.5 and type(layer) is int and layer == 19 and mode == 'router_logit_probability_proportional':
        return dict(MOE20_R4_ROUTING_CONTROL if hold_layer == 19 else MOE20_R3_ROUTING_CONTROL)
    raise ValueError('Checkpoint contains an unsupported routing control')


def validate_model_spec(payload):
    if not isinstance(payload, dict) or set(payload) not in ({"model_tag", "architecture", "checkpoints"},
            {"model_tag", "architecture", "checkpoints", "routing_control"}):
        raise ValueError("Model spec requires exactly model_tag, architecture and checkpoints")
    if payload["model_tag"] != MOE20_TAG:
        raise ValueError(f"Model spec is restricted to {MOE20_TAG}")
    architecture = payload["architecture"]
    if not isinstance(architecture, dict) or set(architecture) != set(MOE20_ARCHITECTURE):
        raise ValueError("Model spec requires the complete reference MoE20 architecture")
    for key, expected in MOE20_ARCHITECTURE.items():
        value = architecture[key]
        valid_type = (type(value) in (int, float) if type(expected) is float else type(value) is type(expected))
        if not valid_type or value != expected:
            raise ValueError(f"Unsupported model spec architecture {key}: {value!r}")
    checkpoints = payload["checkpoints"]
    if (not isinstance(checkpoints, dict) or set(checkpoints) != {"base", "sft"}
            or any(type(value) is not int or value <= 0 for value in checkpoints.values())):
        raise ValueError("Model spec requires actual positive integer base and SFT checkpoint steps")
    if checkpoints["base"] != MOE20_BASE_STEP:
        raise ValueError(f"Model spec requires the completed MoE20 base step {MOE20_BASE_STEP}")
    result = {"model_tag": MOE20_TAG, "architecture": dict(architecture), "checkpoints": dict(checkpoints)}
    if 'routing_control' in payload:
        result['routing_control'] = validate_routing_control(payload['routing_control'])
    return result


def read_model_spec(path):
    return validate_model_spec(json.loads(Path(path).read_text(encoding="utf-8-sig")))


def expected_weight_shapes(architecture, routing_control=None):
    """State dict tensor shapes; no model imports, tensors, GPU or weight allocation."""
    width, depth, vocab = architecture["n_embd"], architecture["n_layer"], architecture["vocab_size"]
    kv_width = architecture["n_kv_head"] * (width // architecture["n_head"])
    shapes = {"transformer.wte.weight": (vocab, width), "lm_head.weight": (vocab, width),
        "resid_lambdas": (depth,), "x0_lambdas": (depth,), "smear_gate.weight": (1, 24),
        "smear_lambda": (1,), "backout_lambda": (1,)}
    for layer in range(depth):
        prefix = f"transformer.h.{layer}"
        shapes.update({f"{prefix}.attn.c_q.weight": (width, width),
            f"{prefix}.attn.c_k.weight": (kv_width, width), f"{prefix}.attn.c_v.weight": (kv_width, width),
            f"{prefix}.attn.c_proj.weight": (width, width)})
        if layer % 2 == (depth - 1) % 2:
            shapes[f"{prefix}.attn.ve_gate.weight"] = (architecture["n_kv_head"], 12)
            shapes[f"value_embeds.{layer}.weight"] = (vocab, kv_width)
        if (layer + 1) % architecture["moe_every"] == 0:
            hidden = round(width * architecture["moe_hidden_mult"])
            shapes[f"{prefix}.mlp.router.weight"] = (architecture["moe_num_experts"], width)
            for expert in range(architecture["moe_num_experts"]):
                shapes[f"{prefix}.mlp.experts.{expert}.c_fc.weight"] = (hidden, width)
                shapes[f"{prefix}.mlp.experts.{expert}.c_proj.weight"] = (width, hidden)
        else:
            shapes[f"{prefix}.mlp.c_fc.weight"] = (4 * width, width)
            shapes[f"{prefix}.mlp.c_proj.weight"] = (width, 4 * width)
    if routing_control is not None:
        validate_routing_control(routing_control)
        shapes[MOE20_R2_BIAS_KEY] = (8,)
    return shapes


def validate_weight_shapes(path, architecture, routing_control=None):
    """Inspect finalized tensor metadata via a CPU memory map before evaluation."""
    import torch
    weights = torch.load(path, weights_only=True, mmap=True, map_location="meta")
    expected = expected_weight_shapes(architecture, routing_control)
    if not isinstance(weights, dict) or set(weights) != set(expected):
        raise ValueError("Checkpoint tensor keys do not match the reference MoE20 architecture")
    for key, shape in expected.items():
        if not isinstance(weights[key], torch.Tensor) or tuple(weights[key].shape) != shape:
            raise ValueError(f"Checkpoint tensor shape disagrees with model spec: {key}")
    evidence = {"tensor_shapes_verified": True, "tensor_count": len(expected), "load": "weights_only CPU mmap/meta"}
    if routing_control is not None:
        bias = torch.load(path, weights_only=True, mmap=True, map_location='cpu')[MOE20_R2_BIAS_KEY]
        if bias.device.type != 'cpu' or bias.dtype != torch.float32 or not torch.isfinite(bias).all().item():
            raise ValueError('Persistent routing control requires actual finite CPU FP32[8] data')
        evidence.update(routing_control=validate_routing_control(routing_control), selection_bias=bias.tolist())
        if routing_control == MOE20_R4_ROUTING_CONTROL:
            router = torch.load(path, weights_only=True, mmap=True, map_location='cpu')[MOE20_HELD_ROUTER_KEY]
            if router.dtype != torch.float32 or tuple(router.shape) != (8, 1280) or not torch.isfinite(router).all().item():
                raise ValueError('R4 held router requires actual finite FP32[8,1280] data')
            evidence['held_router_identity'] = {'shape': [8, 1280], 'dtype': 'torch.float32',
                'sha256': hashlib.sha256(router.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}
    return evidence
