"""CPU-only, fail-closed checks for the reference MoE20 routing records.

The caller supplies an architecture already verified against its checkpoint.
Training JSONL contains counters, not enough evidence to verify tensor widths.
No checkpoint, GPU, network, or process operations are performed here.
"""
import hashlib
import json
import math

try:
    from .model_spec import MOE20_ARCHITECTURE, MOE20_TAG
except ImportError:  # The training logger loads this file by its absolute path.
    from model_spec import MOE20_ARCHITECTURE, MOE20_TAG


LAYERS = list(range(1, 20, 2))
GLOBAL_PRETRAIN_TOKENS = 524288
AGGREGATION = "all ranks and all gradient accumulation microbatches in this optimizer step"
R3_MODE = "router_logit_probability_proportional"
R3_FEEDBACK = "global_optimizer_step_probability_proportional_clipped_centered"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, message, positive=False):
    _require(type(value) is int and value >= int(positive), message)
    return value


def _number(value, message):
    _require(type(value) in (int, float) and math.isfinite(value), message)
    return value


def _near(actual, expected, message, tolerance=2e-6):
    _number(actual, message)
    _require(math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance), message)


def _finite_tree(value):
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def _architecture(architecture):
    _require(isinstance(architecture, dict) and set(architecture) == set(MOE20_ARCHITECTURE),
             "Caller must supply the full checkpoint-verified MoE20 architecture")
    for key, expected in MOE20_ARCHITECTURE.items():
        value = architecture[key]
        valid_type = type(value) in (int, float) if type(expected) is float else type(value) is type(expected)
        _require(valid_type and value == expected, f"Unsupported MoE20 architecture: {key}")


def _counter(counter, full=True):
    _require(isinstance(counter, dict), "Counter must be an object")
    n = _integer(counter.get("token_count"), "Invalid token_count")
    assignments = _integer(counter.get("assignment_count"), "Invalid assignment_count")
    _require(counter.get("top_k") == 2 and type(counter.get("top_k")) is int, "Counter top_k must be 2")
    _require(assignments == 2 * n, "assignment_count must equal 2N")
    experts = counter.get("experts")
    _require(isinstance(experts, list) and len(experts) == 8, "Counter requires exactly 8 experts")
    for expert_id, expert in enumerate(experts):
        _require(isinstance(expert, dict) and type(expert.get("expert")) is int
                 and expert["expert"] == expert_id, "Expert indices must be exactly 0..7")
        selections = _integer(expert.get("selections"), "Invalid expert selections")
        _require(selections <= n, "Expert selections must be <= N")
        ratios = [("assignment_share", assignments, selections), ("token_hit_rate", n, selections)]
        if full:
            top1 = _integer(expert.get("top1_count"), "Invalid top1_count")
            _require(top1 <= selections, "Expert top1_count must be <= selections")
            ratios.append(("top1_share", n, top1))
        for key, denominator, numerator in ratios:
            _near(expert.get(key), numerator / denominator if denominator else 0,
                  f"Inconsistent {key}", tolerance=1e-12)
        probability = _number(expert.get("mean_router_probability"), "Invalid mean_router_probability")
        _require(0 <= probability <= 1, "Probability must be in [0,1]")
        if n == 0:
            _near(probability, 0, "Empty counter must have zero router probability", tolerance=1e-12)
        if full:
            gate = _number(expert.get("gate_share"), "Invalid gate_share")
            _require(0 <= gate <= 1, "Gate share must be in [0,1]")
            _require(gate <= expert["token_hit_rate"] + 2e-6, "Gate share exceeds token hit rate")
            if selections == 0:
                _near(gate, 0, "Unselected expert must have zero gate share", tolerance=1e-12)
    _require(sum(e["selections"] for e in experts) == assignments, "Selections do not sum to 2N")
    sums = [("assignment_share", 1), ("token_hit_rate", 2), ("mean_router_probability", 1)]
    if full:
        _require(sum(e["top1_count"] for e in experts) == n, "Top1 counts do not sum to N")
        sums.extend((("top1_share", 1), ("gate_share", 1)))
    for key, expected in sums:
        _near(sum(e[key] for e in experts), expected if n else 0, f"{key} conservation failed")
    unused = [e["expert"] for e in experts if e["selections"] == 0]
    _require(isinstance(counter.get("unused_experts"), list)
             and all(type(item) is int for item in counter["unused_experts"])
             and counter["unused_experts"] == unused, "unused_experts disagrees with selections")
    mean = assignments / 8
    cv = math.sqrt(sum((e["selections"] - mean) ** 2 for e in experts) / 8) / mean if mean else 0
    _near(counter.get("load_cv"), cv, "load_cv disagrees with selection counts", tolerance=1e-12)
    pair = sorted(range(8), key=lambda i: (-experts[i]["selections"], i))[:2]
    return {"token_count": n, "assignment_count": assignments, "load_cv": cv,
            "unused_experts": unused, "largest_pair": pair,
            "largest_pair_share": sum(experts[i]["assignment_share"] for i in pair),
            "pair_0_5_share": experts[0]["assignment_share"] + experts[5]["assignment_share"],
            "assignment_shares": [e["assignment_share"] for e in experts],
            "mean_router_probabilities": [e["mean_router_probability"] for e in experts]}


def _phases(layer):
    phases = layer.get("phases")
    _require(isinstance(phases, dict) and set(phases) == {"prefill", "decode", "other"},
             "Require full prefill/decode/other phase counters")
    for counter in phases.values():
        _counter(counter)
    for key in ("token_count", "assignment_count"):
        _require(sum(counter[key] for counter in phases.values()) == layer[key], f"Phase {key} does not conserve")
    n = layer["token_count"]
    for expert in range(8):
        for key in ("selections", "top1_count"):
            _require(sum(counter["experts"][expert][key] for counter in phases.values())
                     == layer["experts"][expert][key], f"Phase expert {key} does not conserve")
        for key in ("gate_share", "mean_router_probability"):
            weighted = sum(counter["token_count"] * counter["experts"][expert][key]
                           for counter in phases.values()) / n if n else 0
            _near(layer["experts"][expert][key], weighted, f"Phase expert {key} does not conserve")


def _router_diagnostics(diagnostics):
    _require(isinstance(diagnostics, dict) and diagnostics.get("aggregation") == AGGREGATION,
             "Missing global router_diagnostics aggregation")
    for key in ("all_weights_finite", "all_gradients_finite"):
        _require(diagnostics.get(key) is True, f"router_diagnostics.{key} must be true")
    rates = diagnostics.get("router_initial_lrs")
    _require(isinstance(rates, list) and 1 <= len(rates) <= 10, "Require actual router optimizer-group learning rates")
    for rate in rates:
        _require(_number(rate, "Invalid router learning rate") >= 0, "Negative router learning rate")
    layers = diagnostics.get("layers")
    _require(isinstance(layers, list) and all(isinstance(item, dict) for item in layers)
             and [item.get("layer") for item in layers] == LAYERS,
             "router_diagnostics requires all 10 MoE layers")
    for layer in layers:
        for key in ("logits_finite", "probabilities_finite", "weights_finite", "gradients_finite"):
            _require(layer.get(key) is True, f"Layer {layer['layer']} {key} must be true")
        low = _number(layer.get("logit_min"), "Invalid logit_min")
        high = _number(layer.get("logit_max"), "Invalid logit_max")
        _require(low <= high, "Logit range reversed")
        low = _number(layer.get("prob_min"), "Invalid prob_min")
        high = _number(layer.get("prob_max"), "Invalid prob_max")
        _require(0 <= low <= high <= 1, "Probability range must be in [0,1]")
        for key in ("router_gradient_norm", "router_weight_norm"):
            _require(_number(layer.get(key), f"Invalid {key}") >= 0, f"Negative {key}")


def audit_r3_feedback(record):
    """Verify the actual effective-probability controller, including SFT masks."""
    recipe = record.get("metadata", {}).get("load_bias_recipe", {})
    required_recipe = {"moe_load_bias_rate": .5, "moe_load_bias_layer": 19,
        "moe_load_bias_mode": R3_MODE, "moe_load_bias_max_step": .05,
        "selection": "router_logits_plus_bias_softmax", "gate": "biased_router_probs_renormalized",
        "feedback": R3_FEEDBACK, "aux": "biased_router_probs_switch_0.01", "eval": "frozen"}
    _require(isinstance(recipe, dict) and all(recipe.get(key) == value for key, value in required_recipe.items()),
             "Missing or wrong actual R3 probability-controller recipe")
    rows = record.get("load_bias_feedback")
    _require(isinstance(rows, list) and len(rows) == 1 and isinstance(rows[0], dict),
             "R3 requires exactly one complete last-layer feedback update")
    row = rows[0]
    for key, value in (("layer", 19), ("rate", .5), ("mode", R3_MODE), ("max_step", .05),
                       ("world_size", 8), ("aggregation", AGGREGATION), ("mask", "targets != -1"),
                       ("selection", required_recipe["selection"]), ("gate", required_recipe["gate"]),
                       ("feedback", R3_FEEDBACK), ("eval", "frozen")):
        _require(row.get(key) == value, f"R3 actual feedback {key} differs")
    _require(row.get("updated") is True and row.get("all_ranks_bias_consistent") is True,
             "R3 feedback is missing a complete rank-consistent update")
    n = _integer(row.get("global_valid_tokens"), "Invalid R3 valid tokens", positive=True)
    _require(n == record["valid_loss_tokens"], "R3 masked feedback N differs from valid loss tokens")
    counts = row.get("global_expert_counts")
    _require(isinstance(counts, list) and len(counts) == 8
             and all(type(count) is int and 0 <= count <= n for count in counts) and sum(counts) == 2 * n,
             "R3 masked feedback selection counts do not conserve2N")
    sums, means = row.get("global_probability_sums"), row.get("global_mean_router_probs")
    _require(isinstance(sums, list) and isinstance(means, list) and len(sums) == len(means) == 8,
             "R3 requires actual probability sums and means for all8 experts")
    for total, mean in zip(sums, means):
        _require(0 <= _number(total, "Invalid R3 probability sum") <= n
                 and 0 <= _number(mean, "Invalid R3 probability mean") <= 1,
                 "R3 probability mass outside valid range")
        _near(mean, total / n, "R3 mean is not global probability sum / valid N")
    _near(sum(means), 1., "R3 effective probability mass does not conserve")
    _near(row.get("target_probability"), .125, "R3 probability target must be1/8")
    layer = record["routing"]["layers"][-1]
    _require(layer["layer"] == 19, "R3 feedback bound to wrong layer")
    for index, expert in enumerate(layer["experts"]):
        _require(counts[index] <= expert["selections"], "Masked R3 selections exceed full forward count")
        _require(sums[index] / layer["token_count"] <= expert["mean_router_probability"] + 2e-6,
                 "Masked R3 probability mass exceeds full forward mass")
        if record["stage"] == "pretrain":
            _require(n == layer["token_count"] and counts[index] == expert["selections"],
                     "Pretrain R3 feedback does not match full global forward counts")
            _near(means[index], expert["mean_router_probability"],
                  "R3 feedback probability differs from actual effective routing probability")
    before, after = row.get("bias_before"), row.get("bias_after")
    _require(isinstance(before, list) and isinstance(after, list) and len(before) == len(after) == 8,
             "R3 requires finite before/after FP32 bias vectors")
    candidate = [_number(value, "Invalid R3 bias_before") + max(-.05, min(.05, .5 * (.125 - mean)))
                 for value, mean in zip(before, means)]
    center = sum(candidate) / 8
    for actual, expected in zip(after, candidate):
        _near(actual, expected - center, "R3 bias update is not clipped probability feedback then centered")
    # FP32 centering can leave a few ulps when calibrated intercepts are large.
    _require(abs(sum(after)) <= max(2e-6, 2e-7 * sum(abs(value) for value in after)),
             "R3 bias_after is not centered within FP32 arithmetic")
    return {"mode": R3_MODE, "valid_tokens": n, "global_mean_router_probs": means,
            "bias_before": before, "bias_after": after, "feedback": R3_FEEDBACK}


def audit_router_hold(record):
    """Verify per-rank real AdamW updates followed by exact last-router restore."""
    row = record.get("router_hold", {})
    _require(record.get("metadata", {}).get("load_bias_recipe", {}).get("moe_router_hold_layer") == 19,
             "R4 requires explicit last-router hold metadata")
    required = {"layer": 19, "parameter_name": "transformer.h.19.mlp.router.weight",
        "policy": "restore_after_real_optimizer_step",
        "optimizer_state_policy": "real_gradients_moments_and_step_advance", "optimizer_kind": "adamw", "world_size": 8}
    _require(isinstance(row, dict) and all(row.get(key) == value for key, value in required.items()),
             "R4 actual hold policy/world/parameter differs")
    for key in ("optimizer_updated", "weight_exactly_equal", "all_ranks_weight_equal", "all_ranks_state_steps_consistent"):
        _require(row.get(key) is True, f"R4 {key} must be true")
    _require(_number(row.get("weight_max_delta"), "Invalid R4 weight delta") == 0., "Held router weight changed")
    before = _integer(row.get("step_before"), "Invalid actual AdamW step_before")
    after = _integer(row.get("step_after"), "Invalid actual AdamW step_after", positive=True)
    _require(after == before + 1, "R4 must advance actual AdamW clock exactly once")
    if record["stage"] == "pretrain":
        _require(after == record["optimizer_step"] - 3200, "R4 target AdamW clock differs from retained actual300 state")
    for key in ("group_initial_lr", "group_lr"):
        _require(_number(row.get(key), f"Invalid actual R4 {key}") >= 0., "R4 optimizer group rate is invalid")
    proofs = row.get("rank_proofs")
    _require(isinstance(proofs, list) and [item.get("rank") for item in proofs] == list(range(8)),
             "R4 requires eight actual ordered rank proofs")
    weights = set()
    for item in proofs:
        _require(item.get("weight_exactly_equal") is True and item.get("weight_max_delta") == 0.
                 and item.get("step_before") == before and item.get("step_after") == after
                 and item.get("moments_finite") is True and item.get("local_state_shape") == [1, 1280],
                 "R4 rank clock/state shape/weight proof differs")
        weight = item.get("weight_before_sha256")
        _require(isinstance(weight, str) and len(weight) == 64 and all(c in '0123456789abcdef' for c in weight)
                 and item.get("weight_after_sha256") == weight, "R4 rank did not restore exact weight bytes")
        weights.add(weight)
        for phase in ("moments_before_sha256", "moments_after_sha256"):
            moments = item.get(phase, {})
            uninitialized_sft = (record["stage"] == "sft" and before == 0 and after == 1
                                 and phase == "moments_before_sha256" and moments == {"exp_avg": None, "exp_avg_sq": None})
            _require(isinstance(moments, dict) and set(moments) == {"exp_avg", "exp_avg_sq"}
                     and (uninitialized_sft or all(isinstance(value, str) and len(value) == 64
                             and all(c in '0123456789abcdef' for c in value) for value in moments.values())),
                     "R4 lacks actual finite moment fingerprints")
    _require(len(weights) == 1, "Held full router bytes differ across ranks")
    return {"layer": 19, "parameter_name": required["parameter_name"], "policy": required["policy"],
            "optimizer_state_policy": required["optimizer_state_policy"], "world_size": 8, "weight_sha256": weights.pop(),
            "step_before": before, "step_after": after, "rank_proofs": proofs}


def audit_record(record, *, architecture, require_router_diagnostics=False):
    """Validate one complete global record; raise ValueError on invalid evidence."""
    _architecture(architecture)
    _require(isinstance(record, dict) and _finite_tree(record), "Record missing or contains nonfinite numbers")
    _require(type(record.get("schema_version")) is int and record["schema_version"] == 1, "Expected schema_version 1")
    stage = record.get("stage")
    _require(stage in ("pretrain", "sft"), "Expected pretrain or sft stage")
    step = _integer(record.get("optimizer_step"), "Invalid optimizer_step", positive=True)
    if "model_tag" in record:
        _require(record["model_tag"] == MOE20_TAG, "Record model tag mismatch")
    if "architecture" in record:
        _architecture(record["architecture"])
    metadata = record.get("metadata")
    _require(isinstance(metadata, dict) and type(metadata.get("world_size")) is int
             and metadata["world_size"] == 8, "Full world_size=8 required")
    full = metadata.get("routing_detail") == "full_training_forward_counters"
    _require(full or (stage == "sft" and metadata.get("routing_detail") == "selection_counts_and_router_probabilities"),
             "Unknown routing_detail or incomplete pretrain record")
    for key, expected in (("scope", "supervised_loss_tokens"), ("mask", "targets != -1"),
                          ("aggregation", AGGREGATION),
                          ("aux_loss_definition", "mean per-layer Switch load-balancing loss; no z-loss")):
        _require(metadata.get(key) == expected, f"Training metadata {key} mismatch")
    ce = _number(record.get("ce_loss_token_weighted"), "Invalid CE loss")
    aux = _number(record.get("aux_loss_token_weighted"), "Invalid aux loss")
    _require(ce >= 0 and aux >= 0, "Loss values must be nonnegative")
    _near(record.get("total_loss_token_weighted"), ce + .01 * aux, "CE + 0.01 aux != total loss")
    valid = _integer(record.get("valid_loss_tokens"), "Invalid valid_loss_tokens", positive=True)
    microbatches = _integer(record.get("rank_microbatches"), "Invalid rank_microbatches", positive=True)
    _require(microbatches % 8 == 0, "rank_microbatches must include all 8 ranks")
    route = record.get("routing")
    _require(isinstance(route, dict) and type(route.get("num_experts")) is int and route["num_experts"] == 8
             and type(route.get("top_k")) is int and route["top_k"] == 2, "Routing requires E8 K2")
    _require(route.get("label") == f"{stage}_step_{step}", "Routing stage/step label mismatch")
    layers = route.get("layers")
    _require(isinstance(layers, list) and len(layers) == 10
             and all(isinstance(item, dict) and type(item.get("layer")) is int for item in layers)
             and [item["layer"] for item in layers] == LAYERS, "Expected exactly layers 1,3,...,19")
    summaries = []
    for layer in layers:
        counter = _counter(layer, full=full)
        _require(counter["token_count"] > 0, "Empty global training counter")
        if stage == "pretrain":
            _require(counter["token_count"] == GLOBAL_PRETRAIN_TOKENS, "Pretrain requires global N=524288")
        if full:
            _phases(layer)
            _require(layer["phases"]["prefill"]["token_count"] == 0
                     and layer["phases"]["decode"]["token_count"] == 0, "Training counters must be in phase other")
        summaries.append({"layer": layer["layer"], **counter})
    n = summaries[0]["token_count"]
    _require(all(layer["token_count"] == n for layer in summaries), "All layers must observe the same N")
    if stage == "pretrain":
        _require(n % (microbatches * 2048) == 0, "Forward positions do not match global rank_microbatches and sequence length")
    else:
        _require(n == valid, "SFT routing N must equal supervised loss-token count")
    _require(valid <= n and (stage != "pretrain" or valid == n), "Loss-token count inconsistent with full forward N")
    if (require_router_diagnostics and full) or "router_diagnostics" in record:
        _router_diagnostics(record.get("router_diagnostics"))
    result = {"step": step, "stage": stage, "world_size": 8, "token_count": n, "complete_snapshot": full,
            "ce_loss": ce, "aux_loss": aux, "total_loss": record["total_loss_token_weighted"],
            "layers": summaries}
    recipe = metadata.get("load_bias_recipe", {})
    mode = recipe.get("moe_load_bias_mode") if isinstance(recipe, dict) else None
    feedback_mode = any(isinstance(row, dict) and row.get("mode") == R3_MODE
                        for row in record.get("load_bias_feedback", []) or [])
    if full and (mode == R3_MODE or feedback_mode):
        result["routing_control_feedback"] = audit_r3_feedback(record)
    if recipe.get("moe_router_hold_layer", -1) == 19 or "router_hold" in record:
        result["router_hold"] = audit_router_hold(record)
    return result


class RoutingGuard:
    """Stateful stop decision; duplicate, stale and unfinished records add no streak.

    Only adjacent scheduled snapshots contribute to the two-snapshot rule.
    The startup step1/step100 pair is adjacent. Stage changes clear history.
    Once a stop is observed, the stop is latched.
    """
    def __init__(self, *, architecture, snapshot_interval=100, require_router_diagnostics=False):
        _architecture(architecture)
        self.architecture = dict(architecture)
        self.snapshot_interval = _integer(snapshot_interval, "Invalid snapshot_interval", positive=True)
        self.require_router_diagnostics = require_router_diagnostics
        self.last_step = None
        self.last_stage = None
        self.previous_zero = set()
        self.previous_concentration = set()
        self.stopped = False
        self.held_weight_sha256 = None

    def _stop(self, step, issues):
        self.stopped = True
        return {"decision": "stop", "step": step, "issues": issues}

    def observe_line(self, line):
        """Only newline-terminated JSONL is complete; a tail fragment is ignored."""
        if self.stopped:
            return {"decision": "stop", "step": None, "issues": ["guard_stop_latched"]}
        if not isinstance(line, str) or not line.endswith("\n"):
            return {"decision": "ignore", "step": None, "issues": ["unfinished_jsonl_line"]}
        if not line.strip():
            return {"decision": "ignore", "step": None, "issues": ["blank_jsonl_line"]}
        try:
            record = json.loads(line)
        except (ValueError, TypeError):
            return self._stop(None, ["invalid_complete_jsonl"])
        return self.observe(record)

    def observe(self, record):
        step = record.get("optimizer_step") if isinstance(record, dict) else None
        try:
            summary = audit_record(record, architecture=self.architecture,
                                   require_router_diagnostics=self.require_router_diagnostics)
        except (ValueError, KeyError, TypeError, AttributeError, OverflowError) as error:
            return self._stop(step, [f"invalid_snapshot: {error}"])
        if self.stopped:
            return {"decision": "stop", **summary, "issues": ["guard_stop_latched"]}
        if not summary["complete_snapshot"]:
            return {"decision": "ignore", **summary, "issues": ["sft_selection_only_tail_not_full_snapshot"]}
        if self.last_stage == summary["stage"] and self.last_step is not None and step <= self.last_step:
            return {"decision": "ignore", **summary, "issues": ["duplicate_or_out_of_order"]}
        if "router_hold" in summary:
            weight_sha = summary["router_hold"]["weight_sha256"]
            if self.held_weight_sha256 is not None and weight_sha != self.held_weight_sha256:
                return self._stop(step, ["held_router_weight_changed_between_snapshots"])
            self.held_weight_sha256 = weight_sha
        adjacent = (self.last_step is not None and (step - self.last_step == self.snapshot_interval
                    or self.last_step == 1 and step == self.snapshot_interval))
        if self.last_stage != summary["stage"] or not adjacent:
            self.previous_zero = set()
            self.previous_concentration = set()
        zero = {(layer["layer"], expert) for layer in summary["layers"] for expert in layer["unused_experts"]}
        repeated = sorted(zero & self.previous_zero)
        issues = [f"zero_hits: layer={layer} expert={expert}" for layer, expert in sorted(zero)]
        concentrated = set()
        for layer in summary["layers"]:
            if layer["largest_pair_share"] >= .95:
                concentrated.add(layer["layer"])
                issues.append(f"pair_concentration: layer={layer['layer']} pair={layer['largest_pair']} share={layer['largest_pair_share']:.9f}")
        repeated_concentration = sorted(concentrated & self.previous_concentration)
        self.previous_zero = zero
        self.previous_concentration = concentrated
        self.last_step, self.last_stage = step, summary["stage"]
        if repeated or repeated_concentration:
            issues.extend(f"consecutive_zero_hits: layer={layer} expert={expert}" for layer, expert in repeated)
            issues.extend(f"consecutive_pair_concentration: layer={layer}" for layer in repeated_concentration)
            self.stopped = True
        return {"decision": "stop" if self.stopped else "warn" if issues else "continue", **summary,
                "issues": issues, "zero_hits": [list(pair) for pair in sorted(zero)],
                "consecutive_zero_hits": [list(pair) for pair in repeated],
                "consecutive_concentrated_layers": repeated_concentration}


def evaluate_records(records, **guard_kwargs):
    """Replay until the first stop, preserving exactly the live stop decision."""
    guard = RoutingGuard(**guard_kwargs)
    events = []
    for record in records:
        event = guard.observe_line(record) if isinstance(record, str) else guard.observe(record)
        events.append(event)
        if event["decision"] == "stop":
            break
    return {"decision": "stop" if guard.stopped else "warn" if any(e["decision"] == "warn" for e in events) else "continue",
            "first_warn": next((e["step"] for e in events if e["decision"] == "warn"), None),
            "first_stop": next((e["step"] for e in events if e["decision"] == "stop"), None), "events": events}


def _sha(value, message):
    _require(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value), message)


def _bpb_nodes(receipt, architecture, start_step):
    _require(isinstance(receipt, dict) and isinstance(receipt.get("nodes"), dict)
             and set(receipt["nodes"]) == {"before", "after", "update100"}, "Require before/after/update100 BPB nodes")
    nodes = receipt["nodes"]
    shared = None
    for label, step in (("before", start_step), ("after", start_step), ("update100", start_step + 100)):
        node = nodes[label]
        _require(isinstance(node, dict) and _finite_tree(node), "Invalid BPB node")
        _require(type(node.get("schema_version")) is int and node["schema_version"] == 1 and node.get("label") == label
                 and type(node.get("checkpoint_step")) is int and node["checkpoint_step"] == step, "BPB node identity mismatch")
        _require(_number(node.get("bpb"), "Invalid BPB") > 0, "BPB must be positive")
        _require(type(node.get("world_size")) is int and node["world_size"] == 8, "BPB requires 8 ranks")
        budget = _integer(node.get("eval_steps"), "Invalid eval_steps", positive=True)
        batch = _integer(node.get("device_batch_size"), "Invalid validation batch", positive=True)
        _require(type(node.get("sequence_len")) is int and node["sequence_len"] == 2048, "BPB sequence mismatch")
        _require(type(node.get("global_forward_positions")) is int
                 and node["global_forward_positions"] == budget * batch * 2048 * 8, "BPB evaluation budget mismatch")
        ranks = node.get("rank_window_sha256")
        _require(isinstance(ranks, list) and len(ranks) == 8, "BPB requires actual input/target SHA for all 8 ranks")
        for digest in ranks:
            _sha(digest, "Invalid rank window SHA")
        window = hashlib.sha256(json.dumps(ranks, separators=(",", ":")).encode()).hexdigest()
        _require(node.get("window_sha256") == window, "BPB canonical window SHA mismatch")
        _sha(node.get("token_bytes_sha256"), "Missing token-byte SHA")
        identity = node.get("evaluation_identity")
        _require(isinstance(identity, dict), "Missing BPB evaluation identity")
        _architecture(identity.get("architecture"))
        _require(identity["architecture"] == architecture and identity.get("mask") == "targets != -1",
                 "BPB architecture/mask mismatch")
        _sha(identity.get("tokenizer_sha256"), "Missing BPB tokenizer SHA")
        _require(identity.get("dtype") == "torch.bfloat16", "Frozen BPB protocol requires torch.bfloat16 compute")
        if "master_dtype" in identity:
            _require(identity["master_dtype"] == "torch.float32", "Frozen BPB protocol requires torch.float32 master weights")
        source = identity.get("source_sha256")
        _require(isinstance(source, dict) and bool(source), "Missing BPB source hashes")
        for name, digest in source.items():
            _require(isinstance(name, str) and bool(name), "Invalid BPB source path")
            _sha(digest, "Invalid BPB source SHA")
        checkpoint = node.get("checkpoint_identity")
        _require(isinstance(checkpoint, dict) and checkpoint.get("model_tag") == MOE20_TAG
                 and type(checkpoint.get("step")) is int and checkpoint["step"] == step
                 and checkpoint.get("tensor_shapes_verified") is True, "Missing checkpoint shape/identity evidence")
        _sha(checkpoint.get("weights_sha256"), "Missing actual checkpoint weight SHA")
        comparable = {key: node[key] for key in ("rank_window_sha256", "window_sha256", "token_bytes_sha256",
                      "world_size", "eval_steps", "device_batch_size", "sequence_len", "global_forward_positions", "evaluation_identity")}
        if shared is None:
            shared = comparable
        _require(comparable == shared, "BPB nodes did not use identical fixed inputs and evaluation protocol")
    _require(nodes["update100"]["bpb"] <= nodes["before"]["bpb"] * 1.10,
             "Endpoint BPB exceeds frozen baseline by more than 10%")
    return {"before": nodes["before"]["bpb"], "after": nodes["after"]["bpb"],
            "update100": nodes["update100"]["bpb"], "endpoint_limit": nodes["before"]["bpb"] * 1.10,
            "window_sha256": nodes["before"]["window_sha256"], "protocol": "same fixed input/target window across all three nodes"}


def check_recovery_gate(records, *, architecture, start_step=3200, bpb_receipt, held_router_identity=None):
    """Fail closed for the frozen 100-update repair candidate, without training.

    Require global diagnostics at relative updates 1/10/50/100. The reference
    Candidates start at3200/3300/3400/3500/3600; quality thresholds are identical.
    """
    issues, summaries = [], []
    try:
        _architecture(architecture)
        _require(type(start_step) is int and start_step in (3200, 3300, 3400, 3500, 3600),
                 "Recovery is restricted to the reference step3200,3300,3400,3500,3600 candidates")
        records = list(records)
        for record in records:
            summaries.append(audit_record(record, architecture=architecture, require_router_diagnostics=True))
        expected = [start_step + relative for relative in (1, 10, 50, 100)]
        _require([item["step"] for item in summaries] == expected
                 and all(item["stage"] == "pretrain" for item in summaries), "Require exactly pretrain diagnostic updates 1/10/50/100")
        if start_step in (3400, 3500, 3600):
            _require(all(item.get("routing_control_feedback", {}).get("mode") == R3_MODE for item in summaries),
                     "R3 recovery requires actual probability-controller feedback at all four updates")
        if start_step in (3500, 3600):
            _require(isinstance(held_router_identity, dict) and held_router_identity.get("shape") == [8, 1280]
                     and held_router_identity.get("dtype") == "torch.float32"
                     and isinstance(held_router_identity.get("sha256"), str) and len(held_router_identity["sha256"]) == 64,
                     f"Held-router recovery requires the actual source{start_step} tensor identity")
            _require(all(item.get("router_hold", {}).get("weight_sha256") == held_router_identity["sha256"]
                         for item in summaries), f"All four updates must preserve actual source{start_step} router bytes")
        replay = evaluate_records(records, architecture=architecture, require_router_diagnostics=True)
        _require(replay["first_stop"] is None, f"Routing guard stopped within recovery: {replay['first_stop']}")
        last = summaries[-1]["layers"][-1]
        _require(all(share >= .005 for share in last["assignment_shares"]), "Endpoint last-layer expert assignment share below 0.005")
        _require(last["pair_0_5_share"] < .8, "Endpoint last-layer pair 0+5 share must be below 0.8")
        _require(last["load_cv"] < 1, "Endpoint last-layer CV must be below 1")
        _require(all(probability > 1e-4 for probability in last["mean_router_probabilities"]), "Endpoint last-layer mean probability must exceed 1e-4")
        _require(all(not layer["unused_experts"] for layer in summaries[-1]["layers"][:-1]), "Endpoint other layers contain a dead expert")
        bpb = _bpb_nodes(bpb_receipt, architecture, start_step)
    except (ValueError, KeyError, TypeError, AttributeError, IndexError, OverflowError) as error:
        issues.append(str(error))
        bpb = None
    return {"schema_version": 1, "status": "failed" if issues else "passed", "issues": issues,
            "model_tag": MOE20_TAG, "start_step": start_step, "endpoint_step": start_step + 100 if type(start_step) is int else None,
            "diagnostic_steps": [item["step"] for item in summaries],
            "loss_samples": [{key: item[key] for key in ("step", "ce_loss", "aux_loss", "total_loss")} for item in summaries],
            "endpoint": summaries[-1] if summaries else None, "bpb": bpb}
