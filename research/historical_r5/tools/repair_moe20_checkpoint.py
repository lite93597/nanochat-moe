"""CPU-only, fail-closed audit and derivation of the interrupted MoE20 checkpoint.

The command writes a derived checkpoint to a separate output directory. The model and
optimizer ordering are reconstructed by the pinned original GPT.setup_optimizer,
not by matching tensor shapes.  A model-only audit is explicitly incomplete.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

import torch

STEP = 3200
HORIZON = 6641
WORLD_SIZE = 8
TARGET = "transformer.h.19.mlp.router.weight"
SEED = 202610073200
STD = 0.01
LR_FACTOR = 0.1
ORIGINAL_TORCH_VERSION = "2.9.1+cu128"
ARCHITECTURE = {
    "sequence_len": 2048, "vocab_size": 32768, "n_layer": 20,
    "n_head": 10, "n_kv_head": 10, "n_embd": 1280,
    "window_pattern": "L", "moe_num_experts": 8, "moe_top_k": 2,
    "moe_hidden_mult": 2.0, "moe_every": 2, "moe_aux_loss_coef": 0.01,
}
REQUIRED_PINS = {
    "nanochat/gpt.py", "nanochat/optim.py", "scripts/base_train.py",
    "scripts/chat_sft.py", "nanochat/dataloader.py",
    "nanochat/checkpoint_manager.py", "nanochat/moe_training_metrics.py",
    "nanochat/engine.py", "nanochat/moe_report.py", "nanochat/core_eval.py",
    "nanochat/loss_eval.py", "scripts/base_eval.py", "scripts/chat_eval.py",
    "scripts/chat_cli.py", "scripts/infer_bench.py", "nanochat/__init__.py",
    "nanochat/common.py", "nanochat/flash_attention.py",
}


class AuditError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path):
    path = Path(path)
    return {"bytes": path.stat().st_size, "sha256": file_sha(path)}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def checkpoint_names():
    return [f"model_{STEP:06d}.pt", f"meta_{STEP:06d}.json", *[
        f"optim_{STEP:06d}_rank{rank}.pt" for rank in range(WORLD_SIZE)
    ]]


def _safe_relative(value):
    path = Path(value)
    require(not path.is_absolute() and ".." not in path.parts and not path.drive,
            "source pin must be a relative path inside the runtime source")
    return path


def verify_source_pins(runtime_source, pins_path):
    """Accept the deployed manifest or the root's independently saved pin list."""
    runtime_source = Path(runtime_source).resolve()
    payload = json.loads(Path(pins_path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        require(all(isinstance(item, dict) and "path" in item and "sha256" in item for item in payload),
                "invalid source pin list")
        require(len({item["path"] for item in payload}) == len(payload), "duplicate source pins")
        pins = {item["path"]: item["sha256"] for item in payload}
    else:
        require(isinstance(payload, dict), "invalid source pin manifest")
        pins = {}
        for key in ("preserve_reviewed_moe_source", "preserve_training_source", "original_source_sha256"):
            values = payload.get(key, {})
            require(isinstance(values, dict), f"invalid {key}")
            for name, digest in values.items():
                require(name not in pins or pins[name] == digest, "conflicting source pins")
                pins[name] = digest
    require(REQUIRED_PINS <= pins.keys(), "missing original training source pins")
    checked = {}
    for name, digest in sorted(pins.items()):
        require(isinstance(digest, str) and len(digest) == 64 and
                all(c in "0123456789abcdef" for c in digest), f"invalid SHA256 for {name}")
        path = runtime_source / _safe_relative(name)
        require(path.is_file() and not path.is_symlink(), f"missing pinned source: {name}")
        actual = file_identity(path)
        require(actual["sha256"] == digest, f"original source SHA mismatch: {name}")
        checked[name] = actual
    return checked


@contextlib.contextmanager
def original_runtime(runtime_source):
    """Import the actual GPT/optimizer with CUDA detection disabled, no forward."""
    previous = {name: module for name, module in list(sys.modules.items())
                if name == "nanochat" or name.startswith("nanochat.")}
    for name in previous:
        del sys.modules[name]
    old_dtype = os.environ.get("NANOCHAT_DTYPE")
    old_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    os.environ["NANOCHAT_DTYPE"] = "bfloat16"
    sys.path.insert(0, str(Path(runtime_source).resolve()))
    try:
        with patch.object(torch.cuda, "is_available", return_value=False):
            module = importlib.import_module("nanochat.gpt")
        # Setup's shape grouping uses the real eight-rank world size.  No process
        # group or collective is initialized by this structural CPU audit.
        with patch.object(module, "get_dist_info", return_value=(True, 0, 0, WORLD_SIZE)), \
                patch.object(module, "print0", lambda *args, **kwargs: None):
            yield module
    finally:
        sys.path.pop(0)
        for name in list(sys.modules):
            if name == "nanochat" or name.startswith("nanochat."):
                del sys.modules[name]
        sys.modules.update(previous)
        sys.dont_write_bytecode = old_bytecode
        if old_dtype is None:
            os.environ.pop("NANOCHAT_DTYPE", None)
        else:
            os.environ["NANOCHAT_DTYPE"] = old_dtype


def validate_metadata(meta):
    require(isinstance(meta, dict) and type(meta.get("step")) is int and meta["step"] == STEP,
            "requires the actual step-3200 checkpoint")
    require(meta.get("model_config") == ARCHITECTURE, "original MoE20 architecture/aux differs")
    user = meta.get("user_config", {})
    expected = {"num_iterations": HORIZON, "model_tag": "moe_d20_e8_k2",
                "total_batch_size": 524288, "device_batch_size": 8,
                "embedding_lr": 0.3, "unembedding_lr": 0.008, "matrix_lr": 0.02,
                "scalar_lr": 0.5, "weight_decay": 0.28, "router_lr": None,
                "warmup_steps": 40, "warmdown_ratio": 0.65, "final_lr_frac": 0.05,
                "fp8": False}
    require(all(_same_value(user.get(key), value) for key, value in expected.items()),
            "original horizon/batch/learning-rate recipe differs")
    for key in ("moe_num_experts", "moe_top_k", "moe_hidden_mult", "moe_every", "moe_aux_loss_coef"):
        require(user.get(key) == ARCHITECTURE[key], f"user/model configuration differs: {key}")
    require(user.get("max_seq_len") == ARCHITECTURE["sequence_len"] and
            user.get("depth") == ARCHITECTURE["n_layer"], "user/model depth or sequence differs")
    require(meta.get("device_batch_size") == 8 and meta.get("total_batch_size") == 524288 and
            meta.get("max_seq_len") == ARCHITECTURE["sequence_len"], "checkpoint batch/sequence differs")
    cursor = meta.get("dataloader_state_dict")
    require(isinstance(cursor, dict) and set(cursor) == {"pq_idx", "rg_idx", "epoch"},
            "missing original dataloader cursor")
    require(all(type(value) is int for value in cursor.values()), "invalid cursor integer")
    require(0 <= cursor["pq_idx"] < 120 and cursor["rg_idx"] >= 0 and
            cursor["rg_idx"] % WORLD_SIZE == 0 and cursor["epoch"] >= 1, "invalid rank-zero cursor")
    loop = meta.get("loop_state")
    require(isinstance(loop, dict) and set(loop) == {"min_val_bpb", "smooth_train_loss", "total_training_time"},
            "missing original training loop state")
    require(all(type(value) in (float, int) and math.isfinite(value) and value > 0 for value in loop.values()),
            "invalid training loop state")
    require(type(meta.get("val_bpb")) in (float, int) and math.isfinite(meta["val_bpb"]) and meta["val_bpb"] > 0,
            "invalid validation BPB")


def _schedulers(runtime_source, meta):
    """Execute only the pinned pure schedule functions, not the training script."""
    tree = ast.parse((Path(runtime_source) / "scripts/base_train.py").read_text(encoding="utf-8"))
    names = {"get_lr_multiplier", "get_muon_momentum", "get_weight_decay"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    require({node.name for node in functions} == names, "original schedule functions missing")
    namespace = {"args": SimpleNamespace(**meta["user_config"]), "num_iterations": HORIZON,
                 "weight_decay_scaled": meta["user_config"]["weight_decay"], "math": math}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "pinned_base_train_schedulers", "exec"), namespace)
    return namespace


def reconstruct_mapping(runtime_source, meta):
    """Return the original source's exact ordered names, group fields and IDs."""
    with original_runtime(runtime_source) as module:
        with torch.device("meta"):
            model = module.GPT(module.GPTConfig(**meta["model_config"]))
            model.init_weights()
        user = meta["user_config"]
        optimizer = model.setup_optimizer(unembedding_lr=user["unembedding_lr"],
            embedding_lr=user["embedding_lr"], matrix_lr=user["matrix_lr"],
            scalar_lr=user["scalar_lr"], router_lr=user["router_lr"], weight_decay=user["weight_decay"])
        by_object = {id(param): name for name, param in model.named_parameters()}
        expected_tensors = {name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
                            for name, tensor in model.state_dict().items()}
        serialized = optimizer.state_dict()
        schedules = _schedulers(runtime_source, meta)
        groups, parameters = [], {}
        seen = set()
        for index, (live, saved) in enumerate(zip(optimizer.param_groups, serialized["param_groups"], strict=True)):
            require(len(live["params"]) == len(saved["params"]), "source parameter serialization mismatch")
            names = [by_object[id(param)] for param in live["params"]]
            fields = copy.deepcopy({key: value for key, value in saved.items() if key != "params"})
            # A checkpoint at loop step 3200 was saved before update 3200;
            # current group hyperparameters belong to completed update 3199.
            fields["lr"] = fields["initial_lr"] * schedules["get_lr_multiplier"](STEP - 1)
            if fields["kind"] == "muon":
                fields["momentum"] = schedules["get_muon_momentum"](STEP - 1)
                fields["weight_decay"] = schedules["get_weight_decay"](STEP - 1)
            groups.append({"index": index, "parameter_names": names, "parameter_ids": saved["params"],
                           "expected_fields": fields})
            for name, param, identifier in zip(names, live["params"], saved["params"], strict=True):
                require(identifier not in seen and name not in parameters, "duplicate source optimizer mapping")
                seen.add(identifier)
                parameters[name] = {"id": identifier, "group": index,
                                    "shape": list(param.shape), "dtype": str(param.dtype)}
        require(set(parameters) == set(by_object.values()), "source optimizer does not cover all model parameters")
        router_names = [f"transformer.h.{layer}.mlp.router.weight" for layer in range(1, 20, 2)]
        require(all(name in parameters for name in router_names), "ten named original routers missing")
        router_groups = sorted({parameters[name]["group"] for name in router_names})
        for index in router_groups:
            require(groups[index]["expected_fields"]["kind"] == "adamw" and
                    set(groups[index]["parameter_names"]) <= set(router_names), "router group contains other parameters")
        require(parameters[TARGET]["shape"] == [8, ARCHITECTURE["n_embd"]] and
                parameters[TARGET]["dtype"] == "torch.float32", "last router must be the named FP32 [8,width] parameter")
        del optimizer, model
    return {"groups": groups, "parameters": parameters, "model_tensors": expected_tensors,
            "router_names": router_names, "router_group_indices": router_groups,
            "target_optimizer_id": parameters[TARGET]["id"],
            "method": "pinned GPT.named_parameters + GPT.setup_optimizer + Optimizer.state_dict ordered IDs; world_size=8"}


def tensor_identity(tensor):
    require(isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu" and tensor.layout == torch.strided,
            "requires ordinary CPU checkpoint tensors")
    require(tensor.is_floating_point(), "unexpected non-floating checkpoint tensor")
    flat = tensor.detach().contiguous().view(-1)
    digest = hashlib.sha256()
    for block in flat.split(1024 * 1024):
        require(bool(torch.isfinite(block).all()), "non-finite checkpoint tensor")
        digest.update(block.view(torch.uint8).numpy().tobytes())
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "sha256": digest.hexdigest()}


def audit_model(path, mapping):
    data = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    require(isinstance(data, dict) and set(data) == set(mapping["model_tensors"]), "model tensor names differ from pinned GPT")
    result = {}
    for name, expected in mapping["model_tensors"].items():
        actual = tensor_identity(data[name])
        require(actual["shape"] == expected["shape"] and actual["dtype"] == expected["dtype"],
                f"model tensor shape/dtype mismatch: {name}")
        result[name] = actual
    del data
    return result


def _same_value(actual, expected):
    if type(expected) is float:
        return type(actual) in (int, float) and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15)
    if isinstance(expected, (list, tuple)):
        return isinstance(actual, (list, tuple)) and len(actual) == len(expected) and all(_same_value(a, b) for a, b in zip(actual, expected))
    return type(actual) is type(expected) and actual == expected


def optimizer_fingerprint(data):
    """Every tensor and non-tensor field is included in the change proof."""
    require(isinstance(data, dict) and set(data) == {"state", "param_groups"}, "unexpected optimizer top-level fields")
    fields = copy.deepcopy(data["param_groups"])
    states = {}
    for identifier, state in data["state"].items():
        require(type(identifier) is int and isinstance(state, dict), "invalid optimizer state ID")
        states[str(identifier)] = {name: tensor_identity(value) if isinstance(value, torch.Tensor) else value
                                   for name, value in state.items()}
    return {"param_groups": fields, "state": states}


def validate_optimizer(data, mapping, rank):
    require(isinstance(data, dict) and set(data) == {"state", "param_groups"}, "unexpected optimizer top-level fields")
    require(isinstance(data["param_groups"], list) and len(data["param_groups"]) == len(mapping["groups"]), "optimizer group count mismatch")
    expected_states = {}
    for actual, original in zip(data["param_groups"], mapping["groups"], strict=True):
        fields = original["expected_fields"]
        require(isinstance(actual, dict) and set(actual) == {"params", *fields}, "optimizer group fields differ from original source")
        require(actual["params"] == original["parameter_ids"] and
                all(type(value) is int for value in actual["params"]), "ordered optimizer IDs differ from source; shapes cannot establish identity")
        require(all(_same_value(actual[key], value) for key, value in fields.items()),
                f"optimizer group {original['index']} field/schedule mismatch")
        if fields["kind"] == "adamw":
            for name in original["parameter_names"]:
                param = mapping["parameters"][name]
                shape = param["shape"]
                sharded = math.prod(shape) >= 1024 and shape[0] % WORLD_SIZE == 0
                local_shape = [shape[0] // WORLD_SIZE, *shape[1:]] if sharded else shape
                expected_states[param["id"]] = {"keys": {"step", "exp_avg", "exp_avg_sq"},
                    "shapes": {"exp_avg": local_shape, "exp_avg_sq": local_shape}, "dtype": param["dtype"],
                    "name": name, "row_interval": [rank * local_shape[0], (rank + 1) * local_shape[0]] if sharded else None}
        elif fields["kind"] == "muon":
            first = mapping["parameters"][original["parameter_names"][0]]
            rows, cols = first["shape"]
            chunk = math.ceil(len(original["parameter_ids"]) / WORLD_SIZE)
            second = [chunk, rows, 1] if rows >= cols else [chunk, 1, cols]
            expected_states[first["id"]] = {"keys": {"momentum_buffer", "second_momentum_buffer"},
                "shapes": {"momentum_buffer": [chunk, rows, cols], "second_momentum_buffer": second},
                "dtype": first["dtype"], "name": original["parameter_names"][0]}
        else:
            raise AuditError("unknown optimizer kind")
    require(isinstance(data["state"], dict) and set(data["state"]) == set(expected_states),
            "missing/extra rank optimizer state; Muon state belongs only to each group's first parameter")
    for identifier, expected in expected_states.items():
        state = data["state"][identifier]
        require(isinstance(state, dict) and set(state) == expected["keys"], f"optimizer state keys mismatch: ID {identifier}")
        if "step" in state:
            require(type(state["step"]) is int and state["step"] == STEP, f"AdamW step mismatch: ID {identifier}")
        for key, shape in expected["shapes"].items():
            tensor = state[key]
            require(isinstance(tensor, torch.Tensor) and list(tensor.shape) == shape and str(tensor.dtype) == expected["dtype"],
                    f"rank {rank} optimizer slice shape/dtype mismatch: {expected['name']}.{key}")
            if key in ("exp_avg_sq", "second_momentum_buffer"):
                require(bool((tensor >= 0).all()), f"negative optimizer second moment: ID {identifier}")
    fingerprint = optimizer_fingerprint(data)
    target_state = expected_states[mapping["target_optimizer_id"]]
    require(target_state["row_interval"] == [rank, rank + 1], "last router must be one expert row per rank")
    return {"rank": rank, "target_parameter_id": mapping["target_optimizer_id"],
            "target_row_interval": target_state["row_interval"], "target_state_shape": target_state["shapes"]["exp_avg"],
            "fingerprint": fingerprint}


def audit_checkpoint(source, runtime_source, source_pins, *, model_only=False):
    source = Path(source).resolve()
    require(source.is_dir() and not source.is_symlink(), "checkpoint source must be an existing real directory")
    pins = verify_source_pins(runtime_source, source_pins)
    names = checkpoint_names()
    for name in names[:2]:
        require((source / name).is_file() and not (source / name).is_symlink(), f"missing checkpoint: {name}")
    meta = json.loads((source / names[1]).read_text(encoding="utf-8"))
    validate_metadata(meta)
    mapping = reconstruct_mapping(runtime_source, meta)
    model = audit_model(source / names[0], mapping)
    files = {name: file_identity(source / name) for name in names if (source / name).is_file()}
    missing = [name for name in names[2:] if name not in files]
    ranks = []
    if not model_only:
        require(not missing, "actual eight optimizer shards missing; incomplete audit cannot derive a checkpoint")
        for rank, name in enumerate(names[2:]):
            require(not (source / name).is_symlink(), "optimizer source symlink forbidden")
            data = torch.load(source / name, map_location="cpu", weights_only=True, mmap=True)
            ranks.append(validate_optimizer(data, mapping, rank))
            del data
    # Detect changes during the audit, including files read with mmap.
    require(all(file_identity(source / name) == identity for name, identity in files.items()), "source changed during audit")
    return {"format": 1, "status": "incomplete" if model_only else "passed", "model_only": model_only,
            "source": str(source), "world_size": WORLD_SIZE, "step": STEP, "horizon": HORIZON,
            "completed_token_positions": STEP * 524288, "planned_token_positions": HORIZON * 524288,
            "torch_version": str(torch.__version__), "required_apply_torch_version": ORIGINAL_TORCH_VERSION,
            "torch_version_matches_original": str(torch.__version__) == ORIGINAL_TORCH_VERSION,
            "source_pins": pins, "source_files": files, "missing_optimizer_shards": missing,
            "metadata": meta, "dataloader_cursor": meta["dataloader_state_dict"],
            "data_resume_limitations": "Original checkpoint stores rank-zero pq/rg/epoch only. The pinned loader resumes at the next rank-aligned row group; prefetched/doc_buffer contents and RNG states were never checkpointed. Byte-exact data/RNG continuation cannot be claimed.",
            "mapping": mapping, "model_tensor_hashes": model, "optimizer_ranks": ranks,
            "audited_at_epoch": time.time()}


def _model_changes(before, after):
    require(set(before) == set(after), "derived model names changed")
    return [name for name in before if before[name] != after[name]]


def _optimizer_changes(before, after, mapping):
    """Compare the entire structure; permit only the explicitly named recipe."""
    expected = copy.deepcopy(before)
    changed = []
    for index in mapping["router_group_indices"]:
        for field in ("lr", "initial_lr"):
            expected["param_groups"][index][field] *= LR_FACTOR
            changed.append(f"param_groups.{index}.{field}")
    identifier = str(mapping["target_optimizer_id"])
    expected["state"][identifier]["step"] = 0
    changed.append(f"state.{identifier}.step")
    for field in ("exp_avg", "exp_avg_sq"):
        shape = before["state"][identifier][field]["shape"]
        dtype = getattr(torch, before["state"][identifier][field]["dtype"].removeprefix("torch."))
        expected["state"][identifier][field] = tensor_identity(torch.zeros(shape, dtype=dtype, device="cpu"))
        changed.append(f"state.{identifier}.{field}")
    require(after == expected, "derived optimizer changed outside the permitted target state/router group LRs")
    return changed


def apply_repair(source, destination, receipt_path, runtime_source, source_pins):
    """Publish a new derived directory only after complete CPU proof; originals stay immutable.

    If publication succeeds but the external receipt write fails, the internal
    repair_receipt.json remains beside the verified files.  Do not reapply:
    verify_derived_receipt(destination, destination / 'repair_receipt.json') and
    regenerate the external receipt from that JSON. Never delete published data.
    """
    source, destination, receipt_path = Path(source).resolve(), Path(destination).resolve(), Path(receipt_path).resolve()
    require(destination != source and source not in destination.parents and destination not in source.parents,
            "destination must be separate from the original checkpoint tree")
    require(not destination.exists(), "destination already exists; refusing overwrite/reapply; verify its internal repair_receipt.json")
    require(source not in receipt_path.parents and receipt_path != source, "receipt cannot be written into original checkpoint")
    require(destination not in receipt_path.parents and receipt_path != destination,
            "external receipt must be outside unpublished destination")
    require(str(torch.__version__) == ORIGINAL_TORCH_VERSION,
            f"apply requires original Torch {ORIGINAL_TORCH_VERSION}; this CPU environment is {torch.__version__}")
    audit = audit_checkpoint(source, runtime_source, source_pins)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=destination.name + ".staging-", dir=destination.parent))
    published = False
    try:
        mapping, names = audit["mapping"], checkpoint_names()
        data = torch.load(source / names[0], map_location="cpu", weights_only=True, mmap=True)
        generator = torch.Generator(device="cpu").manual_seed(SEED)
        reset = torch.empty(mapping["parameters"][TARGET]["shape"], dtype=torch.float32, device="cpu")
        reset.normal_(mean=0.0, std=STD, generator=generator)
        data[TARGET] = reset
        torch.save(data, staging / names[0])
        del data
        derived_model = audit_model(staging / names[0], mapping)
        changes = _model_changes(audit["model_tensor_hashes"], derived_model)
        require(changes == [TARGET], "derived model must change only the named last router")
        shutil.copyfile(source / names[1], staging / names[1])
        require(file_identity(staging / names[1]) == audit["source_files"][names[1]], "metadata bytes changed")
        optimizer_changes, derived_optimizer_hashes = {}, []
        for rank, name in enumerate(names[2:]):
            data = torch.load(source / name, map_location="cpu", weights_only=True, mmap=True)
            for index in mapping["router_group_indices"]:
                data["param_groups"][index]["lr"] *= LR_FACTOR
                data["param_groups"][index]["initial_lr"] *= LR_FACTOR
            state = data["state"][mapping["target_optimizer_id"]]
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(state["exp_avg"], device="cpu")
            state["exp_avg_sq"] = torch.zeros_like(state["exp_avg_sq"], device="cpu")
            torch.save(data, staging / name)
            del data
            saved = torch.load(staging / name, map_location="cpu", weights_only=True, mmap=True)
            fingerprint = optimizer_fingerprint(saved)
            optimizer_changes[str(rank)] = _optimizer_changes(audit["optimizer_ranks"][rank]["fingerprint"], fingerprint, mapping)
            derived_optimizer_hashes.append({"rank": rank, "fingerprint": fingerprint})
            del saved
        require(all(file_identity(source / name) == identity for name, identity in audit["source_files"].items()),
                "original checkpoint changed during derivation")
        require(verify_source_pins(runtime_source, source_pins) == audit["source_pins"], "original runtime source changed during derivation")
        effective_lr = mapping["groups"][mapping["router_group_indices"][0]]["expected_fields"]["initial_lr"] * LR_FACTOR
        receipt = {"format": 1, "status": "derived", "source": str(source), "destination": str(destination),
                   "world_size": WORLD_SIZE, "step": STEP, "horizon": HORIZON, "source_audit": audit,
                   "source_files": audit["source_files"], "derived_files": {name: file_identity(staging / name) for name in names},
                   "mapping": mapping, "derived_model_tensor_hashes": derived_model,
                   "derived_optimizer_tensor_hashes": derived_optimizer_hashes,
                   "intervention": {"target": TARGET, "seed": SEED, "std": STD, "router_lr_factor": LR_FACTOR,
                       "router_group_indices": mapping["router_group_indices"], "target_optimizer_id": mapping["target_optimizer_id"],
                       "effective_router_initial_lr": effective_lr, "router_cli_baseline_lr": 0.0008,
                       "global_optimizer_resume_step": STEP, "only_target_adamw_step_reset": True},
                   "proof": {"model_changed": changes, "optimizer_changed": optimizer_changes,
                             "metadata_preserved": True, "source_unchanged": True},
                   "created_at_epoch": time.time(), "tool_sha256": file_sha(__file__),
                   "limitations": audit["data_resume_limitations"], "quality_recovery_proven": False}
        atomic_json(staging / "repair_receipt.json", receipt)
        verify_derived_receipt(staging, staging / "repair_receipt.json", check_destination=False)
        require(not destination.exists(), "destination appeared during derivation")
        os.rename(staging, destination)
        published = True
        atomic_json(receipt_path, receipt)
        return receipt
    finally:
        # Only this mkdtemp-created unpublished directory is removed. Never the
        # source tree or a published checkpoint, even if writing external receipt fails.
        if not published and staging.exists():
            require(not staging.is_symlink(), "refusing recursive cleanup of a replaced staging symlink")
            resolved = staging.resolve()
            require(resolved.parent == destination.parent.resolve() and
                    resolved.name.startswith(destination.name + ".staging-") and
                    resolved != source and source not in resolved.parents and resolved not in source.parents,
                    "staging recursive cleanup escaped its verified destination parent")
            shutil.rmtree(resolved)
        gc.collect()


def verify_derived_receipt(checkpoint_dir, receipt_path, *, check_destination=True):
    directory = Path(checkpoint_dir).resolve()
    receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    require(receipt.get("format") == 1 and receipt.get("status") == "derived", "not a derived repair receipt")
    require(receipt.get("step") == STEP and receipt.get("horizon") == HORIZON and receipt.get("world_size") == WORLD_SIZE,
            "repair step/horizon/world mismatch")
    if check_destination:
        require(Path(receipt["destination"]).resolve() == directory, "repair receipt destination mismatch")
    audit = receipt.get("source_audit", {})
    require(audit.get("status") == "passed" and audit.get("model_only") is False and
            audit.get("missing_optimizer_shards") == [] and len(audit.get("optimizer_ranks", [])) == WORLD_SIZE and
            audit.get("torch_version_matches_original") is True and
            audit.get("torch_version") == audit.get("required_apply_torch_version") == ORIGINAL_TORCH_VERSION,
            "repair lacks complete original eight-rank audit")
    require(receipt.get("tool_sha256") == file_sha(__file__), "repair tool SHA differs from proof producer")
    require(set(receipt.get("derived_files", {})) == set(checkpoint_names()), "repair must contain model/meta/all eight optimizer files")
    for name, identity in receipt["derived_files"].items():
        require((directory / name).is_file() and not (directory / name).is_symlink() and
                file_identity(directory / name) == identity, f"derived checkpoint SHA mismatch: {name}")
    intervention, proof = receipt.get("intervention", {}), receipt.get("proof", {})
    require(intervention.get("target") == TARGET and intervention.get("seed") == SEED and
            intervention.get("std") == STD and intervention.get("router_lr_factor") == LR_FACTOR,
            "repair intervention recipe differs")
    require(proof.get("model_changed") == [TARGET] and proof.get("metadata_preserved") is True and
            proof.get("source_unchanged") is True and len(proof.get("optimizer_changed", {})) == WORLD_SIZE,
            "repair allowed-change proof incomplete")
    require(receipt.get("mapping") == audit.get("mapping") and receipt.get("source_files") == audit.get("source_files"),
            "repair mapping/source identities disagree with audit")
    mapping = receipt["mapping"]
    require(intervention.get("target_optimizer_id") == mapping["target_optimizer_id"] and
            intervention.get("router_group_indices") == mapping["router_group_indices"], "repair named optimizer target differs")
    require(_model_changes(audit["model_tensor_hashes"], receipt["derived_model_tensor_hashes"]) == [TARGET],
            "repair per-tensor model proof differs")
    require(REQUIRED_PINS <= audit.get("source_pins", {}).keys(), "repair lacks original source/dependency pins")
    validate_metadata(audit["metadata"])
    require(audit.get("dataloader_cursor") == audit["metadata"]["dataloader_state_dict"], "repair cursor differs from metadata")
    require(len(receipt.get("derived_optimizer_tensor_hashes", [])) == WORLD_SIZE and
            set(proof["optimizer_changed"]) == {str(rank) for rank in range(WORLD_SIZE)}, "repair rank proof incomplete")
    for rank, derived in enumerate(receipt["derived_optimizer_tensor_hashes"]):
        original = audit["optimizer_ranks"][rank]
        require(derived.get("rank") == original.get("rank") == rank and
                original.get("target_row_interval") == [rank, rank + 1], "repair optimizer rank/row proof differs")
        allowed = _optimizer_changes(original["fingerprint"], derived["fingerprint"], mapping)
        require(proof["optimizer_changed"][str(rank)] == allowed, "repair optimizer allowed-change list differs")
    expected_lr = mapping["groups"][mapping["router_group_indices"][0]]["expected_fields"]["initial_lr"] * LR_FACTOR
    require(_same_value(intervention.get("effective_router_initial_lr"), expected_lr), "repair effective router LR differs")
    require(receipt["derived_files"][f"meta_{STEP:06d}.json"] == receipt["source_files"][f"meta_{STEP:06d}.json"],
            "repair metadata changed")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("audit", "apply"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--source", required=True, type=Path)
        sub.add_argument("--runtime-source", required=True, type=Path)
        sub.add_argument("--source-pins", required=True, type=Path)
        if command == "audit":
            sub.add_argument("--output", required=True, type=Path)
            sub.add_argument("--model-only", action="store_true", help="incomplete audit; cannot derive a checkpoint")
        else:
            sub.add_argument("--destination", required=True, type=Path)
            sub.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args(argv)
    output = args.output if args.command == "audit" else args.receipt
    # Refuse any error/success receipt write into the immutable original directory.
    source = args.source.resolve()
    require(source not in output.resolve().parents and output.resolve() != source,
            "receipt cannot be written into original checkpoint")
    try:
        if args.command == "audit":
            receipt = audit_checkpoint(source, args.runtime_source, args.source_pins, model_only=args.model_only)
            atomic_json(output, receipt)
        else:
            receipt = apply_repair(source, args.destination, output, args.runtime_source, args.source_pins)
        print(json.dumps({"status": receipt["status"], "receipt": str(output.resolve()),
                          "step": STEP, "world_size": WORLD_SIZE,
                          "limitations": "No GPU recovery or quality gate performed."}, ensure_ascii=False))
        return 0
    except (AuditError, OSError, RuntimeError, KeyError, TypeError, json.JSONDecodeError) as exc:
        atomic_json(output, {"format": 1, "status": "failed", "command": args.command,
                    "error": str(exc), "source": str(source), "no_training_started": True})
        print(json.dumps({"status": "failed", "receipt": str(output.resolve()), "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
