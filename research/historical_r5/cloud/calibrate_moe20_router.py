"""Capture two real training windows, fit the R3 router intercept on CPU.

No optimizer is constructed or advanced. The source checkpoint is immutable.
The second window is only checked, never used to fit the intercept. Official
validation data is not read. A passed receipt is still not a BPB/recovery gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import torch


R3_MODE = "router_logit_probability_proportional"
R3_RECIPE = {"kind": "last_layer_router_intercept", "layer": 19, "rate": .5,
             "max_step": .05, "score": "router_logits_plus_bias_softmax",
             "gate": "biased_router_probs_renormalized",
             "feedback": "global_optimizer_step_probability_proportional_clipped_centered",
             "aux": "biased_router_probs_switch_0.01", "eval": "frozen",
             "calibration": "training_window_softmax_intercept_newton_float64"}
R4_RECIPE = dict(R3_RECIPE, kind="last_layer_router_intercept_held_router_weight",
                 hold_layer=19, hold_policy="restore_after_real_optimizer_step",
                 optimizer_state_policy="real_gradients_moments_and_step_advance")
RUNTIME_FILES = ("nanochat/gpt.py", "nanochat/common.py", "nanochat/flash_attention.py",
                 "nanochat/checkpoint_manager.py", "nanochat/dataloader.py",
                 "nanochat/tokenizer.py", "nanochat/dataset.py")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path):
    return {"bytes": Path(path).stat().st_size, "sha256": sha256(path)}


def source_identity(checkpoint_dir, step):
    root = Path(checkpoint_dir)
    names = [f"model_{step:06d}.pt", f"meta_{step:06d}.json"]
    names += [f"optim_{step:06d}_rank{rank}.pt" for rank in range(8)]
    return {name: file_identity(root / name) for name in names}


def validate_source_recipe(step, config):
    """Explicit prior recipe; step3600 alone may carry the already-held W."""
    if type(step) is not int or step not in (3400, 3500, 3600):
        raise ValueError('Unsupported immutable calibration source step')
    expected_mode = 'selection_probability_sign' if step == 3400 else R3_MODE
    expected_rate = .005 if step == 3400 else .5
    if (config.get('moe_load_bias_mode', 'selection_probability_sign') != expected_mode
            or config.get('moe_load_bias_rate') != expected_rate
            or config.get('moe_load_bias_layer') != 19
            or config.get('moe_load_bias_max_step', .05) != .05
            or config.get('moe_router_hold_layer', -1) != (19 if step == 3600 else -1)):
        raise ValueError('Calibration source does not match the exact prior R2/R3/R4 recipe')


def _objective(logits, bias):
    scores = logits + bias
    probabilities = scores.softmax(-1)
    mean = probabilities.mean(0)
    objective = scores.logsumexp(-1).mean() - bias.mean()
    return objective, probabilities, mean


def fit_intercept(logits, initial_bias, *, tolerance=1e-6, max_iterations=50):
    """Deterministic damped Newton in the centered eight-dimensional gauge."""
    logits = logits.to(device="cpu", dtype=torch.float64)
    bias = initial_bias.to(device="cpu", dtype=torch.float64).clone()
    if logits.ndim != 2 or logits.shape[1] != 8 or logits.shape[0] == 0:
        raise ValueError("Calibration requires nonempty N x 8 real logits")
    if bias.shape != (8,) or not torch.isfinite(logits).all() or not torch.isfinite(bias).all():
        raise ValueError("Nonfinite logits or initial bias")
    bias -= bias.mean()
    history = []
    gauge = torch.ones(8, 8, dtype=torch.float64) / 8
    ridge = torch.eye(8, dtype=torch.float64) * 1e-10
    for iteration in range(max_iterations + 1):
        objective, probabilities, mean = _objective(logits, bias)
        gradient = mean - 1 / 8
        residual = gradient.abs().max().item()
        node = {"iteration": iteration, "objective": objective.item(),
                "max_probability_residual": residual, "bias": bias.tolist()}
        history.append(node)
        if residual <= tolerance:
            # Runtime stores FP32; certify the actual stored value, not only FP64.
            result = bias.float()
            result -= result.mean()
            actual_residual = (_objective(logits, result.double())[2] - 1 / 8).abs().max().item()
            if actual_residual > tolerance:
                raise RuntimeError("FP32 calibrated bias does not meet the fitted tolerance")
            return result, {"method": "centered_damped_newton_float64", "dtype": "torch.float64",
                "target_mean_probability": 1 / 8, "tolerance": tolerance,
                "maximum_iterations": max_iterations, "ridge": 1e-10,
                "maximum_direction_linf": 2., "armijo": 1e-4,
                "history": history, "converged": True,
                "stored_fp32_max_probability_residual": actual_residual}
        if iteration == max_iterations:
            break
        hessian = torch.diag(mean) - probabilities.T @ probabilities / logits.shape[0]
        direction = torch.linalg.solve(hessian + gauge + ridge, -gradient)
        direction -= direction.mean()
        direction /= max(1., direction.abs().max().item() / 2.)
        slope = gradient.dot(direction).item()
        if not torch.isfinite(direction).all() or slope >= 0:
            raise RuntimeError("Newton direction is not a finite descent direction")
        scale = 1.
        for _ in range(25):
            candidate = bias + direction * scale
            candidate -= candidate.mean()
            candidate_objective = _objective(logits, candidate)[0].item()
            if candidate_objective <= objective.item() + 1e-4 * scale * slope:
                bias = candidate
                node["accepted_scale"] = scale
                break
            scale *= .5
        else:
            raise RuntimeError("Calibration line search did not converge")
    raise RuntimeError("Calibration did not converge within the fixed 50 iterations")


def routing_profile(logits, bias):
    # Match the actual FP32 runtime arithmetic and top-k, including ties.
    logits = logits.to(device="cpu", dtype=torch.float32)
    bias = bias.to(device="cpu", dtype=torch.float32)
    probabilities = (logits + bias).softmax(-1)
    selected = probabilities.topk(2, -1).indices
    counts = torch.bincount(selected.flatten(), minlength=8)
    shares = counts.double() / (2 * len(logits))
    gates = (logits + bias).gather(-1, selected).softmax(-1)
    gate_mass = torch.bincount(selected.flatten(), weights=gates.double().flatten(), minlength=8) / len(logits)
    mean = probabilities.double().mean(0)
    cv = (shares.std(unbiased=False) / shares.mean()).item()
    pair = shares.topk(2).values.sum().item()
    issues = []
    if not torch.isfinite(probabilities).all() or not torch.isfinite(gates).all():
        issues.append("Nonfinite calibrated probabilities or gates")
    if not torch.all(shares >= .005):
        issues.append("Expert assignment share below original 0.005 threshold")
    if cv >= 1:
        issues.append("Load CV reaches original threshold 1")
    if (shares[0] + shares[5]).item() >= .8:
        issues.append("Pair 0+5 reaches original threshold 0.8")
    if pair >= .95:
        issues.append("Largest pair reaches original live threshold 0.95")
    if not torch.all(mean > 1e-4):
        issues.append("Actual router mean probability below original 1e-4 threshold")
    return {"valid_tokens": len(logits), "assignment_count": int(counts.sum()),
            "expert_counts": counts.tolist(), "assignment_shares": shares.tolist(),
            "mean_router_probabilities": mean.tolist(), "gate_shares": gate_mass.tolist(),
            "load_cv": cv, "largest_pair_share": pair,
            "pair_0_5_share": (shares[0] + shares[5]).item(),
            "status": "passed" if not issues else "failed", "issues": issues}


def verify_calibration_receipt(checkpoint_dir, output_dir):
    """Fail closed against actual files and repeat the recorded CPU calculation."""
    root = Path(output_dir)
    receipt = json.loads((root / "calibration_receipt.json").read_text(encoding="utf-8"))
    source_meta_path = Path(checkpoint_dir)
    source_step = receipt.get("step")
    if type(source_step) is not int or source_step not in (3400, 3500, 3600):
        raise ValueError("Calibration source is restricted to actual3400/R3,3500/R4,3600/R5")
    expected_recipe = R3_RECIPE if source_step == 3400 else R4_RECIPE
    expected_kind = {3400: 'moe20_r3_router_intercept_calibration',
                     3500: 'moe20_r4_router_intercept_calibration',
                     3600: 'moe20_r5_router_intercept_calibration'}[source_step]
    if (receipt.get("status") != "passed"
            or receipt.get('kind') != expected_kind
            or receipt.get("world_size") != 8 or receipt.get("routing_control") != expected_recipe
            or receipt.get("optimizer_updates") != 0 or receipt.get("official_validation_used") is not False
            or receipt.get("source_cursor_unchanged") is not True
            or receipt.get("dtype") != 'torch.bfloat16' or receipt.get("master_dtype") != 'torch.float32'):
        raise ValueError("Calibration receipt does not certify the exact reference R3 protocol")
    if receipt["source_files"] != source_identity(checkpoint_dir, source_step):
        raise ValueError("Actual source checkpoint or an optimizer shard changed")
    if set(receipt["output_files"]) != {"calibrated_bias.pt", "fit_logits.pt", "check_logits.pt", "source_audit.json", "data_manifest.json"}:
        raise ValueError("Calibration requires exactly its five actual output files")
    for name, identity in receipt["output_files"].items():
        if name not in ("calibrated_bias.pt", "fit_logits.pt", "check_logits.pt", "source_audit.json", "data_manifest.json"):
            raise ValueError("Unknown calibration output file")
        if identity != file_identity(root / name):
            raise ValueError("Calibration output identity changed")
    audit = json.loads((root / "source_audit.json").read_text(encoding="utf-8"))
    if audit.get("status") != "passed" or audit.get("model_only") is not False or audit.get("step") != source_step:
        raise ValueError("Calibration requires the real complete 3400 eight-optimizer source audit")
    if audit.get("source_files") != receipt["source_files"]:
        raise ValueError("Source audit does not bind the captured checkpoint")
    actual_meta = json.loads((source_meta_path / f"meta_{source_step:06d}.json").read_text(encoding="utf-8"))
    if actual_meta.get('step') != source_step:
        raise ValueError("Source step is not the actual checkpoint metadata step")
    validate_source_recipe(source_step, actual_meta['model_config'])
    if source_step == 3600 and (actual_meta.get('routing_control') not in (None, R4_RECIPE)
            or audit.get('target_adamw_step') != 400 or audit.get('other_adamw_step') != 3600
            or len(audit.get('optimizer_ranks', [])) != 8):
        raise ValueError("R5 calibration requires the genuine held-router3600 eight-optimizer source")
    if receipt["source_dataloader_cursor"] != actual_meta["dataloader_state_dict"]:
        raise ValueError("Calibration does not preserve the actual source data cursor")
    if receipt["data_manifest_sha256"] != sha256(root / "data_manifest.json"):
        raise ValueError("Captured training-data manifest identity differs")
    if receipt["capture_utility_sha256"] != sha256(__file__):
        raise ValueError("Captured utility differs from the audited replay producer")
    runtime = Path(receipt["runtime_source"])
    if receipt["runtime_source_sha256"] != {name: sha256(runtime / name) for name in RUNTIME_FILES}:
        raise ValueError("Runtime capture sources changed")
    bias = torch.load(root / "calibrated_bias.pt", map_location="cpu", weights_only=True)
    if bias.shape != (8,) or bias.dtype != torch.float32 or not torch.isfinite(bias).all():
        raise ValueError("Derived bias must be actual finite FP32[8]")
    if bias.tolist() != receipt["bias_after"] or abs(bias.double().mean().item()) > 1e-6:
        raise ValueError("Calibrated bias receipt or centered gauge mismatch")
    if (receipt["windows"]["fit"]["rank_window_sha256"] == receipt["windows"]["check"]["rank_window_sha256"]
            or not receipt["solver"].get("converged")):
        raise ValueError("Independent check window or genuine convergence missing")
    for role in ("fit", "check"):
        window = receipt["windows"][role]
        if (window.get("split") != "train" or window.get("fresh_loader") is not True
                or window.get("valid_tokens") != 524288 or window.get("microbatches_per_rank") != 4
                or window.get("device_batch_size") != 8 or window.get("sequence_len") != 2048):
            raise ValueError("Calibration window is not the declared complete training protocol")
        hashes = window["rank_window_sha256"]
        if len(hashes) != 8 or any(len(value) != 64 for value in hashes):
            raise ValueError("Missing actual eight-rank input/target identities")
        window_sha = hashlib.sha256(json.dumps(hashes, separators=(",", ":")).encode()).hexdigest()
        if window["window_sha256"] != window_sha or window["used_to_fit"] is not (role == "fit"):
            raise ValueError("Training-window identity or fit-role mismatch")
        logits = torch.load(root / f"{role}_logits.pt", map_location="cpu", weights_only=True)
        if logits.shape != (524288, 8) or logits.dtype != torch.float32 or not torch.isfinite(logits).all():
            raise ValueError("Capture is not the complete real eight-rank global window")
        profile = routing_profile(logits, bias)
        claimed = window["profile"]
        if (profile["status"] != "passed" or profile.keys() != claimed.keys()
                or any(profile[key] != claimed[key] for key in
                       ("valid_tokens", "assignment_count", "expert_counts", "status", "issues"))):
            raise ValueError("Actual frozen calibrated routing failed or profile differs")
        for key in ("assignment_shares", "mean_router_probabilities", "gate_shares",
                    "load_cv", "largest_pair_share", "pair_0_5_share"):
            actual = profile[key] if isinstance(profile[key], list) else [profile[key]]
            stored = claimed[key] if isinstance(claimed[key], list) else [claimed[key]]
            if len(actual) != len(stored) or any(not math.isfinite(value) or abs(value - expected) > 1e-10
                                               for value, expected in zip(stored, actual)):
                raise ValueError("Actual frozen calibration numeric profile differs")
        if role == "fit" and max(abs(value - 1 / 8) for value in profile["mean_router_probabilities"]) > 1e-6:
            raise ValueError("Actual stored bias does not fit the declared convex target")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--step", type=int, required=True, choices=(3400, 3500, 3600))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--runtime-source", required=True, type=Path)
    parser.add_argument("--source-audit", required=True, type=Path)
    parser.add_argument("--data-manifest", required=True, type=Path)
    parser.add_argument("--global-tokens", type=int, default=524288, choices=(524288,))
    parser.add_argument("--device-batch-size", type=int, default=8, choices=(8,))
    args = parser.parse_args()
    sys.path.insert(0, str(args.runtime_source.resolve()))
    os.environ["NANOCHAT_DTYPE"] = "bfloat16"
    from nanochat.common import compute_init, compute_cleanup, COMPUTE_DTYPE, get_base_dir
    from nanochat.checkpoint_manager import build_model
    from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit
    import torch.distributed as dist

    _, rank, _, world, device = compute_init("cuda")
    if world != 8 or COMPUTE_DTYPE != torch.bfloat16:
        raise RuntimeError("Calibration requires exactly eight CUDA ranks and BF16 compute")

    def rank_zero(function):
        packet = [None]
        if rank == 0:
            try:
                packet[0] = {"result": function()}
            except Exception as error:
                packet[0] = {"error": f"{type(error).__name__}: {error}"}
        dist.broadcast_object_list(packet, src=0, device=device)
        if "error" in packet[0]:
            raise RuntimeError(packet[0]["error"])
        return packet[0]["result"]

    try:
        def prepare():
            source = args.checkpoint_dir.resolve()
            output = args.output_dir.resolve()
            if output == source or source in output.parents:
                raise ValueError("Calibration output must be outside the immutable checkpoint directory")
            if args.output_dir.exists():
                raise FileExistsError("Calibration output already exists; never overwrite a prior capture")
            audit = json.loads(args.source_audit.read_text(encoding="utf-8"))
            identity = source_identity(args.checkpoint_dir, args.step)
            if (audit.get("status") != "passed" or audit.get("model_only") is not False
                    or audit.get("world_size") != 8 or audit.get("step") != args.step
                    or audit.get("source_files") != identity):
                raise ValueError("Source audit is not the actual complete 3400 checkpoint")
            if args.step == 3600 and (audit.get('target_adamw_step') != 400
                    or audit.get('other_adamw_step') != 3600 or len(audit.get('optimizer_ranks', [])) != 8):
                raise ValueError("R5 capture requires the actual complete held-router3600 optimizer audit")
            args.output_dir.mkdir(parents=True)
            shutil.copyfile(args.source_audit, args.output_dir / "source_audit.json")
            shutil.copyfile(args.data_manifest, args.output_dir / "data_manifest.json")
            return identity
        identity = rank_zero(prepare)
        model, tokenizer, meta = build_model(str(args.checkpoint_dir), args.step, device, phase="eval")
        if (meta.get("step") != args.step or model.config.n_layer != 20 or model.config.moe_num_experts != 8
                or model.config.moe_top_k != 2 or model.config.sequence_len != 2048
                or model.config.moe_load_bias_layer != 19):
            raise ValueError("Capture source does not match the actual MoE20/R2 step3400 identity")
        validate_source_recipe(args.step, vars(model.config))
        original_cursor = dict(meta["dataloader_state_dict"])
        layer = model.transformer.h[19].mlp
        initial_bias = layer.selection_bias.detach().cpu().clone()
        # Fresh training loader: this is deliberately separate from the source
        # resume cursor and from the official validation loader/window.
        loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(tokenizer,
            args.device_batch_size, 2048, split="train", device=device, resume_state_dict=None)
        microbatches = args.global_tokens // (args.device_batch_size * 2048 * world)
        windows = {}
        active = {"mask": None, "pieces": []}
        def capture(_module, _inputs, logits):
            active["pieces"].append(logits.detach()[active["mask"]].cpu())
        hook = layer.router.register_forward_hook(capture)
        try:
            for role in ("fit", "check"):
                digest = hashlib.sha256()
                active["pieces"] = []
                positions = []
                with torch.no_grad():
                    for _ in range(microbatches):
                        x, y, position = next(loader)
                        for value in (x, y):
                            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
                        active["mask"] = (y != -1).reshape(-1)
                        model(x, y)
                        positions.append(position)
                local = torch.cat(active["pieces"]).to(device=device, dtype=torch.float32)
                invalid = torch.tensor(int(local.shape != (args.global_tokens // world, 8)
                                           or not torch.isfinite(local).all()), device=device)
                dist.all_reduce(invalid)
                if invalid.item():
                    raise RuntimeError("Incomplete or nonfinite rank capture")
                gathered = [torch.empty_like(local) for _ in range(world)] if rank == 0 else None
                dist.gather(local, gather_list=gathered, dst=0)
                hashes, cursors = [None] * world, [None] * world
                dist.all_gather_object(hashes, digest.hexdigest())
                dist.all_gather_object(cursors, positions)
                if rank == 0:
                    logits = torch.cat(gathered).cpu()
                    torch.save(logits, args.output_dir / f"{role}_logits.pt")
                    windows[role] = {"split": "train", "fresh_loader": True,
                        "used_to_fit": role == "fit", "rank_window_sha256": hashes,
                        "window_sha256": hashlib.sha256(json.dumps(hashes, separators=(",", ":")).encode()).hexdigest(),
                        "rank_dataloader_positions": cursors, "valid_tokens": args.global_tokens,
                        "microbatches_per_rank": microbatches, "device_batch_size": 8, "sequence_len": 2048}
        finally:
            hook.remove()

        def solve_and_record():
            torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))
            logits = torch.load(args.output_dir / "fit_logits.pt", map_location="cpu", weights_only=True)
            calibrated, solver = fit_intercept(logits, initial_bias)
            torch.save(calibrated, args.output_dir / "calibrated_bias.pt")
            for role in ("fit", "check"):
                raw = torch.load(args.output_dir / f"{role}_logits.pt", map_location="cpu", weights_only=True)
                windows[role]["profile"] = routing_profile(raw, calibrated)
            if source_identity(args.checkpoint_dir, args.step) != identity:
                raise RuntimeError("Immutable source checkpoint or optimizer changed during capture")
            passed = all(window["profile"]["status"] == "passed" for window in windows.values())
            receipt = {"format": 1, "kind": {3400: "moe20_r3_router_intercept_calibration",
                       3500: "moe20_r4_router_intercept_calibration", 3600: "moe20_r5_router_intercept_calibration"}[args.step],
                "status": "passed" if passed else "failed", "step": args.step,
                "recorded_at_epoch": time.time(), "world_size": 8, "routing_control": (R3_RECIPE if args.step == 3400 else R4_RECIPE),
                "checkpoint_dir": str(args.checkpoint_dir.resolve()), "source_files": identity,
                "source_dataloader_cursor": original_cursor, "source_cursor_unchanged": True,
                "runtime_source": str(args.runtime_source.resolve()),
                "runtime_source_sha256": {name: sha256(args.runtime_source / name) for name in RUNTIME_FILES},
                "capture_utility_sha256": sha256(__file__),
                "tokenizer_sha256": sha256(Path(get_base_dir()) / "tokenizer/tokenizer.pkl"),
                "data_manifest_sha256": sha256(args.data_manifest), "dtype": str(COMPUTE_DTYPE),
                "master_dtype": str(next(model.parameters()).dtype),
                "bias_before": initial_bias.tolist(), "bias_after": calibrated.tolist(),
                "optimizer_updates": 0, "official_validation_used": False,
                "solver": solver, "windows": windows,
                "output_files": {name: file_identity(args.output_dir / name) for name in
                    ("calibrated_bias.pt", "fit_logits.pt", "check_logits.pt", "source_audit.json", "data_manifest.json")},
                "limitations": "Calibration certifies frozen routing on two training windows; it does not certify BPB, inference routing or successful 100-update recovery."}
            (args.output_dir / "calibration_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
            if passed:
                verify_calibration_receipt(args.checkpoint_dir, args.output_dir)
            print(json.dumps({"status": receipt["status"], "fit_cv": windows["fit"]["profile"]["load_cv"],
                              "check_cv": windows["check"]["profile"]["load_cv"], "optimizer_updates": 0}), flush=True)
            return passed
        passed = rank_zero(solve_and_record)
        if not passed:
            raise RuntimeError("Actual frozen calibration/check routing failed the original thresholds")
    finally:
        compute_cleanup()


if __name__ == "__main__":
    main()
