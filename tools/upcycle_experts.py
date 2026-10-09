"""Copy one donor FFN to the other experts in a NEW model-only state dict.

This portable tool exposes the R5 expert-copy operation for your own inputs.
It does NOT modify metadata, reset optimizer shards, calibrate routing, or pass
a validation gate. Its output is not a resumable training checkpoint. See the
historical repair tools for the exact optimizer mapping/reset used in R5.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_experts(state, layer, donor, num_experts):
    if layer < 0 or num_experts < 2 or not 0 <= donor < num_experts:
        raise ValueError("require layer >= 0, experts >= 2, and a valid donor")
    copied = dict(state)
    changed = []
    for matrix in ("c_fc", "c_proj"):
        prefix = f"transformer.h.{layer}.mlp.experts"
        source = f"{prefix}.{donor}.{matrix}.weight"
        if source not in state or not isinstance(state[source], torch.Tensor):
            raise ValueError("missing donor tensor: " + source)
        for expert in range(num_experts):
            target = f"{prefix}.{expert}.{matrix}.weight"
            if target not in state or state[target].shape != state[source].shape:
                raise ValueError("missing or incompatible target: " + target)
            if expert != donor:
                copied[target] = state[source].clone()
                changed.append(target)
    return copied, changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path,
                        help="raw tensor state dict saved by NanoChat")
    parser.add_argument("--output", required=True, type=Path, help="new model-only .pt file")
    parser.add_argument("--layer", required=True, type=int, help="zero-based MoE layer")
    parser.add_argument("--donor", required=True, type=int,
                        help="select using a declared training-data diagnostic, not benchmark scores")
    parser.add_argument("--num-experts", type=int, default=8)
    args = parser.parse_args()
    receipt = args.output.with_suffix(args.output.suffix + ".json")
    if args.output.exists() or receipt.exists() or args.output.resolve() == args.weights.resolve():
        parser.error("output and receipt must be new files, outside the immutable input")
    state = torch.load(args.weights, map_location="cpu", mmap=True, weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        parser.error("input must be a raw tensor-only state dict")
    copied, changed = copy_experts(state, args.layer, args.donor, args.num_experts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(copied, args.output)
    report = {"scope": "expert_copy_model_only", "layer": args.layer, "donor": args.donor,
              "num_experts": args.num_experts, "changed_tensors": changed,
              "source_sha256": sha256(args.weights), "output_sha256": sha256(args.output),
              "optimizer_reset": False, "calibration": "not_run", "quality_gate": "not_run",
              "resumable_checkpoint": False}
    receipt.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"changed_tensors": len(changed), "output": str(args.output),
                      "resumable_checkpoint": False}))


if __name__ == "__main__":
    main()
