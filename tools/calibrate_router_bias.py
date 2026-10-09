"""Fit a centered router bias from saved TRAIN logits, without loading a model.

This portable adapter reuses the historical R5 calibration solver. It produces
a calibration artifact, not a repaired/resumable checkpoint or a quality pass.
Only use logits captured from training data. Validate on a separate TRAIN window
and run a fixed validation quality check before adopting the bias.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research/historical_r5/cloud"))
from calibrate_moe20_router import fit_intercept, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-logits", type=Path, required=True,
                        help="torch.save tensor with shape N x 8; training data only")
    parser.add_argument("--check-logits", type=Path, required=True,
                        help="independent training window tensor with shape M x 8")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="new, empty destination directory")
    parser.add_argument("--initial-bias", type=Path,
                        help="optional torch.save tensor with shape 8")
    args = parser.parse_args()
    if args.train_logits.resolve() == args.check_logits.resolve():
        parser.error("fit and check must use separate training-window files")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output directory must be new or empty")
    logits = torch.load(args.train_logits, map_location="cpu", weights_only=True)
    check = torch.load(args.check_logits, map_location="cpu", weights_only=True)
    if not isinstance(logits, torch.Tensor) or not isinstance(check, torch.Tensor):
        parser.error("both logits files must contain a tensor")
    if check.ndim != 2 or check.shape[1] != 8 or check.shape[0] == 0 or not torch.isfinite(check).all():
        parser.error("check logits must be finite, nonempty M x 8")
    initial = (torch.load(args.initial_bias, map_location="cpu", weights_only=True)
               if args.initial_bias else torch.zeros(8))
    bias, fit = fit_intercept(logits, initial)
    with torch.no_grad():
        mean = (check.double() + bias.double()).softmax(-1).mean(0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(bias, args.output_dir / "router_bias.pt")
    report = {
        "scope": "training_logits_calibration_only",
        "data_partition": "caller_declares_both_windows_are_training_data",
        "training_examples": logits.shape[0], "check_examples": check.shape[0],
        "train_logits_sha256": sha256(args.train_logits),
        "check_logits_sha256": sha256(args.check_logits),
        "bias_sha256": sha256(args.output_dir / "router_bias.pt"),
        "fit": fit, "check_mean_probability": mean.tolist(),
        "check_max_probability_residual": (mean - 1 / 8).abs().max().item(),
        "quality_gate": "not_run", "optimizer_modified": False,
        "resumable_checkpoint": False,
    }
    (args.output_dir / "calibration.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir),
                      "check_max_probability_residual": report["check_max_probability_residual"],
                      "quality_gate": "not_run"}))


if __name__ == "__main__":
    main()
