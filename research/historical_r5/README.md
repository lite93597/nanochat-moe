# Historical R5 repair implementation

These are the algorithms used to audit and derive the recorded MoE20 recovery
checkpoint. They are retained for source review, not advertised as a generic
one-command recovery recipe.

* `tools/repair_moe20_checkpoint_r5.py`: copy donor-0 FFN weights into seven
  destinations, map their 14 matrices through the actual optimizer ordering,
  and reset the corresponding 28 Muon momentum/second-momentum slices.
* `tools/repair_moe20_checkpoint.py`: source and optimizer audit helper.
* `cloud/calibrate_moe20_router.py`: two-window TRAIN capture and FP64 damped
  Newton router-intercept fitting; the check window is not used for fitting.
* `cloud/model_spec.py`: recorded architecture and routing-control contract.

The historical audit deliberately requires the original source hashes, eight
optimizer shards, metadata, and R4 validation records. The source recovery
checkpoints and validation inputs are not bundled here. The fixed hashes verify
checkpoint and source identity. Reproduction requires the matching inputs.

For your own saved training logits, use `tools/calibrate_router_bias.py` at the
repository root. For a model-only expert-copy experiment, use
`tools/upcycle_experts.py`. Neither portable adapter resets optimizer state,
passes a BPB/routing gate, or creates a resumable checkpoint. A new recovery run
needs its own explicit source/optimizer mapping and quality validation.
