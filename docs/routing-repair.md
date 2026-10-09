# Case study: detecting and recovering expert collapse

[Back to README](../README.md) · [Architecture](architecture.md) · [Experiments](experiments.md)

This documents the actual **MoE20-R5** recovery. It is an engineering intervention on a specific checkpoint, not a claim of a new general routing algorithm or a universal recipe for stable MoE training.

## Failure that scalar loss missed

Around updates 2900–3100, the final MoE layer (zero-based layer 19) assigned about **99.8% of Top-2 selections to two experts**, with load CV about **1.73** and several zero-hit experts. BPB was still improving.

Task loss can keep improving through the favored experts. Meanwhile, infrequently selected experts receive little task gradient, reinforcing the route preference. A ten-layer mean auxiliary loss can also hide one pathological layer.

The diagnostic response was to inspect per-layer counts, minimum share, zero experts, gates/probabilities, finite values, and fixed-window BPB together. Earlier R1–R4 trials did not pass their quality gates; they are not relabeled as successful runs.

## R5 intervention

### 1. Start from a complete source checkpoint

The source was the 3600-step recovery checkpoint, with all required optimizer shards and metadata. The donor was expert 0, chosen from predeclared training-window global gate mass, not selected by final benchmark accuracy.

### 2. Give cold experts a capable initialization

Copy donor `c_fc` and `c_proj` into seven target experts in the last layer: **14 matrix tensors**. Copies have independent parameter storage; aliasing one `Parameter` into eight slots would not create independent experts.

This temporarily reduces expert diversity and changes the layer function on formerly other-expert routes. Different routed inputs/gradients can subsequently separate the experts. There is no immediate evidence that they become named-domain specialists.

### 3. Reset only replaced-parameter optimizer state

Reset the target matrices' Muon momentum and second-momentum state: **28 owned state slices**. Preserve other parameter states, clocks, and source recovery metadata. Replacing weights while keeping their incompatible old momentum would be a different intervention.

### 4. Calibrate an FP32 router bias from training windows

Only layer 19 changes from `softmax(Wx)` to:

$$p(x)=\operatorname{softmax}(Wx+b).$$

Two training windows of 524288 valid positions each were used to fit/check the bias. The solver uses FP64 damped Newton steps, stores FP32 bias, and targets approximately uniform mean softmax probability. This does not guarantee perfectly uniform hard Top-2 counts.

Bias is part of the actual logits: it affects selections, selected normalized gates, and auxiliary probabilities. This is not a separate selection-only bias with unmodified gate probabilities. Inference does not recalibrate it against benchmark examples.

### 5. Apply bounded whole-update feedback

Across all eight ranks and accumulation microbatches, compute valid-position-weighted average router probability $\bar p_e$. Then:

$$\Delta b_e=\operatorname{clip}\left(0.5\left(\frac18-\bar p_e\right),-0.05,0.05\right),$$

$$b\leftarrow b+\Delta b,\qquad b\leftarrow b-\operatorname{mean}(b).$$

This is proportional probability feedback with a capped step. It is not hard-count sign feedback. Centering removes the softmax-invariant common bias shift.

### 6. Hold the last router matrix while states advance

The historical R5 runtime performs the router's actual gradient/Adam update, advances its optimizer state, then restores the held FP32 router matrix bytes. Bias remains adaptive; the upstream representations and other router layers still train.

This is more specific than `requires_grad=False`. It limits last-layer router plasticity and is a source-bound stability measure. The experiment does not prove it beats a simpler weight freeze or isolates a benefit from advancing the hidden optimizer state.

## Acceptance before continuation

The source, derived checkpoint, and a 100-update continuation were checked using the same fixed validation window:

| Check | BPB ↓ |
|---|---:|
| Before repair | 0.920247 |
| After clone/calibration | 0.965741 |
| After 100 updates | 0.908361 |

Donor selection and calibration used training inputs. **Validation BPB was used for quality acceptance**; the whole recovery is not a validation-free operation. Initial quality degradation and an intermediate CV excursion are retained in the record rather than omitted.

After acceptance, the model continued to the complete 6641-update pretraining endpoint and completed 932 updates of the full SFT mixture.

| Final last-layer training diagnostic | Pretrain 6641 | SFT 932 |
|---|---:|---:|
| Load CV | 0.128178 | 0.155499 |
| Minimum assignment share | 10.4135% | 10.1877% |

All eight experts were used in these final diagnostics and full task aggregates. A short prefill with 14 or 60 positions can still have unused experts; this is not sustained training collapse. Task routing remains nonuniform.

## A separate SFT engineering failure

After Base pretraining, the original SFT attempt also encountered a supervised-token counting mismatch, a missing `json` import, and save/barrier problems. These were code-path failures, distinct from the original expert collapse.

The accepted SFT entry is [`scripts/chat_sft_recovery_r1.py`](../bundle/moe/scripts/chat_sft_recovery_r1.py): it treats `N` as valid supervised SFT positions and corrects serialization/synchronization without weakening route quality thresholds. The incomplete first-attempt checkpoint was not used as a valid restart. SFT restarted from complete Base6641 plus its optimizer state and traversed the full mixed epoch.

## Using the code in a new experiment

Historical calibration, derivation, and audit helpers in [`research/historical_r5`](../research/historical_r5/) retain explicit source/configuration/provenance checks. They require the corresponding checkpoint, optimizer state, calibration inputs, and validation records; changing or removing a SHA guard is not reproduction.

The portable root tools [`upcycle_experts.py`](../tools/upcycle_experts.py) and [`calibrate_router_bias.py`](../tools/calibrate_router_bias.py) expose individual operations for your own inputs. The first copies model tensors only; the second fits a bias from two caller-declared TRAIN-logit tensors. Neither resets optimizer shards, establishes a BPB pass, or produces a resumable repair checkpoint.

For a new collapse, first save a complete checkpoint and evidence, identify its layer/count/gradient/quality scope, then design a new intervention with an independent quality gate. Do not blindly copy this last-layer-only recipe into an unrelated model.

An inference export contains final model weights and metadata, not every historical source optimizer. See [weights](weights.md) and [reproduction](reproduction.md) for available scope.
