# Result validation

The result tables report aggregate measurements for Dense12, MoE12, and MoE20-R5. Model configuration and checkpoint fingerprints identify the evaluated models.

## Published evidence scope

- Dense12 and MoE12 complete the same 22-task CORE and five-task chat evaluation; the 20-layer R5 run completes base pretraining at update 6641, one full prepared SFT-mixture epoch at update 932, and all 31 evaluation stages.
- Checkpoint validation covers file checksums, Base/SFT tensor shapes, finite router bias and the held-router identity. [Checkpoint fingerprints](../results/checkpoint_identity.json) identify the evaluated weights; see [weight availability](weights.md) for their distribution status.
- All nine public CSVs retain their original measured values. Undefined measurements remain empty. The SHA manifest fingerprints the distributed files.
- The quality and routing figures use the measured aggregates. The recovery figure shows three final-layer diagnostic snapshots: pre-repair pretraining update 3100, final Base update 6641 and final SFT update 932. Different windows and denominators are labeled; the plot is not a reconstructed training trajectory.

## Scientific caveats

Read [the complete metric and limitation notes](../results/README.md) before citing the scores. In particular, SFT validation overlaps parts of final benchmark test pools; the MoE20 model changes width, depth, training amount and repair recipe together; there are no repeated-seed significance claims; Dense inference is measured on different hardware; and short nonempty prefills contain disclosed zero hits.

The release makes no claim that eight-GPU data parallel training is expert parallelism, that each expert has a proven domain specialization, or that all tasks have uniform routing. No raw third-party dataset text or large checkpoint is bundled in GitHub.
