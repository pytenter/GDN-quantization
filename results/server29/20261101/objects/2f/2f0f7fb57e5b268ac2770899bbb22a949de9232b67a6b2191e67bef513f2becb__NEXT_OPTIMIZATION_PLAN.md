# Next optimization plan

## Primary direction

Change one major factor: **checkpoint selection**, not architecture or BPTT length.

On a fixed, independent non-AIME development corpus, replay identical teacher-forced token paths with fully persistent cache and canonical C128 writeback.  For every candidate checkpoint, compare against Hadamard pairwise at predeclared early and late positions.  Select using a frozen downstream score built from full `linear_attn.out_proj` fidelity and fixed-next-token ΔNLL/margin, with an explicit cap on the number/magnitude of Hadamard regressions.  Do not select on C5-state and C6-functional primaries separately.

This recommendation follows from three observations: learned endpoint behavior has substantial rescue/regression churn; local QDQ differences do not explain that churn; and the recovered formal H64 validation matrix shows objective-aligned specialization but also a concrete C5 state-selection/function-fidelity tradeoff.  Neither retained primary metric is a direct downstream trajectory or regression-control objective.

## Minimal future experiment (not executed here)

1. Freeze an independently sourced non-AIME development set before training or checkpoint inspection.  If no suitable licensed set is already present, record that as a blocker rather than silently downloading or synthesizing one.
2. Preserve Qwen3.5-9B, INT8-C128, H128 initialization, persistent recurrent writeback, seed/determinism, and the frozen deployment path.
3. Keep Hadamard, current C5, and current C6 as controls.
4. Change only the checkpoint-selection rule described above.  Do not simultaneously change loss, parameterization, horizon, or learned layer mask.
5. Report paired rescue/regression identities as well as mean fidelity metrics.  A net score without regression accounting is insufficient.
6. After selection is frozen on non-AIME development data, AIME26 may be used only as a previously exposed diagnostic benchmark, not an untouched test set.

## Falsification rule

Reject this direction if the frozen downstream selection score does not rank Hadamard/current C5/current C6 in a way that predicts paired regressions on held-out non-AIME documents, or if selecting by it fails to reduce regression count without eliminating rescues.  Do not rescue the hypothesis by changing samples, capture positions, or score weights after seeing results.

## Backup direction

Only after validating the selection signal, test a single conservative Hadamard-anchor/trust-region strength chosen entirely on the independent development set.  Do not infer a layer mask from the present AIME panel: the same high-delta layers appeared in both regressions and rescues.

No new training or generation is authorized by this plan.
