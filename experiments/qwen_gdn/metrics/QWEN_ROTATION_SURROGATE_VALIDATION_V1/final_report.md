# Qwen Rotation Surrogate Validation V1

## Executive result

**Raw M5 p95 is not validated as a cross-rotation selection objective.** On the common numerically clean 59-trajectory panel, Hadamard has much better reasoning accuracy (41/59 = 69.5%) than Identity (18/59 = 30.5%), yet its hierarchical M5 p95 is higher (4.21873e+06 vs 15.046). The observable rotation ordering is therefore reversed relative to the hypothesis “lower raw M5 p95 means better reasoning.” M0 p95 and Decision p95 are reversed in the same way.

This does not negate the previous within-rotation result that M5 p95 predicts which trajectories fail (AUROC 0.851). It shows that raw M5 scale is not demonstrably comparable across rotation bases and therefore cannot yet select rotations.

## Fixed analysis contract

The metric aggregation was fixed before joining correctness labels. M0/M5 means average each token's layer mean; p95 is the 95th percentile of each token's layer-p95 (a hierarchical tail); max is the global maximum of token layer-max. CVaR95(M5) is the mean of the worst 5% of token M5 layer-p95 values. Decision uses its direct token statistic. Non-finite risk is treated as +Infinity.

The primary panel is clean59: the sole numerically invalid pair (`aime26_28`, seed 2) is excluded from both rotations without inspecting correctness. Full60 results remain in `metric_results.*`; its M5 CVaR95 is infinite because late overflow enters the worst tail.

## Primary comparison (clean59)

| Rotation | AIME accuracy | M0 p95 | M5 p95 | M5 CVaR95 | Decision p95 |
|---|---:|---:|---:|---:|---:|
| Identity | 18/59 (30.5%) | 106.837 | 15.046 | 22.6865 | 3 |
| Hadamard | 41/59 (69.5%) | 3.10272e+07 | 4.21873e+06 | 4.85954e+09 | 5 |

Candidate-level Spearman for risk versus accuracy is +1.0 for all three p95 metrics, whereas a successful lower-is-better surrogate requires a negative sign. With only two eligible rotations this is just one discordant pair, not an inferential correlation estimate; Pearson and confidence intervals are intentionally not reported. Pairwise ranking consistency is 0/1 for M0, M5, Decision, and M5+Decision.

As a secondary paired-trajectory diagnostic, among the 23 clean trajectories where Hadamard changes an Identity failure into a success, Hadamard has lower trajectory M5 p95 in only 0. This reinforces the cross-rotation scale warning; it is not treated as an independent rotation sample.

## Required answers

1. **What candidates were evaluated?** Identity and Hadamard were fully evaluated with matched INT8-C128 metrics and held-out AIME outcomes. Dense-State and Dense-Functional were audited using existing checkpoints/local evaluations but are ineligible for reasoning correlation. Existing random R_seed0/1/2 matrices were audited but their prior gate had quantization off, so no new expensive evaluation was invented.

2. **Does M5 p95 correlate with reasoning quality?** Not in the required cross-rotation sense. The only comparable pair is discordant: Hadamard is much more accurate while having higher raw M5 p95. The n=2 rotation sample is too small for a reliable correlation magnitude, but it is enough to falsify consistent ordering on this observed pair.

3. **Is M5 p95 better than M0?** No. Both produce the same wrong Identity-before-Hadamard risk ranking (0/1 pairwise consistency). M5's stronger within-rotation failure AUROC does not rescue its cross-rotation calibration.

4. **Does M5 p95 justify becoming the primary rotation objective?** No. Raw M5 p95—and especially raw CVaR95—should not be optimized across rotations until it is normalized or otherwise made basis-comparable and validated on more rotations with genuine held-out reasoning outcomes.

5. **Does Decision sensitivity add useful information?** Not for rotation ranking here. Decision p95 also ranks Identity as safer, and a simple M5+Decision rank combination remains discordant. It may remain a secondary within-trajectory diagnostic.

6. **Is there sufficient evidence to train a new learnable rotation?** No. There are only two fully comparable rotations, one candidate pair is discordant, and Dense/random candidates lack the required joined outcomes. Training now would optimize an unvalidated scale.

7. **What should the next objective be?** Do not yet choose a training objective. First create a cheap, fixed held-out candidate panel (including the existing Dense and random matrices) with the exact same teacher-forced M0/M5/Decision capture plus real pre-existing or separately held-out reasoning outcomes. Pre-register a basis-normalized tail metric—for example M5 divided by an FP functional-scale reference per layer/token—and compare normalized p95/CVaR95 against raw M5, M0, and Decision. Only if lower normalized tail risk ranks multiple rotations consistently should a learnable objective use normalized CVaR95(M5), with Decision as a secondary constraint rather than the primary target.

## Dense-Functional limitation

Whether Dense-Functional improves exact M5 p95 is **not identifiable** from saved artifacts. Its local mean `out_proj_error` is 15.1% below Hadamard and local mean `state_error` is 4.5% below Hadamard; its persistent Future-KL AUC is 23.4% lower. These are not M5 p95 and cannot be relabeled as such. Because no Dense AIME result exists, translation to reasoning quality is also not identifiable.

## Decision

**Do not train against raw CVaR95(M5) yet.** The current evidence supports M5 as a failure-detection diagnostic inside a fixed rotation, but contradicts its use as an unnormalized selector across rotations.
