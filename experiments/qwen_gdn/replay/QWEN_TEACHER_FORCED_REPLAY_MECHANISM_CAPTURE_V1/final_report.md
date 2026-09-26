# Final report — QWEN_TEACHER_FORCED_REPLAY_MECHANISM_CAPTURE_V1

## Scope

All 60 existing FP AIME26 token sequences were replayed under aligned FP_STATE, INT8_C128, and INT8_C128 + Key-Hadamard paths. The capture covers 3,708,902 token-condition rows and all 24 GDN layers. No free-running generation, training, scorer change, quantizer change, or rotation modification was performed.

## Primary results

Failure AUROC on 52 FP-correct trajectories (34 INT8 failures):

- M0 p95: 0.776 (95% CI [0.637453022875817, 0.900157939519894])
- M5 p95: 0.851 (95% CI [0.7315244932432432, 0.946218487394958])
- Persistence p95: 0.154 (95% CI [0.05741059980611064, 0.276072443181818])
- Decision sensitivity p95: 0.824 (95% CI [0.6932773109243697, 0.9378356569900687])
- Equal-weight M5+Persistence+Decision: 0.810

Hadamard rescue AUROC on Group B vs C, using Native-minus-Hadamard risk reduction (23 rescues, 11 failures):

- M0 p95 reduction: 0.668
- M5 p95 reduction: 0.676
- Persistence p95 reduction: 0.656
- Decision p95 reduction: 0.690
- Equal-weight M5+Persistence+Decision reduction: 0.700

## Answers

### 1. Does state error magnitude predict AIME failure?

M0 p95 AUROC is 0.776, with Spearman 0.455. Its confidence interval and the full mean/median/max variants are in `correlation_analysis.json`. This is useful evidence of prediction.

### 2. Does M5 predict failure better than M0?

M5 p95 AUROC is 0.851 versus M0 p95 0.776. On this frozen comparison, M5 outperforms M0. No metric weights or directions were refit after labels.

### 3. Does persistence add information beyond M5?

No. Persistence p95 has AUROC 0.154 in the preregistered higher-is-riskier direction and Spearman -0.571; failures have *lower* normalized persistence ratios. Its partial Spearman given M5 is -0.099. Adding it to M5 lowers AUROC from 0.851 to 0.482. The current ratio is likely denominator-confounded and should not be optimized as a positive penalty.

### 4. Does decision-boundary sensitivity explain Hadamard rescue?

Partially. Decision p95 risk-reduction AUROC is 0.690 (95% CI [0.4946518759018759, 0.8616621376811594]); M0, M5, and persistence reductions are 0.668, 0.676, and 0.656. It is the strongest single rescue separator, but the sample is only 23 rescues versus 11 failures and its interval reaches approximately chance. Exact logit ties are retained through the declared margin floor and reported.

### 5. Most promising rotation objective

Use **upper-tail M5 (M5 p95)** as the primary rotation objective. It is the strongest single failure separator (0.851) and beats M0 p95 (0.776) numerically. Treat decision p95 as a secondary held-out constraint or tie-breaker because it is the best rescue separator. Do **not** include persistence as currently defined: the equal-weight M5+Persistence+Decision composite lowers failure AUROC to 0.810; its small rescue increase to 0.700 is within uncertainty.

### 6. Additional experiment before learning a rotation

Prospectively validate M5 p95, M5+Decision, and a redesigned denominator-independent persistence metric on a held-out reasoning set or preregistered AIME split. First run a non-training candidate-ranking study over existing rotations and predeclare overflow/censoring rules for very long trajectories. Do not train a learnable rotation until the frozen metric ranks both failure and rescue out of sample.

## Numerical limitation

One 81,920-token Group D trajectory (`aime26_28`, seed 2) develops quantized-state overflow in both quantized conditions. This is preserved in the CSV and audited in the JSON. It does not enter either primary statistical population, so the reported failure and rescue results are unchanged.

## Bottom line

Before learning a new recurrent-state rotation, optimize **tail functional amplification (M5 p95)**, not mean representation error and not the current normalized persistence ratio. Use decision-boundary sensitivity as a secondary validation signal. Exact results and confidence intervals are machine-readable in `correlation_analysis.json`; token-level values and overflow audit are in `token_level_metrics.csv` and `token_level_metrics.json`.
