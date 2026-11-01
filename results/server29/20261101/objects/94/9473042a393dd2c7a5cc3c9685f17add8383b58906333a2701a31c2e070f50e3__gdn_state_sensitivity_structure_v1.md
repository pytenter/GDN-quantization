# GDN State Sensitivity Structure Audit V1

## Protocol

- mode: mechanism audit, not method design
- teacher-forced path: FP32 greedy token sequence
- perturbation: one target token, one target layer, one selected head/support
- after perturbation: FP32 execution resumes, no repeated perturbation
- recurrent fake quant target: GDN recurrent state only

## Correctness

- fp32_reference_reproducible: `True`
- fp32_repro_agreement: `1.0`
- teacher_forced_token_path_length: `2048`
- one_shot_perturb_records: `6`
- one_shot_norm_mismatch_max: `4.781006547224122e-08`
- target_layer_head_checked: `{'layer': 9, 'head': 0}`
- full_attention_recurrent_state_record_count: `0`
- full_attention_layers_with_recurrent_state: `[]`
- hooks_used: `False`
- model_source_modified: `False`
- model_weights_modified: `False`

## Propagation Map

Observation: same-norm single-head IID perturbations have heterogeneous recurrent consequences across layers and heads.

Top G128 cases:

| layer | head | G1 | G8 | G128 | AUC | label |
|---:|---:|---:|---:|---:|---:|---|
| 25 | 8 | 1.001 | 0.9885 | 0.9227 | 0.9754 | persistent |
| 17 | 0 | 1.011 | 0.8761 | 0.92 | 0.9423 | persistent |
| 9 | 24 | 1.009 | 0.9817 | 0.8942 | 0.9642 | persistent |
| 17 | 16 | 1.01 | 0.8771 | 0.8742 | 0.9359 | persistent |
| 29 | 8 | 0.9674 | 0.9326 | 0.8683 | 0.9358 | persistent |

Most damped cases:

| layer | head | G1 | G8 | G128 | AUC | label |
|---:|---:|---:|---:|---:|---:|---|
| 25 | 16 | 0.9412 | 0.3655 | 3.458e-07 | 0.4317 | strong_damping |
| 9 | 0 | 0.8726 | 0.5025 | 5.915e-06 | 0.465 | strong_damping |
| 9 | 8 | 0.9276 | 0.6826 | 8.986e-06 | 0.5386 | strong_damping |
| 17 | 31 | 0.8516 | 0.3936 | 1.195e-05 | 0.4124 | strong_damping |
| 9 | 16 | 0.9693 | 0.6902 | 1.867e-05 | 0.5534 | strong_damping |

Alternative explanation: one prompt and one perturbation geometry were used.
What this rules out: total Frobenius norm alone is not sufficient to predict propagation impact.

## Hot/Cold Causal Perturbation

Observation: hot/cold support sensitivity is mixed and must not be equated with raw magnitude without confirmation.

G128 means:

- hot_row: `0.3995`
- cold_row: `0.5375`
- hot_column: `0.5227`
- cold_column: `0.5212`

What this does NOT prove: hot rows/columns are not yet proven causal targets for protection.

## SVD Subspace Causal Perturbation

- random mean G128: `0.5221`
- top1_left mean G128: `0.2537`
- top4_left mean G128: `0.3575`
- top8_left mean G128: `0.4074`
- bottom8_left mean G128: `0.539`
- top8_right mean G128: `0.5194`
- bottom8_right mean G128: `0.5213`

Observation: subspace-dependent propagation was measured with equal-norm perturbations. Treat as mechanism evidence only.

## Hybrid Full Attention Divergence

- 8-column mean FA_ratio: `0.9466`; min `0.8326`; max `1.163`; mean KL `0.001468`
- 4-row mean FA_ratio: `0.9901`; min `0.9277`; max `1.218`; mean KL `0.6925`
- 4-column mean FA_ratio: `0.9454`; min `0.7923`; max `1.164`; mean KL `0.1428`

Do not conclude: FA_ratio < 1 only means hidden divergence decreased across that layer transition, not proven error correction.

## Cross-Prompt Persistence Result

PARTIAL_CONFIRMATION: not run as full 8-prompt cross-prompt audit in this pass. Prior temporal persistence remains a candidate mechanism.

## Bitwidth Confirmation

- 5-bit row: mean KL `0.5806`, top1 `0.7793`, mean state error `0.03534`, first failure `None`
- 4-bit row: mean KL `0.1788`, top1 `0.9634`, mean state error `0.04931`, first failure `None`
- 3-bit row: mean KL `0.5215`, top1 `0.8838`, mean state error `0.05177`, first failure `None`

## Cadence Confirmation

- K=1: mean KL `0.4609`, max KL `8.629`, mean global state divergence `0.2586`
- K=4: mean KL `0.3182`, max KL `8.355`, mean global state divergence `0.1804`
- K=16: mean KL `0.1639`, max KL `5.491`, mean global state divergence `0.1321`
- K=32: mean KL `0.1599`, max KL `4.734`, mean global state divergence `0.1157`

# Research Questions Answered

Q1: Does same-norm state error have different recurrent consequences across layers and heads? Yes, supported by teacher-forced layer/head propagation map.

Q2: Are persistent hot directions also dynamically sensitive directions? Partially unresolved. Hot/cold causal results are mixed and need cross-prompt confirmation.

Q3: Is recurrent sensitivity aligned with dominant state singular subspaces? Partially supported or unresolved depending on the subspace means; more layers/heads are needed.

Q4: Do Full Attention layers systematically damp or amplify GDN-induced representation divergence? Measured in teacher-forced mode; interpretation depends on FA_ratio table and is not yet a correction claim.

Q5: Does current evidence suggest total reconstruction error is insufficient? Yes. Total reconstruction error alone is insufficient to predict recurrent impact.

# Insight Ranking

| Insight | Evidence Strength | Novelty Potential | GDN Specificity | Prior-art Risk | Next Value |
|---|---|---|---|---|---|
| Layer/head propagation heterogeneity | strong | high | high | medium | highest |
| Hot/cold causal sensitivity | moderate | high | high | medium | high |
| SVD subspace causal sensitivity | moderate | high | high | medium | high |
| Hybrid Full Attention transition effect | moderate | high | high | low | high |
| Row/column orientation sensitivity | strong | medium-high | high | medium | high |
| Bitwidth teacher-forced behavior | moderate | medium | high | high | medium |
| Cadence teacher-forced behavior | moderate | medium-high | high | medium | medium |

# Method Implication

If layer/head propagation heterogeneity is confirmed across prompts, future work may consider propagation-aware evaluation objectives.

If SVD or hot/cold sensitivity consistently predicts propagation, future work may consider structure-aware precision allocation. This report does not implement such a method.

# FINAL_RESEARCH_CLASSIFICATION

`MULTIPLE_STRUCTURE_SIGNALS_FOUND`

METHOD_DESIGN_READY: NO

Reason: at least one teacher-forced causal mechanism is supported, but multi-prompt and broader head confirmation are still required before method design.
