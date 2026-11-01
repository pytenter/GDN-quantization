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
- teacher_forced_token_path_length: `192`
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
| 9 | 0 | 0.9431 | 0.3747 | None | 0.5614 | unknown |

Most damped cases:

| layer | head | G1 | G8 | G128 | AUC | label |
|---:|---:|---:|---:|---:|---:|---|
| 9 | 0 | 0.9431 | 0.3747 | None | 0.5614 | unknown |

Alternative explanation: one prompt and one perturbation geometry were used.
What this rules out: total Frobenius norm alone is not sufficient to predict propagation impact.

## Hot/Cold Causal Perturbation

Observation: hot/cold support sensitivity is mixed and must not be equated with raw magnitude without confirmation.

G128 means:

- hot_row: `None`
- cold_row: `None`
- hot_column: `None`
- cold_column: `None`

What this does NOT prove: hot rows/columns are not yet proven causal targets for protection.

## SVD Subspace Causal Perturbation

- random mean G128: `None`
- top1_left mean G128: `None`
- top4_left mean G128: `None`
- top8_left mean G128: `None`
- bottom8_left mean G128: `None`
- top8_right mean G128: `None`
- bottom8_right mean G128: `None`

Observation: subspace-dependent propagation was measured with equal-norm perturbations. Treat as mechanism evidence only.

## Hybrid Full Attention Divergence

- 8-column mean FA_ratio: `0.9996`; min `0.9072`; max `1.11`; mean KL `0.0002296`
- 4-row mean FA_ratio: `1.004`; min `0.9293`; max `1.167`; mean KL `0.1931`

Do not conclude: FA_ratio < 1 only means hidden divergence decreased across that layer transition, not proven error correction.

## Cross-Prompt Persistence Result

PARTIAL_CONFIRMATION: not run as full 8-prompt cross-prompt audit in this pass. Prior temporal persistence remains a candidate mechanism.

## Bitwidth Confirmation

- 5-bit row: mean KL `0.2382`, top1 `0.9062`, mean state error `0.05693`, first failure `None`
- 4-bit row: mean KL `0.3959`, top1 `0.875`, mean state error `0.08216`, first failure `None`
- 3-bit row: mean KL `0.638`, top1 `0.8125`, mean state error `0.1032`, first failure `None`

## Cadence Confirmation

- K=1: mean KL `0.3959`, max KL `4.894`, mean global state divergence `0.1931`
- K=4: mean KL `0.2201`, max KL `5.227`, mean global state divergence `0.1473`
- K=16: mean KL `0.1011`, max KL `2.224`, mean global state divergence `0.1066`
- K=32: mean KL `0.07236`, max KL `2.618`, mean global state divergence `0.08943`

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
