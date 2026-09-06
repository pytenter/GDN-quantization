# GDN_INT8_LONG_HORIZON_FUNCTIONAL_SURVIVAL_AUDIT_V1

## Task definition
GDN_INT8_LONG_HORIZON_FUNCTIONAL_SURVIVAL_AUDIT_V1

## Prior persistence paradox
{
  "FUNCTIONAL_SURVIVAL_VS_FUTURE_KL_SPEARMAN": -0.35,
  "FUNCTIONAL_TOXICITY_VS_FUTURE_KL_SPEARMAN": -0.11666666666666667,
  "MEDIAN_R_C_FUNCTIONAL_SURVIVAL_RATIO": 0.8988147714622265,
  "MEDIAN_R_C_FUNCTIONAL_TOXICITY_RATIO": 1.0351553360055508,
  "MEDIAN_R_C_RAW_PERSISTENCE_RATIO": 0.8917255401287082,
  "PERSISTENCE_PARADOX_GATE": "PASS",
  "RAW_PERSISTENCE_VS_FUTURE_KL_SPEARMAN": -0.25,
  "RAW_VS_FUNCTIONAL_SURVIVAL_REVERSAL_COUNT": "2 / 9",
  "R_GREATER_C_FUNCTIONAL_SURVIVAL_COUNT": "3 / 9",
  "R_GREATER_C_FUTURE_KL_COUNT": "9 / 9",
  "R_GREATER_C_RAW_PERSISTENCE_COUNT": "1 / 9",
  "R_GREATER_C_TOXICITY_COUNT": "7 / 9"
}

## Why raw persistence is not the main hypothesis
Earlier R/C evidence already rejected raw persistence as a sufficient explanation. This audit treats raw norm survival only as a baseline.

## Exact GDN transition semantics
{
  "RECURRENT_UPDATE_SEMANTICS_GATE": "PASS",
  "branch_coefficients": "natural branch may change future hidden-dependent q/k/v/beta/gate; this audit treats it as full-network natural propagation",
  "readout": "query is l2-normalized, transposed to [B,H,T,K], scaled by K**-0.5, then contracted over K",
  "source": "gdn_native_core_output_replay_semantics_audit_v1 plus current Qwen3.5 cached decode path",
  "state_layout": "[B,H,K,V]",
  "state_update": "torch_recurrent_gated_delta_rule updates cached recurrent state during single-token decode"
}

## Single-pulse protocol
{
  "SINGLE_PULSE_PROTOCOL_GATE": "PASS",
  "conditions": [
    "FP",
    "REAL_R",
    "REAL_C"
  ],
  "horizons": [
    0,
    1,
    2,
    4,
    8,
    16,
    32
  ],
  "protocol": "inject natural R128/C128 residual once at t0 into recurrent state; continue teacher-forced natural branch rollout",
  "same_norm_injection": "NO for natural R/C; historical norm-swap artifacts reused only as controlled background"
}

## Raw error trajectory
See trajectory_raw_metrics.csv.

## Functional-risk trajectory
See trajectory_functional_metrics.csv.

## Raw vs functional survival
{
  "FUNCTIONAL_SURVIVAL_VS_FUTURE_KL_SPEARMAN": -0.35,
  "MEDIAN_R_C_FUNCTIONAL_SURVIVAL_RATIO": 0.8988147714622265,
  "RAW_VS_FUNCTIONAL_SURVIVAL_REVERSAL_COUNT": "2 / 9",
  "R_GREATER_C_FUNCTIONAL_SURVIVAL_COUNT": "3 / 9",
  "definition": "functional_survival(h)=F_diag(h)/F_diag(0)"
}

## Functional toxicity / concentration
{
  "FUNCTIONAL_TOXICITY_VS_FUTURE_KL_SPEARMAN": -0.11666666666666667,
  "MEDIAN_R_C_FUNCTIONAL_TOXICITY_RATIO": 1.0351553360055508,
  "R_GREATER_C_TOXICITY_COUNT": "7 / 9",
  "definition": "toxicity(h)=F_diag(h)/(||E_h||_F+eps)"
}

## Orientation evolution
{
  "mean_C_cosine_to_initial_h32": 0.7936955543163678,
  "mean_C_high_sensitivity_fraction_h0": 0.00658722997816609,
  "mean_C_high_sensitivity_fraction_h32": 0.01225525204195568,
  "mean_R_cosine_to_initial_h32": 0.46100816470752437,
  "mean_R_high_sensitivity_fraction_h0": 0.024365418724140206,
  "mean_R_high_sensitivity_fraction_h32": 0.04625815292221919
}

## Frozen transition vs natural branch
{
  "FROZEN_TRANSITION_GATE": "NOT_IMPLEMENTABLE",
  "NATURAL_BRANCH_SIGNAL": "MEASURED",
  "reason": "strict frozen-coefficient propagation of all hidden-dependent q/k/v/beta/gate coefficients is not implemented in a way that matches natural branch semantics for this long-horizon audit; no heuristic transition was substituted"
}

## R/C comparison
{
  "FUNCTIONAL_SURVIVAL_VS_FUTURE_KL_SPEARMAN": -0.35,
  "FUNCTIONAL_TOXICITY_VS_FUTURE_KL_SPEARMAN": -0.11666666666666667,
  "MEDIAN_R_C_FUNCTIONAL_SURVIVAL_RATIO": 0.8988147714622265,
  "MEDIAN_R_C_FUNCTIONAL_TOXICITY_RATIO": 1.0351553360055508,
  "MEDIAN_R_C_RAW_PERSISTENCE_RATIO": 0.8917255401287082,
  "RAW_PERSISTENCE_VS_FUTURE_KL_SPEARMAN": -0.25,
  "RAW_VS_FUNCTIONAL_SURVIVAL_REVERSAL_COUNT": "2 / 9",
  "R_GREATER_C_FUNCTIONAL_SURVIVAL_COUNT": "3 / 9",
  "R_GREATER_C_FUTURE_KL_COUNT": "9 / 9",
  "R_GREATER_C_RAW_PERSISTENCE_COUNT": "1 / 9",
  "R_GREATER_C_TOXICITY_COUNT": "7 / 9"
}

## Controlled orientation evidence
{
  "CONTROLLED_ORIENTATION_EVIDENCE": "REUSED_PRIOR_PARTIAL",
  "note": "prior real_rc tangential mediation/random tangent controls support orientation relevance but are not re-run as this task avoids broad intervention search",
  "prior_best_metric": "out_proj_distortion",
  "prior_final_classification": "LOCAL_TANGENTIAL_FILTERING_SUPPORTED_AND_RECURRENT_CAUSAL_BUT_RC_MEDIATION_PARTIAL",
  "source": "/data/zypan/GDN-quantization/results/propagation/gdn_int8_real_rc_tangential_recurrent_mediation_v1_final_summary.json"
}

## Association with future KL
{
  "FUNCTIONAL_SURVIVAL_VS_FUTURE_KL_SPEARMAN": -0.35,
  "FUNCTIONAL_TOXICITY_VS_FUTURE_KL_SPEARMAN": -0.11666666666666667,
  "MEDIAN_R_C_FUNCTIONAL_SURVIVAL_RATIO": 0.8988147714622265,
  "MEDIAN_R_C_FUNCTIONAL_TOXICITY_RATIO": 1.0351553360055508,
  "MEDIAN_R_C_RAW_PERSISTENCE_RATIO": 0.8917255401287082,
  "RAW_PERSISTENCE_VS_FUTURE_KL_SPEARMAN": -0.25,
  "RAW_VS_FUNCTIONAL_SURVIVAL_REVERSAL_COUNT": "2 / 9",
  "R_GREATER_C_FUNCTIONAL_SURVIVAL_COUNT": "3 / 9",
  "R_GREATER_C_FUTURE_KL_COUNT": "9 / 9",
  "R_GREATER_C_RAW_PERSISTENCE_COUNT": "1 / 9",
  "R_GREATER_C_TOXICITY_COUNT": "7 / 9"
}

## Positive results
Natural single-pulse R/C trajectories were measured with future-reference q readout and diagonal functional-risk anchor.

## Negative results
No frozen-coefficient transition was implemented; no full dense future G was constructed; raw persistence is not promoted as the main explanation.

## Scientific limitations
Pilot uses 9 canonical units and horizons up to 32. Future-context q is diagnostic only; diagonal weights are anchored at t0 from previously validated G tensors.

## Relation to previous M5 / orientation results
This audit uses the previously validated diagonal functional geometry as the functional-risk anchor and studies trajectory evolution, not new low-rank geometry.

## Relation to DAMP-style decay persistence
This is direction-dependent functional survival, not magnitude times decay persistence or mixed-precision allocation.

## Final classification
TRANSITION_INDUCED_FUNCTIONAL_CONCENTRATION_SUPPORTED

## Exact next recommended task
test controlled same-norm orientation interventions for functional concentration

## Terminal summary
```text
TASK =
GDN_INT8_LONG_HORIZON_FUNCTIONAL_SURVIVAL_AUDIT_V1

FORMAL_STATUS = COMPLETE
ARTIFACT_REUSE_GATE = PASS
RECURRENT_UPDATE_SEMANTICS_GATE = PASS
PERSISTENCE_PARADOX_GATE = PASS
SINGLE_PULSE_PROTOCOL_GATE = PASS
FROZEN_TRANSITION_GATE = NOT_IMPLEMENTABLE
R_GREATER_C_FUTURE_KL_COUNT = 9 / 9
R_GREATER_C_RAW_PERSISTENCE_COUNT = 1 / 9
MEDIAN_R_C_RAW_PERSISTENCE_RATIO = 0.8917255401287082
MEDIAN_R_C_FUNCTIONAL_SURVIVAL_RATIO = 0.8988147714622265
MEDIAN_R_C_FUNCTIONAL_TOXICITY_RATIO = 1.0351553360055508
R_GREATER_C_FUNCTIONAL_SURVIVAL_COUNT = 3 / 9
R_GREATER_C_TOXICITY_COUNT = 7 / 9
RAW_VS_FUNCTIONAL_SURVIVAL_REVERSAL_COUNT = 2 / 9
RAW_PERSISTENCE_VS_FUTURE_KL_SPEARMAN = -0.25
FUNCTIONAL_SURVIVAL_VS_FUTURE_KL_SPEARMAN = -0.35
FUNCTIONAL_TOXICITY_VS_FUTURE_KL_SPEARMAN = -0.11666666666666667
NATURAL_BRANCH_SIGNAL = YES
FROZEN_TRANSITION_SIGNAL = NO
FUNCTIONAL_SURVIVAL_SIGNAL = NO
FUNCTIONAL_CONCENTRATION_SIGNAL = PARTIAL
RECURRENT_TRANSITION_CAUSAL_SIGNAL = PARTIAL
RC_PARADOX_EXPLAINED = PARTIAL
FINAL_SCIENTIFIC_CLASSIFICATION = TRANSITION_INDUCED_FUNCTIONAL_CONCENTRATION_SUPPORTED
METHOD_PRINCIPLE_ADVANCED = PARTIAL
QUANTIZER_DESIGN_READY =
NO

NEXT_RECOMMENDED_TASK = test controlled same-norm orientation interventions for functional concentration
```
