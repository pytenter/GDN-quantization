# Experiment Index

## Current formal and rotation lines

| Family | Experiment | Code/evidence | Status |
|---|---|---|---|
| Qwen/GDN | AIME26 81,920 FP_STATE / INT8_C128 / Key-Hadamard | qwen3090 server inventory required | COMPLETE; cross-server evidence pending |
| Ling/KDA | AIME26 81,920 FP_STATE / INT8_R128 / Value-Hadamard | `experiments/ling_kda/aime26/81920/`, `results/aime26/81920/ling_kda/` | COMPLETE |
| Ling/KDA | AIME26 256K INT8_R128 + Value-Hadamard | `results/aime26/256k/ling_kda/IN_PROGRESS.md` | IN_PROGRESS |
| Ling/KDA | Corrected Value-Hadamard post-fix rebaseline | `experiments/ling_kda/rotation/value_hadamard/`, `results/rotation/ling_value_hadamard/postfix_rebaseline_v1/` | COMPLETE/PASS |
| Ling/KDA | Architecture/state-semantics audit | `reports/mechanism/ling_kda_architecture_and_state_semantics_audit_v1.md` | COMPLETE_STATIC_RUNTIME_BLOCKED; NEEDS_REVIEW against later runtime evidence |

## Existing Qwen/GDN mechanism evidence

| Stage | Experiment | Script | Result | Report | Status |
|---|---|---|---|---|---|
| Phenomenon | INT8 row E2E completion | `experiments/orientation/analyze_int8_row_e2e_completion_v1.py` | `results/orientation/gdn_int8_row_e2e_completion_v1.json` | `reports/orientation/gdn_int8_row_e2e_completion_v1.md` | COMPLETE |
| Phenomenon | End-to-end bit/axis screening | `experiments/orientation/run_end2end_bit_axis_screening.py` | `results/orientation/gdn_end2end_bit_axis_screening_v1.json` | missing | COMPLETE |
| Mechanism | Orientation state change | `experiments/orientation/run_int8_orientation_state_change_mechanism.py` | `results/orientation/gdn_int8_orientation_state_change_mechanism_v1.json` | `reports/orientation/gdn_int8_orientation_state_change_mechanism_v1.md` | COMPLETE |
| Negative result | Axis geometry rescue | `experiments/orientation/run_int8_axis_geometry_rescue_diagnostic.py` | `results/orientation/gdn_int8_axis_geometry_rescue_v1.json` | `reports/orientation/gdn_int8_axis_geometry_rescue_v1.md` | NEGATIVE/INCONCLUSIVE |
| Metric | Effective-update audit | `experiments/effective_update/run_int8_effective_update_metric_audit.py` | `results/effective_update/gdn_int8_effective_update_metric_audit_v1.json` | `reports/effective_update/gdn_int8_effective_update_metric_audit_v1.md` | COMPLETE |
| Validation | Effective-update multi-prompt validation | `experiments/effective_update/run_int8_effective_update_small_prompt_validation.py` | `results/effective_update/gdn_int8_effective_update_small_prompt_validation_v1.json` | `reports/effective_update/gdn_int8_effective_update_small_prompt_validation_v1.md` | COMPLETE |
| Causal | Residual strength | `experiments/causal_intervention/run_int8_residual_strength_causal_intervention.py` | `results/causal_intervention/gdn_int8_residual_strength_causal_intervention_v1.json` | `reports/causal_intervention/gdn_int8_residual_strength_causal_intervention_v1.md` | SUPPORTED |
| Causal | Residual geometry | `experiments/causal_intervention/run_int8_residual_geometry_causal_intervention.py` | `results/causal_intervention/gdn_int8_residual_geometry_causal_intervention_v1.json` | `reports/causal_intervention/gdn_int8_residual_geometry_causal_intervention_v1.md` | SUPPORTED |
| Dynamics | Single-pulse residual propagation | `experiments/propagation/run_int8_single_pulse_residual_propagation.py` | `results/propagation/gdn_int8_single_pulse_residual_propagation_v1.json` | `reports/propagation/gdn_int8_single_pulse_residual_propagation_v1.md` | SUPPORTED |
| Dynamics | Residual direction sensitivity | `experiments/propagation/run_int8_residual_direction_sensitivity_panel.py` | `results/propagation/gdn_int8_residual_direction_sensitivity_panel_v1.json` | `reports/propagation/gdn_int8_residual_direction_sensitivity_panel_v1.md` | SUPPORTED |
| Readout | Readout-aware propagation audit | `experiments/propagation/run_int8_readout_aware_propagation_audit.py` | `results/propagation/gdn_int8_readout_aware_propagation_audit_v1.json` | `reports/propagation/gdn_int8_readout_aware_propagation_audit_v1.md` | INCONCLUSIVE |

## Versioning cautions

The server contains failed smokes, pre-metric-fix shards, historical KDA basis conventions, and pre-fix endpoint-rotation outputs. They are not automatically merged with the current formal line. See `docs/server_inventory/ling4090.*` for classification and source paths.
