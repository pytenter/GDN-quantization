# LING_L6_L7_CANONICAL_60_CONFIRMATORY_V1 Final Report

## Canonical endpoint results

| Condition | Correct / 60 | Accuracy |
|---|---:|---:|
| FP | 14/20 available subset | 70.00% on subset |
| Hadamard | 35/60 | 58.33% |
| L6 | 36/60 | 60.00% |
| L7 | 33/60 | 55.00% |

FP is a reference-only frozen canonical subset. No mixed-hardware FP generation was performed.

## Paired endpoint comparisons

| Comparison | Rescue | Regress | Net | Exact p | Holm p |
|---|---:|---:|---:|---:|---:|
| L7 vs H | 2 | 4 | -2 | 0.6875 | 1 |

- L7_vs_H rescues: aime26_11_seed1, aime26_12_seed1; regressions: aime26_10_seed2, aime26_21_seed2, aime26_25_seed2, aime26_26_seed1.
| L6 vs H | 3 | 2 | +1 | 1 | 1 |

- L6_vs_H rescues: aime26_09_seed1, aime26_23_seed1, aime26_23_seed2; regressions: aime26_10_seed2, aime26_26_seed1.
| L7 vs L6 | 2 | 5 | -3 | 0.453125 | 1 |

- L7_vs_L6 rescues: aime26_11_seed1, aime26_12_seed1; regressions: aime26_09_seed1, aime26_21_seed2, aime26_23_seed1, aime26_23_seed2, aime26_25_seed2.

## Trajectory summaries

All deltas are left minus right and are exploratory. Global shortening or repetition reduction was not a preregistered success criterion.

| Comparison | Metric | Mean Δ | Median Δ | 95% paired-bootstrap CI | Direction |
|---|---|---:|---:|---|---|
| L7 vs H | generated_tokens | 9400.27 | 0 | [-7870.19, 27011.1] | HIGHER_IN_LEFT_DESCRIPTIVE_ONLY |
| L7 vs H | repeated_4gram_fraction | -0.00851993 | -0.0116099 | [-0.0229183, 0.00666959] | LOWER_IN_LEFT_DESCRIPTIVE_ONLY |
| L7 vs H | first_committed_claim_position | 4511.85 | -935 | [-8016.24, 19715.2] | HIGHER_IN_LEFT_DESCRIPTIVE_ONLY |
| L7 vs H | num_answer_changes | -1.33333 | 0 | [-4.75, 1.56667] | LOWER_IN_LEFT_DESCRIPTIVE_ONLY |
| L7 vs H | retracted_event_count | -0.933333 | -0.5 | [-4.11667, 2.31667] | LOWER_IN_LEFT_DESCRIPTIVE_ONLY |
| L6 vs H | generated_tokens | -11139 | 0 | [-28740.3, 5835.08] | LOWER_IN_LEFT_DESCRIPTIVE_ONLY |
| L6 vs H | repeated_4gram_fraction | 0.0135984 | 0.00623803 | [-0.0048219, 0.0319662] | HIGHER_IN_LEFT_DESCRIPTIVE_ONLY |
| L6 vs H | first_committed_claim_position | -3733.19 | -78 | [-17626.1, 8682.04] | LOWER_IN_LEFT_DESCRIPTIVE_ONLY |
| L6 vs H | num_answer_changes | -0.183333 | 0 | [-4.05042, 3.2] | LOWER_IN_LEFT_DESCRIPTIVE_ONLY |
| L6 vs H | retracted_event_count | 1.96667 | 1.5 | [-4.93333, 7.76667] | HIGHER_IN_LEFT_DESCRIPTIVE_ONLY |
| L7 vs L6 | generated_tokens | 20539.3 | 0 | [-1388.6, 43710.2] | HIGHER_IN_LEFT_DESCRIPTIVE_ONLY |
| L7 vs L6 | repeated_4gram_fraction | -0.0221184 | -0.0109871 | [-0.0382026, -0.00670501] | LOWER_IN_LEFT_DESCRIPTIVE_ONLY |
| L7 vs L6 | first_committed_claim_position | 13195.1 | -303.5 | [-4374.67, 33411.6] | HIGHER_IN_LEFT_DESCRIPTIVE_ONLY |
| L7 vs L6 | num_answer_changes | -1.15 | 0 | [-4.45, 2.20042] | LOWER_IN_LEFT_DESCRIPTIVE_ONLY |
| L7 vs L6 | retracted_event_count | -2.9 | -1 | [-7.71667, 2.8] | LOWER_IN_LEFT_DESCRIPTIVE_ONLY |

Rescue/regression termination, length-ceiling, stable-commitment, answer-change, retraction, and repetition summaries are in `analysis/rescue_regression_analysis.json`. They are descriptive associations and do not establish an internal causal mechanism.

## Gates and interpretation

- `PROVENANCE_GATE = PASS`
- `CANONICAL_60_COMPLETENESS_GATE = PASS`
- `SCORER_GATE = PASS`
- `ROTATION_IDENTITY_GATE = PASS`
- `INT8_HARDWARE_PARITY_GATE = PASS_EXISTING_FROZEN_EVIDENCE`
- `FP_CANONICAL_COMPLETENESS = 20/60_CANONICAL_AVAILABLE_SUBSET_ONLY_NO_NEW_FP_AUTHORIZED`
- `L7_ENDPOINT_SIGNAL = NEGATIVE_OR_MIXED`
- `L6_ENDPOINT_SIGNAL = SUPPORTIVE_BUT_UNDERPOWERED`
- `L7_STRUCTURED_TRAJECTORY_SIGNAL = INCONCLUSIVE_DESCRIPTIVE_ONLY`
- `L6_STRUCTURED_TRAJECTORY_SIGNAL = INCONCLUSIVE_DESCRIPTIVE_ONLY`
- `CURRENT_RESULT_INTERPRETATION = L7_PRIMARY_NEGATIVE_OR_MIXED; L6_SECONDARY_SUPPORTIVE_BUT_UNDERPOWERED`
- `NEXT_ACTION = REASSESS_LEARNED_ROTATION_OBJECTIVE`

## Resource and audit metadata

- New missing40 assignment per condition: 20 samples on the Ling 2×RTX4090 server and 20 samples on the independent Ling 8×RTX3090 server using only GPU3/GPU4.
- Across H/L6/L7 new outputs: 60 condition outputs on RTX4090 and 60 condition outputs on RTX3090.
- 4090 GPU count used: 2. 3090 GPU count used: 2.
- Retries: 0. Infrastructure failures recorded in canonical outputs: 0.
- Qwen 4×RTX3090 used: NO. 48GB vGPU used: NO.
- Protocol: 262,144 total context, dynamic `max_new_tokens = 262144 - prompt_tokens - 512`, Frozen V4, seed 1/2, INT8-R128.
- Branch: `exp/ling-l6-l7-canonical-60-confirmatory-v1`.
- Frozen preregistration commit: `69b23bd999dc23a9c852497b98df0f56abdbdff0`.

Generated UTC: 2026-09-30T05:45:58.137457+00:00
