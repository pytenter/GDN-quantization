# Ling/KDA 256K Post-Divergence Stability Report

## 1. Objective

This study analyzes behavior after the first generated-token mismatch versus FP_STATE. It does not test prevention of first divergence, and it does not rerun generation.

The hypothesis evaluated here is: Value-Hadamard improves post-divergence trajectory stability rather than delaying the initial token-level mismatch.

## 2. Frozen Experimental Setup

Frozen branch/artifact root: `experiments/LING_256K_LONG_HORIZON_V1`. Conditions are FP_STATE, INT8_R128, and INT8_R128_VALUE_HADAMARD on AIME26 with a 256K-token generation horizon and frozen V4 scoring.

All metrics are computed from existing artifacts only. Token-level logits are unavailable; divergence uses generated-token ID LCP approximation. Answer commitment position uses the V4 selected evidence span mapped proportionally to token position. Regex answer-like markers are used only as termination-failure indicators, not as committed final answers.

## 3. First Divergence Result

Prior frozen analysis found median first divergence of INT8_R128 = 42 tokens and Value-Hadamard = 32 tokens. Value-Hadamard therefore does not significantly delay the first token mismatch in these artifacts.

## 4. Post-Divergence Stability Analysis

| Condition | Post-div mean | Post-div median | Length-limit | No commit | Commit rate | Post-div 16g mean | 16g onset median | Commit lag median |
|-|-:|-:|-:|-:|-:|-:|-:|-:|
| FP_STATE | 46608.2 | 34204.0 | 1 | 0 | 100.0% | 0.0617 | 406.5 | 34197.0 |
| INT8_R128 | 159792.6 | 261246.5 | 16 | 10 | 66.7% | 0.0976 | 623.5 | 48733.0 |
| INT8_R128_VALUE_HADAMARD | 126459.3 | 109310.5 | 9 | 7 | 76.7% | 0.0412 | 877.0 | 42801 |

Plain INT8 has the longest post-divergence continuations and the most length-limit/no-commit failures. Value-Hadamard shortens post-divergence continuation relative to INT8 and lowers high-order post-divergence repetition.

## 5. Failure Taxonomy

| Condition | Oscillation | Repetition | Termination | Drift |
|-|-:|-:|-:|-:|
| FP_STATE | 7 | 1 | 0 | 1 |
| INT8_R128 | 3 | 6 | 11 | 1 |
| INT8_R128_VALUE_HADAMARD | 4 | 0 | 9 | 0 |

Taxonomy is rule-based only: repetition thresholds, length-limit/no-commit indicators, and phrase-count indicators. It deliberately avoids external semantic judgement.

## 6. Rescue Case Analysis

| Problem | INT8 failure | Recovery behavior | INT8 post-div len | Value-H post-div len | INT8 16g | Value-H 16g |
|-|-|-|-:|-:|-:|-:|
| aime26_04 | repetition_loop | lower_post_divergence_repetition, shorter_post_divergence_continuation | 47091 | 13951 | 0.1502 | 0.0837 |
| aime26_10 | termination_failure | avoids_length_limit, recovers_answer_commitment, shorter_post_divergence_continuation | 261398 | 209337 | 0.0300 | 0.0675 |
| aime26_19 | oscillation | shorter_post_divergence_continuation | 50395 | 25473 | 0.0104 | 0.0111 |
| aime26_21 | termination_failure | avoids_length_limit, recovers_answer_commitment, lower_post_divergence_repetition, shorter_post_divergence_continuation | 261387 | 42805 | 0.0498 | 0.0212 |
| aime26_22 | termination_failure | avoids_length_limit, shorter_post_divergence_continuation | 261466 | 123895 | 0.0085 | 0.1179 |
| aime26_24 | termination_failure | avoids_length_limit, shorter_post_divergence_continuation | 261485 | 52283 | 0.0474 | 0.0602 |
| aime26_25 | repetition_loop | avoids_length_limit, recovers_answer_commitment, lower_post_divergence_repetition, shorter_post_divergence_continuation | 261470 | 146100 | 0.3732 | 0.0137 |
| aime26_26 | repetition_loop | avoids_length_limit, recovers_answer_commitment, lower_post_divergence_repetition, shorter_post_divergence_continuation | 261508 | 214355 | 0.4047 | 0.0296 |

Across rescue cases, Value-H commonly recovers by avoiding INT8's length-limit/no-answer path and reducing post-divergence repetition or continuation length, while still diverging early from FP_STATE.

## 7. Updated Mechanistic Hypothesis

Observed trajectory factorization:

`INT8 recurrent perturbation -> early token mismatch -> post-divergence trajectory instability -> repetition / excessive continuation / termination failure -> reasoning failure`

Value-Hadamard does not eliminate the initial perturbation in these artifacts. The supported observational claim is narrower: after early divergence, Value-Hadamard improves trajectory stability enough to reduce repetition, excessive continuation, and failed commitment on several samples.

This is not a formal causal mechanism claim. It is an artifact-level trajectory-stability finding from frozen generations.
