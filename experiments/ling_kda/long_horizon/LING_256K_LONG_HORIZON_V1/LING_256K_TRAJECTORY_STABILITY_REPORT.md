# Ling/KDA 256K Trajectory Stability Report

## 1. Experimental setup

This report freezes and analyzes existing Ling-3.0-tiny KDA recurrent-state AIME26 generations at a 256K-token horizon. No model code, quantizer code, rotation implementation, or runtime behavior was changed.

Conditions: `FP_STATE`, `INT8_R128`, and `INT8_R128_VALUE_HADAMARD`. Scoring uses frozen AIME26 Strict V4 candidate extraction.

Important limitation: token-level logits were not present in the frozen artifacts, so first-divergence analysis uses the first generated-token ID mismatch versus `FP_STATE` as the closest available approximation.

## 2. Frozen results

| Condition | Correct | Accuracy |
|-|-:|-:|
| FP_STATE | 21/30 | 70.0% |
| INT8_R128 | 9/30 | 30.0% |
| INT8_R128_VALUE_HADAMARD | 17/30 | 56.7% |

## 3. Generation length analysis

| Condition | Mean | Median | Max | P95 |
|-|-:|-:|-:|-:|
| FP_STATE | 46608.2 | 34204.0 | 261523 | 125187.4 |
| INT8_R128 | 159845.6 | 261314.0 | 261553 | 261532.8 |
| INT8_R128_VALUE_HADAMARD | 126495.8 | 109348.0 | 261540 | 261468.6 |

INT8_R128 generated on average 113237.4 more tokens than FP_STATE. Value-Hadamard generated on average -33349.8 tokens relative to plain INT8.

## 4. Termination analysis

| Condition | EOS | Length limit | No answer | Repeated reasoning | Other |
|-|-:|-:|-:|-:|-:|
| FP_STATE | 29 | 1 | 0 | 3 | 0 |
| INT8_R128 | 14 | 16 | 10 | 9 | 0 |
| INT8_R128_VALUE_HADAMARD | 21 | 9 | 7 | 2 | 0 |

Plain INT8 has the largest termination damage: many samples hit the length limit and V4 abstains. Value-Hadamard reduces no-answer cases but does not eliminate length-limit failures.

## 5. Repetition / oscillation analysis

| Condition | 4-gram mean | 8-gram mean | 16-gram mean |
|-|-:|-:|-:|
| FP_STATE | 0.5109 | 0.2042 | 0.0617 |
| INT8_R128 | 0.5649 | 0.2260 | 0.0977 |
| INT8_R128_VALUE_HADAMARD | 0.4839 | 0.1602 | 0.0414 |

Value-Hadamard changes mean repeated 16-gram rate by -0.0563 relative to plain INT8. The main observed rescue is not a complete removal of repetition, but fewer no-answer/failed-commit trajectories.

## 6. First divergence analysis

| Comparison | Mean | Median | P25 | P75 | P95 |
|-|-:|-:|-:|-:|-:|
| FP_STATE vs INT8_R128 | 53.1 | 42.0 | 19.2 | 56.5 | 166.6 |
| FP_STATE vs INT8_R128_VALUE_HADAMARD | 36.5 | 32.0 | 11.5 | 48.5 | 84.8 |

By generated-token LCP approximation, Value-Hadamard shifts the median first divergence by -10.0 tokens relative to plain INT8. In this approximation it does not delay the first token mismatch; the stability improvement appears later in the trajectory through fewer length-limit/no-answer failures and lower high-order repetition. Treat this as trajectory evidence, not logits-level proof.

## 7. Value-Hadamard rescue analysis

Rescue cases where INT8_R128 is wrong and Value-Hadamard is correct: 8.

| Problem | Gold | INT8 pred | Value-H pred | INT8 len | Value-H len | INT8 div | Value-H div |
|-|-:|-:|-:|-:|-:|-:|-:|
| aime26_04 | 70 | 71 | 70 | 47099 | 13959 | 8 | 8 |
| aime26_10 | 156 | None | 156 | 261509 | 209345 | 111 | 8 |
| aime26_19 | 279 | 271 | 279 | 50430 | 25505 | 35 | 32 |
| aime26_21 | 50 | None | 50 | 261553 | 42808 | 166 | 3 |
| aime26_22 | 754 | None | 754 | 261505 | 123934 | 39 | 39 |
| aime26_24 | 669 | 3 | 669 | 261524 | 52448 | 39 | 165 |
| aime26_25 | 850 | None | 850 | 261472 | 146107 | 2 | 7 |
| aime26_26 | 132 | None | 132 | 261515 | 214389 | 7 | 34 |

## 8. Conclusions

Under a 256K-token horizon, plain INT8 recurrent-state quantization substantially destabilizes reasoning trajectories relative to FP_STATE: accuracy drops from 21/30 to 9/30, output lengths grow, length-limit/no-answer failures rise, and many samples diverge from the FP trajectory very early.

Value-Hadamard improves long-horizon stability in this frozen run: it recovers 8 additional correct samples over plain INT8, reduces V4 no-answer failures, and lowers high-order repetition. The generated-token LCP approximation does not show delayed first divergence; instead, the evidence supports the weaker claim that Value-Hadamard reduces some downstream long-horizon trajectory failures after early divergence. It does not establish formal mechanism closure.

The most important remaining failure modes are termination failure and reasoning oscillation on long generations. Further claims would require logits-level traces or controlled reruns, which are outside this frozen-analysis scope.

## Generated artifacts

- `manifest.json`
- `analysis/trajectory_metadata.{json,csv}`
- `analysis/generation_statistics.json`
- `analysis/termination_statistics.json`
- `analysis/repetition_statistics.json`
- `analysis/divergence_analysis.json`
- `analysis/failure_taxonomy.json`
- `analysis/rescued_cases_analysis.json`
- `trajectories/`
