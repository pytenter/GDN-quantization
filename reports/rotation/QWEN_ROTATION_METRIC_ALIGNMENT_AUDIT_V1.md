# QWEN_ROTATION_METRIC_ALIGNMENT_AUDIT_V1

## Scope and provenance

This is a retrospective mechanism audit over the existing frozen Qwen3.5-9B AIME26@81920 outputs. No AIME generation was rerun, Frozen V4 was not modified, and no tensor trace was generated. The compact trajectory metrics were produced by the existing `LONG_GENERATION_FAILURE_AUDIT_V1` analyzer and joined to the frozen V4 outcomes by `(problem_id, seed)`.

- Native condition: `INT8_C128`
- Rotated condition: `INT8_C128 + Key-Hadamard`
- Paired observations: 60
- Frozen V4 accuracy: Native 18/60; Hadamard 41/60
- Pair groups: RESCUED 23; BOTH_CORRECT 18; BOTH_WRONG 19; LOST 0
- Generic WikiText teacher-forced mean Future-KL AUC: Native 0.00134640; Hadamard 0.00330097

`GENERIC_FUTURE_KL_REASONING_ALIGNMENT = MISMATCH_OBSERVED`. The WikiText documents and AIME problems are different observation units, so no invalid sample-level correlation was computed.

## Termination and answer commitment

Within the 23 rescued pairs:

| Diagnostic | Native | Hadamard |
|---|---:|---:|
| Frozen V4 abstain | 21 | 0 |
| Reached the 81,920-token ceiling | 18 | 0 |
| EOS reliably observed | 5 | 23 |
| Explicit terminal answer extracted | 2 | 23 |

- `ABSTAIN_TO_CORRECT = 21`
- `TRUNCATED_TO_CORRECT = 18`
- `TERMINATION_SIGNAL = STRONG`
- `ANSWER_COMMITMENT_SIGNAL = STRONG`

For rescued samples with a strong committed-answer position in both methods (coverage 10/23), the median Hadamard-minus-Native fraction generated after the first commit was -0.2384, with a 10,000-resample paired-bootstrap 95% CI of [-0.6989, -0.0150]. First-commit position itself had incomplete coverage and an interval crossing zero, so the commitment verdict is supported principally by the 2-to-23 explicit-terminal-answer transition and the reduction after a detectable first commit, not by a claim of universally earlier first commitment.

## Existing canonical failure categories

The existing classifier applies only to Native INT8 ceiling-hit outputs. It covers 18/23 rescued pairs; five rescued Native outputs are outside that classifier's ceiling-hit scope.

| Native category among rescued | Count |
|---|---:|
| REASONING_OSCILLATION | 2 |
| TEXTUAL_REPETITION_LOOP | 2 |
| MIXED_FAILURE | 0 |
| SEMANTIC_DEGENERATION | 2 |
| UNCLEAR / insufficient evidence | 12 |
| Not covered (non-ceiling) | 5 |

No individual canonical category was significantly enriched in RESCUED versus BOTH_WRONG under two-sided Fisher exact tests (all p > 0.08). Consequently the evidence does not authorize a single-category causal story.

## Paired compact trajectory metrics

All deltas below are Hadamard minus Native medians in the RESCUED group; confidence intervals are paired 10,000-resample bootstrap intervals.

| Metric | Median delta | 95% CI |
|---|---:|---:|
| Generated tokens | -51,800 | [-54,886, -44,836] |
| Repeated 4-gram rate | -0.12131 | [-0.12547, -0.09112] |
| Repeated 8-gram rate | -0.04889 | [-0.09096, -0.01941] |
| Repeated 16-gram rate | -0.01310 | [-0.03887, -0.00013] |
| Unique-token ratio | +0.02438 | [+0.02012, +0.02825] |
| Longest repeated span | -32 tokens | [-32, 0] |
| `OSCILLATION_PROXY_EXPLORATORY` correction-marker density per 1k tokens | -3.1738 | [-5.2153, -1.8361] |
| Fraction after first strong answer commit (10/23 coverage) | -0.23835 | [-0.69887, -0.01495] |

The transparent correction-marker density is exploratory and is not the canonical human category. Both lower repetition and lower correction-marker density occur strongly in rescued pairs. They are not sufficient explanations by themselves: BOTH_CORRECT also improves on several of these metrics, while BOTH_WRONG becomes shorter and has lower correction-marker density without becoming correct; its repeated 16-gram delta can increase.

## Verdict

- `TERMINATION_SIGNAL = STRONG`
- `OSCILLATION_SIGNAL = STRONG`
- `TEXT_REPETITION_SIGNAL = STRONG`
- `ANSWER_COMMITMENT_SIGNAL = STRONG`
- `GENERIC_FUTURE_KL_REASONING_ALIGNMENT = MISMATCH_OBSERVED`
- `QWEN_METRIC_ALIGNMENT_VERDICT = GENERIC_FUTURE_KL_MISALIGNED; termination, oscillation, text-repetition, and answer-commitment signals co-occur; no single causal signal is identified`
- `candidate metric for independent validation = TERMINATION_AND_OSCILLATION_TRAJECTORY_STABILITY`

The candidate is a diagnostic construct, not a new loss and not a validated optimization target. It must be locked and tested on an independent non-AIME reasoning panel before any training use. The corresponding protocol-only plan is in `docs/experiments/NON_AIME_TRAJECTORY_METRIC_VALIDATION_V1_PLAN.md`.

