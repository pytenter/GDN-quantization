# NON_AIME_TRAJECTORY_METRIC_VALIDATION_V1_PLAN

## Purpose

Independently test whether the retrospective Qwen AIME signal named `TERMINATION_AND_OSCILLATION_TRAJECTORY_STABILITY` generalizes to non-AIME reasoning data. This document is a plan only. It does not authorize rotation training, metric fitting, checkpoint selection, or AIME reuse.

## Frozen hypothesis

Compared with Native INT8_C128, Key-Hadamard should more often produce a stable terminal answer without reaching the generation ceiling, and should reduce transparent oscillation/repetition diagnostics. A useful metric must separate successful from unsuccessful trajectories on new non-AIME prompts and must preserve a paired Hadamard-versus-Native effect.

The candidate is a predeclared diagnostic vector, not a fitted scalar:

1. EOS or explicit terminal-answer completion before the fixed cutoff;
2. generated-token count and terminal-answer distance from cutoff;
3. correction/restart-marker density per 1,000 tokens (`OSCILLATION_PROXY_EXPLORATORY`);
4. repeated 4-, 8-, and 16-gram rates and unique-token ratio;
5. fraction of the generation after the first strong answer commitment, when detectable.

No weights will be fitted on AIME. A scalar composite, if ever needed, must be specified before viewing validation outcomes and must be justified independently.

## Independent panel

- Use a versioned, immutable panel of at least 100 non-AIME reasoning prompts.
- Include multiple domains such as contest-style but non-AIME mathematics, symbolic logic, algorithmic reasoning, and verifiable word problems.
- Exclude AIME problems, paraphrases, derivatives, and any prompt used for rotation training or checkpoint selection.
- Freeze prompt text, tokenizer/token IDs, references, maximum generation length, decoding parameters, scorer, and sample IDs before inference.
- Use problems with machine-verifiable answers where possible; blind human adjudication must be specified in advance for the remainder.

## Paired execution

- Evaluate the same model checkpoint and each prompt/seed under Native INT8_C128 and Key-Hadamard.
- Keep all non-rotation runtime settings identical; sampling should be off unless a separately frozen multi-seed protocol is approved.
- Capture only generation text/token IDs and compact diagnostics. Do not create recurrent-state, q/k/v, or logit tensor traces.
- Blind the trajectory analyzer and adjudicator to method labels until scores and diagnostics are frozen.

## Primary analysis

Predeclare paired groups using the independent panel's frozen correctness scorer: rescued, both correct, both wrong, and lost. Report:

- paired accuracy difference with an exact or paired-bootstrap interval;
- termination and abstention transitions;
- median paired deltas and 10,000-resample bootstrap 95% intervals for every diagnostic component;
- coverage for answer-commitment metrics;
- whether the same direction appears specifically in rescued pairs and whether it also appears without rescue in both-wrong pairs.

Do not rank many candidate metrics and report only the winner. Apply multiplicity control or label all component tests descriptive. Report all predeclared metrics and counterexamples.

## Success and failure criteria

Validation succeeds only if all of the following hold:

1. the paired correctness effect favors Hadamard without an offsetting excess of lost cases;
2. terminal completion improves in rescued pairs under the frozen definitions;
3. at least one oscillation/repetition component has a paired interval excluding zero in the predicted direction;
4. the signal is not explained solely by shorter generation, assessed by reporting comparable-length strata or a predeclared length-aware sensitivity analysis;
5. conclusions remain qualitatively stable across at least two prompt domains.

If these criteria fail, record `TRAJECTORY_METRIC_VALIDATION = FAIL` or `INCONCLUSIVE`; do not tune the metric on the validation panel and do not promote it to a training objective.

## Allowed next decision

Only after a locked independent validation passes may a separate protocol assess whether the validated diagnostic is computationally differentiable, resistant to shortcut optimization, and suitable for a future-aware objective. That later feasibility study must use training/validation data independent of both AIME and this test panel.

