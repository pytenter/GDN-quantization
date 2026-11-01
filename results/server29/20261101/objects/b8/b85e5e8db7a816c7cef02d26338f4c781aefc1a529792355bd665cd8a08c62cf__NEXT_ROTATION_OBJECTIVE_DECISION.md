# Next Rotation Objective Decision

Status: **FINAL**.

## Decision

`PROCEED_TO_SHORT_HORIZON_SURROGATE_VALIDATION`

## Why

The primary visible-minus-low Future-KL AUC contrast was positive in development and both prospective validations, for both intervention signs, with all frozen mechanical and integrity gates passing. The result also survives the endpoint-magnitude sanity check in the conservative direction: visible branches ended with substantially smaller target-state error norms than LOW branches.

The evidence is strong enough to stop asking whether direction can matter in this pilot and start asking whether an outcome-free, training-time-available short-horizon proxy can rank harmful directions. It is not yet evidence to train a new rotation.

## Minimal next validation

Using the existing frozen artifacts and these three documents, pre-register a compact candidate set of local/short-horizon query-readout proxies and test whether their direction ranking agrees with the frozen long-horizon causal `D`/branch ranking. Keep long-horizon outcomes blinded until proxy definitions and scoring rules are frozen. Require prospective document-level agreement and retain propagation diagnostics as covariates; do not optimize a new rotation in that validation.

If later work needs to distinguish direct readout mediation from recurrent/quantizer feedback, run a separately frozen endpoint-state-controlled mediation experiment. That is optional follow-up evidence, not a prerequisite for the short-horizon surrogate validation selected here.

No L9, AIME, free generation, Qwen work, old 48/128 recovery, or new rotation training is authorized by this decision.
