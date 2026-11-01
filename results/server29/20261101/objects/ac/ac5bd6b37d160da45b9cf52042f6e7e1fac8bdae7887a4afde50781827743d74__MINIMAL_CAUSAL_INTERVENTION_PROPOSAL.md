# Minimal causal intervention proposal

## Question

Holding propagated-error magnitude fixed, does changing only its location relative to the actual KDA query/readout-sensitive subspace change functional damage?

## Stage 1: exact local readout diagnostic

Use the historical KDA readout definition rather than inventing a proxy. For a frozen state-error tensor `U` and frozen query `q_hat`, the readout is linear in the state:

`R_q(U)[b,h,v] = sum_k U[b,h,k,v] * q_hat[b,h,k]`.

The historical closure implementation defines the visible component by projecting each batch/head/value slice onto `q_hat`:

`U_visible = q_hat * (sum_k U_k q_hat_k / ||q_hat||^2)`

and `U_null = U - U_visible`, with `R_q(U_null)` numerically zero up to the frozen gate.

For one preregistered document/condition/anchor, reconstruct the actual error from the frozen trace replay. Form two interventions on the same FP base state:

1. readout-visible error: normalized `U_visible`;
2. readout-null error: a normalized nondegenerate `U_null` (or a deterministic null-space basis vector if the observed null component is zero).

Scale both to exactly the same Frobenius norm as the original error. Freeze document, token, layer set, state layout, query, future input, injection site, quantizer, and evaluation code. The only allowed difference is error direction. Verify byte identities, norm equality, visible readout equality/zero gates, and no accidental change to other state components.

Stage 1 measures local readout sensitivity only. A positive result cannot be called full recurrent functional causality.

## Stage 2: frozen-trace recurrent propagation

If Stage 1 passes, inject the two norm-matched errors at the same anchor and replay the identical frozen `q/k/v/beta/log_decay` sequence with the original persistent every-token QDQ and quantized writeback semantics. Compare preregistered trajectory readout damage over a fixed short and full available horizon. Do not change the POR definition after seeing outcomes.

This stage tests whether visible/null orientation survives the recurrent dynamics under identical future drivers. Because archived traces lack full logits, a trace-only result remains a recurrent readout-causality test, not a FutureKL causal test.

## Stage 3: optional full functional test

Only if Stages 1--2 produce a stable separation, repeat the same two interventions through the full model/logit path and evaluate the frozen FutureKL metric. This requires GPU/model execution and must be separately authorized. Keep persistent QDQ active after injection; otherwise the intervention would no longer test the deployed recurrent semantics.

## Validity gates

- same FP base state and same reconstructed original error norm;
- same document, anchor, endpoint, future tokens, query/readout records, and checkpoint;
- no use of FutureKL to construct or select the direction;
- direction construction and thresholds frozen before outcome evaluation;
- exact visible/null readout gate and norm-match tolerance inherited from historical closure, not newly relaxed;
- no pooling across runtime or GPU unless explicitly qualified;
- all counterexamples reported.

## Feasibility verdict

Stage 1 and the trace-replay portion of Stage 2 are feasible from existing frozen traces after deterministic state reconstruction; no new 32K model-forward rollout is intrinsically required. The archived canonical scalars alone are insufficient. Stage 3 is the missing high-value causal bridge and should not be run until the cheaper direction manipulation demonstrates separation.

Execution status: **proposal only; not authorized or launched**.
