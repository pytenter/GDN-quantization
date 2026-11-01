# RowBal + WScale combined numerical contract

Task: `QWEN_GDN_SYM8_G32_WSCALE_ROWBAL_AIME26_SEED1_V1`.
Method: `H+G32+Sym8+RowBal+WScale`.

This is a strict composition of `ROWBAL_NUMERICAL_CONTRACT.md` and
`WSCALE_NUMERICAL_CONTRACT.md`, implemented through one shared quantizer.
It has no separately calibrated omega or separately selected (a,b).

1. Compute the identical RowBal r from the current updated state X and the
   shared frozen actual-head omega, with the identical selected CAL pair.
2. Compute FP32 `Y=X/r`, original canonical group absmax scale s0, and all
   five WScale candidates `[.60,.75,.875,1,1.125]` with the original effective
   minimum scale.
3. Encode each candidate with the original symmetric signed INT8
   nearest-even/clamp/negative-zero convention.
4. Decode to FP32 Yhat first, then compute FP32 `Xhat=Yhat*r`.
5. Select with `sum_Key32 omega_i*(X_i,j-Xhat_i,j)^2` in **original X space**.
6. Exact ties use priority `[1,.875,1.125,.75,.60]`, as WScale does.
7. Immediately overwrite the actual FP32 state carrier with selected Xhat.

Optimizing `sum omega*(Y-Yhat)^2` is forbidden: it drops the original-space
row factor squared and changes the scientific objective. Merely multiplying
the target loss by a groupwide constant is harmless in exact arithmetic, but
it does not license a different FP32 operation graph without preregistered
qualification. Current-state FP32 candidate work is temporary and may not
remain as a higher precision shadow for future tokens.

The lambda=1 candidate is exactly RowBal's original canonical scale candidate.
Thus under identical computed arithmetic its selected local weighted loss
cannot exceed RowBal's loss. This local inequality is a diagnostic, not a
readout/long-propagation/AIME acceptance threshold. No candidate is dropped
from the 30-question AIME run because of a poor proxy score.

Reduction, transcendental tolerances, FP32 association, and kernel winner
margin checks are frozen with the shared implementation before formal AIME.
Mathematical edge cases include zero/positive/negative groups, tiny dynamic
range, the original peak floor, exact candidate ties, signed zeros, finite
positive r, illegal values and strict layer/head/Key32/Value identity.
With r=1 and lambda=1 the entire base Sym32 field/decode regression must pass.

No changes to Hadamard placement, original update-before-readout, token0 QDQ,
per-token/per-layer writeback, next-helper consumption, sampler, RNG/cache
freshness, seed1, 81,920-new-token cap, or fixed input IDs are part of this
method. This contract does not certify GPU qualification or formal completion.

Metadata includes WScale's static omega/candidate work and RowBal's dynamic
r. One chosen FP32 scale per group remains; the current carrier remains
decoded post-QDQ FP32, never a claimed packed-INT8 persistent allocation.
