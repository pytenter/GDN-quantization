# WScale numerical contract

Task: `QWEN_GDN_SYM8_G32_WSCALE_ROWBAL_AIME26_SEED1_V1`.
Method: `H+G32+Sym8+WScale`. This specifies mathematics and qualification;
it is not evidence of GPU execution or AIME completion.

## Original canonical baseline

State is FP32 `[batch, actual_recurrent_head, Key128, Value128]` in the original
Key-H128 coordinates. Only the post-update, post-current-readout QDQ stage is
replaced. Quantization groups contain 32 adjacent **Key** rows of one Value
column in one batch/head/layer. No mixing of layers, heads, columns, or groups.

Authoritative original source is the read-only
`QWEN_GDN_GROUP32_FOUR_CODEC_AIME26_SEED1_V1/inherited/group32/src/group_codecs.py`.
Its symmetric codec applies `EPS=1e-12` to the **absolute group peak before
division**, not to the scale:

```
safe_peak = max_FP32(absmax_FP32(X_group), FP32(1e-12))
s0 = safe_peak / 127
```

The inherited CUDA scalar-divisor operation graph uses
`FP32(safe_peak * FP32(1/127))`; its independently supported CPU graph uses
`FP32(safe_peak / FP32(127))`. These graphs can differ by one scale ULP.
Production retains the actual inherited CUDA graph, and CPU regression tests
name the selected graph explicitly. `original_scale_floor` therefore means
the corresponding effective scale `canonical_scale(0)`, approximately
`7.8740157e-15`, **never a literal scale floor of 1e-12**.

Encode: FP32 division by scale, nearest-even rounding, clamp to `[-127,127]`,
signed INT8 codes. Zero point is exactly zero. Negative zero is recorded from
the rounded/clipped floating result before integer conversion and restored
after FP32 `q*scale` decode. No `-128` code is legal. Reject nonfinite input,
scale, level, reconstruction, or weighted objective and nonpositive scales.

## Shared static sensitivity

All three new methods use the same frozen FP32 omega for every actual
layer/recurrent-head/Key-row. The non-AIME calibration documents and their
CAL/VAL split must be frozen before collection and remain document-disjoint.
No AIME input, output, answers, error lists, or current/future evaluation query
is used. Per-head raw omega is the CAL mean of squared transported query.

The original helper first multiplies prior state by `alpha=exp(g)` and then
applies the delta-rule update with actual `beta=sigmoid(b)` and actual normalized
Hadamard-coordinate key. Thus

```
A = alpha * (I - beta * k k^T)
q_readout = q_helper / sqrt(128)
transport = A^T q_readout
          = alpha * (q_readout - beta*k*dot(k,q_readout))
```

`g=-exp(A_log)*softplus(a+dt_bias)` is the actual log-gate, not alpha itself.
The helper q already includes original BF16 normalization and FP32 H128
transform; do not normalize or rotate it again. There are 16 independent q/k
heads repeated to 32 recurrent heads, but alpha and beta are actual recurrent
head parameters. Shared q/k never licenses copying paired omega without
checking these parameters. Current token's readout uses updated state before
QDQ; error injected by QDQ at t first affects readout t+1 through A_(t+1).

Stabilization, chosen before AIME: FP32 raw positive floor `1e-12`, geometric
mean normalization separately within each 32 Key rows of one actual head,
then clamp to `[1/16,16]`, with **no renormalization after clamp**. Log/exp
operations and reduction arithmetic are recorded in the implementation pin.
Preserve raw, normalized, and clamped distributions and head/row identities.

## Online selection

Candidate lambda values in storage order are
`[0.60,0.75,0.875,1.00,1.125]`, represented as FP32. For each group/column:

```
s_lambda = max_FP32(FP32(lambda*s0), canonical_scale(0))
d_lambda = original_Sym8_decode(original_Sym8_encode(X_group,s_lambda))
loss_lambda = sum_Key32 omega * (X_group-d_lambda)^2
```

All current-state candidate quantities use FP32. The lambda=1 candidate must
retain the original canonical scale/codes/negative-zero bytes. Exact equal
finite losses use deterministic priority **1.00, 0.875, 1.125, 0.75, 0.60**:
nearest 1, then numerically smaller lambda. There is no fuzzy tie threshold.
Weights are positive. Therefore minimum selected computed objective is no
larger than the included lambda=1 objective under the same arithmetic. This
is a numerical diagnostic, not a proxy-quality admission gate or an AIME
performance guarantee.

The independent scalar reference uses explicit FP32 subtraction, square,
omega multiplication, and a fixed halves-first binary 32-term FP32 sum.
GPU reduction/transcendental implementation and tolerances must be declared
before GPU qualification; a close comparison is never silently changed into
an exact tie. Candidate objectives and winner margins are available for
diagnosing operation-order differences. Kernel optimization may not change
the objective, candidates, clipping, grouping, or search into fixed lambda.

## Carrier, lifecycle, and storage

Only final selected decoded values immediately overwrite the original FP32
state. The next original helper must consume those values/bytes. Prompt and
decode retain QDQ from token0, at every actual recurrent layer, after each
forward. No unquantized persistent state, delayed writeback, future query,
or repeated AIME attempt is authorized. No Python per-group production loop
or per-group CPU synchronization is allowed.

The independent CPU reference deliberately loops and is not the production
kernel. Current carrier is post-QDQ FP32; it is not packed INT8. WScale adds
static FP32 omega and candidate work, with one final FP32 scale per group.
Codes, scales, negative-zero flags, and static metadata costs are reported
separately. Non-AIME math/field/finiteness failures stop the branch. A poor
non-AIME proxy metric does not remove WScale from its 30 formal samples.
