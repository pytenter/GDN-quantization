# RowBal numerical contract

Task: `QWEN_GDN_SYM8_G32_WSCALE_ROWBAL_AIME26_SEED1_V1`.
Method: `H+G32+Sym8+RowBal`. The canonical codec, original helper/readout,
coordinate/shape, shared transported-query omega, failure rules and lifecycle
are exactly those in `WSCALE_NUMERICAL_CONTRACT.md`.

## Current-state row factors

One dynamic positive FP32 factor is computed for each Key row of each current
actual recurrent state, using all 128 Value columns of that row:

```
m_i = sqrt_FP32(mean_Value128_FP32(X_i,j^2) + FP32(1e-12))
M_g = geometric_mean_Key32(m)
O_g = geometric_mean_Key32(shared_clamped_omega)
r_i = clamp(FP32(FP32((m_i/M_g)^a) * FP32((omega_i/O_g)^(-b))), .25, 4)
```

Every indicated input, intermediate, factor, and reduction is FP32; epsilon
is added to the mean-square, before sqrt. Per-group geometric mean uses
positive FP32 inputs and `exp(mean(log(inputs)))`. Do not substitute row norm
across heads or a Value-column scale, do not recompute/calibrate omega during
AIME, and do not change the normalization group. Clamp once, without a final
geometric renormalization. Reject nonfinite intermediates and r<=0.

The proposed GPU implementation evaluates each power as FP32
`exp(log(ratio)*exponent)` with fusion disabled. The independent stdlib
reference uses independently rounded libm power. Before any production GPU
qualification, the finite synthetic qualification tolerance is fixed at
**16 positive-FP32 ULPs for r** (not a state-error threshold). The GPU fields
test first checks r against its independent formula with that bound, then
requires exact codes/scales/negative-zero/candidate-choice/decode bytes from
an independent scalar codec given the actual validated FP32 r. This separates
transcendental rounding from discrete encoding/search correctness, without
relaxing signed-zero or candidate-tie rules. It may not be enlarged after
seeing qualification or AIME results.

The finite CAL parameter grid is `(.25,.25), (.50,.25), (.50,.50)` in this
frozen order. Exactly one pair is selected on non-AIME CAL via the already
frozen readout proxy and tie rule; VAL reports document-disjoint generalization
without a second selection. The selection receipt binds data, code, scores,
pair and metadata SHA before formal launch. RowBal and RowBal+WScale share
that exact pair. A fixed pair is not asserted selected in this contract.

## QDQ and exact multiplication association

For every current-state group/column:

```
Y_i,j = FP32(X_i,j / r_i)
s_g,j = original_canonical_scale(absmax_Key32(Y))
(q,negative_zero) = original_Sym8_encode(Y,s)
Yhat = original_Sym8_decode(q,s,negative_zero)  # FP32(q*s), restore -0
Xhat = FP32(Yhat * r)                         # positive r preserves -0
```

This decode-first association is frozen. The real-arithmetic expression
`r*s*q` is **not** permission to reassociate FP32 `r*s` before q. For r=1
and lambda=1, codec fields and decoded state recover original Sym32 bytes.
The original peak floor remains applied to Y group peak before /127, so
the minimum scale in Y coordinates is approximately 7.8740157e-15. No zero
point, extra rotation, GDN recursion change, or floating protection branch.

The full current factor depends only on current updated post-readout state
and frozen omega. Temporaries Y, RMS, r, codes and scales cannot become an
unquantized continuation shadow; Xhat immediately overwrites the carrier.

## Metadata and scope

Current FP32 decoded carrier still uses 32*128*128*4 bytes per layer,
48 MiB across 24 recurrent layers. Theoretical packed codes are
32*128*128 bytes per layer, but a deployable packed RowBal state additionally
needs its latest FP32 r for every Key row and its group scales, together with
whatever signed-zero convention the packed implementation promises.
Static omega and dynamic row factors are each 32*128*4 bytes per layer,
384 KiB across 24 layers. Scales are 32*4*128*4 bytes per layer,
1.5 MiB across 24 layers. These are separate storage quantities, not an
assertion that production retains all temporary metadata after writeback.

All three candidate methods retain formal eligibility if math/finite/resource
qualification passes, irrespective of proxy error ranking. Row balancing is
a proposed candidate inspired by scaling ideas, not a full reproduction of
any named paper's formula or a guarantee of improved AIME accuracy.
