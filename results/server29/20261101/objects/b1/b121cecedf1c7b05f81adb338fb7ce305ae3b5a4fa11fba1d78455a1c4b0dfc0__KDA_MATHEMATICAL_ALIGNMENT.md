# KDA mathematical alignment clarification

This note makes the frozen protocol's shorthand explicit; it changes neither measurements nor definitions, windows, normalization, aggregation or acceptance rules.

State is `S[b,h,k,v]`, with `(b,h,k,v)=(1,16,128,128)` for each of 18 layers. The rotation acts on the **value axis**: `S_rot=S R`; quantization groups that value axis rowwise; physical cache writeback is `QDQ(S R) R^T`. State comparisons use physical coordinates.

K3 contracts the **key axis**, not the value axis:

`r[l,h,v] = sum_k E[l,h,k,v] * q_prepared_FP[l,h,k]`.

Equivalently, each head's column-vector form is `E^T q`. The frozen text's `E @ q` is shorthand for this actual K3 readout, not a literal dimension-mismatched multiplication. The evaluator imports the exact historical `query_readout` function body, whose implementation is `(E * q.unsqueeze(-1)).sum(dim=-2).unsqueeze(1)`; the CPU tests exercise non-square key/value dimensions as well.

Prepared queries come from the exact historical `prepared_query` / `_prepared_drivers` code: FP32 `q / sqrt(sum(q^2)+1e-6) / sqrt(128)`. The primary score is mean square over all layer/head/value coordinates, not a norm ratio or a summed magnitude.

P1 uses `E_t = S_quant_postupdate_postQDQ_t - S_FP_postupdate_t` at the start token. This error includes the real persistent-quantization history of the short prefix; it is not only the current token's isolated quantization residual.

P2 uses the non-centered second moment `G_t = mean_{i=t-128}^{t-1} q_i q_i^T`, and `mean_{l,h,v} E_{lh,:,v}^T G_{lh} E_{lh,:,v}`. Neither centered covariance nor future-window queries are substituted.

P3 is the mean K3 readout energy of actual recurrent errors through the next 32/128 tokens. The quantized model keeps the full hybrid cache and uses its own trajectory-dependent hidden states; FP queries and FP states are the same clean teacher reference for every rotation.

The output at token t is emitted before t's QDQ writeback, so B2's `KL(FP_t || quant_t)` contains persistent QDQ feedback through t-1. This one-token distinction is retained. All positions are zero-based, windows are `[256,384)` and `[1024,1152)`, and P2 histories are `[128,256)` and `[896,1024)`.

The two window scores are averaged arithmetically with no outcome-driven reweighting. Layer20 results are auxiliary; no layer is selected after observing target rankings.
