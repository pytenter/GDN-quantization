# GDN_INT8_FUNCTIONAL_RISK_STATE_SPACE_PULLBACK_V1

## Pullback Derivation

For a GDN layer and token, let `E = S_hat - S` with shape `[head, key_row, value_col]`.

The true readout contraction is:

```
x[h,v] = sum_k E[h,k,v] * q[h,k]
```

The diagonal functional risk from the previous experiment is:

```
M_diag,t^2 = sum_{h,v} w[h,v] * x[h,v]^2
w[h,v] = diag(G_t)[head,value]
```

Substituting the contraction gives the exact state-space pullback:

```
M_diag,t^2 = sum_{h,v} w[h,v] * (sum_k E[h,k,v] q[h,k])^2
           = e_state^T H_t e_state
H_t = B_t^T diag(w_t) B_t
```

Per head/value column, this has `w[h,v] * outer(q[h], q[h])` over key rows. Therefore functional diagonal geometry is not the same as element-wise state diagonal weighting.
