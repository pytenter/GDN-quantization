# GDN_INT8_FUNCTIONAL_RISK_STATE_SPACE_PULLBACK_V1

## Tensor Semantics

- recurrent state shape: `[1, 32, 128, 128]`
- per-head state shape: `[d_k=128, d_v=128]`
- q shape: `[32, 128]` after batch squeeze
- functional readout: `x[h,v] = sum_k E[h,k,v] * q[h,k]`
- functional dimension: `32 * 128 = 4096`
- vectorization: state `[head, key_row, value_col]`, functional `[head, value_col]`

A direct reshape from 4096 functional weights to 128x128 state weights is invalid because the exact pullback contains `q_i q_k` cross-row interactions inside each value column.
