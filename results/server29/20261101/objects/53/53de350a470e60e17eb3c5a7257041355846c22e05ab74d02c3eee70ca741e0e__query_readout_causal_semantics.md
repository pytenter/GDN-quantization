# LING_KDA_QUERY_READOUT_CAUSAL_CLOSURE_V1 Query Readout Causal Semantics

QUERY_READOUT_SEMANTICS_GATE = PASS

Input residual tensor: K2 update-error tensor in implementation KV layout `[B, HV, K, V]`.
Query tensor: reference-prepared query `q_hat` with shape `[B, HV, K]` after value-head expansion, optional q/k L2 normalization, and kernel scale.
Head/layout semantics: `HV` is the KDA value-head axis; `K` is the key/channel axis; `V` is the value axis.
Normalization and scale: exactly reused from `pytorch_kda_reference._prepared_drivers`, including q/k L2 norm flags and `K ** -0.5` query scaling.
Dtype: K3 readout is evaluated in float32; the local core-output endpoint additionally applies the same output dtype cast used by the reference.
Readout equation: `r[b, 0, h, v] = sum_k U[b, h, k, v] * q_hat[b, h, k]`.
Output shape: `[B, 1, HV, V]`.

Visible projection: `U_vis = q_hat * (sum_k U_k q_hat_k / ||q_hat||^2)` per batch/head/value. `U_null = U - U_vis`, and `R_q(U_null) ~= 0`.
