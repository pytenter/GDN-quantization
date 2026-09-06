# Candidate Definitions

- `M0`: raw Frobenius reconstruction.
- `query_only`: `||E^T q_t||^2`.
- `dynq_staticw`: `sum_j w_bar_j (q_t^T E[:,j])^2`.
- `q2_staticw`: `sum_ij q_t,i^2 w_bar_j E_ij^2`; drops cross-row terms.
- `groupq_staticw`: same as q2/static-w with q2 averaged over key groups `{4,8,16,32}`.
- `headq_staticw`: head-level query energy times static value weights.
- `static_column`: `sum_j w_bar_j ||E[:,j]||^2`.

No candidate computes online full G, dense H, Jacobian, or extra future signal.
