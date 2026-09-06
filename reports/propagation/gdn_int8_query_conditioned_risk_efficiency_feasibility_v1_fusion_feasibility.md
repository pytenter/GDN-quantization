# Fusion Feasibility

Fusable single pass candidates: q2/static-w, grouped-q/static-w, head-q/static-w, static-column.

During an existing state scan, each loaded `E_ij` can update weighted statistics using either `q2_i * w_j`, `group_q2_g * w_j`, `head_energy_h * w_j`, or `w_j`. This does not require reloading the recurrent state.

Exact query-only and dyn-q/static-w require column reductions `E^T q`, which are scientifically useful but less directly compatible with a per-element quantization statistic.
