# LING_KDA_KERNEL_INTERNAL_DIRECTIONAL_GEOMETRY_DECOMPOSITION_V1 Local KDA Source Audit

LOCAL_KDA_SOURCE_PROVENANCE_GATE = PASS

The canonical decode path calls `fused_recurrent_kda` for `q_len <= 64`; the current formal probe uses one post-t0 decode token.
The local Ling wrapper computes q/k/v through projections and `ShortConvolution`, computes beta with sigmoid before kernel, and passes raw first-stage `g` with `use_gate_in_kernel=True` plus `A_log` and `dt_bias`.
The reference reproduces the local fused recurrent loop: q/k l2 normalization, q scaling, gate decay construction, state decay, beta-scaled key correction, state update, and post-update query readout.
