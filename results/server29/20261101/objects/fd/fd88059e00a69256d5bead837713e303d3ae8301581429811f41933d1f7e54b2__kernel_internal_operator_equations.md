# LING_KDA_KERNEL_INTERNAL_DIRECTIONAL_GEOMETRY_DECOMPOSITION_V1 Kernel Internal Operator Equations

For one decode token and fixed local drivers:

`q_hat = l2norm(q) * K^-0.5`; `k_hat = l2norm(k)` when kernel q/k norm is enabled.
`d = -exp(A_log) * softplus(g + dt_bias)` unless safe lower-bound gate is active.
`H_d = exp(d) * H0`.
`v_delta = beta * (v - contraction(H_d, k_hat))`.
`H_u = H_d + outer(k_hat, v_delta)`.
`o = contraction(H_u, q_hat)`.

For state perturbation `E` with all drivers frozen:

`E_d = exp(d) * E`.
`E_u = E_d - outer(k_hat, beta * contraction(E_d, k_hat))`.
`delta_o = contraction(E_u, q_hat)`.

Thus the corrected one-step state-error transfer is exact linear in `E` up to kernel/reference numeric tolerance.
