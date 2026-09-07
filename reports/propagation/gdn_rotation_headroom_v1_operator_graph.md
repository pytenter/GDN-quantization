# Operator Graph

True Qwen3.5 GatedDeltaNet operator graph:

1. hidden_states are masked for padding.
2. in_proj_qkv(hidden_states) produces concatenated q/k/value source channels.
3. The concatenated q/k/value channels pass through learned depthwise short convolution.
4. The convolved tensor is split into query, key, value and reshaped to heads.
5. beta = sigmoid(in_proj_b(hidden_states)).
6. g = -exp(A_log) * softplus(in_proj_a(hidden_states) + dt_bias).
7. If value heads exceed key heads, query/key are repeat_interleave'd to value heads.
8. recurrent core uses l2norm(query/key), query scaling, scalar per-head decay exp(g), prediction S^T k, rank-one update k delta^T, and read S^T q.
9. recurrent state layout is [batch, value_head, key_dim, value_dim].
10. core output is reshaped by value head, then Qwen3_5RMSNormGated(core, z) is applied.
11. out_proj maps value heads back to hidden size.

Key-basis rotation can preserve the recurrent core if S' = R S, q' = R q, k' = R k.
RMSNormGated/out_proj are after the core and see identical core output under exact core equivariance.
l2norm(q/k) commutes with orthogonal R. Scalar per-head decay commutes with R.

The blocking native-architecture operator is the learned depthwise short convolution before q/k enter the core.
For each convolution lag, q/k channels see a diagonal operator D_l. A dense basis rotation R would require
R D_l = D_l R to fold the rotation into upstream projections while keeping the same depthwise convolution.
Actual per-coordinate learned filters are not scalar multiples of identity, so this commutation condition fails.
