# KDA_ROTATION_PREFILL_FULL_PATH_FIRST_DIVERGENCE_V1

FULL_PATH_TRACE_AUDIT = PASS

## Actual runtime sequence

1. decoder layer input hidden
2. input RMSNorm
3. q/k/v projections
4. causal short convolutions
5. recurrence decay and beta projections
6. Value rotation immediately before KDA operator
7. token-wise fused_recurrent_kda
8. raw core output
9. inverse Value map-back
10. FusedRMSNormGated
11. head merge
12. o_proj
13. attention residual
14. post-attention RMSNorm
15. MLP/MoE
16. final residual/layer output
17. prefill endpoint recurrent-state basis handling
18. INT8_R128 endpoint Q/DQ

## Coordinate rules

- q, k, beta, decay and model hidden tensors: BASIS_INVARIANT/NATIVE_BASIS.
- v and recurrent state inside RS KDA: ROTATED_VALUE_BASIS.
- raw RS core output: ROTATED_VALUE_BASIS.
- runtime mapped output and all downstream tensors: NATIVE_BASIS.
- state comparison: S_rot @ R.T; Value/output comparison: x_rot @ R.T.

## Fused or absent stages

- RMSNorm normalized output before learned scale is fused
- dynamic sigmoid gate output is fused into FusedRMSNormGated
- head merge is a view/rearrange rather than a module
