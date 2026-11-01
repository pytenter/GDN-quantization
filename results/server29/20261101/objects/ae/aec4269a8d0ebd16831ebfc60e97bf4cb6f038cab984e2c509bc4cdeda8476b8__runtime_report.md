# Unified final-R runtime report

Status: **PASS**

L2, L6, and L7 use the same loader, tensor boundary, FP32 dense GEMM mechanism, BF16 boundary, canonical INT8-R128 QDQ, and recurrent writeback. They differ only in the values of `R_final`.

- `UNIFIED_FINAL_R_LOADER`: PASS
- `UNIFIED_H_VS_L2_PARITY`: PASS
- `PREFILL_ENDPOINT_DOUBLE_ROTATION`: NO
- `PREFILL_DECODE_BASIS_CONTINUITY`: PASS
- `TP_ROTATION_MAPPING_GATE`: PASS (TP=1, all 18 audited KDA layers mapped)
- `BASELINE_REUSE_GATE`: PASS

The original same-host Step0 parity panel passed exact matrix equality and the required FP/INT8 state, prefill endpoint, first/short decode, logits, scale, qcode, and post-QDQ checks. This authorizes reuse of the complete frozen L2 result (9/20).

Cross-hardware FP bitwise parity failed, so FP_STATE mixed-hardware pooling is forbidden. Under the prospectively frozen `LING_RECURRENT_DENSE_L6_L7_MIXED_GPU_AMENDMENT_V1`, the audited condition-specific INT8-R128 gates passed before generation: L6=PASS, L7=PASS. These claims are restricted to the audited model, final-R, runtime, and INT8-R128 configuration.
