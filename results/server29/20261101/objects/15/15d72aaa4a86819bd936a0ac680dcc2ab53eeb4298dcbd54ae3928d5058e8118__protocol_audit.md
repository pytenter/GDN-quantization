# GDN Functional Column Protection V1 Protocol Audit

```json
{
  "C128_REPRODUCTION_GATE": "PASS",
  "CALIBRATION_EVAL_DISJOINT": "YES",
  "DIAG_M5_DECOMPOSITION_GATE": "PASS",
  "DIAG_M5_DECOMPOSITION_MAX_RELERR": 1.4437312450086395e-15,
  "FP_IDENTITY_GATE": "PASS",
  "HEAD": "5c262b27cf3a8edfb33d18856a0d254599bf072a",
  "HYBRID_PROTECTION_SEMANTICS_GATE": "PASS",
  "M5_implementation_located": true,
  "TASK": "GDN_FUNCTIONAL_COLUMN_PROTECTION_INTERVENTION_V1",
  "branch": "research-sync-2026-09-02",
  "calibration_eval_split_available": true,
  "calibration_units": [
    "test/algebra/1332.json|64",
    "test/algebra/1332.json|128",
    "test/algebra/1332.json|256",
    "test/counting_and_probability/119.json|64",
    "test/counting_and_probability/119.json|128",
    "test/counting_and_probability/119.json|256"
  ],
  "canonical_C128_implementation_located": true,
  "canonical_C128_source": "axis.grouped_quant / state_fake_quant per-column semantics",
  "canonical_FP_state_located": true,
  "cuda_version": "12.1",
  "diagonal_functional_geometry_located": "/data/zypan/runs/gdn_int8_functional_orientation_low_rank_and_stability_audit_v1/geometry_tensors",
  "evaluation_units": [
    "test/geometry/477.json|64",
    "test/geometry/477.json|128",
    "test/geometry/477.json|256"
  ],
  "git_status_start": "?? docs/gdn_functional_column_protection_v1/\n?? experiments/propagation/run_gdn_functional_column_protection_intervention.py\n?? results/propagation/gdn_functional_column_protection_v1_selector_score_audit.json\n?? results/propagation/gdn_functional_column_protection_v1_selector_scores.csv\n?? results/propagation/gdn_functional_column_protection_v1_selector_scores.json",
  "hybrid_state_intervention_feasible": true,
  "model_path": "/data/zypan/modelscope_models/Qwen3.5-9B",
  "quantization": {
    "qrange": [
      -127,
      127
    ],
    "rounding": "torch.round",
    "scale": "amax over K for each [B,H,1,V] C128 group",
    "zero_point": "symmetric zero point"
  },
  "quantization_timing_verified": "after each forward step, cached recurrent state is rewritten",
  "tensor_semantics_verified": "[B,H,K,V], Value column is axis V",
  "timestamp": "2026-09-07 00:12:18 +0800",
  "torch_version": "2.5.1+cu121"
}
```
