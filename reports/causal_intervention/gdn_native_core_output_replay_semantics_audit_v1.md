# GDN Native Core Output Replay Semantics Audit V1

## 1. Task
GDN_NATIVE_CORE_OUTPUT_REPLAY_SEMANTICS_AUDIT_V1

## 2. Previous Hard-Stop Context
Previous layer-local causal Stage0 stopped at RECURRENCE_REPLAY_IDENTITY_GATE because native core output differed from FP32 replay by 0.0015655454 relative error.

## 3. Active GDN Backend
{
  "ACTIVE_BACKEND": "torch_recurrent_gated_delta_rule via Qwen3_5GatedDeltaNet cached decode path",
  "module_path": "transformers.models.qwen3_5.modeling_qwen3_5",
  "source_file": "/data/zypan/transformers-qwen35/src/transformers/models/qwen3_5/modeling_qwen3_5.py",
  "class": "Qwen3_5GatedDeltaNet",
  "forward_lines": {
    "qkv_creation": [
      446,
      504
    ],
    "cached_decode_recurrent_call": [
      505,
      518
    ],
    "cache_update": [
      533,
      535
    ],
    "post_kernel_norm_and_projection": [
      537,
      544
    ]
  },
  "function": "torch_recurrent_gated_delta_rule",
  "function_lines": {
    "qk_l2norm": [
      342,
      345
    ],
    "transpose_and_fp32_compute_cast": [
      346,
      353
    ],
    "state_update_and_readout": [
      364,
      375
    ],
    "output_cast_to_initial_dtype": [
      377,
      380
    ]
  },
  "fused_vs_unfused": "Python torch recurrent fallback wrapped by kernel decorators; active captured path calls torch_recurrent_gated_delta_rule for single-token cached decode."
}

## 4. Native Recurrence Source Trace
{
  "ACTIVE_BACKEND": "torch_recurrent_gated_delta_rule via Qwen3_5GatedDeltaNet cached decode path",
  "module_path": "transformers.models.qwen3_5.modeling_qwen3_5",
  "source_file": "/data/zypan/transformers-qwen35/src/transformers/models/qwen3_5/modeling_qwen3_5.py",
  "class": "Qwen3_5GatedDeltaNet",
  "forward_lines": {
    "qkv_creation": [
      446,
      504
    ],
    "cached_decode_recurrent_call": [
      505,
      518
    ],
    "cache_update": [
      533,
      535
    ],
    "post_kernel_norm_and_projection": [
      537,
      544
    ]
  },
  "function": "torch_recurrent_gated_delta_rule",
  "function_lines": {
    "qk_l2norm": [
      342,
      345
    ],
    "transpose_and_fp32_compute_cast": [
      346,
      353
    ],
    "state_update_and_readout": [
      364,
      375
    ],
    "output_cast_to_initial_dtype": [
      377,
      380
    ]
  },
  "fused_vs_unfused": "Python torch recurrent fallback wrapped by kernel decorators; active captured path calls torch_recurrent_gated_delta_rule for single-token cached decode."
}

## 5. Definition Of Native Core_Output
The tensor previously called native core_output is the raw return from torch_recurrent_gated_delta_rule before gated RMSNorm/out_proj, after q/k L2 normalization, q scaling, FP32 recurrent state update/readout, transpose to [B,T,H,V], and final cast back to the query input dtype.

## 6. Runtime Tensor Boundary Map
{
  "q_entering_kernel": {
    "shape": [
      1,
      1,
      32,
      128
    ],
    "dtype": "torch.bfloat16",
    "device": "cuda:0",
    "stride": [
      4096,
      4096,
      128,
      1
    ],
    "contiguous": true,
    "min": -0.279296875,
    "max": 5.75,
    "mean": -0.01500696036964655,
    "norm": 17.601667404174805
  },
  "q_after_l2norm_transpose": {
    "shape": [
      1,
      32,
      1,
      128
    ],
    "dtype": "torch.float32",
    "device": "cuda:0",
    "stride": [
      4096,
      128,
      4096,
      1
    ],
    "contiguous": true,
    "min": -0.24609375,
    "max": 0.98828125,
    "mean": -0.007755442522466183,
    "norm": 5.657442092895508
  },
  "q_after_scale": {
    "shape": [
      1,
      32,
      128
    ],
    "dtype": "torch.float32",
    "device": "cuda:0",
    "stride": [
      4096,
      128,
      1
    ],
    "contiguous": true,
    "min": -0.02175181917846203,
    "max": 0.08735254406929016,
    "mean": -0.0006854907260276377,
    "norm": 0.5000519156455994
  },
  "k_entering_kernel": {
    "shape": [
      1,
      1,
      32,
      128
    ],
    "dtype": "torch.bfloat16",
    "device": "cuda:0",
    "stride": [
      4096,
      4096,
      128,
      1
    ],
    "contiguous": true,
    "min": -0.279296875,
    "max": 5.8125,
    "mean": -0.05525457486510277,
    "norm": 21.954750061035156
  },
  "k_after_l2norm_transpose": {
    "shape": [
      1,
      32,
      1,
      128
    ],
    "dtype": "torch.float32",
    "device": "cuda:0",
    "stride": [
      4096,
      128,
      4096,
      1
    ],
    "contiguous": true,
    "min": -0.1767578125,
    "max": 0.86328125,
    "mean": -0.013682454824447632,
    "norm": 5.6526689529418945
  },
  "k_t": {
    "shape": [
      1,
      32,
      128
    ],
    "dtype": "torch.float32",
    "device": "cuda:0",
    "stride": [
      4096,
      128,
      1
    ],
    "contiguous": true,
    "min": -0.1767578125,
    "max": 0.86328125,
    "mean": -0.013682454824447632,
    "norm": 5.6526689529418945
  },
  "v_entering_kernel": {
    "shape": [
      1,
      1,
      32,
      128
    ],
    "dtype": "torch.bfloat16",
    "device": "cuda:0",
    "stride": [
      4096,
      4096,
      128,
      1
    ],
    "contiguous": true,
    "min": -0.279296875,
    "max": 1.984375,
    "mean": -0.042116131633520126,
    "norm": 10.396050453186035
  },
  "g_entering_kernel": {
    "shape": [
      1,
      1,
      32
    ],
    "dtype": "torch.float32",
    "device": "cuda:0",
    "stride": [
      32,
      32,
      1
    ],
    "contiguous": true,
    "min": -0.9491671919822693,
    "max": -0.00010886428935918957,
    "mean": -0.11068613827228546,
    "norm": 1.3545353412628174
  },
  "beta_entering_kernel": {
    "shape": [
      1,
      1,
      32
    ],
    "dtype": "torch.bfloat16",
    "device": "cuda:0",
    "stride": [
      32,
      32,
      1
    ],
    "contiguous": true,
    "min": 0.08154296875,
    "max": 0.9609375,
    "mean": 0.6671600341796875,
    "norm": 3.9591968059539795
  },
  "state_before": {
    "shape": [
      1,
      32,
      128,
      128
    ],
    "dtype": "torch.float32",
    "device": "cuda:0",
    "stride": [
      524288,
      16384,
      128,
      1
    ],
    "contiguous": true,
    "min": -0.8478134274482727,
    "max": 1.6916308403015137,
    "mean": 0.0007838820456527174,
    "norm": 16.50672149658203
  },
  "state_after_native": {
    "shape": [
      1,
      32,
      128,
      128
    ],
    "dtype": "torch.float32",
    "device": "cuda:0",
    "stride": [
      524288,
      16384,
      128,
      1
    ],
    "contiguous": true,
    "min": -0.8452103137969971,
    "max": 1.660636067390442,
    "mean": 0.0007666089804843068,
    "norm": 16.530778884887695
  },
  "kernel_raw_output_native_core_output": {
    "shape": [
      1,
      1,
      32,
      128
    ],
    "dtype": "torch.bfloat16",
    "device": "cuda:0",
    "stride": [
      4096,
      128,
      128,
      1
    ],
    "contiguous": true,
    "min": -0.0235595703125,
    "max": 0.123046875,
    "mean": -0.000498517241794616,
    "norm": 0.3506099581718445
  },
  "post_kernel_output_before_norm": {
    "shape": [
      32,
      128
    ],
    "dtype": "torch.bfloat16",
    "device": "cuda:0",
    "stride": [
      128,
      1
    ],
    "contiguous": true,
    "min": -0.0235595703125,
    "max": 0.123046875,
    "mean": -0.000498517241794616,
    "norm": 0.3506099581718445
  },
  "preproj_after_gated_rmsnorm": {
    "shape": [
      32,
      128
    ],
    "dtype": "torch.float32",
    "device": "cuda:0",
    "stride": [
      128,
      1
    ],
    "contiguous": true,
    "min": -5.21875,
    "max": 3.34375,
    "mean": 0.006917234510183334,
    "norm": 15.27873706817627
  },
  "postproj_after_out_proj": {
    "shape": [
      1,
      1,
      4096
    ],
    "dtype": "torch.float32",
    "device": "cuda:0",
    "stride": [
      4096,
      4096,
      1
    ],
    "contiguous": true,
    "min": -0.7265625,
    "max": 6.71875,
    "mean": 0.0005521345883607864,
    "norm": 14.750527381896973
  }
}

## 7. Query Semantics
Native readout uses query after optional l2norm(dim=-1, eps=1e-6), transpose to [B,H,T,K], float32 cast, and scale by K**-0.5 before contraction with state.

## 8. Dtype Semantics
query/key/value/beta/g enter with model dtype; kernel transposes and casts to FP32 for recurrence/readout; core_attn_out is finally cast to initial_dtype before return. Missing this final cast caused the previous mismatch.

## 9. Layout / Transpose Semantics
State layout is [B,H,K,V]; q_t layout is [B,H,K]; readout is sum over K producing [B,H,V], then stored as [B,T,H,V] by transpose(1,2).contiguous().

## 10. Fused-Kernel Semantics
The active audited path is the torch recurrent fallback for single-token cached decode, wrapped by kernel decorators. State update and readout occur in one function invocation; no additional postprocessing is included in native_core_output beyond output dtype cast.

## 11. Readout Ladder
{
  "fp32_scaled_no_cast": {
    "median": 0.0015655453728488894,
    "p95": 0.0018173032246957734,
    "max": 0.001861445419468337
  },
  "fp32_scaled_then_cast_native": {
    "median": 0.0,
    "p95": 0.0,
    "max": 0.0
  },
  "fp32_unscaled_then_scale_cast_native": {
    "median": 0.0,
    "p95": 1.8660859191904082e-06,
    "max": 3.0826003969133697e-06
  },
  "native_dtype_product_sum": {
    "median": 0.002185571583585448,
    "p95": 0.002746783102032564,
    "max": 0.002781641469642601
  }
}

## 12. FP Native-vs-Reference Baseline
{
  "old_fp32_no_cast_relative_error": {
    "median": 0.0015655453728488894,
    "p95": 0.0018173032246957734,
    "max": 0.001861445419468337
  },
  "fixed_native_cast_relative_error": {
    "median": 0.0,
    "p95": 0.0,
    "max": 0.0
  },
  "fixed_native_cast_absolute_error": {
    "median": 0.0,
    "p95": 0.0,
    "max": 0.0
  }
}

## 13. Perturbed Native-vs-Reference Comparison
{
  "R perturbed relative error": {
    "median": 0.0,
    "p95": 0.0,
    "max": 0.0
  },
  "C perturbed relative error": {
    "median": 0.0,
    "p95": 0.0,
    "max": 0.0
  },
  "R perturbed max relative error": 0.0,
  "C perturbed max relative error": 0.0
}

## 14. Root Cause
DTYPE_SEMANTICS

## 15. Gate Results
{
  "SOURCE_TRACE_GATE": "PASS",
  "NATIVE_OUTPUT_SEMANTICS_GATE": "PASS",
  "QUERY_SEMANTICS_GATE": "PASS",
  "DTYPE_SEMANTICS_GATE": "PASS",
  "LAYOUT_SEMANTICS_GATE": "PASS",
  "NATIVE_FP_BASELINE_GATE": "PASS",
  "INSTRUMENTATION_NONINTERFERENCE_GATE": "PASS",
  "CORE_OUTPUT_REPLAY_IDENTITY_GATE": "PASS"
}

## 16. Replay Fix If Applicable
In recurrence replay, compare core output after casting FP32 readout to rec['query'].dtype, matching modeling_qwen3_5.py line 379.

## 17. Threshold Assessment
{
  "OLD_THRESHOLD": 1e-05,
  "THRESHOLD_CHANGE_REQUIRED": "NO",
  "PROPOSED_THRESHOLD": 1e-05,
  "JUSTIFICATION": "Exact semantic bug fixed by matching native final output cast; no tolerance relaxation is needed."
}

## 18. Whether Previous Causal Stage A May Resume
YES

## 19. Claims Not Supported
UPDATE_TRANSDUCTION_CAUSAL remains NOT_YET_TESTED. No Stage A, Pilot, Formal, or method design was run.

## 20. Artifact Paths
{
  "script": "/data/zypan/experiments/qwen35_gdn_quant/audit_gdn_native_core_output_replay_semantics.py",
  "stage0": "/data/zypan/results/gdn_native_core_output_replay_semantics_audit_v1_stage0.json",
  "tensor_trace": "/data/zypan/results/gdn_native_core_output_replay_semantics_audit_v1_tensor_trace.json",
  "fp_baseline": "/data/zypan/results/gdn_native_core_output_replay_semantics_audit_v1_fp_baseline.json",
  "replay_candidates": "/data/zypan/results/gdn_native_core_output_replay_semantics_audit_v1_replay_candidates.json",
  "raw": "/data/zypan/results/gdn_native_core_output_replay_semantics_audit_v1_raw.npz",
  "report": "/data/zypan/reports/gdn_native_core_output_replay_semantics_audit_v1.md",
  "figures": "/data/zypan/results/gdn_native_core_output_replay_semantics_audit_v1_figures"
}
