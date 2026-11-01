# GDN Quantization Protocol Consistency Audit V1

## Summary
Final classification: `LEGACY_PROTOCOL_MISMATCH_IDENTIFIED`; protocol gate: `PASS`; method design ready: `NO`.

## Table 1: Legacy Protocol Diff
|field|ordinary|cadence_K1|same|
|-|-|-|-|
|aggregation|per prompt mean over tokens, then macro average over prompts|per prompt mean over tokens, then macro average over prompts|True|
|cache_reset_policy|new cache for each prompt and quantizer run|new cache for each prompt and cadence run|False|
|cadence_semantics|ordinary profile has no cadence scheduler; effectively every token for selected layers|token_idx % K == 0; K=1 quantizes indices 0..steps-1; counter resets per prompt|False|
|function|quantizer_profile_one|cadence_one|False|
|hook_timing|after recurrent update, copy fake-quantized state back before next token|after recurrent update, copy fake-quantized state back before next token|True|
|kl_direction|KL(P_fp32 || P_quant)|KL(P_fp32 || P_quant)|True|
|model_path|/data/zypan/modelscope_models/Qwen3.5-9B|/data/zypan/modelscope_models/Qwen3.5-9B|True|
|prefill_quantized|False|False|True|
|prompt_set|PROMPTS[0:8] from dynamic script|PROMPTS[0:8] from dynamic script|True|
|quantized_layer_scope|selected layers only|all 24 GDN layers|False|
|quantized_layers|[1, 9, 17, 25, 29]|[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30]|False|
|quantizer|{'bits': 4, 'signed': True, 'symmetric': True, 'qmin': -7, 'qmax': 7, 'actual_int4_range': '[-7, 7]', 'scale_formula': 'max(abs(x))/(2^(bits-1)-1)', 'epsilon': 1e-12, 'rounding': 'torch.round nearest', 'clamp': 'clamp(-qmax, qmax)', 'zero_point': None, 'granularity': 'per-row', 'state_shape': [1, 32, 128, 128], 'reduction_dims': [-1], 'logical_scale_shape': [32, 128], 'broadcast_scale_shape': [1, 32, 128, 1], 'scale_dtype': 'torch.float32', 'state_dtype_before': 'torch.float32', 'dequantized_state_dtype': 'torch.float32', 'state_dtype_after_copy': 'torch.float32'}|{'bits': 4, 'signed': True, 'symmetric': True, 'qmin': -7, 'qmax': 7, 'actual_int4_range': '[-7, 7]', 'scale_formula': 'max(abs(x))/(2^(bits-1)-1)', 'epsilon': 1e-12, 'rounding': 'torch.round nearest', 'clamp': 'clamp(-qmax, qmax)', 'zero_point': None, 'granularity': 'per-row', 'state_shape': [1, 32, 128, 128], 'reduction_dims': [-1], 'logical_scale_shape': [32, 128], 'broadcast_scale_shape': [1, 32, 128, 1], 'scale_dtype': 'torch.float32', 'state_dtype_before': 'torch.float32', 'dequantized_state_dtype': 'torch.float32', 'state_dtype_after_copy': 'torch.float32'}|True|
|script_path|/data/zypan/experiments/qwen35_gdn_quant/run_dynamic_sensitivity_validation.py|/data/zypan/experiments/qwen35_gdn_quant/run_dynamic_sensitivity_validation.py|True|
|teacher_forced_horizon|128|64|False|

## Table 2: Legacy Reproduction
|metric|reported_value|reproduced_value|absolute_difference|relative_difference|
|-|-|-|-|-|
|ordinary_INT4_row_KL|0.05477945233754376|0.05477945233754376|0.0|0.0|
|cadence_K1_KL|0.2474213619729797|0.4222424483718781|0.17482108639889837|0.7065723226355458|

## Table 3: Prompt-Set Factorial Isolation
|prompt_set|mode|relative_error|KL|top1|global_divergence|quant_calls|
|-|-|-|-|-|-|-|
|P_dynamic=P_cadence|legacy_ordinary_selected_layers|0.08563457872122322|0.05477945233754376|0.947265625|0.07681998569741821|5120|
|P_dynamic=P_cadence|legacy_cadence_all_layers_K1|0.08449023963992704|0.4222424483718781|0.837890625|0.1803414340365131|24576|
|P_dynamic=P_cadence|canonical_ordinary_every_token|0.08449023963992704|0.4222424483718781|0.837890625|0.1803414340365131|24576|
|P_dynamic=P_cadence|canonical_cadence_K1|0.08449023963992704|0.4222424483718781|0.837890625|0.1803414340365131|24576|

## Table 4: Invariance Tests
|test_name|pass|threshold|observed_maximum_difference|notes|
|-|-|-|-|-|
|FP32 reproducibility|True|logits atol=1e-06, state atol=1e-07|0.0|same prompt and same FP32 token path|
|No-op hook equivalence|True|same as FP32|0.0|state.copy_(state) after update|
|Zero perturbation equivalence|True|same as FP32|0.0|add zero tensor after update|
|Cache reset A-B-A|True|same as FP32|0.0|prompt A, prompt B, prompt A|
|Quantization call-count|True|observed == expected|0|K=1 quantizes every evaluated token after update|

## Table 5: Canonical K=1 Parity
|metric|ordinary_every_token|cadence_K1|difference|
|-|-|-|-|
|mean_logit_kl|0.02403084527128385|0.02403084527128385|0.0|
|top1_agreement|1.0|1.0|0.0|
|mean_global_state_divergence|0.14525641672434764|0.14525641672434764|0.0|
|quantization_call_count|384|384|0|

## Table 6: Root-Cause Evidence
|candidate_cause|code_evidence|controlled_run_evidence|supported|
|-|-|-|-|
|LAYER_SCOPE_DIFFERENCE|run_dynamic_sensitivity_validation.py quantizer_profile_one loops over argument layers; cadence_one loops over GDN_LAYERS.|selected layers KL=0.05477945233754376; all GDN K=1 KL=0.4222424483718781; canonical all-layer ordinary and cadence K=1 are equal.|True|
|HORIZON_DIFFERENCE|latest dynamic run used quant_steps=128 and cadence_steps=64.|main audit reruns both compared modes at 128 teacher-forced tokens; reported legacy cadence value is therefore not directly comparable without horizon annotation.|True|
|PROMPT_SET_DIFFERENCE|both dynamic ordinary and dynamic cadence use the same PROMPTS[0:8] in run_dynamic_sensitivity_validation.py.|prompt text hashes and input_ids hashes are identical within the canonical harness.|False|
|QUANTIZER_QRANGE_DIFFERENCE|both paths call fake_quant_state; qmax=(2^(bits-1)-1), clamp(-qmax,qmax). INT4 range is [-7,7].|canonical ordinary and cadence K=1 share the same fake_quant_state and match.|False|

## Scientific Interpretation
{
  "Hypothesis": "larger KL in the legacy cadence result mostly comes from quantizing all 24 GDN states rather than only selected representative layers.",
  "Implementation finding": "the ordinary dynamic quantizer profile used the sampled layer set passed as an argument; cadence used all GDN layers.",
  "Negative result": "the mismatch is not evidence for a cadence mechanism by itself.",
  "Observation": "legacy ordinary INT4-row and cadence K=1 were not run under the same layer scope and, in the cited dynamic run, not the same horizon.",
  "Supported conclusion": "ordinary every-token INT4-row and cadence K=1 are equivalent in the canonical harness when layer scope, horizon, scheduler, quantizer and metrics are unified."
}
