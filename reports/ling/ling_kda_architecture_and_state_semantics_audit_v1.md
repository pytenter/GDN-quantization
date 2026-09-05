# LING_KDA_ARCHITECTURE_AND_STATE_SEMANTICS_AUDIT_V1

## Formal Status
COMPLETE_STATIC_RUNTIME_BLOCKED

## Layer Pattern
- Layer 00: KDA (BailingMoeV3KimiDeltaAttention)
- Layer 01: KDA (BailingMoeV3KimiDeltaAttention)
- Layer 02: KDA (BailingMoeV3KimiDeltaAttention)
- Layer 03: MLA (BailingMoeV3MultiLatentAttention)
- Layer 04: KDA (BailingMoeV3KimiDeltaAttention)
- Layer 05: KDA (BailingMoeV3KimiDeltaAttention)
- Layer 06: KDA (BailingMoeV3KimiDeltaAttention)
- Layer 07: MLA (BailingMoeV3MultiLatentAttention)
- Layer 08: KDA (BailingMoeV3KimiDeltaAttention)
- Layer 09: KDA (BailingMoeV3KimiDeltaAttention)
- Layer 10: KDA (BailingMoeV3KimiDeltaAttention)
- Layer 11: MLA (BailingMoeV3MultiLatentAttention)
- Layer 12: KDA (BailingMoeV3KimiDeltaAttention)
- Layer 13: KDA (BailingMoeV3KimiDeltaAttention)
- Layer 14: KDA (BailingMoeV3KimiDeltaAttention)
- Layer 15: MLA (BailingMoeV3MultiLatentAttention)
- Layer 16: KDA (BailingMoeV3KimiDeltaAttention)
- Layer 17: KDA (BailingMoeV3KimiDeltaAttention)
- Layer 18: KDA (BailingMoeV3KimiDeltaAttention)
- Layer 19: MLA (BailingMoeV3MultiLatentAttention)
- Layer 20: KDA (BailingMoeV3KimiDeltaAttention)
- Layer 21: KDA (BailingMoeV3KimiDeltaAttention)
- Layer 22: KDA (BailingMoeV3KimiDeltaAttention)
- Layer 23: MLA (BailingMoeV3MultiLatentAttention)

## KDA Code Map
```json
{
  "CONFIG_CLASS": {
    "class": "BailingMoeV3Config",
    "file": "/data/zypan/models/Ling-3.0-tiny/configuration_bailing_moe_v3.py",
    "line": 6
  },
  "DECODER_LAYER_SCHEDULE": {
    "class": "BailingMoeV3DecoderLayer",
    "file": "/data/zypan/models/Ling-3.0-tiny/modeling_bailing_moe_v3.py",
    "line": 999
  },
  "KDA_CHUNK_BACKEND": {
    "file": "/data/zypan/models/Ling-3.0-tiny/modeling_bailing_moe_v3.py",
    "function": "chunk_kda",
    "line": 863
  },
  "KDA_CONV_STATE_READ": {
    "file": "/data/zypan/models/Ling-3.0-tiny/modeling_bailing_moe_v3.py",
    "line": 830,
    "statement": "conv_state_q, conv_state_k, conv_state_v = past_key_value.layers[self.layer_idx].values"
  },
  "KDA_MODULE": {
    "class": "BailingMoeV3KimiDeltaAttention",
    "file": "/data/zypan/models/Ling-3.0-tiny/modeling_bailing_moe_v3.py",
    "function": "forward",
    "line": 722
  },
  "KDA_OUT_PROJ": {
    "file": "/data/zypan/models/Ling-3.0-tiny/modeling_bailing_moe_v3.py",
    "line": 784,
    "module": "self.o_proj"
  },
  "KDA_POST_CORE_RMS_GATE": {
    "class": "FusedRMSNormGated",
    "file": "/data/zypan/models/Ling-3.0-tiny/modeling_bailing_moe_v3.py",
    "line": 917,
    "statement": "o = self.o_norm(o, g)"
  },
  "KDA_QKV_SHORT_CONV": {
    "classes": [
      "ShortConvolution"
    ],
    "file": "/data/zypan/models/Ling-3.0-tiny/modeling_bailing_moe_v3.py",
    "lines": {
      "k_conv1d": 754,
      "q_conv1d": 749,
      "v_conv1d": 759
    }
  },
  "KDA_RECURRENT_BACKEND": {
    "file": "/data/zypan/models/Ling-3.0-tiny/modeling_bailing_moe_v3.py",
    "function": "fused_recurrent_kda",
    "line": 880
  },
  "KDA_RECURRENT_STATE_READ": {
    "file": "/data/zypan/models/Ling-3.0-tiny/modeling_bailing_moe_v3.py",
    "line": 824,
    "statement": "recurrent_state = past_key_value.layers[self.layer_idx].keys"
  },
  "KDA_STATE_WRITE": {
    "file": "/data/zypan/models/Ling-3.0-tiny/modeling_bailing_moe_v3.py",
    "line": 909,
    "statement": "past_key_value.layers[self.layer_idx].keys = recurrent_state"
  },
  "MLA_MODULE": {
    "class": "BailingMoeV3MultiLatentAttention",
    "file": "/data/zypan/models/Ling-3.0-tiny/modeling_bailing_moe_v3.py",
    "function": "forward",
    "line": 585
  },
  "MODEL_CACHE_INIT": {
    "class": "BailingMoeV3Model",
    "file": "/data/zypan/models/Ling-3.0-tiny/modeling_bailing_moe_v3.py",
    "line": 1307,
    "statement": "if use_cache and past_key_values is None: past_key_values = DynamicCache()"
  },
  "RMS_NORM": {
    "class": "BailingMoeV3RMSNorm",
    "file": "/data/zypan/models/Ling-3.0-tiny/modeling_bailing_moe_v3.py",
    "function": "forward",
    "line": 207
  }
}
```

## State Semantics
- State runtime type: NOT_CAPTURED_RUNTIME_BLOCKED
- State shape: NOT_CAPTURED_RUNTIME_BLOCKED; code-level expected recurrent_state from Cache.layers[layer].keys passed to chunk_kda/fused_recurrent_kda
- State axes: batch=NOT_VERIFIED_RUNTIME_BLOCKED, head=NOT_VERIFIED_RUNTIME_BLOCKED, key=CODE_SEMANTICS_INFERRED_NOT_RUNTIME_VERIFIED: recurrent matrix axis contracted/written by k, value=CODE_SEMANTICS_INFERRED_NOT_RUNTIME_VERIFIED: recurrent matrix axis written/read as value channel
- R128: axis=NOT_RUNTIME_VERIFIED; mathematical row axis should be Key-side if state is [B,H,d_k,d_v], meaning=Key-side grouping by recurrence semantics, not storage layout; blocked from runtime confirmation
- C128: axis=NOT_RUNTIME_VERIFIED; mathematical column axis should be Value-side if state is [B,H,d_k,d_v], meaning=Value-side grouping by recurrence semantics, not storage layout; blocked from runtime confirmation

## Runtime Blocker
```json
{
  "FP_SMOKE_GATE": "NOT_RUN",
  "MODEL_LOAD_GATE": "BLOCKED",
  "error": "ImportError(\"cannot import name 'is_torch_fx_available' from 'transformers.utils.import_utils' (/data/zypan/transformers-qwen35/src/transformers/utils/import_utils.py)\")",
  "secondary_probe_after_transformers_symbol_monkeypatch": {
    "FP_SMOKE_GATE": "NOT_RUN",
    "MODEL_LOAD_GATE": "BLOCKED",
    "error": "ModuleNotFoundError(\"No module named 'fla.ops'\")",
    "note": "non-persistent monkeypatch only: transformers.utils.import_utils.is_torch_fx_available=lambda: False",
    "traceback_tail": [
      "    model = AutoModelForCausalLM.from_pretrained(",
      "  File \"/data/zypan/transformers-qwen35/src/transformers/models/auto/auto_factory.py\", line 378, in from_pretrained",
      "    model_class = get_class_from_dynamic_module(",
      "  File \"/data/zypan/transformers-qwen35/src/transformers/dynamic_module_utils.py\", line 623, in get_class_from_dynamic_module",
      "    return get_class_in_module(class_name, final_module, force_reload=force_download)",
      "  File \"/data/zypan/transformers-qwen35/src/transformers/dynamic_module_utils.py\", line 309, in get_class_in_module",
      "    module_spec.loader.exec_module(module)",
      "  File \"<frozen importlib._bootstrap_external>\", line 883, in exec_module",
      "  File \"<frozen importlib._bootstrap>\", line 241, in _call_with_frames_removed",
      "  File \"/data/zypan/.cache/huggingface/modules/transformers_modules/Ling_hyphen_3_dot_0_hyphen_tiny/e80739a86f176561/modeling_bailing_moe_v3.py\", line 60, in <module>",
      "    from fla.ops.simple_gla.fused_recurrent import fused_recurrent_simple_gla",
      "ModuleNotFoundError: No module named 'fla.ops'"
    ]
  },
  "traceback_tail": [
    "    model = AutoModelForCausalLM.from_pretrained(",
    "  File \"/data/zypan/transformers-qwen35/src/transformers/models/auto/auto_factory.py\", line 378, in from_pretrained",
    "    model_class = get_class_from_dynamic_module(",
    "  File \"/data/zypan/transformers-qwen35/src/transformers/dynamic_module_utils.py\", line 623, in get_class_from_dynamic_module",
    "    return get_class_in_module(class_name, final_module, force_reload=force_download)",
    "  File \"/data/zypan/transformers-qwen35/src/transformers/dynamic_module_utils.py\", line 309, in get_class_in_module",
    "    module_spec.loader.exec_module(module)",
    "  File \"<frozen importlib._bootstrap_external>\", line 883, in exec_module",
    "  File \"<frozen importlib._bootstrap>\", line 241, in _call_with_frames_removed",
    "  File \"/data/zypan/.cache/huggingface/modules/transformers_modules/Ling_hyphen_3_dot_0_hyphen_tiny/e80739a86f176561/modeling_bailing_moe_v3.py\", line 49, in <module>",
    "    from transformers.utils.import_utils import is_torch_fx_available",
    "ImportError: cannot import name 'is_torch_fx_available' from 'transformers.utils.import_utils' (/data/zypan/transformers-qwen35/src/transformers/utils/import_utils.py)"
  ]
}
```

## Final Summary
```json
{
  "BETA_SHAPE": "CODE_LEVEL: [B,T,H]",
  "C128_FUNCTIONAL_MEANING": "Value-side grouping by recurrence semantics, not storage layout; blocked from runtime confirmation",
  "C128_TENSOR_AXIS": "NOT_RUNTIME_VERIFIED; mathematical column axis should be Value-side if state is [B,H,d_k,d_v]",
  "COMMIT": null,
  "CONFIG_AUDIT": {
    "attention_dropout": 0.0,
    "expert_swiglu_limit_list": null,
    "gated_attention_proj_granularity_type": "head_wise",
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 1536,
    "intermediate_size": 4608,
    "kda_lower_bound": -5,
    "kda_safe_gate": true,
    "kv_lora_rank": 512,
    "layer_group_size": 4,
    "max_window_layers": 20,
    "moe_intermediate_size": 512,
    "moe_router_enable_expert_bias": true,
    "moe_shared_expert_intermediate_size": 512,
    "mtp_use_kda": false,
    "no_kda_lora": true,
    "num_attention_heads": 16,
    "num_experts": 128,
    "num_experts_per_tok": 8,
    "num_hidden_layers": 24,
    "num_key_value_heads": 16,
    "num_kv_heads_for_linear_attn": 0,
    "num_nextn_predict_layers": 0,
    "num_shared_experts": 1,
    "output_router_logits": false,
    "q_lora_rank": 256,
    "qk_head_dim": 192,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "rms_norm_eps": 1e-06,
    "rope_interleave": true,
    "rope_scaling": null,
    "rope_theta": 6000000,
    "router_dtype": "fp32",
    "scale_router_input": false,
    "share_expert_swiglu_limit_list": null,
    "short_conv_kernel_size": 4,
    "use_kda_lora": false,
    "use_mla_nope": false,
    "use_qk_norm": true,
    "use_qkv_bias": false,
    "v_head_dim": 128,
    "value_norm": false
  },
  "CUDA_VERSION": "12.1",
  "DECAY_SHAPE": "CODE_LEVEL: A_log [H], dt_bias [H*128]",
  "DECAY_TYPE": "A_log per head plus dt_bias per value channel are passed to KDA backend; exact scalar/channel semantics not runtime verified",
  "DECODE_STATE_SEMANTICS": "For q_len <= 64 code switches to fused_recurrent_kda and uses same Cache.layers[layer].keys slot; decode runtime not captured",
  "FINAL_SCIENTIFIC_CLASSIFICATION": "LING_KDA_STATIC_ARCHITECTURE_VERIFIED_RUNTIME_BLOCKED",
  "FORMAL_STATUS": "COMPLETE_STATIC_RUNTIME_BLOCKED",
  "FP_SMOKE_GATE": "NOT_RUN",
  "HIDDEN_SIZE": 1536,
  "INSTRUMENTATION_NONINTERFERENCE_GATE": "NOT_RUN_RUNTIME_BLOCKED",
  "KDA_CODE_MAP": {
    "CONFIG_CLASS": {
      "class": "BailingMoeV3Config",
      "file": "/data/zypan/models/Ling-3.0-tiny/configuration_bailing_moe_v3.py",
      "line": 6
    },
    "DECODER_LAYER_SCHEDULE": {
      "class": "BailingMoeV3DecoderLayer",
      "file": "/data/zypan/models/Ling-3.0-tiny/modeling_bailing_moe_v3.py",
      "line": 999
    },
    "KDA_CHUNK_BACKEND": {
      "file": "/data/zypan/models/Ling-3.0-tiny/modeling_bailing_moe_v3.py",
      "function": "chunk_kda",
      "line": 863
    },
    "KDA_CONV_STATE_READ": {
      "file": "/data/zypan/models/Ling-3.0-tiny/modeling_bailing_moe_v3.py",
      "line": 830,
      "statement": "conv_state_q, conv_state_k, conv_state_v = past_key_value.layers[self.layer_idx].values"
    },
    "KDA_MODULE": {
      "class": "BailingMoeV3KimiDeltaAttention",
      "file": "/data/zypan/models/Ling-3.0-tiny/modeling_bailing_moe_v3.py",
      "function": "forward",
      "line": 722
    },
    "KDA_OUT_PROJ": {
      "file": "/data/zypan/models/Ling-3.0-tiny/modeling_bailing_moe_v3.py",
      "line": 784,
      "module": "self.o_proj"
    },
    "KDA_POST_CORE_RMS_GATE": {
      "class": "FusedRMSNormGated",
      "file": "/data/zypan/models/Ling-3.0-tiny/modeling_bailing_moe_v3.py",
      "line": 917,
      "statement": "o = self.o_norm(o, g)"
    },
    "KDA_QKV_SHORT_CONV": {
      "classes": [
        "ShortConvolution"
      ],
      "file": "/data/zypan/models/Ling-3.0-tiny/modeling_bailing_moe_v3.py",
      "lines": {
        "k_conv1d": 754,
        "q_conv1d": 749,
        "v_conv1d": 759
      }
    },
    "KDA_RECURRENT_BACKEND": {
      "file": "/data/zypan/models/Ling-3.0-tiny/modeling_bailing_moe_v3.py",
      "function": "fused_recurrent_kda",
      "line": 880
    },
    "KDA_RECURRENT_STATE_READ": {
      "file": "/data/zypan/models/Ling-3.0-tiny/modeling_bailing_moe_v3.py",
      "line": 824,
      "statement": "recurrent_state = past_key_value.layers[self.layer_idx].keys"
    },
    "KDA_STATE_WRITE": {
      "file": "/data/zypan/models/Ling-3.0-tiny/modeling_bailing_moe_v3.py",
      "line": 909,
      "statement": "past_key_value.layers[self.layer_idx].keys = recurrent_state"
    },
    "MLA_MODULE": {
      "class": "BailingMoeV3MultiLatentAttention",
      "file": "/data/zypan/models/Ling-3.0-tiny/modeling_bailing_moe_v3.py",
      "function": "forward",
      "line": 585
    },
    "MODEL_CACHE_INIT": {
      "class": "BailingMoeV3Model",
      "file": "/data/zypan/models/Ling-3.0-tiny/modeling_bailing_moe_v3.py",
      "line": 1307,
      "statement": "if use_cache and past_key_values is None: past_key_values = DynamicCache()"
    },
    "RMS_NORM": {
      "class": "BailingMoeV3RMSNorm",
      "file": "/data/zypan/models/Ling-3.0-tiny/modeling_bailing_moe_v3.py",
      "function": "forward",
      "line": 207
    }
  },
  "KDA_KEY_DIM": 128,
  "KDA_RUNTIME_BACKEND": "BLOCKED_MISSING_COMPATIBLE_FLA_OPS_KDA",
  "KDA_VALUE_DIM": 128,
  "KDA_VS_GDN_DIFFERENCE_AUDIT": {
    "RMS_path": "SAME_AT_HIGH_LEVEL: RMS/gated post-core path exists",
    "head_merge_semantics": "CONFIRMED_CODE_LEVEL: [B,T,H,D] -> [B,T,H*D] before o_proj",
    "out_proj": "SAME_AT_HIGH_LEVEL: flattened heads through o_proj",
    "output_gate": "CONFIRMED_CODE_LEVEL: FusedRMSNormGated sigmoid gate",
    "query_key_normalization": "DIFFERENT/CONFIRMED_CODE_LEVEL: use_qk_l2norm_in_kernel=True",
    "scalar_decay_vs_channel_wise_decay": "NOT_YET_CONFIRMED; Ling passes A_log[H] and dt_bias[H*D] to KDA backend, exact backend recurrence unavailable",
    "short_convolution": "DIFFERENT: q/k/v use ShortConvolution before KDA backend",
    "state_shape": "NOT_YET_CONFIRMED_RUNTIME_BLOCKED",
    "update_gate": "CONFIRMED_CODE_LEVEL: beta sigmoid per head"
  },
  "KEY_SHAPE": "CODE_LEVEL: k rearranged as [B,T,H,128]",
  "LAYER_PATTERN": "0:KDA,1:KDA,2:KDA,3:MLA,4:KDA,5:KDA,6:KDA,7:MLA,8:KDA,9:KDA,10:KDA,11:MLA,12:KDA,13:KDA,14:KDA,15:MLA,16:KDA,17:KDA,18:KDA,19:MLA,20:KDA,21:KDA,22:KDA,23:MLA",
  "LAYER_PATTERN_ROWS": [
    {
      "attention_layer_type": "linear_attention",
      "layer": 0,
      "module_class": "BailingMoeV3KimiDeltaAttention",
      "semantic_type": "KDA"
    },
    {
      "attention_layer_type": "linear_attention",
      "layer": 1,
      "module_class": "BailingMoeV3KimiDeltaAttention",
      "semantic_type": "KDA"
    },
    {
      "attention_layer_type": "linear_attention",
      "layer": 2,
      "module_class": "BailingMoeV3KimiDeltaAttention",
      "semantic_type": "KDA"
    },
    {
      "attention_layer_type": "attention",
      "layer": 3,
      "module_class": "BailingMoeV3MultiLatentAttention",
      "semantic_type": "MLA"
    },
    {
      "attention_layer_type": "linear_attention",
      "layer": 4,
      "module_class": "BailingMoeV3KimiDeltaAttention",
      "semantic_type": "KDA"
    },
    {
      "attention_layer_type": "linear_attention",
      "layer": 5,
      "module_class": "BailingMoeV3KimiDeltaAttention",
      "semantic_type": "KDA"
    },
    {
      "attention_layer_type": "linear_attention",
      "layer": 6,
      "module_class": "BailingMoeV3KimiDeltaAttention",
      "semantic_type": "KDA"
    },
    {
      "attention_layer_type": "attention",
      "layer": 7,
      "module_class": "BailingMoeV3MultiLatentAttention",
      "semantic_type": "MLA"
    },
    {
      "attention_layer_type": "linear_attention",
      "layer": 8,
      "module_class": "BailingMoeV3KimiDeltaAttention",
      "semantic_type": "KDA"
    },
    {
      "attention_layer_type": "linear_attention",
      "layer": 9,
      "module_class": "BailingMoeV3KimiDeltaAttention",
      "semantic_type": "KDA"
    },
    {
      "attention_layer_type": "linear_attention",
      "layer": 10,
      "module_class": "BailingMoeV3KimiDeltaAttention",
      "semantic_type": "KDA"
    },
    {
      "attention_layer_type": "attention",
      "layer": 11,
      "module_class": "BailingMoeV3MultiLatentAttention",
      "semantic_type": "MLA"
    },
    {
      "attention_layer_type": "linear_attention",
      "layer": 12,
      "module_class": "BailingMoeV3KimiDeltaAttention",
      "semantic_type": "KDA"
    },
    {
      "attention_layer_type": "linear_attention",
      "layer": 13,
      "module_class": "BailingMoeV3KimiDeltaAttention",
      "semantic_type": "KDA"
    },
    {
      "attention_layer_type": "linear_attention",
      "layer": 14,
      "module_class": "BailingMoeV3KimiDeltaAttention",
      "semantic_type": "KDA"
    },
    {
      "attention_layer_type": "attention",
      "layer": 15,
      "module_class": "BailingMoeV3MultiLatentAttention",
      "semantic_type": "MLA"
    },
    {
      "attention_layer_type": "linear_attention",
      "layer": 16,
      "module_class": "BailingMoeV3KimiDeltaAttention",
      "semantic_type": "KDA"
    },
    {
      "attention_layer_type": "linear_attention",
      "layer": 17,
      "module_class": "BailingMoeV3KimiDeltaAttention",
      "semantic_type": "KDA"
    },
    {
      "attention_layer_type": "linear_attention",
      "layer": 18,
      "module_class": "BailingMoeV3KimiDeltaAttention",
      "semantic_type": "KDA"
    },
    {
      "attention_layer_type": "attention",
      "layer": 19,
      "module_class": "BailingMoeV3MultiLatentAttention",
      "semantic_type": "MLA"
    },
    {
      "attention_layer_type": "linear_attention",
      "layer": 20,
      "module_class": "BailingMoeV3KimiDeltaAttention",
      "semantic_type": "KDA"
    },
    {
      "attention_layer_type": "linear_attention",
      "layer": 21,
      "module_class": "BailingMoeV3KimiDeltaAttention",
      "semantic_type": "KDA"
    },
    {
      "attention_layer_type": "linear_attention",
      "layer": 22,
      "module_class": "BailingMoeV3KimiDeltaAttention",
      "semantic_type": "KDA"
    },
    {
      "attention_layer_type": "attention",
      "layer": 23,
      "module_class": "BailingMoeV3MultiLatentAttention",
      "semantic_type": "MLA"
    }
  ],
  "LING_KDA_QUANTIZATION_PROTOCOL_READY": "NO",
  "LOGIT_MAX_ABS_DIFF_WITH_INSTRUMENTATION": null,
  "LOGIT_RELATIVE_DIFF_WITH_INSTRUMENTATION": null,
  "MODEL_LOAD_GATE": "BLOCKED",
  "MODEL_PATH": "/data/zypan/models/Ling-3.0-tiny",
  "NEXT_RECOMMENDED_TASK": "Install/locate the exact Ling-compatible Flash Linear Attention backend providing fla.modules and fla.ops.kda, then rerun LING_KDA_ARCHITECTURE_AND_STATE_SEMANTICS_AUDIT_V1 runtime stages",
  "NUM_KDA_HEADS": 16,
  "NUM_KDA_LAYERS": 18,
  "NUM_LAYERS": 24,
  "NUM_MLA_LAYERS": 6,
  "POST_CORE_GATE": "YES: g_proj(hidden_states) reshaped to [B,T,H,128], consumed by o_norm(o,g)",
  "POST_CORE_OUT_PROJ": "YES: o flattened [B,T,H*128] then o_proj to hidden_size",
  "POST_CORE_RMS": "YES: FusedRMSNormGated(self.head_dim, eps=rms_norm_eps, activation='sigmoid')",
  "PREFILL_STATE_SEMANTICS": "KDA receives initial_state from Cache.layers[layer].keys and writes final recurrent_state back when use_cache; prefill runtime not captured",
  "PYTHON_VERSION": "3.10.18",
  "PYTORCH_VERSION": "2.5.1+cu121",
  "QUERY_READOUT_EQUATION": "Code-level KDA backend returns o from q,k,v,g,beta and recurrent_state; exact q^T S form is inside missing FLA backend and not verified",
  "QUERY_SHAPE": "CODE_LEVEL: q rearranged as [B,T,H,128]",
  "R128_FUNCTIONAL_MEANING": "Key-side grouping by recurrence semantics, not storage layout; blocked from runtime confirmation",
  "R128_TENSOR_AXIS": "NOT_RUNTIME_VERIFIED; mathematical row axis should be Key-side if state is [B,H,d_k,d_v]",
  "STATE_BATCH_AXIS": "NOT_VERIFIED_RUNTIME_BLOCKED",
  "STATE_CAPTURE_GATE": "NOT_RUN_RUNTIME_BLOCKED",
  "STATE_DTYPE": "NOT_CAPTURED_RUNTIME_BLOCKED",
  "STATE_HEAD_AXIS": "NOT_VERIFIED_RUNTIME_BLOCKED",
  "STATE_KEY_AXIS": "CODE_SEMANTICS_INFERRED_NOT_RUNTIME_VERIFIED: recurrent matrix axis contracted/written by k",
  "STATE_RUNTIME_TYPE": "NOT_CAPTURED_RUNTIME_BLOCKED",
  "STATE_SHAPE": "NOT_CAPTURED_RUNTIME_BLOCKED; code-level expected recurrent_state from Cache.layers[layer].keys passed to chunk_kda/fused_recurrent_kda",
  "STATE_UPDATE_EQUATION": "Code passes q,k,v,g,beta,A_log,dt_bias,initial_state to chunk_kda/fused_recurrent_kda; delta/decay recurrence is inside missing FLA backend and was not runtime/source verified here",
  "STATE_VALUE_AXIS": "CODE_SEMANTICS_INFERRED_NOT_RUNTIME_VERIFIED: recurrent matrix axis written/read as value channel",
  "TASK": "LING_KDA_ARCHITECTURE_AND_STATE_SEMANTICS_AUDIT_V1",
  "TRANSFORMERS_VERSION": "5.16.0.dev0",
  "UPDATE_GATE_TYPE": "beta = sigmoid(b_proj(hidden_states)), shape [B,T,H] before backend",
  "VALUE_SHAPE": "CODE_LEVEL: v rearranged as [B,T,H,128]",
  "dependency_probe": {
    "fla": {
      "file": null,
      "status": "OK",
      "version": "UNKNOWN"
    },
    "fla.modules": {
      "error": "ModuleNotFoundError(\"No module named 'fla.modules'\")",
      "status": "FAIL"
    },
    "fla.ops.kda": {
      "error": "ModuleNotFoundError(\"No module named 'fla.ops'\")",
      "status": "FAIL"
    },
    "modelscope": {
      "file": "/data/zypan/.local/lib/python3.10/site-packages/modelscope/__init__.py",
      "status": "OK",
      "version": "1.39.1"
    },
    "torch": {
      "file": "/data/ydai/miniconda3/envs/bitdecode/lib/python3.10/site-packages/torch/__init__.py",
      "status": "OK",
      "version": "2.5.1+cu121"
    },
    "transformers": {
      "file": "/data/zypan/transformers-qwen35/src/transformers/__init__.py",
      "status": "OK",
      "version": "5.16.0.dev0"
    },
    "triton": {
      "file": "/data/ydai/miniconda3/envs/bitdecode/lib/python3.10/site-packages/triton/__init__.py",
      "status": "OK",
      "version": "3.1.0"
    }
  },
  "git": {
    "branch": "research-sync-2026-09-02",
    "head": "73c204c3eee30c37cbb09c4d7d43cb8ed63063c4",
    "status_start": "?? experiments/ling/\n?? reports/ling/\n?? results/ling/"
  },
  "isolated_dependency_probe": {
    "fla": {
      "file": null,
      "status": "OK",
      "version": "UNKNOWN"
    },
    "fla.modules": {
      "error": "ModuleNotFoundError(\"No module named 'fla.modules'\")",
      "status": "FAIL"
    },
    "fla.ops.kda": {
      "error": "ModuleNotFoundError(\"No module named 'fla.ops'\")",
      "status": "FAIL"
    },
    "modelscope": {
      "file": "/data/zypan/.local/lib/python3.10/site-packages/modelscope/__init__.py",
      "status": "OK",
      "version": "1.39.1"
    },
    "torch": {
      "file": "/data/ydai/miniconda3/envs/bitdecode/lib/python3.10/site-packages/torch/__init__.py",
      "status": "OK",
      "version": "2.5.1+cu121"
    },
    "transformers": {
      "file": "/data/zypan/transformers-qwen35/src/transformers/__init__.py",
      "status": "OK",
      "version": "5.16.0.dev0"
    },
    "triton": {
      "file": "/data/ydai/miniconda3/envs/bitdecode/lib/python3.10/site-packages/triton/__init__.py",
      "status": "OK",
      "version": "3.1.0"
    }
  },
  "output_artifact_policy": "small metadata/report only; no large tensors saved",
  "runtime_probe": {
    "FP_SMOKE_GATE": "NOT_RUN",
    "MODEL_LOAD_GATE": "BLOCKED",
    "error": "ImportError(\"cannot import name 'is_torch_fx_available' from 'transformers.utils.import_utils' (/data/zypan/transformers-qwen35/src/transformers/utils/import_utils.py)\")",
    "secondary_probe_after_transformers_symbol_monkeypatch": {
      "FP_SMOKE_GATE": "NOT_RUN",
      "MODEL_LOAD_GATE": "BLOCKED",
      "error": "ModuleNotFoundError(\"No module named 'fla.ops'\")",
      "note": "non-persistent monkeypatch only: transformers.utils.import_utils.is_torch_fx_available=lambda: False",
      "traceback_tail": [
        "    model = AutoModelForCausalLM.from_pretrained(",
        "  File \"/data/zypan/transformers-qwen35/src/transformers/models/auto/auto_factory.py\", line 378, in from_pretrained",
        "    model_class = get_class_from_dynamic_module(",
        "  File \"/data/zypan/transformers-qwen35/src/transformers/dynamic_module_utils.py\", line 623, in get_class_from_dynamic_module",
        "    return get_class_in_module(class_name, final_module, force_reload=force_download)",
        "  File \"/data/zypan/transformers-qwen35/src/transformers/dynamic_module_utils.py\", line 309, in get_class_in_module",
        "    module_spec.loader.exec_module(module)",
        "  File \"<frozen importlib._bootstrap_external>\", line 883, in exec_module",
        "  File \"<frozen importlib._bootstrap>\", line 241, in _call_with_frames_removed",
        "  File \"/data/zypan/.cache/huggingface/modules/transformers_modules/Ling_hyphen_3_dot_0_hyphen_tiny/e80739a86f176561/modeling_bailing_moe_v3.py\", line 60, in <module>",
        "    from fla.ops.simple_gla.fused_recurrent import fused_recurrent_simple_gla",
        "ModuleNotFoundError: No module named 'fla.ops'"
      ]
    },
    "traceback_tail": [
      "    model = AutoModelForCausalLM.from_pretrained(",
      "  File \"/data/zypan/transformers-qwen35/src/transformers/models/auto/auto_factory.py\", line 378, in from_pretrained",
      "    model_class = get_class_from_dynamic_module(",
      "  File \"/data/zypan/transformers-qwen35/src/transformers/dynamic_module_utils.py\", line 623, in get_class_from_dynamic_module",
      "    return get_class_in_module(class_name, final_module, force_reload=force_download)",
      "  File \"/data/zypan/transformers-qwen35/src/transformers/dynamic_module_utils.py\", line 309, in get_class_in_module",
      "    module_spec.loader.exec_module(module)",
      "  File \"<frozen importlib._bootstrap_external>\", line 883, in exec_module",
      "  File \"<frozen importlib._bootstrap>\", line 241, in _call_with_frames_removed",
      "  File \"/data/zypan/.cache/huggingface/modules/transformers_modules/Ling_hyphen_3_dot_0_hyphen_tiny/e80739a86f176561/modeling_bailing_moe_v3.py\", line 49, in <module>",
      "    from transformers.utils.import_utils import is_torch_fx_available",
      "ImportError: cannot import name 'is_torch_fx_available' from 'transformers.utils.import_utils' (/data/zypan/transformers-qwen35/src/transformers/utils/import_utils.py)"
    ]
  },
  "six_question_answers": {
    "Q1": "24 layers: 18 KDA and 6 MLA by config+modeling schedule; runtime instantiation blocked by missing FLA backend.",
    "Q2": "KDA code/config use 16 heads, d_k=128, d_v=128; 128x128 per-head recurrent state is the intended mathematical state but runtime tensor was not captured.",
    "Q3": "Runtime recurrent state shape was not captured because model load is blocked by missing compatible fla.modules/fla.ops.kda.",
    "Q4": "By recurrence API semantics, Key-side is the k-associated state axis and Value-side is the v-associated axis; tensor axis indices remain not runtime verified.",
    "Q5": "If runtime state is [B,H,d_k,d_v], R128 maps to Key-side axis d_k and C128 maps to Value-side axis d_v; this is not yet a passed gate.",
    "Q6": "Safe instrumentation is not yet verified; canonical point should be read-only hooks around BailingMoeV3KimiDeltaAttention.forward and Cache.layers[layer].keys after compatible backend is available."
  }
}
```
