# GDN INT8 Layer-Local Operator-Invariant Update Transduction Causal V1

## Task
GDN_INT8_LAYER_LOCAL_OPERATOR_INVARIANT_UPDATE_TRANSDUCTION_CAUSAL_V1

## Stage0 Result
STAGE0 = FAIL

## Recurrence Boundary
{
  "module_path": "transformers.models.qwen3_5.modeling_qwen3_5",
  "source_file": "/data/zypan/transformers-qwen35/src/transformers/models/qwen3_5/modeling_qwen3_5.py",
  "function_name": "torch_recurrent_gated_delta_rule",
  "function_start_line": 330,
  "relevant_source_lines": {
    "function_signature": [
      330,
      340
    ],
    "qk_l2norm_and_transpose": [
      342,
      353
    ],
    "initial_state_before_update": [
      358,
      362
    ],
    "per_token_recurrence_update": [
      364,
      375
    ]
  },
  "boundary_semantics": "Function entry receives already-computed query/key/value/g/beta and initial_state before recurrence update is applied."
}

## Canonical Unit
{
  "problem_id": "test/algebra/1332.json",
  "role": "INT8-row truncated pathological candidate",
  "t0": 64,
  "teacher_forced_continuation_source": "P0_FP_STATE_DECODED_RESPONSE_RETOKENIZED",
  "layer_selection": {
    "layer_idx": 25,
    "selection_rule": "max prior primary R_STRUCT_C_NORM/REAL_C J_key ratio for same prompt/t0",
    "prior_ratio": 4.825513786648506,
    "prior_R_J_key": 0.014004440648803095,
    "prior_C_J_key": 0.0029021657098412664
  },
  "head_selection": {
    "head_idx": 30,
    "selection_rule": "max R_STRUCT_C_NORM residual norm within predetermined layer",
    "selected_head_residual_norm": 0.08219899982213974,
    "num_heads": 32
  },
  "target_layer": 25,
  "target_head": 30,
  "tensor_shapes": {
    "query": [
      1,
      1,
      32,
      128
    ],
    "key": [
      1,
      1,
      32,
      128
    ],
    "value": [
      1,
      1,
      32,
      128
    ],
    "g": [
      1,
      1,
      32
    ],
    "beta": [
      1,
      1,
      32
    ],
    "initial_state": [
      1,
      32,
      128,
      128
    ],
    "final_state": [
      1,
      32,
      128,
      128
    ],
    "core_output": [
      1,
      1,
      32,
      128
    ],
    "use_qk_l2norm_in_kernel": true
  }
}

## Gates
{
  "PROTOCOL_GATE": "PASS",
  "STATE_SEMANTICS_GATE": "PASS",
  "RECURRENCE_BOUNDARY_GATE": "PASS",
  "RESIDUAL_CONSTRUCTION_GATE": "PASS",
  "NORM_MATCH_GATE": "PASS",
  "OPERATOR_INVARIANCE_GATE": "PASS",
  "RECURRENCE_REPLAY_IDENTITY_GATE": "FAIL",
  "ALGEBRA_IDENTITY_GATE": "PASS",
  "BRANCH_CONSTRUCTION_GATE": "PASS",
  "INSTRUMENTATION_NONINTERFERENCE_GATE": "PASS",
  "METRIC_GATE": "PASS",
  "REPRODUCIBILITY_GATE": "PASS"
}

## Operator Identity Errors
{
  "max_operator_relative_error": 0.0,
  "threshold": 1e-08,
  "max_q_relative_error": 0.0,
  "max_k_relative_error": 0.0,
  "max_v_relative_error": 0.0,
  "max_g_relative_error": 0.0,
  "max_beta_relative_error": 0.0
}

## Replay Identity
{
  "max_relative_error": 0.0015655453728488894,
  "threshold": 1e-05,
  "by_condition": {
    "R_STRUCT_C_NORM": {
      "fp_next_state": {
        "max_abs_error": 0.0,
        "relative_error": 0.0,
        "cosine": 0.9999999403953552,
        "exact_equal": true
      },
      "fp_core_output": {
        "max_abs_error": 0.00020609050989151,
        "relative_error": 0.0015655453728488894,
        "cosine": 0.9999988079071045,
        "exact_equal": false
      },
      "perturbed_next_state": {
        "max_abs_error": 0.0,
        "relative_error": 0.0,
        "cosine": 1.0,
        "exact_equal": true
      },
      "perturbed_core_output": {
        "max_abs_error": 0.00020609050989151,
        "relative_error": 0.0015654946726491625,
        "cosine": 0.9999987483024597,
        "exact_equal": false
      }
    },
    "REAL_C": {
      "fp_next_state": {
        "max_abs_error": 0.0,
        "relative_error": 0.0,
        "cosine": 0.9999999403953552,
        "exact_equal": true
      },
      "fp_core_output": {
        "max_abs_error": 0.00020609050989151,
        "relative_error": 0.0015655453728488894,
        "cosine": 0.9999988079071045,
        "exact_equal": false
      },
      "perturbed_next_state": {
        "max_abs_error": 0.0,
        "relative_error": 0.0,
        "cosine": 1.0,
        "exact_equal": true
      },
      "perturbed_core_output": {
        "max_abs_error": 0.00020609050989151,
        "relative_error": 0.0015652634914332518,
        "cosine": 0.9999988079071045,
        "exact_equal": false
      }
    }
  }
}

## Algebra Identity
{
  "max_relative_error": 2.8430711730328304e-06,
  "threshold": 1e-05,
  "by_condition": {
    "R_STRUCT_C_NORM": {
      "max_abs_error": 1.019798219203949e-07,
      "relative_error": 2.8430711730328304e-06,
      "cosine": 1.0,
      "exact_equal": false
    },
    "REAL_C": {
      "max_abs_error": 5.21540641784668e-08,
      "relative_error": 2.5774466461864757e-06,
      "cosine": 1.0,
      "exact_equal": false
    }
  }
}

## Branch Construction
{
  "max_relative_error": 9.232182372078334e-07,
  "threshold": 1e-05,
  "by_condition": {
    "R_STRUCT_C_NORM": {
      "additive_identity": {
        "max_abs_error": 1.4901161193847656e-08,
        "relative_error": 7.627841093932616e-07,
        "cosine": 1.0,
        "exact_equal": false
      },
      "FULL_matches_native_perturbed": {
        "max_abs_error": 1.1920928955078125e-07,
        "relative_error": 1.605488536947604e-08,
        "cosine": 1.0,
        "exact_equal": false
      }
    },
    "REAL_C": {
      "additive_identity": {
        "max_abs_error": 2.9802322387695312e-08,
        "relative_error": 9.232182372078334e-07,
        "cosine": 1.0,
        "exact_equal": false
      },
      "FULL_matches_native_perturbed": {
        "max_abs_error": 2.9802322387695312e-08,
        "relative_error": 1.4425174791230093e-08,
        "cosine": 1.0,
        "exact_equal": false
      }
    }
  }
}

## Local D/U Metrics
{
  "R_STRUCT_C_NORM": {
    "E_norm": 0.08219899982213974,
    "addressed_error_norm": 0.01567913219332695,
    "addressed_error_over_E": 0.19074602157036388,
    "D_norm": 0.08215565234422684,
    "D_over_E": 0.999472651990825,
    "U_norm": 0.008618003688752651,
    "U_over_E": 0.10484316971368558,
    "U_over_D": 0.1048984877210746,
    "full_next_error_norm": 0.08095071464776993,
    "full_next_error_over_E": 0.9848138641825857,
    "cos_D_U": -0.19124040007591248,
    "cancellation_fraction": 0.014666522181500552,
    "consumed_energy_fraction": 0.02911793746215122
  },
  "REAL_C": {
    "E_norm": 0.08219899237155914,
    "addressed_error_norm": 0.008423160761594772,
    "addressed_error_over_E": 0.10247279824815363,
    "D_norm": 0.08215565234422684,
    "D_over_E": 0.9994727425838035,
    "U_norm": 0.004629768431186676,
    "U_over_E": 0.05632390735646356,
    "U_over_D": 0.056353620219968474,
    "full_next_error_norm": 0.08180972188711166,
    "full_next_error_over_E": 0.9952642912739957,
    "cos_D_U": -0.10273787379264832,
    "cancellation_fraction": 0.004210671429660695,
    "consumed_energy_fraction": 0.008403613080048068
  }
}

## Allowed Conclusion
No D/U causal interpretation is allowed because a Stage0 gate failed.

## Most Important Corrective Result
The previous all-layer construction remains negative evidence; this script does not reinterpret or overwrite it.

## Recommended Next Task
Run Stage A 9-unit layer/head-local operator audit only if explicitly requested; do not start Pilot/Formal automatically.
