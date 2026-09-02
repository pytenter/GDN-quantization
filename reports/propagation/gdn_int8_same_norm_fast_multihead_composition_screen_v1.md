# GDN INT8 Same-Norm Fast Multihead Composition Screen V1

## 1. Task

GDN_INT8_SAME_NORM_FAST_MULTIHEAD_COMPOSITION_SCREEN_V1

## 2. Reason previous experiment was stopped

The prior comprehensive run mixed scope/headwise matching with single-head baselines, pairwise interactions, direction-related work, and 64-token rollouts in one serial GPU0 process; it reached only 2/9 units after about 3 hours.

## 3. Preserved 2/9 partial diagnostic

{
  "PARTIAL_DIAGNOSTIC_ONLY": true,
  "PREVIOUS_TASK": "GDN_INT8_SAME_NORM_DISTRIBUTED_COMPOSITION_FUNCTIONAL_ALIGNMENT_V1",
  "completed": "2 / 9",
  "completed_units": [
    "test/algebra/1332.json|128",
    "test/algebra/1332.json|64"
  ],
  "headwise_norm_match": {
    "S1": {
      "R_gt_C": "1/2",
      "median_gap": 7.72e-06
    },
    "S2": {
      "R_gt_C": "2/2",
      "median_gap": 5.6e-05
    },
    "S4": {
      "R_gt_C": "2/2",
      "median_gap": 2.19e-05
    },
    "S8": {
      "R_gt_C": "1/2",
      "median_gap": -1.05e-06
    }
  },
  "pairwise_partial": {
    "I_R_gt_I_C": "6/24",
    "median_I_R_minus_I_C": -3.31e-05,
    "pairs": 24
  },
  "preserved_fields_observed": [
    "canonical_head",
    "composition",
    "head_set",
    "pairwise",
    "problem_id",
    "singles",
    "t0",
    "target_layer"
  ],
  "scope_level_norm_match": {
    "S1": {
      "R_gt_C": "1/2",
      "median_gap": 7.72e-06
    },
    "S2": {
      "R_gt_C": "1/2",
      "median_gap": 8.62e-06
    },
    "S4": {
      "R_gt_C": "1/2",
      "median_gap": 1.29e-05
    },
    "S8": {
      "R_gt_C": "2/2",
      "median_gap": 5.23e-05
    }
  }
}

## 4. Scientific question

Does the R/C behavioral gap grow with multi-head scope, and does it survive per-head norm matching?

## 5. Protocol

Canonical 9 units; horizon 64; scopes [1, 2, 4, 8]; families ['scope', 'headwise']; no pairwise, direction, epsilon ladder, gradient, or extra tracing.

## 6. Canonical units

[
  "test/algebra/1332.json|64",
  "test/algebra/1332.json|128",
  "test/algebra/1332.json|256",
  "test/counting_and_probability/119.json|64",
  "test/counting_and_probability/119.json|128",
  "test/counting_and_probability/119.json|256",
  "test/geometry/477.json|64",
  "test/geometry/477.json|128",
  "test/geometry/477.json|256"
]

## 7. Head-set manifest

S1 is the canonical StageA head; S2/S4/S8 use the deterministic stored previous head-set rule, verified against prior partial units where stored.

## 8. Stage0 gates

{
  "STAGE0": "PASS",
  "canonical_units": [
    "test/algebra/1332.json|64",
    "test/algebra/1332.json|128",
    "test/algebra/1332.json|256",
    "test/counting_and_probability/119.json|64",
    "test/counting_and_probability/119.json|128",
    "test/counting_and_probability/119.json|256",
    "test/geometry/477.json|64",
    "test/geometry/477.json|128",
    "test/geometry/477.json|256"
  ],
  "gates": {
    "HEADWISE_NORM_MATCH_GATE": "PASS",
    "HEAD_SET_GATE": "PASS",
    "INSTRUMENTATION_NONINTERFERENCE_GATE": "PASS",
    "METRIC_GATE": "PASS",
    "OPERATOR_INVARIANCE_GATE": "PASS",
    "PROTOCOL_GATE": "PASS",
    "REPLAY_REGRESSION_GATE": "PASS",
    "SCOPE_NORM_MATCH_GATE": "PASS"
  },
  "git_commit": "78b566659776a32183fcc0d6271fe2219bcf5c58",
  "head_set_manifest_source": "deterministic previous script head_set(canonical_head); verified against stored previous partial units when available",
  "headwise_max_norm_match_error_S8": 9.875252842457871e-08,
  "previous_partial": {
    "PARTIAL_DIAGNOSTIC_ONLY": true,
    "PREVIOUS_TASK": "GDN_INT8_SAME_NORM_DISTRIBUTED_COMPOSITION_FUNCTIONAL_ALIGNMENT_V1",
    "completed": "2 / 9",
    "completed_units": [
      "test/algebra/1332.json|128",
      "test/algebra/1332.json|64"
    ],
    "headwise_norm_match": {
      "S1": {
        "R_gt_C": "1/2",
        "median_gap": 7.72e-06
      },
      "S2": {
        "R_gt_C": "2/2",
        "median_gap": 5.6e-05
      },
      "S4": {
        "R_gt_C": "2/2",
        "median_gap": 2.19e-05
      },
      "S8": {
        "R_gt_C": "1/2",
        "median_gap": -1.05e-06
      }
    },
    "pairwise_partial": {
      "I_R_gt_I_C": "6/24",
      "median_I_R_minus_I_C": -3.31e-05,
      "pairs": 24
    },
    "preserved_fields_observed": [
      "canonical_head",
      "composition",
      "head_set",
      "pairwise",
      "problem_id",
      "singles",
      "t0",
      "target_layer"
    ],
    "scope_level_norm_match": {
      "S1": {
        "R_gt_C": "1/2",
        "median_gap": 7.72e-06
      },
      "S2": {
        "R_gt_C": "1/2",
        "median_gap": 8.62e-06
      },
      "S4": {
        "R_gt_C": "1/2",
        "median_gap": 1.29e-05
      },
      "S8": {
        "R_gt_C": "2/2",
        "median_gap": 5.23e-05
      }
    }
  },
  "regression_head_set": [
    30,
    0,
    4,
    8,
    12,
    16,
    20,
    24
  ],
  "regression_unit": "test/algebra/1332.json|64",
  "scope_norm_match_error_S8": 6.025309795773408e-08,
  "task": "GDN_INT8_SAME_NORM_FAST_MULTIHEAD_COMPOSITION_SCREEN_V1",
  "timestamp": "2026-09-02 15:52:39 +0800"
}

## 9. Scope-level norm-match results

{
  "S1": {
    "R_gt_C": 6,
    "median_R_KL_AUC": 0.00010562757971345096,
    "median_C_KL_AUC": 9.464411847716486e-05,
    "median_RC_GAP": 2.6022942278171955e-05,
    "IQR_RC_GAP": [
      -1.162065343744835e-05,
      4.026269205981809e-05
    ],
    "RC_GAP_by_unit": {
      "test/algebra/1332.json|64": -4.979335838388347e-05,
      "test/algebra/1332.json|128": 6.523465687347963e-05,
      "test/algebra/1332.json|256": -1.162065343744835e-05,
      "test/counting_and_probability/119.json|64": 2.8493180241183754e-05,
      "test/counting_and_probability/119.json|128": 1.8006105697890914e-05,
      "test/counting_and_probability/119.json|256": 2.6022942278171955e-05,
      "test/geometry/477.json|64": -8.231740032602465e-05,
      "test/geometry/477.json|128": 0.0001592666426488203,
      "test/geometry/477.json|256": 4.026269205981809e-05
    }
  },
  "S2": {
    "R_gt_C": 5,
    "median_R_KL_AUC": 9.143102188154811e-05,
    "median_C_KL_AUC": 0.0001423992396023877,
    "median_RC_GAP": 1.6419750620594062e-05,
    "IQR_RC_GAP": [
      -2.498341215016861e-05,
      5.8737167198536446e-05
    ],
    "RC_GAP_by_unit": {
      "test/algebra/1332.json|64": -6.502181823459652e-05,
      "test/algebra/1332.json|128": 8.227008748654639e-05,
      "test/algebra/1332.json|256": 5.8737167198536446e-05,
      "test/counting_and_probability/119.json|64": 6.66424867002625e-05,
      "test/counting_and_probability/119.json|128": 1.6419750620594062e-05,
      "test/counting_and_probability/119.json|256": -1.6800909244485098e-07,
      "test/geometry/477.json|64": -5.09682177208396e-05,
      "test/geometry/477.json|128": 1.9516906492633365e-05,
      "test/geometry/477.json|256": -2.498341215016861e-05
    }
  },
  "S4": {
    "R_gt_C": 5,
    "median_R_KL_AUC": 0.00010935246517811852,
    "median_C_KL_AUC": 6.684432096539121e-05,
    "median_RC_GAP": 6.231694731486737e-06,
    "IQR_RC_GAP": [
      -1.935772821360488e-05,
      3.372416528469251e-05
    ],
    "RC_GAP_by_unit": {
      "test/algebra/1332.json|64": -4.390623994837932e-05,
      "test/algebra/1332.json|128": 6.96358480047349e-05,
      "test/algebra/1332.json|256": 2.86525276772793e-05,
      "test/counting_and_probability/119.json|64": 3.372416528469251e-05,
      "test/counting_and_probability/119.json|128": -1.8010542652542854e-05,
      "test/counting_and_probability/119.json|256": -1.935772821360488e-05,
      "test/geometry/477.json|64": 6.328718257978477e-05,
      "test/geometry/477.json|128": -4.311828608813883e-05,
      "test/geometry/477.json|256": 6.231694731486737e-06
    }
  },
  "S8": {
    "R_gt_C": 6,
    "median_R_KL_AUC": 0.00015587279252965062,
    "median_C_KL_AUC": 9.968501007129619e-05,
    "median_RC_GAP": 4.545219079041539e-05,
    "IQR_RC_GAP": [
      -9.8543648480729e-06,
      6.819002946046907e-05
    ],
    "RC_GAP_by_unit": {
      "test/algebra/1332.json|64": 6.819002946046907e-05,
      "test/algebra/1332.json|128": 3.6448342118075745e-05,
      "test/algebra/1332.json|256": -3.510133576361501e-05,
      "test/counting_and_probability/119.json|64": -2.977764316307676e-05,
      "test/counting_and_probability/119.json|128": 4.854625684651611e-05,
      "test/counting_and_probability/119.json|256": -9.8543648480729e-06,
      "test/geometry/477.json|64": 0.00010559432316117879,
      "test/geometry/477.json|128": 8.151374388974848e-05,
      "test/geometry/477.json|256": 4.545219079041539e-05
    }
  }
}

## 10. Head-wise norm-match results

{
  "S1": {
    "R_gt_C": 6,
    "median_R_KL_AUC": 0.00010562757971345096,
    "median_C_KL_AUC": 9.464411847716486e-05,
    "median_RC_GAP": 2.6022942278171955e-05,
    "IQR_RC_GAP": [
      -1.162065343744835e-05,
      4.026269205981809e-05
    ],
    "RC_GAP_by_unit": {
      "test/algebra/1332.json|64": -4.979335838388347e-05,
      "test/algebra/1332.json|128": 6.523465687347963e-05,
      "test/algebra/1332.json|256": -1.162065343744835e-05,
      "test/counting_and_probability/119.json|64": 2.8493180241183754e-05,
      "test/counting_and_probability/119.json|128": 1.8006105697890914e-05,
      "test/counting_and_probability/119.json|256": 2.6022942278171955e-05,
      "test/geometry/477.json|64": -8.231740032602465e-05,
      "test/geometry/477.json|128": 0.0001592666426488203,
      "test/geometry/477.json|256": 4.026269205981809e-05
    }
  },
  "S2": {
    "R_gt_C": 5,
    "median_R_KL_AUC": 0.00013251135641772572,
    "median_C_KL_AUC": 0.0001423992396023877,
    "median_RC_GAP": 2.2089674245684985e-05,
    "IQR_RC_GAP": [
      -6.343974523776065e-06,
      3.536576052488882e-05
    ],
    "RC_GAP_by_unit": {
      "test/algebra/1332.json|64": 2.2089674245684985e-05,
      "test/algebra/1332.json|128": 8.989746261364511e-05,
      "test/algebra/1332.json|256": 3.474038276384339e-05,
      "test/counting_and_probability/119.json|64": 4.445125288932725e-05,
      "test/counting_and_probability/119.json|128": -5.262603900313084e-05,
      "test/counting_and_probability/119.json|256": -6.608676653218105e-07,
      "test/geometry/477.json|64": -5.168728284075076e-05,
      "test/geometry/477.json|128": 3.536576052488882e-05,
      "test/geometry/477.json|256": -6.343974523776065e-06
    }
  },
  "S4": {
    "R_gt_C": 7,
    "median_R_KL_AUC": 0.00010597401141039745,
    "median_C_KL_AUC": 6.684432096539121e-05,
    "median_RC_GAP": 2.0685526443295606e-05,
    "IQR_RC_GAP": [
      1.2296160770661194e-05,
      2.2599536365669806e-05
    ],
    "RC_GAP_by_unit": {
      "test/algebra/1332.json|64": 2.2599536365669806e-05,
      "test/algebra/1332.json|128": 2.1121858137684396e-05,
      "test/algebra/1332.json|256": 2.0685526443295606e-05,
      "test/counting_and_probability/119.json|64": -2.145459994017462e-05,
      "test/counting_and_probability/119.json|128": 1.2296160770661194e-05,
      "test/counting_and_probability/119.json|256": 1.3638576610493599e-05,
      "test/geometry/477.json|64": 4.385448108210855e-05,
      "test/geometry/477.json|128": -6.733576283941226e-06,
      "test/geometry/477.json|256": 5.323676637576961e-05
    }
  },
  "S8": {
    "R_gt_C": 8,
    "median_R_KL_AUC": 7.779257014156344e-05,
    "median_C_KL_AUC": 9.968501007129619e-05,
    "median_RC_GAP": 1.7923396455662852e-05,
    "IQR_RC_GAP": [
      3.2894612580553937e-06,
      2.874463555188649e-05
    ],
    "RC_GAP_by_unit": {
      "test/algebra/1332.json|64": 2.8981400452667065e-05,
      "test/algebra/1332.json|128": -3.1080381720015684e-05,
      "test/algebra/1332.json|256": 2.2897270690438944e-05,
      "test/counting_and_probability/119.json|64": 1.7923396455662852e-05,
      "test/counting_and_probability/119.json|128": 2.874463555188649e-05,
      "test/counting_and_probability/119.json|256": 3.2894612580553937e-06,
      "test/geometry/477.json|64": 5.645306708310853e-06,
      "test/geometry/477.json|128": 3.2363909193576968e-06,
      "test/geometry/477.json|256": 4.6197724687624734e-05
    }
  }
}

## 11. R/C gap vs scope

{
  "scope": {
    "S1": 2.6022942278171955e-05,
    "S2": 1.6419750620594062e-05,
    "S4": 6.231694731486737e-06,
    "S8": 4.545219079041539e-05
  },
  "headwise": {
    "S1": 2.6022942278171955e-05,
    "S2": 2.2089674245684985e-05,
    "S4": 2.0685526443295606e-05,
    "S8": 1.7923396455662852e-05
  }
}

## 12. Gap-growth analysis

{
  "scope": {
    "S2_minus_S1": {
      "positive_count": 4,
      "median": -1.5863550772968513e-06,
      "IQR": [
        -2.6190951370616806e-05,
        3.1349182605185056e-05
      ],
      "by_unit": {
        "test/algebra/1332.json|64": -1.5228459850713047e-05,
        "test/algebra/1332.json|128": 1.703543061306676e-05,
        "test/algebra/1332.json|256": 7.035782063598479e-05,
        "test/counting_and_probability/119.json|64": 3.814930645907875e-05,
        "test/counting_and_probability/119.json|128": -1.5863550772968513e-06,
        "test/counting_and_probability/119.json|256": -2.6190951370616806e-05,
        "test/geometry/477.json|64": 3.1349182605185056e-05,
        "test/geometry/477.json|128": -0.00013974973615618695,
        "test/geometry/477.json|256": -6.52461042099867e-05
      }
    },
    "S4_minus_S1": {
      "positive_count": 5,
      "median": 4.40119113125526e-06,
      "IQR": [
        -3.601664835043377e-05,
        5.887118435504154e-06
      ],
      "by_unit": {
        "test/algebra/1332.json|64": 5.887118435504154e-06,
        "test/algebra/1332.json|128": 4.40119113125526e-06,
        "test/algebra/1332.json|256": 4.027318111472765e-05,
        "test/counting_and_probability/119.json|64": 5.230985043508759e-06,
        "test/counting_and_probability/119.json|128": -3.601664835043377e-05,
        "test/counting_and_probability/119.json|256": -4.5380670491776835e-05,
        "test/geometry/477.json|64": 0.00014560458290580942,
        "test/geometry/477.json|128": -0.00020238492873695914,
        "test/geometry/477.json|256": -3.4030997328331355e-05
      }
    },
    "S8_minus_S1": {
      "positive_count": 4,
      "median": -2.3480682326166663e-05,
      "IQR": [
        -3.5877307126244855e-05,
        3.0540151148625194e-05
      ],
      "by_unit": {
        "test/algebra/1332.json|64": 0.00011798338784435254,
        "test/algebra/1332.json|128": -2.878631475540389e-05,
        "test/algebra/1332.json|256": -2.3480682326166663e-05,
        "test/counting_and_probability/119.json|64": -5.8270823404260515e-05,
        "test/counting_and_probability/119.json|128": 3.0540151148625194e-05,
        "test/counting_and_probability/119.json|256": -3.5877307126244855e-05,
        "test/geometry/477.json|64": 0.00018791172348720344,
        "test/geometry/477.json|128": -7.775289875907183e-05,
        "test/geometry/477.json|256": 5.189498730597301e-06
      }
    }
  },
  "headwise": {
    "S2_minus_S1": {
      "positive_count": 5,
      "median": 1.5958072648143495e-05,
      "IQR": [
        -4.6606666583594156e-05,
        3.063011748527389e-05
      ],
      "by_unit": {
        "test/algebra/1332.json|64": 7.188303262956846e-05,
        "test/algebra/1332.json|128": 2.466280574016548e-05,
        "test/algebra/1332.json|256": 4.636103620129174e-05,
        "test/counting_and_probability/119.json|64": 1.5958072648143495e-05,
        "test/counting_and_probability/119.json|128": -7.063214470102175e-05,
        "test/counting_and_probability/119.json|256": -2.6683809943493766e-05,
        "test/geometry/477.json|64": 3.063011748527389e-05,
        "test/geometry/477.json|128": -0.0001239008821239315,
        "test/geometry/477.json|256": -4.6606666583594156e-05
      }
    },
    "S4_minus_S1": {
      "positive_count": 4,
      "median": -5.70994492722972e-06,
      "IQR": [
        -4.411279873579524e-05,
        3.230617988074395e-05
      ],
      "by_unit": {
        "test/algebra/1332.json|64": 7.239289474955328e-05,
        "test/algebra/1332.json|128": -4.411279873579524e-05,
        "test/algebra/1332.json|256": 3.230617988074395e-05,
        "test/counting_and_probability/119.json|64": -4.9947780181358375e-05,
        "test/counting_and_probability/119.json|128": -5.70994492722972e-06,
        "test/counting_and_probability/119.json|256": -1.2384365667678356e-05,
        "test/geometry/477.json|64": 0.0001261718814081332,
        "test/geometry/477.json|128": -0.00016600021893276154,
        "test/geometry/477.json|256": 1.297407431595152e-05
      }
    },
    "S8_minus_S1": {
      "positive_count": 5,
      "median": 5.935032627806642e-06,
      "IQR": [
        -2.273348102011656e-05,
        3.4517924127887294e-05
      ],
      "by_unit": {
        "test/algebra/1332.json|64": 7.877475883655054e-05,
        "test/algebra/1332.json|128": -9.631503859349532e-05,
        "test/algebra/1332.json|256": 3.4517924127887294e-05,
        "test/counting_and_probability/119.json|64": -1.0569783785520901e-05,
        "test/counting_and_probability/119.json|128": 1.0738529853995577e-05,
        "test/counting_and_probability/119.json|256": -2.273348102011656e-05,
        "test/geometry/477.json|64": 8.796270703433551e-05,
        "test/geometry/477.json|128": -0.00015603025172946261,
        "test/geometry/477.json|256": 5.935032627806642e-06
      }
    }
  }
}

## 13. Scope vs head-wise comparison

{
  "median_scope_S8_gap": 4.545219079041539e-05,
  "median_headwise_S8_gap": 1.7923396455662852e-05,
  "median_absolute_S8_gap_reduction": 1.9801621294629617e-05,
  "median_HEADWISE_GAP_RETENTION": 0.03970362254167937,
  "allocation_reduction_votes": 5
}

## 14. Energy-allocation assessment

NOT_SUPPORTED

## 15. Within-head geometry survival assessment

NOT_SUPPORTED

## 16. Negative/corrective findings

The previous 2/9 comprehensive run is preserved only as PARTIAL_DIAGNOSTIC_ONLY and is not used for classification.

## 17. Allowed conclusion

This FAST screen only tests whether the R/C KL-AUC gap grows with same-layer multi-head scope, and whether that growth survives per-head residual norm matching. Pairwise interaction, functional direction, gradients, and method design remain out of scope for this task.

## 18. Primary interpretation

INCONCLUSIVE

## 19. Recommended next task

do not launch follow-up automatically

## 20. Artifact paths

{
  "script": "/data/zypan/experiments/qwen35_gdn_quant/run_int8_same_norm_fast_multihead_composition_screen.py",
  "stage0": "/data/zypan/results/gdn_int8_same_norm_fast_multihead_composition_screen_v1_stage0.json",
  "results": "/data/zypan/results/gdn_int8_same_norm_fast_multihead_composition_screen_v1_results.json",
  "checkpoint": "/data/zypan/results/gdn_int8_same_norm_fast_multihead_composition_screen_v1_checkpoint.json",
  "raw": "/data/zypan/results/gdn_int8_same_norm_fast_multihead_composition_screen_v1_raw.npz",
  "report": "/data/zypan/reports/gdn_int8_same_norm_fast_multihead_composition_screen_v1.md",
  "figures": "/data/zypan/results/gdn_int8_same_norm_fast_multihead_composition_screen_v1_figures",
  "previous_partial_marker": "/data/zypan/results/gdn_int8_same_norm_distributed_composition_functional_alignment_v1_partial_diagnostic_only.json"
}
