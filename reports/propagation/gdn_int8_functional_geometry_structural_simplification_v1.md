# GDN_INT8_FUNCTIONAL_GEOMETRY_STRUCTURAL_SIMPLIFICATION_V1

## Question
Can stable but high-dimensional GDN M5 functional geometry be simplified to isotropic, diagonal, or natural head-block structures without losing decision-relevant orientation signal?

## Protocol
{
  "BLOCK_SEMANTICS_GATE": "PASS",
  "GEOMETRY_APPROXIMATION_GATE": "PASS",
  "HEAD": "2842db12c873a98c23b23d4aad32dc552edd25f2",
  "M5_QUADRATIC_IDENTITY_GATE": "PASS",
  "M5_QUADRATIC_IDENTITY_MAX_RELERR": 1.0590213118295197e-07,
  "M5_QUADRATIC_IDENTITY_MEDIAN_RELERR": 2.753419336616738e-08,
  "NO_FUTURE_INFORMATION_LEAKAGE_GATE": "PASS",
  "PROTOCOL_GATE": "PASS",
  "TASK": "GDN_INT8_FUNCTIONAL_GEOMETRY_STRUCTURAL_SIMPLIFICATION_V1",
  "branch": "research-sync-2026-09-02",
  "diagonal_definition": "diag(diag(G)); no off-diagonal entries retained",
  "expected_geometry_tensor_count": 216,
  "frozen_m5_formula": "M_G^2 = q^T E G_approx E^T q, G_t = A_t^T A_t, A_t = W_O D_g D_w J_RMS",
  "future_KL_use": "evaluation only; not used to fit iso/diag/block/static means",
  "geometry_tensor_count": 216,
  "geometry_tensor_source": "/data/zypan/runs/gdn_int8_functional_orientation_low_rank_and_stability_audit_v1/geometry_tensors",
  "git_status_start": "?? experiments/propagation/run_int8_functional_geometry_structural_simplification.py\n?? reports/propagation/gdn_int8_functional_geometry_structural_simplification_v1_stage0_protocol_audit.md\n?? results/propagation/gdn_int8_functional_geometry_structural_simplification_v1_stage0_protocol_audit.json",
  "head_block_definition": "blockdiag over true GDN [head,value] layout, H=32, V=128 contiguous value blocks",
  "instrumentation": "post-hoc state/readout reconstruction only; no forward hook or model mutation",
  "isotropic_definition": "alpha I, alpha=trace(G)/d",
  "matched_pairs_source": "/data/zypan/GDN-quantization/results/propagation/gdn_int8_functional_risk_beyond_reconstruction_controlled_v1_matched_pair_results.json",
  "source_cases": "/data/zypan/GDN-quantization/results/propagation/gdn_int8_functional_proxy_natural_config_generalization_v1_formal_raw_results.jsonl",
  "tensor_shape_checks_sample": [
    {
      "G_shape": [
        4096,
        4096
      ],
      "e_dim": 4096,
      "layer": 0,
      "q_shape": [
        1,
        32,
        128
      ],
      "state_shape": [
        1,
        32,
        128,
        128
      ],
      "unit_id": "test/algebra/1332.json|64"
    },
    {
      "G_shape": [
        4096,
        4096
      ],
      "e_dim": 4096,
      "layer": 1,
      "q_shape": [
        1,
        32,
        128
      ],
      "state_shape": [
        1,
        32,
        128,
        128
      ],
      "unit_id": "test/algebra/1332.json|64"
    },
    {
      "G_shape": [
        4096,
        4096
      ],
      "e_dim": 4096,
      "layer": 2,
      "q_shape": [
        1,
        32,
        128
      ],
      "state_shape": [
        1,
        32,
        128,
        128
      ],
      "unit_id": "test/algebra/1332.json|64"
    },
    {
      "G_shape": [
        4096,
        4096
      ],
      "e_dim": 4096,
      "layer": 4,
      "q_shape": [
        1,
        32,
        128
      ],
      "state_shape": [
        1,
        32,
        128,
        128
      ],
      "unit_id": "test/algebra/1332.json|64"
    },
    {
      "G_shape": [
        4096,
        4096
      ],
      "e_dim": 4096,
      "layer": 5,
      "q_shape": [
        1,
        32,
        128
      ],
      "state_shape": [
        1,
        32,
        128,
        128
      ],
      "unit_id": "test/algebra/1332.json|64"
    },
    {
      "G_shape": [
        4096,
        4096
      ],
      "e_dim": 4096,
      "layer": 6,
      "q_shape": [
        1,
        32,
        128
      ],
      "state_shape": [
        1,
        32,
        128,
        128
      ],
      "unit_id": "test/algebra/1332.json|64"
    },
    {
      "G_shape": [
        4096,
        4096
      ],
      "e_dim": 4096,
      "layer": 8,
      "q_shape": [
        1,
        32,
        128
      ],
      "state_shape": [
        1,
        32,
        128,
        128
      ],
      "unit_id": "test/algebra/1332.json|64"
    },
    {
      "G_shape": [
        4096,
        4096
      ],
      "e_dim": 4096,
      "layer": 9,
      "q_shape": [
        1,
        32,
        128
      ],
      "state_shape": [
        1,
        32,
        128,
        128
      ],
      "unit_id": "test/algebra/1332.json|64"
    },
    {
      "G_shape": [
        4096,
        4096
      ],
      "e_dim": 4096,
      "layer": 10,
      "q_shape": [
        1,
        32,
        128
      ],
      "state_shape": [
        1,
        32,
        128,
        128
      ],
      "unit_id": "test/algebra/1332.json|64"
    },
    {
      "G_shape": [
        4096,
        4096
      ],
      "e_dim": 4096,
      "layer": 12,
      "q_shape": [
        1,
        32,
        128
      ],
      "state_shape": [
        1,
        32,
        128,
        128
      ],
      "unit_id": "test/algebra/1332.json|64"
    }
  ],
  "timestamp": "2026-09-06 11:01:48 +0800"
}

## Structure
{
  "CROSS_HEAD_ENERGY_FRACTION": 0.2164935839834684,
  "DIAGONAL_ENERGY_FRACTION": 0.7024531421022508,
  "HEAD_BLOCK_ENERGY_FRACTION": 0.7835064160165316,
  "OFFDIAGONAL_ENERGY_FRACTION": 0.2975468578977492,
  "diag_fro_fraction_mean": 0.7219779156361978,
  "head_block_fro_fraction_mean": 0.8072112844816828,
  "n_layers": 24,
  "n_units": 9
}

## Dynamic Approximation To Full M5
{
  "block": {
    "C-C_pairwise_vs_full": 1.0,
    "R-C_pairwise_vs_full": 0.9861111111111112,
    "R-R_pairwise_vs_full": 1.0,
    "kendall_vs_full": 0.9632237871674492,
    "log_pearson_vs_full": 0.9940841305111264,
    "mean_relative_error": 0.03467203576598977,
    "median_relative_error": 0.016162861679745226,
    "pairwise_vs_full": 0.9920634920634921,
    "spearman_vs_full": 0.9966557334876841
  },
  "block_vs_future": {
    "C-C_pairwise": 0.48148148148148145,
    "R-C_pairwise": 0.8611111111111112,
    "R-R_pairwise": 0.9074074074074074,
    "kendall": 0.52660406885759,
    "log_pearson": 0.7456052877004087,
    "pairwise": 0.7896825396825397,
    "spearman": 0.7158338156794649
  },
  "diag": {
    "C-C_pairwise_vs_full": 1.0,
    "R-C_pairwise_vs_full": 0.9583333333333334,
    "R-R_pairwise_vs_full": 1.0,
    "kendall_vs_full": 0.9405320813771518,
    "log_pearson_vs_full": 0.9920696898603956,
    "mean_relative_error": 0.053850652021050494,
    "median_relative_error": 0.044753998235236764,
    "pairwise_vs_full": 0.9761904761904762,
    "spearman_vs_full": 0.9936008746543186
  },
  "diag_vs_future": {
    "C-C_pairwise": 0.48148148148148145,
    "R-C_pairwise": 0.8333333333333334,
    "R-R_pairwise": 0.9074074074074074,
    "kendall": 0.514866979655712,
    "log_pearson": 0.736515628201491,
    "pairwise": 0.7738095238095238,
    "spearman": 0.7034214418933693
  },
  "full_vs_future": {
    "C-C_pairwise": 0.48148148148148145,
    "R-C_pairwise": 0.875,
    "R-R_pairwise": 0.9074074074074074,
    "kendall": 0.5305164319248826,
    "log_pearson": 0.7472720645958529,
    "pairwise": 0.7976190476190477,
    "spearman": 0.7158338156794649
  },
  "iso": {
    "C-C_pairwise_vs_full": 0.9444444444444444,
    "R-C_pairwise_vs_full": 0.8819444444444444,
    "R-R_pairwise_vs_full": 1.0,
    "kendall_vs_full": 0.687793427230047,
    "log_pearson_vs_full": 0.8740504852956777,
    "mean_relative_error": 11.938913755472258,
    "median_relative_error": 11.980396883820397,
    "pairwise_vs_full": 0.9206349206349206,
    "spearman_vs_full": 0.859540806482732
  },
  "iso_vs_future": {
    "C-C_pairwise": 0.46296296296296297,
    "R-C_pairwise": 0.9791666666666666,
    "R-R_pairwise": 0.9074074074074074,
    "kendall": 0.5156494522691706,
    "log_pearson": 0.7202419326799097,
    "pairwise": 0.8531746031746031,
    "spearman": 0.7130683645250498
  }
}

## Matched R-C Decision Retention
{
  "M5_block": {
    "concordance": 1.0,
    "correct": 14,
    "total": 14
  },
  "M5_diag": {
    "concordance": 1.0,
    "correct": 14,
    "total": 14
  },
  "M5_full_recomputed": {
    "concordance": 1.0,
    "correct": 14,
    "total": 14
  },
  "M5_iso": {
    "concordance": 1.0,
    "correct": 14,
    "total": 14
  }
}

## Static Held-Out Calibration
{
  "dynamic_block_reference": {
    "C-C_pairwise": 0.48148148148148145,
    "R-C_pairwise": 0.8611111111111112,
    "R-R_pairwise": 0.9074074074074074,
    "kendall": 0.52660406885759,
    "log_pearson": 0.7456052877004087,
    "pairwise": 0.7896825396825397,
    "spearman": 0.7158338156794649
  },
  "dynamic_diag_reference": {
    "C-C_pairwise": 0.48148148148148145,
    "R-C_pairwise": 0.8333333333333334,
    "R-R_pairwise": 0.9074074074074074,
    "kendall": 0.514866979655712,
    "log_pearson": 0.736515628201491,
    "pairwise": 0.7738095238095238,
    "spearman": 0.7034214418933693
  },
  "dynamic_full_reference": {
    "C-C_pairwise": 0.48148148148148145,
    "R-C_pairwise": 0.875,
    "R-R_pairwise": 0.9074074074074074,
    "kendall": 0.5305164319248826,
    "log_pearson": 0.7472720645958526,
    "pairwise": 0.7976190476190477,
    "spearman": 0.7158338156794649
  },
  "matched_rc_14": {
    "M5_block_dynamic": {
      "concordance": 1.0,
      "correct": 14,
      "total": 14
    },
    "M5_diag_dynamic": {
      "concordance": 1.0,
      "correct": 14,
      "total": 14
    },
    "M5_full_dynamic": {
      "concordance": 1.0,
      "correct": 14,
      "total": 14
    },
    "M5_static_block": {
      "concordance": 1.0,
      "correct": 14,
      "total": 14
    },
    "M5_static_diag": {
      "concordance": 1.0,
      "correct": 14,
      "total": 14
    },
    "M5_static_iso": {
      "concordance": 1.0,
      "correct": 14,
      "total": 14
    }
  },
  "protocol": "leave-one-unit-out; arithmetic mean of calibration-unit G diagonals/head-blocks/alpha; no future_KL fitting",
  "static_block": {
    "C-C_pairwise": 0.46296296296296297,
    "R-C_pairwise": 0.8958333333333334,
    "R-R_pairwise": 0.9074074074074074,
    "kendall": 0.32472613458528954,
    "log_pearson": 0.5461690914313433,
    "pairwise": 0.8055555555555556,
    "spearman": 0.47466074988745255
  },
  "static_diag": {
    "C-C_pairwise": 0.46296296296296297,
    "R-C_pairwise": 0.875,
    "R-R_pairwise": 0.9074074074074074,
    "kendall": 0.3137715179968701,
    "log_pearson": 0.5395936600608651,
    "pairwise": 0.7936507936507936,
    "spearman": 0.46247347096276287
  },
  "static_iso": {
    "C-C_pairwise": 0.46296296296296297,
    "R-C_pairwise": 0.9791666666666666,
    "R-R_pairwise": 0.9074074074074074,
    "kendall": 0.42644757433489827,
    "log_pearson": 0.6256691676984517,
    "pairwise": 0.8531746031746031,
    "spearman": 0.6095247282783459
  }
}

## Final Classification
`FUNCTIONAL_GEOMETRY_COORDINATE_SEPARABLE_SUPPORTED`

## Next Task
design a minimal static diagonal functional-risk quantizer candidate
