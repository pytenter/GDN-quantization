# LING_ERROR_COHERENCE_DIAGNOSIS_V1

```json
{
  "AIME_EXECUTED": "NO",
  "CONFIRMATION_EXECUTED": 0,
  "CONFIRMATORY_STATUS": "NOT_EVALUATED",
  "CROSS_HARDWARE_FP_BITWISE": "FAIL_PRESERVED",
  "DECOMPOSITION_CLOSURE_GATE": "PASS",
  "DISCOVERY_COMPLETE": "64/64",
  "ENERGY_DECOMPOSITION_IDENTITY_IS_TAUTOLOGICAL": "YES",
  "ERROR_COHERENCE_DISCOVERY_LABEL": "DESCRIPTIVE_SIGNAL_NOT_PRESENT",
  "EXPERIMENT": "LING_ERROR_COHERENCE_DIAGNOSIS_V1",
  "FINAL_ACTION": "STOP_AFTER_DISCOVERY",
  "FP_REFERENCES_MIXED_ACROSS_HARDWARE": "NO",
  "L9_TRAINING_EXECUTED": "NO",
  "NO_NEW_MODEL_FORWARD": "NO; Route B teacher-forced FP capture",
  "PARENT_GLOBAL_GATE": "FAIL_PRESERVED",
  "PREMEASUREMENT_COMMIT_SHA": "d4906196ab2921035f4c175804fe27af15eb1314",
  "PREREGISTRATION_GATE": "PASS",
  "PRIMARY_NEXT_STEP_CATEGORY": "CONSIDER_READOUT_SENSITIVITY_DIAGNOSIS",
  "PROVENANCE_GATE": "PASS",
  "RECOMMENDATION_ONLY": true,
  "RUNTIME_IDENTITY_GATE": "PASS",
  "SCIENTIFIC_QUESTION": "IS_ALIGNMENT_STRUCTURED_AND_RANK_DECISIVE",
  "closure_summary": {
    "global_max": {
      "max_abs_residual": 1.1102230246251565e-16,
      "max_energy_abs_residual": 7.63429852668196e-09,
      "max_energy_scaled_residual": 4.496333930483613e-16,
      "relative_L2_residual": 9.331083041648673e-18
    },
    "per_document": {
      "pg19-test-26493": {
        "max_abs_residual": 1.1102230246251565e-16,
        "max_energy_abs_residual": 1.007549599307822e-09,
        "max_energy_scaled_residual": 4.496333930483613e-16,
        "relative_L2_residual": 9.331083041648673e-18
      },
      "pg19-test-28988": {
        "max_abs_residual": 1.1102230246251565e-16,
        "max_energy_abs_residual": 4.762114258483052e-09,
        "max_energy_scaled_residual": 4.006168586217735e-16,
        "relative_L2_residual": 7.94585297057409e-18
      },
      "pg19-test-37702": {
        "max_abs_residual": 1.1102230246251565e-16,
        "max_energy_abs_residual": 3.342019994079237e-10,
        "max_energy_scaled_residual": 4.075950139256283e-16,
        "relative_L2_residual": 9.282901618399175e-18
      },
      "pg19-test-5396": {
        "max_abs_residual": 1.1102230246251565e-16,
        "max_energy_abs_residual": 7.63429852668196e-09,
        "max_energy_scaled_residual": 4.014848006119953e-16,
        "relative_L2_residual": 8.451639082401059e-18
      }
    }
  },
  "injection_scope": "actual Q-minus-native-F, includes numerical basis/write rounding; pure QDQ separately audited",
  "median_horizon_diagnostics": {
    "2048": {
      "J_INJ_rho": 0.8955882352941177,
      "J_NOALIGN_concordance": 1.0,
      "J_NOALIGN_rho": 1.0,
      "J_PROP_rho": 0.8573529411764707,
      "alignment_decisive_pair_rate": 0.0,
      "median_NET_COHERENCE_RATIO": 0.001687820239378096
    },
    "32768": {
      "J_INJ_rho": 0.5602941176470588,
      "J_NOALIGN_concordance": 1.0,
      "J_NOALIGN_rho": 1.0,
      "J_PROP_rho": 0.7705882352941177,
      "alignment_decisive_pair_rate": 0.0,
      "median_NET_COHERENCE_RATIO": 0.0005062027470742862
    },
    "8192": {
      "J_INJ_rho": 0.8794117647058823,
      "J_NOALIGN_concordance": 1.0,
      "J_NOALIGN_rho": 1.0,
      "J_PROP_rho": 0.6485294117647058,
      "alignment_decisive_pair_rate": 0.0,
      "median_NET_COHERENCE_RATIO": 0.0009196322563917818
    }
  },
  "per_document_horizon": [
    {
      "J_INJ_rho": 0.8941176470588236,
      "J_NOALIGN_concordance": 1.0,
      "J_NOALIGN_rho": 1.0,
      "J_PROP_rho": 0.9029411764705882,
      "alignment_decisive_pair_count": 0,
      "alignment_decisive_pair_rate": 0.0,
      "document_id": "pg19-test-37702",
      "horizon": 2048,
      "median_NET_COHERENCE_RATIO": 0.0017985878669982292,
      "median_abs_NET_COHERENCE_RATIO": 0.0017985878669982292,
      "total_non_tied_pairs": 120
    },
    {
      "J_INJ_rho": 0.9,
      "J_NOALIGN_concordance": 1.0,
      "J_NOALIGN_rho": 1.0,
      "J_PROP_rho": 0.7088235294117647,
      "alignment_decisive_pair_count": 0,
      "alignment_decisive_pair_rate": 0.0,
      "document_id": "pg19-test-37702",
      "horizon": 8192,
      "median_NET_COHERENCE_RATIO": 0.0008914148736898205,
      "median_abs_NET_COHERENCE_RATIO": 0.0008914148736898205,
      "total_non_tied_pairs": 120
    },
    {
      "J_INJ_rho": 0.5176470588235295,
      "J_NOALIGN_concordance": 1.0,
      "J_NOALIGN_rho": 1.0,
      "J_PROP_rho": 0.7970588235294118,
      "alignment_decisive_pair_count": 0,
      "alignment_decisive_pair_rate": 0.0,
      "document_id": "pg19-test-37702",
      "horizon": 32768,
      "median_NET_COHERENCE_RATIO": 0.00046295166232905344,
      "median_abs_NET_COHERENCE_RATIO": 0.00046295166232905344,
      "total_non_tied_pairs": 120
    },
    {
      "J_INJ_rho": 0.85,
      "J_NOALIGN_concordance": 1.0,
      "J_NOALIGN_rho": 1.0,
      "J_PROP_rho": 0.9264705882352942,
      "alignment_decisive_pair_count": 0,
      "alignment_decisive_pair_rate": 0.0,
      "document_id": "pg19-test-5396",
      "horizon": 2048,
      "median_NET_COHERENCE_RATIO": 0.001692327509074364,
      "median_abs_NET_COHERENCE_RATIO": 0.001692327509074364,
      "total_non_tied_pairs": 120
    },
    {
      "J_INJ_rho": 0.9,
      "J_NOALIGN_concordance": 1.0,
      "J_NOALIGN_rho": 1.0,
      "J_PROP_rho": 0.6941176470588235,
      "alignment_decisive_pair_count": 0,
      "alignment_decisive_pair_rate": 0.0,
      "document_id": "pg19-test-5396",
      "horizon": 8192,
      "median_NET_COHERENCE_RATIO": 0.0009513770438421915,
      "median_abs_NET_COHERENCE_RATIO": 0.0009513770438421915,
      "total_non_tied_pairs": 120
    },
    {
      "J_INJ_rho": 0.6970588235294117,
      "J_NOALIGN_concordance": 1.0,
      "J_NOALIGN_rho": 1.0,
      "J_PROP_rho": 0.9088235294117647,
      "alignment_decisive_pair_count": 0,
      "alignment_decisive_pair_rate": 0.0,
      "document_id": "pg19-test-5396",
      "horizon": 32768,
      "median_NET_COHERENCE_RATIO": 0.000549453831819519,
      "median_abs_NET_COHERENCE_RATIO": 0.000549453831819519,
      "total_non_tied_pairs": 120
    },
    {
      "J_INJ_rho": 0.8970588235294118,
      "J_NOALIGN_concordance": 1.0,
      "J_NOALIGN_rho": 1.0,
      "J_PROP_rho": 0.8117647058823529,
      "alignment_decisive_pair_count": 0,
      "alignment_decisive_pair_rate": 0.0,
      "document_id": "pg19-test-28988",
      "horizon": 2048,
      "median_NET_COHERENCE_RATIO": 0.0016641205487684112,
      "median_abs_NET_COHERENCE_RATIO": 0.0016641205487684112,
      "total_non_tied_pairs": 120
    },
    {
      "J_INJ_rho": 0.8588235294117647,
      "J_NOALIGN_concordance": 1.0,
      "J_NOALIGN_rho": 1.0,
      "J_PROP_rho": 0.5911764705882353,
      "alignment_decisive_pair_count": 0,
      "alignment_decisive_pair_rate": 0.0,
      "document_id": "pg19-test-28988",
      "horizon": 8192,
      "median_NET_COHERENCE_RATIO": 0.0009478496390937432,
      "median_abs_NET_COHERENCE_RATIO": 0.0009478496390937432,
      "total_non_tied_pairs": 120
    },
    {
      "J_INJ_rho": 0.5823529411764706,
      "J_NOALIGN_concordance": 1.0,
      "J_NOALIGN_rho": 1.0,
      "J_PROP_rho": 0.7441176470588236,
      "alignment_decisive_pair_count": 0,
      "alignment_decisive_pair_rate": 0.0,
      "document_id": "pg19-test-28988",
      "horizon": 32768,
      "median_NET_COHERENCE_RATIO": 0.00061077684339658,
      "median_abs_NET_COHERENCE_RATIO": 0.00061077684339658,
      "total_non_tied_pairs": 120
    },
    {
      "J_INJ_rho": 0.9470588235294117,
      "J_NOALIGN_concordance": 1.0,
      "J_NOALIGN_rho": 1.0,
      "J_PROP_rho": 0.7470588235294118,
      "alignment_decisive_pair_count": 0,
      "alignment_decisive_pair_rate": 0.0,
      "document_id": "pg19-test-26493",
      "horizon": 2048,
      "median_NET_COHERENCE_RATIO": 0.0016833129696818276,
      "median_abs_NET_COHERENCE_RATIO": 0.0016833129696818276,
      "total_non_tied_pairs": 120
    },
    {
      "J_INJ_rho": 0.8352941176470589,
      "J_NOALIGN_concordance": 1.0,
      "J_NOALIGN_rho": 1.0,
      "J_PROP_rho": 0.6029411764705882,
      "alignment_decisive_pair_count": 0,
      "alignment_decisive_pair_rate": 0.0,
      "document_id": "pg19-test-26493",
      "horizon": 8192,
      "median_NET_COHERENCE_RATIO": 0.0008399790326998194,
      "median_abs_NET_COHERENCE_RATIO": 0.0008399790326998194,
      "total_non_tied_pairs": 120
    },
    {
      "J_INJ_rho": 0.538235294117647,
      "J_NOALIGN_concordance": 1.0,
      "J_NOALIGN_rho": 1.0,
      "J_PROP_rho": 0.5352941176470588,
      "alignment_decisive_pair_count": 0,
      "alignment_decisive_pair_rate": 0.0,
      "document_id": "pg19-test-26493",
      "horizon": 32768,
      "median_NET_COHERENCE_RATIO": 0.0004598026107900978,
      "median_abs_NET_COHERENCE_RATIO": 0.0004598026107900978,
      "total_non_tied_pairs": 120
    }
  ],
  "signed_magnitude_diagnostics": {
    "2048": {
      "NEG_ALIGNMENT": -0.7767271414889017,
      "POS_ALIGNMENT": 3.4587354757055317,
      "SUM_BASIS_DISCREPANCY": 0.0009884671659704603,
      "SUM_PURE_QDQ": 0.42121930769880067,
      "negative_alignment_fraction": 0.21796747967479674,
      "positive_alignment_fraction": 0.7820325203252032
    },
    "32768": {
      "NEG_ALIGNMENT": -54.04328549427138,
      "POS_ALIGNMENT": 272.69348602405,
      "SUM_BASIS_DISCREPANCY": 0.01811442308648397,
      "SUM_PURE_QDQ": 9.249867503498098,
      "negative_alignment_fraction": 0.20192243747921806,
      "positive_alignment_fraction": 0.7980775625207819
    },
    "8192": {
      "NEG_ALIGNMENT": -8.965169406593514,
      "POS_ALIGNMENT": 39.545617834112036,
      "SUM_BASIS_DISCREPANCY": 0.005334690235959395,
      "SUM_PURE_QDQ": 2.4334999012910328,
      "negative_alignment_fraction": 0.21434881666434186,
      "positive_alignment_fraction": 0.7856511833356582
    }
  }
}
```

Discovery-only. No reasoning, AIME or training conclusion. Algebraic closure is tautological, not predictive evidence.

## Segment, amplification and cancellation details

See horizon_segment_metrics.csv for both full-segment and primary-support intersection summaries; per_rotation_metrics.csv reports positive/negative fractions and POS/NEG_ALIGNMENT magnitudes.
