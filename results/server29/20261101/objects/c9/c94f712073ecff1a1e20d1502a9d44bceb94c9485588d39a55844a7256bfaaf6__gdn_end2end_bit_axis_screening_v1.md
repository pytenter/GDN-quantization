# GDN End-to-End Bit/Axis Screening V1

- **TASK**: `GDN_END2END_BIT_AXIS_SCREENING_V1`
- **SMOKE_GATE**: `PASS`
- **QUANTIZER_PROTOCOL_GATE**: `PASS`
- **RUN_STATUS**: `COMPLETE`
- **INT8_ROW_CLASSIFICATION**: `LONG_HORIZON_RISK`
- **INT8_COLUMN_CLASSIFICATION**: `LONG_HORIZON_RISK`
- **INT4_ROW_CLASSIFICATION**: `LONG_HORIZON_RISK`
- **INT4_COLUMN_CLASSIFICATION**: `NUMERICALLY_UNSTABLE`
- **RECOMMENDED_RESEARCH_BIT**: `INCONCLUSIVE`
- **RECOMMENDED_RESEARCH_AXIS**: `INCONCLUSIVE`
- **END2END_BIT_AXIS_SCREENING_GATE**: `PASS`
- **CURRENT_STRONGEST_END2END_CONCLUSION**: `Under identical 1x128 recurrent-state granularity, recommended bit=INCONCLUSIVE axis=INCONCLUSIVE; classes={'INT8-row': 'LONG_HORIZON_RISK', 'INT8-column': 'LONG_HORIZON_RISK', 'INT4-row': 'LONG_HORIZON_RISK', 'INT4-column': 'NUMERICALLY_UNSTABLE'}.`

## Table A: MATH-500 fixed subset N=50
```json
{
  "MATH-500|FP_STATE": {
    "N": 50,
    "accuracy": 0.5,
    "completed": 20,
    "correct": 25,
    "mean_tokens": 1569.32,
    "median_tokens": 2048,
    "nonfinite": 0,
    "truncated": 30
  },
  "MATH-500|INT4-column": {
    "N": 50,
    "accuracy": 0.32,
    "completed": 18,
    "correct": 16,
    "mean_tokens": 1601.34,
    "median_tokens": 1625,
    "nonfinite": 17,
    "truncated": 15
  },
  "MATH-500|INT4-row": {
    "N": 50,
    "accuracy": 0.02,
    "completed": 0,
    "correct": 1,
    "mean_tokens": 2048.0,
    "median_tokens": 2048,
    "nonfinite": 0,
    "truncated": 50
  },
  "MATH-500|INT8-column": {
    "N": 50,
    "accuracy": 0.42,
    "completed": 20,
    "correct": 21,
    "mean_tokens": 1633.9,
    "median_tokens": 2048,
    "nonfinite": 0,
    "truncated": 30
  },
  "MATH-500|INT8-row": {
    "N": 50,
    "accuracy": 0.52,
    "completed": 20,
    "correct": 26,
    "mean_tokens": 1640.78,
    "median_tokens": 2048,
    "nonfinite": 0,
    "truncated": 30
  }
}
```

## Table B: AIME-2024 all 30
```json
{
  "AIME-2024|FP_STATE": {
    "N": 30,
    "accuracy": 0.2,
    "completed": 3,
    "correct": 6,
    "mean_tokens": 3915.0333333333333,
    "median_tokens": 4096,
    "nonfinite": 0,
    "truncated": 27
  },
  "AIME-2024|INT4-column": {
    "N": 30,
    "accuracy": 0.03333333333333333,
    "completed": 19,
    "correct": 1,
    "mean_tokens": 1465.0,
    "median_tokens": 1360,
    "nonfinite": 11,
    "truncated": 0
  },
  "AIME-2024|INT4-row": {
    "N": 30,
    "accuracy": 0.0,
    "completed": 8,
    "correct": 0,
    "mean_tokens": 3916.9,
    "median_tokens": 4096,
    "nonfinite": 0,
    "truncated": 22
  },
  "AIME-2024|INT8-column": {
    "N": 30,
    "accuracy": 0.23333333333333334,
    "completed": 4,
    "correct": 7,
    "mean_tokens": 3830.4666666666667,
    "median_tokens": 4096,
    "nonfinite": 0,
    "truncated": 26
  },
  "AIME-2024|INT8-row": {
    "N": 30,
    "accuracy": 0.03333333333333333,
    "completed": 1,
    "correct": 1,
    "mean_tokens": 4046.4666666666667,
    "median_tokens": 4096,
    "nonfinite": 0,
    "truncated": 29
  }
}
```

## Table C: Paired flip analysis
```json
{
  "AIME-2024": {
    "INT4-column": {
      "FP_correct_quant_wrong_count": 6,
      "FP_correct_to_Q_correct": 0,
      "FP_correct_to_Q_wrong": 6,
      "FP_wrong_quant_correct_count": 1,
      "FP_wrong_to_Q_correct": 1,
      "FP_wrong_to_Q_wrong": 23,
      "net_correct_change": -5
    },
    "INT4-row": {
      "FP_correct_quant_wrong_count": 6,
      "FP_correct_to_Q_correct": 0,
      "FP_correct_to_Q_wrong": 6,
      "FP_wrong_quant_correct_count": 0,
      "FP_wrong_to_Q_correct": 0,
      "FP_wrong_to_Q_wrong": 24,
      "net_correct_change": -6
    },
    "INT8-column": {
      "FP_correct_quant_wrong_count": 2,
      "FP_correct_to_Q_correct": 4,
      "FP_correct_to_Q_wrong": 2,
      "FP_wrong_quant_correct_count": 3,
      "FP_wrong_to_Q_correct": 3,
      "FP_wrong_to_Q_wrong": 21,
      "net_correct_change": 1
    },
    "INT8-row": {
      "FP_correct_quant_wrong_count": 5,
      "FP_correct_to_Q_correct": 1,
      "FP_correct_to_Q_wrong": 5,
      "FP_wrong_quant_correct_count": 0,
      "FP_wrong_to_Q_correct": 0,
      "FP_wrong_to_Q_wrong": 24,
      "net_correct_change": -5
    }
  },
  "MATH-500": {
    "INT4-column": {
      "FP_correct_quant_wrong_count": 14,
      "FP_correct_to_Q_correct": 11,
      "FP_correct_to_Q_wrong": 14,
      "FP_wrong_quant_correct_count": 5,
      "FP_wrong_to_Q_correct": 5,
      "FP_wrong_to_Q_wrong": 20,
      "net_correct_change": -9
    },
    "INT4-row": {
      "FP_correct_quant_wrong_count": 25,
      "FP_correct_to_Q_correct": 0,
      "FP_correct_to_Q_wrong": 25,
      "FP_wrong_quant_correct_count": 1,
      "FP_wrong_to_Q_correct": 1,
      "FP_wrong_to_Q_wrong": 24,
      "net_correct_change": -24
    },
    "INT8-column": {
      "FP_correct_quant_wrong_count": 8,
      "FP_correct_to_Q_correct": 17,
      "FP_correct_to_Q_wrong": 8,
      "FP_wrong_quant_correct_count": 4,
      "FP_wrong_to_Q_correct": 4,
      "FP_wrong_to_Q_wrong": 21,
      "net_correct_change": -4
    },
    "INT8-row": {
      "FP_correct_quant_wrong_count": 7,
      "FP_correct_to_Q_correct": 18,
      "FP_correct_to_Q_wrong": 7,
      "FP_wrong_quant_correct_count": 8,
      "FP_wrong_to_Q_correct": 8,
      "FP_wrong_to_Q_wrong": 17,
      "net_correct_change": 1
    }
  }
}
```

## Table D: Output-length analysis
```json
{
  "AIME-2024": {
    "257-512": {
      "FP_STATE": {
        "N": 0,
        "accuracy": 0.0,
        "underpowered": true
      },
      "INT4-column": {
        "N": 0,
        "accuracy": 0.0,
        "underpowered": true
      },
      "INT4-row": {
        "N": 0,
        "accuracy": 0.0,
        "underpowered": true
      },
      "INT8-column": {
        "N": 0,
        "accuracy": 0.0,
        "underpowered": true
      },
      "INT8-row": {
        "N": 0,
        "accuracy": 0.0,
        "underpowered": true
      }
    },
    "513-1024": {
      "FP_STATE": {
        "N": 0,
        "accuracy": 0.0,
        "underpowered": true
      },
      "INT4-column": {
        "N": 0,
        "accuracy": 0.0,
        "underpowered": true
      },
      "INT4-row": {
        "N": 0,
        "accuracy": 0.0,
        "underpowered": true
      },
      "INT8-column": {
        "N": 0,
        "accuracy": 0.0,
        "underpowered": true
      },
      "INT8-row": {
        "N": 0,
        "accuracy": 0.0,
        "underpowered": true
      }
    },
    "<=256": {
      "FP_STATE": {
        "N": 0,
        "accuracy": 0.0,
        "underpowered": true
      },
      "INT4-column": {
        "N": 0,
        "accuracy": 0.0,
        "underpowered": true
      },
      "INT4-row": {
        "N": 0,
        "accuracy": 0.0,
        "underpowered": true
      },
      "INT8-column": {
        "N": 0,
        "accuracy": 0.0,
        "underpowered": true
      },
      "INT8-row": {
        "N": 0,
        "accuracy": 0.0,
        "underpowered": true
      }
    },
    ">1024": {
      "FP_STATE": {
        "N": 30,
        "accuracy": 0.2,
        "underpowered": false
      },
      "INT4-column": {
        "N": 30,
        "accuracy": 0.03333333333333333,
        "underpowered": false
      },
      "INT4-row": {
        "N": 30,
        "accuracy": 0.0,
        "underpowered": false
      },
      "INT8-column": {
        "N": 30,
        "accuracy": 0.23333333333333334,
        "underpowered": false
      },
      "INT8-row": {
        "N": 30,
        "accuracy": 0.03333333333333333,
        "underpowered": false
      }
    }
  },
  "MATH-500": {
    "257-512": {
      "FP_STATE": {
        "N": 2,
        "accuracy": 0.5,
        "underpowered": true
      },
      "INT4-column": {
        "N": 2,
        "accuracy": 1.0,
        "underpowered": true
      },
      "INT4-row": {
        "N": 2,
        "accuracy": 0.0,
        "underpowered": true
      },
      "INT8-column": {
        "N": 2,
        "accuracy": 0.0,
        "underpowered": true
      },
      "INT8-row": {
        "N": 2,
        "accuracy": 1.0,
        "underpowered": true
      }
    },
    "513-1024": {
      "FP_STATE": {
        "N": 13,
        "accuracy": 0.6923076923076923,
        "underpowered": false
      },
      "INT4-column": {
        "N": 13,
        "accuracy": 0.38461538461538464,
        "underpowered": false
      },
      "INT4-row": {
        "N": 13,
        "accuracy": 0.0,
        "underpowered": false
      },
      "INT8-column": {
        "N": 13,
        "accuracy": 0.46153846153846156,
        "underpowered": false
      },
      "INT8-row": {
        "N": 13,
        "accuracy": 0.7692307692307693,
        "underpowered": false
      }
    },
    "<=256": {
      "FP_STATE": {
        "N": 0,
        "accuracy": 0.0,
        "underpowered": true
      },
      "INT4-column": {
        "N": 0,
        "accuracy": 0.0,
        "underpowered": true
      },
      "INT4-row": {
        "N": 0,
        "accuracy": 0.0,
        "underpowered": true
      },
      "INT8-column": {
        "N": 0,
        "accuracy": 0.0,
        "underpowered": true
      },
      "INT8-row": {
        "N": 0,
        "accuracy": 0.0,
        "underpowered": true
      }
    },
    ">1024": {
      "FP_STATE": {
        "N": 35,
        "accuracy": 0.42857142857142855,
        "underpowered": false
      },
      "INT4-column": {
        "N": 35,
        "accuracy": 0.2571428571428571,
        "underpowered": false
      },
      "INT4-row": {
        "N": 35,
        "accuracy": 0.02857142857142857,
        "underpowered": false
      },
      "INT8-column": {
        "N": 35,
        "accuracy": 0.42857142857142855,
        "underpowered": false
      },
      "INT8-row": {
        "N": 35,
        "accuracy": 0.4,
        "underpowered": false
      }
    }
  },
  "quantizer_length_distribution": {
    "AIME-2024|FP_STATE": {
      "N": 30,
      "max": 4096,
      "mean": 3915.0333333333333,
      "min": 2116
    },
    "AIME-2024|INT4-column": {
      "N": 30,
      "max": 3791,
      "mean": 1465.0,
      "min": 584
    },
    "AIME-2024|INT4-row": {
      "N": 30,
      "max": 4096,
      "mean": 3916.9,
      "min": 2675
    },
    "AIME-2024|INT8-column": {
      "N": 30,
      "max": 4096,
      "mean": 3830.4666666666667,
      "min": 1220
    },
    "AIME-2024|INT8-row": {
      "N": 30,
      "max": 4096,
      "mean": 4046.4666666666667,
      "min": 2610
    },
    "MATH-500|FP_STATE": {
      "N": 50,
      "max": 2048,
      "mean": 1569.32,
      "min": 468
    },
    "MATH-500|INT4-column": {
      "N": 50,
      "max": 2048,
      "mean": 1601.34,
      "min": 464
    },
    "MATH-500|INT4-row": {
      "N": 50,
      "max": 2048,
      "mean": 2048.0,
      "min": 2048
    },
    "MATH-500|INT8-column": {
      "N": 50,
      "max": 2048,
      "mean": 1633.9,
      "min": 485
    },
    "MATH-500|INT8-row": {
      "N": 50,
      "max": 2048,
      "mean": 1640.78,
      "min": 302
    }
  }
}
```

## Failure summary
```json
{
  "exceptions": 0,
  "max_token_hit": 259,
  "missing_records": 0,
  "nonfinite": 28,
  "truncated": 259
}
```

## Limitations
- AIME-2024 uses one sample per problem; one problem equals 3.33pp.
- Answer judging is normalized exact/numeric matching, not CAS.
- Mechanism status unchanged: STATE_CHANGE_SWAMPING_SIGNAL_IDENTIFIED; METHOD_DESIGN_READY=NO.

## Next Recommended Experiment
Do not auto-launch; review screening before choosing next experiment.
