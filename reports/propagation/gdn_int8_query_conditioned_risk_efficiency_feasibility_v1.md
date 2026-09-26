# GDN_INT8_QUERY_CONDITIONED_RISK_EFFICIENCY_FEASIBILITY_V1

## Task Definition
GDN_INT8_QUERY_CONDITIONED_RISK_EFFICIENCY_FEASIBILITY_V1

## Previous State
{
  "pullback_classification": "FUNCTIONAL_PULLBACK_VALID_BUT_STATIC_STATE_METRIC_NOT_SUPPORTED",
  "pullback_commit": "52761e4"
}

## Why Dynamic-q Is Not Automatically Free
q_t is available in the model, but using it for quantizer scale/objective construction still requires arithmetic over state errors. The red line is zero extra full-state passes.

## Reference Functional-Risk Formula
R_dyn_dyn = sum_j w_t,j (q_t^T E[:,j])^2.

## Candidate Approximations
# Candidate Definitions

- `M0`: raw Frobenius reconstruction.
- `query_only`: `||E^T q_t||^2`.
- `dynq_staticw`: `sum_j w_bar_j (q_t^T E[:,j])^2`.
- `q2_staticw`: `sum_ij q_t,i^2 w_bar_j E_ij^2`; drops cross-row terms.
- `groupq_staticw`: same as q2/static-w with q2 averaged over key groups `{4,8,16,32}`.
- `headq_staticw`: head-level query energy times static value weights.
- `static_column`: `sum_j w_bar_j ||E[:,j]||^2`.

No candidate computes online full G, dense H, Jacobian, or extra future signal.

## Scientific Fidelity
{
  "M0": {
    "vs_dynamic_diag": {
      "C-C_pairwise": 1.0,
      "R-C_pairwise": 0.875,
      "R-R_pairwise": 1.0,
      "kendall": 0.6549295774647887,
      "mean_relative_error": 1.4366860398996213,
      "median_relative_error": 1.3648121267172941,
      "pairwise": 0.9285714285714286,
      "spearman": 0.8407614637597273
    },
    "vs_full": {
      "C-C_pairwise": 1.0,
      "R-C_pairwise": 0.875,
      "R-R_pairwise": 1.0,
      "kendall": 0.662754303599374,
      "mean_relative_error": 1.430697980566813,
      "median_relative_error": 1.378454914428505,
      "pairwise": 0.9285714285714286,
      "spearman": 0.8383497331018072
    },
    "vs_future_KL": {
      "C-C_pairwise": 0.48148148148148145,
      "R-C_pairwise": 0.7638888888888888,
      "R-R_pairwise": 0.9074074074074074,
      "kendall": 0.2809076682316119,
      "pairwise": 0.7341269841269841,
      "spearman": 0.42083092160267543
    }
  },
  "dynq_staticw": {
    "vs_dynamic_diag": {
      "C-C_pairwise": 0.9814814814814815,
      "R-C_pairwise": 0.9444444444444444,
      "R-R_pairwise": 1.0,
      "kendall": 0.636150234741784,
      "mean_relative_error": 0.16187838382530126,
      "median_relative_error": 0.10399933342215147,
      "pairwise": 0.9642857142857143,
      "spearman": 0.8109846292366069
    },
    "vs_full": {
      "C-C_pairwise": 0.9814814814814815,
      "R-C_pairwise": 0.9583333333333334,
      "R-R_pairwise": 1.0,
      "kendall": 0.661189358372457,
      "mean_relative_error": 0.15607866274199816,
      "median_relative_error": 0.08492093902344472,
      "pairwise": 0.9722222222222222,
      "spearman": 0.82477972859991
    },
    "vs_future_KL": {
      "C-C_pairwise": 0.46296296296296297,
      "R-C_pairwise": 0.875,
      "R-R_pairwise": 0.9074074074074074,
      "kendall": 0.3137715179968701,
      "pairwise": 0.7936507936507936,
      "spearman": 0.46247347096276287
    }
  },
  "group16_q_staticw": {
    "vs_dynamic_diag": {
      "C-C_pairwise": 1.0,
      "R-C_pairwise": 0.8472222222222222,
      "R-R_pairwise": 1.0,
      "kendall": 0.568075117370892,
      "mean_relative_error": 1.3971098554496217,
      "median_relative_error": 1.3419554681612405,
      "pairwise": 0.9126984126984127,
      "spearman": 0.7609171007781851
    },
    "vs_full": {
      "C-C_pairwise": 1.0,
      "R-C_pairwise": 0.8888888888888888,
      "R-R_pairwise": 1.0,
      "kendall": 0.5915492957746479,
      "mean_relative_error": 1.3712931648205524,
      "median_relative_error": 1.3144539499763197,
      "pairwise": 0.9365079365079365,
      "spearman": 0.7840697150942183
    },
    "vs_future_KL": {
      "C-C_pairwise": 0.48148148148148145,
      "R-C_pairwise": 0.9722222222222222,
      "R-R_pairwise": 0.9074074074074074,
      "kendall": 0.3161189358372457,
      "pairwise": 0.8531746031746031,
      "spearman": 0.4765258215962441
    }
  },
  "group32_q_staticw": {
    "vs_dynamic_diag": {
      "C-C_pairwise": 1.0,
      "R-C_pairwise": 0.8472222222222222,
      "R-R_pairwise": 1.0,
      "kendall": 0.5876369327073553,
      "mean_relative_error": 1.4462507193381482,
      "median_relative_error": 1.5231503023465534,
      "pairwise": 0.9126984126984127,
      "spearman": 0.7751945462730723
    },
    "vs_full": {
      "C-C_pairwise": 1.0,
      "R-C_pairwise": 0.8888888888888888,
      "R-R_pairwise": 1.0,
      "kendall": 0.6111111111111112,
      "mean_relative_error": 1.4158998513130276,
      "median_relative_error": 1.355348981205732,
      "pairwise": 0.9365079365079365,
      "spearman": 0.7980899093189273
    },
    "vs_future_KL": {
      "C-C_pairwise": 0.48148148148148145,
      "R-C_pairwise": 0.9722222222222222,
      "R-R_pairwise": 0.9074074074074074,
      "kendall": 0.33568075117370894,
      "pairwise": 0.8531746031746031,
      "spearman": 0.49601260531223873
    }
  },
  "group4_q_staticw": {
    "vs_dynamic_diag": {
      "C-C_pairwise": 1.0,
      "R-C_pairwise": 0.8888888888888888,
      "R-R_pairwise": 1.0,
      "kendall": 0.57981220657277,
      "mean_relative_error": 0.9750030845780273,
      "median_relative_error": 0.8225424339049785,
      "pairwise": 0.9365079365079365,
      "spearman": 0.7728149720239244
    },
    "vs_full": {
      "C-C_pairwise": 1.0,
      "R-C_pairwise": 0.9166666666666666,
      "R-R_pairwise": 1.0,
      "kendall": 0.6048513302034428,
      "mean_relative_error": 0.9648621126672554,
      "median_relative_error": 0.8418068177156592,
      "pairwise": 0.9523809523809523,
      "spearman": 0.791948035243424
    },
    "vs_future_KL": {
      "C-C_pairwise": 0.48148148148148145,
      "R-C_pairwise": 0.9305555555555556,
      "R-R_pairwise": 0.9074074074074074,
      "kendall": 0.2793427230046948,
      "pairwise": 0.8293650793650794,
      "spearman": 0.4296739340150492
    }
  },
  "group8_q_staticw": {
    "vs_dynamic_diag": {
      "C-C_pairwise": 1.0,
      "R-C_pairwise": 0.875,
      "R-R_pairwise": 1.0,
      "kendall": 0.564945226917058,
      "mean_relative_error": 1.1345187253359925,
      "median_relative_error": 0.9344931842678306,
      "pairwise": 0.9285714285714286,
      "spearman": 0.7549038523377709
    },
    "vs_full": {
      "C-C_pairwise": 1.0,
      "R-C_pairwise": 0.9166666666666666,
      "R-R_pairwise": 1.0,
      "kendall": 0.594679186228482,
      "mean_relative_error": 1.1204919436094616,
      "median_relative_error": 0.998151481972542,
      "pairwise": 0.9523809523809523,
      "spearman": 0.778635282011705
    },
    "vs_future_KL": {
      "C-C_pairwise": 0.48148148148148145,
      "R-C_pairwise": 0.9444444444444444,
      "R-R_pairwise": 0.9074074074074074,
      "kendall": 0.28169014084507044,
      "pairwise": 0.8373015873015873,
      "spearman": 0.43436876969580035
    }
  },
  "headq_staticw": {
    "vs_dynamic_diag": {
      "C-C_pairwise": 1.0,
      "R-C_pairwise": 0.8333333333333334,
      "R-R_pairwise": 1.0,
      "kendall": 0.5884194053208138,
      "mean_relative_error": 1.5528031464748746,
      "median_relative_error": 1.5961626009907612,
      "pairwise": 0.9047619047619048,
      "spearman": 0.7845520612258023
    },
    "vs_full": {
      "C-C_pairwise": 1.0,
      "R-C_pairwise": 0.875,
      "R-R_pairwise": 1.0,
      "kendall": 0.6150234741784038,
      "mean_relative_error": 1.5187635753255813,
      "median_relative_error": 1.4333683758690885,
      "pairwise": 0.9285714285714286,
      "spearman": 0.8102771882436169
    },
    "vs_future_KL": {
      "C-C_pairwise": 0.48148148148148145,
      "R-C_pairwise": 0.9722222222222222,
      "R-R_pairwise": 0.9074074074074074,
      "kendall": 0.3380281690140845,
      "pairwise": 0.8531746031746031,
      "spearman": 0.5061418740755033
    }
  },
  "q2_staticw": {
    "vs_dynamic_diag": {
      "C-C_pairwise": 1.0,
      "R-C_pairwise": 0.9652777777777778,
      "R-R_pairwise": 1.0,
      "kendall": 0.6338028169014085,
      "mean_relative_error": 0.15272552691836622,
      "median_relative_error": 0.08833085939950146,
      "pairwise": 0.9801587301587301,
      "spearman": 0.7870281047012669
    },
    "vs_full": {
      "C-C_pairwise": 1.0,
      "R-C_pairwise": 0.9236111111111112,
      "R-R_pairwise": 1.0,
      "kendall": 0.6416275430359938,
      "mean_relative_error": 0.16388392042065256,
      "median_relative_error": 0.09575093485929353,
      "pairwise": 0.9563492063492064,
      "spearman": 0.7889896456363753
    },
    "vs_future_KL": {
      "C-C_pairwise": 0.48148148148148145,
      "R-C_pairwise": 0.7986111111111112,
      "R-R_pairwise": 0.9074074074074074,
      "kendall": 0.27856025039123633,
      "pairwise": 0.753968253968254,
      "spearman": 0.4196411344781015
    }
  },
  "query_only": {
    "vs_dynamic_diag": {
      "C-C_pairwise": 0.9444444444444444,
      "R-C_pairwise": 0.8402777777777778,
      "R-R_pairwise": 1.0,
      "kendall": 0.6964006259780907,
      "mean_relative_error": 0.9645722466672009,
      "median_relative_error": 0.9594816113295183,
      "pairwise": 0.8968253968253969,
      "spearman": 0.8514373914721204
    },
    "vs_full": {
      "C-C_pairwise": 0.9444444444444444,
      "R-C_pairwise": 0.8819444444444444,
      "R-R_pairwise": 1.0,
      "kendall": 0.7151799687010955,
      "mean_relative_error": 0.9656889255273126,
      "median_relative_error": 0.9608130839746594,
      "pairwise": 0.9206349206349206,
      "spearman": 0.8713100520933822
    },
    "vs_future_KL": {
      "C-C_pairwise": 0.46296296296296297,
      "R-C_pairwise": 0.9791666666666666,
      "R-R_pairwise": 0.9074074074074074,
      "kendall": 0.4522691705790297,
      "pairwise": 0.8531746031746031,
      "spearman": 0.6397195961155058
    }
  },
  "static_column": {
    "vs_dynamic_diag": {
      "C-C_pairwise": 1.0,
      "R-C_pairwise": 0.8333333333333334,
      "R-R_pairwise": 1.0,
      "kendall": 0.5868544600938967,
      "mean_relative_error": 325.8975996679259,
      "median_relative_error": 331.24152574019035,
      "pairwise": 0.9047619047619048,
      "spearman": 0.7840054022766737
    },
    "vs_full": {
      "C-C_pairwise": 1.0,
      "R-C_pairwise": 0.875,
      "R-R_pairwise": 1.0,
      "kendall": 0.6134585289514867,
      "mean_relative_error": 321.5401391624221,
      "median_relative_error": 310.63514927280096,
      "pairwise": 0.9285714285714286,
      "spearman": 0.8096340600681716
    },
    "vs_future_KL": {
      "C-C_pairwise": 0.48148148148148145,
      "R-C_pairwise": 0.9722222222222222,
      "R-R_pairwise": 0.9074074074074074,
      "kendall": 0.3364632237871675,
      "pairwise": 0.8531746031746031,
      "spearman": 0.5046626792719789
    }
  }
}

## Strict Held-Out Generalization
{
  "candidates": {
    "M0": {
      "vs_dynamic_diag": {
        "C-C_pairwise": 1.0,
        "R-C_pairwise": 0.875,
        "R-R_pairwise": 1.0,
        "kendall": 0.6549295774647887,
        "mean_relative_error": 1.4366860398996213,
        "median_relative_error": 1.3648121267172941,
        "pairwise": 0.9285714285714286,
        "spearman": 0.8407614637597273
      },
      "vs_full": {
        "C-C_pairwise": 1.0,
        "R-C_pairwise": 0.875,
        "R-R_pairwise": 1.0,
        "kendall": 0.662754303599374,
        "mean_relative_error": 1.430697980566813,
        "median_relative_error": 1.378454914428505,
        "pairwise": 0.9285714285714286,
        "spearman": 0.8383497331018072
      },
      "vs_future_KL": {
        "C-C_pairwise": 0.48148148148148145,
        "R-C_pairwise": 0.7638888888888888,
        "R-R_pairwise": 0.9074074074074074,
        "kendall": 0.2809076682316119,
        "pairwise": 0.7341269841269841,
        "spearman": 0.42083092160267543
      }
    },
    "dynq_staticw": {
      "vs_dynamic_diag": {
        "C-C_pairwise": 0.9814814814814815,
        "R-C_pairwise": 0.9444444444444444,
        "R-R_pairwise": 1.0,
        "kendall": 0.636150234741784,
        "mean_relative_error": 0.16187838382530126,
        "median_relative_error": 0.10399933342215147,
        "pairwise": 0.9642857142857143,
        "spearman": 0.8109846292366069
      },
      "vs_full": {
        "C-C_pairwise": 0.9814814814814815,
        "R-C_pairwise": 0.9583333333333334,
        "R-R_pairwise": 1.0,
        "kendall": 0.661189358372457,
        "mean_relative_error": 0.15607866274199816,
        "median_relative_error": 0.08492093902344472,
        "pairwise": 0.9722222222222222,
        "spearman": 0.82477972859991
      },
      "vs_future_KL": {
        "C-C_pairwise": 0.46296296296296297,
        "R-C_pairwise": 0.875,
        "R-R_pairwise": 0.9074074074074074,
        "kendall": 0.3137715179968701,
        "pairwise": 0.7936507936507936,
        "spearman": 0.46247347096276287
      }
    },
    "group16_q_staticw": {
      "vs_dynamic_diag": {
        "C-C_pairwise": 1.0,
        "R-C_pairwise": 0.8472222222222222,
        "R-R_pairwise": 1.0,
        "kendall": 0.568075117370892,
        "mean_relative_error": 1.3971098554496217,
        "median_relative_error": 1.3419554681612405,
        "pairwise": 0.9126984126984127,
        "spearman": 0.7609171007781851
      },
      "vs_full": {
        "C-C_pairwise": 1.0,
        "R-C_pairwise": 0.8888888888888888,
        "R-R_pairwise": 1.0,
        "kendall": 0.5915492957746479,
        "mean_relative_error": 1.3712931648205524,
        "median_relative_error": 1.3144539499763197,
        "pairwise": 0.9365079365079365,
        "spearman": 0.7840697150942183
      },
      "vs_future_KL": {
        "C-C_pairwise": 0.48148148148148145,
        "R-C_pairwise": 0.9722222222222222,
        "R-R_pairwise": 0.9074074074074074,
        "kendall": 0.3161189358372457,
        "pairwise": 0.8531746031746031,
        "spearman": 0.4765258215962441
      }
    },
    "group32_q_staticw": {
      "vs_dynamic_diag": {
        "C-C_pairwise": 1.0,
        "R-C_pairwise": 0.8472222222222222,
        "R-R_pairwise": 1.0,
        "kendall": 0.5876369327073553,
        "mean_relative_error": 1.4462507193381482,
        "median_relative_error": 1.5231503023465534,
        "pairwise": 0.9126984126984127,
        "spearman": 0.7751945462730723
      },
      "vs_full": {
        "C-C_pairwise": 1.0,
        "R-C_pairwise": 0.8888888888888888,
        "R-R_pairwise": 1.0,
        "kendall": 0.6111111111111112,
        "mean_relative_error": 1.4158998513130276,
        "median_relative_error": 1.355348981205732,
        "pairwise": 0.9365079365079365,
        "spearman": 0.7980899093189273
      },
      "vs_future_KL": {
        "C-C_pairwise": 0.48148148148148145,
        "R-C_pairwise": 0.9722222222222222,
        "R-R_pairwise": 0.9074074074074074,
        "kendall": 0.33568075117370894,
        "pairwise": 0.8531746031746031,
        "spearman": 0.49601260531223873
      }
    },
    "group4_q_staticw": {
      "vs_dynamic_diag": {
        "C-C_pairwise": 1.0,
        "R-C_pairwise": 0.8888888888888888,
        "R-R_pairwise": 1.0,
        "kendall": 0.57981220657277,
        "mean_relative_error": 0.9750030845780273,
        "median_relative_error": 0.8225424339049785,
        "pairwise": 0.9365079365079365,
        "spearman": 0.7728149720239244
      },
      "vs_full": {
        "C-C_pairwise": 1.0,
        "R-C_pairwise": 0.9166666666666666,
        "R-R_pairwise": 1.0,
        "kendall": 0.6048513302034428,
        "mean_relative_error": 0.9648621126672554,
        "median_relative_error": 0.8418068177156592,
        "pairwise": 0.9523809523809523,
        "spearman": 0.791948035243424
      },
      "vs_future_KL": {
        "C-C_pairwise": 0.48148148148148145,
        "R-C_pairwise": 0.9305555555555556,
        "R-R_pairwise": 0.9074074074074074,
        "kendall": 0.2793427230046948,
        "pairwise": 0.8293650793650794,
        "spearman": 0.4296739340150492
      }
    },
    "group8_q_staticw": {
      "vs_dynamic_diag": {
        "C-C_pairwise": 1.0,
        "R-C_pairwise": 0.875,
        "R-R_pairwise": 1.0,
        "kendall": 0.564945226917058,
        "mean_relative_error": 1.1345187253359925,
        "median_relative_error": 0.9344931842678306,
        "pairwise": 0.9285714285714286,
        "spearman": 0.7549038523377709
      },
      "vs_full": {
        "C-C_pairwise": 1.0,
        "R-C_pairwise": 0.9166666666666666,
        "R-R_pairwise": 1.0,
        "kendall": 0.594679186228482,
        "mean_relative_error": 1.1204919436094616,
        "median_relative_error": 0.998151481972542,
        "pairwise": 0.9523809523809523,
        "spearman": 0.778635282011705
      },
      "vs_future_KL": {
        "C-C_pairwise": 0.48148148148148145,
        "R-C_pairwise": 0.9444444444444444,
        "R-R_pairwise": 0.9074074074074074,
        "kendall": 0.28169014084507044,
        "pairwise": 0.8373015873015873,
        "spearman": 0.43436876969580035
      }
    },
    "headq_staticw": {
      "vs_dynamic_diag": {
        "C-C_pairwise": 1.0,
        "R-C_pairwise": 0.8333333333333334,
        "R-R_pairwise": 1.0,
        "kendall": 0.5884194053208138,
        "mean_relative_error": 1.5528031464748746,
        "median_relative_error": 1.5961626009907612,
        "pairwise": 0.9047619047619048,
        "spearman": 0.7845520612258023
      },
      "vs_full": {
        "C-C_pairwise": 1.0,
        "R-C_pairwise": 0.875,
        "R-R_pairwise": 1.0,
        "kendall": 0.6150234741784038,
        "mean_relative_error": 1.5187635753255813,
        "median_relative_error": 1.4333683758690885,
        "pairwise": 0.9285714285714286,
        "spearman": 0.8102771882436169
      },
      "vs_future_KL": {
        "C-C_pairwise": 0.48148148148148145,
        "R-C_pairwise": 0.9722222222222222,
        "R-R_pairwise": 0.9074074074074074,
        "kendall": 0.3380281690140845,
        "pairwise": 0.8531746031746031,
        "spearman": 0.5061418740755033
      }
    },
    "q2_staticw": {
      "vs_dynamic_diag": {
        "C-C_pairwise": 1.0,
        "R-C_pairwise": 0.9652777777777778,
        "R-R_pairwise": 1.0,
        "kendall": 0.6338028169014085,
        "mean_relative_error": 0.15272552691836622,
        "median_relative_error": 0.08833085939950146,
        "pairwise": 0.9801587301587301,
        "spearman": 0.7870281047012669
      },
      "vs_full": {
        "C-C_pairwise": 1.0,
        "R-C_pairwise": 0.9236111111111112,
        "R-R_pairwise": 1.0,
        "kendall": 0.6416275430359938,
        "mean_relative_error": 0.16388392042065256,
        "median_relative_error": 0.09575093485929353,
        "pairwise": 0.9563492063492064,
        "spearman": 0.7889896456363753
      },
      "vs_future_KL": {
        "C-C_pairwise": 0.48148148148148145,
        "R-C_pairwise": 0.7986111111111112,
        "R-R_pairwise": 0.9074074074074074,
        "kendall": 0.27856025039123633,
        "pairwise": 0.753968253968254,
        "spearman": 0.4196411344781015
      }
    },
    "query_only": {
      "vs_dynamic_diag": {
        "C-C_pairwise": 0.9444444444444444,
        "R-C_pairwise": 0.8402777777777778,
        "R-R_pairwise": 1.0,
        "kendall": 0.6964006259780907,
        "mean_relative_error": 0.9645722466672009,
        "median_relative_error": 0.9594816113295183,
        "pairwise": 0.8968253968253969,
        "spearman": 0.8514373914721204
      },
      "vs_full": {
        "C-C_pairwise": 0.9444444444444444,
        "R-C_pairwise": 0.8819444444444444,
        "R-R_pairwise": 1.0,
        "kendall": 0.7151799687010955,
        "mean_relative_error": 0.9656889255273126,
        "median_relative_error": 0.9608130839746594,
        "pairwise": 0.9206349206349206,
        "spearman": 0.8713100520933822
      },
      "vs_future_KL": {
        "C-C_pairwise": 0.46296296296296297,
        "R-C_pairwise": 0.9791666666666666,
        "R-R_pairwise": 0.9074074074074074,
        "kendall": 0.4522691705790297,
        "pairwise": 0.8531746031746031,
        "spearman": 0.6397195961155058
      }
    },
    "static_column": {
      "vs_dynamic_diag": {
        "C-C_pairwise": 1.0,
        "R-C_pairwise": 0.8333333333333334,
        "R-R_pairwise": 1.0,
        "kendall": 0.5868544600938967,
        "mean_relative_error": 325.8975996679259,
        "median_relative_error": 331.24152574019035,
        "pairwise": 0.9047619047619048,
        "spearman": 0.7840054022766737
      },
      "vs_full": {
        "C-C_pairwise": 1.0,
        "R-C_pairwise": 0.875,
        "R-R_pairwise": 1.0,
        "kendall": 0.6134585289514867,
        "mean_relative_error": 321.5401391624221,
        "median_relative_error": 310.63514927280096,
        "pairwise": 0.9285714285714286,
        "spearman": 0.8096340600681716
      },
      "vs_future_KL": {
        "C-C_pairwise": 0.48148148148148145,
        "R-C_pairwise": 0.9722222222222222,
        "R-R_pairwise": 0.9074074074074074,
        "kendall": 0.3364632237871675,
        "pairwise": 0.8531746031746031,
        "spearman": 0.5046626792719789
      }
    }
  },
  "protocol": "leave-one-unit-out static w_bar"
}

## Dynamic-q Incremental Value
{
  "best_efficient": "q2_staticw",
  "best_group": "group32_q_staticw",
  "dynq_staticw_minus_static_column_spearman": 0.026979226959933156
}

## Temporal Causality
{
  "future_oracle4_q2_staticw": {
    "ORACLE_ONLY": "YES",
    "vs_dynamic_diag": {
      "C-C_pairwise": 1.0,
      "R-C_pairwise": 0.9305555555555556,
      "R-R_pairwise": 1.0,
      "kendall": 0.6009389671361502,
      "mean_relative_error": 0.17024172477015814,
      "median_relative_error": 0.12252297874452156,
      "pairwise": 0.9603174603174603,
      "spearman": 0.7597916264711557
    },
    "vs_future_KL": {
      "C-C_pairwise": 0.48148148148148145,
      "R-C_pairwise": 0.7777777777777778,
      "R-R_pairwise": 0.9074074074074074,
      "kendall": 0.24726134585289514,
      "pairwise": 0.7420634920634921,
      "spearman": 0.3760370441829057
    }
  },
  "prev1_q2_staticw": {
    "ORACLE_ONLY": "NO",
    "vs_dynamic_diag": {
      "C-C_pairwise": 1.0,
      "R-C_pairwise": 0.8958333333333334,
      "R-R_pairwise": 1.0,
      "kendall": 0.607981220657277,
      "mean_relative_error": 0.1565395945195761,
      "median_relative_error": 0.11249577066701394,
      "pairwise": 0.9404761904761905,
      "spearman": 0.7740690719660428
    },
    "vs_future_KL": {
      "C-C_pairwise": 0.48148148148148145,
      "R-C_pairwise": 0.7986111111111112,
      "R-R_pairwise": 0.9074074074074074,
      "kendall": 0.2652582159624413,
      "pairwise": 0.753968253968254,
      "spearman": 0.397774776512959
    }
  },
  "q2_staticw": {
    "ORACLE_ONLY": "NO",
    "vs_dynamic_diag": {
      "C-C_pairwise": 1.0,
      "R-C_pairwise": 0.9652777777777778,
      "R-R_pairwise": 1.0,
      "kendall": 0.6338028169014085,
      "mean_relative_error": 0.15272552691836622,
      "median_relative_error": 0.08833085939950146,
      "pairwise": 0.9801587301587301,
      "spearman": 0.7870281047012669
    },
    "vs_future_KL": {
      "C-C_pairwise": 0.48148148148148145,
      "R-C_pairwise": 0.7986111111111112,
      "R-R_pairwise": 0.9074074074074074,
      "kendall": 0.27856025039123633,
      "pairwise": 0.753968253968254,
      "spearman": 0.4196411344781015
    }
  },
  "recent16_q2_staticw": {
    "ORACLE_ONLY": "NO",
    "vs_dynamic_diag": {
      "C-C_pairwise": 1.0,
      "R-C_pairwise": 0.9166666666666666,
      "R-R_pairwise": 1.0,
      "kendall": 0.6173708920187794,
      "mean_relative_error": 0.15782864410889275,
      "median_relative_error": 0.12069701018723888,
      "pairwise": 0.9523809523809523,
      "spearman": 0.7781207794713486
    },
    "vs_future_KL": {
      "C-C_pairwise": 0.48148148148148145,
      "R-C_pairwise": 0.7777777777777778,
      "R-R_pairwise": 0.9074074074074074,
      "kendall": 0.2715179968701095,
      "pairwise": 0.7420634920634921,
      "spearman": 0.40690719660428326
    }
  },
  "recent4_q2_staticw": {
    "ORACLE_ONLY": "NO",
    "vs_dynamic_diag": {
      "C-C_pairwise": 1.0,
      "R-C_pairwise": 0.9166666666666666,
      "R-R_pairwise": 1.0,
      "kendall": 0.6087636932707355,
      "mean_relative_error": 0.1654449681895397,
      "median_relative_error": 0.11593461473340831,
      "pairwise": 0.9523809523809523,
      "spearman": 0.7698565824168757
    },
    "vs_future_KL": {
      "C-C_pairwise": 0.48148148148148145,
      "R-C_pairwise": 0.7916666666666666,
      "R-R_pairwise": 0.9074074074074074,
      "kendall": 0.27543035993740217,
      "pairwise": 0.75,
      "spearman": 0.40565309666216476
    }
  },
  "recent8_q2_staticw": {
    "ORACLE_ONLY": "NO",
    "vs_dynamic_diag": {
      "C-C_pairwise": 1.0,
      "R-C_pairwise": 0.9166666666666666,
      "R-R_pairwise": 1.0,
      "kendall": 0.5970266040688575,
      "mean_relative_error": 0.1668906115458523,
      "median_relative_error": 0.12623063400666118,
      "pairwise": 0.9523809523809523,
      "spearman": 0.7606598495080069
    },
    "vs_future_KL": {
      "C-C_pairwise": 0.48148148148148145,
      "R-C_pairwise": 0.7777777777777778,
      "R-R_pairwise": 0.9074074074074074,
      "kendall": 0.2683881064162754,
      "pairwise": 0.7420634920634921,
      "spearman": 0.399929255900701
    }
  }
}

## Future-query Oracle Gap
-0.027236478230111216

## FLOP Analysis
{
  "rows": [
    {
      "Candidate": "M0",
      "Dynamic metadata": "none",
      "Extra FLOPs": 0,
      "Extra state bytes": 0,
      "Extra state pass": 0,
      "Fidelity": "baseline",
      "Fusable": "YES",
      "Future-KL": "evaluation only",
      "Practical": "YES",
      "Selected": false,
      "Static dimensions": 0
    },
    {
      "Candidate": "query_only exact",
      "Dynamic metadata": "q_t O(d_k)",
      "Extra FLOPs": 1048576,
      "Extra state bytes": 0,
      "Extra state pass": 0,
      "Fidelity": "high dynamic baseline",
      "Fusable": "YES",
      "Future-KL": "evaluation only",
      "Practical": "PARTIAL",
      "Selected": false,
      "Static dimensions": 0
    },
    {
      "Candidate": "dyn-q/static-w exact",
      "Dynamic metadata": "q_t O(d_k)",
      "Extra FLOPs": 1052672,
      "Extra state bytes": 0,
      "Extra state pass": 0,
      "Fidelity": "best query-conditioned exact",
      "Fusable": "YES",
      "Future-KL": "evaluation only",
      "Practical": "PARTIAL",
      "Selected": false,
      "Static dimensions": 4096
    },
    {
      "Candidate": "q2/static-w",
      "Dynamic metadata": "q2 O(d_k)",
      "Extra FLOPs": 1572864,
      "Extra state bytes": 0,
      "Extra state pass": 0,
      "Fidelity": "separable no cross-row terms",
      "Fusable": "YES",
      "Future-KL": "evaluation only",
      "Practical": "YES",
      "Selected": true,
      "Static dimensions": 4096
    },
    {
      "Candidate": "group-q/static-w",
      "Dynamic metadata": "group q2 O(d_k/group)",
      "Extra FLOPs": 1572864,
      "Extra state bytes": 0,
      "Extra state pass": 0,
      "Fidelity": "coarser q2",
      "Fusable": "YES",
      "Future-KL": "evaluation only",
      "Practical": "YES",
      "Selected": false,
      "Static dimensions": 4096
    },
    {
      "Candidate": "head-q/static-w",
      "Dynamic metadata": "head scalar O(head)",
      "Extra FLOPs": 1572864,
      "Extra state bytes": 0,
      "Extra state pass": 0,
      "Fidelity": "head energy only",
      "Fusable": "YES",
      "Future-KL": "evaluation only",
      "Practical": "YES",
      "Selected": false,
      "Static dimensions": 4096
    },
    {
      "Candidate": "static-column",
      "Dynamic metadata": "none",
      "Extra FLOPs": 1048576,
      "Extra state bytes": 0,
      "Extra state pass": 0,
      "Fidelity": "no dynamic q",
      "Fusable": "YES",
      "Future-KL": "evaluation only",
      "Practical": "YES",
      "Selected": false,
      "Static dimensions": 4096
    }
  ]
}

## Memory-Traffic Analysis
Best efficient candidate has zero extra state read bytes because it is fusible into the existing state scan.

## Single-Pass Fusion Analysis
# Fusion Feasibility

Fusable single pass candidates: q2/static-w, grouped-q/static-w, head-q/static-w, static-column.

During an existing state scan, each loaded `E_ij` can update weighted statistics using either `q2_i * w_j`, `group_q2_g * w_j`, `head_energy_h * w_j`, or `w_j`. This does not require reloading the recurrent state.

Exact query-only and dyn-q/static-w require column reductions `E^T q`, which are scientifically useful but less directly compatible with a per-element quantization statistic.

## Optional Microbenchmark
{
  "MICROBENCH_BACKEND_LIMITED": "YES",
  "MICROBENCH_OVERHEAD": 0.4362282620141247,
  "baseline_current_int8_stat_proxy": {
    "median_ms": 0.08806400001049042,
    "p25_ms": 0.0859839990735054,
    "p75_ms": 0.08908800035715103
  },
  "group16_staticw_proxy": {
    "median_ms": 0.1258240044116974,
    "p25_ms": 0.12492799758911133,
    "p75_ms": 0.12697599828243256
  },
  "note": "PyTorch elementwise proxy benchmark only; not production CUDA latency.",
  "q2_staticw_proxy": {
    "median_ms": 0.12648000568151474,
    "p25_ms": 0.12572799623012543,
    "p75_ms": 0.12800000607967377
  }
}

## Complexity-vs-Fidelity Pareto Frontier
[
  {
    "M0": 0.8407614637597273
  },
  {
    "query_only": 0.8514373914721204
  },
  {
    "dynq_staticw": 0.8109846292366069
  },
  {
    "q2_staticw": 0.7870281047012669
  },
  {
    "group4_q_staticw": 0.7728149720239244
  },
  {
    "group8_q_staticw": 0.7549038523377709
  },
  {
    "group16_q_staticw": 0.7609171007781851
  },
  {
    "group32_q_staticw": 0.7751945462730723
  },
  {
    "headq_staticw": 0.7845520612258023
  },
  {
    "static_column": 0.7840054022766737
  }
]

## Positive Results
Zero extra state pass is feasible for q2/group/head/static-column candidates, and the cheap q2/group approximations are fusible into a single state scan.

## Negative Results
Exact dyn-q/static-w and cheap q2/group variants do not substantially beat static-column or query-only fidelity under strict held-out evaluation; current/recent q gives limited future-KL signal.

## Difference From DAMP
This audit does not allocate mixed precision or FP16 protection; it evaluates same-bit INT8 objective weights.

## Final Scientific Classification
QUERY_CONDITIONED_RISK_NOT_SUPPORTED

## Readiness Gates
{
  "ALGORITHMIC_EFFICIENCY_READY": "YES",
  "QUANTIZER_DESIGN_READY": "NO",
  "SCIENTIFIC_SIGNAL_READY": "PARTIAL",
  "TEMPORAL_CAUSALITY_READY": "PARTIAL"
}

## Exact Next Task
investigate alternative causal state-risk predictors
