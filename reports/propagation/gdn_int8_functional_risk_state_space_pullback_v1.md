# GDN_INT8_FUNCTIONAL_RISK_STATE_SPACE_PULLBACK_V1

## Task
GDN_INT8_FUNCTIONAL_RISK_STATE_SPACE_PULLBACK_V1

## Previous Scientific State
{
  "structural_simplification_classification": "FUNCTIONAL_GEOMETRY_COORDINATE_SEPARABLE_SUPPORTED",
  "structural_simplification_commit": "fd4b846"
}

## Why Direct Reshape Is Invalid
The exact metric is sum_j w_j (q^T E[:,j])^2, which expands to cross-row terms q_i q_k E_ij E_kj. A 4096 functional vector cannot be directly reshaped into a 128x128 element-wise state map.

## Exact Tensor Semantics
{
  "FUNCTIONAL_CHANNEL_LAYOUT_GATE": "PASS",
  "READOUT_CONTRACTION_GATE": "PASS",
  "STATE_TENSOR_SEMANTICS_GATE": "PASS",
  "functional_dimension": 4096,
  "per_head_state_shape": "[128, 128]",
  "q_shape": "[32, 128]",
  "readout_contraction": "x[h,v] = sum_k E[h,k,v] * q[h,k]",
  "recurrent_state_shape": "[1, 32, 128, 128]"
}

## Functional-Space Metric
M_diag,t^2 = x_t^T diag(diag(G_t)) x_t, x_t = E_t^T q_t.

## State-Space Pullback Derivation
H_t = B_t^T diag(diag(G_t)) B_t, implemented without materializing dense H.

## Numerical Identity Validation
{
  "PULLBACK_MAX_REL_ERROR": 3.4945910020894807e-07,
  "PULLBACK_MEAN_REL_ERROR": 4.444676882174263e-08,
  "PULLBACK_MEDIAN_REL_ERROR": 0.0,
  "STATE_PULLBACK_IDENTITY_GATE": "PASS",
  "n_checks": 1728
}

## Candidate State Metrics
{
  "M0": {
    "vs_dynamic_diag": {
      "C-C_pairwise": 1.0,
      "R-C_pairwise": 0.875,
      "R-R_pairwise": 1.0,
      "kendall": 0.6549295774647887,
      "mean_relative_error": 1.436687866443041,
      "median_relative_error": 1.3648182243528872,
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
  "query_only": {
    "vs_dynamic_diag": {
      "C-C_pairwise": 0.9444444444444444,
      "R-C_pairwise": 0.8402777777777778,
      "R-R_pairwise": 1.0,
      "kendall": 0.6964006259780907,
      "mean_relative_error": 0.9645719569836546,
      "median_relative_error": 0.9594815032723002,
      "pairwise": 0.8968253968253969,
      "spearman": 0.8514373914721204
    },
    "vs_full": {
      "C-C_pairwise": 0.9444444444444444,
      "R-C_pairwise": 0.8819444444444444,
      "R-R_pairwise": 1.0,
      "kendall": 0.7151799687010955,
      "mean_relative_error": 0.9656886675921462,
      "median_relative_error": 0.9608131056503599,
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
      "mean_relative_error": 1.552853689268002,
      "median_relative_error": 1.5942582819726259,
      "pairwise": 0.9047619047619048,
      "spearman": 0.7840054022766737
    },
    "vs_full": {
      "C-C_pairwise": 1.0,
      "R-C_pairwise": 0.875,
      "R-R_pairwise": 1.0,
      "kendall": 0.6134585289514867,
      "mean_relative_error": 1.5188110577732594,
      "median_relative_error": 1.433760364299903,
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
  },
  "static_cov": {
    "vs_dynamic_diag": {
      "C-C_pairwise": 0.9814814814814815,
      "R-C_pairwise": 0.9375,
      "R-R_pairwise": 1.0,
      "kendall": 0.6251956181533647,
      "mean_relative_error": 0.1524183509037932,
      "median_relative_error": 0.08477987070157687,
      "pairwise": 0.9603174603174603,
      "spearman": 0.7926233198276417
    },
    "vs_full": {
      "C-C_pairwise": 0.9814814814814815,
      "R-C_pairwise": 0.9375,
      "R-R_pairwise": 1.0,
      "kendall": 0.6424100156494522,
      "mean_relative_error": 0.1665575119706039,
      "median_relative_error": 0.09473144745338845,
      "pairwise": 0.9603174603174603,
      "spearman": 0.80153064505756
    },
    "vs_future_KL": {
      "C-C_pairwise": 0.46296296296296297,
      "R-C_pairwise": 0.8125,
      "R-R_pairwise": 0.9074074074074074,
      "kendall": 0.2652582159624413,
      "pairwise": 0.7579365079365079,
      "spearman": 0.4068107273779664
    }
  },
  "static_element": {
    "vs_dynamic_diag": {
      "C-C_pairwise": 1.0,
      "R-C_pairwise": 0.9236111111111112,
      "R-R_pairwise": 1.0,
      "kendall": 0.6126760563380281,
      "mean_relative_error": 0.16403227221856284,
      "median_relative_error": 0.13468853250847626,
      "pairwise": 0.9563492063492064,
      "spearman": 0.7747443565502604
    },
    "vs_full": {
      "C-C_pairwise": 1.0,
      "R-C_pairwise": 0.8819444444444444,
      "R-R_pairwise": 1.0,
      "kendall": 0.6205007824726134,
      "mean_relative_error": 0.1776711293305208,
      "median_relative_error": 0.13763727176241725,
      "pairwise": 0.9325396825396826,
      "spearman": 0.7760949257186958
    },
    "vs_future_KL": {
      "C-C_pairwise": 0.48148148148148145,
      "R-C_pairwise": 0.7708333333333334,
      "R-R_pairwise": 0.9074074074074074,
      "kendall": 0.24647887323943662,
      "pairwise": 0.7380952380952381,
      "spearman": 0.38079619268120135
    }
  },
  "static_row": {
    "vs_dynamic_diag": {
      "C-C_pairwise": 1.0,
      "R-C_pairwise": 0.22916666666666666,
      "R-R_pairwise": 1.0,
      "kendall": 0.002347417840375587,
      "mean_relative_error": 0.9745654895073189,
      "median_relative_error": 0.6539637510354633,
      "pairwise": 0.5595238095238095,
      "spearman": -0.07312367354813815
    },
    "vs_full": {
      "C-C_pairwise": 1.0,
      "R-C_pairwise": 0.1875,
      "R-R_pairwise": 1.0,
      "kendall": -0.01486697965571205,
      "mean_relative_error": 1.055934536795612,
      "median_relative_error": 0.7058679123243934,
      "pairwise": 0.5357142857142857,
      "spearman": -0.11846420991703646
    },
    "vs_future_KL": {
      "C-C_pairwise": 0.48148148148148145,
      "R-C_pairwise": 0.09027777777777778,
      "R-R_pairwise": 0.9074074074074074,
      "kendall": -0.18544600938967137,
      "pairwise": 0.3492063492063492,
      "spearman": -0.28381246382404013
    }
  },
  "static_separable": {
    "vs_dynamic_diag": {
      "C-C_pairwise": 1.0,
      "R-C_pairwise": 0.9236111111111112,
      "R-R_pairwise": 1.0,
      "kendall": 0.6048513302034428,
      "mean_relative_error": 0.16803688334390302,
      "median_relative_error": 0.1318448978239431,
      "pairwise": 0.9563492063492064,
      "spearman": 0.7699852080519648
    },
    "vs_full": {
      "C-C_pairwise": 1.0,
      "R-C_pairwise": 0.8819444444444444,
      "R-R_pairwise": 1.0,
      "kendall": 0.6142410015649452,
      "mean_relative_error": 0.181378290773668,
      "median_relative_error": 0.14136823268725376,
      "pairwise": 0.9325396825396826,
      "spearman": 0.7710142131326774
    },
    "vs_future_KL": {
      "C-C_pairwise": 0.48148148148148145,
      "R-C_pairwise": 0.7708333333333334,
      "R-R_pairwise": 0.9074074074074074,
      "kendall": 0.24021909233176839,
      "pairwise": 0.7380952380952381,
      "spearman": 0.374879413467104
    }
  },
  "static_w_dynamic_q": {
    "vs_dynamic_diag": {
      "C-C_pairwise": 0.9814814814814815,
      "R-C_pairwise": 0.9444444444444444,
      "R-R_pairwise": 1.0,
      "kendall": 0.636150234741784,
      "mean_relative_error": 0.16188048496910457,
      "median_relative_error": 0.10400050523078906,
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
  }
}

## Natural Replay Results
{
  "M0": {
    "vs_dynamic_diag": {
      "C-C_pairwise": 1.0,
      "R-C_pairwise": 0.875,
      "R-R_pairwise": 1.0,
      "kendall": 0.6549295774647887,
      "mean_relative_error": 1.436687866443041,
      "median_relative_error": 1.3648182243528872,
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
  "query_only": {
    "vs_dynamic_diag": {
      "C-C_pairwise": 0.9444444444444444,
      "R-C_pairwise": 0.8402777777777778,
      "R-R_pairwise": 1.0,
      "kendall": 0.6964006259780907,
      "mean_relative_error": 0.9645719569836546,
      "median_relative_error": 0.9594815032723002,
      "pairwise": 0.8968253968253969,
      "spearman": 0.8514373914721204
    },
    "vs_full": {
      "C-C_pairwise": 0.9444444444444444,
      "R-C_pairwise": 0.8819444444444444,
      "R-R_pairwise": 1.0,
      "kendall": 0.7151799687010955,
      "mean_relative_error": 0.9656886675921462,
      "median_relative_error": 0.9608131056503599,
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
      "mean_relative_error": 1.552853689268002,
      "median_relative_error": 1.5942582819726259,
      "pairwise": 0.9047619047619048,
      "spearman": 0.7840054022766737
    },
    "vs_full": {
      "C-C_pairwise": 1.0,
      "R-C_pairwise": 0.875,
      "R-R_pairwise": 1.0,
      "kendall": 0.6134585289514867,
      "mean_relative_error": 1.5188110577732594,
      "median_relative_error": 1.433760364299903,
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
  },
  "static_cov": {
    "vs_dynamic_diag": {
      "C-C_pairwise": 0.9814814814814815,
      "R-C_pairwise": 0.9375,
      "R-R_pairwise": 1.0,
      "kendall": 0.6251956181533647,
      "mean_relative_error": 0.1524183509037932,
      "median_relative_error": 0.08477987070157687,
      "pairwise": 0.9603174603174603,
      "spearman": 0.7926233198276417
    },
    "vs_full": {
      "C-C_pairwise": 0.9814814814814815,
      "R-C_pairwise": 0.9375,
      "R-R_pairwise": 1.0,
      "kendall": 0.6424100156494522,
      "mean_relative_error": 0.1665575119706039,
      "median_relative_error": 0.09473144745338845,
      "pairwise": 0.9603174603174603,
      "spearman": 0.80153064505756
    },
    "vs_future_KL": {
      "C-C_pairwise": 0.46296296296296297,
      "R-C_pairwise": 0.8125,
      "R-R_pairwise": 0.9074074074074074,
      "kendall": 0.2652582159624413,
      "pairwise": 0.7579365079365079,
      "spearman": 0.4068107273779664
    }
  },
  "static_element": {
    "vs_dynamic_diag": {
      "C-C_pairwise": 1.0,
      "R-C_pairwise": 0.9236111111111112,
      "R-R_pairwise": 1.0,
      "kendall": 0.6126760563380281,
      "mean_relative_error": 0.16403227221856284,
      "median_relative_error": 0.13468853250847626,
      "pairwise": 0.9563492063492064,
      "spearman": 0.7747443565502604
    },
    "vs_full": {
      "C-C_pairwise": 1.0,
      "R-C_pairwise": 0.8819444444444444,
      "R-R_pairwise": 1.0,
      "kendall": 0.6205007824726134,
      "mean_relative_error": 0.1776711293305208,
      "median_relative_error": 0.13763727176241725,
      "pairwise": 0.9325396825396826,
      "spearman": 0.7760949257186958
    },
    "vs_future_KL": {
      "C-C_pairwise": 0.48148148148148145,
      "R-C_pairwise": 0.7708333333333334,
      "R-R_pairwise": 0.9074074074074074,
      "kendall": 0.24647887323943662,
      "pairwise": 0.7380952380952381,
      "spearman": 0.38079619268120135
    }
  },
  "static_row": {
    "vs_dynamic_diag": {
      "C-C_pairwise": 1.0,
      "R-C_pairwise": 0.22916666666666666,
      "R-R_pairwise": 1.0,
      "kendall": 0.002347417840375587,
      "mean_relative_error": 0.9745654895073189,
      "median_relative_error": 0.6539637510354633,
      "pairwise": 0.5595238095238095,
      "spearman": -0.07312367354813815
    },
    "vs_full": {
      "C-C_pairwise": 1.0,
      "R-C_pairwise": 0.1875,
      "R-R_pairwise": 1.0,
      "kendall": -0.01486697965571205,
      "mean_relative_error": 1.055934536795612,
      "median_relative_error": 0.7058679123243934,
      "pairwise": 0.5357142857142857,
      "spearman": -0.11846420991703646
    },
    "vs_future_KL": {
      "C-C_pairwise": 0.48148148148148145,
      "R-C_pairwise": 0.09027777777777778,
      "R-R_pairwise": 0.9074074074074074,
      "kendall": -0.18544600938967137,
      "pairwise": 0.3492063492063492,
      "spearman": -0.28381246382404013
    }
  },
  "static_separable": {
    "vs_dynamic_diag": {
      "C-C_pairwise": 1.0,
      "R-C_pairwise": 0.9236111111111112,
      "R-R_pairwise": 1.0,
      "kendall": 0.6048513302034428,
      "mean_relative_error": 0.16803688334390302,
      "median_relative_error": 0.1318448978239431,
      "pairwise": 0.9563492063492064,
      "spearman": 0.7699852080519648
    },
    "vs_full": {
      "C-C_pairwise": 1.0,
      "R-C_pairwise": 0.8819444444444444,
      "R-R_pairwise": 1.0,
      "kendall": 0.6142410015649452,
      "mean_relative_error": 0.181378290773668,
      "median_relative_error": 0.14136823268725376,
      "pairwise": 0.9325396825396826,
      "spearman": 0.7710142131326774
    },
    "vs_future_KL": {
      "C-C_pairwise": 0.48148148148148145,
      "R-C_pairwise": 0.7708333333333334,
      "R-R_pairwise": 0.9074074074074074,
      "kendall": 0.24021909233176839,
      "pairwise": 0.7380952380952381,
      "spearman": 0.374879413467104
    }
  },
  "static_w_dynamic_q": {
    "vs_dynamic_diag": {
      "C-C_pairwise": 0.9814814814814815,
      "R-C_pairwise": 0.9444444444444444,
      "R-R_pairwise": 1.0,
      "kendall": 0.636150234741784,
      "mean_relative_error": 0.16188048496910457,
      "median_relative_error": 0.10400050523078906,
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
  }
}

## Held-Out Static Results
{
  "M0_VS_DYNAMIC_DIAG_SPEARMAN": 0.8407614637597273,
  "QUERY_ONLY_VS_DYNAMIC_DIAG_SPEARMAN": 0.8514373914721204,
  "STATIC_COLUMN_HELDOUT_PAIRWISE": 0.9047619047619048,
  "STATIC_COLUMN_VS_DYNAMIC_DIAG_SPEARMAN": 0.7840054022766737,
  "STATIC_COV_HELDOUT_PAIRWISE": 0.9603174603174603,
  "STATIC_COV_VS_DYNAMIC_DIAG_SPEARMAN": 0.7926233198276417,
  "STATIC_ELEMENT_HELDOUT_PAIRWISE": 0.9563492063492064,
  "STATIC_ELEMENT_VS_DYNAMIC_DIAG_SPEARMAN": 0.7747443565502604,
  "STATIC_ROW_HELDOUT_PAIRWISE": 0.5595238095238095,
  "STATIC_ROW_VS_DYNAMIC_DIAG_SPEARMAN": -0.07312367354813815,
  "STATIC_SEPARABLE_HELDOUT_PAIRWISE": 0.9563492063492064,
  "STATIC_SEPARABLE_VS_DYNAMIC_DIAG_SPEARMAN": 0.7699852080519648,
  "candidates": {
    "M0": {
      "vs_dynamic_diag": {
        "C-C_pairwise": 1.0,
        "R-C_pairwise": 0.875,
        "R-R_pairwise": 1.0,
        "kendall": 0.6549295774647887,
        "mean_relative_error": 1.436687866443041,
        "median_relative_error": 1.3648182243528872,
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
    "query_only": {
      "vs_dynamic_diag": {
        "C-C_pairwise": 0.9444444444444444,
        "R-C_pairwise": 0.8402777777777778,
        "R-R_pairwise": 1.0,
        "kendall": 0.6964006259780907,
        "mean_relative_error": 0.9645719569836546,
        "median_relative_error": 0.9594815032723002,
        "pairwise": 0.8968253968253969,
        "spearman": 0.8514373914721204
      },
      "vs_full": {
        "C-C_pairwise": 0.9444444444444444,
        "R-C_pairwise": 0.8819444444444444,
        "R-R_pairwise": 1.0,
        "kendall": 0.7151799687010955,
        "mean_relative_error": 0.9656886675921462,
        "median_relative_error": 0.9608131056503599,
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
        "mean_relative_error": 1.552853689268002,
        "median_relative_error": 1.5942582819726259,
        "pairwise": 0.9047619047619048,
        "spearman": 0.7840054022766737
      },
      "vs_full": {
        "C-C_pairwise": 1.0,
        "R-C_pairwise": 0.875,
        "R-R_pairwise": 1.0,
        "kendall": 0.6134585289514867,
        "mean_relative_error": 1.5188110577732594,
        "median_relative_error": 1.433760364299903,
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
    },
    "static_cov": {
      "vs_dynamic_diag": {
        "C-C_pairwise": 0.9814814814814815,
        "R-C_pairwise": 0.9375,
        "R-R_pairwise": 1.0,
        "kendall": 0.6251956181533647,
        "mean_relative_error": 0.1524183509037932,
        "median_relative_error": 0.08477987070157687,
        "pairwise": 0.9603174603174603,
        "spearman": 0.7926233198276417
      },
      "vs_full": {
        "C-C_pairwise": 0.9814814814814815,
        "R-C_pairwise": 0.9375,
        "R-R_pairwise": 1.0,
        "kendall": 0.6424100156494522,
        "mean_relative_error": 0.1665575119706039,
        "median_relative_error": 0.09473144745338845,
        "pairwise": 0.9603174603174603,
        "spearman": 0.80153064505756
      },
      "vs_future_KL": {
        "C-C_pairwise": 0.46296296296296297,
        "R-C_pairwise": 0.8125,
        "R-R_pairwise": 0.9074074074074074,
        "kendall": 0.2652582159624413,
        "pairwise": 0.7579365079365079,
        "spearman": 0.4068107273779664
      }
    },
    "static_element": {
      "vs_dynamic_diag": {
        "C-C_pairwise": 1.0,
        "R-C_pairwise": 0.9236111111111112,
        "R-R_pairwise": 1.0,
        "kendall": 0.6126760563380281,
        "mean_relative_error": 0.16403227221856284,
        "median_relative_error": 0.13468853250847626,
        "pairwise": 0.9563492063492064,
        "spearman": 0.7747443565502604
      },
      "vs_full": {
        "C-C_pairwise": 1.0,
        "R-C_pairwise": 0.8819444444444444,
        "R-R_pairwise": 1.0,
        "kendall": 0.6205007824726134,
        "mean_relative_error": 0.1776711293305208,
        "median_relative_error": 0.13763727176241725,
        "pairwise": 0.9325396825396826,
        "spearman": 0.7760949257186958
      },
      "vs_future_KL": {
        "C-C_pairwise": 0.48148148148148145,
        "R-C_pairwise": 0.7708333333333334,
        "R-R_pairwise": 0.9074074074074074,
        "kendall": 0.24647887323943662,
        "pairwise": 0.7380952380952381,
        "spearman": 0.38079619268120135
      }
    },
    "static_row": {
      "vs_dynamic_diag": {
        "C-C_pairwise": 1.0,
        "R-C_pairwise": 0.22916666666666666,
        "R-R_pairwise": 1.0,
        "kendall": 0.002347417840375587,
        "mean_relative_error": 0.9745654895073189,
        "median_relative_error": 0.6539637510354633,
        "pairwise": 0.5595238095238095,
        "spearman": -0.07312367354813815
      },
      "vs_full": {
        "C-C_pairwise": 1.0,
        "R-C_pairwise": 0.1875,
        "R-R_pairwise": 1.0,
        "kendall": -0.01486697965571205,
        "mean_relative_error": 1.055934536795612,
        "median_relative_error": 0.7058679123243934,
        "pairwise": 0.5357142857142857,
        "spearman": -0.11846420991703646
      },
      "vs_future_KL": {
        "C-C_pairwise": 0.48148148148148145,
        "R-C_pairwise": 0.09027777777777778,
        "R-R_pairwise": 0.9074074074074074,
        "kendall": -0.18544600938967137,
        "pairwise": 0.3492063492063492,
        "spearman": -0.28381246382404013
      }
    },
    "static_separable": {
      "vs_dynamic_diag": {
        "C-C_pairwise": 1.0,
        "R-C_pairwise": 0.9236111111111112,
        "R-R_pairwise": 1.0,
        "kendall": 0.6048513302034428,
        "mean_relative_error": 0.16803688334390302,
        "median_relative_error": 0.1318448978239431,
        "pairwise": 0.9563492063492064,
        "spearman": 0.7699852080519648
      },
      "vs_full": {
        "C-C_pairwise": 1.0,
        "R-C_pairwise": 0.8819444444444444,
        "R-R_pairwise": 1.0,
        "kendall": 0.6142410015649452,
        "mean_relative_error": 0.181378290773668,
        "median_relative_error": 0.14136823268725376,
        "pairwise": 0.9325396825396826,
        "spearman": 0.7710142131326774
      },
      "vs_future_KL": {
        "C-C_pairwise": 0.48148148148148145,
        "R-C_pairwise": 0.7708333333333334,
        "R-R_pairwise": 0.9074074074074074,
        "kendall": 0.24021909233176839,
        "pairwise": 0.7380952380952381,
        "spearman": 0.374879413467104
      }
    },
    "static_w_dynamic_q": {
      "vs_dynamic_diag": {
        "C-C_pairwise": 0.9814814814814815,
        "R-C_pairwise": 0.9444444444444444,
        "R-R_pairwise": 1.0,
        "kendall": 0.636150234741784,
        "mean_relative_error": 0.16188048496910457,
        "median_relative_error": 0.10400050523078906,
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
    }
  },
  "matched_rc_14": {
    "M0": {
      "concordance": 0.7142857142857143,
      "correct": 10,
      "total": 14
    },
    "M5_diag_dynamic": {
      "concordance": 1.0,
      "correct": 14,
      "total": 14
    },
    "M5_full": {
      "concordance": 1.0,
      "correct": 14,
      "total": 14
    },
    "query_only": {
      "concordance": 1.0,
      "correct": 14,
      "total": 14
    },
    "static_column": {
      "concordance": 1.0,
      "correct": 14,
      "total": 14
    },
    "static_cov": {
      "concordance": 0.8571428571428571,
      "correct": 12,
      "total": 14
    },
    "static_element": {
      "concordance": 0.7857142857142857,
      "correct": 11,
      "total": 14
    },
    "static_row": {
      "concordance": 0.0,
      "correct": 0,
      "total": 14
    },
    "static_separable": {
      "concordance": 0.7857142857142857,
      "correct": 11,
      "total": 14
    },
    "static_w_dynamic_q": {
      "concordance": 1.0,
      "correct": 14,
      "total": 14
    }
  },
  "protocol": "leave-one-unit-out; only calibration units used for w_bar, C_q, I_element, separable, row, and column factors; no future_KL fitting"
}

## Element vs Separable vs Row vs Column
{
  "STATE_IMPORTANCE_EFFECTIVE_RANK_MEDIAN": 2.2862675655643048,
  "STATE_IMPORTANCE_RANK1_REL_FRO_ERROR_MEDIAN": 0.16100059335426647,
  "STATE_IMPORTANCE_SEPARABLE": "YES",
  "STATE_IMPORTANCE_TOP1_SVD_ENERGY_MEDIAN": 0.9740788057938392,
  "STATE_IMPORTANCE_TOP4_SVD_ENERGY_MEDIAN": 0.9994581106836924,
  "column_vs_dynamic_diag_spearman": 0.7840054022766737,
  "interpretation": "state importance is evaluated in recurrent [key_row,value_col] space; this is distinct from full functional G low-rank analysis",
  "row_vs_dynamic_diag_spearman": -0.07312367354813815
}

## Controlled Intervention Validation
{
  "CONTROLLED_MAPPING_VALIDATION": "POSITIVE",
  "note": "local perturbation pilot only; no future KL rollout for synthetic perturbations",
  "static_column_vs_dynamic_diag_spearman": 0.7068503010206497,
  "static_column_vs_full_local_spearman": 0.6974627734035748,
  "static_cov_vs_dynamic_diag_spearman": 0.9224581507624124,
  "static_cov_vs_full_local_spearman": 0.9229160789388551,
  "static_element_vs_dynamic_diag_spearman": 0.9159708349294744,
  "static_element_vs_full_local_spearman": 0.9159708349294744,
  "static_row_vs_dynamic_diag_spearman": 0.8939902824602257,
  "static_row_vs_full_local_spearman": 0.8948298174503706,
  "static_separable_vs_dynamic_diag_spearman": 0.9265795043503965,
  "static_separable_vs_full_local_spearman": 0.9258926120857325
}

## Quantizer Compatibility
See quantizer_compatibility.md.

## Positive Results
Exact state pullback identity holds and static state metrics are evaluated under leave-one-unit-out calibration.

## Negative Results
Matched R-C 14/14 is reported only as consistency because isotropic also achieved 14/14 previously; no candidate is accepted on that evidence alone.

## Scientific Limitations
Synthetic perturbation validation is local-output only and does not run future KL. Static calibration uses the existing 9 canonical units.

## Difference From DAMP-Style Precision Allocation
This task scores same-bit recurrent-state quantization objectives; it does not allocate FP16/INT8/Top-K precision.

## Final Scientific Classification
FUNCTIONAL_PULLBACK_VALID_BUT_STATIC_STATE_METRIC_NOT_SUPPORTED

## STATE_IMPORTANCE_READY_FOR_QUANTIZER
NO

## Exact Next Recommended Task
investigate dynamic q-dependent state metrics before quantizer design
