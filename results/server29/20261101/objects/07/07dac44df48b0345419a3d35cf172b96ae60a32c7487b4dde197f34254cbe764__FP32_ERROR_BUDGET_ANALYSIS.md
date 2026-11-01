V2 canonical path is fixed M1, not selected by crossing the old threshold. Unit roundoff u=2^-24; gamma128=128u/(1-128u). For each head, the diagonal representation budget uses norm(C,2)*(2*norm(DeltaA,2)+norm(DeltaA,2)^2), plus actual M0 residual and an explicit FP64 evaluation budget. Near-zero mu<=1e-24 uses absolute errors. C32 conversion and two FP32 matrix products have separate norm/componentwise budgets. A32 orthogonality is checked against the representation perturbation bound and2*gamma128. This does not replace actual9B FP functional qualification. V1 remains FAIL under its original threshold.

{
  "M0": {
    "maximum": 9.551248320045331e-15,
    "P50": 4.332101885244269e-15,
    "P95": 7.106862926960982e-15,
    "P99": 8.329759394483562e-15
  },
  "M1": {
    "maximum": 1.6890838568648173e-07,
    "P50": 9.088166575281548e-08,
    "P95": 1.448462951089674e-07,
    "P99": 1.570809589997834e-07
  },
  "M2_V1_construct_FP32_evaluate_FP64": {
    "maximum": 2.7452575492717835e-06,
    "P50": 1.1484034377352961e-06,
    "P95": 1.8824256171447872e-06,
    "P99": 2.4192025745055413e-06
  },
  "M3_requested_FP32_checker": {
    "maximum": 2.1366414188318343e-06,
    "P50": 7.046280597514704e-07,
    "P95": 1.211803550909624e-06,
    "P99": 1.625445007181426e-06
  },
  "C_cast_actual": {
    "maximum": 1.6773987792942e-07,
    "P50": 5.997163031891564e-08,
    "P95": 1.1362976290239093e-07,
    "P99": 1.4099737755532064e-07
  },
  "checker_actual": {
    "maximum": 2.1011433147703683e-06,
    "P50": 6.976880329372714e-07,
    "P95": 1.2174065581809036e-06,
    "P99": 1.594207325133644e-06
  },
  "representation_budget": {
    "maximum": 1.2777660262560071e-05,
    "P50": 7.584207466772085e-06,
    "P95": 1.2223699151013066e-05,
    "P99": 1.2537158646163684e-05
  },
  "FP32_A_orth_spectral": {
    "maximum": 7.680727240586925e-08,
    "P50": 7.107655541127806e-08,
    "P95": 7.383744931621897e-08,
    "P99": 7.515416109425408e-08
  },
  "FP64_eigen_relative": {
    "maximum": 2.4496088435345183e-15,
    "P50": 1.4315138421904117e-15,
    "P95": 1.9122441907963584e-15,
    "P99": 2.1052645239628737e-15
  },
  "FP64_U_orth_max": {
    "maximum": 3.885780586188048e-15,
    "P50": 2.220446049250313e-15,
    "P95": 3.1086244689504383e-15,
    "P99": 3.552713678800501e-15
  },
  "delta_A_spectral": {
    "maximum": 5.207700416169671e-08,
    "P50": 4.948331663143789e-08,
    "P95": 5.0944462050329736e-08,
    "P99": 5.133075089139892e-08
  }
}

Failures: []
