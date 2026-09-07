# Protocol Audit

```json
{
  "CALIBRATION_EVAL_DISJOINT": "YES",
  "FUNCTIONAL_SCORE_DEFINITION_MATCH_PREVIOUS": "YES",
  "HEAD": "c164dab6bd2279b8aca1afe5c13ff8886df70a7d",
  "MAGNITUDE_DEFINITION_MATCH_PREVIOUS": "YES",
  "PREVIOUS_RESULT_REPRODUCTION_GATE": "PASS",
  "PROTOCOL_GATE": "PASS",
  "TASK": "GDN_MAGNITUDE_CONDITIONED_FUNCTIONAL_INCREMENTAL_UTILITY_V1",
  "branch": "research-sync-2026-09-02",
  "calibration_units": [
    "test/algebra/1332.json|64",
    "test/algebra/1332.json|128",
    "test/algebra/1332.json|256",
    "test/counting_and_probability/119.json|64",
    "test/counting_and_probability/119.json|128",
    "test/counting_and_probability/119.json|256"
  ],
  "candidate_band": "magnitude rank 25%-50% within each layer/head",
  "evaluation_units": [
    "test/geometry/477.json|64",
    "test/geometry/477.json|128",
    "test/geometry/477.json|256"
  ],
  "functional_definition": "FAVOR_j = E_t g_t,j * (q_t^T e_t,j^C128)^2 from previous selector_scores.csv",
  "git_status_start": "?? docs/gdn_magnitude_conditioned_functional_v1/\n?? experiments/propagation/run_gdn_magnitude_conditioned_functional_incremental_utility.py\n?? reports/propagation/gdn_magnitude_conditioned_functional_v1_protocol_audit.md\n?? results/propagation/gdn_magnitude_conditioned_functional_v1_formal_pair_results.json\n?? results/propagation/gdn_magnitude_conditioned_functional_v1_matched_pair_manifest.csv\n?? results/propagation/gdn_magnitude_conditioned_functional_v1_matched_pair_manifest.json\n?? results/propagation/gdn_magnitude_conditioned_functional_v1_pair_eval_unit_rows.csv\n?? results/propagation/gdn_magnitude_conditioned_functional_v1_pair_eval_unit_rows.partial.csv\n?? results/propagation/gdn_magnitude_conditioned_functional_v1_pair_results.partial.json\n?? results/propagation/gdn_magnitude_conditioned_functional_v1_pilot_pair_results.json\n?? results/propagation/gdn_magnitude_conditioned_functional_v1_protocol_audit.json",
  "magnitude_definition": "E_t ||S_t,:,j||_2^2 from previous selector_scores.csv",
  "previous_final": "/data/zypan/runs/gdn_functional_column_protection_v1/final_classification.json",
  "previous_scores": "/data/zypan/runs/gdn_functional_column_protection_v1/selector_scores.csv",
  "primary_protection_ratio": 0.375,
  "primary_threshold": 0.02,
  "threshold_panel": [
    0.01,
    0.02,
    0.05
  ],
  "timestamp": "2026-09-07 11:23:46 +0800"
}
```
