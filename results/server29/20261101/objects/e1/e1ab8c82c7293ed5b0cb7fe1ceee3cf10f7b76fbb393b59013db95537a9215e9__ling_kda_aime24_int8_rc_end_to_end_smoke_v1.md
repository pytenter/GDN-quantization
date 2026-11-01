# LING_KDA_AIME24_INT8_RC_END_TO_END_SMOKE_V1

## Summary
- FORMAL_STATUS: `RUNNING`
- FP_STATE: 1/30 (0.5000)
- INT8_R128: 0/30 (0.0000)
- INT8_C128: 6/30 (0.2000)
- FINAL_SCIENTIFIC_CLASSIFICATION: `LING_KDA_AIME24_PROTOCOL_OR_RUNTIME_INCONCLUSIVE`
- INITIAL_RC_TRANSFER_SIGNAL: `INCONCLUSIVE`

## Gates

- R128_SEMANTICS_GATE: `PASS`
- C128_SEMANTICS_GATE: `PASS`
- PREFILL_QUANTIZATION_GATE: `PASS`
- DECODE_QUANTIZATION_GATE: `PASS`
- FP_IDENTITY_GATE: `PASS`
- INT8_R128_SANITY_GATE: `PASS`
- INT8_C128_SANITY_GATE: `PASS`
- AIME_DATASET_GATE: `PASS`
- EVALUATOR_GATE: `PASS`

## Health

```json
{
  "FP_STATE": {
    "accuracy": 0.5,
    "attempted": 2,
    "correct": 1,
    "max_tokens": 32768,
    "mean_tokens": 18467.5,
    "median_tokens": 18467.5,
    "min_tokens": 4167,
    "nonfinite": 0,
    "runtime_errors": 0,
    "success": 1,
    "truncated": 1
  },
  "INT8_C128": {
    "accuracy": 0.2,
    "attempted": 30,
    "correct": 6,
    "max_tokens": 32768,
    "mean_tokens": 28500.866666666665,
    "median_tokens": 32768.0,
    "min_tokens": 6338,
    "nonfinite": 0,
    "runtime_errors": 0,
    "success": 6,
    "truncated": 24
  },
  "INT8_R128": {
    "accuracy": 0.0,
    "attempted": 2,
    "correct": 0,
    "max_tokens": 32768,
    "mean_tokens": 18371.5,
    "median_tokens": 18371.5,
    "min_tokens": 3975,
    "nonfinite": 0,
    "runtime_errors": 0,
    "success": 1,
    "truncated": 1
  }
}
```

## Paired Problem Comparison

| problem_id | FP | R128 | C128 | FP ans | R ans | C ans | FP tok | R tok | C tok | FP trunc | R trunc | C trunc |
|---|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 60 | 1 | 0 | 0 | 204 | 240 | 240 minutes? But wait, if t=168, then case 2: walking time = 0, so the walk takes 168 minutes? But then the walk takes 168 minutes? But the question asks: "Find the number of minutes the walk takes her, including the t minutes spent in the coffee shop | 4167 | 3975 | 32768 | 0 | 0 | 1 |
| 61 | 0 | 0 | 0 | 5 | 450 | 4 | 32768 | 32768 | 32768 | 1 | 1 | 1 |

## Final Summary JSON

```json
{
  "AIME_DATASET_GATE": "PASS",
  "C128_SEMANTICS_GATE": "PASS",
  "COMMIT": null,
  "COMPLETED_GENERATIONS": 34,
  "CONFIG_STATS": {
    "FP_STATE": {
      "accuracy": 0.5,
      "attempted": 2,
      "correct": 1,
      "max_tokens": 32768,
      "mean_tokens": 18467.5,
      "median_tokens": 18467.5,
      "min_tokens": 4167,
      "nonfinite": 0,
      "runtime_errors": 0,
      "success": 1,
      "truncated": 1
    },
    "INT8_C128": {
      "accuracy": 0.2,
      "attempted": 30,
      "correct": 6,
      "max_tokens": 32768,
      "mean_tokens": 28500.866666666665,
      "median_tokens": 32768.0,
      "min_tokens": 6338,
      "nonfinite": 0,
      "runtime_errors": 0,
      "success": 6,
      "truncated": 24
    },
    "INT8_R128": {
      "accuracy": 0.0,
      "attempted": 2,
      "correct": 0,
      "max_tokens": 32768,
      "mean_tokens": 18371.5,
      "median_tokens": 18371.5,
      "min_tokens": 3975,
      "nonfinite": 0,
      "runtime_errors": 0,
      "success": 1,
      "truncated": 1
    }
  },
  "DATASET": "HuggingFaceH4/aime_2024 train",
  "DECODE_QUANTIZATION_GATE": "PASS",
  "DECODING_PROTOCOL": "do_sample=True, temperature=0.6, top_p=0.95, repo canonical AIME protocol",
  "EVALUATOR_GATE": "PASS",
  "EXPECTED_GENERATIONS": 90,
  "FINAL_SCIENTIFIC_CLASSIFICATION": "LING_KDA_AIME24_PROTOCOL_OR_RUNTIME_INCONCLUSIVE",
  "FORMAL_STATUS": "RUNNING",
  "FP_CORRECT_C_WRONG": 1,
  "FP_CORRECT_R_WRONG": 1,
  "FP_IDENTITY_GATE": "PASS",
  "FP_STATE_ACCURACY": 0.5,
  "FP_STATE_CORRECT": 1,
  "FP_STATE_MEAN_TOKENS": 18467.5,
  "FP_STATE_MEDIAN_TOKENS": 18467.5,
  "FP_STATE_NONFINITE": 0,
  "FP_STATE_RUNTIME_ERRORS": 0,
  "FP_STATE_SUCCESS": 1,
  "FP_STATE_TRUNCATED": 1,
  "INITIAL_RC_TRANSFER_SIGNAL": "INCONCLUSIVE",
  "INT8_C128_ACCURACY": 0.2,
  "INT8_C128_CORRECT": 6,
  "INT8_C128_MEAN_TOKENS": 28500.866666666665,
  "INT8_C128_MEDIAN_TOKENS": 32768.0,
  "INT8_C128_NONFINITE": 0,
  "INT8_C128_RUNTIME_ERRORS": 0,
  "INT8_C128_SANITY_GATE": "PASS",
  "INT8_C128_SUCCESS": 6,
  "INT8_C128_TRUNCATED": 24,
  "INT8_R128_ACCURACY": 0.0,
  "INT8_R128_CORRECT": 0,
  "INT8_R128_MEAN_TOKENS": 18371.5,
  "INT8_R128_MEDIAN_TOKENS": 18371.5,
  "INT8_R128_NONFINITE": 0,
  "INT8_R128_RUNTIME_ERRORS": 0,
  "INT8_R128_SANITY_GATE": "PASS",
  "INT8_R128_SUCCESS": 1,
  "INT8_R128_TRUNCATED": 1,
  "LING_KDA_END_TO_END_INT8_PHENOTYPE": "NO",
  "MAX_NEW_TOKENS": 32768,
  "MODEL_PATH": "/data/zypan/models/Ling-3.0-tiny",
  "MODEL_REVISION": "e3a47d5b986e7141b6efd62597d598ebb392060d",
  "NEXT_RECOMMENDED_TASK": "teacher-forced / future-KL transfer diagnostic",
  "N_CONFIGS": 3,
  "N_PROBLEMS": 30,
  "N_SEEDS": 1,
  "PREFILL_QUANTIZATION_GATE": "PASS",
  "R128_SEMANTICS_GATE": "PASS",
  "R_CORRECT_C_WRONG": 0,
  "R_C_DISAGREEMENT_COUNT": 0,
  "R_WRONG_C_CORRECT": 0,
  "SEED": 20260827,
  "STATE_QUANTIZATION_SCOPE": "PREFILL_AND_DECODE",
  "TASK": "LING_KDA_AIME24_INT8_RC_END_TO_END_SMOKE_V1",
  "paired_problem_comparison_path": "/data/zypan/repos/GDN-quantization/runs/ling_kda_aime24_int8_rc_end_to_end_smoke_v1/paired_problem_comparison.json",
  "protocol_path": "/data/zypan/repos/GDN-quantization/runs/ling_kda_aime24_int8_rc_end_to_end_smoke_v1/protocol.json",
  "quantizer_semantics_path": "/data/zypan/repos/GDN-quantization/runs/ling_kda_aime24_int8_rc_end_to_end_smoke_v1/quantizer_semantics.json",
  "records": {
    "FP_STATE": [
      "/data/zypan/repos/GDN-quantization/runs/ling_kda_aime24_int8_rc_end_to_end_smoke_v1/fp_state.jsonl"
    ],
    "INT8_C128": [
      "/data/zypan/repos/GDN-quantization/runs/ling_kda_aime24_int8_rc_end_to_end_smoke_v1/int8_c128_shard000-of-002.jsonl",
      "/data/zypan/repos/GDN-quantization/runs/ling_kda_aime24_int8_rc_end_to_end_smoke_v1/int8_c128_shard001-of-002.jsonl"
    ],
    "INT8_R128": [
      "/data/zypan/repos/GDN-quantization/runs/ling_kda_aime24_int8_rc_end_to_end_smoke_v1/int8_r128.jsonl"
    ]
  },
  "stage0_gates_path": "/data/zypan/repos/GDN-quantization/runs/ling_kda_aime24_int8_rc_end_to_end_smoke_v1/stage0_gates.json"
}
```
