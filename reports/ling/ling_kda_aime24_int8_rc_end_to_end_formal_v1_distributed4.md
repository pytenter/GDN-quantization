# LING_KDA_AIME24_INT8_RC_END_TO_END_FORMAL_V1_DISTRIBUTED4

## Summary
- FORMAL_STATUS: `COMPLETE`
- COMPLETED_GENERATIONS: `90/90`
- FP_STATE: 17/30 (0.5667)
- INT8_R128: 14/30 (0.4667)
- INT8_C128: 10/30 (0.3333)
- OVERALL: 41/90 (0.4556)
- RUNTIME_ERRORS: `0`
- NONFINITE: `0`
- TRUNCATED: `53`

## Run Layout
- Local completed shards: `2/4`, `3/4`, `4/8`, `5/8`.
- Remote completed shards: `0/8`, `1/8`, plus helper `29/30` for `problem_id=89`.
- Final aggregation deduplicates by `(problem_id, method)`.

## Method Health

| method | correct | attempted | accuracy | truncated | runtime_errors | nonfinite | mean_tokens | median_tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FP_STATE | 17 | 30 | 0.5667 | 15 | 0 | 0 | 23472.2 | 30860.0 |
| INT8_R128 | 14 | 30 | 0.4667 | 19 | 0 | 0 | 24902.8 | 32768.0 |
| INT8_C128 | 10 | 30 | 0.3333 | 19 | 0 | 0 | 23765.2 | 32768.0 |

## Paired Problem Comparison

| problem_id | FP | R128 | C128 | FP ans | R128 ans | C128 ans | FP tok | R128 tok | C128 tok | FP trunc | R128 trunc | C128 trunc |
|---|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 60 | 1 | 1 | 1 | 204 | 204 | 204 | 4018 | 1935 | 2824 | 0 | 0 | 0 |
| 61 | 0 | 0 | 0 | likely correct | 4 | 2 | 32768 | 32768 | 32768 | 1 | 1 | 1 |
| 62 | 0 | 0 | 0 | 4 | 7 | 6 | 32768 | 32768 | 32768 | 1 | 1 | 1 |
| 63 | 0 | 0 | 0 | 2 | 2 | 1 | 32768 | 32768 | 32768 | 1 | 1 | 1 |
| 64 | 1 | 1 | 0 | 110 | 110 | 4 | 27518 | 32768 | 32768 | 0 | 1 | 1 |
| 65 | 0 | 1 | 0 | 2 | 104 | 2 | 32768 | 21389 | 32768 | 1 | 0 | 1 |
| 66 | 0 | 0 | 0 | 27 | exactly the maximum of the half-diagonals? No! If we can move them freely, we can place all boxes inside the sphere | the maximum of half-diagonal | 32768 | 32768 | 32768 | 1 | 1 | 1 |
| 67 | 1 | 1 | 1 | 25 | 25 | 25 | 3466 | 1577 | 4464 | 0 | 0 | 0 |
| 68 | 1 | 1 | 1 | 809 | 809 | 809 | 16262 | 10234 | 3979 | 0 | 0 | 0 |
| 69 | 1 | 1 | 1 | 116 | 116 | 116 | 8838 | 6224 | 2573 | 0 | 0 | 0 |
| 70 | 1 | 0 | 0 | 104 | 33 | 33 | 32768 | 32768 | 32768 | 1 | 1 | 1 |
| 71 | 1 | 1 | 1 | 294** | 294 | 294 | 20780 | 15402 | 12008 | 0 | 0 | 0 |
| 72 | 1 | 1 | 1 | 540 | 540 | 540 | 2558 | 1912 | 5567 | 0 | 0 | 0 |
| 73 | 0 | 0 | 0 | 0 | 1 | 1 | 32768 | 32768 | 32768 | 1 | 1 | 1 |
| 74 | 1 | 0 | 0 | is 480 | the infimum or the minimum depending on whether it's achieved | the infimum | 32768 | 32768 | 32768 | 1 | 1 | 1 |
| 75 | 1 | 0 | 1 | 73 | |all 4|, and the question says 'Find the number of residents who own all four of these things | 73 | 17125 | 32768 | 3275 | 0 | 1 | 0 |
| 76 | 1 | 0 | 0 | 468 | 0 | 78(1+√5)? Let's verify if there is another way to derive it | 22792 | 32768 | 32768 | 0 | 1 | 1 |
| 77 | 1 | 1 | 0 | 601 | 601 | 201 solutions | 28952 | 32768 | 32768 | 0 | 1 | 1 |
| 78 | 0 | 0 | 0 | that there is no such point | 0 | 2 | 32768 | 32768 | 32768 | 1 | 1 | 1 |
| 79 | 1 | 1 | 1 | 321 | 321 | 321 | 8417 | 21459 | 11740 | 0 | 0 | 0 |
| 80 | 0 | 0 | 0 | 105 | 6 | 1 b-eautiful integer (n=25, a=4, c=1) | 32768 | 32768 | 32768 | 1 | 1 | 1 |
| 81 | 0 | 0 | 0 | 240 | 3 | not the same as the number of inscribed rectangles (which is C(6,2)=15) because an inscribed rectangle has vertices at vertices of the polygon, not necessarily all 30 intersection points of the lines | 32768 | 32768 | 32768 | 1 | 1 | 1 |
| 82 | 1 | 1 | 0 | 236 | 236 | 2 | 27784 | 30373 | 32768 | 0 | 0 | 1 |
| 83 | 1 | 1 | 1 | 45 | 45 | 45 | 12977 | 32768 | 18949 | 0 | 1 | 0 |
| 84 | 1 | 1 | 1 | 33 | 33 | 33 | 3876 | 5037 | 2986 | 0 | 0 | 0 |
| 85 | 0 | 0 | 0 | 2 | 120 | 6 | 32768 | 32768 | 32768 | 1 | 1 | 1 |
| 86 | 1 | 1 | 0 | 55 | 55 | 3 | 7283 | 8949 | 32768 | 0 | 0 | 1 |
| 87 | 0 | 0 | 0 | 2 | 3 | 629 | 32768 | 32768 | 21998 | 1 | 1 | 0 |
| 88 | 0 | 0 | 0 | 0 | 3 | 3 | 32768 | 32768 | 32768 | 1 | 1 | 1 |
| 89 | 0 | 0 | 0 | 1 | 2 | 5 | 32768 | 32768 | 32768 | 1 | 1 | 1 |

## Artifacts
- Run summary: `runs/ling_kda_aime24_int8_rc_end_to_end_formal_v1_distributed4/final_summary.json`
- Result summary: `results/ling/ling_kda_aime24_int8_rc_end_to_end_formal_v1_distributed4_final_summary.json`
- Paired comparison: `runs/ling_kda_aime24_int8_rc_end_to_end_formal_v1_distributed4/paired_problem_comparison.json`
- Raw shard JSONL files: `runs/ling_kda_aime24_int8_rc_end_to_end_formal_v1_distributed4`

## Deduplication
Keep one record per (problem_id, method), preferring the planned resumed distributed shards: num_shards=8 shard_id in {0,1,4,5}, then completed num_shards=4 shard_id in {2,3}, then helper num_shards=30 shard_id=problem_index. One old duplicate (64, FP_STATE) was discarded.

Duplicate keys:

```json
{
  "64:FP_STATE": [
    {
      "correct": true,
      "dedup_score": [
        0,
        0,
        2
      ],
      "file": "runs/ling_kda_aime24_int8_rc_end_to_end_formal_v1_distributed4/fp_state_shard00.jsonl",
      "line": 2,
      "num_shards": 4,
      "shard_id": 0,
      "truncated": true
    },
    {
      "correct": true,
      "dedup_score": [
        100,
        1,
        1
      ],
      "file": "runs/ling_kda_aime24_int8_rc_end_to_end_formal_v1_distributed4/fp_state_shard04.jsonl",
      "line": 1,
      "num_shards": 8,
      "shard_id": 4,
      "truncated": false
    }
  ]
}
```

