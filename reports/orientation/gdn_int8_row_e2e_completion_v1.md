# GDN INT8-row E2E Completion V1

## OBSERVATION
- Protocol gates: SMOKE=PASS, QUANTIZER=PASS.
- `max_new_tokens=32768`; no `max_length` cap is used by the harness.
- Truncation means `generated_new_tokens == 32768` and no configured EOS/stop token was reached.

## RESULTS
```json
{
  "FP_STATE": {
    "EOS_count": 19,
    "N": 30,
    "accuracy": 0.7666666666666667,
    "completed_count": 19,
    "correct": 23,
    "exactly_32768_count": 11,
    "exception_count": 0,
    "max_generation_length": 32768,
    "mean_generation_length": 17201.466666666667,
    "median_generation_length": 13494.0,
    "nonfinite_count": 0,
    "nonfinite_rate": 0.0,
    "p90_generation_length": 32768.0,
    "p95_generation_length": 32768.0,
    "truncation_count": 11,
    "truncation_rate": 0.36666666666666664
  },
  "INT8-column": {
    "EOS_count": 17,
    "N": 30,
    "accuracy": 0.6,
    "completed_count": 17,
    "correct": 18,
    "exactly_32768_count": 13,
    "exception_count": 0,
    "max_generation_length": 32768,
    "mean_generation_length": 22671.2,
    "median_generation_length": 28715.5,
    "nonfinite_count": 0,
    "nonfinite_rate": 0.0,
    "p90_generation_length": 32768.0,
    "p95_generation_length": 32768.0,
    "truncation_count": 13,
    "truncation_rate": 0.43333333333333335
  },
  "INT8-row": {
    "EOS_count": 4,
    "N": 30,
    "accuracy": 0.23333333333333334,
    "completed_count": 4,
    "correct": 7,
    "exactly_32768_count": 26,
    "exception_count": 0,
    "max_generation_length": 32768,
    "mean_generation_length": 32202.666666666668,
    "median_generation_length": 32768.0,
    "nonfinite_count": 0,
    "nonfinite_rate": 0.0,
    "p90_generation_length": 32768.0,
    "p95_generation_length": 32768.0,
    "truncation_count": 26,
    "truncation_rate": 0.8666666666666667
  }
}
```

## HYPOTHESIS
- INT8-row may be a finite-but-pathological orientation under every-token recurrent feedback quantization.

## SUPPORTED CONCLUSION
- INT8-row remains strongly degraded relative to FP_STATE and INT8-column on the fixed MATH-500 N=30 subset.

## NEGATIVE RESULT
- This P0 output alone does not establish causality or a deployable method.

## PROPOSED FOLLOW-UP
- If P0 is complete and row degradation remains, run P1-A state-change mechanism on selected row-pathological and control prompts.
