# Final report: QWEN_GDN_RECURRENT_C5_C6_H64_FORMAL_V1

## Environment

Training used one `NVIDIA vGPU-48GB` with 47.382 GiB visible VRAM. No distributed framework was used.

## Protocol

The prospective protocol was frozen before training. Both conditions used H64, 1024-token documents, eight canonical captures per document, real recurrent INT8-C128 writeback, H64 graph detachment, and exactly one optimizer step per document. C5 and C6 independently initialized from theta=0 / Hadamard.

## Results

| Condition | Status | Updates | Exposures | Validation primary | Selected checkpoint | Peak reserved GiB | Memory creep | NaN/Inf | Retry | Deployment |
|---|---|---:|---:|---:|---|---:|---|---|---:|---|
| C5 | COMPLETE | 125 | 1000 | 0.00791618513176 | `checkpoints/C5/candidate_update_063_exposure_0504.pt` | 31.244 | NO | NO | 0 | YES |
| C6 | COMPLETE | 125 | 1000 | 0.0490164864968 | `checkpoints/C6/candidate_update_125_exposure_1000.pt` | 31.244 | NO | NO | 0 | YES |

## Comparison

Only training, held-out local/recurrent validation, and frozen-loader diagnostics are compared here. No downstream benchmark was run and no AIME conclusion is made.

C5 state-primary minimum: `0.00791618513176`. C6 functional-primary minimum: `0.0490164864968`. These are different preregistered metrics and are reported without treating their raw magnitudes as a direct superiority test.

## Deployment

`C5_DEPLOYMENT_ARTIFACT_READY=YES`

`C6_DEPLOYMENT_ARTIFACT_READY=YES`

Both deployment packages require directly loading the frozen FP32 matrices; inference-host reconstruction from theta is non-canonical.

## Downstream

`AIME_EVALUATION=NOT_STARTED`
