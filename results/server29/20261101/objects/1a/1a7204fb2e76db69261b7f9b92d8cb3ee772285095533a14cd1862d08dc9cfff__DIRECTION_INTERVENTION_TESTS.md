# Direction Intervention Tests

## Preserved engineering attempts

- attempt01: failed before scientific measurement because inference tensors were used in an autograd-tracked matrix multiply.
- attempt02: client SSH session disconnected; only STARTED existed, no scientific or canonical output.
- attempt03: failed before scientific measurement because inference-cache writeback occurred outside inference mode.
- attempt04: failed before scientific measurement on an explicit Float/Double operator-test dtype mismatch.
- attempt05: complete mechanical PASS after append-only fixes.

No failed attempt was deleted or retroactively relabeled.

## Mechanical attempt05

- Fork parity: PASS.
- Actual post-QDQ norm gates: PASS.
- Post-QDQ signed direction gates: PASS.
- Visible/low sensitivity separation: PASS (mechanical micro ratio about 62,452x).
- Five branches used the same base and completed real every-token L6 QDQ continuation with finite outputs.
- Runtime: 29.63 seconds / 0.00823 GPU-hours.

The micro uses checkpoint 8 and four scored tokens. Its Future-KL values are explicitly non-scientific and cannot support the mechanism hypothesis.
## Formal intervention results

All formal engineering gates passed on the frozen 24K checkpoint.

| Document | Role | D | D(+) | D(-) | Norm gate | Direction gate | Sensitivity gate | Fork gate |
|---|---|---:|---:|---:|---|---|---|---|
| pg19-test-37702 | development | 70.268568 | 90.849051 | 49.688085 | PASS | PASS | PASS | PASS |
| pg19-test-28988 | validation | 141.175802 | 96.593700 | 185.757905 | PASS | PASS | PASS | PASS |
| pg19-test-26493 | validation | 52.382955 | 73.885682 | 30.880227 | PASS | PASS | PASS | PASS |

Both sign-specific contrasts were positive on development and on both prospective validation documents. No outcome-dependent branch, threshold, direction, checkpoint, or document choice was made.
