# H1 prerequisite forward gate: FAIL

The two ZeRO-3 ranks agree with each other, but not with the frozen canonical single-GPU result. Source and sample SHA256 match.

| Metric | Canonical | ZeRO-3 | Conservative error/bound | Frozen threshold |
| --- | ---: | ---: | ---: | ---: |
| Teacher loss | 16.969072342 | 16.860229492 | relative 0.00641419 | 1e-05 |
| Logits max | 17.375 | 17.25 | max-abs ≥ 0.125 | 0.0625 |
| Logits L2 | 1803.78 | 1801.04 | relative L2 ≥ 0.00151592 | 0.001 |
| Layer 16 state max | 0.810676 | 0.806271 | max-abs ≥ 0.00440449 | 0.001 |

These are lower bounds from retained summaries, not reconstructed tensor-level errors. Each exceeds its pre-registered threshold. Layer 0 state SHA is exact; layer 16/30 and logits SHA differ. Exact root cause is unknown. No recurrent QDQ, backward, Adam update, H4 or later horizon was attempted after this failure.
