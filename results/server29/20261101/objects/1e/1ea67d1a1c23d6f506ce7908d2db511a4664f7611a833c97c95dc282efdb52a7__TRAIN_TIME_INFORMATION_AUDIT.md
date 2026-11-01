# Train-time information audit

This audit was frozen with the protocol and is updated only with implementation evidence.

| Quantity | Information class | Train-feasible? | Long-horizon dependency |
|---|---|---:|---:|
| Current recurrent state and token query | A: current token | Yes | None |
| P1 | A | Yes | None |
| Previous 128 FP-teacher queries | B: finite history | Yes, with a ring buffer or teacher pass | None |
| P2_W128 | A+B | Yes | None |
| H32/H128 teacher-forced tokens and FP queries | C: short minibatch | Yes | None |
| P3_H32/H128 | A+B+C | Forward-feasible; backward readiness separate | None |
| B0/B1 | A/C | Yes | None |
| B2 short KL | C plus FP teacher logits | Yes | None |
| Historical JPROP/POR | D: 32K trajectory | No; retrospective only | Required |
| Historical Future-KL | D: 32K trajectory | Label only | Required |

The evaluator computes a new early-prefix FP teacher on the current runtime. Archived 32K FP chunks are not used by P1/P2/P3. A canonical scalar norm is never substituted for an error tensor. KDA state layout is `[layer, head, key, value]`; the query contracts the key axis, leaving a value-vector readout. Physical coordinates are restored after rotated INT8 QDQ before state comparison.

P1 and P2 are local objectives. P3 requires unrolling 32 or 128 real recurrent steps. Exact gradients through rounding are zero/discontinuous and would require an explicit STE or differentiable quantizer surrogate; no optimizer step or checkpoint creation is authorized here. Prediction and trainability are therefore evaluated separately.
