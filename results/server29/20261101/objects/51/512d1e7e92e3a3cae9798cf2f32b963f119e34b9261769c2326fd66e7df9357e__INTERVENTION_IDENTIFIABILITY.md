# Intervention identifiability

Baseline P uses `theta_t^P = qparam(u_t^P)`. Treatment F uses `theta_t^clean = qparam(u_t^FP)`, while recurrence and integer-code selection consume F's own perturbed state `u_t^F`. Because the parent persistent path differs from the paired clean FP path after quantized writeback, `u_t^P` is generally not `u_t^FP`; the qparam source is therefore genuinely intervened upon. Unlike the rejected Path R, F does not share both state and qparams with P, so equality is not closed by induction.

`INTERVENTION_IDENTIFIABILITY = PASS`
