# Result Summary

| Experiment | Question | Units | Main Metric | Result | Evidence Level | Interpretation |
|---|---|---:|---|---|---|---|
| Repeated Accumulation Formal | Does repeated R128 exposure accumulate trajectory error? | 18 | cadence KL/state AUC | SUPPORTED | FORMAL | Recurrent exposure is a core causal pathway. |
| V-Space Scale Contamination | Is R128 source error representation dependent? | 9 pilot plus screens | residual/KL rescue | V-Hadamard rescue 9/9; median KL rescue fraction about 0.9655 | SUPPORTED | Cross-Value scale contamination is a strong source-side explanation. |
| Source-to-Temporal Formal Bridge | Does source rescue attenuate repeated accumulation? | 18 | residual, STATE_AUC, KL_AUC | 18/18 improves; SOURCE_TO_TEMPORAL_BRIDGE SUPPORTED | FORMAL | Source-side representation rescue propagates to long-horizon behavior. |
| Natural Residual Norm-Swap | Is residual magnitude causal and sufficient? | 9 | KL_AUC | REAL_R > REAL_C 9/9; R shrink improves 9/9; same-norm R structure worse 9/9 | CAUSAL PILOT | Magnitude matters, but structure remains. |
| Frozen Observability | Does persistence explain same-norm harm? | 9 | J_state/J_key/full KL | J_state R/C 0.6665, J_key R/C 2.4732, full KL R/C 2.0473 | SUPPORTED PATH SIGNAL | Persistence is not relevance; future-key interaction is implicated. |
| Future-Key Causal Construction | Is J_key independently causal? | 9 | intervention cleanliness and KL | CONSTRUCTION_NOT_CLEAN_ENOUGH | NEGATIVE | J_key cannot be claimed as standalone causal scalar. |
| K-Space Rotation Audit | Is future-key signal just K coordinate basis? | screen | rotation counterfactual | ROTATION_MANIFOLD_DOES_NOT_CLEANLY_DECOUPLE_KEY | NEGATIVE | K-space rotation does not cleanly explain the path signal. |
| Native Core Replay Audit | Are replay semantics correct? | regression | replay relative error | next_state = 0, core_output = 0 after dtype fix | IMPLEMENTATION SUPPORTED | Later causal replays rest on corrected native semantics. |
| Local Operator Coupling / U Causal V2 | Does local U transduction explain downstream harm? | 9 | U/E, consumed energy, KL rescue | R U/E > C 8/9; R_STATE_ONLY improves KL only 3/9 | MIXED: SUPPORTED diagnostic, NEGATIVE causal | Local coupling is real but not the downstream causal mechanism. |
| Downstream Feedback Propagation | Does single-head downstream feedback explain harm? | 9 | hidden/operator/state drift | R>C approximately 0/9 across downstream drift metrics | NEGATIVE | Single-head downstream feedback is not supported. |
| FAST Multi-Head Composition | Does single-layer multi-head scope amplify R/C gap? | 9 | RC_GAP KL_AUC | scope/head-wise growth PARTIAL; primary interpretation INCONCLUSIVE | INCONCLUSIVE | No clean single-layer scope-amplification explanation. |
| Headwise S8 Subset Robustness | Is head-wise S8 R-bias subset robust? | 18 | median subset delta | PARTIAL; distributed weak R-bias CANDIDATE; heterogeneity STRONG | CANDIDATE / INCONCLUSIVE | Head identity likely matters. |
