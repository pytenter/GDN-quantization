# Experiment index

| Area | Model / condition | Canonical code | Compact result/report | Status |
|---|---|---|---|---|
| AIME26 81920 | Qwen FP, INT8_C128, Key-Hadamard | `experiments/qwen_gdn/aime26/81920/` | `results/aime26/81920/qwen_gdn/` | FROZEN |
| AIME26 81920 | Ling FP, INT8_R128, Value-Hadamard | `experiments/ling_kda/aime26/81920/` | `results/aime26/81920/ling_kda/` | FROZEN |
| Scoring | Shared Strict V4 | `experiments/shared/scoring/aime26_scorer_v4.py` | `docs/SCORER_PROTOCOL.md` | FROZEN |
| Rotation | Qwen Key-Hadamard | integrated formal runner; `experiments/qwen_gdn/rotation/key_hadamard/` | `results/aime26/81920/qwen_gdn/evidence/reference_closure/` | CANONICAL FIXED |
| Rotation | Ling Value-Hadamard | `experiments/ling_kda/rotation/value_hadamard/` | `results/rotation/ling_value_hadamard/` | CANONICAL CORRECTED V2 |
| Rotation prototypes | Qwen structured prototypes | `experiments/qwen_gdn/rotation/prototypes/` | `results/rotation/qwen_key_hadamard/prototypes/` | FAILED/CLOSED |
| Mechanism | Qwen orientation/readout/residual chains | `experiments/qwen_gdn/mechanism/` | `results/mechanism/qwen_gdn/`, `reports/mechanism/` | MIXED; SEE REPORT GATES |
| Mechanism | Ling commutator | `experiments/ling_kda/mechanism/run_ling_kda_commutator_functional_causal_v1.py` | `results/mechanism/ling_kda/ling_kda_commutator_functional_causal_v1/` | PARTIAL/NOT CLOSED |
| Mechanism | Ling decay scalarization | `experiments/ling_kda/mechanism/run_ling_kda_decay_scalarization_causal_v1.py` | `results/mechanism/ling_kda/ling_kda_decay_scalarization_causal_v1/` | PARTIAL |
| Mechanism | Ling persistent-error decomposition | `experiments/ling_kda/mechanism/run_ling_kda_persistent_error_decomposition_causal_v1.py` | `results/mechanism/ling_kda/ling_kda_persistent_error_decomposition_causal_v1/` | OBSERVATIONAL COMPLETE |
| Mechanism | Ling prefill localization/equivariance/writeback | `experiments/ling_kda/rotation/value_hadamard/` | `results/mechanism/ling_kda/kda_rotation_prefill_*` | COMPLETE WITH PER-CHAIN GATES |
| AIME26 256K | Ling FP_STATE | status only | `results/aime26/256k/ling_kda/fp_state/STATUS.md` | GENERATION COMPLETE; NOT FINAL |
| AIME26 256K | Ling INT8_R128 | status only | `results/aime26/256k/ling_kda/int8_r128/STATUS.md` | IN PROGRESS |
| AIME26 256K | Ling INT8_R128 + Value-Hadamard | status only | `results/aime26/256k/ling_kda/int8_r128_value_hadamard/STATUS.md` | IN PROGRESS |

Server and model provenance are indexed in `docs/server_inventory/` and `docs/MODEL_PROVENANCE.md`.
