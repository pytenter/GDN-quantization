# Artifact audit

This validation reused existing artifacts only. No rotation was trained, no model or quantizer was changed, and no AIME generation was run.

## Available candidates

| Candidate | Artifact / implementation | INT8-C128 M0/M5/Decision | Held-out AIME | Primary analysis eligibility |
|---|---|---:|---:|---:|
| Identity | no rotation | yes | yes | yes |
| Hadamard | existing key-side Hadamard | yes | yes | yes |
| Dense-State | `results/rotation/qwen_dense_orthogonal_oracle_v1/training/dense_state/best_dense_state_seed0.pt` | no exact tail capture | no | no |
| Dense-Functional | `results/rotation/qwen_dense_orthogonal_oracle_v1/training/dense_functional/best_dense_functional_seed0.pt` | no exact tail capture | no | no |
| R_seed0/1/2 | `results/rotation/qwen_arbitrary_orthogonal_gate_v1/rotation_manifest.json` | no; gate had INT8 OFF | no | no |

## Evidence and exclusions

- Identity and Hadamard reuse 60 teacher-forced AIME trajectories and 3,708,902 token-condition rows from `QWEN_TEACHER_FORCED_REPLAY_MECHANISM_CAPTURE_V1`.
- Canonical V4 correctness labels come from the fixed metric-alignment audit, not from artificial labels.
- One paired trajectory (`aime26_28`, seed 2) contains late non-finite values in both quantized conditions. The full60 panel is retained; the predeclared primary sensitivity panel excludes that pair symmetrically for numerical validity, giving clean59.
- Dense checkpoints exist and pass artifact reuse checks, but their saved evaluation contains local mean errors and Future-KL only. It has `AIME26_used=false`; exact M5 p95/CVaR95 and downstream reasoning are absent.
- The random-rotation experiment is an FP equivalence gate (`int8=OFF`, `state_quantization=OFF`), so its matrices are audited but not evaluated as INT8-C128 candidates.

## Checkpoint hashes

- Dense-State: `a3597986bf9fd57e8af8dfcd13cbb8b443e1e641459444510a41ee556955074c`
- Dense-Functional: `758a5ab138d23fa5dae5666cd51039550ec662b565d9b4f4a3dcbb50ca61cc4c`
