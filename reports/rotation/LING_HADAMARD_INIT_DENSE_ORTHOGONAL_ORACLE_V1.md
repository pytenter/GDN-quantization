# LING_HADAMARD_INIT_DENSE_ORTHOGONAL_ORACLE_V1

## Safety

- Server: `user2@10.32.128.46:20598`, 2 x RTX 4090.
- Existing experiments impacted: NO.
- AIME26 used: NO.
- Model weights changed: NO. All model parameters stayed frozen; only side-car Cayley parameters were optimized.
- Environment changed: NO.
- Third 8 x 3090 server used: NO.

## Calibration

- Dataset: the already-cached WikiText-2 raw corpus; no dataset was downloaded.
- Raw-corpus SHA256: `37f38795847da8daa776be9d7dd3b6083442dd7a27e0d14f0dedbeaa3d4884d0`.
- Split sizes: 64 TRAIN, 16 VALIDATION, 16 HELDOUT.
- Maximum sequence length: 1024 model tokens.
- Sampling: 8 fixed positions per document, all at position >=128.
- TRAIN/VALIDATION overlap: 0. TRAIN/HELDOUT overlap: 0. AIME overlap: 0.
- Qwen and Ling use the same frozen raw texts and their own tokenizers.

## Parameterization

- Rotation: `R = DeltaR after canonical H128`; the side-car executes canonical H first and the learned correction second. The canonical runtime probe receives the equivalent full row-vector rotation `H @ DeltaR`.
- `DeltaR`: FP32 Cayley transform of a skew-symmetric matrix, implemented with `torch.linalg.solve`.
- Sharing: one 128 x 128 correction per recurrent layer, shared across heads.
- Recurrent layers: 18. Independent degrees of freedom per layer: 8,128. Total trainable parameters: 146,304.
- Model trainable parameters: 0. Rotation master/matmul dtype: FP32.
- The required deployed boundary is retained: FP32 Value rotation -> BF16 intermediate -> canonical INT8_R128 QDQ.
- `PER_HEAD_ORACLE_NOT_TESTED = YES`.

## Corrected KDA semantics and gates

| Gate | Result |
|---|---:|
| `KDA_ROTATION_SEMANTICS_VERSION` | `CORRECTED_PREFILL_ENDPOINT_V2` |
| `LING_KDA_PREFILL_ENDPOINT_STATE_BASIS_GATE` | PASS |
| `REDUNDANT_PREFILL_ENDPOINT_ROTATION` | NO |
| `AUDIT_PATH_SIDE_EFFECT` | NO |
| theta=0 -> DeltaR=I | PASS |
| Dense theta=0 vs canonical Hadamard rotation/QDQ/local output | PASS |
| Held-out runtime logits at theta=0 | PASS at horizons 1/4/8/16/32/64/128 (max abs 0) |
| STE forward vs canonical INT8_R128 QDQ | PASS (max abs 0) |
| Local replay | PASS (max relative L2 0.00420709, tolerance 0.005) |
| Model weight gradient | PASS (0 trainable model parameters) |
| Orthogonality runtime | PASS |

The first version of the persistent evaluator passed only bare `DeltaR` to a probe that expects the full Value-basis rotation. That made the alleged Hadamard branch an identity branch, visible because Native and Hadamard AUC were exactly equal. Its JSON and logs were renamed `INVALID_MISSING_CANONICAL_H` and excluded. The corrected evaluator passes `H @ DeltaR`; initialization-logit parity was rerun and the complete 16-document panel was regenerated from scratch.

## LR probe

| Objective | LR | Best validation primary | Selection |
|---|---:|---:|---|
| State | 1e-3 | 6.5339512e-05 | |
| State | 3e-3 | 6.0761399e-05 | selected |
| Functional | 1e-3 | 7.1024759e-05 | |
| Functional | 3e-3 | 6.3546768e-05 | selected |

## Training

| Variant | Best step | Last step | Best validation primary | Mean / max `||DeltaR-I||F` | Max orthogonality residual |
|---|---:|---:|---:|---:|---:|
| Dense-State | 1000 | 1000 | 5.1220350e-05 | 4.40312 / 5.17784 | 1.19e-06 |
| Dense-Functional | 1000 | 1000 | 5.3097597e-05 | 13.62340 / 14.34902 | 6.56e-07 |

Both corrections are substantial rather than almost identity; Dense-Functional moves much farther. All Cayley matrices remain finite and proper with determinant numerically near +1. Checkpoint SHA256 values are recorded in `results/rotation/ling_dense_orthogonal_oracle_v1/summary.json`; raw checkpoints remain server-side and are not committed.

## Held-out local metrics

The prescribed local panel is the frozen VALIDATION split (16 documents x 8 positions = 128 paired samples).

| Method | State error | Core error | Post-norm/gate error | Out-proj error | Saturation |
|---|---:|---:|---:|---:|---:|
| Native INT8 | 1.18463e-04 | 6.08570e-05 | 6.95330e-05 | 1.01552e-04 | 0.0078974 |
| Hadamard | 1.69076e-04 | 4.08870e-05 | 4.58113e-05 | 7.30419e-05 | 0.0079736 |
| Dense-State | 7.44979e-05 | 2.84244e-05 | 3.53560e-05 | 5.59854e-05 | 0.0079743 |
| Dense-Functional | 5.52327e-05 | 2.61519e-05 | 3.32759e-05 | 5.30975e-05 | 0.0079728 |

`LING_LOCAL_HEADROOM = PRESENT`: Dense-State reduces state error by about 55.9% and Dense-Functional reduces the primary local out-proj error by about 27.3%, both relative to Hadamard.

## Persistent future-KL

Mean Future-KL on 16 independent held-out documents under corrected full `H @ DeltaR` runtime semantics:

| Horizon | Native INT8 | Hadamard | Dense-State | Dense-Functional |
|---:|---:|---:|---:|---:|
| 1 | 0.0057875 | 0.0794746 | 0.1805767 | 0.1194892 |
| 4 | 0.0057445 | 0.0131524 | 0.0281408 | 0.0458034 |
| 8 | 0.1262676 | 0.0258425 | 0.0339935 | 0.0267474 |
| 16 | 0.0425176 | 0.0585229 | 0.0417960 | 0.0577671 |
| 32 | 0.0366003 | 0.0219697 | 0.0204997 | 0.0208323 |
| 64 | 0.0403792 | 0.0298994 | 0.0203023 | 0.0242221 |
| 128 | 0.0219770 | 0.0109970 | 0.0203324 | 0.0188324 |

Normalized trapezoidal AUC: Native 0.03792503, Hadamard 0.02627500, Dense-State 0.02513393, Dense-Functional 0.02723232.

The lowest mean learned AUC is Dense-State. For paired `Hadamard AUC - Dense-State AUC`, the observed median effect is -0.00037864, the 10,000-resample paired median-bootstrap 95% CI is [-0.00297469, 0.00306924], and the median relative reduction is -3.66%. The lower mean AUC is driven by a subset of documents; the paired median document is slightly worse, which is why the strict conclusion follows the paired median rather than the unpaired mean.

`LING_PERSISTENT_STATISTICAL_VERDICT = NO_CLEAR_HEADROOM`.

Because local headroom is present while persistent headroom is not, the protocol-level classification is `LING_PERSISTENT_HEADROOM = OBJECTIVE_TRANSFER_GAP`.

## 512-token stress

One independent held-out WikiText document was run for 512 teacher-forced transitions.

| Method | Mean / max / final future-KL | Mean logit relL2 | Mean state relL2 | Top-1 match | Nonfinite |
|---|---:|---:|---:|---:|---:|
| Hadamard | 0.018776 / 0.382280 / 0.036890 | 0.051759 | 0.108731 | 93.36% | 0 |
| Dense-State | 0.016977 / 0.364302 / 0.107529 | 0.050719 | 0.091502 | 93.95% | 0 |

Dense-State is numerically stable and has slightly better mean KL/state error/top-1 rate on this one trajectory, but its final-token KL is worse. This single stress trajectory does not override the paired 16-document verdict.

## Headroom verdict

- `LING_LOCAL_HEADROOM = PRESENT`
- `LING_PERSISTENT_STATISTICAL_VERDICT = NO_CLEAR_HEADROOM`
- `LING_PERSISTENT_HEADROOM = OBJECTIVE_TRANSFER_GAP`
- `PER_HEAD_ORACLE_NOT_TESTED = YES`
- `HADAMARD_NEAR_LAYER_SHARED_OPTIMUM_LING = INCONCLUSIVE`
- Seed 0: complete. Seed 1: not triggered because the strict verdict is not CLEAR_HEADROOM.

The result does not show stable layer-shared persistent headroom under a local state/output objective. It also does not establish `NO_ORTHOGONAL_HEADROOM`, because per-head capacity was not tested and the strong local gains did not transfer reliably.

## Recommendation

The next scientifically justified step is **C: a future-aware/recurrent-aware training objective**, not Butterfly/HARP. A per-head dense oracle remains a later capacity diagnostic, but the immediate failure mode is transfer from local replay metrics to long-horizon behavior.

## Reproducibility and Git

- Branch: `exp/hadamard-init-dense-orthogonal-oracle-v1-ling`
- Baseline: `7c1196f7df1a06093a4082245a65e61a0a9af6dc`
- Implementation/evaluator tip before this report: `ef443bb`
- Checkpoints are stored under `results/rotation/ling_dense_orthogonal_oracle_v1/training/` on the Ling server, with hashes in the committed summary.
- Branch was pushed; main was not merged and no force push was used.
