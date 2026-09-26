# QWEN_HADAMARD_INIT_DENSE_ORTHOGONAL_ORACLE_V1

## Safety

- Server: `zypan@202.38.247.55`, 4 x RTX 3090.
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

- Rotation: `R = DeltaR after canonical H128`; the implementation executes canonical H first and the learned correction second and never pre-composes a dense H matrix.
- `DeltaR`: FP32 Cayley transform of a skew-symmetric matrix, implemented with `torch.linalg.solve`.
- Sharing: one 128 x 128 correction per recurrent layer, shared across heads.
- Recurrent layers: 24. Independent degrees of freedom per layer: 8,128. Total trainable parameters: 195,072.
- Model trainable parameters: 0. Master/matmul dtype for the correction: FP32.
- `PER_HEAD_ORACLE_NOT_TESTED = YES`.

## Gates

| Gate | Result |
|---|---:|
| theta=0 -> DeltaR=I | PASS |
| Dense theta=0 vs canonical Hadamard rotation/QDQ/local output | PASS (max abs 0) |
| STE forward vs canonical INT8_C128 QDQ | PASS (max abs 0) |
| Local replay | PASS (max relative L2 0.00220608, tolerance 0.005) |
| Model weight gradient | PASS (0 trainable model parameters) |
| Orthogonality runtime | PASS |
| Held-out runtime logits at theta=0 | PASS at horizons 1/4/8/16/32/64/128 (max abs 0) |

## LR probe

| Objective | LR | Best validation primary | Selection |
|---|---:|---:|---|
| State | 1e-3 | 4.2823451e-05 | |
| State | 3e-3 | 4.2823424e-05 | selected |
| Functional | 1e-3 | 2.5437345e-04 | selected |
| Functional | 3e-3 | 2.7117267e-04 | |

## Training

| Variant | Best step | Last step | Best validation primary | Mean / max `||DeltaR-I||F` | Max orthogonality residual |
|---|---:|---:|---:|---:|---:|
| Dense-State | 100 | 600 (early stop) | 4.2823313e-05 | 0.000489 / 0.000874 | 2.77e-11 |
| Dense-Functional | 900 | 1000 | 2.3939875e-04 | 5.44157 / 6.08392 | 1.07e-06 |

Dense-State learned an almost-identity correction. Dense-Functional learned a substantial but still proper/finite rotation. Checkpoint SHA256 values are recorded in `results/rotation/qwen_dense_orthogonal_oracle_v1/summary.json`; raw checkpoints remain server-side and are not committed.

## Held-out local metrics

The protocol names this section “held-out local metrics”, but its prescribed paired panel is the frozen VALIDATION split (16 documents x 8 positions = 128 samples).

| Method | State error | Core error | Post-norm/gate error | Out-proj error | Saturation |
|---|---:|---:|---:|---:|---:|
| Native INT8 | 1.62722e-04 | 1.17760e-04 | 2.06525e-04 | 1.91808e-04 | 0.0079187 |
| Hadamard | 4.28540e-05 | 2.70493e-05 | 2.93522e-04 | 2.81956e-04 | 0.0080484 |
| Dense-State | 4.28537e-05 | 2.69717e-05 | 2.93399e-04 | 2.81972e-04 | 0.0080484 |
| Dense-Functional | 4.09282e-05 | 2.74142e-05 | 2.49270e-04 | 2.39399e-04 | 0.0080431 |

`QWEN_LOCAL_HEADROOM = PRESENT`: Dense-Functional reduces the primary local out-proj error by about 15.1% relative to Hadamard. Dense-State is effectively tied with Hadamard.

## Persistent future-KL

Mean Future-KL on 16 independent held-out documents:

| Horizon | Native INT8 | Hadamard | Dense-State | Dense-Functional |
|---:|---:|---:|---:|---:|
| 1 | 0.0003530 | 0.0007245 | 0.0007898 | 0.0005817 |
| 4 | 0.0006007 | 0.0007281 | 0.0006263 | 0.0005228 |
| 8 | 0.0006248 | 0.0007430 | 0.0006949 | 0.0008050 |
| 16 | 0.0006551 | 0.0025114 | 0.0015629 | 0.0020974 |
| 32 | 0.0005116 | 0.0026019 | 0.0022908 | 0.0012382 |
| 64 | 0.0011774 | 0.0032424 | 0.0031195 | 0.0035848 |
| 128 | 0.0019004 | 0.0062299 | 0.0048306 | 0.0035806 |

Normalized trapezoidal AUC: Native 0.00113269, Hadamard 0.00358796, Dense-State 0.00303620, Dense-Functional 0.00274858.

The best learned variant is Dense-Functional. For paired `Hadamard AUC - Dense-Functional AUC`, the observed median effect is 0.00028912, the 10,000-resample paired median-bootstrap 95% CI is [-0.00062934, 0.00073927], and the median relative reduction is 11.45%.

`QWEN_PERSISTENT_HEADROOM = PROMISING_HEADROOM`: the point estimate and relative reduction pass the directional/10% tests, but the confidence interval crosses zero. This is not `CLEAR_HEADROOM`.

Native INT8 also has lower AUC than all rotated conditions on this panel. This experiment answers incremental dense-over-Hadamard headroom and does not claim that the rotated family beats Native INT8 for Qwen future-KL.

## 512-token stress

One independent held-out WikiText document was run for 512 teacher-forced transitions.

| Method | Mean / max / final future-KL | Mean logit relL2 | Mean state relL2 | Top-1 match | Nonfinite |
|---|---:|---:|---:|---:|---:|
| Hadamard | 0.016760 / 4.344824 / 0.002002 | 0.067263 | 0.072910 | 93.36% | 0 |
| Dense-Functional | 0.009170 / 1.228717 / 0.002038 | 0.061029 | 0.070549 | 96.09% | 0 |

The learned matrix is numerically stable through 512 transitions and reduces the average/maximum KL spike and top-1 divergence count on this stress trajectory. The final-token KL is essentially tied and does not replace the multi-document bootstrap verdict.

## Headroom verdict

- `QWEN_LOCAL_HEADROOM = PRESENT`
- `QWEN_PERSISTENT_HEADROOM = PROMISING_HEADROOM`
- `PER_HEAD_ORACLE_NOT_TESTED = YES`
- `HADAMARD_NEAR_LAYER_SHARED_OPTIMUM_QWEN = INCONCLUSIVE`
- Seed 0: complete. Seed 1: not triggered because the strict verdict is not CLEAR_HEADROOM.

There is evidence that a layer-shared dense functional correction can improve Hadamard locally and may transfer to recurrent history, but the held-out uncertainty is too large for a clear claim. Qwen alone therefore does not authorize Butterfly/HARP work.

## Reproducibility and Git

- Branch: `exp/hadamard-init-dense-orthogonal-oracle-v1-qwen`
- Baseline: `a0ff66857f91c30b829ecb7603ddae013644b2d1`
- Implementation tip before this report: `0b6d598`
- Checkpoints are stored under `results/rotation/qwen_dense_orthogonal_oracle_v1/training/` on the Qwen server, with hashes in the committed summary.
- Branch was pushed; main was not merged and no force push was used.
