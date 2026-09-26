# QWEN_AIME26_LAST20_SEED1_DENSE_E2E_V1

Status: **COMPLETE**

This is a 20-question, single-generation-seed method evaluation. One question equals five percentage points; it does not establish cross-seed stability or an independent final generalization result.

## Frozen sample and protocol audit

The immutable 30-row manifest was verified by SHA-256 and original order; rows 11 through 30 are exactly `aime26_11` through `aime26_30`, each scheduled once at generation seed 1. `aime26_28/seed1` is retained. The inherited formal sampling protocol uses `max_new_tokens=81920`, temperature 1.0, top-p 0.95, top-k 20, sampling enabled, generated-token-only presence penalty 1.5, repetition penalty 1.0, thinking enabled, and the canonical manual HF runtime (not SGLang).

## Results

| ID | Condition | Status | Completed | Correct / 20 | Accuracy | Abstain | Truncated | EOS | Correct+EOS | Median tokens | Total tokens | E2E seconds | Peak MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | FP_STATE | COMPLETE | 20/20 | 17/20 | 85.0% | 2 | 2 | 18 | 17 | 29756.5 | 725261 | 53374.5 | 19865.7 |
| C1 | INT8_C128_NATIVE | COMPLETE | 20/20 | 4/20 | 20.0% | 15 | 13 | 7 | 4 | 81920.0 | 1474292 | 123479.3 | 19877.0 |
| C2 | INT8_C128_KEY_HADAMARD | COMPLETE | 20/20 | 13/20 | 65.0% | 2 | 0 | 20 | 13 | 31180.5 | 667976 | 71585.5 | 19295.9 |
| C3 | INT8_C128_DENSE_STATE_OLD | COMPLETE | 20/20 | 12/20 | 60.0% | 4 | 0 | 20 | 12 | 33941.0 | 727200 | 79134.8 | 19351.4 |
| C4 | INT8_C128_DENSE_FUNCTIONAL_OLD | COMPLETE | 20/20 | 8/20 | 40.0% | 4 | 0 | 20 | 8 | 33799.0 | 765128 | 82368.4 | 19771.8 |

## Paired comparisons

Primary C4 vs C2: rescued 1, regressed 6, net -5 question(s) (-25.0 pp); question-bootstrap 95% CI [-0.5, 0.0]; exact McNemar p=0.125.

- Rescue IDs: aime26_22

- Regression IDs: aime26_14, aime26_17, aime26_21, aime26_25, aime26_26, aime26_29

| Comparison | Status | Rescued | Regressed | Net | Rescue IDs | Regression IDs | Bootstrap 95% CI | McNemar p |
|---|---:|---:|---:|---:|---|---|---|---:|
| C4 vs C2 (primary) | COMPLETE | 1 | 6 | -5 | aime26_22 | aime26_14, aime26_17, aime26_21, aime26_25, aime26_26, aime26_29 | [-0.5, 0.0] | 0.125 |
| C3 vs C2 | COMPLETE | 3 | 4 | -1 | aime26_11, aime26_13, aime26_22 | aime26_17, aime26_26, aime26_27, aime26_29 | [-0.3, 0.2] | 1 |
| C4 vs C3 | COMPLETE | 1 | 5 | -4 | aime26_27 | aime26_11, aime26_13, aime26_14, aime26_21, aime26_25 | [-0.4, 0.0] | 0.21875 |
| C1 vs C2 | COMPLETE | 0 | 9 | -9 | none | aime26_14, aime26_17, aime26_18, aime26_21, aime26_23, aime26_24, aime26_26, aime26_27, aime26_29 | [-0.65, -0.25] | 0.00390625 |

## Required conclusions

1. The last 20 original questions and seed1 are frozen correctly: **YES**.
2. Baseline results are C0 17/20, C1 4/20, and C2 13/20 under Frozen V4.
3. Dense-State checkpoint exists, matches its recorded SHA-256, and its current integration gate is **PASS**.
4. Dense-Functional provenance is verifiable: WikiText-2 raw calibration, AIME26 unused, seed0 training, fixed minimum-validation-primary selection at step 900; current integration gate is **PASS**.
5. AIME improvement status: see completed paired table above.
6. All rescue/regression IDs are listed in the paired table; no net-only claim is used.
7. Dense step0 vs canonical C2 path: gate **PASS**, C2B required: **False**; final logit max-abs 0.0, final state max-abs 0.0.
8. Generation length, truncation, and EOS counts are reported per condition in the results table.
9. Missing/numerical/infrastructure status: none among required formal units.
10. Recommendation on later task-loss training: use the primary/secondary paired evidence cautiously; proceed only if the learned checkpoint has a positive effect that is scientifically meaningful despite the wide 20-question single-seed uncertainty.

Local error/Future-KL improvements are not treated as AIME improvements. No causal mechanism claim is made.

## Artifacts

Experiment root: `/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/QWEN_AIME26_LAST20_SEED1_DENSE_E2E_V1`

See `manifests/artifact_manifest.json` for hashes. No new rotation training was started.
