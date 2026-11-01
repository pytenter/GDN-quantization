```text
EXPERIMENT = QWEN_GDN_A800_CLEANROOM_C5R_C6R_C7_MATCHED_V1
CLOSURE = QWEN_GDN_A800_POSTTRAIN_SHARED16_REPRODUCTION_CLOSURE_V1
C5R_TRAINING = COMPLETE / PASS
C6R_TRAINING = COMPLETE / PASS
C7_TRAINING = COMPLETE / PASS
TRAINING_RERUN = NO
HASH_FAILURE_AUDIT = PASS
HASH_CONSTANT_RECORD_ERROR = SUPPORTED
ARTIFACT_MUTATION_SUPPORTED = NO
POSTTRAIN_RUNNER_SCIENTIFIC_PATH_IDENTITY_GATE = PASS
POSTTRAIN_EFFECTIVE_PROVENANCE_GATE = PASS
POSTTRAIN_SHARED16_GATE = PASS
QWEN_FINAL_HIDDEN_INDEXING_GATE = PASS
C5R_REPRODUCTION = INCONCLUSIVE
C5R_AIME = SKIP
C6R_FROZEN_ROTATION_EXPORT = PASS
C7_FROZEN_ROTATION_EXPORT = PASS
C6R_EXPORT_PARITY_GATE = PASS
C7_EXPORT_PARITY_GATE = PASS
TRAINING_ARTIFACT_BYTE_IDENTITY = PASS
3090_DEPLOYMENT_STARTED = NO
C5R_AIME_STARTED = NO
C6R_AIME_STARTED = NO
C7_AIME_STARTED = NO
NEW_AIME_GENERATION = 0
PUSH_PERFORMED = NO
```

# Shared16 post-training closure

All rows below are direct same-runtime trusted A800 clean-teacher measurements, not AIME. Frozen seven-condition panel preserved.

| Condition | State | Functional | Final Hidden |
|---|---:|---:|---:|
| Hadamard | 0.00838398559245 | 0.0478101125227 | 0.0167575784972 |
| historical_C5_u063 | 0.00808766196315 | 0.0505878723088 | 0.0179230450813 |
| historical_C5_u113 | 0.00836844233527 | 0.0525381964035 | 0.0179863769554 |
| historical_C6_u125 | 0.00825962756426 | 0.0499474076569 | 0.0179860840544 |
| C5R | 0.00783597432383 | 0.0487577369313 | 0.0188712133659 |
| C6R | 0.00814104696838 | 0.0489342293695 | 0.0173984808471 |
| C7 | 0.00856255854706 | 0.0508051724537 | 0.0167887384387 |

## Historical background (different runtime, not direct comparison)

Historical C5 selected u63/e504: State 0.00791618513176, Functional 0.0495092943711. Old Final Hidden was not recorded.

## C5R reproduction

C5R selected u113/e904; historical u63/e504. Layer A protocol reproduction PASS. Layers B/C supply full paired objective and document evidence; Layer D is descriptive only. Matrix distance alone did not determine verdict.

No prospective mechanical criterion/equivalence envelope for objective direction/magnitude or 16-document association was frozen. Same-runtime evidence is complete; do not invent post-hoc 2%, 5% or correlation thresholds.

| Metric | Historical C5 vs H | C5R vs H | Sign agreement /16 | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|
| state | -3.534401% | -6.536405% | 10 | 0.6361834341928431 | 0.5205882352941177 |
| functional | +5.809984% | +1.982059% | 11 | 0.3974134778356688 | 0.5147058823529411 |
| final_hidden | +6.954863% | +12.613009% | 10 | 0.48532482803603266 | 0.3588235294117647 |

All 16 documents retained. Per-capture/per-document CSVs contain 896/112 rows; all values finite. Largest paired differences, exact document IDs and complete signed distributions are recorded in c5r_reproduction_analysis.json. No outliers removed and no thresholds invented.

{
  "C5R_vs_historical_C5_u063": {
    "relative_Frobenius_min": 0.4078328408614075,
    "relative_Frobenius_median": 0.6431351164037069,
    "relative_Frobenius_max": 0.76667519157009,
    "descriptive_only": true
  },
  "C5R_vs_historical_C5_u113": {
    "relative_Frobenius_min": 0.4242809579539696,
    "relative_Frobenius_median": 0.679701618830899,
    "relative_Frobenius_max": 0.796934137292844,
    "descriptive_only": true
  },
  "C6R_vs_historical_C6_u125": {
    "relative_Frobenius_min": 0.147366232188737,
    "relative_Frobenius_median": 0.2546483027697102,
    "relative_Frobenius_max": 0.33629434692597665,
    "descriptive_only": true
  }
}

## C6R / C7 and exports

C6R is the primary matched control, C7 the primary new objective. Compare their three Shared16 fidelities only; no reasoning superiority claim. The selected existing 24x128x128 FP32 exports passed per-layer SHA, frozen orthogonality threshold and exact A800 checkpoint-loaded matrix parity. No export rewritten, no selection changed.

## Hash failure and scientific path

All three archived EXPECTED constants were malformed 65-character hex strings. GNU sha256sum/Python hashlib agree with independent 3090 copies and preexisting identity records. Old failing source and logs remain byte-identical. Append-only amendment fixes EXPECTED resolution only. The original archived main, payload, model loader, teacher cache, evaluator, QDQ, writeback and metrics are used unchanged; audit output names differ to preserve failure records.

Protected inventory: 500 files / 52000371751 bytes, exact hash/size/mtime revalidated. Teachers were not rebuilt.

## Provenance / scope / commit

Branch: exp/qwen-gdn-a800-cleanroom-c5r-c6r-c7-matched-v1. Pre-task HEAD: ad1b9fc19bd5fd5cc9014b7473f31edce22976a3. One local closure commit is created only after schema/hash/diff review; its SHA is in logs/posttrain_closure_commit.json (excluded from pre-commit hash manifest to avoid self-reference). No push.

Host/GPU/driver: autodl-container-027f4cad3d-47d7863c / NVIDIA A800 80GB PCIe / GPU-8430c7aa-cf91-9075-2959-ea3a04760eda / 580.82.07. Python 3.10.18 (main, Jun  5 2025, 13:14:17) [GCC 11.2.0]; Torch 2.5.1+cu121, CUDA12.1. Fresh direct CPU checkpoint and all 429 full GPU state items verified before any Shared16 computation.

Reproduction commands: scripts/posttrain_closure_pipeline.sh (one-shot; existing evidence blocks repetition). Sources, audits and raw compact evidence are indexed by manifests/posttrain_closure_artifact_sha256.txt.

STOP: no training/backward/optimizer, no transfer of new R to3090, no deployment, no AIME, no scoring. Even PASS does not authorize endpoints.
