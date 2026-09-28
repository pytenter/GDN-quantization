# Ling 256K Protocol Freeze Report

Date: 2026-09-28

Repository: `pytenter/GDN-quantization`

Primary protocol-freeze commit: `ecc0f040267267717b68f52b29ec7ed2182c51b8`

Primary commit subject: `docs: freeze Ling KDA 256K canonical evaluation protocol`

## 1. Modified documentation

The primary protocol-freeze commit modified only documentation:

- `README.md`
- `docs/EXPERIMENT_INDEX.md`
- `docs/HANDOFF_2026-09-26.md`
- `docs/HANDOFF_ARTIFACT_MATRIX.md`
- `docs/PROTOCOL.md`
- `docs/RESEARCH_STATUS.md`
- `docs/RESEARCH_STATUS_2026-09-26.md`
- `docs/SCORER_PROTOCOL.md`
- `docs/LING_CANONICAL_256K_PROTOCOL.md` (new)

This report commit also clarifies the historical-snapshot status of:

- `docs/server_inventory/SERVER_MAP.md`
- `docs/server_inventory/ling4090.md`
- `LING_256K_PROTOCOL_FREEZE_REPORT.md` (new)

No training code, evaluation code, checkpoint, experiment output, configuration, manifest, or frozen artifact was changed. No generation or scoring job was run.

## 2. Reason for the change

Current top-level documentation still described Ling's fixed 81,920-token AIME26 run as canonical. That wording conflicted with the frozen `LING_L6_L7_CANONICAL_60_CONFIRMATORY_V1` evaluation, which uses a 262,144-token context/server budget and a dynamic per-sample generation budget.

The documentation now makes the protocol boundary explicit:

- 81,920 fixed-budget Ling results are immutable historical reference and regression evidence.
- Current canonical Ling/KDA evaluation uses the 256K dynamic-budget long-horizon protocol.
- Outputs from the two protocols must not be merged; cross-protocol comparisons must be explicitly labeled historical.

## 3. Current canonical Ling/KDA protocol

### Context and generation budget

```ini
context_length = 262144
server_max_total_tokens = 262144
safety_margin = 512
max_new_tokens = 262144 - prompt_tokens - 512
```

### Decoding

```ini
thinking = true
do_sample = true
temperature = 1.0
top_p = 0.95
top_k = 20
repetition_penalty = 1.0
sampling_seed = 1 or 2
stop = null
```

### Runtime

```text
SGLang 0.5.19
Torch 2.9.1+cu128
Triton 3.5.1
FLA 0.5.2
dtype=bfloat16
radix_cache=false
cuda_graph=false
tp_size=1
sampling_backend=pytorch
```

The machine-readable evidence was audited from branch `exp/ling-l6-l7-canonical-60-confirmatory-v1` at commit `69b23bd999dc23a9c852497b98df0f56abdbdff0`, principally `experiments/LING_L6_L7_CANONICAL_60_CONFIRMATORY_V1/configs/frozen_eval_config.json` and `configs/runtime_identity.json`.

## 4. Remaining 81,920 references

Repository searches were run for `81920`, `81,920`, `256K`, and `262144` across `README.md`, `docs/`, `experiments/`, and `reports/`.

### Historical reference (allowed)

- `results/aime26/81920/**` and `experiments/*/aime26/81920/**`: immutable historical code/result provenance.
- `README.md`, `docs/PROTOCOL.md`, `docs/EXPERIMENT_INDEX.md`, `docs/SCORER_PROTOCOL.md`: explicitly labeled historical Ling baseline.
- `docs/HANDOFF_2026-09-26.md` and `docs/RESEARCH_STATUS_2026-09-26.md`: dated snapshots with an added 2026-09-28 protocol-boundary notice.
- `docs/server_inventory/**`: historical asset-inventory labels; map and Ling inventory now carry an explicit current-protocol notice.
- `reports/audits/FINAL_THREE_SERVER_CONSOLIDATION.md`: frozen 2026-09-21 audit report describing the then-current designation; preserved unchanged as historical evidence.
- Qwen/GDN 81,920 references: outside the scope of this Ling-only protocol redesignation and left unchanged.
- Mechanism/replay reports that name the 81,920 source trajectories: historical provenance and allowed.

### Current protocol (allowed)

- `README.md`
- `docs/LING_CANONICAL_256K_PROTOCOL.md`
- `docs/PROTOCOL.md`
- `docs/EXPERIMENT_INDEX.md`
- `docs/RESEARCH_STATUS.md`
- updated handoff/status notices

These references identify 256K/262144 as current and distinguish the dynamic budget from the earlier fixed-cap seed-1 256K precursor study.

### Ambiguous wording

No unresolved ambiguous 81,920/256K wording remains in the current top-level Ling protocol, index, scorer, status, handoff, or server-map documentation. Historical audit/artifact labels were intentionally not rewritten; their snapshot context and classification are documented above.

## 5. Verification

- `git diff --check`: PASS for the protocol-freeze commit.
- Scope check: PASS; the primary commit contains documentation files only.
- Frozen artifacts modified: none.
- Generation rerun: no.
- Configuration changed: no.

Primary protocol-freeze commit: `ecc0f040267267717b68f52b29ec7ed2182c51b8`.
