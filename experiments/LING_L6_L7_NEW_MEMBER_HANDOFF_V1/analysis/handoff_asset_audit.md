# LING L6/L7 new-member handoff — Stage 0 audit

Status: `AUDIT_COMPLETE_UPLOAD_BLOCKED`

This audit was performed without starting a model job, using a formal GPU, changing the formal branch, or moving any canonical output. The active confirmatory experiment was observed at H 37/40 with three clients alive; it was not interrupted.

## Outcome

The four canonical L6/L7 binaries are intact, and all 80 frozen recurrent traces passed a full per-file SHA256 audit. Exact official-model revision and a content-addressed actual runtime source tree were recovered. Upload is blocked because neither server is authenticated to Hugging Face and because the public WikiText license evidence is internally inconsistent about license version and does not by itself settle the redistribution classification of the derived recurrent tensors.

No Hugging Face upload or readback was attempted.

## Canonical identities

| Asset | Bytes | SHA256 | GitHub | HF |
|---|---:|---|---|---|
| L6 best checkpoint | 591269 | `0f5a7666193ae8a8c8166329a41e6e47e952f6ec74ca08bcfe644b193dcd83b3` | No | No |
| L7 best checkpoint | 591269 | `ec3e0916796d63dfb7dce84ba62f2062873f066565e965af14798998746d28ae` | No | No |
| L6 final rotation | 1186141 | `ca6ef0d398d0b030dd5c295f62b8c0f43e11d41b92b319bd750ffa064977d71a` | No | No |
| L7 final rotation | 1186205 | `57eaebb47ff031c2b171cbb94563f81481d4f4632098af8339a27d01347060a2` | No | No |

Training curves, summaries, training config, data manifests, checkpoint manifest, materialization manifests, trainer, pipeline script, and preregistration are already tracked by GitHub at the frozen baseline. The checkpoint manifest and preregistration are absent from the canonical training directory copy but present in Git with exact hashes recorded in the JSON audit.

## Frozen trace audit

| Split | Files | Bytes | Count | Total bytes | Per-file SHA256 |
|---|---:|---:|---|---|---|
| Train | 64 | 16,952,122,432 | PASS | PASS | PASS |
| Validation | 16 | 4,238,042,928 | PASS | PASS | PASS |
| Total | 80 | 21,190,165,360 | PASS | PASS | PASS |

Every manifest row was opened and hashed. No file, size, or SHA mismatch was found.

## Exact runtime provenance

The canonical training script used:

`/data/zypan/experiments/LING_RECURRENT_DENSE_L6_L7_V1/runtime_repo`

This is not a standalone Git repository. It contains fixed copied shared-rotation files and symlinks to the live GDN source tree. The canonical `run_training_pipeline.sh` sets this exact path as `REPO` and passes it via `--legacy-repo` to smoke, horizon selection, L6 train/materialize, L7 train/materialize, and sanity. The successful log runs from `2026-09-26T11:51:02+08:00` to `2026-09-26T13:53:30+08:00`.

Recursive dependency inspection found 16 project-owned files needed by the imported/executed chain. Their sorted content manifest has SHA256:

`acd09637631bfe2185bee2eb17ed00df364d0416dd4a6fdeffad3064df150f63`

One source path is referenced only by a lazy, unexecuted helper and is absent from the actual runtime tree. Because canonical training succeeded without it, the audit does not substitute a similarly named file from another worktree. Exact details and all 16 file hashes are in the JSON audit.

`EXACT_RUNTIME_PROVENANCE_GATE = PASS`

## Model provenance

- Official repository: `inclusionAI/Ling-3.0-tiny`
- Exact revision: `e3a47d5b986e7141b6efd62597d598ebb392060d`
- Recovery: all 42 local HF metadata files agree; public HF API confirms the revision and repository.
- License: MIT
- Local canonical path: `/data/zypan/models/Ling-3.0-tiny`
- `trust_remote_code`: required
- Dtype: BF16
- Model shards: 32, totaling 15,787,992,416 bytes
- Config, tokenizer, custom-model code, chat-template, and index hashes are recorded in the JSON audit.

No model shard was uploaded or rehashed in Stage 0. HF etags are not mislabeled as locally computed SHA256 values.

`MODEL_PROVENANCE_GATE = PASS`

## Environment provenance

Canonical training used `/data/zypan/envs/ling-kda/bin/python3`, not the later formal SGLang environment.

Training environment highlights:

- Python 3.11.16
- Torch 2.7.1+cu128
- CUDA 12.8 / cuDNN 90701
- Triton 3.3.1
- FLA 0.5.2
- Transformers 4.57.6
- Accelerate 1.14.0
- huggingface_hub 0.36.2

The formal evaluation environment is separately preserved as Torch 2.9.1+cu128, Triton 3.5.1, FLA 0.5.2, SGLang 0.5.19, and SGLang source commit `0bcd822377da7b5718e674eaf9c870d349424dd1`.

`ENVIRONMENT_PROVENANCE_GATE = PASS`

## Hugging Face access

Repository `pytenter/ling-kda-artifacts` is publicly readable, ungated, and currently contains only `.gitattributes`. Both the 4090 and 3090 hosts return `LocalTokenNotFoundError` for `whoami`.

`HF_AUTHENTICATION_GATE = PASS` on the local upload host as account `pytenter`. The two GPU servers remain intentionally unauthenticated.

The user must run `hf auth login` directly on the intended upload host and verify that `hf auth whoami` returns `pytenter`. A token must not be sent to Codex.

## Public trace licensing

The official Salesforce WikiText page identifies WikiText and allows sharing/adaptation, but its metadata tags report `cc-by-sa-3.0` and `gfdl` while the card prose says `CC BY-SA 4.0`. The model is MIT licensed. The exact public-redistribution obligations for token IDs and recurrent tensors derived from both sources have not been conclusively classified.

`TRACE_PUBLIC_LICENSE_GATE = BLOCKED_PENDING_LICENSE_REVIEW`

Small canonical binaries and metadata remain eligible for a later Phase 1 after HF authentication and packaging/security checks. Trace upload must remain blocked until the license review resolves version, attribution, share-alike, and derived-artifact treatment.

## Stage 0 gates

| Gate | Result |
|---|---|
| L6 checkpoint | PASS |
| L7 checkpoint | PASS |
| L6 rotation | PASS |
| L7 rotation | PASS |
| Local trace integrity | PASS |
| Exact runtime provenance | PASS |
| Model provenance | PASS |
| Environment provenance | PASS |
| Candidate secret scan | PASS |
| HF authentication | PASS (local upload host) |
| Public trace license | BLOCKED_PENDING_LICENSE_REVIEW |
| HF upload/readback | NOT RUN |
| Runtime snapshot build | PASS (16 files, exact SHA256) |
| Clean-room smokes | NOT RUN |

## Required next decisions

1. Upload and read back the Phase 1 small artifacts from the authenticated local upload host.
2. Obtain an explicit license determination for public distribution of the 80 derived trace files.
3. Run the clean-room smokes without using or interrupting the active formal-generation GPUs. Trace upload remains a separate Phase 2 gate.
