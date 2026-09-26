# Server provenance

This file records non-secret scientific provenance only. Credentials, tokens, private keys, and credential-bearing environment variables are intentionally excluded.

## Server roles

| Stable handoff name | Audited hardware | Principal roots used | Main archived evidence | Snapshot safety state |
|---|---|---|---|---|
| Qwen server | 4 x RTX 3090 | `/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/` and experiment subtrees | Qwen replay, M5 validation, last-20 E2E, C3/C4 provenance, C5/C6, memory and topology closure | Trajectory run active on GPU0; read-only collection only |
| Ling server | 2 x RTX 4090 | `/data01/user2/worktrees/` | Ling L4/L5 provenance, original L6/L7 preparation and H materialization | GPUs idle at snapshot; old diagnostic sessions left untouched |
| Third server | 8 x RTX 3090 | `/data/zypan/worktrees/hadamard-init-dense-oracle-v1-ling-3090-v2/`, `/data/zypan/worktrees/ling-256k-length-sensitivity-v1/`, `/data/zypan/experiments/LING_RECURRENT_DENSE_L6_L7_V1/` | Deterministic V2, 256K final offline analysis, L6/L7 training and formal evaluation | L6 formal generation active on GPUs 3/4; read-only collection only |

The historical inventory name `new4090` is retained in some pre-existing paths for traceability, but it must not be interpreted as the hardware type of the third server.

## Provenance policy

- Existing Git branches were inventoried before server collection. `origin/codex/final-consolidation-2026-09-21` is the base; the cumulative Qwen and Ling branches were preferred over their intermediate ancestors.
- Byte-identical or already canonical files were not recopied under alternate names. Twenty-two previously verified cross-server duplicates remain represented by their canonical copy.
- `research-sync-2026-09-02` was selectively imported only for useful mechanism paths absent from the handoff tree. Existing paths were not overwritten.
- Server artifacts from 2026-09-23 through 2026-09-26 were copied as code/config/manifest/compact result/report. Checkpoints, full responses, raw token traces, growing outputs, caches, and model data were excluded.
- No same-name/different-content source was silently overwritten. No unresolved content conflict was found in the selected archive.
- Directory SHA256 values are not invented. Where a raw directory was excluded, its size/mtime and a committed manifest hash are recorded when available; live directories are marked `LIVE_OUTPUT_NOT_HASHED`.

## Runtime snapshot qualifications

The runtime statements in this handoff are observations, not process control actions. At the final read-only snapshot:

- Qwen trajectory equivalence was running; `single_01` through `single_08` were immutable and archived, while the active ninth run was excluded.
- Ling L6 formal evaluation was running on the third server. Seven L6 JSON outputs were present, but there was no final report or completion marker; none of those live outputs was copied.
- Ling recurrent memory V2 was PARTIAL and deliberately not extended while formal generation occupied the runtime.

No server experiment, process, environment, model, or original artifact was modified, stopped, restarted, or deleted by this consolidation.
