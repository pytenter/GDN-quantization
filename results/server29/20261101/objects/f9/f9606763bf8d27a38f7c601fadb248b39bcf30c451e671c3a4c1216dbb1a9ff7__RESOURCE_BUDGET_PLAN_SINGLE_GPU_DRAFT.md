# C6R single-GPU resource plan

CPU planning only; no GPU forward or new worker was launched. Historical assets remain unchanged.

| Item | GPUh |
|---|---:|
| Historical five C128 plus five Dual-M actual reservations | 14.357984323 |
| Ten trajectories using each codec's slowest historical reservation | 14.691678917 |
| Ten predictions with 25% planning margin | 18.364598646 |
| New C6R initial/fresh128 qualification allowance | 0.500000 |
| Additional GPU-held IO risk allowance | 0.250000 |
| Exit forecast reserve | 0.250000 |
| Total initial projection | 19.364598646 |

The projection is below protective19.5GPUh by 8.12 minutes. It is an empirical estimate, not a proven upper bound. The20GPUh hard limit and19.5GPUh protective stop remain fixed. Actual qualification replaces its allowance; observed canary slowdown or complete32K phase overruns can only increase future predictions. A whole-plan forecast that no longer fits causes STOP before another scientific forward.

Historical cost includes model loading (about30–32 seconds per process), full32768-token forward/QDQ, scoring IO, four checkpoint writes and release. Do not count these twice. Per-document archive transfers run after GPU release; they consume walltime and disk/network resources, not extra fabricated GPUh.

## Minimal safe IO

- Put only this task's checkpoint directory at `/tmp/qwen_c6r_dual_m_5doc_checkpoints_20261016`, backed by/tmp. Bound one checkpoint at1.52GB and atomic CURRENT+tmp at3.04GB, plus512MiB explicitly preregistered disk reserve: need3576870912 bytes versus current3804200960 bytes. The matrix/cache audit must verify C6R adds no unbudgeted cache payload.
- Store new scores on/data. Keep at most the current document's two formal score files locally; budget two384MB scores, one384MB write allowance,64MiB canary scores,512MiB metadata and512MiB disk reserve: need2292850688 bytes versus current3655974912 bytes.
- After both conditions for a document finish and its GPU process is independently released, copy its two scores and current checkpoint to a fresh29 DOC directory; independently hash every file with Python streaming SHA256 and sha256sum, verify bytes, then atomically seal. Only then remove the exact new local tensor copies and write immutable archive receipt plus current LOCATION mapping. Old tensors are never removed.
- Permanent29 base: `/data/zypan/qwen_c6r_dual_m_interaction_archive/QWEN_GDN_C6R_DUAL_M_INTERACTION_REUSE_5DOC_V1`; new whole-task archive cap16GiB covers all five retained per-doc checkpoints, ten scores and metadata. Historical10GiB cap is not silently reused.

## Reusable control and stops

Reuse original single-card process identity/termination helpers, tmux ancestry proof and resident regular-file stdout. Use one fixedGPU0 UUID, a single-attempt root guard, no old prefixes and no automatic retries. Copy helper source unchanged; explicitly bind new20/19.5 budgets in the new control layer. Reject foreign compute processes, wrongUUID/type, stale sources/documents/C6R/export, QDQ/parity/finite/scale risk failure, disk reserve failure, archive verification failure, absent completion receipt, or30-minute lack of log progress. Record first failure and all actual reservations through release.

The full ten trajectories remain NOT_RUN until original FP assets are CPU-rescored, C6R math/runtime identity is valid, protocol/source files are frozen, and the new128/fresh canary and resource gates pass. No training, AIME, multidevice run or successor experiment is authorized here.
