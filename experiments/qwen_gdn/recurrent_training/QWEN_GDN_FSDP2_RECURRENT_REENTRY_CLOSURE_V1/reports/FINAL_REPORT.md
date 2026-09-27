# QWEN_GDN_FSDP2_RECURRENT_REENTRY_CLOSURE_V1 — final report

**Verdict: BLOCKED; root cause UNRESOLVED.** Parent H4 zero-storage failure remains real on both ranks; it was not an OOM. This task found no minimal reduced reproduction and no legal fix. Consequently H4 compatibility was not rerun and H32 feasibility remains **NOT_DETERMINED**. Do not start formal training on this evidence.

## Provenance

- Branch: `exp/qwen-gdn-fsdp2-recurrent-reentry-closure-v1`; parent `ad5007de643f55d92fccdb37562318f22a2e734b`. The current branch HEAD and commit list are given by Git; the report does not hard-code its own future commit ID.
- Parent status: BLOCKED. H1 QDQ/writeback, gradient/update and same-topology repeatability passed; H4 backward failed; FSDP2 recurrent training feasibility NOT_ESTABLISHED.
- Parent rank error SHA256 and exact wrap plan/script hashes: `configs/parent_failure_manifest.json`. Parent error has `setStorage sizes [8192,1,1,4] ... storage size 0` at `c5_loss.backward()` on both ranks, without a deeper Python model/FSDP frame.
- Runtime: PyTorch 2.5.1+cu121, CUDA 12.1, NCCL 2.21.5, Transformers 5.16.0.dev0, Triton 3.1.0. FLA version is unavailable, not guessed. API: `torch.distributed._composable.fsdp.fully_shard`, two ranks on physical GPUs 0/1. Installed FSDP source hashes are in the analysis manifest.

## Direct answers

| Question | Result |
|---|---|
| `FAILED_STORAGE_OWNER`; failed parameter/view | **UNKNOWN**; `linear_attn.conv1d.weight` is a shape/source candidate only |
| Parent-style minimal reproducer | NO, in tested real reduced paths |
| Minimum re-entry count causing parent-style storage failure | NOT_FOUND_IN_REDUCED_TESTS |
| No-cache GDN N1/N2/N4, reshard=True | PASS / PASS / nonzero-gradient gate FAIL, **not storage** |
| No-cache GDN N1/N2/N4, reshard=False | PASS / PASS / same nonzero-gradient gate FAIL, **not storage** |
| Canonical V1 cache + QDQ/rotation N1/N2/N4, reshard=True | PASS / PASS / PASS |
| Canonical V1 cache + QDQ/rotation N4, reshard=False | PASS |
| One real decoder N1/N2/N4 reshard=True; N4 reshard=False | PASS / PASS / PASS; PASS |
| Nested 2/3 real GDN decoders N1/N4, reshard=True; N4 reshard=False | All PASS |
| Recurrent state, QDQ or rotation *required for parent failure* | UNKNOWN / UNKNOWN / UNKNOWN |
| First storage-invalid lifecycle event | UNKNOWN |
| FSDP internal pre-backward unshard in failed parent | UNKNOWN; passing reduced graph only has public module hook counts |
| Failed parent saved tensor a parameter view? | UNKNOWN; passing reduced graph contains conv-weight views |
| Conv1d involved in failed parent? | UNKNOWN; candidate, not identified |
| Reshard/re-entry causal signal | NO in tested reduced conditions; parent unresolved |
| PyTorch-version compatibility suspected? | NO positive evidence; not excluded |
| Legal fix / preserves parameter-sharding memory benefit | NO / not applicable |
| H4 after fix / peak memory | NOT_RUN / not available |
| H8 / H16 / H32 / formal C5-C6 / AIME / DeepSpeed started | NO / NO / NO / NO / NO / NO |
| Other user tasks interrupted | NO |

The three-block N4 peak allocated memory increased from 2,117,376,000 bytes (reshard=True) to 2,907,156,480 bytes (False) on rank 0, without changing the PASS outcome. Direct lifecycle evidence and caveats are in `reports/STORAGE_LIFECYCLE_AUDIT.md`; exact condition/rank files in `analysis/minimal_reentry_matrix.json`.

## Stop point

The parent failure owner and first invalidation remain unobserved; the critical parent 9B/H4 backward and reduced synthetic loss/graph are not equivalent. There is no authorized H4 retry or full-model no-reshard control. No production Qwen, FSDP, QDQ, rotation or cache semantics were changed. New research would require a separately authorized diagnostic capable of observing the parent failing graph without violating the 24GB memory gate. This task stops here.
