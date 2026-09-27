# QWEN_GDN_FSDP2_OUTPUT_HOOK_CAUSAL_CLOSURE_V1 — final report

**Status: BLOCKED; root cause UNRESOLVED.** The finite zero-output anchor was exactly neutral on single-GPU H1, but did not rescue the full 9B FSDP2 H4 backward. Both ranks failed on a **new** zero-storage tensor shape. This task stopped H4 testing at the required new-failure-mode gate. The old parent remains BLOCKED, with H4 backward FAIL and H32 feasibility NOT_DETERMINED.

## Provenance and scope

- Branch: `exp/qwen-gdn-fsdp2-output-hook-causal-closure-v1`, based on re-entry closure `f730e3dca7589d8cc272de839d6e5b3a77cd1eaa`; original FSDP feasibility parent `ad5007de643f55d92fccdb37562318f22a2e734b`.
- Parent H4 error hashes are in `configs/parent_manifest.json`. Runtime is PyTorch 2.5.1+cu121 and the composable `torch.distributed._composable.fsdp.fully_shard` API.
- No old experiment, production Qwen operator, C128 QDQ, recurrent writeback, C5 objective, rotation parameterization, frozen backbone, Torch/CUDA installation, or other user's process was changed.
- The current branch HEAD is the commit containing this file; use `git rev-parse HEAD` or the final handoff commit SHA to identify it without creating a self-referential hash.

## Required gates and observations

| Gate / question | Result |
|---|---|
| C5 depends on ordinary model output? | **PARTIAL**: no direct path through final logits or same block's returned output; upstream decoder outputs can feed downstream internal states |
| Exact zero anchor tensor | finite final captured-token model logits `[1,1,248320]`, grad-connected; `0.0 * logits.float().sum()` |
| Single-GPU original vs anchored H1 | loss exact; 24/24 gradients exact; 24/24 one-step Adam theta exact; QDQ/writeback exact — **PASS** |
| Frozen baseline full 9B H4 | both ranks `setStorage [8192,1,1,4] ... storage size 0` at C5 backward; not rerun |
| Anchored full 9B H4 | both ranks `setStorage [1024,4096] ... storage size 0` during backward — **NEW_FAILURE_MODE** |
| `ZERO_ANCHOR_H4_RESCUE` | **NO** |
| Anchored H4 peak allocated / reserved | 14,201,228,288 / 17,190,354,944 bytes per rank; no OOM |
| H1 reduced hook noninterference | **PASS**: exact loss/gradient/update; 0-byte GPU peak increase; two named FSDP pre-backward events/rank |
| Baseline / anchored H4 pre-backward trace | **NOT_CAPTURED / NOT_CAPTURED**; H4 hook-only rerun forbidden by stop rule |
| Baseline missing / anchor restored hook; first missing unit | **UNKNOWN / UNKNOWN** |
| `FSDP_PRE_BACKWARD_HOOK_ANOMALY`; `FAILED_STORAGE_OWNER` | **UNKNOWN / UNKNOWN** |
| Output-view/in-place signal | **NOT_ESTABLISHED**; conv weight and cache paths contain views/mutations, not proven to mutate a wrapped decoder output |
| `OUTPUT_HOOK_BYPASS_CAUSAL_SIGNAL`; output-hook hypothesis | **UNRESOLVED / UNRESOLVED** |
| DeepSpeed ZeRO-3 recommended next? | **NO, not justified by this protocol's evidence**; no DeepSpeed install or runtime test |
| H4 compatibility after intervention | **BLOCKED** |
| H8 / H16 / H32 / formal C5-C6 / AIME started | **NO / NO / NO / NO / NO** |
| Other user tasks interrupted | **NO** |

Static C5 graph and output/view audits are in their own reports; exact rank and hook evidence are in `analysis/`. The [related PyTorch FSDP2 output-view issue](https://github.com/pytorch/pytorch/issues/181832) illustrates a possible mechanism, but it is **not** proof that this Qwen graph has that bug. With no H4 hook trace and no anchor rescue, no failed tensor owner or causal hook bypass can be asserted. A future separately approved full-graph localization diagnostic would be more discriminating than installing another distributed runtime now.
