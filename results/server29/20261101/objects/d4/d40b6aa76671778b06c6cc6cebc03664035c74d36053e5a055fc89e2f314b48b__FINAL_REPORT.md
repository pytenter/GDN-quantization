# FSDP2 recurrent-aware training feasibility — early-stop report

Status: **BLOCKED**. The two-card FSDP2 implementation shards frozen Qwen parameters and preserves canonical teacher forward and H1 C128 QDQ/writeback. Five independent H1 two-rank processes yielded identical loss, gradients and one-step rotation updates. The first H4 update then failed on **both ranks during backward**, before optimizer step, with `setStorage ... storage of size 0`. This is a reproducible-in-rank framework/storage error in the single attempted H4 run, **not an OOM**. It was not retried or masked.

## Scope and provenance

- Base: `9f2df90d49d3b692fbc0da5286ce735de43fa1db` (verified latest causal-isolation branch at preregistration).
- Branch: `exp/qwen-gdn-fsdp2-recurrent-training-feasibility-v1`.
- Server: original four-RTX3090 host; selected GPU0/GPU1. Other user processes were not interrupted.
- Runtime: PyTorch 2.5.1+cu121, NCCL, `torch.distributed._composable.fsdp.fully_shard`; block-wise FSDP2, plus embedding, final norm, distinct lm_head and root. Rotation bank was replicated, small, and solely trainable.
- Canonical GDN `torch.cumsum`, C128 QDQ, Key-side rotation and recurrent writeback were unchanged. No fixed-prefix, `device_map` dual, PP, TP, activation checkpointing, formal C5/C6 or AIME was used.
- DeepSpeed was checked only after the H4 block, is absent in the frozen environment, and was not installed or run.

## Gates

| Gate | Result | Evidence |
|---|---|---|
| FSDP runtime / parameter sharding / trainable inventory | PASS | 427 model parameter entries have `S(0)` and exactly half local numel on each rank; 8,953,803,264 logical model params, 4,476,901,632 local per rank; 195,072 trainable rotation params. Initial load checker falsely looked for `Shard` rather than actual `S(0)`; original artifact and separate correction retained. |
| Canonical teacher forward | PASS | Single vs 3 fresh two-rank runs: selected GDN states, logits and loss bitwise equal. |
| H1 QDQ/recurrent writeback | PASS | All 24 layers: pre-state, scale, qcodes, post-state hashes and cache writeback exact; loss exact. |
| H1 gradient/one Adam update | PASS at H1 | All 24 rotation gradients exist, finite and nonzero on both ranks; all 24 theta tensors change; sampled frozen model hashes unchanged. |
| Same-topology H1 repeatability | PASS | Five fresh two-rank processes: exact forward/loss, gradient norms (maximum range 0), and post-Adam theta hashes. |
| H4 backward compatibility | FAIL | Both ranks: `RuntimeError: setStorage: sizes [8192,1,1,4] ... storage of size 0` at `c5_loss.backward()`; no OOM. Root cause is unproven; repeated invocation/reshard interaction is only a hypothesis. |
| H32, lifetime, H64/H128, portability | BLOCKED | H4 error triggered preregistered stop. No extrapolation to H32, no sustained updates, no checkpoint export. |

## GPU memory (bytes)

| Condition | Single GPU | FSDP2 rank0 | FSDP2 rank1 |
|---|---:|---:|---:|
| Static full-model load / sharded load | 17,907,635,200 | 8,957,748,224 | 8,957,748,224 |
| H1 peak allocated | 18,595,188,224 (forward only) | 13,697,982,976 (update) | 13,697,982,976 (update) |
| H4 | — | backward storage error; peak not captured | backward storage error; peak not captured |
| H8 / H16 | — | NOT_RUN | NOT_RUN |
| H32 | historical single-GPU OOM | NOT_RUN | NOT_RUN |
| H64 / H128 | — | NOT_AUTHORIZED | NOT_AUTHORIZED |

FSDP2 cut measured static allocation by 8,949,886,976 bytes per GPU (~50%). H1 FSDP update peak reserved was 15,103,688,704 bytes per rank. The single H1 number is a forward-only diagnostic and **must not** be interpreted as an apples-to-apples update comparison. The old single-GPU memory audit found `NO_LEAK_H32_TOO_LARGE`; its H32 OOM is historical. Neither an H32 FSDP peak nor an H-dependent activation slope can be estimated from this stopped run. `STATIC_VS_ACTIVATION_CONTRIBUTION=PARTIALLY_IDENTIFIED`.

## Deferred audits and conclusion

The saved-tensor family audit was not run after the H4 backward failure; no largest-family claim is made. The 5 H1 probes were independent one-step runs, **not** 5 sustained updates; the eight-update checkpoint-portability trigger was not met. Memory leak status under FSDP2 is undetermined. DeepSpeed ZeRO-3 fallback was unavailable in this environment and no packages were changed.

`FSDP2_RECURRENT_TRAINING_FEASIBLE=NOT_ESTABLISHED`; `H32_FEASIBILITY_GATE=BLOCKED`. This negative feasibility result does not overturn prior frozen experiments. A separate protocol is needed to investigate the FSDP2 H4 backward/storage incompatibility before any H32 memory claim. Formal C5/C6 and AIME remain **NO**.

See `analysis/final_verdict.json`, `analysis/fsdp_h4_backward_failure.json`, per-rank H4 `.error.json`, `analysis/recurrent_qdq_semantics.json`, `analysis/same_topology_repeatability.json`, and `hashes/artifact_sha256.txt`.
