# QWEN_GDN_TWO_PROCESS_STATIC_PP2_V1 — final report

**Verdict: `PP2_STATUS=FAIL` at teacher-forward parity.** The static partition and stage-local loading worked, but the two-rank forward was not mathematically equivalent to the validated canonical single-GPU reference. The single allowed localization identified the first difference within block 0's GDN/linear-attention output, before the stage boundary. The root cause remains unresolved. Per the frozen stop rule, this task stopped without backward or training; `PP2_RECURRENT_TRAINING_FEASIBLE=NOT_ESTABLISHED`, `H32_FEASIBILITY=NOT_DETERMINED`.

## Provenance and ownership

- Branch: `exp/qwen-gdn-two-process-static-pp2-v1`; base: `exp/qwen-gdn-deepspeed-zero3-recurrent-feasibility-v1` at `1edd108db04ff4f2ddfcc3d26868b6a5ba662d08`. Final HEAD and commit list are provided by Git history after commit.
- Canonical C5 implementation SHA256: `48bd12d252c827ae57c48b2228d95e30de9f92baad47ebc2e8b64405b49e516e`; rotation/QDQ SHA256: `62e769dfad73170279daf0bdb56855b7d7e4aaf77478de897958fa87b2608067`; Qwen modeling SHA256: `90d929129ffc835d2652c604925c4f3842bc6e401e174ec6f0db2285dfb8f85a`.
- Physical GPUs: rank0→GPU0, rank1→GPU1, both RTX 3090. Rank0 owns embedding and blocks 0–15; rank1 owns blocks 16–31, final norm and LM head. Static checkpoint weights: rank0 `8.338878870010376 GiB`, rank1 `8.338886499404907 GiB` (BF16 plus FP32 tensors as inventoried).
- Rank0 GDN IDs: `0,1,2,4,5,6,8,9,10,12,13,14`; rank1: `16,17,18,20,21,22,24,25,26,28,29,30`. Planned owning-rank rotation parameter counts are `97,536` each, `195,072` global, without replication. These rotations were inventoried from the canonical design; no trainable rotations were instantiated or trained after forward failure.
- The canonical environment was not modified: Python 3.10.18, Torch 2.5.1+cu121, CUDA 12.1, NCCL 2.21.5, Transformers 5.16.0.dev0, Triton 3.1.0. No FSDP, DeepSpeed, ZeRO, tensor parallelism, dynamic weight gathering, or cross-device `device_map` autograd was used.

## Frozen C5/C6 semantics

At each capture, the exact C5 objective is `L_C5 = (Σ_{l∈24 GDN} ℓ_state,l)/24`, where each layer loss is the batch/head mean of squared error of recovered pre-update K,V state versus detached teacher state, normalized by that teacher state's K,V energy (floor `1e-12`). Each stage's **local sum must divide by the global 24**, never by its local 12. C6 is `(Σ ℓ_functional,l)/24 + 0.1 L_C5`. Formal document training additionally averages eight frozen capture positions; the historical H1 diagnostic uses only token index 31 of the first 32 TRAIN tokens and no `/8` factor. Full derivation is in `reports/CANONICAL_C5_C6_MATH_DERIVATION.md`.

Teacher targets come from the complete unrotated 32-block model in no-grad mode, with continuous cache across token prefixes; captured pre-update recurrent states are FP32 and detached, along with out-projection targets. The student would use the original GDN core with key-side Hadamard+Cayley rotation, INT8-C128 QDQ and differentiable cache writeback. At each global H-token BPTT boundary, canonical `detach_cache` detaches recurrent, convolution and key/value caches and trims convolution history; both stages would need the same token boundary. These training semantics were audited but **not executed** in PP2 because teacher-forward parity failed.

## Gate results

| Gate / measurement | Result |
| --- | --- |
| Stage-local load, 16 blocks per rank, frozen base | PASS |
| Canonical single-GPU teacher versus historical reference | Exact on input, selected states, logits, loss |
| PP2 rank0 send versus rank1 receive | Exact |
| PP2 boundary hidden versus canonical single-GPU | FAIL |
| All 24 teacher GDN recurrent states | 1 exact (layer 0), 23 mismatched |
| PP2 final logits and loss | FAIL; reference loss `16.969072341918945`, PP2 `15.68132209777832` |
| Student QDQ/writeback | NOT_RUN |
| First forward divergence | Block 0, after exact input-layer norm, at GDN/linear attention output |
| H1 boundary gradient / rotation gradient / one-step Adam / PP2_H1_GATE | NOT_RUN |
| PP2_H4_GATE; H4 rank0/rank1 peaks | NOT_RUN; no peaks measured |
| H8 / H16 / H32; H32 rank0/rank1 peaks; H32×5 stability | NOT_RUN; no BPTT peaks measured |
| PP2 memory leak | UNKNOWN |
| Checkpoint portability | NOT_RUN; no checkpoint created |

The measured teacher-forward boundary wire carried `[1,64,4096]` BF16 elements: `524,288 bytes` total, or `8,192 bytes/token` at batch 1. Backward wire bytes/token are **NOT_RUN**; the frozen BF16 protocol would require `8,192 bytes/token`, but no gradient was sent. Communication and compute wall times were not profiled. Stage-load memory snapshots must not be interpreted as H4 or H32 peaks.

The only localization showed full-model `_attn_implementation=sdpa` versus standalone stage `None`, with a matching block-0 input-layer norm but mismatched linear-attention output. This is an investigation lead, not a proven unique root cause; no patch or rerun was permitted by this protocol. See `reports/FORWARD_PARITY_REPORT.md` and `analysis/first_forward_divergence_localization.json`.

Formal C5/C6 training: **NO**. AIME generation: **NO**. Other users' processes: **not interrupted**. Existing FSDP2, ZeRO-3, single-GPU and one-process dual conclusions and artifacts were not changed. The next decision requires a new user-authorized protocol; this task does not start another framework or experiment.
