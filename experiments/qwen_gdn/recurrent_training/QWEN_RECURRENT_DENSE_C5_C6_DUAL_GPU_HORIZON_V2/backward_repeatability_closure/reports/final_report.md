# QWEN_BACKWARD_REPEATABILITY_AND_TOPOLOGY_CLOSURE_V1

**Verdict: DUAL_GPU_TRAINING_TOPOLOGY = FAIL under the prospectively frozen single-GPU numerical-envelope protocol.** Stop here. This does not prove a particular kernel is the cause; it means at least one required dual comparison exceeded the permitted intrinsic same-topology single-GPU variability. No 8-update trajectory, four-GPU topology, H128/H64 search, formal training, or AIME generation was started.

The five initial independent single-GPU runs had exact input, teacher target, 240 sampled state/QDQ points and losses, but bitwise gradient repeatability failed. The first divergent parameter was `bank.layer(0).theta`; 21/24 GDN layers varied. The independent diagnostic reproduces the prior V2 reference's eight loss components and all 240 sampled forward/QDQ hashes exactly.

Strict PyTorch deterministic mode stopped before backward at `cumsum_cuda_kernel` inside Qwen's gated-delta fallback. The operator was not bypassed or replaced. The observed backward variability source is classified UNRESOLVED; the cumsum error is evidence of an unsupported deterministic operator, not proof of causation.

Ten additional fresh-process single-GPU one-step runs formed 45 pairs. Their frozen global relative-L2 maxima were gradient 0.00463565494 and Adam update 0.0757620924. The strict per-layer and global maxima had no safety multiplier and were hashed before the five new dual-GPU runs. The five dual runs kept forward, QDQ, writeback and BPTT exact.

| Gate | Result | Violations |
|---|---|---:|
| Single-vs-dual gradient | FAIL | 119 |
| Dual internal repeatability | FAIL | 8 |
| Single-vs-dual one-step update | FAIL | 96 |

All 50 cross-topology gradient pairs, 10 dual-internal pairs and 50 update pairs were retained; no outlier, layer or metric was dropped. Exact hashes remain provenance only because the single topology itself is not bitwise backward-repeatable.

Required next action: manual decision on a new, explicit protocol. This task does not authorize loosening the envelope or proceeding to resource search.
