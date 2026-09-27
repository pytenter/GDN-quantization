# QWEN_GDN_DETERMINISM_CAUSAL_ISOLATION_V1 — FAIL

The previous V2 numerical early stop remains intact and is archived at commit `52be0e5bc9561067231f87ded788a2d0147800a1`. This new experiment isolated the prefix intervention under normal runtime: P0R0 and P1R0 completed, with identical input through the first GDN prefix input and first divergence at its prefix output. P1R0 fails every frozen full-model numerical criterion; full-model logits max abs 0.203125, relative L2 0.01116291292. The prefix difference is small locally (max abs 1.525878906e-05) but propagates to logits and C128 qcodes. Per-layer qcode counts are in `analysis/final_verdict.json`.

The exact prefix input was replayed 10 times in fresh processes (five repeats per process). Canonical and candidate each repeat exactly, yet differ from each other; the canonical result is closer to a CPU FP64 mathematical reference on this tensor. This demonstrates a reproducible arithmetic-path difference, not a claim that the candidate is mathematically invalid.

By the preregistered stop rule, P0R1/P1R1 and strict-runtime subfactors were not run. Therefore strict-runtime main effect and interaction remain unidentifiable. The current fixed left-to-right prefix candidate is rejected for canonical compatibility. Full-model forward semantics: **FAIL**; fresh-process full-model repeatability and backward: **NOT RUN**; PP2 retry authorization: **NO**.

PP2, H32/H64/H128, formal C5/C6 training and AIME were **not started**. No other user task was interrupted. All model tests used the original Qwen server, one idle GPU. Diagnostic tensors are temporary and excluded from Git; compact hashes and summaries are retained.
