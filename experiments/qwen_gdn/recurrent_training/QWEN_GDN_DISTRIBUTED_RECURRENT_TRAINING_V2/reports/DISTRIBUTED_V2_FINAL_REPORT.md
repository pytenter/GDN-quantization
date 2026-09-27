# QWEN_GDN_DISTRIBUTED_RECURRENT_TRAINING_V2 — stopped at numerical gate

`DISTRIBUTED_V2_STATUS=FAIL`; `DISTRIBUTED_RECURRENT_TRAINING_READY=NO`. This is an early-stop result, not a completed PP2 training study. The original V1 conclusion remains unchanged: forward exact PASS; single-versus-dual trajectory and fixed non-AIME endpoint gates FAIL; next stage NO.

The user first designated a second server's GPUs 6/7, then chose to return to the original four-RTX3090 server. The new-server hardware and environment evidence is preserved, but no work continued there after the reversion. The original server's GPUs 0/1 were idle before each bounded diagnostic. No unrelated process was modified or interrupted.

## Required questions

1. **NVLink:** No active NVLink was found on the original RTX3090 server.
2. **Selected pair:** Original-server GPUs 0/1, PHB-connected on the same NUMA node, with no direct PyTorch P2P access. Selection was conditional on fresh idle checks.
3. **Measured bandwidth:** For 128 MiB, effective NCCL send/recv was 11.459 GB/s (0→1) and 11.848 GB/s (1→0); all-reduce was 7.177 GB/s. Direct cross-device copy was 5.901 and 5.927 GB/s. These are bounded application measurements, not physical link peaks. Raw 1/8/32/128 MiB data: `analysis/p2p_bandwidth.json`.
4. **V1 versus V2:** V1 was one Python process using HF `device_map` and implicit cross-device autograd. The proposed V2 is two ranks, one GPU each, explicit NCCL stage-boundary activation/gradient transfer, initially one microbatch and no overlap. See `reports/V1_VS_DISTRIBUTED_V2_DESIGN.md`.
5. **Cumsum/scan nondeterminism:** Not closed. The old first strict-mode failure was CUDA `cumsum` during teacher-target forward. A fixed GPU prefix-sum operator and copied chunk function passed bounded microdiagnostics, but the pre-registered full-model teacher-forward test failed. The cause is not isolated because that test also enabled global strict deterministic mode.
6. **PP2 same-topology reproducibility:** NOT_RUN after the numerical stop.
7. **Distributed gradient correctness:** NOT_RUN.
8. **Rotation checkpoint portability to canonical inference:** NOT_RUN; no diagnostic PP2 checkpoint exists.
9. **H32/H64/H128 peak memory:** NOT_RUN. No horizon smoke or sustained test was authorized after the failed gate.
10. **Formal C5/C6 authorization:** NO. No formal C5/C6 training or AIME generation was started.

## Failure evidence

The fixed non-AIME full-model diagnostic compared canonical teacher forward with the explicit deterministic chunk candidate under strict deterministic mode. Its pre-frozen checks all failed: logits max absolute difference 0.203125, logits relative L2 0.0111629, relative loss difference 0.0061993, and C128 codes differed in all 24 GDN layers. The worst recorded recurrent-state max absolute difference was 0.03402 (layer 25). The result SHA256 is `41638d46c535478640c801d85dbd83f7f8952f1d55a5cb2184c785b06e442c66`; the companion error JSON SHA256 is `26def9a3b67885bd70f5ba95adda3a1dddd982f7e1797b86bf7b18518ff4a62b`. See `analysis/deterministic_full_model_teacher_forward_gate.json` and `reports/RECURRENT_DETERMINISM_AUDIT.md`.

The two causes changed together in this diagnostic, so the result does not prove that the candidate prefix sum itself is mathematically wrong. Nevertheless, the recorded V2 gate failed. Per protocol, no failed unit was retried, no tolerance was relaxed, no PP2 implementation was used for training, and no formal experiment continued.

## State and provenance

- Branch: `exp/qwen-gdn-distributed-recurrent-training-v2`, based on main commit `78b566659776a32183fcc0d6271fe2219bcf5c58` when this independent checkout was created. No main merge.
- Canonical Qwen3.5 modeling source SHA256: `90d929129ffc835d2652c604925c4f3842bc6e401e174ec6f0db2285dfb8f85a`.
- `SOURCE_DELTA_GATE=PASS`: the diagnostic chunk function differs only in the explicit prefix-sum call after AST normalization.
- `DETERMINISTIC_CUMSUM_MICRO_GATE=PASS` and `DETERMINISTIC_CHUNK_FUNCTION_GATE=PASS`, but neither is a full-model or PP2 pass.
- The old V1 artifacts were read-only. No package install/upgrade was performed; no existing runtime was modified.
- The original server had approximately 20 GiB free on `/data`; no large tensor trace was generated.
- Formal C5/C6 training: NOT_STARTED. AIME: NOT_STARTED. Other users' processes interrupted: NO.

No commit or push was made because the prescribed validation sequence stopped at a failed numerical gate, before the requested completed-artifact stage.
