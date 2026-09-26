# Research status

## Frozen formal closure

- AIME26 81,920-token protocol: frozen and archived for Qwen3.5-9B/GDN and Ling-3.0-tiny/KDA.
- Scorer: `AIME26_STRICT_V4_CANDIDATE`, SHA256 `fa5dd904c7d1887df4b8c23613f37b8d438ea494cb72607b89be8e07d55a343b`.
- Dataset manifest: SHA256 `5cdda617c7bbf73a0277de75a0f25ed52976e385b01e1264b806085d33649b0f`; three-server match.
- Qwen formal provenance: `qwen3090` / SERVER_B.
- Ling formal provenance: `ling4090` / SERVER_A.

The paired fixed-rotation effects are Qwen rescued 23, lost 0, net +23; Ling rescued 7, lost 2, net +5.

## Corrected rotation closure

Qwen uses Key-side H128 with INT8_C128. Ling uses Value-side Hadamard with INT8_R128 and corrected prefill-endpoint V2 semantics. For Ling, kernel-returned recurrent state is already in rotated Value coordinates; cache writeback and first decode consume that same basis with no redundant endpoint rotation.

The later Ling post-fix code/evidence on `new4090` is byte-identical to the selected organized copies from `ling4090`. There is no unresolved canonical-code conflict.

## Mechanism status

The repository retains compact causal and diagnostic chains without upgrading partial observations into proofs. In particular:

- Ling commutator closure remains partial/not closed and `METHOD_DESIGN_READY=NO`.
- Ling decay-anisotropy signal remains partial and `METHOD_DESIGN_READY=NO`.
- Ling persistent-error decomposition is observational/formal-complete and `METHOD_DESIGN_READY=NO`.
- Historical prefill endpoint double rotation is causally localized and closed, but applies only to the invalid old implementation path.
- Qwen mechanism reports retain their own stated gates, including partial and method-not-ready outcomes.

## 256K length-sensitivity follow-up

The table below is retained as a historical snapshot from `2026-09-21T16:07:03+08:00`:

| Condition | Snapshot | Status |
|---|---:|---|
| Ling FP_STATE | 30/30 | `COMPLETED_GENERATION_PENDING_FINAL_FREEZE` |
| Ling INT8_R128 | 13/30 | `IN_PROGRESS` |
| Ling INT8_R128 + Value-Hadamard | 10/30 | `IN_PROGRESS` |

Those entries are historical progress counts, not accuracy values. The later completed offline analysis is archived under `experiments/ling_kda/long_horizon/LING_256K_LONG_HORIZON_V1/` and reports FP_STATE 21/30, INT8_R128 9/30, and INT8_R128 + Value-Hadamard 17/30. It remains separate from the canonical 81,920 two-seed result.

## Next stage

The current phase is recurrent-aware learnable rotation design and validation. Qwen C5/C6 semantic gates pass, but formal AIME has not started; Ling L6 evaluation is running/partial, L7 evaluation is pending, and Ling memory V2 remains partial. ButterflyQuant and HARP have not been executed. The open question is whether a learnable orthogonal rotation trained under real recurrent INT8 history can outperform fixed Hadamard.
