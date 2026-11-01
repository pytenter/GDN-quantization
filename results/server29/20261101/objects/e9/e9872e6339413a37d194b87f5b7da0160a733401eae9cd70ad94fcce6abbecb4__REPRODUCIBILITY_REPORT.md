# Phase A reproducibility snapshot — 2026-10-11

Scope: Phase A is complete; this is not a final Phase B or full-task completion report. No persistent32K outcome has been generated for the two newly selected documents at this snapshot.

## Frozen source and matching

Measurement commit: `a74255b1b173d27a3e51f16dbc4cb26b46104d08`. CPU analysis commit: `fb2c838a01bcbc09d2c2aef9ad4e166bb9c42c48`. Published report/input-selection commit: `b4b3ecc4a301e49b57b3ba462f761ba2d479a7f4`. New, separately frozen calibration adapter commit: `c7131f3f144846ab9d3f0558210493b37a896097`.

The mathematical helpers are recovered unchanged from the historical L7 trainer, SHA256 `d1c433598875ea84c80c6f6d4fb5fe48088d5ca0bf44ccff9d950a77560fd9ae`. Measurement runner SHA256 `6f1b66553a4a30d8e39e1cd144f8a1f77fa98cec25027198cb600834c8179d1c`; original Phase A code freeze SHA256 `5e107371d691a4f88c6c9b210425f9931ed2cd4b0b41495dacb65f4c48e9d449`.

Both documents recompute every metric from a fresh shared FP reference on their assigned GPU; no incompatible historical score join. Windows256/1024, H128, all18 KDA layers/all16heads, all16 natural rotation conditions. These are matched-window full-model deployment scores under the original loss definitions, not a replay of historical512-token FP-conditioned training captures or optimization.

Model: `/data/zypan/models/Ling-3.0-tiny`; all32 weight shards freshly rehashed, metadata freshness retained. Model config SHA256 `9750d847957913f665a13c0b5a6537199e33c6f3ec970d9fcb55a0e5076d4012`; tokenizer SHA256 `40fb9d7d7795b8bd305aeff39ce9963f3f450915b9553f2938e009be9a1fed60`. Full identities are in `SOURCE_IDENTITIES.json` and `FRESH_MODEL_WEIGHT_AUDIT.json`.

Rotation H/L6/L7 SHA256 respectively: `872d4e45d21b03ccf8c3c11421331a07684f3b11ec6e5523ce7ced9254a44412`, `51a532aad371e9af1d65ed2d2d5d0c41b1369f0635ba2c11adb713283038e964`, `fdf4863869066ca5d42e9e64cb44a3ab7061b17e24ac34487d8ef42073df7ce3`.

Runtime: NVIDIA driver580.178.04, kernel7.0.0-38-generic, PyTorch2.7.1+cu128, CUDA12.8, selectorBT16_W8, seed0. GPU6 UUID `GPU-de53f243-b271-521c-d875-d96611ed8574`; GPU3 UUID `GPU-759526ed-efb2-e7f6-881c-ab098e0b05ae`.

## Canonical results

| Document | Token SHA256 | FP tensor-identity SHA256 | Canonical SHA256 | Sidecar SHA256 |
|---|---|---|---|---|
|37702→GPU6|7158fe70f187e37a3eeda995632686b249580abebf7c37a25d0cb3d6c9e45a4d|1199346f87d997df380d4954c0c9d76a84e505219cf7155dbfcf27d748d5b2de|7f4b4ef23bde72f8d670b2b7d409e0330b5f3421c049ef5b1d96c34a064e09cf|7871717f21f7e0c68948501170c3988bddb5ffaa772603bdfd0238a8e8e51def|
|5396→GPU3|2f800a5e94d247ca0f3686ae4ed1733439e3251f6fb0d61630d83382545b198c|560329ff4c8b9df57d32e7cd4740f370574e42e2c8e46f205807ebb472dc65d9|e2d4d67d7bc9eddc8630aad6c9c194765672e9ee8fe7a20f8c828aa319bb5311|057a59e2a4c03e37d1e7647407e39933d8be6a8a4c21b85006f7a8ea4158bed7|

Per condition: actual20736 QDQ/writebacks and20718 verified next-kernel consumptions. All72 real post-o_proj helper checks per document are bitwise equal with zero identity loss. FP replay, finite results, smoke repeat, canonical/sidecar pairing, and historical160-file byte identity pass. These are measured GPU gates, not CPU-only claims.

## Analysis and publication

Analysis code SHA256 `e8f260576e022cbfbe2a60d02b10b04cb6f6d6b0f59b50929137cd060f40408a`; methods freeze SHA256 `b168ed29def9a6b83ff1faa095ca2b34c784f39e64c4e6c5fb4dbce2dd388b32`. Analysis JSON SHA256 `280b004696dc00ebd18e84e9724a71cec59c3b7041420799f51e57cfd52a6480`; Route A JSON SHA256 `284f7e417f6956266b0c677013bd117f65c600a05a194abc0cfa6791e01b6346`.

Route A is frozen from calibration-only rankings before historical target analysis. Aggregate P1/L7_TOTAL Spearman0.9000/0.852941176 with2/5 eligible opposing pairs. This is high overlap with informative non-identical rankings, not gradient equivalence or optimization superiority. Positive exploratory P1 residual-rank associations use old pulse/no-further-QDQ labels across a different driver; persistent incremental information remains unconfirmed.

All8 CSVs are real complete exports with recorded SHA256 in `CSV_AUTHORING_VERIFICATION.json`. Artifact-backed typed cell roundtrips/recalculation and independent per-cell source matching pass. All126 ranking correlations independently recomputed from exact-equality midranks pass. Every rendered preview was visually inspected. The post-export Node process exit1 is retained as an engineering incident; outputs were independently verified, not blindly re-run or overwritten.

Phase A runner GPU-hours0.9995773686965307; conservative receipt-to-completion upper bound0.999650581743982. The actual ledger is this snapshot folder's `GPU_COST_LEDGER.json`, SHA256 `caa83a5e2caa82aa405f37a5749fa92415e883e0dc9563ce362c59eace73b3f8`; the initial root0 ledger remains immutable. All loading/FP/smoke time is included; CPU, waiting and unrelated experiments are excluded.

## Conditional continuation, not results

Independent new inputs selected by frozen seed/hash order:860→GPU6 and5956→GPU3. Selection SHA256 `aeb19c2602f58f2e1f35b5142e20ac56c4c1e2f247404d5c0be96adf7c6069a0`. Previous measurement documents are excluded, including48693/51410. New calibration adapter CPU tests pass but do not establish GPU correctness; the original incorrect shape-assumption sources and append-only repair are preserved in engineering appendices005/006. B calibration freeze SHA256 `4841b38e4a151cd0598d557d42d37daa88a38e9c76c51e531206db35fba247dd`.

At this snapshot no rotation pair or32K comparison is available. Both new documents must independently yield an opposing pair with both midrank gaps≥3 under the frozen tie-break. A separately frozen full-persistent runner, hybrid-state qualification, same-GPU FP, real cost/storage/continuity gate and total≤16GPUh are still required before all four32K units. Phase C remains proposal-only; no training, optimizer.step, new rotations, AIME, free generation, Qwen edits, or modifications to historical failures/results were performed.
