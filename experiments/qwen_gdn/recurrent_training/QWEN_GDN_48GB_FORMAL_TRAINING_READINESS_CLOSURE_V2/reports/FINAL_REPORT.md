# QWEN_GDN_48GB_FORMAL_TRAINING_READINESS_CLOSURE_V2 — final report

Parent: `origin/exp/qwen-gdn-48gb-single-recurrent-training-v1` at `d3f1ab61925dc91d7215de0c505cbc01c9043d65`. Training host: NVIDIA vGPU-48GB. Canonical inference host: four RTX 3090 (24GB each).

V1 recomputed-R exact-hash portability: **FAIL** (unchanged). V2 frozen-R deployment portability: **PASS**. Theta SHA-256 `554b59891cc3c5b37a48203fd553e3992e6b7759f5867080ea377e165377f228`; frozen-R SHA-256 `cb9eca306fa4e3589fd4d5a67fe285f90ef94085a27715e1ed517ef60b3bbe17`; 24/24 per-layer hashes exact.

Full-document training has 1024 tokens and eight zero-based captures at 625, 663, 813, 819, 881, 940, 957, 989. There are at most two simultaneously stored capture losses before a segment backward. Theta gradients accumulate over H segments; one optimizer step follows the document. The original preregistered token digest had one extra character; the explicit erratum and preflight stop are preserved.

C5 H32: **PASS**; peak allocated 23.215 GiB, peak reserved 23.746 GiB; sustained **PASS**, memory creep NO.
C6 H32: **PASS**; peak allocated 23.215 GiB, peak reserved 23.746 GiB; sustained **PASS**, memory creep NO.
C5 H64: single **PASS**, sustained **PASS**, single peak reserved 31.092 GiB. H128: **BLOCKED** by CUDA OOM, last free 16.7 MiB; no retry.
C6 H64: single **PASS**, sustained **PASS**, single peak reserved 31.111 GiB. H128: **BLOCKED** by CUDA OOM, last free 16.7 MiB; no retry.

Historical horizon rule: largest_joint_sustained_feasible_horizon. Selected horizon: 64. FORMAL_RECURRENT_TRAINING_READY=YES.
Formal C5/C6 training: NOT_STARTED. AIME: NOT_STARTED. Distributed framework: NO. Original inference source modification: NO. Other user processes were not interrupted.

See `ROTATION_DEPLOYMENT_PORTABILITY_V2.md`, `FULL_DOCUMENT_GRAPH_LIFETIME_AUDIT.md`, `FULL_DOCUMENT_MEMORY_CLOSURE.md`, and `FORMAL_TRAINING_READINESS.md` for stage evidence.
