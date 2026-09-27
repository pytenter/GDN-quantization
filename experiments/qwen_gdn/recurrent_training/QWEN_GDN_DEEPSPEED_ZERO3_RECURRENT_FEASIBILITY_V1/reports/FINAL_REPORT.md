# QWEN_GDN_DEEPSPEED_ZERO3_RECURRENT_FEASIBILITY_V1 — FAIL at H1

Branch: `exp/qwen-gdn-deepspeed-zero3-recurrent-feasibility-v1`; base `c4fa1819633398696b3fed98e82b95fcc07473fa`; FSDP2 parent `ad5007de643f55d92fccdb37562318f22a2e734b`. HEAD and commit list are recorded by Git at delivery.

Isolated environment: `/data/zypan/worktrees/qwen-gdn-zero3-v1/.venv-qwen-zero3-recurrent-v1`. Canonical environment unchanged: **YES** (pip freeze byte-equal to pre-install snapshot; DeepSpeed absent there). Python 3.10.18, Torch 2.5.1+cu121, CUDA 12.1, NCCL (2, 21, 5), DeepSpeed 0.16.4, Transformers 5.16.0.dev0, Triton 3.1.0, FLA UNKNOWN (not a registered distribution in this environment). GPUs 0/1, ZeRO stage 3, CPU/NVMe offload NO, PP NO, TP NO, activation checkpoint NO.

Trainable Qwen base parameters: **0**. Logical Qwen base parameters: **8,953,803,264**. Trainable rotation parameters: **195,072**. Optimizer initially contained rotation only; DeepSpeed ZeRO-managed the bank. Two-rank NCCL and load/sharding gates PASS. First load setup attempt failed with `AttributeError: 'Parameter' object has no attribute 'partition_numel'`; its evidence was preserved and one parameter-creation-scope correction allowed load.

H1 canonical teacher forward: **FAIL**. Historical loss 16.969072342; ZeRO-3 loss 16.860229492; relative error 0.00641419 vs frozen 1e-5 threshold. Logits max-abs lower bound 0.125 > 0.0625; logits relative-L2 lower bound 0.00151592 > 0.001. Layer 16 state max-abs lower bound 0.00440449 > 0.001. Exact cause **unknown**. No claim that recurrent BPTT itself failed is supported.

H1 C128 QDQ: NOT RUN; writeback: NOT RUN; gradient graph: NOT RUN; Adam update: NOT RUN; repeatability: NOT RUN. H4/H8/H16/H32: NOT RUN. H4 first error/root-cause diagnostic: NOT APPLICABLE. Max stable horizon: NONE ESTABLISHED. H32 peak memory: NOT RUN. Memory leak: UNKNOWN. External parameter audit: potential rotation `.matrix()` and conv-weight-view risks, not runtime verified; no registration was applied. Rotation checkpoint portability: NOT RUN. `ZERO3_RECURRENT_TRAINING_FEASIBLE=NOT_DETERMINED`; `H32_FEASIBILITY=NOT_DETERMINED`; route `STOP_AT_H1` by protocol.

Formal C5/C6 training started: **NO**. AIME started: **NO**. Other user tasks interrupted: **NO**. Old FSDP2 conclusions and artifacts remain unchanged. The result is a valid negative feasibility gate, not a comparative verdict about DeepSpeed versus FSDP2. Further investigation would require a separately authorized forward-parity protocol; this task stops here.
