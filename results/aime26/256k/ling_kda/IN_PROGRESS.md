# Ling/KDA AIME26 256K status

```text
STATUS: IN_PROGRESS
server: ling4090 (hostname lthpc1)
last_checked: 2026-09-21T13:13:13+08:00
tmux_session: ling256_4090_helper
experiment: LING_256K_LENGTH_SENSITIVITY_4090_HELPER_V1
condition: INT8_R128 + Value-Hadamard
stage: formal256k
seed: 1
max_new_tokens: 262144
context_length: 262144
progress: 8/30 completed records
```

## Execution

- Two independent TP=1 SGLang instances use physical GPUs 0 and 1.
- Odd problem IDs run on GPU 0; even problem IDs run on GPU 1.
- Deterministic inference is enabled.
- Radix cache and Mamba radix cache are disabled.
- CUDA graph is disabled.
- Sampling uses temperature 1.0, top-p 0.95, and top-k 20.
- YaRN factor 2.0 extends the model-native 131,072 context to 262,144.
- KDA semantics are `CORRECTED_PREFILL_ENDPOINT_V2`; redundant prefill endpoint rotation is disabled.

## Source locations (not committed)

```text
experiment_path: /data01/user2/worktrees/aime26-sglang-rotation-v1/experiments/ling_256k_length_sensitivity_v1
launcher: /data01/user2/worktrees/aime26-sglang-rotation-v1/experiments/ling_256k_length_sensitivity_v1/run_ling_256k_4090_helper.sh
config: /data01/user2/worktrees/aime26-sglang-rotation-v1/experiments/ling_256k_length_sensitivity_v1/LING_256K_4090_HELPER_CONFIG.json
runner: /data01/user2/worktrees/aime26-sglang-rotation-v1/experiments/ling_256k_length_sensitivity_v1/run_ling_256k_length_sensitivity.py
output: /data01/user2/worktrees/aime26-sglang-rotation-v1/artifacts/ling_256k_length_sensitivity_4090_helper_v1
```

## Source hashes

```text
launcher sha256: 1c10d5fa8dbc3dd0e29a95c76d5cab929bcf911eea6a3eb4c1cb29cf4feb9e41
config sha256: d1b6fd61794cd5895d2c126b4141a02a455a6a6767c4264ff5a8dd243bc0c473
runner sha256: 6a0725cff4e889e51ddf591f9482eaac524032b84ea5daf32e342ae49414655b
scorer sha256: fa5dd904c7d1887df4b8c23613f37b8d438ea494cb72607b89be8e07d55a343b
```

At the snapshot time both server logs were actively updating at about 20 generated token/s per GPU while the clients processed the next samples. No final score or final accuracy is claimed. Growing outputs and logs were not copied.
