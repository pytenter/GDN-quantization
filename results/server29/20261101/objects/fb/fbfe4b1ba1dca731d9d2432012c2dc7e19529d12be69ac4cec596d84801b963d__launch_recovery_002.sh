#!/bin/bash
set -eu
task_rec=/data/zypan/worktrees/qwen-gdn-deterministic-scale-discovery-v3/experiments/QWEN_GDN_DETERMINISTIC_RECOVERY_TO_SCALE_DISCOVERY_V3/recovery/attempt_002
task_runtime=/data/zypan/experiment_runs/QWEN_GDN_DETERMINISTIC_RECOVERY_TO_SCALE_DISCOVERY_V3_runtime
test -f "$task_rec/RECOVERY_PREMEASUREMENT_FREEZE.json"
test ! -e "$task_runtime/recovery_002_controller.log"
tmux new-session -d -s qwen_v3_recovery002 "env CUDA_VISIBLE_DEVICES= CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1 /data/ydai/miniconda3/envs/bitdecode/bin/python $task_rec/scripts/resume_v3.py controller > $task_runtime/recovery_002_controller.log 2>&1"
