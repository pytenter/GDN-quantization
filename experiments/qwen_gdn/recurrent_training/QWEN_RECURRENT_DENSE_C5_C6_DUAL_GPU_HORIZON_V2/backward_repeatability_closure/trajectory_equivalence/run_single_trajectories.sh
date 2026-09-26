#!/usr/bin/env bash
set -eu

experiment=/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/QWEN_RECURRENT_DENSE_C5_C6_DUAL_GPU_HORIZON_V2
closure="$experiment/backward_repeatability_closure/trajectory_equivalence"
python_bin=/data/ydai/miniconda3/envs/bitdecode/bin/python
cd "$experiment"

while [ ! -f "$closure/analysis/single_01.json" ]; do
  if [ -f "$closure/analysis/single_01.error.json" ]; then
    echo FIRST_TRAJECTORY_FAILED
    exit 1
  fi
  sleep 60
done

for run_id in 2 3 4 5 6 7 8 9 10; do
  if [ -e "$closure/analysis/single_$(printf '%02d' "$run_id").json" ] ||
     [ -e "$closure/analysis/single_$(printf '%02d' "$run_id").error.json" ]; then
    echo "EXISTING_RESULT_OR_ERROR_FOR_RUN_$run_id"
    exit 1
  fi
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$python_bin" "$closure/trajectory_equivalence.py" \
    --phase run --mode single --run-id "$run_id" \
    > "$closure/logs/single_$(printf '%02d' "$run_id").log" 2>&1
  echo "SINGLE_TRAJECTORY_${run_id}_COMPLETE"
done

echo SINGLE_TRAJECTORIES_10_COMPLETE
