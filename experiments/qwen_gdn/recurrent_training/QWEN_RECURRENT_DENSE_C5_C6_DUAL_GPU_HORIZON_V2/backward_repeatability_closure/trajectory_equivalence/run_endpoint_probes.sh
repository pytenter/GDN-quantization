#!/usr/bin/env bash
set -euo pipefail

mode=${1:?single or dual required}
if [[ "$mode" != single && "$mode" != dual ]]; then
  echo "mode must be single or dual" >&2
  exit 2
fi
cd /data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/QWEN_RECURRENT_DENSE_C5_C6_DUAL_GPU_HORIZON_V2
closure=backward_repeatability_closure/trajectory_equivalence
python=/data/ydai/miniconda3/envs/bitdecode/bin/python
if [[ "$mode" == single ]]; then
  end=10
  devices=0
else
  end=5
  devices=0,1
fi
for index in $(seq 1 "$end"); do
  stem=$(printf '%s_%02d' "$mode" "$index")
  if [[ ! -f "$closure/analysis/$stem.json" ]]; then
    echo "missing complete trajectory $stem" >&2
    exit 1
  fi
  if [[ -e "$closure/analysis/${stem}_probe.json" || -e "$closure/analysis/${stem}_probe.error.json" ]]; then
    echo "refusing duplicate probe $stem" >&2
    exit 1
  fi
  CUDA_VISIBLE_DEVICES="$devices" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$python" "$closure/trajectory_endpoint_probe.py" --mode "$mode" --run-id "$index" \
    > "$closure/logs/${stem}_probe.log" 2>&1 || {
      echo "${stem}_PROBE_FAILED" >&2
      tail -n 30 "$closure/logs/${stem}_probe.log" >&2
      exit 1
    }
  echo "${stem}_PROBE_COMPLETE"
done
