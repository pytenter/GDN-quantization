#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/zypan/experiment_runs/LING_KDA_PERSISTENT_SURROGATE_PROSPECTIVE_VALIDATION_V1
PY=/data/zypan/envs/ling-kda/bin/python
RUNNER="$ROOT/scripts/run_calibration.py"
CONDITIONS=DA_fromH,DA_fromL6,H,H_to_L6_a25,H_to_L6_a50,H_to_L7R_a25,H_to_L7R_a50,H_to_L7_a25,H_to_L7_a50,H_to_L8_a25,H_to_L8_a50,IDENTITY,L6,L7,L7R,L8

uuid3=GPU-759526ed-efb2-e7f6-881c-ab098e0b05ae
uuid6=GPU-de53f243-b271-521c-d875-d96611ed8574
occupants="$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits 2>/dev/null || true)"
if grep -qE "^(${uuid3}|${uuid6})," <<<"$occupants"; then
  printf '%s\n' "$occupants" > "$ROOT/logs/calibration_gpu_occupied.txt"
  exit 41
fi

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export LING_GATED_RMSNORM_FIXED_CONFIG=BT16_W8
export PYTHONUNBUFFERED=1

run_one() {
  local gpu="$1" doc="$2" tag="calibration_${2}_gpu${1}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$RUNNER" \
    --document "$doc" --gpu "$gpu" --conditions "$CONDITIONS" \
    --starts 256,1024 --horizons 32,128 --tag "$tag" \
    --max-wall-seconds 14400 >"$ROOT/logs/${tag}.log" 2>&1
  printf '%s\n' "$?" > "$ROOT/logs/${tag}.exitcode"
}

run_one 6 pg19-test-51410 & p6=$!
run_one 3 pg19-test-48693 & p3=$!
set +e
wait "$p6"; r6=$?
wait "$p3"; r3=$?
set -e
printf '{"gpu6":%d,"gpu3":%d}\n' "$r6" "$r3" > "$ROOT/logs/calibration_pair_exitcodes.json"
if (( r6 != 0 || r3 != 0 )); then
  exit 42
fi
