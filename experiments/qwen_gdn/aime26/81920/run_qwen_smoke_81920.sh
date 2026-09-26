#!/usr/bin/env bash
set -euo pipefail

REPO="/data/zypan/worktrees/aime26-sglang-rotation-v1"
PYTHON="/data/ydai/miniconda3/envs/bitdecode/bin/python3.10"
ROOT="$REPO/artifacts/aime26_v2/official_sampling_81920/qwen"
LAUNCH_ROOT="$ROOT/launch"
PIDS=()

cleanup() {
  local pid
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

cd "$REPO"
mkdir -p "$LAUNCH_ROOT" "$ROOT/smoke"

run_method() {
  local method="$1"
  local gpu="$2"
  local out="$ROOT/smoke/${method}.jsonl"
  local log="$LAUNCH_ROOT/smoke_${method}.log"
  if [[ -s "$out" ]]; then
    echo "Refusing to overwrite existing V2 smoke artifact: $out" >&2
    return 1
  fi
  echo "START method=$method gpu=$gpu timestamp=$(date --iso-8601=seconds)"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PYTHON" -u     experiments/aime26/run_qwen_aime26_formal.py     --stage smoke --method "$method" >"$log" 2>&1
  echo "DONE method=$method timestamp=$(date --iso-8601=seconds)"
}

run_method fp_state 0 &
PIDS+=("$!")
run_method int8_c128 1 &
PIDS+=("$!")
run_method int8_c128_key_h 2 &
PIDS+=("$!")

CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 "$PYTHON" -u   experiments/aime26/qwen_deterministic_replay.py   --output "$LAUNCH_ROOT/deterministic_replay.json"   >"$LAUNCH_ROOT/deterministic_replay.stdout.log" 2>&1 &
PIDS+=("$!")

for pid in "${PIDS[@]}"; do
  wait "$pid"
done
PIDS=()

"$PYTHON" experiments/aime26/audit_qwen_formal_preflight.py   >"$LAUNCH_ROOT/preflight_report.stdout.json"

echo "QWEN_SMOKE_81920_AND_PREFLIGHT_COMPLETE"
