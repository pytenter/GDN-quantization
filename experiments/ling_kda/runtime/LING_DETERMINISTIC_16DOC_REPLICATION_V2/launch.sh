#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/zypan/worktrees/hadamard-init-dense-oracle-v1-ling-3090-v2"
OUT="$ROOT/experiments/LING_DETERMINISTIC_16DOC_REPLICATION_V2"
PYTHON_BIN="/data/zypan/envs/ling-kda/bin/python"
LEGACY_REPO="/data/zypan/repos/GDN-quantization"
FIXED_CONFIG_DIR="$ROOT/experiments/ling_kda/runtime/fresh_process_first_divergence_closure_v2/fixed_fla_configs"
TRACE_DIR="$ROOT/results/rotation/ling_dense_orthogonal_oracle_v1/traces"
CORPUS="$ROOT/results/rotation/ling_persistent_headroom_confirmation_v1/panel/PERSISTENT_HEADROOM_64DOC_RAW_TEXTS.jsonl"
STATE_CHECKPOINT="$ROOT/results/rotation/ling_dense_orthogonal_oracle_v1/training/dense_state/best_dense_state_seed0.pt"
FUNCTIONAL_CHECKPOINT="$ROOT/results/rotation/ling_dense_orthogonal_oracle_v1/training/dense_functional/best_dense_functional_seed0.pt"
SEED=20260922
REPEATS=5

cd "$ROOT"
if [[ -e "$OUT/.launch_started" ]]; then
  echo "Refusing to overwrite an existing launch: $OUT/.launch_started" >&2
  exit 2
fi
mkdir -p "$OUT/logs" "$OUT/smoke/gpu3" "$OUT/full/gpu2" "$OUT/full/gpu3"

export FLA_CACHE_MODE=default
export FLA_CONFIG_DIR="$FIXED_CONFIG_DIR"
export LING_GATED_RMSNORM_FIXED_CONFIG=BT16_W8
export LING_MODEL_PATH="/data/zypan/models/Ling-3.0-tiny"
export CUDA_VISIBLE_DEVICES=2,3
unset TRITON_CACHE_DIR CUBLAS_WORKSPACE_CONFIG CUDA_LAUNCH_BLOCKING PYTORCH_CUDA_ALLOC_CONF

host="$(hostname -s)"
mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
if [[ "$host" != "nlpg-SYS-4029GP-TRT" || "${#gpu_names[@]}" -ne 8 ]]; then
  echo "Host/GPU topology gate failed" >&2
  exit 3
fi
for name in "${gpu_names[@]}"; do
  [[ "$name" == "NVIDIA GeForce RTX 3090" ]] || { echo "Unexpected GPU: $name" >&2; exit 3; }
done
if [[ -n "$(nvidia-smi -i 2,3 --query-compute-apps=pid --format=csv,noheader,nounits)" ]]; then
  echo "GPU compute process gate failed" >&2
  exit 4
fi

{
  date -u --iso-8601=seconds
  hostname
  pwd
  nvidia-smi
  printf 'CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES"
  git status --short --branch
  git rev-parse HEAD
  git branch --show-current
  df -h /data "$ROOT"
  tmux list-panes -a -F '#{session_name}|dead=#{pane_dead}|pid=#{pane_pid}|cmd=#{pane_current_command}' || true
} > "$OUT/stage0_audit.txt"

"$PYTHON_BIN" "$OUT/manage_replication.py" prepare | tee "$OUT/logs/prepare.log"
touch "$OUT/.launch_started"
RUNTIME_HASH="$("$PYTHON_BIN" -c 'import json; print(json.load(open("experiments/LING_DETERMINISTIC_16DOC_REPLICATION_V2/launch_metadata.json"))["runtime_manifest_sha256"])')"
KERNEL_HASH="$("$PYTHON_BIN" -c 'import json; print(json.load(open("experiments/LING_DETERMINISTIC_16DOC_REPLICATION_V2/launch_metadata.json"))["kernel_configuration_sha256"])')"
GENERATION_HASH="$("$PYTHON_BIN" -c 'import json; print(json.load(open("experiments/LING_DETERMINISTIC_16DOC_REPLICATION_V2/launch_metadata.json"))["generation_configuration_sha256"])')"
printf '%s\n' "$RUNTIME_HASH" > "$OUT/runtime_manifest.sha256"
printf '%s\n' "$KERNEL_HASH" > "$OUT/kernel_configuration.sha256"
printf '%s\n' "$GENERATION_HASH" > "$OUT/generation_configuration.sha256"

common=(
  "$OUT/run_one.py"
  --mode single
  --legacy-repo "$LEGACY_REPO"
  --max-memory-gib 22
  --trace-dir "$TRACE_DIR"
  --corpus "$CORPUS"
  --state-checkpoint "$STATE_CHECKPOINT"
  --functional-checkpoint "$FUNCTIONAL_CHECKPOINT"
  --seed-mode fixed
  --seed "$SEED"
)

for condition in Native_INT8 Hadamard Dense_State Dense_Functional; do
  for repeat in 0 1; do
    CUDA_VISIBLE_DEVICES=3 "$PYTHON_BIN" -u "${common[@]}" \
      --output-dir "$OUT/smoke/gpu3" --doc-index 0 --condition "$condition" --repeat-id "$repeat" \
      2>&1 | tee -a "$OUT/logs/smoke.log"
  done
done
"$PYTHON_BIN" "$OUT/manage_replication.py" check-smoke | tee "$OUT/logs/check_smoke.log"

if [[ -n "$(nvidia-smi -i 2,3 --query-compute-apps=pid --format=csv,noheader,nounits)" ]]; then
  echo "GPU compute process appeared after smoke; refusing formal run" >&2
  exit 5
fi

worker() {
  local gpu="$1" first="$2" last="$3"
  local doc condition repeat
  for ((doc=first; doc<=last; doc++)); do
    for condition in Native_INT8 Hadamard Dense_State Dense_Functional; do
      for ((repeat=0; repeat<REPEATS; repeat++)); do
        CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u "${common[@]}" \
          --output-dir "$OUT/full/gpu${gpu}" --doc-index "$doc" \
          --condition "$condition" --repeat-id "$repeat"
      done
    done
  done
}

worker 2 0 7 > "$OUT/logs/full_gpu2.log" 2>&1 &
pid0=$!
worker 3 8 15 > "$OUT/logs/full_gpu3.log" 2>&1 &
pid1=$!
set +e
wait "$pid0"
status0=$?
wait "$pid1"
status1=$?
set -e
if [[ "$status0" -ne 0 || "$status1" -ne 0 ]]; then
  echo "Formal worker exit status: gpu2=$status0 gpu3=$status1" >&2
  exit 6
fi

"$PYTHON_BIN" "$OUT/manage_replication.py" finalize | tee "$OUT/logs/finalize.log"
touch "$OUT/.complete"
