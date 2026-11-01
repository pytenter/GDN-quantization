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
[[ -e "$OUT/.launch_started" ]] || { echo "Original launch marker missing" >&2; exit 2; }
[[ ! -e "$OUT/.complete" ]] || { echo "Experiment is already complete" >&2; exit 2; }
[[ ! -e "$OUT/.resume_gpu37_started" ]] || { echo "GPU3/7 resume already started" >&2; exit 2; }

host="$(hostname -s)"
[[ "$host" == "nlpg-SYS-4029GP-TRT" ]] || { echo "Host gate failed: $host" >&2; exit 3; }
for gpu in 3 7; do
  name="$(nvidia-smi -i "$gpu" --query-gpu=name --format=csv,noheader | xargs)"
  [[ "$name" == "NVIDIA GeForce RTX 3090" ]] || { echo "GPU $gpu model gate failed: $name" >&2; exit 3; }
done
if [[ -n "$(nvidia-smi -i 3,7 --query-compute-apps=pid --format=csv,noheader,nounits)" ]]; then
  echo "GPU3/GPU7 compute process gate failed" >&2
  exit 4
fi

existing="$(find "$OUT/full" -path '*/fresh_process_runs/*.json' -type f | wc -l)"
[[ "$existing" -eq 28 ]] || { echo "Expected 28 preserved results, found $existing" >&2; exit 5; }

mkdir -p "$OUT/full/gpu3" "$OUT/full/gpu7" "$OUT/logs"
export FLA_CACHE_MODE=default
export FLA_CONFIG_DIR="$FIXED_CONFIG_DIR"
export LING_GATED_RMSNORM_FIXED_CONFIG=BT16_W8
export LING_MODEL_PATH="/data/zypan/models/Ling-3.0-tiny"
export CUDA_VISIBLE_DEVICES=3,7
unset TRITON_CACHE_DIR CUBLAS_WORKSPACE_CONFIG CUDA_LAUNCH_BLOCKING PYTORCH_CUDA_ALLOC_CONF

{
  date --iso-8601=seconds
  printf 'resume_pair=3,7\n'
  printf 'preserved_results=%s\n' "$existing"
  nvidia-smi -i 3,7
} > "$OUT/gpu37_resume_audit.txt"
touch "$OUT/.resume_gpu37_started"

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

worker() {
  local gpu="$1" first="$2" last="$3"
  local doc condition repeat pattern
  local -a matches
  shopt -s nullglob
  for ((doc=first; doc<=last; doc++)); do
    for condition in Native_INT8 Hadamard Dense_State Dense_Functional; do
      for ((repeat=0; repeat<REPEATS; repeat++)); do
        printf -v pattern 'doc%02d_%s_fixed_r%02d.json' "$doc" "$condition" "$repeat"
        matches=("$OUT"/full/gpu*/fresh_process_runs/"$pattern")
        if [[ "${#matches[@]}" -eq 1 ]]; then
          printf 'skip existing %s\n' "${matches[0]}"
          continue
        fi
        if [[ "${#matches[@]}" -gt 1 ]]; then
          printf 'duplicate result identity %s\n' "$pattern" >&2
          return 20
        fi
        CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u "${common[@]}" \
          --output-dir "$OUT/full/gpu${gpu}" --doc-index "$doc" \
          --condition "$condition" --repeat-id "$repeat"
      done
    done
  done
}

# Keep the original upper-half assignment on GPU3; GPU7 replaces busy GPU2.
worker 7 0 7 > "$OUT/logs/resume_full_gpu7.log" 2>&1 &
pid7=$!
worker 3 8 15 > "$OUT/logs/resume_full_gpu3.log" 2>&1 &
pid3=$!
set +e
wait "$pid7"; status7=$?
wait "$pid3"; status3=$?
set -e
if [[ "$status7" -ne 0 || "$status3" -ne 0 ]]; then
  echo "Resume worker exit status: gpu7=$status7 gpu3=$status3" >&2
  exit 6
fi

count="$(find "$OUT/full" -path '*/fresh_process_runs/*.json' -type f | wc -l)"
[[ "$count" -eq 320 ]] || { echo "Final raw result count is $count, expected 320" >&2; exit 7; }
duplicates="$(find "$OUT/full" -path '*/fresh_process_runs/*.json' -type f -printf '%f\n' | sort | uniq -d)"
[[ -z "$duplicates" ]] || { printf 'Duplicate result identities:\n%s\n' "$duplicates" >&2; exit 7; }

"$PYTHON_BIN" "$OUT/finalize_gpu37.py" | tee "$OUT/logs/finalize_gpu37.log"
touch "$OUT/.complete"
