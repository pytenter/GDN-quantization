#!/usr/bin/env bash
set -euo pipefail

REPO=/data/zypan/worktrees/ling-256k-length-sensitivity-v1
EXP="$REPO/experiments/ling_256k_length_sensitivity_v1"
OUT="$REPO/artifacts/ling_256k_length_sensitivity_v1"
STAGE0="$OUT/stage0"
PARITY_SUMMARY="$STAGE0/parity/runtime_parity_summary.json"
VALIDATION="$STAGE0/stage0_validation.json"
GUARD_LOG="$STAGE0/stage0_to_formal_guard.log"

exec >>"$GUARD_LOG" 2>&1
echo "GUARD_START=$(date --iso-8601=seconds)"

while tmux list-windows -t ling256_stage0 -F '#{window_name}' 2>/dev/null | grep -qx parity; do
  sleep 30
done

for _ in $(seq 1 120); do
  [[ -s "$PARITY_SUMMARY" ]] && break
  sleep 5
done
[[ -s "$PARITY_SUMMARY" ]] || { echo 'BLOCK: parity summary missing'; exit 1; }

"/data/zypan/runtime_compat/ling256_exact/.conda/envs/ling-sglang-aime26/bin/python" - \
  "$OUT" "$EXP/LING_256K_EXPERIMENT_CONFIG.json" "$PARITY_SUMMARY" "$VALIDATION" \
  "$STAGE0/gpu_memory_timeseries.csv" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

out, config_path, parity_path, validation_path, memory_path = map(Path, sys.argv[1:])
expected_ids = {f"aime26_{index:02d}" for index in range(1, 7)}
config_bytes = config_path.read_bytes()
config_hash = hashlib.sha256(config_bytes).hexdigest()
config = json.loads(config_bytes)

errors = []
if config.get("context_length") != 262144:
    errors.append("CONTEXT_LENGTH")
if config.get("tp_size") != 2 or config.get("allowed_physical_gpus") != [2, 3]:
    errors.append("GPU_ISOLATION_CONFIG")
rope = config.get("rope_scaling", {})
if rope.get("rope_type") != "yarn" or rope.get("factor") != 2.0 or rope.get("original_max_position_embeddings") != 131072:
    errors.append("YARN_CONFIG")

smoke_paths = sorted((out / "records/smoke256/fp_state").glob("*.json"))
if len(smoke_paths) != 1:
    errors.append("SMOKE_RECORD_COUNT")
smoke = json.loads(smoke_paths[0].read_text()) if len(smoke_paths) == 1 else {}
for key, expected in {
    "successful_sample": True,
    "token_provenance_valid": True,
    "nonfinite": False,
    "generated_tokens": 256,
    "max_context_length": 262144,
    "tp_size": 2,
}.items():
    if smoke.get(key) != expected:
        errors.append(f"SMOKE_{key}")
if smoke.get("gpu_ids") != [2, 3] or smoke.get("runtime_error") is not None:
    errors.append("SMOKE_RUNTIME")

records = {}
for path in sorted((out / "records/parity81920/fp_state").glob("*.json")):
    row = json.loads(path.read_text())
    records[row.get("problem_id")] = row
if set(records) != expected_ids:
    errors.append("PARITY_RECORD_SET")
for problem_id, row in sorted(records.items()):
    required = (
        row.get("successful_sample") is True
        and row.get("token_provenance_valid") is True
        and row.get("nonfinite") is False
        and row.get("runtime_error") is None
        and row.get("max_context_length") == 262144
        and row.get("tp_size") == 2
        and row.get("gpu_ids") == [2, 3]
        and row.get("experiment_config_hash") == config_hash
    )
    if not required:
        errors.append(f"PARITY_GATE_{problem_id}")

parity = json.loads(parity_path.read_text())
parity_rows = parity.get("rows", [])
if len(parity_rows) != 6 or any(row.get("status") != "COMPARED" for row in parity_rows):
    errors.append("PARITY_REPORT_INCOMPLETE")
if any(row.get("prompt_ids_equal") is not True for row in parity_rows):
    errors.append("PROMPT_PARITY")
if parity.get("runtime_parity_status") not in {"EXACT", "EARLY_TOKEN_PARITY", "DIVERGED"}:
    errors.append("PARITY_STATUS_UNKNOWN")

peaks = {}
if memory_path.exists():
    for line in memory_path.read_text().splitlines()[1:]:
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 4:
            index, used = int(fields[1]), int(fields[3])
            peaks[index] = max(peaks.get(index, 0), used)
if set(peaks) != {2, 3} or any(value >= 24576 for value in peaks.values()):
    errors.append("MEMORY_AUDIT")

result = {
    "stage_0_status": "PASS" if not errors else "FAIL",
    "errors": errors,
    "gpu_selection": [2, 3],
    "memory_status": "PASS" if "MEMORY_AUDIT" not in errors else "FAIL",
    "gpu_peak_memory_mib": peaks,
    "yarn_status": "PASS" if "YARN_CONFIG" not in errors else "FAIL",
    "length_only_intervention_valid": False,
    "matched_3090_81920_baseline_required": True,
    "runtime_parity_status": parity.get("runtime_parity_status"),
    "recommended_long_max_new_tokens": "262144 - prompt_tokens - 512",
    "safe_fixed_max_new_tokens": 261273,
    "smoke_v4_status": smoke.get("v4_result", {}).get("status"),
    "smoke_generated_tokens": smoke.get("generated_tokens"),
    "parity_correct": sum(bool(row.get("v4_correct")) for row in records.values()),
    "parity_count": len(records),
    "experiment_config_sha256": config_hash,
}
temporary = validation_path.with_suffix(validation_path.suffix + f".tmp.{os.getpid()}")
temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
os.replace(temporary, validation_path)
print(json.dumps(result, sort_keys=True))
if errors:
    raise SystemExit(1)
PY

python3 - "$VALIDATION" "$STAGE0/LING_256K_STAGE0_REPORT.md" <<'PY'
import json, os, sys
from pathlib import Path
src, dst = map(Path, sys.argv[1:])
x = json.loads(src.read_text())
text = f"""# LING 256K Stage 0 report

STAGE_0_STATUS = {x['stage_0_status']}
GPU_SELECTION = physical GPU 2,3 (TP=2)
MEMORY_STATUS = {x['memory_status']}
YARN_STATUS = {x['yarn_status']}
LENGTH_ONLY_INTERVENTION_VALID = NO
3090_VS_4090_PARITY = {x['runtime_parity_status']}
RECOMMENDED_LONG_MAX_NEW_TOKENS = {x['recommended_long_max_new_tokens']}
SAFE_FIXED_MAX_NEW_TOKENS = {x['safe_fixed_max_new_tokens']}
STAGE0_FP_CORRECT = {x['parity_correct']}/{x['parity_count']}
MATCHED_3090_81920_BASELINE_REQUIRED = YES
EXPERIMENT_CONFIG_SHA256 = {x['experiment_config_sha256']}

The old 4090 run used TP=1 without YaRN. The new run uses TP=2 with 256K
YaRN, so the old 4090 output is not a strict length-only control. The formal
driver must first produce a matched 3090/TP2/YaRN 81920 baseline.
"""
tmp = dst.with_suffix(dst.suffix + f".tmp.{os.getpid()}")
tmp.write_text(text)
os.replace(tmp, dst)
PY

printf 'STAGE_0_STATUS=PASS\n' > "$STAGE0/STAGE0_PASS.tmp"
mv "$STAGE0/STAGE0_PASS.tmp" "$STAGE0/STAGE0_PASS"
echo "STAGE0_GATE_WRITTEN=$(date --iso-8601=seconds)"

SERVER_PID=$(pgrep -f 'launch_ling_sglang.py.*--port 31000' | head -n 1 || true)
if [[ -n "$SERVER_PID" ]]; then
  kill -TERM "$SERVER_PID"
  for _ in $(seq 1 60); do
    kill -0 "$SERVER_PID" 2>/dev/null || break
    sleep 2
  done
fi
if pgrep -f 'launch_ling_sglang.py.*--port 31000' >/dev/null; then
  echo 'BLOCK: Stage 0 server did not stop cleanly'
  exit 1
fi

for _ in $(seq 1 60); do
  busy=0
  for index in 2 3; do
    uuid=$(nvidia-smi -i "$index" --query-gpu=uuid --format=csv,noheader,nounits)
    if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits 2>/dev/null | grep -Fxq "$uuid"; then
      busy=1
    fi
  done
  [[ "$busy" -eq 0 ]] && break
  sleep 2
done
[[ "${busy:-1}" -eq 0 ]] || { echo 'BLOCK: GPU 2 or 3 is not free after Stage 0 shutdown'; exit 1; }

"$EXP/launch_ling_256k_formal_tmux.sh"
echo "FORMAL_LAUNCHED=$(date --iso-8601=seconds)"
