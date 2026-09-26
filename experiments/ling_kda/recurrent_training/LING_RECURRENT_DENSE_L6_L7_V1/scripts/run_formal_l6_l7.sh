#!/usr/bin/env bash
set -euo pipefail

EXP=/data01/user2/worktrees/ling-recurrent-dense-l6-l7-v1/experiments/LING_RECURRENT_DENSE_L6_L7_V1
PY=/data01/user2/.conda/envs/ling-sglang-aime26/bin/python
MODEL=/data01/user2/models/Ling-3.0-tiny
PATCH="$EXP/scripts/sglang_kda_unified_final_r_patch.py"
ODD=(aime26_11 aime26_13 aime26_15 aime26_17 aime26_19 aime26_21 aime26_23 aime26_25 aime26_27 aime26_29)
EVEN=(aime26_12 aime26_14 aime26_16 aime26_18 aime26_20 aime26_22 aime26_24 aime26_26 aime26_28 aime26_30)
OWN_PIDS=()

cleanup() {
  for pid in "${OWN_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; fi
  done
}
trap cleanup EXIT INT TERM

run_condition() {
  local condition="$1" rotation="$2"
  local d0="$EXP/logs/formal_${condition}_gpu0" d1="$EXP/logs/formal_${condition}_gpu1"
  local p0 p1 c0 c1 s0=0 s1=0
  [[ -f "$rotation" ]] || { echo "missing rotation: $rotation" >&2; return 2; }
  mkdir -p "$d0" "$d1" "$EXP/outputs/$condition"
  "$EXP/scripts/launch_parity_server.sh" 0 31400 int8_r128_unified_final_r "$d0" "$PATCH" "$rotation" > "$d0/server.log" 2>&1 & p0=$!
  "$EXP/scripts/launch_parity_server.sh" 1 31401 int8_r128_unified_final_r "$d1" "$PATCH" "$rotation" > "$d1/server.log" 2>&1 & p1=$!
  OWN_PIDS=("$p0" "$p1")
  "$PY" "$EXP/scripts/run_formal_recurrent.py" --base-url http://127.0.0.1:31400 --model "$MODEL" --experiment-root "$EXP" --condition "$condition" --problem-ids "${ODD[@]}" --physical-gpu 0 --instance-id "${condition}_gpu0" --rotation-file "$rotation" --resume > "$d0/client.log" 2>&1 & c0=$!
  "$PY" "$EXP/scripts/run_formal_recurrent.py" --base-url http://127.0.0.1:31401 --model "$MODEL" --experiment-root "$EXP" --condition "$condition" --problem-ids "${EVEN[@]}" --physical-gpu 1 --instance-id "${condition}_gpu1" --rotation-file "$rotation" --resume > "$d1/client.log" 2>&1 & c1=$!
  wait "$c0" || s0=$?; wait "$c1" || s1=$?
  kill "$p0" "$p1" 2>/dev/null || true; wait "$p0" 2>/dev/null || true; wait "$p1" 2>/dev/null || true
  OWN_PIDS=()
  [[ "$s0" -eq 0 && "$s1" -eq 0 ]] || { echo "formal client failure $condition gpu0=$s0 gpu1=$s1" >&2; return 1; }
  "$PY" - "$EXP/outputs/$condition" <<'PY'
import json, sys
from pathlib import Path
rows = [json.loads(p.read_text()) for p in sorted(Path(sys.argv[1]).glob('aime26_*_seed1.json'))]
assert len(rows) == 20, len(rows)
assert all(row['successful_sample'] for row in rows)
assert len({row['question_id'] for row in rows}) == 20
print('FORMAL_CONDITION_COMPLETE', sys.argv[1])
PY
}

"$PY" - "$EXP" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])

def load(relative):
    path = root / relative
    if not path.is_file():
        raise RuntimeError(f"missing formal prerequisite: {path}")
    return json.loads(path.read_text(encoding="utf-8"))

parity = load("analysis/unified_h_parity.json")
baseline = load("analysis/baseline_reuse_audit.json")
training = load("analysis/TRAINING_PIPELINE_COMPLETE.json")
leakage = load("analysis/training_data_leakage.json")
materialization = load("analysis/final_rotation_materialization.json")
train_manifest = load("manifests/training_data_manifest.json")
validation_manifest = load("manifests/validation_data_manifest.json")

checks = {
    "UNIFIED_H_VS_L2_PARITY": parity.get("UNIFIED_H_VS_L2_PARITY") == "PASS",
    "BASELINE_REUSE_GATE": baseline.get("status") == "PASS",
    "TRAINING_PIPELINE_COMPLETE": training.get("status") == "PASS",
    "AIME26_TRAINING_OVERLAP_NO": leakage.get("status") == "PASS" and leakage.get("AIME26_overlap") == "NO",
    "TRAIN_MANIFEST": train_manifest.get("status") == "PASS",
    "VALIDATION_MANIFEST": validation_manifest.get("status") == "PASS",
    "FINAL_R_MATERIALIZATION": materialization.get("status") == "PASS",
    "L6_ROTATION_PRESENT": (root / "rotations/L6_final_rotation.pt").is_file(),
    "L7_ROTATION_PRESENT": (root / "rotations/L7_final_rotation.pt").is_file(),
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise RuntimeError(f"formal generation gate failed: {failed}")
print("FORMAL_PREFLIGHT_PASS", json.dumps(checks, sort_keys=True))
PY
run_condition L6 "$EXP/rotations/L6_final_rotation.pt"
run_condition L7 "$EXP/rotations/L7_final_rotation.pt"
echo L6_L7_FORMAL_COMPLETE
