#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 2 ]] || { echo "usage: $0 CONDITION ROTATION" >&2; exit 2; }
CONDITION="$1"; ROTATION="$2"
[[ "$CONDITION" == L6 || "$CONDITION" == L7 ]] || { echo "invalid condition" >&2; exit 2; }
EXP=/data01/user2/worktrees/ling-recurrent-dense-l6-l7-v1/experiments/LING_RECURRENT_DENSE_L6_L7_V1
PY=/data01/user2/.conda/envs/ling-sglang-aime26/bin/python
MODEL=/data01/user2/models/Ling-3.0-tiny
PATCH="$EXP/scripts/sglang_kda_unified_final_r_patch.py"
REFERENCE_JSON="$EXP/logs/unified_h_parity/unified_h_int8/client_result.json"
ROOT="$EXP/logs/cross_host_${CONDITION}_int8"
RUN="$ROOT/reference_4090"
PORT=$([[ "$CONDITION" == L6 ]] && echo 31406 || echo 31407)
server=""
cleanup() { [[ -z "$server" ]] || { kill "$server" 2>/dev/null || true; wait "$server" 2>/dev/null || true; }; }
trap cleanup EXIT INT TERM
[[ -f "$EXP/mixed_gpu_protocol_amendment_v1.json" && -f "$EXP/manifests/hardware_assignment.json" ]] || { echo "mixed-GPU amendment is not frozen" >&2; exit 2; }
[[ -f "$ROTATION" ]] || { echo "missing rotation" >&2; exit 2; }
[[ ! -e "$RUN/client_result.json" ]] || { echo "refusing to overwrite completed reference" >&2; exit 2; }
mkdir -p "$RUN"
"$EXP/scripts/launch_parity_server.sh" 0 "$PORT" int8_r128_unified_final_r "$RUN" "$PATCH" "$ROTATION" > "$RUN/server.log" 2>&1 & server=$!
"$PY" "$EXP/scripts/run_runtime_gate_client.py" --base-url "http://127.0.0.1:$PORT" --model "$MODEL" --mode int8_r128_unified_final_r --run-dir "$RUN" --reference-json "$REFERENCE_JSON" --wait-seconds 1200 > "$RUN/client.log" 2>&1
cleanup; server=""
"$PY" "$EXP/scripts/tensor_bitwise_manifest.py" --run-dir "$RUN" --rotation "$ROTATION" --condition "$CONDITION" --host-class RTX4090 --output "$EXP/analysis/${CONDITION}_4090_int8_tensor_manifest.json"
echo "${CONDITION}_4090_INT8_REFERENCE_COMPLETE"
