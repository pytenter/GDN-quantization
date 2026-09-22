#!/usr/bin/env bash
set -u

ROOT="${ROOT:-/data01/user2/worktrees/hadamard-init-dense-oracle-v1-ling}"
PYTHON_BIN="${PYTHON_BIN:-/data01/user2/.conda/envs/ling-kda/bin/python}"
GPU="${GPU:-1}"
OUT="results/runtime/ling_repeatability_closure_v1"
LOGS="logs/ling_repeatability_closure_v1"
cd "$ROOT"
mkdir -p "$OUT" "$LOGS" "$OUT/temporary_first_divergence"

while tmux has-session -t ling_repeatability_fresh_v1 2>/dev/null; do
  sleep 10
done

GPU="$GPU" bash experiments/ling_kda/runtime/repeatability_closure_v1/run_deterministic_factor.sh \
  > "$LOGS/deterministic_factor.log" 2>&1
echo $? > "$OUT/deterministic_factor.exit_code"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -u \
  experiments/ling_kda/runtime/repeatability_closure_v1/run_kda_reference_repeatability.py \
  --legacy-repo /data01/user2/repos/GDN-quantization --max-memory-gib 22 \
  --trace-dir results/rotation/ling_dense_orthogonal_oracle_v1/traces \
  --corpus results/rotation/ling_persistent_headroom_confirmation_v1/panel/PERSISTENT_HEADROOM_64DOC_RAW_TEXTS.jsonl \
  --state-checkpoint results/rotation/ling_dense_orthogonal_oracle_v1/training/dense_state/best_dense_state_seed0.pt \
  --functional-checkpoint results/rotation/ling_dense_orthogonal_oracle_v1/training/dense_functional/best_dense_functional_seed0.pt \
  --output "$OUT/reference_path_summary.json" \
  > "$LOGS/reference_path.log" 2>&1
echo $? > "$OUT/reference_path.exit_code"

for repeat in 0 1; do
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -u \
    experiments/ling_kda/runtime/repeatability_closure_v1/capture_prefill_layer0_temp.py \
    --legacy-repo /data01/user2/repos/GDN-quantization --max-memory-gib 22 \
    --trace-dir results/rotation/ling_dense_orthogonal_oracle_v1/traces \
    --corpus results/rotation/ling_persistent_headroom_confirmation_v1/panel/PERSISTENT_HEADROOM_64DOC_RAW_TEXTS.jsonl \
    --state-checkpoint results/rotation/ling_dense_orthogonal_oracle_v1/training/dense_state/best_dense_state_seed0.pt \
    --functional-checkpoint results/rotation/ling_dense_orthogonal_oracle_v1/training/dense_functional/best_dense_functional_seed0.pt \
    --condition Dense_State --output "$OUT/temporary_first_divergence/repeat_${repeat}.pt" \
    >> "$LOGS/first_divergence_capture.log" 2>&1
done

"$PYTHON_BIN" - <<'PY'
import hashlib
import json
from pathlib import Path
import torch

root = Path("results/runtime/ling_repeatability_closure_v1")
temporary = root / "temporary_first_divergence"
left = torch.load(temporary / "repeat_0.pt", map_location="cpu", weights_only=False)
right = torch.load(temporary / "repeat_1.pt", map_location="cpu", weights_only=False)
x, y = left["state"].double(), right["state"].double()
d = x - y
raw = lambda value: hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
nonzero = (d != 0).nonzero()
result = {
    "document_id": left["document_id"],
    "condition": left["condition"],
    "stage": "prefill_cache_pre_quant",
    "layer": int(left["layer"]),
    "shape": list(x.shape),
    "dtype": str(left["state"].dtype),
    "left_sha256": raw(left["state"]),
    "right_sha256": raw(right["state"]),
    "bitwise_equal": bool(torch.equal(x, y)),
    "max_abs": float(d.abs().max()),
    "relative_l2": float(torch.linalg.vector_norm(d) / torch.linalg.vector_norm(x).clamp_min(1e-12)),
    "different_elements": int(nonzero.shape[0]),
    "first_different_index": None if nonzero.numel() == 0 else [int(v) for v in nonzero[0].tolist()],
    "temporary_tensor_bytes_before_cleanup": sum(p.stat().st_size for p in temporary.glob("*.pt")),
    "temporary_tensors_cleaned": True,
}
(root / "first_divergence_exact.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
for path in temporary.glob("*.pt"):
    path.unlink()
temporary.rmdir()
print(json.dumps(result, indent=2, sort_keys=True))
PY
