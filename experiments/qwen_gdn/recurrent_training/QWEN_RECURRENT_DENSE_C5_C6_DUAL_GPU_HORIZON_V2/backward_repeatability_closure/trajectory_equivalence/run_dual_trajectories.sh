#!/usr/bin/env bash
set -euo pipefail

cd /data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/QWEN_RECURRENT_DENSE_C5_C6_DUAL_GPU_HORIZON_V2
closure=backward_repeatability_closure/trajectory_equivalence
python=/data/ydai/miniconda3/envs/bitdecode/bin/python

"$python" - "$closure" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
meta = json.loads((root / "analysis/frozen_single_gpu_trajectory_envelope_hash.json").read_text())
path = root / "analysis/frozen_single_gpu_trajectory_envelope.json"
if hashlib.sha256(path.read_bytes()).hexdigest() != meta["sha256"]:
    raise SystemExit("frozen single trajectory envelope hash mismatch")
for index in range(1, 11):
    path = root / "analysis" / f"single_{index:02d}.json"
    if not path.is_file():
        raise SystemExit(f"single trajectory incomplete: {path}")
PY

for index in 1 2 3 4 5; do
  stem=$(printf 'dual_%02d' "$index")
  if [[ -e "$closure/analysis/$stem.json" || -e "$closure/analysis/$stem.error.json" ]]; then
    echo "refusing duplicate $stem" >&2
    exit 1
  fi
  CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$python" "$closure/trajectory_equivalence.py" --phase run --mode dual --run-id "$index" \
    > "$closure/logs/$stem.log" 2>&1 || {
      echo "${stem}_FAILED" >&2
      tail -n 30 "$closure/logs/$stem.log" >&2
      exit 1
    }
  "$python" - "$closure" "$stem" <<'PY'
import json, sys
from pathlib import Path
root, stem = Path(sys.argv[1]), sys.argv[2]
single = json.loads((root / "analysis/single_01.json").read_text())
dual = json.loads((root / "analysis" / f"{stem}.json").read_text())
checks = []
for i in range(8):
    a = json.loads((root / "raw" / single["updates"][i]["forward_artifact"]).read_text())
    b = json.loads((root / "raw" / dual["updates"][i]["forward_artifact"]).read_text())
    fields = ("document_id", "input_token_ids_sha256", "teacher_token_ids_sha256",
              "target_positions_sha256", "teacher_target_tensor_hashes")
    metadata = all(a[field] == b[field] for field in fields)
    gates = all(a[name] == b[name] == "PASS" for name in
                ("REAL_RECURRENT_WRITEBACK_GATE", "BPTT_BOUNDARY_GATE"))
    sampled = len(a["trace"]) == len(b["trace"]) == 240 and set(a["trace"]) == set(b["trace"])
    exact = True
    if i == 0:
        trace_fields = ("consumed_prev_state_sha256", "pre_qdq_state_sha256",
                        "scale_sha256", "qcodes_sha256", "post_qdq_state_sha256")
        exact = sampled and all(a["trace"][key].get(field) == b["trace"][key].get(field)
                                for key in a["trace"] for field in trace_fields)
        exact = exact and a["loss_components"] == b["loss_components"]
    checks.append({"update": i+1, "metadata_exact": metadata, "gates_pass": gates,
                   "sampled_240": sampled, "theta0_student_exact": exact if i == 0 else None,
                   "pass": metadata and gates and sampled and exact})
result = {"stem": stem, "DUAL_GPU_FORWARD_GATE": "PASS" if all(x["pass"] for x in checks) else "FAIL",
          "checks": checks}
path = root / "analysis" / f"{stem}_forward_gate.json"
with path.open("x") as handle:
    json.dump(result, handle, indent=2, sort_keys=True)
    handle.write("\n")
if result["DUAL_GPU_FORWARD_GATE"] != "PASS":
    raise SystemExit(f"{stem} forward exact gate failed; stopping before next dual run")
PY
  echo "${stem}_COMPLETE"
done
