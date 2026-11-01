#!/usr/bin/env python3
"""Finalize the fail-closed GPU smoke after its pre-token identity gate failed."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path("/data/zypan/worktrees/qwen-gdn-formal-equivalent-mechanism-v2/experiments/qwen_gdn/QWEN_GDN_STAGE_B_GPU_STREAMING_SMOKE_V3")
EXECUTION_HEAD = "1ff3e4803875e08b7a27d13869e62ef8837b4075"
ACTUAL = {
    "aime26_01_seed1": {
        "teacher_tokens": 9572,
        "prefix_sha256": "96f073d722f052c40859bf9341fff4932df640fc0f13aabb9aef3bffaedf8221",
        "full_sha256": "5c06393606cdf51f14e02e1c0c3d387101b1e49590ab4d1c37e6607ed0e8bd9c",
    },
    "aime26_01_seed2": {
        "teacher_tokens": 13887,
        "prefix_sha256": "adc97bde9131a910ff6c4421a1ebcbe4a6ecb0ee52922c92312065c224525b48",
        "full_sha256": "d21664b87826470380b6ba58267be2ee22dfca82db9ba502dc32d45b0ec51536",
    },
}


def dump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def sha(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


frozen = json.loads((ROOT / "smoke_identity.json").read_text())
comparisons = []
for sample_id, actual in ACTUAL.items():
    expected = frozen["samples"][sample_id]
    comparisons.append({
        "sample_id": sample_id,
        "expected": expected,
        "actual": actual,
        "teacher_tokens_match": expected["teacher_tokens"] == actual["teacher_tokens"],
        "prefix_sha256_match": expected["prefix_sha256"] == actual["prefix_sha256"],
        "full_sha256_match": expected["full_sha256"] == actual["full_sha256"],
        "expected_prefix_length": len(expected["prefix_sha256"]),
        "actual_prefix_length": len(actual["prefix_sha256"]),
    })

incident = {
    "incident": "SMOKE_IDENTITY_GATE_FAILURE",
    "classification": "PRE_TOKEN_FROZEN_IDENTITY_TRANSCRIPTION_ERROR",
    "execution_head": EXECUTION_HEAD,
    "exit_code": 1,
    "model_loaded": True,
    "tokens_processed": 0,
    "sample_artifacts_committed": 0,
    "cause": "Both frozen prefix SHA256 literals contain one duplicated terminal-wrap character and are 65 rather than 64 hex characters.",
    "comparisons": comparisons,
    "retry_started": False,
    "protocol_action": "FAIL_CLOSED_STOP",
    "FULL60_STARTED": False,
}
dump(ROOT / "validation/smoke_identity_failure.json", incident)
dump(ROOT / "runtime/timing.json", {
    "status": "NOT_REACHED_IDENTITY_GATE_FAILED_BEFORE_FIRST_TOKEN",
    "tokens_processed": 0,
    "samples_completed": 0,
    "THROUGHPUT_MEASUREMENT": "NOT_RUN",
    "PROJECTED_FULL60_RUNTIME": "NOT_GENERATED",
    "PROJECTED_FULL60_GPU_HOURS": "NOT_GENERATED",
})
dump(ROOT / "runtime/memory.json", {
    "status": "NOT_REACHED_IDENTITY_GATE_FAILED_BEFORE_PER_SAMPLE_MEASUREMENT",
    "sample_measurements": [],
    "MEMORY_CREEP": "NOT_MEASURED",
    "MEMORY_GATE": "NOT_RUN",
})
dump(ROOT / "artifacts/size_summary.json", {
    "status": "NOT_REACHED_IDENTITY_GATE_FAILED_BEFORE_ARTIFACT_COMMIT",
    "sample_artifact_count": 0,
    "ARTIFACT_SIZE_MEASUREMENT": "NOT_RUN",
})
dump(ROOT / "projection/full60_resource_projection.json", {
    "status": "NOT_GENERATED_DUE_TO_SMOKE_IDENTITY_GATE_FAILURE",
    "PROJECTED_FULL60_RUNTIME": "NOT_GENERATED",
    "PROJECTED_FULL60_GPU_HOURS": "NOT_GENERATED",
    "PROJECTED_FULL60_FINAL_ARTIFACT_GB": "NOT_GENERATED",
    "PROJECTED_FULL60_TEMPORARY_GB": "NOT_GENERATED",
    "PROJECTED_FREE_SPACE_AFTER_FULL60_GB": "NOT_GENERATED",
    "FULL60_RESOURCE_READINESS": "FAIL",
    "FULL60_STARTED": False,
})
dump(ROOT / "validation/semantic_gates.json", {
    "SOURCE_GATE": "PASS",
    "SMOKE_IDENTITY_GATE": "FAIL",
    "COMMON_TOKEN_SEQUENCE_GATE": "NOT_RUN",
    "COMMON_FP_STATE_INPUT_GATE": "NOT_RUN",
    "STATE_RESET_TO_COMMON_FP_EACH_TOKEN": "NOT_RUN",
    "PERSISTENT_QDQ_WRITEBACK": "NOT_RUN",
    "FORMAL_ROTATION_PATH_GATE": "NOT_RUN",
    "FORMAL_QUANTIZER_SHARED_PATH_GATE": "NOT_RUN",
    "reason": "Fail-closed identity check stopped execution before the first formal forward token.",
})
dump(ROOT / "validation/atomic_test.json", {
    "ATOMIC_ARTIFACT_GATE": "NOT_RUN",
    "reason": "Pre-token identity gate failed; no reducer artifact was permitted.",
})
dump(ROOT / "validation/fail_closed_test.json", {
    "FAIL_CLOSED_ORCHESTRATION_GATE": "PASS_OBSERVED",
    "observed_case": "identity mismatch returned non-zero, committed zero sample artifacts, launched no downstream stage",
    "worker_exit_code": 1,
    "tokens_processed": 0,
    "sample_artifacts_committed": 0,
    "downstream_launch_authorized": False,
})
gates = {
    "SOURCE_GATE": "PASS",
    "SMOKE_IDENTITY_GATE": "FAIL",
    "COMMON_TOKEN_SEQUENCE_GATE": "NOT_RUN",
    "COMMON_FP_STATE_INPUT_GATE": "NOT_RUN",
    "STATE_RESET_TO_COMMON_FP_EACH_TOKEN": "NOT_RUN",
    "PERSISTENT_QDQ_WRITEBACK": "NOT_RUN",
    "FORMAL_ROTATION_PATH_GATE": "NOT_RUN",
    "FORMAL_QUANTIZER_SHARED_PATH_GATE": "NOT_RUN",
    "MEMORY_GATE": "NOT_RUN",
    "THROUGHPUT_MEASUREMENT": "NOT_RUN",
    "ARTIFACT_SIZE_MEASUREMENT": "NOT_RUN",
    "ATOMIC_ARTIFACT_GATE": "NOT_RUN",
    "FAIL_CLOSED_ORCHESTRATION_GATE": "PASS_OBSERVED",
    "PROJECTED_FULL60_RUNTIME": "NOT_GENERATED",
    "PROJECTED_FULL60_GPU_HOURS": "NOT_GENERATED",
    "PROJECTED_FULL60_ARTIFACT_GB": "NOT_GENERATED",
    "FULL60_RESOURCE_READINESS": "FAIL",
    "GPU_STREAMING_SMOKE_V3": "FAIL",
    "FIRST_FAILED_GATE": "SMOKE_IDENTITY_GATE",
    "MODEL_LOADED": True,
    "TOKENS_PROCESSED": 0,
    "SAMPLE_ARTIFACT_COUNT": 0,
    "RETRY_STARTED": False,
    "NEXT_STAGE": "STOPPED_IDENTITY_FAILURE",
    "FULL60_STARTED": False,
    "NEW_AIME_GENERATION": False,
    "RECURRENT_REPLAY_STARTED": False,
    "MULTI_HORIZON_STARTED": False,
    "PRE_DIVERGENCE_STARTED": False,
    "LAYER_INTERVENTION_STARTED": False,
    "C7_STARTED": False,
    "NEW_TRAINING_STARTED": False,
}
dump(ROOT / "validation/final_gates.json", gates)

report = """# FINAL GPU STREAMING SMOKE V3 REPORT

## Verdict

`GPU_STREAMING_SMOKE_V3 = FAIL`

The fail-closed pre-token identity gate stopped the run. Both frozen 32-token prefix SHA256 literals were accidentally recorded with one duplicated terminal-wrap character (65 hex characters instead of 64). The formal model loaded, but **zero forward tokens ran**, **zero sample artifacts were committed**, and **no retry or downstream stage was started**.

Q1. GPU smoke preserved formal semantics? **NOT RUN**; identity failed before the first token.

Q2. Rotation path still identical? Frozen artifacts and runner identities were verified, but the runtime rotation gate was **NOT RUN**.

Q3. Quantizer path still identical? **NOT RUN**.

Q4. State-reset semantics preserved? **NOT RUN**; no recurrent token executed.

Q5. GPU peak memory? **NOT MEASURED** under the required per-sample protocol.

Q6. Memory creep? **NOT MEASURED**.

Q7. Tokens/s? **NOT MEASURED** (0 tokens processed).

Q8. Full60 projected runtime? **NOT GENERATED**.

Q9. Full60 projected GPU-hours? **NOT GENERATED**.

Q10. Final artifact projected size? **NOT GENERATED**.

Q11. Temporary peak size? **NOT GENERATED**.

Q12. Disk/resource gate? **FAIL / NOT REACHED**.

Q13. Atomic artifact gate? **NOT RUN**; no artifact was allowed past identity validation.

Q14. Fail-closed orchestration? **PASS_OBSERVED**: the process returned non-zero and launched no downstream work.

Q15. Was full60 run? **NO**. `FULL60_STARTED = FALSE`.

## Required terminal state

`FIRST_FAILED_GATE = SMOKE_IDENTITY_GATE`

`RETRY_STARTED = FALSE`

`NEXT_STAGE = STOPPED_IDENTITY_FAILURE`

All forbidden follow-on flags are `FALSE`.
"""
(ROOT / "reports").mkdir(parents=True, exist_ok=True)
(ROOT / "reports/FINAL_GPU_SMOKE_REPORT.md").write_text(report, encoding="utf-8")

manifest = ROOT / "hashes/SHA256SUMS"
manifest.parent.mkdir(parents=True, exist_ok=True)
paths = sorted(p for p in ROOT.rglob("*") if p.is_file() and p != manifest and "__pycache__" not in p.parts)
manifest.write_text("".join(f"{sha(p)}  {p.relative_to(ROOT)}\n" for p in paths), encoding="utf-8")
print(json.dumps({"GPU_STREAMING_SMOKE_V3": "FAIL", "FIRST_FAILED_GATE": "SMOKE_IDENTITY_GATE", "tokens_processed": 0, "full60_started": False, "hashed_artifacts": len(paths)}, sort_keys=True))
