#!/usr/bin/env python3
"""CPU-only V1 containment and V2 Stage-A static preparation."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path


OLD_REPO = Path("/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen")
NEW_REPO = Path("/data/zypan/worktrees/qwen-gdn-formal-equivalent-mechanism-v2")
TASK = "QWEN_GDN_FORMAL_EQUIVALENT_MECHANISM_CLOSURE_V2"
ROOT = NEW_REPO / "experiments/qwen_gdn" / TASK
V1 = OLD_REPO / "experiments/qwen_gdn/QWEN_GDN_HADAMARD_VS_LEARNED_ROTATION_MECHANISM_CLOSURE_V1"
FORMAL = OLD_REPO / "experiments/qwen_gdn/QWEN_C5_C6_AIME26_CANONICAL60_V1"
DEPLOY = OLD_REPO / "experiments/qwen_gdn/QWEN_C5_C6_DEPLOYMENT_CLOSURE_V1"
TEACHER = Path("/data/zypan/worktrees/aime26-sglang-rotation-v1/artifacts/aime26_v2/official_sampling_81920/qwen/formal/shards")
SCORER = Path("/data/zypan/consolidation/GDN-quantization-qwen3090-20260921/experiments/shared/scoring/aime26_scorer_v4.py")

EXPECTED_HASHES = {
    "C5_FILE_SHA256": (DEPLOY / "deployment/C5/frozen_rotation_fp32.pt", "0266c5065267cf452d72cfe29a2a4d9a24fdfb6a89d7bec04b4d796c86f857c9"),
    "C6_FILE_SHA256": (DEPLOY / "deployment/C6/frozen_rotation_fp32.pt", "67e3731213ad3fb0c088c781ddd912aaaad5af75f9532b289442690a7b09f39c"),
    "FORMAL_CONFIG_SHA256": (FORMAL / "configs/formal_config.json", "557ac6670fe6e88606be226ccf0dea289ebd34dd4c2be54e7e54b9b6ee9e0836"),
    "HARDWARE_ASSIGNMENT_SHA256": (FORMAL / "configs/hardware_assignment.json", "75b5df954f64f606f30143df3bf5329e238a6eabe8253bfc287141b0443c475a"),
    "FROZEN_V4_SCORER_SHA256": (SCORER, "fa5dd904c7d1887df4b8c23613f37b8d438ea494cb72607b89be8e07d55a343b"),
    "FORMAL_RUNNER_SHA256": (FORMAL / "scripts/run_qwen_c5_c6_formal.py", "77b1e1f6dad8a0dd53a27cf5a47c02be86c26d2cf30732604ba6d3747e178f69"),
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def live_state_once() -> dict:
    ps = subprocess.run(["ps", "-eo", "pid,ppid,args"], text=True, stdout=subprocess.PIPE, check=True).stdout
    old_workers = [line.strip() for line in ps.splitlines() if "mechanism_capture.py" in line and str(V1) in line]
    wrappers = [line.strip() for line in ps.splitlines() if "run_mechanism_capture.sh" in line and str(V1) in line]
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True, stdout=subprocess.PIPE, check=True,
    ).stdout.splitlines()
    free_bytes = int(subprocess.run(["df", "-B1", "--output=avail", "/data"], text=True, stdout=subprocess.PIPE, check=True).stdout.splitlines()[-1])
    modes = []
    for line in old_workers:
        mode = re.search(r"--mode\s+(\S+)\s+--condition-set\s+(\S+)", line)
        if mode:
            modes.append(f"{mode.group(1)}:{mode.group(2)}")
    return {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "OLD_V1_GPU_ACTIVE": "YES" if old_workers else "NO",
        "OLD_V1_WRAPPER_ACTIVE": "YES" if wrappers else "NO",
        "OLD_V1_CURRENT_PHASE": sorted(set(modes)) or ["NONE"],
        "OLD_V1_ACTIVE_WORKERS": len(old_workers),
        "old_worker_processes": old_workers,
        "old_wrapper_processes": wrappers,
        "GPU_STATUS": {f"GPU{i}_STATUS": row for i, row in enumerate(gpu)},
        "DATA_FREE_GB": free_bytes / (1024 ** 3),
    }


def expected_rows():
    rows = []
    for path in sorted(TEACHER.glob("worker*/fp_state.jsonl")):
        rows.extend(load_jsonl(path))
    rows.sort(key=lambda row: (int(row["canonical_index"]), int(row["seed"])))
    if len(rows) != 60:
        raise RuntimeError(f"expected 60 teacher rows, found {len(rows)}")
    return rows


SAMPLE_RE = re.compile(r'"sample_id"\s*:\s*"([^"]+)"')


def completed_from_artifact(path: Path) -> set[str]:
    completed = set()
    with path.open(encoding="utf-8", errors="strict") as f:
        for line in f:
            match = SAMPLE_RE.search(line)
            if match:
                completed.add(match.group(1))
    return completed


def last_started_sample(path: Path) -> str | None:
    last = None
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if '"event": "start"' not in line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            last = f"{row['problem']}_seed{int(row['seed'])}"
    return last


def length_summary(rows):
    values = [len(row.get("generated_token_ids", [])) for row in rows]
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
        "values": values,
    }


def source_identity():
    observed = {}
    for name, (path, expected) in EXPECTED_HASHES.items():
        actual = sha(path)
        observed[name] = {"path": str(path), "expected": expected, "actual": actual, "bytes": path.stat().st_size, "match": actual == expected}
    gate = all(row["match"] for row in observed.values())
    if not gate:
        raise RuntimeError("frozen identity mismatch")
    return {"FORMAL_ARTIFACT_IDENTITY_GATE": "PASS", "artifacts": observed}


def probe_identity():
    c5 = json.loads((DEPLOY / "analysis/C5_functional_probe_revalidated.json").read_text())
    c6 = json.loads((DEPLOY / "analysis/C6_functional_probe.json").read_text())
    repeat = json.loads((DEPLOY / "analysis/repeatability_and_hadamard_guard_v2.json").read_text())
    c5_repeat = repeat["conditions"]["C5"]
    c6_repeat = repeat["conditions"]["C6"]
    # The functional probe hashes raw tensor bytes (core.tensor_sha256), while
    # the formal repeatability runner hashes the token-ID serialization
    # (runner.ids_sha256).  They intentionally have different digests for the
    # same IDs, so validate each frozen hashing domain separately.
    functional_same = c5["sample_token_sha256"] == c6["sample_token_sha256"] == "1ab158a4061d9f8bf3b75dcede826b0f5451bdcf9b58394310c0cf19e68a606d"
    formal_repeat_same = c5_repeat["run1"]["input_ids_sha256"] == c6_repeat["run1"]["input_ids_sha256"]
    gate = all([
        c5["FUNCTIONAL_PROBE_GATE"] == "PASS",
        c6["FUNCTIONAL_PROBE_GATE"] == "PASS",
        c5_repeat["REPEATABILITY_GATE"] == "PASS",
        c6_repeat["REPEATABILITY_GATE"] == "PASS",
        functional_same,
        formal_repeat_same,
    ])
    return {
        "PARITY_PROBE_GATE": "PASS" if gate else "FAIL",
        "PARITY_PROBE_SOURCE": "EXISTING_DEPLOYMENT_CLOSURE_NON_AIME_PROBE",
        "probe_identity": "TRAIN:first-row first 32 input tokens",
        "AIME_used": False,
        "input_token_count": 32,
        "input_hash": c5["sample_token_sha256"],
        "input_hash_algorithm": "core.tensor_sha256 over torch.long tensor bytes",
        "formal_repeatability_input_hash": c5_repeat["run1"]["input_ids_sha256"],
        "formal_repeatability_input_hash_algorithm": "formal runner ids_sha256 over token-ID serialization",
        "hash_domain_note": "The two input digests differ by design because the existing probes use different frozen serialization functions; C5/C6 identity is checked within each domain.",
        "teacher_target_hash": c5["teacher_target_sha256"],
        "repeatability_seed": c5_repeat["run1"]["seed"],
        "repeatability_generated_token_hash_C5": hashlib.sha256(json.dumps(c5_repeat["run1"]["generated_token_ids"], separators=(",", ":")).encode()).hexdigest(),
        "repeatability_generated_token_hash_C6": hashlib.sha256(json.dumps(c6_repeat["run1"]["generated_token_ids"], separators=(",", ":")).encode()).hexdigest(),
        "canonical_source_sha256": c5["canonical_source_sha256"],
        "rotation_source_sha256": c5["rotation_source_sha256"],
        "existing_gates": {
            "C5_FUNCTIONAL_PROBE_GATE": c5["FUNCTIONAL_PROBE_GATE"],
            "C6_FUNCTIONAL_PROBE_GATE": c6["FUNCTIONAL_PROBE_GATE"],
            "C5_REPEATABILITY_GATE": c5_repeat["REPEATABILITY_GATE"],
            "C6_REPEATABILITY_GATE": c6_repeat["REPEATABILITY_GATE"],
        },
    }


def write_static_outputs(live):
    for name in ("provenance", "analysis", "parity", "logs", "hashes", "reports", "scripts", "config"):
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    rows = expected_rows()
    by_id = {f"{row['problem_id']}_seed{int(row['seed'])}": row for row in rows}
    assignment = {i: [f"{row['problem_id']}_seed{int(row['seed'])}" for j, row in enumerate(rows) if j % 4 == i] for i in range(4)}
    completed = set()
    shard_status = {}
    oom_workers = []
    oom_samples = {}
    for worker in range(4):
        unit = V1 / f"raw/formal/controlled_all_shard{worker:02d}"
        status = json.loads((unit / "status.json").read_text())
        actual = completed_from_artifact(unit / "group_rows.jsonl")
        completed.update(actual)
        log = V1 / f"logs/worker{worker}_controlled_all.log"
        failure_sample = last_started_sample(log) if status.get("status") == "FAILED" else None
        if "OutOfMemoryError" in status.get("error", ""):
            oom_workers.append(worker)
            oom_samples[str(worker)] = failure_sample
        shard_status[str(worker)] = {
            "expected": assignment[worker],
            "artifact_completed": sorted(actual),
            "status": status.get("status"),
            "status_completed": status.get("completed"),
            "failure_sample": failure_sample,
            "error_class": status.get("error", "").split(":", 1)[0] or None,
        }
    expected = set(by_id)
    missing = sorted(expected - completed)
    if len(completed) != 48 or len(missing) != 12:
        raise RuntimeError(f"controlled completeness changed: {len(completed)}/60, missing {len(missing)}")

    recurrent = {}
    recurrent_completed = 0
    for condition in ("baseline", "learned"):
        for worker in range(4):
            path = V1 / f"raw/formal/recurrent_{condition}_shard{worker:02d}/status.json"
            key = f"{condition}_shard{worker:02d}"
            if path.is_file():
                status = json.loads(path.read_text())
                recurrent[key] = {k: status.get(k) for k in ("status", "completed", "total", "last", "error")}
                recurrent_completed += int(status.get("completed", 0))
            else:
                recurrent[key] = {"status": "NOT_STARTED", "completed": 0, "total": 15}

    snapshot = {
        **live,
        "V1_CURRENT_RECURRENT_ALLOWED_TO_FINISH": True,
        "V1_PHASE_BOUNDARY_CONTAINMENT": {
            "method": "SIGSTOP_WRAPPER_PID_ONLY",
            "signal_scope": "four wrapper shell PIDs only; no process-group signal",
            "wrapper_pids": [39368, 39370, 39377, 39386],
            "current_worker_pids_explicitly_untouched": [3571, 16958, 13717, 10627],
            "command_return_code": 0,
            "effect": "current recurrent-baseline workers continue; stopped shells cannot launch recurrent learned after the current worker exits",
            "follow_up_rule": "after every current baseline worker has naturally exited, retire the stopped wrapper shells without SIGCONT; never resume the V1 stage sequence",
        },
        "V1_CONTROLLED_EXPECTED": 60,
        "V1_CONTROLLED_COMPLETED": len(completed),
        "V1_CONTROLLED_MISSING": len(missing),
        "V1_CONTROLLED_COMPLETENESS_GATE": "PASS" if not missing else "FAIL",
        "CONTROLLED_COMPLETED_SAMPLE_IDS": sorted(completed),
        "CONTROLLED_MISSING_SAMPLE_IDS": missing,
        "controlled_missing_details": [{"sample_id": sample, "shard": next(i for i, ids in assignment.items() if sample in ids), "generated_token_count": len(by_id[sample].get("generated_token_ids", []))} for sample in missing],
        "completed_length_summary": length_summary([by_id[sample] for sample in sorted(completed)]),
        "missing_length_summary": length_summary([by_id[sample] for sample in missing]),
        "CONTROLLED_MISSINGNESS": "OOM_ASSOCIATED_NONRANDOM_POSSIBLE",
        "missingness_note": "Each failed shard stops at its OOM sample, so that sample and later shard-assigned samples are absent; missing-at-random is not assumed and no correction is applied.",
        "V1_RECURRENT_EXPECTED": 120,
        "V1_RECURRENT_COMPLETED_AT_SNAPSHOT": recurrent_completed,
        "V1_RECURRENT_CURRENT_STATUS": recurrent,
        "V1_OOM_COUNT": len(oom_workers),
        "V1_OOM_WORKERS": oom_workers,
        "V1_OOM_SAMPLE_AT_FAILURE": oom_samples,
        "controlled_shards": shard_status,
        "V1_FORMAL_PATH_STATUS": "PARTIAL_MISMATCH_MATMUL_ASSOCIATION_DTYPE_DUPLICATED_QUANTIZER",
        "V1_SCIENTIFIC_STATUS": "INCOMPLETE_DESCRIPTIVE_DIAGNOSTIC_ONLY",
        "V1_RETRY_STARTED": False,
        "V1_MISSING_SAMPLE_REPAIR_STARTED": False,
        "V1_CONTROLLED_RETRY_STARTED": False,
        "V1_OOM_SAMPLE_RETRY_STARTED": False,
        "V1_NEW_DOWNSTREAM_PHASE_AFTER_RECURRENT": False,
    }
    atomic_json(ROOT / "provenance/V1_STATUS_SNAPSHOT.json", snapshot)

    identity = source_identity()
    identity.update({
        "V2_WORKTREE": str(NEW_REPO),
        "V2_WORKSPACE_KIND": "independent shared-object clone because installed Git lacks the worktree subcommand",
        "V2_BRANCH": "exp/qwen-gdn-formal-equivalent-mechanism-v2",
        "V2_BASE_COMMIT": "fa7918d82c144a30ba3459114abb53e70df73d98",
        "V2_WORKTREE_CLEAN_AT_CREATION": True,
        "large_model_data_duplicated": False,
        "git_object_alternate": str(OLD_REPO / ".git/objects"),
    })
    atomic_json(ROOT / "provenance/workspace_and_artifact_identity.json", identity)

    formal_runner = FORMAL / "scripts/run_qwen_c5_c6_formal.py"
    rotation = {
        "gate": "PASS_STATIC_SOURCE_AND_EXISTING_RUNTIME_PROBE",
        "source_file": str(formal_runner),
        "source_file_sha256": sha(formal_runner),
        "function": "QwenFrozenRotationPatch._transform",
        "source_lines": [248, 254],
        "exact_expression": "value.float().matmul(hadamard(128, dtype=torch.float32, device=value.device)).matmul(delta)",
        "P0_input_dtype": "torch.bfloat16 expected from formal model load; must be captured at Stage A",
        "value_float_dtype": "torch.float32",
        "H_dtype": "torch.float32",
        "Delta_dtype": "torch.float32",
        "P1_after_H_dtype": "torch.float32",
        "P2_after_Delta_dtype": "torch.float32",
        "quantizer_input_relation": "rotation affects q/k; recurrent kernel produces cache state subsequently passed unchanged to quantize_c128",
        "existing_runtime_probe_rotated_qk_dtype": "torch.float32",
        "CAPTURE_IS_READ_ONLY_REQUIRED": True,
        "forbidden_fusion": "R=H@Delta; value@R",
    }
    atomic_json(ROOT / "provenance/formal_rotation_path.json", rotation)
    quantizer = {
        "gate": "PASS_STATIC_SOURCE_AND_EXISTING_RUNTIME_PROBE",
        "source_file": str(formal_runner), "source_file_sha256": sha(formal_runner),
        "formal_function_identity": "quantize_c128", "source_lines": [269, 285],
        "GROUP_SIZE": 128, "GROUP_AXIS": "state dim=-2 (K axis)", "QMIN": -127, "QMAX": 127,
        "SYMMETRIC": True, "ZERO_POINT": 0, "EPSILON": 1e-12,
        "SCALE_FORMULA": "state.detach().float().abs().amax(dim=-2, keepdim=True).clamp_min(EPS) / 127.0",
        "ROUNDING": "torch.round", "CLAMP": "[-127,127]", "DEQUANT": "codes * scale",
        "existing_runtime_probe_dtypes": {"pre_qdq_state": "torch.float32", "scale": "torch.float32", "qcodes": "torch.float32", "dequant": "torch.float32"},
        "V2_SHARED_PATH_RULE": "Stage A must call this exact function; no quantizer arithmetic is reimplemented",
    }
    atomic_json(ROOT / "provenance/formal_quantizer_path.json", quantizer)
    writeback = {
        "gate": "PASS_STATIC_SOURCE_AND_EXISTING_RUNTIME_PROBE",
        "source_file": str(formal_runner), "source_file_sha256": sha(formal_runner),
        "formal_function": "quantize_c128", "writeback_line": 280,
        "writeback_expression": "state.copy_(dequant.to(state.dtype))",
        "writeback_tensor": "get_state(cache, layer_idx)",
        "writeback_dtype": "same as cache state; existing deployment probe observed torch.float32",
        "decode_call_line": 421, "quantize_call_line": 424,
        "next_read_location": "next loop iteration passes the same past_key_values cache to model(input_ids=current, past_key_values=past, use_cache=True)",
        "existing_probe_writeback_exact_all_24_layers": True,
    }
    atomic_json(ROOT / "provenance/formal_writeback_path.json", writeback)
    probe = probe_identity()
    atomic_json(ROOT / "provenance/parity_probe_identity.json", probe)

    budget = {
        "checked_utc": live["checked_utc"],
        "DATA_FREE_GB": live["DATA_FREE_GB"],
        "MIN_FREE_SPACE_BEFORE_STAGE_A_GB": 25,
        "PROJECTED_STAGE_A_ARTIFACT_GB": 1.0,
        "projection_basis": "three selected layers, P0-P11 hashes/metrics and only essential cloned tensors for four repeat runs; no 60-sample dense dump",
        "DISK_GATE": "PASS" if live["DATA_FREE_GB"] >= 25 else "FAIL",
    }
    atomic_json(ROOT / "analysis/disk_budget.json", budget)

    audit = """# V1 orchestration audit

`run_mechanism_capture.sh` uses `set -euo pipefail`, but wraps the full three-stage compound list in:

```bash
{
  run_stage controlled all
  run_stage recurrent baseline
  run_stage recurrent learned
} || code=$?
```

In Bash, `errexit` is disabled for commands/functions whose enclosing compound command is tested by `||`. Therefore the failing `python | tee` pipeline did not abort `run_stage`; execution reached its unconditional `echo COMPLETE`, the function returned zero, and the wrapper proceeded. `pipefail` existed but its nonzero status was effectively overwritten. There was also no explicit worker-exit/artifact-completeness gate between stages.

`WHY_DID_WRAPPER_CONTINUE_AFTER_OOM = PIPELINE_STATUS_MASKED_BY_COMPOUND_OR_CONTEXT_AND_MISSING_COMPLETENESS_GATE`

The running V1 source is not modified. V2 freezes an explicit rule: any nonzero worker exit or `actual_success != expected_success` makes the stage FAIL and forbids launching the next stage.

## Live phase-boundary containment

The one allowed live-state check found four recurrent-baseline Python workers under four independent tmux wrapper shells. To preserve the already-started scientific computation while preventing the next launch, only wrapper PIDs `39368,39370,39377,39386` received `SIGSTOP` (return code 0). No signal was sent to worker PIDs `3571,16958,13717,10627`, to their process groups, or to tmux. When each current worker naturally exits, its stopped wrapper cannot advance to `recurrent learned`. The wrappers must later be retired without `SIGCONT`; V1 must never be resumed.

`V1_CURRENT_RECURRENT_ALLOWED_TO_FINISH = TRUE`

`V1_NEW_DOWNSTREAM_PHASE_AFTER_RECURRENT = FALSE`
"""
    (ROOT / "analysis/v1_orchestration_audit.md").write_text(audit, encoding="utf-8")
    atomic_json(ROOT / "config/v2_orchestration_rule.json", {
        "rule": [
            "if any WORKER_EXIT_CODE != 0: STAGE_GATE=FAIL; DO_NOT_START_NEXT_STAGE",
            "if ACTUAL_SUCCESS_COUNT != EXPECTED_SAMPLE_COUNT: STAGE_GATE=FAIL; DO_NOT_START_NEXT_STAGE",
            "if ARTIFACT_COMPLETENESS != PASS: STAGE_GATE=FAIL; DO_NOT_START_NEXT_STAGE",
        ],
        "partial_stage_may_continue": False,
        "running_v1_boundary_containment": {
            "current_recurrent_baseline_allowed_to_finish": True,
            "new_downstream_phase_after_recurrent": False,
            "never_resume_stopped_v1_wrappers": True,
        },
    })

    status = "NOT_RUN_OLD_CAPTURE_ACTIVE" if live["OLD_V1_GPU_ACTIVE"] == "YES" else "NOT_RUN"
    for name in ("formal_self_repeat", "instrumented_self_repeat", "formal_vs_instrumented_summary", "qcode_parity"):
        atomic_json(ROOT / f"parity/{name}.json", {"status": status, "FORMAL_REPLAY_PARITY_CLOSURE": status, "MECHANISM_REPLAY_AUTHORIZED": "NO"})
    with (ROOT / "parity/tensor_parity.csv").open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=["condition", "node", "status", "shape", "dtype", "device", "exact_equal", "sha256_a", "sha256_b", "max_abs_diff", "relative_l2", "num_equal", "num_different", "fraction_different"]).writeheader()

    static_gate = all([identity["FORMAL_ARTIFACT_IDENTITY_GATE"] == "PASS", probe["PARITY_PROBE_GATE"] == "PASS", budget["DISK_GATE"] == "PASS"])
    report = f"""# Stage A status report

This execution stopped at the required old-capture gate. No model was loaded and no CUDA allocation was made by V2.

1. V1 controlled completeness: **{len(completed)}/60; FAIL**.
2. Missing samples: `{', '.join(missing)}`. They are the OOM sample and subsequent assigned samples in failed shards; `OOM_ASSOCIATED_NONRANDOM_POSSIBLE`.
3. Wrapper continuation: Bash `errexit` was neutralized by the compound-list `||` context; the failing pipeline status was overwritten by a later successful echo, with no completeness gate.
4. V1 is `INCOMPLETE_DESCRIPTIVE_DIAGNOSTIC_ONLY` due to 12 missing controlled units plus formal-path matmul/dtype/quantizer mismatch.
5. V2 uses the independent shared-object clone `{NEW_REPO}`, branch `exp/qwen-gdn-formal-equivalent-mechanism-v2`, base `fa7918d82c144a30ba3459114abb53e70df73d98`; clean at creation and no model data duplicated.
6. Formal C5/C6 rotation: `value.float().matmul(H_fp32).matmul(Delta_fp32)` in `QwenFrozenRotationPatch._transform`.
7. Formal C128 quantizer: the unchanged `quantize_c128` function, state K axis `-2`, symmetric `[-127,127]`, scale `absmax.clamp_min(1e-12)/127`, `torch.round`, dequant `codes*scale`.
8. Formal writeback: `state.copy_(dequant.to(state.dtype))`; the same cache is passed into the next decode call.
9. Probe reuse: `{probe['PARITY_PROBE_SOURCE']}`, identity `TRAIN:first-row`, input hash `{probe['input_hash']}`; gate `{probe['PARITY_PROBE_GATE']}`.
10–20. Formal/instrumented repeat and P0–P11 parity: **{status}**.
21. `FIRST_FORMAL_PARITY_DIVERGENCE = NOT_EVALUATED`.
22. `FORMAL_REPLAY_PARITY_CLOSURE = {status}`.
23. `MECHANISM_REPLAY_AUTHORIZED = NO`.

## Required terminal state

- `OLD_V1_ACTIVE = {live['OLD_V1_GPU_ACTIVE']}`
- `V2_STATIC_PREPARATION = {'PASS' if static_gate else 'FAIL'}`
- `V2_GPU_EXECUTION = {status}`
- `V1_RETRY_STARTED = FALSE`
- `V1_MISSING_SAMPLE_REPAIR_STARTED = FALSE`
- `V1_CURRENT_RECURRENT_ALLOWED_TO_FINISH = TRUE`
- `V1_CONTROLLED_RETRY_STARTED = FALSE`
- `V1_OOM_SAMPLE_RETRY_STARTED = FALSE`
- `V1_NEW_DOWNSTREAM_PHASE_AFTER_RECURRENT = FALSE`
- `NEW_AIME_GENERATION_STARTED = FALSE`
- `STRICT_FP_FANOUT_STARTED = FALSE`
- `RECURRENT_REPLAY_STARTED = FALSE`
- `MULTI_HORIZON_STARTED = FALSE`
- `PRE_DIVERGENCE_STARTED = FALSE`
- `LAYER_INTERVENTION_STARTED = FALSE`
- `NEW_ROTATION_TRAINING_STARTED = FALSE`
- `C7_STARTED = FALSE`
"""
    (ROOT / "reports/STAGE_A_FINAL_REPORT.md").write_text(report, encoding="utf-8")
    return {"snapshot": snapshot, "identity": identity, "probe": probe, "budget": budget, "static_gate": "PASS" if static_gate else "FAIL", "parity_status": status}


def write_hashes():
    target = ROOT / "hashes/SHA256SUMS"
    rows = []
    for path in sorted(p for p in ROOT.rglob("*") if p.is_file() and p != target and "logs" not in p.relative_to(ROOT).parts):
        rows.append(f"{sha(path)}  {path.relative_to(ROOT)}")
    target.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-state-json", type=Path, help="Use the already captured one-shot live state; do not execute a second live check")
    args = parser.parse_args()
    live = json.loads(args.live_state_json.read_text(encoding="utf-8")) if args.live_state_json else live_state_once()
    result = write_static_outputs(live)
    write_hashes()
    print(json.dumps({
        "OLD_V1_ACTIVE": live["OLD_V1_GPU_ACTIVE"],
        "V2_STATIC_PREPARATION": result["static_gate"],
        "V2_GPU_EXECUTION": result["parity_status"],
        "FORMAL_REPLAY_PARITY_CLOSURE": result["parity_status"],
        "V1_CONTROLLED_COMPLETED": result["snapshot"]["V1_CONTROLLED_COMPLETED"],
        "V1_CONTROLLED_MISSING": result["snapshot"]["V1_CONTROLLED_MISSING"],
        "missing": result["snapshot"]["CONTROLLED_MISSING_SAMPLE_IDS"],
    }, indent=2))


if __name__ == "__main__":
    main()
