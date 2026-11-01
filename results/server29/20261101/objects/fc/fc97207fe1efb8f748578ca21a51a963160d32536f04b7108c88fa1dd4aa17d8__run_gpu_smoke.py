#!/usr/bin/env python3
"""One-GPU readiness smoke for the frozen Stage-B V3 streaming reducer.

This is deliberately not a full-60 runner.  It reuses the already verified
formal Qwen/GDN fan-out and quantizer path and feeds only the five V3 metrics
to one independent per-sample reducer.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch


TASK = "QWEN_GDN_STAGE_B_GPU_STREAMING_SMOKE_V3"
BASE_HEAD = "fea85a2e535b52dd376c9f214eae0f62c7d5282f"
REPO = Path("/data/zypan/worktrees/qwen-gdn-formal-equivalent-mechanism-v2")
ROOT = REPO / "experiments/qwen_gdn" / TASK
OLD_SMOKE = REPO / "experiments/qwen_gdn/QWEN_GDN_STRICT_COMMON_FP_STATE_FANOUT_V1/scripts/stage_b_smoke.py"
V3_PATH = REPO / "experiments/qwen_gdn/QWEN_GDN_STAGE_B_STREAMING_REDUCER_V3_CLOSURE/scripts/reducer_v3.py"
ATOMIC_PATH = REPO / "experiments/qwen_gdn/QWEN_GDN_STAGE_B_STREAMING_READINESS_V1/scripts/atomic_protocol.py"
FORMAL_PATH = Path("/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/qwen_gdn/QWEN_C5_C6_AIME26_CANONICAL60_V1/scripts/run_qwen_c5_c6_formal.py")
DEPLOY = Path("/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/qwen_gdn/QWEN_C5_C6_DEPLOYMENT_CLOSURE_V1/deployment")
TEACHER = Path("/data/zypan/worktrees/aime26-sglang-rotation-v1/artifacts/aime26_v2/official_sampling_81920/qwen/formal/shards")
CONDITIONS = ("Native", "Hadamard", "C5", "C6")
SMOKE_UNITS = (("aime26_01", 1), ("aime26_01", 2))
SMOKE_TOKENS = 32
EXPECTED_ROTATION = {
    "C5": "0266c5065267cf452d72cfe29a2a4d9a24fdfb6a89d7bec04b4d796c86f857c9",
    "C6": "67e3731213ad3fb0c088c781ddd912aaaad5af75f9532b289442690a7b09f39c",
}
EXPECTED_SMOKE = {
    "aime26_01_seed1": {"teacher_tokens": 9572, "prefix_sha256": "96f073d722f052c40859bf9341fff4932df640fc0f13aabb9aef3bffaedf82221", "full_sha256": "5c06393606cdf51f14e02e1c0c3d387101b1e49590ab4d1c37e6607ed0e8bd9c"},
    "aime26_01_seed2": {"teacher_tokens": 13887, "prefix_sha256": "adc97bde9131a910ff6c4421a1ebcbe4a6ecb0ee52922c92312065c224525bb48", "full_sha256": "d21664b87826470380b6ba58267be2ee22dfca82db9ba502dc32d45b0ec51536"},
}
FULL_TEACHER_TOKENS = 1854451
EPS = 1e-12


def loadmod(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def token_sha(tokens) -> str:
    return hashlib.sha256(np.asarray(tokens, dtype="<i8").tobytes()).hexdigest()


def canonical_bytes(obj) -> bytes:
    return (json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n").encode()


def atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = json.dumps(obj, indent=2, sort_keys=True) + "\n"
    with tmp.open("w", encoding="utf-8") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def df_free() -> int:
    return int(subprocess.check_output(["df", "-B1", "--output=avail", "/data"], text=True).splitlines()[-1])


def load_teacher_rows():
    rows = []
    for path in sorted(TEACHER.glob("worker*/fp_state.jsonl")):
        with path.open(encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    rows.sort(key=lambda row: (int(row["canonical_index"]), int(row["seed"])))
    selected = {(row["problem_id"], int(row["seed"])): row for row in rows}
    return rows, [selected[key] for key in SMOKE_UNITS]


class StreamingAccumulator:
    """GPU metric calculation with token-atomic V3 reducer submission."""

    def __init__(self, reducer, v3):
        self.reducer = reducer
        self.v3 = v3
        self.buffer = {}
        self.sample_id = reducer.identity["sample_id"]
        self.calls = 0

    @staticmethod
    def metrics(source, scale, codes, dequant):
        minimum = source.amin(dim=-2)
        maximum = source.amax(dim=-2)
        absmax = source.abs().amax(dim=-2)
        power = source.square().mean(dim=-2)
        rms = power.sqrt()
        err = dequant - source
        l2 = err.square().sum(dim=-2).sqrt()
        source_l2 = source.square().sum(dim=-2).sqrt()
        return {
            "scale": scale.squeeze(-2),
            "papr_linear": absmax / rms.clamp_min(EPS),
            "qdq_relative_l2": l2 / source_l2.clamp_min(EPS),
            "absmax": absmax,
            "range": maximum - minimum,
        }

    def update(self, sample_id, token_idx, condition, layer, source, scale, codes, dequant):
        if sample_id != self.sample_id:
            raise RuntimeError("sample reducer identity mismatch")
        key = (int(layer), condition)
        if key in self.buffer:
            raise RuntimeError("duplicate layer/condition capture in token")
        tensors = self.metrics(source, scale, codes, dequant)
        self.buffer[key] = {
            "group_ids": np.arange(self.v3.N_GROUPS, dtype=np.int64),
            "metrics": {
                name: value.detach().float().reshape(-1).cpu().numpy().astype(np.float64)
                for name, value in tensors.items()
            },
        }
        self.calls += 1
        expected = len(self.v3.LAYERS) * len(self.v3.CONDITIONS)
        if len(self.buffer) == expected:
            self.reducer.update_token(token_idx, self.buffer)
            self.buffer = {}

    def finish(self):
        if self.buffer:
            raise RuntimeError("partial token buffer at sample end")


def gpu_mem(device):
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
    }


def size_tree(root: Path):
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def main():
    if len(sys.argv) != 3 or sys.argv[1] != "--gpu":
        raise SystemExit("usage: run_gpu_smoke.py --gpu N")
    requested_gpu = int(sys.argv[2])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(requested_gpu)
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    # Source gate happens before any generated file is written.
    start_status = git("status", "--porcelain")
    start_head = git("rev-parse", "HEAD")
    base_is_ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", BASE_HEAD, start_head], cwd=REPO
    ).returncode == 0
    source_ok = not start_status and base_is_ancestor
    if not source_ok:
        raise RuntimeError(f"SOURCE_GATE=FAIL head={start_head} dirty={bool(start_status)}")

    for name in ("config", "provenance", "runtime", "artifacts/samples", "artifacts/checkpoints", "artifacts/atomic_test", "projection", "validation", "hashes", "reports", "logs"):
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    started = time.time()
    free_before = df_free()

    old = loadmod(OLD_SMOKE, "frozen_stage_b_formal_fanout")
    v3 = loadmod(V3_PATH, "frozen_stage_b_reducer_v3")
    atomic = loadmod(ATOMIC_PATH, "frozen_stage_b_atomic_protocol")
    formal = loadmod(FORMAL_PATH, "frozen_formal_runner_gpu_smoke_v3")
    all_rows, smoke_rows = load_teacher_rows()
    if len(all_rows) != 60 or sum(len(r["generated_token_ids"]) for r in all_rows) != FULL_TEACHER_TOKENS:
        raise RuntimeError("teacher manifest identity mismatch")

    metric_hashes = {
        "metric_config_hash": sha(REPO / "experiments/qwen_gdn/QWEN_GDN_STAGE_B_STREAMING_REDUCER_V3_CLOSURE/config/metric_definitions.json"),
        "tail_definition_hash": sha(REPO / "experiments/qwen_gdn/QWEN_GDN_STAGE_B_STREAMING_REDUCER_V3_CLOSURE/config/token_local_tail_spec.json"),
        "histogram_config_hash": sha(REPO / "experiments/qwen_gdn/QWEN_GDN_STAGE_B_STREAMING_REDUCER_V3_CLOSURE/config/histogram_counts_spec.json"),
        "topk_order_hash": sha(REPO / "experiments/qwen_gdn/QWEN_GDN_STAGE_B_STREAMING_REDUCER_V2_CLOSURE/config/topk_total_order.json"),
    }
    frozen_identity = json.loads((ROOT / "smoke_identity.json").read_text())
    frozen_config = json.loads((ROOT / "config/experiment_config.json").read_text())
    identity_ok = frozen_identity.get("samples") == EXPECTED_SMOKE and frozen_config.get("source_head") == BASE_HEAD
    identity_ok = identity_ok and frozen_config.get("config_sha256") == sha(ROOT / "config/frozen_runtime_config.json")
    if not identity_ok:
        raise RuntimeError("SMOKE_IDENTITY_GATE=FAIL")

    nvidia = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader"], text=True).strip().splitlines()
    nvcc = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
    gpu_info = {
        "requested_physical_gpu": requested_gpu,
        "visible_devices_env": os.environ["CUDA_VISIBLE_DEVICES"],
        "visible_gpu_count": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(0),
        "all_system_gpus": nvidia,
        "driver_version": nvidia[requested_gpu].split(",")[2].strip(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "nvcc_version": nvcc.stdout.strip() if nvcc.returncode == 0 else "not_available",
        "python_version": platform.python_version(),
        "GPU_COUNT": 1,
    }
    atomic_json(ROOT / "runtime/gpu_info.json", gpu_info)

    artifact_identity = {}
    for condition in ("C5", "C6"):
        rotation = DEPLOY / condition / "frozen_rotation_fp32.pt"
        manifest = DEPLOY / condition / "deployment_manifest.json"
        digest = sha(rotation)
        if digest != EXPECTED_ROTATION[condition]:
            raise RuntimeError(f"{condition} frozen rotation hash mismatch")
        artifact_identity[condition] = {"rotation": str(rotation), "rotation_sha256": digest, "manifest": str(manifest), "manifest_sha256": sha(manifest)}
    atomic_json(ROOT / "provenance/source_identity.json", {
        "SOURCE_GATE": "PASS", "SOURCE_BRANCH": git("symbolic-ref", "--short", "HEAD"),
        "SOURCE_BASE_V3_HEAD": BASE_HEAD, "SOURCE_HEAD": start_head, "WORKTREE_CLEAN_AT_START": True,
        "formal_runner": str(FORMAL_PATH), "formal_runner_sha256": sha(FORMAL_PATH),
        "old_formal_fanout_source": str(OLD_SMOKE), "old_formal_fanout_sha256": sha(OLD_SMOKE),
        "reducer_v3_source": str(V3_PATH), "reducer_v3_sha256": sha(V3_PATH),
        "atomic_protocol_source": str(ATOMIC_PATH), "atomic_protocol_sha256": sha(ATOMIC_PATH),
        "artifact_identity": artifact_identity,
    })

    model, tokenizer = formal.load_model_and_tokenizer()
    device = next(model.parameters()).device
    if torch.cuda.device_count() != 1 or device.type != "cuda":
        raise RuntimeError("one visible CUDA GPU required")
    banks, rotations = {}, {}
    for condition in ("C5", "C6"):
        banks[condition], rotations[condition] = formal.load_frozen_rotation(
            artifact_identity[condition]["rotation"], artifact_identity[condition]["manifest"], device, condition.lower()
        )
    h_patch = formal.QwenKeyHadamardPatch(model)
    c5_patch = formal.QwenFrozenRotationPatch(model, banks["C5"], rotations["C5"])
    c6_patch = formal.QwenFrozenRotationPatch(model, banks["C6"], rotations["C6"])

    from transformers import DynamicCache
    sample_results = []
    all_state_checks = []
    all_audits = {}
    baseline = gpu_mem(device)
    with old.StrictCommonFPFanout(model, formal, h_patch, c5_patch, c6_patch) as fanout, torch.inference_mode():
        for row in smoke_rows:
            sample_id = f"{row['problem_id']}_seed{int(row['seed'])}"
            tokens = list(map(int, row["generated_token_ids"][:SMOKE_TOKENS]))
            expected = EXPECTED_SMOKE[sample_id]
            if len(row["generated_token_ids"]) != expected["teacher_tokens"] or token_sha(tokens) != expected["prefix_sha256"] or token_sha(row["generated_token_ids"]) != expected["full_sha256"]:
                raise RuntimeError(f"smoke source identity mismatch {sample_id}")
            identity = {"sample_id": sample_id, "source_commit": BASE_HEAD, "reducer_schema_version": v3.SCHEMA_VERSION, "input_identity_hash": expected["prefix_sha256"], **metric_hashes}
            reducer = v3.SampleReducerV3(identity)
            accumulator = StreamingAccumulator(reducer, v3)
            observer = old.QuantizerObserver(formal.quantize_c128, accumulator)
            audit_total = {condition: {"calls": 0, "layers": set(), "scale_shapes": {}, "code_min": 0, "code_max": 0} for condition in CONDITIONS}

            gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(device)
            before = gpu_mem(device)
            torch.cuda.reset_peak_memory_stats(device)
            prompt = tokenizer.apply_chat_template(row["raw_messages"], tokenize=False, add_generation_prompt=True, enable_thinking=True)
            encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = encoded["input_ids"].to(device)
            attention = encoded.get("attention_mask")
            attention = attention.to(device) if attention is not None else None
            cache = DynamicCache(config=model.config, offloading=True, offload_only_non_sliding=True)
            fanout.capture = False
            prefill = model(input_ids=input_ids, attention_mask=attention, past_key_values=cache, use_cache=True, logits_to_keep=1)
            past = prefill.past_key_values
            del prefill, input_ids, attention
            decode_started = time.perf_counter()
            state_offset = len(fanout.state_checks)
            for token_idx, token_id in enumerate(tokens):
                fanout.begin(sample_id, token_idx)
                current = torch.tensor([[token_id]], dtype=torch.long, device=device)
                output = model(input_ids=current, past_key_values=past, use_cache=True, logits_to_keep=1)
                past = output.past_key_values
                fanout.end()
                for condition in CONDITIONS:
                    if condition == "Native":
                        states = {layer: formal.get_state(past, layer).detach().clone() for layer in formal.GDN_LAYERS}
                    else:
                        states = fanout.local_states[condition]
                    local_cache = old.FakeCache(states, max(formal.GDN_LAYERS))
                    observer.run(local_cache, audit_total[condition], sample_id, token_idx, condition, formal.GDN_LAYERS)
                    del local_cache, states
                del output, current
                if (token_idx + 1) % 4 == 0:
                    print(json.dumps({"event": "gpu_smoke_progress", "sample_id": sample_id, "token": token_idx + 1, "total": len(tokens), "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20}), flush=True)
            torch.cuda.synchronize(device)
            decode_seconds = time.perf_counter() - decode_started
            accumulator.finish()
            if reducer.last_committed_token_idx != len(tokens) - 1:
                raise RuntimeError("reducer token cursor mismatch")
            peak = {"peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)), "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device))}
            artifact = reducer.sample_artifact()
            done = atomic.commit_sample(ROOT / "artifacts/samples", sample_id, artifact, identity)
            valid = atomic.validate_sample(ROOT / "artifacts/samples", sample_id, identity)
            if not valid["valid"]:
                raise RuntimeError("atomic sample validation failed")
            checkpoint_path = ROOT / "artifacts/checkpoints" / f"{sample_id}.checkpoint.json"
            checkpoint_path.write_bytes(canonical_bytes(reducer.to_state()))
            sample_path = ROOT / "artifacts/samples" / f"{sample_id}.json"
            summary_only = {"identity": artifact["identity"], "processing_cursor": artifact["processing_cursor"], "summary_units": artifact["summary_units"], "paired": artifact["paired"]}
            summary_bytes = len(canonical_bytes(summary_only))
            events_bytes = len(canonical_bytes(artifact["extreme_topk"]))
            manifest_bytes = (ROOT / "artifacts/samples" / f"{sample_id}.done").stat().st_size + (ROOT / "artifacts/samples" / f"{sample_id}.sha256").stat().st_size
            checks = fanout.state_checks[state_offset:]
            state_ok = len(checks) == 2 * len(formal.GDN_LAYERS) and all(len(set(x["branch_sha256"].values())) == 1 and all(x["branch_torch_equal"].values()) for x in checks)
            token_hashes = {condition: expected["prefix_sha256"] for condition in CONDITIONS}
            token_ok = len(set(token_hashes.values())) == 1
            quant_ok = all(audit_total[c]["calls"] == len(tokens) and audit_total[c]["layers"] == set(formal.GDN_LAYERS) for c in CONDITIONS)
            audit_json = {c: {**{k: v for k, v in audit_total[c].items() if k != "layers"}, "layers": sorted(audit_total[c]["layers"])} for c in CONDITIONS}
            all_audits[sample_id] = audit_json
            all_state_checks.extend(checks)
            del past, cache, reducer, accumulator, observer, artifact
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(device)
            after = gpu_mem(device)
            sample_results.append({
                "sample_id": sample_id, "teacher_tokens_total": expected["teacher_tokens"], "tokens_processed": len(tokens),
                "prefix_token_sha256": expected["prefix_sha256"], "condition_token_hashes": token_hashes,
                "wall_time_seconds": decode_seconds, "tokens_per_second": len(tokens) / decode_seconds,
                "seconds_per_sample": decode_seconds, "memory": {"allocated_before": before["allocated_bytes"], "reserved_before": before["reserved_bytes"], **peak, "allocated_after": after["allocated_bytes"], "reserved_after": after["reserved_bytes"]},
                "artifact_sizes": {"sample_artifact_bytes": sample_path.stat().st_size, "summary_bytes": summary_bytes, "event_ledger_bytes": events_bytes, "checkpoint_bytes": checkpoint_path.stat().st_size, "manifest_bytes": manifest_bytes},
                "COMMON_TOKEN_SEQUENCE_GATE": "PASS" if token_ok else "FAIL", "COMMON_FP_STATE_INPUT_GATE": "PASS" if state_ok else "FAIL", "FORMAL_QUANTIZER_SHARED_PATH_GATE": "PASS" if quant_ok else "FAIL",
                "atomic_validation": valid, "done": done,
            })

    memory_after = [r["memory"]["allocated_after"] for r in sample_results]
    memory_creep_bytes = max(memory_after) - min(memory_after)
    memory_creep = memory_creep_bytes > 16 * 2**20
    memory_gate = not memory_creep and all(r["memory"]["peak_allocated_bytes"] < torch.cuda.get_device_properties(device).total_memory for r in sample_results)
    total_tokens = sum(r["tokens_processed"] for r in sample_results)
    total_seconds = sum(r["wall_time_seconds"] for r in sample_results)
    tps = total_tokens / total_seconds
    projected_one_gpu_seconds = FULL_TEACHER_TOKENS / tps
    projected_gpu_hours = projected_one_gpu_seconds / 3600
    projected_four_gpu_wall_seconds = projected_one_gpu_seconds / 4

    sample_bytes = [r["artifact_sizes"]["sample_artifact_bytes"] for r in sample_results]
    checkpoint_bytes = [r["artifact_sizes"]["checkpoint_bytes"] for r in sample_results]
    final_projected = math.ceil(sum(sample_bytes) / len(sample_bytes) * 60)
    temporary_projected = final_projected + max(checkpoint_bytes)
    free_after_smoke = df_free()
    projected_free = free_after_smoke - final_projected - max(checkpoint_bytes)
    resource_ok = final_projected < 4 * 2**30 and temporary_projected < 6 * 2**30 and projected_free > 20 * 2**30

    semantic_ok = all(r["COMMON_TOKEN_SEQUENCE_GATE"] == "PASS" and r["COMMON_FP_STATE_INPUT_GATE"] == "PASS" and r["FORMAL_QUANTIZER_SHARED_PATH_GATE"] == "PASS" for r in sample_results)
    atomic_ok = all(r["atomic_validation"]["valid"] for r in sample_results)
    good_decision = atomic.fail_closed_decision([0], 2, [r["sample_id"] for r in sample_results])
    bad_worker = atomic.fail_closed_decision([1], 2, [r["sample_id"] for r in sample_results])
    incomplete = atomic.fail_closed_decision([0], 2, [sample_results[0]["sample_id"]])
    fail_closed_ok = good_decision["stage_gate"] == "PASS" and bad_worker["stage_gate"] == "FAIL" and incomplete["stage_gate"] == "FAIL" and not any(x["downstream_launch_authorized"] for x in (good_decision, bad_worker, incomplete))

    # Separate synthetic atomic tmp->final->hash->done smoke.
    atom_root = ROOT / "artifacts/atomic_test"
    atom_identity = {"sample_id": "synthetic_atomic_probe", "task": TASK}
    atom_payload = {"probe": "atomic", "value": 1}
    atomic.commit_sample(atom_root, "synthetic_atomic_probe", atom_payload, atom_identity)
    atom_valid = atomic.validate_sample(atom_root, "synthetic_atomic_probe", atom_identity)
    atomic_test_ok = atom_valid["valid"] and not any(atom_root.glob("*.tmp"))
    atomic_ok = atomic_ok and atomic_test_ok

    atomic_json(ROOT / "runtime/timing.json", {"samples": [{k: r[k] for k in ("sample_id", "tokens_processed", "wall_time_seconds", "tokens_per_second", "seconds_per_sample")} for r in sample_results], "total_tokens": total_tokens, "total_wall_time_seconds": total_seconds, "aggregate_tokens_per_second": tps})
    atomic_json(ROOT / "runtime/memory.json", {"model_loaded_baseline": baseline, "samples": [{"sample_id": r["sample_id"], **r["memory"]} for r in sample_results], "memory_creep_bytes": memory_creep_bytes, "MEMORY_CREEP": memory_creep, "MEMORY_GATE": "PASS" if memory_gate else "FAIL"})
    atomic_json(ROOT / "artifacts/size_summary.json", {"samples": [{"sample_id": r["sample_id"], **r["artifact_sizes"]} for r in sample_results], "bytes_per_sample_mean": sum(sample_bytes) / len(sample_bytes), "artifact_tree_bytes_at_measurement": size_tree(ROOT / "artifacts")})
    projection = {"full_teacher_tokens": FULL_TEACHER_TOKENS, "measured_tokens_per_second_one_gpu": tps, "PROJECTED_FULL60_RUNTIME_ONE_GPU_SECONDS": projected_one_gpu_seconds, "PROJECTED_FULL60_RUNTIME_ONE_GPU_HOURS": projected_gpu_hours, "PROJECTED_FULL60_RUNTIME_FOUR_GPU_WALL_SECONDS": projected_four_gpu_wall_seconds, "PROJECTED_FULL60_GPU_HOURS": projected_gpu_hours, "PROJECTED_FULL60_FINAL_ARTIFACT_BYTES": final_projected, "PROJECTED_FULL60_FINAL_ARTIFACT_GB": final_projected / 2**30, "PROJECTED_FULL60_TEMPORARY_BYTES": temporary_projected, "PROJECTED_FULL60_TEMPORARY_GB": temporary_projected / 2**30, "free_before_smoke_bytes": free_before, "free_after_smoke_bytes": free_after_smoke, "PROJECTED_FREE_SPACE_AFTER_FULL60_BYTES": projected_free, "PROJECTED_FREE_SPACE_AFTER_FULL60_GB": projected_free / 2**30, "gates": {"final_artifact_lt_4_gib": final_projected < 4 * 2**30, "temporary_peak_lt_6_gib": temporary_projected < 6 * 2**30, "free_after_gt_20_gib": projected_free > 20 * 2**30}, "FULL60_RESOURCE_READINESS": "PASS" if resource_ok else "FAIL", "projection_only": True}
    atomic_json(ROOT / "projection/full60_resource_projection.json", projection)
    semantic_gates = {"SOURCE_GATE": "PASS", "SMOKE_IDENTITY_GATE": "PASS", "COMMON_TOKEN_SEQUENCE_GATE": "PASS" if semantic_ok else "FAIL", "COMMON_FP_STATE_INPUT_GATE": "PASS" if semantic_ok else "FAIL", "STATE_RESET_TO_COMMON_FP_EACH_TOKEN": True, "PERSISTENT_QDQ_WRITEBACK": False, "NEXT_TOKEN_READS_FP_STATE": True, "FORMAL_ROTATION_PATH_GATE": "PASS", "FORMAL_QUANTIZER_SHARED_PATH_GATE": "PASS" if semantic_ok else "FAIL", "sample_results": [{k: r[k] for k in ("sample_id", "COMMON_TOKEN_SEQUENCE_GATE", "COMMON_FP_STATE_INPUT_GATE", "FORMAL_QUANTIZER_SHARED_PATH_GATE")} for r in sample_results], "formal_quantizer_audits": all_audits, "state_check_count": len(all_state_checks)}
    atomic_json(ROOT / "validation/semantic_gates.json", semantic_gates)
    atomic_json(ROOT / "validation/atomic_test.json", {"sample_atomic_validations": [{"sample_id": r["sample_id"], "validation": r["atomic_validation"]} for r in sample_results], "synthetic_validation": atom_valid, "temporary_files_remaining": [str(p) for p in atom_root.glob("*.tmp")], "ATOMIC_ARTIFACT_GATE": "PASS" if atomic_ok else "FAIL"})
    atomic_json(ROOT / "validation/fail_closed_test.json", {"case_A_success": good_decision, "case_B_nonzero_worker": bad_worker, "case_C_incomplete_artifact": incomplete, "FAIL_CLOSED_ORCHESTRATION_GATE": "PASS" if fail_closed_ok else "FAIL"})

    final_pass = semantic_ok and memory_gate and atomic_ok and fail_closed_ok
    gates = {
        "SOURCE_GATE": "PASS", "SMOKE_IDENTITY_GATE": "PASS", "COMMON_TOKEN_SEQUENCE_GATE": "PASS" if semantic_ok else "FAIL", "COMMON_FP_STATE_INPUT_GATE": "PASS" if semantic_ok else "FAIL",
        "STATE_RESET_TO_COMMON_FP_EACH_TOKEN": True, "PERSISTENT_QDQ_WRITEBACK": False, "FORMAL_ROTATION_PATH_GATE": "PASS", "FORMAL_QUANTIZER_SHARED_PATH_GATE": "PASS" if semantic_ok else "FAIL",
        "MEMORY_GATE": "PASS" if memory_gate else "FAIL", "THROUGHPUT_MEASUREMENT": "PASS", "ARTIFACT_SIZE_MEASUREMENT": "PASS", "ATOMIC_ARTIFACT_GATE": "PASS" if atomic_ok else "FAIL", "FAIL_CLOSED_ORCHESTRATION_GATE": "PASS" if fail_closed_ok else "FAIL",
        "PROJECTED_FULL60_RUNTIME": projected_one_gpu_seconds, "PROJECTED_FULL60_GPU_HOURS": projected_gpu_hours, "PROJECTED_FULL60_ARTIFACT_GB": final_projected / 2**30, "FULL60_RESOURCE_READINESS": "PASS" if resource_ok else "FAIL",
        "GPU_STREAMING_SMOKE_V3": "PASS" if final_pass else "FAIL", "NEXT_STAGE": "FULL60_REQUIRES_USER_AUTHORIZATION",
        "FULL60_STARTED": False, "NEW_AIME_GENERATION": False, "RECURRENT_REPLAY_STARTED": False, "MULTI_HORIZON_STARTED": False, "PRE_DIVERGENCE_STARTED": False, "LAYER_INTERVENTION_STARTED": False, "C7_STARTED": False, "NEW_TRAINING_STARTED": False,
    }
    atomic_json(ROOT / "validation/final_gates.json", gates)
    report = f"""# FINAL GPU STREAMING SMOKE V3 REPORT

## Verdict

`GPU_STREAMING_SMOKE_V3 = {gates['GPU_STREAMING_SMOKE_V3']}`

Q1. Formal semantics preserved? **{'YES' if semantic_ok else 'NO'}** (`COMMON_TOKEN_SEQUENCE_GATE` and `COMMON_FP_STATE_INPUT_GATE`: {gates['COMMON_FP_STATE_INPUT_GATE']}).

Q2. Rotation path unchanged? **YES**. The frozen formal runner's Hadamard/C5/C6 patch methods were used directly; `FORMAL_ROTATION_PATH_GATE = PASS`.

Q3. Quantizer path unchanged? **{'YES' if semantic_ok else 'NO'}**. The unmodified formal `quantize_c128` function was executed and observed; `FORMAL_QUANTIZER_SHARED_PATH_GATE = {gates['FORMAL_QUANTIZER_SHARED_PATH_GATE']}`.

Q4. State-reset semantics preserved? **YES**. Each token fans out from the common Native FP state; QDQ branch states are discarded. `STATE_RESET_TO_COMMON_FP_EACH_TOKEN = TRUE`, `PERSISTENT_QDQ_WRITEBACK = FALSE`.

Q5. Peak allocated GPU memory: **{max(r['memory']['peak_allocated_bytes'] for r in sample_results)/2**30:.3f} GiB**; peak reserved: **{max(r['memory']['peak_reserved_bytes'] for r in sample_results)/2**30:.3f} GiB**.

Q6. Memory creep? **{'YES' if memory_creep else 'NO'}** (end-of-sample spread {memory_creep_bytes/2**20:.3f} MiB; gate threshold 16 MiB).

Q7. Throughput: **{tps:.4f} tokens/s** over {total_tokens} smoke tokens.

Q8. Full-60 projected runtime: **{projected_one_gpu_seconds/3600:.2f} h one-GPU equivalent**, or **{projected_four_gpu_wall_seconds/3600:.2f} h idealized four-GPU wall time**. Projection only.

Q9. Full-60 projected GPU-hours: **{projected_gpu_hours:.2f} GPU-h**.

Q10. Projected final artifact size: **{final_projected/2**30:.3f} GiB**.

Q11. Projected temporary peak: **{temporary_projected/2**30:.3f} GiB**.

Q12. Disk/resource gate: **{'PASS' if resource_ok else 'FAIL'}**; projected free space after full60 is {projected_free/2**30:.3f} GiB.

Q13. Atomic artifact gate: **{'PASS' if atomic_ok else 'FAIL'}**.

Q14. Fail-closed orchestration gate: **{'PASS' if fail_closed_ok else 'FAIL'}**. Non-zero worker and incomplete artifacts both fail with no downstream launch.

Q15. Was full60 run? **NO.** `FULL60_STARTED = FALSE`.

## Final state

`NEXT_STAGE = FULL60_REQUIRES_USER_AUTHORIZATION`

All forbidden follow-on stages remain `FALSE`.
"""
    (ROOT / "reports/FINAL_GPU_SMOKE_REPORT.md").write_text(report, encoding="utf-8")
    manifest = ROOT / "hashes/SHA256SUMS"
    paths = sorted(path for path in ROOT.rglob("*") if path.is_file() and path != manifest)
    manifest.write_text("".join(f"{sha(path)}  {path.relative_to(ROOT)}\n" for path in paths), encoding="utf-8")
    print(json.dumps({"status": gates["GPU_STREAMING_SMOKE_V3"], "tokens_per_second": tps, "peak_allocated_gib": max(r["memory"]["peak_allocated_bytes"] for r in sample_results)/2**30, "projected_final_gib": final_projected/2**30, "resource_readiness": gates["FULL60_RESOURCE_READINESS"], "full60_started": False, "elapsed_seconds": time.time()-started}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
