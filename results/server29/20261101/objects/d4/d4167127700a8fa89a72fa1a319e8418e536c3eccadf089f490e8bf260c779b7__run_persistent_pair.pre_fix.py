#!/usr/bin/env python3
"""Prospective paired 32K persistent-QDQ scorer for frozen Ling rotations.

The FP reference and all selected rotations advance in one process on the
same physical GPU.  Full-vocabulary KL is reduced online; no logits are
archived.  Every branch performs R128 QDQ and writeback after every token.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import socket
import sys
import time
import traceback
from pathlib import Path


TASK = "LING_KDA_PERSISTENT_SURROGATE_PROSPECTIVE_VALIDATION_V1"
ROOT = Path("/data/zypan/experiment_runs") / TASK
BRIDGE = Path("/data/zypan/worktrees/ling-long-horizon-observability-bridge-v3/experiments/LING_LONG_HORIZON_OBSERVABILITY_BRIDGE_V1")
DIRECTION = Path("/data/zypan/experiment_runs/LING_KDA_LONG_HORIZON_DIRECTION_CAUSAL_V1/scripts/run_direction_causal.py")
LAYERS = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22]
GPU_UUIDS = {3: "GPU-759526ed-efb2-e7f6-881c-ab098e0b05ae", 6: "GPU-de53f243-b271-521c-d875-d96611ed8574"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"NO_OVERWRITE:{path}")
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.link(tmp, path)
    tmp.unlink()


def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def finite_tree(value) -> bool:
    if isinstance(value, dict):
        return all(finite_tree(x) for x in value.values())
    if isinstance(value, list):
        return all(finite_tree(x) for x in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def percentile(values, fraction: float) -> float:
    values = sorted(values)
    if not values:
        raise RuntimeError("EMPTY_PERCENTILE")
    at = fraction * (len(values) - 1)
    low = int(math.floor(at)); high = int(math.ceil(at))
    if low == high:
        return float(values[low])
    return float(values[low] * (high - at) + values[high] * (at - low))


def load_tokens_and_rotations(torch, transport, document: str, conditions: list[str]):
    selection = json.loads((ROOT / "PROSPECTIVE_DOCUMENT_SELECTION.json").read_text())
    doc = next(x for x in selection["selected_documents"] if x["document_id"] == document)
    token_path = Path(doc["token_file"]["path"])
    if sha256(token_path) != doc["token_file"]["sha256"]:
        raise RuntimeError("TOKEN_HASH_MISMATCH")
    tokens = json.loads(token_path.read_text())
    if len(tokens) < 32768:
        raise RuntimeError("TOKEN_LENGTH_LT_32768")
    manifest_path = BRIDGE / "rotation_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    bank = {x["condition_id"]: x for x in manifest["bank"]["conditions"]}
    rotations, identities = {}, {}
    for condition in conditions:
        item = bank[condition]
        path = Path(transport.mapped(item["artifact"]["path"]))
        digest = sha256(path)
        if digest != item["artifact"]["sha256"]:
            raise RuntimeError(f"ROTATION_HASH_MISMATCH:{condition}")
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if [int(x) for x in obj["layer_ids"]] != LAYERS:
            raise RuntimeError(f"ROTATION_LAYER_MISMATCH:{condition}")
        mapping = obj["rotations"]
        rotations[condition] = torch.stack([
            (mapping[layer] if layer in mapping else mapping[str(layer)]).float() for layer in LAYERS
        ])
        identities[condition] = {"path": str(path), "sha256": digest, "manifest_entry": item}
    return doc, [int(x) for x in tokens], rotations, identities, {"path": str(manifest_path), "sha256": sha256(manifest_path)}


def state_diagnostic(torch, direction, capture, fp_cache, branch_cache):
    per_layer = {}
    total_sq = 0.0
    total_n = 0
    for layer in LAYERS:
        fp = direction.cache_state(capture, fp_cache, layer).float()
        branch = direction.cache_state(capture, branch_cache, layer).float()
        error = branch - fp
        sq = float(error.square().sum().item())
        n = error.numel()
        per_layer[str(layer)] = {
            "l2": float(math.sqrt(sq)),
            "mse": float(sq / n),
            "max_abs": float(error.abs().max().item()),
        }
        total_sq += sq; total_n += n
    return {"global_l2": math.sqrt(total_sq), "global_mse": total_sq / total_n, "per_layer": per_layer}


def summarize_kl(values: list[float], start: int, end: int):
    if len(values) != end - start:
        raise RuntimeError(f"KL_COUNT_MISMATCH:{start}:{end}:{len(values)}")
    return {
        "window": [start, end], "count": len(values),
        "mean": sum(values) / len(values), "auc": sum(values),
        "p50": percentile(values, 0.50), "p99": percentile(values, 0.99),
        "min": min(values), "max": max(values),
    }


def run(document: str, gpu: int, conditions: list[str], role: str, tag: str):
    if len(conditions) != len(set(conditions)) or not conditions:
        raise RuntimeError("CONDITION_SET_INVALID")
    config = json.loads((ROOT / "frozen_config.json").read_text())
    if config["document_gpu_map"].get(document) != gpu:
        raise RuntimeError("FROZEN_DOCUMENT_GPU_MAP_VIOLATION")
    if any(x not in config["conditions"] for x in conditions):
        raise RuntimeError("FROZEN_ROTATION_ROSTER_VIOLATION")
    if role == "correctness_smoke":
        endpoint, primary_start, secondary_start = 512, 256, 128
        diagnostic_positions = {127, 255, 511}
    elif role in {"stage1", "stage2"}:
        endpoint, primary_start, secondary_start = 32768, 8192, 2048
        diagnostic_positions = {2047, 8191, 16383, 24575, 32767}
    else:
        raise ValueError(role)
    run_root = ROOT / "runs" / tag
    if run_root.exists():
        raise FileExistsError(f"NO_OVERWRITE:{run_root}")
    run_root.mkdir(parents=True)
    started = time.time()
    direction = import_file(DIRECTION, f"prospective_direction_{gpu}_{os.getpid()}")
    transport, _, capture_module, core = direction.load_assets(gpu)
    torch = core.torch
    torch.set_grad_enabled(False)
    hardware = direction.gpu_guard(torch, gpu)
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG_MISMATCH")
    if os.environ.get("LING_GATED_RMSNORM_FIXED_CONFIG") != "BT16_W8":
        raise RuntimeError("LING_SELECTOR_MISMATCH")
    if not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("DETERMINISTIC_ALGORITHMS_DISABLED")
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise RuntimeError("TF32_ENABLED")
    doc, tokens, rotations_cpu, rotation_ids, rotation_manifest = load_tokens_and_rotations(
        torch, transport, document, conditions
    )
    started_receipt = {
        "task": TASK, "role": role, "tag": tag, "document": document, "physical_gpu": gpu,
        "conditions": conditions, "endpoint": endpoint, "primary_window": [primary_start, endpoint],
        "secondary_window": [secondary_start, endpoint], "hardware": hardware,
        "token_file": doc["token_file"], "rotation_identities": rotation_ids,
        "rotation_manifest": rotation_manifest, "source_sha256": sha256(Path(__file__)),
        "direction_runner_sha256": sha256(DIRECTION), "pid": os.getpid(), "started_epoch": started,
        "streaming_full_vocabulary_kl": True, "logits_archived": False,
    }
    atomic_json(run_root / "STARTED.json", started_receipt)
    capture_instance = capture_module.Capture()
    try:
        capture = capture_instance
        rotations = {k: v.to(capture.device) for k, v in rotations_cpu.items()}
        caches = {condition: None for condition in conditions}
        fp_cache = None
        previous = {condition: {} for condition in conditions}
        active_condition = [None]
        consumed = {condition: 0 for condition in conditions}
        qdq_calls = {condition: 0 for condition in conditions}
        qscale = {condition: {"min": float("inf"), "max": 0.0, "sum": 0.0, "count": 0} for condition in conditions}
        trajectories = {condition: [] for condition in conditions}
        diagnostics = {condition: {} for condition in conditions}
        original_kernel = capture.kernel

        def audited_kernel(**kwargs):
            condition = active_condition[0]
            if condition is not None:
                layer = int(capture.probe.current_layer)
                initial = kwargs.get("initial_state")
                if layer in previous[condition]:
                    if initial is None or not torch.equal(initial, previous[condition][layer]):
                        raise RuntimeError(f"PERSISTENT_WRITEBACK_CONSUMPTION_FAIL:{condition}:{layer}")
                    consumed[condition] += 1
                elif initial is not None and bool(initial.count_nonzero()):
                    raise RuntimeError(f"DOCUMENT_START_NOT_ZERO:{condition}:{layer}")
            return original_kernel(**kwargs)

        capture.kernel = audited_kernel
        for pos in range(endpoint):
            active_condition[0] = None
            fp_cache, fp_logits, _ = direction.model_step(torch, capture, fp_cache, tokens[pos], pos)
            if not bool(torch.isfinite(fp_logits).all()):
                raise RuntimeError(f"NONFINITE_FP_LOGITS:{pos}")
            for condition in conditions:
                active_condition[0] = condition
                caches[condition], logits, _ = direction.model_step(torch, capture, caches[condition], tokens[pos], pos)
                if not bool(torch.isfinite(logits).all()):
                    raise RuntimeError(f"NONFINITE_BRANCH_LOGITS:{condition}:{pos}")
                meta = direction.quantize_l6_cache(torch, core, capture, caches[condition], rotations[condition])
                if sorted(meta) != LAYERS:
                    raise RuntimeError(f"QDQ_LAYER_COVERAGE_FAIL:{condition}:{pos}")
                qdq_calls[condition] += len(meta)
                previous[condition] = {
                    layer: direction.cache_state(capture, caches[condition], layer).detach().clone()
                    for layer in LAYERS
                }
                for item in meta.values():
                    qscale[condition]["min"] = min(qscale[condition]["min"], item["scale_min"])
                    qscale[condition]["max"] = max(qscale[condition]["max"], item["scale_max"])
                    qscale[condition]["sum"] += item["scale_mean"]
                    qscale[condition]["count"] += 1
                if pos >= secondary_start:
                    value = direction.kl_metrics(torch, fp_logits, logits)["KL"]
                    if not math.isfinite(value):
                        raise RuntimeError(f"NONFINITE_KL:{condition}:{pos}")
                    trajectories[condition].append(float(value))
                del logits
            active_condition[0] = None
            if pos in diagnostic_positions:
                for condition in conditions:
                    diagnostics[condition][str(pos + 1)] = state_diagnostic(
                        torch, direction, capture, fp_cache, caches[condition]
                    )
            if (pos + 1) % 512 == 0:
                print("PERSISTENT_PROGRESS", document, role, pos + 1, endpoint,
                      round(time.time() - started, 2), flush=True)
            del fp_logits
        capture.kernel = original_kernel
        results = []
        for condition in conditions:
            secondary_values = trajectories[condition]
            primary_values = secondary_values[primary_start - secondary_start:]
            expected_qdq = endpoint * len(LAYERS)
            expected_consumed = (endpoint - 1) * len(LAYERS)
            integrity = {
                "every_token_layer_qdq_count": qdq_calls[condition],
                "expected_every_token_layer_qdq_count": expected_qdq,
                "actual_next_kernel_equal_post_qdq_count": consumed[condition],
                "expected_next_kernel_equal_post_qdq_count": expected_consumed,
                "passed": qdq_calls[condition] == expected_qdq and consumed[condition] == expected_consumed,
            }
            if not integrity["passed"]:
                raise RuntimeError(f"PERSISTENT_SEMANTICS_FAIL:{condition}:{integrity}")
            scale = qscale[condition]
            trajectory_path = run_root / f"KL_TRAJECTORY_{condition}.json"
            trajectory = {
                "document": document, "condition": condition, "token_positions": [secondary_start, endpoint],
                "alignment": "logits at position t reflect persistent QDQ/writeback through t-1; state is QDQ-written at t",
                "full_vocabulary_KL_FP_to_branch": secondary_values,
            }
            trajectory["content_sha256_without_self"] = canonical_hash(trajectory)
            atomic_json(trajectory_path, trajectory)
            result = {
                "document": document, "condition": condition, "role": role, "physical_gpu": gpu,
                "primary": summarize_kl(primary_values, primary_start, endpoint),
                "secondary": summarize_kl(secondary_values, secondary_start, endpoint),
                "state_diagnostics": diagnostics[condition],
                "qdq_scale": {"min": scale["min"], "mean_over_layer_steps": scale["sum"] / scale["count"], "max": scale["max"], "count": scale["count"]},
                "persistent_integrity": integrity,
                "trajectory": {"path": str(trajectory_path), "sha256": sha256(trajectory_path)},
                "all_finite": True,
            }
            atomic_json(run_root / f"RESULT_{condition}.json", result)
            results.append(result)
        elapsed = time.time() - started
        receipt = {
            "gate": "PASS", **started_receipt, "completed_epoch": time.time(),
            "elapsed_seconds": elapsed, "gpu_hours": elapsed / 3600.0,
            "results": results, "persistent_every_token_qdq_writeback": True,
            "same_process_same_gpu_fp_pairing": True, "model_weights_modified": False,
            "optimizer_step": False, "training": False, "free_generation": False, "aime": False,
        }
        if not finite_tree(receipt):
            raise RuntimeError("NONFINITE_RECEIPT")
        receipt["content_sha256_without_self"] = canonical_hash(receipt)
        atomic_json(run_root / "COMPLETE.json", receipt)
        print("PERSISTENT_PAIR_COMPLETE", document, role, conditions, round(elapsed, 2), flush=True)
    finally:
        capture_instance.close()
        gc.collect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--document", required=True)
    ap.add_argument("--gpu", type=int, choices=[3, 6], required=True)
    ap.add_argument("--conditions", required=True)
    ap.add_argument("--role", choices=["correctness_smoke", "stage1", "stage2"], required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    run(args.document, args.gpu, args.conditions.split(","), args.role, args.tag)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else f"untagged_{os.getpid()}"
        path = ROOT / "runs" / f"{tag}.STOP.json"
        if not path.exists():
            atomic_json(path, {"task": TASK, "gate": "FAIL", "error": repr(exc), "traceback": traceback.format_exc(),
                               "host": socket.gethostname(), "stopped_epoch": time.time(), "final_action": "FAIL_AND_STOP"})
        raise
