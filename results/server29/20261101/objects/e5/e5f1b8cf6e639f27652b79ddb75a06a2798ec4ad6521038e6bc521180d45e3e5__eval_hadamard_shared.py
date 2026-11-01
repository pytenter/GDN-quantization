#!/usr/bin/env python3
"""Reevaluate canonical Hadamard on the exact frozen H64 validation protocol."""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path


ROOT = Path("experiments/QWEN_GDN_SHARED_OBJECTIVE_PARETO_AND_DEV_SCREEN_V1")
SOURCE_ROOT = Path("/data/zypan")
MODEL = SOURCE_ROOT / "modelscope_models/Qwen3.5-9B"
CANONICAL_REPO = SOURCE_ROOT / "worktrees/hadamard-init-dense-oracle-v1-qwen"
SOURCE = CANONICAL_REPO / "experiments/QWEN_RECURRENT_DENSE_C5_C6_V1/run_recurrent_dense.py"
TRANSFORMERS_ROOT = SOURCE_ROOT / "transformers-qwen35/src"
OUTPUT = ROOT / "analysis/hadamard_shared_objective.json"
ERROR = ROOT / "analysis/hadamard_shared_objective.error.json"


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def theta_digest(bank) -> str:
    h = hashlib.sha256()
    for name, value in sorted(bank.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main() -> None:
    if OUTPUT.exists() or ERROR.exists():
        raise RuntimeError("Hadamard shared-objective measurement already attempted")
    protocol = json.loads((ROOT / "analysis/shared_validation_protocol.json").read_text(encoding="utf-8"))
    if protocol["status"] != "PASS" or protocol["ORIGINAL_VALIDATION_PROTOCOL_MATCH"] != "YES":
        raise RuntimeError("exact shared validation protocol did not pass")

    os.environ.setdefault("QWEN_TRANSFORMERS_SRC", str(TRANSFORMERS_ROOT))
    os.environ.setdefault("QWEN_MODEL_PATH", str(MODEL))
    os.environ.setdefault("HF_HOME", "/data/zypan/.cache/huggingface")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    sys.path.insert(0, str(TRANSFORMERS_ROOT))
    import torch
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected exactly one visible GPU, got {torch.cuda.device_count()}")
    device = torch.device("cuda:0")
    started = time.time()
    try:
        core = import_file(SOURCE, "qwen_shared_objective_hadamard_core")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL), torch_dtype=torch.bfloat16, device_map=None,
            trust_remote_code=True, local_files_only=True, low_cpu_mem_usage=True,
        )
        model.to(device)
        core.freeze_model(model)
        bank = core.CAYLEY.PerLayerCayleyRotations(core.GDN_LAYERS).to(device)
        before_theta = theta_digest(bank)
        if any(bool(torch.count_nonzero(parameter).detach().cpu()) for parameter in bank.parameters()):
            raise RuntimeError("Hadamard theta=0 initialization mismatch")
        h128 = core.hadamard(device)
        rows = core.corpus_rows("VALIDATION")
        expected_ids = [x["document_id"] for x in protocol["document_checks"]]
        if [x["document_id"] for x in rows] != expected_ids:
            raise RuntimeError("validation document order drift")
        torch.cuda.reset_peak_memory_stats(device)
        with core.RecurrentExperimentPatch(model, bank, h128) as patch:
            values = core.evaluate_validation(model, tokenizer, bank, patch, rows, 64, "DENSE_STATE")
            qdq = {
                "calls": patch.qdq_audit["calls"],
                "layers": sorted(patch.qdq_audit["layers"]),
                "code_min": patch.qdq_audit["code_min"],
                "code_max": patch.qdq_audit["code_max"],
                "scale_shapes": patch.qdq_audit["scale_shapes"],
            }
        torch.cuda.synchronize(device)
        elapsed = time.time() - started
        expected_calls = len(core.GDN_LAYERS) * 16 * 1024
        if values["documents"] != 16 or values["samples"] != 128:
            raise RuntimeError(f"validation coverage mismatch: {values}")
        if qdq["calls"] != expected_calls or qdq["layers"] != list(core.GDN_LAYERS):
            raise RuntimeError(f"QDQ audit mismatch: {qdq['calls']} != {expected_calls}")
        if not (-127 <= qdq["code_min"] <= qdq["code_max"] <= 127):
            raise RuntimeError("QDQ code range violation")
        after_theta = theta_digest(bank)
        if before_theta != after_theta:
            raise RuntimeError("Hadamard bank changed during no-grad evaluation")
        free, total = torch.cuda.mem_get_info(device)
        result = {
            "status": "PASS",
            "label": "CANONICAL_HADAMARD_REEVALUATED_BASELINE",
            "historical_update0_row": False,
            "ORIGINAL_VALIDATION_PROTOCOL_MATCH": "YES",
            "state_relative_mse": values["state"],
            "functional_relative_mse": values["functional"],
            "combined_as_c5": values["state"],
            "combined_as_c6": values["functional"] + 0.1 * values["state"],
            "documents": values["documents"],
            "captures": values["samples"],
            "horizon": 64,
            "theta_zero": True,
            "theta_digest_before": before_theta,
            "theta_digest_after": after_theta,
            "qdq_audit": qdq,
            "expected_qdq_calls": expected_calls,
            "runtime_seconds_including_model_load": elapsed,
            "gpu_hours": elapsed / 3600.0,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "post_run_free_bytes": free,
            "device_total_bytes": total,
            "environment": {
                "hostname": platform.node(), "platform": platform.platform(),
                "python": sys.version, "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(device),
                "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
            "interpretation_limits": [
                "reevaluated baseline, not a recovered historical update-0 row",
                "different physical GPU class from the original 48GB training host is recorded",
                "shared-objective values are not AIME accuracy",
            ],
        }
        atomic_json(OUTPUT, result)
        print(json.dumps({"status": "PASS", "state": values["state"], "functional": values["functional"], "gpu_hours": result["gpu_hours"]}, sort_keys=True), flush=True)
        del model, bank, tokenizer, h128
        gc.collect()
        torch.cuda.empty_cache()
    except BaseException as exc:
        evidence = {
            "status": "FAIL", "error_type": type(exc).__name__, "error": str(exc),
            "traceback": traceback.format_exc(), "runtime_seconds": time.time() - started,
        }
        try:
            import torch
            evidence["cuda_allocated_bytes"] = torch.cuda.memory_allocated(device)
            evidence["cuda_reserved_bytes"] = torch.cuda.memory_reserved(device)
        except BaseException:
            pass
        atomic_json(ERROR, evidence)
        raise


if __name__ == "__main__":
    main()
