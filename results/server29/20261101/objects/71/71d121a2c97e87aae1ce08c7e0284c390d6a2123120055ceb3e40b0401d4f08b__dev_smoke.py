#!/usr/bin/env python3
"""Tiny cost/semantics smoke for the frozen independent non-AIME dev screen."""

from __future__ import annotations

import contextlib
import gc
import hashlib
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path


ROOT = Path("experiments/QWEN_GDN_SHARED_OBJECTIVE_PARETO_AND_DEV_SCREEN_V1")
LEGACY = Path("/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen")
FORMAL_RUNNER = LEGACY / "experiments/qwen_gdn/QWEN_C5_C6_AIME26_CANONICAL60_V1/scripts/run_qwen_c5_c6_formal.py"
DEPLOY = LEGACY / "experiments/qwen_gdn/QWEN_C5_C6_DEPLOYMENT_CLOSURE_V1/deployment"
SOURCE = Path("/data/zypan/results/gdn_end2end_bit_axis_screening_v1_math_subset.json")
SOURCE_SHA = "b197ab1779a65f7e7ca2425c5bb63b0982f4ffe8557c9ae3fdec4fee029da77a"
OUT = ROOT / "analysis/independent_dev_smoke.json"
ERR = ROOT / "analysis/independent_dev_smoke_attempt2.error.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("formal_dev_smoke", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def init_audit() -> dict:
    return {"calls": 0, "layers": set(), "scale_shapes": {}, "code_min": 0, "code_max": 0}


def patch_for(formal, model, condition: str, banks: dict, identities: dict):
    if condition == "FP":
        return contextlib.nullcontext()
    if condition == "HADAMARD":
        return formal.QwenKeyHadamardPatch(model)
    return formal.QwenFrozenRotationPatch(model, banks[condition], identities[condition])


def main() -> None:
    if OUT.exists() or ERR.exists():
        raise RuntimeError("independent dev smoke already attempted")
    started = time.time()
    try:
        if sha256(SOURCE) != SOURCE_SHA:
            raise RuntimeError("MATH-500 subset source SHA mismatch")
        source = json.loads(SOURCE.read_text())
        if source.get("n") != 30 or source.get("task") != "GDN_END2END_BIT_AXIS_SCREENING_V1":
            raise RuntimeError("source identity mismatch")
        records = list(source["records"])
        ordered = sorted(records, key=lambda x: hashlib.sha256(x["problem_id"].encode()).hexdigest())
        smoke_record = ordered[8]
        formal = load_module(FORMAL_RUNNER)
        model, tokenizer = formal.load_model_and_tokenizer()
        device = next(model.parameters()).device
        banks, identities = {}, {}
        canonical = {
            "C5_U63": (DEPLOY / "C5/frozen_rotation_fp32.pt", DEPLOY / "C5/deployment_manifest.json", "C5"),
            "C6_U125": (DEPLOY / "C6/frozen_rotation_fp32.pt", DEPLOY / "C6/deployment_manifest.json", "C6"),
            "C5_U113": (ROOT / "candidate_artifacts/C5_U113/frozen_rotation_fp32.pt", ROOT / "candidate_artifacts/C5_U113/deployment_manifest.json", "C5_U113"),
            "C6_U119": (ROOT / "candidate_artifacts/C6_U119/frozen_rotation_fp32.pt", ROOT / "candidate_artifacts/C6_U119/deployment_manifest.json", "C6_U119"),
        }
        for label, (rotation, manifest, expected) in canonical.items():
            banks[label], identities[label] = formal.load_frozen_rotation(rotation, manifest, device, expected)
        _messages, prompt_ids = formal.render_input(tokenizer, smoke_record["problem"])
        prompt_ids = prompt_ids.to(device)
        teacher_tokens = tokenizer(smoke_record["solution"], add_special_tokens=False)["input_ids"]
        if len(teacher_tokens) < 65:
            raise RuntimeError("smoke teacher sequence shorter than 65 tokens")
        steps = 64
        units = []
        for condition in ("FP", "HADAMARD", "C5_U63", "C5_U113", "C6_U119", "C6_U125"):
            formal.setup_seed(0)
            audit = init_audit()
            gc.collect()
            __import__("torch").cuda.empty_cache()
            torch = __import__("torch")
            before = int(torch.cuda.memory_allocated(device))
            torch.cuda.reset_peak_memory_stats(device)
            unit_started = time.time()
            with patch_for(formal, model, condition, banks, identities) as patch, torch.inference_mode():
                output = model(input_ids=prompt_ids, attention_mask=torch.ones_like(prompt_ids), use_cache=True)
                past = output.past_key_values
                for index in range(steps):
                    token = torch.tensor([[teacher_tokens[index]]], dtype=torch.long, device=device)
                    output = model(input_ids=token, past_key_values=past, use_cache=True)
                    past = output.past_key_values
                    if condition != "FP":
                        formal.quantize_c128(past, audit)
                patch_record = None if condition == "FP" else {
                    "calls": int(patch.calls),
                    "layers": sorted(int(x) for x in patch.layers),
                    "phase_counts": dict(patch.phase_counts),
                }
            torch.cuda.synchronize(device)
            elapsed = time.time() - unit_started
            peak = int(torch.cuda.max_memory_allocated(device))
            del output, past
            gc.collect()
            torch.cuda.empty_cache()
            after = int(torch.cuda.memory_allocated(device))
            if condition != "FP":
                # quantize_c128 is called once per recurrent step and internally
                # loops over all 24 GDN layers; audit["calls"] therefore counts
                # steps, while audit["layers"] proves the layer coverage.
                if audit["calls"] != steps:
                    raise RuntimeError(f"{condition} QDQ call count mismatch")
                if sorted(audit["layers"]) != list(formal.GDN_LAYERS):
                    raise RuntimeError(f"{condition} QDQ layer coverage mismatch")
                if patch_record["phase_counts"].get("recurrent_decode") != steps * len(formal.GDN_LAYERS):
                    raise RuntimeError(f"{condition} recurrent patch count mismatch")
            units.append(
                {
                    "condition": condition,
                    "steps": steps,
                    "elapsed_seconds": elapsed,
                    "tokens_per_second": steps / elapsed,
                    "quantizer_calls": int(audit["calls"]),
                    "quantizer_layers": sorted(int(x) for x in audit["layers"]),
                    "patch_audit": patch_record,
                    "memory_allocated_before": before,
                    "peak_memory_allocated": peak,
                    "memory_allocated_after_cleanup": after,
                }
            )
        total = time.time() - started
        result = {
            "status": "PASS",
            "purpose": "cost and execution-semantics smoke only; no candidate metric was retained or inspected",
            "source_path": str(SOURCE),
            "source_sha256": SOURCE_SHA,
            "source_task": source["task"],
            "source_declared_arrow": source.get("source_arrow"),
            "source_declared_arrow_present": Path(source.get("source_arrow", "")).is_file(),
            "smoke_problem_id": smoke_record["problem_id"],
            "smoke_problem_id_order_key": hashlib.sha256(smoke_record["problem_id"].encode()).hexdigest(),
            "smoke_problem_sha256": hashlib.sha256(smoke_record["problem"].encode()).hexdigest(),
            "smoke_solution_sha256": hashlib.sha256(smoke_record["solution"].encode()).hexdigest(),
            "prompt_tokens": int(prompt_ids.shape[-1]),
            "teacher_solution_tokens": len(teacher_tokens),
            "steps": steps,
            "conditions": [x["condition"] for x in units],
            "units": units,
            "process_wall_seconds_including_model_load": total,
            "gpu_hours": total / 3600.0,
            "canonical_runner": str(FORMAL_RUNNER),
            "canonical_runner_sha256": sha256(FORMAL_RUNNER),
            "no_metrics_retained": True,
            "no_training": True,
            "no_free_generation": True,
            "AIME_used": False,
        }
        atomic_json(OUT, result)
        print(json.dumps({"status": "PASS", "gpu_hours": result["gpu_hours"], "units": len(units)}, sort_keys=True))
    except BaseException as exc:
        atomic_json(
            ERR,
            {
                "status": "FAIL",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "runtime_seconds": time.time() - started,
                "gpu_hours": (time.time() - started) / 3600.0,
            },
        )
        raise


if __name__ == "__main__":
    main()
