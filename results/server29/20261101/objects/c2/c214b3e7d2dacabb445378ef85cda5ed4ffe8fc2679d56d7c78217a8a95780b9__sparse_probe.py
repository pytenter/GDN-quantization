#!/usr/bin/env python3
"""Budgeted FP-path persistent recurrent replay for C5/C6 diagnosis.

This is inference-only.  It imports the frozen formal runner and calls its
rotation patches and canonical quantize_c128 implementation directly.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
import torch


ROOT = Path("experiments/QWEN_GDN_C5_C6_EVIDENCE_GUIDED_DIAGNOSIS_V1")
LEGACY = Path("/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen")
AIME = LEGACY / "experiments/qwen_gdn/QWEN_C5_C6_AIME26_CANONICAL60_V1"
DEPLOY = LEGACY / "experiments/qwen_gdn/QWEN_C5_C6_DEPLOYMENT_CLOSURE_V1"
FORMAL_RUNNER = AIME / "scripts/run_qwen_c5_c6_formal.py"
BASELINE_CSV = Path("/data/zypan/consolidation/GDN-quantization-qwen3090-20260921/results/aime26/81920/qwen_gdn/frozen_v4_per_sample.csv")
PLAN = ROOT / "hypothesis_and_probe_plan.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    os.replace(tmp, path)


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("canonical_c5c6_formal", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_fp_rows(sample_ids: list[str]) -> dict[str, dict]:
    wanted = set(sample_ids)
    source_files = set()
    with BASELINE_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sample_id = f"{row['problem_id']}_seed{row['seed']}"
            if row["condition"] == "QWEN_FP_STATE" and sample_id in wanted:
                source_files.add(row["source_file"])
    found = {}
    for source in sorted(source_files):
        with open(source, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                sample_id = f"{row['problem_id']}_seed{row['seed']}"
                if row.get("method") == "QWEN_FP_STATE" and sample_id in wanted:
                    found[sample_id] = row
    if set(found) != wanted:
        raise RuntimeError(f"FP token source identity mismatch missing={sorted(wanted-set(found))}")
    return found


def state_metric(value: torch.Tensor, reference: torch.Tensor) -> dict:
    a = value.detach().float()
    b = reference.detach().float().to(a.device)
    diff = a - b
    denom = float(torch.linalg.vector_norm(b).item())
    absolute = float(torch.linalg.vector_norm(diff).item())
    return {
        "absolute_l2": absolute,
        "reference_l2": denom,
        "relative_l2": absolute / max(denom, 1e-12),
        "near_zero_denominator": denom <= 1e-12,
        "max_abs_error": float(diff.abs().max().item()),
        "mean_abs_error": float(diff.abs().mean().item()),
    }


def physical_state(formal, condition: str, layer: int, state: torch.Tensor, banks: dict) -> torch.Tensor:
    x = state.detach().float()
    if condition == "FP":
        return x
    h = formal.hadamard(128, dtype=torch.float32, device=x.device)
    if condition in ("C5", "C6"):
        delta = banks[condition].layer(layer).matrix().float()
        x = torch.einsum("ij,...jv->...iv", delta, x)
    return torch.einsum("ij,...jv->...iv", h, x)


class OutputCapture:
    def __init__(self, formal, model):
        self.formal = formal
        self.active = False
        self.values = {}
        self.handles = []
        layers = model.model.layers if hasattr(model.model, "layers") else model.model.language_model.layers
        for idx in formal.GDN_LAYERS:
            module = layers[idx].linear_attn.out_proj
            self.handles.append(module.register_forward_hook(self._hook(idx)))

    def _hook(self, idx):
        def fn(_module, _args, output):
            if self.active:
                value = output[0] if isinstance(output, (tuple, list)) else output
                self.values[idx] = value.detach().float().cpu().clone()
        return fn

    def close(self):
        for handle in self.handles:
            handle.remove()


def init_audit():
    return {"calls": 0, "layers": set(), "scale_shapes": {}, "code_min": 0, "code_max": 0}


def condition_patch(formal, model, condition, banks, identities):
    if condition == "FP":
        return contextlib.nullcontext()
    if condition == "H":
        return formal.QwenKeyHadamardPatch(model)
    return formal.QwenFrozenRotationPatch(model, banks[condition], identities[condition])


def model_metrics(fp_logits: torch.Tensor, logits: torch.Tensor, target: int) -> dict:
    fp = fp_logits.float().to(logits.device)
    cur = logits.float()
    fp_logp = torch.log_softmax(fp, dim=-1)
    cur_logp = torch.log_softmax(cur, dim=-1)
    fp_p = torch.exp(fp_logp)
    kl = torch.sum(fp_p * (fp_logp - cur_logp), dim=-1)
    fp_top = int(torch.argmax(fp, dim=-1).item())
    mask = torch.ones(cur.shape[-1], dtype=torch.bool, device=cur.device)
    mask[fp_top] = False
    fp_margin = float((fp[..., fp_top] - fp[..., mask].max(dim=-1).values).item())
    cur_margin = float((cur[..., fp_top] - cur[..., mask].max(dim=-1).values).item())
    return {
        "kl_fp_to_condition": float(kl.item()),
        "delta_nll_fixed_fp_next_token": float((-cur_logp[..., target] + fp_logp[..., target]).item()),
        "fp_reference_token": fp_top,
        "fixed_next_token": int(target),
        "fp_reference_margin": fp_margin,
        "condition_margin_for_fp_reference": cur_margin,
        "margin_degradation": fp_margin - cur_margin,
        "top1_agreement": fp_top == int(torch.argmax(cur, dim=-1).item()),
    }


def run_condition(formal, model, tokenizer, source, condition, positions, max_steps,
                  banks, identities, fp_reference, rows):
    device = next(model.parameters()).device
    formal.setup_seed(int(source["seed"]))
    _messages, input_ids = formal.render_input(tokenizer, source["problem"])
    input_ids = input_ids.to(device)
    tokens = [int(x) for x in source["generated_token_ids"]]
    steps = min(int(max_steps), max(0, len(tokens) - 1))
    legal = {int(x) for x in positions if int(x) < steps}
    audit = init_audit()
    capture = OutputCapture(formal, model)
    sample_id = f"{source['problem_id']}_seed{source['seed']}"
    started = time.time()
    torch.cuda.reset_peak_memory_stats(device)
    before_alloc = int(torch.cuda.memory_allocated(device))
    try:
        with condition_patch(formal, model, condition, banks, identities) as patch, torch.inference_mode():
            output = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=True)
            past = output.past_key_values
            for token_index in range(steps):
                capture.active = token_index in legal
                capture.values = {}
                current = torch.tensor([[tokens[token_index]]], device=device, dtype=torch.long)
                output = model(input_ids=current, past_key_values=past, use_cache=True)
                past = output.past_key_values
                logits = output.logits[:, -1, :].detach().float()
                pre = None
                if token_index in legal and condition != "FP":
                    pre = {layer: formal.get_state(past, layer).detach().float().clone() for layer in formal.GDN_LAYERS}
                if condition != "FP":
                    formal.quantize_c128(past, audit)
                if token_index not in legal:
                    continue
                if set(capture.values) != set(formal.GDN_LAYERS):
                    raise RuntimeError(f"out_proj capture incomplete at {sample_id}/{condition}/{token_index}")
                target = tokens[token_index + 1]
                if condition == "FP":
                    fp_reference["logits"][token_index] = logits.cpu().clone()
                    fp_reference["outputs"][token_index] = {k: v.clone() for k, v in capture.values.items()}
                    fp_reference["states"][token_index] = {
                        layer: physical_state(formal, "FP", layer, formal.get_state(past, layer), banks).cpu().clone()
                        for layer in formal.GDN_LAYERS
                    }
                    continue
                mm = model_metrics(fp_reference["logits"][token_index], logits, target)
                rows.append({
                    "sample_id": sample_id, "question_id": source["problem_id"], "seed": int(source["seed"]),
                    "condition": condition, "semantic": "FP_PATH_PERSISTENT_REPLAY",
                    "token_index_consumed": token_index, "next_token_target_index": token_index + 1,
                    "scope": "model_output", "layer": -1, **mm,
                })
                for layer in formal.GDN_LAYERS:
                    post_native = formal.get_state(past, layer).detach().float()
                    post_physical = physical_state(formal, condition, layer, post_native, banks)
                    drift = state_metric(post_physical, fp_reference["states"][token_index][layer])
                    local = state_metric(post_native, pre[layer])
                    rows.append({
                        "sample_id": sample_id, "question_id": source["problem_id"], "seed": int(source["seed"]),
                        "condition": condition, "semantic": "FP_PATH_PERSISTENT_REPLAY",
                        "token_index_consumed": token_index, "next_token_target_index": token_index + 1,
                        "scope": "state_fp_relative_post_qdq_physical", "layer": int(layer), **drift,
                    })
                    rows.append({
                        "sample_id": sample_id, "question_id": source["problem_id"], "seed": int(source["seed"]),
                        "condition": condition, "semantic": "FP_PATH_PERSISTENT_REPLAY",
                        "token_index_consumed": token_index, "next_token_target_index": token_index + 1,
                        "scope": "state_local_qdq_native_basis", "layer": int(layer), **local,
                    })
                    functional = state_metric(capture.values[layer], fp_reference["outputs"][token_index][layer])
                    rows.append({
                        "sample_id": sample_id, "question_id": source["problem_id"], "seed": int(source["seed"]),
                        "condition": condition, "semantic": "FP_PATH_PERSISTENT_REPLAY",
                        "token_index_consumed": token_index, "next_token_target_index": token_index + 1,
                        "scope": "linear_attn_out_proj_output_fp_relative", "layer": int(layer), **functional,
                    })
                del pre
            patch_record = None if condition == "FP" else {
                "calls": int(patch.calls), "layers": sorted(int(x) for x in patch.layers),
                "phase_counts": dict(patch.phase_counts),
            }
        elapsed = time.time() - started
        peak = int(torch.cuda.max_memory_allocated(device))
        del output, past
        gc.collect(); torch.cuda.empty_cache()
        after_alloc = int(torch.cuda.memory_allocated(device))
        return {
            "sample_id": sample_id, "condition": condition, "steps": steps,
            "capture_positions": sorted(legal), "elapsed_seconds": elapsed,
            "tokens_per_second": steps / elapsed if elapsed else None,
            "quantizer_calls": int(audit["calls"]), "quantizer_layers": sorted(int(x) for x in audit["layers"]),
            "patch_audit": patch_record, "memory_allocated_before": before_alloc,
            "memory_allocated_after_cleanup": after_alloc, "peak_memory_allocated": peak,
        }
    finally:
        capture.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "main"), required=True)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    cfg = plan["sparse_protocol"][args.mode]
    sample_ids = list(cfg["sample_ids"])
    positions = list(cfg["capture_positions"])
    max_steps = int(cfg["max_decode_steps"])
    sources = load_fp_rows(sample_ids)
    formal = load_module(FORMAL_RUNNER)
    process_start = time.time()
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    run_path = ROOT / "analysis" / f"{args.mode}_run_summary.json"
    error_path = ROOT / "analysis" / f"{args.mode}_run_error.json"
    if error_path.exists():
        raise RuntimeError(f"refusing to overwrite prior failure evidence: {error_path}")
    rows, units = [], []
    try:
        model, tokenizer = formal.load_model_and_tokenizer()
        device = next(model.parameters()).device
        banks, identities = {}, {}
        for condition in ("C5", "C6"):
            banks[condition], identities[condition] = formal.load_frozen_rotation(
                DEPLOY / f"deployment/{condition}/frozen_rotation_fp32.pt",
                DEPLOY / f"deployment/{condition}/deployment_manifest.json",
                device, condition,
            )
        for sample_id in sample_ids:
            fp_reference = {"states": {}, "outputs": {}, "logits": {}}
            for condition in ("FP", "H", "C5", "C6"):
                units.append(run_condition(formal, model, tokenizer, sources[sample_id], condition,
                                           positions, max_steps, banks, identities, fp_reference, rows))
            del fp_reference
            gc.collect(); torch.cuda.empty_cache()
        output = ROOT / ("sparse_metrics.parquet" if args.mode == "main" else "analysis/smoke_metrics.parquet")
        atomic_parquet(output, rows)
        elapsed = time.time() - process_start
        result = {
            "status": "PASS", "mode": args.mode, "started_utc": started_utc,
            "ended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "process_wall_seconds": elapsed, "gpu_count": 1, "gpu_hours": elapsed / 3600.0,
            "sample_ids": sample_ids, "max_decode_steps": max_steps,
            "capture_positions": positions, "metric_rows": len(rows),
            "metrics_file": str(output), "metrics_sha256": sha256(output), "units": units,
            "canonical_runner": str(FORMAL_RUNNER), "canonical_runner_sha256": sha256(FORMAL_RUNNER),
            "no_free_generation": True, "no_training": True,
        }
        atomic_json(run_path, result)
        print(json.dumps({k: result[k] for k in ("status", "mode", "process_wall_seconds", "gpu_hours", "metric_rows")}, sort_keys=True))
    except Exception:
        elapsed = time.time() - process_start
        atomic_json(error_path, {
            "status": "FAIL", "mode": args.mode, "started_utc": started_utc,
            "ended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "process_wall_seconds": elapsed, "gpu_hours": elapsed / 3600.0,
            "traceback": traceback.format_exc(), "completed_units": units,
        })
        raise


if __name__ == "__main__":
    main()
