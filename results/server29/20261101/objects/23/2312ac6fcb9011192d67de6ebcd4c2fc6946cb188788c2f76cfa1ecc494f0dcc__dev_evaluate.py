#!/usr/bin/env python3
"""Persistent teacher-forced independent dev screen, partitioned SCREEN/CHECK."""

from __future__ import annotations

import argparse
import contextlib
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


ROOT = Path("experiments/QWEN_GDN_SHARED_OBJECTIVE_PARETO_AND_DEV_SCREEN_V1")
ANALYSIS = ROOT / "analysis"
PANEL = ANALYSIS / "independent_dev_panel.jsonl"
PROTOCOL = ANALYSIS / "independent_dev_protocol.json"
LEGACY = Path("/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen")
FORMAL_RUNNER = LEGACY / "experiments/qwen_gdn/QWEN_C5_C6_AIME26_CANONICAL60_V1/scripts/run_qwen_c5_c6_formal.py"
DEPLOY = LEGACY / "experiments/qwen_gdn/QWEN_C5_C6_DEPLOYMENT_CLOSURE_V1/deployment"
CONDITIONS = ("FP", "HADAMARD", "C5_U63", "C5_U113", "C6_U119", "C6_U125")


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


def atomic_parquet(path: Path, rows: list[dict]) -> None:
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    os.replace(tmp, path)


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("formal_independent_dev", str(path))
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


def physical_state(formal, condition: str, layer: int, state, banks: dict):
    torch = __import__("torch")
    value = state.detach().float()
    if condition == "FP":
        return value
    h = formal.hadamard(128, dtype=torch.float32, device=value.device)
    if condition not in ("HADAMARD", "FP"):
        delta = banks[condition].layer(layer).matrix().float()
        value = torch.einsum("ij,...jv->...iv", delta, value)
    return torch.einsum("ij,...jv->...iv", h, value)


def exact_state_relative_mse(value, reference) -> float:
    torch = __import__("torch")
    a = value.float()
    b = reference.float().to(a.device)
    numerator = (a - b).square().sum(dim=(-2, -1))
    denominator = b.square().sum(dim=(-2, -1)).clamp_min(1e-12)
    return float((numerator / denominator).mean().item())


def exact_functional_relative_mse(value, reference) -> float:
    a = value.float()
    b = reference.float().to(a.device)
    return float(((a - b).square().sum() / b.square().sum().clamp_min(1e-12)).item())


class Capture:
    def __init__(self, formal, model):
        self.active = False
        self.outputs = {}
        self.final_hidden = None
        self.handles = []
        layers = model.model.layers if hasattr(model.model, "layers") else model.model.language_model.layers
        for layer in formal.GDN_LAYERS:
            self.handles.append(layers[layer].linear_attn.out_proj.register_forward_hook(self._out_hook(layer)))
        self.handles.append(model.lm_head.register_forward_pre_hook(self._lm_head_hook))

    def _out_hook(self, layer: int):
        def hook(_module, _args, output):
            if self.active:
                value = output[0] if isinstance(output, (tuple, list)) else output
                self.outputs[layer] = value.detach().float().cpu().clone()
        return hook

    def _lm_head_hook(self, _module, args):
        if self.active:
            self.final_hidden = args[0].detach().float().cpu().clone()

    def reset(self, active: bool) -> None:
        self.active = active
        self.outputs = {}
        self.final_hidden = None

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def output_metrics(fp_logits, logits, target: int) -> dict:
    torch = __import__("torch")
    fp = fp_logits.float().to(logits.device)
    cur = logits.float()
    fp_logp = torch.log_softmax(fp, dim=-1)
    cur_logp = torch.log_softmax(cur, dim=-1)
    kl = torch.sum(torch.exp(fp_logp) * (fp_logp - cur_logp), dim=-1)
    fp_top = int(torch.argmax(fp, dim=-1).item())
    fp_other = torch.cat((fp[..., :fp_top], fp[..., fp_top + 1 :]), dim=-1)
    cur_other = torch.cat((cur[..., :fp_top], cur[..., fp_top + 1 :]), dim=-1)
    fp_margin = float((fp[..., fp_top] - fp_other.max(dim=-1).values).item())
    cur_margin = float((cur[..., fp_top] - cur_other.max(dim=-1).values).item())
    return {
        "full_vocab_kl_fp_to_condition": float(kl.item()),
        "fp_fixed_reference_nll": float((-fp_logp[..., target]).item()),
        "condition_fixed_reference_nll": float((-cur_logp[..., target]).item()),
        "delta_nll_fixed_reference_token": float((-cur_logp[..., target] + fp_logp[..., target]).item()),
        "fixed_reference_token": int(target),
        "fixed_reference_token_source": "MATH-500 archived solution next token",
        "fp_top1_reference_token": fp_top,
        "fp_reference_margin": fp_margin,
        "condition_margin_for_fp_reference": cur_margin,
        "fp_reference_margin_degradation": fp_margin - cur_margin,
        "top1_agreement_with_fp": fp_top == int(torch.argmax(cur, dim=-1).item()),
    }


def run_condition(formal, model, tokenizer, document: dict, condition: str, banks: dict, identities: dict, fp_ref: dict, rows: list[dict]):
    torch = __import__("torch")
    device = next(model.parameters()).device
    _messages, prompt_ids = formal.render_input(tokenizer, document["problem"])
    prompt_ids = prompt_ids.to(device)
    teacher = tokenizer(document["solution"], add_special_tokens=False)["input_ids"]
    if len(teacher) != int(document["teacher_solution_token_count"]):
        raise RuntimeError(f"teacher tokenization drift for {document['document_id']}")
    steps = int(document["recurrent_updates"])
    captures = set(int(x) for x in document["capture_positions"])
    if steps != min(2048, len(teacher) - 1):
        raise RuntimeError("frozen update count drift")
    formal.setup_seed(0)
    audit = init_audit()
    capture = Capture(formal, model)
    started = time.time()
    torch.cuda.reset_peak_memory_stats(device)
    before = int(torch.cuda.memory_allocated(device))
    try:
        with patch_for(formal, model, condition, banks, identities) as patch, torch.inference_mode():
            output = model(input_ids=prompt_ids, attention_mask=torch.ones_like(prompt_ids), use_cache=True)
            past = output.past_key_values
            for index in range(steps):
                active = index in captures
                capture.reset(active)
                pre_states = None
                if active:
                    pre_states = {
                        layer: physical_state(formal, condition, layer, formal.get_state(past, layer), banks).cpu().clone()
                        for layer in formal.GDN_LAYERS
                    }
                token = torch.tensor([[teacher[index]]], dtype=torch.long, device=device)
                output = model(input_ids=token, past_key_values=past, use_cache=True)
                past = output.past_key_values
                logits = output.logits[:, -1, :].detach().float()
                if condition != "FP":
                    formal.quantize_c128(past, audit)
                if not active:
                    continue
                if set(capture.outputs) != set(formal.GDN_LAYERS) or capture.final_hidden is None:
                    raise RuntimeError(f"capture incomplete {document['document_id']} {condition} {index}")
                target = int(teacher[index + 1])
                if condition == "FP":
                    fp_ref[index] = {
                        "states": pre_states,
                        "outputs": {k: v.clone() for k, v in capture.outputs.items()},
                        "final_hidden": capture.final_hidden.clone(),
                        "logits": logits.cpu().clone(),
                    }
                    continue
                state_by_layer = [exact_state_relative_mse(pre_states[layer], fp_ref[index]["states"][layer]) for layer in formal.GDN_LAYERS]
                functional_by_layer = [exact_functional_relative_mse(capture.outputs[layer], fp_ref[index]["outputs"][layer]) for layer in formal.GDN_LAYERS]
                row = {
                    "document_id": document["document_id"],
                    "partition": document["partition"],
                    "condition": condition,
                    "token_index_consumed": index,
                    "next_token_target_index": index + 1,
                    "state_relative_mse": sum(state_by_layer) / len(state_by_layer),
                    "functional_relative_mse": sum(functional_by_layer) / len(functional_by_layer),
                    "downstream_final_hidden_relative_mse": exact_functional_relative_mse(capture.final_hidden, fp_ref[index]["final_hidden"]),
                    **output_metrics(fp_ref[index]["logits"], logits, target),
                }
                if not all(math.isfinite(float(v)) for k, v in row.items() if isinstance(v, float)):
                    raise FloatingPointError(f"nonfinite metric {document['document_id']} {condition} {index}")
                rows.append(row)
            patch_record = None if condition == "FP" else {
                "calls": int(patch.calls),
                "layers": sorted(int(x) for x in patch.layers),
                "phase_counts": dict(patch.phase_counts),
            }
        torch.cuda.synchronize(device)
        elapsed = time.time() - started
        peak = int(torch.cuda.max_memory_allocated(device))
        if condition != "FP":
            if audit["calls"] != steps or sorted(audit["layers"]) != list(formal.GDN_LAYERS):
                raise RuntimeError(f"QDQ audit mismatch for {document['document_id']} {condition}")
            if patch_record["phase_counts"].get("recurrent_decode") != steps * len(formal.GDN_LAYERS):
                raise RuntimeError(f"recurrent patch count mismatch for {document['document_id']} {condition}")
        del output, past
        gc.collect()
        torch.cuda.empty_cache()
        after = int(torch.cuda.memory_allocated(device))
        return {
            "document_id": document["document_id"],
            "partition": document["partition"],
            "condition": condition,
            "steps": steps,
            "capture_positions": sorted(captures),
            "elapsed_seconds": elapsed,
            "tokens_per_second": steps / elapsed,
            "quantizer_calls": int(audit["calls"]),
            "quantizer_layers": sorted(int(x) for x in audit["layers"]),
            "patch_audit": patch_record,
            "memory_allocated_before": before,
            "peak_memory_allocated": peak,
            "memory_allocated_after_cleanup": after,
        }
    finally:
        capture.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", choices=("SCREEN", "CHECK"), required=True)
    args = parser.parse_args()
    lower = args.partition.lower()
    output = ANALYSIS / f"{lower}_capture_metrics.parquet"
    summary_path = ANALYSIS / f"{lower}_run_summary.json"
    error_path = ANALYSIS / f"{lower}_run_error.json"
    if output.exists() or summary_path.exists() or error_path.exists():
        raise RuntimeError(f"{args.partition} run already attempted")
    protocol = json.loads(PROTOCOL.read_text())
    if protocol.get("status") != "FROZEN_BEFORE_CANDIDATE_METRICS" or sha256(PANEL) != protocol["panel_sha256"]:
        raise RuntimeError("frozen panel identity mismatch")
    documents = [json.loads(line) for line in PANEL.read_text().splitlines() if line.strip()]
    documents = [x for x in documents if x["partition"] == args.partition]
    expected_ids = protocol[f"{lower}_document_ids"]
    if [x["document_id"] for x in documents] != expected_ids or len(documents) != 4:
        raise RuntimeError("partition identity mismatch")
    started = time.time()
    rows, units = [], []
    try:
        formal = load_module(FORMAL_RUNNER)
        model, tokenizer = formal.load_model_and_tokenizer()
        device = next(model.parameters()).device
        banks, identities = {}, {}
        specs = {
            "C5_U63": (DEPLOY / "C5/frozen_rotation_fp32.pt", DEPLOY / "C5/deployment_manifest.json", "C5"),
            "C6_U125": (DEPLOY / "C6/frozen_rotation_fp32.pt", DEPLOY / "C6/deployment_manifest.json", "C6"),
            "C5_U113": (ROOT / "candidate_artifacts/C5_U113/frozen_rotation_fp32.pt", ROOT / "candidate_artifacts/C5_U113/deployment_manifest.json", "C5_U113"),
            "C6_U119": (ROOT / "candidate_artifacts/C6_U119/frozen_rotation_fp32.pt", ROOT / "candidate_artifacts/C6_U119/deployment_manifest.json", "C6_U119"),
        }
        for condition, (rotation, manifest, expected) in specs.items():
            banks[condition], identities[condition] = formal.load_frozen_rotation(rotation, manifest, device, expected)
        for document in documents:
            fp_ref = {}
            for condition in CONDITIONS:
                units.append(run_condition(formal, model, tokenizer, document, condition, banks, identities, fp_ref, rows))
            if set(fp_ref) != set(document["capture_positions"]):
                raise RuntimeError(f"FP capture coverage mismatch for {document['document_id']}")
            del fp_ref
            gc.collect()
            __import__("torch").cuda.empty_cache()
        atomic_parquet(output, rows)
        elapsed = time.time() - started
        result = {
            "status": "PASS",
            "partition": args.partition,
            "documents": expected_ids,
            "conditions": list(CONDITIONS),
            "capture_metric_rows": len(rows),
            "metrics_path": str(output),
            "metrics_sha256": sha256(output),
            "units": units,
            "process_wall_seconds_including_model_load": elapsed,
            "gpu_hours": elapsed / 3600.0,
            "panel_sha256": protocol["panel_sha256"],
            "canonical_runner": str(FORMAL_RUNNER),
            "canonical_runner_sha256": sha256(FORMAL_RUNNER),
            "no_training": True,
            "no_free_generation": True,
            "AIME_used": False,
        }
        atomic_json(summary_path, result)
        print(json.dumps({"status": "PASS", "partition": args.partition, "gpu_hours": result["gpu_hours"], "rows": len(rows)}, sort_keys=True))
    except BaseException as exc:
        elapsed = time.time() - started
        atomic_json(
            error_path,
            {
                "status": "FAIL",
                "partition": args.partition,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "runtime_seconds": elapsed,
                "gpu_hours": elapsed / 3600.0,
                "completed_units": units,
            },
        )
        raise


if __name__ == "__main__":
    main()
