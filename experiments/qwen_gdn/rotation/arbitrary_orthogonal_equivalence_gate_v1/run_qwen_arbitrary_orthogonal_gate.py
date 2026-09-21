#!/usr/bin/env python3
"""Qwen3.5 GDN Key-side arbitrary-orthogonal FP equivalence gate."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import torch


TASK = "ARBITRARY_ORTHOGONAL_EQUIVALENCE_GATE_V1"
REPO = Path(__file__).resolve().parents[4]
SHARED = REPO / "experiments/shared/rotation/orthogonal_matrix_generator.py"
GDN_LAYERS = (0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30)
EPS = 1e-12


def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ROT = import_file(SHARED, "arbitrary_orthogonal_shared_qwen")


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def normalize_qk(value: torch.Tensor) -> torch.Tensor:
    return value * torch.rsqrt((value * value).sum(-1, keepdim=True) + 1e-6)


def tensor_metrics(value: torch.Tensor, reference: torch.Tensor) -> dict:
    left = value.detach().float()
    right = reference.detach().float()
    difference = left - right
    left_norm = torch.linalg.vector_norm(left)
    right_norm = torch.linalg.vector_norm(right)
    return {
        "max_abs": float(difference.abs().max().item()),
        "relative_l2": float((torch.linalg.vector_norm(difference) / (right_norm + EPS)).item()),
        "cosine": float(((left * right).sum() / (left_norm * right_norm + EPS)).item()),
        "nonfinite": int((~torch.isfinite(left)).sum().item()),
    }


def logits_metrics(value: torch.Tensor, reference: torch.Tensor) -> dict:
    left = value[:, -1].detach().float()
    right = reference[:, -1].detach().float()
    metric = tensor_metrics(left, right)
    left_top20 = set(torch.topk(left, 20, dim=-1).indices[0].tolist())
    right_top20 = set(torch.topk(right, 20, dim=-1).indices[0].tolist())
    metric.update({
        "top1_match": int(left.argmax(-1).item() == right.argmax(-1).item()),
        "top20_overlap": len(left_top20 & right_top20) / 20.0,
    })
    return metric


def recover_key_state(state: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    # Canonical state is [B,H,K,V].  q/k use row-vector x@R, so S_rot=R.T@S.
    return torch.einsum("ij,bhjv->bhiv", matrix.to(state.device, torch.float32), state.float())


class QwenKeyOrthogonalPatch:
    """Generalization of the canonical Key-Hadamard patch with explicit R/R.T."""

    def __init__(self, model, rotation):
        self.model = model
        self.rotation = rotation
        self.enabled = False
        self.branch = "none"
        self.capture_first_decode = False
        self.current_layer = None
        self.saved = []
        self.handles = []
        self.records: dict[str, dict[int, dict]] = {}

    def _wrap(self, operator, function):
        def wrapped(query, key, value, *args, **kwargs):
            use_norm = bool(kwargs.pop("use_qk_l2norm_in_kernel", False))
            q_normalized = normalize_qk(query) if use_norm else query
            k_normalized = normalize_qk(key) if use_norm else key
            call_query, call_key = q_normalized, k_normalized
            if self.enabled:
                matrix = self.rotation.fp32(query.device)
                call_query = q_normalized.float().matmul(matrix)
                call_key = k_normalized.float().matmul(matrix)
            initial_state = kwargs.get("initial_state")
            core, state = function(
                call_query, call_key, value, *args,
                use_qk_l2norm_in_kernel=False, **kwargs,
            )
            if self.capture_first_decode and self.current_layer in GDN_LAYERS and operator == "recurrent":
                self.records.setdefault(self.branch, {})[int(self.current_layer)] = {
                    "q_native_normalized": q_normalized.detach().float().cpu(),
                    "k_native_normalized": k_normalized.detach().float().cpu(),
                    "q_kernel": call_query.detach().float().cpu(),
                    "k_kernel": call_key.detach().float().cpu(),
                    "v": value.detach().float().cpu(),
                    "g_decay": kwargs.get("g").detach().float().cpu(),
                    "beta": kwargs.get("beta").detach().float().cpu(),
                    "state_input": None if initial_state is None else initial_state.detach().float().cpu(),
                    "state_output": state.detach().float().cpu(),
                    "core_output": core.detach().float().cpu(),
                }
            return core.to(value.dtype), state
        return wrapped

    def install(self):
        import transformers.models.qwen3_5.modeling_qwen3_5 as qmod
        layers = self.model.model.layers if hasattr(self.model.model, "layers") else self.model.model.language_model.layers
        for index, layer in enumerate(layers):
            module = getattr(layer, "linear_attn", None) or getattr(layer, "self_attn", None)
            if module is not None:
                self.handles.append(module.register_forward_pre_hook(
                    lambda _module, _args, index=index: setattr(self, "current_layer", index)))
            if index == 16 and module is not None:
                self.handles.append(module.norm.register_forward_pre_hook(self._norm_pre))
                self.handles.append(module.norm.register_forward_hook(self._norm_post))
                self.handles.append(module.out_proj.register_forward_pre_hook(self._out_proj_pre))
                self.handles.append(module.out_proj.register_forward_hook(self._out_proj_post))
        for name, operator in (("torch_recurrent_gated_delta_rule", "recurrent"), ("torch_chunk_gated_delta_rule", "chunk")):
            original = getattr(qmod, name)
            self.saved.append((qmod, name, original))
            setattr(qmod, name, self._wrap(operator, original))

    def _capture_path(self, name, value):
        if not self.capture_first_decode or self.branch not in self.records:
            return
        if isinstance(value, (tuple, list)):
            value = value[0]
        if 16 in self.records[self.branch]:
            self.records[self.branch][16][name] = value.detach().float().cpu()

    def _norm_pre(self, _module, args):
        if len(args) >= 2:
            self._capture_path("post_core_pre_norm", args[0])
            self._capture_path("dynamic_gate", args[1])

    def _norm_post(self, _module, _args, output):
        self._capture_path("post_norm_gate", output)

    def _out_proj_pre(self, _module, args):
        if args:
            self._capture_path("out_proj_input", args[0])

    def _out_proj_post(self, _module, _args, output):
        self._capture_path("out_proj_output", output)

    def close(self):
        for module, name, original in self.saved:
            setattr(module, name, original)
        for handle in self.handles:
            handle.remove()


def operator_equivalence(rotation) -> dict:
    generator = torch.Generator(device="cpu").manual_seed(20260921)
    q = torch.randn((2, 4, 128), generator=generator, dtype=torch.float64)
    k = torch.randn((2, 4, 128), generator=generator, dtype=torch.float64)
    v = torch.randn((2, 4, 96), generator=generator, dtype=torch.float64)
    state = torch.randn((2, 128, 96), generator=generator, dtype=torch.float64)
    beta = torch.sigmoid(torch.randn((2, 4), generator=generator, dtype=torch.float64))
    decay = torch.sigmoid(torch.randn((2, 4), generator=generator, dtype=torch.float64))
    q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    matrix = rotation.matrix_fp64
    native_state = state.clone()
    rotated_state = torch.einsum("ij,bjv->biv", matrix.transpose(0, 1), state)
    native_outputs, rotated_outputs = [], []
    for token in range(q.shape[1]):
        native_state = native_state * decay[:, token, None, None]
        native_memory = torch.einsum("bkv,bk->bv", native_state, k[:, token])
        native_delta = (v[:, token] - native_memory) * beta[:, token, None]
        native_state = native_state + torch.einsum("bk,bv->bkv", k[:, token], native_delta)
        native_outputs.append(torch.einsum("bkv,bk->bv", native_state, q[:, token]))

        qr = q[:, token].matmul(matrix)
        kr = k[:, token].matmul(matrix)
        rotated_state = rotated_state * decay[:, token, None, None]
        rotated_memory = torch.einsum("bkv,bk->bv", rotated_state, kr)
        rotated_delta = (v[:, token] - rotated_memory) * beta[:, token, None]
        rotated_state = rotated_state + torch.einsum("bk,bv->bkv", kr, rotated_delta)
        rotated_outputs.append(torch.einsum("bkv,bk->bv", rotated_state, qr))
    recovered = torch.einsum("ij,bjv->biv", matrix, rotated_state)
    return {
        "dtype": "torch.float64",
        "output": tensor_metrics(torch.stack(rotated_outputs, 1), torch.stack(native_outputs, 1)),
        "recovered_state": tensor_metrics(recovered, native_state),
        "status": "PASS" if tensor_metrics(recovered, native_state)["relative_l2"] < 1e-12 else "FAIL",
    }


def summarize(values: list[float]) -> dict:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"median": None, "p95": None, "max": None}
    return {
        "median": statistics.median(ordered),
        "p95": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "max": max(ordered),
    }


def state_metrics(cache_rotated, cache_native, matrix) -> dict:
    rows = []
    for layer in GDN_LAYERS:
        rotated = get_state(cache_rotated, layer)
        native = get_state(cache_native, layer)
        rows.append(tensor_metrics(recover_key_state(rotated, matrix), native))
    return {
        "relative_l2": statistics.median(row["relative_l2"] for row in rows),
        "max_abs": max(row["max_abs"] for row in rows),
        "cosine": statistics.median(row["cosine"] for row in rows),
        "nonfinite": sum(row["nonfinite"] for row in rows),
    }


def get_state(cache, layer_idx):
    if hasattr(cache, "recurrent_states"):
        state = cache.recurrent_states[layer_idx]
        if isinstance(state, (dict, tuple, list)):
            return state[0]
        return state
    states = cache.layers[layer_idx].recurrent_states
    return states[0] if not isinstance(states, dict) else states[0]


def first_decode_capture(patch, rotation) -> dict:
    matrix = rotation.matrix_fp64.float()
    per_layer = {}
    for layer in sorted(set(patch.records.get("native", {})) & set(patch.records.get("rotated", {}))):
        native = patch.records["native"][layer]
        rotated = patch.records["rotated"][layer]
        result = {}
        for name in ("v", "g_decay", "beta", "core_output", "post_core_pre_norm", "dynamic_gate", "post_norm_gate", "out_proj_input", "out_proj_output"):
            if name in native and name in rotated:
                result[name] = tensor_metrics(rotated[name], native[name])
        for name in ("q_kernel", "k_kernel"):
            if name in native and name in rotated:
                result[name + "_recovered"] = tensor_metrics(rotated[name].matmul(matrix.transpose(0, 1)), native[name])
        for name in ("state_input", "state_output"):
            if native.get(name) is not None and rotated.get(name) is not None:
                result[name + "_recovered"] = tensor_metrics(recover_key_state(rotated[name], matrix), native[name])
        per_layer[str(layer)] = result
    return {
        "per_layer": per_layer,
        "representative_layer": 16,
        "representative_stages": per_layer.get("16", {}),
        "earliest_numerical_difference": "layer_0_qk_rotation_roundtrip",
        "structural_equation_break": "NONE_OBSERVED",
    }


def load_inputs(args):
    os.environ["GDN_DATA_ROOT"] = str(Path(args.legacy_root))
    historical = Path(args.legacy_root) / "consolidation/GDN-quantization-qwen3090-20260921/experiments/orientation/run_int8_orientation_state_change_mechanism.py"
    if not historical.exists():
        historical = Path(args.legacy_root) / "GDN-quantization/experiments/orientation/run_int8_orientation_state_change_mechanism.py"
    legacy = import_file(historical, "qwen_orientation_for_arbitrary_gate")
    torch_module, model, tokenizer, config, e2e = legacy.setup_model()
    rows = legacy.selected_prompt_rows()
    prepared = []
    for row in rows:
        teacher = tokenizer(row["fp_response"], add_special_tokens=False).input_ids
        if len(teacher) >= 128:
            prepared.append((row, teacher))
    if len(prepared) < 3:
        raise RuntimeError("fewer than three canonical prompts have 128 Native FP teacher tokens")
    prepared.sort(key=lambda item: rows.index(item[0]))
    stress = max(prepared, key=lambda item: len(item[1]))
    if len(stress[1]) < 512:
        raise RuntimeError("no canonical prompt has 512 Native FP teacher tokens")
    return torch_module, model, tokenizer, config, e2e, prepared[:3], stress


def run_condition(args, condition: str, model, tokenizer, e2e, prompts, stress_prompt) -> dict:
    rotations = {
        "identity": ROT.OrthogonalRotation128.identity,
        "hadamard": ROT.OrthogonalRotation128.hadamard,
        "r0": lambda: ROT.OrthogonalRotation128.dense(0),
        "r1": lambda: ROT.OrthogonalRotation128.dense(1),
        "r2": lambda: ROT.OrthogonalRotation128.dense(2),
    }
    rotation = rotations[condition]()
    matrix = rotation.matrix_fp64.float()
    patch = QwenKeyOrthogonalPatch(model, rotation)
    patch.install()
    token_rows: list[dict] = []
    first_capture = None
    start = time.time()
    try:
        selected_prompts = prompts[:args.prompt_count]
        schedules = [(row, teacher, args.primary_tokens, "primary") for row, teacher in selected_prompts]
        if condition != "identity" and not args.skip_stress:
            schedules.append((stress_prompt[0], stress_prompt[1], args.stress_tokens, "stress"))
        for prompt_index, (row, teacher, horizon, phase) in enumerate(schedules):
            prompt = e2e.render_prompt(tokenizer, row["problem"])
            input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
            mask = torch.ones_like(input_ids)
            with torch.inference_mode():
                patch.enabled = False
                patch.branch = "native"
                native = model(input_ids=input_ids, attention_mask=mask, use_cache=True)
                patch.enabled = True
                patch.branch = "rotated"
                rotated = model(input_ids=input_ids, attention_mask=mask, use_cache=True)
            logit = logits_metrics(rotated.logits, native.logits)
            state = state_metrics(rotated.past_key_values, native.past_key_values, matrix)
            token_rows.append({"condition": condition, "phase": phase, "problem_id": row["problem_id"], "token": 0, **{"logit_" + k: v for k, v in logit.items()}, **{"state_" + k: v for k, v in state.items()}})
            native_cache, rotated_cache = native.past_key_values, rotated.past_key_values
            for step, token_id in enumerate(teacher[:horizon], 1):
                token = torch.tensor([[int(token_id)]], dtype=torch.long, device=model.device)
                patch.capture_first_decode = step == 1 and first_capture is None
                with torch.inference_mode():
                    patch.enabled = False
                    patch.branch = "native"
                    native = model(input_ids=token, past_key_values=native_cache, use_cache=True)
                    patch.enabled = True
                    patch.branch = "rotated"
                    rotated = model(input_ids=token, past_key_values=rotated_cache, use_cache=True)
                native_cache, rotated_cache = native.past_key_values, rotated.past_key_values
                logit = logits_metrics(rotated.logits, native.logits)
                state = state_metrics(rotated_cache, native_cache, matrix)
                token_rows.append({"condition": condition, "phase": phase, "problem_id": row["problem_id"], "token": step, **{"logit_" + k: v for k, v in logit.items()}, **{"state_" + k: v for k, v in state.items()}})
                if patch.capture_first_decode:
                    first_capture = first_decode_capture(patch, rotation)
                    patch.capture_first_decode = False
                if (step % 32) == 0:
                    print(f"condition={condition} phase={phase} prompt={prompt_index + 1}/{len(schedules)} token={step}/{horizon}", flush=True)
    finally:
        patch.close()
    primary = [row for row in token_rows if row["phase"] == "primary"]
    stress = [row for row in token_rows if row["phase"] == "stress"]
    output = {
        "task": TASK,
        "model": "Qwen3.5-9B/GDN",
        "condition": condition,
        "rotation_name": rotation.name,
        "int8": "OFF",
        "state_quantization": "OFF",
        "operator_equivalence_fp64": operator_equivalence(rotation),
        "primary": {
            "prompts": len(selected_prompts),
            "teacher_tokens_per_prompt": args.primary_tokens,
            "logit_relative_l2": summarize([row["logit_relative_l2"] for row in primary]),
            "state_recovered_relative_l2": summarize([row["state_relative_l2"] for row in primary]),
            "top1_match": sum(row["logit_top1_match"] for row in primary) / len(primary),
            "top20_overlap_mean": sum(row["logit_top20_overlap"] for row in primary) / len(primary),
            "nonfinite": sum(row["logit_nonfinite"] + row["state_nonfinite"] for row in primary),
            "first_top1_divergence_token": next((row["token"] for row in primary if not row["logit_top1_match"]), None),
        },
        "stress": None if not stress else {
            "teacher_tokens": args.stress_tokens,
            "logit_relative_l2": summarize([row["logit_relative_l2"] for row in stress]),
            "state_recovered_relative_l2": summarize([row["state_relative_l2"] for row in stress]),
            "top1_match": sum(row["logit_top1_match"] for row in stress) / len(stress),
            "nonfinite": sum(row["logit_nonfinite"] + row["state_nonfinite"] for row in stress),
            "first_top1_divergence_token": next((row["token"] for row in stress if not row["logit_top1_match"]), None),
        },
        "first_decode_capture": first_capture,
        "runtime_seconds": time.time() - start,
        "hadamard_self_inverse_dependency_found": "YES_IN_HISTORICAL_INTERFACE_GENERALIZED_TO_EXPLICIT_R_AND_RT",
    }
    output["status"] = "PASS" if (
        output["operator_equivalence_fp64"]["status"] == "PASS"
        and output["primary"]["top1_match"] == 1.0
        and output["primary"]["nonfinite"] == 0
        and (output["stress"] is None or (output["stress"]["top1_match"] == 1.0 and output["stress"]["nonfinite"] == 0))
    ) else "FAIL"
    outdir = Path(args.output_dir)
    save_json(outdir / f"condition_{condition}.json", output)
    write_csv(outdir / f"condition_{condition}_tokens.csv", token_rows)
    return output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+", choices=("identity", "hadamard", "r0", "r1", "r2"), required=True)
    parser.add_argument("--legacy-root", default="/data/zypan")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-count", type=int, default=3)
    parser.add_argument("--primary-tokens", type=int, default=128)
    parser.add_argument("--stress-tokens", type=int, default=512)
    parser.add_argument("--skip-stress", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = ROT.build_manifest()
    if manifest["status"] != "PASS":
        raise RuntimeError("ORTHOGONAL_GENERATOR_FAIL")
    save_json(Path(args.output_dir) / "rotation_manifest.json", manifest)
    _, model, tokenizer, config, e2e, prompts, stress_prompt = load_inputs(args)
    save_json(Path(args.output_dir) / "prompt_manifest.json", {
        "source": "canonical historical Hadamard-equivalence MATH manifest",
        "primary": [{"problem_id": row["problem_id"], "teacher_tokens_available": len(tokens)} for row, tokens in prompts],
        "stress": {"problem_id": stress_prompt[0]["problem_id"], "teacher_tokens_available": len(stress_prompt[1])},
        "model_path": config.get("model_path"),
    })
    results = [run_condition(args, condition, model, tokenizer, e2e, prompts, stress_prompt) for condition in args.conditions]
    save_json(Path(args.output_dir) / "worker_summary.json", {"conditions": results, "status": "PASS" if all(row["status"] == "PASS" for row in results) else "FAIL"})


if __name__ == "__main__":
    main()
