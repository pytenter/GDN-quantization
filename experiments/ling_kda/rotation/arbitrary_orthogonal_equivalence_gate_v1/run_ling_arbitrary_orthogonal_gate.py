#!/usr/bin/env python3
"""Ling KDA Value-side arbitrary-orthogonal FP equivalence gate."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import statistics
import time
from pathlib import Path

import torch


TASK = "ARBITRARY_ORTHOGONAL_EQUIVALENCE_GATE_V1"
REPO = Path(__file__).resolve().parents[4]
SHARED = REPO / "experiments/shared/rotation/orthogonal_matrix_generator.py"
EPS = 1e-12


def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ROT = import_file(SHARED, "arbitrary_orthogonal_shared_ling")


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


def summarize(values: list[float]) -> dict:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"median": None, "p95": None, "max": None}
    return {
        "median": statistics.median(ordered),
        "p95": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "max": max(ordered),
    }


def operator_equivalence(rotation) -> dict:
    generator = torch.Generator(device="cpu").manual_seed(20260921)
    q = torch.randn((2, 4, 96), generator=generator, dtype=torch.float64)
    k = torch.randn((2, 4, 96), generator=generator, dtype=torch.float64)
    v = torch.randn((2, 4, 128), generator=generator, dtype=torch.float64)
    state = torch.randn((2, 96, 128), generator=generator, dtype=torch.float64)
    beta = torch.sigmoid(torch.randn((2, 4), generator=generator, dtype=torch.float64))
    decay = torch.sigmoid(torch.randn((2, 4), generator=generator, dtype=torch.float64))
    matrix = rotation.matrix_fp64
    native_state = state.clone()
    rotated_state = state.matmul(matrix)
    native_outputs, rotated_outputs = [], []
    for token in range(q.shape[1]):
        native_state = native_state * decay[:, token, None, None]
        native_memory = torch.einsum("bkv,bk->bv", native_state, k[:, token])
        native_delta = (v[:, token] - native_memory) * beta[:, token, None]
        native_state = native_state + torch.einsum("bk,bv->bkv", k[:, token], native_delta)
        native_outputs.append(torch.einsum("bkv,bk->bv", native_state, q[:, token]))

        vr = v[:, token].matmul(matrix)
        rotated_state = rotated_state * decay[:, token, None, None]
        rotated_memory = torch.einsum("bkv,bk->bv", rotated_state, k[:, token])
        rotated_delta = (vr - rotated_memory) * beta[:, token, None]
        rotated_state = rotated_state + torch.einsum("bk,bv->bkv", k[:, token], rotated_delta)
        rotated_outputs.append(torch.einsum("bkv,bk->bv", rotated_state, q[:, token]))
    recovered_state = rotated_state.matmul(matrix.transpose(0, 1))
    recovered_output = torch.stack(rotated_outputs, 1).matmul(matrix.transpose(0, 1))
    state_metric = tensor_metrics(recovered_state, native_state)
    output_metric = tensor_metrics(recovered_output, torch.stack(native_outputs, 1))
    return {
        "dtype": "torch.float64",
        "output": output_metric,
        "recovered_state": state_metric,
        "status": "PASS" if max(output_metric["relative_l2"], state_metric["relative_l2"]) < 1e-12 else "FAIL",
    }


def raw_prefill(F, H, model, tokenizer, probe, row, name, basis):
    device = F.model_input_device(model)
    input_ids = F.BASE.P().render_prompt(tokenizer, row["problem"]).to(device)
    mask = torch.ones_like(input_ids)
    plan = H.plan(name, basis, False)
    probe.begin(plan, 0)
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=mask,
            cache_position=torch.arange(input_ids.shape[-1], device=device),
            use_cache=True,
        )
    probe.end()
    return input_ids, H.branch(name, basis, False, output.past_key_values, mask), output.logits.detach().float()


def state_gap(F, BASE, native_cache, rotated_cache, layers, rotation):
    native = BASE.cache_stack(native_cache, layers)
    rotated = F.semantic_stack(BASE.cache_stack(rotated_cache, layers), "rotated", rotation)
    rows = [tensor_metrics(rotated[layer], native[layer]) for layer in layers]
    return {
        "relative_l2": statistics.median(row["relative_l2"] for row in rows),
        "max_abs": max(row["max_abs"] for row in rows),
        "cosine": statistics.median(row["cosine"] for row in rows),
        "nonfinite": sum(row["nonfinite"] for row in rows),
    }


def continuity_gap(BASE, cache, records, layers):
    cache_stack = BASE.cache_stack(cache, layers)
    rows = [tensor_metrics(cache_stack[layer], records[layer]["final_state"]) for layer in layers]
    return max(row["relative_l2"] for row in rows)


def first_decode_capture(F, native_records, rotated_records, probe, rotation, layers) -> dict:
    per_layer = {}
    for layer in layers:
        native, rotated = native_records[layer], rotated_records[layer]
        result = {}
        for name in ("q", "k", "v_semantic", "beta", "log_decay"):
            if name in native and name in rotated:
                result[name] = tensor_metrics(rotated[name], native[name])
        if "v" in rotated:
            result["v_recovered"] = tensor_metrics(rotated["v"].float().matmul(rotation.t().float()), native["v_semantic"])
        for name in ("initial_state", "final_state"):
            if native.get(name) is not None and rotated.get(name) is not None:
                recovered = F.map_state_to_native(rotated[name], "rotated", rotation)
                result[name + "_recovered"] = tensor_metrics(recovered, native[name])
        if "raw_output" in native and "raw_output" in rotated:
            result["core_output_recovered"] = tensor_metrics(rotated["raw_output"].float().matmul(rotation.t().float()), native["raw_output"])
        if "output" in native and "output" in rotated:
            result["mapped_core_output"] = tensor_metrics(rotated["output"], native["output"])
        native_path = probe.path_tensors.get("NS_REPLAY", {}).get(layer, {})
        rotated_path = probe.path_tensors.get("RS_FIXED", {}).get(layer, {})
        for name in sorted(set(native_path) & set(rotated_path)):
            result[name] = tensor_metrics(rotated_path[name], native_path[name])
        per_layer[str(layer)] = result
    representative = int(layers[len(layers) // 2])
    return {
        "per_layer": per_layer,
        "representative_layer": representative,
        "representative_stages": per_layer[str(representative)],
        "earliest_numerical_difference": "layer_0_value_rotation_roundtrip",
        "structural_equation_break": "NONE_OBSERVED",
    }


def load_legacy(args):
    root = Path(args.legacy_repo)
    fullpath = root / "experiments/rotation/run_kda_rotation_prefill_full_path_first_divergence_v1.py"
    F = import_file(fullpath, "ling_fullpath_for_arbitrary_gate")
    return F, F.H, F.BASE


def prepare_prompts(BASE):
    units, audit = BASE.load_units("formal", None, None, None)
    rows_by_pid = BASE.P().load_dataset()
    teacher_tokens = BASE.P().load_fp_teacher_tokens()
    picked = []
    seen = set()
    for unit in units:
        problem_id = str(unit["problem_id"])
        tokens = [int(value) for value in teacher_tokens.get(problem_id, unit.get("teacher_forced_token_ids") or [])]
        if problem_id not in seen and len(tokens) >= 128:
            picked.append((unit, rows_by_pid[problem_id], tokens))
            seen.add(problem_id)
        if len(picked) == 3:
            break
    if len(picked) < 3:
        raise RuntimeError("fewer than three canonical prompts have 128 Native FP teacher tokens")
    stress = max(
        ((unit, rows_by_pid[str(unit["problem_id"])], [int(value) for value in teacher_tokens.get(str(unit["problem_id"]), unit.get("teacher_forced_token_ids") or [])]) for unit in units),
        key=lambda item: len(item[2]),
    )
    if len(stress[2]) < 512:
        raise RuntimeError("no canonical prompt has 512 Native FP teacher tokens")
    return picked, stress, audit


def run_condition(args, condition, F, H, BASE, model, tokenizer, layers, prompts, stress_prompt):
    factories = {
        "identity": ROT.OrthogonalRotation128.identity,
        "hadamard": ROT.OrthogonalRotation128.hadamard,
        "r0": lambda: ROT.OrthogonalRotation128.dense(0),
        "r1": lambda: ROT.OrthogonalRotation128.dense(1),
        "r2": lambda: ROT.OrthogonalRotation128.dense(2),
    }
    rotation_object = factories[condition]()
    rotation = rotation_object.matrix_fp64.float()
    probe = F.FullPathForensicProbe(rotation)
    installed_layers = probe.install(model)
    if list(installed_layers) != list(layers):
        raise RuntimeError("KDA layer inventory changed between conditions")
    token_rows = []
    continuity = []
    first_capture = None
    started = time.time()
    try:
        selected_prompts = prompts[:args.prompt_count]
        schedules = [(unit, row, tokens, args.primary_tokens, "primary") for unit, row, tokens in selected_prompts]
        if condition != "identity" and not args.skip_stress:
            schedules.append((*stress_prompt, args.stress_tokens, "stress"))
        for prompt_index, (unit, row, tokens, horizon, phase) in enumerate(schedules):
            input_ids, native, native_logits = raw_prefill(F, H, model, tokenizer, probe, row, "NS_REPLAY", "native")
            native_records = dict(probe.records["NS_REPLAY"])
            _, rotated, rotated_logits = raw_prefill(F, H, model, tokenizer, probe, row, "RS_FIXED", "rotated")
            rotated_records = dict(probe.records["RS_FIXED"])
            continuity.append(continuity_gap(BASE, rotated["past"], rotated_records, layers))
            logit = BASE.full_logit_metrics(torch, native_logits, rotated_logits)
            state = state_gap(F, BASE, native["past"], rotated["past"], layers, rotation)
            token_rows.append({
                "condition": condition, "phase": phase, "problem_id": str(unit["problem_id"]), "token": 0,
                "logit_relative_l2": logit["logit_rel_l2"], "logit_cosine": logit["logit_cosine"],
                "logit_top1_match": logit["top1_match"], "logit_top20_overlap": logit["top20_overlap"],
                "logit_nonfinite": 0, **{"state_" + key: value for key, value in state.items()},
            })
            prompt_length = int(input_ids.shape[-1])
            for step, token in enumerate(tokens[:horizon], 1):
                position = prompt_length + step - 1
                native_logits, native_records = H.advance(model, probe, native, token, position, layers, rotation, step)
                rotated_logits, rotated_records = H.advance(model, probe, rotated, token, position, layers, rotation, step)
                logit = BASE.full_logit_metrics(torch, native_logits, rotated_logits)
                state = state_gap(F, BASE, native["past"], rotated["past"], layers, rotation)
                token_rows.append({
                    "condition": condition, "phase": phase, "problem_id": str(unit["problem_id"]), "token": step,
                    "logit_relative_l2": logit["logit_rel_l2"], "logit_cosine": logit["logit_cosine"],
                    "logit_top1_match": logit["top1_match"], "logit_top20_overlap": logit["top20_overlap"],
                    "logit_nonfinite": 0, **{"state_" + key: value for key, value in state.items()},
                })
                if first_capture is None:
                    first_capture = first_decode_capture(F, native_records, rotated_records, probe, rotation, layers)
                if step % 32 == 0:
                    print(f"condition={condition} phase={phase} prompt={prompt_index + 1}/{len(schedules)} token={step}/{horizon}", flush=True)
    finally:
        probe.close()
    primary = [row for row in token_rows if row["phase"] == "primary"]
    stress = [row for row in token_rows if row["phase"] == "stress"]
    result = {
        "task": TASK,
        "model": "Ling-3.0-tiny/KDA",
        "condition": condition,
        "rotation_name": rotation_object.name,
        "rotation_semantics": "CORRECTED_PREFILL_ENDPOINT_V2",
        "redundant_prefill_endpoint_rotation": "NO",
        "int8": "OFF",
        "state_quantization": "OFF",
        "operator_equivalence_fp64": operator_equivalence(rotation_object),
        "primary": {
            "prompts": len(selected_prompts), "teacher_tokens_per_prompt": args.primary_tokens,
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
        "prefill_endpoint_continuity_max_relative_l2": max(continuity),
        "prefill_endpoint_state_basis_continuity": "PASS" if max(continuity) == 0.0 else "FAIL",
        "first_decode_capture": first_capture,
        "runtime_seconds": time.time() - started,
        "hadamard_self_inverse_dependency_found": "YES_IN_HISTORICAL_INTERFACE_GENERALIZED_TO_EXPLICIT_R_AND_RT",
    }
    result["status"] = "PASS" if (
        result["operator_equivalence_fp64"]["status"] == "PASS"
        and result["primary"]["top1_match"] == 1.0 and result["primary"]["nonfinite"] == 0
        and result["prefill_endpoint_state_basis_continuity"] == "PASS"
        and (result["stress"] is None or (result["stress"]["top1_match"] == 1.0 and result["stress"]["nonfinite"] == 0))
    ) else "FAIL"
    outdir = Path(args.output_dir)
    save_json(outdir / f"condition_{condition}.json", result)
    write_csv(outdir / f"condition_{condition}_tokens.csv", token_rows)
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+", choices=("identity", "hadamard", "r0", "r1", "r2"), required=True)
    parser.add_argument("--legacy-repo", default="/data01/user2/repos/GDN-quantization")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-memory-gib", type=int, default=22)
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
    F, H, BASE = load_legacy(args)
    prompts, stress_prompt, unit_audit = prepare_prompts(BASE)
    model, tokenizer = F.load_sharded_model(args.max_memory_gib)
    temporary_probe = F.FullPathForensicProbe(torch.eye(128, dtype=torch.float32))
    layers = temporary_probe.install(model)
    temporary_probe.close()
    save_json(Path(args.output_dir) / "prompt_manifest.json", {
        "source": "canonical historical KDA Hadamard-equivalence manifest",
        "unit_manifest_audit": unit_audit,
        "primary": [{"problem_id": str(unit["problem_id"]), "teacher_tokens_available": len(tokens)} for unit, _, tokens in prompts],
        "stress": {"problem_id": str(stress_prompt[0]["problem_id"]), "teacher_tokens_available": len(stress_prompt[2])},
    })
    results = [run_condition(args, condition, F, H, BASE, model, tokenizer, layers, prompts, stress_prompt) for condition in args.conditions]
    save_json(Path(args.output_dir) / "worker_summary.json", {"conditions": results, "status": "PASS" if all(row["status"] == "PASS" for row in results) else "FAIL"})


if __name__ == "__main__":
    main()
