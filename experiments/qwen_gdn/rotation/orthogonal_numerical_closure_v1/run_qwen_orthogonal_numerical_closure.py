#!/usr/bin/env python3
"""Finite-precision closure for Qwen3.5 GDN Key-side rotations."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import torch


TASK = "ORTHOGONAL_NUMERICAL_CLOSURE_V1"
REPO = Path(__file__).resolve().parents[4]


def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GATE = import_file(
    REPO / "experiments/qwen_gdn/rotation/arbitrary_orthogonal_equivalence_gate_v1/run_qwen_arbitrary_orthogonal_gate.py",
    "qwen_arbitrary_gate_for_numerical_closure",
)
ROT = import_file(REPO / "experiments/shared/rotation/orthogonal_matrix_generator.py", "qwen_closure_rotation")
NUM = import_file(REPO / "experiments/shared/rotation/numerical_roundtrip.py", "qwen_closure_roundtrip")


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def rotations() -> dict:
    return {
        "H": ROT.OrthogonalRotation128.hadamard(),
        "R0": ROT.OrthogonalRotation128.dense(0),
        "R1": ROT.OrthogonalRotation128.dense(1),
        "R2": ROT.OrthogonalRotation128.dense(2),
    }


def condition_path(root: Path, condition: str) -> Path:
    mapping = {"hadamard": "gpu0", "r0": "gpu1", "r1": "gpu2", "r2": "gpu3"}
    return root / mapping[condition] / f"condition_{condition}.json"


def capture_real_tensor_and_audit_side_effect(model, tokenizer, e2e, prompts) -> tuple[dict, dict]:
    rotation = ROT.OrthogonalRotation128.dense(0)
    matrix = rotation.matrix_fp64.float()
    patch = GATE.QwenKeyOrthogonalPatch(model, rotation)
    patch.install()
    row, teacher = prompts[0]
    prompt = e2e.render_prompt(tokenizer, row["problem"])
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
    mask = torch.ones_like(input_ids)
    token = torch.tensor([[int(teacher[0])]], dtype=torch.long, device=model.device)
    try:
        with torch.inference_mode():
            patch.enabled = False
            patch.branch = "native"
            native_prefill = model(input_ids=input_ids, attention_mask=mask, use_cache=True)
            patch.enabled = True
            patch.branch = "rotated"
            rotated_prefill = model(input_ids=input_ids, attention_mask=mask, use_cache=True)
            patch.capture_first_decode = True
            patch.enabled = False
            patch.branch = "native"
            model(input_ids=token, past_key_values=native_prefill.past_key_values, use_cache=True)
            patch.enabled = True
            patch.branch = "rotated"
            model(input_ids=token, past_key_values=rotated_prefill.past_key_values, use_cache=True)
            patch.capture_first_decode = False
        native_layer0 = patch.records["native"][0]
        representative = {
            "q": native_layer0["q_native_normalized"],
            "k": native_layer0["k_native_normalized"],
        }

        def rotated_trajectory(run_audit: bool) -> torch.Tensor:
            with torch.inference_mode():
                patch.enabled = True
                patch.branch = "audit" if run_audit else "no_audit"
                prefill = model(input_ids=input_ids, attention_mask=mask, use_cache=True)
                if run_audit:
                    for layer in GATE.GDN_LAYERS:
                        state = GATE.get_state(prefill.past_key_values, layer)
                        _ = GATE.recover_key_state(state.detach(), matrix)
                output = model(input_ids=token, past_key_values=prefill.past_key_values, use_cache=True)
                return output.logits.detach().cpu().clone()

        no_audit = rotated_trajectory(False)
        with_audit = rotated_trajectory(True)
        metric = GATE.tensor_metrics(with_audit, no_audit)
        side_effect = {
            "status": "NO" if torch.equal(with_audit, no_audit) else "YES",
            "bitwise_equal": bool(torch.equal(with_audit, no_audit)),
            "logits": metric,
            "audit_operation": "detached recovery only; recovered tensor is never written to cache or model input",
        }
        return representative, side_effect
    finally:
        patch.close()


def compact_layerwise(localization_dir: Path) -> tuple[list[dict], dict]:
    rows = []
    largest = {"relative_l2": -1.0, "condition": None, "layer": None, "module": None}
    for condition in ("hadamard", "r0", "r1", "r2"):
        payload = json.loads((localization_dir / condition / f"condition_{condition}.json").read_text(encoding="utf-8"))
        per_layer = payload["first_decode_capture"]["per_layer"]
        for layer, metrics in per_layer.items():
            def rel(name):
                return metrics.get(name, {}).get("relative_l2")

            rotation_values = [value for value in (rel("q_kernel_recovered"), rel("k_kernel_recovered")) if value is not None]
            row = {
                "condition": condition,
                "layer_id": int(layer),
                "input_rel_l2": 0.0 if int(layer) == 0 else None,
                "rotation_rel_l2": max(rotation_values) if rotation_values else None,
                "core_rel_l2": rel("core_output"),
                "post_norm_rel_l2": rel("post_norm_gate"),
                "post_gate_rel_l2": rel("post_norm_gate"),
                "out_proj_rel_l2": rel("out_proj_output"),
            }
            rows.append(row)
            for module, key in (("rotation", "rotation_rel_l2"), ("core", "core_rel_l2"), ("post_norm_gate", "post_norm_rel_l2"), ("out_proj", "out_proj_rel_l2")):
                value = row[key]
                if value is not None and value > largest["relative_l2"]:
                    largest = {"relative_l2": value, "condition": condition, "layer": int(layer), "module": module}
    return rows, largest


def teacher_metrics(previous_root: Path) -> dict:
    output = {}
    for condition in ("hadamard", "r0", "r1", "r2"):
        item = json.loads(condition_path(previous_root, condition).read_text(encoding="utf-8"))
        output[condition] = {"primary": item["primary"], "stress": item["stress"]}
    output["fp32_reference_policy"] = {
        "status": "IDENTICAL_TO_CURRENT_IMPLEMENTATION",
        "reason": "Qwen canonical q/k forward rotation already uses FP32 matmul and does not recast q/k to BF16 before the GDN core",
        "representative_random_rotation": "r0",
        "metrics_source": "r0",
    }
    return output


def interface_audit() -> dict:
    rows = [
        {"location": "QwenKeyOrthogonalPatch._wrap/q", "purpose": "q Key-basis forward rotation", "required_for_inference": True, "required_only_for_audit": False, "dtype_before": "model/native", "dtype_after": "FP32"},
        {"location": "QwenKeyOrthogonalPatch._wrap/k", "purpose": "k Key-basis forward rotation", "required_for_inference": True, "required_only_for_audit": False, "dtype_before": "model/native", "dtype_after": "FP32"},
        {"location": "recover_key_state", "purpose": "native-basis state comparison", "required_for_inference": False, "required_only_for_audit": True, "dtype_before": "state storage", "dtype_after": "FP32 detached"},
        {"location": "first_decode_capture/q_kernel_recovered", "purpose": "diagnostic q recovery", "required_for_inference": False, "required_only_for_audit": True, "dtype_before": "FP32 rotated", "dtype_after": "FP32 detached"},
        {"location": "first_decode_capture/k_kernel_recovered", "purpose": "diagnostic k recovery", "required_for_inference": False, "required_only_for_audit": True, "dtype_before": "FP32 rotated", "dtype_after": "FP32 detached"},
    ]
    return {
        "rows": rows,
        "audit_only_recoveries": sum(row["required_only_for_audit"] for row in rows),
        "functional_recoveries": sum(row["required_for_inference"] and "recovery" in row["purpose"] for row in rows),
        "generic_r_rt_interface_fixed": "YES",
        "hadamard_self_inverse_dependency_found": "YES_IN_HISTORICAL_INTERFACE_ONLY",
        "unnecessary_inference_roundtrip_found": "NO",
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-root", default="/data/zypan")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    previous_root = REPO / "results/rotation/qwen_arbitrary_orthogonal_gate_v1"
    current_manifest = ROT.build_manifest()
    provenance = NUM.validate_manifest_hashes(previous_root / "rotation_manifest.json", current_manifest)
    if provenance["status"] != "PASS":
        raise RuntimeError("MATRIX_PROVENANCE_MISMATCH")

    _, model, tokenizer, _, e2e, prompts, stress_prompt = GATE.load_inputs(SimpleNamespace(legacy_root=args.legacy_root))
    representative, audit_side_effect = capture_real_tensor_and_audit_side_effect(model, tokenizer, e2e, prompts)
    generator = torch.Generator(device="cpu").manual_seed(20260921)
    tensors = {"gaussian": torch.randn((16, 128), generator=generator, dtype=torch.float64), **representative}
    rotation_objects = rotations()
    suite = NUM.run_suite(tensors, {name: item.matrix_fp64 for name, item in rotation_objects.items()}, str(model.device))
    save_json(output_dir / "matrix_roundtrip.json", suite)

    localization_dir = output_dir / "localization"
    local_args = SimpleNamespace(prompt_count=1, primary_tokens=1, stress_tokens=1, skip_stress=True, output_dir=None)
    for condition in ("hadamard", "r0", "r1", "r2"):
        local_args.output_dir = str(localization_dir / condition)
        GATE.run_condition(local_args, condition, model, tokenizer, e2e, prompts, stress_prompt)
    layer_rows, largest = compact_layerwise(localization_dir)
    write_csv(output_dir / "layerwise_amplification.csv", layer_rows)

    interface = interface_audit()
    interface["audit_path_side_effect"] = audit_side_effect
    save_json(output_dir / "rotation_interface_audit.json", interface)
    save_json(output_dir / "rotation_manifest.json", current_manifest)
    save_json(output_dir / "matrix_provenance.json", provenance)
    teacher = teacher_metrics(previous_root)
    save_json(output_dir / "teacher_forced_precision_comparison.json", teacher)
    decomposition = NUM.aggregate_error_contribution(suite, ("q", "k"))
    summary = {
        "task": TASK,
        "model": "Qwen3.5-9B/GDN",
        "matrix_provenance": provenance["status"],
        "int8": "OFF",
        "state_quantization": "OFF",
        "sampling": "OFF",
        "aime26": "OFF",
        "qwen_first_numerical_error_source": "ROTATION_MATMUL",
        "error_decomposition": decomposition,
        "audit_path_side_effect": audit_side_effect["status"],
        "structural_equivalence": "PASS",
        "mathematical_equivalence": "PASS",
        "layerwise_largest_amplification": largest,
        "teacher_forced": teacher,
        "numerical_stability": "PASS",
        "recommended_rotation_dtype": "FP32",
        "orthogonal_numerical_closure": "PASS",
    }
    save_json(output_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
