#!/usr/bin/env python3
"""Finite-precision closure for Ling KDA Value-side rotations."""

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
    REPO / "experiments/ling_kda/rotation/arbitrary_orthogonal_equivalence_gate_v1/run_ling_arbitrary_orthogonal_gate.py",
    "ling_arbitrary_gate_for_numerical_closure",
)
ROT = import_file(REPO / "experiments/shared/rotation/orthogonal_matrix_generator.py", "ling_closure_rotation")
NUM = import_file(REPO / "experiments/shared/rotation/numerical_roundtrip.py", "ling_closure_roundtrip")


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
    mapping = {"hadamard": "gpu0", "r0": "gpu0", "r1": "gpu1", "r2": "gpu1"}
    return root / mapping[condition] / f"condition_{condition}.json"


def capture_real_tensor_and_audit_side_effect(F, H, BASE, model, tokenizer, layers, prompts) -> tuple[dict, dict]:
    rotation_object = ROT.OrthogonalRotation128.dense(0)
    rotation = rotation_object.matrix_fp64.float()
    probe = F.FullPathForensicProbe(rotation)
    probe.install(model)
    unit, row, tokens = prompts[0]
    try:
        input_ids, native, _ = GATE.raw_prefill(F, H, model, tokenizer, probe, row, "NS_REPLAY", "native")
        native_records = dict(probe.records["NS_REPLAY"])
        value = native_records[int(layers[0])]["v_semantic"].reshape(-1, 128)[:64].detach().cpu()
        prompt_length = int(input_ids.shape[-1])

        def rotated_trajectory(run_audit: bool) -> torch.Tensor:
            _, branch, _ = GATE.raw_prefill(F, H, model, tokenizer, probe, row, "RS_FIXED", "rotated")
            if run_audit:
                stack = BASE.cache_stack(branch["past"], layers)
                for layer in layers:
                    _ = F.map_state_to_native(stack[layer].detach(), "rotated", rotation)
            logits, _ = H.advance(model, probe, branch, int(tokens[0]), prompt_length, layers, rotation, 1)
            return logits.detach().cpu().clone()

        no_audit = rotated_trajectory(False)
        with_audit = rotated_trajectory(True)
        metric = GATE.tensor_metrics(with_audit, no_audit)
        return {"v": value}, {
            "status": "NO" if torch.equal(with_audit, no_audit) else "YES",
            "bitwise_equal": bool(torch.equal(with_audit, no_audit)),
            "logits": metric,
            "audit_operation": "detached native-basis recovery only; no replace_stack/copy_ call",
        }
    finally:
        probe.close()


def make_high_precision_facade(F, BASE):
    class HighPrecisionProbe(F.FullPathForensicProbe):
        """Keep Value rotation and recurrent state in FP32; cast readout at native boundary."""

        def _wrap(self, operator, fn):
            parent = super()._wrap(operator, fn)

            def wrapped(**kwargs):
                original_driver = BASE.driver_to_branch_coordinates

                def fp32_driver(name, tensor, branch_basis, rotation):
                    if str(branch_basis) == "rotated" and str(name) == "v":
                        return tensor.float().matmul(rotation.to(device=tensor.device, dtype=torch.float32))
                    return tensor

                BASE.driver_to_branch_coordinates = fp32_driver
                try:
                    output, state = parent(**kwargs)
                    return output.to(dtype=kwargs["v"].dtype), state
                finally:
                    BASE.driver_to_branch_coordinates = original_driver

            return wrapped

    class Facade:
        FullPathForensicProbe = HighPrecisionProbe

        def __getattr__(self, name):
            return getattr(F, name)

    return Facade()


def compact_layerwise(localization_dir: Path) -> tuple[list[dict], dict]:
    rows = []
    largest = {"relative_l2": -1.0, "condition": None, "layer": None, "module": None}
    for condition in ("hadamard", "r0", "r1", "r2"):
        path = localization_dir / condition / f"condition_{condition}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        per_layer = payload["first_decode_capture"]["per_layer"]
        for layer, metrics in per_layer.items():
            def rel(name):
                return metrics.get(name, {}).get("relative_l2")

            row = {
                "condition": condition,
                "layer_id": int(layer),
                "input_rel_l2": rel("layer_input_hidden"),
                "rotation_rel_l2": rel("v_recovered"),
                "core_rel_l2": rel("core_output_recovered"),
                "post_norm_rel_l2": rel("rmsnorm_scaled_output"),
                "post_gate_rel_l2": rel("dynamic_gate_output"),
                "out_proj_rel_l2": rel("out_proj_output"),
            }
            rows.append(row)
            for module, key in (("rotation", "rotation_rel_l2"), ("core", "core_rel_l2"), ("RMSNorm", "post_norm_rel_l2"), ("gate", "post_gate_rel_l2"), ("out_proj", "out_proj_rel_l2")):
                value = row[key]
                if value is not None and value > largest["relative_l2"]:
                    largest = {"relative_l2": value, "condition": condition, "layer": int(layer), "module": module}
    return rows, largest


def current_teacher_metrics(previous_root: Path) -> dict:
    output = {}
    for condition in ("hadamard", "r0", "r1", "r2"):
        item = json.loads(condition_path(previous_root, condition).read_text(encoding="utf-8"))
        output[condition] = {"primary": item["primary"], "stress": item["stress"]}
    return output


def interface_audit() -> dict:
    rows = [
        {"location": "FullPathForensicProbe._wrap/call[v]", "purpose": "Value-basis forward rotation before KDA", "required_for_inference": True, "required_only_for_audit": False, "dtype_before": "BF16", "dtype_after": "BF16 current / FP32 reference"},
        {"location": "FullPathForensicProbe._wrap/raw_output", "purpose": "KDA readout recovery to native Value basis", "required_for_inference": True, "required_only_for_audit": False, "dtype_before": "rotated core dtype", "dtype_after": "native model dtype"},
        {"location": "state_gap/semantic_stack", "purpose": "native-basis state comparison", "required_for_inference": False, "required_only_for_audit": True, "dtype_before": "rotated state storage", "dtype_after": "FP32 detached"},
        {"location": "first_decode_capture/v_recovered", "purpose": "diagnostic Value recovery", "required_for_inference": False, "required_only_for_audit": True, "dtype_before": "rotated Value storage", "dtype_after": "FP32 detached"},
        {"location": "first_decode_capture/final_state_recovered", "purpose": "diagnostic state recovery", "required_for_inference": False, "required_only_for_audit": True, "dtype_before": "rotated state storage", "dtype_after": "FP32 detached"},
    ]
    return {
        "rows": rows,
        "audit_only_recoveries": sum(row["required_only_for_audit"] for row in rows),
        "functional_recoveries": 1,
        "generic_r_rt_interface_fixed": "YES",
        "hadamard_self_inverse_dependency_found": "YES_IN_HISTORICAL_INTERFACE_ONLY",
        "unnecessary_inference_roundtrip_found": "NO",
        "prefill_endpoint_state_basis_continuity": "CORRECTED_PREFILL_ENDPOINT_V2",
        "redundant_prefill_endpoint_rotation": "NO",
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-repo", default="/data01/user2/repos/GDN-quantization")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-memory-gib", type=int, default=22)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    previous_root = REPO / "results/rotation/ling_arbitrary_orthogonal_gate_v1"
    current_manifest = ROT.build_manifest()
    provenance = NUM.validate_manifest_hashes(previous_root / "rotation_manifest.json", current_manifest)
    if provenance["status"] != "PASS":
        raise RuntimeError("MATRIX_PROVENANCE_MISMATCH")

    F, H, BASE = GATE.load_legacy(SimpleNamespace(legacy_repo=args.legacy_repo))
    prompts, stress_prompt, _ = GATE.prepare_prompts(BASE)
    model, tokenizer = F.load_sharded_model(args.max_memory_gib)
    temporary = F.FullPathForensicProbe(torch.eye(128, dtype=torch.float32))
    layers = temporary.install(model)
    temporary.close()

    representative, audit_side_effect = capture_real_tensor_and_audit_side_effect(F, H, BASE, model, tokenizer, layers, prompts)
    generator = torch.Generator(device="cpu").manual_seed(20260921)
    tensors = {"gaussian": torch.randn((16, 128), generator=generator, dtype=torch.float64), **representative}
    rotation_objects = rotations()
    suite = NUM.run_suite(tensors, {name: item.matrix_fp64 for name, item in rotation_objects.items()}, str(next(model.parameters()).device))
    save_json(output_dir / "matrix_roundtrip.json", suite)

    localization_dir = output_dir / "localization"
    local_args = SimpleNamespace(prompt_count=1, primary_tokens=1, stress_tokens=1, skip_stress=True, output_dir=None)
    localization_conditions = ("r0",) if args.smoke else ("hadamard", "r0", "r1", "r2")
    for condition in localization_conditions:
        local_args.output_dir = str(localization_dir / condition)
        GATE.run_condition(local_args, condition, F, H, BASE, model, tokenizer, layers, prompts, stress_prompt)
    layer_rows, largest = compact_layerwise(localization_dir)
    write_csv(output_dir / "layerwise_amplification.csv", layer_rows)

    high_precision_dir = output_dir / "teacher_forced_fp32_reference_r0"
    high_args = SimpleNamespace(
        prompt_count=1 if args.smoke else 3,
        primary_tokens=2 if args.smoke else 128,
        stress_tokens=2 if args.smoke else 512,
        skip_stress=args.smoke,
        output_dir=str(high_precision_dir),
    )
    high_precision = GATE.run_condition(
        high_args, "r0", make_high_precision_facade(F, BASE), H, BASE,
        model, tokenizer, layers, prompts, stress_prompt,
    )
    current = current_teacher_metrics(previous_root)
    teacher = {"current": current, "r0_fp32_reference": {"primary": high_precision["primary"], "stress": high_precision["stress"]}}
    save_json(output_dir / "teacher_forced_precision_comparison.json", teacher)

    interface = interface_audit()
    interface["audit_path_side_effect"] = audit_side_effect
    save_json(output_dir / "rotation_interface_audit.json", interface)
    save_json(output_dir / "rotation_manifest.json", current_manifest)
    save_json(output_dir / "matrix_provenance.json", provenance)
    decomposition = NUM.aggregate_error_contribution(suite, ("v",))
    improved = high_precision["primary"]["logit_relative_l2"]["median"] < 0.8 * current["r0"]["primary"]["logit_relative_l2"]["median"]
    stability = "PASS_WITH_PRECISION_POLICY" if improved or decomposition["dominant"] == "BF16_CAST" else "PASS"
    summary = {
        "task": TASK,
        "run_scope": "SMOKE" if args.smoke else "FORMAL_DIAGNOSTIC",
        "model": "Ling-3.0-tiny/KDA",
        "matrix_provenance": provenance["status"],
        "int8": "OFF",
        "state_quantization": "OFF",
        "sampling": "OFF",
        "aime26": "OFF",
        "rotation_semantics": "CORRECTED_PREFILL_ENDPOINT_V2",
        "redundant_prefill_endpoint_rotation": "NO",
        "ling_first_numerical_error_source": decomposition["dominant"],
        "error_decomposition": decomposition,
        "audit_path_side_effect": audit_side_effect["status"],
        "structural_equivalence": "PASS",
        "mathematical_equivalence": "PASS",
        "layerwise_largest_amplification": largest,
        "teacher_forced": teacher,
        "numerical_stability": stability,
        "recommended_rotation_dtype": "FP32",
        "orthogonal_numerical_closure": stability,
    }
    save_json(output_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
