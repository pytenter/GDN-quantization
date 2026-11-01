#!/usr/bin/env python3
"""Static regression guard for the additive C5/C6 canonical-runner integration."""

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path("/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/qwen_gdn/QWEN_C5_C6_DEPLOYMENT_CLOSURE_V1")
ORIGINAL = Path("/data/zypan/consolidation/GDN-quantization-qwen3090-20260921/experiments/qwen_gdn/aime26/81920/run_qwen_aime26_formal.py")
INTEGRATED = ROOT / "scripts/run_qwen_c5_c6_formal.py"
LOCAL_LOADER = Path("/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/QWEN_GDN_48GB_FORMAL_TRAINING_READINESS_CLOSURE_V2/scripts/runtime.py")
EXPECTED_ORIGINAL = "9ff56d00b6b2dbaf9be3c3186bd30be8d3b9cc9d9b580a680316bcc8c01247dd"
EXPECTED_LOCAL_LOADER = "fda4484ab2a9de6541bf2cd9401ff534ea00ad0a3fac0616f25b00d6a7edad75"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nodes(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}


def ast_hash(node):
    return hashlib.sha256(ast.dump(node, annotate_fields=True, include_attributes=False).encode()).hexdigest()


def main():
    original_nodes = nodes(ORIGINAL)
    integrated_nodes = nodes(INTEGRATED)
    unchanged = [
        "setup_seed", "hadamard", "normalize_qk", "QwenKeyHadamardPatch", "get_state",
        "quantize_c128", "load_model_and_tokenizer", "render_input", "build_warpers",
        "sample_next", "build_top_k_audit", "top_k_audit_valid", "stop_ids",
    ]
    comparisons = {}
    for name in unchanged:
        expected = ast_hash(original_nodes[name])
        actual = ast_hash(integrated_nodes[name])
        comparisons[name] = {"original_ast_sha256": expected, "integrated_ast_sha256": actual, "exact": expected == actual}

    text = INTEGRATED.read_text(encoding="utf-8")
    patch = integrated_nodes["QwenFrozenRotationPatch"]
    loader = integrated_nodes["load_frozen_rotation"]
    patch_text = ast.get_source_segment(text, patch) or ""
    loader_text = ast.get_source_segment(text, loader) or ""
    loader_gate = all([
        "--rotation-file" in text,
        "torch.load(rotation_file, map_location=\"cpu\"" in loader_text,
        "FrozenRotationBank" in loader_text,
        "tensor_sha256" in loader_text,
        "PerLayerCayleyRotations" not in loader_text,
        ".matmul(hadamard(" in patch_text,
        ".matmul(delta)" in patch_text,
        "theta_reconstruction = False" in patch_text,
        "QwenFrozenRotationPatch(model, rotation_bank" in text,
    ])
    original_hash = sha256(ORIGINAL)
    local_loader_hash = sha256(LOCAL_LOADER)
    report = {
        "canonical_original_runner": str(ORIGINAL),
        "canonical_original_runner_expected_sha256": EXPECTED_ORIGINAL,
        "canonical_original_runner_actual_sha256": original_hash,
        "canonical_original_runner_identity_gate": "PASS" if original_hash == EXPECTED_ORIGINAL else "FAIL",
        "integrated_runner": str(INTEGRATED),
        "integrated_runner_sha256": sha256(INTEGRATED),
        "verified_local_frozen_loader_source": str(LOCAL_LOADER),
        "verified_local_frozen_loader_expected_sha256": EXPECTED_LOCAL_LOADER,
        "verified_local_frozen_loader_actual_sha256": local_loader_hash,
        "verified_local_frozen_loader_gate": "PASS" if local_loader_hash == EXPECTED_LOCAL_LOADER else "FAIL",
        "unchanged_canonical_components": comparisons,
        "unchanged_components_all_exact": all(row["exact"] for row in comparisons.values()),
        "direct_load_static_checks": {
            "rotation_file_cli": "--rotation-file" in text,
            "cpu_torch_load": "torch.load(rotation_file, map_location=\"cpu\"" in loader_text,
            "per_layer_sha_verification": "tensor_sha256" in loader_text,
            "no_cayley_reconstruction": "PerLayerCayleyRotations" not in loader_text,
            "H_then_Delta_order": ".matmul(hadamard(" in patch_text and ".matmul(delta)" in patch_text,
            "theta_reconstruction_false": "theta_reconstruction = False" in patch_text,
            "integrated_patch_selected": "QwenFrozenRotationPatch(model, rotation_bank" in text,
        },
        "FROZEN_LOADER_INTEGRATION_STATIC_GATE": "PASS" if loader_gate else "FAIL",
        "HADAMARD_REGRESSION_STATIC_GATE": "PASS" if all(row["exact"] for row in comparisons.values()) else "FAIL",
    }
    path = ROOT / "analysis/loader_integration_static_audit_v2.json"
    with path.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps({k: report[k] for k in ("canonical_original_runner_identity_gate", "verified_local_frozen_loader_gate", "FROZEN_LOADER_INTEGRATION_STATIC_GATE", "HADAMARD_REGRESSION_STATIC_GATE")}, indent=2))
    if any(report[k] != "PASS" for k in ("canonical_original_runner_identity_gate", "verified_local_frozen_loader_gate", "FROZEN_LOADER_INTEGRATION_STATIC_GATE", "HADAMARD_REGRESSION_STATIC_GATE")):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
