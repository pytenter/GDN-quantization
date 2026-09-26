from pathlib import Path
import importlib.util
import torch


ROOT = Path(__file__).resolve().parents[3]


def load(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_precision_decomposition_has_distinct_pre_and_post_cast_metrics():
    diag = load("experiments/shared/rotation/numerical_roundtrip.py", "numerical_roundtrip_test")
    rot = load("experiments/shared/rotation/orthogonal_matrix_generator.py", "rotation_generator_test")
    value = torch.randn(3, 128, generator=torch.Generator().manual_seed(7))
    result = diag.precision_roundtrip(value, rot.dense_so_matrix(0), "cpu")
    p2 = result["p2_bf16_storage_fp32_rotation"]
    p2b = result["p2b_bf16_intermediate_storage_fp32_rotation"]
    assert p2["before_final_bf16_cast"]["relative_l2"] >= 0.0
    assert p2["after_final_bf16_cast"]["relative_l2"] >= 0.0
    assert result["p3_native_bf16"]["accumulation_dtype"] == "ACCUMULATION_DTYPE_UNKNOWN"
    assert p2b["intermediate_storage_dtype"] == "torch.bfloat16"


def test_manifest_mismatch_fails_closed(tmp_path):
    diag = load("experiments/shared/rotation/numerical_roundtrip.py", "numerical_roundtrip_manifest_test")
    previous = {"matrices": [{"name": "R_seed0", "sha256_fp64_c_order_bytes": "old"}]}
    current = {"matrices": [{"name": "R_seed0", "sha256_fp64_c_order_bytes": "new"}]}
    path = tmp_path / "manifest.json"
    path.write_text(__import__("json").dumps(previous), encoding="utf-8")
    assert diag.validate_manifest_hashes(path, current)["status"] == "MATRIX_PROVENANCE_MISMATCH"
