"""Append-only CPU source audit; AST extraction avoids GPU-driver imports.

v1 is retained unchanged: importing FLA ops in CUDA-hidden mode required a
Triton active driver. No model/GPU work occurred. This v3 extracts only the
original pure KL and QDQ functions; it cannot qualify the FLA GPU kernel.
"""
import ast
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import torch
from objective import digest
from run_matched_l7 import ROOT, publish
from phase_b_calibration_common_v2 import verify, audit_b, checked_json
from select_phase_b_pairs import validate_completed
from persistent_engine_v1 import full_kl


def extract(path, name):
    node = next(x for x in ast.parse(path.read_text()).body
                if isinstance(x, ast.FunctionDef) and x.name == name)
    namespace = dict(torch=torch, EPS=1e-12)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    cfg = verify()
    selection = checked_json(ROOT / "PROSPECTIVE_DOCUMENT_AND_PAIR_SELECTION.json")
    assert selection["status"] == "BOTH_DOCUMENTS_DISCRIMINATIVE_PAIRS_FROZEN"
    _, identities = validate_completed(cfg)
    assert all(identities[x["document"]] == x["calibration_identity"] for x in selection["documents"])
    ledger = audit_b()
    assert all(x["status"] == "COMPLETE_MECHANICAL_PASS" for x in ledger["phase_b_calibration_runs"])
    supplement = json.loads((ROOT / "PREMEASUREMENT_ASSET_SUPPLEMENT.json").read_text())
    assert digest(ROOT / "SOURCE_IDENTITIES.json") == supplement["files"]["SOURCE_IDENTITIES.json"]
    sources = json.loads((ROOT / "SOURCE_IDENTITIES.json").read_text())["source_hashes"]
    direction_path = next(Path(x) for x in sources if x.endswith("/run_direction_causal.py"))
    capture_path = next(Path(x) for x in sources if x.endswith("/capture.py"))
    transition_path = capture_path.parent / "transition.py"
    inherited = Path("/data/zypan/worktrees/ling-long-horizon-propagation-confirm-v1/inherited/data01/user2/worktrees/ling-l6-l7-objective-horizon-diagnosis-v1/experiments")
    source_paths = {direction_path, transition_path, capture_path,
        inherited / "LING_L6_L7_OBJECTIVE_HORIZON_DIAGNOSIS_V1/scripts/phase2_final_hidden_512_smoke.py",
        inherited / "LING_L7R_L8_LAYER_AWARE_MATCHED_TRAINING_V4_DETERMINISTIC/scripts/v4_common.py",
        Path("/data/zypan/models/Ling-3.0-tiny/modeling_bailing_moe_v3.py"),
        Path("/data/zypan/models/Ling-3.0-tiny/configuration_bailing_moe_v3.py")}
    import transformers.cache_utils
    source_paths.add(Path(inspect.getfile(transformers.cache_utils)))
    fla_root = Path(importlib.util.find_spec("fla").origin).parent
    source_paths.update({fla_root / "ops/kda/__init__.py", fla_root / "ops/kda/fused_recurrent.py"})
    model_source = Path("/data/zypan/models/Ling-3.0-tiny/modeling_bailing_moe_v3.py").read_text()
    for expression in ("past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)",
        "past_key_value.layers[self.layer_idx].keys = recurrent_state",
        "past_key_value.layers[self.layer_idx].values = (conv_state_q, conv_state_k, conv_state_v)",
        "past_key_values = DynamicCache()"):
        assert expression in model_source, "HYBRID_SOURCE_LAYOUT_NOT_IDENTIFIED"
    original_kl = extract(direction_path, "kl_metrics")
    original_qdq = extract(transition_path, "qdq_r128")
    torch.manual_seed(0)
    count = 0
    for dtype in (torch.float32, torch.bfloat16):
        for magnitude in (1., 30., 100.):
            a, b = (torch.randn(1, 1, 509).to(dtype) * magnitude for _ in range(2))
            assert full_kl(torch, a, b) == original_kl(torch, a, b)["KL"]
            count += 1
    qdq_checks = []
    for matrix in (torch.eye(128), torch.linalg.qr(torch.randn(128, 128))[0]):
        state = torch.randn((1, 2, 4, 128))
        rotated = state.matmul(matrix)
        dequant, codes, scale = original_qdq(rotated)
        expected_scale = rotated.float().abs().amax(-1, keepdim=True).clamp_min(1e-12) / 127.
        expected_code = torch.round(rotated.float() / expected_scale).clamp(-127, 127).to(torch.int8)
        assert codes.dtype == torch.int8 and torch.equal(codes, expected_code)
        assert torch.equal(scale, expected_scale)
        assert torch.equal(dequant, expected_code.float() * expected_scale)
        assert torch.isfinite(dequant.matmul(matrix.t())).all()
        qdq_checks.append(dict(original_source_ast_qdq_codes_scales_dequant_bitwise_equal=True,
                               group_width=128, inverse_finite=True))
    assert not torch.cuda.is_initialized()
    failure = ROOT / "PERSISTENT_SOURCE_CPU_ENGINEERING_FAILURE_V1.json"
    publish(failure, dict(status="PRE_GPU_CPU_IMPORT_ENGINEERING_FAILURE_RETAINED",
        code_sha256=digest(ROOT / "audit_persistent_source_cpu_v1.py"),
        error="RuntimeError: 0 active drivers ([]). There should only be one.",
        second_cpu_engineering_failure=dict(code_sha256=digest(ROOT / "audit_persistent_source_cpu_v2.py"),
            error="FileNotFoundError: SOURCE_IDENTITIES.sha256.json",
            fix="Use actual SOURCE_IDENTITIES hash binding in immutable asset supplement; no new sidecar or source write"),
        location="direction.load_assets -> FLA ops -> Triton autotuner driver initialization",
        cuda_visible_devices="", model_execution=False, gpu_measurements=0, gpu_hours=0,
        preserved_v1=True, fix="Versioned v3 extracts pure QDQ/KL AST; no FLA op import or kernel execution",
        scientific_or_quantization_definition_changed=False, no_gpu_retry=True))
    publish(failure.with_suffix(".sha256.json"), dict(path=str(failure), sha256=digest(failure)))
    value = dict(status="CPU_SOURCE_AND_ARITHMETIC_AUDIT_PASS_REAL_GPU_PENDING",
        actual_model_or_gpu_executed=False, cuda_initialized=False, additional_gpu_hours=0,
        pair_selection_sha256=digest(ROOT / "PROSPECTIVE_DOCUMENT_AND_PAIR_SELECTION.json"),
        completed_total_gpu_hours_upper_bound=ledger["total_gpu_hours_upper_bound"],
        kl_bitwise_original_cpu_cases=count, original_qdq_cpu_checks=qdq_checks,
        original_quantizer_source=str(transition_path), pure_original_function_ast_only=True,
        hybrid_source_identified_not_gpu_qualified=True,
        source_identities={str(path): digest(path) for path in sorted(source_paths)},
        package_versions={x: importlib.metadata.version(x) for x in ("torch", "transformers", "triton", "fla-core")},
        package_name_resolution=dict(import_name="fla", distribution_name="fla-core",
                                     generic_flash_linear_attention_metadata_absent=True),
        engine_sha256=digest(ROOT / "persistent_engine_v1.py"), audit_code_sha256=digest(__file__),
        engineering_failure_receipt_sha256=digest(failure),
        calibration_canonical_identities=identities, long_run_authorized=False,
        remaining_requirements=["frozen_worker_launcher_and_runtime_assets", "true_gpu_hybrid_repeat_parity_and_consumption",
                               "same_engine_microbenchmark_and_storage_vram_continuity", "complete_four_units_plus_fp_total_cost_gate"])
    path = ROOT / "PERSISTENT_SOURCE_CPU_AUDIT_V3.json"
    publish(path, value); publish(path.with_suffix(".sha256.json"), dict(path=str(path), sha256=digest(path)))
    print(json.dumps(dict(status=value["status"], sha256=digest(path), source_files=len(source_paths),
                         kl_original_cpu_cases=count, cuda_initialized=False, long_run_authorized=False)))


if __name__ == "__main__":
    main()
