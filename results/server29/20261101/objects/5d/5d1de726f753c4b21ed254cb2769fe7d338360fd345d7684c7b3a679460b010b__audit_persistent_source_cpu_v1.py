"""Append-only CPU source/quantizer/scorer audit; no model or GPU execution."""
import ast
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import torch
from objective import digest
from run_matched_l7 import ROOT, PRIOR, load_module, publish
from phase_b_calibration_common_v2 import verify, audit_b, checked_json
from select_phase_b_pairs import validate_completed
from persistent_engine_v1 import full_kl


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    cfg = verify()
    selection = checked_json(ROOT / "PROSPECTIVE_DOCUMENT_AND_PAIR_SELECTION.json")
    assert selection["status"] == "BOTH_DOCUMENTS_DISCRIMINATIVE_PAIRS_FROZEN"
    runs, identities = validate_completed(cfg)
    assert all(identities[x["document"]] == x["calibration_identity"] for x in selection["documents"])
    ledger = audit_b()
    assert all(x["status"] == "COMPLETE_MECHANICAL_PASS" for x in ledger["phase_b_calibration_runs"])
    base = load_module(PRIOR / "run_short_horizon_proxy_v2.py", "persistent_source_cpu_base")
    direction = load_module(base.DIRECTION_RUNNER, "persistent_source_cpu_direction")
    transport, common, capture, core = direction.load_assets(3)
    source_paths = {Path(base.DIRECTION_RUNNER), Path(inspect.getfile(core.qdq_r128)),
                    Path(inspect.getfile(core.update_pre_q)), Path(inspect.getfile(capture)),
                    capture.PARENTS[2] / "scripts/phase2_final_hidden_512_smoke.py",
                    capture.V4 / "scripts/v4_common.py",
                    Path("/data/zypan/models/Ling-3.0-tiny/modeling_bailing_moe_v3.py"),
                    Path("/data/zypan/models/Ling-3.0-tiny/configuration_bailing_moe_v3.py")}
    import transformers.cache_utils
    import fla.ops.kda
    source_paths.add(Path(inspect.getfile(transformers.cache_utils)))
    source_paths.add(Path(inspect.getfile(fla.ops.kda)))
    model_source = Path("/data/zypan/models/Ling-3.0-tiny/modeling_bailing_moe_v3.py").read_text()
    # Static evidence, not execution or correctness of the real hybrid cache.
    for expression in ("past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)",
                       "past_key_value.layers[self.layer_idx].keys = recurrent_state",
                       "past_key_value.layers[self.layer_idx].values = (conv_state_q, conv_state_k, conv_state_v)",
                       "past_key_values = DynamicCache()"):
        assert expression in model_source, "HYBRID_SOURCE_LAYOUT_NOT_IDENTIFIED"
    namespace = dict(torch=torch, EPS=1e-12)
    node = next(x for x in ast.parse(Path(base.DIRECTION_RUNNER).read_text()).body
                if isinstance(x, ast.FunctionDef) and x.name == "kl_metrics")
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(base.DIRECTION_RUNNER), "exec"), namespace)
    torch.manual_seed(0)
    count = 0
    for dtype in (torch.float32, torch.bfloat16):
        for magnitude in (1., 30., 100.):
            a, b = (torch.randn(1, 1, 509).to(dtype) * magnitude for _ in range(2))
            assert full_kl(torch, a, b) == namespace["kl_metrics"](torch, a, b)["KL"]
            count += 1
    matrices = torch.stack([torch.eye(128), torch.linalg.qr(torch.randn(128, 128))[0]])
    qdq_checks = []
    for matrix in matrices:
        state = torch.randn((1, 2, 4, 128))
        rotated = state.matmul(matrix)
        dequant, codes, scale = core.qdq_r128(rotated)
        expected_scale = rotated.float().abs().amax(-1, keepdim=True).clamp_min(1e-12) / 127.
        expected_code = torch.round(rotated.float() / expected_scale).clamp(-127, 127).to(torch.int8)
        assert codes.dtype == torch.int8 and torch.equal(codes, expected_code)
        assert torch.equal(scale, expected_scale)
        assert torch.equal(dequant, expected_code.float() * expected_scale)
        inverse = dequant.matmul(matrix.t())
        assert torch.isfinite(inverse).all()
        qdq_checks.append(dict(real_original_quantizer_cpu_codes_scales_dequant_bitwise_equal=True,
                               group_width=128, inverse_finite=True))
    assert not torch.cuda.is_initialized()
    value = dict(status="CPU_SOURCE_AND_ARITHMETIC_AUDIT_PASS_REAL_GPU_PENDING",
        actual_model_or_gpu_executed=False, cuda_initialized=False, additional_gpu_hours=0,
        pair_selection_sha256=digest(ROOT / "PROSPECTIVE_DOCUMENT_AND_PAIR_SELECTION.json"),
        completed_total_gpu_hours_upper_bound=ledger["total_gpu_hours_upper_bound"],
        kl_bitwise_original_cpu_cases=count, original_qdq_cpu_checks=qdq_checks,
        original_quantizer_source=str(inspect.getfile(core.qdq_r128)),
        hybrid_source_identified_not_gpu_qualified=True,
        source_identities={str(path): digest(path) for path in sorted(source_paths)},
        package_versions={x: importlib.metadata.version(x) for x in ("torch", "transformers", "triton", "fla-core")},
        package_name_resolution=dict(import_name="fla", distribution_name="fla-core",
                                     generic_flash_linear_attention_metadata_absent=True),
        engine_sha256=digest(ROOT / "persistent_engine_v1.py"), audit_code_sha256=digest(__file__),
        calibration_canonical_identities=identities, long_run_authorized=False,
        remaining_requirements=["frozen_worker_launcher_and_runtime_assets", "true_gpu_hybrid_repeat_parity_and_consumption",
                               "same_engine_microbenchmark_and_storage_vram_continuity", "complete_four_units_plus_fp_total_cost_gate"])
    path = ROOT / "PERSISTENT_SOURCE_CPU_AUDIT_V1.json"
    publish(path, value); publish(path.with_suffix(".sha256.json"), dict(path=str(path), sha256=digest(path)))
    print(json.dumps(dict(status=value["status"], sha256=digest(path), source_files=len(source_paths),
                         kl_original_cpu_cases=count, cuda_initialized=False, long_run_authorized=False)))


if __name__ == "__main__":
    main()
