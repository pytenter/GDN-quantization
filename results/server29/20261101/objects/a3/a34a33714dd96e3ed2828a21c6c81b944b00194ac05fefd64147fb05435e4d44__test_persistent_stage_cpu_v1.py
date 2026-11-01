"""CPU protocol/identity/projection tests. No imports of GPU-only FLA operations."""
import ast
import json
import os
from pathlib import Path
import torch
from objective import digest
from run_matched_l7 import ROOT, publish
from persistent_stage_common_v1 import MODES, project_document, fingerprint


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized()
    assert MODES["formal"]["endpoint"] == 32768 and MODES["formal"]["primary"] == 8192
    assert MODES["formal"]["secondary"] == 2048 and MODES["micro"]["endpoint"] > 8192
    assert MODES["qualification"]["repeats"] == 2
    expected_keys = ("endpoint", "fp_reference_full_logit_sha256", "fp_replay_parity_tokens",
        "hybrid_cache_inventories", "results", "full_model_teacher_forced",
        "every_token_qdq_including_after_anchors", "same_gpu_live_fp_reference")
    x = {k: k for k in expected_keys}; y = dict(x, elapsed_seconds=99.)
    assert fingerprint(x) == fingerprint(y)
    y["fp_reference_full_logit_sha256"] = "changed"
    assert fingerprint(x) != fingerprint(y)
    micro = dict(endpoint=8448, total_wall_seconds=900.,
        stream=dict(blocks=[dict(end=2048, tokens=512, elapsed_seconds=500.),
                            dict(end=8192, tokens=512, elapsed_seconds=51.2),
                            dict(end=8448, tokens=256, elapsed_seconds=25.6)],
                    hybrid_cache_inventories={"8447": {k:dict(cache_bytes=1024**3) for k in ("FP", "A", "B")}}),
        gpu_memory=dict(peak_allocated_bytes=20*1024**3, total_bytes=48*1024**3))
    p = project_document(micro)
    assert abs(p["max_late_block_seconds_per_token"] - .1) < 1e-12 and p["memory_pass"]
    assert p["conservative_seconds"] > 900 + (32768-8448)*.1
    assert p["fp_plus_two_rotations"]
    micro["gpu_memory"]["total_bytes"] = 24*1024**3
    assert not project_document(micro)["memory_pass"]
    names = ["persistent_engine_v1.py", "persistent_stage_common_v1.py", "run_persistent_stage_v1.py",
             "launch_persistent_stage_v1.py", "audit_persistent_stage_v1.py", "budget_persistent_complete_v1.py"]
    for name in names:
        source = (ROOT / name).read_text(); ast.parse(source)
        assert "optimizer.step(" not in source and "os.kill(" not in source
        assert "kill-session" not in source and "kill-window" not in source
    assert not torch.cuda.is_initialized()
    value = dict(status="CPU_PROTOCOL_AND_COST_TESTS_PASS_GPU_PENDING", cuda_initialized=False,
        actual_model_or_gpu_executed=False, additional_gpu_hours=0, long_run_authorized=False,
        mode_contracts_pass=True, repeat_fingerprint_detects_numerical_change=True,
        conservative_full_three_stream_cost_and_memory_gate_pass=True,
        no_kill_no_optimizer_source_checks=True, code_sha256={x:digest(ROOT/x) for x in names},
        test_sha256=digest(__file__))
    path = ROOT / "PERSISTENT_STAGE_CPU_TESTS_V1.json"
    publish(path, value); publish(path.with_suffix(".sha256.json"), dict(path=str(path), sha256=digest(path)))
    print(json.dumps(dict(status=value["status"], sha256=digest(path), gpu_work=False)))


if __name__ == "__main__":
    main()
