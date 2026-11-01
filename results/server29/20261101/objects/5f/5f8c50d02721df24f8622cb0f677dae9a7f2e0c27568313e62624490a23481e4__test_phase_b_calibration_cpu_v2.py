"""CPU qualification of the new input adapter; explicitly NOT GPU qualification."""
import ast
import json
import os
import time
from pathlib import Path
from objective import digest
from run_matched_l7 import ROOT, PRIOR, load_module, publish
from phase_b_calibration_common_v2 import checked_json, read
from run_phase_b_calibration_v2 import load_inputs


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU_ONLY_TEST_REQUIRED"
    import torch
    assert not torch.cuda.is_initialized(), "UNEXPECTED_CUDA_INITIALIZATION"
    selection = checked_json(ROOT / "PROSPECTIVE_DOCUMENT_SELECTION.json")
    cfg = read(ROOT / "PHASE_A_CONFIG.json")
    cfg["documents"] = [x["document_id"] for x in selection["selected_documents"]]
    cfg["document_gpu_map"] = {x["document_id"]: x["gpu"] for x in selection["selected_documents"]}
    base = load_module(PRIOR / "run_short_horizon_proxy_v2.py", "l7_b_cpu_base")
    transport = load_module(base.TRANSPORT, "l7_b_cpu_transport")
    _, _, _, original, identities = base.load_tokens_and_rotations(
        torch, transport, cfg, "pg19-test-37702", cfg["conditions"])
    rows = []
    for doc in cfg["documents"]:
        entry, tokens, path, matrices, new_identities = load_inputs(torch, cfg, doc, cfg["conditions"])
        assert len(matrices) == 16 and set(new_identities) == set(identities)
        assert all(torch.equal(matrices[k], original[k]) and new_identities[k] == identities[k] for k in matrices)
        rows.append(dict(document=doc, token_count=len(tokens), token_sha256=digest(path),
                         full16_rotation_tensors_bitwise_equal_to_original_adapter=True,
                         frozen_tokenizer_reencoding_exact=entry["frozen_tokenizer_reencoding_exact"]))
    sources = ["phase_b_calibration_common_v2.py", "run_phase_b_calibration_v2.py", "launch_phase_b_calibration_v2.py",
               "budget_phase_b_calibration_v2.py", "audit_phase_b_calibration_v2.py", "freeze_phase_b_calibration_v2.py"]
    for name in sources:
        ast.parse((ROOT / name).read_text())
    assert not torch.cuda.is_initialized()
    path = ROOT / "PHASE_B_CALIBRATION_CPU_TESTS.json"
    publish(path, dict(status="CPU_INPUT_ADAPTER_AND_SOURCE_CHECK_PASS_GPU_NOT_RUN", epoch=time.time(),
        rows=rows, frozen_roster_mathematics_unchanged=True, gpu_hours=0,
        gpu_qualification_pass=False, no_new_future_outcomes_accessed=True,
        source_sha256={name: digest(ROOT / name) for name in sources}))
    publish(path.with_suffix(".sha256.json"), dict(path=str(path), sha256=digest(path)))
    print(json.dumps(dict(status="CPU_INPUT_ADAPTER_PASS_NOT_GPU_PASS", path=str(path), sha256=digest(path))))


if __name__ == "__main__":
    main()
