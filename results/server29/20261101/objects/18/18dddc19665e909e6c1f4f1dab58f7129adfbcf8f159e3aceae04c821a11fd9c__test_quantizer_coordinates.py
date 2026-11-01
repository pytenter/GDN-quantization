"""CPU-only exact quantizer/basis check against inherited deployed code."""
import ast
import json
import torch
from pathlib import Path
from run_short_horizon_proxy_v2 import ROOT, BRIDGE, TRANSPORT, import_file, atomic_json, sha256

torch.set_num_threads(2)
transport = import_file(TRANSPORT, "proxy_cpu_transport_audit")
config = json.loads((ROOT / "frozen_config.json").read_text())
manifest = json.loads((BRIDGE / "rotation_manifest.json").read_text())
direction = import_file(Path("/data/zypan/experiment_runs/LING_KDA_LONG_HORIZON_DIRECTION_CAUSAL_V1/scripts/run_direction_causal.py"), "proxy_cpu_direction")
_, _, _, core = direction.load_assets(6)
torch.manual_seed(1)
x = torch.randn(2, 3, 128, 128)
dequant, codes, scale = core.qdq_r128(x)
gold_scale = x.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 127
gold_codes = torch.round(x/gold_scale).clamp(-127, 127).to(torch.int8)
assert codes.dtype == torch.int8 and torch.equal(codes, gold_codes)
assert torch.equal(scale, gold_scale) and torch.equal(dequant, gold_codes.float()*gold_scale)
rows = []
for condition in manifest["bank"]["conditions"]:
    if condition["condition_id"] not in config["conditions"]:
        continue
    path = Path(transport.mapped(condition["artifact"]["path"]))
    assert sha256(path) == condition["artifact"]["sha256"]
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for layer in config["layers"]:
        matrices = payload["rotations"]
        r = (matrices[layer] if layer in matrices else matrices[str(layer)]).float()
        source = x[0,0]
        restored = (source @ r) @ r.T
        rel = float(torch.linalg.vector_norm(restored-source)/torch.linalg.vector_norm(source))
        assert torch.isfinite(r).all() and rel <= 1e-5
        physical = core.qdq_r128(source @ r)[0] @ r.T
        assert torch.isfinite(physical).all()
        rows.append({"condition": condition["condition_id"], "layer": layer,
                     "rotation_sha256": sha256(path), "unquantized_coordinate_identity_relative_l2": rel})
assert len(rows) == 16*18
atomic_json(ROOT / "QUANTIZER_AND_COORDINATE_TESTS.json", {
    "status": "PASS", "device": "CPU_ONLY", "torch_round_int8_rowwise_exact": True,
    "layer_rotation_checks": rows, "no_gpu_time": True, "no_gradients_or_optimization": True})
print("CPU_EXACT_QDQ_COORDINATE_TESTS_PASS", len(rows))
