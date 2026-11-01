"""Read-only source/asset audit; records identities, not target rankings."""
import ast
import json
import subprocess
from pathlib import Path
from run_short_horizon_proxy_v2 import ROOT, BRIDGE, HIST_SOURCE, DIRECTION_RUNNER, TRANSPORT, sha256, atomic_json

parent = Path("/data/zypan/worktrees/ling-long-horizon-propagation-confirm-v1")
trainer = parent / "inherited/data01/user2/worktrees/ling-l6-l7-objective-horizon-diagnosis-v1/experiments/LING_L7R_L8_LAYER_AWARE_MATCHED_TRAINING_V3_CONTEMPORARY/inputs/historical_trainer.py"
cayley = trainer.parent / "runtime_repo/experiments/shared/rotation/cayley_rotation.py"
trace_root = Path("/data/zypan/worktrees/ling-error-coherence-diagnosis-v1/experiments/LING_ERROR_COHERENCE_DIAGNOSIS_V1/discovery/traces")
chunks = sorted(trace_root.rglob("*.pt"))
config = json.loads((ROOT / "frozen_config.json").read_text())
paths = [trainer, cayley, DIRECTION_RUNNER, TRANSPORT,
         HIST_SOURCE / "run_ling_kda_query_readout_causal_closure_v1.py", HIST_SOURCE / "pytorch_kda_reference.py",
         BRIDGE / "rotation_manifest.json", BRIDGE / "document_manifest.json",
         Path("/data/zypan/models/Ling-3.0-tiny/config.json"),
         Path("/data/zypan/models/Ling-3.0-tiny/model.safetensors.index.json")]
audit = {"source_hashes": {str(p): sha256(p) for p in paths}, "archived_fp_trace_chunks": len(chunks),
         "trace_bytes": sum(p.stat().st_size for p in chunks), "archived_trace_used_for_proxy": False,
         "trace_file_inventory": [str(p) for p in chunks],
         "source_commits": {str(p): subprocess.check_output(["git", "-C", str(p), "rev-parse", "HEAD"], text=True).strip()
                             for p in [parent, BRIDGE.parents[1]]},
         "model_identity": json.loads(Path("/data/zypan/experiment_runs/LING_KDA_LONG_HORIZON_DIRECTION_CAUSAL_V1/SOURCE_IDENTITIES.json").read_text())["model"],
         "semantic_findings": {
             "state": "[B,H,K,V] = [1,16,128,128]; 18 KDA layers; hybrid MLA cache also retained",
             "physical_qdq": "S -> S @ R -> rowwise symmetric INT8 over V -> dequant @ R.T -> FP32 cache; every token",
             "k3": "historical query_readout and prepared_query function bodies reused unchanged",
             "query": "q/sqrt(sum(q^2)+1e-6)/sqrt(128); FP query, post-update E",
             "historical_label": "FP||pulse full-vocabulary KL AUC, natural persistent-QDQ prefix to anchor, no further QDQ after anchor",
             "trainer": "Cayley SO(128), 8128 parameters/layer; exact-forward identity-backward STE; recurrent truncated BPTT",
             "training_boundary_difference": "historical trainer casts quant boundary/dequant to BF16; deployed inherited Phase2 physical cache is FP32",
             "L7_objective": "local gate+o_norm+o_proj relative MSE + 0.1 state loss, not identical to present full-logit KL baseline",
             "autograd": "historical readout helper detaches E and q; present evaluator is inference-only, not training-ready"},
         "no_optimizer_step": True, "no_new_rotation": True}
assert len(chunks) == 256
assert audit["model_identity"]["config_sha256"] == sha256(Path("/data/zypan/models/Ling-3.0-tiny/config.json"))
atomic_json(ROOT / "SOURCE_AND_ASSET_AUDIT.json", audit)
print("SOURCE_ASSET_AUDIT_RECORDED", len(chunks))
