"""Observer plumbing tests; deliberately not a full-model GPU qualification."""
from types import SimpleNamespace
from pathlib import Path
import hashlib
import json
import sys
sys.dont_write_bytecode = True
import torch
from objective import original_helpers
from run_matched_l7 import OutputTap, ObservedRunner, ObservedCore, augment, publish, ROOT, TRAINER

torch.set_num_threads(2)
torch.set_grad_enabled(False)
torch.manual_seed(12)
h = original_helpers(torch, TRAINER)

class Norm(torch.nn.Module):
    def forward(self, x, gate):
        return x + gate

layers = [0,1]
modules = [SimpleNamespace(attention=SimpleNamespace(
    o_norm=Norm(), o_proj=torch.nn.Linear(8,5,bias=False))) for _ in layers]
capture = SimpleNamespace(model=SimpleNamespace(model=SimpleNamespace(layers=modules)))
cache = {i: torch.ones(1,16,128,128) * (i+1) for i in layers}
fixture = torch.randn(1,1,2,4)
gate = torch.randn_like(fixture)

def step(torch, capture, cache, token, pos):
    for layer in layers:
        a = capture.model.model.layers[layer].attention
        a.o_proj(a.o_norm(fixture+layer,gate).reshape(1,1,-1))
    return cache, torch.tensor([[[0.0,1.0,2.0]]]), {}

def qdq(torch, core, capture, cache, r):
    for value in cache.values():
        value.add_(0.1)
    return {i: {"scale": 1.0} for i in layers}

direction = SimpleNamespace(model_step=step, quantize_l6_cache=qdq,
    tensor_sha=lambda x: hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())
base = SimpleNamespace(stack_cache=lambda torch,runner,capture,cache,layers:
    torch.cat([cache[i] for i in layers],0))
core = SimpleNamespace(update_pre_q=lambda state,rec: torch.cat([cache[i] for i in layers],0).clone())
tap = OutputTap(torch,capture,layers)
obs = ObservedRunner(base,direction,torch,core,capture,tap,h,[256],layers)
wrapped = ObservedCore(core,obs)
for pos in [256,257]:
    obs.model_step(torch,capture,cache,4,pos)
assert len(obs.helper_forward_checks) == 4
assert all(x["bitwise_equal"] and x["identity_loss"] == 0 for x in obs.helper_forward_checks)
obs.mode = "student"
for pos in range(256,384):
    obs.fp_outputs[pos] = obs.fp_outputs[256]
    wrapped.update_pre_q(None,None)
    obs.model_step(torch,capture,cache,4,pos)
    obs.quantize_l6_cache(torch,wrapped,capture,cache,None)
assert len(obs.scores) == 128
assert all(v[0] == 0 and v[1] > 0 and
           v[2] == float(torch.tensor(v[1],dtype=torch.float32)*0.1)
           for v in obs.scores.values())
result = augment({"windows": [{"start": 256}], "aggregate": {}},obs,[256])
assert result["aggregate"]["L7_FUNCTIONAL_ORIGINAL"] == 0
assert result["aggregate"]["L7_STATE_ORIGINAL"] > 0
tap.close()
publish(ROOT / "CPU_OBSERVER_PLUMBING_TESTS.json", {
    "status": "CPU_OBSERVER_PLUMBING_PASS", "real_model_or_gpu_used": False,
    "post_o_proj_coverage": len(layers), "scored_tokens": len(obs.scores),
    "real_gpu_parity": "NOT_EXECUTED", "gpu_hours": 0,
    "checks": ["hook output exact helper replay", "teacher-reference handoff",
               "pre-QDQ functional / post-QDQ state timing", "H128 token mean"],
    "torch": torch.__version__})
print("CPU_OBSERVER_PLUMBING_PASS; NOT a GPU/model qualification")
