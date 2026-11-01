"""V4 only: read-only scientific parents and mechanically frozen qualified launch selections."""
from pathlib import Path
import ast,hashlib,json,os,sys
import torch
D=Path(__file__).resolve().parents[1];R=D.parents[1]
V=R/'experiments/LING_L7R_L8_LAYER_AWARE_MATCHED_TRAINING_V3_CONTEMPORARY'
C=R/'experiments/LING_FUSED_RMSNORM_GATED_DETERMINISM_CLOSURE_V1'
sys.path.insert(0,str(V/'scripts'))
import common as c
import contemporary_smoke as s
def emit(p,x):c.emit(D/p,x)
def load(p):return c.load(p)
def freeze_check():
    for r in load(D/'runtime/readiness_source_freeze.json')['files']:assert c.sha(r['path'])==r['sha256'],r['path']
    assert load(C/'reports/FINAL_VERDICT.json')['DETERMINISTIC_TRAINING_RUNTIME_QUALIFICATION']=='PASS'
    assert os.environ['CUDA_VISIBLE_DEVICES']=='0' and c.hardware()['gpus'][0]['gpu_uuid']==c.UUIDS[0]
def apply_qualified_runtime():
    import triton
    hm=load(D/'runtime/historical_fixed_forward_manifest.json')
    for k,v in hm['selector'].items():assert os.environ[k]==v
    source=Path(hm['helper']).read_text();assert c.sha(hm['helper'])==hm['helper_sha256']
    fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==hm['function'])
    ns={'os':os};exec(compile(ast.Module(body=[fn],type_ignores=[]),hm['helper'],'exec'),ns);assert ns[fn.name]()
    import fla.modules.fused_norm_gate as f
    q=load(D/'runtime/qualified_backward_selection.json')['configuration'];kernel=f.layer_norm_gated_bwd_kernel
    while kernel is not None:
        if hasattr(kernel,'configs'):
            kernel.configs=[triton.Config(q['kwargs'],num_warps=q['num_warps'],num_stages=q['num_stages'],num_ctas=q['num_ctas'])]
            if isinstance(getattr(kernel,'cache',None),dict):kernel.cache.clear()
        kernel=getattr(kernel,'fn',None)
    return f
