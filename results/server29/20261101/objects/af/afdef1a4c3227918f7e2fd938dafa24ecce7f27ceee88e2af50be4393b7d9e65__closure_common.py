"""Engineering-only helpers; no training, generation, scoring or propagation metrics."""
import ast, hashlib, importlib.util, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
PARENT = REPO / 'experiments/QWEN_GDN_PROPAGATION_SURROGATE_VALIDATION_V1'
SRC = Path('/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen')
ARCH = Path('/data/zypan/transformers-qwen35/src/transformers/models/qwen3_5/modeling_qwen3_5.py')
FORMAL = SRC / 'experiments/qwen_gdn/QWEN_C5_C6_AIME26_CANONICAL60_V1/scripts/run_qwen_c5_c6_formal.py'
READINESS = SRC / 'experiments/QWEN_GDN_48GB_FORMAL_TRAINING_READINESS_CLOSURE_V2/scripts/runtime.py'
SEED = 73619

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''): h.update(b)
    return h.hexdigest()

def read(p): return json.loads(Path(p).read_text())
def check(ok, msg):
    if not ok: raise RuntimeError(msg)
def save(p, obj):
    p = ROOT / p; p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f: json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False); f.write('\n')
def tensor_sha(t):
    import torch
    return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def record(t):
    import torch
    return dict(shape=list(t.shape), dtype=str(t.dtype), device=str(t.device), sha256=tensor_sha(t),
                finite=bool(torch.isfinite(t).all()), minimum=float(t.min()), maximum=float(t.max()))
def pt_save(p, obj):
    import torch
    p=ROOT/p; p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('xb') as f: torch.save(obj,f); f.flush(); os.fsync(f.fileno())
    p.chmod(0o444)
    return dict(path=str(p.relative_to(ROOT)), sha256=sha(p), bytes=p.stat().st_size)
def load_pt(p):
    import torch
    return torch.load(ROOT/p, map_location='cpu', weights_only=True)
def mod(name,p):
    sys.path.insert(0,str(p.parent)); s=importlib.util.spec_from_file_location(name,p)
    m=importlib.util.module_from_spec(s); sys.modules[name]=m; s.loader.exec_module(m); return m
def configure(reference=False):
    import torch
    check(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','CUBLAS workspace config')
    check(os.environ.get('CUDA_VISIBLE_DEVICES')=='0','qualification uses GPU0')
    torch.set_num_threads(1); torch.use_deterministic_algorithms(not reference)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    F=mod('closure_unchanged_formal',FORMAL); F.setup_seed(SEED)
    return F
def init_model(reference=False):
    F=configure(reference); R=mod('closure_unchanged_readiness',READINESS)
    model,tok=F.load_model_and_tokenizer()
    import transformers.models.qwen3_5.modeling_qwen3_5 as qmod
    return F,R.load_core(),model,tok,qmod
def cpu(v):
    import torch
    if torch.is_tensor(v): return v.detach().cpu().clone()
    if isinstance(v,dict): return {k:cpu(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)): return type(v)(cpu(x) for x in v)
    return v
def cuda(v):
    import torch
    if torch.is_tensor(v): return v.to('cuda:0')
    if isinstance(v,dict): return {k:cuda(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)): return type(v)(cuda(x) for x in v)
    return v
def verify_parent():
    rows=[]
    for line in (PARENT/'FINAL_SHA256SUMS').read_text().splitlines():
        h,rel=line.split(None,1); rel=rel.lstrip('*'); actual=sha(PARENT/rel)
        check(actual==h,'frozen parent drift '+rel); rows.append(dict(path=rel,sha256=actual))
    p=read(PARENT/'protocol/preregistration_repaired_001.json')
    sources=[]
    for path,h in p['source_sha256'].items():
        check(sha(path)==h,'frozen source drift '+path); sources.append(dict(path=path,sha256=h))
    return dict(manifest_sha256=sha(PARENT/'FINAL_SHA256SUMS'), files=rows, sources=sources,
                final_verdict=read(PARENT/'analysis/final_verdict.json'), hash_gate='PASS',
                repeated_recovery=False, repeated_rotation_reconstruction=False)
def metric(a,b):
    import torch
    a=a.detach().double().cpu(); b=b.detach().double().cpu(); d=(a-b).abs()
    denom=a.abs().clamp_min(1e-12)
    return dict(max_abs=float(d.max()), max_rel=float((d/denom).max()),
                relative_L2=float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(a).clamp_min(1e-12)),
                exact_equal_fraction=float((a==b).double().mean()), byte_exact=bool(torch.equal(a,b)),
                reference_max_abs=float(a.abs().max()), finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()))
