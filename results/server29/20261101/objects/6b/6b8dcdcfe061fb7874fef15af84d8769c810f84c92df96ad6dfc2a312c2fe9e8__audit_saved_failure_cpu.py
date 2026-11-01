"""Independent CPU-only recomputation of the already-saved GPU first failure."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,pathlib,json,hashlib,time,subprocess,importlib.metadata
R=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
import torch
import gpu_worker as W
torch.set_num_threads(1)
assert not torch.cuda.is_initialized()
p=R/'analysis/FIRST_FAILURE_TENSORS.pt';t=torch.load(p,map_location='cpu',weights_only=True)
print('saved_keys',list(t))
a=t['reference'];b=t['candidate'];_,N,_=W.originals();got=N.compare(a,b,'model_bf16');old=W.read(R/'FIRST_FAILURE.json')['first_failure']
assert got==old['comparison'];assert W.tsha(a)==old['reference_sha'] and W.tsha(b)==old['candidate_sha']
assert not got['PASS'] and got['finite'];assert got['relative_L2']>got['relative_L2_cap']
assert got['max_abs']<=got['max_abs_cap'] and got['max_rel_floor_stabilized']<=got['max_rel_cap']
assert not torch.cuda.is_initialized()
versions={}
for n in ['torch','transformers','tokenizers','triton','flash-linear-attention','fla']:
 try:versions[n]=importlib.metadata.version(n)
 except importlib.metadata.PackageNotFoundError:versions[n]='DISTRIBUTION_METADATA_NOT_FOUND'
repos=[]
paths=['/data/zypan/transformers-qwen35','/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen','/data/zypan/worktrees/qwen-gdn-deterministic-scale-discovery-v3']
for path in paths:
 proc=subprocess.run(['git','-C',path,'rev-parse','HEAD'],capture_output=True,text=True)
 repos.append(dict(path=path,head=proc.stdout.strip() if proc.returncode==0 else 'NOT_GIT_REPOSITORY',returncode=proc.returncode))
result=dict(status='PASS_REPRODUCED_FROZEN_GATE_FAILURE',scientific_gate_status='FAIL',analysis='Read-only saved logits; no model load/forward/CUDA context',saved_tensor_file_sha256=W.sha(p),reference_sha256=W.tsha(a),candidate_sha256=W.tsha(b),comparison=got,comparison_exactly_matches_GPU_receipt=True,failing_predicates=['relative_L2'],context=old['context'],versions=versions,repositories=repos,CUDA_initialized=False,GPU_HOURS_USED=0,root_cause_limit='Evidence establishes a real native-vs-H full-model logit mismatch under the frozen caps. Saved evidence contains final logits only; the earliest internal layer/op causing it is NOT_LOCALIZED. Floating-point coordinate-transform/reduction/cast effects are a hypothesis, not a verified cause. No basis to alter caps, skip A0, or infer Dual-M quality.',verified_unix=time.time())
with (R/'analysis/NUMERICAL_FIRST_FAILURE_CPU_AUDIT.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(dict(status=result['status'],relative_L2=got['relative_L2'],cap=got['relative_L2_cap'],CUDA_initialized=False)))
