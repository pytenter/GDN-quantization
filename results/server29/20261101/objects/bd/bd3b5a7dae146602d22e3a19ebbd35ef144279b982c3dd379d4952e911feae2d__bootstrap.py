"""CPU-only prospective identity/provenance preparation. No model or GPU forward."""
from pathlib import Path
import ast, hashlib, json, os, shutil, subprocess, sys
import torch
R=Path.cwd(); NAME='LING_FUSED_RMSNORM_GATED_DETERMINISM_CLOSURE_V1'; D=R/'experiments'/NAME
Q=R/'experiments/LING_V3_FIXED_CONFIG_REUSE_QUALIFICATION_V1'
V=R/'experiments/LING_L7R_L8_LAYER_AWARE_MATCHED_TRAINING_V3_CONTEMPORARY'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def emit(p,x):
    p=D/p;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def git(*args):return subprocess.check_output(['git',*args],cwd=R)
assert git('rev-parse','HEAD').decode().strip()=='a4ab59f4cb4e2a721eda7250177df9ec06188c0b'
assert not git('status','--porcelain').strip() and not D.exists()
# Verify the entire sealed qualification and recursively protected historical files before writes.
paths={}
old=load(Q/'provenance/protected_parent_inventory.json')
for r in old['files']:paths[r['path']]=r
sealed=load(Q/'manifests/final_sha256.json')
for p in Q.rglob('*'):
    if p.is_file() and '__pycache__' not in p.parts:
        paths[str(p)]=dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
for i,r in enumerate(paths.values()):
    p=Path(r['path']);assert sha(p)==r['sha256'] and p.stat().st_size==r['bytes'],str(p)
    if i%300==0:print('protected checked',i,flush=True)
# Parent's own manifest may use relative files. Its canonical Git blob is also checked below.
for p in Q.rglob('*'):
    if p.is_file() and '__pycache__' not in p.parts:
        rel=str(p.relative_to(R));assert hashlib.sha256(git('show','HEAD:'+rel)).hexdigest()==sha(p),rel
git('switch','-c','exp/ling-fused-rmsnorm-gated-determinism-closure-v1')
D.mkdir(parents=True)
emit(Path('provenance/protected_parent_inventory.json'),dict(gate='PASS',PARENT_IMMUTABILITY_GATE='PASS',files=sorted(paths.values(),key=lambda r:r['path']),pre_task_head='a4ab59f4cb4e2a721eda7250177df9ec06188c0b',parent_verdicts_immutable=True))
src=Path(load(Q/'analysis/call37_frozen_input_manifest.json').get('bundle_file',Q/'runtime/call37_frozen_tensors.pt'))
assert src.is_file() and src.resolve()==(Q/'runtime/call37_frozen_tensors.pt').resolve()
m=load(Q/'analysis/call37_frozen_input_manifest.json');assert sha(src)==m['bundle_sha256'] and src.stat().st_size==m['bundle_bytes']
dest=D/'runtime/call37_frozen_tensors.pt';dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
assert src.read_bytes()==dest.read_bytes()
b=torch.load(dest,map_location='cpu',weights_only=True);assert set(b)==set(m['tensors'])
rows={}
for k,x in b.items():
    h=hashlib.sha256(x.detach().reshape(-1).contiguous().view(torch.uint8).numpy().tobytes()).hexdigest();p=m['tensors'][k]
    assert h==p['sha256'] and str(x.dtype)==p['dtype'] and list(x.shape)==p['shape'] and list(x.stride())==p['stride']
    rows[k]=dict(name=k,sha256=h,dtype=str(x.dtype),shape=list(x.shape),stride=list(x.stride()),storage_offset=x.storage_offset(),storage_nbytes=x.untyped_storage().nbytes(),layout=str(x.layout),bytes=x.numel()*x.element_size(),source_parent_artifact=str(src),original_capture_metadata=p,saved_bundle_requires_grad=x.requires_grad)
assert sha(dest)==m['bundle_sha256']
emit(Path('protocol/prospective_call37_input_manifest.json'),dict(gate='PASS',PROSPECTIVE_CALL37_FROZEN_INPUT_GATE='PASS',source=str(src),source_manifest=str(Q/'analysis/call37_frozen_input_manifest.json'),source_manifest_sha256=sha(Q/'analysis/call37_frozen_input_manifest.json'),bundle=str(dest),bundle_sha256=sha(dest),bundle_bytes=dest.stat().st_size,tensor_count=len(b),tensor_bytes=sum(x['bytes'] for x in rows.values()),tensors=rows,norm_attributes=m['norm_attributes'],scalars=dict(residual=None,prenorm=False,residual_in_fp32=False,is_rms_norm=True),byte_identical_copy=True,no_mutation=True,storage_note='CPU serialization rebases storage offsets; saved strides/values match parent. Original GPU offsets are separately retained, not falsely asserted to be CPU offsets.'))
hist=git('rev-parse','origin/exp/ling-fresh-process-first-divergence-closure-v2').decode().strip()
assert hist==git('rev-parse','d757a35').decode().strip()=='d757a35fab7d901527804747e7bfe78712d7f6ad'
hm=load(Q/'runtime/historical_fixed_config_manifest.json');resolution=load(Q/'runtime/historical_fixed_config_provenance_resolution.json')
assert resolution['HISTORICAL_FIXED_CONFIG_PROVENANCE_GATE']=='PASS'
hp='experiments/ling_kda/runtime/repeatability_closure_v1/run_ling_repeatability.py'
hb=git('show',hist+':'+hp);assert hashlib.sha256(hb).hexdigest()==hm['gated_fix']['source_sha256']
helper=D/'runtime/historical/run_ling_repeatability.py';helper.parent.mkdir(parents=True,exist_ok=True);helper.write_bytes(hb)
tree=ast.parse(hb);fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==hm['gated_fix']['function'])
assert ast.get_source_segment(hb.decode(),fn)==hm['gated_fix']['source']
selector=next(n for n in ast.walk(fn) if isinstance(n,ast.Compare) and isinstance(n.left,ast.Call) and getattr(n.left.func,'attr',None)=='get')
envkey=ast.literal_eval(selector.left.args[0]);envvalue=ast.literal_eval(selector.comparators[0])
conf=next(n for n in ast.walk(fn) if isinstance(n,ast.Call) and getattr(n.func,'attr',None)=='Config')
configuration=dict(kwargs=ast.literal_eval(conf.args[0]),**{k.arg:ast.literal_eval(k.value) for k in conf.keywords})
assert configuration=={k:v for k,v in hm['gated_fix']['evidence'].get('fixed_config',configuration).items()} if 'fixed_config' in hm['gated_fix']['evidence'] else True
cfgdir=D/'runtime/historical/fixed_fla_configs';cfgdir.mkdir()
fixed=[]
for row in resolution['fixed_kernels']:
    if 'config_sha256' not in row:continue
    p='experiments/ling_kda/runtime/fresh_process_first_divergence_closure_v2/fixed_fla_configs/'+row['kernel']+'.json'
    raw=git('show',hist+':'+p);assert hashlib.sha256(raw).hexdigest()==row['config_sha256']
    (cfgdir/(row['kernel']+'.json')).write_bytes(raw)
    fixed.append(dict(git_path=p,sha256=row['config_sha256'],config=json.loads(raw)))
rep=Q/'runtime/historical_replication/runtime_manifest.json';shutil.copyfile(rep,D/'runtime/historical/runtime_manifest.json')
ss=Q/'runtime/source_snapshot/V3_env/fla/modules/fused_norm_gate.py';actual=V.parents[0]/'LING_L7R_L8_LAYER_AWARE_MATCHED_TRAINING_V2_CLEANROOM/runtime/env/lib/python3.11/site-packages/fla/modules/fused_norm_gate.py'
assert sha(ss)==sha(actual)
emit(Path('runtime/historical_fixed_forward_manifest.json'),dict(gate='PASS',HISTORICAL_FIXED_FORWARD_PROVENANCE_GATE='PASS',commit=hist,ref_verified=True,helper=str(helper),helper_sha256=sha(helper),function=fn.name,source=ast.get_source_segment(hb.decode(),fn),configuration=configuration,selector={envkey:envvalue},fixed_fla_configs=fixed,historical_runtime_manifest=str(rep),historical_runtime_manifest_sha256=sha(rep),provenance_resolution=str(Q/'runtime/historical_fixed_config_provenance_resolution.json'),provenance_limit=resolution['provenance_limit'],fla_source=str(actual),fla_source_sha256=sha(actual),launch_only=True,mathematical_semantics_unchanged=True,historical_backward_configs=[],backward_config_audit='All six committed fixed JSON kernel names are forward/inference; gated helper modifies only layer_norm_gated_fwd_kernel. No validated backward fixed configuration recorded in this finite recovered closure.'))
emit(Path('protocol/measurement_plan.json'),dict(protocol='User attachment SHA separately copied',hardware='PHYSICAL_RTX4090_GPU0_ONLY',same_process_forward_calls=5,fresh_forward_processes=5,whole_update_processes=5,input='immutable parent call37 bundle; V3 TRAIN prefix for disposable whole update',recipe='unchanged V3 historical Local + 0.1 State, seed0, Adam lr.003 wd0 clip1',forward_fix='original helper AST only; historical JSON files byte-identical',no_config_search=True,no_tolerance=True,formal_training=False,Fresh_B=0,AIME=0,backward_policy='If whole update FAIL, compare ordered loss/gradient boundaries; isolated same frozen exact direct backward inputs five processes; only provenance-backed historical backward fixes; otherwise isolated mathematical reference five forward and five backward then STOP.',success_policy='Both forward and whole update exact => automatic V4 preparation; no formal training until all V4 readiness gates PASS'))
# Mechanical extraction of the original numerical main. Only parent gate calls and artifact destination are removed.
sp=V/'scripts/contemporary_smoke.py';original=sp.read_text();mt=ast.parse(original);main=next(n for n in mt.body if isinstance(n,ast.FunctionDef) and n.name=='main')
start=main.lineno; body=main.body
removed=[];kept=[]
for n in body:
    seg=ast.get_source_segment(original,n)
    if ('active_stop()' in seg or "runtime/v3_runtime_identity.json')['gate']" in seg or seg.startswith('source_freeze=') or "source_freeze['files']" in seg or "condition=='L7R' or (condition=='L8'" in seg):removed.append(seg)
    else:kept.append(n)
main.body=kept
for i,n in enumerate(main.body):
    if ast.get_source_segment(original,n)=='values[0].backward()':
        main.body.insert(i,ast.parse('observe_graph(values[0])').body[0]);break
else:raise AssertionError('backward boundary not found')
text=ast.unparse(main)+'\n'
p=D/'runtime/mechanical_v3_whole_main.py';p.write_text(text)
emit(Path('provenance/mechanical_whole_main.json'),dict(source=str(sp),source_sha256=sha(sp),generated=str(p),generated_sha256=sha(p),removed_parent_orchestration_statements=removed,added_non_numerical_observer='observe_graph(values[0]) before unchanged backward',numeric_AST_statements_unchanged=True,artifact_writer_redirected_by_execution_namespace=True))
print('CPU_PREFLIGHT_PASS',len(paths),'parent files',len(b),'tensors',sha(dest),configuration,flush=True)
