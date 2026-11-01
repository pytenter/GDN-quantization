"""V3 read-only parent audit and isolated copies. Never launches a GPU process."""
import ast, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path

R=Path(sys.argv[1]).resolve()
P=Path('/data/zypan/analyses/qwen_gdn_oscar_sym32_p1_c1_s3_unified_recovery_aime26_v2_rerun_20261022')
V1=Path('/data/zypan/analyses/qwen_gdn_oscar_sym32_p1_c1_s3_aime26_seed1_v1')
V2=Path('/data/zypan/analyses/qwen_gdn_oscar_sym32_p1_c1_s3_unified_recovery_aime26_v2')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def put(n,x):
 p=R/n;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
def row(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def cp(s,n):
 d=R/n;d.parent.mkdir(parents=True,exist_ok=True)
 assert not d.exists();shutil.copyfile(s,d);assert sha(s)==sha(d)
 return dict(original=row(s),copy_path=n)
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
assert not (R/'analysis/INPUT_COLLECTION_STARTED.json').exists()
put('analysis/INPUT_COLLECTION_STARTED.json',dict(pid=os.getpid(),monotonic=time.monotonic(),GPU_dispatches=0))
freeze=read(P/'SOURCE_FREEZE.json');before=[]
for f in freeze['files']:
 p=P/f['path'];assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256'];before.append(f)
assert len(before)==622 and sha(P/'SOURCE_FREEZE.json')=='3955f03f2eaa56f83245a85ccad1c56d7d98b70eaa046154a1d95f5b79a7fa14'
sys.path.insert(0,str(P/'src'));import execution_control as C
bindings=C.bindings(P);external=[]
for name,f in bindings.items():
 p=Path(name);assert sha(p)==f['sha256'] and p.stat().st_size==f['bytes'];external.append(row(p))
assert len(external)==61
models=[]
for f in read(P/'analysis/MODEL_TOKENIZER_FRESH_SHA.json')['files']:
 s=Path(f['path']).stat();assert (s.st_size,s.st_ino,s.st_mtime_ns)==(f['bytes'],f['inode'],f['mtime_ns']);models.append(f)
assert len(models)==17
gpu=C.query_gpu();proc=[]
for f in (P/'formal').glob('*/DISPATCH.json'):
 d=read(f);s=f.parent/'STARTED.json'
 if s.exists():
  ident=read(s).get('identity');
  if ident and C.same_process(ident):proc.append(dict(unit=f.parent.name,identity=ident,dispatch=d,progress=read(f.parent/'PROGRESS.json') if (f.parent/'PROGRESS.json').exists() else None))
copy_names=['CANONICAL_UH_MATRICES_AND_SHA.json','FP32_M0_M3_NUMERICAL_ATTRIBUTION.json','V2_NUMERICAL_QUALIFICATION_CONTRACT.json','FP64_MATH_QUALIFICATION.json','COVARIANCE_ASSET_COMPLETENESS.json','QUERY_HEAD_SHARING_AND_MAPPING.json','SOURCE_MODEL_RUNTIME_IDENTITY.json','QUESTION_GPU_ASSIGNMENT.json','FORMAL_JOB_MANIFEST.json','AIME26_DATASET_FREEZE.json','AIME26_SEED1_PROTOCOL.json','AIME26_SEED1_PROTOCOL_FREEZE.json','SOURCE_FREEZE.json','GPU_PERMIT.json','RESTART_AUTHORIZATION.json','REPAIR_CHANGES_AND_INVARIANTS.json','reuse/QUERY_SECOND_MOMENT_AUDIT.json','reuse/QUERY_CALIBRATION_DATASET_FREEZE.json','reuse/V1_FIRST_FAILURE.json','reuse/V1_build_eigenbasis_cpu.py','reuse/V1_EIGENBASIS_NUMERICAL_CONTRACT.json','src/rotation_adapter.py','src/gpu_rotation_qualification.py','src/runtime_engine.py','src/rotation_codec.py','inherited/group32/src/canonical_h128.py','inherited/group32/src/group_codecs.py','inherited/group32/src/persistent_adapter.py','inherited/group32/src/inherited_device_guards.py','calibration/doc_05_tokens.json']
copies=[cp(P/n,'reuse/'+n) for n in copy_names]
asset_names=['reuse/QUERY_SECOND_MOMENTS_FP64.npz']+[f['path'] for f in read(P/'CANONICAL_UH_MATRICES_AND_SHA.json')['layers']]
assets=[cp(P/n,'assets/'+n) for n in asset_names]
failure_files=[]
for p in (P/'execution/GPU2_FP_EQ').iterdir():
 if p.is_file():failure_files.append(cp(p,'evidence/token22/'+p.name))
for g in [1,3]:
 for name in ['DISPATCH.json','STARTED.json','PROGRESS.json','RESOURCE_INTERVAL.json','worker.log']:
  p=P/'execution'/('GPU'+str(g)+'_FP_EQ')/name
  if p.exists():failure_files.append(cp(p,'evidence/FP_GPU'+str(g)+'/'+name))
past=[]
for label,p in [('V1',V1),('V2_RAM_STOP',V2)]:
 for name in ['ROOT.json','FIRST_FAILURE.json','FINALIZER_FAILURE.json','FINAL_ARCHIVE_VERIFICATION.json','FINAL_SEAL_RECEIPT.json','FINAL_SEAL.json','SHA256_MANIFEST.json','SESSION_RESULT.json']:
  f=p/name
  if f.exists():past.append(cp(f,'evidence/'+label+'/'+name))
for name in ['operations/PRIOR_RAM_ARCHIVE_VERIFICATION.json','operations/PRIOR_RAM_RELEASE_AFTER_DURABLE_ARCHIVE.json','operations/PRIOR_STOP_DELIVERY_PROVENANCE.json']:
 f=P/name
 if f.exists():past.append(cp(f,'evidence/prior_preservation/'+f.name))
native=Path('/data/zypan/transformers-qwen35/src/transformers/models/qwen3_5/modeling_qwen3_5.py')
copies.append(cp(native,'reuse/actual_modeling_qwen3_5.py'))
copies.append(cp(Path('/data/zypan/modelscope_models/Qwen3.5-9B/config.json'),'reuse/actual_model_config.json'))
source=native.read_text();tree=ast.parse(source)
recurrent=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='torch_recurrent_gated_delta_rule')
chunk=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='torch_chunk_gated_delta_rule')
klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Qwen3_5GatedDeltaNet')
forward=next(n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
config=read(R/'reuse/actual_model_config.json')['text_config']
assert (config['linear_num_key_heads'],config['linear_num_value_heads'],config['linear_key_head_dim'],config['linear_value_head_dim'])==(16,32,128,128)
assert 'repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)' in ast.get_source_segment(source,forward)
put('analysis/ACTUAL_NATIVE_SOURCE_SEMANTICS.json',dict(status='PASS_STATIC_BOUND_SOURCE_AUDIT',architecture=row(native),actual_functions={n.name:dict(line=n.lineno,sha256=hashlib.sha256(ast.get_source_segment(source,n).encode()).hexdigest()) for n in [recurrent,chunk,forward]},config=config,normalization='BF16 normalize before FP32 cast; same original native l2norm expression',injection='after causal convolution and SiLU and repeat_interleave and before GDN arithmetic; no rotation through nonlinear convolution',alpha='exp(g.float()), g=-exp(A_log.float())*softplus(a.float()+dt_bias), head scalar',beta='sigmoid(b), head scalar',state='batch,32,Key128,Value128; recurrent native FP32',readout='state Key-axis reduction with query divided by sqrt128',single_token_prefill='chunk helper first token, padding64; recurrent helper later; both must be independently checked',fused_kernel='Runtime uses bound torch helpers, not an assumed FLA fused equivalence',rotation_dtype='normalize in original input dtype, castFP32 then matmul A32; core cast back to original value dtypeBF16',matrix_orientation='column kA=A.T k; stored row k transforms row@A; stateA=A.T S; backprojection A stateA',GPU_dispatches=0))
put('HEAD_MAPPING_AND_U_SHARING.json',dict(status='PASS_ACTUAL_SOURCE_CONFIG_AND_ARRAY_BINDINGS_PENDING_CPU_ASSET_SCAN',unique_heads=16,recurrent_heads=32,mapping=[i//2 for i in range(32)],source_function='Qwen3_5GatedDeltaNet.forward',repeat_interleave_factor=2,samples_per_unique_head=4096,duplicate_shared_Query_independent_samples=False,all_layers=24,source_sha256=sha(native)))
put('QUERY_CQ_ASSET_COMPLETENESS.json',dict(status='PASS_BOUND_COMPLETE_COPIES_NO_NEW_QUERY',copies=assets,samples_per_unique_head=4096,unique_heads_per_layer=16,recurrent_heads=32,new_Query_forwards=0,U_refit=0))
put('V1_V2_FAILURE_IDENTITY_AND_REUSE.json',dict(status='HISTORICAL_STOPS_IMMUTABLE_NO_RELABEL',prior=past,token22=failure_files,source_copies=copies,original_V1_status='QUERY_EIGENBASIS_QUALIFICATION_FAIL_CLOSED_STOP',old_V2_RAM_STOP_new_AIME=0,current_C1_S3_FP_STOP_new_AIME=0,P1_continues=True))
put('P1_RUNNING_TASK_IMMUTABILITY_AUDIT.json',dict(status='PASS_READONLY_ALL622_SOURCE61_EXTERNAL17_MODELSTAT',parent_root=str(P),source_freeze=row(P/'SOURCE_FREEZE.json'),source_files=before,external_dependencies=external,model_stat=models,active_processes=proc,observed_GPU=gpu,monotonic=time.monotonic(),parent_writes=0,new_GPU_dispatches=0,no_P1_raw_read_for_accuracy=True))
print(json.dumps(dict(status='PASS_CPU_INPUT_COLLECTION',copied_assets=len(assets),copied_failure_files=len(failure_files),P1_active=len(proc),source_files=len(before),external_dependencies=len(external),model_stat=len(models),GPU_dispatches=0)),flush=True)
