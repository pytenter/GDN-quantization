#!/usr/bin/env python3
"""Synthetic stdlib control/byte tests only; never invokes a model/torch/CUDA."""
import ast,builtins,hashlib,json,os,struct,symtable,sys,tempfile
from pathlib import Path
R=Path(__file__).resolve().parent.parent
from cublas_compare import SnapshotStore,atomic_new,canonical,difference,reduce_qualification,reduce_performance,sha_bytes,read
from cublas_harness import build_stage_schedule,fresh_environment,build_command,numeric_group_verdict,performance_verdict
import cublas_worker as W
checks=[]
def test(name,fn):
 try:fn();checks.append(dict(name=name,status='PASS'))
 except Exception as e:checks.append(dict(name=name,status='FAIL',error=str(e)));raise

def deny():
 prior=os.environ.get('ALLOW_GPU_DIAG');os.environ['ALLOW_GPU_DIAG']='NO'
 try:
  try:W.admission(R,type('Args',(),{})())
  except RuntimeError as e:assert 'GPU_NOT_AUTHORIZED' in str(e)
  else:raise AssertionError('Default gate admitted')
 finally:
  if prior is None:os.environ.pop('ALLOW_GPU_DIAG',None)
  else:os.environ['ALLOW_GPU_DIAG']=prior
 assert 'torch' not in sys.modules and not any(x.startswith('transformers') for x in sys.modules)
test('default_NO_denies_before_any_engine_import',deny)

def controller_gate():
 with tempfile.TemporaryDirectory(prefix='CPU_CONTROLLER_MOCK_',dir=R/'qualification') as tmp:
  t=Path(tmp);unit=build_stage_schedule()[0];auth=t/'AUTH.json';atomic_new(auth,dict(role='SYNTHETIC_ONLY'));atomic_new(t/'RUN_COPY_ORIGIN.json',dict(role='SYNTHETIC_ONLY'))
  argv=['/fake/python','-B',str(t/'scripts/run_cublas_controller.py'),'--root',str(t)];digest=W.sha(auth);meta=dict(controller_pid=123,controller_starttick=456,controller_cmdline=argv,authorization_sha256=digest,source_manifest_sha256='SYNTHETIC')
  atomic_new(t/'execution/CONTROLLER_STARTED.json',meta);dp=t/'execution'/unit['unit_id']/'DISPATCH.json';atomic_new(dp,dict(unit=unit,controller_pid=123,controller_starttick=456,authorization_sha256=digest,source_manifest_sha256='SYNTHETIC',budget_snapshot=dict(completed_observed_seconds_upper=0,unit_allowance_seconds=180,protective_seconds=1650,hard_seconds=1800)))
  previous=os.environ.get('QWEN_CONTROLLER_DISPATCH_SHA256');os.environ['QWEN_CONTROLLER_DISPATCH_SHA256']=W.sha(dp)
  try:
   W.controller_admission(t,auth,dict(source_manifest_sha256='SYNTHETIC'),unit,dict(future_execution_root=str(t)),reader=lambda pid:(456,argv),parent_pid=123)
   try:W.controller_admission(t,auth,dict(source_manifest_sha256='SYNTHETIC'),unit,dict(future_execution_root=str(t)),reader=lambda pid:(456,argv),parent_pid=999)
   except RuntimeError:pass
   else:raise AssertionError('Wrong directparent admitted')
   os.environ['QWEN_CONTROLLER_DISPATCH_SHA256']='wrong'
   try:W.controller_admission(t,auth,dict(source_manifest_sha256='SYNTHETIC'),unit,dict(future_execution_root=str(t)),reader=lambda pid:(456,argv),parent_pid=123)
   except RuntimeError:pass
   else:raise AssertionError('Tampered dispatch admitted')
  finally:
   if previous is None:os.environ.pop('QWEN_CONTROLLER_DISPATCH_SHA256',None)
   else:os.environ['QWEN_CONTROLLER_DISPATCH_SHA256']=previous
 test_marker=True
 assert test_marker
test('mock_exact_controller_parent_starttick_dispatch_and_budget_binding',controller_gate)

def override_gates():
 keys=['NVIDIA_TF32_OVERRIDE','TORCH_ALLOW_TF32_CUBLAS_OVERRIDE','CUDA_LAUNCH_BLOCKING'];prior={key:os.environ.get(key) for key in keys};contract=dict(environment=dict(python=sys.version.split()[0]))
 try:
  for key in keys:os.environ.pop(key,None)
  W.ambient_controls(contract)
  for key in keys:
   os.environ[key]='1'
   try:W.ambient_controls(contract)
   except RuntimeError:pass
   else:raise AssertionError('Ambient override admitted:'+key)
   os.environ.pop(key)
  try:W.ambient_controls(dict(environment=dict(python='wrong')))
  except RuntimeError:pass
  else:raise AssertionError('WrongPython admitted')
 finally:
  for key,value in prior.items():
   if value is None:os.environ.pop(key,None)
   else:os.environ[key]=value
test('preengine_Python_and_TF32_async_ambient_override_rejection',override_gates)

def astchange():
 p=R.parent/'QWEN_GDN_RUNTIME_THROUGHPUT_ROOT_CAUSE_V1/evidence/28f10b0f2a7e75dc606b38f63d526a91e4b220d6af98a28e536dfef4f10ebe8f/core.py';s=p.read_text();assert hashlib.sha256(s.encode()).hexdigest()=='aa338b8b83cde62b97c94fd205d1c62999f783e7ec3242cbb813dc877e01f575'
 for value in [':4096:8',':16:8']:
  t=W.transform_core(s,value);assert ast.dump(next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='configure'),include_attributes=False)==ast.dump(next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name=='configure'),include_attributes=False)
 try:W.transform_core(s,':32:8')
 except RuntimeError:pass
 else:raise AssertionError('unregistered workspace accepted')
 try:W.transform_core(s+s.split('import ast,')[0],':16:8')
 except RuntimeError:pass
 else:raise AssertionError('duplicate startup literal accepted')
test('pinned_core_AST_only_one_env_literal_math_configure_unchanged',astchange)

def globals_resolved():
 p=R.parent/'QWEN_GDN_RUNTIME_THROUGHPUT_ROOT_CAUSE_V1/evidence/05184e00b87e909c4bf25106a7e1f56fa64aeb1be5246a08d8217e9f641a6855/baseline_worker.py';s=p.read_text();assert hashlib.sha256(s.encode()).hexdigest()=='0961d96bdafa92d6c3bca7d1980b74e9d70fe3aabff404417878ced20c7f1e41'
 supplied={'K','C','torch','collections','CURRENT','Observe','dual','describe','Path','contextlib','random','time','gc','math','statistics','sys','os','sha','check','WARMUP','TASK','json','replay','summary','profile_rows'}|set(dir(builtins))
 top=symtable.symtable(s,str(p),'exec');found={}
 def walk(t):
  for symbol in t.get_symbols():
   if symbol.is_referenced() and symbol.is_global():found.setdefault(t.get_name(),set()).add(symbol.get_name())
  for child in t.get_children():walk(child)
 for t in top.get_children():
  if t.get_name() in ['replay','summary','profile_rows']:walk(t)
 assert all(v<=supplied for v in found.values()),str({k:v-supplied for k,v in found.items()})
test('original_replay_profile_summary_global_closure_resolved_without_execution',globals_resolved)

def loaded_map_bindings():
 with tempfile.TemporaryDirectory(prefix='CPU_FAKE_MAPS_ONLY_',dir=R/'qualification') as tmp:
  td=Path(tmp);bindings=[];lines=[]
  for name in ['libtorch.so','libc10.so','libcublas.so.12','libcudnn.so.9','libcudart.so.12']:
   p=td/name;p.write_bytes(('CPU synthetic file '+name).encode());bindings.append(dict(path=str(p),resolved_path=str(p),bytes=p.stat().st_size,sha256=W.sha(p)));lines.append('00000000-00001000 r-xp 00000000 00:00 1 '+str(p))
  contract=dict(prospective_library_bindings=bindings+[dict(status='MISSING',directory='/fake',pattern='*.so')]);maps='\n'.join(lines)
  selected=W.loaded_library_identity(contract,maps);assert len(selected)==5 and next(x for x in selected if Path(x['path']).name=='libcudart.so.12')['role']=='CUDA_runtime'
  bad=td/'libtorch_unbound.so';bad.write_bytes(b'fake');extra=maps+'\n00000000-00001000 r-xp 00000000 00:00 1 '+str(bad)
  try:W.loaded_library_identity(contract,extra)
  except RuntimeError:pass
  else:raise AssertionError('unbound selected library accepted')
  (td/'libcublas.so.12').write_bytes(b'corrupt')
  try:W.loaded_library_identity(contract,maps)
  except RuntimeError:pass
  else:raise AssertionError('corrupt selected library accepted')
test('injected_fake_proc_maps_actualpaths_SHA_bytes_prefix_and_unbound_rejection',loaded_map_bindings)

def profile_ast():
 p=R.parent/'QWEN_GDN_RUNTIME_THROUGHPUT_ROOT_CAUSE_V1/evidence/05184e00b87e909c4bf25106a7e1f56fa64aeb1be5246a08d8217e9f641a6855/baseline_worker.py';tree=W.profile_replay_ast(p.read_text());assert tree.body[0].name=='stack_profile_replay';compile(tree,str(p),'exec')
 ks=[]
 for node in ast.walk(tree):
  if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='profile':ks.append({k.arg:k.value.value for k in node.keywords if isinstance(k.value,ast.Constant)})
 assert len(ks)==1 and ks[0]['record_shapes'] is True and ks[0]['with_stack'] is True and ks[0]['profile_memory'] is False
test('profile_copy_changes_instrumentation_only_reverse_AST_exact',profile_ast)

def envschedule():
 ss=build_stage_schedule();assert len(ss)==16 and len({x['unit_id'] for x in ss})==16;assert [u['unit_id'] for u in ss[:4]]==['B0_FP_Q0','B0_FP_Q1','B1_FP_Q0','B1_FP_Q1'];assert [u['unit_id'] for u in ss[4:10]]==['B0_FP_P0','B1_FP_P0','B1_FP_P1','B0_FP_P1','B0_FP_P2','B1_FP_P2']
 for u in ss:
  e=fresh_environment(u['condition'],{});assert e['CUBLAS_WORKSPACE_CONFIG']=={ 'B0':':4096:8','B1':':16:8'}[u['condition']] and e['CUDA_VISIBLE_DEVICES']=='0' and e['OMP_NUM_THREADS']==e['MKL_NUM_THREADS']=='1';cmd=build_command('/fake/python',R,'AUTH.json',u);assert cmd[1]=='-B' and '--unit-id' in cmd
 try:build_command('/fake',R,'AUTH',dict(ss[0],repeat=3))
 except ValueError:pass
 else:raise AssertionError('unknown unit admitted')
test('fixed_fresh_schedule_beforeimport_env_and_unknownunit_denial',envschedule)

def rawdiffs():
 r=difference(struct.pack('<f',0.),struct.pack('<f',-0.),'torch.float32',[1]);assert r['byte_equal'] is False and r['max_abs_error']==0 and r['first_mismatched_byte']==3
 r=difference(struct.pack('<d',0),struct.pack('<d',1e308),'torch.float64',[1]);assert r['max_relative_error']=='Infinity';json.dumps(r,allow_nan=False)
 r=difference(struct.pack('<H',0x3f80),struct.pack('<H',0x4000),'torch.bfloat16',[1]);assert r['max_abs_error']==1
 r=difference(struct.pack('<f',float('nan')),struct.pack('<f',1.),'torch.float32',[1]);assert r['nonfinite_flat_indices_first16']==[0]
test('rawbyte_signedzero_BF16_nonfinite_extreme_relative_diff',rawdiffs)

with tempfile.TemporaryDirectory(prefix='CPU_SYNTHETIC_ONLY_',dir=R/'qualification') as td:
 T=Path(td);ident={k:'SYNTHETIC_CONTROL_TEST_NOT_REAL_GPU' for k in ['contract_sha256','source_manifest_sha256','preregistration_sha256','fixture_canonical_sha256','GPU_UUID','runtime_library_digest','runtime_override_env_digest','authorization_sha256']}
 def populate():
  for j,(condition,repeat) in enumerate([('B0',0),('B0',1),('B1',0),('B1',1)]):
   u=f'{condition}_FP_Q{repeat}';D=T/'execution'/u;reference=None if j==0 else T/'execution/B0_FP_Q0/numeric';store=SnapshotStore(D/'numeric',reference)
   for token in range(256):store.record(token,[('logits','torch.float32',[1],[1],struct.pack('<f',token+.5))],dict(token=token,input_id=token,codec='FP'))
   ix=store.finish();ip=D/'numeric/NUMERICAL_INDEX.json';proc=dict(pid=100+j,starttick=1000+j);atomic_new(D/'STARTED.json',dict(process_identity=proc));atomic_new(D/'RESULT.json',dict(status='PASS',mode='qualify',consumed_tokens=256,native_core_calls=6144,next_token_consumption_checks=6120,all_tokens_finite=True,strict_byte_pass=True,unit_id=u,condition=condition,codec='FP',repeat=repeat,workspace={'B0':':4096:8','B1':':16:8'}[condition],process_identity=proc,seed=20261015,numerical_index=dict(path='numeric/NUMERICAL_INDEX.json',sha256=sha_bytes(ip.read_bytes()),bytes=ip.stat().st_size),**ident))
  assert reduce_qualification(T)['status']=='PASS';assert numeric_group_verdict(T,'FP')['status']=='PASS'
 test('synthetic_full256_four_fresh_actualbyte_and_transitive_determinism',populate)
 def badbyte():
  store=SnapshotStore(T/'bad',T/'execution/B0_FP_Q0/numeric');x=store.record(0,[('logits','torch.float32',[1],[1],struct.pack('<f',1))],dict(token=0,input_id=0,codec='FP'));assert not x['byte_equal'];assert (T/'bad/token_000.bin').is_file()
  try:store.finish()
  except RuntimeError:pass
  else:raise AssertionError('partial/mismatched accepted')
  try:store.record(0,[],{})
  except RuntimeError:pass
  else:raise AssertionError('duplicate cursor accepted')
 test('first_mismatch_retained_incomplete_and_duplicate_denied',badbyte)
 def malformed():
  fp=T/'execution/B1_FP_Q1/numeric/token_255.json';original=fp.read_bytes();fp.write_bytes(original+b' ')
  try:reduce_qualification(T)
  except RuntimeError:pass
  else:raise AssertionError('tampered receipt accepted')
  fp.write_bytes(original)
 test('token_receipt_tamper_fails_before_performance',malformed)
 def badraw():
  fp=T/'execution/B0_FP_Q0/numeric/token_255.bin';original=fp.read_bytes();fp.write_bytes(b'bad!')
  try:reduce_qualification(T)
  except RuntimeError:pass
  else:raise AssertionError('tampered fullreference raw accepted')
  fp.write_bytes(original)
 test('reference_raw_tensor_bytes_SHA_independently_verified',badraw)

 def perf():
  for repeat in range(3):
   for j,c in enumerate(['B0','B1']):
    u=f'{c}_FP_P{repeat}';D=T/'execution'/u;tps=10+.01*repeat if c=='B0' else 11+.01*repeat;proc=dict(pid=200+2*repeat+j,starttick=2000+2*repeat+j);atomic_new(D/'STARTED.json',dict(process_identity=proc));atomic_new(D/'RESULT.json',dict(status='PASS',unit_id=u,mode='benchmark',condition=c,repeat=repeat,codec='FP',workspace={'B0':':4096:8','B1':':16:8'}[c],seed=20261015,process_identity=proc,tokens_per_second=tps,decode_wall_seconds=128/tps,whole_process_wall_seconds=30,measured_tokens=128,warmup_tokens=128,benchmark_reference_result=dict(instrumented=False,consumed_tokens=256,native_core_calls=6144,QDQ_calls=0,condition='FP_NATIVE'),**ident))
  assert performance_verdict(T)['status']=='PASS_BENEFIT'
 test('synthetic_six_performance_identity_and_5percent_nonoverlap',perf)
 def badperf():
  pp=T/'execution/B1_FP_P1/RESULT.json';orig=pp.read_bytes();x=json.loads(orig);x['tokens_per_second']=float('nan');pp.write_text(json.dumps(x));assert performance_verdict(T)['status']=='FAIL';pp.write_bytes(orig);x=json.loads(orig);x['source_manifest_sha256']='wrong';pp.write_text(json.dumps(x));assert performance_verdict(T)['status']=='FAIL';pp.write_bytes(orig)
 test('performance_nonfinite_wrongsource_rejected',badperf)
assert 'torch' not in sys.modules
out=dict(status='PASS_CPU_SYNTHETIC_CONTROLS_ONLY',GPU='NOT_RUN_NO_AUTHORIZATION',model_forward='NOT_RUN',torch_imported=False,checks=checks,numeric_examples_are='Synthetic struct-packed scalars only; not model/codec evidence')
target=R/'qualification/CUBLAS_CPU_TEST_RESULTS.json'
if target.exists():
 previous=target.read_bytes();archive=R/'qualification'/('CUBLAS_CPU_TEST_RESULTS_PREVIOUS_'+hashlib.sha256(previous).hexdigest()+'.json')
 if not archive.exists():archive.write_bytes(previous)
 target.unlink()
atomic_new(target,out);print(json.dumps(out,indent=2))
