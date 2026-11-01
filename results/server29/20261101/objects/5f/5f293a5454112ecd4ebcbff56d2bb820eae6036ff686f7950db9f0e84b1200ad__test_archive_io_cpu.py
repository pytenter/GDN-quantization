#!/usr/bin/env python3
"""Independent synthetic archive/byte-I/O tests. No SSH, model or GPU calls."""
import ast,hashlib,io,json,shlex,struct,sys,tempfile,time
from pathlib import Path
sys.dont_write_bytecode=True
import archive_io as A
import cublas_compare as C
ROOT=Path(__file__).resolve().parent.parent
checks=[]
def check(name,fn):
 fn();checks.append(dict(name=name,status='PASS'))
def expect_fail(fn):
 try:fn()
 except (RuntimeError,OSError,ValueError):return
 raise AssertionError('Failure was required')
class FakeTransport:
 data={};reads=0;writes=0
 def __init__(self,c):self.c=c
 def put_immutable(self,path,chunks,n,wanted):
  if path in self.data:raise FileExistsError(path)
  raw=b''.join(chunks);assert len(raw)==n and A.digest(raw)==wanted
  self.data[path]=raw;type(self).writes+=1
  # Independent read, not trusting put parameters.
  reread=bytes(bytearray(self.data[path]));return dict(path=path,bytes=len(reread),sha256=A.digest(reread),independent_reread=True,atomic_no_clobber_published=True)
 def open_verified(self,path,n,h):
  raw=self.data[path];A.require(len(raw)==n and A.digest(raw)==h,'Synthetic archive identity changed');type(self).reads+=1;return io.BytesIO(raw)
 def preflight(self):return dict(status='PASS',available_bytes=2*1024**4,required_quota_bytes=self.c['required_quota_bytes'])
 def verify_objects(self,payload):
  out=[]
  for x in payload['objects']:
   raw=self.data[x['path']];assert len(raw)==x['bytes'] and A.digest(raw)==x['sha256'];out.append(dict(path=x['path'],bytes=len(raw),sha256=A.digest(raw),fresh_independent_stream_SHA=True))
  return dict(status='PASS_FRESH_REMOTE_RAW_SHA_BYTES',objects=out)
def config(root,card='gpu0'):
 v=dict(task_id=C.TASK,execution_run_id='QWEN_GDN_FAST_RUNTIME_QUALIFICATION_REPAIR_20261019_V1',host='zypan@202.38.247.29',control_path='/tmp/qwen_replication_archive29.sock',archive_root='/synthetic/archive/RUN',card_key=card,required_quota_bytes=200*1024**3,global_reservation_bytes=800*1024**3,global_headroom_bytes=100*1024**3,remote_python='python3',io_timeout_seconds=30)
 (root/'ARCHIVE_CONFIG.json').write_text(json.dumps(v));return v
def tensors(t,bad=False):return [('logits','torch.float32',[1],[1],struct.pack('<f',float(t)+(1 if bad else 0))),('state','torch.bfloat16',[1],[1],struct.pack('<H',t%2**16))]
def synthetic_library_receipts(directory):
 identities=[dict(path='/synthetic/libtorch_cpu.so',bytes=3,sha256=C.sha_bytes(b'CPU'),role='torch')];digest=C.sha_bytes(C.canonical(identities));out=dict(runtime_library_identity=identities,runtime_library_digest=digest)
 for stage,key in [('PREFLIGHT','runtime_library_preflight'),('POSTFLIGHT','runtime_library_postflight')]:
  mp=directory/'library_identity'/(stage+'_MAPS.txt');mp.parent.mkdir(parents=True,exist_ok=True);mp.write_bytes(b'SYNTHETIC_CPU_FIXTURE_NOT_REAL_PROCESS_MAPS\n')
  mr=dict(path=str(mp.relative_to(directory)),bytes=mp.stat().st_size,sha256=C.sha_bytes(mp.read_bytes()))
  report=dict(schema='QWEN_LOADED_LIBRARY_IDENTITY_V2',stage=stage,status='PASS',identities=identities,identity_digest=digest,maps_receipt=mr,maps_final_receipt=mr,issues=[],missing_bindings=[],mismatches=[])
  ip=mp.with_name(stage+'_IDENTITY.json');C.atomic_new(ip,report);out[key]=dict(report,identity_receipt=dict(path=str(ip.relative_to(directory)),bytes=ip.stat().st_size,sha256=C.sha_bytes(ip.read_bytes())))
 return out
oldfactory=A.TRANSPORT_FACTORY;A.TRANSPORT_FACTORY=FakeTransport
try:
 for name in ['_PUT','_GET','_PREFLIGHT','_FINAL_VERIFY']:compile(getattr(A,name),name,'exec')
 checks.append(dict(name='all_exact_embedded_remote_scripts_compile_without_execution',status='PASS'))
 with tempfile.TemporaryDirectory() as d:
  root=Path(d);config(root);units=[];ident={k:'SYNTHETIC' for k in ['contract_sha256','source_manifest_sha256','preregistration_sha256','fixture_canonical_sha256','GPU_UUID','runtime_library_digest','runtime_override_env_digest','authorization_sha256']}
  for j,(condition,repeat) in enumerate([('B0',0),('B0',1),('B1',0),('B1',1)]):
   name=f'{condition}_FP_Q{repeat}';directory=root/'execution'/name;ref=None if j==0 else root/'execution/B0_FP_Q0/numeric';store=C.SnapshotStore(directory/'numeric',ref)
   for t in range(256):assert store.record(t,tensors(t),dict(token=t,input_id=t,codec='FP'))['byte_equal']
   ix=store.finish();ip=directory/'numeric/NUMERICAL_INDEX.json';proc=dict(pid=100+j,starttick=1000+j);C.atomic_new(directory/'STARTED.json',dict(process_identity=proc));identity=dict(ident);identity.update(synthetic_library_receipts(directory));C.atomic_new(directory/'RESULT.json',dict(status='PASS',mode='qualify',consumed_tokens=256,native_core_calls=6144,next_token_consumption_checks=6120,all_tokens_finite=True,strict_byte_pass=True,unit_id=name,condition=condition,codec='FP',repeat=repeat,workspace={'B0':':4096:8','B1':':16:8'}[condition],process_identity=proc,seed=20261015,numerical_index=dict(sha256=C.sha_bytes(ip.read_bytes()),bytes=ip.stat().st_size),**identity));units.append(directory)
  check('all256_four_fresh_raw_remote_no_local_large_reference',lambda: (None if not list(root.rglob('*.bin')) and len(FakeTransport.data)==256 else (_ for _ in ()).throw(AssertionError())))
  check('original_qualification_reducer_with_fresh_raw_reverification',lambda: (None if C.reduce_qualification(root)['status']=='PASS' else (_ for _ in ()).throw(AssertionError())))
  assert FakeTransport.reads==3*256+256
  check('every_comparison_and_final_reducer_reads_actual_remote_bytes',lambda: (None if FakeTransport.reads==1024 else (_ for _ in ()).throw(AssertionError())))
  pre=A.archive_preflight(root,card='gpu0');assert pre['status']=='PASS';expect_fail(lambda:A.archive_preflight(root,card='gpu1'));checks.append(dict(name='archive_config_card_quota_and_auth_identity_metadata',status='PASS'))
  bad=C.SnapshotStore(root/'execution/B1_FP_BAD/numeric',root/'execution/B0_FP_Q0/numeric');item=bad.record(0,tensors(0,True),dict(token=0,input_id=0,codec='FP'));assert not item['byte_equal'] and 'failure_raw_archive' in item and not list((root/'execution/B1_FP_BAD').rglob('*.bin'))
  diff=C.compare_failure(root/'execution/B0_FP_Q0/numeric/token_000.json',root/'execution/B1_FP_BAD/numeric/token_000.json');assert diff['differences'][0]['result']['max_abs_error']==1.0;checks.append(dict(name='first_difference_raw_preserved_remote_and_same_numeric_diff',status='PASS'))
  objects=A.enumerate_archive_objects(root);assert len(objects)==257 and A.final_verify_archive(root,False)['objects']==257;checks.append(dict(name='metadata_catalog_and_final_fresh_remote_all_raw_verification',status='PASS'))
  refpath=objects[0]['path'];saved=FakeTransport.data[refpath];FakeTransport.data[refpath]=saved+b'X';expect_fail(lambda:C.reduce_qualification(root));FakeTransport.data[refpath]=saved;checks.append(dict(name='archive_trailing_or_changed_raw_fails_closed',status='PASS'))
  p=root/'execution/B0_FP_Q0/numeric/token_000.json';v=json.loads(p.read_text());v['tensors'][0]['sha256']='0'*64;p.write_text(json.dumps(v));expect_fail(lambda:C.SnapshotStore(root/'execution/B1_FP_TAMPER/numeric',p.parent).record(0,tensors(0),dict(token=0,input_id=0,codec='FP')));checks.append(dict(name='per_tensor_SHA_rechecked_before_any_actual_comparison',status='PASS'))
 # Test the exact remote scripts/stream implementation on local isolated CPU files.
 class LocalScriptTransport(A.SSHArchiveTransport):
  def _command(self,code,args):return [sys.executable,'-B','-c',code]+[str(x) for x in args]
 class StalledLocalPipe(A.SSHArchiveTransport):
  def _command(self,code,args):return [sys.executable,'-B','-c','import time;time.sleep(1)']
 stalled=StalledLocalPipe(dict(global_headroom_bytes=0,io_timeout_seconds=.05));start=time.monotonic();rawstall=b'x'*1048576;expect_fail(lambda:stalled.put_immutable('unused',iter([rawstall]),len(rawstall),A.digest(rawstall)));assert time.monotonic()-start<.8;checks.append(dict(name='stalled_owned_pipe_write_obeys_inactivity_deadline_CPU',status='PASS'))
 with tempfile.TemporaryDirectory() as d:
  t=LocalScriptTransport(dict(global_headroom_bytes=0,io_timeout_seconds=30));path=str(Path(d)/'raw with spaces ;$(literal).bin');raw=b'abc\x00'+bytes(range(256))*8192;receipt=t.put_immutable(path,iter([raw[:3],raw[3:]]),len(raw),A.digest(raw));assert receipt['independent_reread']
  with t.open_verified(path,len(raw),A.digest(raw)) as f:assert f.read(7)+f.read(len(raw)-7)==raw and not f.read(1)
  expect_fail(lambda:t.put_immutable(path,iter([raw]),len(raw),A.digest(raw)));assert Path(path).read_bytes()==raw;checks.append(dict(name='exact_remote_put_get_scripts_atomic_no_clobber_SHA_stream_EOF_CPU',status='PASS'))
  expect_fail(lambda:t.open_verified(path,len(raw),'0'*64));checks.append(dict(name='exact_remote_stream_fresh_identity_mismatch_rejected_CPU',status='PASS'))
 with tempfile.TemporaryDirectory() as d:
  r=Path(d);v=config(r);v['archive_root']=str(r/'remote_archive'/('long_component_'+'x'*180));(r/'ARCHIVE_CONFIG.json').write_text(json.dumps(v));c=A.load_archive_config(r);n=r/'execution/B0_FP_Q0/numeric';n.mkdir(parents=True)
  for i in range(512):
   obj=Path(v['archive_root'])/f'gpu0/execution/B0_FP_Q0/numeric/token_{i:03d}.bin';obj.parent.mkdir(parents=True,exist_ok=True);obj.write_bytes(b'FINAL_SYNTHETIC_RAW'+str(i).encode());receipt=dict(path=str(obj),bytes=obj.stat().st_size,sha256=A.digest(obj.read_bytes()),host=c['host'],card_key=c['card_key'],archive_config_sha256=c['config_sha256']);(n/f'token_{i:03d}.json').write_text(json.dumps(dict(raw_archive=receipt)))
  payload_size=len(json.dumps(A.enumerate_archive_objects(r)).encode());assert payload_size>131072
  A.TRANSPORT_FACTORY=LocalScriptTransport;result=A.final_verify_archive(r,True);cat=result['cards'][0]['catalog'];assert result['objects']==512 and A.digest(Path(cat['path']).read_bytes())==cat['sha256'];expect_fail(lambda:A.final_verify_archive(r,True));assert A.final_verify_archive(r,False)['objects']==512;A.TRANSPORT_FACTORY=FakeTransport;checks.append(dict(name='exact_remote_final_512_longpath_objects_stdin_exceeds128KiB_SHA_atomic_catalog_CPU',status='PASS',synthetic_catalog_input_bytes=payload_size))
  transport=A.SSHArchiveTransport(c);literal='/safe/path with spaces;$(not_executed)';assert shlex.split(transport._command(A._GET,[literal])[-1])[-1]==literal;checks.append(dict(name='SSH_command_arguments_shell_quoted_without_execution',status='PASS'))
 # Confirm numerical difference/reducer functions were not rewritten as I/O work.
 original=Path('/Users/pytenter/Documents/临时项目1/QWEN_GDN_FAST_RUNTIME_QUALIFICATION_V1/scripts/cublas_compare.py')
 class StripIdentityGate(ast.NodeTransformer):
  def visit_Expr(self,node):
   if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='verify_library_gates':return None
   return self.generic_visit(node)
 orig={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(original.read_text()).body if isinstance(n,ast.FunctionDef)};new={n.name:ast.dump(StripIdentityGate().visit(n),include_attributes=False) for n in ast.parse(Path(C.__file__).read_text()).body if isinstance(n,ast.FunctionDef)}
 assert all(orig[k]==new[k] for k in ['numeric_values','difference','reduce_performance']);checks.append(dict(name='numeric_values_difference_performance_reducer_AST_unchanged',status='PASS'))
 assert 'torch' not in sys.modules;checks.append(dict(name='no_torch_GPU_model_or_SSH_used_in_CPU_tests',status='PASS'))
finally:A.TRANSPORT_FACTORY=oldfactory
out=dict(status='PASS_CPU_SYNTHETIC_ARCHIVE_IO_NOT_REMOTE_GPU_QUALIFICATION',checks=checks,checks_count=len(checks),GPU_calls=0,SSH_calls=0,model_forward=0,large_data_downloaded=False,numerical_examples='synthetic struct-packed scalars only',source_sha256={p.name:A.digest(p.read_bytes()) for p in [Path(A.__file__),Path(C.__file__),Path(__file__)]})
(ROOT/'analysis/ARCHIVE_IO_CPU_TEST_RESULTS.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
