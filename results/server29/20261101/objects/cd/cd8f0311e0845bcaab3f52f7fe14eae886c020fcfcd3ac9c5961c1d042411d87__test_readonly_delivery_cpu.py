#!/usr/bin/env python3
"""Isolated stdlib fixture tests only: no SSH, NVML, Torch or GPU execution."""
from pathlib import Path
import argparse,base64,copy,hashlib,importlib.util,json,os,subprocess,sys,tarfile,tempfile,time,io
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_readonly_delivery_tested',ROOT/'operations/download_verified_delivery.py');D=importlib.util.module_from_spec(spec);spec.loader.exec_module(D)
CHECKS=[]
def check(n,ok):
 if not ok:raise AssertionError(n)
 CHECKS.append(dict(name=n,status='PASS'))
def rejects(n,fn):
 try:fn()
 except(Exception,AssertionError):check(n,True)
 else:raise AssertionError('Expected rejection:'+n)
def put(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(D.canonical(v))
def metadata(r,n):p=r/n;return dict(path=n,bytes=p.stat().st_size,sha256=D.sha(p))
def execute(code):
 return subprocess.run([sys.executable,'-B','-c',code],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),capture_output=True)
def must_json(code):
 p=execute(code);check('fixture_remote_program_return0',p.returncode==0)
 return json.loads(p.stdout)
def fixture(r):
 (r/'src').mkdir(parents=True);(r/'assets').mkdir();(r/'analysis').mkdir()
 (r/'src/cpu_source.py').write_text('# frozen CPU fixture only\n')
 (r/'assets/small_tensor.bin').write_bytes(bytes(range(64)))
 put(r/'analysis/EXTERNAL_BINDINGS.json',dict(files=[]))
 native_rows=[]
 external=r.parent/(r.name+'_CPU_FIXTURE_ELF_ONLY');external.mkdir()
 for i in range(3):
  p=external/('fixture_%d.so'%i);p.write_bytes(b'\x7fELFCPU_FIXTURE_NO_EXECUTION'+bytes([i]));b=p.stat()
  native_rows.append(dict(path=str(p.resolve()),bytes=b.st_size,inode=b.st_ino,device=b.st_dev,mtime_ns=b.st_mtime_ns,sha256=D.sha(p),provenance={'kind':'ISOLATED_CPU_BYTES_NEVER_IMPORTED'}))
 probes=[]
 for i in range(2):
  name='operations/CPU_PROBE_%d.json'%i;ident=dict(pid=99999970+i,starttick=123+i,cmdline='CPU_FIXTURE_ONLY_NEVER_EXECUTED_%d'%i)
  put(r/name,dict(status='PASS_FRESH_CPU_IMPORTS_AND_PROVENANCED_NATIVE_CLOSURE_NO_GPU',identity=ident,CUDA_VISIBLE_DEVICES='',Torch_CUDA_initialized=False,CPU_PID_absent_from_NVML=True,
   GPU_contexts_created=0,GPU_dispatches=0,model_instances_created=0,Tensor_forwards=0,files=native_rows,count=len(native_rows),bytes=sum(x['bytes']for x in native_rows),base_external_bindings_sha256=D.sha(r/'analysis/EXTERNAL_BINDINGS.json')))
  probes.append(dict(metadata(r,name),identity=ident,CPU_only=True,CUDA_initialized=False,CPU_PID_absent_from_NVML=True))
 put(r/'analysis/RUNTIME_NATIVE_CLOSURE.json',dict(schema_version=1,status='PASS_TWO_FRESH_CPU_IMPORTS_AND_PROVENANCED_NATIVE_CLOSURE_NO_GPU',
  base_external_bindings_sha256=D.sha(r/'analysis/EXTERNAL_BINDINGS.json'),files=native_rows,count=len(native_rows),bytes=sum(x['bytes']for x in native_rows),cpu_probe_receipts=probes,aliases=[],loader_only_extension=None))
 source=[metadata(r,n)for n in('src/cpu_source.py','assets/small_tensor.bin','analysis/EXTERNAL_BINDINGS.json','analysis/RUNTIME_NATIVE_CLOSURE.json')]
 put(r/'SOURCE_FREEZE.json',dict(files=source));s=D.sha(r/'SOURCE_FREEZE.json')
 identity=dict(pid=99999988,starttick=123,cmdline=['CPU_FIXTURE_ONLY'])
 put(r/'CONTROLLER_STARTED.json',dict(identity=identity))
 put(r/'FINISHER_STARTED.json',dict(identity=dict(identity,pid=99999989)))
 put(r/'SESSION_RESULT.json',dict(status='FAIL_CLOSED_STOP',formal_completed=0,formal_receipts=[],own_release_all_PASS=True,all_own_processes_released=True))
 put(r/'TASK_RESULT.json',dict(status='FAIL_CLOSED_STOP',actual_formal_completed=0))
 rows=[metadata(r,str(p.relative_to(r)))for p in sorted(r.rglob('*'))if p.is_file()]
 m=dict(task_id=D.TASK,execution_attempt=r.name,terminal='FAIL_CLOSED_STOP',formal_completed=0,source_freeze_sha256=s,count=len(rows),bytes=sum(x['bytes']for x in rows),files=rows,
   raw_remote_only_files=[x for x in rows if x['path'].endswith('.bin')],raw_remote_only_count=1,
   excluded_exact_relative_paths=sorted(set(D.CONTROLS)|{'FINISHER_RESULT.json','FINISHER_FAILURE.json','operations/ATOMIC_ARCHIVE_PUBLICATION.json'}),excluded_live_descriptors={},finisher_identity=dict(identity,pid=99999989))
 put(r/'SHA256_MANIFEST.json',m);(r/'SHA256SUMS').write_bytes(''.join(x['sha256']+'  '+x['path']+'\n'for x in rows).encode())
 v=dict(status='PASS_INDEPENDENT_ARCHIVE29_EVERY_FILE_FRESH_SHA_AND_BYTES',task_id=D.TASK,terminal='FAIL_CLOSED_STOP',count=len(rows),bytes=m['bytes'],files=rows,source_freeze_sha256=s,manifest_sha256=D.sha(r/'SHA256_MANIFEST.json'))
 put(r/'FINAL_ARCHIVE_VERIFICATION.json',v)
 seal=dict(status='PASS_ATOMIC_NOREPLACE_SEALED_SCIENCE',task_id=D.TASK,execution_attempt=r.name,terminal='FAIL_CLOSED_STOP',formal_completed=0,planned_formal=60,source_freeze_sha256=s,archive_root=str(r),atomic='renameat2_RENAME_NOREPLACE',manifest_sha256=D.sha(r/'SHA256_MANIFEST.json'),SHA256SUMS_sha256=D.sha(r/'SHA256SUMS'),archive_verification_sha256=D.sha(r/'FINAL_ARCHIVE_VERIFICATION.json'))
 put(r/'FINAL_SEAL_RECEIPT.json',seal)
 return dict(remote_root=str(r),archive_root=str(r),remote_python=sys.executable,ssh_control_path='CPU_TEST_NO_SSH'),s,m

def helper(r,code):
 return D.REMOTE_HELPERS+'\nr=Path('+repr(str(r))+');remote_root=r\n'+code

def native_tests(base,r):
 code="m,c=native_contract(r);f=fresh_native_files(r,m,c,{});print(json.dumps(f))"
 positive=must_json(helper(r,code));check('dynamic_full_native_count_3_not_293',positive['count']==3 and len(positive['files'])==3)
 check('all_actual_native_rows_have_fresh_SHA_stat_ELF',all(x['ELF_magic']=='7f454c46'and x['inode']>0 and x['device']>=0 for x in positive['files']))
 closure=r/'analysis/RUNTIME_NATIVE_CLOSURE.json';original=closure.read_bytes();model=json.loads(original)
 def reject_model(name,change):
  altered=copy.deepcopy(model);change(altered);put(closure,altered)
  check(name,execute(helper(r,code)).returncode!=0);closure.write_bytes(original)
 reject_model('native_manifest_duplicate_exact_path_rejected',lambda x:x['files'].append(x['files'][0]))
 reject_model('native_manifest_count_mismatch_rejected',lambda x:x.update(count=293))
 reject_model('native_manifest_missing_original61_binding_rejected',lambda x:x.update(base_external_bindings_sha256='0'*64))
 reject_model('native_manifest_stat_mismatch_rejected',lambda x:x['files'][0].update(inode=x['files'][0]['inode']+1))
 reject_model('native_manifest_SHA_mismatch_rejected',lambda x:x['files'][0].update(sha256='0'*64))
 reject_model('native_manifest_empty_provenance_rejected',lambda x:x['files'][0].update(provenance={}))
 reject_model('native_manifest_CPU_probe_context_rejected',lambda x:x['cpu_probe_receipts'][0].update(CUDA_initialized=True))
 p=Path(model['files'][0]['path']);old=p.read_bytes();before=p.stat()
 p.write_bytes(b'NOT_ELF'+old[7:]);os.utime(p,ns=(before.st_atime_ns,before.st_mtime_ns))
 altered=copy.deepcopy(model);altered['files'][0]['sha256']=D.sha(p);put(closure,altered)
 check('non_ELF_native_row_rejected_even_with_matching_SHA',execute(helper(r,code)).returncode!=0)
 p.write_bytes(old);os.utime(p,ns=(before.st_atime_ns,before.st_mtime_ns));closure.write_bytes(original)
 # A complete frozen row added before dispatch is checked dynamically, no293 constant.
 extra=p.parent/'fourth_cpu_fixture.so';extra.write_bytes(b'\x7fELFDYNAMIC_ADDITIONAL_CPU_FIXTURE');b=extra.stat()
 added=dict(path=str(extra.resolve()),bytes=b.st_size,inode=b.st_ino,device=b.st_dev,mtime_ns=b.st_mtime_ns,sha256=D.sha(extra),provenance={'kind':'CPU_FIXTURE'})
 altered=copy.deepcopy(model);altered['files'].append(added);altered['count']+=1;altered['bytes']+=b.st_size;put(closure,altered)
 check('native_addition_without_reviewed_extension_rejected',execute(helper(r,code)).returncode!=0)
 receipt_name='operations/EXTENSION_PROBE_FIXTURE.json';extension_name='operations/EXTENSION_REVIEW_FIXTURE.json'
 put(r/receipt_name,dict(identity=dict(pid=99999996,starttick=300,cmdline=['CPU_FIXTURE_NO_GPU']),fixture_only=True))
 receipt=dict(metadata(r,receipt_name),identity=json.loads((r/receipt_name).read_text())['identity'])
 put(r/extension_name,dict(status='PASS_EXACT_LOADER_PROBE_ELF_PROVENANCE_CPU_REVIEW',base_external_bindings_sha256=model['base_external_bindings_sha256'],files=[added],probe_receipt=receipt))
 altered['loader_only_extension']=dict(path=extension_name,sha256=D.sha(r/extension_name),probe_receipt=receipt,additional_file_count=1);put(closure,altered)
 expanded=must_json(helper(r,code));check('extended_native_manifest_all_4_rows_fresh_verified',expanded['count']==4 and expanded['files'][-1]['path']==str(extra))
 closure.write_bytes(original);(r/receipt_name).unlink();(r/extension_name).unlink()
 check('same_PID_starttick_cannot_be_false_exit_due_to_argv_type',execute(helper(r,"expected=dict(pid=1,starttick=1,cmdline='STRING');identity=lambda pid:dict(pid=1,starttick=1,cmdline=['DIFFERENT']);exited_probe(expected)" )).returncode!=0)

def probe_fixture(r):
 worker=dict(pid=99999992,starttick=456,cmdline=['CPU_FIXTURE_ENGINEERING_WORKER'])
 supervisor=dict(pid=99999993,starttick=457,cmdline=['CPU_FIXTURE_ENGINEERING_SUPERVISOR'])
 put(r/'HUMAN_REPAIR_AUTHORIZATION.json',dict(fixture=True))
 frozen=dict(status='SEPARATELY_PINNED_ENGINEERING_LOADER_PROBE_NO_SCIENTIFIC_QUALIFICATION',files=[],count=0,model_forward_calls_allowed=0,QDQ_calls_allowed=0,AIME_generations_allowed=0,scientific_qualification=False)
 put(r/'ENGINEERING_LOAD_ONLY_PROBE_SOURCE_FREEZE.json',frozen)
 permit=dict(status='ALLOW_ONE_ENGINEERING_LOADER_ONLY_OBSERVATION',source_freeze_sha256=D.sha(r/'ENGINEERING_LOAD_ONLY_PROBE_SOURCE_FREEZE.json'),repair_authorization_sha256=D.sha(r/'HUMAN_REPAIR_AUTHORIZATION.json'),gpu_uuid='CPU_FIXTURE_NO_GPU',model_forward_calls_allowed=0,QDQ_calls_allowed=0,AIME_generations_allowed=0,counts_as_scientific_qualification=False)
 put(r/'ENGINEERING_LOAD_ONLY_PROBE_PERMIT.json',permit)
 mapped=[]
 for row in json.loads((r/'analysis/RUNTIME_NATIVE_CLOSURE.json').read_text())['files']:
  mapped.append(dict(canonical_path=row['path'],ELF_magic='7f454c46',sha256=row['sha256'],errors=[],stat={k:row[k]for k in('bytes','inode','device','mtime_ns')}))
 observations=[];stages=[]
 for stage in('PRE_TORCH','POST_TORCH','MODEL_LOADED','POST_SCALAR_PROFILER'):
  name='operations/PROBE/MAPS_'+stage+'.json';put(r/name,dict(identity=worker,observation_only=True,status='OBSERVATION_ONLY_NO_NATIVE_ADMISSION',errors=[],raw_maps='ISOLATED_CPU_FIXTURE_ONLY\n',mapped_native=mapped))
  observations.append(metadata(r,name));stages.append(dict(path=name,sha256=D.sha(r/name),mapped_native_count=len(mapped),assessment_status='PASS_ALL_ACTUAL_NATIVE_MAPPINGS_EXACTLY_PINNED'))
 result=dict(status='PASS_ENGINEERING_LOAD_ONLY_OBSERVATION_NOT_QUALIFICATION',identity=worker,observations=observations,model_forward_calls=0,QDQ_calls=0,AIME_generations=0,counts_as_scientific_qualification=False,native_maps_are_observations_not_admission=True)
 result_name='operations/PROBE/RESULT.json';put(r/result_name,result)
 interval=dict(dispatch_monotonic_lower=10.,identity_monotonic_upper=11.,exit_first_observed_monotonic=20.,release_first_observed_monotonic=21.,monotonic_same_host=True)
 completion=dict(status='PASS_ENGINEERING_LOAD_ONLY_PROBE_EXITED_RELEASED',worker_identity=worker,supervisor_identity=supervisor,actual_wait_returncode=0,all_exact_own_processes_exited=True,own_GPU_UUID_released=True,model_forward_calls=0,QDQ_calls=0,AIME_generations=0,counts_as_scientific_qualification=False,probe_source_freeze_sha256=D.sha(r/'ENGINEERING_LOAD_ONLY_PROBE_SOURCE_FREEZE.json'),probe_permit_sha256=D.sha(r/'ENGINEERING_LOAD_ONLY_PROBE_PERMIT.json'),result=metadata(r,result_name),resource_interval=interval,EXACT_physical_GPUh='UNRESOLVED',cost_scope='ENGINEERING_LOADER_ONLY_SEPARATE_FROM_58_QUALIFICATION_AND_60_FORMAL')
 comp='operations/ENGINEERING_LOAD_ONLY_PROBE_COMPLETION.json';put(r/comp,completion)
 review=dict(status='PASS_ALL_LOAD_ONLY_NATIVE_MAP_STAGES_EXACTLY_PINNED',native_manifest_sha256=D.sha(r/'analysis/RUNTIME_NATIVE_CLOSURE.json'),completion_sha256=D.sha(r/comp),stages=stages,actual_fresh_native_closure_count=len(mapped))
 rev='operations/ENGINEERING_NATIVE_MAPS_FINAL_REVIEW.json';put(r/rev,review)
 return comp,rev,result_name

def probe_tests(r):
 comp,rev,result_name=probe_fixture(r)
 def code():
  paths=[metadata(r,str(p.relative_to(r)))for p in sorted(r.rglob('*'))if p.is_file()and not p.is_symlink()]
  return helper(r,"by_path="+repr({x['path']:x for x in paths})+"\nm=read(r/'analysis/RUNTIME_NATIVE_CLOSURE.json');x=loader_probe_scope(r,by_path,sha(r/'analysis/RUNTIME_NATIVE_CLOSURE.json'),m['count']);print(json.dumps(x))")
 positive=must_json(code());check('loader_probe_separate_engineering_cost_not_58_or60',positive['scientific_qualification_PASS_from_probe']==positive['formal_generations_from_probe']==0 and positive['observed_dispatch_to_release_upper_seconds']==11. and positive['EXACT_physical_GPUh']=='UNRESOLVED')
 check('loader_probe_worker_and_supervisor_exits_independent',positive['exit']['exact_identity_exited']and positive['supervisor_exit']['exact_identity_exited'])
 check('all_original_raw_maps_observations_bound_to_final_native_rows',len(positive['native_map_stages'])==4 and all(x['all_observed_exact_native_rows_match_final_manifest']for x in positive['native_map_stages']))
 def reject_record(name,path,change):
  original=(r/path).read_bytes();value=json.loads(original);change(value);put(r/path,value)
  check(name,execute(code()).returncode!=0);(r/path).write_bytes(original)
 reject_record('real_probe_exit_minus11_never_becomes_PASS',comp,lambda x:x.update(status='FAIL_ENGINEERING_LOAD_ONLY_PROBE',actual_wait_returncode=-11))
 reject_record('probe_zero_exit_required_after_RESULT',comp,lambda x:x.update(actual_wait_returncode=-11))
 reject_record('probe_model_forward_forbidden',comp,lambda x:x.update(model_forward_calls=1))
 reject_record('probe_not_scientific_qualification_required',comp,lambda x:x.update(counts_as_scientific_qualification=True))
 reject_record('probe_release_required',comp,lambda x:x.update(own_GPU_UUID_released=False))
 reject_record('probe_same_host_monotonic_required',comp,lambda x:x['resource_interval'].update(monotonic_same_host=False))
 reject_record('probe_monotonic_order_required',comp,lambda x:x['resource_interval'].update(release_first_observed_monotonic=9.))
 reject_record('final_probe_native_manifest_identity_required',rev,lambda x:x.update(native_manifest_sha256='0'*64))
 reject_record('final_probe_dynamic_native_count_required',rev,lambda x:x.update(actual_fresh_native_closure_count=293))
 reject_record('probe_RESULT_SHA_bytes_required',result_name,lambda x:x.update(QDQ_calls=1))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
 check('new_engineering_repair_attempt_identity',D.ATTEMPT==ROOT.name.upper() and D.ATTEMPT.endswith('_EXIT_RACE_REPAIR_20261023'))
 config_actual=D.read(ROOT/'ROOT.json')
 check('actual_remote_archive_identity_is_new_lowercase_attempt',Path(config_actual['remote_root']).name==D.ATTEMPT.lower() and Path(config_actual['archive_root']).name==D.ATTEMPT.lower())
 check('old_GPU_continuation_is_not_current_attempt',not D.ATTEMPT.endswith('_GPU_CONTINUATION_20261022'))
 with tempfile.TemporaryDirectory(prefix='resident_readonly_fixture_')as temp:
  base=Path(temp);r=base/'cpu_remote_slug';r.mkdir();config,s,m=fixture(r)
  for name in('../escape','/absolute','a//b','a/./b','a/../b','a\nfile','a\x00file'):
   rejects('unsafe_path_rejected_'+repr(name),lambda name=name:D.safe_relative(name))
  check('valid_nested_metadata_path_allowed',D.safe_relative('nested/SOURCE_FREEZE.json')==Path('nested/SOURCE_FREEZE.json'))
  for suffix in D.RAW_SUFFIXES:check('raw_suffix_blocked_'+suffix,D.raw_path('assets/test'+suffix))
  native_tests(base,r)
  positive=must_json(D.audit_code(config,s,'29'))
  check('dynamic_source_count_not_68',positive['source_count']==4)
  check('archive_native_contract_has_dynamic_full_catalog',positive['native_closure_contract']['count']==3)
  check('dynamic_science_count_not_104',positive['count']==m['count'])
  check('all_raw_dual_catalog_stays_remote',len(positive['raw_remote_only_files'])==1)
  controls=D.decode_controls(positive);check('control_bytes_copied_verbatim',controls['SHA256_MANIFEST.json']==(r/'SHA256_MANIFEST.json').read_bytes())
  changed=copy.deepcopy(positive);changed['controls']['FINAL_SEAL_RECEIPT.json']['base64']=base64.b64encode(b'{}\n').decode()
  rejects('raw_control_bytes_serialization_mismatch_rejected',lambda:D.decode_controls(changed))
  extra=r/'UNKNOWN.json';extra.write_text('{}');check('archive_unknown_file_rejected',execute(D.audit_code(config,s,'29')).returncode!=0);extra.unlink()
  extra=r/'UNEXPECTED_EMPTY_DIR';extra.mkdir();check('archive_unknown_empty_directory_rejected',execute(D.audit_code(config,s,'29')).returncode!=0);extra.rmdir()
  tensor=r/'assets/small_tensor.bin';old=tensor.read_bytes();tensor.write_bytes(b'X'*len(old));check('changed_tensor_actual_SHA_rejected',execute(D.audit_code(config,s,'29')).returncode!=0);tensor.write_bytes(old)
  p=r/'src/cpu_source.py';old=p.read_bytes();p.unlink();p.symlink_to(tensor);check('symlink_member_rejected',execute(D.audit_code(config,s,'29')).returncode!=0);p.unlink();p.write_bytes(old)
  rejects('raw_stream_request_forbidden',lambda:D.stream_program(config,m['raw_remote_only_files']))
  rows=[x for x in m['files']if not D.raw_path(x['path'])]
  stream=execute(D.stream_program(config,rows));check('actual_uncompressed_stream_code_return0',stream.returncode==0)
  with tarfile.open(fileobj=io.BytesIO(stream.stdout),mode='r|')as archive:
   actual=[]
   for member in archive:
    data=archive.extractfile(member).read();actual.append(member.name);row=next(x for x in rows if x['path']==member.name)
    check('stream_actual_small_bytes_'+member.name,len(data)==row['bytes']and hashlib.sha256(data).hexdigest()==row['sha256'])
   check('stream_roster_excludes_raw',set(actual)=={x['path']for x in rows}and not any(D.raw_path(n)for n in actual))
  # Remote program argv remains bounded for thousands of manifest rows.
  many=[dict(path='fixture/%08d.json'%i,bytes=1,sha256='0'*64)for i in range(20000)]
  check('large_dynamic_roster_does_not_exceed_single_argv_limit',len(D.stream_program(config,many).encode())<16384)
  check('wrong_stream_roster_binding_rejected',execute(D.stream_program(config,rows[:-1])).returncode!=0)
  # Readiness only probes metadata/proc: these fake exited PIDs cannot touch a real process.
  ready=execute(D.readiness_code(config,s));check('dead_finisher_without_result_is_real_failure',ready.returncode!=0)
  put(r/'FINISHER_RESULT.json',dict(status='PASS_REAL_TERMINAL_SCIENCE_ARCHIVED_ON29'))
  ready=must_json(D.readiness_code(config,s));check('true_metadata_exit_and_seal_readiness',ready['status']=='READY_TRUE_CONTROLLER_AND_FINISHER_EXITED_AND_SEALED')
  put(r/'FINISHER_FAILURE.json',dict(status='SYNTHETIC_EXPECTED_FAILURE'))
  check('finisher_failure_never_waits_as_healthy',execute(D.readiness_code(config,s)).returncode!=0)
  local=base/'local';(local/'operations').mkdir(parents=True)
  atomic=D.local_atomic_selftest(local);check('actual_local_atomic_positive_and_negative',atomic['status']=='PASS_ACTUAL_LOCAL_NOREPLACE_POSITIVE_AND_NEGATIVE')
  # SOURCE_LOCAL accepts only missing raw, never missing nonraw source.
  remote=dict(source_files=json.loads((r/'SOURCE_FREEZE.json').read_text())['files'])
  for row in remote['source_files']:
   if not D.raw_path(row['path']):
    destination=local/row['path'];destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes((r/row['path']).read_bytes())
  verified=D.verify_local_sources(local,remote);check('local_missing_raw_catalog_exception',verified['local_present_count']==3and verified['remote_only_missing_raw_count']==1)
  (local/'src/cpu_source.py').unlink();rejects('local_missing_nonraw_source_rejected',lambda:D.verify_local_sources(local,remote))
  # Both generated remote programs parse without executing any remote command.
  for role in('55','29'):compile(D.audit_code(config,s,role),'audit_'+role,'exec')
  compile(D.readiness_code(config,s),'readiness','exec')
  snap_spec=importlib.util.spec_from_file_location('_snapshot_fixture',ROOT/'operations/read_snapshot.py');snap=importlib.util.module_from_spec(snap_spec);snap_spec.loader.exec_module(snap)
  compile(snap.snapshot_code(config),'snapshot','exec');check('all_generated_remote_programs_compile',True)
  probe_root=base/'probe_cpu_fixture';probe_root.mkdir();fixture(probe_root);probe_tests(probe_root)
  check('test_did_not_import_Torch','torch'not in sys.modules)
  result=dict(status='PASS_ISOLATED_READONLY_DELIVERY_CPU_FIXTURES',checks=CHECKS,check_count=len(CHECKS),CPU_only=True,SSH_commands=0,GPU_dispatches=0,Torch_imported=False,
    actual_local_atomic=atomic,source_pins={str(p.relative_to(ROOT)):D.sha(p)for p in(ROOT/'operations/read_snapshot.py',ROOT/'operations/download_verified_delivery.py',Path(__file__))},
    cannot_claim_real_55_29_archive_delivery=True,monotonic=time.monotonic())
  put(args.output,result);print(json.dumps(dict(status=result['status'],check_count=len(CHECKS))))
if __name__=='__main__':main()
