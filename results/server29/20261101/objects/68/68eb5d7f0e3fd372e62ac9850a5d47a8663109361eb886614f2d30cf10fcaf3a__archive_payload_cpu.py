#!/usr/bin/env python3
"""Temporary stdlib fake transport/archive tests; no network or GPU APIs."""
import copy,ctypes,hashlib,importlib.util,io,json,os,shutil,struct,sys,tarfile,tempfile,types
from pathlib import Path
from unittest import mock
sys.dont_write_bytecode=True
RUN=Path(__file__).resolve().parents[1];spec=importlib.util.spec_from_file_location('archive_payload_cpu_fixture',RUN/'scripts/archive_payload_once.py');A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)
checks=[]
def ok(name,condition=True):assert condition,name;checks.append(dict(name=name,status='PASS'))
def reject(name,fn):
 try:fn()
 except (RuntimeError,OSError,AssertionError):ok(name);return
 raise AssertionError(name+' unexpectedly passed')
def save(path,obj):path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(A.encoded(obj))
def binding(payload,path):return A.hash_file(payload,path)
def fixture(top,complete=False):
 root=top.resolve();payload=root/'payload';payload.mkdir(parents=True);save(payload/'SOURCE_FREEZE.json',dict(task_id=A.TASK,files=[dict(path='source.py',bytes=1,sha256='x')]));(payload/'source.py').write_text('SYNTHETIC CPU SOURCE');(payload/'execution/raw').mkdir(parents=True);(payload/'execution/raw/token_000.bin').write_bytes(b'CPU fixture raw bytes, no tensor or GPU');controller=dict(pid=700,starttick=701,cmdline=['python','controller','--run']);save(payload/'execution/CONTROLLER_STARTED.json',dict(controller_identity=controller));workers=[];events=[]
 for i in range(6 if complete else 1):
  who=dict(pid=100+i,starttick=1000+i,cmdline=['python','synthetic-worker',str(i)]);workers.append(dict(who,process_exit_verified=True,all_UUID_NVML_release_verified=True));events.extend([dict(event='FIRST_DISPATCH',worker_identity=who),dict(event='FIRST_PROCESS_EXIT_OBSERVED',worker_identity=who),dict(event='FIRST_ALL_UUID_NVML_RELEASE_OBSERVED',worker_identity=who)])
 ledger=payload/'execution/RESOURCE_EVENTS.jsonl';ledger.write_text(''.join(json.dumps(x)+'\n'for x in events));status='COMPLETE_FINITE_RECOVERY_UNITS'if complete else'FAIL_CLOSED_STOP';save(payload/'execution/CONTROLLER_RESULT.json',dict(task_id=A.TASK,status=status));first=None
 if not complete:save(payload/'FIRST_FAILURE.json',dict(status='FAIL_CLOSED_STOP',error='SYNTHETIC CPU FAILURE'));first=binding(payload,'FIRST_FAILURE.json')
 supervisor=dict(pid=800,starttick=801,cmdline=['python',str(root/'scripts/offline_controller_once.py'),'--root',str(root)]);operations=[('operations/OFFLINE_NATIVE_INVOCATION_STARTED.json',dict(supervisor_identity=supervisor)),('operations/LIVE_INITIAL_PROCESS_SNAPSHOT.json',dict(status='SYNTHETIC_CPU_SNAPSHOT',supervisor_identity=supervisor)),('operations/NATIVE_CONTROLLER_EXIT_RECEIPT.json',dict(returncode=0 if complete else 1,controller_identity=controller))]
 for name,value in operations:save(root/name,value)
 allown=workers+[dict(controller,role='controller',process_exit_verified=True,all_UUID_NVML_release_verified=True),dict(supervisor,role='supervisor',process_exit_verified=True,all_UUID_NVML_release_verified=True)]
 proof=dict(schema='QWEN_RECOVERY_RELEASE_ADMISSION_V1',task_id=A.TASK,status='PASS_ALL_OWN_PROCESSES_EXITED_ALL_UUIDS_RELEASED',payload_root=str(payload),source_freeze_sha256=binding(payload,'SOURCE_FREEZE.json')['sha256'],controller_identity=controller,controller_exit_verified=True,final_owned_NVML_records=[],all_owned_processes_exited=True,all_owned_GPU_resources_released=True,controller_result=dict(binding(payload,'execution/CONTROLLER_RESULT.json'),status=status),resource_ledger=binding(payload,'execution/RESOURCE_EVENTS.jsonl'),first_failure=first,own_processes=allown,owned_worker_identities=[{k:x[k]for k in ['pid','starttick','cmdline']}for x in workers],supervisor_identity=supervisor,supervisor_identities=[supervisor],evidence_receipts=[binding(root,name)for name,_ in operations]);save(root/'release_proof.json',proof);return root,payload,root/'release_proof.json',proof
class FakeTransport:
 def __init__(self,base,fail=False,tamper=False):self.base=base;self.fail=fail;self.tamper=tamper
 def transfer(self,config,payload,rows):
  if self.fail:raise RuntimeError('Synthetic transport outage, retained partial')
  wire=io.BytesIO();A.write_tar(payload,rows,wire);dest=self.base/'published';stage=self.base/'incoming';stage.mkdir(parents=True);expected={x['path']:x for x in rows}
  with tarfile.open(fileobj=io.BytesIO(wire.getvalue()),mode='r:')as archive:
   for item in archive:
    assert item.isfile()and item.name in expected and '..'not in Path(item.name).parts;p=stage/item.name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(archive.extractfile(item).read())
  passes=[]
  for repeat in range(2):
   actual=[A.hash_file(stage,x['path'])for x in rows];assert actual==rows;passes.append(dict(pass_index=repeat+1,files=len(rows),bytes=sum(x['bytes']for x in rows),roster_sha256=A.digest(A.canon(actual))))
  raw=A.encoded(config['catalog']);(stage/'ASSET_CATALOG.json').write_bytes(raw);verification=dict(status='PASS_ARCHIVED_PAYLOAD_BYTES_ONLY_NO_SCIENTIFIC_PROMOTION',task_id=A.TASK,destination=A.DEST,files_verified=len(rows),bytes_verified=sum(x['bytes']for x in rows),source_freeze_sha256=config['source_freeze_sha256'],catalog_sha256=config['catalog_sha256'],release_proof_sha256=config['catalog']['release_proof']['sha256'],native_scientific_controller_status=config['catalog']['native_controller_status'],fresh_independent_remote_SHA_passes=passes,source_raw_downloaded_to_Mac=False,source_deleted=False,all_source_files_roster_exact=True,atomic_no_clobber_publication=True,received_tar_bytes=len(wire.getvalue()));vr=A.encoded(verification);(stage/'ARCHIVE_VERIFICATION.json').write_bytes(vr);assert not dest.exists();stage.rename(dest);result=dict(status=verification['status'],destination=A.DEST,verification=verification,ASSET_CATALOG=dict(path=A.DEST+'/ASSET_CATALOG.json',bytes=len(raw),sha256=A.digest(raw)),ARCHIVE_VERIFICATION=dict(path=A.DEST+'/ARCHIVE_VERIFICATION.json',bytes=len(vr),sha256=A.digest(vr)),retained_stream_archive='synthetic temp only')
  if self.tamper:result['ASSET_CATALOG']['sha256']='0'*64
  return result
with tempfile.TemporaryDirectory(prefix='ARCHIVE_PAYLOAD_CPU_FAKE_')as temporary:
 temp=Path(temporary);root,payload,pp,proof=fixture(temp/'one');release=A.validate_release(root,pp);ok('partial_true_STOP_release_proof_and_native_ledger_accepted',release['native_status']=='FAIL_CLOSED_STOP');before=A.inventory(payload)
 for name,change in [('wrong_payload',dict(payload_root='/different')),('wrong_sourcefreeze',dict(source_freeze_sha256='0'*64)),('controller_not_exited',dict(controller_exit_verified=False)),('owned_NVML_present',dict(final_owned_NVML_records=[dict(pid=100)])),('aggregate_unreleased',dict(all_owned_GPU_resources_released=False)),('wrong_controller_starttick',dict(controller_identity=dict(proof['controller_identity'],starttick=9)))]:
  bad=dict(proof,**change);save(pp,bad);reject(name+'_blocks_before_transport',lambda:A.validate_release(root,pp))
 save(pp,proof);bad=copy.deepcopy(proof);bad['own_processes'][0]['all_UUID_NVML_release_verified']=False;save(pp,bad);reject('worker_unresolved_release_blocks_archive',lambda:A.validate_release(root,pp));save(pp,proof);ledger=payload/'execution/RESOURCE_EVENTS.jsonl';prior=ledger.read_bytes();ledger.write_text('changed');reject('closedledger_changed_SHA_rejected',lambda:A.validate_release(root,pp));ledger.write_bytes(prior)
 (payload/'alias').symlink_to(payload/'source.py');reject('any_source_symlink_rejected',lambda:A.inventory(payload));(payload/'alias').unlink();os.mkfifo(payload/'pipe');reject('source_FIFO_rejected',lambda:A.inventory(payload));(payload/'pipe').unlink();(payload/'too_big.bin').touch()
 with mock.patch.object(A,'MAX_SINGLE',1):reject('single_file_bound_rejected',lambda:A.inventory(payload))
 (payload/'too_big.bin').unlink()
 with mock.patch.object(A,'MAX_TOTAL',1):reject('total_payload_bound_rejected',lambda:A.inventory(payload))
 reject('unsafe_relative_path_rejected',lambda:A.relative('../outside'));ok('safe_file_opens_only_regular_inside_payload',A.hash_file(payload,'source.py')['sha256']==hashlib.sha256((payload/'source.py').read_bytes()).hexdigest());rows=[A.hash_file(payload,x['path'])for x in A.inventory(payload)];altered=copy.deepcopy(rows);altered[0]['sha256']='0'*64;reject('source_second_pass_SHA_mismatch_rejected',lambda:A.write_tar(payload,altered,io.BytesIO()))
 with mock.patch.object(A.subprocess,'Popen',side_effect=AssertionError('No real SSH allowed')):result=A.archive_once(root,pp,FakeTransport(temp/'fake29'))
 ok('full_partial_payload_archived_without_scientific_PASS',result['status']=='PASS_POST_RELEASE_FULL_PAYLOAD_ARCHIVE_BYTES_ONLY'and result['native_scientific_controller_status']=='FAIL_CLOSED_STOP');ok('source_payload_roster_and_bytes_remain_identical',A.inventory(payload)==before and [A.hash_file(payload,x['path'])for x in before]==rows);ok('catalog_and_verification_outside_payload_only',(root/'archive/ASSET_CATALOG.json').is_file()and(root/'archive/ARCHIVE_VERIFICATION.json').is_file()and not(payload/'ASSET_CATALOG.json').exists());ok('remote_independent_two_SHA_passes_recorded',len(result['remote']['verification']['fresh_independent_remote_SHA_passes'])==2);reject('one_shot_archive_cannot_repeat',lambda:A.archive_once(root,pp,FakeTransport(temp/'duplicate')))
 failed,_,failedproof,_=fixture(temp/'failure')
 reject('transport_failure_stops_without_retry',lambda:A.archive_once(failed,failedproof,FakeTransport(temp/'failed29',fail=True)));ok('transport_failure_keeps_source_and_failure_receipt',(failed/'payload/execution/raw/token_000.bin').is_file()and(failed/'archive/ARCHIVE_FAILURE.json').is_file());reject('failed_archive_attempt_cannot_blind_retry',lambda:A.archive_once(failed,failedproof,FakeTransport(temp/'failed29retry')))
 wrong,_,wrongproof,_=fixture(temp/'wrongremote');reject('remote_catalog_wrong_SHA_does_not_pass',lambda:A.archive_once(wrong,wrongproof,FakeTransport(temp/'wrong29',tamper=True)))
 full,_,fullproof,fullvalue=fixture(temp/'complete',complete=True);fullaudit=A.validate_release(full,fullproof);ok('six_complete_dispatches_require_all_six_released',fullaudit['native_dispatches']==6);ok('complete_own_roster_includes_controller_and_supervisor_eight',fullaudit['owned_processes']==8)
 for label,mutator in [
  ('undeclared_extra_own_PID',lambda d:d['own_processes'].append(dict(pid=999,starttick=900,cmdline=['extra'],process_exit_verified=True,all_UUID_NVML_release_verified=True))),
  ('controller_missing_from_own',lambda d:d['own_processes'].pop(-2)),
  ('supervisor_missing_from_own',lambda d:d['own_processes'].pop(-1)),
  ('supervisor_unreleased',lambda d:d['own_processes'][-1].update(all_UUID_NVML_release_verified=False)),
  ('controller_unreleased',lambda d:d['own_processes'][-2].update(process_exit_verified=False)),
  ('worker_identity_roster_missing',lambda d:d['owned_worker_identities'].pop()),
  ('supervisor_declared_as_worker',lambda d:d['owned_worker_identities'].__setitem__(0,d['supervisor_identity'])),
  ('supervisor_list_differs',lambda d:d['supervisor_identities'][0].update(starttick=998)),
  ('missing_supervisor_evidence_binding',lambda d:d['evidence_receipts'].pop()),
  ('supervisor_evidence_SHA_mismatch',lambda d:d['evidence_receipts'][0].update(sha256='0'*64))]:
  bad=copy.deepcopy(fullvalue);mutator(bad);save(fullproof,bad);reject(label+'_rejected',lambda:A.validate_release(full,fullproof))
 save(fullproof,fullvalue);invocation=full/'operations/OFFLINE_NATIVE_INVOCATION_STARTED.json';saved=invocation.read_bytes();save(invocation,dict(supervisor_identity=dict(fullvalue['supervisor_identity'],starttick=998)));bad=copy.deepcopy(fullvalue);bad['evidence_receipts'][0]=binding(full,'operations/OFFLINE_NATIVE_INVOCATION_STARTED.json');save(fullproof,bad);reject('supervisor_freshSHA_receipt_identity_mismatch_rejected',lambda:A.validate_release(full,fullproof));invocation.write_bytes(saved);save(fullproof,fullvalue)
 reject('archive_python_requires_absolute_path',lambda:A.SSHTransport(python_executable='python3'))
 fakeproc=types.SimpleNamespace(returncode=0,stdout=json.dumps(dict(status='PASS_READ_ONLY_ARCHIVE_PYTHON',version_info=[3,12,3],executable='/usr/bin/python3')),stderr='')
 with mock.patch.object(A.subprocess,'run',return_value=fakeproc)as run:
  py=A.SSHTransport().check_python();ok('explicit29Python_readonly_preflight_mock',py['configured_path']=='/usr/bin/python3'and '/usr/bin/python3' in run.call_args.args[0][-1])
 with mock.patch.object(A.subprocess,'run',return_value=types.SimpleNamespace(returncode=1,stdout='',stderr='missing')),mock.patch.object(A.subprocess,'Popen',side_effect=AssertionError('stream must not start')):
  reject('Python_preflight_failure_prevents_archive_stream',lambda:A.SSHTransport().transfer({},payload,rows))
 # Execute the exact embedded receiver against an in-memory tiny tar with only
 # the atomic directory rename and capacity APIs replaced by CPU fakes.
 remote_root,remote_payload,rproof,_=fixture(temp/'remoteprogram');rrows=[A.hash_file(remote_payload,x['path'])for x in A.inventory(remote_payload)];catalog=dict(task_id=A.TASK,files=rrows,release_proof=dict(sha256='a'*64),native_controller_status='FAIL_CLOSED_STOP');dest=temp/'receiver_actual';config=dict(task_id=A.TASK,destination=str(dest),catalog=catalog,catalog_sha256=A.digest(A.encoded(catalog)),source_freeze_sha256=A.hash_file(remote_payload,'SOURCE_FREEZE.json')['sha256'],source_bytes=sum(x['bytes']for x in rrows));tar=io.BytesIO();A.write_tar(remote_payload,rrows,tar);header=A.encoded(config);wire=struct.pack('!Q',len(header))+header+tar.getvalue()
 class Fn:
  def __call__(self,_,src,__,target,flags):assert flags==1;src=os.fsdecode(src);target=os.fsdecode(target);assert not Path(target).exists();os.rename(src,target);return 0
 class Libc:renameat2=Fn()
 output=io.StringIO()
 with mock.patch.object(sys,'stdin',types.SimpleNamespace(buffer=io.BytesIO(wire))),mock.patch.object(sys,'stdout',output),mock.patch.object(ctypes,'CDLL',return_value=Libc()),mock.patch.object(os,'statvfs',return_value=types.SimpleNamespace(f_bavail=1024**4,f_frsize=1)):
  exec(compile(A.REMOTE_RECEIVER,'embedded_receiver_CPU_fake','exec'),{'__name__':'cpu_receiver'})
 real=json.loads(output.getvalue());ok('exact_embedded_remote_program_two_disk_SHA_passes_PASS',len(real['verification']['fresh_independent_remote_SHA_passes'])==2 and dest.is_dir());ok('exact_embedded_remote_catalog_SHA_matches_local',real['ASSET_CATALOG']['sha256']==config['catalog_sha256']);ok('remote_source_roster_plus_exact_two_sidecars',{str(p.relative_to(dest))for p in dest.rglob('*')if p.is_file()}=={x['path']for x in rrows}|{'ASSET_CATALOG.json','ARCHIVE_VERIFICATION.json'});ok('no_GPU_engine_import_or_real_network',not any(x=='torch'or x.startswith('torch.')for x in sys.modules))
print(json.dumps(dict(status='PASS_CPU_FAKE_FULL_PAYLOAD_ARCHIVE_TESTS_NOT_REAL_ARCHIVE',checks_passed=len(checks),checks=checks,GPU_queries=0,Driver_calls=0,real_network_calls=0,real_archive_calls=0,temporary_raw_fixtures_only=True,source_sha256=hashlib.sha256((RUN/'scripts/archive_payload_once.py').read_bytes()).hexdigest()),sort_keys=True,indent=2))
