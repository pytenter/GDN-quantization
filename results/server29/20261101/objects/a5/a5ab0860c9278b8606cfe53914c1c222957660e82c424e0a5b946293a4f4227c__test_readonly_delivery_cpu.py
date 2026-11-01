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
 (r/'src').mkdir(parents=True);(r/'assets').mkdir()
 (r/'src/cpu_source.py').write_text('# frozen CPU fixture only\n')
 (r/'assets/small_tensor.bin').write_bytes(bytes(range(64)))
 source=[metadata(r,n)for n in('src/cpu_source.py','assets/small_tensor.bin')]
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

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
 check('new_engineering_repair_attempt_identity',D.ATTEMPT==ROOT.name and D.ATTEMPT.endswith('_ENGINEERING_REPAIR_20261022'))
 config_actual=D.read(ROOT/'ROOT.json')
 check('actual_remote_archive_identity_is_new_lowercase_attempt',Path(config_actual['remote_root']).name==D.ATTEMPT.lower() and Path(config_actual['archive_root']).name==D.ATTEMPT.lower())
 check('old_GPU_continuation_is_not_current_attempt',not D.ATTEMPT.endswith('_GPU_CONTINUATION_20261022'))
 with tempfile.TemporaryDirectory(prefix='resident_readonly_fixture_')as temp:
  base=Path(temp);r=base/'cpu_remote_slug';r.mkdir();config,s,m=fixture(r)
  for name in('../escape','/absolute','a//b','a/./b','a/../b','a\nfile','a\x00file'):
   rejects('unsafe_path_rejected_'+repr(name),lambda name=name:D.safe_relative(name))
  check('valid_nested_metadata_path_allowed',D.safe_relative('nested/SOURCE_FREEZE.json')==Path('nested/SOURCE_FREEZE.json'))
  for suffix in D.RAW_SUFFIXES:check('raw_suffix_blocked_'+suffix,D.raw_path('assets/test'+suffix))
  positive=must_json(D.audit_code(config,s,'29'))
  check('dynamic_source_count_not_68',positive['source_count']==2)
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
  (local/'src').mkdir();(local/'src/cpu_source.py').write_bytes((r/'src/cpu_source.py').read_bytes())
  verified=D.verify_local_sources(local,remote);check('local_missing_raw_catalog_exception',verified['local_present_count']==1and verified['remote_only_missing_raw_count']==1)
  (local/'src/cpu_source.py').unlink();rejects('local_missing_nonraw_source_rejected',lambda:D.verify_local_sources(local,remote))
  # Both generated remote programs parse without executing any remote command.
  for role in('55','29'):compile(D.audit_code(config,s,role),'audit_'+role,'exec')
  compile(D.readiness_code(config,s),'readiness','exec')
  snap_spec=importlib.util.spec_from_file_location('_snapshot_fixture',ROOT/'operations/read_snapshot.py');snap=importlib.util.module_from_spec(snap_spec);snap_spec.loader.exec_module(snap)
  compile(snap.snapshot_code(config),'snapshot','exec');check('all_generated_remote_programs_compile',True)
  check('test_did_not_import_Torch','torch'not in sys.modules)
  result=dict(status='PASS_ISOLATED_READONLY_DELIVERY_CPU_FIXTURES',checks=CHECKS,check_count=len(CHECKS),CPU_only=True,SSH_commands=0,GPU_dispatches=0,Torch_imported=False,
    actual_local_atomic=atomic,source_pins={str(p.relative_to(ROOT)):D.sha(p)for p in(ROOT/'operations/read_snapshot.py',ROOT/'operations/download_verified_delivery.py',Path(__file__))},
    cannot_claim_real_55_29_archive_delivery=True,monotonic=time.monotonic())
  put(args.output,result);print(json.dumps(dict(status=result['status'],check_count=len(CHECKS))))
if __name__=='__main__':main()
