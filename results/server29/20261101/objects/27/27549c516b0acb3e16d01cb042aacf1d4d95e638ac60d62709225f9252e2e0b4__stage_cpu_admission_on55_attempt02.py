#!/usr/bin/env python3
"""One-shot stdlib small-source staging and CPU-only admission on55.

Default is preview. --run creates only a fresh CPU staging root, uploads the
bounded exact source bundle, verifies all files, then runs the frozen candidate
--check and --print-env with CUDA_VISIBLE_DEVICES='' and authorization NO.
No scientific runner, GPU worker, permit, YES authorization or runtime import.
Failure state is retained; inspect all paths before any further action.
"""
import argparse,datetime,gzip,hashlib,io,json,os,shlex,subprocess,sys,tarfile,time,traceback
from pathlib import Path,PurePosixPath
sys.dont_write_bytecode=True
TASK='QWEN_GDN_HISTORICAL_RUNTIME_GPU_RECOVERY_V1'
TECHNICAL='QWEN_GDN_HISTORICAL_RUNTIME_RESTORATION_V1'
FREEZE='64ca5853e4ed65ab3e2c59d9deb85b4aa026f0f712abd5496495b7b912fcb457'
MAX_FILE=16*1024**2;MAX_TOTAL=512*1024**2

def require(ok,message):
 if not ok:raise RuntimeError(message)
def encoded(x):return(json.dumps(x,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
def digest(b):return hashlib.sha256(b).hexdigest()
def stable(p):
 p=Path(p);require(p.is_file()and not p.is_symlink(),'Unsafe/missing regular file:'+str(p));a=p.stat();b=p.read_bytes();z=p.stat();require((a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns),'Changed file:'+str(p));return b
def read(p):return json.loads(stable(p))
def save_new(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);raw=encoded(value)
 with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
 return dict(path=str(p),bytes=len(raw),sha256=digest(raw))
def rows(payload):
 result=[]
 for p in sorted(payload.rglob('*')):
  require(not p.is_symlink(),'Payload symlink prohibited')
  if not p.is_file():continue
  rel=str(p.relative_to(payload));require(p.stat().st_size<MAX_FILE,'Large asset forbidden');require(p.suffix.lower()not in {'.pt','.pth','.bin','.safetensors','.so','.dll','.dylib','.npy','.npz'}and'.so.'not in p.name,'Model/tensor/native binary prohibited');raw=stable(p);require(not raw.startswith(b'\x7fELF'),'Native binary prohibited');result.append(dict(path=rel,bytes=len(raw),sha256=digest(raw)))
 require(sum(x['bytes']for x in result)<MAX_TOTAL,'Small-source bound exceeded');return result

def verify_local(root):
 state=read(root/'ROOT.json');require(state.get('task_id')==TASK and state.get('ALLOW_GPU_RESTORE')=='NO'and state.get('current_GPU_work_allowed')is False,'Outer authorization must remain NO');require(state.get('GPU_queries')==state.get('GPU_dispatches')==state.get('Driver_API_calls')==0,'Unexpected GPU/Driver use');payload=root/'execution_payload';meta=read(payload/'ROOT.json');auth=read(payload/'HUMAN_GPU_RESTORE_AUTHORIZATION.json');require(meta.get('task_id')==auth.get('task_id')==TECHNICAL and meta.get('outer_task_id')==auth.get('outer_task_id')==TASK,'Explicit task namespaces differ');require(meta.get('ALLOW_GPU_RESTORE')==auth.get('ALLOW_GPU_RESTORE')=='NO','Payload must remain NO');fr=stable(payload/'SOURCE_FREEZE.json');require(digest(fr)==state.get('effective_source_freeze_sha256')==FREEZE,'Wrong latest effective freeze');freeze=json.loads(fr);expected=freeze['files'];require(len(expected)==407,'Effective source count differs');actual=rows(payload);byname={x['path']:x for x in actual};require(len(byname)==len(actual),'Duplicate source path');require(set(byname)=={x['path']for x in expected}|{'SOURCE_FREEZE.json','ROOT.json','HUMAN_GPU_RESTORE_AUTHORIZATION.json'},'Exact407 plus3 metadata roster required')
 for row in expected:require(byname.get(row['path'])==row,'Effective source bytes/SHA mismatch:'+row['path'])
 require(not any('PERMIT' in x['path'].upper()or x['path'].startswith('execution/')for x in actual),'No permit/started unit may be staged');return state,actual

REMOTE_START=r'''import json,os,sys
from pathlib import Path
s=json.load(sys.stdin);p=Path(s['root']);assert p.is_absolute()and'..'not in p.parts;assert not p.exists(),'CPU staging root exists; no blind retry';p.mkdir(parents=True,exist_ok=False);raw=(json.dumps(dict(task_id=s['task_id'],status='CPU_ADMISSION_STARTED_NO_GPU',ALLOW_GPU_RESTORE='NO'),sort_keys=True)+'\n').encode()
with(p/'CPU_ADMISSION_STARTED.json').open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
print(json.dumps(dict(status='PASS_FRESH_CPU_ROOT_RESERVED',root=str(p))))
'''
REMOTE_UPLOAD=r'''import hashlib,json,os,sys
from pathlib import Path
s=json.loads(sys.argv[1]);p=Path(s['path']);tmp=Path(str(p)+'.incoming');assert not p.exists()and not tmp.exists(),'No overwrite remote upload';p.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256();count=0
with tmp.open('xb')as f:
 for b in iter(lambda:sys.stdin.buffer.read(1048576),b''):count+=len(b);assert count<=s['bytes'];h.update(b);f.write(b)
 f.flush();os.fsync(f.fileno())
assert count==s['bytes']and h.hexdigest()==s['sha256'],'Upload byte/SHA mismatch';os.link(tmp,p);tmp.unlink();z=p.stat();q=hashlib.sha256()
with p.open('rb')as f:
 for b in iter(lambda:f.read(1048576),b''):q.update(b)
a=p.stat();assert(z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns)==(a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns);assert z.st_size==s['bytes']and q.hexdigest()==s['sha256'];print(json.dumps(dict(status='PASS_REMOTE_NEW_FILE_SHA_BYTES',path=str(p),bytes=z.st_size,sha256=q.hexdigest())))
'''
REMOTE_CHECK=r'''import ctypes,datetime,hashlib,json,os,subprocess,sys,tarfile,time,traceback
from pathlib import Path,PurePosixPath
s=json.load(sys.stdin);root=Path(s['root']);dest=root/'execution_payload';tmp=root/'execution_payload.incoming';receipt=root/'REMOTE_CPU_ADMISSION_ATTEMPT02.json';assert root.is_dir()and not dest.exists()and not tmp.exists()and not receipt.exists(),'One-shot CPU stage already exists'
def raw(p):
 assert p.is_file()and not p.is_symlink();a=p.stat();b=p.read_bytes();z=p.stat();assert(a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns);return b
def sha(p):return hashlib.sha256(raw(p)).hexdigest()
def roster(d):return[dict(path=str(p.relative_to(d)),bytes=len(raw(p)),sha256=sha(p))for p in sorted(d.rglob('*'))if p.is_file()]
def write(value):
 b=(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
 with receipt.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
 return dict(path=str(receipt),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
record=dict(task_id=s['task_id'],status='CPU_ADMISSION_IN_PROGRESS_NO_GPU',ALLOW_GPU_RESTORE='NO',GPU_queries=0,GPU_dispatches=0,Driver_API_calls=0,Torch_imports=0,commands=[],bundle=s['bundle'],expected_effective_source_sha256=s['effective_sha'],expected_roster=s['rows'])
try:
 bundle=Path(s['bundle']['path']);assert bundle.stat().st_size==s['bundle']['bytes']and sha(bundle)==s['bundle']['sha256'];tmp.mkdir();seen=set();total=0
 with tarfile.open(bundle,'r:gz')as tf:
  for member in tf:
   name=PurePosixPath(member.name);assert member.isfile()and not name.is_absolute()and'..'not in name.parts and member.name not in seen and member.size<16*1024**2;seen.add(member.name);total+=member.size;assert total<512*1024**2;p=tmp/member.name;assert p.resolve().is_relative_to(tmp.resolve());p.parent.mkdir(parents=True,exist_ok=True)
   with tf.extractfile(member)as src,p.open('xb')as out:
    for b in iter(lambda:src.read(1048576),b''):out.write(b)
    out.flush();os.fsync(out.fileno())
   p.chmod(member.mode&0o777)
 actual=roster(tmp);assert actual==s['rows'],'Remote410 source SHA/bytes differs';assert sha(tmp/'SOURCE_FREEZE.json')==s['effective_sha'];libc=ctypes.CDLL(None,use_errno=True);rename=getattr(libc,'renameat2',None)
 if rename is not None:
  rename.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint];rename.restype=ctypes.c_int;rc=rename(-100,os.fsencode(tmp),-100,os.fsencode(dest),1)
 else:
  assert os.uname().machine=='x86_64';libc.syscall.restype=ctypes.c_long;rc=libc.syscall(ctypes.c_long(316),ctypes.c_int(-100),ctypes.c_char_p(os.fsencode(tmp)),ctypes.c_int(-100),ctypes.c_char_p(os.fsencode(dest)),ctypes.c_uint(1))
 if rc!=0:raise OSError(ctypes.get_errno(),'Atomic no-clobber CPU directory publish failed')
 record['remote_roster_before_checks']=roster(dest);assert record['remote_roster_before_checks']==s['rows'];env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',ALLOW_GPU_RESTORE='NO',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1');record['environment_controls']={k:env[k]for k in ['CUDA_VISIBLE_DEVICES','PYTHONDONTWRITEBYTECODE','ALLOW_GPU_RESTORE','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE']}
 for option in ['--check','--print-env']:
  cmd=['/bin/sh',str(dest/'historical_runtime_candidate.sh'),option];before=time.monotonic();p=subprocess.run(cmd,env=env,cwd=str(dest),stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=600);entry=dict(command=cmd,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,wall_seconds=time.monotonic()-before);record['commands'].append(entry);assert p.returncode==0,'Frozen CPU entry returned nonzero:'+option;obj=json.loads(p.stdout);entry['parsed_json']=obj;assert obj.get('GPU_queries')==obj.get('GPU_dispatches')==0 and obj.get('ALLOW_GPU_RESTORE')=='NO','Unexpected GPU allowance/count in CPU entry'
  if option=='--check':assert obj['source_files']==407 and obj['external_runtime_identity']=='PASS_CURRENT_HOST_EXTERNAL_SOURCE_MODEL_NATIVE_SHA'
 record['remote_roster_after_checks']=roster(dest);assert record['remote_roster_after_checks']==s['rows'],'Payload changed after CPU checks';record.update(status='PASS_REMOTE_CPU_ADMISSION_NOT_GPU_AUTHORIZATION',source_files_verified=407,payload_files_verified=len(s['rows']),payload_bytes_verified=sum(x['bytes']for x in s['rows']),source_bindings={name:dict(path=str(dest/name),bytes=(dest/name).stat().st_size,sha256=sha(dest/name))for name in ['scripts/restoration_controller.py','worker/restoration_worker.py','scripts/candidate_check.py','historical_runtime_candidate.sh','SOURCE_FREEZE.json']},completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());proof=write(record);print(json.dumps(dict(status=record['status'],receipt=proof,result=record),ensure_ascii=False))
except BaseException as exc:
 record.update(status='CPU_ADMISSION_FAILED_NO_BLIND_RETRY',error=type(exc).__name__+':'+str(exc),traceback=traceback.format_exc());proof=write(record);print(json.dumps(dict(status=record['status'],receipt=proof,result=record),ensure_ascii=False));sys.exit(2)
'''

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True);ap.add_argument('--run',action='store_true');args=ap.parse_args();root=args.root.resolve();state,roster=verify_local(root);preview=dict(task_id=TASK,status='PREVIEW_CPU_ADMISSION_NO_NETWORK_NO_GPU',source_files=407,payload_files=len(roster),payload_bytes=sum(x['bytes']for x in roster),effective_source_sha256=FREEZE,GPU_queries=0,GPU_dispatches=0,Driver_API_calls=0)
 if not args.run:print(json.dumps(preview));return
 operations=root/'operations';operations.mkdir(exist_ok=True);ledger=operations/'CPU_ADMISSION_ATTEMPT02_STATE.jsonl';require(not ledger.exists(),'One-shot CPU admission state exists: inspect remote paths, no blind retry')
 with ledger.open('x')as f:f.flush();os.fsync(f.fileno())
 def event(kind,**data):
  with ledger.open('a')as f:f.write(json.dumps(dict(event=kind,unix=time.time(),**data),sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
 report=dict(preview,status='CPU_ADMISSION_STARTED_NO_GPU',tool_source=dict(path=str(Path(__file__).resolve()),bytes=Path(__file__).stat().st_size,sha256=digest(stable(__file__))),local_root_snapshot_sha256=digest(stable(root/'ROOT.json')),request_sha256=state['request_sha256'],commands_expected=['--check','--print-env'])
 def remote(code,args=(),data=None):
  ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-S',state['ssh_control_path'],state['remote_user']+'@'+state['remote_host'],shlex.join([state['remote_python'],'-B','-c',code]+list(args))];p=subprocess.run(ssh,input=data,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=720);out=p.stdout.decode(errors='replace');err=p.stderr.decode(errors='replace');event('SSH_ACTION_RETURNED',code_sha256=digest(code.encode()),returncode=p.returncode,stdout_bytes=len(p.stdout),stderr=err);require(p.returncode==0,'CPU SSH action failed; inspect state and remote receipt; no retry. stdout='+out[:2000]+' stderr='+err[:2000]);return json.loads(out)
 event('START',**preview)
 try:
  bundle=operations/'CPU_ADMISSION_SOURCE_ATTEMPT02.tar.gz';require(not bundle.exists(),'Bundle exists: inspect state')
  with bundle.open('xb')as stream,gzip.GzipFile(filename='',mode='wb',fileobj=stream,mtime=0)as gz,tarfile.open(fileobj=gz,mode='w')as tf:
   for row in roster:
    p=root/'execution_payload'/row['path'];raw=stable(p);require(digest(raw)==row['sha256'],'Local source changed before tar');info=tarfile.TarInfo(row['path']);info.size=len(raw);info.mode=p.stat().st_mode&0o777;info.mtime=0;info.uid=info.gid=0;tf.addfile(info,io.BytesIO(raw))
  report['bundle']=dict(path=str(bundle),bytes=bundle.stat().st_size,sha256=digest(stable(bundle)));report['local_roster']=roster;remote_root=state['remote_root'];require(PurePosixPath(remote_root).is_absolute()and'..'not in PurePosixPath(remote_root).parts,'Unsafe remote CPU root');event('RESERVE_FRESH_REMOTE_ROOT_STARTED',remote_root=remote_root);report['remote_start']=remote(REMOTE_START,data=encoded(dict(root=remote_root,task_id=TASK)));event('REMOTE_ROOT_RESERVED');remote_bundle=dict(path=remote_root+'/CPU_ADMISSION_SOURCE_ATTEMPT02.tar.gz',bytes=report['bundle']['bytes'],sha256=report['bundle']['sha256']);report['remote_upload']=remote(REMOTE_UPLOAD,[json.dumps(remote_bundle)],stable(bundle));event('BUNDLE_UPLOADED_AND_SHA_VERIFIED');report['remote_cpu']=remote(REMOTE_CHECK,data=encoded(dict(root=remote_root,task_id=TASK,bundle=remote_bundle,rows=roster,effective_sha=FREEZE)));require(report['remote_cpu']['status']=='PASS_REMOTE_CPU_ADMISSION_NOT_GPU_AUTHORIZATION','Remote CPU admission not PASS');require(verify_local(root)[1]==roster,'Local payload changed during CPU staging');report.update(status='PASS_REMOTE_CPU_ADMISSION_NOT_GPU_AUTHORIZATION',ALLOW_GPU_RESTORE='NO',source_files_verified=407,payload_files_verified=len(roster),no_scientific_or_GPU_execution=True,completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());desc=save_new(root/'qualification/REMOTE_CPU_ADMISSION_ATTEMPT02.json',report);event('PASS',receipt=desc);print(json.dumps(dict(status=report['status'],receipt=desc,source_files=407,payload_files=len(roster),GPU_queries=0,GPU_dispatches=0,Driver_API_calls=0)))
 except BaseException as exc:
  report.update(status='CPU_ADMISSION_FAILED_NO_BLIND_RETRY',error=type(exc).__name__+':'+str(exc),traceback=traceback.format_exc());target=root/'qualification/REMOTE_CPU_ADMISSION_ATTEMPT02.json'
  if not target.exists():save_new(target,report)
  event('FAILED_INSPECT_STATE_NO_BLIND_RETRY',error=report['error']);raise
if __name__=='__main__':main()
