#!/usr/bin/env python3
"""One-shot post-release full payload archive, CPU/file I/O only.

Default previews bounded source metadata. --run requires an independent native
controller/process/all-UUID release proof. It never queries GPU/NVML, reads any
external model, downloads tensors to Mac, modifies payload, or deletes sources.
"""
import argparse,ctypes,hashlib,io,json,os,shlex,stat,struct,subprocess,sys,tarfile,time,traceback
from pathlib import Path,PurePosixPath
sys.dont_write_bytecode=True
TASK='QWEN_GDN_HISTORICAL_RUNTIME_GPU_RECOVERY_V1'
DEST='/data/zypan/qwen_historical_runtime_gpu_recovery_archive/GPU_EXECUTION_20261019_V1/PAYLOAD_ASSETS'
MAX_TOTAL=12*1024**3;MAX_SINGLE=128*1024**2;MAX_FILES=20000;MAX_HEADER=32*1024**2

def require(ok,message):
 if not ok:raise RuntimeError(message)
def encoded(value):return(json.dumps(value,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
def canon(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def digest(raw):return hashlib.sha256(raw).hexdigest()
def identity(s):return(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
def new_bytes(path,raw):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('xb')as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
 return dict(path=str(path),bytes=len(raw),sha256=digest(raw))
def save_new(path,value):return new_bytes(path,encoded(value))
def stable_small(path):
 path=Path(path);require(path.is_file()and not path.is_symlink()and path.stat().st_size<=MAX_HEADER,'Unsafe/oversized evidence');a=path.stat();raw=path.read_bytes();require(identity(a)==identity(path.stat()),'Evidence changed during read');return raw
def read(path):return json.loads(stable_small(path))
def relative(value):
 q=PurePosixPath(value);require(isinstance(value,str)and not q.is_absolute()and q.parts and'..'not in q.parts and'.'not in q.parts and '\\'not in value and not any(c in value for c in ['\n','\r','\0']),'Unsafe relative path');return q

def inventory(payload):
 payload=Path(payload);require(payload.is_dir()and not payload.is_symlink(),'Unsafe payload root');rows=[]
 for path in sorted(payload.rglob('*')):
  st=path.lstat();require(not stat.S_ISLNK(st.st_mode),'Payload symlink forbidden:'+str(path));name=str(path.relative_to(payload));relative(name)
  if stat.S_ISDIR(st.st_mode):continue
  require(stat.S_ISREG(st.st_mode),'Nonregular payload asset forbidden:'+name);require(st.st_size<=MAX_SINGLE,'Single file exceeds128MiB:'+name);require(name not in ['ASSET_CATALOG.json','ARCHIVE_VERIFICATION.json'],'Reserved archive sidecar name already in source');rows.append(dict(path=name,bytes=st.st_size,mode=stat.S_IMODE(st.st_mode)))
 require(len(rows)<=MAX_FILES and sum(x['bytes']for x in rows)<=MAX_TOTAL,'Payload bounded archive quota exceeded');return rows

def safe_file(payload,name):
 parts=relative(name).parts;directory=os.open(payload,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:
  for part in parts[:-1]:
   new=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=directory);os.close(directory);directory=new
  fd=os.open(parts[-1],os.O_RDONLY|os.O_NOFOLLOW,dir_fd=directory)
 finally:os.close(directory)
 require(stat.S_ISREG(os.fstat(fd).st_mode),'Nonregular opened asset');return os.fdopen(fd,'rb')

def hash_file(payload,name):
 path=Path(payload)/name
 with safe_file(payload,name)as stream:
  a=os.fstat(stream.fileno());h=hashlib.sha256();count=0
  for raw in iter(lambda:stream.read(4*1024**2),b''):count+=len(raw);h.update(raw)
  require(identity(a)==identity(os.fstat(stream.fileno()))==identity(path.lstat()),'Asset changed during SHA:'+name)
 return dict(path=name,bytes=count,sha256=h.hexdigest())

def bound_file(payload,row):
 require(isinstance(row,dict),'Native evidence binding absent');name=str(relative(row['path']));actual=hash_file(payload,name);require(all(actual[k]==row[k]for k in ['path','bytes','sha256']),'Native receipt/ledger SHA differs:'+name);return read(Path(payload)/name)

def validate_release(root,proof_path):
 root=Path(root).resolve();payload=root/'payload';proof_path=Path(proof_path).resolve();require(payload not in proof_path.parents,'Independent release proof must be outside archived payload');proof=read(proof_path);require(proof.get('schema')=='QWEN_RECOVERY_RELEASE_ADMISSION_V1'and proof.get('task_id')==TASK and proof.get('status')=='PASS_ALL_OWN_PROCESSES_EXITED_ALL_UUIDS_RELEASED','Independent all-process/all-UUID release verdict missing');require(proof.get('payload_root')==str(payload.resolve()),'Release proof binds another payload root');freeze=hash_file(payload,'SOURCE_FREEZE.json');require(proof.get('source_freeze_sha256')==freeze['sha256'],'Release proof effective source SHA differs');require(proof.get('controller_exit_verified')is True and proof.get('final_owned_NVML_records')==[],'Controller exit or final own NVML release not proved');require(proof.get('all_owned_processes_exited',True)is True and proof.get('all_owned_GPU_resources_released',True)is True,'Contradictory aggregate release proof');controller_marker=payload/'execution/CONTROLLER_STARTED.json'
 if not controller_marker.exists():controller_marker=payload/'execution/CONTROLLER_ATTEMPT_STARTED.json'
 require(controller_marker.is_file()and proof.get('controller_identity')==read(controller_marker).get('controller_identity'),'Exact native controller PID/starttick/cmdline binding missing');native=bound_file(payload,proof.get('controller_result'));require(native.get('task_id')==TASK and native.get('status')in ['COMPLETE_FINITE_RECOVERY_UNITS','FAIL_CLOSED_STOP'],'Native controller terminal receipt missing');require(proof['controller_result'].get('status')==native['status'],'Native terminal status differs');first=proof.get('first_failure')
 if native['status']=='FAIL_CLOSED_STOP':require(first is not None,'True STOP first failure binding required')
 if first is not None:bound_file(payload,first)
 ledger_meta=proof.get('resource_ledger');require(isinstance(ledger_meta,dict),'Closed native resource ledger binding missing');bound_file(payload,ledger_meta)if str(ledger_meta.get('path','')).endswith('.json')else None;actual_ledger=hash_file(payload,str(relative(ledger_meta['path'])));require(all(actual_ledger[k]==ledger_meta[k]for k in ['path','bytes','sha256']),'Closed resource ledger differs');events=[]
 with safe_file(payload,ledger_meta['path'])as source:
  for raw in source:
   if raw.strip():events.append(json.loads(raw))
 workers=proof.get('own_processes');require(isinstance(workers,list),'Owned-process audit roster absent');own={}
 def ident_key(value):return canon({k:value[k]for k in ['pid','starttick','cmdline']})
 for row in workers:
  ident={k:row[k]for k in ['pid','starttick','cmdline']};require(type(ident['pid'])is int and type(ident['starttick'])is int and isinstance(ident['cmdline'],list),'Invalid process identity');require(row.get('process_exit_verified')is True and row.get('all_UUID_NVML_release_verified')is True,'An owned process has unresolved process/GPU release');key=ident_key(ident);require(key not in own,'Duplicate exact own process identity');own[key]=ident
 dispatch=[x for x in events if x.get('event')=='FIRST_DISPATCH'];native_workers={ident_key(x['worker_identity']):x['worker_identity']for x in dispatch};require(len(native_workers)==len(dispatch),'Duplicate native dispatch identity');declared=proof.get('owned_worker_identities');require(isinstance(declared,list)and len(declared)==len(native_workers)and {ident_key(x)for x in declared}==set(native_workers),'Declared worker subset differs from native dispatch identities')
 for key,ident in native_workers.items():
  require(own.get(key)==ident,'Dispatched worker missing from complete independent process proof');require(any(x.get('event')=='FIRST_PROCESS_EXIT_OBSERVED'and x.get('worker_identity')==ident for x in events),'Native first process exit absent');require(any(x.get('event')=='FIRST_ALL_UUID_NVML_RELEASE_OBSERVED'and x.get('worker_identity')==ident for x in events),'Native all-UUID release observation absent')
 supervisors=proof.get('supervisor_identities');require(isinstance(supervisors,list)and len(supervisors)==1 and proof.get('supervisor_identity')==supervisors[0],'One explicitly bound offline supervisor identity required');expected=set(native_workers)|{ident_key(proof['controller_identity'])}|{ident_key(x)for x in supervisors};require(len(expected)==len(native_workers)+2,'Controller, supervisor and worker identities must be distinct');require(set(own)==expected,'Owned-process roster must exactly match controller, declared supervisor and native workers')
 evidence=proof.get('evidence_receipts');require(isinstance(evidence,list),'Supervisor independent evidence receipt roster absent');byname={x['path']:x for x in evidence};require(len(byname)==len(evidence),'Duplicate supervisor evidence path');needed=['operations/OFFLINE_NATIVE_INVOCATION_STARTED.json','operations/LIVE_INITIAL_PROCESS_SNAPSHOT.json','operations/NATIVE_CONTROLLER_EXIT_RECEIPT.json'];require(all(name in byname for name in needed),'Supervisor invocation/initial/exit proof bindings missing')
 for row in evidence:
  actual=hash_file(root,str(relative(row['path'])));require(all(actual[k]==row[k]for k in ['path','bytes','sha256']),'Independent supervisor evidence bytes/SHA changed:'+row['path'])
 invocation=read(root/needed[0]);require(invocation.get('supervisor_identity')==supervisors[0],'Supervisor identity does not match its bound invocation receipt')
 if native['status']=='COMPLETE_FINITE_RECOVERY_UNITS':require(len(native_workers)==6,'Complete finite run must have six released worker dispatch identities')
 return dict(proof=proof,proof_binding=dict(path=str(proof_path),bytes=proof_path.stat().st_size,sha256=digest(stable_small(proof_path))),native_status=native['status'],source_freeze_sha256=freeze['sha256'],native_dispatches=len(dispatch),owned_processes=len(own),payload_root=str(payload.resolve()))

class CheckedReader:
 def __init__(self,stream,expected):self.stream=stream;self.expected=expected;self.h=hashlib.sha256();self.count=0
 def read(self,size=-1):
  raw=self.stream.read(size);self.h.update(raw);self.count+=len(raw);return raw
 def verify(self):require(self.count==self.expected['bytes']and self.h.hexdigest()==self.expected['sha256']and self.stream.read(1)==b'','Source second-pass SHA/EOF differs:'+self.expected['path'])

def write_tar(payload,rows,sink):
 with tarfile.open(fileobj=sink,mode='w|',format=tarfile.PAX_FORMAT)as archive:
  for row in rows:
   path=Path(payload)/row['path']
   with safe_file(payload,row['path'])as source:
    a=os.fstat(source.fileno());reader=CheckedReader(source,row);info=tarfile.TarInfo(row['path']);info.size=row['bytes'];info.mode=stat.S_IMODE(a.st_mode);info.mtime=0;info.uid=info.gid=0;archive.addfile(info,reader);reader.verify();require(identity(a)==identity(os.fstat(source.fileno()))==identity(path.lstat()),'Source changed while streaming:'+row['path'])

REMOTE_RECEIVER=r'''import ctypes,hashlib,json,os,stat,struct,sys,tarfile,time,traceback
from pathlib import Path,PurePosixPath
MAX_TOTAL=12*1024**3;MAX_SINGLE=128*1024**2;MAX_HEADER=32*1024**2
stream=sys.stdin.buffer
size=struct.unpack('!Q',stream.read(8))[0];assert 0<size<=MAX_HEADER
raw=stream.read(size);assert len(raw)==size;config=json.loads(raw);dest=Path(config['destination']);tmp=Path(str(dest)+'.incoming');bundle=Path(str(dest)+'.stream.tar.incoming');assert dest.is_absolute()and'..'not in dest.parts;assert not dest.exists()and not tmp.exists()and not bundle.exists(),'Archive target already exists, no blind retry';dest.parent.mkdir(parents=True,exist_ok=True);free=os.statvfs(dest.parent);assert free.f_bavail*free.f_frsize>=3*config['source_bytes']+100*1024**3,'Insufficient archive headroom'
def stable(p):
 a=p.stat();assert p.is_file()and not p.is_symlink();h=hashlib.sha256();count=0
 with p.open('rb')as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b);count+=len(b)
 z=p.stat();assert(a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns,a.st_ctime_ns)==(z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns,z.st_ctime_ns);return dict(path=str(p.relative_to(tmp)),bytes=count,sha256=h.hexdigest())
count=0
with bundle.open('xb')as out:
 for b in iter(lambda:stream.read(4194304),b''):count+=len(b);assert count<=MAX_TOTAL+64*1024**2;out.write(b)
 out.flush();os.fsync(out.fileno())
expected={x['path']:x for x in config['catalog']['files']};assert len(expected)==len(config['catalog']['files'])<=20000;assert sum(x['bytes']for x in expected.values())==config['source_bytes']<=MAX_TOTAL;assert hashlib.sha256(json.dumps(config['catalog'],sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False).encode()+b'\n').hexdigest()==config['catalog_sha256'];tmp.mkdir();seen=set();total=0
with tarfile.open(bundle,'r:')as archive:
 for member in archive:
  name=PurePosixPath(member.name);assert member.isfile()and not name.is_absolute()and'..'not in name.parts and member.name in expected and member.name not in seen;assert member.size==expected[member.name]['bytes']<=MAX_SINGLE;seen.add(member.name);total+=member.size;assert total<=MAX_TOTAL;p=tmp/member.name;assert p.resolve().is_relative_to(tmp.resolve());p.parent.mkdir(parents=True,exist_ok=True)
  with archive.extractfile(member)as src,p.open('xb')as out:
   for b in iter(lambda:src.read(4194304),b''):out.write(b)
   out.flush();os.fsync(out.fileno())
  p.chmod(member.mode&0o777)
assert seen==set(expected)
passes=[]
for repeat in range(2):
 actual=[stable(p)for p in sorted(tmp.rglob('*'))if p.is_file()];assert len(actual)==len(expected)and{row['path']:row for row in actual}==expected,'Fresh remote per-file SHA/bytes mismatch';passes.append(dict(pass_index=repeat+1,files=len(actual),bytes=sum(x['bytes']for x in actual),roster_sha256=hashlib.sha256(json.dumps(actual,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()))
catalog_raw=(json.dumps(config['catalog'],sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
with(tmp/'ASSET_CATALOG.json').open('xb')as f:f.write(catalog_raw);f.flush();os.fsync(f.fileno())
assert hashlib.sha256(catalog_raw).hexdigest()==config['catalog_sha256'];sourcefreeze=expected['SOURCE_FREEZE.json'];assert sourcefreeze['sha256']==config['source_freeze_sha256'];verification=dict(status='PASS_ARCHIVED_PAYLOAD_BYTES_ONLY_NO_SCIENTIFIC_PROMOTION',task_id=config['task_id'],destination=str(dest),files_verified=len(expected),bytes_verified=total,source_freeze_sha256=config['source_freeze_sha256'],catalog_sha256=config['catalog_sha256'],release_proof_sha256=config['catalog']['release_proof']['sha256'],native_scientific_controller_status=config['catalog']['native_controller_status'],fresh_independent_remote_SHA_passes=passes,source_raw_downloaded_to_Mac=False,source_deleted=False,all_source_files_roster_exact=True,atomic_no_clobber_publication=True,received_tar_bytes=count)
verification_raw=(json.dumps(verification,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
with(tmp/'ARCHIVE_VERIFICATION.json').open('xb')as f:f.write(verification_raw);f.flush();os.fsync(f.fileno())
libc=ctypes.CDLL(None,use_errno=True);rename=getattr(libc,'renameat2',None)
if rename is not None:
 rename.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint];rename.restype=ctypes.c_int;rc=rename(-100,os.fsencode(tmp),-100,os.fsencode(dest),1)
else:
 assert os.uname().machine=='x86_64';libc.syscall.restype=ctypes.c_long;rc=libc.syscall(ctypes.c_long(316),ctypes.c_int(-100),ctypes.c_char_p(os.fsencode(tmp)),ctypes.c_int(-100),ctypes.c_char_p(os.fsencode(dest)),ctypes.c_uint(1))
if rc!=0:raise OSError(ctypes.get_errno(),'Atomic no-clobber publication failed; all partial archive evidence retained')
fd=os.open(dest.parent,os.O_RDONLY)
try:os.fsync(fd)
finally:os.close(fd)
assert{str(p.relative_to(dest))for p in dest.rglob('*')if p.is_file()}==set(expected)|{'ASSET_CATALOG.json','ARCHIVE_VERIFICATION.json'}
print(json.dumps(dict(status=verification['status'],destination=str(dest),verification=verification,ASSET_CATALOG=dict(path=str(dest/'ASSET_CATALOG.json'),bytes=len(catalog_raw),sha256=hashlib.sha256(catalog_raw).hexdigest()),ARCHIVE_VERIFICATION=dict(path=str(dest/'ARCHIVE_VERIFICATION.json'),bytes=len(verification_raw),sha256=hashlib.sha256(verification_raw).hexdigest()),retained_stream_archive=str(bundle))))
'''

class SSHTransport:
 def __init__(self,host='zypan@202.38.247.29',control='/tmp/qwen_replication_archive29.sock',python_executable='/usr/bin/python3'):
  require(PurePosixPath(python_executable).is_absolute(),'Archive interpreter must be an explicitly configured absolute path');self.host=host;self.control=control;self.python=python_executable
 def check_python(self):
  code="import json,sys; assert sys.version_info >= (3,9); print(json.dumps(dict(executable=sys.executable,version=sys.version,version_info=list(sys.version_info[:3]),status='PASS_READ_ONLY_ARCHIVE_PYTHON')))";command=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-S',self.control,self.host,shlex.join([self.python,'-B','-c',code])];p=subprocess.run(command,capture_output=True,text=True,timeout=45);require(p.returncode==0,'Configured29 Python unavailable; no archive stream started, inspect and do not blindly retry: '+p.stderr[:3000]);result=json.loads(p.stdout);require(result.get('status')=='PASS_READ_ONLY_ARCHIVE_PYTHON'and result.get('version_info',[0,0])[:2]>=[3,9],'29 Python does not support bounded verifier');return dict(configured_path=self.python,actual=result,stderr=p.stderr)
 def transfer(self,configuration,payload,rows):
  python_identity=self.check_python();command=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-S',self.control,self.host,shlex.join([self.python,'-B','-c',REMOTE_RECEIVER])];p=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE);header=encoded(configuration);require(len(header)<=MAX_HEADER,'Archive catalog envelope too large')
  try:p.stdin.write(struct.pack('!Q',len(header)));p.stdin.write(header);write_tar(payload,rows,p.stdin);p.stdin.close();p.stdin=None;stdout,stderr=p.communicate(timeout=7200)
  except BaseException:
   if p.stdin is not None:
    try:p.stdin.close()
    except OSError:pass
   # Kill only this local archive SSH transport on exception; never a GPU PID.
   if p.poll()is None:
    p.terminate()
    try:p.wait(timeout=30)
    except subprocess.TimeoutExpired:p.kill();p.wait(timeout=30)
   raise
  require(p.returncode==0,'Archive SSH failed; all source and partial archive retained, no blind retry: '+stderr.decode(errors='replace')[:4000]);result=json.loads(stdout);result['transport_python_preflight']=python_identity;return result

def archive_once(root,proof_path,transport=None):
 root=Path(root).resolve();payload=root/'payload';release=validate_release(root,proof_path);initial=inventory(payload);archive=root/'archive';archive.mkdir(exist_ok=True);state=archive/'ARCHIVE_STATE.jsonl';require(not state.exists(),'Archive already attempted; inspect source/destination, no blind retry')
 with state.open('x')as f:f.flush();os.fsync(f.fileno())
 def event(kind,**values):
  with state.open('a')as f:f.write(json.dumps(dict(event=kind,unix=time.time(),**values),sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
 event('ARCHIVE_STARTED_POST_RELEASE',release_proof=release['proof_binding'],payload_root=str(payload),source_freeze_sha256=release['source_freeze_sha256'])
 try:
  rows=[hash_file(payload,x['path'])for x in initial];require(inventory(payload)==initial,'Source roster changed during first SHA pass');catalog=dict(task_id=TASK,status='POST_RELEASE_SOURCE_ASSET_CATALOG_NO_SCIENCE_PROMOTION',payload_root=str(payload),archive_destination=DEST,source_freeze_sha256=release['source_freeze_sha256'],release_proof=release['proof_binding'],native_controller_status=release['native_status'],files_count=len(rows),total_bytes=sum(x['bytes']for x in rows),files=rows);raw=encoded(catalog);catalog_desc=new_bytes(archive/'ASSET_CATALOG.json',raw);event('SOURCE_FIRST_SHA_PASS_CATALOG_READY',catalog=catalog_desc,files=len(rows),bytes=catalog['total_bytes']);config=dict(task_id=TASK,destination=DEST,catalog=catalog,catalog_sha256=digest(raw),source_freeze_sha256=release['source_freeze_sha256'],source_bytes=catalog['total_bytes']);result=(transport or SSHTransport()).transfer(config,payload,rows);require(result['status']=='PASS_ARCHIVED_PAYLOAD_BYTES_ONLY_NO_SCIENTIFIC_PROMOTION'and result.get('destination')==DEST and result['verification']['catalog_sha256']==catalog_desc['sha256']and result['verification']['files_verified']==len(rows)and result['verification']['bytes_verified']==catalog['total_bytes'],'Remote archive verification differs');require(result['ASSET_CATALOG']['path']==DEST+'/ASSET_CATALOG.json'and result['ASSET_CATALOG']['bytes']==len(raw)and result['ASSET_CATALOG']['sha256']==digest(raw),'Remote catalog raw bytes binding differs');vr=encoded(result['verification']);require(result['ARCHIVE_VERIFICATION']['path']==DEST+'/ARCHIVE_VERIFICATION.json'and result['ARCHIVE_VERIFICATION']['bytes']==len(vr)and result['ARCHIVE_VERIFICATION']['sha256']==digest(vr),'Remote independent verification bytes binding differs');passes=result['verification'].get('fresh_independent_remote_SHA_passes');require(isinstance(passes,list)and len(passes)==2 and all(x['files']==len(rows)and x['bytes']==catalog['total_bytes']and x['roster_sha256']==digest(canon(rows))for x in passes),'Two remote fresh SHA/roster passes required');require(inventory(payload)==initial,'Source changed while streaming second SHA pass');require(validate_release(root,proof_path)['proof_binding']==release['proof_binding'],'Release proof or native closed ledger changed');verification=dict(status='PASS_POST_RELEASE_FULL_PAYLOAD_ARCHIVE_BYTES_ONLY',task_id=TASK,source_catalog=catalog_desc,remote=result,source_two_stable_SHA_passes=True,remote_two_independent_disk_SHA_passes=True,source_deleted=False,large_tensor_downloaded_to_Mac=False,payload_unchanged=True,native_scientific_controller_status=release['native_status']);desc=save_new(archive/'ARCHIVE_VERIFICATION.json',verification);event('PASS_ATOMIC_ARCHIVE_NO_SOURCE_DELETE',verification=desc);return verification
 except BaseException as exc:
  failure=dict(status='ARCHIVE_IO_FAILED_NO_BLIND_RETRY',error=type(exc).__name__+':'+str(exc),traceback=traceback.format_exc(),all_source_and_partial_remote_assets_retained=True,source_deleted=False)
  if not(archive/'ARCHIVE_FAILURE.json').exists():save_new(archive/'ARCHIVE_FAILURE.json',failure)
  event('FAILED_INSPECT_STATE_NO_BLIND_RETRY',error=failure['error']);raise

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True);ap.add_argument('--release-proof',type=Path);ap.add_argument('--run',action='store_true');ap.add_argument('--archive-python',default='/usr/bin/python3');args=ap.parse_args();root=args.root.resolve()
 if not args.run:
  rows=inventory(root/'payload');print(json.dumps(dict(status='PREVIEW_ONLY_NO_ARCHIVE_NO_NETWORK_NO_GPU',payload_root=str(root/'payload'),files=len(rows),bytes=sum(x['bytes']for x in rows),maximum_total_bytes=MAX_TOTAL,maximum_single_bytes=MAX_SINGLE,release_proof_required_for_run=True,destination=DEST)));return
 require(args.release_proof is not None,'Independent release proof required');print(json.dumps(archive_once(root,args.release_proof,SSHTransport(python_executable=args.archive_python)),sort_keys=True))
if __name__=='__main__':main()
