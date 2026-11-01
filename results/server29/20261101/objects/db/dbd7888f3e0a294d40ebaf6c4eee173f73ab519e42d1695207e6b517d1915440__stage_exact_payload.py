"""One-shot deployment of frozen small files, then independent CPU-only SHA checks."""
import hashlib,io,json,shlex,subprocess,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
STAGING=ROOT/'staging/payload_v1'
remote=json.loads((ROOT/'ROOT.json').read_text())['remote_root']
freeze=json.loads((STAGING/'SOURCE_FREEZE.json').read_text())
buffer=io.BytesIO()
with tarfile.open(fileobj=buffer,mode='w')as tar:
 for path in sorted(STAGING.rglob('*')):
  if path.is_file():
   if path.is_symlink():raise RuntimeError('Symlink in payload')
   tar.add(path,arcname=str(path.relative_to(STAGING)),recursive=False)
payload=buffer.getvalue()
program=r'''
import hashlib,json,os,sys,tarfile,time
from pathlib import Path
root=Path(REMOTE)
root.mkdir(parents=True,exist_ok=False)
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|')as tar:
 for member in tar:
  relative=Path(member.name)
  if relative.is_absolute()or'..'in relative.parts or not member.isfile():raise RuntimeError('Unsafe payload member')
  path=root/relative;path.parent.mkdir(parents=True,exist_ok=True)
  with path.open('xb')as target,tar.extractfile(member)as source:
   for block in iter(lambda:source.read(4*1024**2),b''):target.write(block)
def sha(path):
 before=path.stat();h=hashlib.sha256()
 with path.open('rb')as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 after=path.stat()
 if(before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):raise RuntimeError('File changed during fresh SHA:'+str(path))
 return h.hexdigest()
def check(path,row):
 if path.stat().st_size!=row['bytes']or sha(path)!=row['sha256']:raise RuntimeError('Fresh SHA/bytes mismatch:'+str(path))
def read(path):return json.loads(path.read_text())
freeze=read(root/'SOURCE_FREEZE.json')
if freeze!=EXPECTED:raise RuntimeError('Remote source freeze differs from local external anchor')
for name,key in [('CONTROL_SOURCE_MANIFEST.json','control_manifest_sha256'),('PAYLOAD_BUILD_MANIFEST.json','payload_build_manifest_sha256')]:
 if sha(root/name)!=freeze[key]:raise RuntimeError('Common anchor differs:'+name)
payload=read(root/'PAYLOAD_BUILD_MANIFEST.json');control=read(root/'CONTROL_SOURCE_MANIFEST.json')
for row in payload['files']+control['files']:check(root/row['path'],row)
external={};card_counts={}
for number in range(4):
 card=root/('gpu'+str(number));source=read(card/'SOURCE_MANIFEST.json')
 for row in source['execution_sources']:check(card/row['path'],row)
 for row in source['external_dependencies']:
  old=external.get(row['path'])
  if old and(old['bytes'],old['sha256'])!=(row['bytes'],row['sha256']):raise RuntimeError('Conflicting external identities')
  external[row['path']]=row
 card_counts[card.name]=dict(execution_sources=len(source['execution_sources']),external_dependencies=len(source['external_dependencies']))
for i,row in enumerate(external.values()):
 check(Path(row['path']),row)
 if i%25==0:print('fresh external SHA progress '+str(i+1)+'/'+str(len(external)),file=sys.stderr,flush=True)
out=dict(status='PASS_EXACT_PAYLOAD_AND_FRESH_EXTERNAL_SHA_BYTES_CPU_ONLY',root=str(root),payload_files=len(payload['files']),control_files=len(control['files']),card_counts=card_counts,unique_external_dependencies=len(external),control_manifest_sha256=freeze['control_manifest_sha256'],payload_build_manifest_sha256=freeze['payload_build_manifest_sha256'],GPU_dispatches=0,engine_imports=0,remote_unix=time.time())
target=root/'REMOTE_STAGE_VERIFICATION.json'
with target.open('x')as f:json.dump(out,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps(out))
'''.replace('Path(REMOTE)','Path('+repr(remote)+')').replace('freeze!=EXPECTED','freeze!='+repr(freeze))
command='/data/ydai/miniconda3/envs/bitdecode/bin/python -B -c '+shlex.quote(program)
proc=subprocess.run(['ssh','-S','/tmp/qwen_gdn_replication_20261013.sock','-o','BatchMode=yes','-o','ConnectTimeout=15','zypan@202.38.247.55',command],input=payload,capture_output=True,timeout=900)
(ROOT/'analysis/REMOTE_STAGE_STDERR.txt').write_bytes(proc.stderr)
if proc.returncode:
 (ROOT/'analysis/REMOTE_STAGE_ERROR_STDOUT.txt').write_bytes(proc.stdout)
 raise RuntimeError('Deployment or SHA verification failed; inspect preserved state, do not blindly repeat')
result=json.loads(proc.stdout);result['local_transfer_tar_bytes']=len(payload);result['local_transfer_tar_sha256']=hashlib.sha256(payload).hexdigest()
with(ROOT/'analysis/REMOTE_STAGE_VERIFICATION.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result))
