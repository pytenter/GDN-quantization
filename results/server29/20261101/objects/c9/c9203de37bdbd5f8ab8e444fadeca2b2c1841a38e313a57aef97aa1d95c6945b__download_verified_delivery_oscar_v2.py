"""One bounded read-only streaming delivery waiter; excludes remote raw/model."""
import sys,json,time,subprocess,shlex,tarfile,hashlib,os,fcntl
from pathlib import Path
R=Path(__file__).resolve().parents[1];cfg=json.loads((R/'ROOT.json').read_text());lock=(R/'operations/DOWNLOAD.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if (R/'operations/LOCAL_DELIVERY_VERIFICATION.json').exists():raise SystemExit('Delivery already verified; no duplicate')
(R/'operations/LOCAL_DOWNLOADER_STARTED.json').write_text(json.dumps(dict(pid=os.getpid(),argv=sys.argv,unix=time.time()),indent=2)+'\n')
def cmd(code,*args):return ['ssh','-S',cfg['ssh_control_path'],'-o','BatchMode=yes','-o','ConnectTimeout=15','zypan@202.38.247.55',shlex.join([cfg['remote_python'],'-u','-B','-c',code,cfg['remote_root'],*args])]
ready='''import sys,json,hashlib
from pathlib import Path
R=Path(sys.argv[1]);seal=R/'FINAL_SEAL_RECEIPT.json'
if (R/'FINALIZER_FAILURE.json').exists():print(json.dumps(dict(status='READONLY_FINISHER_RECOVERY_NEEDED',failure=json.loads((R/'FINALIZER_FAILURE.json').read_text()))));raise SystemExit
if not seal.exists():print(json.dumps(dict(status='PENDING')));raise SystemExit
x=json.loads((R/'FINALIZER_STARTED.json').read_text())['identity'];p=Path('/proc')/str(x['pid'])
try:alive=int((p/'stat').read_text().rsplit(')',1)[1].split()[19])==x['starttick'] and (p/'cmdline').read_bytes().rstrip(b'\\0').decode().split('\\0')==x['cmdline']
except (FileNotFoundError,ProcessLookupError):alive=False
if alive:print(json.dumps(dict(status='PENDING_FINALIZER_EXIT')));raise SystemExit
m=json.loads((R/'SHA256_MANIFEST.json').read_text());s=json.loads(seal.read_text());assert hashlib.sha256((R/'SHA256_MANIFEST.json').read_bytes()).hexdigest()==s['manifest_sha256'];rows=[x for x in m['files'] if not x['path'].endswith(('.bin','.npz','.npy','.f32','.pt','.pth','.safetensors'))]
for n in ['SHA256_MANIFEST.json','SHA256SUMS.txt','FINAL_ARCHIVE_VERIFICATION.json','FINAL_SEAL_RECEIPT.json']:
 p=R/n;rows.append(dict(path=n,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
print(json.dumps(dict(status='READY',files=rows,remote_only_raw=[x for x in m['files'] if x['path'].endswith(('.bin','.npz','.npy','.f32','.pt','.pth','.safetensors'))],scientific_status=m['status'],manifest_sha256=s['manifest_sha256'])))
'''
try:
 while True:
  try:
   p=subprocess.run(cmd(ready),capture_output=True,timeout=90)
   if p.returncode==255:raise ConnectionError(p.stderr.decode(errors='replace'))
  except (subprocess.TimeoutExpired,ConnectionError) as connection_error:
   state=dict(status='WAITING_FOR_SSH_RECONNECTION',error=repr(connection_error),pid=os.getpid(),unix=time.time(),GPU_retries=0,remote_mutations=0)
   (R/'operations/LOCAL_DOWNLOADER_CONNECTION_STATUS.json').write_text(json.dumps(state,indent=2)+'\n')
   with (R/'operations/LOCAL_DOWNLOADER_CONNECTION_EVENTS.jsonl').open('a') as connection_events:connection_events.write(json.dumps(state)+'\n')
   if '--wait' not in sys.argv:raise
   time.sleep(30);continue
  assert p.returncode==0,p.stderr.decode()
  (R/'operations/LOCAL_DOWNLOADER_CONNECTION_STATUS.json').write_text(json.dumps(dict(status='CONNECTED_READONLY_POLLING',pid=os.getpid(),unix=time.time(),GPU_retries=0,remote_mutations=0),indent=2)+'\n')
  data=json.loads(p.stdout)
  if data['status']=='READY':break
  if data['status']=='READONLY_FINISHER_RECOVERY_NEEDED':raise RuntimeError(json.dumps(data))
  if '--wait' not in sys.argv:raise SystemExit(data['status'])
  time.sleep(30)
 D=R/'delivery';D.mkdir(exist_ok=True);rows={x['path']:x for x in data['files']}
 code='''import sys,json,tarfile,hashlib
from pathlib import Path
R=Path(sys.argv[1]);expected=sys.argv[2];m=json.loads((R/'SHA256_MANIFEST.json').read_text());assert hashlib.sha256((R/'SHA256_MANIFEST.json').read_bytes()).hexdigest()==expected;names=[x['path'] for x in m['files'] if not x['path'].endswith(('.bin','.npz','.npy','.f32','.pt','.pth','.safetensors'))]+['SHA256_MANIFEST.json','SHA256SUMS.txt','FINAL_ARCHIVE_VERIFICATION.json','FINAL_SEAL_RECEIPT.json']
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as t:
 for n in names:
  rel=Path(n);assert not rel.is_absolute() and '..' not in rel.parts;p=R/rel;assert p.is_file() and not p.is_symlink();i=t.gettarinfo(str(p),arcname=n)
  with p.open('rb') as f:t.addfile(i,f)
'''
 p=subprocess.Popen(cmd(code,data['manifest_sha256']),stdout=subprocess.PIPE,stderr=(R/'operations/DOWNLOAD_SSH_STDERR.log').open('wb'));seen=set()
 with tarfile.open(fileobj=p.stdout,mode='r|gz') as tar:
  for member in tar:
   assert member.isfile() and member.name in rows and member.name not in seen;row=rows[member.name];assert member.size==row['bytes'];rel=Path(member.name);assert not rel.is_absolute() and '..' not in rel.parts;q=D/rel;q.parent.mkdir(parents=True,exist_ok=True);tmp=q.with_name(q.name+'.download-tmp');h=hashlib.sha256()
   with tar.extractfile(member) as stream,tmp.open('wb') as f:
    for b in iter(lambda:stream.read(4*1024**2),b''):h.update(b);f.write(b)
   assert tmp.stat().st_size==row['bytes'] and h.hexdigest()==row['sha256'],member.name
   if q.exists():assert hashlib.sha256(q.read_bytes()).hexdigest()==row['sha256'];tmp.unlink()
   else:os.replace(tmp,q)
   seen.add(member.name)
 assert p.wait(timeout=60)==0 and seen==set(rows)
 h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();seal=json.loads((D/'FINAL_SEAL_RECEIPT.json').read_text());archive=json.loads((D/'FINAL_ARCHIVE_VERIFICATION.json').read_text());assert h(D/'SHA256_MANIFEST.json')==seal['manifest_sha256']==archive['manifest_sha256'] and h(D/'FINAL_ARCHIVE_VERIFICATION.json')==seal['archive_verification_sha256']
 raw_catalog={x['path']:x for x in data['remote_only_raw']};preexisting_raw_sources=[]
 for x in json.loads((D/'SOURCE_FREEZE.json').read_text())['files']:
  if x['path'].endswith(('.bin','.npz','.npy','.f32','.pt','.pth','.safetensors')):
   z=raw_catalog[x['path']];assert z['bytes']==x['bytes'] and z['sha256']==x['sha256']
   local=R/x['path']
   if local.exists():assert local.stat().st_size==x['bytes'] and h(local)==x['sha256'];preexisting_raw_sources.append(x['path'])
  else:assert (D/x['path']).stat().st_size==x['bytes'] and h(D/x['path'])==x['sha256']
 audit=subprocess.run(['/usr/bin/python3','-u','-B',str(R/'operations/final_readonly_audit.py')],stdout=(R/'operations/FINAL_AUDIT.log').open('ab'),stderr=subprocess.STDOUT);assert audit.returncode==0,'Independent final audit failed; preserve evidence, no science retry'
 v=dict(status='PASS_DOWNLOADED_SMALL_SHA_AND_REMOTE_ONLY_RAW_CATALOG',preexisting_local_raw_source_files=preexisting_raw_sources,downloaded_files=len(rows),downloaded_bytes=sum(x['bytes'] for x in rows.values()),files=list(rows.values()),remote_only_raw=data['remote_only_raw'],scientific_status=data['scientific_status'],manifest_sha256=seal['manifest_sha256'],report_sha256=h(D/'FINAL_SCIENTIFIC_REPORT.md'),source_count=len(json.loads((D/'SOURCE_FREEZE.json').read_text())['files']),GPU_retries=0,model_or_raw_downloads=0,unix=time.time());(R/'operations/LOCAL_DELIVERY_VERIFICATION.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps({k:v[k] for k in ['status','downloaded_files','downloaded_bytes','scientific_status','manifest_sha256','report_sha256']}),flush=True)
except Exception as exc:
 (R/'operations/LOCAL_DOWNLOADER_FAILURE.json').write_text(json.dumps(dict(error=repr(exc),unix=time.time(),GPU_retries=0),indent=2)+'\n');raise
