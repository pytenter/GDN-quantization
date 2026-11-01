"""Small CPU delivery only. Atomically publishes an independent archive on29."""
import base64,hashlib,io,json,os,shlex,shutil,subprocess,tarfile
from pathlib import Path
R=Path(__file__).resolve().parents[1];cfg=json.loads((R/'ROOT.json').read_text());D=R/'delivery';O=R/'operations';O.mkdir(exist_ok=True)
def sha(b):return hashlib.sha256(b).hexdigest()
def write(p,v):
 b=(json.dumps(v,indent=2,ensure_ascii=False)+'\n').encode();p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(b)
 return dict(path=str(p),bytes=len(b),sha256=sha(b))
def run55(code,args=(),payload=None,timeout=120):
 cmd=shlex.join([cfg['remote_python'],'-B','-c',code]+list(args))
 p=subprocess.run(['ssh','-S',cfg['ssh_control_path'],'-o','BatchMode=yes','-o','ConnectTimeout=10','zypan@202.38.247.55',cmd],input=payload,capture_output=True,timeout=timeout)
 if p.returncode:raise RuntimeError(p.stderr.decode()[-3000:])
 return p.stdout
assert not D.exists(),'Delivery already exists; inspect before rerun, do not overwrite seal'
D.mkdir();rows=[]
excluded={'SOURCE_FREEZE.json','SHA256_MANIFEST.json','SHA256SUMS.txt','FINAL_SEAL_RECEIPT.json','LOCAL_DELIVERY_VERIFICATION.json'}
for p in sorted(R.rglob('*')):
 if not p.is_file() or p.is_symlink():continue
 rel=p.relative_to(R)
 if any(x in rel.parts for x in ('delivery','operations','__pycache__')) or str(rel) in excluded:continue
 b=p.read_bytes();assert len(b)<4*1024**2,'Only bounded small files permitted:'+str(p)
 rows.append(dict(path=str(rel),bytes=len(b),sha256=sha(b)))
source=[x for x in rows if x['path'].startswith(('src/','scripts/','inherited/','fixtures/')) or x['path'] in ['CODEC_NUMERICAL_CONTRACT.md','HISTORICAL_PORT_INT8_PROTOCOL.md','GPU_PLAN.json','USER_REQUEST.txt','CPU_CODEC_QUALIFICATION.json']]
source_raw=(json.dumps(dict(task_id=R.name,status='CPU_QUALIFIED_GPU_NOT_AUTHORIZED_SOURCE_FREEZE',files=source),indent=2)+'\n').encode()
# SOURCE_FREEZE is finalized once here, before any authorized GPU execution.
(R/'SOURCE_FREEZE.json').write_bytes(source_raw);rows.append(dict(path='SOURCE_FREEZE.json',bytes=len(source_raw),sha256=sha(source_raw)));rows.sort(key=lambda x:x['path'])
for x in rows:
 p=R/x['path'];q=D/x['path'];q.parent.mkdir(parents=True,exist_ok=True);b=p.read_bytes();assert sha(b)==x['sha256'];q.write_bytes(b)
manifest=dict(task_id=R.name,status='CPU_SCOPE_COMPLETE_GPU_NOT_AUTHORIZED',source_freeze_sha256=sha(source_raw),files=rows,large_tensor_or_model_downloaded=False)
write(R/'SHA256_MANIFEST.json',manifest);shutil.copyfile(R/'SHA256_MANIFEST.json',D/'SHA256_MANIFEST.json')
sums=''.join(x['sha256']+'  '+x['path']+'\n' for x in rows)+sha((D/'SHA256_MANIFEST.json').read_bytes())+'  SHA256_MANIFEST.json\n'
(R/'SHA256SUMS.txt').write_text(sums);(D/'SHA256SUMS.txt').write_text(sums)
allrows=rows+[dict(path=n,bytes=(D/n).stat().st_size,sha256=sha((D/n).read_bytes())) for n in ('SHA256_MANIFEST.json','SHA256SUMS.txt')]
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w') as t:
 for x in allrows:t.add(D/x['path'],arcname=x['path'])
delivery55=cfg['remote_root']+'/delivery';delivery29=cfg['archive_root']+'/delivery'
put=r'''import tarfile,sys,hashlib,json,uuid,os
from pathlib import Path
p=Path(sys.argv[1]);assert not p.exists(),'Final delivery already exists'
t=p.with_name(p.name+'.staging.'+uuid.uuid4().hex);t.mkdir(parents=True)
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|*') as a:
 for m in a:
  assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
  q=t/m.name;q.parent.mkdir(parents=True,exist_ok=True)
  with a.extractfile(m) as f,q.open('xb') as out:out.write(f.read());out.flush();os.fsync(out.fileno())
v=json.loads((t/'SHA256_MANIFEST.json').read_text());rows=[]
for x in v['files']:
 q=t/x['path'];b=q.read_bytes();assert len(b)==x['bytes'] and hashlib.sha256(b).hexdigest()==x['sha256'];rows.append(x)
os.rename(t,p)
print(json.dumps(dict(status='PASS_INDEPENDENT_FRESH_SHA_BYTES_ATOMIC_PUBLICATION',root=str(p),files=len(rows),bytes=sum(x['bytes'] for x in rows),manifest_sha256=hashlib.sha256((p/'SHA256_MANIFEST.json').read_bytes()).hexdigest(),sums_sha256=hashlib.sha256((p/'SHA256SUMS.txt').read_bytes()).hexdigest())))'''
v55=json.loads(run55(put,[delivery55],buf.getvalue(),120));write(O/'CPU_DELIVERY_55_RECEIPT.json',v55)
forward=r'''import subprocess,shlex,sys
from pathlib import Path
code=sys.argv[3];cmd=shlex.join(['/usr/bin/python3','-B','-c',code,sys.argv[2]])
p=subprocess.Popen(['ssh','-S','/tmp/qwen_replication_archive29.sock','-o','BatchMode=yes','zypan@202.38.247.29',cmd],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
with open(sys.argv[4],'rb') as f:
 out,err=p.communicate(f.read(),timeout=180)
assert p.returncode==0,err.decode();sys.stdout.buffer.write(out)'''
# CPU bundle remains on55 in this new task only. It contains no model or large raw.
bundle=cfg['remote_root']+'/CPU_SMALL_DELIVERY.tar'
run55("import sys;from pathlib import Path;p=Path(sys.argv[1]);p.write_bytes(sys.stdin.buffer.read());print(p.stat().st_size)",[bundle],buf.getvalue(),120)
v29=json.loads(run55(forward,[delivery55,delivery29,put,bundle],timeout=210));write(O/'CPU_DELIVERY_29_RECEIPT.json',v29)
assert v55['manifest_sha256']==v29['manifest_sha256']==sha((D/'SHA256_MANIFEST.json').read_bytes())
assert v55['sums_sha256']==v29['sums_sha256']==sha((D/'SHA256SUMS.txt').read_bytes())
seal=dict(task_id=R.name,status='PASS_CPU_SCOPE_FINAL_SEAL',scope='CPU_CODEC_QUALIFIED_GPU_NOT_AUTHORIZED',indexed_files=len(rows),indexed_bytes=sum(x['bytes'] for x in rows),source_files=len(source),source_freeze_sha256=sha(source_raw),manifest_sha256=v55['manifest_sha256'],sums_sha256=v55['sums_sha256'],CPU_delivery55=v55,CPU_archive29=v29,old_assets_modified=False,GPU_queries=0,GPU_dispatches=0,actual_GPU_qualification='NOT_RUN',INT8_TPS='NOT_RUN')
sealrow=write(R/'FINAL_SEAL_RECEIPT.json',seal);shutil.copyfile(R/'FINAL_SEAL_RECEIPT.json',D/'FINAL_SEAL_RECEIPT.json')
publish=r'''import sys,json,hashlib,os,subprocess,shlex,base64
from pathlib import Path
p=Path(sys.argv[1]);b=sys.stdin.buffer.read();p.parent.mkdir(parents=True,exist_ok=True)
with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
assert p.read_bytes()==b
print(json.dumps(dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())))'''
def replicate_anchor(name):
 b=(D/name).read_bytes();a=json.loads(run55(publish,[delivery55+'/'+name],b))
 forward_anchor='import subprocess,shlex,sys;cmd=shlex.join(["/usr/bin/python3","-B","-c",sys.argv[1],sys.argv[2]]);p=subprocess.run(["ssh","-S","/tmp/qwen_replication_archive29.sock","-o","BatchMode=yes","zypan@202.38.247.29",cmd],input=sys.stdin.buffer.read(),capture_output=True,timeout=30);assert p.returncode==0,p.stderr.decode();sys.stdout.buffer.write(p.stdout)'
 z=json.loads(run55(forward_anchor,[publish,delivery29+'/'+name],b));assert a['sha256']==z['sha256']==sha(b);return dict(local_sha256=sha(b),remote55=a,archive29=z)
anchor=replicate_anchor('FINAL_SEAL_RECEIPT.json')
for x in allrows:
 b=(D/x['path']).read_bytes();assert sha(b)==x['sha256'] and len(b)==x['bytes']
verification=dict(status='PASS_ALL_THREE_HOSTS_CPU_SMALL_DELIVERY',indexed_files=len(rows),all_indexed_SHA_bytes=True,all_three_hosts_manifest_SHA_equal=True,seal=anchor,report_sha256=sha((D/'FINAL_SCIENTIFIC_REPORT.md').read_bytes()),source_freeze_sha256=sha(source_raw),remote55=v55,archive29=v29,GPU_NOT_RUN=True)
write(R/'LOCAL_DELIVERY_VERIFICATION.json',verification);shutil.copyfile(R/'LOCAL_DELIVERY_VERIFICATION.json',D/'LOCAL_DELIVERY_VERIFICATION.json');replicate_anchor('LOCAL_DELIVERY_VERIFICATION.json')
write(O/'FINAL_DELIVERY_RESULT.json',dict(status='PASS',files=len(rows),bytes=sum(x['bytes'] for x in rows),manifest_sha256=v55['manifest_sha256'],seal_sha256=anchor['local_sha256'],local_delivery=str(D),remote55=delivery55,archive29=delivery29))
print(json.dumps(dict(status='PASS_THREE_HOST_CPU_DELIVERY',files=len(rows),bytes=sum(x['bytes'] for x in rows),report=str(D/'FINAL_SCIENTIFIC_REPORT.md'),manifest_sha256=v55['manifest_sha256']),ensure_ascii=False))
