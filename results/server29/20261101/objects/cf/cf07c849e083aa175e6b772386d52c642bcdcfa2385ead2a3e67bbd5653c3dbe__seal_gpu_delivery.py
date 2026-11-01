"""Final small-file manifest and independent atomic55/29 publications."""
import hashlib,io,json,os,shutil,tarfile
from pathlib import Path
from collect_terminal import R,cfg,ssh
D=R/'delivery';assert (D/'TASK_RESULT.json').exists()
sha=lambda b:hashlib.sha256(b).hexdigest()
def write(p,v):
 b=(json.dumps(v,indent=2,ensure_ascii=False)+'\n').encode()
 with p.open('xb') as f:f.write(b)
 return dict(bytes=len(b),sha256=sha(b))
analysis=D/'independent_analysis';analysis.mkdir(exist_ok=True)
for p in (R/'operations').glob('*.py'):shutil.copyfile(p,analysis/p.name)
for name in ('TERMINAL_ADMISSION.json','REMOTE_SMALL_FILE_INVENTORY.json','EXPECTED_CURRENT_RAW_OBJECTS.json','DELIVERY_TOOL_MANIFEST.json','REMOTE_CPU_ADMISSION.json','LINUX_CPU_ADMISSION.json','LAUNCH_RECEIPT.json'):
 shutil.copyfile(R/'operations'/name,analysis/name)
# Independent review binds real dispatch identities, all three field suites,
# unchanged science, finite actual receipts, raw archive and true process release.
import sys
sys.path.insert(0,str(D/'src'));import gpu_control as C
source=C.verify_sources(D);permit=json.loads((D/'GPU_PERMIT.json').read_text());assert source==permit['source_freeze_sha256']
parent=D/'inputs/parent_gpu_delivery'
before={x['path']:x for x in json.loads((parent/'SOURCE_FREEZE.json').read_text())['files']};after={x['path']:x for x in json.loads((D/'SOURCE_FREEZE.json').read_text())['files']}
assert set(before).issubset(after) and len(after)==500
changes=[k for k in before if before[k]!=after[k]];assert set(changes)=={'GPU_PLAN.json','src/gpu_control.py','scripts/gpu_controller.py'}
assert set(after)-set(before)=={'src/continuation_control.py','analysis/RESOURCE_AUTHORIZATION_AMENDMENT.json','scripts/test_continuation_controls.py'}
assert (parent/'SHA256_MANIFEST.json').read_bytes()==(R.parent/'gpu_execution_20261019_v1/delivery/SHA256_MANIFEST.json').read_bytes()
for x in json.loads((parent/'SHA256_MANIFEST.json').read_text())['files']:
 b=(parent/x['path']).read_bytes();assert len(b)==x['bytes'] and sha(b)==x['sha256'],'Parent GPU seal member differs:'+x['path']
archive=json.loads((D/'archive/FINAL_ARCHIVE_VERIFICATION.json').read_text());assert archive['status'].startswith('PASS_') and not archive['unindexed_bin']
dep=json.loads((D/'archive/POST_RUN_SOURCE_AND_DEPENDENCY_VERIFICATION.json').read_text());assert dep['source_freeze_sha256']==source
terminal=json.loads((D/'independent_analysis/TERMINAL_ADMISSION.json').read_text());assert terminal['status'].startswith('PASS_')
original_CPU=R.parent/'delivery'
for name in ('SHA256_MANIFEST.json','SHA256SUMS.txt','FINAL_SEAL_RECEIPT.json','LOCAL_DELIVERY_VERIFICATION.json','SOURCE_FREEZE.json'):
 assert (original_CPU/name).read_bytes()==(parent/'provenance/cpu_seal'/name).read_bytes(),'Original CPU seal changed:'+name
for x in json.loads((original_CPU/'SHA256_MANIFEST.json').read_text())['files']:
 b=(original_CPU/x['path']).read_bytes();assert len(b)==x['bytes'] and sha(b)==x['sha256'],'Original CPU file changed:'+x['path']
review=dict(status='PASS_INDEPENDENT_FINAL_REVIEW',science_source_files=len(after),resource_only_changed_parent_rows=changes,added_resource_control_rows=sorted(set(after)-set(before)),scientific_worker_byte_identical=True,math_changes=0,source_freeze_sha256=source,external_current_SHA_files=len(dep['external_files']),archive_objects=archive['object_count'],all_current_raw_fresh_SHA_bytes=True,actual_process_and_NVML_release=True,CPU_original_seal_unchanged=True)
write(D/'FINAL_INDEPENDENT_REVIEW.json',review)
exclude={'SHA256_MANIFEST.json','SHA256SUMS.txt','FINAL_SEAL_RECEIPT.json','LOCAL_DELIVERY_VERIFICATION.json'}
rows=[]
for p in sorted(D.rglob('*')):
 if not p.is_file() or p.is_symlink():continue
 rel=str(p.relative_to(D))
 if rel in exclude or '__pycache__' in p.parts:continue
 b=p.read_bytes();assert len(b)<4*1024**2,'Large raw/model is not a small delivery asset'
 rows.append(dict(path=rel,bytes=len(b),sha256=sha(b)))
manifest=dict(task_id=cfg['task_id'],execution_run_id=permit['execution_run_id'],source_freeze_sha256=source,files=rows,large_raw='Remote29 only; catalog is indexed',scope=json.loads((D/'TASK_RESULT.json').read_text())['status'])
write(D/'SHA256_MANIFEST.json',manifest)
sums=''.join(x['sha256']+'  '+x['path']+'\n' for x in rows)+sha((D/'SHA256_MANIFEST.json').read_bytes())+'  SHA256_MANIFEST.json\n';(D/'SHA256SUMS.txt').write_text(sums)
bundle_rows=rows+[dict(path=n,bytes=(D/n).stat().st_size,sha256=sha((D/n).read_bytes())) for n in ('SHA256_MANIFEST.json','SHA256SUMS.txt')]
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w') as t:
 for x in bundle_rows:t.add(D/x['path'],arcname=x['path'])
dest55=cfg['remote_root']+'/FINAL_DELIVERY';dest29=cfg['archive_root']+'/FINAL_DELIVERY'
put=r'''import tarfile,sys,hashlib,json,uuid,os
from pathlib import Path
p=Path(sys.argv[1]);assert not p.exists(),'Final delivery already exists'
t=p.with_name(p.name+'.staging.'+uuid.uuid4().hex);t.mkdir(parents=True)
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|*') as a:
 for m in a:
  assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
  q=t/m.name;q.parent.mkdir(parents=True,exist_ok=True)
  with a.extractfile(m) as f,q.open('xb') as out:out.write(f.read());out.flush();os.fsync(out.fileno())
v=json.loads((t/'SHA256_MANIFEST.json').read_text())
for x in v['files']:
 q=t/x['path'];b=q.read_bytes();assert len(b)==x['bytes'] and hashlib.sha256(b).hexdigest()==x['sha256'],x['path']
os.rename(t,p)
print(json.dumps(dict(status='PASS_FRESH_INDEPENDENT_DELIVERY_SHA_BYTES',root=str(p),files=len(v['files']),manifest_sha256=hashlib.sha256((p/'SHA256_MANIFEST.json').read_bytes()).hexdigest(),sums_sha256=hashlib.sha256((p/'SHA256SUMS.txt').read_bytes()).hexdigest())))'''
v55=json.loads(ssh(put,[dest55],buf.getvalue(),120));write(R/'operations/GPU_DELIVERY_55_RECEIPT.json',v55)
bundle=cfg['remote_root']+'/GPU_SMALL_FINAL_DELIVERY.tar';ssh('import sys;from pathlib import Path;p=Path(sys.argv[1]);assert not p.exists();p.write_bytes(sys.stdin.buffer.read())',[bundle],buf.getvalue(),120)
forward=r'''import sys,subprocess,shlex
command=shlex.join(['/usr/bin/python3','-B','-c',sys.argv[1],sys.argv[2]])
with open(sys.argv[3],'rb') as f:p=subprocess.run(['ssh','-S','/tmp/qwen_replication_archive29.sock','-o','BatchMode=yes','zypan@202.38.247.29',command],input=f.read(),capture_output=True,timeout=180)
assert p.returncode==0,p.stderr.decode();sys.stdout.buffer.write(p.stdout)'''
v29=json.loads(ssh(forward,[put,dest29,bundle],timeout=200));write(R/'operations/GPU_DELIVERY_29_RECEIPT.json',v29)
assert v55['manifest_sha256']==v29['manifest_sha256']==sha((D/'SHA256_MANIFEST.json').read_bytes())
assert v55['sums_sha256']==v29['sums_sha256']==sha((D/'SHA256SUMS.txt').read_bytes())
seal=dict(status='PASS_FINAL_ACTUAL_GPU_RESULT_DELIVERY_SEAL',task_status=manifest['scope'],indexed_files=len(rows),indexed_bytes=sum(x['bytes'] for x in rows),manifest_sha256=v55['manifest_sha256'],sums_sha256=v55['sums_sha256'],source_freeze_sha256=source,raw_archive_catalog=archive['catalog'],remote55=v55,archive29=v29,no_scientific_failure_erased=True,old_assets_untouched=True)
write(D/'FINAL_SEAL_RECEIPT.json',seal)
pub=r'''import sys,json,hashlib,os
from pathlib import Path
p=Path(sys.argv[1]);b=sys.stdin.buffer.read()
with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
assert p.read_bytes()==b
print(json.dumps(dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())))'''
forward_anchor=r'''import sys,subprocess,shlex
command=shlex.join(['/usr/bin/python3','-B','-c',sys.argv[1],sys.argv[2]])
p=subprocess.run(['ssh','-S','/tmp/qwen_replication_archive29.sock','-o','BatchMode=yes','zypan@202.38.247.29',command],input=sys.stdin.buffer.read(),capture_output=True,timeout=30)
assert p.returncode==0,p.stderr.decode();sys.stdout.buffer.write(p.stdout)'''
def anchors(name):
 b=(D/name).read_bytes();a=json.loads(ssh(pub,[dest55+'/'+name],b));z=json.loads(ssh(forward_anchor,[pub,dest29+'/'+name],b));assert a['sha256']==z['sha256']==sha(b);return dict(local_sha256=sha(b),remote55=a,archive29=z)
seal_hosts=anchors('FINAL_SEAL_RECEIPT.json')
for x in bundle_rows:
 b=(D/x['path']).read_bytes();assert len(b)==x['bytes'] and sha(b)==x['sha256']
verification=dict(status='PASS_ALL_THREE_HOSTS_FINAL_SMALL_DELIVERY',indexed_files=len(rows),all_downloaded_SHA_bytes_verified=True,all_three_hosts_manifest_equal=True,seal=seal_hosts,report_sha256=sha((D/'FINAL_SCIENTIFIC_REPORT.md').read_bytes()),manifest_sha256=v55['manifest_sha256'],raw_catalog=archive['catalog'],source_freeze_sha256=source,actual_owned_resources_released=True,remote55=dest55,archive29=dest29)
write(D/'LOCAL_DELIVERY_VERIFICATION.json',verification);anchors('LOCAL_DELIVERY_VERIFICATION.json')
write(R/'operations/GPU_FINAL_DELIVERY_RESULT.json',dict(status='PASS',task_status=manifest['scope'],files=len(rows),bytes=sum(x['bytes'] for x in rows),manifest_sha256=v55['manifest_sha256'],seal_sha256=seal_hosts['local_sha256'],report=str(D/'FINAL_SCIENTIFIC_REPORT.md'),remote55=dest55,archive29=dest29))
print(json.dumps(dict(status='PASS_FINAL_THREE_HOST_DELIVERY',task_status=manifest['scope'],files=len(rows),manifest_sha256=v55['manifest_sha256']),ensure_ascii=False))
