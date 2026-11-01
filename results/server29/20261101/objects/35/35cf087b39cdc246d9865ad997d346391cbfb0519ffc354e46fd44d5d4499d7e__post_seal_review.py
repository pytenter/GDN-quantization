"""Independent post-publication byte verification, with no GPU execution."""
import hashlib,json,sys,shlex,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[1];D=R/'delivery';cfg=json.loads((R/'ROOT.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
v=json.loads((D/'LOCAL_DELIVERY_VERIFICATION.json').read_text());seal=json.loads((D/'FINAL_SEAL_RECEIPT.json').read_text());m=json.loads((D/'SHA256_MANIFEST.json').read_text());task=json.loads((D/'TASK_RESULT.json').read_text())
assert v['status']=='PASS_ALL_THREE_HOSTS_FINAL_SMALL_DELIVERY' and task['status']=='INT8_SHORT_RUNTIME_QUALIFIED'
assert v['manifest_sha256']==seal['manifest_sha256']==sha(D/'SHA256_MANIFEST.json')
assert seal['sums_sha256']==sha(D/'SHA256SUMS.txt') and v['report_sha256']==sha(D/'FINAL_SCIENTIFIC_REPORT.md')
assert len(m['files'])==len({x['path'] for x in m['files']})==seal['indexed_files']
for x in m['files']:
 p=D/x['path'];assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256'],x['path']
expected=''.join(x['sha256']+'  '+x['path']+'\n' for x in m['files'])+sha(D/'SHA256_MANIFEST.json')+'  SHA256_MANIFEST.json\n'
assert (D/'SHA256SUMS.txt').read_text()==expected
sys.path.insert(0,str(D/'src'));import gpu_control as C
assert C.verify_sources(D)=='be6786c37ab508f6749fce33cfd00606a2c2ac5bbbd2fb429a673143b40dfddc'
archive=json.loads((D/'archive/FINAL_ARCHIVE_VERIFICATION.json').read_text());assert archive['object_count']==2016 and archive['total_bytes']==366819508224 and not archive['unindexed_bin'] and not archive['orphan_tmp']
assert all(x['fresh_stream_SHA'] for x in archive['objects'])
anchors={n:dict(bytes=(D/n).stat().st_size,sha256=sha(D/n)) for n in ('SHA256_MANIFEST.json','SHA256SUMS.txt','FINAL_SEAL_RECEIPT.json','LOCAL_DELIVERY_VERIFICATION.json','FINAL_SCIENTIFIC_REPORT.md')}
remote=r'''import json,hashlib,sys
from pathlib import Path
p=Path(sys.argv[1]);anchors=json.loads(sys.argv[2]);m=json.loads((p/'SHA256_MANIFEST.json').read_text())
for x in m['files']:
 q=p/x['path'];b=q.read_bytes();assert len(b)==x['bytes'] and hashlib.sha256(b).hexdigest()==x['sha256'],x['path']
for n,x in anchors.items():
 b=(p/n).read_bytes();assert len(b)==x['bytes'] and hashlib.sha256(b).hexdigest()==x['sha256'],n
print(json.dumps(dict(status='PASS_POST_PUBLICATION_ALL_SMALL_BYTES',root=str(p),files=len(m['files']))))'''
def ssh(code,args=(),payload=None):
 p=subprocess.run(['ssh','-S',cfg['ssh_control_path'],'-o','BatchMode=yes','zypan@202.38.247.55',shlex.join([cfg['remote_python'],'-B','-c',code]+[str(x) for x in args])],input=payload,capture_output=True,timeout=180);assert p.returncode==0,p.stderr.decode();return p.stdout
f55=json.loads(ssh(remote,[cfg['remote_root']+'/FINAL_DELIVERY',json.dumps(anchors)]))
forward=r'''import subprocess,shlex,sys
p=subprocess.run(['ssh','-S','/tmp/qwen_replication_archive29.sock','-o','BatchMode=yes','zypan@202.38.247.29',shlex.join(['/usr/bin/python3','-B','-c',sys.argv[1],sys.argv[2],sys.argv[3]])],capture_output=True,timeout=150);assert p.returncode==0,p.stderr.decode();sys.stdout.buffer.write(p.stdout)'''
f29=json.loads(ssh(forward,[remote,cfg['archive_root']+'/FINAL_DELIVERY',json.dumps(anchors)]))
out=dict(status='PASS_INDEPENDENT_POST_SEAL_ALL_THREE_HOSTS',scientific_status=task['status'],files=len(m['files']),source_freeze_sha256=v['source_freeze_sha256'],manifest_sha256=v['manifest_sha256'],raw_objects=2016,raw_bytes=366819508224,all_raw_fresh_hashes_verified=True,remote55=f55,archive29=f29,GPU_dispatches=0)
with (R/'operations/POST_SEAL_INDEPENDENT_REVIEW.json').open('x') as f:json.dump(out,f,indent=2)
print(json.dumps(out))
