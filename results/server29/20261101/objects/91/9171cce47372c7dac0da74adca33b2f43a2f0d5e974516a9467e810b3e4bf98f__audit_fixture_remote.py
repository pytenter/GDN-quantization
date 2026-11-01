"""Read frozen input fixture via SSH using stdlib; keep only bound prefix256."""
from pathlib import Path
import hashlib
import json
import shlex
import subprocess
import sys

here=Path(__file__).resolve().parent
root=here.parents[1]
cfg=json.loads((root/'ROOT.json').read_text())
program=r'''
from pathlib import Path
import datetime,hashlib,json,os,sys
root=Path(ROOT)
contracts=[json.loads((root/('gpu'+str(i))/'analysis/STRICT_BASELINE_CONTRACT.json').read_text()) for i in range(4)]
fi=contracts[0]['fixture']
assert all(x['fixture']==fi for x in contracts),'Fixture bindings differ across cards'
p=Path(fi['tokens_path']);before=p.stat();data=p.read_bytes();after=p.stat()
identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
assert identity(before)==identity(after),'Fixture changed while read'
ids=json.loads(data)
rawsha=hashlib.sha256(data).hexdigest()
canonicalsha=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
assert rawsha==fi['tokens_file_sha256'] and canonicalsha==fi['tokens_canonical_sha256']
assert len(ids)==32768==fi['length'] and all(type(x)is int and 0<=x<248320 for x in ids)
prefix=ids[:256]
receipt=dict(status='PASS_CURRENT_FROZEN_FIXTURE_IDENTITY_PREFIX256_ONLY',
 task_id=contracts[0]['task_id'],execution_run_id=contracts[0]['execution_run_id'],
 observation_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 fixture_path=str(p),fixture_raw_SHA256=rawsha,fixture_raw_bytes=len(data),
 fixture_canonical_SHA256=canonicalsha,full_length=len(ids),
 all_four_frozen_fixture_bindings_identical=True,
 prefix_length=len(prefix),prefix256=prefix,
 prefix_canonical_SHA256=hashlib.sha256(json.dumps(prefix,separators=(',',':')).encode()).hexdigest(),
 native_byteorder=sys.byteorder,
 metadata_before=dict(device=before.st_dev,inode=before.st_ino,bytes=before.st_size,mtime_ns=before.st_mtime_ns,ctime_ns=before.st_ctime_ns),
 metadata_after=dict(device=after.st_dev,inode=after.st_ino,bytes=after.st_size,mtime_ns=after.st_mtime_ns,ctime_ns=after.st_ctime_ns),
 expected_next_input_rule='ids[t+1] when t+1<256, otherwise None; frozenworker line270',
 GPU_imports=False,GPU_queries=False,GPU_dispatches=False,
 current_identity_is_not_historical_dynamic_library_PASS=True)
target=root/'independent_delivery/source/FIXTURE_PREFIX256_IDENTITY.json'
target.parent.mkdir(parents=True,exist_ok=True)
encoded=(json.dumps(receipt,sort_keys=True,ensure_ascii=False,indent=2)+'\n').encode()
with target.open('xb')as f:f.write(encoded);f.flush();os.fsync(f.fileno())
sys.stdout.buffer.write(encoded)
'''
program='ROOT='+repr(cfg['remote_root'])+'\n'+program
cmd=['ssh','-S',cfg['ssh_control_path'],'-o','BatchMode=yes','-o','ConnectTimeout=10',
     '-p',str(cfg['remote_port']),cfg['remote_user']+'@'+cfg['remote_host'],
     shlex.quote(cfg['remote_python'])+' -B -']
result=subprocess.run(cmd,input=program.encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=60)
if result.returncode:
    print(result.stderr.decode(errors='replace'));sys.exit(result.returncode)
receipt=json.loads(result.stdout)
target=here/'FIXTURE_PREFIX256_IDENTITY.json'
with target.open('xb')as f:f.write(result.stdout)
print(json.dumps(dict(status=receipt['status'],path=str(target),sha256=hashlib.sha256(result.stdout).hexdigest(),prefix_length=receipt['prefix_length'])))
