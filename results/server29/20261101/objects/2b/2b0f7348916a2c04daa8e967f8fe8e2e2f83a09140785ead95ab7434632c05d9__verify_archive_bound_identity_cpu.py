#!/usr/bin/env python3
"""Fresh small-file identity checks only; no models, raw archive or GPU calls."""
import hashlib,json,shlex,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[2];r=json.loads((R/'ROOT.json').read_text());payload=dict(root=r['remote_root'],cards=r['cards'])
code='''from pathlib import Path
import hashlib,json,sys,os,time
v=json.load(sys.stdin);rows=[]
for x in v['cards']:
 root=Path(x['remote_root'])
 for name,key in [('ARCHIVE_CONFIG.json','archive_config_sha256'),('AUTHORIZATION.json','authorization_sha256'),('SOURCE_MANIFEST.json','source_manifest_sha256')]:
  p=root/name;a=p.stat();raw=p.read_bytes();b=p.stat();h=hashlib.sha256(raw).hexdigest()
  if h!=x[key] or (a.st_ino,a.st_size,a.st_mtime_ns)!=(b.st_ino,b.st_size,b.st_mtime_ns):raise RuntimeError('Frozen archive authority identity mismatch:'+str(p))
  rows.append(dict(card=x['card_key'],path=str(p),bytes=len(raw),sha256=h,status='PASS_FRESH_MATCH_LOCAL_ROOT_ANCHOR'))
result=dict(status='PASS_ARCHIVE_AUTH_CONFIG_SOURCE_IDENTITY',rows=rows,unix=time.time(),GPU_calls=0)
p=Path(v['root'])/'independent_delivery/ARCHIVE_BOUND_IDENTITY.json';raw=(json.dumps(result,indent=2)+'\\n').encode()
with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
print(json.dumps(dict(result=result,receipt=dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))))
'''
cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-S',r['ssh_control_path'],r['remote_user']+'@'+r['remote_host'],shlex.join([r['remote_python'],'-B','-c',code])];p=subprocess.run(cmd,input=json.dumps(payload).encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace'))
v=json.loads(p.stdout);(R/'independent_delivery/ARCHIVE_BOUND_IDENTITY.json').write_text(json.dumps(v,indent=2)+'\n');print(v['result']['status'],len(v['result']['rows']))
