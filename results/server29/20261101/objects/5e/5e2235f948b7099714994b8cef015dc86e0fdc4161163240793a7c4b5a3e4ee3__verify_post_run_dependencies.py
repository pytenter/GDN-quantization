"""Post-release CPU streaming file verification; never imports GPU runtime."""
import json
from collect_terminal import R,cfg,ssh
D=R/'delivery';assert D.exists()
code=r'''import sys,json
from pathlib import Path
r=Path(sys.argv[1]);sys.path.insert(0,str(r/'src'));import gpu_control as C
source=C.verify_sources(r);rows=[]
for path,x in C.bindings(r).items():
 p=Path(path);a=p.stat();h=C.sha(p);z=p.stat()
 assert (a.st_ino,a.st_size,a.st_mtime_ns)==(z.st_ino,z.st_size,z.st_mtime_ns)
 assert z.st_size==x['bytes'] and h==x['sha256'],path
 rows.append(dict(path=path,bytes=z.st_size,sha256=h,stable=True))
print(json.dumps(dict(status='PASS_FRESH_POST_RUN_SOURCE_AND_DEPENDENCY_SHA_BYTES',source_freeze_sha256=source,external_files=rows,GPU_execution=False)))'''
v=json.loads(ssh(code,[cfg['remote_root']],timeout=240))
(D/'archive/POST_RUN_SOURCE_AND_DEPENDENCY_VERIFICATION.json').write_text(json.dumps(v,indent=2)+'\n')
print(json.dumps(dict(status=v['status'],external_files=len(v['external_files']),source_sha256=v['source_freeze_sha256'])))
