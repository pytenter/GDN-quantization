"""CPU-only staging to the exact new revision, refusing every post-freeze edit."""
from pathlib import Path
import base64
import hashlib
import importlib.util
import json

ROOT=Path(__file__).resolve().parents[1]
REMOTE='/tmp/QWEN_GDN_SYM8_G32_WSCALE_ROWBAL_AIME26_SEED1_V1_BYTEVIEW_REPAIR_20261025'
OLD='/tmp/QWEN_GDN_SYM8_G32_WSCALE_ROWBAL_AIME26_SEED1_V1_REDUCTION_REPAIR_20261025'
spec=importlib.util.spec_from_file_location('existing_transport',ROOT.parent/'QWEN_GDN_GROUP64_FOUR_CODEC_AIME26_SEED1_V1/operations/remote.py')
transport=importlib.util.module_from_spec(spec);spec.loader.exec_module(transport)


def stage(paths):
    rows=[]
    for p in sorted(set(map(Path,paths))):
        assert not p.is_symlink() and p.is_file()
        before=p.stat();raw=p.read_bytes();after=p.stat()
        assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
        n=str(p.relative_to(ROOT));assert '..' not in Path(n).parts
        rows.append(dict(path=n,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),data=base64.b64encode(raw).decode()))
    code='rows='+repr(rows)+'\nROOT='+repr(REMOTE)+'\nOLD='+repr(OLD)+'\n'+r'''
from pathlib import Path
import base64,hashlib,json,os
r=Path(ROOT);old=Path(OLD)
assert r!=old and r.name.endswith('_BYTEVIEW_REPAIR_20261025')
assert hashlib.sha256((old/'SOURCE_FREEZE.json').read_bytes()).hexdigest()=='6e87495d2dac7ccc7d97363256be3862ab7a52159271b848a58f06e2eabd1f4e'
assert not r.is_symlink();r.mkdir(mode=0o700,exist_ok=True)
frozen=(r/'SOURCE_FREEZE.json').exists() or (r/'GPU_PERMIT.json').exists() or (r/'GPU_RUN_STARTED.json').exists()
done=[]
for x in rows:
 n=Path(x['path']);assert not n.is_absolute() and '..' not in n.parts
 p=r/n;assert not p.is_symlink();p.parent.mkdir(parents=True,exist_ok=True)
 raw=base64.b64decode(x['data']);assert len(raw)==x['bytes'] and hashlib.sha256(raw).hexdigest()==x['sha256']
 prior=hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
 if p.exists() and p.read_bytes()==raw:pass
 elif frozen:raise RuntimeError('Final revision is frozen; no overwrite/addition via staging:'+str(n))
 else:
  t=p.with_name(p.name+'.cpu_tmp_'+str(os.getpid()))
  with t.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
  os.replace(t,p)
 assert p.stat().st_size==x['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256']
 done.append(dict(path=str(n),bytes=x['bytes'],sha256=x['sha256'],previous_SHA=prior))
assert hashlib.sha256((old/'SOURCE_FREEZE.json').read_bytes()).hexdigest()=='6e87495d2dac7ccc7d97363256be3862ab7a52159271b848a58f06e2eabd1f4e'
print(json.dumps(dict(status='PASS_CPU_STAGING_EXACT_NEW_PREFREEZE_REVISION',root=str(r),files=done,old_root_unchanged=True,GPU_calls=0)))
'''
    return transport.run(code,role='55',python='/data/ydai/miniconda3/envs/bitdecode/bin/python',timeout=90)
