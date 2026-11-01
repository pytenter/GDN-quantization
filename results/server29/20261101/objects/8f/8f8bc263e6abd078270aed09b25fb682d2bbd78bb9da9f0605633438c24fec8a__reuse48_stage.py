"""CPU-only staging of this exact new execution, refusing post-freeze edits."""
from pathlib import Path
import argparse,base64,hashlib,importlib.util,json

ROOT=Path(__file__).resolve().parents[1]
REMOTE='/dev/shm/qwen_wscale_rowbal_cupti_repair_20261025'
OLD='/tmp/QWEN_GDN_SYM8_G32_WSCALE_ROWBAL_AIME26_SEED1_V1_BYTEVIEW_REPAIR_20261025'
SOURCE='05901486d72e6f62aba67e61961d6d62043dabef14340b46034a072a004ce5e1'
PYTHON='/data/ydai/miniconda3/envs/bitdecode/bin/python'
spec=importlib.util.spec_from_file_location('transport',ROOT/'operations/remote.py')
transport=importlib.util.module_from_spec(spec);spec.loader.exec_module(transport)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def stage(paths):
    rows=[]
    for p in sorted(set(map(Path,paths))):
        assert p.is_file() and not p.is_symlink()
        raw=p.read_bytes();name=str(p.relative_to(ROOT))
        assert name not in ['SOURCE_FREEZE.json','GPU_PERMIT.json']
        rows.append(dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),data=base64.b64encode(raw).decode()))
    code='rows='+repr(rows)+'\nROOT='+repr(REMOTE)+'\nOLD='+repr(OLD)+'\nSOURCE='+repr(SOURCE)+'\n'+r'''
from pathlib import Path
import json,base64,hashlib,os
r=Path(ROOT);old=Path(OLD)
assert ROOT=='/dev/shm/qwen_wscale_rowbal_cupti_repair_20261025' and not r.is_symlink()
assert hashlib.sha256((old/'SOURCE_FREEZE.json').read_bytes()).hexdigest()==SOURCE
r.mkdir(mode=0o700,exist_ok=True)
assert not any((r/n).exists() for n in ['SOURCE_FREEZE.json','GPU_PERMIT.json','GPU_RUN_STARTED.json','SESSION_RESULT.json']), 'Frozen/started: no staging edits'
for x in rows:
 n=Path(x['path']);assert not n.is_absolute() and '..' not in n.parts
 p=r/n;assert not p.is_symlink();p.parent.mkdir(parents=True,exist_ok=True)
 raw=base64.b64decode(x['data']);assert len(raw)==x['bytes'] and hashlib.sha256(raw).hexdigest()==x['sha256']
 if not p.exists() or p.read_bytes()!=raw:
  tmp=p.with_name(p.name+'.preparation_'+str(os.getpid()))
  with tmp.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
  os.replace(tmp,p)
 assert p.stat().st_size==x['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256']
print(json.dumps(dict(status='PASS_NEW_CPU_PREFREEZE_STAGING',count=len(rows),bytes=sum(x['bytes'] for x in rows),root=ROOT,GPU_calls=0)))
'''
    compile(code,'<prevalidated-staging>','exec')
    return transport.run(code,role='55',python=PYTHON,timeout=55)

def stage_raw():
    code='ROOT='+repr(REMOTE)+'\nOLD='+repr(OLD)+'\n'+r'''
from pathlib import Path
import json,hashlib,shutil,os
r=Path(ROOT);old=Path(OLD)
assert not any((r/n).exists() for n in ['SOURCE_FREEZE.json','GPU_PERMIT.json','GPU_RUN_STARTED.json'])
catalog=json.loads((old/'calibration/RAW_CLONE_VERIFICATION.json').read_text());assert catalog['count']==4323 and len(catalog['files'])==4323
total=0
for x in catalog['files']:
 n=Path(x['path']);assert not n.is_absolute() and '..' not in n.parts and n.parts[:2]==('calibration','raw')
 src=old/n;dst=r/n;assert src.is_file() and not src.is_symlink() and src.stat().st_size==x['bytes']
 assert hashlib.sha256(src.read_bytes()).hexdigest()==x['sha256']
 dst.parent.mkdir(parents=True,exist_ok=True);assert not dst.is_symlink()
 if dst.exists():assert dst.stat().st_size==x['bytes'] and hashlib.sha256(dst.read_bytes()).hexdigest()==x['sha256']
 else:
  with src.open('rb') as f,dst.open('xb') as g:shutil.copyfileobj(f,g,1024**2);g.flush();os.fsync(g.fileno())
 assert hashlib.sha256(dst.read_bytes()).hexdigest()==x['sha256'];total+=x['bytes']
assert total==catalog['bytes']==1300245388
print(json.dumps(dict(status='PASS_REUSED_CALIBRATION_RAW_EXACT_BYTES_IN_NEW_RAM_SCRATCH',count=4323,bytes=total,GPU_calls=0,new_calibration_credit=0,Mac_tensor_downloads=0)))
'''
    compile(code,'<prevalidated-raw-clone>','exec')
    return transport.run(code,role='55',python=PYTHON,timeout=120)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage-assets',action='store_true');ap.add_argument('--stage-raw',action='store_true');args=ap.parse_args()
    assert ROOT.name.endswith('_CUPTI_REPAIR_20261025')
    if args.stage_assets:
        paths=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and str(p.relative_to(ROOT)) not in ['SOURCE_FREEZE.json','GPU_PERMIT.json'] and not p.suffix=='.pyc']
        batches=[];current=[];size=0
        for p in sorted(paths):
            if current and size+p.stat().st_size>2*1024**2:batches.append(current);current=[];size=0
            current.append(p);size+=p.stat().st_size
        if current:batches.append(current)
        print(json.dumps({'assets':[stage(b) for b in batches]}))
    if args.stage_raw:print(json.dumps({'raw':stage_raw()}))

if __name__=='__main__':main()
