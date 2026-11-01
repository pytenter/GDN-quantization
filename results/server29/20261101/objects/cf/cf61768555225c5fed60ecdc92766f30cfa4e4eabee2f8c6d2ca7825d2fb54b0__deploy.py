"""Deploy only this new independent development capsule, never the baseline."""
import argparse,base64,hashlib,io,json,subprocess,tarfile
from pathlib import Path
from transport import SSH55
ROOT=Path(__file__).resolve().parents[1]
REMOTE='/data/zypan/analyses/pace48_leap_unroll_dp8_dp16_v1/releases/FORMAL_V2'
PYTHON='/data/ydai/miniconda3/envs/bitdecode/bin/python'
def deploy(initial=False):
    files=[p for p in ROOT.rglob('*') if p.is_file() and p.suffix!='.lock' and not any(x in p.parts for x in ['__pycache__','formal','performance','runtime','validation','monitoring','independent_delivery'])]
    records=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
    stream=io.BytesIO()
    with tarfile.open(fileobj=stream,mode='w:gz') as tar:
        for p in files:tar.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
    data=base64.b64encode(stream.getvalue()).decode()
    code=f'''import base64,io,tarfile,pathlib,json,hashlib
r=pathlib.Path({REMOTE!r})
initial={initial!r}
if initial:r.mkdir()
assert r.is_dir()
assert not (r/'FORMAL_SOURCE_MANIFEST.json').exists(),'Production source is frozen'
b=base64.b64decode({data!r})
with tarfile.open(fileobj=io.BytesIO(b),mode='r:gz') as t:
 for x in t.getmembers():
  assert x.isfile() and not pathlib.PurePosixPath(x.name).is_absolute() and '..' not in pathlib.PurePosixPath(x.name).parts
  p=r/x.name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(t.extractfile(x).read())
rows={records!r}
for row in rows:
 p=r/row['path'];assert p.stat().st_size==row['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
print(json.dumps(dict(status='PASS_EXACT_NEW_CAPSULE_DEPLOY',files=len(rows),bytes=sum(x['bytes'] for x in rows),root=str(r))))
'''
    import shlex
    cmd=SSH55+['exec env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 '+shlex.quote(PYTHON)+' -B -']
    p=subprocess.run(cmd,input=code,text=True,capture_output=True)
    if p.returncode:raise RuntimeError(p.stderr[-3000:]+p.stdout[-1000:])
    print(p.stdout.strip())
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--initial',action='store_true');args=a.parse_args();deploy(args.initial)
