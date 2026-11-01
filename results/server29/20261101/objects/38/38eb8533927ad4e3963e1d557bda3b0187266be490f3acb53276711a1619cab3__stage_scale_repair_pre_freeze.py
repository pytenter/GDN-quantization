"""CPU-only staging to the new root, before any frozen permit or STARTED."""
import argparse
import base64
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile

L = Path(__file__).resolve().parents[1]
REMOTE = '/data/zypan/analyses/qwen_gdn_pace48_aime26_seed1_60_v1_scale_repair_20261025'
TRANSPORT_SHA = 'db0ea71612d633b0fc33feef1d95c42e273ab2bd9ffc971b8d7e4ddc97f74f40'
GRANT_SHA = 'f9cfbada775455554bf3d33a4d22522e48ad0a75fab03ed2caa3611b72232ef3'


def run(math_only=False):
    transport = L / 'operations/remote.py'
    assert hashlib.sha256(transport.read_bytes()).hexdigest() == TRANSPORT_SHA
    spec = importlib.util.spec_from_file_location('scale_repair_pinned_transport', transport)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if math_only:
        names = ['ROOT.json', 'operations/HUMAN_PACE_SCALE_REPAIR_AUTHORIZATION.json',
                 'src/pace_quantizer.py', 'src/pace_buffer.py', 'src/pace_scale_diagnostics.py',
                 'tests/test_pace_scale_boundary.py', 'tests/test_pace_scale_diagnostics.py']
    else:
        names = [str(p.relative_to(L)) for p in sorted(L.rglob('*'))
                 if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'
                 and str(p.relative_to(L)) not in ('source_manifest.json', 'GPU_PERMIT.json', 'SOURCE_FREEZE.json')]
    stream = io.BytesIO()
    rows = []
    with tarfile.open(fileobj=stream, mode='w:gz', dereference=True) as archive:
        for name in names:
            p = L / name
            raw = p.read_bytes()
            rows.append(dict(path=name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
            archive.add(p, arcname=name, recursive=False)
    program = '''
import base64,hashlib,io,json,os,tarfile
from pathlib import Path
R=Path(REMOTE)
assert str(R)=='/data/zypan/analyses/qwen_gdn_pace48_aime26_seed1_60_v1_scale_repair_20261025'
R.mkdir(parents=True,exist_ok=True)
assert not R.is_symlink()
assert not any((R/n).exists() for n in ['GPU_PERMIT.json','SESSION_STARTED.json','GPU_RUN_STARTED.json','OFFLINE_PARENT_STARTED.json','DELIVERY_MANIFEST.json'])
def ancestors(p):
 current=p
 while current!=Path('/'):
  assert not current.is_symlink()
  current=current.parent
ancestors(R)
expected={x['path']:x for x in ROWS};assert len(expected)==len(ROWS)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(DATA)),mode='r:gz') as archive:
 members=archive.getmembers();assert len(members)==len(expected) and {m.name for m in members}==set(expected)
 payload=[]
 for member in members:
  p=Path(member.name);assert not p.is_absolute() and '..' not in p.parts and member.isfile()
  data=archive.extractfile(member).read();row=expected[member.name]
  assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
  ancestors(R/p);payload.append((p,data))
 for p,data in payload:
  target=R/p;target.parent.mkdir(parents=True,exist_ok=True);ancestors(target)
  temp=target.with_name(target.name+'.prefreeze_new_inode.'+str(os.getpid()))
  with temp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
  os.replace(temp,target)
for row in ROWS:
 p=R/row['path'];assert p.is_file() and not p.is_symlink() and p.stat().st_size==row['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
assert hashlib.sha256((R/'operations/HUMAN_PACE_SCALE_REPAIR_AUTHORIZATION.json').read_bytes()).hexdigest()==GRANT_SHA
print(json.dumps(dict(status='PASS_NEW_PACE_SCALE_REPAIR_PREFREEZE_CPU_STAGE',count=len(ROWS),bytes=sum(x['bytes'] for x in ROWS),GPU_calls=0,old_assets_modified=False,root=str(R))))
'''
    code = ('REMOTE=' + repr(REMOTE) + '\nGRANT_SHA=' + repr(GRANT_SHA)
            + '\nDATA=' + repr(base64.b64encode(stream.getvalue()).decode())
            + '\nROWS=json.loads(' + repr(json.dumps(rows)) + ')\n' + program)
    return module.run('import json\n' + code, role='55',
                      python='/data/ydai/miniconda3/envs/bitdecode/bin/python', timeout=120)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--math-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.math_only), ensure_ascii=False))
