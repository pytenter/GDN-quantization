"""Synchronize only before SESSION_STARTED and before issuance of a GPU permit."""
import base64
import hashlib
import io
import json
from pathlib import Path
import tarfile
import remote

L = Path(__file__).resolve().parents[1]
output = io.BytesIO()
rows = []
with tarfile.open(fileobj=output, mode='w:gz', dereference=True) as archive:
    for p in sorted(L.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts or p.suffix == '.pyc':
            continue
        name = str(p.relative_to(L))
        if name in ('GPU_PERMIT.json', 'source_manifest.json', 'SOURCE_FREEZE.json'):
            continue
        rows.append(dict(path=name, bytes=p.stat().st_size,
                         sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        archive.add(p, arcname=name, recursive=False)
body = '''
import pathlib,tarfile,io,base64,json,hashlib
R=pathlib.Path('/data/zypan/analyses/qwen_gdn_pace48_aime26_seed1_60_v1')
assert R.is_dir() and not (R/'SESSION_STARTED.json').exists() and not (R/'GPU_PERMIT.json').exists()
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(DATA)),mode='r:gz') as t:
 for m in t.getmembers():
  p=pathlib.Path(m.name);assert not p.is_absolute() and '..' not in p.parts and m.isfile()
 t.extractall(R)
for x in ROWS:
 p=R/x['path'];assert p.stat().st_size==x['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256']
print(json.dumps(dict(status='PASS_PRE_FREEZE_STAGED_EXACT_SHA_NO_START',count=len(ROWS))))
'''
code = ('DATA = ' + repr(base64.b64encode(output.getvalue()).decode()) + '\nROWS = ' +
        repr(rows) + '\n' + body)
# Validate the complete program before transport. Placeholder replacement must
# never rewrite coincidental bytes inside an encoded archive.
compile(code, '<PACE_PRE_FREEZE_STAGE>', 'exec')
print(json.dumps(remote.run(code, role='55', python='/data/ydai/miniconda3/envs/bitdecode/bin/python', timeout=60)))
