"""Independent expected GPU fields prepared without any CUDA initialization."""
import sys,os,json,hashlib
from pathlib import Path
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
import numpy as np
from independent_reference import reference
out=R/'fixtures';out.mkdir(exist_ok=True);rows=[]
for p in sorted((R/'history/group32/inherited/group32/fixtures/micro').glob('*.npz')):
 with np.load(p,allow_pickle=False)as z:x=z['input'].copy()
 arrays={'input':x}
 for m in ['B1','B2']:
  for k,v in reference(x,m,'CUDA_LEGACY_EMULATED',group_size=64).items():
   if v is not None:arrays[m+'_'+k]=v
 q=out/p.name;np.savez(q,**arrays);rows.append(dict(path=str(q.relative_to(R)),bytes=q.stat().st_size,sha256=hashlib.sha256(q.read_bytes()).hexdigest(),origin=str(p.relative_to(R)),origin_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
assert len(rows)>=30
(R/'GPU_MICRO_INPUT_MANIFEST.json').write_text(json.dumps(dict(status='CPU_PREPARED_INDEPENDENT_GROUP64_FIELDS',GPU_dispatches=0,CPU_semantics='CUDA_LEGACY_EMULATED',files=rows),indent=2)+'\n')
print(json.dumps(dict(status='PASS',fixtures=len(rows),GPU_dispatches=0)))
