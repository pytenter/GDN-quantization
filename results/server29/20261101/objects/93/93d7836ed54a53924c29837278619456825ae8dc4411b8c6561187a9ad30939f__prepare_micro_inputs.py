import os,sys,json,hashlib
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
import torch,numpy as np
from canonical_h128 import hadamard
from independent_reference import reference
torch.set_num_threads(1);torch.random.default_generator.manual_seed(20261019)
shape=(1,2,128,128);base=torch.randn(shape);a=torch.arange(128).reshape(1,1,128,1).float().expand(shape).clone()
fixtures={'zero':torch.zeros(shape),'positive':base.abs()+.01,'negative':-base.abs()-.01,
 'mixed':base+.3,'floor':torch.nextafter(torch.ones(shape)*1e-12,torch.zeros(shape)),
 'halves':a-63.5,'tiny':torch.full(shape,torch.finfo(torch.float32).tiny)*torch.where(base<0,-1.,1.),
 'large':torch.linspace(-1e20,1e20,128).reshape(1,1,128,1).expand(shape).clone(),
 'layout':base+torch.tensor([0.,100.]).reshape(1,2,1,1),'signedzero':torch.where(base<0,-1e-15,1.)}
fixtures['signedzero'][0,0,0,0]=-0.
h=hadamard(128)
for row in json.loads((R/'fixtures/REAL_STATE_INPUT_MANIFEST.json').read_text())['records']:
 x=torch.from_numpy(np.frombuffer((R/row['file']).read_bytes(),dtype='<f4').copy()).reshape(row['shape'])
 fixtures[Path(row['file']).stem]=h.T@x
d=R/'fixtures/micro';d.mkdir(exist_ok=True);rows=[]
for name,x in fixtures.items():
 arrays={'input':x.numpy()}
 for m in ('B0','B1','B2'):
  for k,v in reference(x.numpy(),m,'CUDA_LEGACY_EMULATED').items():
   if v is not None:arrays[m+'_'+k]=v
 p=d/(name+'.npz');np.savez(p,**arrays)
 rows.append(dict(file=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(R/'fixtures/GPU_MICRO_INPUT_MANIFEST.json').write_text(json.dumps(dict(status='CPU_PREPARED_GPU_NOT_RUN',GPU_dispatches=0,files=rows),indent=2)+'\n')
print('prepared',len(rows),'GPU micro inputs and independent reference arrays')
