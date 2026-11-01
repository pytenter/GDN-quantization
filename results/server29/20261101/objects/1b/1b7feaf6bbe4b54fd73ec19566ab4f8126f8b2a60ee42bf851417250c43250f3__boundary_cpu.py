import os,sys,json
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
import torch,numpy as np
from group_codecs import encode,decode,tensor_bytes
from independent_reference import reference
from fractions import Fraction
checks=[]
def check(n,ok,**kw):checks.append(dict(name=n,passed=bool(ok),**kw));assert ok,n
for endpoint,expected in ((10.5,10),(11.5,12),(126.5,126),(127.5,128)):
 x=torch.zeros(1,1,128,128);x[:,:,0,:]=-endpoint;x[:,:,1,:]=255-endpoint
 e,debug=encode(x,'B2',debug=True)
 check('zero_point_tie_'+str(endpoint),int(e.zero_points[0,0,0,0])==expected and float(e.scales[0,0,0,0])==1.,exact_ratio=str(Fraction.from_float(endpoint)),expected=round(Fraction.from_float(endpoint)))
x=torch.zeros(1,1,128,128);x[:,:,0,:]=127.;x[:,:,1,:]=.5;x[:,:,2,:]=1.5;x[:,:,3,:]=-.5;x[:,:,4,:]=-1.5
for m in ('B0','B1'):
 e=encode(x,m);c=e.codes[0,0,0,:,0]
 check('symmetric_code_half_ties_'+m,c[:5].tolist()==[127,0,2,0,-2] and e.negative_zero_flags[0,0,0,3,0].item())
for m in ('B0','B1','B2'):
 e=encode(torch.zeros_like(x),m)
 check('zero_safe_scale_'+m,bool((e.scales>0).all()) and e.scales.dtype==torch.float32 and bool((decode(e)==0).all()))
for invalid in (torch.zeros(1,1,128,128,dtype=torch.float16),torch.zeros(1,1,64,128)):
 try:encode(invalid,'B1');caught=False
 except ValueError:caught=True
 check('shape_dtype_reject_'+str(invalid.shape)+'_'+str(invalid.dtype),caught)
extreme=torch.zeros_like(x);extreme[:,:,0,:]=torch.finfo(torch.float32).max;extreme[:,:,1,:]=-torch.finfo(torch.float32).max
try:encode(extreme,'B2');caught=False
except FloatingPointError:caught=True
check('Asym_FP32_span_overflow_FAIL_closed',caught)
# Independent exact-rational proof that changing division graph can change FP32 scale.
v=np.float32(735.27685546875);direct=np.float32(v/np.float32(127));recip=np.float32(v*np.float32(1/127))
check('CPU_CUDA_scalar_division_risk_real_fixture',direct.view(np.uint32)!=recip.view(np.uint32),direct_bits=int(direct.view(np.uint32)),reciprocal_bits=int(recip.view(np.uint32)))
(R/'qualification/ADDITIONAL_BOUNDARY_CPU.json').write_text(json.dumps(dict(status='PASS',checks=checks,GPU_dispatches=0),indent=2)+'\n')
print('boundary checks',len(checks))
