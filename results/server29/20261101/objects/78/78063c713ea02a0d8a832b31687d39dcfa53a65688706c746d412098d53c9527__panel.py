import hashlib
import torch
from references import FORMAL
SEED=20261015
CASES=('normal','hadamard_synthetic','row_imbalance','column_imbalance','sparse','zeros','single_outlier','signed_ties','batch_two')
PARAMS={'seed':SEED,'shape':[1,32,128,128],'batch_two_shape':[2,32,128,128],
        'imbalance_log10_range':[-3,3],'sparse_keep_probability':0.01,'outlier_value':10000.,
        'tie_values':[-127.,-126.5,-2.5,-1.5,-0.5,0.,0.5,1.5,2.5,126.5,127.],
        'coordinate':'Hadamard key coordinate; Hadamard case transforms a native synthetic state exactly once as H.T @ S',
        'stress_cases_G3':{'tiny':1e-30,'large':1e10,'finite_fp32_max':float(torch.finfo(torch.float32).max)},
        'metrics_zero_reference_relative_mse':0.,'latency_warmup':2,'latency_repeats':7,
        'recurrent':{'shape':[2,3,128,128],'steps':[1,8,32,128],'decay':0.97,'noise':0.01,'rank_one_amplitude':0.001,'seed':20262015}}
def thash(t):return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()
def make_panel():
    p={}
    for i,name in enumerate(CASES):
        g=torch.Generator(device='cpu').manual_seed(SEED+i)
        shape=(2 if name=='batch_two' else 1,32,128,128)
        x=torch.randn(shape,generator=g,dtype=torch.float32)
        if name=='hadamard_synthetic':x=FORMAL['hadamard'](128,device='cpu').t()@x
        if name=='row_imbalance':x=x*torch.logspace(-3,3,128)[None,None,:,None]
        if name=='column_imbalance':x=x*torch.logspace(-3,3,128)[None,None,None,:]
        if name=='sparse':x=x*(torch.rand(shape,generator=g)<0.01)
        if name=='zeros':x.zero_()
        if name=='single_outlier':x[0,0,0,0]=10000.
        if name=='signed_ties':x=torch.tensor(PARAMS['tie_values']).repeat((x.numel()+10)//11)[:x.numel()].reshape(shape).clone()
        p[name]=x.contiguous()
    return p
