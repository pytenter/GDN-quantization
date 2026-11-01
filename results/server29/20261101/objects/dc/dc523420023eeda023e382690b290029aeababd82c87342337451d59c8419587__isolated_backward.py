"""Frozen direct-input current-backward and mathematical reference diagnostics only."""
from closure_common import *
import argparse
def main(label,mode,bundle):
    verify();f,hm,_=apply_fixed()
    assert Path(bundle).is_file();x={k:gpu(v) for k,v in torch.load(bundle,map_location='cpu',weights_only=True).items()}
    if mode=='current':
        obs=LaunchObserver(f.layer_norm_gated_bwd_kernel,'layer_norm_gated_bwd_kernel')
        result=f.layer_norm_gated_bwd(**x);forward=None;launch=obs.rows
    else:
        assert x['activation']=='sigmoid' and x['is_rms_norm'] and x['bias'] is None and not x['has_residual'] and x['dresidual'] is None
        # Mathematical FP32 RMSNorm/sigmoid; disposable isolated reference, never substituted into training.
        core=x['x'].detach().requires_grad_(True);gate=x['g'].detach().requires_grad_(True);weight=x['weight'].detach().requires_grad_(True)
        y=(core.float()*torch.rsqrt(core.float().square().mean(-1,keepdim=True)+x['eps'])*weight.float()*torch.sigmoid(gate.float())).to(core.dtype)
        forward=m(y);result=torch.autograd.grad(y,(core,gate,weight),x['dy']);launch=[]
    torch.cuda.synchronize()
    emit(Path('analysis')/(mode+'_backward_'+label+'.json'),dict(pid=os.getpid(),label=label,mode=mode,physical_gpu=0,gpu_uuid=c.UUIDS[0],bundle=bundle,bundle_sha256=c.sha(bundle),inputs={k:m(v) for k,v in x.items()},forward=forward,outputs=[m(v) for v in result],launches=launch,formal_updates=0,Fresh_B=0,AIME=0))
    print(mode,label,[m(v)['sha256'] if torch.is_tensor(v) else None for v in result],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--mode',choices=['current','reference'],required=True);p.add_argument('--bundle',required=True);a=p.parse_args();main(a.label,a.mode,a.bundle)
