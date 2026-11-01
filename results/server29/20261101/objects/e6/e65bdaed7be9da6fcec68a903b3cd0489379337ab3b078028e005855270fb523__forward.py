from closure_common import *
import argparse
def main(label,repeats):
    bm=verify();torch.manual_seed(0)
    f,hm,chain=apply_fixed();observer=LaunchObserver(f.layer_norm_gated_fwd_kernel,'layer_norm_gated_fwd_kernel')
    b={k:gpu(v) for k,v in torch.load(bm['bundle'],map_location='cpu',weights_only=True).items()}
    for k,x in b.items():
        a=bm['tensors'][k];z=m(x);assert z['sha256']==a['sha256'] and z['shape']==a['shape'] and z['dtype']==a['dtype'] and z['stride']==a['stride']
    n=make_norm(bm,b);rows=[]
    with torch.no_grad():
        for i in range(repeats):
            y=n(b['native_core'],b['dynamic_gate']);torch.cuda.synchronize();rows.append(m(y))
    # Descriptive only, not equality/tolerance qualification.
    core=b['native_core'];gate=b['dynamic_gate']
    ref=(core.float()*torch.rsqrt(core.float().square().mean(-1,keepdim=True)+n.eps)*n.weight.float()*torch.sigmoid(gate.float())).to(core.dtype)
    dif=(y.float()-ref.float());desc=dict(max_abs_diff=float(dif.abs().max()),relative_l2=float(dif.norm()/ref.float().norm()),cosine=float(torch.nn.functional.cosine_similarity(y.float().reshape(1,-1),ref.float().reshape(1,-1))))
    assert c.sha(bm['source'])==c.sha(bm['bundle'])==bm['bundle_sha256']
    result=dict(label=label,pid=os.getpid(),physical_gpu=0,gpu_uuid=c.UUIDS[0],bundle_sha256=bm['bundle_sha256'],outputs=rows,same_process_exact=len({x['sha256'] for x in rows})==1,HISTORICAL_FIX_ACTUALLY_APPLIED_GATE=observer.check_forward(hm),launches=observer.rows,configuration_chain=chain,reference_descriptive=desc,reference=m(ref),flags=s.flags(),formal_updates=0,Fresh_B=0,AIME=0)
    emit(Path('analysis')/('forward_'+label+'.json'),result)
    print(json.dumps(dict(label=label,output=rows[0]['sha256'],same_process_exact=result['same_process_exact'],actually_applied=result['HISTORICAL_FIX_ACTUALLY_APPLIED_GATE'])),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--repeats',type=int,required=True);a=p.parse_args();main(a.label,a.repeats)
