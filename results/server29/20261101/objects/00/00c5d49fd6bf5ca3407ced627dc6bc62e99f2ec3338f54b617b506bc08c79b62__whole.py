"""Execute mechanically extracted unchanged V3 one-update numeric main with launch-only historical fix."""
from closure_common import *
import argparse, random, traceback
def main(label):
    verify();f,hm,chain=apply_fixed();launch=LaunchObserver(f.layer_norm_gated_fwd_kernel,'layer_norm_gated_fwd_kernel')
    backlaunch=LaunchObserver(f.layer_norm_gated_bwd_kernel,'layer_norm_gated_bwd_kernel')
    rows=[];graph_rows=[];orig=f.layer_norm_gated_bwd;signature=inspect.signature(orig)
    def observed(*args,**kwargs):
        bound=signature.bind(*args,**kwargs);bound.apply_defaults();inp={k:m(v) for k,v in bound.arguments.items()}
        idx=len(rows);path=Path('runtime/backward_bundles')/label/(str(idx)+'.pt');(D/path).parent.mkdir(parents=True,exist_ok=True)
        torch.save({k:cpu(v) for k,v in bound.arguments.items()},D/path)
        result=orig(*args,**kwargs)
        rows.append(dict(call=idx,inputs=inp,outputs=[m(v) for v in result],bundle=str(D/path),bundle_sha256=c.sha(D/path)))
        return result
    f.layer_norm_gated_bwd=observed
    # Namespace delegation retains original common ROOT/config/input pointers for reads only.
    ns=dict(vars(s));ns.update(random=random,argparse=argparse)
    def observe_graph(loss):
        nodes=[];seen=set()
        def walk(node):
            if node is None or node in seen:return
            seen.add(node);nodes.append(node)
            for nxt,_ in node.next_functions:walk(nxt)
        walk(loss.grad_fn)
        for idx,node in enumerate(nodes):
            def hook(inputs,outputs,idx=idx,node=node):
                graph_rows.append(dict(execution=len(graph_rows),node=idx,operation=type(node).__name__,incoming=[m(v) for v in outputs],outgoing=[m(v) for v in inputs]))
                return None
            node.register_hook(hook)
    ns['observe_graph']=observe_graph
    def receipt_writer(parent_destination,receipt):
        assert str(parent_destination).endswith('_L7R_smoke.json')
        receipt.update(historical_fixed_forward_manifest_sha256=c.sha(D/'runtime/historical_fixed_forward_manifest.json'),source_freeze_new_sha256=c.sha(D/'runtime/source_freeze.json'),HISTORICAL_FIX_ACTUALLY_APPLIED_GATE=launch.check_forward(hm))
        emit(Path('runtime')/('whole_'+label+'.json'),receipt)
    ns['emit']=receipt_writer
    source=D/'runtime/mechanical_v3_whole_main_observed.py';exec(compile(source.read_text(),str(source),'exec'),ns)
    ns['main'](0,label,'L7R')
    emit(Path('analysis')/('whole_boundaries_'+label+'.json'),dict(norm_backward=rows,autograd_execution=graph_rows,norm_forward_launches=launch.rows,norm_backward_launches=backlaunch.rows,observer_scope='hashing/copy/launch recording only; no changed arguments or numerical operations',formal_updates=0,disposable_updates=1,Fresh_B=0,AIME=0))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--label',required=True);a=p.parse_args()
    try:main(a.label)
    except BaseException as e:
        emit(Path('runtime')/('whole_'+a.label+'_failure.json'),dict(error=repr(e),traceback=traceback.format_exc(),formal_updates=0));raise
