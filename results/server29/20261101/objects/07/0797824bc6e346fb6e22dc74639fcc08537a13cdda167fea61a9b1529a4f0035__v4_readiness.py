"""Frozen V3 numerical one-update body, L8 intervention only; no formal training."""
from v4_common import *
import argparse,random,traceback
def main(label,condition):
    freeze_check();f=apply_qualified_runtime()
    # Exact qualification observer implementation used read-only; no source patch to parents.
    observer_module=c.import_file(C/'scripts/closure_common.py','v4_qualified_launch_observer')
    fwd=observer_module.LaunchObserver(f.layer_norm_gated_fwd_kernel,'layer_norm_gated_fwd_kernel')
    bwd=observer_module.LaunchObserver(f.layer_norm_gated_bwd_kernel,'layer_norm_gated_bwd_kernel')
    ns=dict(vars(s));ns.update(random=random,argparse=argparse,observe_graph=lambda loss:None)
    def writer(parent_destination,receipt):
        n=receipt['numeric'];q=load(D/'runtime/qualified_backward_selection.json')['configuration']
        assert all(r['launch'].get('BT')==q['kwargs']['BT'] and r['compiled']['num_warps']==q['num_warps'] and r['compiled']['num_stages']==q['num_stages'] for r in bwd.rows)
        fwd.check_forward(load(D/'runtime/historical_fixed_forward_manifest.json'))
        init=load(C/'runtime/whole_P1.json')['numeric']
        if condition=='L7R':assert n['parameter_before']==init['parameter_before'] and n['initial_rotations']==init['initial_rotations']
        # Original numerical main asserts L8 early theta/rotations fixed, active middle/late updated, and base frozen.
        receipt.update(v4_runtime_manifest_sha256=c.sha(D/'runtime/v4_deterministic_runtime_manifest.json'),fwd_launches=fwd.rows,bwd_launches=bwd.rows,V4_L7R_UPDATE0_IDENTITY_GATE='PASS' if condition=='L7R' else 'NOT_APPLICABLE',V4_L8_EARLY_FIXED_H_GATE='PASS' if condition=='L8' else 'NOT_APPLICABLE',V4_L8_MIDDLE_LATE_TRAINABLE_GATE='PASS' if condition=='L8' else 'NOT_APPLICABLE',V4_BASE_WEIGHT_FREEZE_GATE='PASS')
        emit(Path('runtime')/('readiness_'+condition+'_'+label+'.json'),receipt)
    ns['emit']=writer;path=C/'runtime/mechanical_v3_whole_main_observed.py';exec(compile(path.read_text(),str(path),'exec'),ns);ns['main'](0,label,condition)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--condition',choices=['L7R','L8'],required=True);a=p.parse_args()
    try:main(a.label,a.condition)
    except BaseException as e:emit(Path('run/STOP.json'),dict(stage='V4_READINESS',condition=a.condition,error=repr(e),traceback=traceback.format_exc(),formal_training_started=False,Fresh_B=0,AIME=0));raise
