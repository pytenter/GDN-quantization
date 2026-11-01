"""Independent CPU-only contract tests. Synthetic fixtures are not GPU evidence."""
import json
import math
import sys
from pathlib import Path
import numpy as np

if 'calibration_math' not in sys.modules:
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'calibration'))
import calibration_math as C


def run():
    passed=[]
    def check(name,fn):
        fn();passed.append(name)
    def equal(a,b,tol=1e-6):
        if not np.allclose(a,b,rtol=tol,atol=tol):raise AssertionError('numerical mismatch')
    def rejects(fn):
        try:fn()
        except ValueError:return
        raise AssertionError('did not reject illegal input')
    rng=np.random.default_rng(319)
    q=rng.normal(size=(32,128)).astype('float32')*.1
    k=rng.normal(size=(32,128)).astype('float32');k/=np.linalg.norm(k,axis=-1,keepdims=True)
    g=np.linspace(-4.,-.001,32,dtype='float32');b=np.linspace(0.,1.,32,dtype='float32')
    def explicit_matrix():
        expected=np.stack([math.exp(float(g[h]))*(np.eye(128)-float(b[h])*np.outer(k[h],k[h]))@q[h]/math.sqrt(128) for h in range(32)])
        equal(C.transported_query(q,k,g,b),expected,tol=1e-7)
    check('transport_against_explicit_128x128_A_transpose',explicit_matrix)
    check('actual_native_alpha_bytes_can_bind_transport',lambda:equal(C.transported_query(q,k,g,b,actual_alpha=np.exp(g).astype('float32')),C.transported_query(q,k,np.zeros(32),b)*np.exp(g).astype('float32')[:,None],1e-7))
    check('actual_native_alpha_wrong_head_shape_rejected',lambda:rejects(lambda:C.transported_query(q,k,g,b,actual_alpha=np.ones(16))))
    check('effective_query_scaled_exactly_once',lambda:equal(C.transported_query(q,k,g,b),C.transported_query(q/math.sqrt(128),k,g,b,True),1e-7))
    check('decay_is_exp_of_log_g_not_g',lambda:equal(C.transported_query(q,k,np.full(32,-math.log(2)),np.zeros(32)),q/(2*math.sqrt(128)),1e-7))
    def head_gate_independence():
        qq=np.repeat(q[:1],32,axis=0);kk=np.repeat(k[:1],32,axis=0)
        v=C.transported_query(qq,kk,g,b)
        if np.array_equal(v[0],v[1]):raise AssertionError('paired-head gate improperly copied')
    check('shared_query_does_not_copy_transport_across_heads',head_gate_independence)
    check('reject_nonfinite_query',lambda:rejects(lambda:C.transported_query(q*np.nan,k,g,b)))
    check('reject_decay_outside_native_range',lambda:rejects(lambda:C.transported_query(q,k,-g,b)))
    check('reject_beta_outside_native_range',lambda:rejects(lambda:C.transported_query(q,k,g,b+2)))
    check('reject_shared_gate_shape',lambda:rejects(lambda:C.transported_query(q,k,g[:16],b[:16])))
    def omega_zero():
        w,r=C.stabilize_omega(np.zeros((32,128)))
        equal(w,np.ones_like(w),0)
        if r['floor_count']!=4096:raise AssertionError('zero floor coverage missing')
    check('zero_omega_has_finite_positive_unit_weights',omega_zero)
    def omega_per_group():
        raw=np.ones((2,128));raw[0,:32]=np.exp(np.linspace(-1,1,32));raw[1,32:64]=4
        w,r=C.stabilize_omega(raw)
        equal(w[1],np.ones(128));equal(np.exp(np.log(w[0,:32]).mean()),1.,1e-6)
        equal(w[0,32:],np.ones(96),0)
    check('omega_group_key_only_no_head_or_group_mix',omega_per_group)
    def omega_clamp():
        raw=np.ones((32,128));raw[0,0]=1e-30;raw[0,1]=1e30
        w,r=C.stabilize_omega(raw)
        if w.min()<1/16 or w.max()>16 or not r['clamp_low_count'] or not r['clamp_high_count']:raise AssertionError('clamp missing')
    check('omega_floor_geometric_normalization_and_finite_clamp',omega_clamp)
    check('negative_omega_rejected',lambda:rejects(lambda:C.stabilize_omega(-np.ones((32,128)))))
    def accum():
        a=C.OmegaAccumulator([0,1]);v=np.ones((32,128),'float32')
        for layer in (0,1):a.add(layer,'CAL_document_1',1,v)
        w,r=a.finalize(expected_count=1);equal(w[0],v,0)
        rejects(lambda:a.add(0,'CAL_document_1',1,v))
        rejects(lambda:a.add(0,'VAL_document',2,v,split='VAL'))
    check('omega_actual_layer_coverage_document_identity_and_duplicates',accum)
    check('omega_incomplete_layers_rejected',lambda:rejects(lambda:C.OmegaAccumulator([0,1]).finalize()))
    x=rng.normal(size=(32,128,128)).astype('float32');w=np.ones((32,128),'float32')
    check('unit_a_b_recovers_r_one',lambda:equal(C.row_factors(x,w,0,0),w,0))
    check('zero_state_row_factors_finite',lambda:equal(C.row_factors(x*0,w,.5,.25),w,0))
    def row_amplitude():
        xx=np.ones((32,128,128),'float32');xx[:,0,:]=16
        rr=C.row_factors(xx,w,.5,0)
        equal(rr[:,0]/rr[:,1],np.full(32,4.),1e-6)
        equal(rr[:,32:],np.ones((32,96)),0)
    check('RowBal_Key_RMS_amplitude_and_group_independence',row_amplitude)
    def row_sensitivity():
        ww=np.ones_like(w);ww[:,0]=16
        rr=C.row_factors(np.ones_like(x),ww,0,.5)
        equal(rr[:,0]/rr[:,1],np.full(32,.25),1e-6)
    check('higher_sensitivity_gets_finer_effective_step',row_sensitivity)
    check('RowBal_clamp_avoids_overflow',lambda:equal(C.row_factors(x,w,1e30,1e30).clip(.25,4),C.row_factors(x,w,1e30,1e30),0))
    check('RowBal_nonpositive_omega_rejected',lambda:rejects(lambda:C.row_factors(x,w*0,.5,.25)))
    check('RowBal_wrong_axis_shape_rejected',lambda:rejects(lambda:C.row_factors(x[:,:,:64],w,.5,.25)))
    def one_step_proxy():
        e=np.zeros((2,128,128));e[:,3,7]=3
        qq=np.zeros((2,128));qq[:,3]=2
        equal(C.readout_proxy_mse(np.zeros_like(e),e,qq),36/128,0)
    check('readout_proxy_is_original_State_space',one_step_proxy)
    cal=C.CalibrationSample('cal','a'*64,'CAL',0,31,32,x,np.ones((32,128),'float32'))
    val=C.CalibrationSample('val','b'*64,'VAL',0,31,32,x,np.ones((32,128),'float32')*2)
    def selection():
        calls=[]
        def qdq(state,omega,a,b):
            calls.append((a,b));return state+(.0 if (a,b)==(.5,.25) else 1.)
        out=C.select_rowbal_parameters([cal],[val],{0:w},qdq)
        if (out['selected_a'],out['selected_b'])!=(.5,.25) or out['VAL_used_for_selection']:raise AssertionError('selection policy failed')
        if len(calls)!=6:raise AssertionError('finite grid coverage differs')
    check('finite_grid_CAL_selection_reports_VAL_only',selection)
    def tie():
        out=C.select_rowbal_parameters([cal],[val],{0:w},lambda s,w,a,b:s)
        if out['selected_grid_index']!=0:raise AssertionError('grid tie not deterministic')
    check('finite_grid_exact_tie_first_frozen_order',tie)
    check('CAL_VAL_same_document_rejected',lambda:rejects(lambda:C.select_rowbal_parameters([cal],[C.CalibrationSample('cal','b'*64,'VAL',0,31,32,x,w)],{0:w},lambda s,w,a,b:s)))
    check('CAL_VAL_same_raw_document_rejected',lambda:rejects(lambda:C.select_rowbal_parameters([cal],[C.CalibrationSample('val','a'*64,'VAL',0,31,32,x,w)],{0:w},lambda s,w,a,b:s)))
    check('misaligned_next_token_rejected',lambda:rejects(lambda:C.select_rowbal_parameters([C.CalibrationSample('cal','a'*64,'CAL',0,31,33,x,w)],[val],{0:w},lambda s,w,a,b:s)))
    check('finite_grid_change_rejected',lambda:rejects(lambda:C.select_rowbal_parameters([cal],[val],{0:w},lambda s,w,a,b:s,((.2,.2),))))
    return dict(status='PASS_CPU_CALIBRATION_CONTRACT',test_count=len(passed),checks=passed,
                fixtures='SYNTHETIC_CPU_ONLY_NOT_GPU_OR_SCIENCE_EVIDENCE',AIME_used=False,model_loads=0,GPU_calls=0)

if __name__=='__main__':print(json.dumps(run(),sort_keys=True))
