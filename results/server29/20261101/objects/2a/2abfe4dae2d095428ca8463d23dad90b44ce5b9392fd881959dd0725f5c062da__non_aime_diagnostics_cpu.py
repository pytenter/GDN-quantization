"""Independent diagnostics arithmetic tests; syntheticCPU, never GPU evidence."""
import numpy as np

def run(module):
    checks=[]
    def ok(name,fn):fn();checks.append(name)
    def eq(a,b):
        if not np.allclose(a,b,rtol=1e-12,atol=1e-12):raise AssertionError('wrong independent metric')
    def reject(fn):
        try:fn()
        except RuntimeError:return
        raise AssertionError('illegal diagnostic input accepted')
    x=np.ones((2,128,128),np.float32);g=np.zeros((2,128),np.float32);g[:,3]=.5;w=np.ones((2,128),np.float32)
    def amplitude():
        r=module._metric_arrays_cpu(x,2*x,g,2*w)
        eq(r['state_mse'],[1,1]);eq(r['weighted_state_mse'],[2,2]);eq(r['relative_State_amplitude_error'],[1,1]);eq(r['State_direction_one_minus_cosine'],[0,0]);eq(r['one_step_transported_readout_mse'],[.25,.25])
    ok('known_amplitude_and_weighted_State_and_readout_errors',amplitude)
    def direction():
        r=module._metric_arrays_cpu(x,-x,g,w);eq(r['State_direction_one_minus_cosine'],[2,2]);eq(r['relative_State_amplitude_error'],[0,0])
    ok('opposite_State_direction_error_two',direction)
    def zero():
        r=module._metric_arrays_cpu(x*0,x*0,g,w)
        for value in r.values():eq(value,[0,0])
    ok('both_zero_State_is_finite_zero_error',zero)
    def key_axis():
        z=np.zeros_like(x);d=z.copy();d[:,3,7]=6;ww=w.copy();ww[:,3]=4
        r=module._metric_arrays_cpu(z,d,g,ww);eq(r['state_mse'],[36/16384]*2);eq(r['weighted_state_mse'],[144/16384]*2);eq(r['one_step_transported_readout_mse'],[9/128]*2)
    ok('Key_axis_transport_does_not_mix_Value_columns',key_axis)
    def separate_heads():
        d=x.copy();d[0]+=1;d[1]+=2;r=module._metric_arrays_cpu(x,d,g,w);eq(r['state_mse'],[1,4]);eq(r['one_step_transported_readout_mse'],[.25,1])
    ok('actual_recurrent_head_identity_no_averaging_before_error',separate_heads)
    def equal_state():
        r=module._metric_arrays_cpu(x,x,g,w)
        for value in r.values():eq(value,[0,0])
    ok('exact_reconstruction_metrics_zero',equal_state)
    ok('nonfinite_State_rejected',lambda:reject(lambda:module._metric_arrays_cpu(x*np.nan,x,g,w)))
    ok('nonpositive_omega_rejected',lambda:reject(lambda:module._metric_arrays_cpu(x,x,g,w*0)))
    ok('wrong_State_axes_rejected',lambda:reject(lambda:module._metric_arrays_cpu(x[:,:,:64],x[:,:,:64],g,w)))
    ok('wrong_transport_head_shape_rejected',lambda:reject(lambda:module._metric_arrays_cpu(x,x,g[:1],w)))
    def stats():
        s=module._Statistics();s.add(np.array([1.,2.,3.]));s.add(np.array([4.]));v=s.finish();eq(v['mean'],2.5);eq(v['minimum'],1);eq(v['maximum'],4);assert v['count']==4 and v['quantiles_exact_for_all_reported_metadata_values']
    ok('exact_bounded_metadata_quantiles_all_values',stats)
    ok('nonfinite_metadata_rejected',lambda:reject(lambda:module._Statistics().add(np.array([np.inf]))))
    return dict(status='PASS_CPU_DIAGNOSTIC_METRIC_CONTRACT',test_count=len(checks),checks=checks,fixtures='SYNTHETIC_CPU_ONLY_NOT_GPU_OR_SCIENCE_EVIDENCE',model_loads=0,GPU_calls=0,AIME_used=False)
