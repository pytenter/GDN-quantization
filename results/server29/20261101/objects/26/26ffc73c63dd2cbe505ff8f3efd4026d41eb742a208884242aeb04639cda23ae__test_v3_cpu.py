"""Small necessary regression set, not another historical deep audit."""
import ast
import copy
import os
from pathlib import Path
from v3_common import *
from v3_guard import evaluate

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    cfg=read(OLD/'PHASE_B_CALIBRATION_CONFIG.json'); u=cfg['gpu_uuids']['6']
    identity=dict(pid=123,start_ticks=7,uid=1000,nspid=[123],boot_id='synthetic')
    physical=dict(returncode=0,stdout=f'6, {u}, GPU, 580.178.04, bus, 24576, 100, 10')
    compute=dict(returncode=0,stdout=f'{u}, 123, 80')
    shared=[dict(returncode=0,stdout='123 456',command=['fuser','/dev/nvidiactl'])]
    valid=evaluate(6,u,physical,compute,shared,identity,identity,True)
    assert valid['exclusive']
    wrong=copy.deepcopy(compute); wrong['stdout']+=f'\n{u}, 456, 80'
    assert not evaluate(6,u,physical,wrong,shared,identity,identity,True)['exclusive']
    wrong=copy.deepcopy(physical); wrong['stdout']=wrong['stdout'].replace(u,'WRONG_UUID')
    assert not evaluate(6,u,wrong,compute,shared,identity,identity,True)['exclusive']
    wrong=copy.deepcopy(identity); wrong['start_ticks']=8
    assert not evaluate(6,u,physical,compute,shared,identity,wrong,True)['exclusive']
    wrong=copy.deepcopy(compute); wrong['returncode']=1
    assert not evaluate(6,u,physical,wrong,shared,identity,identity,True)['exclusive']
    wrong=copy.deepcopy(shared); wrong[0]['returncode']=2
    assert not evaluate(6,u,physical,compute,wrong,identity,identity,True)['exclusive']
    wrong=copy.deepcopy(compute); wrong['stdout']+='\nGPU-other, 456, 80'
    assert evaluate(6,u,physical,wrong,shared,identity,identity,True)['exclusive']
    assert not evaluate(6,u,physical,dict(returncode=0,stdout=''),shared,identity,identity,True)['exclusive']
    assert digest(CODE/'persistent_engine_v1.py')==ENGINE_SHA
    vf=checked(V2/'V2_CODE_AND_ASSET_FREEZE.json')
    for name in ('v2_worker.py','v2_runtime.py','persistent_engine_v1.py','run_phase_b_calibration_v2.py','objective.py'):
        h=next(h for p,h in vf['code_sha256'].items() if Path(p).name==name)
        assert digest(CODE/name)==h
    import torch
    from v2_runtime import initialize,snapshot
    assert not torch.cuda.is_initialized()
    init=initialize(torch,0); snapshot(torch,'V3_CPU_regression',0)
    assert init['deterministic_algorithms_enabled'] and not init['warn_only']
    from persistent_engine_v1 import summarize,full_kl
    import math
    values=[1e-12,1e-4,1e-8,1e-3]*128
    original=summarize(values,0,512)['mean']; stable=math.fsum(values)/len(values)
    reducer_error=abs(original-stable)
    x=torch.tensor([[[1.,2.,3.]]]); assert full_kl(torch,x,x)==0.
    for p in CODE.glob('v3_*.py'): ast.parse(p.read_text())
    assert not torch.cuda.is_initialized()
    emit(OUT/'CPU_REGRESSION_RESULTS.json',dict(status='PASS',gpu_qualification='NOT_RUN',gpu_work=False,
        checks=['legal_single_compute_worker','shared_fd_not_compute_ownership','unauthorized_target_context_rejected',
        'UUID_index_error_rejected','PID_lifetime_error_rejected','NVML_failure_not_safe','diagnostic_failure_not_safe',
        'other_UUID_context_not_target','missing_own_context_rejected','original_engine_and_worker_and_runtime_byte_identical',
        'strict_deterministic_true_warn_only_false','original_reducer_and_identity_KL_unchanged'],
        runtime_cpu=init,synthetic_reducer_rounding_difference=reducer_error,original_engine_sha256=ENGINE_SHA))
    print('V3_CPU_REGRESSION_PASS_GPU_NOT_YET_QUALIFIED')

if __name__=='__main__': main()
