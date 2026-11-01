"""CPU chain probe or minimal GPU initialization canary; never loads a model."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback
from recovery_state import atomic_json
from execution04_launcher import selected_environment,process_observation

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['cpu','gpu'],required=True);p.add_argument('--stage',required=True);p.add_argument('--result',type=Path,required=True);p.add_argument('--expected',type=Path,required=True);a=p.parse_args()
    observation=process_observation();atomic_json(a.result.with_suffix('.OBSERVED_ENV.json'),observation)
    try:
        assert observation['environment']==json.loads(a.expected.read_text()),'ACTUAL_WORKER_ENVIRONMENT_DIFFERS_FROM_LAUNCHER'
        if a.mode=='cpu':assert os.environ['CUDA_VISIBLE_DEVICES']==''
        import torch
        assert not torch.cuda.is_initialized()
        from v2_runtime import initialize,snapshot
        runtime=[initialize(torch,0)]
        assert observation['hash_randomization']==0,'HASH_SEED_NOT_EFFECTIVE_AT_INTERPRETER_START'
        assert observation['environment']['PYTHONHASHSEED']=='0'
        memory=[];uuid=None
        if a.mode=='gpu':
            from prospective_common import OUT,verify
            from gpu_admission import route,device,proc_identity
            cfg=verify();gpu=route(cfg);uuid=cfg['selected_gpu_uuid']
            assert os.environ['CUDA_VISIBLE_DEVICES']==str(gpu)
            evidence=a.result.with_name(a.result.stem+'.GPU_EXCLUSIVITY.jsonl')
            assert device(gpu,cfg,evidence=evidence)['exclusive']
            assert torch.cuda.device_count()==1
            torch.cuda.init();torch.cuda.synchronize();assert torch.cuda.is_initialized()
            torch.cuda.reset_peak_memory_stats();tiny=torch.empty(1,dtype=torch.uint8,device='cuda');torch.cuda.synchronize()
            check=device(gpu,cfg,os.getpid(),binding=proc_identity(os.getpid()),evidence=evidence,require_context=True)
            assert check['exclusive'],check['failure_conditions']
            memory=[dict(peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved())]
            runtime.append(snapshot(torch,'runtime_canary_after_CUDA_init_no_model',0));del tiny
            assert not (OUT/'STOP.json').exists()
        else:assert not torch.cuda.is_initialized()
        atomic_json(a.result,dict(status='PASS',kind='RUNTIME_INITIALIZATION_CANARY' if a.mode=='gpu' else 'CPU_LAUNCH_CHAIN_PROBE',stage=a.stage,actual_environment=selected_environment(os.environ),hash_randomization=observation['hash_randomization'],hash_probe=observation['hash_probe'],runtime=runtime,gpu_uuid=uuid,memory=memory,model_loaded=False,science_trajectory_count=0,numerical_qualification=False,completed_epoch=time.time()))
        return 0
    except BaseException as e:
        failure=dict(status='FAIL',kind='RUNTIME_ENVIRONMENT_PROBE',stage=a.stage,reason=repr(e),traceback=traceback.format_exc(),environment=selected_environment(os.environ),no_retry=True,epoch=time.time(),science_trajectory_count=0)
        atomic_json(a.result.with_suffix('.PROBE_FAILURE.json'),failure)
        if a.mode=='gpu':
            d=Path(os.environ['LING_PROSPECTIVE_RESULTS'])
            if not (d/'STOP.json').exists():atomic_json(d/'STOP.json',failure)
        return 1

if __name__=='__main__':sys.exit(main())
