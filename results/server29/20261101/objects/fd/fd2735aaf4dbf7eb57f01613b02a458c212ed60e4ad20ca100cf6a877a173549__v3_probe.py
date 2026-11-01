"""One tiny actual CUDA context; no model load, no science outcome."""
import argparse
import os
import time
import traceback
from v3_common import *
from v3_guard import proc_identity
from v2_runtime import initialize,snapshot

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--gpu',type=int,required=True); ap.add_argument('--document',required=True)
    ap.add_argument('--repeat',type=int,default=0); a=ap.parse_args(); t=tag('guardprobe',a.document,a.gpu,a.repeat)
    start=time.time(); fail=RUNS/(t+'.STOP.json')
    try:
        cfg,_,_=verify(); r=checked(RUNS/(t+'.LAUNCH_AUTHORIZATION.json'))
        assert r['wrapper_sha256']==digest(__file__) and os.environ['CUDA_VISIBLE_DEVICES']==str(a.gpu)
        binding=proc_identity(os.getpid())
        emit(RUNS/(t+'.PROCESS_IDENTITY.json'),dict(identity=binding,gpu=a.gpu,gpu_uuid=r['gpu_uuid'],wrapper_sha256=digest(__file__)))
        emit(RUNS/(t+'.STARTED.json'),dict(pid=os.getpid(),started_epoch=start,gpu_uuid=r['gpu_uuid'],mode='guardprobe'))
        import torch
        runtime=initialize(torch,cfg['seed']); assert torch.cuda.device_count()==1
        x=torch.ones((1024,1024),dtype=torch.float32,device='cuda'); torch.cuda.synchronize()
        snapshots=[]
        for i in range(8 if a.repeat else 2):
            assert float(x.sum())==1048576.
            hw=device(a.gpu,cfg,os.getpid()); snapshots.append(hw)
            assert hw['exclusive'],('ACTUAL_GPU_GUARD_PROBE_FAIL',hw['failure_conditions'])
            if a.repeat: time.sleep(1)
        runtime_end=snapshot(torch,'after_tiny_cuda_guard_probe',cfg['seed']); end=time.time()
        emit(RUNS/(t+'.json'),dict(status='PASS',gpu=a.gpu,gpu_uuid=r['gpu_uuid'],document=a.document,mode='guardprobe',
            runtime=[runtime,runtime_end],hardware=snapshots,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            started_epoch=start,completed_epoch=end,accounted_gpu_hours=(end-start)/3600,model_loaded=False,
            scientific_result=False,numerical_512_qualification='NOT_RUN',runner_sha256=digest(__file__)))
        print('TINY_REAL_GPU_GUARD_PROBE_PASS',a.gpu,flush=True)
    except Exception as e:
        emit(fail,dict(status='STOP',error=repr(e),traceback=traceback.format_exc(),pid=os.getpid(),gpu=a.gpu,
            document=a.document,mode='guardprobe',started_epoch=start,stopped_epoch=time.time()))
        stop(repr(e),'guardprobe',run_stop=str(fail)); raise

if __name__=='__main__': main()
