"""Actual frozen initializer in CPU-only child interpreters; no model/CUDA."""
import json
import os
from pathlib import Path
import subprocess
import sys
from strict_launch_environment import strict_launch_environment

ROOT='/data/zypan/worktrees/ling-kda-p1-l7-efficient-recovery-v4'
PROBE="""import json,os,sys
sys.path.insert(0, ROOT_PATH)
import torch
from v2_runtime import initialize
assert not torch.cuda.is_initialized()
s=initialize(torch,0)
assert not torch.cuda.is_initialized() and not s['cuda_initialized']
assert s['deterministic_algorithms_enabled'] and not s['warn_only']
assert not s['matmul_tf32'] and not s['cudnn_tf32']
print(json.dumps(dict(status='PASS_FROZEN_RUNTIME_CPU_ONLY',hash_seed=os.environ['PYTHONHASHSEED'],hash_probe=hash('ling-kda-frozen-runtime'),cuda_initialized=s['cuda_initialized'])))
""".replace('ROOT_PATH',repr(ROOT))

if __name__=='__main__':
    env=strict_launch_environment(os.environ,0,'CPU_TEST_ONLY',2)
    assert env['CUDA_VISIBLE_DEVICES']=='2' and env['PYTHONHASHSEED']=='0'
    # No CUDA devices exposed even though the operational env builder is
    # tested for physical index routing. The child calls no model or CUDA op.
    env['CUDA_VISIBLE_DEVICES']=''
    positives=[]
    for _ in range(2):
        r=subprocess.run([sys.executable,'-B','-c',PROBE],env=env,capture_output=True,text=True)
        assert r.returncode==0,(r.stdout,r.stderr)
        positives.append(json.loads(r.stdout.strip()))
    assert positives[0]['hash_probe']==positives[1]['hash_probe']
    missing=dict(env);missing.pop('PYTHONHASHSEED',None)
    r=subprocess.run([sys.executable,'-B','-c',PROBE],env=missing,capture_output=True,text=True)
    assert r.returncode!=0 and 'PYTHONHASHSEED' in r.stderr
    report=dict(status='PASS_ENVIRONMENT_FIX_CPU_ONLY',positive_processes=positives,missing_seed_negative_case_rejected=True,CUDA_exposed=False,CUDA_model_inference=0,GPU_scientific_qualification=False,current_stopped_execution_retried=False)
    p=Path(sys.argv[1]);p.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report))
