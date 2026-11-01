"""Read-only CPU evidence after GPU STOP. Does not repair or relaunch anything."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import ast,hashlib,json,pathlib,sys,traceback
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import gpu_worker as W
import torch
F,N,arch=W.originals()
result=dict(audit='CPU_ONLY_FIRST_FAILURE_CAUSE',GPU_HOURS_USED=0,cuda_initialized_before=torch.cuda.is_initialized(),frozen_worker_sha256=hashlib.sha256((ROOT/'src/gpu_worker.py').read_bytes()).hexdigest(),comparison_function_source='unchanged frozen numerical_recovery.compare AST',missing_global='tsha',global_present='tsha' in N.compare.__globals__,attempt_modified=False,GPU_retry=False)
try:
    N.compare(torch.ones(1,1,2,2),torch.ones(1,1,2,2),'model_fp32')
except NameError as exc:
    result.update(reproduced=True,error=repr(exc),traceback=traceback.format_exc())
else:raise RuntimeError('Expected original recorded namespace failure did not reproduce')
coordinator=(ROOT/'src/coordinator.py').read_text()
result['dispatch_guard_audit']=dict(minimal_codec_CPU_check_required='MINIMAL_ADAPTER_CPU_REGRESSION.json' in coordinator,engineering_import_PASS_required='CPU_ENGINEERING_IMPORT_CHECK.json' in coordinator,source_hash_freeze_required='Frozen new source changed' in coordinator,root_cause='A missing dependency in new AST execution namespace, plus no successful-engineering-import prerequisite on dispatch. Original comparison algorithm and source files unchanged.')
result['cuda_initialized_after']=torch.cuda.is_initialized();assert result['cuda_initialized_after'] is False
with (ROOT/'analysis/FIRST_FAILURE_CAUSE_AUDIT.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(dict(reproduced=result['reproduced'],error=result['error'],GPU_retry=False,cuda_initialized=False)))
