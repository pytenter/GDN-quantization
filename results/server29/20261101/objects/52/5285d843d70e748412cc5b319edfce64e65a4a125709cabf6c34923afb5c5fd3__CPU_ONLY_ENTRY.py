
from pathlib import Path
import json,os,runpy,sys,time
R=Path(sys.argv[1]);OUT=Path(sys.argv[2]);OLD=sys.argv[3]
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
assert os.environ.get('PYTHONDONTWRITEBYTECODE')=='1'
assert os.environ.get('OMP_NUM_THREADS')==os.environ.get('MKL_NUM_THREADS')=='1'
sys.dont_write_bytecode=True
diag=runpy.run_path(str(R/'tests/test_pace_scale_diagnostics.py'),run_name='scale_failure_context_cpu_fixture')
diagnostic=diag['run']()
assert diagnostic['status']=='PASS' and diagnostic['cases']==6 and diagnostic['Torch_imported'] is False
with (OUT/'DIAGNOSTICS_CPU.json').open('x') as f:json.dump(diagnostic,f,sort_keys=True,allow_nan=False);f.write('\n')
print(json.dumps(dict(phase='CPU_DIAGNOSTICS_COMPLETE',cases=6,Torch_imported=False)),flush=True)
import torch
assert not torch.cuda._initialized
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
blocked=[]
def forbidden(name):
 def stop(*a,**kw):
  blocked.append(name)
  raise AssertionError('CUDA operation forbidden in CPU-only test: '+name)
 return stop
guarded=[]
for name in ('_lazy_init','init','set_device','current_device','device_count','synchronize',
             'manual_seed','manual_seed_all','set_rng_state','set_rng_state_all',
             'reset_peak_memory_stats','memory_allocated','max_memory_allocated',
             'memory_reserved','max_memory_reserved','empty_cache','stream','current_stream'):
 if hasattr(torch.cuda,name):setattr(torch.cuda,name,forbidden('torch.cuda.'+name));guarded.append('torch.cuda.'+name)
torch.Tensor.cuda=forbidden('Tensor.cuda');guarded.append('Tensor.cuda')
torch.nn.Module.cuda=forbidden('Module.cuda');guarded.append('Module.cuda')
sys.argv=[str(R/'tests/test_pace_scale_boundary.py'),'--device','cpu','--old-quantizer',OLD,'--out',str(OUT/'BOUNDARY_AND_STRESS_CPU.json')]
runpy.run_path(sys.argv[0],run_name='__main__')
result=json.loads((OUT/'BOUNDARY_AND_STRESS_CPU.json').read_bytes())
assert result['status']=='PASS' and result['Torch_tests_run'] is True and result['GPU_tests_run'] is False
assert result['boundary']['actual_cuda'] is False and result['boundary']['model_loads']==0 and result['boundary']['AIME_inputs']==0
assert result['cpu_stress']['actual_cuda'] is False and result['cpu_stress']['GPU_calls']==0
assert [(x['decode_period'],x['tokens']) for x in result['cpu_stress']['cases']]==[(1,16384),(8,16384)]
assert not blocked and not torch.cuda._initialized
assert 'pace_runtime' not in sys.modules and 'src.pace_runtime' not in sys.modules
proof=dict(status='PASS_CPU_ONLY_CUDA_ENTRYPOINTS_BLOCKED_AND_UNUSED',CVD='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
 guarded_CUDA_entrypoints=guarded,forbidden_CUDA_calls=blocked,CUDA_context_initialized=torch.cuda._initialized,
 model_runtime_module_imported=False,model_loads=0,AIME_inputs=0,qualification_GPU_credit=0)
with (OUT/'CPU_ONLY_ASSERTIONS.json').open('x') as f:json.dump(proof,f,sort_keys=True,allow_nan=False);f.write('\n')
print(json.dumps(dict(phase='CPU_ONLY_ENTRY_COMPLETED',status='PASS',CUDA_context_initialized=False)),flush=True)
