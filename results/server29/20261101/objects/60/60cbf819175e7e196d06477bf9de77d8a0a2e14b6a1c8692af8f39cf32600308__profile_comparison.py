#!/usr/bin/env python3
"""Summarize actual matched trace evidence; no causal attribution by subtraction."""
import argparse,json,math
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();r=a.root
j=json.loads((r/'MATCHED_RUNTIME_BENCHMARK.json').read_bytes());out=[]
for p in j['profiles']:
 ph=p['physical_activity'];cpu={x['name']:x for x in p['CPU_averages']};kernels=ph['CUDA_device_activity_costs_by_name']
 def c(name):
  x=cpu.get(name)
  return dict(count=x['count'],count_per_token=x['count']/128,self_CPU_ms_per_token=x['self_cpu_us']/128000,CPU_total_ms_per_token=x['cpu_total_us']/128000) if x else {'status':'NOT_OBSERVED'}
 out.append(dict(condition=p['condition'],trace_identity=ph['trace_identity'],profiled_wall_seconds=p['wall_seconds'],kernel_launches_per_token=ph['CUDA_kernel_activity_count_excluding_memcpy_memset']/128,GPU_kernel_sum_ms_per_token=ph['CUDA_kernel_duration_sum_us']/128000,GPU_active_union_ms_per_token=ph['CUDA_active_interval_union_us']/128000,CPU_owner_scopes=ph['CPU_owner_scopes_from_chrome'],selected_CPU_ops={n:c(n) for n in ['aten::mm','aten::matmul','aten::bmm','aten::isfinite','aten::all','aten::item','aten::_local_scalar_dense','aten::empty','aten::fill_','cudaLaunchKernel','cudaStreamSynchronize','cudaEventSynchronize','MATCH/model_forward','MATCH/finite_and_QDQ']},top_CPU_self=sorted(p['CPU_averages'],key=lambda x:x['self_cpu_us'],reverse=True)[:25],top_GPU_sum=sorted([x for x in kernels if x['classification']=='KERNEL_ACTIVITY'],key=lambda x:x['summed_duration_us'],reverse=True)[:25],runtime_APIs=ph['actual_cuda_runtime_API_events'],correlations=ph['correlation_audit']))
result=dict(status='ACTUAL_PROFILE_COMPARISON_ONLY',profiles=out,scope='Each128tokens, separateprofilerunits, neverpooledwithordinaryTPS',limits=['CPUtotal includesnestedops; self ishostoperatoraccounting notcriticalpathshare.','GPUactivitysum canoverlap, activeunion excludesgaps; neitherreplaceswall.','No exclusivepercentattribution fromwall-union oroperatorprofiles.','No individualguard/observer-off/CUBLAS-toggletrialwasauthorized in frozenbatch; residualsinglecause remainsunknown.'])
with (r/'PERFORMANCE_PROFILE_COMPARISON.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(result,ensure_ascii=False))
