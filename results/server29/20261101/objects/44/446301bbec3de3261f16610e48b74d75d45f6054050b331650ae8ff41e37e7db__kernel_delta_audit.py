#!/usr/bin/env python3
from pathlib import Path
import json,math
B=Path(__file__).resolve().parents[1];j=json.loads((B/'MATCHED_RUNTIME_BENCHMARK.json').read_bytes());p={x['condition']:x for x in j['profiles']}
a={x['name']:x for x in p['OLD_FP']['physical_activity']['CUDA_device_activity_costs_by_name'] if x['classification']=='KERNEL_ACTIVITY'}
b={x['name']:x for x in p['NEW_FP']['physical_activity']['CUDA_device_activity_costs_by_name'] if x['classification']=='KERNEL_ACTIVITY'}
rows=[]
for name in sorted(a.keys()|b.keys()):
 old=a.get(name,{});new=b.get(name,{})
 rows.append(dict(name=name,old_count=old.get('count',0),new_count=new.get('count',0),delta_per_token=(new.get('count',0)-old.get('count',0))/128,old_kernel_sum_ms_per_token=old.get('summed_duration_us',0)/128000,new_kernel_sum_ms_per_token=new.get('summed_duration_us',0)/128000))
changed=[x for x in rows if x['delta_per_token']]
mm=[]
for cond,x in p.items():
 op=next(z for z in x['CPU_averages'] if z['name']=='aten::mm');mm.append(dict(condition=cond,count_per_token=op['count']/128,CPU_self_ms_per_token=op['self_cpu_us']/128000,CPU_total_ms_per_token=op['cpu_total_us']/128000,operator_associated_device_ms_per_token=op['device_attributed_us']/128000))
obj=dict(status='ACTUAL_TWO_TRACE_NAME_COUNT_AUDIT',OLD_distinct_kernel_names=len(a),NEW_distinct_kernel_names=len(b),shared_names=len(a.keys()&b.keys()),OLD_only_names=sorted(a.keys()-b.keys()),NEW_only_names=sorted(b.keys()-a.keys()),kernel_count_delta_per_token=math.fsum(x['delta_per_token'] for x in changed),changed_name_counts=changed,all_kernel_names=rows,aten_mm=mm,limits=['Name/count equality does not prove equal arguments, arithmetic results, or algorithm state.','Host self time can include uninstrumented library work or waiting; it is not guaranteed pure CPU arithmetic or an exclusive causal share.','Captured profiler timing is not ordinary unprofiled timing.','No new single-setting test for CUBLAS or individual safety checks; root trigger remainsUNRESOLVED.'])
(B/'KERNEL_NAME_DELTA_AUDIT.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:obj[k] for k in ['OLD_distinct_kernel_names','NEW_distinct_kernel_names','shared_names','kernel_count_delta_per_token','aten_mm']}))
