#!/usr/bin/env python3
"""CPU-only read-only reconstruction of the once-only matched batch."""
import argparse, csv, hashlib, json, math, os
from pathlib import Path
import statistics, subprocess
from chrome_trace_categories import corrected_profile
TASK='QWEN_GDN_HISTORICAL_THROUGHPUT_RECONCILIATION_V1'
MAIN=['OLD_FP','NEW_FP','OLD_H_C128','NEW_H_C128']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def save(p,j):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(j,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def stats(values):
 if not values:return None
 x=sorted(values);return dict(N=len(x),min=x[0],max=x[-1],mean=math.fsum(x)/len(x),median=statistics.median(x),p95=x[max(0,math.ceil(.95*len(x))-1)])
def same_process(ident):
 p=Path('/proc')/str(ident['pid'])/'stat'
 if not p.exists():return False
 return p.read_text().rsplit(')',1)[1].split()[19]==str(ident['start_tick'])
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();r=a.root
 batch=read(r/'execution/BATCH_RESULT.json');ledger=read(r/'GPU_RESOURCE_LEDGER.json');manifest=read(r/'SOURCE_SHA256_MANIFEST.json')
 assert batch['ledger_sha256']==sha(r/'GPU_RESOURCE_LEDGER.json')
 assert ledger['cumulative_GPU_hours']<=.75 and ledger['all_own_GPUs_released']
 assert not same_process(read(r/'execution/BATCH_STARTED.json')['identity'])
 assert all(not same_process(d['identity']) for d in ledger['dispatches'])
 live=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True)
 assert not {d['pid'] for d in ledger['dispatches']}.intersection(int(x.split(',')[0]) for x in live.splitlines() if x.strip())
 for row in manifest['execution_sources']:
  p=r/row['path'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
 for row in manifest['external_dependencies']:
  p=Path(row['path']);assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
 units=[];raw=[];profiles=[];telemetry=[json.loads(x) for x in (r/'execution/RESOURCE_TELEMETRY.jsonl').read_text().splitlines() if x.strip()]
 for done in batch['completed_units']:
  p=r/done['path'];assert p.stat().st_size==done['bytes'] and sha(p)==done['sha256'];j=read(p)
  assert j['status']=='PASS' and j['source_manifest_sha256']==sha(r/'SOURCE_SHA256_MANIFEST.json')
  assert j['model_forward_calls']==256 and j['decoder_layer_calls']==8192
  assert j['GDN_chunk_calls']==24 and j['GDN_recurrent_calls']==6120 and j['measured_GDN_recurrent_calls']==3072
  assert j['measured_tokens']==128 and j['cache_retained_at_measurement_start']==128 and j['cache_final_length']==256
  assert all(row['before']==t and row['after']==t+1 and row['all8attention_KV_lengths']==[t+1]*8 for t,row in enumerate(j['actual_KV_growth']))
  assert j['actual_positions']==[[[[t]],[[t]],[[t]]] for t in range(256)]
  env=read(p.parent/'ENVIRONMENT.json');assert env['architecture_sha256']=='90d929129ffc835d2652c604925c4f3842bc6e401e174ec6f0db2285dfb8f85a'
  sample=[x for x in telemetry if x['unit']==done['unit'] and j['decode_window_start_monotonic']<=x['monotonic']<=j['decode_window_end_monotonic']]
  values={};cpu=[]
  for x in sample:
   fields=[v.strip() for v in x['GPU'].split(',')]
   assert fields[1]==j['GPU_UUID']
   for i,key in [(4,'memory_free_MiB'),(5,'GPU_utilization_percent'),(6,'power_limit_W'),(7,'power_draw_W'),(8,'SM_clock_MHz'),(9,'memory_clock_MHz'),(10,'temperature_C')]:
    try:values.setdefault(key,[]).append(float(fields[i]))
    except ValueError:pass
  for first,second in zip(sample,sample[1:]):
   if 'worker_proc_stat' not in first or 'worker_proc_stat' not in second:continue
   def ticks(z):
    f=z['worker_proc_stat'].rsplit(')',1)[1].split();return int(f[11])+int(f[12])
   delta=second['monotonic']-first['monotonic']
   if delta>0:cpu.append((ticks(second)-ticks(first))/os.sysconf('SC_CLK_TCK')/delta*100)
  covariates={key:stats(v) for key,v in values.items()};covariates['worker_CPU_percent']=stats(cpu);covariates['sample_count']=len(sample)
  row=dict(unit=done['unit'],condition=j['condition'],repeat=j['repeat'],profile=j['profile'],status='PASS',
   measured_tokens=128,decode_wall_seconds=j['decode_wall_seconds'],tokens_per_second=j['tokens_per_second'],wall_ms_per_token=j['wall_ms_per_token'],
   host_iteration_p50_ms=1000*j['host_iteration_p50_seconds'],host_iteration_p95_ms=1000*j['host_iteration_p95_seconds'],
   peak_allocated_bytes=j['peak_allocated_bytes'],peak_reserved_bytes=j['peak_reserved_bytes'],GPU_UUID=j['GPU_UUID'],
   deterministic=env['deterministic'],intraop_threads=env['intraop_threads'],interop_threads=env['interop_threads'],attention_backend=env['attention_backend'],
   GPU_SM_clock_median_MHz=covariates.get('SM_clock_MHz',{}).get('median'),GPU_power_median_W=covariates.get('power_draw_W',{}).get('median'),
   GPU_utilization_median_percent=covariates.get('GPU_utilization_percent',{}).get('median'),CPU_utilization_observed_mean_percent=covariates['worker_CPU_percent']['mean'] if covariates['worker_CPU_percent'] else None,
   model_forwards=256,GDN_recurrent_calls=6120,measured_GDN_recurrent_calls=3072,cache_measurement_start=128,cache_final=256,
   result_SHA256=done['sha256'],source_manifest_SHA256=j['source_manifest_sha256'])
  units.append(row);raw.append(dict(unit=done['unit'],record=j,environment=env,measured_window_telemetry=covariates))
  if j['profile']:
   t=j['chrome_trace'];corrected=corrected_profile(r/t['path'],t['sha256'],t['bytes'],('MATCH/model_forward','MATCH/finite_and_QDQ'))
   profiles.append(dict(unit=done['unit'],condition=j['condition'],wall_seconds=j['decode_wall_seconds'],CPU_averages=j['CPU_averages'],physical_activity=corrected))
 groups={}
 for cond in MAIN+['DIAG_NEW_DETERMINISM_OFF','DIAG_NEW_INTRAOP24']:
  rr=[x for x in raw if x['record']['condition']==cond and not x['record']['profile']]
  groups[cond]=dict(status='ACTUAL_THREE_REPEATS' if len(rr)==3 else 'NOT_RUN' if not rr else 'INCOMPLETE_REPEATS',N=len(rr),
   TPS=stats([x['record']['tokens_per_second'] for x in rr]),wall_ms_per_token=stats([x['record']['wall_ms_per_token'] for x in rr]),
   pooled_iteration_host_ms=stats([v*1000 for x in rr for v in x['record']['host_iteration_seconds']]),
   environments=[x['environment'] for x in rr],units=[x['unit'] for x in rr])
 result=dict(task_id=TASK,status='CPU_REDUCED_COMPLETED_BATCH' if batch['status']=='PASS' else 'ACTUAL_PARTIAL_BATCH_AFTER_FAIL_CLOSED_STOP',
  batch_status=batch['status'],ledger_SHA256=sha(r/'GPU_RESOURCE_LEDGER.json'),source_manifest_SHA256=sha(r/'SOURCE_SHA256_MANIFEST.json'),
  cumulative_GPU_hours=ledger['cumulative_GPU_hours'],all_own_GPUs_released=True,true_process_identities_exited=True,
  groups=groups,units=units,profiles=profiles,auxiliary_admission=batch.get('auxiliary_decision'),
  protocol='One initial1token prefill,128warmup totalforwards,128measured; native actualrotarypositions0..255/KVgrowth+32layers+24GDNverified. Wholemodelsamefixedinput sameGPU/timing, independentcaches; no sampling.',
  limits=['Comparing recoveredoldsource on current shared checkpoint/environment is not full recreation of unrecorded historical clocks/UUID/flags/packages.',
   'BothFPstates nativeFP32/BF16weights, not fullFP32-weightmodel.',
   'H old/new source math AST same but pref illQDQandmatrixmemoization differ; no numeric equivalence claimed.',
   'Auxiliarysingle-switchsettings are independentdiagnostics, never substitutesfor scientificNEWbaseline.',
   'CPUelapsed,self,GPUactivitysum,unionandwall are distinct; nestedCPU/GPUrows notadditive. Wall-active isnotpureCPUoverhead.',
   'Telemetrysampled every2seconds withinactualmeasurementwindow; missingcovariatesUNKNOWN, no inventedhistoricalhardwarevalues.'])
 save(r/'MATCHED_RUNTIME_BENCHMARK.json',result)
 with (r/'MATCHED_RUNTIME_BENCHMARK.csv').open('x',newline='') as f:
  if units:w=csv.DictWriter(f,fieldnames=list(units[0]));w.writeheader();w.writerows(units)
  else:w=csv.DictWriter(f,fieldnames=['status','reason']);w.writeheader();w.writerow(dict(status='NOT_RUN',reason='No valid completed GPU units; no numerical TPS invented'))
 save(r/'MATCHED_RECORD_VERIFICATION.json',dict(task_id=TASK,status='PASS_RECORD_AND_RELEASE_VERIFICATION',valid_completed_units=len(units),
  complete_main_4x3=all(groups[c]['N']==3 for c in MAIN),source_unchanged=True,actual_per_token_position_KV_GDN_full_forward=True,
  physical_traces_parsed=len(profiles),GPU_occupied_hours=ledger['cumulative_GPU_hours'],all_bound_processes_exited=True,NVML_tracked_PIDs_absent=True))
 print(json.dumps({'status':result['status'],'groups':{k:dict(N=v['N'],TPS=v['TPS']) for k,v in groups.items()},'profiles':len(profiles),'GPUh':ledger['cumulative_GPU_hours']}))

if __name__=='__main__':main()
