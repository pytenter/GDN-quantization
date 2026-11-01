#!/usr/bin/env python3
"""Independent read-only completed Phase0 verifier. No Torch, signal or writes."""
import argparse,collections,hashlib,json,math,subprocess,sys,time
from pathlib import Path
T='QWEN_GDN_DUAL_M_EQUIVALENT_SPEEDUP_V1'
UUID='GPU-a78176ed-f97f-c54f-6201-5a01b570f422'
CONDITIONS=['FP_NATIVE','H_C128','H_DUAL_M_REFERENCE']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def process(pid):
 p=Path('/proc')/str(pid)
 if not sys.platform.startswith('linux'):return {'status':'NOT_CHECKED_NON_LINUX'}
 if not p.exists():return {'status':'GONE','pid':pid}
 raw=(p/'stat').read_text();return {'status':'LIVE','pid':pid,'start_tick':raw[raw.rfind(')')+2:].split()[19],'cmdline':(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();r=a.root;checks={};details=[]
 def check(name,value):checks[name]=bool(value)
 A=read(r/'AUTHORIZATION_PHASE0.json');M=read(r/'SOURCE_MANIFEST.json');P=read(r/'PLAN.json');B=read(r/'REFERENCE_BINDING.json')
 check('task_auth_profile_only',A['task_id']==T and A['status']=='PASS' and A['allowed_modes']==['profile'] and A['gpu']==0 and A['gpu_uuid']==UUID)
 check('source_manifest_auth_binding',sha(r/'SOURCE_MANIFEST.json')==A['source_manifest_sha256'])
 check('plan_auth_binding',sha(r/'PLAN.json')==A['plan_sha256'])
 check('reference_auth_binding',sha(r/'REFERENCE_BINDING.json')==A['reference_binding_sha256'] and B['status']=='PASS')
 for i,e in enumerate(M['execution_sources']):
  p=Path(e['path']);p=p if p.is_absolute() else r/p
  check('execution_source_'+str(i),p.is_file() and sha(p)==e['sha256'] and p.stat().st_size==e['bytes'])
 phase=r/'execution/profile';res=read(phase/'RESULT.json');ctrl=read(phase/'CONTROLLER_RESULT.json');L=read(r/'GPU_RESOURCE_LEDGER.json')
 check('no_failure',(not (r/'FIRST_FAILURE.json').exists()) and (not (r/'CONTROLLER_FAILURE.json').exists()))
 check('worker_and_controller_PASS',res['status']=='PASS' and ctrl['status']=='PASS')
 check('worker_result_identity',res['task_id']==T and res['mode']=='profile' and res['authorization_sha256']==sha(r/'AUTHORIZATION_PHASE0.json') and res['source_manifest_sha256']==sha(r/'SOURCE_MANIFEST.json') and res['plan_sha256']==sha(r/'PLAN.json') and res['reference_binding_sha256']==sha(r/'REFERENCE_BINDING.json'))
 units=res['completed_units'];check('exact12records',len(units)==12)
 expected={(c,128,j,False,'SHORT_NONINSTRUMENTED_BASELINE') for c in CONDITIONS for j in range(3)}|{(c,128,0,True,'INSTRUMENTED_PROFILE') for c in CONDITIONS}
 actual=[]
 for i,e in enumerate(units):
  p=r/e['path'];row=read(p);u=e['unit'];key=(u['condition'],u['length'],u['repeat'],u['instrumented'],u['stage']);actual.append(key)
  check('record_hash_'+str(i),sha(p)==e['sha256'] and p.stat().st_size==e['bytes'])
  check('record_identity_'+str(i),row['status']=='PASS' and row['task_id']==T and row['condition']==u['condition'] and row['length']==128 and row['repeat']==u['repeat'] and row['instrumented']==u['instrumented'] and row['source_manifest_sha256']==A['source_manifest_sha256'] and row['plan_sha256']==A['plan_sha256'] and row['reference_binding_sha256']==A['reference_binding_sha256'] and row['input_ids_sha256']==B['fixture']['tokens_canonical_sha256'])
  q=u['condition']!='FP_NATIVE';n=256;check('record_consumption_'+str(i),row['warmup_tokens']==128 and row['consumed_tokens']==n and row['measured_decode_tokens']==128 and row['native_core_calls']==24*n and row['QDQ_calls']==(n if q else 0) and row['layer_writebacks']==(24*n if q else 0) and row['measured_QDQ_calls']==(128 if q else 0) and row['next_token_consumption_checks']==(24*(n-1) if q else 0))
  if q:
   h=row['hadamard_audit'];check('H_counts_'+str(i),h['calls']==24*n and len(h['layers'])==24 and h['phase_counts']=={'chunk_prefill':24,'recurrent_decode':24*(n-1)})
  check('record_timing_'+str(i),math.isfinite(row['decode_window_wall_seconds']) and row['decode_window_wall_seconds']>0 and abs(row['decode_tokens_per_second']-128/row['decode_window_wall_seconds'])<1e-12 and len(row['per_token_wall_dispatch_seconds'])==128)
  check('FP32_carrier_'+str(i),row['persistent_recurrent_state_tensor_bytes']==50331648)
  if u['instrumented']:
   x=row['chrome_trace'];p=r/x['path'];check('trace_hash_'+str(i),p.is_file() and sha(p)==x['sha256'] and p.stat().st_size==x['bytes']);check('GPU_profile_observed_'+str(i),row['profile']['CUDA_device_activity_count']>0 and len(row['per_token_CUDA_stream_ms'])==128)
  else:check('uninstrumented_separate_'+str(i),row['profile'] is None and row['per_token_CUDA_stream_ms']==[])
  details.append({'path':e['path'],'unit':u,'consumed_tokens':row['consumed_tokens'],'QDQ_calls':row['QDQ_calls'],'writebacks':row['layer_writebacks'],'next_consumption_checks':row['next_token_consumption_checks'],'decode_wall_seconds':row['decode_window_wall_seconds'],'tokens_per_second':row['decode_tokens_per_second']})
 check('unique_expected_exact_units',set(actual)==expected and len(set(actual))==len(actual))
 micro=read(phase/'CODEC_MICRO.json');d=micro['stage_diagnostic'];check('real_fixture_shape_dtype',micro['status']=='PASS' and micro['fixture_shape']==[1,32,128,128] and micro['fixture_dtype']=='torch.float32' and micro['fixture_bytes']==2097152 and len(micro['fixture_sha256'])==64)
 check('codec_three_timings',len(micro['rows'])==3)
 check('stage_reference_output_byte_exact',d['status']=='PASS_BYTE_EXACT_INSTRUMENTED_VS_ORIGINAL' and set(d['output_parity'])=={'codes','rows','columns','QDQ'} and all(x['status']=='BYTE_EXACT' for x in d['output_parity'].values()))
 check('stage_LS_missing_stays_unknown',d['LS_device_attributed_us'] is not None or d['LS_device_attribution_status']=='UNKNOWN_NOT_MEASURED')
 jobs=L['dispatches'];check('single_profile_dispatch_only',len(jobs)==1 and jobs[0]['mode']=='profile' and jobs[0]['gpu']==0 and jobs[0]['gpu_uuid']==UUID)
 check('exit_and_NVML_release_ledger',all(j['process_exited'] and j['NVML_released'] and j['returncode']==0 for j in jobs) and L['all_own_GPUs_released'] and not L['final_NVML_error'])
 check('budget_under2GPUh',L['cumulative_GPU_seconds']<=7200 and abs(L['cumulative_GPU_hours']*3600-L['cumulative_GPU_seconds'])<1e-8 and L['hard_limit_not_exceeded'])
 started=read(phase/'CONTROLLER_STARTED.json');proc={'controller':process(started['pid']),'workers':[process(j['pid']) for j in jobs]}
 if sys.platform.startswith('linux'):
  check('actual_controller_gone',proc['controller']['status']=='GONE');check('actual_workers_gone',all(x['status']=='GONE' for x in proc['workers']))
  try:
   out=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True);apps=[s.strip() for s in out.splitlines() if s.strip()];check('actual_GPU0_released',not any(UUID in s for s in apps))
  except Exception as exc:apps={'error':str(exc)};check('actual_GPU0_released',False)
 else:apps='NOT_CHECKED_NON_LINUX'
 output={'task_id':T,'status':'PASS' if all(checks.values()) else 'FAIL','unix':time.time(),'scope':'Read-only exact12profilequalification audit; nooptimizedcandidate/newscience','checks':checks,'records':details,'micro_fixture_sha256':micro['fixture_sha256'],'LS_attribution_status':d['LS_device_attribution_status'],'processes':proc,'NVML_apps':apps,'cumulative_GPU_hours':L['cumulative_GPU_hours'],'limitations':['Profiledeviceintervals do not establish pureCPUdispatch attribution.','NoFP_NATIVEvsHscientificequivalence or optimizedcandidate speedup claimed.','If run offLinux, actualprocess/NVMLrelease requires separatefreshremoteproof.']}
 print(json.dumps(output,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
