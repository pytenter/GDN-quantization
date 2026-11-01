"""CPU identity audit and prospective control registration, before any new GPU."""
from pathlib import Path
import ast,hashlib,json,os,subprocess,sys,time
W=Path(__file__).resolve().parent;D=Path(os.environ['LING_PROSPECTIVE_RESULTS']);O=Path('/data/zypan/worktrees/ling-kda-prospective-execution05');H=Path('/data/zypan/experiment_runs/LING_KDA_P1_L7_PROSPECTIVE_EXTENSION_4DOC_V1/execution_05')
sys.dont_write_bytecode=True
from cleanroom_common import read,sha,publish as new_publish
def publish(path,value):
 if Path(path).exists():
  assert read(path)==value,('CPU_PREPARATION_IDENTITY_CONFLICT',str(path));return
 new_publish(path,value)
from gpu_telemetry import sample,inventory
from gpu_admission import assess_window
from frozen_runtime_profile import validate_files

def main():
 assert not (D/'RUN_CONTROLLER.json').exists() and not (D/'STOP.json').exists()
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=O,text=True).strip()=='902d7974c89465df882403e96cc8dcc8dc1c0e85'
 validate_files();assert read(D/'audits/G0_ACTUAL_MODEL_INPUT_ROTATION_SHA.json')['status']=='PASS_ACTUAL_BYTES'
 math_names=['functional_worker.py','formal_worker.py','persistent_engine_v1.py','v2_runtime.py','objective.py','run_matched_l7.py','frozen_runtime_profile.py','prospective_common.py']
 math_audit={}
 for n in math_names:
  assert (W/n).read_bytes()==(O/n).read_bytes(),('MATHEMATICAL_IDENTITY_CONFLICT',n)
  math_audit[n]=sha(W/n)
 from formal_checkpoint_control import adapted_source
 adapted=adapted_source()
 assert 'primary = sum(x[\'KL\'] for x in kl)/384' in adapted
 publish(D/'audits/MATHEMATICAL_IDENTITY.json',dict(status='PASS',unchanged_file_SHA256=math_audit,formal_control_adapter='registered reversible string edits; reversing every checkpoint/cursor/output hook exactly restores original source byte-for-byte',adapted_formal_AST_SHA256=hashlib.sha256(ast.dump(ast.parse(adapted),include_attributes=False).encode()).hexdigest(),mathematical_kernel_changes=False))
 # Actual three-sample resource window; retain timestamps and both API values.
 inventory_rows,_=inventory()
 eligible=[r for r in inventory_rows if r['name']=='NVIDIA GeForce RTX 3090' and float(r['used_MiB'])<=64 and float(r['GPU_util'])<=2 and int(r['index'])!=6]
 eligible.sort(key=lambda r:([2,3,0,1,4,5].index(int(r['index']))))
 chosen=eligible[:2];assert chosen,'NO_IDLE_COMPATIBLE_GPU_AT_INITIAL_DISCOVERY'
 uuids={r['uuid']:int(r['index']) for r in chosen}
 publish(D/'audits/INITIAL_DEVICE_DISCOVERY.json',dict(status='INITIAL_SELECTION_NOT_MIGRATION',inventory=inventory_rows,chosen=chosen,prior_preferred_pair_rejected_before_any_binding_or_GPU_launch='GPU2 acquired by UID1003/PID2357739 during CPU preparation; no model worker was launched and no numerical gate retried',human_steering='找.29服务器上面的空闲的GPU来运行就可以了'))
 rows={u:[] for u in uuids}
 for i in range(3):
  for u in uuids:rows[u].append(sample(u))
  if i<2:time.sleep(5)
 publish(D/'audits/INITIAL_RESOURCE_SAMPLES.json',dict(samples=rows))
 results={u:assess_window(rows[u],u) for u in uuids}
 for u,index in uuids.items():assert all(int(s['target'][0]['index'])==index for s in rows[u])
 publish(D/'resource_admission.json',dict(status='PASS_INITIAL_THREE_SAMPLE_WINDOW',results=results,samples=rows))
 publish(D/'resource_baselines.json',{u:j['graphics_baseline'] for u,j in results.items()})
 selection=read(D/'FROZEN_4DOC_SELECTION.json')
 expected=[('pg19-test-41603','H_to_L7_a50','H'),('pg19-test-2544','H_to_L6_a50','H_to_L7_a50'),('pg19-test-7987','H_to_L7_a50','H_to_L7R_a25'),('pg19-test-26183','H_to_L6_a50','H_to_L7R_a25')]
 assert [(x['document'],x['P1_predicted_rotation'],x['L7_predicted_rotation']) for x in selection['documents']]==expected
 ordered=list(uuids)
 mapping={}
 for name,index in [('pg19-test-41603',2),('pg19-test-26183',3)]:
  same=[u for u,i in uuids.items() if i==index];mapping[name]=same[0] if same else min(ordered,key=lambda u:list(mapping.values()).count(u))
 for name in ['pg19-test-2544','pg19-test-7987']:mapping[name]=min(ordered,key=lambda u:list(mapping.values()).count(u))
 publish(D/'EXECUTION_ROUTING.json',dict(execution=str(D),source_worktree=str(W),reference_experiment=str(D),authorized_devices=uuids,document_devices=mapping,formal_global_order=selection['execution_order'],formal_serial=True,qualification_parallel_lanes=len(uuids),original_checkpoint_contract_sha256=sha(D/'FROZEN_SPEC.json'),initial_device_selection='latest human authorized idle compatible RTX3090s on .29; idle compatible devices selected before any new GPU run; same-UUID FP-prefix reuse prioritized'))
 publish(D/'FP_PAIRING_CONTRACT_GATE.json',dict(status='PASS',reference_experiment=str(D),spec_sha256=sha(D/'FROZEN_SPEC.json'),mode='CLEANROOM_OWN_FP_SAME_DOC_UUID_FROZEN_RUNTIME'))
 # Verify the two allowed historical FP assets before symlinking. No old writes.
 import torch
 import functional_worker as f
 assert not torch.cuda.is_initialized()
 reuse={};not_reused=[]
 for doc,h in [('pg19-test-41603','367ad2083285fd5a74c69ead6b84e9dbe443dfd22e628ab038cf0966cc10a0a7'),('pg19-test-26183','2e6abbe74fc5266fd26c3b74da4977cdd3849382534852503fb68a10e0f77f4f')]:
  key=doc+'__FP';cp=H/'runs'/(key+'__8192.pt');parent=H/'runs'/(key+'__parent.json');oldq=read(H/'qualification_units'/(key+'.json'))
  assert sha(cp)==h==oldq['checkpoint_sha256'];p=read(parent);assert sha(parent)==oldq['parent_sha256']
  if p['host']['gpu_uuid']!=mapping[doc]:
   not_reused.append(dict(document=doc,reason='different initial selected UUID: rebuild own prefix, do not impose cross-GPU equality or grandfather labels',old_UUID=p['host']['gpu_uuid'],new_UUID=mapping[doc],old_checkpoint_sha256=h));continue
  for stage in ['parent','resumed_parity']:
   profile=read(H/'runs'/(key+'__'+stage+'.RUNTIME_PROFILE.json'));assert profile['status']=='PASS_ACTUAL_FROZEN_RUNTIME_PROFILE' and profile['forward_kernels_executed']
  saved=torch.load(cp,map_location='cpu',weights_only=False);entry=next(x for x in selection['documents'] if x['document']==doc)
  assert saved['next_position']==8192 and saved['frozen_spec_sha256']==sha(D/'FROZEN_SPEC.json') and saved['token_sha256']==entry['document_identity']['token_file_sha256']
  assert saved['audit_state'] is None and saved['quantizer_state']['rotation_sha256'] is None
  assert f.object_fingerprint(torch,saved['cache'])==saved['cache_fingerprint']
  assert all(k in saved for k in ['torch_rng','cuda_rng','python_rng','numpy_rng'])
  reuse[key]=dict(original_checkpoint=str(cp),checkpoint_sha256=h,original_parent=str(parent),parent_sha256=sha(parent),original_qualification_sha256=sha(H/'qualification_units'/(key+'.json')),current_fresh64_required=True)
  (D/'runs'/(key+'__8192.pt')).symlink_to(cp);(D/'runs'/(key+'__parent.json')).symlink_to(parent)
  del saved
 publish(D/'reused_FP_bindings.json',dict(status='ELIGIBLE_SAME_UUID_FP_PREFIXES_VERIFIED_PENDING_CURRENT_FRESH64',units=reuse,not_reused=not_reused,other_historical_checkpoint_reuse=False))
 old=Path('/data/zypan/experiment_runs/LING_KDA_P1_L7_MIXED_OUTCOME_RESOLUTION_V1');hist=[]
 for p in old.rglob('*__formal.NATURAL_EXIT.json'):
  j=read(p)
  if j.get('natural_exit'):hist.append(dict(path=str(p),sha256=sha(p),seconds=j['wall_seconds']))
 publish(D/'preregistration.json',dict(TASK='LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1',scientific_protocol='LING_KDA_P1_L7_PROSPECTIVE_EXTENSION_4DOC_V1',preregistration_sha256='54d6d24fbc3142a692ef1a6c04a5617cf6b0abe6702a662722f34fc1b6141722',selection_sha256=sha(D/'FROZEN_4DOC_SELECTION.json'),FROZEN_SPEC_sha256=sha(D/'FROZEN_SPEC.json'),frozen_epoch=selection['frozen_epoch'],FULL_32K_EXECUTION_AUTHORIZED=True,MAX_CONCURRENT_GPUS=2,MAX_TOTAL_NEW_GPUH=36.0,initial_devices=uuids,historical_formal_measurements=hist,historical_formal_worker_seconds=[r['seconds'] for r in hist],new_qualification_workers_planned=24-len(reuse),new_parent_workers=12-len(reuse),current_fresh64_workers=12,canary_model_workers=len(uuids),formal_workers=24,formal_scientific_trajectories=12,planned_long_checkpoint_cursor=16384,planned_overlap_tokens=64,global_formal_order=selection['execution_order'],no_new_scientific_keys=True,stats_unit='document',old_MIXED_not_combined_n6=True))
 publish(D/'engineering_change_manifest.json',dict(status='REGISTERED_BEFORE_GPU',base_commit='902d7974c89465df882403e96cc8dcc8dc1c0e85',changes=['new single scheduler/launch envelope and budget ledger','new3sample resource admission with<=2% low-util warning','same frozen runtime/profile mathematical files','control-only atomic16384 checkpoint and64-step exact planned resume overlap','CPU result and SHA delivery'],unchanged_math=math_audit,formal_adapter_exact_source_reversibility=True,extra_headroom_bytes=2*2**30,low_util_threshold_percent=2,resource_window_samples=3,resource_sample_interval_seconds=5,hard_new_GPUh=36.,protective_stop_GPUh=35.8,own_process_graceful_stop_seconds=60,own_identity_verified_SIGTERM_then15seconds_SIGKILL_for_budget_or_safety_only=True,automatic_failed_science_retry=False,automatic_second_attempt=False,allowlisted_devices_frozen_before_gpu=uuids,epoch=time.time()))
 print('CLEANROOM_CPU_PREPARATION_PASS; GPU_NOT_STARTED',len(hist))

if __name__=='__main__':main()
