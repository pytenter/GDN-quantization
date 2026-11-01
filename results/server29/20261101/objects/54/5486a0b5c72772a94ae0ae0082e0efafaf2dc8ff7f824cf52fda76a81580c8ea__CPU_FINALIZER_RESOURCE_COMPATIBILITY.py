"""Six bounded CPU-only resource/finalizer fixtures; no model, CUDA, or science."""
from pathlib import Path
import ast,copy,hashlib,json,os,sys,tempfile,time,unittest
os.environ['CUDA_VISIBLE_DEVICES']=''
W=Path('/data/zypan/worktrees/ling-kda-4doc-cleanroom-gpu3-single-v2')
D=Path('/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU3_SINGLE_V2')
sys.path.insert(0,str(W))
import cleanroom_finalize as f
import resource_evidence as re
UUID=f.UUID
class FinalizerResourceFixtures(unittest.TestCase):
 def root(self):
  td=tempfile.TemporaryDirectory(prefix='cpu_finalizer_fixture_',dir=D/'audits');self.addCleanup(td.cleanup);return Path(td.name)
 def dump(self,p,value):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,sort_keys=True)+'\n')
 def owner(self):return dict(pid=987654321,uid=1000,start_ticks=4242)
 def snapshot(self,owner):
  foreign=987654322
  return dict(uuid=UUID,epoch=time.time(),direct_NVML_processes=dict(compute=dict(status='PASS',processes=[dict(pid=owner['pid'],usedGpuMemory=20000000000),dict(pid=foreign,usedGpuMemory=12345)]),graphics=dict(status='PASS',processes=[])),SMI_all_processes=dict(status='PASS',processes=[dict(pid=owner['pid'],type='C'),dict(pid=foreign,type='C')]),process_ownership=[dict(status='PASS',**owner),dict(pid=foreign,uid=9999,start_ticks=444,status='PASS')])
 def test_1_no_STOP_is_not_resource_failure(self):
  r=f.resource_stop_evidence(None,self.root());self.assertFalse(r['is_resource_gate_STOP']);self.assertEqual(r['errors'],[]);self.assertEqual(r['pinned_failures'],[])
 def test_2_actual_prereg_budget_separates_4090(self):
  prereg=json.loads((D/'preregistration.json').read_text());r=f.resource_budget_context(prereg,1.25)
  self.assertEqual(r['prior_same_task_RTX3090_GPUh'],0);self.assertEqual(r['same_task_cumulative_RTX3090_GPUh'],1.25);self.assertEqual(r['remaining_same_task_RTX3090_GPUh'],34.75)
  self.assertEqual(r['independent_RTX4090_qualification_GPUh'],3.447347587855119);self.assertFalse(r['other_task_resources']['rtx4090_qualification']['charged_to_current_3090_budget'])
 def test_3_actual_events_partial_counters_not_canary_only(self):
  root=self.root();jobs=root/'jobs';jobs.mkdir()
  receipts={}
  for tag,event,kind,position,allocated,reserved in [('p','qualification_progress','parent',2560,16027651072,16125001728),('f','formal_progress','formal1',8704,17000000000,18000000000)]:
   (jobs/(tag+'.log')).write_text(json.dumps(dict(event=event,position=position,peak_allocated=allocated,peak_reserved=reserved))+'\n')
   receipts[tag]=dict(kind=kind,status='FAIL',document='pg19-test-41603',condition='FP')
  rows=f.progress_log_audit(jobs,receipts);r=f.cuda_observations([dict(peak_allocated=15856,peak_reserved=2097152)],rows)
  self.assertEqual(r['max_observed_CUDA_peak_allocated_bytes'],17000000000);self.assertEqual(r['max_observed_CUDA_peak_reserved_bytes'],18000000000)
  self.assertEqual({x['kind'] for x in rows},{'parent','formal1'});self.assertEqual({x['last_confirmed_consumed_tokens'] for x in rows},{2560,8704})
  self.assertTrue(all(x['exact_terminal_cursor'] is None and x['unlogged_tokens_not_inferred'] for x in rows));self.assertEqual(len(r['CUDA_partial_failed_worker_peak_records']),2)
 def test_4_first_STOP_pin_raw_and_exact_owner_NVML(self):
  root=self.root();owner=self.owner();tag='unit';self.dump(root/'jobs'/f'{tag}.LAUNCH.json',dict(identity=owner))
  def fail(*a):raise RuntimeError('UNAUTHORIZED_EXTERNAL_COMPUTE_PROCESS')
  try:re.assess_with_evidence(self.snapshot(owner),UUID,owner,raw_path=root/'jobs/unit.RESOURCE.jsonl',pinned_path=root/'FAILED_RESOURCE_SNAPSHOT.json',assess_fn=fail)
  except re.ResourceEvidenceError as error:
   stop=dict(stage='RUNNING_RESOURCE_SAFETY',reason=str(error),tag=tag,resource_evidence=dict(path=error.evidence_path,sha256=error.evidence_sha256),forbiddenPIDs=error.forbidden_pids)
  self.dump(root/'STOP.json',stop);original=f.sha(root/'STOP.json');r=f.resource_stop_evidence(stop,root)
  self.assertEqual(r['errors'],[]);self.assertEqual(r['first_STOP_sha256'],original);self.assertEqual(f.sha(root/'STOP.json'),original);self.assertTrue(r['pinned_failures'][0]['raw_line_and_snapshot_SHA_verified'])
  own=[x for x in r['observed_process_records'] if x['owned_process_snapshot_identity_status']=='MATCH_PID_UID_STARTTIME'];self.assertTrue(own);self.assertTrue(all(x['owned_NVML_usedGpuMemory_bytes']==[20000000000] for x in own));self.assertEqual(r['pinned_failures'][0]['forbidden_pids'],[987654322])
 def test_5_failed_query_remains_UNKNOWN(self):
  root=self.root();owner=self.owner()
  try:re.record_sampling_failure(RuntimeError('CPU_SYNTHETIC_QUERY_FAILURE'),UUID,owner,raw_path=root/'jobs/unit.RESOURCE.jsonl',pinned_path=root/'FAILED_RESOURCE_SNAPSHOT.json')
  except re.ResourceEvidenceError as error:stop=dict(stage='RUNNING_RESOURCE_SAFETY',reason=str(error),resource_evidence=dict(path=error.evidence_path,sha256=error.evidence_sha256))
  self.dump(root/'STOP.json',stop);r=f.resource_stop_evidence(stop,root)
  self.assertTrue(r['pinned_failures'][0]['raw_line_and_snapshot_SHA_verified']);self.assertEqual(r['pinned_failures'][0]['status'],'UNKNOWN');self.assertTrue(any(x['reason']=='RESOURCE_SNAPSHOT_QUERY_OR_VISIBILITY_UNKNOWN' for x in r['errors']));self.assertTrue(all(x['owned_NVML_usedGpuMemory_bytes'] is None for x in r['observed_process_records']))
 def test_6_formal_in_worker_failed_raw_exact_unit(self):
  root=self.root();owner=self.owner();doc='pg19-test-41603';condition='FP';tag=doc+'__'+condition+'__formal1'
  self.dump(root/'jobs'/(tag+'.LAUNCH.json'),dict(identity=owner));self.dump(root/'jobs'/(tag+'.EXIT.json'),dict(status='FAIL'))
  record=dict(exclusive=False,failure_conditions=['UNAUTHORIZED_EXTERNAL_COMPUTE_PROCESS'],gpu_uuid=UUID,gpu=3,epoch=100.0,snapshot=self.snapshot(owner));path=root/'formal'/(doc+'__'+condition+'__formal.GPU_EXCLUSIVITY.jsonl');self.dump(path,record)
  stop=dict(stage='formal',document=doc,condition=condition,epoch=101.0,reason='UNAUTHORIZED_EXTERNAL_COMPUTE_PROCESS');self.dump(root/'STOP.json',stop);r=f.resource_stop_evidence(stop,root)
  self.assertEqual(r['errors'],[]);self.assertEqual(len(r['in_worker_failure_bindings']),1);b=r['in_worker_failure_bindings'][0];self.assertEqual(b['sha256'],f.sha(path));self.assertTrue(b['reason_matches_first_STOP']);self.assertEqual(b['owner'],owner)
if __name__=='__main__':
 suite=unittest.defaultTestLoader.loadTestsFromTestCase(FinalizerResourceFixtures);result=unittest.TextTestRunner(verbosity=2).run(suite)
 report=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),CPU_ONLY=True,model_loaded=False,CUDA_initialized=False,GPU_worker_count=0,science_trajectory_count=0,numerical_qualification=False,fixture_scope='synthetic resource/finalizer branches only; no scientific results',preregistration_sha256=f.sha(D/'preregistration.json'),source_sha256={n:f.sha(W/n) for n in ['cleanroom_finalize.py','cleanroom_run.py','resource_evidence.py','test_cleanroom_control.py','test_resource_evidence.py']},fixture_script_sha256=f.sha(Path(__file__)))
 out=D/'audits/CPU_FINALIZER_RESOURCE_COMPATIBILITY.json'
 with out.open('x') as stream:json.dump(report,stream,ensure_ascii=False,sort_keys=True,indent=2);stream.write('\n')
 print(json.dumps(dict(report_path=str(out),sha256=f.sha(out),**report),sort_keys=True));sys.exit(0 if result.wasSuccessful() else 1)
