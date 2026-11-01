"""Two actual CPU child faults through frozen Controller.job registration cleanup."""
from pathlib import Path
import copy,hashlib,json,os,subprocess,sys,tempfile,time,unittest
from unittest import mock
os.environ['CUDA_VISIBLE_DEVICES']=''
W=Path('/data/zypan/worktrees/ling-kda-4doc-cleanroom-gpu3-single-v2');D=Path('/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU3_SINGLE_V2');sys.path.insert(0,str(W))
import cleanroom_run as m
import cleanroom_common as c
records=[]
class RegistrationFixtures(unittest.TestCase):
 def run_fault(self,which):
  with tempfile.TemporaryDirectory(prefix='cpu_registration_fixture_',dir=D/'audits') as temp:
   root=Path(temp);routing=c.read(D/'EXECUTION_ROUTING.json');c.publish(root/'EXECUTION_ROUTING.json',routing);c.publish(root/'SOURCE_CODE_FREEZE.json',dict(source_commit='CPU_SYNTHETIC_REGISTRATION_FIXTURE_ONLY'))
   baseline={m.TARGET_UUID:dict(CPU_SYNTHETIC_GRAPHICS_BASELINE=True)};c.publish(root/'resource_baselines.json',baseline)
   original_publish=c.publish;original_popen=subprocess.Popen;original_identity=m.g.proc_identity;children=[];injected=[]
   def publish(path,value,**kwargs):
    if which=='LAUNCH_PUBLICATION' and str(path).endswith('.LAUNCH.json') and not injected:
     injected.append(which);raise OSError('CPU_SYNTHETIC_LAUNCH_PUBLICATION_FAILURE')
    return original_publish(path,value,**kwargs)
   def fail_identity(pid):
    injected.append(which);raise OSError('CPU_SYNTHETIC_POST_SPAWN_IDENTITY_FAILURE')
   def stop(reason,**kwargs):
    p=root/'STOP.json'
    if not p.exists():original_publish(p,dict(status='STOP',reason=reason,epoch=time.time(),CPU_SYNTHETIC=True,**kwargs))
   def cpu_child(command,**kwargs):
    env=dict(kwargs['env']);env['CUDA_VISIBLE_DEVICES']='';env['CPU_REGISTRATION_FIXTURE']='1';kwargs['env']=env
    process=original_popen([sys.executable,'-B','-u','-c','import os,time; assert os.environ["CUDA_VISIBLE_DEVICES"]==""; print("CPU_ONLY_CHILD_NO_MODEL_OR_CUDA",flush=True); time.sleep(0.2)'],**kwargs);children.append(process);return process
   with mock.patch.object(m,'D',root),mock.patch.object(m,'verify_launch_registration',return_value=True),mock.patch.object(m,'publish',side_effect=publish),mock.patch.object(m,'stop',side_effect=stop),mock.patch.object(m.g,'admit',return_value=dict(graphics_baseline=baseline[m.TARGET_UUID])),mock.patch.object(m.subprocess,'Popen',side_effect=cpu_child),mock.patch.object(m.g,'proc_identity',side_effect=fail_identity if which=='IDENTITY' else original_identity):
    controller=m.Controller()
    with self.assertRaisesRegex(RuntimeError,'POST_SPAWN_REGISTRATION_FAILED_CHILD_EXITED'):controller.job(m.TARGET_UUID,'parent','pg19-test-41603','FP')
    self.assertEqual(len(children),1);child=children[0];self.assertEqual(child.poll(),0);self.assertEqual(child.wait(),0);self.assertFalse((Path('/proc')/str(child.pid)).exists());self.assertEqual(controller.active,{})
    launches=list((root/'jobs').glob('*.LAUNCH.json'));exits=list((root/'jobs').glob('*.EXIT.json'));self.assertEqual(len(launches),1);self.assertEqual(len(exits),1)
    launch=c.read(launches[0]);receipt=c.read(exits[0]);self.assertTrue(receipt['registration_failure']);self.assertTrue(receipt['wait_confirmed_child_exit']);self.assertEqual(receipt['returncode'],0);self.assertEqual(receipt['wall_seconds'],receipt['completed_monotonic']-receipt['started_monotonic']);self.assertEqual(receipt['spawned_PID'],child.pid)
    if which=='IDENTITY':self.assertIsNone(launch['identity']);self.assertIsNone(receipt['cleanup_identity'])
    else:self.assertEqual(launch['identity'],receipt['cleanup_identity']);self.assertEqual(launch['identity']['pid'],child.pid)
    first=c.sha(root/'STOP.json');self.assertEqual(c.read(root/'STOP.json')['stage'],'POST_SPAWN_REGISTRATION')
    with self.assertRaisesRegex(AssertionError,'EXISTING_FIRST_STOP'):controller.job(m.TARGET_UUID,'parent','pg19-test-41603','FP')
    self.assertEqual(c.sha(root/'STOP.json'),first);self.assertEqual(len(children),1)
    ledger=c.read(root/'gpu_resource_ledger.json');self.assertEqual(ledger['active_workers'],[]);self.assertEqual(ledger['completed_launch_to_exit_GPU_hours'],receipt['wall_seconds']/3600)
    records.append(dict(case=which,status='PASS',actual_CPU_child_PID=child.pid,child_returncode=0,child_wait_confirmed=True,child_no_longer_in_proc=True,source_job_path_called=True,first_STOP_sha256=first,first_STOP_unchanged_on_attempted_reentry=True,launch_sha256=c.sha(launches[0]),exit_sha256=c.sha(exits[0]),actual_CPU_child_launch_to_exit_seconds=receipt['wall_seconds'],synthetic_controller_ledger_hours_not_GPU_scientific_usage=ledger['completed_launch_to_exit_GPU_hours'],all_paths_in_temporary_CPU_fixture_root=True))
 def test_1_metadata_failure_actual_child_waited(self):self.run_fault('IDENTITY')
 def test_2_LAUNCH_publication_failure_actual_child_waited(self):self.run_fault('LAUNCH_PUBLICATION')
if __name__=='__main__':
 suite=unittest.defaultTestLoader.loadTestsFromTestCase(RegistrationFixtures);result=unittest.TextTestRunner(verbosity=2).run(suite)
 report=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=2,tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),CPU_ONLY=True,no_model_or_CUDA=True,GPU_workers=0,science_trajectory_count=0,numerical_qualification=False,actual_CPU_children=len(records),mock_scope='admission/source-freeze validation stubbed only inside temporary CPU fixture root; Popen replaced actual scientific command with real CPU child; original Controller.job registration/cleanup/STOP/ledger called; actual live /proc identity used except intentional identity fault',records=records,source_SHA256={n:c.sha(W/n) for n in ['cleanroom_run.py','resource_evidence.py','cleanroom_finalize.py','gpu_admission.py']},test_script_sha256=c.sha(Path(__file__)))
 out=D/'audits/CPU_POST_SPAWN_REGISTRATION_REGRESSION.json'
 with out.open('x') as f:json.dump(report,f,sort_keys=True,indent=2);f.write('\n')
 print(json.dumps(dict(report_path=str(out),sha256=c.sha(out),**report),sort_keys=True));sys.exit(0 if result.wasSuccessful() else 1)
