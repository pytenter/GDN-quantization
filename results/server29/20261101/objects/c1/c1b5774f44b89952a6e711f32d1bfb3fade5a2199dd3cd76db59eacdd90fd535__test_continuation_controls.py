"""Stdlib tests of resource amendment, negative admission and stage dependencies."""
import json,sys,tempfile,unittest,copy,subprocess,os
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
import gpu_control as C
import continuation_control as K
class Controls(unittest.TestCase):
 def test_actual_parent_and_science(self):
  results,v=K.admit_parent(R,C.read(R/'GPU_PLAN.json'));self.assertEqual(len(results),7);self.assertTrue(v['science_and_worker_unchanged'])
 def test_finite_original_roster(self):
  plan=C.read(R/'GPU_PLAN.json');old=C.read(R/'inputs/parent_gpu_delivery/GPU_PLAN.json')
  self.assertEqual(plan['units'],old['units'][7:]);self.assertEqual(len(plan['units']),15)
  for k in ('hard_seconds','protective_seconds','max_GPU_hours'):self.assertIsNone(plan[k])
  for k in ('device','same_inputs','source_protocol','automatic_long_horizon'):self.assertEqual(plan[k],old[k])
 def test_no_reauthorization_budget(self):
  p=C.authorize(R,R/'GPU_PERMIT.json');self.assertEqual(p['resource_policy'],'NO_GPU_TIME_BUDGET')
  for key,value in [('ALLOW_GPU_CANARY','NO'),('source_freeze_sha256','x'),('plan_sha256','x'),('max_GPU_hours',.5),('hard_seconds',1800),('protective_seconds',1650),('resource_policy','LIMITED'),('human_separate_authorization',False),('human_authorization_text',''),('no_progress_watchdog_seconds',1799),('cleanup_seconds',0),('device',{})]:
   q=copy.deepcopy(p);q[key]=value
   with tempfile.TemporaryDirectory() as d:
    f=Path(d)/'permit.json';f.write_text(json.dumps(q))
    with self.assertRaises(RuntimeError,msg=key):C.authorize(R,f)
 def test_watchdog_is_not_elapsed_budget(self):
  self.assertFalse(K.stalled(100000,100001));self.assertFalse(K.stalled(100000,101799.9));self.assertTrue(K.stalled(100000,101800));self.assertFalse(K.stalled(101799,101800))
 def test_exact_progress_only(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);a=K.progress_signature(p);(p/'worker.log').write_text('polling chatter');self.assertEqual(a,K.progress_signature(p))
   (p/'STARTED.json').write_text('{}');b=K.progress_signature(p);self.assertNotEqual(a,b);self.assertEqual(b,K.progress_signature(p))
   (p/'PROGRESS_0000.json').write_text('{}');self.assertNotEqual(b,K.progress_signature(p))
 def test_stage_dependencies(self):
  u={x['id']:x for x in C.read(R/'GPU_PLAN.json')['units']};complete={}
  self.assertFalse(K.ready(u['B0_128_R0'],complete));complete.update(B0_32_R0={},B0_32_R1={});self.assertTrue(K.ready(u['B0_128_R0'],complete));self.assertFalse(K.ready(u['B0_128_R1'],complete));complete['B0_128_R0']={};self.assertTrue(K.ready(u['B0_128_R1'],complete));self.assertFalse(K.ready(u['B0_512_R0'],complete));complete['B0_128_R1']={};self.assertTrue(K.ready(u['B0_512_R0'],complete));self.assertFalse(K.ready(u['B0_TPS'],complete));complete.update(B0_512_R0={},B0_512_R1={});self.assertTrue(K.ready(u['B0_TPS'],complete));self.assertFalse(K.ready(u['B1_TPS'],complete))
 def test_resource_bounds(self):
  self.assertEqual(K.reservation(1,2,4),dict(reservation_lower_seconds=1,reservation_upper_seconds=3))
  with self.assertRaises(RuntimeError):K.reservation(3,2,4)
 def test_foreign_identity_cleanup_rejected(self):
  class P:
   pid=123
   def terminate(self):raise AssertionError('Foreign process killed')
  with self.assertRaises(RuntimeError):C.terminate_exact(P(),dict(pid=123,starttick=1),reader=lambda _:dict(pid=123,starttick=2))
 def test_check_does_not_query_GPU(self):
  p=subprocess.run([sys.executable,'-B',str(R/'scripts/gpu_controller.py'),'--root',str(R)],capture_output=True,text=True);self.assertEqual(p.returncode,0,p.stderr);v=json.loads(p.stdout);self.assertEqual(v['GPU_queries'],0);self.assertEqual(v['GPU_dispatches'],0);self.assertEqual(v['finite_units'],15)
if __name__=='__main__':
 s=unittest.defaultTestLoader.loadTestsFromTestCase(Controls);result=unittest.TextTestRunner(verbosity=2).run(s)
 print(json.dumps(dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),GPU_queries=0,GPU_dispatches=0)))
 sys.exit(0 if result.wasSuccessful() else 1)
