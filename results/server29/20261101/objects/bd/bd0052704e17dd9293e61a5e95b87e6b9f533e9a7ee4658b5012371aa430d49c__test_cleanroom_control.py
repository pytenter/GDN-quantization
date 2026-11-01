"""Bounded CPU regressions for the new controls; never numerical GPU evidence."""
import ast,copy,hashlib,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from cleanroom_common import D,W,publish,stop,read,sha
import gpu_admission as g

def idle(smi=0,nvml=0):
 total=24576*2**20;reserved=452*2**20;used=15*2**20
 owner=dict(pid=4040,uid=120,start_ticks=3403,comm='Xorg',status='PASS')
 return dict(uuid='GPU_TEST',target=[dict(uuid='GPU_TEST',name='NVIDIA GeForce RTX 3090',total_MiB='24576',used_MiB='15',GPU_util=str(smi))],direct_NVML=dict(status='PASS',memory_v2=dict(total=total,free=total-reserved-used,used=used,reserved=reserved),utilization=dict(gpu=nvml)),direct_NVML_processes=dict(compute=dict(status='PASS',processes=[]),graphics=dict(status='PASS',processes=[dict(pid=4040,usedGpuMemory=4677632)])),SMI_all_processes=dict(status='PASS',processes=[dict(pid='4040',type='G')]),process_ownership=[owner])

class Controls(unittest.TestCase):
 def test_low_util_nonatomic_1_0_is_admitted(self):self.assertEqual(g.assess_window([idle(),idle(1,0),idle()],'GPU_TEST')['status'],'PASS')
 def test_2_1_nonblocking(self):self.assertEqual(g.assess_window([idle(2,1)]*3,'GPU_TEST')['status'],'PASS')
 def test_foreign_compute_rejected(self):
  x=idle();x['direct_NVML_processes']['compute']['processes']=[dict(pid=777,usedGpuMemory=1000)]
  with self.assertRaisesRegex(AssertionError,'EXTERNAL_COMPUTE'):g.assess_window([x]*3,'GPU_TEST')
 def test_sustained_use_rejected(self):
  with self.assertRaisesRegex(AssertionError,'OCCUPANCY'):g.assess_window([idle(50,50)]*3,'GPU_TEST')
 def test_headroom_counts_reserved_and_NVML(self):
  x=idle();x['direct_NVML']['memory_v2']['free']=g.PEAK+g.HEADROOM-1;x['direct_NVML']['memory_v2']['reserved']=x['direct_NVML']['memory_v2']['total']-x['direct_NVML']['memory_v2']['used']-x['direct_NVML']['memory_v2']['free']
  with self.assertRaisesRegex(AssertionError,'HEADROOM'):g.assess_window([x]*3,'GPU_TEST')
 def test_unknown_process_visibility_rejected(self):
  x=idle();x['direct_NVML_processes']['compute']['status']='UNKNOWN'
  with self.assertRaises(AssertionError):g.assess_window([x]*3,'GPU_TEST')
 def test_new_idle_GPU_index_maps_into_original_worker_cfg(self):
  routing=read(D/'EXECUTION_ROUTING.json');uuid,index=next(iter(routing['authorized_devices'].items()))
  cfg=dict(selected_gpu_uuid=uuid,gpu_uuids={})
  with patch.object(g,'inventory',lambda:([dict(uuid=uuid,index=str(index))],None)):
   self.assertEqual(g.route(cfg),index);self.assertEqual(cfg['gpu_uuids'][str(index)],uuid)
 def test_uuid_mismatch_rejected(self):
  with self.assertRaises(AssertionError):g.assess_window([idle()]*3,'OTHER')
 def test_immutable_first_failure(self):
  import cleanroom_common as c
  with tempfile.TemporaryDirectory() as td,patch.object(c,'D',Path(td)):
   c.stop('FIRST');h=c.sha(Path(td)/'STOP.json');c.stop('SECOND');self.assertEqual(c.sha(Path(td)/'STOP.json'),h)
 def test_atomic_outputs_no_overwrite(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'data.json';publish(p,dict(value=1))
   with self.assertRaises(FileExistsError):publish(p,dict(value=2))
   self.assertEqual(read(p)['value'],1)
 def test_formal_source_control_only_reversible(self):
  from formal_checkpoint_control import adapted_source
  src=adapted_source();old=(W/'formal_worker.py').read_text()
  for marker in ["primary = sum(x['KL'] for x in kl)/384",'value = w.engine.full_kl(torch,ref_logits,student_logits)','cache,logits,_ = direction.model_step(torch,capture,cache,tokens[position],position)','metadata = audit.quantize(a.condition,cache,rotation,position,False)']:
   self.assertIn(marker,src);self.assertIn(marker,old)
 def test_checkpoint_array_state_roundtrip_CPU(self):
  import torch
  from formal_checkpoint_control import atomic_torch_save
  import functional_worker as f
  v=dict(cursor=16384,processed_score_positions=list(range(8192,16384,64)),kl=[dict(position=i,KL=float(i)/123456) for i in range(8192,16384,64)],state=torch.arange(12).reshape(3,4))
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'cp.pt';atomic_torch_save(torch,v,p);again=torch.load(p,map_location='cpu',weights_only=False)
   self.assertEqual(f.object_fingerprint(torch,v),f.object_fingerprint(torch,again));self.assertEqual(sum(x['KL'] for x in v['kl']),sum(x['KL'] for x in again['kl']))
   atomic_torch_save(torch,v,p)
   with self.assertRaises(AssertionError):atomic_torch_save(torch,dict(v,cursor=99),p)
 def test_hard_budget_counts_both_active_devices_and_stops_early(self):
  import cleanroom_run as runner,cleanroom_common as common,time
  with tempfile.TemporaryDirectory() as td,patch.object(runner,'D',Path(td)),patch.object(common,'D',Path(td)):
   d=Path(td);publish(d/'EXECUTION_ROUTING.json',dict(authorized_devices={}))
   controller=runner.Controller();controller.completed=[dict(wall_seconds=35.7*3600)]
   controller.active={'a':dict(started_monotonic=time.monotonic()-200),'b':dict(started_monotonic=time.monotonic()-200)}
   value=controller.ledger();self.assertGreater(value['actual_new_GPU_hours'],35.8);self.assertLess(value['actual_new_GPU_hours'],36.0);self.assertTrue((d/'STOP.json').exists())
 def test_slow_actual_parent_forecast_stops_before_large_work(self):
  import cleanroom_run as runner,cleanroom_common as common
  with tempfile.TemporaryDirectory() as td,patch.object(runner,'D',Path(td)),patch.object(common,'D',Path(td)):
   d=Path(td);(d/'runs').mkdir();(d/'formal').mkdir();(d/'qualification_units').mkdir()
   publish(d/'EXECUTION_ROUTING.json',dict(authorized_devices={}))
   publish(d/'preregistration.json',dict(historical_formal_worker_seconds=[3000]))
   publish(d/'runs/fixture__parent.json',dict(elapsed_seconds=2000,loading_seconds=10,memory=[dict(position=512,elapsed=1),dict(position=1024,elapsed=1000)]))
   c=runner.Controller()
   with self.assertRaisesRegex(RuntimeError,'BUDGET_FORECAST'):c.forecast()
   self.assertGreater(read(d/'budget_forecast.json')['projected_total_GPUh'],36)
 def test_only_gpu3_and_all_four_documents_serial(self):
  import cleanroom_run as runner
  routing=read(D/'EXECUTION_ROUTING.json')
  self.assertEqual(runner.validate_single_device_routing(routing),[runner.TARGET_UUID])
  for change in ['device','document','lanes']:
   wrong=copy.deepcopy(routing)
   if change=='device':wrong['authorized_devices']={'GPU_OTHER':2}
   if change=='document':wrong['document_devices']['pg19-test-41603']='GPU_OTHER'
   if change=='lanes':wrong['qualification_parallel_lanes']=2
   with self.assertRaises(AssertionError):runner.validate_single_device_routing(wrong)
 def test_no_other_uuid_can_reach_admission(self):
  import cleanroom_run as runner
  c=runner.Controller()
  with patch.object(runner.g,'admit') as admit:
   with self.assertRaisesRegex(AssertionError,'NON_GPU3'):c.job('GPU_OTHER','canary','pg19-test-41603')
   admit.assert_not_called()
 def test_controller_executor_is_single_worker(self):
  code=ast.parse((W/'cleanroom_run.py').read_text())
  pools=[n for n in ast.walk(code) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='ThreadPoolExecutor']
  self.assertEqual(len(pools),2)
  self.assertTrue(all(next(k.value.value for k in n.keywords if k.arg=='max_workers')==1 for n in pools))
 def test_launch_environment_before_fresh_interpreter(self):
  from cleanroom_common import environment
  routing=read(D/'EXECUTION_ROUTING.json');u=next(iter(routing['authorized_devices']));env=environment(u);env['CUDA_VISIBLE_DEVICES']=''
  command=[sys.executable,'-B','-c',"import os,sys,json;from frozen_runtime_profile import validate_environment;v=validate_environment();print(json.dumps(dict(hash=hash('cleanroom-fixed-seed'),flags=sys.flags.hash_randomization,env=v['environment'])))"]
  values=[]
  for unused in range(2):
   p=subprocess.run(command,cwd=W,env=env,text=True,capture_output=True);self.assertEqual(p.returncode,0,p.stderr);values.append(json.loads(p.stdout))
  self.assertEqual(values[0],values[1]);self.assertEqual(values[0]['flags'],0)
  env.pop('PYTHONHASHSEED');p=subprocess.run(command,cwd=W,env=env,text=True,capture_output=True);self.assertNotEqual(p.returncode,0)

if __name__=='__main__':
 import io,time
 stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls))
 out=Path(sys.argv[1]) if len(sys.argv)>1 else D/'audits/CPU_CONTROL_TESTS.json'
 report=dict(status='PASS_CPU_CONTROL_ONLY' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,skips=len(result.skipped),no_GPU_or_model=True,source_SHA256={n:sha(W/n) for n in ['gpu_admission.py','cleanroom_common.py','cleanroom_worker.py','formal_checkpoint_control.py','cleanroom_run.py','cleanroom_finalize.py','test_cleanroom_control.py']},log=stream.getvalue(),epoch=time.time())
 publish(out,report);print(stream.getvalue());assert result.wasSuccessful() and not result.skipped
