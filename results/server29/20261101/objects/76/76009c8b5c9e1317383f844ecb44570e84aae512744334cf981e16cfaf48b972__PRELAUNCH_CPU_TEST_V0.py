"""Hardware port and source/wait guards; CPU fixtures never qualify a GPU."""
from pathlib import Path
import sys,ast,copy,tempfile,unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'control/runtime'))
import resource_monitor as rm
from test_resource_monitor_cpu import Sampler,Clock,FakeGraphicsPolicy
class HardwarePort(unittest.TestCase):
 def test_3090_exact_model_index_uuid_total(self):
  clock=Clock();sample=Sampler(clock);u='GPU-07d9c381-888d-1111-a944-56abcc583573'
  s=sample(u);s['target'][0].update(index='5',name='NVIDIA GeForce RTX 3090',total_MiB='24576')
  record=dict(snapshot=s)
  with tempfile.TemporaryDirectory() as td,patch.dict(rm.PAIRS,{5:u},clear=True):
   m=rm.ResourceMonitor(td,5,u,lambda u:s,FakeGraphicsPolicy())
   self.assertEqual(m._safety(record)['free_bytes'],s['direct_NVML']['memory_v2']['free'])
   for key,value in [('name','NVIDIA GeForce RTX 4090'),('index','6'),('total_MiB','24564'),('uuid',rm.PAIRS.get(0,'WRONG'))]:
    t=copy.deepcopy(record);t['snapshot']['target'][0][key]=value
    with self.assertRaises(rm.ResourceFailure):m._safety(t)
 def test_unrequested_device_rejected(self):
  with tempfile.TemporaryDirectory() as td:
   with self.assertRaises(rm.ResourceFailure):rm.ResourceMonitor(td,4,'UNREQUESTED',lambda u:{},FakeGraphicsPolicy())
 def test_unlimited_time_preserves_visibility_and_no_progress(self):
  r=Path(__file__).resolve().parents[1];import json
  c=json.loads((r/'authorization/ACTIVE_QUALIFICATION_BUDGET.json').read_text())
  for k in ('MAX_TOTAL_GPUH','MAX_4090_GPUH','MAX_3090_GPUH','MAX_GLOBAL_WALL_SECONDS','MAX_PER_JOB_WALL_SECONDS'):self.assertIsNone(c[k])
  self.assertEqual(c['no_progress_watchdog_seconds'],1800);self.assertFalse(c['formal_AIME_authorized'])
if __name__=='__main__':unittest.main()
