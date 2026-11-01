"""CPU-only recovery/routing tests; no fixture qualifies a science device."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import gpu_admission as g
import qualified_completion as q
import recovery_controller as c
from recovery_state import atomic_json,sha

def pair():
    a=dict(status='PASS',begin=0,endpoint=8256,observer_enabled=True,document='fixture',condition='FP',host=dict(gpu_uuid='OLD',kernel='k',driver='d',torch='t',cuda_runtime='c',selector='BT16_W8'))
    for field in q.FIELDS:a[field]='FROZEN_SHA'
    b=copy.deepcopy(a);b.update(begin=8192,observer_enabled=False);b['host']['gpu_uuid']='NEW'
    return a,b

class QualificationTests(unittest.TestCase):
    def test_real_target_device_required(self):
        a,b=pair();self.assertTrue(q.compare_exact(a,b,'NEW'))
        with self.assertRaises(AssertionError):q.compare_exact(a,b,'OLD')
    def test_each_exact_hash_failure_stops(self):
        for field in q.FIELDS:
            a,b=pair();b[field]='CHANGED'
            with self.assertRaises(AssertionError):q.compare_exact(a,b,'NEW')
    def test_parent_only_is_not_qualified(self):
        a,b=pair();b['observer_enabled']=True
        with self.assertRaises(AssertionError):q.compare_exact(a,b,'NEW')
    def test_runtime_identity_failure_stops(self):
        a,b=pair();b['host']['selector']='OTHER'
        with self.assertRaises(AssertionError):q.compare_exact(a,b,'NEW')
    def test_no_gpu6_label_inheritance(self):
        a,b=pair();b['host']['gpu_uuid']='OLD'
        with self.assertRaises(AssertionError):q.compare_exact(a,b,'NEW')
    def test_worker_ticket_isolation(self):
        self.assertNotEqual(g.ticket_path('UUID4'),g.ticket_path('UUID5'))
    def test_qualified_lane_resume_without_pid(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td);atomic_json(d/'LANE_COMPLETE_UUID4.json',dict(status='PASS',uuid='UUID4',units=[]))
            with patch.object(q,'D',d):q.wait_lanes([('UUID4',None,None)])
    def test_completed_new_unit_never_launches_again(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td);(d/'formal').mkdir();(d/'preregistration').mkdir()
            atomic_json(d/'SOURCE_CODE_FREEZE.json',dict(source_commit='FIXTURE',config_sha256='CONFIG'))
            protocol=d/'preregistration/DOCUMENT_SELECTION_AND_EXCLUSION_RULES.json';atomic_json(protocol,dict(synthetic=True))
            identity=dict(document='fixture',rotation='FP',stage='formal',source_commit='FIXTURE',config_sha256='CONFIG',token_sha256='TOKEN',contract_sha256=sha(protocol),execution=str(d),target_gpu_uuid='NEW')
            r=d/'formal/fixture__FP__formal.json';e=d/'formal/fixture__FP__formal.NATURAL_EXIT.json'
            atomic_json(r,dict(status='PASS',CPU_synthetic_only=True));atomic_json(e,dict(natural_exit=True,result_sha256=sha(r),unit_identity=identity))
            with patch.object(c,'D',d),patch.object(c.p,'verify',lambda:dict(selected_gpu_uuid='NEW')),patch.object(c,'entry_for',lambda _:dict(token_file_sha256='TOKEN')),patch.object(c,'admit',side_effect=AssertionError('DUPLICATE_SCIENTIFIC_LAUNCH')):
                c.run('fixture','FP','formal');c.run('fixture','FP','formal')

    def test_actual_master_formal_order_and_twelve_barrier(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td);(d/'runs').mkdir()
            selection=dict(experiment='CPU_FIXTURE_ONLY',execution_order=[dict(document='fixture'+str(i),conditions=['FP','P1','L7']) for i in range(4)])
            units=[dict(document=e['document'],rotation=r) for e in selection['execution_order'] for r in e['conditions']]
            mapping={e['document']:'NEW' for e in selection['execution_order']}
            spec=dict(new_units=units);atomic_json(d/'FROZEN_SPEC.json',spec);atomic_json(d/'PHASE0_HISTORICAL_ASSET_AUDIT.json',dict(status='PASS'))
            atomic_json(d/'FROZEN_4DOC_SELECTION.json',selection)
            atomic_json(d/'EXECUTION_ROUTING.json',dict(original_checkpoint_contract_sha256=sha(d/'FROZEN_SPEC.json'),document_devices=mapping,authorized_devices={'NEW':4}))
            gates=[]
            for u in units:
                a,b=pair();a.update(document=u['document'],condition=u['rotation']);b.update(document=u['document'],condition=u['rotation'])
                if u['rotation']!='FP':
                    a['integrity']=b['integrity']=dict(passed=True,every_token_layer_qdq_count=8256*18,actual_next_kernel_equal_post_qdq_count=8255*18)
                k=u['document']+'__'+u['rotation'];p=d/'runs'/(k+'__parent.json');f=d/'runs'/(k+'__resumed_parity.json');cp=d/'runs'/(k+'__8192.pt')
                cp.write_bytes(b'CPU_SYNTHETIC_ONLY');a['checkpoint_sha256']=b['checkpoint_sha256']=sha(cp)
                atomic_json(p,a);atomic_json(f,b)
                gates.append(dict(u,status='PASS',qualified_gpu_uuid='NEW',parent_sha256=sha(p),resumed_sha256=sha(f),checkpoint_sha256=sha(cp)))
            gate=d/'ENGINEERING_QUALIFICATION_PRE_FORMAL.json';atomic_json(gate,dict(status='PASS',count=12,units=gates));calls=[]
            with patch.object(q,'D',d),patch.object(c,'D',d),patch.object(c,'frozen_selection',lambda:selection),patch.object(q,'resource_plan',lambda:dict(status='CPU_FIXTURE')),patch.object(c,'run',lambda doc,rot,stage:calls.append((doc,rot,stage))),patch.dict(os.environ,{'LING_EXECUTION_LANE':''}):q.run_main()
            self.assertEqual(calls,[(u['document'],u['rotation'],'formal') for u in units])
            self.assertTrue((d/'PROSPECTIVE_COMPLETE.json').exists())
            bad=dict(status='PASS',count=12,units=gates[:-1]);gate.write_text(json.dumps(bad));calls.clear()
            with patch.object(q,'D',d),patch.object(c,'D',d),patch.object(c,'frozen_selection',lambda:selection),patch.object(c,'run',lambda *args:calls.append(args)),patch.dict(os.environ,{'LING_EXECUTION_LANE':''}):
                with self.assertRaises(AssertionError):q.run_main()
            self.assertEqual(calls,[])

class GraphicsTests(unittest.TestCase):
    def setUp(self):
        self.owner=dict(status='PASS',pid=4040,uid=120,start_ticks=3403,comm='Xorg')
        self.nvml=dict(pid=4040,usedGpuMemory=4677632,gpuInstanceId=4294967295,computeInstanceId=4294967295)
        self.xml=dict(pid='4040',type='G',name='/usr/lib/xorg/Xorg',used_memory='4 MiB')
        self.s=dict(uuid='UUID',direct_NVML_processes=dict(compute=dict(status='PASS',processes=[]),graphics=dict(status='PASS',processes=[self.nvml])),SMI_all_processes=dict(status='PASS',processes=[self.xml]),process_ownership=[self.owner])
        self.b=dict(NVML_graphics=[self.nvml],SMI_graphics=[self.xml],ownership=[self.owner])
    def test_baseline_remains_visible(self):self.assertTrue(g.graphics_allowed(self.s,self.b))
    def test_unknown_graphics_is_not_idle(self):
        self.s['direct_NVML_processes']['graphics']['status']='UNKNOWN';self.assertFalse(g.graphics_allowed(self.s,self.b))
    def test_pid_reuse_is_rejected(self):
        self.s['process_ownership']=[dict(self.owner,start_ticks=9999)];self.assertFalse(g.graphics_allowed(self.s,self.b))
    def test_new_graphics_process_is_rejected(self):
        self.s['SMI_all_processes']['processes']=self.s['SMI_all_processes']['processes']+[dict(self.xml,pid='5555')];self.assertFalse(g.graphics_allowed(self.s,self.b))

if __name__=='__main__':
    import io,time
    stream=io.StringIO();r=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__)))
    d=Path(os.environ['LING_PROSPECTIVE_RESULTS']);(d/'NEW_RECOVERY_TESTS.log').write_text(stream.getvalue())
    atomic_json(d/'NEW_RECOVERY_TEST_REPORT.json',dict(status='PASS' if r.wasSuccessful() else 'FAIL',test_count=r.testsRun,skips=len(r.skipped),CPU_synthetic_only=True,GPU_scientific_units=0,source_sha256=sha(__file__),epoch=time.time()))
    print(stream.getvalue());assert r.wasSuccessful() and not r.skipped
