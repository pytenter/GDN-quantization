"""Deterministic original/candidate production-controller race; CPU fixtures only."""
import importlib.util,json,os,sys,tempfile,time,unittest
from pathlib import Path
from unittest import mock
HERE=Path(__file__).resolve().parent
FROZEN=Path('/data/zypan/analyses/qwen_gdn_pace48_aime26_seed1_60_v1_4gpu_continue_control_v2_20261026')
sys.path.insert(0,str(FROZEN/'src'))
import pace_control as C

def module(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

class ActualControllerRaceTests(unittest.TestCase):
    def run_fixture(self,controller_path,label):
        M=module(controller_path,label)
        with tempfile.TemporaryDirectory(prefix='PACE4_exit_guard_CPU_') as name:
            root=Path(name);(root/'scripts').mkdir()
            for n in ['source_manifest.json','GPU_PERMIT.json']:(root/n).write_text('{}')
            (root/'REUSE_ADMISSION.json').write_text('{"reused_ids":[]}');(root/'FOUR_GPU_PLAN.json').write_text('{"jobs":[]}')
            code="""import pathlib,sys,time
r=pathlib.Path(sys.argv[sys.argv.index('--root')+1]);u=sys.argv[sys.argv.index('--unit')+1];t=r/'execution'/u
(t/'RESULT.json').write_text('{"CPU_MOCK_ONLY":true}')
while not (r/'EXIT_NOW').exists():time.sleep(.001)
raise SystemExit(0)
"""
            (root/'scripts/pace_four_gpu_worker.py').write_text(code)
            calls=[]
            def exit_during_identity_check(identity):
                calls.append(identity);(root/'EXIT_NOW').touch();until=time.monotonic()+5
                while time.monotonic()<until:
                    p=Path('/proc')/str(identity['pid'])/'stat'
                    if not p.exists() or p.read_text().rsplit(')',1)[1].split()[0]=='Z':break
                    time.sleep(.001)
                return C.process_identity(identity['pid'])==identity if (Path('/proc')/str(identity['pid'])/'cmdline').exists() else False
            smoke=dict(status='PASS',actual_cuda=True,configs={'R2':{},'R3':{}},actual_mask_gate=dict(status='PASS',cases=[dict(status='PASS')]*48))
            # Receipt/device mocks are labelled CPU-only. The worker Popen,
            # production dispatch, identity checks, owning exit and release are real.
            with mock.patch.object(C,'authorize',lambda r:None),mock.patch.object(C,'model_guard',lambda r:None),mock.patch.object(C,'storage_guard',lambda r:None),mock.patch.object(C,'same_process',exit_during_identity_check),mock.patch.object(C,'query_gpu',lambda:dict(apps=[])),mock.patch.object(C,'verify_result',lambda *a:dict(result=dict(paired_full_model_smoke=smoke))),mock.patch.object(M,'publish_unit',lambda *a:dict(manifest_sha256='CPU_MOCK')):
                c=M.Controller(root);c.device=lambda *a,**k:None;ok=c.run_unit('scale_repair_gates','SCALE_REPAIR_GATES_GPU0',0)
            self.assertTrue(calls);self.assertEqual(c.intervals[0]['returncode'],0);self.assertTrue(c.intervals[0]['released'])
            return ok,c.first,c.completed
    def test_original_production_controller_false_stops_legal_exit0(self):
        ok,failure,completed=self.run_fixture(FROZEN/'scripts/pace_four_gpu_controller.py','original_exit_race_controller')
        self.assertFalse(ok);self.assertIn('PID/starttick/argv changed',failure['error']);self.assertFalse(completed)
    def test_candidate_production_controller_reaps_exit0_and_completes_once(self):
        ok,failure,completed=self.run_fixture(HERE/'candidate/pace_four_gpu_controller.py','candidate_exit_race_controller')
        self.assertTrue(ok);self.assertIsNone(failure);self.assertEqual(len(completed),1)

if __name__=='__main__':unittest.main(verbosity=2)
