"""CPU-only partition/auth and real owning-Popen lifecycle checks. No CUDA credit."""
import ast,copy,json,os,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from unittest import mock
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
import pace_control as C
import pace_four_gpu_admission as A
import pace_four_gpu_controller as M

def jobs():
    return [dict(id='q%02d_%s'%(i,c),condition=c) for i in range(30) for c in ('R2','R3')]

def permit_pair():
    g=dict(status='EXPLICIT_HUMAN_PACE_FOUR_GPU_CONTINUATION',WSCALE_paused=True,TP=1,unique_formal_target=60,server29_CPU_only=True,
           authorized_GPUs=[0,1,2,3],UUIDs=list(A.UUIDS.values()),remote_root=A.REMOTE,archive_root=A.ARCHIVE)
    p=dict(ALLOW_GPU='YES',execution_root=A.REMOTE,execution_revision=A.REVISION,physical_indices=[0,1,2,3],devices=list(A.UUIDS.values()),
           source_manifest_sha256='s',human_authorization_sha256='g',four_gpu_plan_sha256='p',seed=1,max_new_tokens=81920,method_variant=A.METHOD,state_scale_format=A.FORMAT,
           formal_new_samples=46,reused_formal_samples=14,no_automatic_GPU_retry=True,watchdog_seconds=1800,cleanup_seconds=120,
           GPUh_hard_cap=None,total_wall_cap=None,unit_elapsed_cap=None)
    return p,g

class FourGPUAdmissionTests(unittest.TestCase):
    def test_partition_exact_no_duplicate_and_balanced(self):
        j=jobs()
        for n in (0,1,3,14,15,59,60):
            reuse=[x['id'] for x in j[:n]];out=A.partition_jobs(j,reuse)
            self.assertEqual({x['unit'] for x in out},set(x['id'] for x in j)-set(reuse));self.assertEqual(len(out),60-n)
            self.assertTrue(all(x['gpu']==i%4 and x['gpu_uuid']==A.UUIDS[i%4] for i,x in enumerate(out)))
            counts=[sum(x['gpu']==g for x in out) for g in range(4)];self.assertLessEqual(max(counts)-min(counts),1)
    def test_partition_preserves_canonical_sequence(self):
        j=jobs();reuse=[j[i]['id'] for i in (1,5,9)]
        self.assertEqual([x['unit'] for x in A.partition_jobs(j,reuse)],[x['id'] for x in j if x['id'] not in reuse])
    def test_reject_bad_rosters(self):
        for j,r in [(jobs()[:-1],[]),(jobs()+[jobs()[0]],[]),(jobs(),['unknown']),(jobs(),[jobs()[0]['id']]*2)]:
            with self.assertRaises(ValueError):A.partition_jobs(j,r)
    def test_full_exact_grant(self):
        p,g=permit_pair();self.assertTrue(A.check_permit(p,g,'s','g','p'))
    def test_reject_scope_source_protocol_and_cap_mutations(self):
        changes=dict(ALLOW_GPU='NO',execution_root='/tmp/foreign',physical_indices=[0],devices=list(reversed(A.UUIDS.values())),source_manifest_sha256='x',human_authorization_sha256='x',four_gpu_plan_sha256='x',seed=2,max_new_tokens=100,method_variant='other',state_scale_format='FP16_ONLY',formal_new_samples=47,no_automatic_GPU_retry=False,watchdog_seconds=10,cleanup_seconds=10,GPUh_hard_cap=1,total_wall_cap=1,unit_elapsed_cap=1)
        for k,v in changes.items():
            p,g=permit_pair();p[k]=v
            with self.subTest(k=k),self.assertRaises(RuntimeError):A.check_permit(p,g,'s','g','p')
    def test_reject_grant_device_and_TP_mutations(self):
        for k,v in dict(authorized_GPUs=[0],UUIDs=[A.UUIDS[0]],remote_root='/foreign',archive_root='/foreign',TP=4,server29_CPU_only=False,WSCALE_paused=False,unique_formal_target=120).items():
            p,g=permit_pair();g[k]=v
            with self.subTest(k=k),self.assertRaises(RuntimeError):A.check_permit(p,g,'s','g','p')
    def test_progress_is_real_fields_not_timestamp(self):
        a=dict(phase='FORWARD_COMMITTED',generated_tokens=128,input_forwards=144,monotonic=1)
        b=dict(a,monotonic=999);self.assertEqual(M.progress_signature(a),M.progress_signature(b))
        b['input_forwards']=145;self.assertNotEqual(M.progress_signature(a),M.progress_signature(b))
    def test_numeric_and_protocol_sources_byte_identical(self):
        manifest=json.loads((ROOT/'provenance/PRE4GPU_SOURCE_MANIFEST.json').read_bytes());by={x['path']:x for x in manifest['files']}
        import hashlib
        for n in ('src/pace_quantizer.py','src/pace_buffer.py','src/pace_runtime.py','src/pace_exponent_gates.py','src/pace_scale_diagnostics.py','protocol/sampler.py','protocol/common.py','protocol/aime26_scorer_v4.py','protocol/aime26_frozen.jsonl','method_spec.json','vendor/modeling_qwen3_5.py'):
            self.assertEqual(hashlib.sha256((ROOT/n).read_bytes()).hexdigest(),by[n]['sha256'],n)
    def test_formal_math_validator_unchanged(self):
        current=ast.parse((ROOT/'src/pace_control.py').read_text());oldroot=ROOT.parent/'QWEN_GDN_PACE48_AIME26_SEED1_60_V1_SCALE_EXPONENT_REPAIR_20261025'
        # On target55, the original immutable source is available at the predecessor path.
        if not (oldroot/'src/pace_control.py').exists():oldroot=Path('/data/zypan/analyses/qwen_gdn_pace48_aime26_seed1_60_v1_scale_exponent_repair_20261025')
        old=ast.parse((oldroot/'src/pace_control.py').read_text())
        find=lambda t:next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='validate_formal')
        self.assertEqual(ast.dump(find(current)),ast.dump(find(old)))

@unittest.skipUnless(sys.platform.startswith('linux'),'Real production /proc lifecycle runs on target55 CPU, not Mac')
class ActualOwnedCPULifecycleTests(unittest.TestCase):
    def test_foreign_or_reused_identity_never_signalled(self):
        p=subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(30)'])
        try:
            identity=C.process_identity(p.pid);bad=dict(identity,starttick=identity['starttick']+1)
            with self.assertRaises(RuntimeError):C.terminate_owned(p,bad)
            self.assertIsNone(p.poll());C.terminate_owned(p,identity);p.wait(timeout=5)
            self.assertEqual(p.returncode,-15);self.assertTrue(C.release_state(True,dict(apps=[]),p.pid,identity))
        finally:
            if p.poll() is None:p.terminate();p.wait()
    def execute_fixture(self,returncode):
        with tempfile.TemporaryDirectory(prefix='pace4_CPU_Popen_') as td:
            r=Path(td);(r/'scripts').mkdir();(r/'source_manifest.json').write_text('{}');(r/'GPU_PERMIT.json').write_text('{}')
            (r/'REUSE_ADMISSION.json').write_text('{"reused_ids":[]}');(r/'FOUR_GPU_PLAN.json').write_text('{"jobs":[]}')
            code='''import pathlib,sys,json,os,time
r=pathlib.Path(sys.argv[sys.argv.index('--root')+1]);u=sys.argv[sys.argv.index('--unit')+1];t=r/'execution'/u
(t/'RESULT.json').write_text('{"status":"PASS"}');time.sleep(.1)
raise SystemExit('''+str(returncode)+')\n'
            (r/'scripts/pace_four_gpu_worker.py').write_text(code)
            smoke=dict(status='PASS',actual_cuda=True,configs={k:dict(status='COMPLETE',nonfinite=False,runtime_error=None,condition=k,head_mask_sha256='CPU_MOCK',quantizer_audit=dict(dense_recurrent_carrier=False)) for k in ('R2','R3')},actual_mask_gate=dict(status='PASS',actual_cuda=True,layers=24,head_mask_sha256='CPU_MOCK',cases=[dict(status='PASS')]*48))
            # Synthetic gate receipt exercises lifecycle control only; never CUDA credit.
            with mock.patch.object(C,'authorize',lambda r:None),mock.patch.object(C,'model_guard',lambda r:None),mock.patch.object(C,'storage_guard',lambda r:None),mock.patch.object(C,'unit_target',lambda r,s,u:Path(r)/'execution'/u),mock.patch.object(C,'verify_result',lambda *a:dict(status='PASS',result=dict(paired_full_model_smoke=smoke))),mock.patch.object(C,'query_gpu',lambda:dict(apps=[])),mock.patch.object(M,'publish_unit',lambda *a:dict(manifest_sha256='CPU_ONLY')):
                c=M.Controller(r);c.device=lambda *a,**k:dict(apps=[])
                value=c.run_unit('scale_repair_gates','CPU_FIXTURE',0)
                self.assertEqual(c.intervals[0]['returncode'],returncode);self.assertTrue(c.intervals[0]['released'])
                self.assertFalse(C.same_process(c.intervals[0]['identity']))
                return value,c.completed,c.first
    def test_real_positive_result_with_exit1_rejected(self):
        ok,completed,failure=self.execute_fixture(1);self.assertFalse(ok);self.assertFalse(completed);self.assertIsNotNone(failure)
    def test_real_exit0_release_accepted(self):
        ok,completed,failure=self.execute_fixture(0);self.assertTrue(ok);self.assertEqual(len(completed),1);self.assertIsNone(failure)

if __name__=='__main__':unittest.main(verbosity=2)
