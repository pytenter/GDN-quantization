"""CPU-only terminal counting and real one-parent/two-owning-Popen tests."""
import copy,json,os,shlex,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest import mock
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
import pace_control as C
import finalize_four_gpu_cpu as F
import offline_four_gpu_parent as P

class FinishTests(unittest.TestCase):
    def fixture(self,r):
        (r/'source_manifest.json').write_text('{}');(r/'GPU_PERMIT.json').write_text('{}')
        jobs=[dict(id='q%02d_%s'%(i,c),condition=c) for i in range(30) for c in ('R2','R3')];reuse=[j['id'] for j in jobs[:15]];plan=[dict(unit=j['id']) for j in jobs[15:]]
        (r/'REUSE_ADMISSION.json').write_text(json.dumps(dict(reused_ids=reuse)));(r/'FOUR_GPU_PLAN.json').write_text(json.dumps(dict(jobs=plan)))
        pairs=[(j,dict(source_manifest_sha256='OLD' if i<15 else C.sha(r/'source_manifest.json'),method_variant=F.METHOD_VARIANT,state_scale_format=F.STATE_SCALE_FORMAT),r) for i,j in enumerate(jobs)]
        intervals=[dict(stage='scale_repair_gates',unit='SCALE_REPAIR_GATES_GPU'+str(g),gpu=g,returncode=0,released=True) for g in range(4)]+[dict(stage='formal',unit=j['id'],returncode=0,released=True) for j in jobs[15:]]
        (r/'RESOURCE_LEDGER.json').write_text(json.dumps(dict(intervals=intervals)))
        return pairs,intervals
    def test_full60_exact_union_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d).resolve();pairs,_=self.fixture(r);full,a,count=F.full_completion(r,dict(status='GPU_COMPLETE_FOURGPU_REUSE_PLUS_REMAINING60'),pairs,[])
            self.assertTrue(full);self.assertEqual((a['reused_formal_completed'],a['current_formal_completed']),(15,45));self.assertEqual(dict(count),dict(R2=30,R3=30))
    def test_partial_exit_release_duplicate_or_missing_gate_never_full(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d).resolve();pairs,intervals=self.fixture(r);session=dict(status='GPU_COMPLETE_FOURGPU_REUSE_PLUS_REMAINING60')
            self.assertFalse(F.full_completion(r,session,pairs[:-1],[])[0]);self.assertFalse(F.full_completion(r,dict(status='FAIL_CLOSED_STOP'),pairs,[])[0]);self.assertFalse(F.full_completion(r,session,pairs,[dict(error='invalid')])[0])
            for mutate in ('nonzero','unreleased','duplicate','missing_gate'):
                v=copy.deepcopy(intervals)
                if mutate=='nonzero':v[-1]['returncode']=1
                elif mutate=='unreleased':v[-1]['released']=False
                elif mutate=='duplicate':v[-1]['unit']=v[-2]['unit']
                else:v=v[1:]
                (r/'RESOURCE_LEDGER.json').write_text(json.dumps(dict(intervals=v)));self.assertFalse(F.full_completion(r,session,pairs,[])[0],mutate)
            with self.assertRaises(RuntimeError):F.completion_accounting(r,pairs+[pairs[0]])
    def test_partial_seals_without_scoring(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d).resolve();pairs,_=self.fixture(r);(r/'SESSION_RESULT.json').write_text(json.dumps(dict(status='FAIL_CLOSED_STOP',failure=dict(reason='CPU fixture'))))
            with mock.patch.object(C,'process_identity',lambda pid:dict(pid=pid,starttick=0,cmdline=['CPU_MOCK'])),mock.patch.object(C,'REMOTE',str(r)),mock.patch.object(F,'terminal_guard',lambda r:dict(status='CPU_MOCK_RELEASE')),mock.patch.object(F,'complete_rows',lambda r,invalid:pairs[:15]),mock.patch.object(F,'archive',lambda r:dict(status='CPU_MOCK_NO_NETWORK')),mock.patch.object(F,'run_score_child',side_effect=AssertionError('Partial scoring forbidden')):
                self.assertEqual(F.finalize(r),0)
            v=C.read(r/'final_verdict.json');self.assertEqual(v['formal_completed'],15);self.assertFalse(v['full_accuracy_reported']);self.assertFalse((r/'comparison').exists())

@unittest.skipUnless(sys.platform.startswith('linux'),'Actual /proc process identity requires target55')
class RealParentTests(unittest.TestCase):
    def test_genuine_Bash_wait_preserves_zero_nonzero_and_signal_integer(self):
        for rc in (0,7,143):
            with self.subTest(rc=rc),tempfile.TemporaryDirectory(prefix='PACE4_wait_CPU_') as d:
                r=Path(d).resolve();(r/'scripts').mkdir();(r/'source_manifest.json').write_text('{}');(r/'GPU_PERMIT.json').write_text('{}')
                ending='import os,signal;os.kill(os.getpid(),signal.SIGTERM)' if rc==143 else 'raise SystemExit('+str(rc)+')'
                (r/'scripts/offline_four_gpu_parent.py').write_text('import time;time.sleep(1)\n'+ending+'\n')
                code="import sys;sys.path.insert(0,"+repr(str(ROOT/'scripts'))+");import record_four_gpu_wait as W;W.C.REMOTE="+repr(str(r))+";W.C.authorize=lambda r:None;W.main()"
                recorder=shlex.join([sys.executable,'-B','-c',code,'--root',str(r)])
                child=shlex.join([sys.executable,str(r/'scripts/offline_four_gpu_parent.py'),'--root',str(r),'--execute'])
                script=recorder+' --mode start --bash-pid $$\n'+child+' &\nfixture_child=$!\n'+recorder+' --mode child --bash-pid $$ --child-pid "$fixture_child"\nwait "$fixture_child"\nfixture_rc=$?\n'+recorder+' --mode exit --bash-pid $$ --returncode "$fixture_rc"\nexit "$fixture_rc"\n'
                process=subprocess.run(['/bin/bash','-c',script],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,timeout=10)
                self.assertEqual(process.returncode,rc,process.stderr.decode());v=C.read(r/'OFFLINE_CHAIN_OBSERVED_EXIT.json');self.assertEqual(v['actual_shell_wait_returncode'],rc);self.assertFalse(v['python_negative_signal_conversion']);self.assertFalse(C.same_process(v['parent']))

    def test_controller_exit1_then_one_cpu_finisher_exit0_no_restart(self):
        with tempfile.TemporaryDirectory(prefix='PACE4_parent_CPU_') as d:
            r=Path(d).resolve();(r/'scripts').mkdir();(r/'source_manifest.json').write_text('{}');(r/'GPU_PERMIT.json').write_text('{}')
            for name,rc in [('pace_four_gpu_controller.py',1),('finalize_four_gpu_cpu.py',0)]:
                (r/'scripts'/name).write_text('import time;time.sleep(.15);raise SystemExit('+str(rc)+')\n')
            with mock.patch.object(C,'authorize',lambda r:None):
                parent=P.Parent(r);self.assertEqual(parent.run(),0)
                with self.assertRaises((FileExistsError,RuntimeError)):P.Parent(r).run()
            for name,rc in [('CONTROLLER',1),('FINISHER',0)]:
                v=C.read(r/('PARENT_'+name+'_EXIT.json'));self.assertEqual(v['returncode'],rc);self.assertTrue(v['actual_exited'] and v['identity_gone']);self.assertFalse(C.same_process(v['identity']))
            self.assertEqual(len(list(r.glob('PARENT_*_DISPATCH.json'))),2)

if __name__=='__main__':unittest.main(verbosity=2)
