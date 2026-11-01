"""Real CPU Popen checks for the controller exit/identity race, no Torch/CUDA."""
import importlib.util,os,subprocess,sys,time,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from owned_exit_guard import owned_child_running
FROZEN=Path('/data/zypan/analyses/qwen_gdn_pace48_aime26_seed1_60_v1_4gpu_continue_control_v2_20261026/src/inherited_device_guards.py')
s=importlib.util.spec_from_file_location('frozen_process_identity',FROZEN);C=importlib.util.module_from_spec(s);s.loader.exec_module(C)

class ExitGuardTests(unittest.TestCase):
    def child(self,rc=0,sleep=0):
        p=subprocess.Popen([sys.executable,'-B','-c',f'import sys,time;sys.stdin.buffer.read(1);time.sleep({sleep});raise SystemExit({rc})'],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        identity=C.process_identity(p.pid);self.addCleanup(self.cleanup,p);return p,identity
    @staticmethod
    def cleanup(p):
        if p.poll() is None:p.terminate();p.wait(timeout=5)
        if p.stdin:p.stdin.close()
    @staticmethod
    def finish_after_initial_poll(p,rc):
        def reader(_):
            p.stdin.write(b'x');p.stdin.flush()
            until=time.monotonic()+5
            # The child has exited, but this reader never reaps it. On Linux
            # the zombie has an empty argv: exactly the original guard boundary.
            while time.monotonic()<until:
                stat=Path('/proc')/str(p.pid)/'stat'
                if not stat.exists() or stat.read_text().rsplit(')',1)[1].split()[0]=='Z':break
                time.sleep(.001)
            return C.same_process(_)
        return reader
    def test_original_guard_deterministically_false_stops_real_exit0(self):
        p,i=self.child(0);self.assertIsNone(p.poll());reader=self.finish_after_initial_poll(p,0)
        with self.assertRaises(RuntimeError):C.require(reader(i),'Owned child PID/starttick/argv changed; never signal replacement')
        self.assertEqual(p.wait(timeout=5),0)
    def test_candidate_accepts_real_exit0_between_poll_and_identity(self):
        p,i=self.child(0);self.assertFalse(owned_child_running(p,i,self.finish_after_initial_poll(p,0)));self.assertEqual(p.returncode,0)
    def test_real_exit1_is_preserved_not_reclassified_success(self):
        p,i=self.child(1);self.assertFalse(owned_child_running(p,i,self.finish_after_initial_poll(p,1)));self.assertEqual(p.returncode,1)
    def test_live_exact_owned_identity_stays_running(self):
        p,i=self.child();self.assertTrue(owned_child_running(p,i,C.same_process));self.assertIsNone(p.poll())
    def test_live_wrong_starttick_fails_without_signal(self):
        p,i=self.child();bad=dict(i,starttick=i['starttick']+1)
        with self.assertRaises(RuntimeError):owned_child_running(p,bad,C.same_process)
        self.assertIsNone(p.poll());self.assertTrue(C.same_process(i))
    def test_live_same_tick_changed_argv_fails_without_signal(self):
        p,i=self.child();bad=dict(i,cmdline=['FOREIGN'])
        with self.assertRaises(RuntimeError):owned_child_running(p,bad,C.same_process)
        self.assertIsNone(p.poll());self.assertTrue(C.same_process(i))
    def test_delayed_own_exit_is_reaped_within_exit_classification_grace(self):
        p,i=self.child(0,sleep=.02)
        def transitioning(_):p.stdin.write(b'x');p.stdin.flush();return False
        self.assertFalse(owned_child_running(p,i,transitioning));self.assertEqual(p.returncode,0)
    def test_already_reaped_never_reads_reused_PID(self):
        p,i=self.child(0);p.stdin.write(b'x');p.stdin.flush();p.wait(timeout=5)
        def forbidden(_):raise AssertionError('An already reaped child must not be treated as a new PID owner')
        self.assertFalse(owned_child_running(p,i,forbidden));self.assertEqual(p.returncode,0)

if __name__=='__main__':unittest.main(verbosity=2)
