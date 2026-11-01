"""Real Linux owning exits plus adversarial identity observations, zero GPU use."""
import os,signal,subprocess,sys,time,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from owning_lifecycle import OwningLifecycle,OwnershipViolation,ExitReleaseTimeout,read_snapshot
import pace_control as C

class Fake:
    pid=12345
    def __init__(self):self.returncode=None
    def poll(self):return self.returncode

class IdentityCases(unittest.TestCase):
    def fixture(self):
        self.p=Fake();self.expected=dict(pid=self.p.pid,starttick=5,cmdline=['fixed','worker'])
        self.snapshot=dict(self.expected,ppid=os.getpid(),state='R',stable=True,cmdline_empty=False)
        self.now=1.;self.events=[]
        return OwningLifecycle(self.p,self.expected,dict(pid=os.getpid()),self.events.append,reader=lambda _:dict(self.snapshot),clock=lambda:self.now)
    def test_exact_live_has_no_totalwall_cap(self):
        l=self.fixture();self.now=100000.;self.assertEqual(l.observe()['phase'],'RUNNING');self.assertTrue(l.can_signal())
    def test_live_wrong_argv_rejected_without_signal_authority(self):
        l=self.fixture();self.snapshot['cmdline']=['foreign']
        with self.assertRaises(OwnershipViolation):l.observe()
        self.assertFalse(l.can_signal());self.p.returncode=0;self.assertEqual(l.closed(True)['status'],'FAIL_CLOSED_OWNING_LIFECYCLE')
    def test_wrong_starttick_ppid_unstable_snapshot_rejected(self):
        for key,value in [('starttick',6),('ppid',os.getpid()+1),('stable',False)]:
            l=self.fixture();self.snapshot[key]=value
            with self.assertRaises(OwnershipViolation):l.observe()
            self.assertFalse(l.can_signal())
    def test_empty_cmdline_is_pending_and_never_signal_target(self):
        l=self.fixture();self.snapshot.update(cmdline=[],cmdline_empty=True)
        self.assertEqual(l.observe()['phase'],'EXIT_PENDING');self.assertFalse(l.can_signal())
    def test_leader_Z_with_live_other_threads_is_pending(self):
        l=self.fixture();self.snapshot['state']='Z';self.assertEqual(l.observe()['phase'],'EXIT_PENDING');self.now=1.8;self.p.returncode=0
        self.assertEqual(l.observe()['phase'],'EXITED');self.assertEqual(l.closed(True)['status'],'PASS_OWNING_LIFECYCLE_EXIT0_RELEASED')
    def test_missing_proc_does_not_infer_exit_or_PID_reuse(self):
        l=self.fixture();l.reader=lambda _:(_ for _ in ()).throw(FileNotFoundError('vanished'))
        self.assertEqual(l.observe()['phase'],'EXIT_PENDING');self.assertIsNone(self.p.returncode);self.assertFalse(l.can_signal())
        l.reader=lambda _:dict(self.snapshot);self.assertEqual(l.observe()['phase'],'RUNNING')
    def test_only120s_own_exit_phase_deadline_not_extra120_after_exit(self):
        l=self.fixture();self.snapshot['state']='Z';l.observe();self.now=100.;self.p.returncode=0;l.observe();l.begin_cleanup()
        self.assertEqual(l.deadline(),121.)
    def test_pending_timeout_is_failure_never_exit_success(self):
        l=self.fixture();self.snapshot['state']='Z';l.observe();self.now=121.
        with self.assertRaises(ExitReleaseTimeout):l.observe()
        self.assertFalse(l.can_signal());self.assertFalse(l.closed(False)['actual_exited'])
    def test_actual_nonzero_and_unreleased_never_accepted(self):
        for rc,released in [(1,True),(-15,True),(0,False)]:
            l=self.fixture();self.p.returncode=rc;self.assertEqual(l.closed(released)['status'],'FAIL_CLOSED_OWNING_LIFECYCLE')
    def test_reaped_child_does_not_read_reused_pid(self):
        l=self.fixture();self.p.returncode=0;l.reader=lambda _:(_ for _ in ()).throw(AssertionError('must not inspect replacement'))
        self.assertEqual(l.observe()['returncode'],0)

@unittest.skipUnless(sys.platform.startswith('linux'),'Target55 real /proc CPU evidence')
class RealOwningChildren(unittest.TestCase):
    def child(self,rc):
        code='''import threading,time,os,sys,ctypes
ready=threading.Event()
def cleanup():
 ready.wait();time.sleep(.8);os._exit(RC)
threading.Thread(target=cleanup).start()
sys.stdin.buffer.readline();ready.set();ctypes.CDLL(None).syscall(60,0)
'''.replace('RC',str(rc))
        p=subprocess.Popen([sys.executable,'-B','-c',code],stdin=subprocess.PIPE)
        try:
            expected=C.process_identity(p.pid);events=[];life=OwningLifecycle(p,expected,C.process_identity(os.getpid()),events.append)
            p.stdin.write(b'GO\n');p.stdin.flush();p.stdin.close();phases=[]
            deadline=time.monotonic()+10
            while time.monotonic()<deadline:
                state=life.observe();phases.append(state['phase'])
                if state['phase']=='EXIT_PENDING':self.assertFalse(life.can_signal())
                if state['phase']=='EXITED':break
                time.sleep(.03)
            self.assertEqual(p.wait(timeout=5),rc);self.assertIn('EXIT_PENDING',phases)
            result=life.closed(True);self.assertEqual(result['actual_returncode'],rc)
            self.assertEqual(result['status'],'PASS_OWNING_LIFECYCLE_EXIT0_RELEASED' if rc==0 else 'FAIL_CLOSED_OWNING_LIFECYCLE')
            self.assertFalse(Path('/proc',str(p.pid)).exists());self.assertTrue(any(e['phase']=='EXIT_PENDING' and len(e['observed']['tasks'])>=2 for e in events))
        finally:
            if p.poll() is None:
                if C.same_process(expected):C.terminate_owned(p,expected)
                p.wait(timeout=5)
    def test_actual_main_thread_exit_cleanup_later_OS0(self):self.child(0)
    def test_actual_main_thread_exit_cleanup_later_OS1(self):self.child(1)
    def test_live_exec_changed_argv_never_signal_authorized(self):
        p=subprocess.Popen([sys.executable,'-B','-c',"import os,sys;sys.stdin.readline();os.execv('/bin/sleep',['/bin/sleep','2'])"],stdin=subprocess.PIPE)
        expected=C.process_identity(p.pid);l=OwningLifecycle(p,expected,C.process_identity(os.getpid()),lambda _:None)
        p.stdin.write(b'GO\n');p.stdin.flush();p.stdin.close();time.sleep(.15)
        with self.assertRaises(OwnershipViolation):l.observe()
        self.assertFalse(l.can_signal());self.assertEqual(p.wait(timeout=5),0);self.assertEqual(l.closed(True)['status'],'FAIL_CLOSED_OWNING_LIFECYCLE')

if __name__=='__main__':unittest.main(verbosity=2)
