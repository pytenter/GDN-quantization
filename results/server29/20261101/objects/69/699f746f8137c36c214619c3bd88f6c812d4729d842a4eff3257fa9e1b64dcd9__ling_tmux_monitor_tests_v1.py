import copy
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('monitor',Path(__file__).with_name('ling_tmux_monitor_v1.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.unit={'gate':'PASS','task':'LING_LONG_HORIZON_OBSERVABILITY_BRIDGE_V1',
         'document':'pg19-test-37702','hardware':dict(physical_gpu=6,gpu_uuid=m.UUIDS[6]),
         'anchor':8192,'endpoint':32768,'premeasurement_commit':m.PARENT,
         'token_sha256':'token','rotation_sha256':'rotation',
         'scores':{'E_anchor_sha256':'error'},'target':{'E_anchor_sha256':'error',
          'FP_REFERENCE_PAIRING_GATE':'PASS','INITIAL_PULSE_IDENTITY_GATE':'PASS',
          'NO_FURTHER_QDQ_GATE':'PASS','teacher_tokens':24576}}
    def check(self,u):m.validate(u,{'token_ids_sha256':'token'},{'artifact':{'sha256':'rotation'}},{m.PARENT})
    def test_valid(self):self.check(self.unit)
    def test_wrong_hardware(self):
        self.unit['hardware']['physical_gpu']=3
        with self.assertRaises(AssertionError):self.check(self.unit)
    def test_wrong_token(self):
        self.unit['token_sha256']='changed'
        with self.assertRaises(AssertionError):self.check(self.unit)
    def test_wrong_anchor_pairing(self):
        self.unit['target']['E_anchor_sha256']='changed'
        with self.assertRaises(AssertionError):self.check(self.unit)
    def test_nonfinite(self):
        self.unit['scores']['value']=float('nan')
        with self.assertRaises(AssertionError):self.check(self.unit)
    def test_no_execution_control(self):
        source=Path(m.__file__).read_text()
        for forbidden in ('os.kill(', 'signal.SIG', 'Popen(', 'tmux new-session', 'write_text('):
            self.assertNotIn(forbidden,source)
    def test_ledger_not_process_count(self):
        self.assertIn("'cleanup_ledger_bookkeeping_remaining':1",Path(m.__file__).read_text())

if __name__=='__main__':unittest.main()
