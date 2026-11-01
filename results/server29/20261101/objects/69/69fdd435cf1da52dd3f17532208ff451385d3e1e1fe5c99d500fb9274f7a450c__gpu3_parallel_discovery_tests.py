"""Pure scheduling/provenance tests. No torch, CUDA or scientific measurements."""
import importlib.util, tempfile, unittest
from pathlib import Path
sp=importlib.util.spec_from_file_location('parallel',Path(__file__).with_name('gpu3_parallel_discovery_v1.py'))
m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)

class Tests(unittest.TestCase):
    def test_doc_grouping(self):
        self.assertEqual(m.check_assignment(3,m.ASSIGNMENT),['pg19-test-5396','pg19-test-26493'])
        self.assertEqual(m.check_assignment(6,m.ASSIGNMENT),['pg19-test-37702','pg19-test-28988'])
    def test_assignment_change_blocks(self):
        changed=dict(m.ASSIGNMENT);changed['pg19-test-5396']=6
        with self.assertRaises(AssertionError):m.check_assignment(3,changed)
    def test_wrong_gpu_blocks(self):
        with self.assertRaises(AssertionError):m.check_assignment(5,m.ASSIGNMENT)
    def test_no_partial_pair_rerun(self):
        with tempfile.TemporaryDirectory() as root:
            a,b=Path(root)/'a',Path(root)/'b'
            self.assertFalse(m.existing_pair([a,b]));a.touch()
            with self.assertRaises(AssertionError):m.existing_pair([a,b])
            b.touch();self.assertTrue(m.existing_pair([a,b]))
    def unit(self):
        return dict(gate='PASS',task='LING_LONG_HORIZON_OBSERVABILITY_BRIDGE_V1',document='pg19-test-5396',condition='H',anchor=8192,endpoint=32768,
            hardware=dict(physical_gpu=3,gpu_uuid=m.UUIDS[3]),premeasurement_commit='frozen',token_sha256='tokens',rotation_sha256='rot',
            scores=dict(E_anchor_sha256='e'),target=dict(E_anchor_sha256='e',teacher_tokens=24576,INITIAL_PULSE_IDENTITY_GATE='PASS',FP_REFERENCE_PAIRING_GATE='PASS',NO_FURTHER_QDQ_GATE='PASS'))
    def check(self,u):m.validate_unit(u,3,'pg19-test-5396','H',8192,'tokens','rot',['frozen'])
    def test_valid_unit(self):self.check(self.unit())
    def test_cross_gpu_reference_not_masqueraded(self):
        u=self.unit();u['hardware']=dict(physical_gpu=6,gpu_uuid=m.UUIDS[6])
        with self.assertRaises(AssertionError):self.check(u)
    def test_unknown_commit_blocks(self):
        u=self.unit();u['premeasurement_commit']='unknown'
        with self.assertRaises(AssertionError):self.check(u)
    def test_wrong_anchor_or_document_blocks(self):
        for key,value in [('anchor',2048),('document','pg19-test-37702')]:
            u=self.unit();u[key]=value
            with self.assertRaises(AssertionError):self.check(u)
    def test_pulse_mismatch_blocks(self):
        u=self.unit();u['scores']['E_anchor_sha256']='different'
        with self.assertRaises(AssertionError):self.check(u)
    def test_semantic_gate_failure_blocks(self):
        u=self.unit();u['target']['NO_FURTHER_QDQ_GATE']='FAIL'
        with self.assertRaises(AssertionError):self.check(u)
    def test_no_numerical_reimplementation(self):
        source=Path(m.__file__).read_text()
        for name in ['replay','pulse_target','target_step','k3','parity','smoke']:
            self.assertNotIn('def '+name+'(',source);self.assertNotIn('b.'+name+'=',source)
        for token in ['os.kill','killpg','b.gate()','b.discovery()','SIGTERM','SIGSTOP']:
            self.assertNotIn(token,source)

if __name__=='__main__':unittest.main()
