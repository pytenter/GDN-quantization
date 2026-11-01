"""Pure control-boundary tests. No model, CUDA, signal, or measurement."""
from pathlib import Path
import importlib.util,tempfile,unittest

spec=importlib.util.spec_from_file_location('controller',Path(__file__).with_name('gpu3_controlled_handoff_v2.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class Tests(unittest.TestCase):
    def test_expected_guard_exit(self):
        m.assert_expected_guard_exit('Traceback (most recent call last):\nAssertionError: PRESERVED_FORMAL_STOP\n')
    def test_other_exception_blocks(self):
        with self.assertRaises(AssertionError):m.assert_expected_guard_exit('Traceback (most recent call last):\nRuntimeError: CUDA failure\n')
    def test_multiple_exceptions_block(self):
        with self.assertRaises(AssertionError):m.assert_expected_guard_exit('Traceback (most recent call last):\nTraceback (most recent call last):\nAssertionError: PRESERVED_FORMAL_STOP\n')
    def test_nonfinite_blocks(self):
        with self.assertRaises(AssertionError):m.finite({'value':float('nan')})
    def test_finite_nested_passes(self):m.finite({'value':[1.0,2.0]})
    def test_fixed_original_pids(self):self.assertEqual(m.ORIGINAL_PIDS,[2726684,2726685])
    def test_fixed_gpu_identity(self):
        self.assertEqual(set(m.UUIDS),{3,6})
    def test_preserved_attempt01_stop_is_nonblocking(self):
        self.assertEqual(m.classify_engineering_stop('ENGINEERING_STOP.json',m.PRESERVED_ATTEMPT01_STOP_SHA256),'PRESERVED_ATTEMPT01_NONBLOCKING')
    def test_changed_attempt01_stop_blocks(self):
        self.assertEqual(m.classify_engineering_stop('ENGINEERING_STOP.json','0'*64),'BLOCKING')
    def test_new_attempt_stop_blocks(self):
        self.assertEqual(m.classify_engineering_stop('ENGINEERING_STOP_ATTEMPT02.json',m.PRESERVED_ATTEMPT01_STOP_SHA256),'BLOCKING')
    def test_no_process_signals_or_scientific_reimplementation(self):
        source=Path(m.__file__).read_text()
        for forbidden in ('os.kill(', 'killpg(', 'SIGTERM', 'SIGSTOP', 'def replay(', 'def pulse_target(', 'def target_step(', 'def k3('):
            self.assertNotIn(forbidden,source)

if __name__=='__main__':unittest.main()
