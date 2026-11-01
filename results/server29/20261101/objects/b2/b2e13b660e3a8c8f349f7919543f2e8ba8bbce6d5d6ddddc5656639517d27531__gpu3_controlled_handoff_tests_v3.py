"""Pure control-boundary tests. No model, CUDA, signal, or measurement."""
from pathlib import Path
import importlib.util,tempfile,unittest

spec=importlib.util.spec_from_file_location('controller',Path(__file__).with_name('gpu3_controlled_handoff_v3.py'))
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
    def test_preserved_attempt03_advance_stop_is_nonblocking(self):
        self.assertEqual(m.classify_engineering_stop('CONTROLLED_HANDOFF_advance_STOP.json',m.PRESERVED_ATTEMPT03_ADVANCE_STOP_SHA256),'PRESERVED_ATTEMPT03_ADVANCE_NONBLOCKING')
    def test_preserved_attempt03_gpu6_stop_is_nonblocking(self):
        self.assertEqual(m.classify_engineering_stop('GPU6_postcommit_STOP.json',m.PRESERVED_ATTEMPT03_GPU6_POSTCOMMIT_STOP_SHA256),'PRESERVED_ATTEMPT03_GPU6_POSTCOMMIT_NONBLOCKING')
    def test_changed_attempt03_stops_block(self):
        self.assertEqual(m.classify_engineering_stop('CONTROLLED_HANDOFF_advance_STOP.json','0'*64),'BLOCKING')
        self.assertEqual(m.classify_engineering_stop('GPU6_postcommit_STOP.json','0'*64),'BLOCKING')
    def test_new_attempt_stop_blocks(self):
        self.assertEqual(m.classify_engineering_stop('ENGINEERING_STOP_ATTEMPT02.json',m.PRESERVED_ATTEMPT01_STOP_SHA256),'BLOCKING')
    def test_no_process_signals_or_scientific_reimplementation(self):
        source=Path(m.__file__).read_text()
        for forbidden in ('os.kill(', 'killpg(', 'SIGTERM', 'SIGSTOP', 'def replay(', 'def pulse_target(', 'def target_step(', 'def k3('):
            self.assertNotIn(forbidden,source)


class RunnerGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec=importlib.util.spec_from_file_location('runner_v3',Path(__file__).with_name('gpu3_parallel_discovery_v1_guard_identity_v3.py'))
        cls.r=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.r)
    def test_expected_handoff_receipt_is_only_allowed_parent_change(self):
        from unittest.mock import patch
        r=self.r
        with patch.object(r,'git',return_value=r.PARENT), patch.object(r.subprocess,'check_output',return_value=r.EXPECTED_PARENT_TRACKED_STATUS), patch.object(r,'sha',return_value='a'*64), patch.object(r,'read',side_effect=[{'controlled_handoff_receipt_sha256':'a'*64},{'kind':'ENGINEERING_CONTROLLED_HANDOFF_REQUEST','scientific_failure':False}]):
            r.assert_parent_runtime_tree()
    def test_clean_or_extra_parent_status_blocks(self):
        from unittest.mock import patch
        r=self.r
        for status in ('',' M '+r.PARENT_STOP_REL+'\n M unexpected.py\n'):
            with self.subTest(status=status), patch.object(r,'git',return_value=r.PARENT), patch.object(r.subprocess,'check_output',return_value=status):
                with self.assertRaises(AssertionError):r.assert_parent_runtime_tree()
    def test_changed_handoff_receipt_blocks(self):
        from unittest.mock import patch
        r=self.r
        with patch.object(r,'git',return_value=r.PARENT), patch.object(r.subprocess,'check_output',return_value=r.EXPECTED_PARENT_TRACKED_STATUS), patch.object(r,'sha',return_value='b'*64), patch.object(r,'read',return_value={'controlled_handoff_receipt_sha256':'a'*64}):
            with self.assertRaises(AssertionError):r.assert_parent_runtime_tree()
    def test_runner_delta_is_guard_only(self):
        old=(self.r.P/'inputs'/'gpu3_parallel_discovery_v1_guard_identity_v2.py').read_text()
        new=Path(self.r.__file__).read_text()
        marker="CONFIGS = ['source_identity.json','document_manifest.json','rotation_manifest.json','runtime_manifest.json','preregistration.json']\n"
        block=new[new.index("PARENT_STOP_REL = "):new.index("\ndef read(")]
        normalized=new.replace(block,'')
        normalized=normalized.replace('    assert_parent_runtime_tree()\n',"    assert git('rev-parse','HEAD',cwd=R)==PARENT and not git('status','--porcelain=v1','--untracked-files=no',cwd=R)\n",1)
        normalized=normalized.replace('        assert_parent_runtime_tree()\n',"        assert git('rev-parse','HEAD',cwd=R)==PARENT\n",1)
        self.assertEqual(normalized,old)
        for forbidden in ('def replay(', 'def pulse_target(', 'def target_step(', 'def k3(', 'os.kill(', 'killpg(', 'SIGTERM', 'SIGSTOP'):
            self.assertNotIn(forbidden,new)

if __name__=='__main__':unittest.main()
