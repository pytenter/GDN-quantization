"""Synthetic assignment regression tests; no model or CUDA loaded."""
import importlib.util, os, sys, types, unittest
from pathlib import Path
from unittest.mock import patch
sp=importlib.util.spec_from_file_location('qualification',Path(__file__).with_name('gpu3_expansion_qualification.py'))
m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)

class Tests(unittest.TestCase):
    def check(self,gpu='3',uuid=m.UUID,occupant=None,flags=None):
        flags=flags or dict(deterministic_algorithms=False,cudnn_deterministic=False,cudnn_benchmark=False,matmul_tf32=False)
        torch=types.SimpleNamespace(__version__='2.7.1+cu128',version=types.SimpleNamespace(cuda='12.8'),
            cuda=types.SimpleNamespace(device_count=lambda:1),are_deterministic_algorithms_enabled=lambda:flags['deterministic_algorithms'],
            backends=types.SimpleNamespace(cudnn=types.SimpleNamespace(deterministic=flags['cudnn_deterministic'],benchmark=flags['cudnn_benchmark']),
                cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=flags['matmul_tf32']))))
        old=dict(python=sys.version,torch='2.7.1+cu128',CUDA='12.8',deterministic_algorithms=False,cudnn_deterministic=False,cudnn_benchmark=False,matmul_tf32=False)
        def read(p):
            return dict(hardware=old) if p.name=='original_historical_reference.json' else dict(environment={'CUDA_VISIBLE_DEVICES':'3'})
        def cmd(args,**kwargs):
            if args[0]=='hostname':return 'nlpg-SYS-4029GP-TRT\n'
            if '--query-gpu=index,uuid,name,driver_version' in args:return f'3, {uuid}, NVIDIA GeForce RTX 3090, 580.173.02\n'
            return '' if occupant is None else f'{m.UUID}, {occupant}\n'
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':gpu}),patch.object(m,'read',read),patch.object(m.subprocess,'check_output',cmd):
            return m.hardware(torch)
    def test_correct_assignment(self):self.assertEqual(self.check()['physical_gpu'],3)
    def test_wrong_visible_gpu(self):
        with self.assertRaises(AssertionError):self.check(gpu='6')
    def test_wrong_uuid(self):
        with self.assertRaises(AssertionError):self.check(uuid='GPU-wrong')
    def test_external_occupant(self):
        with self.assertRaises(AssertionError):self.check(occupant=os.getpid()+1000)
    def test_own_context_allowed(self):self.check(occupant=os.getpid())
    def test_determinism_flag_change_rejected(self):
        with self.assertRaises(AssertionError):self.check(flags=dict(deterministic_algorithms=True,cudnn_deterministic=False,cudnn_benchmark=False,matmul_tf32=False))
    def test_original_numeric_functions_not_replaced(self):
        source=Path(m.__file__).read_text()
        for name in ['replay','pulse_target','target_step','k3','parity','smoke']:
            self.assertNotIn('b.'+name+'=',source)
    def test_gpu6_not_launched_or_signalled(self):
        source=Path(__file__).with_name('gpu3_expansion_supervisor.py').read_text()
        for token in ['os.kill','killpg','b.discovery','historical_backend_discovery_runner']:
            self.assertNotIn(token,source)

if __name__=='__main__':unittest.main()
