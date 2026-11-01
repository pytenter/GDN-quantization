"""Same-task prior accounting tests; CPU only, no telemetry/API/model execution."""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import runner as R

W=Path(__file__).resolve().parents[1]
D=Path('/data/zypan/experiment_runs/ling-kda-dual-axis-resume-01/LING_KDA_ROTATION_DUAL_AXIS_FUNCTIONAL_V1')


def budget():
    return dict(max_gpu_hours=16.,protect_gpu_hours=15.5,formal_safety_factor=1.25,
        cleanup_reserve_gpu_hours=.25,prior_this_task_GPUh=R.PRIOR_THIS_TASK_GPU_HOURS,
        prior_binding=dict(R.PRIOR_BINDING))


def job(condition='FP_NATIVE',seconds=360.,kind='qual_long'):
    return dict(kind=kind,condition=condition,charged_seconds=seconds,qualified_completion=True,
        tag=condition+'.'+kind,launch_to_wait_seconds=max(0,seconds-10),terminal_gate_seconds=min(10,seconds))


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='ling_dualm_continuation_CPU_')
        self.r=R.Runner(Path(self.tmp.name),W);self.r.prereg={'budget':budget()}
    def tearDown(self):self.tmp.cleanup()

    def test_actual_prior_sealed_binding_readonly_verified(self):
        value=self.r.verify_prior_budget(budget())
        self.assertEqual(value['status'],'PASS_PINNED_PRIOR_SAME_TASK_BUDGET')
        self.assertEqual(self.r.used_hours(),.47844357310752256)

    def test_current_and_prior_separate_no_double_count(self):
        self.r.prior_gpu_hours=R.PRIOR_THIS_TASK_GPU_HOURS;self.r.jobs=[job(seconds=360.)]
        self.assertEqual(self.r.current_hours(),.1)
        self.assertEqual(self.r.used_hours(),R.PRIOR_THIS_TASK_GPU_HOURS+.1)

    def test_active_launch_time_included_once(self):
        self.r.prior_gpu_hours=R.PRIOR_THIS_TASK_GPU_HOURS;self.r.jobs=[job(seconds=360.)]
        self.r.active=dict(launch_monotonic=100.)
        self.assertEqual(self.r.current_hours(460.),.2)
        self.assertEqual(self.r.used_hours(460.),R.PRIOR_THIS_TASK_GPU_HOURS+.2)
        self.r.active=None

    def test_ledger_explicit_prior_current_cumulative(self):
        self.r.prior_gpu_hours=R.PRIOR_THIS_TASK_GPU_HOURS;self.r.jobs=[job(seconds=360.)]
        value=self.r.ledger()
        self.assertEqual(value['prior_task_gpu_hours'],R.PRIOR_THIS_TASK_GPU_HOURS)
        self.assertEqual(value['current_execution_GPU_hours'],.1)
        self.assertEqual(value['cumulative_same_task_GPU_hours'],R.PRIOR_THIS_TASK_GPU_HOURS+.1)
        self.assertEqual(value['charged_GPU_hours'],value['GPU_HOURS_USED'])
        self.assertEqual(value['charged_GPU_hours'],value['cumulative_same_task_GPU_hours'])
        self.assertEqual(len(value['jobs']),1);self.assertEqual(value['completed_workers'],1)

    def test_zero_default_only_for_pure_CPU_fixture(self):
        self.r.jobs=[job(seconds=360.)]
        self.assertEqual(self.r.prior_gpu_hours,0.);self.assertEqual(self.r.used_hours(),.1)
        self.r.verify_prior_budget(budget());self.assertGreater(self.r.used_hours(),.1)

    def test_protective_15_5_is_cumulative(self):
        self.r.prior_gpu_hours=R.PRIOR_THIS_TASK_GPU_HOURS
        self.r.jobs=[job(seconds=(15.5-R.PRIOR_THIS_TASK_GPU_HOURS)*3600.+.001)]
        self.assertLess(self.r.current_hours(),15.5)
        with self.assertRaises(R.TaskStopped):self.r.budget_guard()
        self.assertEqual(R.read(self.r.d/'FIRST_FAILURE.json')['reason'],'GPU_BUDGET_PROTECTIVE_STOP')

    def test_absolute_16_cannot_reset_on_new_execution(self):
        self.r.prior_gpu_hours=R.PRIOR_THIS_TASK_GPU_HOURS
        self.r.jobs=[job(seconds=(16.-R.PRIOR_THIS_TASK_GPU_HOURS)*3600.)]
        with self.assertRaises(R.TaskStopped):self.r.budget_guard()
        self.assertGreaterEqual(self.r.used_hours(),16.)

    def test_forecast_prior_once_with_unchanged30_and_margins(self):
        self.r.prior_gpu_hours=R.PRIOR_THIS_TASK_GPU_HOURS
        self.r.jobs=[job(c) for c in R.CONDITIONS]
        value=self.r.forecast()
        self.assertEqual(value['formal_trajectories'],30)
        self.assertEqual(value['formal_safety_factor'],1.25)
        self.assertEqual(value['cleanup_exit_reserve_GPUh'],.25)
        self.assertAlmostEqual(value['projected_total_GPUh'],R.PRIOR_THIS_TASK_GPU_HOURS+.5+3.75+.25)
        self.assertAlmostEqual(value['current_execution_qualification_GPUh'],.5)

    def test_forecast_prevents_false_pass_from_prior_omission(self):
        self.r.prior_gpu_hours=R.PRIOR_THIS_TASK_GPU_HOURS
        # Without prior the complete prediction would be15.2 hours and PASS;
        # the unchanged cumulative15.5 policy must STOP this case.
        long_total=(15.2-.25)/(1.+6*1.25)
        self.r.jobs=[job(c,long_total*3600./5.) for c in R.CONDITIONS]
        with self.assertRaises(R.TaskStopped):self.r.forecast()
        value=R.read(self.r.d/'BUDGET_FORECAST.json')
        self.assertGreater(value['projected_total_GPUh'],15.5)
        self.assertEqual(value['status'],'BUDGET_FEASIBILITY_STOP')

    def test_real_preflight_prior_zero_forbidden(self):
        b=budget();b['prior_this_task_GPUh']=0.
        with self.assertRaises(AssertionError):self.r.verify_prior_budget(b)

    def test_real_preflight_missing_prior_binding_forbidden(self):
        b=budget();del b['prior_binding']
        with self.assertRaises(KeyError):self.r.verify_prior_budget(b)

    def test_real_preflight_changed_ledger_binding_forbidden(self):
        b=budget();b['prior_binding']['ledger_sha256']='0'*64
        with self.assertRaises(AssertionError):self.r.verify_prior_budget(b)

    def test_real_preflight_changed_seal_binding_forbidden(self):
        b=budget();b['prior_binding']['seal_receipt_sha256']='0'*64
        with self.assertRaises(AssertionError):self.r.verify_prior_budget(b)

    def test_invalid_prior_values_rejected(self):
        for value in [-1.,float('nan'),float('inf')]:
            with self.subTest(value=value):
                self.r.prior_gpu_hours=value
                with self.assertRaises(AssertionError):self.r.used_hours()

    def test_worker_analysis_and_math_sources_byte_unchanged(self):
        old=Path('/data/zypan/worktrees/ling-kda-rotation-dual-axis-functional-v1')
        for name in ['dual_axis_task/model_worker.py','dual_axis_task/analysis.py','dual_axis_task/codec_adapter.py',
                     'dual_axis_task/cache_tools.py','gpu_admission.py','lifecycle_guard.py','gpu_telemetry.py']:
            with self.subTest(name=name):self.assertEqual(R.sha(W/name),R.sha(old/name))


if __name__=='__main__':
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and 'torch' not in sys.modules
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    inherited=D/'audits/CPU_CONTINUATION_INHERITED24.json'
    proc=subprocess.run([sys.executable,'-B',str(W/'dual_axis_task/test_runner_cpu.py'),'--output',str(inherited)],
        env=os.environ.copy(),capture_output=True,text=True)
    print(proc.stdout);print(proc.stderr)
    inherited_ok=proc.returncode==0 and R.read(inherited)['status']=='PASS_CPU_CONTROL_TESTS'
    report=dict(task_id=R.TASK,status='PASS_CPU_CONTINUATION_BUDGET_TESTS' if result.wasSuccessful() and inherited_ok else 'FAIL_CPU_CONTINUATION_BUDGET_TESTS',
        tests_run=result.testsRun,inherited_control_tests=R.read(inherited)['tests_run'] if inherited.exists() else None,
        inherited_control_status='PASS' if inherited_ok else 'FAIL',inherited_control_sha256=R.sha(inherited) if inherited.exists() else None,
        failures=[dict(test=str(a),traceback=b) for a,b in result.failures],errors=[dict(test=str(a),traceback=b) for a,b in result.errors],
        CPU_ONLY=True,GPU_API_queries=0,GPU_model_workers=0,torch_imported='torch' in sys.modules,
        prior_same_task_GPUh=R.PRIOR_THIS_TASK_GPU_HOURS,prior_binding=R.PRIOR_BINDING,
        max_same_task_GPUh=16.,protect_same_task_GPUh=15.5,no_old_PASS_or_checkpoint_inherited=True,
        source_sha256={p.name:R.sha(p) for p in [Path(__file__).resolve(),Path(R.__file__).resolve()]})
    out=D/'audits/CPU_CONTINUATION_BUDGET_TESTS.json';out.parent.mkdir(exist_ok=True)
    with out.open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(status=report['status'],tests_run=result.testsRun,path=str(out),sha256=R.sha(out))))
    raise SystemExit(0 if report['status'].startswith('PASS') else 1)
