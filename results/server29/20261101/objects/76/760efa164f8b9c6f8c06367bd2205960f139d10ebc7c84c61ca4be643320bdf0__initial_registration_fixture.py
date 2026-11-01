"""CPU children and saved/fake telemetry only; never starts model_worker or CUDA."""
import copy
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
import runner as R

W=Path(__file__).resolve().parents[1]
D=Path('/data/zypan/experiment_runs/LING_KDA_ROTATION_DUAL_AXIS_FUNCTIONAL_V1')


def snapshot(owner=None):
    total=24*2**30;used=(6*2**30 if owner is not None else 4*2**20);free=total-used
    rows=[] if owner is None else [dict(pid=owner['pid'],usedGpuMemory=used)]
    text='' if owner is None else f'{R.UUID}, {owner["pid"]}, 6144\n'
    early=[] if owner is None else [dict(uuid=R.UUID,pid=owner['pid'],used_gpu_memory='6144')]
    times={k:dict(started_monotonic=1.,completed_monotonic=2.) for k in
        ['physical','compute','fuser','direct_NVML','direct_NVML_processes','SMI_XML','process_ownership']}
    return dict(uuid=R.UUID,target=[dict(index='6',uuid=R.UUID,name='NVIDIA GeForce RTX 3090',
        total_MiB='24576',used_MiB=str(used/2**20),GPU_util='0')],
        physical=dict(returncode=0),compute=dict(returncode=0,stdout=text),SMI_XML=dict(returncode=0),
        direct_NVML=dict(status='PASS',memory=dict(total=total,free=free,used=used),
            memory_v2=dict(total=total,free=free,used=used,reserved=0),utilization=dict(gpu=0,memory=0),
            temperature=40,power_mW=50000,pstate=8,SM_clock_MHz=200),
        direct_NVML_processes=dict(compute=dict(status='PASS',processes=rows),graphics=dict(status='PASS',processes=[])),
        SMI_all_processes=dict(status='PASS',processes=[] if owner is None else [dict(pid=str(owner['pid']),type='C')]),
        process_ownership=[] if owner is None else [dict(owner,status='PASS')],
        early_SMI_compute_processes=dict(status='PASS',processes=early),query_timings=times)


class FakeResource(R.OriginalResourcePolicy):
    def __init__(self):
        super().__init__(W);self.runner=None;self.fault=None;self.exit_during_sample=False
    def sample(self,uuid):
        owner=self.runner.active_identity if self.runner is not None and self.runner.process is not None and self.runner.process.poll() is None else None
        value=snapshot(owner)
        if self.exit_during_sample and owner is not None:
            self.runner.process.wait(timeout=3.)
        if self.fault is not None:self.fault(value,owner)
        return value


class DummyRunner(R.Runner):
    def __init__(self,d,resource):
        super().__init__(d,W,resource,poll_seconds=.01,window_seconds=.002,cleanup_timeout=.05)
        resource.runner=self;self.prereg=dict(identity=dict(source_commit='TEST_ONLY_NO_GPU'),controller={},
            budget=dict(cleanup_reserve_gpu_hours=.25))
        self.source_sha='cpu_test_source';self.prereg_sha='cpu_test_prereg';self.dummy_delay=.06;self.dummy_rc=0
    def verify_source(self):return dict(status='PASS_CPU_FIXTURE_ONLY',source_manifest_sha256=self.source_sha)
    def worker_command(self,kind,document,condition,result_path,parent_path=None):
        binding=dict(preregistration_sha256=self.prereg_sha,source_manifest_sha256=self.source_sha,source_commit='TEST_ONLY_NO_GPU')
        expected={k:self.environment()[k] for k in R.ENV_KEYS}
        result=dict(status='PASS',task_id=R.TASK,kind=kind,document=document,condition=condition,UUID=R.UUID,binding=binding,
            endpoint=8448,tokens_consumed_in_this_process=8448,FP_payloads=[],integrity=None,GPU_codec_reference_parity=None)
        profile=dict(status='PASS_ACTUAL_FROZEN_PROFILE',hash_randomization=0,actual_environment=expected,source_binding=binding,
            profile=dict(status='PASS_ACTUAL_FROZEN_RUNTIME_PROFILE',forward_kernels_executed=True,environment=expected))
        script='import os,json,time,sys;from pathlib import Path;time.sleep('+repr(self.dummy_delay)+');p=Path('+repr(str(result_path))+');p.write_text('+repr(json.dumps(result))+');p.with_suffix(".PROFILE.json").write_text('+repr(json.dumps(profile))+');sys.exit('+repr(self.dummy_rc)+')'
        return [sys.executable,'-B','-c',script]


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='ling_dualm_cpu_runner_');self.d=Path(self.tmp.name)
        self.resource=FakeResource();self.runner=DummyRunner(self.d,self.resource)
    def tearDown(self):self.tmp.cleanup()
    def rejected(self,call):
        with self.assertRaises((AssertionError,R.TaskStopped)):call()

    def test_actual_cpu_child_natural_exit_terminal_and_bound_receipt(self):
        path,result=self.runner.run_job('qual_long','CPU_ONLY','FP_NATIVE')
        entry=self.runner.jobs[-1]
        self.assertEqual(entry['status'],'PASS');self.assertTrue(entry['qualified_completion'])
        self.assertEqual(entry['result_sha256'],R.sha(path));self.assertTrue(entry['terminal_gate_sha256'])
        self.assertFalse(R.exact_identity_present(entry['owner']));self.assertGreater(entry['charged_seconds'],entry['launch_to_wait_seconds'])
        self.assertEqual(R.read(self.d/'GPU_RESOURCE_LEDGER.json')['active_workers'],[])

    def test_natural_exit_during_sequential_sampling_is_transition_then_gate(self):
        self.resource.exit_during_sample=True
        self.runner.run_job('qual_long','CPU_ONLY','FP_NATIVE')
        rows=[json.loads(x) for x in next((self.d/'jobs').glob('*/LIFECYCLE.jsonl')).read_text().splitlines()]
        self.assertTrue(any(x['value']['lifecycle_state']=='NATURAL_EXIT_TRANSITION_NOT_RESOURCE_PASS' for x in rows))
        self.assertTrue(self.runner.jobs[-1]['terminal_resource_pass'])

    def test_foreign_pid_rejected_even_same_uid(self):
        owner=R.process_identity(os.getpid());s=snapshot(owner)
        foreign=owner['pid']+999999
        s['direct_NVML_processes']['compute']['processes'].append(dict(pid=foreign,usedGpuMemory=1))
        s['process_ownership'].append(dict(owner,pid=foreign,status='PASS'))
        self.rejected(lambda:self.resource.guard.strict_snapshot_safety(s,R.UUID,owner,[]))

    def test_child_tid_not_whitelisted(self):
        owner=R.process_identity(os.getpid());s=snapshot(owner)
        s['SMI_all_processes']['processes'].append(dict(pid=str(owner['pid']+1),type='C'))
        self.rejected(lambda:self.resource.guard.strict_snapshot_safety(s,R.UUID,owner,[]))

    def test_each_compute_api_foreign_rejected(self):
        owner=R.process_identity(os.getpid())
        for api in ['NVML','XML','CSV']:
            with self.subTest(api=api):
                s=snapshot(owner);foreign=99999999
                if api=='NVML':s['direct_NVML_processes']['compute']['processes'].append(dict(pid=foreign,usedGpuMemory=1))
                elif api=='XML':s['SMI_all_processes']['processes'].append(dict(pid=str(foreign),type='C'))
                else:
                    s['compute']['stdout']+=f'{R.UUID}, {foreign}, 1\n'
                    s['early_SMI_compute_processes']['processes'].append(dict(uuid=R.UUID,pid=foreign,used_gpu_memory='1'))
                self.rejected(lambda:self.resource.guard.strict_snapshot_safety(s,R.UUID,owner,[]))

    def test_unknown_api_or_missing_timing_fail_closed(self):
        owner=R.process_identity(os.getpid())
        for api in ['physical','compute','SMI_XML']:
            s=snapshot(owner);s[api]['returncode']=None
            self.rejected(lambda:self.resource.guard.strict_snapshot_safety(s,R.UUID,owner,[]))
        s=snapshot(owner);del s['query_timings']['SMI_XML'];self.rejected(lambda:self.resource.timing(s))

    def test_unknown_NVML_process_visibility_rejected(self):
        s=snapshot();s['direct_NVML_processes']['compute']['status']='UNKNOWN'
        self.rejected(lambda:self.resource.guard.strict_snapshot_safety(s,R.UUID,None,[]))

    def test_two_GiB_exact_headroom_boundary(self):
        owner=R.process_identity(os.getpid());s=snapshot(owner);m=s['direct_NVML']['memory_v2']
        m['free']=2*2**30;m['used']=m['total']-m['free']
        self.resource.guard.strict_snapshot_safety(s,R.UUID,owner,[])
        m['free']-=1;m['used']+=1
        self.rejected(lambda:self.resource.guard.strict_snapshot_safety(s,R.UUID,owner,[]))

    def test_alive_owner_missing_is_not_excused_by_result(self):
        owner=R.process_identity(os.getpid());s=snapshot()
        m=s['direct_NVML']['memory_v2'];m['used']=6*2**30;m['free']=m['total']-m['used']
        self.rejected(lambda:self.resource.running(s,R.UUID,owner,[],None,None))

    def test_nonzero_exit_transition_rejected(self):
        owner=R.process_identity(os.getpid());s=snapshot(owner)
        self.rejected(lambda:self.resource.running(s,R.UUID,owner,[],None,1))

    def test_first_failure_exclusive_and_rc0_cannot_wash(self):
        self.runner.first_failure('ORIGINAL_FIRST');digest=R.sha(self.d/'FIRST_FAILURE.json')
        self.runner.first_failure('LATER_ERROR')
        self.assertEqual(digest,R.sha(self.d/'FIRST_FAILURE.json'))
        self.rejected(lambda:self.runner.run_job('qual_long','CPU_ONLY','FP_NATIVE'))
        self.assertFalse((self.d/'jobs').exists())

    def test_resource_first_failure_survives_later_worker_rc0(self):
        self.runner.dummy_delay=.02
        def fault(s,owner):
            if owner is not None:
                self.runner.first_failure('RESOURCE_FIRST_BEFORE_RC0')
        self.resource.fault=fault
        self.rejected(lambda:self.runner.run_job('qual_long','CPU_ONLY','FP_NATIVE'))
        self.assertEqual(R.read(self.d/'FIRST_FAILURE.json')['reason'],'RESOURCE_FIRST_BEFORE_RC0')
        self.assertFalse(self.runner.jobs[-1]['qualified_completion'])

    def test_nonzero_cpu_child_stops_and_exits(self):
        self.runner.dummy_rc=3
        self.rejected(lambda:self.runner.run_job('qual_long','CPU_ONLY','FP_NATIVE'))
        self.assertEqual(self.runner.jobs[-1]['returncode'],3);self.assertFalse(self.runner.jobs[-1]['qualified_completion'])

    def test_protective_15_5_and_absolute_16_budget_not_reset(self):
        self.runner.jobs=[dict(charged_seconds=15.5*3600)]
        self.rejected(self.runner.budget_guard)
        self.assertEqual(R.read(self.d/'FIRST_FAILURE.json')['reason'],'GPU_BUDGET_PROTECTIVE_STOP')
        self.runner.jobs=[dict(charged_seconds=16*3600)]
        self.rejected(self.runner.budget_guard)

    def test_forecast_measured_long5_full30_margin(self):
        self.runner.jobs=[dict(kind='qual_long',condition=c,charged_seconds=360,qualified_completion=True,tag=c,
                              launch_to_wait_seconds=350,terminal_gate_seconds=10) for c in R.CONDITIONS]
        self.runner.jobs.append(dict(kind='qual_fresh',condition='FP_NATIVE',charged_seconds=36,
                               qualified_completion=True,tag='fresh',launch_to_wait_seconds=26,terminal_gate_seconds=10))
        result=self.runner.forecast()
        self.assertAlmostEqual(result['formal_prediction_GPUh'],3.75)
        self.assertAlmostEqual(result['projected_total_GPUh'],.51+3.75+.25)

    def test_forecast_infeasible_stop_no_condition_removed(self):
        self.runner.jobs=[dict(kind='qual_long',condition=c,charged_seconds=1800) for c in R.CONDITIONS]
        self.rejected(self.runner.forecast)
        self.assertEqual(R.read(self.d/'BUDGET_FORECAST.json')['formal_trajectories'],30)
        self.assertEqual(R.read(self.d/'FIRST_FAILURE.json')['reason'],'BUDGET_FEASIBILITY_STOP')

    def test_environment_injected_before_fresh_interpreter(self):
        env=self.runner.environment();self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'6')
        self.assertEqual(env['PYTHONHASHSEED'],'0');self.assertEqual(env['LING_TARGET_GPU_UUID'],R.UUID)
        result=subprocess.run([sys.executable,'-B','-c','import os,sys,json;print(json.dumps([os.environ["PYTHONHASHSEED"],sys.flags.hash_randomization,"torch" in sys.modules]))'],env=env,capture_output=True,text=True,check=True)
        self.assertEqual(json.loads(result.stdout),['0',0,False])

    def test_source_and_profile_bad_cannot_complete(self):
        self.runner.run_job('qual_long','CPU_ONLY','FP_NATIVE')
        entry=self.runner.jobs[-1];p=Path(entry['result_path']);j=R.read(p);j['UUID']='WRONG';p.write_text(json.dumps(j))
        self.rejected(lambda:self.runner.validate_assets(p,'qual_long','CPU_ONLY','FP_NATIVE',{k:self.runner.environment()[k] for k in R.ENV_KEYS}))

    def test_phase0_preflight_failure_seals_without_GPU_launch(self):
        def bad_preflight():raise AssertionError('CPU_GATE_NOT_PASS')
        self.runner.preflight=bad_preflight
        sealed=[];original=R.import_file
        try:
            R.import_file=lambda path,name:type('CPUSeal',(),{'seal':staticmethod(lambda d:sealed.append(str(d)))})
            self.runner.execute()
        finally:R.import_file=original
        self.assertEqual(sealed,[str(self.d)]);self.assertTrue((self.d/'FIRST_FAILURE.json').exists())
        self.assertEqual(R.read(self.d/'GPU_RESOURCE_LEDGER.json')['charged_GPU_hours'],0)
        self.assertFalse((self.d/'jobs').exists())

    def test_duplicate_controller_is_read_only_rejection(self):
        R.publish(self.d/'RUN_CONTROLLER.json',dict(owner='EXISTING',CPU_ONLY=True))
        before=R.sha(self.d/'RUN_CONTROLLER.json')
        self.rejected(self.runner.execute)
        self.assertEqual(before,R.sha(self.d/'RUN_CONTROLLER.json'))
        self.assertFalse((self.d/'FIRST_FAILURE.json').exists())

    def registration_fault(self,stage,delay):
        self.runner.dummy_delay=delay;self.runner.cleanup_timeout=.5
        calls=[];original_command=self.runner.worker_command
        def counted(*args,**kwargs):calls.append(1);return original_command(*args,**kwargs)
        self.runner.worker_command=counted
        original_identity=R.process_identity;original_publish=R.publish
        if stage=='proc':
            def broken_identity(pid):
                if pid!=os.getpid():raise OSError('INJECTED_CHILD_PROC_REGISTRATION_FAILURE')
                return original_identity(pid)
            R.process_identity=broken_identity
        else:
            def broken_publish(path,value):
                if Path(path).name==stage:raise OSError('INJECTED_'+stage+'_PUBLICATION_FAILURE')
                return original_publish(path,value)
            R.publish=broken_publish
        original_cleanup=self.runner.stop_own_child;failure_digests=[]
        def observed_cleanup(*args):
            self.assertTrue((self.d/'FIRST_FAILURE.json').exists())
            failure_digests.append(R.sha(self.d/'FIRST_FAILURE.json'))
            return original_cleanup(*args)
        self.runner.stop_own_child=observed_cleanup
        try:self.rejected(lambda:self.runner.run_job('qual_long','CPU_REGISTRATION','FP_NATIVE'))
        finally:R.process_identity=original_identity;R.publish=original_publish
        self.assertEqual(len(calls),1);self.assertEqual(len(self.runner.jobs),1)
        entry=self.runner.jobs[0];cleanup=R.read(next((self.d/'jobs').glob('*/OWN_CHILD_CLEANUP.json')))
        self.assertTrue(entry['registration_failure']);self.assertTrue(entry['wait_confirmed_exit'])
        self.assertFalse(entry['qualified_completion']);self.assertEqual(entry['status'],'FAIL_CLOSED_STOP')
        self.assertEqual(R.sha(self.d/'FIRST_FAILURE.json'),failure_digests[0]);self.assertEqual(len(failure_digests),1)
        self.assertTrue(entry['charged_seconds']>=entry['launch_to_wait_seconds']>0)
        self.assertEqual(R.read(self.d/'GPU_RESOURCE_LEDGER.json')['active_workers'],[])
        self.assertTrue(cleanup['wait_confirmed_exit']);self.assertEqual(cleanup['actions'],[])
        self.assertFalse(R.exact_identity_present(cleanup['owner']) if cleanup['owner'] is not None else False)
        if stage=='proc':
            self.assertIsNone(entry['owner']);self.assertIsNone(entry['own_process_identity_absent_after_wait'])
            self.assertEqual(entry['owner_identity_status'],'UNKNOWN')
            self.assertTrue(cleanup['unknown_identity_never_authorizes_signal'])
        else:self.assertIs(entry['own_process_identity_absent_after_wait'],True)

    def test_registration_proc_failure_fast_actual_CPU_child(self):self.registration_fault('proc',0.)
    def test_registration_proc_failure_live_actual_CPU_child(self):self.registration_fault('proc',.12)
    def test_registration_LAUNCH_publication_failure_actual_CPU_child(self):self.registration_fault('LAUNCH.json',.12)
    def test_registration_ENV_publication_failure_actual_CPU_child(self):self.registration_fault('ENV.json',.12)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=D/'CPU_CONTROL_TESTS.json')
    args=parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU_TESTS_REQUIRE_HIDDEN_CUDA'
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report=dict(task_id=R.TASK,status='PASS_CPU_CONTROL_TESTS' if result.wasSuccessful() else 'FAIL_CPU_CONTROL_TESTS',
        tests_run=result.testsRun,failures=[dict(test=str(a),traceback=b) for a,b in result.failures],
        errors=[dict(test=str(a),traceback=b) for a,b in result.errors],CPU_ONLY=True,
        GPU_API_queries=0,GPU_model_workers=0,torch_imported='torch' in sys.modules,
        scope='Real CPU child processes plus synthetic three-source resource records; never a CUDA/model or actual GPU qualification.',
        source_sha256={p.name:R.sha(p) for p in [Path(__file__).resolve(),Path(R.__file__).resolve()]})
    prior=D/'audits/CPU_CONTROL_TESTS_FINAL20.json'
    if prior.exists():
        assert R.sha(prior)=='1d359aaf17d3294da8838b962d1b8d60ab49c4b6c24b785d68ed0814b5786f1f'
        report['prior_FINAL20_report_preserved_sha256']=R.sha(prior)
        report['registration_boundary_actual_CPU_child_cases']=4
    output=args.output
    with output.open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(status=report['status'],tests_run=result.testsRun,path=str(output),sha256=R.sha(output))))
    raise SystemExit(0 if result.wasSuccessful() else 1)
