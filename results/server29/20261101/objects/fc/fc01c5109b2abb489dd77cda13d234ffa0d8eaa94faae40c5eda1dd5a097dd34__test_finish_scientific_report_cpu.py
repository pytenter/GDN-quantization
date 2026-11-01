#!/usr/bin/env python3
"""Synthetic stdlib tests only; no model, Torch, GPU, network or real finisher."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
RUN = Path(__file__).resolve().parents[1]
SOURCE = RUN / 'scripts/finish_scientific_report.py'
spec = importlib.util.spec_from_file_location('finisher_fixture', SOURCE)
f = importlib.util.module_from_spec(spec); spec.loader.exec_module(f)


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def complete_fixture():
    units = []
    for name in f.UNITS:
        row = dict(unit=name, status='PASS_COMPLETE_UNIT_IDENTITY_AND_RETAINED_METADATA', issues=[], result={})
        if 'FP_R' in name: row['measurement'] = dict(tokens_per_second=14.0, wall_ms_per_token=1000/14)
        units.append(row)
    reducers = dict(pair=dict(status='ACTUAL_REDUCER_RESULT', result=dict(status='PASS_FULL_32TOKEN_RETAINED_CPU_PAIR_COMPARISON', complete_pair=True, all_byte_equal=True, all_torch_equal=True, actual_token_pairs=32, actual_intermediate_pairs=768)))
    for u in f.UNITS[:2]: reducers[u] = dict(status='ACTUAL_REDUCER_RESULT', result=dict(status='PASS_FULL_32TOKEN_24LAYER_RETAINED_FP64_REPLAY', complete_32token_24layer_audit=True, actual_records_audited=768))
    resource = dict(status='PASS_EVENT_DERIVED_INTERVALS', units=[dict(unit=u, worker_returncode=0) for u in f.UNITS])
    return units, reducers, resource, dict(status='COMPLETE_FINITE_RECOVERY_UNITS')


class Decisions(unittest.TestCase):
    def test_full_pass(self):
        self.assertTrue(f.decide(*complete_fixture(), None)['engineering_target_met'])

    def test_slow_qualified_is_actual_partial_speed(self):
        units, red, res, controller = complete_fixture()
        for u in units:
            if u['unit'].startswith('OLD'): u['measurement'].update(tokens_per_second=12, wall_ms_per_token=1000/12)
        d = f.decide(units, red, res, controller, None)
        self.assertTrue(d['FP_short_protocol_qualified']); self.assertFalse(d['engineering_target_met'])
        self.assertEqual(d['status'], 'HISTORICAL_PORT_FP_QUALIFIED_PARTIAL_SPEED')

    def test_only_two_old_cannot_median(self):
        a = list(complete_fixture()); a[0][4]['status'] = 'NOT_RUN_NO_DISPATCH'
        d = f.decide(*a, None); self.assertIsNone(d['current_OLD_three_repeat_median_TPS']); self.assertFalse(d['FP_short_protocol_qualified'])

    def test_reducer_error_with_stale_pass_not_qualified(self):
        a=list(complete_fixture()); a[1]['pair']['status']='REDUCER_ERROR'
        self.assertFalse(f.decide(*a, None)['full_retained_pair_pass'])

    def test_replay_error_with_stale_pass_not_qualified(self):
        a=list(complete_fixture()); a[1][f.UNITS[0]]['status']='REDUCER_ERROR'
        self.assertFalse(f.decide(*a, None)['full_retained_FP64_replay_pass'])

    def test_partial_raw_never_full(self):
        a=list(complete_fixture()); a[1]['pair']['result']['actual_token_pairs']=31
        self.assertFalse(f.decide(*a, None)['FP_short_protocol_qualified'])

    def test_byte_difference_fails_even_numeric_equal(self):
        a=list(complete_fixture()); a[1]['pair']['result'].update(all_byte_equal=False, earliest_difference={'byte_only':True})
        d=f.decide(*a, None); self.assertEqual(d['status'],'FP_NUMERICAL_QUALIFICATION_FAIL'); self.assertFalse(d['FP_short_protocol_qualified'])

    def test_incomplete_replay_count_never_full(self):
        a=list(complete_fixture()); a[1][f.UNITS[0]]['result']['actual_records_audited']=767
        self.assertFalse(f.decide(*a, None)['FP_short_protocol_qualified'])

    def test_prior_first_failure_blocks_success(self):
        self.assertFalse(f.decide(*complete_fixture(), {'status':'FAIL_CLOSED_STOP'})['FP_short_protocol_qualified'])

    def test_nonzero_exit_blocks_qualification(self):
        a=list(complete_fixture()); a[2]['units'][0]['worker_returncode']=1
        self.assertFalse(f.decide(*a, None)['FP_short_protocol_qualified'])

    def test_resource_unresolved_blocks_qualification(self):
        a=list(complete_fixture()); a[2]['status']='UNRESOLVED_OR_INVALID_RESOURCE_EVENTS'
        self.assertFalse(f.decide(*a, None)['FP_short_protocol_qualified'])

    def test_absent_numeric_blocks_qualification(self):
        a=list(complete_fixture()); a[0][1]['status']='DISPATCHED_PARTIAL_OR_FAILED'
        self.assertFalse(f.decide(*a, None)['FP_short_protocol_qualified'])

    def test_no_legal_rows_history_not_filled(self):
        a=list(complete_fixture())
        for u in a[0]: u['status']='NOT_RUN_NO_DISPATCH'
        d=f.decide(*a, None); self.assertIsNone(d['current_OLD_three_repeat_median_TPS']); self.assertEqual(d['historical_OLD_FP_TPS_reference_only'],13.742036038730332)

    def test_optional_new_absence_does_not_block_five_mandatory(self):
        a=list(complete_fixture()); a[0][-1]['status']='NOT_RUN_NO_DISPATCH'; a[2]['units'].pop()
        self.assertTrue(f.decide(*a, None)['FP_short_protocol_qualified'])


class ResourceEvents(unittest.TestCase):
    def fixture(self, returncode=0):
        ident=dict(pid=101,starttick=42,cmdline=['python','worker'])
        rows=[dict(event='FIRST_DISPATCH',unit=dict(id=f.UNITS[0]),worker_identity=ident),
              dict(event='FIRST_PROCESS_EXIT_OBSERVED',unit=dict(id=f.UNITS[0]),worker_identity=ident,returncode=returncode),
              dict(event='FIRST_ALL_UUID_NVML_RELEASE_OBSERVED',unit=dict(id=f.UNITS[0]),worker_identity=ident),
              dict(event='UNIT_RESERVATION_INTERVAL_BOUNDS',unit_id=f.UNITS[0],reservation_seconds_lower=10,reservation_seconds_upper=12),
              dict(event='NVML_OBSERVATION',unit=None,worker_identity=ident,observation=dict(apps=[dict(pid=101,uuid=f.UUID,memory_MiB='1024')]))]
        return rows

    def test_real_interval_not_physical_utilization(self):
        d=f.resource_summary(self.fixture(),[]); self.assertEqual(d['status'],'PASS_EVENT_DERIVED_INTERVALS'); self.assertEqual(d['actual_physical_GPU_hours'],'UNRESOLVED'); self.assertEqual(d['cumulative_reservation_GPU_hours_lower'],10/3600)

    def test_failed_exit_still_billed(self):
        d=f.resource_summary(self.fixture(2),[]); self.assertEqual(d['units'][0]['worker_returncode'],2); self.assertEqual(d['cumulative_reservation_GPU_hours_upper'],12/3600)

    def test_empty_events_not_zero(self):
        d=f.resource_summary([],[]); self.assertNotEqual(d['status'],'PASS_EVENT_DERIVED_INTERVALS'); self.assertEqual(d['cumulative_reservation_GPU_hours_lower'],'UNRESOLVED')

    def test_duplicate_first_exit_rejected(self):
        rows=self.fixture(); rows.append(copy.deepcopy(rows[1])); self.assertNotEqual(f.resource_summary(rows,[])['status'],'PASS_EVENT_DERIVED_INTERVALS')

    def test_wrong_uuid_rejected(self):
        rows=self.fixture(); rows[-1]['observation']['apps'][0]['uuid']='FOREIGN'; self.assertNotEqual(f.resource_summary(rows,[])['status'],'PASS_EVENT_DERIVED_INTERVALS')

    def test_missing_release_rejected(self):
        rows=self.fixture(); del rows[2]; self.assertNotEqual(f.resource_summary(rows,[])['status'],'PASS_EVENT_DERIVED_INTERVALS')

    def test_exit_identity_rejected(self):
        rows=self.fixture(); rows[1]['worker_identity']={**rows[1]['worker_identity'],'starttick':43}; self.assertNotEqual(f.resource_summary(rows,[])['status'],'PASS_EVENT_DERIVED_INTERVALS')

    def test_reversed_interval_rejected(self):
        rows=self.fixture(); rows[3]['reservation_seconds_upper']=9; self.assertNotEqual(f.resource_summary(rows,[])['status'],'PASS_EVENT_DERIVED_INTERVALS')


class ReleaseAndNoExecution(unittest.TestCase):
    def fixture(self, run):
        run=run.resolve();payload=run/'payload'; ident=dict(pid=1,starttick=2,cmdline=['controller']); worker=dict(pid=3,starttick=4,cmdline=['worker'])
        write(payload/'execution/CONTROLLER_STARTED.json',dict(controller_identity=ident))
        write(payload/'execution/NUMERIC_FRESH0/DISPATCH.json',dict(worker_identity=worker))
        (payload/'execution/RESOURCE_EVENTS.jsonl').write_text('{}\n')
        ledger=f.descriptor(payload/'execution/RESOURCE_EVENTS.jsonl',payload)
        proof=dict(schema='QWEN_RECOVERY_RELEASE_ADMISSION_V1',task_id=f.TASK,status='PASS_ALL_OWN_PROCESSES_EXITED_ALL_UUIDS_RELEASED',payload_root=str(payload),execution_root=str(payload),controller_exit_verified=True,controller_exited=True,all_owned_processes_exited=True,all_owned_GPU_resources_released=True,source_freeze_sha256='frozen',controller_identity=ident,owned_worker_identities=[worker],own_processes=[dict(**i,process_exit_verified=True,all_UUID_NVML_release_verified=True) for i in [ident,worker]],final_owned_NVML_records=[],final_NVML_apps=[dict(pid=99,uuid='foreign')],controller_result=None,resource_ledger=ledger,first_failure=None,evidence_receipts=[f.descriptor(payload/'execution/CONTROLLER_STARTED.json',run)])
        path=run/'release.json';write(path,proof)
        return path,proof,dict(identities=dict(source_freeze_sha256='frozen'))

    def test_release_superset_foreign_apps_allowed(self):
        with tempfile.TemporaryDirectory() as d:
            run=Path(d); path,p,b=self.fixture(run); self.assertEqual(f.release_admission(run,path,b)['status'],p['status'])

    def test_own_live_pid_denied(self):
        with tempfile.TemporaryDirectory() as d:
            run=Path(d); path,p,b=self.fixture(run); p['final_NVML_apps'].append(dict(pid=3,uuid=f.UUID));write(path,p)
            with self.assertRaises(RuntimeError):f.release_admission(run,path,b)

    def test_missing_worker_identity_denied(self):
        with tempfile.TemporaryDirectory() as d:
            run=Path(d); path,p,b=self.fixture(run);p['owned_worker_identities']=[];write(path,p)
            with self.assertRaises(RuntimeError):f.release_admission(run,path,b)

    def test_unreleased_controller_denied(self):
        with tempfile.TemporaryDirectory() as d:
            run=Path(d);path,p,b=self.fixture(run);p['own_processes'][0]['process_exit_verified']=False;write(path,p)
            with self.assertRaises(RuntimeError):f.release_admission(run,path,b)

    def test_evidence_tamper_denied(self):
        with tempfile.TemporaryDirectory() as d:
            run=Path(d);path,p,b=self.fixture(run);(run/'payload/execution/CONTROLLER_STARTED.json').write_text('{}')
            with self.assertRaises(RuntimeError):f.release_admission(run,path,b)

    def test_terminal_descriptor_required(self):
        with tempfile.TemporaryDirectory() as d:
            run=Path(d);path,p,b=self.fixture(run);write(run/'payload/FIRST_FAILURE.json',dict(status='FAIL_CLOSED_STOP'))
            with self.assertRaises(RuntimeError):f.release_admission(run,path,b)

    def test_no_raw_never_invokes_reducer(self):
        with tempfile.TemporaryDirectory() as d:
            def forbidden(*a,**k):raise AssertionError('Reducer must not execute')
            out=f.run_reducers(Path(d),forbidden); self.assertTrue(all(x['status'].startswith('NOT_RUN') for x in out.values()))

    def test_reducer_error_saved_once(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); script=root/'dummy.py';script.write_text('# fixture only\n');count=[]
            def fail(*a,**k):count.append(1);raise ValueError('synthetic CPU exception')
            out=f.reducer_once([sys.executable,'-B',str(script)],root/'out',root/'out/result.json',fail)
            self.assertEqual(out['status'],'REDUCER_ERROR');self.assertEqual(len(count),1);self.assertFalse(out['retry'])

    def test_existing_reducer_output_denies_retry(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);script=root/'dummy.py';script.write_text('');out=root/'out';out.mkdir()
            with self.assertRaises(RuntimeError):f.reducer_once([sys.executable,'-B',str(script)],out,out/'x')

    def test_path_traversal_and_symlink_denied(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for value in ['../x','/etc/passwd','a/../x']:
                with self.assertRaises(RuntimeError):f.safe(root,value)
            (root/'alias').symlink_to(root,target_is_directory=True)
            with self.assertRaises(RuntimeError):f.safe(root,'alias/x')

    def test_default_preview_no_writes(self):
        with tempfile.TemporaryDirectory() as d:
            run=Path(d);out=subprocess.run([sys.executable,'-B',str(SOURCE),'--root',str(run)],capture_output=True,text=True,check=True)
            rec=json.loads(out.stdout);self.assertEqual(rec['status'],'PREVIEW_ONLY_NO_REDUCER_NO_WRITES_NO_GPU');self.assertEqual(list(run.iterdir()),[])


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]); result=unittest.TextTestRunner(verbosity=2).run(suite)
    output=dict(status='PASS_CPU_SYNTHETIC_FINISHER_CONTROLS' if result.wasSuccessful() else 'FAIL_CPU_SYNTHETIC_FIXTURE',tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Stdlib synthetic decisions, resource/release admission, fake reducer exceptions and default preview only. No real finish, model, Torch, Driver, NVML, GPU, network or source mutation.',GPU_queries=0,GPU_dispatches=0)
    path=RUN/'tests/SCIENTIFIC_FINISHER_CPU_TEST_RESULTS.json'
    if path.exists():
        with (RUN/'tests/SCIENTIFIC_FINISHER_CPU_TEST_HISTORY.jsonl').open('a') as stream:stream.write(path.read_text().strip()+'\n')
    path.write_text(json.dumps(output,indent=2)+'\n');sys.exit(0 if result.wasSuccessful() else 1)
