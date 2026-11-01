"""Isolated CPU fixtures for repair union, actual scorer waits and delivery guards.

No model, CUDA, AIME scoring, production filesystem or network operation occurs.
"""
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


F = load('repair_finisher_cpu_fixture', 'scripts/finalize_cpu.py')
H = load('repair_delivery_cpu_fixture', 'operations/pace_independent_delivery.py')
Z = load('repair_freeze_cpu_fixture', 'operations/freeze_static_once.py')
P = load('repair_parent_cpu_fixture', 'scripts/offline_parent.py')
CASES = []


def check(value, name):
    if not value:
        raise AssertionError(name)
    CASES.append(name)


def rejects(action, name):
    try:
        action()
    except (RuntimeError, AssertionError, KeyError, ValueError, FileNotFoundError, FileExistsError):
        CASES.append(name)
    else:
        raise AssertionError('Unsafe fixture accepted: ' + name)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def union_fixtures():
    jobs = [dict(id=f'q{i:02d}_{arm}', condition=arm) for i in range(30) for arm in ('R2', 'R3')]
    pairs = [(job, dict(source_manifest_sha256='old' if i < 3 else 'current'), Path('/fixture') / job['id'])
             for i, job in enumerate(jobs)]
    intervals = [dict(stage='formal', returncode=0, released=True) for _ in range(57)]
    gate = dict(stage='scale_repair_gates', unit='SCALE_REPAIR_GATES', returncode=0, released=True)
    intervals.append(gate)
    with patch.object(F.C, 'sha', return_value='current'), \
         patch.object(F.C, 'read', return_value={'intervals': intervals}), \
         patch.object(F.K, 'continuation_summary', return_value={'fixture_only': True, 'runtime_reuse_admitted': True, 'admitted_reused_formal_count': 3}):
        session = {'status': 'GPU_COMPLETE_PACE48_REUSE3_PLUS57_FORMAL', 'reused_formal_count': 3, 'current_formal_count': 57, 'formal_total_unique': 60}
        full, accounting, _ = F.full_completion(Path('/fixture'), session, pairs, [])
        check(full and accounting['reused_formal_completed'] == 3 and accounting['current_formal_completed'] == 57,
              'strict_three_old_plus_fiftyseven_current_full_union')
        check(not F.full_completion(Path('/fixture'), session, pairs[:-1], [])[0], 'partial_not_full')
        check(not F.full_completion(Path('/fixture'), session, pairs, [{'invalid': True}])[0], 'invalid_receipt_not_full')
        check(not F.full_completion(Path('/fixture'), {'status': 'GPU_COMPLETE_PACE48_60_FORMAL'}, pairs, [])[0],
              'old_full_status_not_current_success')
        wrong = [(job, dict(wrapper, source_manifest_sha256='current'), target) for job, wrapper, target in pairs]
        check(not F.full_completion(Path('/fixture'), session, wrong, [])[0], 'sixty_current_not_admitted_three_plus57')
        duplicate = list(pairs); duplicate[-1] = pairs[-2]
        rejects(lambda: F.full_completion(Path('/fixture'), session, duplicate, []), 'duplicate_union_rejected')
        intervals.remove(gate)
        check(not F.full_completion(Path('/fixture'), session, pairs, [])[0], 'missing_actual_new_repair_gate_refuses_full')
        intervals.append(gate)
        gate['returncode'] = 1
        check(not F.full_completion(Path('/fixture'), session, pairs, [])[0], 'failed_new_repair_gate_refuses_full')
        gate['returncode'] = 0
        intervals[0]['returncode'] = 1
        check(not F.full_completion(Path('/fixture'), session, pairs, [])[0], 'actual_nonzero_exit_refuses_full')
        intervals[0]['returncode'] = 0; intervals[0]['released'] = False
        check(not F.full_completion(Path('/fixture'), session, pairs, [])[0], 'unreleased_refuses_full')


def partial_finisher_fixture():
    with tempfile.TemporaryDirectory(prefix='pace_repair_partial_cpu_') as name:
        root = Path(name).resolve()
        pairs = [(dict(id=n, condition=arm), dict(source_manifest_sha256='old'), root / 'provenance' / n)
                 for n, arm in [('q1_R2', 'R2'), ('q1_R3', 'R3'), ('q2_R2', 'R2')]]
        session = {'status': 'FAIL_CLOSED_STOP', 'failure': {'error': 'fixture_only'}}
        archived = []
        original_read = F.C.read
        def read(path):
            return session if Path(path).name == 'SESSION_RESULT.json' else original_read(path)
        with patch.object(F.C, 'REMOTE', str(root)), patch.object(F.C, 'sha', return_value='current'), \
             patch.object(F.C, 'process_identity', return_value={'pid': os.getpid(), 'starttick': 1, 'cmdline': ['CPU_FIXTURE']}), \
             patch.object(F.C, 'read', side_effect=read), \
             patch.object(F, 'terminal_guard', return_value={'status': 'CPU_MOCK_ONLY'}), \
             patch.object(F, 'complete_rows', return_value=pairs), \
             patch.object(F, 'full_completion', return_value=(False, {'reused_formal_completed': 3, 'current_formal_completed': 0}, {'R2': 2, 'R3': 1})), \
             patch.object(F, 'run_score_child', side_effect=AssertionError('Partial must never score')), \
             patch.object(F, 'archive', side_effect=lambda q: archived.append(str(q))):
            check(F.finalize(root) == 0, 'partial_finisher_closes_truthful_stop')
        verdict = original_read(root / 'final_verdict.json')
        check(verdict['formal_completed'] == 3 and verdict['R2_completed'] == 2 and verdict['R3_completed'] == 1,
              'partial_reused_three_counts_retained')
        check(verdict['full_accuracy_reported'] is False and verdict['partial_scientific_CSV_created'] is False
              and not (root / 'comparison').exists(), 'partial_no_accuracy_no_empty_scientific_csv')
        check(len(archived) == 1, 'one_finisher_archive_operation_in_fixture')
        rejects(lambda: F.finalize(root), 'finisher_restart_started_receipt_rejected')


def real_scorer_wait_fixtures():
    # The child is a small stdlib CPU receipt fixture, not the scientific scorer.
    child = '''from pathlib import Path
import json,os,sys,time
out=Path(sys.argv[sys.argv.index('--score-pass')+1]);mode=(out.parents[0]/'MODE').read_text()
if mode=='no_identity':raise SystemExit(0)
try:starttick=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])
except OSError:starttick=time.monotonic_ns()
identity={'pid':os.getpid(),'starttick':starttick,'cmdline':[sys.executable,*sys.argv]}
identity['cmdline']=[sys.executable,*sys.argv]
# sys.argv starts with the script path; this equals the parent's actual argv.
(out.with_suffix('.STARTED.json')).write_text(json.dumps({'identity':identity}))
time.sleep(.08)
if mode=='nonzero':raise SystemExit(4)
out.write_text(json.dumps({'identity':identity,'payload':{'fixture_only':True}}))
'''
    with tempfile.TemporaryDirectory(prefix='pace_repair_scorer_wait_cpu_') as name:
        root = Path(name).resolve(); (root / 'scripts').mkdir(); (root / 'scripts/finalize_cpu.py').write_text(child)
        def identity(pid):
            if pid == os.getpid():
                return {'pid': pid, 'starttick': 1, 'cmdline': ['CPU_FIXTURE_PARENT']}
            if Path('/proc').is_dir():
                p = Path('/proc') / str(pid)
                return {'pid': pid, 'starttick': int((p/'stat').read_text().rsplit(')',1)[1].split()[19]),
                        'cmdline': [s.decode() for s in (p/'cmdline').read_bytes().split(b'\0') if s]}
            for p in root.glob('FROZEN_V4_FRESH_PASS_*.STARTED.json'):
                value = json.loads(p.read_text())['identity']
                if value['pid'] == pid:
                    return value
            raise FileNotFoundError(pid)
        def alive(value):
            try:os.kill(value['pid'], 0);return True
            except ProcessLookupError:return False
        with patch.object(F, 'PYTHON', sys.executable), patch.object(F.C, 'process_identity', side_effect=identity), \
             patch.object(F.C, 'same_process', side_effect=alive), patch.object(F.C, 'sha', return_value='fixture_sha'):
            (root / 'MODE').write_text('pass')
            first = F.run_score_child(root, 0, dict(os.environ, CUDA_VISIBLE_DEVICES=''))
            second = F.run_score_child(root, 1, dict(os.environ, CUDA_VISIBLE_DEVICES=''))
            check(first['identity']['pid'] != second['identity']['pid'], 'two_fresh_actual_CPU_children_have_distinct_PIDs')
            for index in (0, 1):
                exit = json.loads((root / f'FROZEN_V4_FRESH_PASS_{index}.PROCESS_EXIT.json').read_text())
                check(exit['returncode'] == 0 and exit['actual_exited'] is True and exit['identity_gone'] is True,
                      f'actual_wait_exit0_gone_receipt_{index}')
            (root / 'MODE').write_text('nonzero')
            rejects(lambda: F.run_score_child(root, 2, dict(os.environ, CUDA_VISIBLE_DEVICES='')), 'actual_exit4_not_fabricated_success')
            exit = json.loads((root / 'FROZEN_V4_FRESH_PASS_2.PROCESS_EXIT.json').read_text())
            check(exit['returncode'] == 4 and exit['actual_exited'] is True, 'actual_nonzero_returncode_preserved')
            (root / 'MODE').write_text('no_identity')
            rejects(lambda: F.run_score_child(root, 3, dict(os.environ, CUDA_VISIBLE_DEVICES='')), 'missing_child_identity_not_invented')



def real_offline_parent_fixtures():
    child = "from pathlib import Path\nimport json,os,sys,time\nroot=Path(sys.argv[sys.argv.index('--root')+1]);label='CONTROLLER' if Path(sys.argv[0]).name=='pace_controller.py' else 'FINISHER'\ntry:starttick=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])\nexcept OSError:starttick=time.monotonic_ns()\nidentity={'pid':os.getpid(),'starttick':starttick,'cmdline':[sys.executable,*sys.argv]}\n(root/('CPU_CHILD_IDENTITY_'+label+'.json')).write_text(json.dumps(identity))\nwith (root/'CPU_CALL_ORDER.jsonl').open('a') as f:f.write(json.dumps({'label':label,'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES')})+'\\n')\ntime.sleep(.08)\nmode=(root/'CPU_MODE').read_text()\nraise SystemExit(1 if mode==label+'_FAIL' else 0)\n"
    for mode in ('PASS', 'CONTROLLER_FAIL', 'FINISHER_FAIL'):
        with tempfile.TemporaryDirectory(prefix='pace_repair_parent_wait_cpu_') as name:
            root = Path(name).resolve();(root/'scripts').mkdir()
            for n in ('pace_controller.py', 'finalize_cpu.py'):(root/'scripts'/n).write_text(child)
            write(root/'source_manifest.json', {'fixture_only': True});write(root/'GPU_PERMIT.json', {'fixture_only': True})
            (root/'CPU_MODE').write_text(mode)
            def identity(pid):
                if pid == os.getpid():return {'pid':pid,'starttick':1,'cmdline':['CPU_FIXTURE_PARENT']}
                if Path('/proc').is_dir():
                    q=Path('/proc')/str(pid)
                    return {'pid':pid,'starttick':int((q/'stat').read_text().rsplit(')',1)[1].split()[19]),
                            'cmdline':[x.decode() for x in (q/'cmdline').read_bytes().split(b'\0') if x]}
                for q in root.glob('CPU_CHILD_IDENTITY_*.json'):
                    value=json.loads(q.read_text())
                    if value['pid']==pid:return value
                raise FileNotFoundError(pid)
            def alive(value):
                try:os.kill(value['pid'],0);return True
                except ProcessLookupError:return False
            with patch.object(P.C,'authorize',return_value={}),patch.object(P.C,'process_identity',side_effect=identity), \
                 patch.object(P.C,'same_process',side_effect=alive):
                parent=P.Parent(root);actual=parent.run()
                check(actual==(1 if mode=='FINISHER_FAIL' else 0),'parent_actual_returncode_'+mode)
                rejects(lambda: parent.run(),'parent_restart_no_second_controller_'+mode)
            order=[json.loads(x) for x in (root/'CPU_CALL_ORDER.jsonl').read_text().splitlines()]
            check([x['label'] for x in order]==['CONTROLLER','FINISHER'] and all(x['CUDA_VISIBLE_DEVICES']=='' for x in order),
                  'one_controller_then_one_CPU_finisher_'+mode)
            for label in ('CONTROLLER','FINISHER'):
                receipt=json.loads((root/('PARENT_'+label+'_EXIT.json')).read_text())
                expected=1 if mode==label+'_FAIL' else 0
                check(receipt['actual_exited'] is True and receipt['identity_gone'] is True and receipt['returncode']==expected
                      and receipt['source_manifest_sha256']==sha(root/'source_manifest.json')
                      and receipt['permit_sha256']==sha(root/'GPU_PERMIT.json'), 'owning_wait_pinned_actual_'+label+'_'+mode)


def active_observer_fixture():
    namespace = {}; exec(H.REMOTE_COMMON, namespace)
    with tempfile.TemporaryDirectory(prefix='pace_repair_active_reader_cpu_') as name:
        root = Path(name).resolve(); unit = 'q01_R2_seed1'
        worker = {'pid': 311, 'starttick': 99, 'cmdline': ['python', 'pace_worker.py', '--unit', unit]}
        parent = {'pid': 201, 'starttick': 77, 'cmdline': ['python', 'offline_parent.py']}
        controller = {'pid': 202, 'starttick': 78, 'cmdline': ['python', 'pace_controller.py']}
        target = root / 'R2_PACE48_DP8/trajectories' / unit
        write(root / 'RESOURCE_LEDGER.json', {'active_unit': unit, 'intervals': [], 'completed_units': []})
        write(root / 'OFFLINE_PARENT_STARTED.json', {'identity': parent})
        write(root / 'SESSION_STARTED.json', {'identity': controller})
        write(root / 'FORMAL_JOB_MANIFEST.json', {'jobs': [{'id': unit, 'condition': 'R2', 'decode_period': 8}]})
        write(target / 'ACTUAL_CHILD_IDENTITY.json', worker)
        dispatch = {'unit': unit, 'gpu_uuid': H.UUID, 'source_manifest_sha256': 'a'*64, 'permit_sha256': 'b'*64}
        write(target / 'DISPATCH.json', dispatch)
        seen = []
        namespace['root_pins'] = lambda _: root
        namespace['compare'] = lambda value: {'status': 'OWNED_ALIVE', 'owned_alive': True, 'expected': value}
        namespace['scoped_processes'] = lambda _: [parent, controller, worker]
        def nvml(values):
            seen.extend(values);return {'own_apps': [{'pid': worker['pid'], 'uuid': H.UUID}], 'foreign_app_count': 0}
        namespace['gpu_observation'] = nvml
        value = namespace['terminal_snapshot']({'source_sha256': 'a'*64, 'permit_sha256': 'b'*64})
        check({x['pid'] for x in seen} == {201, 202, 311}, 'parent_controller_active_worker_registered_before_first_interval')
        check(value['status'] == 'PENDING_NO_ACTION' and value['active_unit'] == unit
              and len(value['NVML']['own_apps']) == 1, 'active_worker_not_mislabelled_foreign_or_terminal')
        dispatch['permit_sha256'] = 'c'*64;write(target / 'DISPATCH.json', dispatch)
        rejects(lambda: namespace['terminal_snapshot']({'source_sha256': 'a'*64, 'permit_sha256': 'b'*64}),
                'active_worker_wrong_permit_refused')
    with tempfile.TemporaryDirectory(prefix='pace_repair_mask_origin_cpu_') as name:
        root = Path(name).resolve();oldmask = root / 'provenance/PRIOR_PACE_TRUE_STOP/calibration/head_mask.json'
        write(oldmask, {'fixed_original_mask_fixture': True})
        check(namespace['formal_mask_sha'](root, False) == sha(oldmask), 'partial_before_new_calibration_copy_uses_original_mask')
        rejects(lambda: namespace['formal_mask_sha'](root, True), 'current_formal_cannot_borrow_missing_new_mask')
    check(H.REMOTE.endswith('_scale_repair_20261025') and H.ARCHIVE.endswith(H.REVISION), 'helper_exact_new_revision_roots')


def freeze_admission_fixtures():
    with tempfile.TemporaryDirectory(prefix='pace_repair_freeze_cpu_') as name:
        root = Path(name).resolve()
        for relative in Z.REQUIRED_CPU_FILES:
            p = root / relative;p.parent.mkdir(parents=True, exist_ok=True);p.write_text('CPU_FIXTURE_' + relative)
        grant = root / Z.GRANT;grant.parent.mkdir(exist_ok=True);grant.write_bytes((ROOT / Z.GRANT).read_bytes())
        admission = root / 'analysis/PACE_REUSE3_DUAL_HOST_FRESH_REVIEW.json';write(admission, {'fixture_only': True})
        report = {'status': 'PASS_TWO_FRESH_FINAL_SCALE_REPAIR_CPU_ADMISSION', 'GPU_calls': 0,
                  'runs': [{'status': 'PASS', 'returncode': 0}, {'status': 'PASS', 'returncode': 0}],
                  'files': [{'path': n, 'bytes': (root/n).stat().st_size, 'sha256': sha(root/n)} for n in sorted(Z.REQUIRED_CPU_FILES)],
                  'human_authorization_sha256': sha(grant), 'reuse_admission_path': admission.relative_to(root).as_posix(),
                  'reuse_admission_sha256': sha(admission), 'reuse_admission_verified': True}
        path = root / 'operations/SCALE_REPAIR_FINAL_CPU_ADMISSION.json';write(path, report)
        check(Z.validate_final_cpu_admission(root)['GPU_calls'] == 0, 'freeze_accepts_two_fresh_exact_final_source_CPU_reports')
        bad = copy.deepcopy(report);bad['runs'][1]['returncode'] = 1;write(path, bad)
        rejects(lambda: Z.validate_final_cpu_admission(root), 'freeze_rejects_one_failed_fresh_run')
        bad = copy.deepcopy(report);bad['files'].append(bad['files'][0]);write(path, bad)
        rejects(lambda: Z.validate_final_cpu_admission(root), 'freeze_rejects_duplicate_source_binding')
        bad = copy.deepcopy(report);bad['files'] = bad['files'][1:];write(path, bad)
        rejects(lambda: Z.validate_final_cpu_admission(root), 'freeze_rejects_missing_required_final_source')
        write(path, report);(root / report['files'][0]['path']).write_text('changed')
        rejects(lambda: Z.validate_final_cpu_admission(root), 'freeze_rejects_post_CPU_source_edit')


def main():
    union_fixtures();partial_finisher_fixture();real_scorer_wait_fixtures();real_offline_parent_fixtures();active_observer_fixture();freeze_admission_fixtures()
    inherited = H.self_test()
    check(inherited['cases'] == 26 and inherited['network_calls'] == inherited['GPU_calls'] == 0,
          'helper_original_claim_failure_reclaim_and_transport_cases')
    print(json.dumps({'status': 'PASS', 'cases': len(CASES), 'case_names': CASES,
                      'GPU_calls': 0, 'network_calls': 0, 'model_loads': 0, 'AIME_score_calls': 0,
                      'actual_stdlib_scorer_fixture_children': 4, 'actual_stdlib_parent_fixture_children': 6,
                      'identity_scope': 'Linux_proc_actual_else_Mac_child_receipt_fixture; real owning_Popen_wait_on_both',
                      'production_roots_mutated': False}, sort_keys=True))


if __name__ == '__main__':
    main()
