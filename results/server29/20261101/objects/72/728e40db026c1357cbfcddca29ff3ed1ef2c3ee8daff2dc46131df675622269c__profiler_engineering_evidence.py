"""Read-only, current-attempt profiler evidence; no dispatch or qualification credit."""
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def describe(root, path):
    path = Path(path)
    assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root)
    return dict(path=str(path.relative_to(root)), bytes=path.stat().st_size, sha256=sha(path))


def validate(root, control):
    root = Path(root).resolve()
    records = []
    for directory, expected in [
        ('profiler_exit_diagnostics', {'EXACT_H_CURRENT_GDB': 0, 'MICRO_NO_PROFILE': 0,
            'MICRO_CPU_PROFILE': -11, 'MICRO_CUDA_PROFILE': 0, 'MICRO_CUDA_PROFILE_GC': 0,
            'MICRO_TORCH_FIRST_CUDA': 0}),
        ('profiler_exit_detach_diagnostics', {'CPU_CURRENT_GDB_ASLR': 255,
            'CPU_CURRENT_EXPLICIT_DETACH': 0, 'CUDA_CURRENT_EXPLICIT_DETACH': 0,
            'CPU_TORCH_FIRST_CONTROL': 0, 'EXACT_H_EXPLICIT_DETACH_R0': 0,
            'EXACT_H_EXPLICIT_DETACH_R1': 0}),
    ]:
        base = root / 'operations' / directory
        complete = read(base / 'INVESTIGATION_OBSERVATIONS_COMPLETE.json')
        assert complete['actual_dispatches'] == 6 and complete['all_new_probe_GPU_processes_released']
        for case, code in expected.items():
            p = base / case / 'OBSERVATION.json'
            row = read(p)
            assert row['case']['id'] == case and row['actual_Popen_wait_returncode'] == code
            assert row['release'] == 'PASS' and not row['counts_as_scientific_qualification']
            assert row['AIME_generations'] == 0
            if case == 'CPU_CURRENT_GDB_ASLR':
                assert row['GDB_reported_SIGSEGV'] is True
            for name in ['subprocess_identity', 'inferior_identity', 'supervisor_identity']:
                assert not control.same_process(row[name]), (case, name)
            lower, exit_observed, release = [row[k] for k in [
                'dispatch_monotonic_lower', 'exit_first_observed_monotonic',
                'release_first_observed_monotonic']]
            assert lower <= exit_observed <= release
            records.append(dict(id=case, receipt=describe(root, p),
                actual_returncode=code, GDB_reported_SIGSEGV=row['GDB_reported_SIGSEGV'],
                actual_model_forwards=row['model_forwards'],
                reservation_GPUh_upper=(release - lower) / 3600,
                resource_interval=dict(dispatch_monotonic_lower=lower,
                    exit_first_observed_monotonic=exit_observed,
                    release_first_observed_monotonic=release, monotonic_same_host=True)))
    base = root / 'operations/actual_profile_guard_integration'
    completed = read(base / 'COMPLETE.json')
    assert completed['status'] == 'PASS_THREE_ACTUAL_UNCHANGED_QUALIFIER_CLI_PROFILE_GUARD_EXITS'
    assert completed['all_actual_exit0_and_own_release'] and len(completed['results']) == 3
    assert completed['original_qualifier_sha256'] == sha(root / 'scripts/qualify_gpu.py')
    mini = base / 'engineering_only_root'
    assert sha(mini / 'src/qualification_guards.py') == sha(root / 'src/qualification_guards.py')
    assert sha(mini / 'src/profiler_detach_lifecycle.py') == sha(root / 'src/profiler_detach_lifecycle.py')
    supervisor = read(base / 'STARTED.json')['identity']
    assert not control.same_process(supervisor)
    for condition in ['H', 'C1', 'S3']:
        name = 'ENGINEERING_' + condition + '_PROFILE_SHORT'
        p = base / (name + '_COMPLETION.json')
        row = read(p)
        assert row['actual_wait_returncode'] == 0 and row['release'] == 'PASS'
        assert row['actual_model_forwards'] == 136 and not control.same_process(row['identity'])
        target = mini / 'qualification' / name
        result = read(target / 'RESULT.json')
        assert sha(target / 'RESULT.json') == row['result_sha256'] and result['status'] == 'PASS'
        assert sha(target / 'CUPTI_PROFILE_DETACH.json') == row['detach_receipt_sha256']
        detach = read(target / 'CUPTI_PROFILE_DETACH.json')
        assert detach['flush_returncode'] == detach['finalize_returncode'] == 0
        assert detach['actual_process_exit_still_required'] and not detach['signal_or_exit_suppression']
        assert sha(target / 'CUDA_PROFILE.json') == detach['trace_sha256']
        lower, exited, released = [row[k] for k in ['dispatch_monotonic_lower',
            'exit_first_observed_monotonic', 'release_first_observed_monotonic']]
        assert lower <= exited <= released
        records.append(dict(id=name, receipt=describe(root, p), actual_returncode=0,
            actual_model_forwards=136, reservation_GPUh_upper=(released - lower) / 3600,
            resource_interval=dict(dispatch_monotonic_lower=lower,
                exit_first_observed_monotonic=exited, release_first_observed_monotonic=released,
                monotonic_same_host=True)))
    frozen = read(root / 'ENGINEERING_PROFILER_DETACH_FREEZE.json')
    copies = read(root / 'provenance/PROFILER_ENGINEERING_ORIGINAL_INPUTS/COPY_MAP.json')
    by = {x['path']: x for x in copies['files']}
    assert len(by) == len(frozen['files']) == 348
    for old in frozen['files']:
        saved = by[old['path']]
        assert saved['sha256'] == old['sha256'] and saved['bytes'] == old['bytes']
        preserved = root / saved['preserved_path']
        assert preserved.stat().st_size == old['bytes'] and sha(preserved) == old['sha256']
    assert len(records) == 15 and sum(x['actual_model_forwards'] for x in records) == 816
    return dict(status='PASS_CURRENT_PROFILER_REPAIR_EVIDENCE_ALL15_EXITED_RELEASED',
        engineering_dispatch_count=15, inherited_loader_engineering_dispatches=12,
        qualification_units_contributed=0, formal_samples_contributed=0,
        actual_nonAIME_model_forwards=816, records=records,
        reservation_GPUh_upper=sum(x['reservation_GPUh_upper'] for x in records),
        EXACT_physical_GPUh='UNRESOLVED_NOT_KERNEL_BUSY_TIME',
        failures_preserved=True, original348_consumed_inputs_preserved=True,
        guard_sha256=sha(root / 'src/qualification_guards.py'),
        lifecycle_sha256=sha(root / 'src/profiler_detach_lifecycle.py'),
        qualifier_sha256=sha(root / 'scripts/qualify_gpu.py'),
        human_authorization_sha256=sha(root / 'HUMAN_PROFILER_REPAIR_AUTHORIZATION.json'),
        scope='Current15 engineering processes only; old12 remain a distinct inherited cost scope')
