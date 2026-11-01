"""Outside-capsule engineering supervisor, no Discovery or scientific advancement.

Runs only after preflight PASS. Children reuse the exact frozen runner and
environment; every run has an exclusive directory. This supervisor must itself
be archived by SHA before a qualification claim. No manifest expansion in VERIFY.
"""
from pathlib import Path
import datetime, hashlib, importlib.util, json, os, subprocess, time, traceback

R = Path('/data/zypan/worktrees/ling-long-horizon-observability-bridge-v3')
C = R / 'provenance_capsule_v3'
A = C / 'attempt_10_ast_audit_association'
UUID = 'GPU-de53f243-b271-521c-d875-d96611ed8574'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def emit(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')

def validate_completion(run):
    complete = read(run / 'PREFLIGHT_COMPLETE.json')
    assert complete['GENERATED_CODE_PREFLIGHT_GATE'] == 'PASS'
    assert complete['FORMAL_MEASUREMENTS'] == 0
    assert complete['GPU6_FORMAL_AUTHORIZED'] == 'NO'
    receipt = read(run / 'generated_capture/GENERATED_CODE_RECEIPT.json')
    assert receipt['preexec_capture'] and receipt['formal_measurements'] == 0
    runtime_sources = {row['path']: row['sha256'] for row in
                       read(C / 'runtime_environment_manifest.json')['environment_files']}
    runtime_sources.update({str((C / row['path']).resolve()): row['sha256'] for row in
                            read(A / 'CLEAN_ENGINEERING_LAUNCH.json')['files']})
    runtime_sources.update({str((C / row['snapshot_relative_path']).resolve()): row['snapshot_sha256']
                            for row in read(C / 'historical_source_snapshot_manifest.json')['files']})
    runtime_sources.update({row['path']: row['sha256'] for row in
                            read(C / 'cached_environment_module_manifest.json')['files']})
    for event in receipt['events']:
        copy = Path(event['canonical_captured_copy_path'])
        assert copy.resolve().is_relative_to(run.resolve())
        assert sha(copy) == event['source_sha256']
        assert copy.stat().st_size == event['source_size']
        producer = event['producer']
        producer_path = str(Path(producer['producer_source_path']).resolve())
        assert runtime_sources.get(producer_path) == producer['producer_source_sha256'], (
            'UNFROZEN_GENERATED_PRODUCER', producer_path)
        assert sha(producer_path) == producer['producer_source_sha256']
    assert all(event['preexec_verified'] for event in receipt['execution_events'])
    identity = complete['identity']
    assert identity['git_subprocess_count'] == 0
    assert not identity['ambient_regular_file_reads_observed']
    return receipt

def run_child(run_id, mode):
    assert not (A / 'STOP.json').exists()
    assert not (A / 'runs' / run_id).exists(), 'NO_DUPLICATE_RUN'
    gpu = subprocess.check_output([
        'nvidia-smi', '-i', '6', '--query-gpu=uuid,memory.free',
        '--format=csv,noheader,nounits'], text=True).strip()
    assert gpu.split(',')[0].strip() == UUID, 'HARDWARE_IDENTITY_FAILURE'
    assert float(gpu.split(',')[1]) > 20000, 'GPU_CAPACITY_UNAVAILABLE'
    procs = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'], text=True)
    assert not any(UUID in row for row in procs.splitlines()), 'GPU_BUSY_NO_INTERFERENCE'
    profile = read(C / 'measurement_launcher_environment.json')
    env = os.environ.copy()
    env.update(profile['set'])
    for key in profile['unset']:
        env.pop(key, None)
    argv = ['/data/zypan/envs/ling-kda/bin/python', '-B',
            str(A / 'sealed_runner_attempt10.py'), mode, run_id]
    with (A / (run_id + '.log')).open('xb') as log:
        proc = subprocess.Popen(argv, cwd=R, env=env, stdout=log,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True)
        emit(A / (run_id + '_LAUNCH.json'), dict(
            pid=proc.pid, argv=argv, gpu=gpu,
            supervisor_path=str(Path(__file__).resolve()), supervisor_sha256=sha(__file__),
            timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            FORMAL_MEASUREMENTS=0, GPU6_FORMAL_AUTHORIZED='NO'))
        print('ENGINEERING_CHILD_LAUNCHED', run_id, proc.pid, flush=True)
        status = proc.wait()
    assert status == 0, ('ENGINEERING_CHILD_FAILURE', run_id, status)
    result = validate_completion(A / 'runs' / run_id)
    print('ENGINEERING_CHILD_COMPLETE', run_id, flush=True)
    return result

def main():
    assert not (A / 'STOP.json').exists()
    assert not (A / 'GENERATED_QUALIFICATION_SUPERVISOR.json').exists()
    emit(A / 'GENERATED_QUALIFICATION_SUPERVISOR.json', dict(
        pid=os.getpid(), path=str(Path(__file__).resolve()), sha256=sha(__file__),
        scope='engineering generated-code qualification only; no Discovery',
        FORMAL_MEASUREMENTS=0, GPU6_FORMAL_AUTHORIZED='NO'))
    first = A / 'runs/preflight_01/PREFLIGHT_COMPLETE.json'
    first_pid = read(A / 'PREFLIGHT_LAUNCH.json')['pid']
    while not first.exists():
        assert not (A / 'STOP.json').exists(), 'PREFLIGHT_STOP_PRESERVED'
        process = Path('/proc') / str(first_pid)
        assert process.exists(), 'PREFLIGHT_TERMINATED_WITHOUT_COMPLETION'
        argv = (process / 'cmdline').read_bytes().split(b'\x00')
        assert str(A / 'sealed_runner_attempt10.py').encode() in argv, 'PREFLIGHT_PID_IDENTITY_CHANGED'
        time.sleep(15)
    preflight = validate_completion(A / 'runs/preflight_01')
    runs = ['fresh_process_01', 'fresh_process_02', 'fresh_process_03']
    receipts = [run_child(name, 'capture') for name in runs]
    spec = importlib.util.spec_from_file_location(
        'frozen_engineering_guard_for_receipt_comparison', A / 'generated_code_guard_attempt10.py')
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    gates = guard.deterministic_receipts(receipts)
    # Preflight is separate, but strict verification uses identical execution
    # coverage; it must match the three qualified fresh processes as well.
    guard.deterministic_receipts([preflight, receipts[0], receipts[1], receipts[2]])
    results = [read(A / 'runs' / name / 'qualification/smoke/repeat0/result.json')
               for name in ['preflight_01', *runs]]
    for field in ['scores', 'target', 'endpoint', 'FP_step_hashes']:
        assert all(row[field] == results[0][field] for row in results), (
            'FRESH_PROCESS_NUMERICAL_DETERMINISM_FAILURE', field)
    parity = [read(A / 'runs' / name / 'qualification/smoke/basis_and_jprop_parity.json')
              for name in ['preflight_01', *runs]]
    for field in ['basis_rows', 'identity_tolerance']:
        assert all(row[field] == parity[0][field] for row in parity), (
            'FRESH_PROCESS_BASIS_NUMERICAL_DETERMINISM_FAILURE', field)
    for row in parity:
        for gate in ['BASIS_SEMANTICS_GATE', 'HISTORICAL_JPROP_PARITY_GATE',
                     'KDA_TRANSITION_IDENTITY_GATE', 'QDQ_IDENTITY_GATE']:
            assert row[gate] == 'PASS', ('PARITY_GATE_FAILURE', gate)
    # Full parity receipts may contain run paths and audit timestamps. Do not
    # silently normalize numerical fields: preserve them and compare by explicit
    # schema only after inspecting the actual historical receipt structure.
    emit(A / 'GENERATED_FRESH_PROCESS_COMPARISON.json', dict(
        **gates, numerical_fields_bitwise_identical=['scores', 'target', 'endpoint', 'FP_step_hashes'],
        run_ids=runs, preflight_also_identical=True,
        source_receipts=[dict(path=str(A / 'runs' / name / 'generated_capture/GENERATED_CODE_RECEIPT.json'),
                              sha256=sha(A / 'runs' / name / 'generated_capture/GENERATED_CODE_RECEIPT.json'))
                         for name in ['preflight_01', *runs]],
        parity_receipt_inventory=[dict(path=str(A / 'runs' / name / 'qualification/smoke/basis_and_jprop_parity.json'),
                                      sha256=sha(A / 'runs' / name / 'qualification/smoke/basis_and_jprop_parity.json'))
                                  for name in ['preflight_01', *runs]],
        original_scientific_gate_aggregation='NOT_PERFORMED_BY_THIS_SUPERVISOR',
        FORMAL_MEASUREMENTS=0, GPU6_FORMAL_AUTHORIZED='NO'))
    manifest = {key: gates[key] for key in [
        'modules', 'GENERATED_EXECUTION_FINGERPRINT_SEQUENCE', 'multiplicities']}
    emit(A / 'generated_executable_manifest.json', manifest)
    frozen_sha = sha(A / 'generated_executable_manifest.json')
    emit(A / 'GENERATED_CAPSULE_FREEZE.json', dict(
        GENERATED_EXECUTABLE_CAPSULE_ID=frozen_sha,
        fresh_process_comparison_sha256=sha(A / 'GENERATED_FRESH_PROCESS_COMPARISON.json'),
        scope='generated executable identity only, not V3 qualification or premeasurement',
        manifest_expansion_allowed=False, FORMAL_MEASUREMENTS=0, GPU6_FORMAL_AUTHORIZED='NO'))
    run_child('strict_verify_01', 'verify')
    assert sha(A / 'generated_executable_manifest.json') == frozen_sha
    emit(A / 'GENERATED_STRICT_VERIFY_COMPLETION.json', dict(
        GENERATED_STRICT_VERIFY_GATE='PASS', manifest_byte_identity='PASS',
        manifest_sha256=frozen_sha, original_scientific_gates='NOT_AGGREGATED',
        FORMAL_MEASUREMENTS=0, GPU6_FORMAL_AUTHORIZED='NO'))
    print('GENERATED_CODE_QUALIFICATION_PASS_NOT_FORMAL_AUTHORIZATION', flush=True)

if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        if not (A / 'STOP.json').exists():
            emit(A / 'STOP.json', dict(reason=repr(error), traceback=traceback.format_exc(),
                phase='GENERATED_CODE_FRESH_PROCESS_AND_STRICT_VERIFY',
                FINAL_ACTION='FAIL_AND_STOP', FORMAL_MEASUREMENTS=0,
                DISCOVERY_PRIMARY_COMPLETE='0/64', DISCOVERY_SECONDARY_COMPLETE='0/64',
                GPU6_FORMAL_AUTHORIZED='NO'))
        raise
