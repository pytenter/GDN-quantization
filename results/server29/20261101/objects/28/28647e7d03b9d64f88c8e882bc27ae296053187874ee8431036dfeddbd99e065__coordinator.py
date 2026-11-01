"""Single-attempt, single-device supervisor. No CUDA imports or automatic retries."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
PYTHON = '/data/ydai/miniconda3/envs/bitdecode/bin/python'
GPU_UUID = 'GPU-a78176ed-f97f-c54f-6201-5a01b570f422'
TASK_ID = 'QWEN_GDN_HADAMARD_DUAL_AXIS_DIRECT_32K_PILOT_V1'
PHASES = ['canary_initial', 'canary_fresh'] + [f'science_doc{d}_{c}' for d in (1, 2) for c in ('FP_NATIVE', 'H_C128', 'H_DUAL_M')]
SCIENCE_PHASES = PHASES[2:]
MAX_GPU_SECONDS = 24.0 * 3600
EXIT_RESERVE_SECONDS = 300.0
SCIENCE_RESERVE_SECONDS = 900.0
MAX_ARCHIVE_BYTES = 10 * 1024**3
MIN_FREE_DISK_BYTES = 1024**3
SAMPLE_SECONDS = 0.5
WATCHDOG_SECONDS = 1800.0

class StopExecution(RuntimeError):
    def __init__(self, message, classification='FAIL_AND_STOP'):
        super().__init__(message)
        self.classification = classification

def require(condition, message, classification='FAIL_AND_STOP'):
    if not condition:
        raise StopExecution(message, classification)

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def write(relative, value, exclusive=False):
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if exclusive:
        with path.open('x') as f:
            f.write(data)
    else:
        temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
        with temp.open('x') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)

def nvml_probe():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.free,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True, timeout=10)
    rows = list(csv.reader(raw.splitlines()))
    matches = [r for r in rows if r and int(r[0]) == 0]
    require(len(matches) == 1, 'GPU0 is absent or ambiguous')
    g = [x.strip() for x in matches[0]]
    appraw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory', '--format=csv,noheader,nounits'], text=True, timeout=10)
    apps = []
    for row in csv.reader(appraw.splitlines()):
        if row and row[0].strip() == GPU_UUID:
            require(len(row) >= 4, 'Malformed NVIDIA process inventory')
            apps.append({'gpu_uuid': row[0].strip(), 'pid': int(row[1]), 'process_name': row[2].strip(), 'used_gpu_memory_MiB': row[3].strip()})
    return {'status': 'PASS', 'unix': time.time(), 'physical_id': int(g[0]), 'uuid': g[1], 'name': g[2], 'total_MiB': int(g[3]), 'free_MiB': int(g[4]), 'used_MiB': int(g[5]), 'utilization_percent': int(g[6]), 'apps': apps}

def safe_probe():
    try:
        return nvml_probe()
    except BaseException as exc:
        return {'status': 'QUERY_FAILED', 'unix': time.time(), 'error': repr(exc)}

def hardware_identity(probe):
    require(probe.get('status') == 'PASS' and probe['physical_id'] == 0 and probe['uuid'] == GPU_UUID and 'RTX 3090' in probe['name'] and probe['total_MiB'] == 24576, 'Frozen GPU0 hardware/UUID identity mismatch; automatic device switch forbidden')

def idle_admission():
    p = nvml_probe()
    hardware_identity(p)
    require(not p['apps'] and p['free_MiB'] >= 23000 and p['utilization_percent'] <= 1, 'GPU0 is not truly idle with >=23000MiB free; stop without switching card', 'BLOCKED_AND_STOP')
    return p

def artifact_usage():
    total = 0
    for directory, dirs, files in os.walk(ROOT, followlinks=False):
        dirs[:] = [d for d in dirs if not (Path(directory) / d).is_symlink()]
        for name in files:
            p = Path(directory) / name
            if p.is_symlink():
                # Symlink targets are external inherited assets, not newly archived bytes.
                total += p.lstat().st_size
            else:
                total += p.stat().st_size
    v = os.statvfs(ROOT)
    return {'new_archive_bytes': total, 'free_disk_bytes': v.f_bavail * v.f_frsize}

def storage_gate():
    usage = artifact_usage()
    require(usage['new_archive_bytes'] < MAX_ARCHIVE_BYTES, 'New artifact archive reached 10GiB cap', 'BLOCKED_AND_STOP')
    require(usage['free_disk_bytes'] >= MIN_FREE_DISK_BYTES, 'Filesystem free space below 1GiB reserve', 'BLOCKED_AND_STOP')
    return usage

def source_check(manifest):
    entries = manifest.get('execution_sources')
    require(isinstance(entries, list) and bool(entries), 'Frozen execution_sources is absent')
    paths = set()
    for entry in entries:
        relative = Path(entry['path'])
        require(not relative.is_absolute() and '..' not in relative.parts, 'execution source must be within the new experiment root')
        p = ROOT / relative
        require(p.is_file() and sha(p) == entry['sha256'], f'Frozen source identity changed: {relative}')
        paths.add(str(relative))
    require({'src/coordinator.py', 'src/worker.py', 'preregistration.json', 'frozen_documents.json'} <= paths, 'Source manifest must freeze coordinator, worker, preregistration and documents')

def frozen_document_check():
    m = read(ROOT / 'frozen_documents.json')
    require(m.get('DOCS_FROZEN') == 2 and str(m.get('status', '')).startswith('PASS'), 'Two prospective documents not frozen')
    require(len(m.get('selected_documents', [])) == 2, 'Document roster size differs from two')
    for d in m['selected_documents']:
        require(d['full_token_count'] >= 32769 and d['consumed_tokens'] == 32768, 'Frozen scientific input length mismatch')
        for name in ('full_token_ids', 'input_ids_32769', 'input_ids_32768'):
            a = d['artifacts'][name]
            p = ROOT / a['path']
            require(p.is_file() and sha(p) == a['file_sha256'], f'Frozen token file changed: {p}')
        require(sha(ROOT / d['raw_text_copy']) == d['raw_text_sha256'], 'Frozen raw context changed')
    verification = read(ROOT / 'data_selection/FROZEN_DOCUMENT_VERIFICATION.json')
    require(str(verification.get('status', '')).startswith('PASS') and verification['manifest_sha256'] == sha(ROOT / 'frozen_documents.json') and verification['CUDA_initialized'] is False, 'Fresh CPU document identity prerequisite missing or stale')

def process_identity(pid):
    proc = Path('/proc') / str(pid)
    stat = (proc / 'stat').read_text()
    fields = stat[stat.rfind(')') + 2:].split()
    cmdline = [x.decode(errors='replace') for x in (proc / 'cmdline').read_bytes().split(b'\0') if x]
    return {'pid': pid, 'start_time_ticks': fields[19], 'cmdline': cmdline}

def exact_child(identity, phase):
    return identity['pid'] > 0 and str(ROOT / 'src/worker.py') in identity['cmdline'] and phase == identity['cmdline'][-1]

def terminate_own_child(child, recorded, phase):
    result = {'termination_requested': False, 'exact_PID_only': True}
    if child.poll() is not None:
        return result
    live = process_identity(child.pid)
    require(recorded is not None and live['start_time_ticks'] == recorded['start_time_ticks'] and exact_child(live, phase), 'Cannot confirm exact own child before termination; no broad process kill permitted')
    child.send_signal(signal.SIGTERM)
    result['termination_requested'] = True
    try:
        child.wait(timeout=20)
    except subprocess.TimeoutExpired:
        live = process_identity(child.pid)
        require(live['start_time_ticks'] == recorded['start_time_ticks'] and exact_child(live, phase), 'Own PID identity changed before forced cleanup')
        child.send_signal(signal.SIGKILL)
        child.wait(timeout=20)
        result['SIGKILL_required'] = True
    return result

def wait_for_release(pid):
    deadline = time.monotonic() + 30
    p = safe_probe()
    while time.monotonic() < deadline:
        if p['status'] == 'PASS' and not any(a['pid'] == pid for a in p['apps']):
            return True, p
        time.sleep(0.5)
        p = safe_probe()
    return False, p

def positive_number(value, label):
    require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0, f'Invalid positive budget projection: {label}')
    return float(value)

def remaining_science_projection(g4, unfinished, ledger):
    projected = positive_number(g4.get('projected_remaining_GPU_seconds'), 'projected_remaining_GPU_seconds')
    per_phase = g4.get('phase_predictions_seconds', {})
    if isinstance(per_phase, dict) and all(p in per_phase for p in SCIENCE_PHASES):
        values = {p: positive_number(per_phase[p], p) for p in SCIENCE_PHASES}
        # Respect any additional conservative aggregate overhead above the explicit sum.
        overhead = max(0.0, projected - sum(values.values()))
        return sum(values[p] for p in unfinished) + overhead, 'explicit_phase_predictions_plus_aggregate_overhead'
    # Only aggregate G4 is required by the worker interface. Use the conservative aggregate
    # fraction, raised if completed science phases reveal a slower per-trajectory duration.
    observed = [x['reservation_seconds'] for x in ledger['phases'] if x['phase'] in SCIENCE_PHASES]
    each = max([projected / 6.0] + [x * 1.25 for x in observed])
    return each * len(unfinished), 'aggregate_six_trajectory_projection_with_observed_1.25x_floor'

def main():
    if (ROOT / 'COORDINATOR_STARTED.json').exists() or (ROOT / 'GATE_STATUS_FINAL.json').exists():
        print(json.dumps({'status': 'REFUSED_EXISTING_ATTEMPT', 'no_GPU_dispatch': True}), flush=True)
        return 2
    (ROOT / 'logs').mkdir(parents=True, exist_ok=True)
    (ROOT / 'analysis').mkdir(parents=True, exist_ok=True)
    gates = {p: 'NOT_RUN' for p in PHASES}
    ledger = {'task_id': TASK_ID, 'GPU_HOURS_USED': 0.0, 'MAX_GPU_HOURS': 24.0, 'old_task_GPU_hours_included': False, 'GPU_UUID': GPU_UUID, 'physical_GPU_ID': 0, 'phases': [], 'NVML_target_sampling_seconds': SAMPLE_SECONDS, 'accounting': 'Exclusive reservation begins at child dispatch and includes model load, all computation/IO, and wait until an actual NVML observation excludes own worker PID. Query failure never counts as verified release. This new task begins at zero; no attempt reset.', 'started_unix': time.time()}
    failure = None
    frozen = None
    manifest_sha = None
    g4 = None
    g4_sha = None
    g4_path = None
    phase = None
    last_accounted_monotonic = None
    def termination_signal(signum, frame):
        raise StopExecution(f'Coordinator received signal {signum}; stopping own worker without retry')
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, termination_signal)
    try:
        require(not (ROOT / 'FIRST_FAILURE.json').exists(), 'Existing first failure forbids later GPU phases')
        prereg = read(ROOT / 'preregistration.json')
        authorization = prereg.get('GPU_EXECUTION_AUTHORIZED', prereg.get('GPU_AUTHORIZATION'))
        require(authorization == 'YES', 'Explicit GPU execution authorization missing', 'BLOCKED_AND_STOP')
        frozen = read(ROOT / 'source_manifest.json')
        manifest_sha = sha(ROOT / 'source_manifest.json')
        source_check(frozen)
        for relative in ['qualification_cpu/ENGINEERING_ADAPTER_CPU.json', 'qualification_cpu/CPU_STRUCTURE_001.json', 'qualification_cpu/MODEL_CONTENT_IDENTITY.json']:
            require(read(ROOT / relative).get('status') == 'PASS', f'Mandatory CPU prerequisite has not passed: {relative}', 'BLOCKED_AND_STOP')
        frozen_document_check()
        storage_gate()
        try:
            write('COORDINATOR_STARTED.json', {'task_id': TASK_ID, 'pid': os.getpid(), 'unix': time.time(), 'phases': PHASES, 'source_manifest_sha256': manifest_sha, 'single_attempt_no_retries': True}, exclusive=True)
        except FileExistsError:
            print(json.dumps({'status': 'REFUSED_SIMULTANEOUS_COORDINATOR', 'no_GPU_dispatch': True}), flush=True)
            return 2
        admission = idle_admission()
        write('gpu_admission.json', {'status': 'PASS', 'probe': admission, 'GPU_UUID': GPU_UUID, 'physical_GPU_ID': 0, 'CUDA_mapping': 'CUDA_VISIBLE_DEVICES=0 -> cuda:0', 'MAX_GPU_COUNT': 1, 'MAX_CONCURRENT_TRAJECTORIES': 1, 'MAX_GPU_HOURS': 24.0}, exclusive=True)
        for index, phase in enumerate(PHASES):
            require(not (ROOT / 'FIRST_FAILURE.json').exists(), 'First failure exists; no later GPU dispatch')
            require(sha(ROOT / 'source_manifest.json') == manifest_sha, 'Source manifest changed after freeze')
            source_check(frozen)
            frozen_document_check()
            require(not (ROOT / f'analysis/RECEIPT_{phase}.json').exists(), f'Phase already has receipt; restart forbidden: {phase}')
            remaining = MAX_GPU_SECONDS - ledger['GPU_HOURS_USED'] * 3600
            prediction = None
            forecast_method = None
            if phase in SCIENCE_PHASES:
                require(g4 is not None and g4_path is not None and sha(g4_path) == g4_sha, 'Qualified runtime budget absent or changed')
                unfinished = [p for p in SCIENCE_PHASES if gates[p] != 'PASS']
                prediction, forecast_method = remaining_science_projection(g4, unfinished, ledger)
                require(prediction <= remaining - SCIENCE_RESERVE_SECONDS, 'Conservative projection cannot complete all remaining paired scientific trajectories inside 24GPUh with 900s reserve', 'BLOCKED_AND_STOP')
            else:
                require(remaining > EXIT_RESERVE_SECONDS + 900, 'Insufficient budget for qualification and safe release', 'BLOCKED_AND_STOP')
            storage = storage_gate()
            probe = idle_admission()
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false', TRANSFORMERS_OFFLINE='1', HF_HUB_OFFLINE='1')
            command = [PYTHON, '-u', str(ROOT / 'src/worker.py'), phase]
            child = None
            recorded_identity = None
            phase_error = None
            cleanup_error = None
            log = None
            start = time.monotonic()
            entry = {'phase': phase, 'dispatch_unix': time.time(), 'command': command, 'budget_remaining_before_seconds': remaining, 'remaining_science_prediction_seconds': prediction, 'forecast_method': forecast_method, 'storage_before': storage, 'peak_sampled_NVML_MiB': probe['used_MiB'], 'NVML_sample_count': 0, 'coarse_NVML_samples': [], 'released': False}
            try:
                log = (ROOT / f'logs/{phase}.log').open('x')
                child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                entry['pid'] = child.pid
                recorded_identity = process_identity(child.pid)
                require(exact_child(recorded_identity, phase), 'Dispatched child cmdline does not match exact worker phase')
                entry['process_identity_at_dispatch'] = recorded_identity
                write(f'analysis/PROCESS_{phase}.json', recorded_identity, exclusive=True)
                last_progress = start
                last_log_size = -1
                last_storage = start
                last_live = start
                last_coarse = start - 30
                while child.poll() is None:
                    time.sleep(SAMPLE_SECONDS)
                    probe = nvml_probe()
                    hardware_identity(probe)
                    elapsed = time.monotonic() - start
                    entry['NVML_sample_count'] += 1
                    entry['peak_sampled_NVML_MiB'] = max(entry['peak_sampled_NVML_MiB'], probe['used_MiB'])
                    require(all(a['pid'] == child.pid for a in probe['apps']), 'Foreign process appeared on reserved GPU0; stopping only own exact worker PID')
                    require(remaining - elapsed > EXIT_RESERVE_SECONDS, '24GPUh deadline reached 300-second safe-stop reserve', 'BLOCKED_AND_STOP')
                    require(not (ROOT / 'FIRST_FAILURE.json').exists(), 'Worker recorded FIRST_FAILURE; dependent phases prohibited')
                    size = (ROOT / f'logs/{phase}.log').stat().st_size
                    if size != last_log_size:
                        last_progress = time.monotonic()
                        last_log_size = size
                    require(time.monotonic() - last_progress < WATCHDOG_SECONDS, '30-minute log no-progress watchdog; stop without retry')
                    if time.monotonic() - last_storage >= 5:
                        entry['storage_latest'] = storage_gate()
                        last_storage = time.monotonic()
                    if time.monotonic() - last_coarse >= 30:
                        entry['coarse_NVML_samples'].append(probe)
                        last_coarse = time.monotonic()
                    if time.monotonic() - last_live >= 10:
                        write('analysis/GPU_RESOURCE_LEDGER_LIVE.json', dict(ledger, active_phase=phase, active_PID=child.pid, active_reserved_seconds=elapsed, active_cumulative_GPUh=ledger['GPU_HOURS_USED'] + elapsed / 3600, active_peak_NVML_MiB=entry['peak_sampled_NVML_MiB']))
                        last_live = time.monotonic()
                entry['returncode'] = child.wait()
            except BaseException as exc:
                phase_error = exc
                entry['phase_exception'] = repr(exc)
            finally:
                if child is not None:
                    try:
                        entry['cleanup'] = terminate_own_child(child, recorded_identity, phase)
                    except BaseException as exc:
                        cleanup_error = exc
                        entry['cleanup_error'] = repr(exc)
                    entry['returncode'] = child.poll()
                    released, release_probe = wait_for_release(child.pid)
                else:
                    release_probe = safe_probe()
                    released = release_probe['status'] == 'PASS' and not release_probe['apps']
                elapsed = time.monotonic() - start
                entry.update(reservation_seconds=elapsed, GPU_hours=elapsed / 3600, release_observation_unix=time.time(), release_probe=release_probe, released=released, reservation_duration_is_lower_bound_if_release_unverified=not released)
                if log is not None:
                    log.close()
                ledger['phases'].append(entry)
                ledger['GPU_HOURS_USED'] += elapsed / 3600
                last_accounted_monotonic = time.monotonic()
                write('analysis/GPU_RESOURCE_LEDGER_LIVE.json', ledger)
            require(entry['released'], 'Own GPU context release could not be verified by NVML; stop')
            require(cleanup_error is None, f'Exact-PID cleanup failed: {cleanup_error}')
            if phase_error is not None:
                raise phase_error
            require(entry['returncode'] == 0, f'Worker exited with code {entry["returncode"]}; no patch/retry')
            require(not (ROOT / 'FIRST_FAILURE.json').exists(), 'First failure exists after worker exit')
            receipt = read(ROOT / f'analysis/RECEIPT_{phase}.json')
            require(receipt.get('status') == 'PASS', 'Worker exited without a valid PASS receipt')
            if 'phase' in receipt:
                require(receipt['phase'] == phase, 'Worker receipt phase identity mismatch')
            require(sha(ROOT / 'source_manifest.json') == manifest_sha, 'Source manifest changed during phase')
            source_check(frozen)
            storage_gate()
            gates[phase] = 'PASS'
            entry['receipt_sha256'] = sha(ROOT / f'analysis/RECEIPT_{phase}.json')
            print(json.dumps({'phase': phase, 'status': 'PASS', 'reservation_seconds': elapsed, 'cumulative_GPUh': ledger['GPU_HOURS_USED'], 'release_verified': True}), flush=True)
            if phase == 'canary_fresh':
                g4_path = ROOT / 'G4_RUNTIME_BUDGET.json'
                # Worker may place analysis outputs under analysis/; preserve one exact path.
                if not g4_path.exists() and (ROOT / 'analysis/G4_RUNTIME_BUDGET.json').exists():
                    g4_path = ROOT / 'analysis/G4_RUNTIME_BUDGET.json'
                g4 = read(g4_path)
                require(g4.get('status') == 'PASS', 'G4 measured-throughput projection is resource-blocked; 32K forbidden', 'BLOCKED_AND_STOP')
                g4_sha = sha(g4_path)
                # Preserve the worker's actual output path; never copy or rewrite its receipt.
                prediction, method = remaining_science_projection(g4, SCIENCE_PHASES, ledger)
                require(prediction <= MAX_GPU_SECONDS - ledger['GPU_HOURS_USED'] * 3600 - SCIENCE_RESERVE_SECONDS, 'G4 forecast lacks budget for both complete documents plus 900-second reserve; do not start 32K', 'BLOCKED_AND_STOP')
        final_status = 'COMPLETE_AND_STOP'
    except BaseException as exc:
        if phase is not None and gates[phase] != 'PASS':
            gates[phase] = 'FAIL' if any(e['phase'] == phase for e in ledger['phases']) else 'NOT_RUN'
        final_status = getattr(exc, 'classification', 'FAIL_AND_STOP')
        failure = {'status': final_status, 'phase': phase, 'error': repr(exc), 'traceback': traceback.format_exc(), 'unix': time.time(), 'no_GPU_retry': True, 'no_later_GPU_work': True}
        if not (ROOT / 'FIRST_FAILURE.json').exists():
            write('FIRST_FAILURE.json', failure, exclusive=True)
    final_probe = safe_probe()
    # An unverified release remains a potentially occupied reservation through this
    # final observation, including cleanup and reporting time; never undercount it.
    if ledger['phases'] and not ledger['phases'][-1]['released'] and last_accounted_monotonic is not None:
        e = ledger['phases'][-1]
        extension = max(0.0, time.monotonic() - last_accounted_monotonic)
        e['reservation_seconds'] += extension
        e['GPU_hours'] += extension / 3600
        ledger['GPU_HOURS_USED'] += extension / 3600
        e['final_release_probe'] = final_probe
        if final_probe['status'] == 'PASS' and 'pid' in e and not any(a['pid'] == e['pid'] for a in final_probe['apps']):
            e.update(released=True, release_confirmed_late=True, release_observation_unix=time.time(), reservation_duration_is_lower_bound_if_release_unverified=False)
    own_pids = [e['pid'] for e in ledger['phases'] if 'pid' in e]
    release_verified = final_probe['status'] == 'PASS' and not any(a['pid'] in own_pids for a in final_probe['apps'])
    ledger.update(final_status=final_status, final_GPU_release_verified=release_verified, final_GPU_release_probe=final_probe, ended_unix=time.time())
    write('gpu_resource_ledger.json', ledger, exclusive=True)
    write('GATE_STATUS_FINAL.json', {'task_id': TASK_ID, 'status': final_status, 'gates': gates, 'coordinator_failure': failure, 'first_failure_file': 'FIRST_FAILURE.json' if (ROOT / 'FIRST_FAILURE.json').exists() else None, 'GPU_HOURS_USED': ledger['GPU_HOURS_USED'], 'MAX_GPU_HOURS': 24.0, 'GPU_release_verified': release_verified, 'GPU_release_query': final_probe, 'no_further_GPU_work': True}, exclusive=True)
    print(json.dumps({'status': final_status, 'gates': gates, 'GPU_HOURS_USED': ledger['GPU_HOURS_USED'], 'GPU_release_verified': release_verified}), flush=True)
    return 0 if final_status == 'COMPLETE_AND_STOP' else 1

if __name__ == '__main__':
    sys.exit(main())
