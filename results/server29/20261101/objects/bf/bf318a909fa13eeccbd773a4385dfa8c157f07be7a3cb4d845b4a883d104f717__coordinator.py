"""New five-document control only; unchanged worker/core/codec/reducer.

Run once from an actual remote tmux pane after CPU preparation/freeze. All GPU
work is delegated to control/worker_entry.py. No automatic retry or resumption.
"""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import time
import traceback

TASK = 'QWEN_GDN_DUAL_AXIS_PROSPECTIVE_REPLICATION_5DOC_V1'
ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ('FP_NATIVE', 'H_C128', 'H_DUAL_M')
SCIENCE_PHASES = tuple(f'science_doc{d}_{c}' for d in range(1, 6) for c in CONDITIONS)
PHASES = ('canary_initial', 'canary_fresh') + SCIENCE_PHASES
HELPER_SHA = '28647e7d03b9d64f88c8e882bc27ae296053187874ee8431036dfeddbd99e065'
CORE_SHAS = {
    'src/codec.py': 'ece859b901145c6d9d646b8befd01916037c9117ef4a80f7b801f27f79f8c23f',
    'src/core.py': 'aa338b8b83cde62b97c94fd205d1c62999f783e7ec3242cbb813dc877e01f575',
    'src/worker.py': 'e04ec85092ee4f2207699366b2aaa61e1e71038e728c9234477b5b2c21794773',
    'src/reducer.py': '7e94c95803acb7ae47f0e3ec44fa97faf7cc8d93c4ecc5791a56ea2cf305b84f',
}
SCORE_DIR = Path('/tmp/qwen_5doc_scores_20261016')


def need(ok, why):
    if not ok:
        raise ValueError(why)


def import_helpers(root):
    path = root / 'src/coordinator.py'
    import hashlib
    need(hashlib.sha256(path.read_bytes()).hexdigest() == HELPER_SHA,
         'Qualified coordinator helper bytes changed')
    spec = importlib.util.spec_from_file_location('qualified_control_helpers_5doc', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT, module.TASK_ID = root, TASK
    module.PHASES, module.SCIENCE_PHASES = list(PHASES), list(SCIENCE_PHASES)
    module.exact_child = lambda identity, phase: exact_child(identity, phase, root)
    need(module.MAX_GPU_SECONDS == 24 * 3600 and module.SCIENCE_RESERVE_SECONDS == 900
         and module.EXIT_RESERVE_SECONDS == 300 and module.WATCHDOG_SECONDS == 1800,
         'Qualified safety limits changed')
    return module


def exact_child(identity, phase, root):
    command = identity.get('cmdline', [])
    return bool(identity.get('pid', 0) > 0 and phase in PHASES and len(command) == 4
                and command[0] == '/data/ydai/miniconda3/envs/bitdecode/bin/python'
                and command[1] == '-u' and command[2] == str(root / 'control/worker_entry.py')
                and command[3] == phase)


def resident_stdio(root):
    path = root / 'logs/COORDINATOR_STDOUT.log'
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    null = os.open('/dev/null', os.O_RDONLY)
    os.dup2(null, 0); os.dup2(fd, 1); os.dup2(fd, 2)
    os.close(null); os.close(fd)
    sys.stdout = open(1, 'w', buffering=1, closefd=False)
    sys.stderr = open(2, 'w', buffering=1, closefd=False)
    need(stat.S_ISREG(os.fstat(1).st_mode) and stat.S_ISREG(os.fstat(2).st_mode),
         'Supervisor stdout/stderr must be resident regular files')
    return str(path)


def tmux_residency():
    need(os.environ.get('TMUX') and os.environ.get('TMUX_PANE'), 'Actual remote tmux pane required')
    pane = os.environ['TMUX_PANE']
    raw = subprocess.check_output(['tmux', 'display-message', '-p', '-t', pane,
                                   '#{pane_pid} #{session_name}'], text=True, timeout=10).strip().split(maxsplit=1)
    fields = os.environ['TMUX'].rsplit(',', 2)
    need(len(raw) == 2 and raw[0].isdigit() and len(fields) == 3
         and fields[0] and fields[1].isdigit() and fields[2].isdigit(), 'Invalid live tmux identity')
    pane_pid, server_pid = int(raw[0]), int(fields[1])
    proc = Path('/proc') / str(server_pid)
    need(proc.stat().st_uid == os.getuid(), 'tmux server UID mismatch')
    comm = (proc / 'comm').read_text().strip()
    cmdline = [x.decode(errors='replace') for x in (proc / 'cmdline').read_bytes().split(b'\0') if x]
    executable = Path(shutil.which('tmux') or '')
    need(comm.startswith('tmux') and cmdline and Path(cmdline[0]).name.startswith('tmux')
         and executable.is_file() and os.path.samefile(proc / 'exe', executable), 'Actual tmux server executable mismatch')
    pid, chain = os.getpid(), []
    while pid > 1 and len(chain) < 64:
        chain.append(pid)
        if pid == server_pid:
            break
        text = (Path('/proc') / str(pid) / 'stat').read_text()
        pid = int(text[text.rfind(')') + 2:].split()[1])
    need(pane_pid in chain and server_pid in chain and chain.index(pane_pid) <= chain.index(server_pid),
         'Controller is not descended from the queried tmux pane and server')
    return dict(status='PASS', pane=pane, pane_PID=pane_pid, server_PID=server_pid,
                server_same_UID=True, server_executable_matches=True, session=raw[1],
                controller_PID=os.getpid(), controller_ancestor_chain=chain)


def emit(value, ledger):
    try:
        print(json.dumps(value, ensure_ascii=False), flush=True)
    except (BrokenPipeError, OSError) as exc:
        ledger.setdefault('informational_stdout_errors', []).append(
            dict(unix=time.time(), error=repr(exc), phase=value.get('phase'), scientific_PASS_preserved=True))


def frozen_documents(C, root):
    m = C.read(root / 'frozen_documents.json')
    C.require(m.get('DOCS_FROZEN') == 5 and str(m.get('status', '')).startswith('PASS')
              and len(m.get('selected_documents', [])) == 5, 'Exactly five prospective documents required')
    raw, ids, names = set(), set(), set()
    for i, d in enumerate(m['selected_documents'], 1):
        C.require(d['document_index'] == i and d['full_token_count'] >= 32769
                  and d['consumed_tokens'] == 32768, 'Frozen input index/length mismatch')
        raw.add(d['raw_text_sha256']); names.add(d['document_id'])
        for name in ('full_token_ids', 'input_ids_32769', 'input_ids_32768'):
            a = d['artifacts'][name]; relative = Path(a['path'])
            C.require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe frozen token path')
            path = root / relative
            C.require(path.is_file() and C.sha(path) == a['file_sha256'], 'Frozen token file changed: ' + str(relative))
        ids.add(d['artifacts']['input_ids_32768']['canonical_json_sha256'])
        relative = Path(d['raw_text_copy'])
        C.require(not relative.is_absolute() and '..' not in relative.parts
                  and C.sha(root / relative) == d['raw_text_sha256'], 'Frozen raw text changed')
    C.require(len(raw) == len(ids) == len(names) == 5, 'Frozen raw/input/document identities are duplicated')
    v = C.read(root / 'data_selection/FROZEN_DOCUMENT_VERIFICATION.json')
    C.require(str(v.get('status', '')).startswith('PASS') and v['manifest_sha256'] == C.sha(root / 'frozen_documents.json')
              and v['CUDA_initialized'] is False, 'Fresh CPU document identity verification missing or stale')
    return m


def positive(value, label):
    need(type(value) in (int, float) and math.isfinite(value) and value > 0, 'Invalid positive budget: ' + label)
    return float(value)


def verify_budget(C, root):
    budget = C.read(root / 'RESOURCE_BUDGET.json')
    C.require(budget.get('status') == 'PASS' and budget.get('task_id') == TASK
              and budget.get('GPU_UUID') == C.GPU_UUID and budget.get('MAX_NEW_GPU_HOURS') == 24.0,
              'New-task resource budget identity failed')
    predictions = budget['phase_predictions_seconds']
    C.require(set(predictions) == set(SCIENCE_PHASES), 'Budget must cover exactly fifteen science phases')
    for phase, value in predictions.items():
        positive(value, phase)
    qualification = positive(budget['qualification_prediction_seconds'], 'qualification_prediction_seconds')
    C.require(budget['safety_exit_reserve_seconds'] == 900, 'Frozen 900-second science reserve changed')
    C.require(len(budget['projection_basis_source_sha256']) == 64, 'Historical full32K cost binding missing')
    basis_path = Path(budget['projection_basis_source_path'])
    C.require(basis_path.is_absolute() and basis_path.is_file()
              and C.sha(basis_path) == budget['projection_basis_source_sha256'], 'Historical full32K resource ledger changed')
    basis = C.read(basis_path)
    C.require(basis['final_status'] == 'COMPLETE_AND_STOP' and basis['final_GPU_release_verified'] is True,
              'Historical cost basis is not a completed released execution')
    old = [row for row in basis['phases'] if row['phase'].startswith('science_doc')]
    C.require(len(old) == 6 and all(row['returncode'] == 0 and row['released'] is True for row in old),
              'Historical six complete science reservations absent')
    for condition in CONDITIONS:
        values = [positive(row['reservation_seconds'], row['phase']) for row in old
                  if row['phase'].split('_', 2)[2] == condition]
        C.require(len(values) == 2, 'Historical per-condition cost sample missing')
        expected = max(values) * 1.15
        C.require(all(predictions[p] == expected for p in SCIENCE_PHASES if p.endswith('_' + condition)),
                  'Frozen phase allowance must equal qualified same-condition worst reservation times1.15')
    storage = budget['storage']
    C.require(Path(storage['score_directory']) == SCORE_DIR, 'Fixed scores filesystem path changed')
    for key in ('score_file_upper_bound_bytes', 'metadata_checkpoint_peak_extra_bytes',
                'metadata_reserve_bytes', 'score_reserve_bytes'):
        positive(storage[key], key)
    C.require(storage['score_file_upper_bound_bytes'] >= 381536430
              and storage['metadata_reserve_bytes'] >= 1024**3 and storage['score_reserve_bytes'] >= 1024**3,
              'Storage bounds below actual qualified assets or original reserve')
    total = sum(predictions.values()) + qualification + 900
    C.require(total < C.MAX_GPU_SECONDS, 'Full five-document projection cannot fit24GPUh', 'BLOCKED_AND_STOP')
    return budget


def science_projection(budget, unfinished, ledger):
    predictions = dict(budget['phase_predictions_seconds'])
    observed = {}
    for row in ledger['phases']:
        if row['phase'] in SCIENCE_PHASES:
            condition = row['phase'].split('_', 2)[2]
            observed[condition] = max(observed.get(condition, 0), row['reservation_seconds'])
    # The initial predictions already include the preregistered 15% margin.
    # Observed slower same-condition full32K reservations can only raise them.
    values = {p: max(predictions[p], observed.get(p.split('_', 2)[2], 0)) for p in unfinished}
    return sum(values.values()), dict(method='frozen15_predictions_with_monotone_same_condition_observed_floor',
                                     remaining_phase_predictions_seconds=values,
                                     observed_same_condition_max_seconds=observed, prediction_downward_updates=False)


def tree_bytes(root):
    total = 0
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if not (Path(directory) / d).is_symlink()]
        for name in files:
            path = Path(directory) / name
            try:
                total += path.lstat().st_size if path.is_symlink() else path.stat().st_size
            except FileNotFoundError:
                # An atomic CURRENT.tmp -> CURRENT.pt replacement can retire a
                # directory entry during the independent disk inventory.
                continue
    return total


def free_bytes(root):
    s = os.statvfs(root)
    return s.f_bavail * s.f_frsize


def storage_check(C, root, budget):
    cfg = budget['storage']; link = root / 'scores'
    C.require(link.is_symlink() and link.resolve() == SCORE_DIR and SCORE_DIR.is_dir()
              and not SCORE_DIR.is_symlink(), 'Scores must use the fixed separate /tmp directory')
    C.require(root.stat().st_dev != SCORE_DIR.stat().st_dev, 'Metadata and scores must use qualified separate filesystems')
    score_bytes = tree_bytes(SCORE_DIR); metadata_bytes = tree_bytes(root)
    # Existing rolling files and score files already occupy part of the frozen
    # total capacity; account for that occupancy exactly once, including temp IO.
    cp_bytes = 0
    if (root / 'checkpoints').exists():
        for p in (root / 'checkpoints').glob('CURRENT*'):
            try:
                if p.is_file():
                    cp_bytes += p.stat().st_size
            except FileNotFoundError:
                continue
    metadata_small = max(0, metadata_bytes - cp_bytes)
    small_allowance = 512 * 1024**2
    checkpoint_peak = cfg['metadata_checkpoint_peak_extra_bytes'] - small_allowance
    C.require(checkpoint_peak >= 2 * 1507209134, 'Rolling checkpoint capacity bound is too small')
    meta_remaining = max(0, checkpoint_peak - cp_bytes) + max(0, small_allowance - metadata_small)
    score_total = 15 * cfg['score_file_upper_bound_bytes'] + 64 * 1024**2
    score_remaining = max(0, score_total - score_bytes)
    mf, sf = free_bytes(root), free_bytes(SCORE_DIR)
    C.require(mf >= meta_remaining + cfg['metadata_reserve_bytes'], 'Metadata/checkpoint filesystem lacks frozen peak+reserve', 'BLOCKED_AND_STOP')
    C.require(sf >= score_remaining + cfg['score_reserve_bytes'], 'Score filesystem lacks all15 full-vocab assets+reserve', 'BLOCKED_AND_STOP')
    C.require(metadata_bytes + score_bytes < C.MAX_ARCHIVE_BYTES, 'Combined new metadata/tensor archive reaches original10GiB cap', 'BLOCKED_AND_STOP')
    return dict(metadata_bytes=metadata_bytes, score_bytes=score_bytes, current_and_tmp_checkpoint_bytes=cp_bytes,
                metadata_free_bytes=mf, score_free_bytes=sf, metadata_remaining_peak_bytes=meta_remaining,
                score_remaining_bytes=score_remaining, separate_filesystems=True,
                score_directory=str(SCORE_DIR), combined_new_archive_bytes=metadata_bytes + score_bytes)


def preflight(C, root):
    C.require(not (root / 'FIRST_FAILURE.json').exists(), 'Prior first failure forbids any GPU dispatch')
    prereg = C.read(root / 'preregistration.json')
    C.require(prereg['TASK_ID'] == TASK and prereg.get('GPU_EXECUTION_AUTHORIZED') == 'YES', 'Explicit task GPU authorization absent')
    C.require(prereg['score_positions'] == list(range(8192, 32768, 64)), 'Score position convention changed')
    manifest = C.read(root / 'source_manifest.json')
    C.require(manifest['task'] == TASK and manifest.get('frozen_before_any_new_GPU_forward') is True, 'Wrong frozen new task source manifest')
    C.source_check(manifest)
    paths = {v['path'] for v in manifest['execution_sources']}
    C.require({'control/coordinator.py', 'control/worker_entry.py', 'RESOURCE_BUDGET.json'} <= paths,
              'Frozen manifest must include the new control and budget')
    for rel, digest in CORE_SHAS.items():
        C.require(C.sha(root / rel) == digest, 'Qualified scientific source bytes changed: ' + rel)
    for rel in ('qualification_cpu/ENGINEERING_ADAPTER_CPU.json', 'qualification_cpu/CPU_STRUCTURE_001.json',
                'qualification_cpu/MODEL_CONTENT_IDENTITY.json'):
        C.require(C.read(root / rel).get('status') == 'PASS', 'Mandatory CPU source/model prerequisite not PASS: ' + rel)
    C.require(not any((root / f'analysis/RECEIPT_{phase}.json').exists() for phase in PHASES),
              'A new independent task cannot inherit completed worker receipts')
    C.require(not (root / 'checkpoints/CURRENT.pt').exists() and not (root / 'checkpoints/CATALOG.jsonl').exists(),
              'A new independent task cannot inherit a prefix checkpoint/catalog')
    C.require(SCORE_DIR.is_dir() and not any(SCORE_DIR.iterdir()),
              'The new task score directory must initially be empty; never mix previous trajectories')
    frozen_documents(C, root)
    C.require(prereg['documents_manifest_sha256'] == C.sha(root / 'frozen_documents.json'), 'Preregistered roster binding changed')
    return manifest, verify_budget(C, root)


def check_fresh_resource(C, root):
    g4 = C.read(root / 'G4_RUNTIME_BUDGET.json')
    # Its six-trajectory time projection is a frozen-worker diagnostic only.
    # Preserve and enforce its original actual-device memory bound without edits.
    C.require(positive(g4['projected_peak_reserved_bytes'], 'projected_peak_reserved_bytes')
              < positive(g4['admissible_memory_bytes'], 'admissible_memory_bytes'),
              'Original canary memory admission failed', 'BLOCKED_AND_STOP')
    C.require(g4['safety_exit_reserve_seconds'] == 900, 'Original canary memory receipt identity changed')
    reducer = C.read(root / 'analysis/CANARY_REDUCER.json')
    C.require(reducer['status'] == 'PASS', 'Fresh canary reducer failed')
    return dict(status='PASS', original_worker_G4_sha256=C.sha(root / 'G4_RUNTIME_BUDGET.json'),
                six_trajectory_time_projection_scope='CANARY_ONLY_DIAGNOSTIC_NOT_15_PHASE_ADMISSION',
                original_memory_admission_enforced=True, old_G4_status=g4['status'])


def run(root):
    root = Path(root).resolve()
    need(root.name == TASK and root.is_dir(), 'Wrong independent task root')
    C = import_helpers(root)
    if any((root / name).exists() for name in ('COORDINATOR_STARTED.json', 'CONTROL_RUN_GUARD.json', 'GATE_STATUS_FINAL.json')):
        return dict(status='REFUSED_EXISTING_ATTEMPT', no_GPU_dispatch=True)
    logfile = resident_stdio(root)
    ledger = dict(task_id=TASK, GPU_HOURS_USED=0.0, MAX_GPU_HOURS=24.0, old_task_GPU_hours_included=False,
                  GPU_UUID=C.GPU_UUID, physical_GPU_ID=0, phases=[], started_unix=time.time(),
                  NVML_target_sampling_seconds=C.SAMPLE_SECONDS, supervisor_stdio_file=logfile,
                  accounting='Exclusive child dispatch through independently verified GPU release; includes load/compute/IO/checkpoints; new task zero, no attempt reset')
    gates = dict.fromkeys(PHASES, 'NOT_RUN')
    phase, failure, last_accounted = None, None, None
    manifest_sha, budget_sha, fresh_resource = None, None, None
    def termination(signum, frame):
        raise C.StopExecution('Coordinator signal ' + str(signum) + '; no GPU retry')
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, termination)
    try:
        manifest, budget = preflight(C, root)
        manifest_sha, budget_sha = C.sha(root / 'source_manifest.json'), C.sha(root / 'RESOURCE_BUDGET.json')
        residency = tmux_residency()
        try:
            C.write('CONTROL_RUN_GUARD.json', dict(pid=os.getpid(), unix=time.time(), source_manifest_sha256=manifest_sha), exclusive=True)
        except FileExistsError:
            # Losing a simultaneous CPU launch race must not poison the owner
            # attempt by writing FIRST_FAILURE into its already active root.
            return dict(status='REFUSED_SIMULTANEOUS_COORDINATOR', no_GPU_dispatch=True)
        C.write('control/TMUX_RESIDENCY.json', residency, exclusive=True)
        storage_check(C, root, budget)
        C.write('COORDINATOR_STARTED.json', dict(task_id=TASK, pid=os.getpid(), unix=time.time(), phases=list(PHASES),
                source_manifest_sha256=manifest_sha, resource_budget_sha256=budget_sha, single_attempt_no_retries=True), exclusive=True)
        admission = C.idle_admission()
        C.write('gpu_admission.json', dict(status='PASS', probe=admission, GPU_UUID=C.GPU_UUID,
                physical_GPU_ID=0, CUDA_mapping='CUDA_VISIBLE_DEVICES=0 -> cuda:0', MAX_GPU_COUNT=1,
                MAX_CONCURRENT_TRAJECTORIES=1, MAX_GPU_HOURS=24.0), exclusive=True)
        for phase in PHASES:
            C.require(not (root / 'FIRST_FAILURE.json').exists(), 'First failure prohibits later dispatch')
            C.require(C.sha(root / 'source_manifest.json') == manifest_sha and C.sha(root / 'RESOURCE_BUDGET.json') == budget_sha,
                      'Frozen source or resource manifest changed')
            C.source_check(manifest); frozen_documents(C, root)
            C.require(not (root / ('analysis/RECEIPT_' + phase + '.json')).exists(), 'Existing phase receipt prohibits restart')
            remaining = C.MAX_GPU_SECONDS - ledger['GPU_HOURS_USED'] * 3600
            forecast = None
            if phase in SCIENCE_PHASES:
                C.require(gates['canary_initial'] == gates['canary_fresh'] == 'PASS' and fresh_resource is not None,
                          'Both new-task canary gates and memory admission must PASS before32K')
                forecast, method = science_projection(budget, [p for p in SCIENCE_PHASES if gates[p] != 'PASS'], ledger)
                C.require(forecast <= remaining - 900, 'All remaining15-phase forecast exceeds24GPUh with900s reserve', 'BLOCKED_AND_STOP')
            else:
                method = dict(method='new_task_actual_qualification_plus_frozen15_phase_prediction')
                C.require(remaining > 1200, 'Insufficient qualification/release reserve', 'BLOCKED_AND_STOP')
            storage = storage_check(C, root, budget); probe = C.idle_admission()
            command = [C.PYTHON, '-u', str(root / 'control/worker_entry.py'), phase]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                       OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
                       TRANSFORMERS_OFFLINE='1', HF_HUB_OFFLINE='1')
            child = recorded = log = None
            phase_error = cleanup_error = None
            start = time.monotonic()
            entry = dict(phase=phase, dispatch_unix=time.time(), command=command, budget_remaining_before_seconds=remaining,
                         remaining_science_prediction_seconds=forecast, forecast_method=method, storage_before=storage,
                         peak_sampled_NVML_MiB=probe['used_MiB'], NVML_sample_count=0, coarse_NVML_samples=[], released=False)
            try:
                log = (root / f'logs/{phase}.log').open('x')
                child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                entry['pid'] = child.pid; recorded = C.process_identity(child.pid)
                C.require(C.exact_child(recorded, phase), 'Exact worker_entry command identity failed')
                entry['process_identity_at_dispatch'] = recorded
                C.write(f'analysis/PROCESS_{phase}.json', recorded, exclusive=True)
                last_progress = last_storage = last_live = start
                last_log_size, last_coarse = -1, start - 30
                while child.poll() is None:
                    time.sleep(C.SAMPLE_SECONDS)
                    probe = C.nvml_probe(); C.hardware_identity(probe)
                    elapsed = time.monotonic() - start
                    entry['NVML_sample_count'] += 1
                    entry['peak_sampled_NVML_MiB'] = max(entry['peak_sampled_NVML_MiB'], probe['used_MiB'])
                    C.require(all(a['pid'] == child.pid for a in probe['apps']), 'Foreign process on reserved GPU0; stop own exact PID')
                    C.require(remaining - elapsed > 300, '24GPUh reached300s safe-stop reserve', 'BLOCKED_AND_STOP')
                    C.require(not (root / 'FIRST_FAILURE.json').exists(), 'Worker recorded FIRST_FAILURE')
                    size = (root / f'logs/{phase}.log').stat().st_size
                    if size != last_log_size:
                        last_progress, last_log_size = time.monotonic(), size
                    C.require(time.monotonic() - last_progress < 1800, '30-minute no-progress watchdog; no retry')
                    if time.monotonic() - last_storage >= 5:
                        entry['storage_latest'] = storage_check(C, root, budget); last_storage = time.monotonic()
                    if time.monotonic() - last_coarse >= 30:
                        entry['coarse_NVML_samples'].append(probe); last_coarse = time.monotonic()
                    if time.monotonic() - last_live >= 10:
                        C.write('analysis/GPU_RESOURCE_LEDGER_LIVE.json', dict(ledger, active_phase=phase,
                                active_PID=child.pid, active_reserved_seconds=elapsed,
                                active_cumulative_GPUh=ledger['GPU_HOURS_USED'] + elapsed / 3600))
                        last_live = time.monotonic()
                entry['returncode'] = child.wait()
            except BaseException as exc:
                phase_error = exc; entry['phase_exception'] = repr(exc)
            finally:
                if child is not None:
                    try:
                        entry['cleanup'] = C.terminate_own_child(child, recorded, phase)
                    except BaseException as exc:
                        cleanup_error = exc; entry['cleanup_error'] = repr(exc)
                    entry['returncode'] = child.poll()
                    released, release_probe = C.wait_for_release(child.pid)
                else:
                    release_probe = C.safe_probe(); released = release_probe['status'] == 'PASS' and not release_probe['apps']
                elapsed = time.monotonic() - start
                entry.update(reservation_seconds=elapsed, GPU_hours=elapsed / 3600, released=released,
                             release_probe=release_probe, release_observation_unix=time.time(),
                             reservation_duration_is_lower_bound_if_release_unverified=not released)
                if log is not None:
                    log.close()
                ledger['phases'].append(entry); ledger['GPU_HOURS_USED'] += elapsed / 3600
                last_accounted = time.monotonic()
                C.write('analysis/GPU_RESOURCE_LEDGER_LIVE.json', ledger)
            C.require(entry['released'], 'GPU release not verified')
            C.require(cleanup_error is None, 'Exact-PID cleanup failed: ' + str(cleanup_error))
            if phase_error is not None:
                raise phase_error
            C.require(entry['returncode'] == 0 and not (root / 'FIRST_FAILURE.json').exists(), 'Worker failed; no retry')
            receipt_path = root / f'analysis/RECEIPT_{phase}.json'; receipt = C.read(receipt_path)
            C.require(receipt.get('status') == receipt.get('result', {}).get('status') == 'PASS', 'Valid PASS receipt absent')
            if phase in SCIENCE_PHASES:
                parts = phase.split('_', 2)
                C.require(receipt['result'] == dict(status='PASS', condition=parts[2], doc=int(parts[1][3:]), consumed=32768, scores=384),
                          'Formal completion is not exactly32768/384 expected doc/condition')
            C.require(C.sha(root / 'source_manifest.json') == manifest_sha, 'Manifest changed during phase')
            C.source_check(manifest); storage_check(C, root, budget)
            gates[phase] = 'PASS'; entry['receipt_sha256'] = C.sha(receipt_path)
            C.write(f'analysis/PASS_{phase}.json', dict(status='PASS', phase=phase,
                    receipt_sha256=entry['receipt_sha256'], cumulative_GPUh=ledger['GPU_HOURS_USED']), exclusive=True)
            emit(dict(phase=phase, status='PASS', cumulative_GPUh=ledger['GPU_HOURS_USED']), ledger)
            if phase == 'canary_fresh':
                fresh_resource = check_fresh_resource(C, root)
                forecast, method = science_projection(budget, SCIENCE_PHASES, ledger)
                C.require(forecast + 900 < C.MAX_GPU_SECONDS - ledger['GPU_HOURS_USED'] * 3600,
                          'Actual qualification plus all15 predictions and900s cannot fit24GPUh', 'BLOCKED_AND_STOP')
                C.write('analysis/FIVE_DOC_RESOURCE_ADMISSION.json', dict(fresh_resource,
                        resource_budget_sha256=budget_sha, actual_qualification_GPUh=ledger['GPU_HOURS_USED'],
                        projected_science_seconds=forecast, safety_exit_reserve_seconds=900, forecast_method=method), exclusive=True)
        final_status = 'COMPLETE_AND_STOP'
    except BaseException as exc:
        if phase is not None and gates[phase] != 'PASS':
            gates[phase] = 'FAIL' if any(p['phase'] == phase for p in ledger['phases']) else 'NOT_RUN'
        final_status = getattr(exc, 'classification', 'FAIL_AND_STOP')
        failure = dict(status=final_status, phase=phase, error=repr(exc), traceback=traceback.format_exc(),
                       unix=time.time(), no_GPU_retry=True, no_later_GPU_work=True)
        if not (root / 'FIRST_FAILURE.json').exists():
            C.write('FIRST_FAILURE.json', failure, exclusive=True)
    final_probe = C.safe_probe()
    if ledger['phases'] and not ledger['phases'][-1]['released'] and last_accounted is not None:
        row = ledger['phases'][-1]; extension = max(0, time.monotonic() - last_accounted)
        row['reservation_seconds'] += extension; row['GPU_hours'] += extension / 3600
        ledger['GPU_HOURS_USED'] += extension / 3600; row['final_release_probe'] = final_probe
        if final_probe['status'] == 'PASS' and 'pid' in row and not any(a['pid'] == row['pid'] for a in final_probe['apps']):
            row.update(released=True, release_confirmed_late=True, release_observation_unix=time.time(),
                       reservation_duration_is_lower_bound_if_release_unverified=False)
    own_pids = {p['pid'] for p in ledger['phases'] if 'pid' in p}
    release = final_probe['status'] == 'PASS' and not any(a['pid'] in own_pids for a in final_probe['apps'])
    ledger.update(final_status=final_status, final_GPU_release_verified=release, final_GPU_release_probe=final_probe, ended_unix=time.time())
    C.write('gpu_resource_ledger.json', ledger, exclusive=True)
    result = dict(task_id=TASK, status=final_status, gates=gates, coordinator_failure=failure,
                  first_failure_file='FIRST_FAILURE.json' if (root / 'FIRST_FAILURE.json').exists() else None,
                  source_manifest_sha256=manifest_sha, resource_budget_sha256=budget_sha,
                  GPU_HOURS_USED=ledger['GPU_HOURS_USED'], MAX_GPU_HOURS=24.0,
                  GPU_release_verified=release, GPU_release_query=final_probe, no_further_GPU_work=True)
    C.write('GATE_STATUS_FINAL.json', result, exclusive=True)
    emit(result, ledger)
    return result


def self_test():
    assert len(PHASES) == 17 and len(SCIENCE_PHASES) == 15 and len(set(PHASES)) == 17
    budget = dict(phase_predictions_seconds={p: 5000.0 for p in SCIENCE_PHASES})
    p0, _ = science_projection(budget, SCIENCE_PHASES, dict(phases=[]))
    assert p0 == 75000.0
    ledger = dict(phases=[dict(phase='science_doc1_H_C128', reservation_seconds=5500)])
    unfinished = [p for p in SCIENCE_PHASES if p != 'science_doc1_H_C128']
    p1, details = science_projection(budget, unfinished, ledger)
    assert p1 == 72000 and all(v == 5500 for k, v in details['remaining_phase_predictions_seconds'].items() if k.endswith('H_C128'))
    assert science_projection(budget, SCIENCE_PHASES, dict(phases=[dict(phase='science_doc1_H_C128', reservation_seconds=4000)]))[0] == p0
    root = Path('/synthetic') / TASK
    command = ['/data/ydai/miniconda3/envs/bitdecode/bin/python', '-u', str(root / 'control/worker_entry.py'), PHASES[-1]]
    assert exact_child(dict(pid=123, cmdline=command), PHASES[-1], root)
    assert not exact_child(dict(pid=123, cmdline=[*command[:2], str(root / 'src/worker.py'), command[-1]]), PHASES[-1], root)
    assert not exact_child(dict(pid=123, cmdline=command), PHASES[-2], root)
    # Disk capacity algebra: existing CURRENT/tmp occupancy is not double-counted.
    peak, small, reserve = 2 * 1507209134, 512 * 1024**2, 1024**3
    for cp in (0, 1507209134, peak):
        initial_free = peak + small + reserve
        current_free = initial_free - cp
        assert current_free >= max(0, peak - cp) + small + reserve
    mocked = mock_execution_tests()
    broken_stdout = dict(phases=[])
    from unittest.mock import patch
    with patch('builtins.print', side_effect=BrokenPipeError('synthetic stdout fixture')):
        emit(dict(phase='SYNTHETIC_ALREADY_PASS', status='PASS'), broken_stdout)
    assert broken_stdout['informational_stdout_errors'][0]['scientific_PASS_preserved'] is True
    return dict(status='PASS', scope='Pure control metadata/algebra mocks; no SSH/NVML/Torch/model/GPU',
                phases=17, science_phases=15, same_condition_only_monotone_forecast='PASS',
                wrong_entry_or_phase_rejected='PASS', rolling_checkpoint_capacity_accounting='PASS',
                supervisor_mock_execution=mocked, broken_stdout_preserves_PASS='PASS')


def mock_execution_tests():
    """Disposable control metadata only; no real subprocess or scientific data."""
    import tempfile
    from types import SimpleNamespace
    from unittest.mock import patch
    class MockStop(RuntimeError):
        def __init__(self, message, classification='FAIL_AND_STOP'):
            super().__init__(message); self.classification = classification
    def scenario(failed_phase=None, release=True, preflight_failure=False,
                 missing_receipt_phase=None, release_failure_phase=None):
        with tempfile.TemporaryDirectory() as td:
            root = (Path(td) / TASK).resolve(); root.mkdir()
            clock = [0.0]; dispatched = []; active = {}; pid = 54321
            probe = dict(status='PASS', apps=[], used_MiB=0)
            def monotonic():
                clock[0] += 0.01; return clock[0]
            def write(rel, value, exclusive=False):
                path = root / rel; path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('x' if exclusive else 'w') as f:
                    json.dump(value, f)
            def read(path):
                return json.loads(Path(path).read_text())
            def require(ok, message, classification='FAIL_AND_STOP'):
                if not ok:
                    raise MockStop(message, classification)
            def mock_stdio(path):
                (path / 'logs').mkdir()
                return 'SYNTHETIC_STDIO'
            def mock_preflight(*args):
                if preflight_failure:
                    raise MockStop('SYNTHETIC_PREFLIGHT_FAILURE_ONLY', 'BLOCKED_AND_STOP')
                return {}, budget
            def released_now():
                return release and (not release_failure_phase or active.get('command', [None])[-1] != release_failure_phase)
            class Child:
                def __init__(self, command, **kwargs):
                    self.command = command; self.pid = pid; self.polls = 0
                    phase = command[-1]; dispatched.append(phase); active['command'] = command
                    if phase in SCIENCE_PHASES:
                        parts = phase.split('_', 2)
                        result = dict(status='PASS', condition=parts[2], doc=int(parts[1][3:]), consumed=32768, scores=384)
                    else:
                        result = dict(status='PASS')
                    if phase == failed_phase:
                        self.code = 1
                        write('FIRST_FAILURE.json', dict(status='FAIL_AND_STOP', phase=phase,
                              error='SYNTHETIC_CONTROL_FAILURE_ONLY', no_GPU_retry=True), True)
                    else:
                        self.code = 0
                        if phase != missing_receipt_phase:
                            write(f'analysis/RECEIPT_{phase}.json', dict(status='PASS', result=result), True)
                def poll(self):
                    self.polls += 1
                    return None if self.polls == 1 else self.code
                def wait(self):
                    return self.code
            C = SimpleNamespace(PYTHON='/data/ydai/miniconda3/envs/bitdecode/bin/python', GPU_UUID='SYNTHETIC_GPU_METADATA',
                    MAX_GPU_SECONDS=86400, SAMPLE_SECONDS=.5, require=require, StopExecution=MockStop,
                    read=read, write=write, sha=lambda path:'SYNTHETIC_FROZEN_HASH', source_check=lambda x:None,
                    idle_admission=lambda:dict(probe), nvml_probe=lambda:dict(probe), hardware_identity=lambda p:None,
                    process_identity=lambda p:dict(pid=p, start_time_ticks='SYNTHETIC_START', cmdline=active['command']),
                    exact_child=lambda identity, phase:exact_child(identity, phase, root),
                    terminate_own_child=lambda *a:dict(exact_PID_only=True, termination_requested=False),
                    wait_for_release=lambda p:(released_now(), dict(probe)),
                    safe_probe=lambda:dict(probe, apps=[] if released_now() else [dict(pid=pid)]))
            budget = dict(phase_predictions_seconds={p:5000 for p in SCIENCE_PHASES})
            with patch.dict(globals(), import_helpers=lambda p:C, preflight=mock_preflight,
                            resident_stdio=mock_stdio, tmux_residency=lambda:dict(status='PASS'),
                            frozen_documents=lambda *a:None, storage_check=lambda *a:dict(status='PASS'),
                            check_fresh_resource=lambda *a:dict(status='PASS'), emit=lambda *a:None), \
                 patch.object(subprocess, 'Popen', Child), patch.object(signal, 'signal', lambda *a:None), \
                 patch.object(time, 'sleep', lambda x:None), patch.object(time, 'monotonic', monotonic):
                result = run(root)
            ledger = read(root / 'gpu_resource_ledger.json')
            assert len(ledger['phases']) == len(dispatched)
            assert ledger['GPU_HOURS_USED'] == sum(p['GPU_hours'] for p in ledger['phases'])
            assert all(p['reservation_seconds'] > 0 for p in ledger['phases'])
            assert not any((root / f).exists() for f in ('science_results.json', 'science_results.csv', 'window_results.csv'))
            if preflight_failure:
                assert result['status'] == 'BLOCKED_AND_STOP' and not dispatched and not ledger['phases']
                assert result['GPU_HOURS_USED'] == 0 and set(result['gates'].values()) == {'NOT_RUN'}
            elif failed_phase:
                assert result['status'] == 'FAIL_AND_STOP' and dispatched[-1] == failed_phase
                assert read(root / 'FIRST_FAILURE.json')['error'] == 'SYNTHETIC_CONTROL_FAILURE_ONLY'
                assert all(result['gates'][p] == 'NOT_RUN' for p in PHASES[len(dispatched):])
            elif missing_receipt_phase:
                assert result['status'] == 'FAIL_AND_STOP' and dispatched[-1] == missing_receipt_phase
                assert result['gates'][missing_receipt_phase] == 'FAIL'
                assert all(result['gates'][p] == 'NOT_RUN' for p in PHASES[len(dispatched):])
            elif release and not release_failure_phase:
                assert result['status'] == 'COMPLETE_AND_STOP' and tuple(dispatched) == PHASES
                assert set(result['gates'].values()) == {'PASS'}
            else:
                assert result['status'] == 'FAIL_AND_STOP'
                assert len(dispatched) == (PHASES.index(release_failure_phase) + 1 if release_failure_phase else 1)
                assert result['GPU_release_verified'] is False and ledger['final_GPU_release_verified'] is False
                assert ledger['phases'][-1]['reservation_duration_is_lower_bound_if_release_unverified'] is True
            return dict(status=result['status'], dispatched=len(dispatched), released=result['GPU_release_verified'])
    return dict(all17_complete=scenario(), failure_stops_remaining=scenario('science_doc2_H_DUAL_M'),
                failed_preflight_no_dispatch=scenario(preflight_failure=True),
                missing_receipt_stops_and_accounts=scenario(missing_receipt_phase='science_doc1_H_C128'),
                unverified_release_stops_and_accounts=scenario(release=False),
                last_science_unreleased_is_not_complete=scenario(release_failure_phase=PHASES[-1]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=str(ROOT)); parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    output = self_test() if args.self_test else run(args.root)
    if args.self_test:
        print(json.dumps(output, ensure_ascii=False), flush=True)
    else:
        sys.exit(0 if output['status'] == 'COMPLETE_AND_STOP' else 1)
