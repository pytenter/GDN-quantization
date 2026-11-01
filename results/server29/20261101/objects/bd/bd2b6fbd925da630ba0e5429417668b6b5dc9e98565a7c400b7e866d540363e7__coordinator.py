"""C6R control only: four lanes, one20GPUh budget, verified new-score archiving.

Never changes a mathematical worker, retries a unit, reads scientific Delta, or
uses an old checkpoint. Run once from a real remote tmux pane after source freeze.
"""
import argparse
import concurrent.futures
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
TASK = 'QWEN_GDN_C6R_DUAL_M_INTERACTION_REUSE_5DOC_V1'
CONDITIONS = ('C6R_C128', 'C6R_DUAL_M')
DOC_GPU = {1: 0, 2: 1, 3: 2, 4: 3, 5: 0}
SCIENCE = tuple(f'science_doc{d}_{c}' for d in range(1, 6) for c in CONDITIONS)
QUAL = tuple(f'gpu{g}_{p}' for g in range(4) for p in ('canary_initial', 'canary_fresh'))
UNITS = QUAL + SCIENCE
GATE_NAMES = ('MODEL_IDENTITY', 'C6R_ROTATION_IDENTITY', 'C6R_COORDINATE_CORRECTNESS',
              'C6R_C128_CANONICAL_PARITY', 'C6R_DUAL_M_CODEC_PARITY',
              'PERSISTENT_QDQ_WRITEBACK', 'NEXT_KERNEL_CONSUMPTION',
              'FRESH_PROCESS_NUMERICAL_QUALIFICATION', 'SCALE_AND_CLIPPING_SAFETY',
              'GPU_RESOURCE_ADMISSION', 'FP_REFERENCE_SCORE_REUSE',
              'RUNTIME_BUDGET_FEASIBILITY')
HARD_SECONDS, PROTECT_SECONDS = 20 * 3600, 19.5 * 3600
HELPER_SHA = '28647e7d03b9d64f88c8e882bc27ae296053187874ee8431036dfeddbd99e065'
SSH29 = ['ssh', '-S', '/tmp/qwen_replication_archive29.sock', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'zypan@202.38.247.29']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def write(path, value, exclusive=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open('x') as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
    else:
        temp = path.with_name(path.name + '.control_tmp')
        with temp.open('w') as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
        os.replace(temp, path)


def helpers(root):
    path = root / 'src/coordinator.py'
    need(sha(path) == HELPER_SHA, 'Qualified process helper bytes changed')
    spec = importlib.util.spec_from_file_location('qualified_c6r_control_helpers', path)
    C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
    C.ROOT, C.TASK_ID, C.MAX_GPU_SECONDS = root, TASK, HARD_SECONDS
    return C


def stdio(root):
    path = root / 'logs/COORDINATOR_STDOUT.log'; path.parent.mkdir(exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    null = os.open('/dev/null', os.O_RDONLY)
    os.dup2(null, 0); os.dup2(fd, 1); os.dup2(fd, 2); os.close(null); os.close(fd)
    sys.stdout = open(1, 'w', buffering=1, closefd=False)
    sys.stderr = open(2, 'w', buffering=1, closefd=False)
    need(stat.S_ISREG(os.fstat(1).st_mode) and stat.S_ISREG(os.fstat(2).st_mode), 'Resident regular-file stdout required')
    return str(path)


def tmux_identity():
    need(os.environ.get('TMUX') and os.environ.get('TMUX_PANE'), 'Actual remote tmux pane required')
    pane = os.environ['TMUX_PANE']; env = os.environ['TMUX'].rsplit(',', 2)
    raw = subprocess.check_output(['tmux', 'display-message', '-p', '-t', pane, '#{pane_pid} #{session_name}'], text=True, timeout=10).strip().split(maxsplit=1)
    need(len(env) == 3 and env[1].isdigit() and len(raw) == 2 and raw[0].isdigit(), 'Invalid tmux metadata')
    server, pane_pid = int(env[1]), int(raw[0]); proc = Path('/proc') / str(server)
    executable = Path(shutil.which('tmux') or '')
    need(proc.stat().st_uid == os.getuid() and executable.is_file() and os.path.samefile(proc/'exe', executable), 'Wrong tmux server executable/UID')
    pid, chain = os.getpid(), []
    while pid > 1 and len(chain) < 64:
        chain.append(pid)
        if pid == server: break
        text = (Path('/proc') / str(pid) / 'stat').read_text(); pid = int(text[text.rfind(')')+2:].split()[1])
    need(pane_pid in chain and server in chain and chain.index(pane_pid) <= chain.index(server), 'Controller is not descended from real tmux pane/server')
    return dict(status='PASS', pane=pane, pane_PID=pane_pid, server_PID=server, ancestor_chain=chain, session=raw[1])


def inventory(uuids):
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.free,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True, timeout=10)
    GPUs = {}
    for row in csv.reader(raw.splitlines()):
        v = [x.strip() for x in row]
        if v and v[0] in uuids:
            need(v[0] not in GPUs and v[1] == uuids[v[0]] and 'RTX 3090' in v[2] and int(v[3]) == 24576, 'Physical GPU UUID/type/capacity mismatch')
            GPUs[v[0]] = dict(physical_GPU_ID=int(v[0]), GPU_UUID=v[1], total_MiB=int(v[3]), free_MiB=int(v[4]), used_MiB=int(v[5]), utilization_percent=int(v[6]), apps=[])
    need(set(GPUs) == set(uuids) == {'0', '1', '2', '3'}, 'Four exact authorized GPU identities required')
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory', '--format=csv,noheader,nounits'], text=True, timeout=10)
    by_uuid = {v:k for k,v in uuids.items()}
    for row in csv.reader(raw.splitlines()):
        v = [x.strip() for x in row]
        if v and v[0] in by_uuid:
            need(len(v) >= 4, 'Malformed compute-process inventory')
            GPUs[by_uuid[v[0]]]['apps'].append(dict(pid=int(v[1]), process_name=v[2], used_MiB=v[3]))
    return dict(status='PASS', unix=time.time(), GPUs=GPUs)


def safe_inventory(uuids):
    try: return inventory(uuids)
    except BaseException as exc: return dict(status='QUERY_FAILED', error=repr(exc), unix=time.time())


def idle(probe, gpu):
    x = probe['GPUs'][str(gpu)]
    need(not x['apps'] and x['free_MiB'] >= 23000 and x['utilization_percent'] <= 1, f'GPU{gpu} not idle with23000MiB free')


def tree_bytes(root):
    total = 0
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if not (Path(directory)/d).is_symlink()]
        for name in files:
            p = Path(directory)/name
            try: total += p.lstat().st_size
            except FileNotFoundError: pass
    return total


def available(root):
    x = os.statvfs(root); return x.f_bavail * x.f_frsize


def storage(root, budget):
    b = budget['storage']; cpdir = Path(b['checkpoint_directory'])
    need((root/'checkpoints').is_symlink() and (root/'checkpoints').resolve() == cpdir and cpdir.is_dir() and not cpdir.is_symlink(), 'Unregistered global checkpoint directory')
    need(root.stat().st_dev != cpdir.stat().st_dev, 'Checkpoints must use independent/tmp filesystem')
    for g in range(4):
        link = root/'qualification_gpu'/f'gpu{g}'/'checkpoints'
        need(link.is_symlink() and link.resolve() == cpdir, 'Canary checkpoint must share registered global slot')
    need((root/'scores').is_dir() and not (root/'scores').is_symlink(), 'New scores must use task-local/data directory')
    snapshots = []
    for p in cpdir.glob('CURRENT*'):
        try:
            if p.is_file(): snapshots.append((str(p), p.stat().st_size))
        except FileNotFoundError:
            pass  # Original atomic CURRENT.tmp replacement may retire this entry.
    occupied = sum(size for _, size in snapshots)
    remaining_cp = max(0, 2*b['checkpoint_file_upper_bound_bytes'] - occupied)
    metadata = tree_bytes(root); scores = tree_bytes(root/'scores')
    for g in range(4): scores += tree_bytes(root/'qualification_gpu'/f'gpu{g}'/'scores')
    non_scores = max(0, metadata-scores)
    score_peak = 4*b['score_file_upper_bound_bytes'] + b['one_score_write_allowance_bytes'] + b['canary_score_allowance_bytes']
    remaining_data = max(0, score_peak-scores) + max(0, b['metadata_allowance_bytes']-non_scores)
    need(available(cpdir) >= remaining_cp + b['checkpoint_free_reserve_bytes'], 'Checkpoint atomic capacity/reserve insufficient')
    need(available(root) >= remaining_data + b['metadata_free_reserve_bytes'], 'Four in-flight scores/metadata capacity insufficient')
    need(all(size <= b['checkpoint_file_upper_bound_bytes'] for _, size in snapshots), 'Actual checkpoint exceeds frozen bound')
    return dict(status='PASS', checkpoint_free_bytes=available(cpdir), data_free_bytes=available(root), checkpoint_occupied_bytes=occupied, new_metadata_bytes=metadata, new_score_bytes=scores)


def source_check(root, manifest):
    for row in manifest['execution_sources']:
        rel = Path(row['path']); need(not rel.is_absolute() and '..' not in rel.parts, 'Unsafe source path')
        path = root/rel; need(path.is_file() and sha(path) == row['sha256'] and path.stat().st_size == row['bytes'], 'Frozen source changed: '+str(rel))


def preflight(root):
    need(root.name == TASK and not (root/'FIRST_FAILURE.json').exists(), 'Wrong root or existing first failure')
    manifest, prereg, budget = read(root/'SOURCE_MANIFEST.json'), read(root/'PREREGISTRATION.json'), read(root/'RESOURCE_BUDGET.json')
    need(manifest.get('task', manifest.get('task_id')) == prereg['TASK_ID'] == budget['task_id'] == TASK, 'Task identity mismatch')
    need(manifest['frozen_before_any_new_GPU_forward'] is True and prereg['GPU_EXECUTION_AUTHORIZED'] == 'YES' and prereg['GPU_COUNT'] == 4, 'Frozen four-GPU authorization absent')
    need(sha(root/'source_manifest.json') == sha(root/'SOURCE_MANIFEST.json') and sha(root/'preregistration.json') == sha(root/'PREREGISTRATION.json'), 'Original worker metadata aliases differ')
    source_check(root, manifest)
    need(budget['MAX_NEW_GPU_HOURS'] == 20 and budget['PROTECTIVE_STOP_GPUH'] == 19.5 and budget['GPU_COUNT'] == 4, 'Shared budget or protective limit changed')
    need(set(budget['GPU_UUIDS']) == {'0','1','2','3'} and len(set(budget['GPU_UUIDS'].values())) == 4, 'Four distinct UUIDs required')
    need(set(budget['phase_predictions_seconds']) == set(SCIENCE), 'Budget must cover exactly10 scientific trajectories')
    need(prereg['score_positions'] == list(range(8192,32768,64)) and prereg['doc_GPU_assignment'] == {str(k):v for k,v in DOC_GPU.items()}, 'Frozen positions/doc-device schedule changed')
    for name in ('HISTORICAL_ASSET_AUDIT.json','C6R_ROTATION_IDENTITY.json','C6R_MATH_RUNTIME_AUDIT.json'):
        need(read(root/name)['status'] == 'PASS', 'Required CPU gate is not PASS: '+name)
    old = Path(budget['projection_basis_source_path']); need(sha(old) == budget['projection_basis_source_sha256'], 'Historical resource source changed')
    rows = read(old)['phases']
    for condition, previous in [('C6R_C128','H_C128'),('C6R_DUAL_M','H_DUAL_M')]:
        actual = [x['reservation_seconds'] for x in rows if x['phase'].startswith('science_doc') and x['phase'].endswith('_'+previous)]
        need(len(actual) == 5 and all(budget['phase_predictions_seconds'][p] == max(actual)*1.20 for p in SCIENCE if p.endswith(condition)), 'Initial forecast is not five real same-codec maximum*1.20')
    need(sha(Path(budget['historical_canary_source_path'])) == budget['historical_canary_source_sha256'], 'Historical128 timing baseline changed')
    need(sum(budget['phase_predictions_seconds'].values()) + budget['qualification_prediction_seconds'] + budget['additional_GPU_held_IO_reserve_seconds'] + budget['safety_exit_reserve_seconds'] < PROTECT_SECONDS, 'Whole plan cannot fit protective19.5GPUh')
    for g in range(4):
        lane = root/'qualification_gpu'/f'gpu{g}'
        for name in ('source_manifest.json','preregistration.json'):
            need(sha(lane/name) == sha(root/name), 'Canary scope source/protocol differs')
    need(not (root/'GPU_RESOURCE_LEDGER.json').exists() and not (root/'FINAL_GPU_STATUS.json').exists(), 'Previous execution exists')
    return manifest, prereg, budget


def total_seconds(ledger, active, now):
    return math.fsum(x['reservation_seconds'] for x in ledger['phases']) + math.fsum(max(0,now-x['start']) for x in active.values())


def forecast(budget, unfinished, ledger, active, now):
    seen = {}
    for row in ledger['phases']:
        if row['phase'] in SCIENCE:
            cond = row['phase'].split('_',2)[2]; seen[cond] = max(seen.get(cond,0), row['reservation_seconds'])
    base = budget.get('adjusted_phase_predictions_seconds',budget['phase_predictions_seconds'])
    predictions = {p:max(base[p],seen.get(p.split('_',2)[2],0)) for p in unfinished}
    elapsed = {x['phase']:max(0,now-x['start']) for x in active.values()}
    return math.fsum(max(0,v-elapsed.get(p,0)) for p,v in predictions.items()), dict(phase_predictions_seconds=predictions, observed_condition_max_seconds=seen, active_elapsed_seconds=elapsed, only_upward=True)


def fresh_gate(root, budget):
    measured = {}; adjusted = {}; old = read(Path(budget['projection_basis_source_path']))['phases']
    for cond, prior in [('C6R_C128','H_C128'),('C6R_DUAL_M','H_DUAL_M')]:
        current=[]
        for g in range(4):
            lane=root/'qualification_gpu'/f'gpu{g}'; runtime=read(lane/'C6R_CANARY_RUNTIME.json')
            need(runtime['status']=='PASS','C6R fresh runtime gate failed')
            need(runtime['projected_peak_reserved_bytes']<runtime['admissible_memory_bytes'],'Original Qwen canary memory admission failed')
            x=runtime['throughput'][cond]['median_seconds'];need(type(x) in (int,float) and math.isfinite(x) and x>0,'Invalid canary median');current.append(x)
        baseline=budget['historical_canary_median_seconds'][cond];ratio=max(current)/baseline
        worst=max(x['reservation_seconds'] for x in old if x['phase'].startswith('science_doc') and x['phase'].endswith('_'+prior))
        value=max(budget['phase_predictions_seconds'][next(p for p in SCIENCE if p.endswith(cond))],worst*ratio)
        for p in SCIENCE:
            if p.endswith(cond):adjusted[p]=value
        measured[cond]=dict(actual_four_median_seconds=current,historical_median_seconds=baseline,ratio=ratio,adjusted_phase_seconds=value)
    checked=[]
    for tag in ('A_FRESH','D_FRESH'):
        views=[]
        for g in range(4):
            a=read(root/'qualification_gpu'/f'gpu{g}'/'analysis'/(tag+'.json'));need(a['status']=='PASS' and a['consumed_tokens']==128 and len(a['rows'])==128,'Fresh exact128 rows absent')
            keys=('t','input_id','logits','cache','cache_cursor','helper','QDQ');views.append([{k:x[k] for k in keys} for x in a['rows']])
        need(all(v==views[0] for v in views[1:]),'Cross-GPU C6R128 exact qualification failed');checked.append(tag)
    write(root/'analysis/C6R_FOUR_GPU_RESOURCE_ADMISSION_INPUTS.json',dict(status='PASS',throughput=measured,adjusted_phase_predictions_seconds=adjusted,cross_GPU_exact_tags=checked,scope='128 only, not a32K numerical uncertainty bound'),True)
    return dict(budget,adjusted_phase_predictions_seconds=adjusted)


def qualification(root, budget, manifest, ledger, unit_gates, archive):
    """Derive named gates from actual receipts, never a synthetic PASS roster."""
    need(all(unit_gates[p] == 'PASS' for p in QUAL), 'Eight real qualification processes required')
    rows = [p for p in ledger['phases'] if p['phase'] in QUAL]
    need(len(rows) == 8 and len({p['phase'] for p in rows}) == 8 and
         all(p['released'] and p['returncode'] == 0 for p in rows), 'Qualification ledger/release incomplete')
    proof = {}
    def record(name, value, paths, scope):
        need(value, 'Named qualification failed: ' + name)
        proof[name] = dict(status='PASS', actual_proof_files=[dict(path=str(p.relative_to(root)), sha256=sha(p)) for p in paths], scope=scope)

    history_path=root/'HISTORICAL_ASSET_AUDIT.json'; history=read(history_path)
    rotation_path=root/'C6R_ROTATION_IDENTITY.json'; rotation=read(rotation_path)
    math_path=root/'C6R_MATH_RUNTIME_AUDIT.json'; mathematical=read(math_path)
    env_paths=[]; initial_paths=[]; fresh_paths=[]; analysis_paths=[]; device_paths=[]; runtime_paths=[]; lock_paths=[]
    all_runs=[]; initials=[]; freshs=[]; runtimes=[]; environments=[]
    layers=set(str(x['layer']) for x in rotation['matrix_checks'])
    need(len(layers)==24 and rotation['status']==mathematical['status']==history['status']=='PASS', 'Actual CPU identity gates absent')
    for g in range(4):
        lane=root/'qualification_gpu'/f'gpu{g}'
        ip=lane/'CANARY_INITIAL.json'; fp=lane/'CANARY_FRESH.json'; rp=lane/'C6R_CANARY_RUNTIME.json'
        initial_paths.append(ip); fresh_paths.append(fp); runtime_paths.append(rp)
        i,f,r=read(ip),read(fp),read(rp); initials.append(i);freshs.append(f);runtimes.append(r)
        need(i['status']==f['status']==r['status']=='PASS', 'Per-card canary proof not PASS')
        for phase in ('canary_initial','canary_fresh'):
            ep=lane/'analysis'/('ENV_'+phase+'.json'); dp=lane/'analysis'/('DEVICE_ADAPTER_'+phase+'.json'); lp=lane/'analysis'/('CHECKPOINT_LOCK_'+phase+'.json')
            env_paths.append(ep);device_paths.append(dp);lock_paths.append(lp)
            env,dev,lock=read(ep),read(dp),read(lp);environments.append(env)
            need(env['UUID']==dev['GPU_UUID']==r['GPU_UUID']==budget['GPU_UUIDS'][str(g)] and env['physical_GPU_ID']==dev['physical_GPU_ID']==g and
                 env['CUDA_VISIBLE_DEVICES']==str(g) and env['strict'] is True and env['TF32'] is False and dev['status']=='PASS', 'Actual device/environment identity mismatch')
            need(lock['checkpoint_function_body_unchanged'] is True and lock['entry_trace_removed_before_main'] is True and lock['main_call_hook_count']==1, 'Complete original checkpoint lock not executed')
        by_tag={}
        for tag in ('A_REFERENCE','A_OFF','A_ON','D_OFF','D_ON','A_FRESH','D_FRESH'):
            path=lane/'analysis'/(tag+'.json');analysis_paths.append(path);run=read(path);all_runs.append(run);by_tag[tag]=run
            need(run['status']=='PASS' and run['consumed_tokens']==128 and len(run['rows'])==128 and run['rotation_calls']==128*24 and
                 set(map(str,run['rotation_layers']))==layers and run['adapter_run_inverse_AST_exact'] is True, 'Actual full128 C6R run/rotation incomplete')
            need(run['C6R']['rotation_sha256']==rotation['rotation_SHA256'] and run['C6R']['deployment_manifest_sha256']==rotation['deployment_manifest_SHA256'] and
                 run['C6R']['direct_frozen_load'] is True and run['C6R']['theta_reconstruction'] is False, 'Canary C6R export identity mismatch')
            need(run['writebacks']==128*24 and run['next_token_consumption_checks']==127*24, 'Persistent QDQ count incomplete')
            need([x['t'] for x in run['rows']]==list(range(128)) and all(x['cache_cursor']==x['t']+1 for x in run['rows']), 'Canary cursor sequence changed')
            need(len(run['performance'])==128 and all(math.isfinite(p['full_model_plus_QDQ_seconds']) and p['full_model_plus_QDQ_seconds']>0 for p in run['performance']), 'Actual canary performance incomplete')
            if tag in ('A_REFERENCE','A_ON','D_ON','A_FRESH','D_FRESH'):
                for n,row in enumerate(run['rows']):
                    need(set(row['QDQ'])==set(row['helper'])==layers, 'Per-layer QDQ/readout proof missing')
                    need(all(row['helper'][l]['initial']==run['rows'][n-1]['QDQ'][l]['post'] for l in layers) if n else
                         all(row['helper'][l]['initial'] is None for l in layers), 'Next kernel did not consume previous QDQ bytes')
        comparable=lambda x,helpers: [{k:r[k] for k in (('t','input_id','logits','cache','cache_cursor','helper','QDQ') if helpers else ('t','input_id','logits','cache','cache_cursor'))} for r in x['rows']]
        need(comparable(by_tag['A_REFERENCE'],True)==comparable(by_tag['A_ON'],True) and comparable(by_tag['A_OFF'],False)==comparable(by_tag['A_ON'],False), 'Actual canonical/observer parity differs')
        need(comparable(by_tag['D_OFF'],False)==comparable(by_tag['D_ON'],False) and
             comparable(by_tag['A_ON'],True)==comparable(by_tag['A_FRESH'],True) and comparable(by_tag['D_ON'],True)==comparable(by_tag['D_FRESH'],True), 'Actual observer/fresh parity differs')
        official=by_tag['D_ON']['mechanism'][0]['stats']
        need(official['official_exact']=='PASS_TORCH_EQUAL_CODES_SCALES_DEQUANT' and official['batched_vs_individual']=='BYTE_EXACT', 'Actual official codec parity absent')
        cp=by_tag['A_ON']['checkpoints'];need(len(cp)==1 and cp[0]['restored_exact'] is True and cp[0]['cursor']==64, 'Actual64 checkpoint restore proof absent')
        for tag in ('D_OFF','D_ON','D_FRESH'):
            for p in by_tag[tag]['performance']:
                # These two maxima can refer to different heads. The original
                # worker tests their conjunction per head before returning PASS;
                # recombining aggregate maxima would invent a stricter gate.
                risk=p['scale_risk']; need(all(math.isfinite(risk[k]) for k in ('max_affected_energy_fraction','max_affected_relative_mse')), 'C6R scale-risk audit incomplete')
        need(not (lane/'scale_safety.json').exists() and not (lane/'FIRST_FAILURE.json').exists(), 'Real canary safety/failure receipt exists')
    architectures={x['sha256'] for x in history['current_model_runtime_hashes'] if x['path']=='/data/zypan/transformers-qwen35/src/transformers/models/qwen3_5/modeling_qwen3_5.py'}
    record('MODEL_IDENTITY', len(architectures)==1 and all(e['load']['parameter_dtypes']==['torch.bfloat16'] and e['load']['architecture_sha256'] in architectures for e in environments), [history_path]+env_paths, 'Fresh-stream CPU model/runtime identities, actual original loader/device/dtype/environment')
    record('C6R_ROTATION_IDENTITY', True,[rotation_path]+initial_paths+fresh_paths,'Exact frozen direct C6R export/manifest in all eight processes; no theta reconstruction')
    record('C6R_COORDINATE_CORRECTNESS',mathematical['minimal_adapter']['GDN_operator_changes'] is False if 'GDN_operator_changes' in mathematical.get('minimal_adapter',{}) else mathematical['runtime']['GDN_operator_changes'] is False,[math_path]+analysis_paths,'CPU exact original rotation AST and every128-token/24-layer actual C6R invocation; axis unchanged')
    record('C6R_C128_CANONICAL_PARITY',all(x['canonical_C128_direct_function'] is True and x['C6R_original_reference']=='BYTE_EXACT_ALL_128' for x in initials),initial_paths+analysis_paths,'Direct original canonical C128; actual original unmemoized versus executor all128 exact')
    record('C6R_DUAL_M_CODEC_PARITY',all(x['DualM_official_all8_ones']=='PASS_TORCH_EQUAL_CODES_SCALES_DEQUANT' for x in initials),initial_paths+analysis_paths,'Actual official all8/ones code/row/column/dequant parity on C6R state')
    record('PERSISTENT_QDQ_WRITEBACK',True,analysis_paths,'Every128-token/24-layer actual QDQ/writeback; FP32 carrier, physical packed INT8 NOT_QUALIFIED')
    record('NEXT_KERNEL_CONSUMPTION',True,analysis_paths+lock_paths,'Actual pointer assertions plus traced prior QDQ-post/next-kernel-initial bytes; original checkpoint64 exact restore')
    record('FRESH_PROCESS_NUMERICAL_QUALIFICATION',all(x['fresh_process']=='BYTE_EXACT_C6R_C128_DUAL_M_ALL128' for x in freshs),fresh_paths+[root/'analysis/C6R_FOUR_GPU_RESOURCE_ADMISSION_INPUTS.json'],'Four distinct fresh processes and cross-card128 exact; not a full32K uncertainty bound')
    record('SCALE_AND_CLIPPING_SAFETY',True,analysis_paths,'Actual frozen finite and per-layer/head affected-energy AND reconstruction-risk checks; nonzero small clipping/floor events permitted')
    record('GPU_RESOURCE_ADMISSION',all(x['projected_peak_reserved_bytes']<x['admissible_memory_bytes'] for x in runtimes) and archive['status']=='PASS',[root/'gpu_admission.json',root/'archive/ARCHIVE_STORAGE_PREFLIGHT.json']+runtime_paths+device_paths,'Actual four idle UUIDs, original Qwen memory projection, independent/tmp global current and29 permanent capacity')
    record('FP_REFERENCE_SCORE_REUSE',history['FP_REFERENCE_REUSE_GATE']=='PASS' and history['HISTORICAL_REUSE_GATE']=='PASS' and history['all15_score_SHA_and384_vector_hashes_freshly_verified'] is True,[history_path],'All five complete FP references and original two-codec KL reproduced on CPU, no new FP/H forward')
    remaining,details=forecast(budget,SCIENCE,ledger,{},time.monotonic());used=total_seconds(ledger,{},time.monotonic())
    projected=used+remaining+budget['additional_GPU_held_IO_reserve_seconds']+budget['safety_exit_reserve_seconds']
    record('RUNTIME_BUDGET_FEASIBILITY',projected<PROTECT_SECONDS,[root/'RESOURCE_BUDGET.json',root/'analysis/C6R_FOUR_GPU_RESOURCE_ADMISSION_INPUTS.json'],'Actual eight reservation durations + current conservative ten-trajectory forecast + IO/exit reserves below19.5; shared hard20GPUh')
    need(set(proof)==set(GATE_NAMES) and len(proof)==12,'Exactly twelve real gates required')
    result=dict(task_id=TASK,status='PASS',gates=proof,unit_gates={p:unit_gates[p] for p in QUAL},physical_GPU_UUIDS=budget['GPU_UUIDS'],qualification_process_count=8,
                actual_qualification_GPUh=used/3600,projected_whole_GPUh=projected/3600,remaining_forecast=details,MAX_GPU_HOURS=20.0,PROTECTIVE_STOP_GPUH=19.5,
                source_manifest_sha256=sha(root/'SOURCE_MANIFEST.json'),resource_budget_sha256=sha(root/'RESOURCE_BUDGET.json'),scope='Preformal128 qualification only; no full32K forward uncertainty bound',verified_unix=time.time())
    write(root/'GPU_QUALIFICATION.json',result,True)
    return result


REMOTE_VERIFY = r'''import pathlib,hashlib,json,subprocess,sys,os,time
stage=pathlib.Path(sys.argv[1]);dest=pathlib.Path(sys.argv[2]);expected=json.loads(sys.argv[3]);checked=[]
assert stage.is_dir() and not dest.exists()
for e in expected:
 p=stage/e['relative_path'];assert p.resolve().is_relative_to(stage.resolve());a=p.stat();h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(16777216),b''):h.update(b)
 independent=subprocess.check_output(['sha256sum','--',str(p)],text=True).split()[0];z=p.stat()
 assert z.st_size==e['bytes'] and h.hexdigest()==independent==e['sha256'] and (a.st_size,a.st_mtime_ns)==(z.st_size,z.st_mtime_ns)
 checked.append(dict(e,python_stream_SHA=True,independent_sha256sum=True))
out=dict(status='PASS',archive_path=str(dest),assets=checked,verified_unix=time.time(),atomic_seal=True)
with (stage/'ARCHIVE_VERIFICATION.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
os.rename(stage,dest);print(json.dumps(out))
'''


def archive_preflight(budget):
    py=subprocess.check_output(SSH29+['command -v python3'],text=True,timeout=20).strip();need(py.startswith('/'),'No usable Python3 on29')
    script='import os,json;v=os.statvfs("/data/zypan");print(json.dumps(dict(free_bytes=v.f_bavail*v.f_frsize)))'
    raw=subprocess.check_output(SSH29+[shlex.join([py,'-c',script])],text=True,timeout=20);free=json.loads(raw)['free_bytes']
    need(free>=budget['storage']['max_new_permanent_archive_bytes'],'Permanent29 archive capacity insufficient')
    return dict(status='PASS',python=py,free_bytes=free,archive_root=budget['storage']['permanent_archive_root'])


def archive_assets(root, budget, python29, label, relative_paths, remove_score=False):
    root=Path(root);base=Path(budget['storage']['permanent_archive_root'])
    need(base.name==TASK and str(base).startswith('/data/zypan/qwen_c6r_dual_m_interaction_archive/'),'Wrong permanent new-task archive root')
    dest=base/label;stage=dest.with_name('.'+dest.name+'.partial');assets=[]
    for rel in relative_paths:
        path=root/rel;need(path.is_file() and not path.is_symlink(),'Missing/unexpected symlink asset')
        allowed=(str(rel).startswith('checkpoints/') and path.resolve().parent==Path(budget['storage']['checkpoint_directory'])) or path.resolve().is_relative_to(root.resolve())
        need(allowed,'Asset escapes new-task scopes');a=path.stat();h=sha(path);z=path.stat();need((a.st_size,a.st_mtime_ns,a.st_ino)==(z.st_size,z.st_mtime_ns,z.st_ino),'Asset changed while hashing');assets.append(dict(relative_path=str(rel),sha256=h,bytes=z.st_size))
    command='mkdir -p '+shlex.quote(str(dest.parent))+' && test ! -e '+shlex.quote(str(dest))+' && mkdir '+shlex.quote(str(stage))
    subprocess.run(SSH29+[command],check=True,timeout=20)
    subprocess.run(['rsync','-aLR','-e',shlex.join(SSH29[:-1]),'--',*map(str,relative_paths),SSH29[-1]+':'+str(stage)+'/'],cwd=root,check=True,timeout=180)
    argv=[python29,'-c',REMOTE_VERIFY,str(stage),str(dest),json.dumps(assets,separators=(',',':'))]
    rec=json.loads(subprocess.check_output(SSH29+[shlex.join(argv)],text=True,timeout=180));need(rec['status']=='PASS','Remote independent archive verification failed')
    write(root/'archive'/(label.replace('/','_')+'_ARCHIVE_VERIFICATION.json'),rec,True)
    for e in assets:e.update(archive_path=str(dest/e['relative_path']),local_copy_removed=False)
    if remove_score:
        need(label in SCIENCE,'Only registered new science scores may be removed')
        expected='scores/'+label+'.pt';e=next(x for x in assets if x['relative_path']==expected);path=root/expected
        need(path.resolve().parent==(root/'scores').resolve() and path.is_file() and not path.is_symlink() and path.stat().st_size==e['bytes'] and sha(path)==e['sha256'],'Exact new score changed before removal')
        doc=int(label.split('_',2)[1][3:]);mapping_path=root/'archive'/f'DOC_{doc:02d}_CURRENT_ASSET_LOCATIONS.json';mapping=read(mapping_path) if mapping_path.exists() else dict(status='PARTIAL',docindex=doc,assets=[])
        need(not any(x['relative_path']==expected for x in mapping['assets']),'Duplicate archived score')
        mapping['assets'].append(e);write(mapping_path,mapping)
        path.unlink();e['local_copy_removed']=True
        mapping['status']='PASS' if len(mapping['assets'])==2 and all(x['local_copy_removed'] for x in mapping['assets']) else 'PARTIAL';write(mapping_path,mapping)
    return dict(status='PASS',label=label,assets=assets,remote_verification=rec)


def archive_phase(root,budget,python29,phase):
    a=read(root/'analysis'/(phase+'.json'));need(a['score_file']=='scores/'+phase+'.pt' and sha(root/a['score_file'])==a['score_file_sha256'],'New score identity/path changed')
    return archive_assets(root,budget,python29,phase,[Path(a['score_file']),Path('analysis')/(phase+'.json'),Path('analysis')/('RECEIPT_'+phase+'.json')],True)


def run(root):
    root=Path(root).resolve();C=helpers(root)
    if any((root/x).exists() for x in ('CONTROL_RUN_GUARD.json','COORDINATOR_STARTED.json','FINAL_GPU_STATUS.json')):return dict(status='REFUSED_EXISTING_ATTEMPT',no_GPU_dispatch=True)
    logfile=stdio(root);ledger=dict(task_id=TASK,GPU_COUNT=4,MAX_GPU_HOURS=20.0,PROTECTIVE_STOP_GPUH=19.5,GPU_HOURS_USED=0.0,phases=[],started_unix=time.time(),supervisor_stdio_file=logfile,accounting='Sum each exact ownedGPU child dispatch through independently verified release; includes load/compute/checkpoint waits/scoringIO. Postrelease archive is CPU walltime only.')
    gates={x:'NOT_RUN' for x in UNITS};active={};pending={};registry={};uuids={};failure=None;phase=None;manifest_sha=None;budget_sha=None;budget=None
    pool=concurrent.futures.ThreadPoolExecutor(max_workers=1)
    def exact(identity,label):return identity.get('pid',0)>0 and label in registry and identity.get('cmdline')==registry[label]
    C.exact_child=exact
    def stop(signum,frame):raise RuntimeError('Supervisor signal '+str(signum))
    for sig in (signal.SIGHUP,signal.SIGTERM):signal.signal(sig,stop)
    def live():
        now=time.monotonic();write(root/'analysis/GPU_RESOURCE_LEDGER_LIVE.json',dict(ledger,active_cumulative_GPUh=total_seconds(ledger,active,now)/3600,active_reservations=[dict(phase=j['phase'],pid=j['child'].pid,physical_GPU_ID=g,active_reserved_seconds=now-j['start']) for g,j in active.items()],pending_CPU_archives=[x['phase'] for x in pending.values()]))
    def finish(gpu,allow_failure=False):
        job=active[gpu];child=job['child'];entry=job['entry'];error=None
        try:entry['cleanup']=C.terminate_own_child(child,job['identity'],job['phase'])
        except BaseException as exc:error=repr(exc);entry['cleanup_error']=error
        deadline=time.monotonic()+30;probe=safe_inventory(uuids)
        while time.monotonic()<deadline and not (probe['status']=='PASS' and not any(x['pid']==child.pid for x in probe['GPUs'][str(gpu)]['apps'])):
            time.sleep(.5);probe=safe_inventory(uuids)
        released=probe['status']=='PASS' and not any(x['pid']==child.pid for x in probe['GPUs'][str(gpu)]['apps']);elapsed=time.monotonic()-job['start']
        entry.update(returncode=child.poll(),reservation_seconds=elapsed,GPU_hours=elapsed/3600,released=released,release_probe=probe,release_observation_unix=time.time(),reservation_duration_is_lower_bound_if_release_unverified=not released,_last_accounted=time.monotonic())
        job['log'].close();ledger['phases'].append(entry);ledger['GPU_HOURS_USED']=math.fsum(x['GPU_hours'] for x in ledger['phases']);del active[gpu]
        recpath=job['output']/'analysis'/('RECEIPT_'+job['worker_phase']+'.json');valid=child.poll()==0 and released and error is None and recpath.is_file()
        if valid:
            rec=read(recpath);valid=rec.get('status')==rec.get('result',{}).get('status')=='PASS'
            if job['phase'] in SCIENCE:
                parts=job['phase'].split('_',2);valid=valid and rec['result']==dict(status='PASS',condition=parts[2],doc=int(parts[1][3:]),consumed=32768,scores=384)
            if valid:entry['receipt_sha256']=sha(recpath);entry['receipt_path']=str(recpath.relative_to(root))
        gates[job['phase']]='PASS' if valid else 'FAIL';live()
        if not allow_failure:
            need(valid and not (job['output']/'FIRST_FAILURE.json').exists(),'Worker/receipt/release failed: '+job['phase'])
            source_check(root,manifest)
        return job
    try:
        manifest,prereg,budget=preflight(root);uuids=budget['GPU_UUIDS'];manifest_sha=sha(root/'SOURCE_MANIFEST.json');budget_sha=sha(root/'RESOURCE_BUDGET.json')
        residency=tmux_identity()
        try:write(root/'CONTROL_RUN_GUARD.json',dict(pid=os.getpid(),unix=time.time(),source_manifest_sha256=manifest_sha),True)
        except FileExistsError:return dict(status='REFUSED_SIMULTANEOUS_COORDINATOR',no_GPU_dispatch=True)
        write(root/'control/TMUX_RESIDENCY.json',residency,True);storage(root,budget);archive=archive_preflight(budget);write(root/'archive/ARCHIVE_STORAGE_PREFLIGHT.json',archive,True)
        initial=inventory(uuids)
        for g in range(4):idle(initial,g)
        write(root/'gpu_admission.json',dict(status='PASS',GPU_COUNT=4,GPU_UUIDS=uuids,probe=initial,shared_MAX_GPU_HOURS=20.0,protective_GPUh=19.5),True)
        write(root/'COORDINATOR_STARTED.json',dict(task_id=TASK,pid=os.getpid(),units=list(UNITS),source_manifest_sha256=manifest_sha,resource_budget_sha256=budget_sha,single_attempt_no_retry=True),True)
        qqueues={g:[f'gpu{g}_canary_initial',f'gpu{g}_canary_fresh'] for g in range(4)};squeues={g:[x for x in SCIENCE if DOC_GPU[int(x.split('_',2)[1][3:])]==g] for g in range(4)}
        for stage,queues in [('QUALIFICATION',qqueues),('SCIENCE',squeues)]:
            if stage=='SCIENCE':
                need(all(gates[x]=='PASS' for x in QUAL),'All four initial/fresh C6R gates required');budget=fresh_gate(root,budget)
                qualification(root,budget,manifest,ledger,gates,archive)
            last_storage=last_live=time.monotonic()
            while active or pending or any(queues.values()):
                need(not (root/'FIRST_FAILURE.json').exists(),'First failure prohibits any later dispatch');need(sha(root/'SOURCE_MANIFEST.json')==manifest_sha and sha(root/'RESOURCE_BUDGET.json')==budget_sha,'Frozen source/budget manifest changed')
                probe=inventory(uuids);now=time.monotonic();used=total_seconds(ledger,active,now)
                need(used<PROTECT_SECONDS and used+300*len(active)<HARD_SECONDS,'Protective19.5/shared hard20GPUh stop')
                for gpu in range(4):need(all(x['pid']==(active[gpu]['child'].pid if gpu in active else None) for x in probe['GPUs'][str(gpu)]['apps']),'Foreign compute process; stop only owned workers')
                if stage=='SCIENCE':
                    future,details=forecast(budget,[x for x in SCIENCE if gates[x]!='PASS'],ledger,active,now)
                    need(used+future+budget['additional_GPU_held_IO_reserve_seconds']+budget['safety_exit_reserve_seconds']<PROTECT_SECONDS,'All remaining10 trajectories cannot fit protective19.5GPUh')
                for gpu,item in list(pending.items()):
                    if item['future'].done():
                        result=item['future'].result();need(result['status']=='PASS','New-score archive failed');write(root/'archive'/('PASS_'+item['phase']+'.json'),result,True);del pending[gpu]
                for gpu in range(4):
                    if gpu in active or gpu in pending or not queues[gpu]:continue
                    phase=queues[gpu].pop(0);idle(probe,gpu);storage(root,budget);source_check(root,manifest)
                    worker_phase=phase.split('_',1)[1] if phase in QUAL else phase;output=root/'qualification_gpu'/f'gpu{gpu}' if phase in QUAL else root
                    need(gates[phase]=='NOT_RUN' and not (output/'analysis'/('RECEIPT_'+worker_phase+'.json')).exists(),'Registered unit cannot repeat')
                    command=[C.PYTHON,'-u',str(root/'control/worker_entry.py'),'--gpu',str(gpu),worker_phase];registry[phase]=command
                    log=(root/'logs'/(phase+'.log')).open('x');start=time.monotonic();child=subprocess.Popen(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',TRANSFORMERS_OFFLINE='1',HF_HUB_OFFLINE='1'),stdout=log,stderr=subprocess.STDOUT)
                    entry=dict(phase=phase,worker_phase=worker_phase,physical_GPU_ID=gpu,GPU_UUID=uuids[str(gpu)],pid=child.pid,command=command,dispatch_unix=time.time(),released=False,peak_sampled_NVML_MiB=probe['GPUs'][str(gpu)]['used_MiB'])
                    active[gpu]=dict(phase=phase,worker_phase=worker_phase,output=output,child=child,log=log,start=start,entry=entry,identity=None,last_progress=start,last_log_size=-1)
                    identity=C.process_identity(child.pid);active[gpu]['identity']=identity;entry['process_identity_at_dispatch']=identity;need(exact(identity,phase),'Exact child identity failed')
                for gpu,job in list(active.items()):
                    need(not (job['output']/'FIRST_FAILURE.json').exists(),'Worker first failure: '+job['phase'])
                    size=(root/'logs'/(job['phase']+'.log')).stat().st_size
                    if size!=job['last_log_size']:job['last_log_size']=size;job['last_progress']=time.monotonic()
                    need(time.monotonic()-job['last_progress']<1800,'30-minute no-progress watchdog')
                    job['entry']['peak_sampled_NVML_MiB']=max(job['entry']['peak_sampled_NVML_MiB'],probe['GPUs'][str(gpu)]['used_MiB'])
                    if job['child'].poll() is not None:
                        done=finish(gpu)
                        if done['phase'] in SCIENCE:pending[gpu]=dict(phase=done['phase'],future=pool.submit(archive_phase,root,budget,archive['python'],done['phase']))
                if now-last_storage>=5:storage(root,budget);last_storage=now
                if now-last_live>=10:live();last_live=now
                if active or pending:time.sleep(.5)
        need(all(gates[x]=='PASS' for x in UNITS),'Incomplete units cannot be complete');status='COMPLETE_AND_STOP'
    except BaseException as exc:
        status='FAIL_CLOSED_STOP';failure=dict(status=status,phase=phase,error=repr(exc),traceback=traceback.format_exc(),unix=time.time(),no_GPU_retry=True,active_phases=[x['phase'] for x in active.values()])
        if not (root/'FIRST_FAILURE.json').exists():write(root/'FIRST_FAILURE.json',failure,True)
        for gpu,job in list(active.items()):
            if job['child'].poll() is None:
                try:
                    ident=C.process_identity(job['child'].pid)
                    if job['identity'] is None and exact(ident,job['phase']):job['identity']=ident
                    need(job['identity'] is not None and ident['start_time_ticks']==job['identity']['start_time_ticks'] and exact(ident,job['phase']),'Cannot prove exact owned PID for cleanup')
                    job['child'].send_signal(signal.SIGTERM);job['entry']['stop_all_exact_SIGTERM_sent']=True
                except BaseException as e:job['entry']['stop_all_signal_error']=repr(e)
        for gpu in list(active):
            try:finish(gpu,True)
            except BaseException as e:failure.setdefault('cleanup_errors',[]).append(dict(gpu=gpu,error=repr(e)))
    finally:
        pool.shutdown(wait=True)
    final=safe_inventory(uuids) if uuids else dict(status='QUERY_FAILED',error='No admitted UUID map');now=time.monotonic()
    for gpu, job in active.items():
        if not any(row is job['entry'] for row in ledger['phases']):
            elapsed=max(0,now-job['start']);job['entry'].update(returncode=job['child'].poll(),reservation_seconds=elapsed,GPU_hours=elapsed/3600,released=False,reservation_duration_is_lower_bound_if_release_unverified=True,_last_accounted=now,cleanup_accounting_fallback=True)
            ledger['phases'].append(job['entry'])
    for row in ledger['phases']:
        if not row['released']:
            extension=max(0,now-row['_last_accounted']);row['reservation_seconds']+=extension;row['GPU_hours']+=extension/3600
            if final['status']=='PASS' and not any(x['pid']==row['pid'] for x in final['GPUs'][str(row['physical_GPU_ID'])]['apps']):row.update(released=True,release_confirmed_late=True,reservation_duration_is_lower_bound_if_release_unverified=False)
        row.pop('_last_accounted',None)
    own={x['pid'] for x in ledger['phases']};released=not active and (not ledger['phases'] or final['status']=='PASS' and not any(x['pid'] in own for y in final['GPUs'].values() for x in y['apps']))
    ledger['GPU_HOURS_USED']=math.fsum(x['GPU_hours'] for x in ledger['phases'])
    if not released or ledger['GPU_HOURS_USED']>=20:
        status='FAIL_CLOSED_STOP';failure=failure or dict(status=status,phase='FINAL_RELEASE_OR_BUDGET',error='Final ownGPU release or hard20 budget not qualified',no_GPU_retry=True)
        if not (root/'FIRST_FAILURE.json').exists():write(root/'FIRST_FAILURE.json',failure,True)
    if status=='COMPLETE_AND_STOP':
        try:
            cp=Path('checkpoints/CURRENT.pt');need((root/cp).is_file(),'Final current checkpoint absent');rec=archive_assets(root,budget,archive['python'],'FINAL_CHECKPOINT',[cp]);write(root/'archive/FINAL_CHECKPOINT_LOCATION.json',rec,True)
        except BaseException as exc:
            status='FAIL_CLOSED_STOP';failure=dict(status=status,phase='FINAL_CHECKPOINT_ARCHIVE',error=repr(exc),no_GPU_retry=True)
            if not (root/'FIRST_FAILURE.json').exists():write(root/'FIRST_FAILURE.json',failure,True)
    ledger.update(final_status=status,final_GPU_release_verified=released,final_GPU_release_probe=final,ended_unix=time.time(),GPU_UUIDS=uuids)
    write(root/'GPU_RESOURCE_LEDGER.json',ledger,True)
    result=dict(task_id=TASK,status=status,COORDINATOR_STATUS=status,gates=gates,GPU_COUNT=4,GPU_UUIDS=uuids,GPU_HOURS_USED=ledger['GPU_HOURS_USED'],MAX_GPU_HOURS=20.0,PROTECTIVE_STOP_GPUH=19.5,GPU_release_verified=released,GPU_release_query=final,no_further_GPU_work=True,source_manifest_sha256=manifest_sha,resource_budget_sha256=budget_sha,coordinator_failure=failure,first_failure_file='FIRST_FAILURE.json' if (root/'FIRST_FAILURE.json').exists() else None)
    write(root/'FINAL_GPU_STATUS.json',result,True);return result


def self_test():
    need(len(UNITS)==18 and len(SCIENCE)==10 and len(QUAL)==8,'Wrong unit counts')
    ledger=dict(phases=[dict(phase='gpu0_canary_initial',reservation_seconds=10)])
    active={0:dict(phase=SCIENCE[0],start=0),1:dict(phase=SCIENCE[2],start=2)}
    need(total_seconds(ledger,active,5)==18,'Four-card sum algebra failed')
    budget=dict(phase_predictions_seconds={p:100 for p in SCIENCE});future,_=forecast(budget,SCIENCE,ledger,active,5)
    need(total_seconds(ledger,active,5)+future==1010,'Active elapsed double counted')
    future,details=forecast(budget,list(SCIENCE[1:]),dict(phases=[dict(phase=SCIENCE[0],reservation_seconds=120)]),{},0)
    need(all(v==120 for p,v in details['phase_predictions_seconds'].items() if p.endswith('C6R_C128')),'Observed overrun did not raise only samecodec')
    return dict(status='PASS',CPU_ONLY=True,GPU_work_performed=False,scope='Pure resource algebra only; archive and physical gates require real admission',units=18,formal=10,qualifications=8)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(ROOT));p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:print(json.dumps(self_test()))
    else:sys.exit(0 if run(a.root)['status']=='COMPLETE_AND_STOP' else 1)
