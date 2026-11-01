"""Manual, one-shot, independent CPU terminal audit and small-file delivery.

Default invocation is inert. Prepare an independent tool manifest only after the
production SOURCE/PERMIT are final, then explicitly request terminal delivery.
No downloaded code is executed. No GPU/model/scorer/finisher/signal is invoked.
A genuine failure after the exclusive claim retains all evidence and forbids an
automatic retry. The original STOP and its source/seals are read-only.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import unicodedata

sys.dont_write_bytecode = True
TASK = 'QWEN_GDN_SYM8_G32_WSCALE_ROWBAL_AIME26_SEED1_V1'
REVISION = TASK + '_CUPTI_REPAIR_20261025'
PRIOR368_REVISION=TASK+'_REUSE48_ARCHIVE29_20261025'
PRIOR368_55='/dev/shm/qwen_wscale_rowbal_reuse48_archive29_20261025'
PRIOR368_29='/data/zypan/qwen_gdn_wscale_rowbal_archive/'+PRIOR368_REVISION
PRIOR368_SOURCE='0ef900e88640bd255bdf29f6da8efe5b3d2e05b00c1c32fb74027e27a3c33d50'
PRIOR368_PERMIT='bb4174d7706c6d9fbb25963f55d3c9deead006898b587630f6eb93d742537385'
PRIOR368_MANIFEST='5afb360dc72b2115e0912e43318c30c276bb0950d8514152e91132211679757c'
BYTEVIEW_REVISION=TASK+'_BYTEVIEW_REPAIR_20261025'
BYTEVIEW55='/tmp/'+BYTEVIEW_REVISION
BYTEVIEW29='/data/zypan/qwen_gdn_wscale_rowbal_archive/'+BYTEVIEW_REVISION
BYTEVIEW_SOURCE='05901486d72e6f62aba67e61961d6d62043dabef14340b46034a072a004ce5e1'
BYTEVIEW_PERMIT='3b03a13eafac5d8997b141f5d099b39f682aee10a1a57bedc79d33dc362c611c'
FRESH_IDS=tuple('GPU1_'+m+'_MATCHED4096' for m in ('base','W','R','RW'))
PARENT_REVISION = TASK + '_REDUCTION_REPAIR_20261025'
PARENT_SOURCE = '6e87495d2dac7ccc7d97363256be3862ab7a52159271b848a58f06e2eabd1f4e'
ROOT = Path(__file__).resolve().parents[1]
REMOTE = '/dev/shm/qwen_wscale_rowbal_cupti_repair_20261025'
ARCHIVE = '/data/zypan/qwen_gdn_wscale_rowbal_archive/' + REVISION
ORIGIN55 = '/tmp/' + TASK
ORIGIN29 = '/data/zypan/qwen_gdn_wscale_rowbal_archive/' + TASK
ORIGIN_SOURCE = 'ee2c01ff3a4963ba76a3df813bf22640de0c2de085ea82a2a297e067ac9f33b7'
REUSE_STATUS = 'PASS_REUSED_FROZEN_NONAIME_CALIBRATION_DUAL_HOST_FRESH_SHA_NO_RECOLLECT_NO_RESELECT'
PYTHON55 = '/data/ydai/miniconda3/envs/bitdecode/bin/python'
UUIDS = {1: 'GPU-596e6591-b065-bb91-fd35-a1e90ba8f9e3',
         2: 'GPU-b7ed673d-18ba-39ca-d7bc-c846af4f6b31',
         3: 'GPU-30b07aba-a728-7649-4899-e09d33f8bc75'}
CONTROLS = ('SHA256_MANIFEST.json', 'SHA256SUMS.txt',
            'FINAL_SEAL_RECEIPT.json', 'FINAL_ARCHIVE_VERIFICATION.json')
TEXT = {'.json', '.jsonl', '.log', '.txt', '.md', '.py', '.sh', '.csv',
        '.yaml', '.yml', '.toml', '.patch', '.c', '.h', '.cu', '.cpp'}
TEXT_IR = {'.ttir', '.ttgir', '.llir', '.ptx'}
OUTPUT_BOUND = 2 * 1024**3
SMALL_PREPARATION_BOUND = 32 * 1024**2
TOOL_MANIFEST = ROOT / 'operations/REUSE48_MONITOR_BINDING.json'
CLAIM = ROOT / 'operations/LOCAL_REUSE48_TERMINAL_DELIVERY_CLAIM.json'
FAILURE = ROOT / 'operations/LOCAL_REUSE48_TERMINAL_DELIVERY_FAILURE.json'
RESULT = ROOT / 'operations/LOCAL_REUSE48_TERMINAL_DELIVERY_VERIFICATION.json'
DEST = ROOT / 'REUSE48_DELIVERY_FROM55'
TRANSPORT = ROOT.parent / 'QWEN_GDN_PACE48_AIME26_SEED1_60_V1/operations/remote.py'
TRANSPORT_SHA='db0ea71612d633b0fc33feef1d95c42e273ab2bd9ffc971b8d7e4ddc97f74f40'


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def safe(name):
    need(isinstance(name, str) and '\x00' not in name and '\\' not in name,
         'Unsafe logical path')
    p = PurePosixPath(name)
    need(name and name != '.' and not p.is_absolute() and '..' not in p.parts and str(p) == name,
         'Unsafe relative path: ' + name)
    return name


def stable_sha(path, forbid_symlink=True):
    p = Path(path)
    need(p.is_file() and (not forbid_symlink or not p.is_symlink()),
         'Missing/nonregular/symlink file: ' + str(p))
    a = p.stat()
    h = hashlib.sha256()
    with p.open('rb') as f:
        opened = os.fstat(f.fileno())
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
        closed = os.fstat(f.fileno())
    z = p.stat()
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    need(identity(a) == identity(opened) == identity(closed) == identity(z),
         'File changed during read: ' + str(p))
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode()


def verify_rows(root, rows):
    root = Path(root)
    seen = set()
    total = 0
    for row in rows:
        n = safe(row['path'])
        need(n not in seen, 'Duplicate manifest path: ' + n)
        seen.add(n)
        p = root / n
        need(all(not q.is_symlink() for q in (p, *p.parents) if q != root.parent),
             'Symlink in manifest path: ' + n)
        need(type(row['bytes']) is int and row['bytes'] >= 0,
             'Invalid byte count: ' + n)
        need(p.stat().st_size == row['bytes'] and stable_sha(p) == row['sha256'],
             'SHA/bytes mismatch: ' + str(p))
        total += row['bytes']
    return total


def classify_terminal(plan, jobs, session, ledger, legal):
    """Count actual unique exit0/releasePASS units; never calculate correctness."""
    qualifications = plan['qualification_units']
    need(plan['qualification_count'] == len(qualifications) == 52,
         'Expected exact frozen 52 qualification roster')
    need(len(jobs) == 90 and all(sum(j['condition'] == m for j in jobs) == 30
                              for m in ('W', 'R', 'RW')), 'Expected 30x3 formal roster')
    allqids = {u['id'] for u in qualifications}
    qids = set(FRESH_IDS)
    fids = {j['id'] for j in jobs}
    need(len(allqids) == 52 and len(qids) == 4 and qids <= allqids and len(fids) == 90 and not qids & fids,
         'Duplicate or overlapping unit roster')
    byid = {u['id']: u for u in qualifications + jobs if u['id'] in qids or u['id'] in fids}
    intervals = ledger['intervals']
    ids = [x['unit'] for x in intervals]
    need(len(ids) == len(set(ids)) and set(ids) <= set(byid),
         'Duplicate/unplanned resource interval')
    for x in intervals:
        u = byid[x['unit']]
        need(x['release'] == 'PASS' and type(x['exit_returncode']) is int,
             'Actual exit/release unresolved: ' + x['unit'])
        need(x['gpu'] == u['gpu'] in UUIDS and u['gpu_uuid'] == UUIDS[u['gpu']],
             'Resource/device binding differs')
        identity = x.get('identity')
        need(isinstance(identity, dict) and type(identity.get('pid')) is int and
             type(identity.get('starttick')) is int and identity.get('cmdline'),
             'Exact child identity not recorded: ' + x['unit'])
    expected_legal = {x['unit'] for x in intervals if x['exit_returncode'] == 0}
    need(set(legal) == expected_legal, 'Exit0 results have not all been validated')
    for label, completed in (('session', session['completed']),
                             ('ledger', ledger['completed_units'])):
        need(len(completed) == len(set(completed)) and set(completed) == expected_legal,
             label + ' completed set differs from legal exit0 units')
    need(session.get('reused_qualification_count') in (0,48), 'Session reused count invalid')
    need(session['own_all_UUID_release'] is True, 'Session release unresolved')
    if set(ids) & fids:
        need(qids <= expected_legal, 'A formal dispatch lacks all52 legal qualification units')
    full = session['status'] == 'GPU_COMPLETE_REUSE48_PLUS4_QUALIFICATION_90_FORMAL'
    need(full or session['status'] == 'FAIL_CLOSED_STOP', 'Not a terminal session')
    if full:
        need(expected_legal == set(byid), 'Full session lacks exact 52+90 units')
    return dict(scientific_status='FORMAL_COMPLETE_90_90_AND_STOP' if full else
                'PARTIAL_RESULTS_PRESERVED_AND_STOP',
                qualification_PASS=48+len(expected_legal & qids), reused_qualification=48, fresh_qualification=len(expected_legal & qids),
                formal_complete=len(expected_legal & fids),
                formal_by_method={m: sum(j['id'] in expected_legal and j['condition'] == m
                                         for j in jobs) for m in ('W', 'R', 'RW')},
                dispatched_intervals=len(intervals), full=full)


def validate_formal(wrapper, job, root):
    """Mirror frozen payload identity/count invariants, without a scorer/tokenizer."""
    row = wrapper['result']
    need(row['status'] == 'COMPLETE' and not row['nonfinite'] and
         row['runtime_error'] is None, 'Incomplete/nonfinite formal result')
    for key in ('condition', 'problem_id', 'seed', 'gpu', 'gpu_uuid',
                'input_ids_sha256', 'input_tokens'):
        need(row[key] == job[key], 'Formal identity differs: ' + key)
    ts = row['generated_token_ids']
    n = len(ts)
    need(0 < n <= 81920 and all(type(t) is int and 0 <= t < 248320 for t in ts)
         and row['generated_token_count'] == n and row['generated_token_ids_sha256'] ==
         hashlib.sha256(canonical(ts)).hexdigest(), 'Formal token payload differs')
    stops = row['stop_token_ids']
    eos = ts[-1] in stops
    need(not any(t in stops for t in ts[:-1]) and row['finish_reason'] ==
         ('eos' if eos else 'length') and (eos or n == 81920), 'Illegal EOS/length')
    total = row['input_tokens'] + n - 1
    q = row['quantizer_audit']
    need(row['actual_model_forward_calls'] == q['tokens'] == total and
         q['qdq_layer_updates'] == 24 * total and
         row['prompt_layer_QDQ'] == 24 * row['input_tokens'] and
         row['decode_layer_QDQ'] == 24 * (n - 1) and
         len(row['writebacks_by_layer']) == 24 and
         all(v == total for v in row['writebacks_by_layer'].values()) and
         row['sampler_calls'] == n, 'Formal forward/QDQ coverage differs')
    expected = 24 * sum(t > 0 and (t in range(1, 128) or t % 2048 == 0)
                        for t in range(total))
    need(q['next_consumption_checks'] == expected, 'Actual next consumption differs')
    for key, name in (('sampler_source_sha256', 'protocol/sampler.py'),
                      ('scorer_source_sha256', 'protocol/aime26_scorer_v4.py'),
                      ('dataset_freeze_sha256', 'AIME26_DATASET_FREEZE.json')):
        need(row[key] == stable_sha(Path(root) / name), 'Formal source differs: ' + key)


def select_small(rows, source_paths):
    """Select sealed scientific text and SOURCE-pinned textual compiler evidence."""
    selected = []
    skipped = []
    output_bytes = 0
    preparation_bytes = 0
    for x in rows:
        n = safe(x['path'])
        parts = PurePosixPath(n).parts
        suffix = PurePosixPath(n).suffix.lower()
        source_ir = suffix in TEXT_IR and n in source_paths
        forbidden = ((n.startswith('calibration/raw/') and suffix not in TEXT) or
                     any(p in ('site-packages', 'cpu_env', '.venv', 'venv', 'envs')
                         for p in parts) or
                     (n.startswith('engineering/triton_cache/') and not source_ir))
        textual = suffix in TEXT or source_ir
        if forbidden or not textual:
            need(n not in source_paths or suffix in {'.npz', '.pt', '.pth', '.bin'},
                 'Unclassified frozen SOURCE file needs explicit review: ' + n)
            skipped.append(dict(x, reason='REMOTE_ONLY_TENSOR_ENV_BINARY_OR_UNSELECTED_TYPE'))
            continue
        is_output = parts[0] in ('execution', 'formal') or n == 'FORMAL_GENERATIONS_90.jsonl'
        bound = OUTPUT_BOUND if is_output else SMALL_PREPARATION_BOUND
        need(type(x['bytes']) is int and 0 <= x['bytes'] <= bound,
             'Selected text exceeds finite per-file bound: ' + n)
        if is_output:
            output_bytes += x['bytes']
        else:
            preparation_bytes += x['bytes']
        selected.append(x)
    need(output_bytes <= OUTPUT_BOUND, 'Selected scientific output exceeds frozen 2GiB bound')
    need(len(selected) <= 100000, 'Unexpected small-file roster size')
    return selected, skipped, dict(scientific_output_bytes=output_bytes,
                                   source_and_preparation_bytes=preparation_bytes,
                                   finite_total_bound=OUTPUT_BOUND + preparation_bytes)


def audit_root(cfg):
    """Read-only remote CPU audit. No imports from the research execution tree."""
    R = Path(cfg['root'])
    need(str(R) == cfg['expected_root'], 'Unexpected audit root')
    for name, expected in cfg['pins'].items():
        need(stable_sha(R / safe(name)) == expected, 'Pinned file differs: ' + name)
    source = read(R / 'SOURCE_FREEZE.json')
    need(source['count'] == len(source['files']) > 0, 'Invalid dynamic SOURCE roster')
    verify_rows(R, source['files'])
    source_paths = {x['path'] for x in source['files']}
    for u in read(R/'GPU_EXECUTION_PLAN.json')['qualification_units']:
        if u['id'] not in FRESH_IDS:need(not (R/'execution'/u['id']).exists(),'Reused evidence appears in current execution')
    need(read(R/'QUALIFICATION_REUSE_ADMISSION.json')['qualification_reused']==48,'Frozen reuse48 missing')
    manifest = read(R / 'SHA256_MANIFEST.json')
    need(manifest['task_id'] == TASK and manifest['count'] == len(manifest['files']),
         'Science manifest identity/count differs')
    total = verify_rows(R, manifest['files'])
    need(total == manifest['total_bytes'], 'Science byte total differs')
    sealed = {x['path']: x for x in manifest['files']}
    need(source_paths <= set(sealed), 'Frozen SOURCE is not fully included in seal')
    expected_sums = ''.join(x['sha256'] + '  ' + x['path'] + '\n' for x in manifest['files'])
    need((R / 'SHA256SUMS.txt').read_bytes() == expected_sums.encode(), 'SHA256SUMS bytes differ')
    seal = read(R / 'FINAL_SEAL_RECEIPT.json')
    verification = read(R / 'FINAL_ARCHIVE_VERIFICATION.json')
    manifest_sha = stable_sha(R / 'SHA256_MANIFEST.json')
    need(seal['status'] == 'PASS_FINAL_SEAL' and verification['status'] ==
         'PASS_INDEPENDENT29_ALL_FILE_SHA_BYTES_ATOMIC_SEAL', 'Final seal not PASS')
    need(seal['manifest_sha256'] == verification['manifest_sha256'] == manifest_sha
         and verification['count'] == manifest['count'] and
         verification['total_bytes'] == total and verification['archive_root'] == ARCHIVE,
         'Final seal manifest/count/root binding differs')
    # The authoritative hash inside seal binds the actual29 verification bytes.
    # Each host's bytes are retained independently; do not reserialize a receipt.
    need(not (R / 'FINALIZER_FAILURE.json').exists(), 'Actual finalizer failure exists')
    session = read(R / 'SESSION_RESULT.json')
    ledger = read(R / 'RESOURCE_LEDGER.json')
    plan = read(R / 'GPU_EXECUTION_PLAN.json')
    jobs = read(R / 'FORMAL_JOB_MANIFEST.json')['jobs']
    byid = {u['id']: u for u in plan['qualification_units'] + jobs if u['id'] in FRESH_IDS or u['id'] in {j['id'] for j in jobs}}
    fids = {j['id'] for j in jobs}
    legal = []
    for interval in ledger['intervals']:
        name = safe(interval['unit'])
        need('/' not in name and name in byid, 'Unknown/unsafe dispatched unit')
        formal = name in fids
        target = R / ('formal' if formal else 'execution') / name
        need(read(target / 'RESOURCE_INTERVAL.json') == interval, 'Interval copy differs')
        dispatch = read(target / 'DISPATCH.json')
        need(dispatch['unit'] == name and dispatch['formal'] is formal and
             dispatch['gpu'] == byid[name]['gpu'] and dispatch['gpu_uuid'] ==
             UUIDS[dispatch['gpu']] and dispatch['permit_sha256'] == cfg['pins']['GPU_PERMIT.json'],
             'Dispatch binding differs')
        need(read(target / 'STARTED.json')['identity'] == interval['identity'],
             'Actual STARTED/ledger identity differs')
        argv = interval['identity']['cmdline']
        expected_script = REMOTE + '/scripts/gpu_unit_worker.py'
        need(expected_script in argv and '--root' in argv and '--unit' in argv and
             argv[argv.index('--root') + 1] == REMOTE and
             argv[argv.index('--unit') + 1] == name and ('--formal' in argv) is formal,
             'Recorded child cmdline is outside the exact new unit/root')
        if interval['exit_returncode'] != 0:
            continue
        wrapper = read(target / 'RESULT.json')
        payload = dict(wrapper)
        digest = payload.pop('payload_sha256')
        need(hashlib.sha256(canonical(payload)).hexdigest() == digest and
             wrapper['status'] == 'PASS' and wrapper['unit'] == name and
             wrapper['identity'] == interval['identity'] and
             wrapper['source_freeze_sha256'] == cfg['pins']['SOURCE_FREEZE.json'] and
             wrapper['permit_sha256'] == cfg['pins']['GPU_PERMIT.json'] and
             wrapper['gpu'] == byid[name]['gpu'] and
             wrapper['gpu_uuid'] == UUIDS[wrapper['gpu']], 'Worker payload identity differs')
        if formal:
            validate_formal(wrapper, byid[name], R)
            for receiptname in ('CUPTI_PRELOAD_ADMISSION.json','PROFILE_COLLECTION_COMPLETED.json','CUPTI_PROFILE_DETACH.json','CUPTI_PROFILE_DETACH_EVENTS.jsonl'):
                need(not (target/receiptname).exists(),'Formal trajectory unexpectedly entered profiling lifecycle')
        if not formal:
            unit=byid[name]
            strict_unit(R,unit,source_sha256=cfg['pins']['SOURCE_FREEZE.json'],permit_sha256=cfg['pins']['GPU_PERMIT.json'],triton_manifest_sha256=stable_sha(R/'analysis/TRITON_RUNTIME_FRESH_SHA.json'),triton_task_root=REMOTE)
            verify_profile_lifecycle(R,unit,cfg['pins'])
        legal.append(name)
    byteview=audit_byteview_origin(cfg,R)
    prior368=audit_prior368_origin(cfg)
    counts = classify_terminal(plan, jobs, session, ledger, legal)
    durable=[]
    for interval in ledger['intervals']:
        name=interval['unit'];target=R/('formal' if name in fids else 'execution')/name;rp=target/'DURABLE29_RECEIPT.json'
        if rp.exists():
            receipt=read(rp);need(receipt['status']=='PASS_DURABLE29_CLOSED_UNIT' and receipt['unit']==name and receipt['actual_exit_returncode']==interval['exit_returncode'] and receipt['release']=='PASS' and receipt['source_freeze_sha256']==cfg['pins']['SOURCE_FREEZE.json'] and receipt['permit_sha256']==cfg['pins']['GPU_PERMIT.json'],'Current durable29 receipt differs')
            mpath=target/'DURABLE29_MANIFEST.json';need(stable_sha(mpath)==receipt['manifest_sha256'],'Durable unit manifest changed');dm=read(mpath);verify_rows(target,dm['files'])
            if cfg['role']=='29':
                live=Path('/data/zypan/qwen_gdn_wscale_rowbal_archive/LIVE_'+REVISION)/'units'/name;need(stable_sha(live/'DURABLE29_MANIFEST.json')==receipt['manifest_sha256'],'Live29 durable manifest differs');verify_rows(live,dm['files']);need(read(live/'DURABLE29_RECEIPT.json')==receipt,'Live29 receipt bytes differ')
            durable.append(name)
    if counts['full']:need(len(durable)==94,'Full result requires all94 independent durable29 units')
    counts['durable_units']=len(durable)
    need(manifest['status'] == read(R / 'FINAL_TASK_STATUS.json')['status'] ==
         seal['scientific_status'] == counts['scientific_status'], 'Terminal status differs')
    if any(x['unit'] in fids for x in ledger['intervals']):
        q = read(R / 'GPU_CODEC_QUALIFICATION.json')
        auth = read(R / 'AUTHORIZATION_FORMAL.json')
        need(q['status'] == 'PASS_ALL_NEW_METHODS_THREE_UUID_REQUIRED_GATES' and
             q['inspection']['count'] == 52 and len(q['units']) == 52 and
             set(q['units']) == {u['id'] for u in plan['qualification_units']} and
             q['source_freeze_sha256'] == cfg['pins']['SOURCE_FREEZE.json'],
             'Full qualification receipt differs')
        need(auth['status'] == 'PASS_ALL10_GATES_ALLOW_FORMAL_ONCE' and
             auth['source_freeze_sha256'] == cfg['pins']['SOURCE_FREEZE.json'] and
             auth['permit_sha256'] == cfg['pins']['GPU_PERMIT.json'] and
             auth['GPU_qualification_sha256'] == stable_sha(R / 'GPU_CODEC_QUALIFICATION.json')
             and auth['jobs_sha256'] == stable_sha(R / 'FORMAL_JOB_MANIFEST.json'),
             'Formal authorization differs')
    if counts['full']:
        need(not (R / 'FIRST_FAILURE.json').exists(), 'Full result contains a real failure')
        need(read(R / 'SCORE_PASS_0.json') == read(R / 'SCORE_PASS_1.json') ==
             read(R / 'FROZEN_V4_SCORES.json'), 'Existing two CPU score receipts disagree')
        need(read(R / 'FORMAL_GENERATIONS_COMPLETE.json')['count'] == 90,
             'Formal completion receipt differs')
    else:
        need((R / 'FIRST_FAILURE.json').is_file(), 'Partial STOP lacks its real failure')
    reuse = read(R / 'CALIBRATION_REUSE_ADMISSION.json')
    need(reuse['status'] == REUSE_STATUS and reuse['origin_source_sha256'] == ORIGIN_SOURCE
         and reuse['origin_source_root'] == ORIGIN55 and reuse['no_reselection'] is True
         and reuse['selected_a_b'] == [.5, .25] and reuse['new_formal_credit'] == 0
         and reuse['new_qualification_credit'] == 0, 'Frozen calibration reuse differs')
    origin = Path(cfg['origin_root'])
    prior = cfg['origin_pins']
    for name, digest in prior.items():
        need(stable_sha(origin / safe(name)) == digest, 'Original sealed bytes changed: ' + name)
    old_source = read(origin / 'SOURCE_FREEZE.json')
    need(old_source['count'] == len(old_source['files']) == 100, 'Origin SOURCE100 differs')
    verify_rows(origin, old_source['files'])
    origin_manifest = read(origin / 'SHA256_MANIFEST.json')
    origin_rows = {x['path']: x for x in origin_manifest['files']}
    required_origin_metadata = ['CALIBRATION_SOURCE_FREEZE.json',
                               'calibration/RAW_CLONE_VERIFICATION.json',
                               'calibration/FP_COLLECTION_MANIFEST.json']
    verify_rows(origin, [origin_rows[n] for n in required_origin_metadata])
    old_cal = read(origin / 'CALIBRATION_SOURCE_FREEZE.json')
    need(len(old_cal['files']) == 31, 'Origin calibration SOURCE31 differs')
    verify_rows(origin, old_cal['files'])
    raw = read(origin / 'calibration/RAW_CLONE_VERIFICATION.json')
    need(raw['count'] == len(raw['files']) == reuse['hosts'][cfg['role']]['raw_clone_count']
         and raw['collection_manifest_sha256'] == stable_sha(origin / 'calibration/FP_COLLECTION_MANIFEST.json'),
         'Origin raw catalog identity differs')
    verify_rows(origin,origin_manifest['files'])
    raw_bytes = verify_rows(origin, raw['files'])
    need(raw_bytes == raw['bytes'] == reuse['hosts'][cfg['role']]['raw_clone_bytes'],
         'Origin raw bytes differ')
    # The immediately preceding reduction STOP is a separate frozen lineage,
    # never current qualification/formal credit or a reused delivery success.
    parent = Path(cfg['parent_root'])
    for name, digest in cfg['parent_pins'].items():
        need(stable_sha(parent / safe(name)) == digest, 'Parent reduction STOP changed: ' + name)
    parent_source = read(parent / 'SOURCE_FREEZE.json')
    need(stable_sha(parent / 'SOURCE_FREEZE.json') == PARENT_SOURCE and
         parent_source['count'] == len(parent_source['files']), 'Parent reduction SOURCE identity differs')
    verify_rows(parent, parent_source['files'])
    parent_seal = read(parent / 'FINAL_SEAL_RECEIPT.json')
    need(parent_seal['status'] == 'PASS_FINAL_SEAL' and
         parent_seal['scientific_status'] == 'PARTIAL_RESULTS_PRESERVED_AND_STOP',
         'Parent true STOP must remain STOP')
    # Preserve parent299 full archive, not only its SOURCE.
    parent_manifest=read(parent/'SHA256_MANIFEST.json');verify_rows(parent,parent_manifest['files'])
    need(parent_seal['manifest_sha256']==stable_sha(parent/'SHA256_MANIFEST.json'),'Parent manifest seal differs')
    # Two-host dependency snapshots are included in all SOURCE/science hashes.
    # Only55 runs these absolute model/native paths;29 is not mislabelled an environment replica.
    external = dict(status='NOT_APPLICABLE_ARCHIVE_IS_NOT_EXECUTION_ENVIRONMENT', count=0, bytes=0)
    cupti_binding=read(R/'analysis/CUPTI_LIFECYCLE_BINDING.json');need(cupti_binding['status']=='PASS_FRESH_EXACT_EXISTING_CUPTI_ELF_AND_HEADER_BINDINGS' and len(cupti_binding['files'])==1,'Explicit CUPTI binding missing')
    cupti=cupti_binding['files'][0];need(Path(cupti['path']).name=='libcupti.so.12','Exactexisting CUPTI ELF basename differs')
    if cfg['role'] == '55':
        for row in cupti_binding['files']+[cupti_binding['header']]:
            cp=Path(row['path']);st=cp.stat();need(cp.is_absolute() and cp.is_file() and not cp.is_symlink() and st.st_size==row['bytes'] and stable_sha(cp)==row['sha256'],'Fresh CUPTI ELF/header SHAbytes differs')
            for key,field in [('inode','st_ino'),('device','st_dev'),('mtime_ns','st_mtime_ns')]:
                if key in row:need(getattr(st,field)==row[key],'CUPTI pinned stat differs')
        base = R / 'inherited/group32/inherited/historical_port_payload/analysis/environment'
        bindings = {}
        for name in ('PROSPECTIVE_MODEL_AND_NATIVE_BINDINGS.json',
                     'CPU_FIXTURE_CRITICAL_NATIVE_BINDINGS.json'):
            obj = read(base / name)
            for key in ('model_weight_shards', 'model_small_files', 'native_library_inventory',
                        'native_library_rows', 'torch_python_files', 'external_source_files'):
                for row in obj.get(key, []):
                    if all(k in row for k in ('path', 'bytes', 'sha256')):
                        previous = bindings.setdefault(str(Path(row['path']).resolve()), row)
                        need((previous['bytes'], previous['sha256']) == (row['bytes'], row['sha256']),
                             'Conflicting external dependency binding')
            row = obj.get('python_executable')
            if isinstance(row, dict) and all(k in row for k in ('path', 'bytes', 'sha256')):
                previous = bindings.setdefault(str(Path(row['path']).resolve()), row)
                need((previous['bytes'], previous['sha256']) == (row['bytes'], row['sha256']),
                     'Conflicting Python binding')
        model = read(R / 'analysis/MODEL_TOKENIZER_FRESH_SHA.json')
        for row in model['files']:
            st=Path(row['path']).stat();need((st.st_size,st.st_ino,st.st_mtime_ns)==(row['bytes'],row['inode'],row['mtime_ns']),'Original model inode/stat differs')
        need(model['count'] == len(model['files']) == 17, 'Actual model/tokenizer17 roster differs')
        triton = read(R / 'analysis/TRITON_RUNTIME_FRESH_SHA.json')
        triton_rows = triton['files'] + triton['distribution_metadata'] + triton['host_c_compiler']
        for row in model['files'] + triton_rows:
            need(Path(row['path']).is_absolute(), 'External dependency must be absolute')
            previous = bindings.setdefault(str(Path(row['path']).resolve()), row)
            need((previous['bytes'], previous['sha256']) == (row['bytes'], row['sha256']),
                 'Conflicting model17/compiler dependency binding')
        need(bindings, 'No absolute execution dependencies found')
        for name, row in bindings.items():
            need(Path(name).stat().st_size == row['bytes'] and
                 stable_sha(name, forbid_symlink=False) == row['sha256'],
                 'Actual execution dependency SHA/bytes differs: ' + name)
        external = dict(status='PASS_FRESH_ALL_BOUND_ABSOLUTE_DEPENDENCY_SHA_BYTES',
                        count=len(bindings), bytes=sum(x['bytes'] for x in bindings.values()),
                        model_tokenizer_full_fresh_SHA_files=17,
                        model17_manifest_sha256=stable_sha(R / 'analysis/MODEL_TOKENIZER_FRESH_SHA.json'),
                        Triton_package_metadata_compiler_files=len(triton_rows),
                        Triton_manifest_sha256=stable_sha(R / 'analysis/TRITON_RUNTIME_FRESH_SHA.json'),
                        roster_sha256=hashlib.sha256(canonical(sorted(
                            (p, x['bytes'], x['sha256']) for p, x in bindings.items()))).hexdigest())
    controls = [dict(path=n, bytes=(R / n).stat().st_size, sha256=stable_sha(R / n))
                for n in CONTROLS]
    selected, skipped, bounds = select_small(manifest['files'], source_paths)
    need(not set(CONTROLS) & {x['path'] for x in selected}, 'Circular control roster')
    return dict(status='PASS_FRESH_INDEPENDENT_SOURCE_SCIENCE_CALIBRATION_SHA_BYTES',
                role=cfg['role'], root=str(R), source_count=len(source['files']),byteview_reuse48=byteview,prior368_true_STOP=prior368,explicit_CUPTI_binding_manifest_sha256=stable_sha(R/'analysis/CUPTI_LIFECYCLE_BINDING.json'),
                science_count=len(manifest['files']), science_bytes=total,
                manifest_sha256=manifest_sha, source_freeze_sha256=cfg['pins']['SOURCE_FREEZE.json'],
                permit_sha256=cfg['pins']['GPU_PERMIT.json'], counts=counts,
                seal=seal, archive_verification=verification, controls=controls,
                origin=dict(source_count=100, calibration_source_count=31,
                            raw_count=len(raw['files']), raw_bytes=raw_bytes,
                            original_seal_bytes_unchanged=True, pins=prior),
                parent_reduction_lineage=dict(status='PASS_FRESH_PARENT_SOURCE_AND_ORIGINAL_SEAL_BYTES',
                    root=str(parent), source_count=len(parent_source['files']),
                    pins=cfg['parent_pins'], scientific_status=parent_seal['scientific_status'],
                    current_qualification_credit=0, current_formal_credit=0),
                execution_dependencies=external,
                small_files=selected + controls, skipped_remote_only=skipped,
                download_bounds=bounds, GPU_calls=0, model_loads=0, signals=0,
                scoring_calls=0, finisher_calls=0, remote_writes=0)


def new(path, value):
    p = Path(path)
    with p.open('x') as f:
        json.dump(value, f, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def import_local(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def local_pins():
    need(ROOT.name==REVISION,'Exact newlocal reuse48 root required')
    meta=read(ROOT/'ROOT.json');need(meta['remote_root']==REMOTE and meta['archive_root']==ARCHIVE,'Newroot identity differs')
    source=read(ROOT/'SOURCE_FREEZE.json');need(source['count']==len(source['files'])>0,'Dynamic current source missing');verify_rows(ROOT,source['files'])
    names=('SOURCE_FREEZE.json','GPU_PERMIT.json','ROOT.json','GPU_EXECUTION_PLAN.json','FORMAL_JOB_MANIFEST.json','QUALIFICATION_REUSE_ADMISSION.json','operations/HUMAN_REUSE48_ARCHIVE29_AUTHORIZATION.json','CALIBRATION_REUSE_ADMISSION.json')
    pins={n:stable_sha(ROOT/n) for n in names};p=read(ROOT/'GPU_PERMIT.json');need(p['ALLOW_GPU']=='YES' and p['execution_root']==REMOTE and p['source_freeze_sha256']==pins['SOURCE_FREEZE.json'] and p['physical_indices']==[1,2,3] and p['devices']==list(UUIDS.values()) and p['formal_new_samples']==90,'Finalnewpermit scope differs')
    need(stable_sha(TRANSPORT)==TRANSPORT_SHA,'Pinned readonlytransport differs')
    return pins


def prepare_tool_manifest():
    pins=local_pins();receipt=ROOT/'operations/OFFLINE_LAUNCH_COMMAND_PROCESS_RECEIPT_LOCAL.json';need(receipt.is_file(),'Actual unique launch receipt required before monitorbinding')
    old=ROOT.parent/BYTEVIEW_REVISION/'operations/TERMINAL_DELIVERY_TOOL_MANIFEST.json';prior=read(old)
    files=('operations/read_reuse48_snapshot_once.py','operations/deliver_reuse48_terminal_cpu_once.py','tests/reuse48_terminal_cpu.py')
    obj=dict(status='FROZEN_REUSE48_MONITOR_AND_TERMINAL_TOOL',task_id=TASK,remote_root=REMOTE,execution_root=REMOTE,archive_root=ARCHIVE,local_root=str(ROOT),source_sha256=pins['SOURCE_FREEZE.json'],permit_sha256=pins['GPU_PERMIT.json'],pins=pins,tool_files={n:stable_sha(ROOT/n) for n in files},files=[dict(path=n,bytes=(ROOT/n).stat().st_size,sha256=stable_sha(ROOT/n)) for n in files],UUIDS={str(k):v for k,v in UUIDS.items()},launch_receipt_sha256=stable_sha(receipt),transport_path=str(TRANSPORT),transport_sha256=TRANSPORT_SHA,original_host_pins=prior['original_host_pins'],parent_reduction_host_pins=prior['parent_reduction_host_pins'],UTC=datetime.now(timezone.utc).isoformat(),no_automatic_retry=True)
    new(TOOL_MANIFEST,obj);return dict(status='LOCAL_REUSE48_BINDING_PREPARED_ONCE',sha256=stable_sha(TOOL_MANIFEST))


def validate_ready(snapshot,pins):
    need(snapshot.get('status')=='TERMINAL_SEAL_READY_FOR_INDEPENDENT_AUDIT' and snapshot.get('terminal_ready_observed') is True and snapshot.get('all_original_processes_exited_observed') is True,'Actual terminal identity/release not ready')
    need(snapshot['source_freeze_sha256']==pins['SOURCE_FREEZE.json'] and snapshot['permit_sha256']==pins['GPU_PERMIT.json'],'Terminal source/permit mismatch')
    need(snapshot['nvml']['status']=='PASS_READONLY_NVML' and not snapshot['nvml']['exact_owned_apps'],'Own NVML release unresolved')
    for x in snapshot.get('scorers',[]):need(x['actual_returncode']==0 and not x['process']['owned_alive'],'Actual scorer exit0 unresolved')
    chain=snapshot.get('offline_chain_wait_observed');need(isinstance(chain,dict) and chain.get('status')=='PASS_CPU_OBSERVED_DIRECT_CHILD_SHELL_WAIT_STATUSES' and chain.get('finisher_shell_wait_status')==0 and chain.get('source_freeze_sha256')==pins['SOURCE_FREEZE.json'] and chain.get('permit_sha256')==pins['GPU_PERMIT.json'],'Actualowned shell wait receipt missing or differs')


def audit_code(config):
    values = dict(TASK=TASK, REVISION=REVISION, PARENT_SOURCE=PARENT_SOURCE,
                  UUIDS=UUIDS, ORIGIN_SOURCE=ORIGIN_SOURCE,
                  ORIGIN55=ORIGIN55, REUSE_STATUS=REUSE_STATUS, ARCHIVE=ARCHIVE,
                  CONTROLS=CONTROLS, REMOTE=REMOTE,FRESH_IDS=FRESH_IDS,BYTEVIEW55=BYTEVIEW55,BYTEVIEW29=BYTEVIEW29,BYTEVIEW_SOURCE=BYTEVIEW_SOURCE,BYTEVIEW_PERMIT=BYTEVIEW_PERMIT,PRIOR368_55=PRIOR368_55,PRIOR368_29=PRIOR368_29,PRIOR368_SOURCE=PRIOR368_SOURCE,PRIOR368_PERMIT=PRIOR368_PERMIT,PRIOR368_MANIFEST=PRIOR368_MANIFEST,TEXT=TEXT, TEXT_IR=TEXT_IR,
                  OUTPUT_BOUND=OUTPUT_BOUND, SMALL_PREPARATION_BOUND=SMALL_PREPARATION_BOUND)
    head = 'from pathlib import Path, PurePosixPath\nimport hashlib,json,os\n'
    head += '\n'.join(k + '=' + repr(v) for k, v in values.items()) + '\n'
    return head + '\n'.join(inspect.getsource(f) for f in REMOTE_FUNCTIONS) + '\nC=type("PureCPU",(),dict(read=staticmethod(read),sha=staticmethod(stable_sha),require=staticmethod(need),objsha=staticmethod(lambda x:hashlib.sha256(canonical(x)).hexdigest())))\n' + \
        '\nprint(json.dumps(audit_root(' + repr(config) + '),sort_keys=True,allow_nan=False))\n'


def path_map(rows):
    mapped = []
    seen = set()
    logical = set()
    for row in rows:
        n = safe(row['path'])
        need(n not in logical, 'Duplicate download logical path')
        logical.add(n)
        physical = 'files_by_path_sha256/' + hashlib.sha256(n.encode()).hexdigest() + '/' + PurePosixPath(n).name
        folded = unicodedata.normalize('NFC', physical).casefold()
        need(folded not in seen, 'Physical path collision')
        seen.add(folded)
        mapped.append(dict(row, local_path=physical))
    return mapped


def unpack_verified(archive, destination, mapped):
    rows = {x['path']: x for x in mapped}
    seen = set()
    with tarfile.open(archive, 'r|') as tf:
        for member in tf:
            n = safe(member.name)
            need(member.isfile() and n in rows and n not in seen,
                 'Unexpected/duplicate/link tar member: ' + n)
            seen.add(n)
            row = rows[n]
            need(member.size == row['bytes'], 'Tar byte count differs')
            dest = Path(destination) / safe(row['local_path'])
            need(not dest.exists(), 'Local destination already exists')
            dest.parent.mkdir(parents=True, exist_ok=True)
            h = hashlib.sha256()
            count = 0
            with tf.extractfile(member) as src, dest.open('xb') as out:
                for block in iter(lambda: src.read(1024**2), b''):
                    count += len(block)
                    need(count <= row['bytes'], 'Tar stream exceeds expected bytes')
                    h.update(block)
                    out.write(block)
                out.flush()
                os.fsync(out.fileno())
            os.chmod(dest, 0o600)  # Downloaded .sh/.py remain non-executable data.
            need(count == row['bytes'] and h.hexdigest() == row['sha256'] == stable_sha(dest),
                 'Downloaded SHA/bytes differs: ' + n)
    need(seen == set(rows), 'Tar omitted expected members')


def download(transport, role, root, rows, mapped, label):
    import resource
    code = '''from pathlib import Path
import hashlib,os,sys,tarfile
R=Path(ROOT_VALUE);rows=ROWS_VALUE
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|',dereference=True) as tar:
 for row in rows:
  p=R/row['path'];a=p.stat();assert p.is_file() and not p.is_symlink() and a.st_size==row['bytes']
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
  assert h.hexdigest()==row['sha256']
  tar.add(p,arcname=row['path'],recursive=False)
  z=p.stat();assert (a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns)
'''.replace('ROOT_VALUE', repr(root)).replace('ROWS_VALUE', repr(rows))
    archive = ROOT / 'operations' / ('LOCAL_TERMINAL_' + label + '.tar')
    limit = sum(x['bytes'] for x in rows) + 4096 * len(rows) + 1024**2
    def finite_local_transport_output():
        # Bound the local SSH writer before bytes are received. This does not
        # set limits or invoke signals in any research/remote process.
        _, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
        cap = limit if hard == resource.RLIM_INFINITY else min(limit, hard)
        resource.setrlimit(resource.RLIMIT_FSIZE, (cap, cap))
    with archive.open('xb') as f:
        p = subprocess.run(transport.command(role=role, python=PYTHON55 if role == '55' else '/usr/bin/python3'),
                           input=code.encode(), stdout=f, stderr=subprocess.PIPE, timeout=3600,
                           preexec_fn=finite_local_transport_output)
    need(p.returncode == 0, 'Read-only small-file stream failed: ' + p.stderr.decode(errors='replace')[-2000:])
    # Bounded by exact expected file sizes plus tar headers/padding, never a model archive.
    need(archive.stat().st_size <= limit, 'Tar exceeds finite expected bound')
    unpack_verified(archive, DEST / label, mapped)
    return dict(path=str(archive), bytes=archive.stat().st_size, sha256=stable_sha(archive))


def _deliver_locked():
    pins = local_pins()
    tool = read(TOOL_MANIFEST)
    need(tool['status'] == 'FROZEN_REUSE48_MONITOR_AND_TERMINAL_TOOL' and
         tool['execution_root'] == REMOTE and tool['archive_root'] == ARCHIVE and tool['pins'] == pins,
         'Independent tool manifest is absent/stale')
    verify_rows(ROOT, tool['files'])
    need(tool['transport_path'] == str(TRANSPORT) and
         tool['transport_sha256'] == stable_sha(TRANSPORT), 'Transport bytes changed')
    need(not any(p.exists() for p in (CLAIM, FAILURE, RESULT, DEST)),
         'Prior delivery claim/result/failure/destination exists; no automatic retry')
    observer = import_local('terminal_gold_blind_reader', ROOT / 'operations/read_reuse48_snapshot_once.py')
    before = observer._query_remote()  # One gold-blind read, no local observation loop.
    if before.get('status') != 'TERMINAL_SEAL_READY_FOR_INDEPENDENT_AUDIT':
        return dict(status='NOT_TERMINAL_READY_NO_CLAIM_NO_RETRY', observed_status=before.get('status'))
    validate_ready(before, pins)
    new(CLAIM, dict(status='EXCLUSIVE_MANUAL_LOCAL_CPU_TERMINAL_DELIVERY_CLAIM', pid=os.getpid(),
        UTC=datetime.now(timezone.utc).isoformat(), pins=pins,
        tool_manifest_sha256=stable_sha(TOOL_MANIFEST), initial_gold_blind_snapshot=before,
        automatic_execution=False, retry_permission=False))
    try:
        transport = import_local('terminal_existing_transport', TRANSPORT)
        audits = {}
        for role, root, origin in (('55', REMOTE, ORIGIN55), ('29', ARCHIVE, ORIGIN29)):
            cfg = dict(role=role, root=root, expected_root=root, pins=pins,
                       origin_root=origin, origin_pins=tool['original_host_pins'][role],
                       parent_root=('/tmp/' if role == '55' else
                           '/data/zypan/qwen_gdn_wscale_rowbal_archive/') + PARENT_REVISION,
                       parent_pins=tool['parent_reduction_host_pins'][role])
            audits[role] = transport.run(audit_code(cfg), role=role,
                python=PYTHON55 if role == '55' else '/usr/bin/python3', timeout=1800)
            new(ROOT / 'operations' / ('FINAL_TERMINAL_ALL' + role + '_SHA_BYTES.json'), audits[role])
        a, b = audits['55'], audits['29']
        for key in ('source_count', 'science_count', 'science_bytes', 'manifest_sha256',
                    'source_freeze_sha256', 'permit_sha256', 'counts', 'seal', 'archive_verification'):
            need(a[key] == b[key], 'Host/archive scientific identity differs: ' + key)
        raw29 = {x['path']: x for x in b['controls']}
        need(a['seal']['archive_verification_sha256'] ==
             raw29['FINAL_ARCHIVE_VERIFICATION.json']['sha256'],
             'Seal does not bind actual archive29 verification bytes')
        small55 = a['small_files']
        small29 = b['controls']  # Preserve actual29 seal/control bytes separately.
        mapped55, mapped29 = path_map(small55), path_map(small29)
        expected_bytes = sum(x['bytes'] for x in small55 + small29)
        fs = os.statvfs(ROOT)
        need(fs.f_bavail * fs.f_frsize >= 2 * expected_bytes + 128 * 1024**2,
             'Insufficient local finite delivery space')
        DEST.mkdir()
        new(DEST / 'LOCAL_PATH_MAP.json', dict(status='EXACT_LOGICAL_TO_CASE_SAFE_PHYSICAL_MAP',
            server55=mapped55, archive29_controls=mapped29,
            original_sealed_manifests_not_rewritten=True))
        streams = [download(transport, '55', REMOTE, small55, mapped55, 'server55'),
                   download(transport, '29', ARCHIVE, small29, mapped29, 'archive29_controls')]
        after = observer._query_remote()
        validate_ready(after, pins)
        need(before['seal_sha256'] == after['seal_sha256'], 'Terminal seal changed during delivery')
        physical = []
        for label, rows in (('server55', mapped55), ('archive29_controls', mapped29)):
            for row in rows:
                rel = label + '/' + row['local_path']
                need(stable_sha(DEST / rel) == row['sha256'] and
                     (DEST / rel).stat().st_size == row['bytes'], 'Final Mac physical bytes differ')
                physical.append(dict(path=rel, bytes=row['bytes'], sha256=row['sha256'],
                                     logical_path=row['path']))
        new(DEST / 'LOCAL_PHYSICAL_FILES.json', dict(files=physical, count=len(physical)))
        with (DEST / 'LOCAL_PHYSICAL_SHA256SUMS.txt').open('x') as f:
            f.write(''.join(x['sha256'] + '  ' + x['path'] + '\n' for x in physical))
            f.flush(); os.fsync(f.fileno())
        out = dict(status='PASS_INDEPENDENT55_29_SOURCE_SCIENCE_CALIBRATION_AND_MAC_TEXT_DELIVERY',
            task_id=TASK, execution_root=REMOTE, archive_root=ARCHIVE, pins=pins,
            tool_manifest_sha256=stable_sha(TOOL_MANIFEST), counts=a['counts'],
            source_count=a['source_count'], science_count=a['science_count'],
            science_bytes=a['science_bytes'], manifest_sha256=a['manifest_sha256'],
            Mac_delivered_files=len(physical), Mac_delivered_bytes=expected_bytes,
            Mac_delivery_root=str(DEST), download_streams=streams,
            local_path_map_sha256=stable_sha(DEST / 'LOCAL_PATH_MAP.json'),
            local_physical_files_sha256=stable_sha(DEST / 'LOCAL_PHYSICAL_FILES.json'),
            local_physical_sums_sha256=stable_sha(DEST / 'LOCAL_PHYSICAL_SHA256SUMS.txt'),
            actual55_control_bytes=a['controls'], actual29_control_bytes=b['controls'],
            calibration_origin55=a['origin'], calibration_origin29=b['origin'],
            byteview_reuse48_55=a['byteview_reuse48'],byteview_reuse48_29=b['byteview_reuse48'],prior368_true_STOP55=a['prior368_true_STOP'],prior368_true_STOP29=b['prior368_true_STOP'],
            parent_reduction_lineage55=a['parent_reduction_lineage'],
            parent_reduction_lineage29=b['parent_reduction_lineage'],
            bound_dependencies55=a['execution_dependencies'],
            archive29_environment_not_misrepresented=b['execution_dependencies'],
            two_host_frozen_source_dependency_snapshots='PASS_FRESH_SHA_BYTES',
            final_exact_process_and_NVML=after, foreign_processes_untouched=True,
            skipped_remote_only=a['skipped_remote_only'], source_mutations=0,
            tensor_model_environment_compiled_binary_downloads=0, GPU_calls=0, signals=0,
            finisher_calls=0, scoring_calls=0, retry_permission=False)
        out['late_identity_supplement']=publish_supplement(transport,before,after,pins,a,b,physical)
        new(RESULT, out)
        return {k: out[k] for k in ('status', 'counts', 'source_count', 'science_count',
                                   'science_bytes', 'Mac_delivered_files', 'Mac_delivered_bytes')}
    except BaseException as exc:
        new(FAILURE, dict(status='TRUE_TERMINAL_DELIVERY_FAILURE_KEEP_CLAIM_NO_AUTORETRY',
            error=repr(exc), UTC=datetime.now(timezone.utc).isoformat(),
            tool_manifest_sha256=stable_sha(TOOL_MANIFEST), retry_permission=False))
        raise


def strict_unit(root, unit, *, source_sha256, permit_sha256, triton_manifest_sha256, triton_task_root):
    root = Path(root)
    t = root / 'execution' / unit['id']
    w = C.read(t / 'RESULT.json')
    i = C.read(t / 'RESOURCE_INTERVAL.json')
    d = C.read(t / 'DISPATCH.json')
    start = C.read(t / 'STARTED.json')
    copy = dict(w)
    payload = copy.pop('payload_sha256')
    C.require(C.objsha(copy) == payload and w['status'] == 'PASS' and (w['unit'] == unit['id']), 'Actual qualification payload mismatch')
    C.require(w['identity'] == i['identity'] == start['identity'] and w['gpu'] == unit['gpu'] and (w['gpu_uuid'] == unit['gpu_uuid']), 'Actual qualification identity/UUID differs')
    C.require(i['exit_returncode'] == 0 and i['release'] == 'PASS', 'Actual exit0/release missing')
    C.require(w['source_freeze_sha256'] == source_sha256 and w['permit_sha256'] == permit_sha256 == d['permit_sha256'], 'Current execution source/permit differs')
    driver = C.read(t / 'DRIVER_IDENTITY.json')
    post = C.read(t / 'POSTFLIGHT_LIBRARY.json')
    C.require(driver['status'] == 'PASS' and driver['actual_uuid'] == unit['gpu_uuid'] and (driver['visible_device_count'] == 1) and (post['status'] == 'PASS'), 'Actual driver/native closure failed')
    r = w['result']
    kind = unit['kind']
    triton = C.read(t / 'TRITON_DEPENDENCY_ADMISSION.json')
    C.require(triton['status'] == 'PASS_FRESH_TRITON_DEPENDENCIES_ACTUAL_IMPORT' and r['triton_dependency_admission_sha256'] == C.sha(t / 'TRITON_DEPENDENCY_ADMISSION.json') and (triton['manifest_sha256'] == triton_manifest_sha256) and (triton['task_root'] == triton_task_root) and (triton['actual_triton_version'] == triton['version'] == '3.1.0'), 'Actual new Triton compiler/source/native admission missing')
    if kind == 'codec_fields':
        C.require(r['status'] == 'PASS' and r['actual_cuda'] is True and (r['independent_reference_pass'] is True), 'Actual GPU fields/independent oracle missing')
        if unit['gpu'] == 1:
            C.require(r['non_aime_diagnostics']['status'] == 'PASS_ACTUAL_NONAIME_DIAGNOSTICS_NO_PROXY_THRESHOLD', 'Actual fixed nonAIME diagnostics missing; no metric quality threshold applies')
    else:
        C.require(C.read(t / 'PREFLIGHT_LIBRARY.json')['status'] == 'PASS', 'Actual preflight nativeclosure failed')
    if kind == 'native_persistent_fresh':
        a = r['audit']
        n = unit['tokens']
        C.require(r['tokens'] == 512 and a['qdq_layer_updates'] == 24 * n and (a['next_consumption_checks'] == 24 * (n - 1)), 'Real32/128/512 all-layer QDQ/next count differs')
        C.require(len(r['writebacks_by_layer']) == 24 and all((v == n for v in r['writebacks_by_layer'].values())), 'Actual512 writeback layers incomplete')
        trace = C.read(t / 'TRACE.json')
        C.require(len(trace['tokens']) == n and [x['token'] for x in trace['tokens']] == list(range(n)) and (trace['status'] == ('REFERENCE_COLLECTED' if unit['repeat'] == 0 else 'PASS_FULL_ACTUAL_TENSOR_BYTES')), 'Full fresh actual byte trace missing')
    elif kind == 'nonAIME_free_generation_fresh':
        n = r['generated_token_count']
        p = r['input_tokens']
        total = p + n - 1
        a = r['quantizer_audit']
        C.require(0 < n <= 64 and r['actual_model_forward_calls'] == total and (a['qdq_layer_updates'] == 24 * total) and (r['sampler_calls'] == n), 'Actual sampler/prompt/fresh canary count differs')
        C.require(r['prompt_layer_QDQ'] == 24 * p and r['decode_layer_QDQ'] == 24 * (n - 1), 'Prompt token0/decode QDQ missing')
        C.require(a['next_consumption_checks'] == 24 * (total - 1), 'Full short actual next-consumption missing')
        ts = r['generated_token_ids']
        eos = ts[-1] in r['stop_token_ids']
        C.require(len(ts) == n and r['finish_reason'] == ('eos' if eos else 'length') and (eos or n == 64), 'Actual EOS/length invalid')
        if unit['repeat'] == 1:
            prior = C.read(root / 'execution' / unit['reference_unit'] / 'RESULT.json')['result']
            for key in ('generated_token_ids', 'input_ids_sha256', 'initial_generator_sha256', 'final_generator_sha256', 'finish_reason'):
                C.require(prior[key] == r[key], 'Two fresh original sampler/RNG payloads differ:' + key)
            C.require(C.read(t / 'TRACE.json')['status'] == 'PASS_FULL_ACTUAL_TENSOR_BYTES', 'Actual sampler full-byte check missing')
    elif kind == 'real_model_memory_pressure':
        C.require(r['status'] == 'PASS' and {x['condition'] for x in r['cases']} == {'W', 'R', 'RW'}, 'Actual three-codec pressure missing')
        for x in r['cases']:
            C.require(x['total_context'] >= 82293 and x['actual_short_QDQ_tokens'] == 16 and (x['free_bytes'] >= 1024 ** 3), 'Maximum-context actual shortQDQ/nativeSDPA resource gate failed')
            C.require(x['pressure_scope'] == 'SYNTHETIC_KV_NATIVE_SDPA_WITH_LIVE_MODEL_NOT81K_GDN_TRAJECTORY', 'Pressure scope claim changed')
    elif kind == 'matched_performance':
        C.require(r['tokens'] == 4096 and r['timing']['short']['TPS'] > 0 and (r['timing']['late']['TPS'] > 0), 'Matched actual early/late TPS missing')
        for window in ('short', 'late'):
            C.require(r['timing'][window]['measured'] == 128 and r['timing'][window]['observer'] is False and (r['timing'][window]['sampling_invoked'] is True), 'Measurement window observer/sampler differs')
        q = r['profiler']
        C.require(q['status'] == 'PASS_ACTUAL16TOKEN_CUDA_PROFILE' and q['actual_CUDA_kernel_calls'] > 0 and (q['quantizer_CUDA_milliseconds'] > 0), 'Actual quantizer kernel/time profile missing')
    return dict(unit=unit['id'], kind=kind, gpu=unit['gpu'], gpu_uuid=unit['gpu_uuid'], result_sha256=C.sha(t / 'RESULT.json'), resource_interval_sha256=C.sha(t / 'RESOURCE_INTERVAL.json'), actualexit0=True, release='PASS')

def audit_byteview_origin(cfg,current):
    origin=Path(BYTEVIEW55 if cfg['role']=='55' else BYTEVIEW29)
    need(stable_sha(origin/'SOURCE_FREEZE.json')==BYTEVIEW_SOURCE and stable_sha(origin/'GPU_PERMIT.json')==BYTEVIEW_PERMIT,'BYTEVIEW fixed origin differs')
    source=read(origin/'SOURCE_FREEZE.json');need(len(source['files'])==334,'BYTEVIEW source334 differs');verify_rows(origin,source['files'])
    manifest=read(origin/'SHA256_MANIFEST.json');total=verify_rows(origin,manifest['files']);need(len(manifest['files'])==5397 and total==1731153792,'BYTEVIEW complete science5397 differs')
    audit=read(current/'provenance/REUSED_QUALIFICATION48/CPU_ORIGIN48_READONLY_AUDIT.json')
    need(stable_sha(current/'provenance/REUSED_QUALIFICATION48/CPU_ORIGIN48_READONLY_AUDIT.json')=='371ec55ebab83fe9059aab0092e880e7eb52dac6bc6370401665231c621b3902','Frozen origin48 audit differs')
    need(audit['qualification_reused_count']==48,'Origin48 count differs')
    for name,row in audit['origin_controls'].items():need(stable_sha(origin/safe(name))==row['sha256'] and (origin/name).stat().st_size==row['bytes'],'Oldorigin control changed '+name)
    for u in audit['units']:
        verify_rows(origin,u['files']);strict_unit(origin,u['original_definition'],source_sha256=BYTEVIEW_SOURCE,permit_sha256=BYTEVIEW_PERMIT,triton_manifest_sha256='bc695cc73e549272d9231cfe1f6b833dbf3519c9964e3820746acfee92dcafe5',triton_task_root=BYTEVIEW55)
    oldbad=read(origin/'execution/GPU1_base_MATCHED4096/RESOURCE_INTERVAL.json');need(oldbad['exit_returncode']==-11,'Oldbase nonzero exit changed')
    return dict(status='PASS_FRESH_FIXED_BYTEVIEW48_PROVENANCE_ALL_SOURCE_SCIENCE_SHA',unit_count=48,source_count=334,science_count=5397,science_bytes=total,old49_credit=0,origin_root=str(origin),source_sha256=BYTEVIEW_SOURCE,permit_sha256=BYTEVIEW_PERMIT)


def verify_profile_lifecycle(root,unit,pins):
    need(unit['id'] in FRESH_IDS and unit['kind']=='matched_performance' and unit['gpu']==1,'Onlyexactcurrent4 matched units may use CUPTI lifecycle')
    target=Path(root)/'execution'/unit['id'];wrapper=read(target/'RESULT.json');identity=wrapper['identity'];digest=hashlib.sha256(canonical(wrapper['result']['profiler'])).hexdigest()
    binding=read(Path(root)/'analysis/CUPTI_LIFECYCLE_BINDING.json');need(binding['status']=='PASS_FRESH_EXACT_EXISTING_CUPTI_ELF_AND_HEADER_BINDINGS' and len(binding['files'])==1,'Explicitfixed CUPTI binding differs');elf=binding['files'][0];msha=stable_sha(Path(root)/'analysis/CUPTI_LIFECYCLE_BINDING.json')
    pre=read(target/'CUPTI_PRELOAD_ADMISSION.json');done=read(target/'PROFILE_COLLECTION_COMPLETED.json');detach=read(target/'CUPTI_PROFILE_DETACH.json')
    for row in (pre,done,detach):need(row['identity']==identity and row['source_freeze_sha256']==pins['SOURCE_FREEZE.json'] and row['permit_sha256']==pins['GPU_PERMIT.json'],'CUPTI lifecycle identity/source/permit differs')
    for row in (pre,detach):need(row['path']==elf['path'] and row['sha256']==elf['sha256'] and row['manifest_sha256']==msha,'Preload/detach same pinnedCUPTI handle differs')
    need(done['status']=='PASS_ACTUAL_PROFILE_COLLECTION_COMPLETED_BEFORE_DETACH' and done['profile_summary_sha256']==digest,'Original actualprofile summary binding missing')
    need(done['profile_summary']==wrapper['result']['profiler'] and wrapper['result']['profiler_native_lifecycle']==detach,'Actualprofile/cleanup resultwrapper differs')
    need(pre['status']=='EXACT_EXISTING_CUPTI_ELF_LOADED_BEFORE_DRIVER_INIT_NO_CUPTI_API_CALL' and pre['already_held'] is False and pre['CUDA_Driver_API_calls']==pre['CUPTI_API_calls']==0,'Actual beforedriver preload scope differs')
    need(detach['CUPTI_API_calls']==2 and detach['profiler_collection_already_completed'] is True and detach['CUDA_or_CUPTI_APIs_not_disabled'] is True and detach['signal_or_exit_suppression'] is False and detach['actual_process_exit_still_required'] is True,'Cleanup scope or actualexit requirement weakened')
    need(detach['status']=='DOCUMENTED_CUPTI_FLUSH_AND_DETACH_RETURNED_SUCCESS' and detach['flush_returncode']==detach['finalize_returncode']==0 and detach['profile_summary_sha256']==digest,'Actual CUPTI flush/finalize did not succeed')
    events=[json.loads(line) for line in (target/'CUPTI_PROFILE_DETACH_EVENTS.jsonl').read_text().splitlines() if line];required=['CYCLIC_GC_WHILE_CUDA_RUNTIME_ALIVE','CUDA_WORK_SYNCHRONIZED_BEFORE_ACTIVITY_FLUSH','CUPTI_ACTIVITY_FLUSH_BEGIN','CUPTI_ACTIVITY_FLUSH_RETURN','CUPTI_FINALIZE_BEGIN','CUPTI_FINALIZE_RETURN'];phases=[x['phase'] for x in events]
    need(phases==required,'CUPTI detachorderedjournal differs')
    need(events[3]['returncode']==events[5]['returncode']==0,'CUPTI journalnonzero API returncode')
    need(all(type(x.get('monotonic')) in (int,float) for x in events) and all(events[n]['monotonic']<=events[n+1]['monotonic'] for n in range(len(events)-1)),'CUPTI events hostmonotonic order differs')
    return dict(status='PASS_EXACT_CURRENT_PROFILE_CUPTI_LIFECYCLE',unit=unit['id'],profile_summary_sha256=digest,detach_sha256=stable_sha(target/'CUPTI_PROFILE_DETACH.json'),actualexit_still_independently_required=True)


def audit_prior368_origin(cfg):
    prior=Path(PRIOR368_55 if cfg['role']=='55' else PRIOR368_29)
    need(stable_sha(prior/'SOURCE_FREEZE.json')==PRIOR368_SOURCE and stable_sha(prior/'GPU_PERMIT.json')==PRIOR368_PERMIT,'Prior368 fixed SOURCE/permit changed')
    source=read(prior/'SOURCE_FREEZE.json');need(source['count']==len(source['files'])==368,'Prior368 sourcecount differs');verify_rows(prior,source['files'])
    need(stable_sha(prior/'SHA256_MANIFEST.json')==PRIOR368_MANIFEST,'Prior368 manifest changed');manifest=read(prior/'SHA256_MANIFEST.json');total=verify_rows(prior,manifest['files'])
    need(len(manifest['files'])==4735 and total==1318027825 and manifest['status']=='PARTIAL_RESULTS_PRESERVED_AND_STOP','Prior368 complete science/STOP differs')
    seal=read(prior/'FINAL_SEAL_RECEIPT.json');need(seal['manifest_sha256']==PRIOR368_MANIFEST and seal['scientific_status']=='PARTIAL_RESULTS_PRESERVED_AND_STOP','Prior368 immutable seal changed')
    verification=prior/'FINAL_ARCHIVE_VERIFICATION.json';need(stable_sha(verification)==seal['archive_verification_sha256'],'Prior368 seal verification byte binding differs')
    failed=read(prior/'execution/GPU1_base_MATCHED4096/RESOURCE_INTERVAL.json');need(failed['exit_returncode']==-11 and failed['release']=='PASS','Prior368 native failure reclassified')
    need(stable_sha(prior/'FIRST_FAILURE.json')=='9a07d11e46625b6093d0c2e5b2ff8e01f04db02a33eecc2f51cb340d2622668f','Prior368 FIRST_FAILURE changed')
    return dict(status='PASS_FRESH_PRIOR368_TRUE_STOP_FULL_SOURCE_SCIENCE_SHA',root=str(prior),source_count=368,science_count=4735,science_bytes=total,source_sha256=PRIOR368_SOURCE,permit_sha256=PRIOR368_PERMIT,manifest_sha256=PRIOR368_MANIFEST,actualexit_minus11_preserved=True,new_qualification_credit=0,new_formal_credit=0)


def read_late_closed_files(transport,pins):
    code='from pathlib import Path\nimport json,hashlib,base64,os\nR=Path('+repr(REMOTE)+')\nrows=[]\n'
    code+="for name,bound in [('OFFLINE_CHAIN_OBSERVED_EXIT.json',1024**2),('finisher.log',32*1024**2)]:\n p=R/name;a=p.stat();assert p.is_file() and not p.is_symlink() and a.st_size<=bound\n raw=p.read_bytes();z=p.stat();assert (a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns)\n rows.append(dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),base64=base64.b64encode(raw).decode()))\nprint(json.dumps(dict(status='PASS_READONLY_LATE_CLOSED_FILE_BYTES',files=rows)))\n"
    result=transport.run(code,role='55',python=PYTHON55,timeout=40);need(result['status']=='PASS_READONLY_LATE_CLOSED_FILE_BYTES' and [x['path'] for x in result['files']]==['OFFLINE_CHAIN_OBSERVED_EXIT.json','finisher.log'],'Lateclosed file scope differs')
    import base64
    for row in result['files']:
        raw=base64.b64decode(row['base64'],validate=True);need(len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'],'Lateclosed file payload SHA/bytes differs')
    chain=json.loads(base64.b64decode(result['files'][0]['base64']));need(chain['source_freeze_sha256']==pins['SOURCE_FREEZE.json'] and chain['permit_sha256']==pins['GPU_PERMIT.json'] and chain['status']=='PASS_CPU_OBSERVED_DIRECT_CHILD_SHELL_WAIT_STATUSES','Latechain SOURCE/permit binding differs')
    return result


def publish_supplement(transport,before,after,pins,a,b,physical):
    # A separate immutable archive binds late observed absence and local delivery.
    # Never invent shell/finisher return codes when no owning wait receipt exists.
    supplement=dict(status='INDEPENDENT_LATE_EXIT_AND_DELIVERY_OBSERVATION',main_archive=ARCHIVE,main_manifest_sha256=a['manifest_sha256'],source_sha256=pins['SOURCE_FREEZE.json'],permit_sha256=pins['GPU_PERMIT.json'],parent_finisher_returncode='UNRESOLVED_NO_OWNING_WAIT_RECEIPT',late_before=before,late_after=after,all_current_original_processes_observed_gone=True,owned_NVML_release=True,audit55=a,audit29=b,Mac_files_sha256=hashlib.sha256(canonical(physical)).hexdigest(),primary_seal_rewritten=False)
    latefiles=read_late_closed_files(transport,pins)
    chain=json.loads(__import__('base64').b64decode(latefiles['files'][0]['base64']))
    need(chain['finisher_shell_wait_status']==0,'Finisher realownedwait not0')
    need(chain['controller']['identity']==after['controller']['expected'] and chain['finisher']['identity']==after['finisher']['expected'] and chain['shell_parent_PID']==after['offline_shell']['expected']['pid'],'Lateactual ownedwait identity/shell binding differs')
    if a['counts']['full']:need(chain['controller_shell_wait_status']==0,'Fullrun controller ownedwait not0')
    supplement['parent_finisher_returncode']='ACTUAL_DIRECT_CHILD_BASH_WAIT_STATUSES_RECORDED_NO_PYTHON_REINTERPRETATION';supplement['offline_chain_wait']=chain;supplement['closed_late_files']=latefiles
    payload=canonical(supplement)+b'\n';digest=hashlib.sha256(payload).hexdigest();final=ARCHIVE+'_INDEPENDENT_SUPPLEMENT_'+digest[:16]
    code='''import json,os,hashlib,ctypes
from pathlib import Path
payload=PAYLOAD;final=Path(FINAL);stage=Path(str(final)+'.staging');assert not final.exists() and not stage.exists();stage.mkdir()
p=stage/'INDEPENDENT_LATE_EXIT_AND_DELIVERY.json'
with p.open('xb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
assert hashlib.sha256(p.read_bytes()).hexdigest()==DIGEST
lib=ctypes.CDLL(None,use_errno=True);fn=lib.renameat2;fn.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint];fn.restype=ctypes.c_int;rc=fn(-100,os.fsencode(stage),-100,os.fsencode(final),1);assert rc==0,os.strerror(ctypes.get_errno());fd=os.open(final.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
assert hashlib.sha256((final/'INDEPENDENT_LATE_EXIT_AND_DELIVERY.json').read_bytes()).hexdigest()==DIGEST
print(json.dumps(dict(status='PASS_INDEPENDENT_SUPPLEMENT_SHA_BYTES_RENAMEAT2_NOREPLACE',root=str(final),sha256=DIGEST,bytes=len(payload))))'''
    code='PAYLOAD='+repr(payload)+'\nFINAL='+repr(final)+'\nDIGEST='+repr(digest)+'\n'+code.replace('payload=PAYLOAD;','payload=PAYLOAD;')
    result=transport.run(code,role='29',python='/usr/bin/python3',timeout=90)
    new(ROOT/'operations/REUSE48_INDEPENDENT_LATE_SUPPLEMENT.json',dict(receipt=result,payload=supplement))
    return result

C=type("PureCPU",(),dict(read=staticmethod(read),sha=staticmethod(stable_sha),require=staticmethod(need),objsha=staticmethod(lambda x:hashlib.sha256(canonical(x)).hexdigest())))
REMOTE_FUNCTIONS=(need,safe,stable_sha,read,canonical,verify_rows,classify_terminal,validate_formal,select_small,strict_unit,verify_profile_lifecycle,audit_byteview_origin,audit_prior368_origin,audit_root)

def deliver():
    import fcntl
    lockpath=ROOT/'operations/REUSE48_READONLY_OBSERVER.lock'
    with lockpath.open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return dict(status='SKIP_ACTIVE_READER_OR_DELIVERY',dispatches=0)
        return _deliver_locked()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--prepare-tool-manifest', action='store_true')
    g.add_argument('--deliver-terminal-once', action='store_true')
    args = ap.parse_args()
    if args.prepare_tool_manifest:
        out = prepare_tool_manifest()
    elif args.deliver_terminal_once:
        out = deliver()
    else:
        out = dict(status='MANUAL_ONLY_NO_ACTION', remote_calls=0, GPU_calls=0,
                   message='Prepare tool manifest after final SOURCE/PERMIT; invoke delivery only explicitly.')
    print(json.dumps(out, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
