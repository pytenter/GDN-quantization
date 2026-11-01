"""Origin-aware applicability and formal admission; never relabel an old GPU execution."""
from pathlib import Path
import time

VERIFIED = set()


def admission(C, root):
    root = Path(root)
    value = C.read(root / 'QUALIFICATION_REUSE_ADMISSION.json')
    auth = C.read(root / 'HUMAN_QUALIFICATION_REUSE_AUTHORIZATION.json')
    C.require(value['human_authorization_sha256'] == C.sha(root / 'HUMAN_QUALIFICATION_REUSE_AUTHORIZATION.json')
        and auth['reused_gate_count'] == value['reused_count'] == 58
        and auth['remaining_gate_count'] == value['remaining_count'] == 0,
        'Reuse lacks the exact latest human request and58/0 scope')
    rows = value['reused']
    C.require(len(rows) == len({x['unit']['id'] for x in rows}) == 58,
        'Reuse must identify exactly58 distinct genuine units')
    return value


def target(C, root, unit):
    value = admission(C, root)
    matches = [x for x in value['reused'] if x['unit']['id'] == unit['id']]
    if not matches:
        return Path(root) / 'qualification' / unit['id'], None
    row = matches[0]
    C.require(row['unit'] == unit,
        'Reused manifest changed or an unpassed unit was reused')
    return Path(row['origin_root']) / 'qualification' / unit['id'], row


def validate(C, root, unit):
    root = Path(root)
    original, row = target(C, root, unit)
    if row is None:
        return C.validate_qualification_result(root, unit)
    origin = Path(row['origin_root'])
    C.require(origin.is_absolute() and origin != root and not origin.is_symlink(), 'Invalid qualification origin')
    origins = [x for x in admission(C, root)['origins'] if x['root'] == str(origin)]
    C.require(len(origins) == 1, 'Qualification origin not in frozen origin allowlist')
    pinned = origins[0]
    C.require(C.sha(origin / 'SOURCE_FREEZE.json') == row['origin_source_freeze_sha256'] == pinned['source_sha256']
        and C.sha(origin / 'FINAL_SEAL_RECEIPT.json') == row['origin_seal_sha256'] == pinned['seal_sha256']
        and C.sha(origin / 'GPU_PERMIT.json') == row['origin_permit_sha256'] == pinned['permit_sha256'],
        'Original source/seal/permit identity changed')
    cache_key = (str(origin), row['origin_source_freeze_sha256'])
    if cache_key not in VERIFIED:
        # The current control intentionally cannot operate another execution
        # root. Independently read/hash the immutable original manifest here;
        # do not relax that execution/dispatch confinement.
        frozen = C.read(origin / 'SOURCE_FREEZE.json')
        C.require(len(frozen['files']) == len({x['path'] for x in frozen['files']}) == pinned['source_count'],
            'Original qualification source manifest differs')
        for file in frozen['files']:
            name = Path(file['path']); path = origin / name
            C.require(not name.is_absolute() and '..' not in name.parts
                and path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(origin)
                and path.stat().st_size == file['bytes'] and C.sha(path) == file['sha256'],
                'Original source bytes changed: ' + str(name))
        VERIFIED.add(cache_key)
    C.require(C.sha(origin / 'GPU_PERMIT.json') == row['origin_permit_sha256'], 'Original qualification permit changed')
    C.require(next(x for x in C.read(origin / 'QUALIFICATION_JOB_MANIFEST.json')['units']
        if x['id'] == unit['id']) == unit, 'Original manifest/UUID/window differs')
    copied = root / row['cached_evidence_prefix']
    C.require(copied.resolve().is_relative_to(root.resolve()) and not copied.is_symlink(), 'Unsafe cached evidence')
    for file in row['files']:
        name = Path(file['path'])
        C.require(not name.is_absolute() and '..' not in name.parts, 'Unsafe qualification evidence name')
        for p in [original / name, copied / name]:
            C.require(p.is_file() and not p.is_symlink() and p.stat().st_size == file['bytes']
                and C.sha(p) == file['sha256'], 'Original or cached qualification bytes changed: ' + str(p))
    if unit['kind'].startswith('profile_'):
        trace = row['profile_trace']; path = original / trace['path']
        C.require(trace['path'] == 'CUDA_PROFILE.json' and path.is_file() and not path.is_symlink()
            and path.stat().st_size == trace['bytes'] and C.sha(path) == trace['sha256'],
            'Reused genuine profile trace differs')
        detached = C.read(original / 'CUPTI_PROFILE_DETACH.json')
        C.require(detached['flush_returncode'] == detached['finalize_returncode'] == 0
            and detached['trace_sha256'] == trace['sha256']
            and detached['actual_process_exit_still_required'] and not detached['signal_or_exit_suppression'],
            'Reused profiler lacks genuine cleanup and trace evidence')
    wrapper = C.validate_qualification_result(origin, unit)
    identity = C.read(original / 'STARTED.json')['identity']
    C._interval_ok(origin, original, identity)
    receipt = row['controller_receipt']
    for key, name in [('result_sha256', 'RESULT.json'), ('resource_interval_sha256', 'RESOURCE_INTERVAL.json'),
                      ('validation_sha256', 'CONTROLLER_VALIDATION.json')]:
        C.require(receipt[key] == C.sha(original / name), 'Old controller receipt SHA differs')
    C.require(not (Path(root) / 'qualification' / unit['id']).exists(),
        'A reused unit also has a new attempt; ambiguous execution scope')
    return wrapper


def seed(C, root):
    value = admission(C, root)
    receipts = {}
    for row in value['reused']:
        validate(C, root, row['unit'])
        receipt = dict(row['controller_receipt'])
        receipt.update(reused_qualification=True, origin_root=row['origin_root'],
            origin_source_freeze_sha256=row['origin_source_freeze_sha256'],
            cached_evidence_prefix=row['cached_evidence_prefix'])
        receipts[row['unit']['id']] = receipt
    return receipts


def authorize_formal(C, root, receipts):
    root = Path(root)
    C.require(not (root / 'FIRST_FAILURE.json').exists(), 'Current failure blocks formal')
    C.require(not (root / 'AUTHORIZATION_FORMAL.json').exists() and not (root / 'FORMAL_GPU_PERMIT.json').exists(),
        'Formal authorization already attempted')
    VERIFIED.clear()  # Fresh origin hashes at the final, rather than cached startup admission.
    value = admission(C, root)
    source = C.verify_sources(root)
    C.verify_model_and_external(root)
    C.validate_prior_p1_stop(root)
    units = C.read(root / 'QUALIFICATION_JOB_MANIFEST.json')['units']
    C.require(len(units) == len(receipts) == 58 and {x['unit'] for x in receipts} == {u['id'] for u in units},
        'Every one of the same58 qualification requirements must be met')
    results, audited = {}, []
    for unit in units:
        results[unit['id']] = validate(C, root, unit)
        path, row = target(C, root, unit)
        identity = C.read(path / 'STARTED.json')['identity']
        C._interval_ok(path.parents[1], path, identity)
        audited.append(dict(unit=unit['id'], identity=identity, result_sha256=C.sha(path/'RESULT.json'),
            interval_sha256=C.sha(path/'RESOURCE_INTERVAL.json'), origin_root=str(path.parents[1]),
            origin_source_freeze_sha256=results[unit['id']]['source_freeze_sha256'],
            reused_qualification=row is not None))
    C.require(sum(x['reused_qualification'] for x in audited) == 58, 'Reuse/current gate counts changed')
    observation = C.query_gpu()
    C.require(not any(x['uuid'] in C.UUIDS.values() for x in observation['apps']), 'Own authorized UUIDs not idle')
    jobs = C.read(root / 'FORMAL_JOB_MANIFEST.json')['jobs']
    C.require(len(jobs) == 60 and len({x['id'] for x in jobs}) == 60 and {x['condition'] for x in jobs} == {'C1','S3'},
        'Original formal roster differs')
    C.require(C.read(root/'HUMAN_CONTINUATION_AUTHORIZATION.json')['formal_conditionally_authorized'], 'Original formal grant absent')
    base = C.read(root/'GPU_PERMIT.json')
    C.require(base['ALLOW_GPU']=='QUALIFICATION_ONLY' and base['source_freeze_sha256']==source
        and base['qualification_reuse_admission_sha256']==C.sha(root/'QUALIFICATION_REUSE_ADMISSION.json'),
        'Source/base permit/reuse binding differs')
    performance = C.performance_audit(root, units, results)
    C.atomic_new(root/'PERFORMANCE_ADMISSION.json',performance)
    C.atomic_new(root/'GPU_QUALIFICATION_FINAL.json',dict(status='PASS_ALL58_REAL_UNITS_EXIT0_AND_RELEASED',
        source_freeze_sha256=source, units=audited, observation=observation, monotonic=time.monotonic(),
        qualification_reused=58, qualification_executed_current=0,
        qualification_reuse_admission_sha256=C.sha(root/'QUALIFICATION_REUSE_ADMISSION.json'),
        no_historical_execution_relabelled_as_current=True))
    auth=dict(status='ALLOW_FIXED60_FORMAL_ONCE_AFTER_ALL58_REAL_GATES_PASS',task_id=C.TASK,
        execution_attempt=root.name,source_freeze_sha256=source,base_permit_sha256=C.sha(root/'GPU_PERMIT.json'),
        qualification_sha256=C.sha(root/'GPU_QUALIFICATION_FINAL.json'),
        performance_admission_sha256=C.sha(root/'PERFORMANCE_ADMISSION.json'),jobs_sha256=C.sha(root/'FORMAL_JOB_MANIFEST.json'),
        human_authorization_sha256=C.sha(root/'HUMAN_CONTINUATION_AUTHORIZATION.json'),
        human_qualification_reuse_authorization_sha256=C.sha(root/'HUMAN_QUALIFICATION_REUSE_AUTHORIZATION.json'),
        planned=60,devices=list(C.UUIDS.values()),monotonic=time.monotonic())
    C.atomic_new(root/'AUTHORIZATION_FORMAL.json',auth)
    C.atomic_new(root/'FORMAL_GPU_PERMIT.json',dict(task_id=C.TASK,ALLOW_GPU='FORMAL_ONLY',source_freeze_sha256=source,
        authorization_sha256=C.sha(root/'AUTHORIZATION_FORMAL.json'),devices=list(C.UUIDS.values()),
        jobs_sha256=auth['jobs_sha256'],human_authorization_sha256=auth['human_authorization_sha256']))
    return auth
