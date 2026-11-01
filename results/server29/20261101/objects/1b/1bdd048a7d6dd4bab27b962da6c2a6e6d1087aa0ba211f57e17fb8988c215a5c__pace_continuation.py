"""One exact closed PACE origin and one finite continuation. CPU-only admissions.

Original results retain their original source, permit, PID, execution root and cost
ledger. Provenance snapshots never count as fresh starts or fresh GPU intervals.
"""
import ast
import copy
import hashlib
import os
from pathlib import Path
import pace_control as C

REVISION = 'QWEN_GDN_PACE48_AIME26_SEED1_60_V1_SCALE_REPAIR_20261025'
OLD_REMOTE = '/data/zypan/analyses/qwen_gdn_pace48_aime26_seed1_60_v1'
OLD_ARCHIVE = '/data/zypan/qwen_gdn_pace48_archive/QWEN_GDN_PACE48_AIME26_SEED1_60_V1'
SNAPSHOT = 'provenance/PRIOR_PACE_TRUE_STOP'
OLD_SOURCE_SHA = '700b707d450a14fb5dbd52aef3df6c0fa6ac59649359eb0712cac87a951113ca'
OLD_PERMIT_SHA = '3cb0900e85fea2124fbcbfd3b1c764ce833e3de3e037cadd06f990c230914c31'
OLD_SCIENCE_SHA = '9943928865d7259e200bc4b2df0e368ee6d230821ae23e230a5b278ed26b4a2e'
MASK_SHA = '09483fb4debca26e9561778adea987d80dfcadb1ef0d4dd1ec311e6915bf7ca6'
REUSED_IDS = ('aime26_01_R2_seed1', 'aime26_01_R3_seed1', 'aime26_02_R2_seed1')
FAILED_ID = 'aime26_02_R3_seed1'
GATE_STAGE, GATE_UNIT = 'scale_repair_gates', 'SCALE_REPAIR_GATES'
FRESH_REVIEW = 'analysis/PACE_REUSE3_DUAL_HOST_FRESH_REVIEW.json'
ORIGIN_CONTROL_PINS = {'source_manifest.json': {'bytes': 23960, 'sha256': '700b707d450a14fb5dbd52aef3df6c0fa6ac59649359eb0712cac87a951113ca'}, 'GPU_PERMIT.json': {'bytes': 1120, 'sha256': '3cb0900e85fea2124fbcbfd3b1c764ce833e3de3e037cadd06f990c230914c31'}, 'SCIENCE_FREEZE.json': {'bytes': 2040, 'sha256': '9943928865d7259e200bc4b2df0e368ee6d230821ae23e230a5b278ed26b4a2e'}, 'DELIVERY_MANIFEST.json': {'bytes': 54316, 'sha256': 'a803c18c7aa8bf90e69b9d01e9ece3d1599a6bf51fa371f4ff1250d08c332c18'}, 'SESSION_RESULT.json': {'bytes': 17421, 'sha256': '33a8c4c424e38961097743cdda351dedd66aa0a5ac55cbf1d250f9fd04758a6f'}, 'FORMAL_ADMISSION.json': {'bytes': 496, 'sha256': '28256800fda4c78f9065739f2796ef42ac1d3b352b3404d60bbbd7e887ac1067'}, 'RESOURCE_LEDGER.json': {'bytes': 12250, 'sha256': 'd4f472796516743aced0869114e7e9c110c56cde68168a9d2824b5d4eafe93d6'}, 'FIRST_FAILURE.json': {'bytes': 3231, 'sha256': 'f18bb4172b004bc37f83e5ed088d8320271909a8471dfd7809ae2957aab3d4f9'}, 'final_verdict.json': {'bytes': 3522, 'sha256': '76a8dce78a3169ac743b4e996d6ddd2fe97fd0965cce22d12677af5bfc13af57'}, 'analysis/MODEL_TOKENIZER_FRESH_SHA.json': {'bytes': 5095, 'sha256': 'e48f1089037eca3f03e30fff77849b4d23dd660019501754f479e2ba35e4e922'}}


def safe(root, name):
    base = Path(root).resolve()
    p = Path(name)
    C.require(isinstance(name, str) and not p.is_absolute() and '..' not in p.parts and
              str(p) == name and name != '.' and not any(x in name for x in ('\x00','\r','\n')),
              'Unsafe continuation artifact path')
    q = base / p
    current = q
    while current != base:
        C.require(not current.is_symlink(), 'Continuation symlink forbidden')
        current = current.parent
    return q


def rows_check(root, rows):
    seen = set()
    total = 0
    for row in rows:
        C.require(set(row) >= {'path', 'bytes', 'sha256'} and row['path'] not in seen,
                  'Duplicate/malformed origin file row')
        seen.add(row['path'])
        p = safe(root, row['path'])
        C.require(p.is_file() and p.stat().st_size == row['bytes'] and C.sha(p) == row['sha256'],
                  'Original byte identity changed: ' + row['path'])
        total += row['bytes']
    return {'count': len(seen), 'bytes': total}


def source_bound(root, name):
    rows = C.read(Path(root) / C.SOURCE)['files']
    found = [x for x in rows if x['path'] == name]
    C.require(len(found) == 1, 'Continuation input was not source frozen: ' + name)
    return rows_check(root, found)


def origin_root(root):
    # Never accept a caller-selected foreign root or a foreign dispatch root.
    p = safe(root, SNAPSHOT)
    C.require(p.is_dir(), 'The one fixed original provenance snapshot is missing')
    return p


def remaining_jobs(root):
    jobs = C.validate_jobs(root)
    ids = [x['id'] for x in jobs]
    C.require(len(ids) == len(set(ids)) == 60 and tuple(ids[:3]) == REUSED_IDS and
              ids[3] == FAILED_ID, 'Exact original formal ordering changed')
    return [x for x in jobs if x['id'] not in REUSED_IDS]


def dispatch_allowed(root, stage, unit):
    C.require((stage, unit) == (GATE_STAGE, GATE_UNIT) or
              (stage == 'formal' and unit in {x['id'] for x in remaining_jobs(root)}),
              'Only one repair gate and the fixed remaining57 may be dispatched')


def verify_equivalence(root, old):
    unchanged = ('src/pace_buffer.py', 'src/pace_runtime.py', 'src/inherited_device_guards.py',
                 'AIME26_DATASET_FREEZE.json', 'FORMAL_JOB_MANIFEST.json',
                 'protocol/calibration_windows.json', 'protocol/sampler.py', 'protocol/common.py',
                 'protocol/aime26_scorer_v4.py', 'protocol/aime26_frozen.jsonl',
                 'vendor/canonical_h128.py')
    for name in unchanged:
        C.require(C.sha(safe(root, name)) == C.sha(safe(old, name)),
                  'Accepted-domain compute/model/protocol changed: ' + name)
    previous = ast.parse(safe(old, 'src/pace_quantizer.py').read_text())
    current = ast.parse(safe(root, 'src/pace_quantizer.py').read_text())
    removed = []
    class ExactPrematureGuard(ast.NodeTransformer):
        def visit_If(self, node):
            if (len(node.body) == 1 and isinstance(node.body[0], ast.Raise) and
                isinstance(node.body[0].exc, ast.Call) and
                isinstance(node.body[0].exc.func, ast.Name) and node.body[0].exc.func.id == 'ValueError' and
                len(node.body[0].exc.args) == 1 and isinstance(node.body[0].exc.args[0], ast.Constant) and
                node.body[0].exc.args[0].value == 'affine group scale exceeds finite FP16 range'):
                expected = ast.parse("if bool((raw_scale > FP16_MAX).any().item()):\n raise ValueError('affine group scale exceeds finite FP16 range')").body[0]
                C.require(ast.dump(node, include_attributes=False) == ast.dump(expected, include_attributes=False),
                          'The removed guard is not the exact original premature range guard')
                removed.append(True)
                return None
            return self.generic_visit(node)
    previous = ExactPrematureGuard().visit(previous)
    # The added diagnostics execute only in the unchanged failed-postcast branch.
    # They may enrich an error; they cannot change any accepted code/scale/ZP.
    allowed_helpers = {'FP16ScaleOverflowError', '_scale_diagnostic_value', '_invalid_scale_diagnostics'}
    current.body = [n for n in current.body if not
                    ((isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in allowed_helpers) or
                     (isinstance(n, ast.Import) and len(n.names)==1 and n.names[0].name=='math'))]
    changed_errors = []
    class ExactPostcastError(ast.NodeTransformer):
        def visit_If(self, node):
            expected = ast.parse("if not bool(torch.isfinite(effective).all().item()) or bool((effective <= 0).any().item()):\n raise FP16ScaleOverflowError(_invalid_scale_diagnostics(groups, bits, lo, hi, raw_scale, effective))").body[0]
            if ast.dump(node, include_attributes=False) == ast.dump(expected, include_attributes=False):
                node.body = ast.parse("raise ValueError('invalid effective FP16 scale')").body
                changed_errors.append(True)
                return node
            return self.generic_visit(node)
    current = ExactPostcastError().visit(current)
    C.require(len(removed) == len(changed_errors) == 1 and ast.dump(previous, include_attributes=False) ==
              ast.dump(current, include_attributes=False),
              'Only premature guard deletion and error-only postcast diagnostics are admitted')
    return {'status': 'PASS_EXACT_ACCEPTED_DOMAIN_AST_AND_UNCHANGED_BUFFER_RUNTIME',
            'old_quantizer_sha256': C.sha(safe(old, 'src/pace_quantizer.py')),
            'new_quantizer_sha256': C.sha(safe(root, 'src/pace_quantizer.py')),
            'unchanged_files': {n: C.sha(safe(root, n)) for n in unchanged},
            'old_accepted_domain': 'Every quantization returned before the old raw_scale>65504 rejection',
            'new_behavior': 'Stored FP16 finite/positive guard remains; no mapping change in old accepted domain'}


def verify_identity(identity, stage, unit):
    C.require(isinstance(identity, dict) and type(identity.get('pid')) is int and identity['pid'] > 0 and
              type(identity.get('starttick')) is int and identity['starttick'] > 0 and
              identity.get('cmdline') == ['/data/ydai/miniconda3/envs/bitdecode/bin/python',
                  OLD_REMOTE + '/scripts/pace_worker.py', '--root', OLD_REMOTE,
                  '--stage', stage, '--unit', unit], 'Original exact worker identity differs')


def verify_old_unit(root, old, stage, unit):
    target = C.unit_target(old, stage, unit)
    dispatch = C.read(target / 'DISPATCH.json')
    identity = C.read(target / 'ACTUAL_CHILD_IDENTITY.json')
    started = C.read(target / 'STARTED.json')
    interval = C.read(target / 'RESOURCE_INTERVAL.json')
    verify_identity(identity, stage, unit)
    C.require(dispatch['task_id'] == C.TASK and dispatch['execution_root'] == OLD_REMOTE and
              dispatch['stage'] == stage and dispatch['unit'] == unit and dispatch['gpu'] == 0 and
              dispatch['gpu_uuid'] == C.UUID and dispatch['source_manifest_sha256'] == OLD_SOURCE_SHA and
              dispatch['permit_sha256'] == OLD_PERMIT_SHA and
              started['dispatch_sha256'] == C.sha(target / 'DISPATCH.json') and
              started['identity'] == interval['identity'] == identity and started['task_id'] == C.TASK and
              started['stage'] == stage and started['unit'] == unit,
              'Original dispatch/start/actual-exit binding differs')
    C.require(interval['task_id'] == C.TASK and interval['stage'] == stage and interval['unit'] == unit and
              interval['pid'] == identity['pid'] and interval['returncode'] == 0 and
              interval['released'] is True and interval['accounting_final'] is True and
              interval['lower_bound_only'] is False and interval['failure'] is None and
              not any(x['pid'] == identity['pid'] for x in interval['NVML']['apps']),
              'Original qualification/formal needs real exit0 and NVML release')
    driver = C.read(target / 'DRIVER_IDENTITY.json')
    C.require(driver['status'] == 'PASS' and driver['actual_uuid'] == C.UUID and
              driver['CUDA_VISIBLE_DEVICES'] == C.UUID and driver['logical_ordinal'] == 0 and
              driver['visible_device_count'] == 1, 'Original actual Driver UUID differs')
    wrapper = C.verify_result(old, stage, unit, identity)
    C.require(wrapper['science_freeze_sha256'] == (OLD_SCIENCE_SHA if stage in ('gate_d','formal') else None),
              'Original stage science binding differs')
    apps = [x for x in wrapper['postflight_NVML']['apps'] if x['pid'] == identity['pid']]
    C.require(apps and all(x['uuid'] == C.UUID for x in apps), 'Original actual NVML ownership differs')
    return wrapper, interval, target


def verify_fresh_review(root, equivalence):
    source_bound(root, FRESH_REVIEW)
    review = C.read(safe(root, FRESH_REVIEW))
    C.require(review['status'] == 'PASS_FRESH_DUAL_HOST_PACE_REUSE3_SOURCE_SCIENCE_MODEL' and
              set(review['hosts']) == {'55','29'} and review['formal_count'] == 3 and
              review['head_mask_sha256'] == MASK_SHA, 'Fresh independent two-host origin audit missing')
    for host, row in review['hosts'].items():
        C.require(row['status'] == 'PASS' and row['source_count'] == 133 and
                  row['source_sha256'] == OLD_SOURCE_SHA and row['science_freeze_sha256'] == OLD_SCIENCE_SHA,
                  'Fresh host origin/source/science audit differs: ' + host)
    C.require(review['hosts']['55']['full_model17_SHA_PASS'] is True,
              'Fresh full model17 bytes must be independently verified on execution host')
    proof = review['accepted_domain_equivalence']
    C.require(proof['status'] == 'PASS' and proof['old_quantizer_sha256'] == equivalence['old_quantizer_sha256'] and
              proof['new_quantizer_sha256'] == equivalence['new_quantizer_sha256'],
              'Fresh mathematical path equivalence proof differs')
    return C.sha(safe(root, FRESH_REVIEW))


def reuse_admission(root, *, require_fresh_review=True):
    root = Path(root).resolve()
    old = origin_root(root)
    for name, row in ORIGIN_CONTROL_PINS.items():
        rows_check(old, [dict(path=name, **row)])
        source_bound(root, SNAPSHOT + '/' + name)
    manifest = C.read(old / 'DELIVERY_MANIFEST.json')
    C.require(manifest['task_id'] == C.TASK and manifest['source_manifest_sha256'] == OLD_SOURCE_SHA and
              manifest['permit_sha256'] == OLD_PERMIT_SHA and manifest['count'] == len(manifest['files']) == 317 and
              manifest['total_bytes'] == 18360743, 'Fixed original science manifest identity/count differs')
    fresh = rows_check(old, manifest['files'])
    frozen = {x['path']: (x['bytes'], x['sha256']) for x in C.read(root/C.SOURCE)['files']}
    C.require(all(frozen.get(SNAPSHOT+'/'+x['path']) == (x['bytes'], x['sha256'])
                  for x in manifest['files']), 'Every original science/proof file must be new-source-bound')
    C.require(fresh['bytes'] == manifest['total_bytes'], 'Original science byte total differs')
    C.require(C.verify_sources(old) == OLD_SOURCE_SHA and C.read(old / C.SOURCE)['count'] == 133 and
              C.verify_science(old) == OLD_SCIENCE_SHA and
              C.sha(old / 'calibration/head_mask.json') == MASK_SHA, 'Original source/science/mask changed')
    C.verify_head_mask(old)
    for name in ('analysis/MODEL_TOKENIZER_FRESH_SHA.json', 'analysis/EXTERNAL_NATIVE_BINDINGS.json',
                 'analysis/EXTERNAL_PYTHON_BINDINGS.json'):
        old_rows = C.read(old / name)['files']; new_rows = C.read(root / name)['files']
        def content(rows):
            return {x['path']: (x['bytes'], x['sha256']) for x in rows}
        C.require(content(old_rows) == content(new_rows), 'Original model/runtime external bytes changed: ' + name)
        if name == 'analysis/MODEL_TOKENIZER_FRESH_SHA.json':
            C.require(len(old_rows) == len(new_rows) == 17 and
                      {x['path']:(x['inode'],x['mtime_ns']) for x in old_rows} ==
                      {x['path']:(x['inode'],x['mtime_ns']) for x in new_rows},
                      'All original model17 file stat identities must remain unchanged')
    eq = verify_equivalence(root, old)
    fresh_review_sha = verify_fresh_review(root, eq) if require_fresh_review else None
    proofs = []
    for stage, unit in [('math_gpu_gates','GATES_ABC'),('calibration','CALIBRATION_ONCE'),('gate_d','FULL_MODEL_D')] + [('formal',u) for u in REUSED_IDS]:
        wrapper, interval, target = verify_old_unit(root, old, stage, unit)
        if stage == 'math_gpu_gates':
            C.require(wrapper['result']['status'] == 'PASS' and wrapper['result']['actual_cuda'] is True and
                      all(wrapper['result']['gates'][g]['status'] == 'PASS' and
                          wrapper['result']['gates'][g]['actual_cuda'] is True for g in ('A','B','C')),
                      'Original CUDA ABC gate evidence differs')
        elif stage == 'calibration':
            C.require(wrapper['result']['status'] == 'PASS' and wrapper['result']['actual_cuda'] is True and
                      wrapper['result']['AIME_used_for_calibration'] is False,
                      'Original once-only non-AIME CUDA calibration differs')
        elif stage == 'gate_d':
            C.require(wrapper['result']['status'] == 'PASS' and wrapper['result']['actual_cuda'] is True and
                      set(wrapper['result']['configs']) == {'R2','R3'} and
                      wrapper['result']['actual_mask_gate']['status'] == 'PASS' and
                      len(wrapper['result']['actual_mask_gate']['cases']) == 48,
                      'Original48 actual-mask cases and dual-config D evidence differs')
        proofs.append(dict(stage=stage, unit=unit, original_root=OLD_REMOTE,
                           snapshot_relative_path=target.relative_to(root).as_posix(),
                           result_sha256=C.sha(target/'RESULT.json'), interval_sha256=C.sha(target/'RESOURCE_INTERVAL.json'),
                           original_identity=wrapper['identity']))
    failed = C.read(C.unit_target(old,'formal',FAILED_ID)/'RESOURCE_INTERVAL.json')
    C.require(failed['returncode'] == 1 and failed['released'] is True and failed['unit'] == FAILED_ID and
              not (C.unit_target(old,'formal',FAILED_ID)/'RESULT.json').exists(),
              'Original failed partial must retain its real failure and zero completion credit')
    session = C.read(old/'SESSION_RESULT.json')
    C.require(session['status'] == 'FAIL_CLOSED_STOP' and session['formal_completed'] == 3 and
              session['own_workers_exit_and_NVML_release'] is True and
              {x['unit'] for x in session['completed_units'] if x['stage']=='formal'} == set(REUSED_IDS),
              'Original true STOP counts changed')
    for unit in REUSED_IDS:
        C.require(not C.unit_target(root,'formal',unit).exists(),
                  'Old reuse must not create a current STARTED/RESULT/interval or duplicated trajectory')
    return dict(status='PASS_STRICT_FIXED_ORIGIN_REUSE3_AND_ONE_CALIBRATION', task_id=C.TASK,
                execution_revision=REVISION, origin_root=OLD_REMOTE, origin_archive=OLD_ARCHIVE,
                snapshot=SNAPSHOT, old_source_sha256=OLD_SOURCE_SHA, old_permit_sha256=OLD_PERMIT_SHA,
                old_science_freeze_sha256=OLD_SCIENCE_SHA, head_mask_sha256=MASK_SHA,
                reused_formal_count=3, reused_formal_ids=list(REUSED_IDS), remaining_formal_ids=[x['id'] for x in remaining_jobs(root)],
                old_failed_partial_credit=0, old_failed_partial_restarts_fresh=True,
                inherited_qualification_units=3, new_required_gate=dict(stage=GATE_STAGE, unit=GATE_UNIT),
                proofs=proofs, accepted_domain_equivalence=eq, independent_fresh_review_sha256=fresh_review_sha,
                historical_costs=dict(scope='Original133 execution only; never current intervals or physical GPU active time',
                    resource_ledger_sha256=C.sha(old/'RESOURCE_LEDGER.json'),
                    intervals=C.read(old/'RESOURCE_LEDGER.json')['intervals'],
                    physical_GPU_active_seconds='UNRESOLVED_NOT_MEASURED'))


def record_admission(root):
    value = reuse_admission(root)
    C.atomic_new(Path(root)/'CONTINUATION_ADMISSION.json', value)
    return value


def verify_current_unit(root, stage, unit):
    dispatch_allowed(root,stage,unit)
    root=Path(root);target=C.unit_target(root,stage,unit)
    interval=C.read(target/'RESOURCE_INTERVAL.json')
    C.require(interval['returncode']==0 and interval['released'] is True and interval['failure'] is None and
              interval['stage']==stage and interval['unit']==unit and interval['task_id']==C.TASK and
              interval['accounting_final'] is True and interval['lower_bound_only'] is False,
              'Current unit is not a real legal exit0/release')
    identity=C.read(target/'ACTUAL_CHILD_IDENTITY.json')
    C.require(identity==interval['identity'] and interval['pid']==identity['pid'] and
              type(identity['starttick']) is int and identity['starttick']>0 and
              identity['cmdline']==['/data/ydai/miniconda3/envs/bitdecode/bin/python',
                  C.REMOTE+'/scripts/pace_worker.py','--root',C.REMOTE,'--stage',stage,'--unit',unit],
              'Current actual child argv/starttick binding differs')
    dispatch=C.read(target/'DISPATCH.json');started=C.read(target/'STARTED.json')
    session=C.read(root/'SESSION_STARTED.json')
    C.require(dispatch['execution_root']==C.REMOTE and dispatch['controller']==session['identity'] and
              dispatch['task_id']==C.TASK and dispatch['stage']==stage and dispatch['unit']==unit and
              dispatch['gpu']==0 and dispatch['gpu_uuid']==C.UUID and
              dispatch['source_manifest_sha256']==session['source_manifest_sha256']==C.sha(root/C.SOURCE) and
              dispatch['permit_sha256']==session['permit_sha256']==C.sha(root/'GPU_PERMIT.json') and
              started['identity']==identity and started['task_id']==C.TASK and
              started['stage']==stage and started['unit']==unit and
              started['dispatch_sha256']==C.sha(target/'DISPATCH.json'),
              'Current dispatch/start/session source identity differs')
    driver=C.read(target/'DRIVER_IDENTITY.json')
    C.require(driver['status']=='PASS' and driver['actual_uuid']==C.UUID and driver['CUDA_VISIBLE_DEVICES']==C.UUID and
              driver['logical_ordinal']==0 and driver['visible_device_count']==1,
              'Current exact Driver UUID/release differs')
    # The frozen controller release_state permits an exited own Popen PID to
    # reappear only with a different actual starttick. A bare PID in the final
    # NVML snapshot therefore must not be reclassified as the exited worker.
    wrapper=C.verify_result(root,stage,unit,identity)
    own=[x for x in wrapper['postflight_NVML']['apps'] if x['pid']==identity['pid']]
    C.require(own and all(x['uuid']==C.UUID for x in own),'Current actual postflight own UUID differs')
    return wrapper,interval,target


def admit_reused_science(root):
    root = Path(root)
    admission = C.read(root/'CONTINUATION_ADMISSION.json')
    C.require(admission == reuse_admission(root), 'Reuse admission changed before new science freeze')
    target = C.unit_target(root,GATE_STAGE,GATE_UNIT)
    verify_current_unit(root,GATE_STAGE,GATE_UNIT)
    old = origin_root(root)
    for name in ('head_mask.json','head_scores.csv','calibration_manifest.json'):
        source, dest = old/'calibration'/name, root/'calibration'/name
        C.require(not dest.exists(), 'Do not overwrite or recalculate the once-only original calibration')
        dest.parent.mkdir(parents=True,exist_ok=True)
        raw=source.read_bytes()
        with dest.open('xb') as f:
            f.write(raw);f.flush();os.fsync(f.fileno())
        C.require(C.sha(dest)==C.sha(source),'Reused calibration copy byte identity differs')
    C.atomic_new(root/'REUSED_CALIBRATION_ADMISSION.json', dict(status='PASS_BYTE_IDENTICAL_ONCE_CALIBRATION_REUSE_AFTER_NEW_GATE',
                 task_id=C.TASK, execution_revision=REVISION, original_root=OLD_REMOTE,
                 continuation_admission_sha256=C.sha(root/'CONTINUATION_ADMISSION.json'),
                 current_gate_result_sha256=C.sha(target/'RESULT.json'),
                 current_gate_interval_sha256=C.sha(target/'RESOURCE_INTERVAL.json'),
                 calibration_GPU_reruns=0, mask=C.verify_head_mask(root)))
    return C.freeze_science(root)


def completed_rows(root, invalid=None):
    root = Path(root)
    rows = []
    admission = reuse_admission(root)
    old = origin_root(root)
    jobs = C.validate_jobs(root)
    for job in jobs:
        unit = job['id']
        if unit in REUSED_IDS:
            wrapper, _interval, target = verify_old_unit(root,old,'formal',unit)
            rows.append((job,wrapper,target))
            continue
        target=C.unit_target(root,'formal',unit)
        if not (target/'RESULT.json').exists() and not (target/'RESOURCE_INTERVAL.json').exists():
            continue
        try:
            wrapper,interval,target=verify_current_unit(root,'formal',unit)
            rows.append((job,wrapper,target))
        except BaseException as exc:
            if invalid is None: raise
            invalid.append(dict(unit=unit, origin='CURRENT_REPAIR_REVISION',error=repr(exc),
                                partial_completion_credit=0))
    C.require(len(rows)==len({j['id'] for j,_w,_t in rows}), 'Duplicate completion credit')
    return rows


def continuation_summary(root):
    root=Path(root)
    admission=reuse_admission(root)
    marker=root/'CONTINUATION_ADMISSION.json'
    runtime_admitted=marker.exists()
    if runtime_admitted:
        C.require(C.read(marker)==admission,'Runtime continuation admission differs from closed original provenance')
    ledger=C.read(root/'RESOURCE_LEDGER.json') if (root/'RESOURCE_LEDGER.json').exists() else {}
    completed=[x for x in ledger.get('completed_units',[]) if x.get('stage')=='formal']
    C.require(len(completed)==len({x['unit'] for x in completed}) and
              {x['unit'] for x in completed} <= {x['id'] for x in remaining_jobs(root)},
              'Current ledger duplicates or claims old completion units')
    return dict(task_id=C.TASK, execution_revision=REVISION, reused_formal_count=3,
                reused_formal_ids=list(REUSED_IDS), current_formal_count=len(completed),
                formal_total_unique=3+len(completed), planned_current_formal_count=57,
                origin_root=OLD_REMOTE, origin_archive=OLD_ARCHIVE, snapshot=SNAPSHOT,
                original_source_sha256=OLD_SOURCE_SHA,original_permit_sha256=OLD_PERMIT_SHA,
                original_science_freeze_sha256=OLD_SCIENCE_SHA,head_mask_sha256=MASK_SHA,
                current_intervals_only=True, original_intervals_are_historical=True,
                historical_formal_completed=3, runtime_reuse_admitted=runtime_admitted,
                admitted_reused_formal_count=3 if runtime_admitted else 0,
                original_failed_partial_credit=0,historical_costs=admission['historical_costs'],
                uninterrupted_continuous60=False)
