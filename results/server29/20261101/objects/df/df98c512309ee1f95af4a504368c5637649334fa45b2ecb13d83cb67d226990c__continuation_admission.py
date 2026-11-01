"""Strict, CPU-only provenance for a user-requested scheduling continuation.

An old payload remains an old payload. No RESULT/STARTED/interval is synthesized in
the new execution directories. No incomplete token stream can be admitted.
"""
from pathlib import Path
import execution_control as C

OLD_ROOT = Path('/data/zypan/analyses/qwen_gdn_group64_four_codec_aime26_seed1_v1/revisions/cross_host_v4')
OLD_ARCHIVE = Path('/data/zypan/qwen_group64_four_codec_aime26_archive/revisions/QWEN_GDN_GROUP64_FOUR_CODEC_AIME26_SEED1_V1/cross_host_v4')
AUTH_SHA = 'ef8e6fcc185c6df73d87be381c2a2dda4c94e9e1ac43268c10b7a63895514931'


def binding(root):
    p = C.read(Path(root) / 'REUSE_ADMISSION_STATIC.json')
    C.require(p['human_authorization_sha256'] == AUTH_SHA, 'Continuation authorization differs')
    C.require(set(p['origins']) == {'server55_gpu0', 'server29_gpu6'}, 'Only two exact v4 origins')
    return p


def origin_root(root, origin_node):
    p = binding(root)
    n = p['origins'][origin_node]
    C.require(n['root'] == str(OLD_ROOT), 'Unknown v4 origin path')
    # On29 the closed peer55 snapshot is a separate, verified frozen source tree.
    source = OLD_ROOT if n['node']['host'] == C.NODE['host'] else OLD_ROOT / 'peer55'
    C.require(C.sha(source / 'SOURCE_FREEZE.json') == n['source_sha256'] and
              C.sha(source / 'GPU_PERMIT.json') == n['permit_sha256'], 'Origin source/permit changed')
    return source


def qualification_source(root, unit_id):
    root = Path(root)
    p = binding(root)
    origin = p['qualification'].get(unit_id)
    if origin is None:
        C.require(unit_id in {u['id'] for u in C.read(root / 'GPU_EXECUTION_PLAN.json')['qualification_units']},
                  'Unknown qualification ID')
        return root
    source = origin_root(root, origin['origin_node'])
    directory = source / 'execution' / unit_id
    C.require(C.sha(directory / 'RESULT.json') == origin['result_sha256'] and
              C.sha(directory / 'RESOURCE_INTERVAL.json') == origin['interval_sha256'],
              'Frozen reused qualification bytes differ')
    if origin.get('trace_sha256') is not None:
        C.require(C.sha(directory / 'TRACE.json') == origin['trace_sha256'], 'Original qualification trace differs')
    return source


def host_reused_formal(root):
    p = binding(root)
    return {k: v for k, v in p['formal'].items() if v['job']['compute_host'] == C.NODE['host']}


def formal_source(root, job):
    root = Path(root)
    p = binding(root)
    reuse = p['formal'].get(job['id'])
    if reuse is not None:
        C.require(job == reuse['job'], 'Reused formal protocol/UUID/job changed')
        source = origin_root(root, reuse['origin_node'])
        directory = source / 'formal' / job['id']
        C.require(C.sha(directory / 'RESULT.json') == reuse['result_sha256'] and
                  C.sha(directory / 'RESOURCE_INTERVAL.json') == reuse['interval_sha256'],
                  'Reused complete formal bytes differ')
        return source
    return root if job['compute_host'] == C.NODE['host'] else root / 'peer55'


def admit_local_reuse(root):
    root = Path(root)
    p = binding(root)
    C.require(C.sha(root / 'HUMAN_THREE_GPU_CONTINUATION_AUTHORIZATION.json') == AUTH_SHA,
              'Human authorization raw bytes differ')
    mine = C.NODE['id']
    C.require(mine in p['origins'], 'Unexpected host controller profile')
    origin = origin_root(root, mine)
    C.verify_sources(origin)
    C.model_guard(origin)
    C.require(C.sha(OLD_ARCHIVE / 'FINAL_SEAL_RECEIPT.json') == p['archive_seal_sha256']
              if C.NODE['host'] == '202.38.247.29' else True, 'Original archive seal differs')
    stop = C.read(origin / 'GLOBAL_STOP_REQUEST.json')
    C.require(stop['status'] == 'USER_REQUESTED_CONTROLLED_STOP_FOR_THREE_GPU_CONTINUATION'
              and stop['human_authorization_sha256'] == AUTH_SHA, 'Not the explicitly authorized scheduling stop')
    C.require(not C.same_process(C.read(origin / 'GPU_RUN_STARTED.json')['identity']),
              'Origin controller still alive')
    qualification = []
    identities = []
    plan = C.read(root / 'GPU_EXECUTION_PLAN.json')['qualification_units']
    for unit in plan:
        if unit['id'] not in p['qualification']:
            continue
        entry = p['qualification'][unit['id']]
        C.require(entry['origin_node'] == mine and unit == entry['unit'], 'Qualification reuse was moved or modified')
        source = qualification_source(root, unit['id'])
        directory = source / 'execution' / unit['id']
        w = C.read(directory / 'RESULT.json')
        i = C.read(directory / 'RESOURCE_INTERVAL.json')
        v = dict(w)
        saved = v.pop('payload_sha256')
        C.require(C.objsha(v) == saved == entry['payload_sha256'] and w['status'] == 'PASS'
                  and w['source_freeze_sha256'] == C.sha(source / 'SOURCE_FREEZE.json')
                  and w['permit_sha256'] == C.sha(source / 'GPU_PERMIT.json'), 'Qualification payload identity changed')
        C.require(w['identity'] == i['identity'] == C.read(directory / 'STARTED.json')['identity']
                  and i['exit_returncode'] == 0 and i['release'] == 'PASS'
                  and w['gpu'] == unit['gpu'] and w['gpu_uuid'] == C.UUIDS[unit['gpu']],
                  'Qualification identity/exit/release/device missing')
        identities.append(i['identity'])
        qualification.append(unit['id'])
    C.require(len(qualification) == 19, 'Exactly one prior device19 is reused on each host')
    jobs = {j['id']: j for j in C.read(root / 'FORMAL_JOB_MANIFEST.json')['jobs']}
    formal = []
    from cross_host_contract import validate_formal_record
    nodes = {n['id']: n for n in C.read(root / 'CROSS_HOST_PLAN.json')['nodes']}
    for unit, entry in host_reused_formal(root).items():
        C.require(unit in jobs and jobs[unit] == entry['job'], 'Reused formal assignment differs')
        source = formal_source(root, jobs[unit])
        directory = source / 'formal' / unit
        w = C.read(directory / 'RESULT.json')
        validate_formal_record(w, jobs[unit], source, nodes[jobs[unit]['node_id']])
        i = C.read(directory / 'RESOURCE_INTERVAL.json')
        C.require(i['identity'] == w['identity'] and i['exit_returncode'] == 0
                  and i['release'] == 'PASS', 'Only completed exited released formal results can be reused')
        C.require(not (root / 'formal' / unit / 'DISPATCH.json').exists(), 'Reused sample was redispatched')
        identities.append(i['identity'])
        formal.append(unit)
    C.require(all(not C.same_process(i) for i in identities), 'Original admitted worker still alive')
    obs = C.query_gpu()
    C.require(not any(x['pid'] in {i['pid'] for i in identities} for x in obs['apps']),
              'Original admitted worker still holds GPU')
    return dict(status='PASS_STRICT_LOCAL_V4_REUSE_NO_NEW_GPU_CREDIT',
                qualification_ids=qualification, formal_ids=formal,
                original_root=str(origin), original_source_sha256=C.sha(origin / 'SOURCE_FREEZE.json'),
                original_permit_sha256=C.sha(origin / 'GPU_PERMIT.json'), new_fresh_qualification_count=0,
                original_identities=identities, no_prefix_splicing=True)
