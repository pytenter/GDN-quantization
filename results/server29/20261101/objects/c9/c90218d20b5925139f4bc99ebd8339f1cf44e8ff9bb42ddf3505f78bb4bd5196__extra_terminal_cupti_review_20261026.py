"""Independent post-delivery CPU verification; no scientific execution or writes."""
import datetime
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path, PurePosixPath
import traceback

ROOT = Path(__file__).resolve().parents[1]
SOURCE = '3ece375b472e11930c26bdbe75b6d528cc41a2c65cdef97e0710d24fae294c60'
PERMIT = '297ce51ed07dac2f7592f5702aa32c5bb10e0b11fc98157749dd9e96dc44c06b'
REV = ROOT.name
REMOTE = '/dev/shm/qwen_wscale_rowbal_cupti_repair_20261025'
ARCHIVE = '/data/zypan/qwen_gdn_wscale_rowbal_archive/' + REV
BASE = 'QWEN_GDN_SYM8_G32_WSCALE_ROWBAL_AIME26_SEED1_V1'
UUIDS = {1: 'GPU-596e6591-b065-bb91-fd35-a1e90ba8f9e3',
         2: 'GPU-b7ed673d-18ba-39ca-d7bc-c846af4f6b31',
         3: 'GPU-30b07aba-a728-7649-4899-e09d33f8bc75'}


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    p = Path(path)
    a = p.stat()
    h = hashlib.sha256()
    with p.open('rb') as f:
        b = os.fstat(f.fileno())
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
        c = os.fstat(f.fileno())
    d = p.stat()
    assert len({(x.st_dev, x.st_ino, x.st_size, x.st_mtime_ns) for x in (a, b, c, d)}) == 1, str(p)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_bytes())


def new(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def remote_review(cfg):
    import base64
    import hashlib
    import json
    import os
    from pathlib import Path, PurePosixPath
    import subprocess

    def read(p):
        return json.loads(Path(p).read_bytes())

    def safe(n):
        p = PurePosixPath(n)
        assert n and not p.is_absolute() and '..' not in p.parts and str(p) == n and '\\' not in n
        return n

    def digest(p):
        p = Path(p)
        a = p.stat()
        h = hashlib.sha256()
        with p.open('rb') as f:
            b = os.fstat(f.fileno())
            for v in iter(lambda: f.read(8 * 1024 * 1024), b''):
                h.update(v)
            c = os.fstat(f.fileno())
        d = p.stat()
        assert len({(x.st_dev, x.st_ino, x.st_size, x.st_mtime_ns) for x in (a,b,c,d)}) == 1, str(p)
        return h.hexdigest()

    def verify(root, rows):
        names = set()
        for row in rows:
            n = safe(row['path'])
            assert n not in names
            names.add(n)
            p = root / n
            assert p.is_file() and not p.is_symlink() and p.stat().st_size == row['bytes'], str(p)
            assert digest(p) == row['sha256'], str(p)
        return sum(x['bytes'] for x in rows)

    def absverify(row, stat=False):
        p = Path(row['path'])
        assert p.is_absolute() and p.is_file() and p.stat().st_size == row['bytes'], str(p)
        assert digest(p) == row['sha256'], str(p)
        if stat:
            s = p.stat()
            for k, field in [('inode','st_ino'),('device','st_dev'),('mtime_ns','st_mtime_ns')]:
                if k in row:
                    assert row[k] == getattr(s, field), (str(p),k)

    roots = []
    identities = []
    for item in cfg['roots']:
        root = Path(item['root'])
        assert digest(root/'SOURCE_FREEZE.json') == item['source_sha256']
        source = read(root/'SOURCE_FREEZE.json')
        assert len(source['files']) == source['count'] == item['source_count']
        verify(root, source['files'])
        manifest = read(root/'SHA256_MANIFEST.json')
        total = verify(root, manifest['files'])
        seal = read(root/'FINAL_SEAL_RECEIPT.json')
        assert digest(root/'SHA256_MANIFEST.json') == seal['manifest_sha256']
        assert digest(root/'FINAL_ARCHIVE_VERIFICATION.json') == seal['archive_verification_sha256']
        assert manifest['status'] == seal['scientific_status'] == 'PARTIAL_RESULTS_PRESERVED_AND_STOP'
        if (root/'RESOURCE_LEDGER.json').exists():
            identities.extend(x['identity'] for x in read(root/'RESOURCE_LEDGER.json')['intervals'] if x.get('identity'))
        roots.append(dict(root=str(root),source_count=source['count'],science_count=len(manifest['files']),
                          science_bytes=total,manifest_sha256=seal['manifest_sha256'],
                          seal_sha256=digest(root/'FINAL_SEAL_RECEIPT.json')))

    current = Path(cfg['roots'][0]['root'])
    assert digest(current/'GPU_PERMIT.json') == cfg['permit']
    origin = Path(cfg['roots'][1]['root'])
    cal = read(origin/'CALIBRATION_SOURCE_FREEZE.json')
    assert len(cal['files']) == 31
    verify(origin, cal['files'])
    raw = read(origin/'calibration/RAW_CLONE_VERIFICATION.json')
    assert raw['count'] == len(raw['files']) == 4323
    rawbytes = verify(origin, raw['files'])
    assert rawbytes == raw['bytes']
    assert digest(origin/'calibration/FP_COLLECTION_MANIFEST.json') == raw['collection_manifest_sha256']

    ledger = read(current/'RESOURCE_LEDGER.json')
    session = read(current/'SESSION_RESULT.json')
    intervals = ledger['intervals']
    assert len(intervals) == len({x['unit'] for x in intervals}) == 40
    assert all(x['release'] == 'PASS' and type(x['exit_returncode']) is int for x in intervals)
    legal = {x['unit'] for x in intervals if x['exit_returncode'] == 0}
    assert legal == set(ledger['completed_units']) == set(session['completed'])
    assert len(legal) == 37 and session['status'] == 'FAIL_CLOSED_STOP' and session['own_all_UUID_release'] is True
    jobs = {x['id']: x for x in read(current/'FORMAL_JOB_MANIFEST.json')['jobs']}
    formal = legal & set(jobs)
    counts = {m:sum(jobs[n]['condition']==m for n in formal) for m in ('W','R','RW')}
    assert counts == {'W':12,'R':11,'RW':10}
    q = read(current/'GPU_CODEC_QUALIFICATION.json')
    assert q['inspection']['count'] == len(q['units']) == 52
    failure = read(current/'FIRST_FAILURE.json')
    assert failure['unit'] == 'aime26_05_R_seed1'
    bad = next(x for x in intervals if x['unit'] == failure['unit'])
    assert bad['exit_returncode'] == 1 and bad['release'] == 'PASS'
    log = (current/'formal'/failure['unit']/'worker.log').read_text()
    assert 'FloatingPointError: nonfinite/illegal quantizer field; no state writeback allowed' in log
    prog = read(current/'formal'/failure['unit']/'PROGRESS.json')
    assert prog['generated_tokens'] == 35328 and prog['input_forwards'] == 35471

    lifecycle = ['CUPTI_PRELOAD_ADMISSION.json','PROFILE_COLLECTION_COMPLETED.json',
                 'CUPTI_PROFILE_DETACH.json','CUPTI_PROFILE_DETACH_EVENTS.jsonl']
    snapshots = []
    native = {}
    for interval in intervals:
        name = interval['unit']
        target = current/('formal' if name in jobs else 'execution')/name
        assert read(target/'RESOURCE_INTERVAL.json') == interval
        assert read(target/'STARTED.json')['identity'] == interval['identity']
        driver = read(target/'DRIVER_IDENTITY.json')
        uuid = cfg['uuids'][str(interval['gpu'])]
        assert driver['status']=='PASS' and driver['actual_uuid']==driver['expected_uuid']==driver['CUDA_VISIBLE_DEVICES']==uuid
        assert driver['visible_device_count']==1 and driver['logical_ordinal']==0
        receipt = read(target/'DURABLE29_RECEIPT.json')
        dm = read(target/'DURABLE29_MANIFEST.json')
        assert digest(target/'DURABLE29_MANIFEST.json') == receipt['manifest_sha256']
        assert receipt['actual_exit_returncode']==interval['exit_returncode'] and receipt['release']=='PASS'
        verify(target,dm['files'])
        if cfg['role']=='29':
            live=Path(cfg['live'])/'units'/name
            assert digest(live/'DURABLE29_MANIFEST.json')==receipt['manifest_sha256']
            assert read(live/'DURABLE29_RECEIPT.json')==receipt
            verify(live,dm['files'])
        if name in jobs:
            assert all(not (target/f).exists() for f in lifecycle)
        else:
            pre=read(target/lifecycle[0]);done=read(target/lifecycle[1]);detach=read(target/lifecycle[2])
            assert pre['identity']==done['identity']==detach['identity']==interval['identity']
            assert all(x['source_freeze_sha256']==cfg['source'] and x['permit_sha256']==cfg['permit'] for x in (pre,done,detach))
            assert detach['flush_returncode']==detach['finalize_returncode']==0 and interval['exit_returncode']==0
            events=[json.loads(x) for x in (target/lifecycle[3]).read_text().splitlines() if x]
            assert [x['phase'] for x in events]==['CYCLIC_GC_WHILE_CUDA_RUNTIME_ALIVE','CUDA_WORK_SYNCHRONIZED_BEFORE_ACTIVITY_FLUSH','CUPTI_ACTIVITY_FLUSH_BEGIN','CUPTI_ACTIVITY_FLUSH_RETURN','CUPTI_FINALIZE_BEGIN','CUPTI_FINALIZE_RETURN']
        for which in ('PREFLIGHT_LIBRARY.json','POSTFLIGHT_LIBRARY.json'):
            p=target/which
            if not p.exists():
                assert which=='POSTFLIGHT_LIBRARY.json' and interval['exit_returncode']!=0
                continue
            snap=read(p)
            assert snap['status']=='PASS' and not snap['missing']
            paths={x['path'] for x in snap['selected']}
            assert paths and len(paths)==len(snap['selected'])
            segments={x['path'] for x in snap['segments']}
            assert segments==paths
            for x in snap['selected']:
                assert x['match'] is True
                prior=native.setdefault(x['path'],x)
                assert (prior['bytes'],prior['sha256'])==(x['bytes'],x['sha256'])
            if cfg['role']=='55':
                for x in snap['segments']:
                    assert x['actual_inode']==x['mapped_inode']==Path(x['path']).stat().st_ino
            snapshots.append(dict(unit=name,phase=which,native_files=len(paths)))

    dependency_result={'status':'ARCHIVE_NOT_EXECUTION_ENVIRONMENT'}
    if cfg['role']=='55':
        bindings={}
        env=current/'inherited/group32/inherited/historical_port_payload/analysis/environment'
        for name in ('PROSPECTIVE_MODEL_AND_NATIVE_BINDINGS.json','CPU_FIXTURE_CRITICAL_NATIVE_BINDINGS.json'):
            obj=read(env/name)
            rows=[]
            for k in ('model_weight_shards','model_small_files','native_library_inventory','native_library_rows','torch_python_files','external_source_files'):
                rows.extend(obj.get(k,[]))
            rows.append(obj.get('python_executable',{}))
            for x in rows:
                if not all(k in x for k in ('path','bytes','sha256')):continue
                key=str(Path(x['path']).resolve());old=bindings.setdefault(key,x)
                assert (old['bytes'],old['sha256'])==(x['bytes'],x['sha256'])
        model=read(current/'analysis/MODEL_TOKENIZER_FRESH_SHA.json')
        assert model['count']==len(model['files'])==17
        triton=read(current/'analysis/TRITON_RUNTIME_FRESH_SHA.json')
        tr=triton['files']+triton['distribution_metadata']+triton['host_c_compiler']
        assert len(tr)==322
        for x in model['files']+tr:
            key=str(Path(x['path']).resolve());old=bindings.setdefault(key,x)
            assert (old['bytes'],old['sha256'])==(x['bytes'],x['sha256'])
        assert len(bindings)==386
        for x in bindings.values():absverify(x)
        for x in model['files']:absverify(x,True)
        for x in native.values():absverify(x)
        cupti=read(current/'analysis/CUPTI_LIFECYCLE_BINDING.json')
        extra=cupti['files']+[cupti['header']]
        assert len(extra)==2
        for x in extra:absverify(x,True)
        dependency_result=dict(status='PASS_FRESH_SHA_BYTES_INODE',external_count=len(bindings),model_count=17,triton_count=len(tr),
                               mapped_unique=len(native),mapped_receipts=len(snapshots),explicit_CUPTI_ELF_header=2)

    if cfg['role']=='55':
        chain=read(current/'OFFLINE_CHAIN_OBSERVED_EXIT.json')
    else:
        latepath=Path(cfg['supplement']['root'])/'INDEPENDENT_LATE_EXIT_AND_DELIVERY.json'
        assert digest(latepath)==cfg['supplement']['sha256']
        chain=read(latepath)['offline_chain_wait']
    assert chain['controller_shell_wait_status']==1 and chain['finisher_shell_wait_status']==0
    assert chain['source_freeze_sha256']==cfg['source'] and chain['permit_sha256']==cfg['permit']
    # Exact production identity fields; extra observational fields are independent data.
    def collect(value):
        if isinstance(value,dict):
            if all(k in value for k in ('pid','starttick','cmdline')) and any(any(r in a for r in cfg['process_roots']) for a in value['cmdline']):
                identities.append({k:value[k] for k in ('pid','starttick','cmdline')})
            for v in value.values():collect(v)
        elif isinstance(value,list):
            for v in value:collect(v)
    collect(chain)
    collect(cfg['launch'])
    process_result={'role':'ARCHIVE_NO_PROCESS_ASSERTION'}
    if cfg['role']=='55':
        distinct={(x['pid'],x['starttick'],tuple(x['cmdline'])) for x in identities}
        exact_alive=[]
        replacements=0
        for pid,tick,argv in distinct:
            p=Path('/proc')/str(pid)
            try:
                stat=(p/'stat').read_text().rsplit(')',1)[1].split();actualtick=int(stat[19])
                actual=tuple(a.decode() for a in (p/'cmdline').read_bytes().split(b'\0') if a)
            except (FileNotFoundError,ProcessLookupError):continue
            if actualtick==tick and actual==argv:exact_alive.append(pid)
            else:replacements+=1
        assert not exact_alive
        scoped=[]
        for p in Path('/proc').iterdir():
            if not p.name.isdigit():continue
            try:argv=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x]
            except (FileNotFoundError,ProcessLookupError,PermissionError):continue
            if any(any(a==r or a.startswith(r+'/') for r in cfg['process_roots']) for a in argv):scoped.append(int(p.name))
        assert not scoped,scoped
        out=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        ownpids={x[0] for x in distinct}
        apps=[x.split(',') for x in out.stdout.strip().splitlines() if x.strip()]
        owned=[x for x in apps if int(x[0].strip()) in ownpids and x[1].strip() in cfg['uuids'].values()]
        assert not owned
        process_result=dict(status='PASS_EXACT_IDENTITY_ABSENCE_OWN_UUID_RELEASE',recorded_identities=len(distinct),
                            replacement_PIDs_preserved=replacements,scoped_processes=scoped,own_NVML_apps=owned,
                            foreign_NVML_records=len(apps),signals=0)

    supplement_result=None
    if cfg['role']=='29':
        s=cfg['supplement'];p=Path(s['root'])/'INDEPENDENT_LATE_EXIT_AND_DELIVERY.json'
        assert p.stat().st_size==s['bytes'] and digest(p)==s['sha256']
        payload=read(p)
        assert payload['primary_seal_rewritten'] is False
        late=payload['closed_late_files']['files']
        for x in late:
            data=base64.b64decode(x['base64'])
            assert len(data)==x['bytes'] and hashlib.sha256(data).hexdigest()==x['sha256']
        supplement_result=dict(status='PASS_FRESH_SUPPLEMENT_BYTES',sha256=s['sha256'],bytes=s['bytes'],late_files=len(late))
    return dict(status='PASS_EXTRA_INDEPENDENT_READONLY_CPU',role=cfg['role'],roots=roots,
                raw_calibration_files=4323,raw_calibration_bytes=rawbytes,source31_count=31,
                qualification_PASS=52,formal_complete=33,formal_by_method=counts,intervals=40,durable_units=40,
                nonzero_intervals=[{'unit':x['unit'],'exit_returncode':x['exit_returncode'],'release':x['release']} for x in intervals if x['exit_returncode']!=0],
                native_receipts=snapshots,dependencies=dependency_result,processes=process_result,supplement=supplement_result,
                first_failure=dict(unit=failure['unit'],actual_exit=1,last_generated=prog['generated_tokens'],input_forwards=prog['input_forwards'],
                                   error='FloatingPointError: nonfinite/illegal quantizer field; no state writeback allowed',exact_field_root_cause='UNRESOLVED_NOT_SAVED'),
                source_mutations=0,remote_writes=0,signals=0,GPU_calls=0,model_loads=0,scoring_calls=0,finisher_calls=0)


def main():
    claim = ROOT/'operations/EXTRA_TERMINAL_CUPTI_REVIEW_CLAIM.json'
    result = ROOT/'operations/EXTRA_TERMINAL_CUPTI_REVIEW.json'
    failure = ROOT/'operations/EXTRA_TERMINAL_CUPTI_REVIEW_FAILURE.json'
    assert not any(p.exists() for p in (claim,result,failure))
    new(claim,dict(UTC=stamp(),pid=os.getpid(),source=SOURCE,permit=PERMIT,auditor_sha256=sha(__file__),no_automatic_retry=True))
    try:
        delivery=load(ROOT/'operations/LOCAL_REUSE48_TERMINAL_DELIVERY_VERIFICATION.json')
        assert delivery['status']=='PASS_INDEPENDENT55_29_SOURCE_SCIENCE_CALIBRATION_AND_MAC_TEXT_DELIVERY'
        assert sha(ROOT/'SOURCE_FREEZE.json')==SOURCE and sha(ROOT/'GPU_PERMIT.json')==PERMIT
        source=load(ROOT/'SOURCE_FREEZE.json')
        for x in source['files']:
            p=ROOT/x['path'];assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256']
        dest=ROOT/'REUSE48_DELIVERY_FROM55'
        physical=load(dest/'LOCAL_PHYSICAL_FILES.json')['files']
        for x in physical:
            p=dest/x['path'];assert p.is_file() and not p.is_symlink() and p.stat().st_size==x['bytes'] and sha(p)==x['sha256']
        assert len(physical)==delivery['Mac_delivered_files']==1049
        launch=load(ROOT/'operations/OFFLINE_LAUNCH_COMMAND_PROCESS_RECEIPT_LOCAL.json')
        tpath=ROOT.parent/'QWEN_GDN_PACE48_AIME26_SEED1_60_V1/operations/remote.py'
        assert sha(tpath)=='db0ea71612d633b0fc33feef1d95c42e273ab2bd9ffc971b8d7e4ddc97f74f40'
        spec=importlib.util.spec_from_file_location('independent_transport',tpath)
        t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
        lineage=[('',100,'ee2c01ff3a4963ba76a3df813bf22640de0c2de085ea82a2a297e067ac9f33b7'),
                 ('_REDUCTION_REPAIR_20261025',299,'6e87495d2dac7ccc7d97363256be3862ab7a52159271b848a58f06e2eabd1f4e'),
                 ('_BYTEVIEW_REPAIR_20261025',334,'05901486d72e6f62aba67e61961d6d62043dabef14340b46034a072a004ce5e1'),
                 ('_REUSE48_ARCHIVE29_20261025',368,'0ef900e88640bd255bdf29f6da8efe5b3d2e05b00c1c32fb74027e27a3c33d50')]
        process_roots=[REMOTE,'/tmp/'+BASE,'/tmp/'+BASE+lineage[1][0],'/tmp/'+BASE+lineage[2][0],'/dev/shm/qwen_wscale_rowbal_reuse48_archive29_20261025']
        audits={}
        for role,current in [('55',REMOTE),('29',ARCHIVE)]:
            roots=[dict(root=current,source_count=448,source_sha256=SOURCE)]
            for suffix,count,digest in lineage:
                p=('/tmp/'+BASE+suffix) if role=='55' else '/data/zypan/qwen_gdn_wscale_rowbal_archive/'+BASE+suffix
                if role=='55' and count==368:p=process_roots[-1]
                roots.append(dict(root=p,source_count=count,source_sha256=digest))
            cfg=dict(role=role,roots=roots,source=SOURCE,permit=PERMIT,uuids={str(k):v for k,v in UUIDS.items()},
                     live='/data/zypan/qwen_gdn_wscale_rowbal_archive/LIVE_'+REV,
                     supplement=delivery['late_identity_supplement'],process_roots=process_roots,launch=launch)
            code='import json\n'+inspect.getsource(remote_review)+'\nprint(json.dumps(remote_review('+repr(cfg)+'),sort_keys=True,allow_nan=False))\n'
            audits[role]=t.run(code,role=role,python='/data/ydai/miniconda3/envs/bitdecode/bin/python' if role=='55' else '/usr/bin/python3',timeout=1800)
            new(ROOT/('operations/EXTRA_TERMINAL_CUPTI_REVIEW_'+role+'.json'),audits[role])
        for a,b in zip(audits['55']['roots'],audits['29']['roots']):
            for k in ('source_count','science_count','science_bytes','manifest_sha256','seal_sha256'):assert a[k]==b[k],k
        out=dict(status='PASS_EXTRA_ALL_SOURCE_SCIENCE_NATIVE_MODEL_ORIGINS_29_SUPPLEMENT_MAC',UTC=stamp(),
                 auditor_sha256=sha(__file__),source=SOURCE,permit=PERMIT,audits=audits,
                 local_frozen_source_files=len(source['files']),Mac_files=len(physical),Mac_bytes=sum(x['bytes'] for x in physical),
                 primary_seals_unchanged=True,full_accuracy_reported=False,GPU_calls=0,signals=0,finisher_calls=0,scoring_calls=0)
        new(result,out)
        print(json.dumps({k:out[k] for k in ('status','local_frozen_source_files','Mac_files','Mac_bytes')}))
    except BaseException as e:
        new(failure,dict(status='FAIL_EXTRA_CPU_REVIEW_NO_AUTOMATIC_RETRY',UTC=stamp(),error=repr(e),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    main()
