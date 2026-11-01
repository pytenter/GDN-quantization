"""Post-STOP stdlib-only identity audit. No imports of model/GPU packages.

Inputs are frozen manifests. The only write is the supplemental receipt beneath
the explicitly supplied independent_delivery/source directory. No qualification
status, source, authorization, failure, or resource record is modified.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time

def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def stat_id(stat):
    return dict(device=stat.st_dev, inode=stat.st_ino, bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns)

def stable_hash(path, cache):
    p = Path(path)
    real = p.resolve(strict=True)
    before = stat_id(real.stat())
    key = tuple(before.values())
    if key not in cache:
        h = hashlib.sha256()
        with real.open('rb') as f:
            opened_before = stat_id(os.fstat(f.fileno()))
            if opened_before != before:
                raise RuntimeError('Changed before open: ' + str(p))
            for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
                h.update(block)
            opened_after = stat_id(os.fstat(f.fileno()))
        if opened_after != before:
            raise RuntimeError('Changed while hashing: ' + str(p))
        cache[key] = h.hexdigest()
    after = stat_id(real.stat())
    if before != after or real != p.resolve(strict=True):
        raise RuntimeError('Changed identity during observation: ' + str(p))
    return dict(path=str(p), resolved_path=str(real), sha256=cache[key],
                bytes=before['bytes'], symlink=p.is_symlink(),
                metadata_before=before, metadata_after=after,
                status='STABLE_CURRENT_CPU_HASH')

def verify_rows(base, rows, cache):
    out=[]
    for row in rows:
        path=Path(row['path'])
        if not path.is_absolute():
            path=base/path
        try:
            actual=stable_hash(path,cache)
            actual.update(expected_sha256=row['sha256'], expected_bytes=row['bytes'],
                          manifest_path=row['path'])
            actual['binding_status']='PASS' if (actual['sha256']==row['sha256'] and actual['bytes']==row['bytes']) else 'FAIL'
        except Exception as exc:
            actual=dict(manifest_path=row['path'],path=str(path),binding_status='FAIL',error=type(exc).__name__+':'+str(exc))
        out.append(actual)
    return dict(status='PASS' if all(x['binding_status']=='PASS' for x in out) else 'FAIL',
                count=len(out),passed=sum(x['binding_status']=='PASS' for x in out),rows=out)

def manifest_identity(path, expected, cache):
    actual=stable_hash(path,cache)
    actual['expected_sha256']=expected
    actual['binding_status']='PASS' if actual['sha256']==expected else 'FAIL'
    return actual

def main(config):
    root=Path(config['remote_root']).resolve(strict=True)
    cache={}
    start=time.time()
    receipt=dict(task_id=config['task_id'],execution_run_id=config['execution_run_id'],
                 audit_kind='POST_STOP_READ_ONLY_SOURCE_AND_DEPENDENCY_IDENTITY',
                 created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 remote_root=str(root),auditor_script_sha256=config['auditor_script_sha256'],
                 GPU_imports=False,GPU_queries=False,GPU_dispatches=False,
                 current_hash_is_not_retrospective_loaded_map_proof=True,
                 frozen_records_unchanged_by_auditor=True)
    payload=read(root/'PAYLOAD_BUILD_MANIFEST.json')
    control=read(root/'CONTROL_SOURCE_MANIFEST.json')
    receipt['payload_manifest_identity']=manifest_identity(root/'PAYLOAD_BUILD_MANIFEST.json',config['payload_build_manifest_sha256'],cache)
    receipt['control_manifest_identity']=manifest_identity(root/'CONTROL_SOURCE_MANIFEST.json',config['control_manifest_sha256'],cache)
    receipt['payload_files']=verify_rows(root,payload['files'],cache)
    receipt['control_files']=verify_rows(root,control['files'],cache)
    receipt['cards']={}
    external_lists=[]
    library_lists=[]
    for card in config['cards']:
        key=card['card_key']
        child=Path(card['remote_root'])
        source=read(child/'SOURCE_MANIFEST.json')
        auth=read(child/'AUTHORIZATION.json')
        origin=read(child/'RUN_COPY_ORIGIN.json')
        contract=read(child/'analysis/STRICT_BASELINE_CONTRACT.json')
        plan=read(child/'qualification/CUBLAS_PREREGISTRATION.json')
        identities={
            name:manifest_identity(child/name,expected,cache)
            for name,expected in [
                ('SOURCE_MANIFEST.json',card['source_manifest_sha256']),
                ('AUTHORIZATION.json',card['authorization_sha256']),
                ('analysis/STRICT_BASELINE_CONTRACT.json',card['contract_sha256']),
                ('qualification/CUBLAS_PREREGISTRATION.json',card['preregistration_sha256']),
                ('ARCHIVE_CONFIG.json',card['archive_config_sha256']),
                ('RUN_COPY_ORIGIN.json',auth['run_copy_origin_sha256']),
                ('HUMAN_GPU_AUTHORIZATION.json',auth['human_authorization_sha256']),
                ('GPU_EXECUTION_AMENDMENT.json',auth['amendment_sha256'])]}
        links={
            'root_exact':auth['root']==str(child) and plan['future_execution_root']==str(child) and origin['execution_root']==str(child),
            'source_hash_auth_origin_exact':auth['source_manifest_sha256']==card['source_manifest_sha256']==origin['source_manifest_sha256'],
            'contract_prereg_archive_hashes_auth_exact':all(auth[a]==card[a] for a in ('contract_sha256','preregistration_sha256','archive_config_sha256')),
            'same_device_source_contract_auth_origin_plan':source['device']==contract['execution_device_binding']==auth['device']==origin['device']==plan['device'],
            'shared_root_card_roster_exact':auth['shared_control_root']==str(root) and auth['card_roots']=={x['card_key']:x['remote_root'] for x in config['cards']},
            'execution_run_exact':all(x.get('execution_run_id')==config['execution_run_id'] for x in [source,auth,origin,contract,plan]),
            'origin_outside_execution_sources':not any(x['path']=='RUN_COPY_ORIGIN.json' for x in source['execution_sources']),
            'all_16_unit_frozen_roster':len(auth['units'])==16 and auth['units']==plan['units'],
            'no_elapsed_GPU_limit':auth['no_GPU_time_limit'] is True and plan['no_GPU_time_limit'] is True and auth['GPU_time_limit'] is None,
        }
        child_row=dict(manifest_identities=identities,binding_links=links,
                       execution_files=verify_rows(child,source['execution_sources'],cache),
                       external_manifest_rows=len(source['external_dependencies']),
                       external_list_canonical_sha256=sha_bytes(json.dumps(source['external_dependencies'],sort_keys=True,separators=(',',':')).encode()),
                       library_bindings_count=len(contract['prospective_library_bindings']))
        receipt['cards'][key]=child_row
        external_lists.append(source['external_dependencies'])
        library_lists.append(contract['prospective_library_bindings'])
        sys.stderr.write(key+' execution closure verified\n');sys.stderr.flush()
    receipt['external_rows_identical_across_cards']=all(x==external_lists[0] for x in external_lists)
    receipt['prospective_library_bindings_identical_across_cards']=all(x==library_lists[0] for x in library_lists)
    receipt['fresh_external_dependencies']=verify_rows(root,external_lists[0],cache)
    librows=[x for x in library_lists[0] if 'path' in x and 'sha256' in x]
    receipt['fresh_bound_library_rows']=verify_rows(root,librows,cache)
    # Supplement only: this path was absent from the frozen prospective bindings.
    audio=Path('/data/ydai/miniconda3/envs/bitdecode/lib/python3.10/site-packages/torchaudio/lib/libtorchaudio.so')
    supplemental=stable_hash(audio,cache)
    allowed={str(Path(x.get('resolved_path',x['path'])).resolve()) for x in librows}
    supplemental.update(in_any_frozen_prospective_library_binding=str(audio.resolve()) in allowed,
                        in_any_frozen_external_dependency=any(Path(x['path']).resolve()==audio.resolve() for x in external_lists[0]),
                        historical_loaded_maps='NOT_RETAINED',runtime_binary_identity='UNRESOLVED_AT_FAILURE',
                        current_disk_hash_does_not_repair_FAILED_unit=True)
    receipt['unbound_lib_current_identity']=supplemental
    # Source/metadata text reads are current CPU observations, never imports.
    package=audio.parents[1]
    metadata_paths=[package/'version.py',package/'_extension/__init__.py',package/'_extension/utils.py']
    site=package.parent
    metadata_paths+=sorted(site.glob('torchaudio-*.dist-info/METADATA'))
    metadata=[]
    for path in metadata_paths:
        if path.exists():
            row=stable_hash(path,cache)
            row['text']=path.read_text(errors='replace')
            metadata.append(row)
    receipt['torchaudio_current_metadata_sources']=metadata
    # Actual failure and unit receipts are retained verbatim as read-only evidence.
    failure=root/'FIRST_FAILURE.json'
    receipt['global_first_failure']=dict(identity=stable_hash(failure,cache),content=read(failure))
    receipt['unit_records']={}
    for card in config['cards']:
        child=Path(card['remote_root'])
        key=card['card_key']
        rows=[]
        for unitdir in sorted((child/'execution').glob('*')):
            if not unitdir.is_dir():continue
            files={}
            for name in ('STARTED.json','RESULT.json','DISPATCH.json'):
                path=unitdir/name
                if path.exists():files[name]=dict(identity=stable_hash(path,cache),content=read(path))
            progress=unitdir/'PROGRESS.jsonl'
            if progress.exists():
                allprogress=[json.loads(x) for x in progress.read_text().splitlines() if x.strip()]
                commits=[x for x in allprogress if x.get('progress_kind')=='token_committed']
                files['PROGRESS.jsonl']=dict(identity=stable_hash(progress,cache),line_count=len(allprogress),last_record=allprogress[-1] if allprogress else None,token_commit_count=len(commits),max_token=max((x.get('token',-1) for x in commits),default=None))
            rows.append(dict(unit_id=unitdir.name,files=files))
        receipt['unit_records'][key]=rows
    binding_ok=(all(receipt[x]['binding_status']=='PASS' for x in ('payload_manifest_identity','control_manifest_identity'))
                and all(receipt[x]['status']=='PASS' for x in ('payload_files','control_files','fresh_external_dependencies','fresh_bound_library_rows'))
                and receipt['external_rows_identical_across_cards'] and receipt['prospective_library_bindings_identical_across_cards']
                and all(x['execution_files']['status']=='PASS' and all(v['binding_status']=='PASS' for v in x['manifest_identities'].values()) and all(x['binding_links'].values()) for x in receipt['cards'].values()))
    receipt['status']='CURRENT_FROZEN_IDENTITIES_PASS_REAL_STOP_REMAINS' if binding_ok else 'CURRENT_IDENTITY_AUDIT_FAILURE'
    receipt['qualification_status']='FAIL_CLOSED_STOP_UNCHANGED'
    receipt['unique_current_physical_files_hashed']=len(cache)
    receipt['elapsed_seconds']=time.time()-start
    receipt['closed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    outroot=root/'independent_delivery/source';outroot.mkdir(parents=True,exist_ok=True)
    target=outroot/config['receipt_name']
    encoded=(json.dumps(receipt,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode()
    with target.open('xb') as f:
        f.write(encoded);f.flush();os.fsync(f.fileno())
    sys.stdout.buffer.write(encoded)

