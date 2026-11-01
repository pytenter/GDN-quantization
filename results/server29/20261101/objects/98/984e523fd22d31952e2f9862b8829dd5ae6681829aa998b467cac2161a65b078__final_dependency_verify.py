"""Run on compute host after GPU release; fresh current dependency stream SHA.

Copy this file into the new experiment's analysis_scripts/ before execution.
No Torch import, tensor deserialization, model load, GPU action, or scientific
score interpretation. Every output is new-only; historical inputs are read-only.
"""
import argparse
import hashlib
import json
import pathlib
import sys
import time
import traceback

TASK='QWEN_GDN_HADAMARD_DUAL_AXIS_DIRECT_32K_PILOT_V1'


def need(ok,message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for block in iter(lambda:f.read(16*1024**2),b''):
            h.update(block)
    return h.hexdigest()


def stat_id(path):
    s=pathlib.Path(path).stat()
    return dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino,device=s.st_dev)


def verify_file(entry):
    p=pathlib.Path(entry['path']);need(p.is_file(),'Missing dependency '+str(p))
    before=stat_id(p);digest=sha(p);after=stat_id(p)
    need(before==after,'Dependency changed during full stream hash: '+str(p))
    need(before['bytes']==entry['bytes'] and digest==entry['sha256'],'Dependency content identity failed: '+str(p))
    return dict(path=str(p),bytes=before['bytes'],sha256=digest,status='PASS',fresh_full_stream_hash=True,
                stat=after,unchanged_during_hash=True,purposes=entry.get('purposes',[]))


def process(root,old_parent):
    root=pathlib.Path(root).resolve();old=pathlib.Path(old_parent).resolve()
    need(root.name==TASK and old!=root,'Use the independent new task root')
    source=pathlib.Path(__file__).resolve();relative=str(source.relative_to(root))
    output=root/'analysis/FINAL_DEPENDENCY_VERIFICATION.json'
    need(not output.exists(),'Existing dependency receipt cannot be overwritten')
    started=time.time();timer=time.monotonic();runtime=[];model=[];failure=None
    result=dict(status='FAIL',task_id=TASK,stage='POST_GPU_FINAL_CURRENT_DEPENDENCY_CONTENT',
                verification_source_relative=relative,verification_source_sha256=sha(source),
                GPU_work_performed=False,torch_imported=False,tensor_deserialization=False)
    try:
        gate=json.loads((root/'GATE_STATUS_FINAL.json').read_text());ledger=json.loads((root/'gpu_resource_ledger.json').read_text())
        need(gate['task_id']==ledger['task_id']==TASK and gate.get('no_further_GPU_work') is True,'Final GPU no-further-work receipt absent')
        need(gate.get('GPU_release_verified') is True and ledger.get('final_GPU_release_verified') is True
             and all(p.get('released') is True for p in ledger['phases']),'Every GPU reservation must be released before final dependency hashing')
        need(gate['GPU_HOURS_USED']==ledger['GPU_HOURS_USED'],'Final GPU accounting mismatch')
        need(started>=ledger['ended_unix'],'Dependency verification started before coordinator final end')
        manifest=json.loads((root/'source_manifest.json').read_text())
        need(manifest['task']==TASK and manifest['frozen_before_any_new_GPU_forward'] is True,'Wrong prospective task manifest')
        need(sha(old/'hashes/artifact_sha256.txt')==manifest['inherited_manifest_sha256'],'Inherited sealed manifest changed')
        bound={line.split('  ',1)[1]:line.split('  ',1)[0] for line in (old/'hashes/artifact_sha256.txt').read_text().splitlines()}
        need(sha(old/'source_manifest.json')==bound['source_manifest.json'],'Inherited source identity manifest is not the sealed original')
        expected={}
        def add(entry,purpose):
            row=dict(path=entry['path'],bytes=entry['bytes'],sha256=entry['sha256'])
            if row['path'] in expected:
                need(all(expected[row['path']][k]==row[k] for k in ('bytes','sha256')),'Conflicting dependency identity')
                expected[row['path']]['purposes'].append(purpose)
            else:expected[row['path']]=dict(row,purposes=[purpose])
        inherited=json.loads((old/'source_manifest.json').read_text())
        for e in inherited['files']:add(e,'INHERITED_CURRENT_RUNTIME_ORIGINAL_SOURCE')
        v1=old/'inherited_cpu_v1'
        need(sha(v1/'source_manifest.json')==bound['inherited_cpu_v1/source_manifest.json'],
             'CPU V1 source identities are not the parent140-sealed original')
        need(sha(v1/'hashes/artifact_sha256.txt')==bound['inherited_cpu_v1/hashes/artifact_sha256.txt'],
             'CPU V1 70-file expected hashes are not the parent140-sealed original')
        cpu=json.loads((v1/'source_manifest.json').read_text());local_origins=[]
        for e in cpu['files']:
            if e['path'].startswith('/Users/'):
                # The sealed manifest explicitly identifies these four local
                # STEP origins and their uploaded runtime copies. Compute uses
                # the copies, not an invented /Users tree on this server.
                copy_path=v1/e['local_copy']
                add(dict(e,path=str(copy_path)),'ACTUAL_SERVER_STEP_REFERENCE_COPY_FROM_EXPLICIT_MANIFEST_LOCAL_COPY')
                local_origins.append(dict(original_local_path=e['path'],server_copy=str(copy_path),
                    expected_sha256=e['sha256'],original_server_path_claim='NOT_A_SERVER_RUNTIME_PATH; local origin rechecked by delivery driver'))
            else:add(e,'CPU_V1_CURRENT_REFERENCE_ORIGINAL_SOURCE')
        cpu_bound={line.split('  ',1)[1]:line.split('  ',1)[0] for line in (v1/'hashes/artifact_sha256.txt').read_text().splitlines()}
        for rel,digest in cpu_bound.items():
            p=v1/rel
            if p.suffix=='.py' and pathlib.PurePosixPath(rel).parts[0] in ('src','reference_sources'):
                # The hash is from the immutable70-file seal; byte count is
                # checked by streaming and recording current unchanged stat.
                add(dict(path=str(p),bytes=p.stat().st_size,sha256=digest),'ACTUAL_IMPORTED_CPU_CODEC_OR_REFERENCE_COPY')
        for entry in sorted(expected.values(),key=lambda e:e['path']):
            runtime.append(verify_file(entry))
        model_receipt=root/'qualification_cpu/MODEL_CONTENT_IDENTITY.json'
        need(sha(model_receipt)==manifest['model_identity_receipt_sha256'],'Frozen pre-run model expected identities changed')
        identities=json.loads(model_receipt.read_text())['model_weight_and_config_current_content_hashes']
        need(len(identities)==14,'Expected exactly14 model/tokenizer/config files')
        for i,e in enumerate(identities,1):
            model.append(verify_file(dict(e,purposes=['POST_RUN_CURRENT_MODEL_TOKENIZER_CONFIG'])))
            print(json.dumps(dict(final_dependency_model_file=i,total=14,path=e['path'],status='FULL_STREAM_SHA_PASS')),file=sys.stderr,flush=True)
        result.update(status='PASS',GPU_coordinator_ended_unix=ledger['ended_unix'],GPU_HOURS_USED=ledger['GPU_HOURS_USED'],
            source_manifest_sha256=sha(root/'source_manifest.json'),pre_run_model_identity_receipt_sha256=sha(model_receipt),
            final_gate_sha256=sha(root/'GATE_STATUS_FINAL.json'),final_resource_ledger_sha256=sha(root/'gpu_resource_ledger.json'),
            local_provenance_origins_mapped_to_explicit_server_copies=local_origins)
    except Exception as exc:
        failure=dict(error=repr(exc),traceback=traceback.format_exc())
    result.update(runtime_source_files=runtime,runtime_source_files_verified=len(runtime),model_weight_and_config_files=model,
        model_files_verified=len(model),model_bytes_freshly_hashed=sum(x['bytes'] for x in model),
        first_failure=failure,verification_started_unix=started,verified_unix=time.time(),CPU_wall_seconds=time.monotonic()-timer)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status=result['status'],runtime_source_files_verified=len(runtime),model_files_verified=len(model),
        model_bytes_freshly_hashed=result['model_bytes_freshly_hashed'],CPU_wall_seconds=result['CPU_wall_seconds'],output=str(output))))
    if failure:
        raise ValueError('Final dependency verification failed; preserve receipt and do not seal')
    return result


def self_test():
    import tempfile
    with tempfile.TemporaryDirectory(prefix='qwen_final_dependency_mock_') as d:
        p=pathlib.Path(d)/'SYNTHETIC_ONLY.txt';p.write_bytes(b'SYNTHETIC_SELFTEST_ONLY')
        expected=dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
        row=verify_file(expected);assert row['status']=='PASS' and row['fresh_full_stream_hash'] is True
        p.write_bytes(b'MUTATED')
        try:verify_file(expected)
        except ValueError:pass
        else:raise AssertionError('Mutated dependency was accepted')
    return dict(status='PASS',scope='Disposable tiny synthetic file only; no real dependencies, scores, GPU, or reports',
                current_stream_hash=True,bytes_stat_recorded=True,mutation_rejected=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root');p.add_argument('--old-parent');p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:print(json.dumps(self_test()))
    else:
        need(a.root and a.old_parent,'--root and --old-parent are required on the compute server')
        process(a.root,a.old_parent)
