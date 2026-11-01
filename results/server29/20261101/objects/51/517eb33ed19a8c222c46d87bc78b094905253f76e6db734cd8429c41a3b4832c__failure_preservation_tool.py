"""Append-only unknown native ingress failure preservation; no rerun or admission."""
from pathlib import Path
import datetime, hashlib, json, subprocess, tarfile

R=Path('/data/zypan/worktrees/ling-long-horizon-observability-bridge-v3')
C=R/'provenance_capsule_v3'; A=C/'attempt_12_alias_namespace_relocation'
AUTH=Path(str(R)+'-authority')
LIB=Path('/data/zypan/.triton/cache/XM4VEAW44Q3HMHE24NDH4IUOZL2OXVS5CEP7VSIDH2GUL4U6JJDA/__triton_launcher.so')
PRODUCER=Path('/data/zypan/envs/ling-kda/lib/python3.11/site-packages/triton/backends/nvidia/driver.py')
UUID='GPU-de53f243-b271-521c-d875-d96611ed8574'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    return json.loads(Path(p).read_text())
def emit(p,value):
    with Path(p).open('x') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def snapshot(p,out):
    before=sha(p)
    with out.open('xb') as f:f.write(p.read_bytes())
    assert sha(p)==before==sha(out)
    return dict(original_path=str(p),evidence_copy=str(out),sha256=before,
                bytes=p.stat().st_size,role='failure evidence only; not admitted runtime input')

def main():
    stop=read(A/'STOP.json')
    assert 'UNKNOWN_IMPORTED_PYTHON_MODULE: __triton_launcher: '+str(LIB) in stop['reason']
    assert 'checked_extension' in stop['traceback'] and 'compile_module_from_src' in stop['traceback']
    clean=read(A/'CLEAN_ENGINEERING_LAUNCH.json')
    launch=read(A/'PREFLIGHT_LAUNCH.json')
    assert sha(A/'CLEAN_ENGINEERING_LAUNCH.json')==launch['launch_inventory_sha256']
    assert clean['commit']==launch['input_commit']
    for row in clean['files']:
        assert sha(C/row['path'])==row['sha256'],('PROTECTED_SOURCE_CHANGED',row['path'])
    for pid in [launch['pid'],read(A/'GENERATED_QUALIFICATION_SUPERVISOR.json')['pid']]:
        assert not Path('/proc',str(pid)).exists(),'FAILED_TASK_STILL_RUNNING'
    gpu_table=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
                                      '--format=csv,noheader,nounits'],text=True)
    gpu_rows=[x for x in gpu_table.splitlines() if UUID in x]
    assert not gpu_rows,'GPU6_COMPUTE_PROCESS_STILL_RUNNING'
    assert not list((A/'runs').glob('*/PREFLIGHT_COMPLETE.json'))
    assert not list((A/'runs').glob('*/qualification/smoke/repeat0/result.json'))
    assert not (A/'generated_executable_manifest.json').exists()
    manifest_names=['runtime_environment_manifest.json','shared_library_manifest.json',
                    'cached_environment_module_manifest.json','executable_runtime_manifest.json',
                    'attempts/file_runtime_attempt04/FILE_RUNTIME_DEPENDENCIES.json']
    membership=[]
    for name in manifest_names:
        raw=(C/name).read_text()
        membership.append(dict(manifest=str(C/name),sha256=sha(C/name),
                               binary_path_present=str(LIB) in raw,binary_hash_present=sha(LIB) in raw))
    assert all(not row['binary_path_present'] and not row['binary_hash_present'] for row in membership)
    frozen_producer=[row for row in read(C/'runtime_environment_manifest.json')['environment_files']
                     if row['path']==str(PRODUCER)]
    assert len(frozen_producer)==1 and sha(PRODUCER)==frozen_producer[0]['sha256']
    evidence=A/'native_ingress_failure_evidence';evidence.mkdir()
    native=snapshot(LIB,evidence/'BLOCKED_NONADMITTED_triton_launcher.so')
    producer=snapshot(PRODUCER,evidence/'frozen_producer_driver.py')
    inputs=A/'failed_engineering_inputs';inputs.mkdir()
    input_copies=[]
    for p in sorted(Path('/data/zypan/attempt12_inputs').iterdir()):
        if p.is_file():input_copies.append(snapshot(p,inputs/p.name))
    supervisor=snapshot(Path('/data/zypan/attempt12-generated-qualification-supervisor.log'),
                        A/'PRESERVED_SUPERVISOR_FAILURE.log')
    captured=read(A/'runs/preflight_01/generated_capture/GENERATED_CODE_RECEIPT.json')
    labels=dict(ATTEMPT_12='FAIL_PRESERVED',FAILURE_CLASS='UNKNOWN_NONADMITTED_NATIVE_CACHE_BINARY',
                PROTECTED_SOURCE_ACTUAL_BYTE_MISMATCH='NO',FORMAL_MEASUREMENTS=0,
                DISCOVERY_PRIMARY_COMPLETE='0/64',DISCOVERY_SECONDARY_COMPLETE='0/64',
                GPU6_FORMAL_AUTHORIZED='NO',SCIENTIFIC_PROTOCOL_CHANGED='NO',MODEL_SEMANTICS_CHANGED='NO',
                QDQ_SEMANTICS_CHANGED='NO',KDA_RECURRENCE_CHANGED='NO',BASIS_SEMANTICS_CHANGED='NO',
                DOCUMENT_SET_CHANGED='NO',ROTATION_SET_CHANGED='NO',HORIZON_CHANGED='NO',METRIC_CHANGED='NO',
                TARGET_CHANGED='NO',TOLERANCE_RELAXED='NO',ADVANCEMENT_RULE_CHANGED='NO',
                TESTS='30_NEW_PLUS_14_OLD_CONTROL_PLUS_12_OLD_ARCHITECTURE_PASS',
                HISTORICAL_NUMERICAL_PARITY_SUITE='NOT_COMPLETED',NATIVE_BINARY_IDENTITY_GATE='FAIL',
                NATIVE_BINARY_LOADER_ADMISSION='DENIED_BEFORE_EXTENSION_CREATE_MODULE',
                NATIVE_BINARY_SOURCE_TOOLCHAIN_ATTRIBUTION='NOT_ESTABLISHED',
                AUTO_WHITELIST='NO',AUTO_RETRY='NO',REQUEST_EXPLICIT_USER_AUTHORITY='YES',
                SUPERVISOR_CLEANUP_LEDGER_REMAINING=1,ACTUAL_TASK_PROCESS_COUNT=0,GPU6_COMPUTE_PROCESS_COUNT=0,
                L9_TRAINING_EXECUTED='NO',NEW_ROTATION_TRAINING_EXECUTED='NO',AIME_EXECUTED='NO',
                FREE_GENERATION_EXECUTED='NO',FINAL_ACTION='FAIL_AND_STOP')
    emit(A/'FAILURE_AUDIT.json',dict(labels=labels,timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
         parent_commit=clean['commit'],STOP_sha256=sha(A/'STOP.json'),
         protected_count=len(clean['files']),protected_byte_identity='PASS',
         generation_count=captured['generation_count'],execution_count=len(captured['execution_events']),
         blocked_library=native,known_frozen_producer=producer,producer_frozen_row=frozen_producer[0],
         frozen_manifest_membership=membership,engineering_input_snapshots=input_copies,
         supervisor_failure=supervisor,actual_gpu_process_table=gpu_rows,
         attribution_boundary='Frozen driver Python source proves the requesting producer, not the historical cached binary build. This cache-hit did not produce a compiler command/input/header/link closure or a source-to-binary identity proof. No retrospective cached-binary whitelist is allowed.',
         next_authority_needed='Explicit prospective native-generated-code qualification: frozen producer/input/toolchain/header/link identities and before-load binary capture; existing unknown cached binary remains rejected. Scientific semantics and tolerances stay unchanged.'))
    with (A/'FINAL_ATTEMPT12_REPORT.md').open('x') as f:
        f.write('# Attempt 12 — unknown native binary FAIL preserved\n\n```ini\n'+
                '\n'.join(k+' = '+str(v) for k,v in labels.items())+'\n```\n\n'+
                'T_IDENTITY_06 closure and all synthetic/old regressions passed. Actual smoke reached the model FLA/Triton convolution launcher. The extension loader denied a pre-existing cached launcher binary before its create_module call. The requesting Triton producer Python source matches the existing frozen runtime manifest, but the cached binary has no frozen identity or recorded source-to-binary/toolchain provenance. Its name/cache path alone does not establish admission. No native whitelist, scientific change or numerical tolerance relaxation was performed. Smoke and parity did not complete: this is not a confirmed numerical parity FAIL, numerical nondeterminism or scientific conclusion. All previous protected files and historical STOP receipts remain byte-identical. The blocked .so is copied as nonexecuted forensic failure evidence only, never as an accepted runtime asset.\n')
    tool=Path(__file__).resolve()
    snapshot(tool,A/'failure_preservation_tool.py')
    names=[str(p.relative_to(R)) for p in sorted(A.rglob('*')) if p.is_file()]
    emit(A/'FAILURE_ARTIFACT_INVENTORY.json',dict(files=[dict(path=n,sha256=sha(R/n),bytes=(R/n).stat().st_size) for n in names]))
    names.append(str((A/'FAILURE_ARTIFACT_INVENTORY.json').relative_to(R)))
    with tarfile.open(AUTH/'attempt12-failure-evidence.tar.gz','x:gz') as tar:
        for n in names:tar.add(R/n,arcname=n,recursive=False)
    print('ATTEMPT12_FAILURE_ARCHIVE_READY',len(names),sha(A/'STOP.json'),flush=True)

if __name__=='__main__':main()
