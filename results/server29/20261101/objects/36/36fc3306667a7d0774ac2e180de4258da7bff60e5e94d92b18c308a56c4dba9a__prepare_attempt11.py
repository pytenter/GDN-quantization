"""Preparation audit only. Never load Torch or mutate historical inputs."""
from pathlib import Path
import datetime, hashlib, json, shutil, subprocess, tarfile
R=Path('/data/zypan/worktrees/ling-long-horizon-observability-bridge-v3')
C=R/'provenance_capsule_v3'; A=C/'attempt_11_static_ast_qualification'
HEAD='ff3fb899f297c332a21cfc7b834b00780d1d1192'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def emit(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,indent=2,sort_keys=True);f.write('\n')
def cmd(*a):return subprocess.check_output(a,cwd=R,text=True).strip()
def main():
    assert cmd('git','rev-parse','HEAD')==HEAD
    assert not cmd('git','status','--porcelain=v1')
    assert not A.exists(),'ATTEMPT11_ALREADY_EXISTS'
    A.mkdir()
    try:
        launch=read(C/'attempts/gpu6_file_recovery_attempt05/FAILED_LAUNCH_IDENTITY.json')
        assert len(launch['files'])==187
        for r in launch['files']:assert sha(C/r['path'])==r['sha256'],r['path']
        prior=C/'attempt_06_generated_code'
        assert sha(prior/'STOP.json')=='b4e34043d1a97c24f0a9f27ffa6b4c27de042e16a018bbdf975350d95286907c'
        assert sha(prior/'PREFLIGHT.log')=='ff201b2af7e47799dd69f0804469193cbdbdabba5c8251a07b1d0c4f524ba264'
        assert sha(prior/'FINAL_ATTEMPT06_REPORT.md')=='fae652a9231d17c09d9e9e1bd471053d1bfbcf5aae9a036f9b0b04814af59687'
        parent=C/'attempt_10_ast_audit_association'
        prior_protected=read(parent/'PROTECTED_10730_IDENTITY.json')['files']
        failure_inventory=parent/'FAILURE_ARTIFACT_INVENTORY.json'
        failed_files=read(failure_inventory)['files']
        expected={row['path']:row['sha256'] for row in prior_protected}
        for row in failed_files:
            assert row['path'] not in expected,('DUPLICATE_PROTECTED_PATH',row['path'])
            assert len(row['sha256'])==64
            expected[row['path']]=row['sha256']
        inv_name=str(failure_inventory.relative_to(R))
        committed=subprocess.check_output(['git','show',HEAD+':'+inv_name],cwd=R)
        assert failure_inventory.read_bytes()==committed,'FROZEN_FAILURE_INVENTORY_BYTE_MISMATCH'
        expected[inv_name]=hashlib.sha256(committed).hexdigest()
        names=cmd('git','ls-files','provenance_capsule_v3').splitlines()
        assert set(names)==set(expected),('PROTECTED_CLOSURE_CHANGED',len(names),len(expected))
        tracked=[]
        for name in names:
            actual=sha(R/name)
            assert actual==expected[name],('PROTECTED_SOURCE_MISMATCH',name)
            tracked.append(dict(path=name,sha256=actual))
        emit(A/'PROTECTED_PREVIOUS_IDENTITY.json',dict(commit=HEAD,files=tracked,
            identity_source='frozen previous protected inventory + frozen failure inventory + exact committed inventory blob',
            source_inventories=[dict(path=str(parent/'PROTECTED_10730_IDENTITY.json'),sha256=sha(parent/'PROTECTED_10730_IDENTITY.json')),
                               dict(path=str(failure_inventory),sha256=sha(failure_inventory))],
            ATTEMPT_06='FAIL_PRESERVED',ATTEMPT_07='FAIL_PRESERVED',ATTEMPT_08='FAIL_PRESERVED',
            ATTEMPT_09='FAIL_PRESERVED',ATTEMPT_10='FAIL_PRESERVED'))
        for old_attempt in ['attempt_07_generated_identity','attempt_08_fixture_closure','attempt_09_output_directory','attempt_10_ast_audit_association']:
            stop_name=str((C/old_attempt/'STOP.json').relative_to(R))
            assert sha(R/stop_name)==expected[stop_name],('PREVIOUS_STOP_MISMATCH',stop_name)
        old=C/'attempts/gpu6_file_recovery_attempt05'
        assert sha(old/'STOP.json')=='886972356efaa75e9fd0fa0fe39604e3d646f34b2348e85b34069c85cb3e667e'
        assert sha(old/'VERIFY_FAILURE.log')=='f330bb9fa287ce678e3fe047a8bfaac0a24bdfa19e36796b541d7d69b0c8c49f'
        manifests={n:sha(C/n) for n in ['historical_source_snapshot_manifest.json','scientific_asset_manifest.json','runtime_environment_manifest.json','known_fixture_manifest.json','frozen_protocol/document_manifest.json','frozen_protocol/rotation_manifest.json']}
        groups={}
        for name,rows,pathkey,hashkey in [
            ('snapshot',read(C/'historical_source_snapshot_manifest.json')['files'],'snapshot_relative_path','snapshot_sha256'),
            ('scientific',read(C/'scientific_asset_manifest.json')['files'],'path','sha256'),
            ('runtime',read(C/'runtime_environment_manifest.json')['environment_files'],'path','sha256'),
            ('historical_inputs',read(C/'target_historical_input_identity.json')['files'],'path','sha256'),
            ('historical_preservation',read(C/'target_historical_preservation_before_qualification.json')['files'],'path','sha256')]:
            count=0
            for r in rows:
                p=C/r[pathkey] if pathkey=='snapshot_relative_path' else Path(r[pathkey])
                assert sha(p)==r[hashkey],('IDENTITY_MISMATCH',name,str(p))
                count+=1
                if count%100==0:print('IDENTITY_PROGRESS',name,count,len(rows),flush=True)
            groups[name]={'count':count,'gate':'PASS'}
        gpu=cmd('nvidia-smi','-i','6','--query-gpu=index,uuid,name,memory.used,memory.free,driver_version','--format=csv,noheader,nounits')
        apps=cmd('nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader,nounits')
        uuid='GPU-de53f243-b271-521c-d875-d96611ed8574'
        assert gpu.split(',')[1].strip()==uuid
        assert not any(uuid in l for l in apps.splitlines()),'GPU6_OCCUPIED'
        assert shutil.disk_usage(C).free>22*1024**3
        emit(A/'GATE0_REVERIFICATION.json',dict(OLD_EVIDENCE_INTEGRITY_GATE='PASS',TRACKED_LAUNCH_IDENTITY_GATE='PASS',old_launch_count=187,old_commit=HEAD,ATTEMPT_05='FAIL_PRESERVED',ATTEMPT_05_GENERATED_BYTES='UNAVAILABLE',manifest_sha256=manifests,reverified_groups=groups,gpu_query=gpu,available_bytes=shutil.disk_usage(C).free,FORMAL_MEASUREMENTS=0,GPU6_FORMAL_AUTHORIZED='NO',SUPERVISOR_CLEANUP_LEDGER_REMAINING=1,ACTUAL_TASK_PROCESS_COUNT=0,GPU6_COMPUTE_PROCESS_COUNT=0,timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        print('ATTEMPT11_GATE0_PASS',flush=True)
    except BaseException as e:
        emit(A/'STOP.json',dict(reason=repr(e),FINAL_ACTION='FAIL_AND_STOP',FORMAL_MEASUREMENTS=0,GPU6_FORMAL_AUTHORIZED='NO'))
        raise
if __name__=='__main__':main()


