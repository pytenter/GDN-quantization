import json
import os
import subprocess
import time
import traceback
from pathlib import Path

from common import (ARMS, CONDITIONS, EXP, PYTHON, REPO, read_json, sha256,
                    write_json)


def status(stage, **extra):
    value={'stage':stage,'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    value.update(extra)
    write_json(EXP/'execution/status.json',value,overwrite=True)


def run(name,command,gpu=None):
    log_path=EXP/'discovery/logs'/f'{name}.log'
    if log_path.exists():
        raise RuntimeError(f'write-once log exists: {log_path}')
    env=os.environ.copy(); env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    if gpu is not None:
        env['CUDA_VISIBLE_DEVICES']=str(gpu)
    with log_path.open('x',encoding='utf-8',buffering=1) as log:
        proc=subprocess.Popen(command,cwd=REPO,env=env,stdout=log,
                              stderr=subprocess.STDOUT,text=True)
        rc=proc.wait()
    write_json(EXP/'execution'/f'{name}.json',{
        'command':[str(x) for x in command],'physical_gpu':gpu,'pid':proc.pid,
        'exitcode':rc,'log_sha256':sha256(log_path)})
    if rc!=0:
        raise RuntimeError(f'{name} failed rc={rc}')


def verify_freeze():
    freeze=read_json(EXP/'PREMEASUREMENT_FREEZE_MANIFEST.json')
    for row in freeze['frozen_files']:
        if sha256(EXP/row['path'])!=row['sha256']:
            raise RuntimeError(f'premeasurement frozen file changed: {row["path"]}')


def premeasurement_commit():
    freeze=read_json(EXP/'PREMEASUREMENT_FREEZE_MANIFEST.json')
    relroot=EXP.relative_to(REPO)
    paths=[str(relroot/row['path']) for row in freeze['frozen_files']]
    paths += [str(relroot/'PREMEASUREMENT_FREEZE_MANIFEST.json'),
              str(relroot/'PREMEASUREMENT_SHA256SUMS')]
    subprocess.run(['git','add','-f','--']+paths,cwd=REPO,check=True)
    subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@local',
                    'commit','-m','Freeze scale feedback direction protocol'],
                   cwd=REPO,check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    auth=read_json(EXP/'execution_authorization.json')
    auth['PREMEASUREMENT_COMMIT_SHA']=commit
    auth['FORMAL_MEASUREMENT_AUTHORIZED']='YES_PENDING_POSTCOMMIT_PARITY'
    write_json(EXP/'execution_authorization.json',auth,overwrite=True)
    write_json(EXP/'PREMEASUREMENT_COMMIT.json',{
        'PREMEASUREMENT_COMMIT_SHA':commit,
        'frozen_file_set_sha256':freeze['frozen_file_set_sha256'],
        'formal_measurement_count_before_commit':0})
    return commit


def seal_release(doc):
    results=[]
    for condition in CONDITIONS:
        for arm in ARMS:
            path=EXP/f'discovery/unit_results/doc_{doc:02d}_{condition}_{arm}.json'
            value=read_json(path)
            if value['status']!='PASS' or not value['paired_clean_scale_schedule']['byte_identity']:
                raise RuntimeError(f'cannot seal result: {path}')
            results.append({'condition':condition,'arm':arm,'path':str(path),
                            'sha256':sha256(path),'bytes':path.stat().st_size})
    seal=EXP/f'discovery/doc_{doc:02d}/result_seal.json'
    write_json(seal,{'RESULT_SEAL_GATE':'PASS','document_index':doc,
                     'unit_count':8,'results':results,'all_result_hashes_verified':True})
    drive=EXP/f'discovery/doc_{doc:02d}/drive'
    manifest=read_json(drive/'drive_manifest.json')
    candidates=[]
    for row in manifest['shards']:
        candidates.append((drive/row['relative_path'],row['file_sha256']))
    for row in manifest['fp_reference_checkpoints']:
        candidates.append((drive/row['relative_path'],row['file_sha256']))
    released=[]; seen=set()
    for path,expected in candidates:
        path=Path(path)
        if str(path) in seen: continue
        seen.add(str(path))
        if sha256(path)!=expected:
            raise RuntimeError(f'release refused for hash mismatch: {path}')
        released.append({'path':str(path),'sha256':expected,'bytes':path.stat().st_size})
    anchor_released=[]
    for condition in CONDITIONS:
        path=EXP/f'qualification/doc_{doc:02d}/anchors/{condition}.pt'
        meta=read_json(path.with_suffix('.json'))
        if sha256(path)!=meta['file_sha256']:
            raise RuntimeError('anchor release refused for hash mismatch')
        anchor_released.append({'path':str(path),'sha256':meta['file_sha256'],
                                'bytes':path.stat().st_size})
    for row in released+anchor_released:
        Path(row['path']).unlink()
    write_json(drive/'drive_release_record.json',{
        'document_index':doc,'drive_manifest_sha256':sha256(drive/'drive_manifest.json'),
        'drive_identity_sha256':sha256(EXP/f'discovery/doc_{doc:02d}/drive_identity.json'),
        'result_seal_sha256':sha256(seal),'all_8_results_complete':True,
        'all_result_hashes_verified':True,'released_paths':released,
        'released_anchor_paths':anchor_released,
        'released_bytes':sum(x['bytes'] for x in released+anchor_released),
        'release_complete':True,'parent_artifacts_deleted':0})


def main():
    qualification_hash_recovery = (
        not (EXP/'qualification/qualification_report.json').exists()
        and len(list((EXP/'qualification').glob('schedule_identity_*.json'))) == 4
        and len(list((EXP/'qualification').glob('smoke_run*_H.json'))) == 2
        and (EXP/'FAIL_AND_STOP_RECOVERY_002.json').exists())
    reducer_recovery = ((EXP/'execution/qualification_doc01_run1.json').exists()
                        and not any((EXP/'qualification').glob('schedule_identity_*.json')))
    if qualification_hash_recovery:
        status('QUALIFICATION_IDENTITY_HASH_RECOVERY_003')
    elif reducer_recovery:
        identity=read_json(EXP/'discovery/doc_01/drive_identity.json')
        if identity['FROZEN_DRIVE_IDENTITY_GATE']!='PASS':
            raise RuntimeError('reducer recovery refused: drive identity not PASS')
        for condition in CONDITIONS:
            if not (EXP/f'qualification/doc_01/anchors/{condition}.json').exists():
                raise RuntimeError('reducer recovery refused: anchor evidence missing')
        status('QUALIFICATION_REDUCER_RECOVERY_002')
        run('qualification_run1_recovery002',[
            str(PYTHON),str(EXP/'scripts/launch_group.py'),'--mode','qualification',
            '--document','1','--run','101'])
        status('QUALIFICATION_DETERMINISM_RECOVERY_002')
        run('smoke_run2_recovery002',[
            str(PYTHON),str(EXP/'scripts/launch_group.py'),'--mode','smoke',
            '--document','1','--run','102'])
    else:
        status('DOC1_CAPTURE')
        capture_name = ('doc01_capture_recovery001'
                        if (EXP/'execution/doc01_capture.json').exists()
                        else 'doc01_capture')
        run(capture_name,[str(PYTHON),str(EXP/'scripts/capture_driver.py'),'--document','1'],gpu=0)
        run('doc01_verify',[str(PYTHON),str(EXP/'scripts/verify_drive.py'),'--document','1'])
        status('DOC1_ANCHORS')
        run('doc01_anchors',[str(PYTHON),str(EXP/'scripts/launch_group.py'),
                             '--mode','build_anchors','--document','1'])
        status('QUALIFICATION_FULL_SCHEDULE_AND_ACTIVATION')
        run('qualification_run1',[str(PYTHON),str(EXP/'scripts/launch_group.py'),
                                  '--mode','qualification','--document','1','--run','1'])
        status('QUALIFICATION_DETERMINISM_RUN2')
        run('smoke_run2',[str(PYTHON),str(EXP/'scripts/launch_group.py'),
                          '--mode','smoke','--document','1','--run','2'])
    qualify_name = ('qualify_recovery003' if qualification_hash_recovery else 'qualify')
    run(qualify_name,[str(PYTHON),str(EXP/'scripts/qualify.py')])
    status('PREMEASUREMENT_COMMIT')
    commit=premeasurement_commit()
    status('POSTCOMMIT_MINIMAL_PARITY',premeasurement_commit=commit)
    run('postcommit_parity',[str(PYTHON),str(EXP/'scripts/postcommit_parity.py')],gpu=0)
    verify_freeze()
    for doc in range(1,5):
        if doc>1:
            status(f'DOC{doc}_CAPTURE')
            run(f'doc{doc:02d}_capture',[str(PYTHON),str(EXP/'scripts/capture_driver.py'),
                                         '--document',str(doc)],gpu=0)
            run(f'doc{doc:02d}_verify',[str(PYTHON),str(EXP/'scripts/verify_drive.py'),
                                        '--document',str(doc)])
            run(f'doc{doc:02d}_anchors',[str(PYTHON),str(EXP/'scripts/launch_group.py'),
                                         '--mode','build_anchors','--document',str(doc)])
        verify_freeze()
        status(f'DOC{doc}_FORMAL',formal_units_complete=(doc-1)*8)
        run(f'doc{doc:02d}_formal',[str(PYTHON),str(EXP/'scripts/launch_group.py'),
                                    '--mode','formal','--document',str(doc)])
        status(f'DOC{doc}_SEAL_RELEASE',formal_units_complete=doc*8)
        seal_release(doc)
    status('ANALYSIS',formal_units_complete=32)
    run('analysis',[str(PYTHON),str(EXP/'scripts/analyze.py')])
    status('COMPLETE',formal_units_complete=32,final_action='STOP')
    write_json(EXP/'controller_exit.json',{'exitcode':0,'DISCOVERY_COMPLETE':'32/32',
                                           'FINAL_ACTION':'STOP'})
    run('finalize',[str(PYTHON),str(EXP/'scripts/finalize.py')])
    print('SCALE_FEEDBACK_DIRECTION_COMPLETE_STOP',flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        candidates=[EXP/'FAIL_AND_STOP_CURRENT.json']+[EXP/f'FAIL_AND_STOP_RECOVERY_{i:03d}.json' for i in range(1,100)]
        path=next(x for x in candidates if not x.exists())
        if not path.exists():
            write_json(path,{'FINAL_ACTION':'FAIL_AND_STOP',
                             'exception':type(exc).__name__,'message':str(exc),
                             'traceback':traceback.format_exc(),'retry_authorized':False})
        status('FAIL_AND_STOP',message=str(exc))
        raise
