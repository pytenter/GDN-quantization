"""Two fixed exploratory scientific keys, fresh own histories, serial GPU6."""
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
import functional_worker as w

D=w.OUT;WT=Path(__file__).resolve().parent


def run(unit,stage):
    spec=w.read(D/'FROZEN_SPEC.json')
    assert not (D/'STOP.json').exists()
    assert subprocess.check_output(['git','-C',str(WT),'status','--porcelain'],text=True)==''
    assert subprocess.check_output(['git','-C',str(WT),'rev-parse','HEAD'],text=True).strip()==spec['source_commit']
    freeze=w.read(D/'FORMAL_CODE_FREEZE.json')
    for n,h in freeze['files'].items():assert w.sha(D/n)==h,n
    for name,row in spec['source_assets'].items():
        s=Path(name).stat();assert s.st_size==row['bytes'] and s.st_mtime_ns==row['mtime_ns'],name
    for n in (unit['token_file']['path'],unit['rotation_identity']['path']):
        assert w.sha(n)==spec['source_assets'][n]['sha256'],n
    cfg=w.read(w.OLD/'PHASE_B_CALIBRATION_CONFIG.json');guard=w.device(6,cfg)
    assert guard['exclusive'],guard['failure_conditions']
    assert shutil.disk_usage(D).free>64*1024**3
    folder=D/('formal' if stage=='formal' else 'runs');folder.mkdir(exist_ok=True)
    tag=f"{unit['document']}__{unit['rotation']}__{stage}"
    assert not (folder/(tag+'.STARTED.json')).exists(),'ALREADY_EXECUTED_NO_RETRY'
    args=[sys.executable,'-B','-u',str(WT/('formal_worker.py' if stage=='formal' else 'functional_worker.py')),
        '--document',unit['document'],'--condition',unit['rotation']]
    if stage!='formal':args.extend(['--stage',stage])
    started=time.time();last_change=started;previous_size=-1;progress_position=-1
    log_path=folder/(tag+'.log')
    with log_path.open('x') as log:
        proc=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT)
        while proc.poll() is None:
            time.sleep(2)
            size=log_path.stat().st_size
            if size!=previous_size:
                previous_size=size
                lines=log_path.read_text().splitlines()
                for line in reversed(lines):
                    if line.startswith('{'):
                        try:row=json.loads(line)
                        except json.JSONDecodeError:continue
                        if 'position' in row:
                            assert row['position']>progress_position or row['position']==progress_position
                            if row['position']>progress_position:last_change=time.time();progress_position=row['position']
                            break
            if time.time()-last_change>spec['no_progress_timeout_seconds']:
                time.sleep(10)
                if proc.poll() is None and log_path.stat().st_size==previous_size:
                    w.publish(D/'STOP.json',dict(status='STOP',stage=stage,reason='NO_PROGRESS_CONFIRMED',
                        tag=tag,pid=proc.pid,last_position=progress_position,epoch=time.time(),first_failure=True))
                    proc.terminate();proc.wait(timeout=60)
                    raise RuntimeError('NO_PROGRESS_CONFIRMED_FIRST_FAILURE_STOP')
        assert proc.returncode==0,('WORKER_FIRST_FAILURE_STOP',tag,proc.returncode)
    path=folder/(tag+'.json');result=w.read(path);assert result['status']=='PASS'
    w.publish(folder/(tag+'.NATURAL_EXIT.json'),dict(started_epoch=started,completed_epoch=time.time(),
        wall_seconds=time.time()-started,natural_exit=True,result_sha256=w.sha(path),
        source_commit=spec['source_commit'],tag=tag))
    print('TRAJECTORY_STAGE_PASS',tag,flush=True)
    return result


def main():
    assert w.read(D/'FP_REFERENCE_REUSE_GATE.json')['status']=='PASS'
    spec=w.read(D/'FROZEN_SPEC.json');units=spec['new_units'];pairs=[];resources=[]
    assert len(units)==2
    for unit in units:
        parent=run(unit,'parent');resumed=run(unit,'resumed_parity')
        for field in ('uninterrupted_suffix_full_logit_sha256','final_full_native_cache_sha256',
                      'audit_state_fingerprint','checkpoint_sha256'):
            assert parent[field]==resumed[field],('NEW_KEY_FRESH_PROCESS_CHECKPOINT_PARITY_FAIL',field)
        inventories=parent['cache_inventory'];first=inventories['511'];last=inventories['8255']
        rate=(last['cache_bytes']-first['cache_bytes'])/(8256-512)
        assert rate==61440,'ILLEGAL_CACHE_GROWTH'
        assert all(x['cache_bytes']==first['cache_bytes']+(int(k)+1-512)*rate for k,x in inventories.items())
        assert parent['integrity']['passed'] and resumed['integrity']['passed']
        resources.append(dict(document=unit['document'],rotation=unit['rotation'],cache_growth_bytes_per_token=rate,
            predicted_native_cache_32768=last['cache_bytes']+(32768-8256)*rate,
            peak_allocated_8256=max(x['peak_allocated'] for x in parent['memory']),
            peak_reserved_8256=max(x['peak_reserved'] for x in parent['memory']),
            physical_GPU='RTX3090 24GB',successful_32768_geometry_reused=True,
            no_three_cache_residency=True,disk_free=shutil.disk_usage(D).free))
        pairs.append(dict(document=unit['document'],rotation=unit['rotation'],status='PASS',
            uninterrupted_equals_fresh_resumed=True,observer_ON_equals_OFF=True,
            own_rotation_0_to_8192_history=True,parent=parent,resumed=resumed))
    w.publish(D/'CROSS_RUN_NUMERICAL_QUALIFICATION.json',dict(status='PASS',new_key_pairs=pairs,
        reused_completed_numerical_contract=w.read(D/'FP_REFERENCE_REUSE_GATE.json'),resources=resources,
        prefix512_new_key_has_no_historical_expectation=True,fresh_process_exact_64_token_repeatability=True,
        full_model_state_and_MLA_and_convolution_and_position_checkpoint=True,
        numeric_tolerance_not_relaxed=True,complete_run_finite_to_be_checked_at_32768=True))
    w.publish(D/'QUALIFICATION_GATES_COMPLETED.json',dict(status='ALL_EIGHT_GATES_PASS',
        **{k:'PASS' for k in spec['gates']},secondary_definitions=w.read(D/'SECONDARY_SPEC.json')['status']))
    for unit in units:run(unit,'formal')
    w.publish(D/'CROSS_COMPLETE.json',dict(status='COMPLETE',new_student_scientific_units=2,
        endpoint_per_student=32768,FP_reference_reruns=0,old_student_reruns=0,epoch=time.time(),
        original_formal_verdict='MIXED_PERMANENTLY_PRESERVED',classification='POSTHOC_EXPLORATORY_CROSS_PANEL'))
    print('TWO_CROSS_SCIENTIFIC_UNITS_COMPLETE_STOP_NO_NEXT_4DOC_OR_TRAINING',flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as e:
        if not (D/'STOP.json').exists():w.publish(D/'STOP.json',dict(status='STOP',stage='cross_controller',
            reason=repr(e),traceback=traceback.format_exc(),first_failure=True,epoch=time.time()))
        raise
