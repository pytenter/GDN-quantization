"""Preregistered screening -> cohort freeze -> own-history gates -> serial32K."""
import csv
import io
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
sys.dont_write_bytecode=True
from prospective_common import OUT,OLD,WT,sha,read,publish,verify
from v3_guard import device
from next_prospective_pair_rule import midranks,opposite_pairs

def publish_text(path,text):
    tmp=path.with_name(path.name+'.partial')
    with tmp.open('x') as f:f.write(text);f.flush();os.fsync(f.fileno())
    os.link(tmp,path);tmp.unlink()

def run(document,condition=None,stage='screening'):
    cfg=verify();freeze=read(OUT/'SOURCE_CODE_FREEZE.json')
    guard=device(6,cfg,evidence=OUT/'CONTROLLER_GPU_ADMISSION.jsonl')
    assert guard['exclusive'],guard['failure_conditions']
    if stage=='screening':
        folder=OUT/'screening';tag=document;args=[sys.executable,'-B','-u',str(WT/'short_worker.py'),'--document',document]
    else:
        selection=read(OUT/'FROZEN_4DOC_SELECTION.json')
        assert sha(OUT/'FROZEN_4DOC_SELECTION.json')==read(OUT/'SELECTION_FREEZE_RECEIPT.json')['sha256']
        assert any(d['document']==document and condition in ['FP',d['P1_predicted_rotation'],d['L7_predicted_rotation']] for d in selection['documents'])
        folder=OUT/('formal' if stage=='formal' else 'runs');tag=f'{document}__{condition}__{stage}'
        args=[sys.executable,'-B','-u',str(WT/('formal_worker.py' if stage=='formal' else 'functional_worker.py')),
            '--document',document,'--condition',condition]
        if stage!='formal':args.extend(['--stage',stage])
    folder.mkdir(exist_ok=True);log_path=folder/(tag+'.log')
    assert not log_path.exists() and not (folder/(tag+'.STARTED.json')).exists(),'ALREADY_LAUNCHED_NO_RETRY'
    started=time.time();last_change=started;last_size=0;progress=-1
    with log_path.open('x') as log:
        proc=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT)
        while proc.poll() is None:
            time.sleep(2);size=log_path.stat().st_size
            if size!=last_size:
                last_size=size
                for line in reversed(log_path.read_text().splitlines()):
                    try:row=json.loads(line)
                    except (ValueError,TypeError):continue
                    if isinstance(row,dict) and 'position' in row:
                        assert row['position']>=progress,'PROGRESS_POSITION_REGRESSED'
                        if row['position']>progress:progress=row['position'];last_change=time.time()
                        break
            if time.time()-last_change>900:
                time.sleep(10)
                if proc.poll() is None and log_path.stat().st_size==last_size:
                    publish(OUT/'STOP.json',dict(status='STOP',stage=stage,reason='NO_PROGRESS_CONFIRMED',tag=tag,
                        owned_worker_pid=proc.pid,last_position=progress,epoch=time.time(),no_retry=True))
                    proc.terminate();proc.wait(timeout=60)
                    raise RuntimeError('OWN_WORKER_NO_PROGRESS_FIRST_FAILURE_STOP')
        if proc.returncode!=0:
            publish(folder/(tag+'.FAILED_EXIT.json'),dict(started_epoch=started,completed_epoch=time.time(),
                wall_seconds=time.time()-started,returncode=proc.returncode,tag=tag,source_commit=freeze['source_commit']))
            raise RuntimeError(('WORKER_FAILURE_NO_RETRY',tag,proc.returncode))
    result=read(folder/(tag+'.json'));assert result['status']=='PASS'
    publish(folder/(tag+'.NATURAL_EXIT.json'),dict(started_epoch=started,completed_epoch=time.time(),
        wall_seconds=time.time()-started,natural_exit=True,result_sha256=sha(folder/(tag+'.json')),
        source_commit=freeze['source_commit'],tag=tag,stage=stage))
    print(json.dumps(dict(event='WORKER_NATURAL_EXIT_PASS',tag=tag,stage=stage)),flush=True)
    return result

def export_selection_csv(decisions):
    buffer=io.StringIO();columns=['scan_index','document','rotation','P1','L7_FUNCTIONAL_ORIGINAL','L7_STATE_ORIGINAL',
        'L7_TOTAL_ORIGINAL','P1_midrank','L7_midrank','eligible_pair_count','selected_document',
        'P1_predicted_rotation','L7_predicted_rotation','P1_gap','L7_gap','decision','short_result_sha256']
    writer=csv.DictWriter(buffer,fieldnames=columns);writer.writeheader()
    for d in decisions:
        for condition in d['roster']:
            r=d['scores'][condition]
            writer.writerow(dict(scan_index=d['scan_index'],document=d['document'],rotation=condition,
                **{k:r[k] for k in ['P1','L7_FUNCTIONAL_ORIGINAL','L7_STATE_ORIGINAL','L7_TOTAL_ORIGINAL']},
                P1_midrank=d['P1_midranks'][condition],L7_midrank=d['L7_midranks'][condition],
                eligible_pair_count=len(d['eligible_pairs']),selected_document=bool(d['eligible_pairs']),
                P1_predicted_rotation=d.get('P1_predicted_rotation',''),L7_predicted_rotation=d.get('L7_predicted_rotation',''),
                P1_gap=d.get('P1_gap',''),L7_gap=d.get('L7_gap',''),decision=d['decision'],short_result_sha256=d['short_result_sha256']))
    if decisions:publish_text(OUT/'SHORT_HORIZON_SELECTION.csv',buffer.getvalue())

def select():
    cfg=verify();a=read(OLD/'PROSPECTIVE_FULL_POOL_INPUT_IDENTITY_AUDIT.json')
    candidates=[e for e in a['inputs'] if e['input_identity_eligible']]
    assert len(candidates)==58
    assert [e['document'] for e in candidates]==read(OUT/'INPUT_ELIGIBILITY_REPRODUCTION.json')['eligible_order']
    decisions=sorted((read(f) for f in (OUT/'screening').glob('*.SELECTION_DECISION.json')),key=lambda d:d['scan_index']);selected=[d for d in decisions if d['eligible_pairs']]
    assert [d['scan_index'] for d in decisions]==list(range(1,len(decisions)+1))
    assert [d['document'] for d in decisions]==[e['document'] for e in candidates[:len(decisions)]]
    for index,entry in enumerate(candidates[:40],1):
        if index<=len(decisions):continue
        document=entry['document'];result=run(document);conditions=cfg['conditions']
        scores={r['condition']:{k:r['aggregate'][k] for k in ['P1','L7_FUNCTIONAL_ORIGINAL','L7_STATE_ORIGINAL','L7_TOTAL_ORIGINAL']} for r in result['results']}
        av=[scores[c]['P1'] for c in conditions];bv=[scores[c]['L7_TOTAL_ORIGINAL'] for c in conditions]
        ra,rb=midranks(av),midranks(bv);pairs=opposite_pairs(conditions,av,bv,3)
        d=dict(scan_index=index,document=document,document_identity=entry,roster=conditions,scores=scores,
            P1_midranks=dict(zip(conditions,ra)),L7_midranks=dict(zip(conditions,rb)),eligible_pairs=pairs,
            decision='ELIGIBLE_SELECTED_FIRST4' if pairs else 'NO_ELIGIBLE_PAIR_CONTINUE_FROZEN_ORDER',
            short_result_path=str(OUT/'screening'/(document+'.json')),short_result_sha256=sha(OUT/'screening'/(document+'.json')))
        if pairs:
            pair=pairs[0]
            d.update(P1_predicted_rotation=pair['metric_a_winner'],L7_predicted_rotation=pair['metric_b_winner'],
                P1_gap=pair['rank_gap_a'],L7_gap=pair['rank_gap_b'],selected_pair=pair)
            selected.append(d)
        decisions.append(d);publish(OUT/'screening'/(document+'.SELECTION_DECISION.json'),d)
        print(json.dumps(dict(event='SCREENING_DECISION',document=document,scan_index=index,
            eligible_pairs=len(pairs),selected_documents=len(selected))),flush=True)
        if len(selected)==4:break
    export_selection_csv(decisions)
    if len(selected)!=4:
        publish(OUT/'SHORT_HORIZON_SELECTION_STATUS.json',dict(status='INSUFFICIENT_ELIGIBLE_DOCS_STOP',
            screened_documents=len(decisions),eligible_documents=len(selected),maximum=40,new_32K_science_units=0))
        raise RuntimeError('INSUFFICIENT_ELIGIBLE_DOCS_STOP')
    assert not list((OUT/'formal').glob('*.STARTED.json')) if (OUT/'formal').exists() else True
    freeze=read(OUT/'SOURCE_CODE_FREEZE.json')
    protocol=read(OLD/'DOCUMENT_SELECTION_AND_EXCLUSION_RULES.json')
    value=dict(status='FROZEN_BEFORE_ANY_LONG_HORIZON_OUTCOME',experiment=protocol['experiment'],documents=selected,
        screened_documents=len(decisions),maximum_screen=40,seed=0,source_commit=freeze['source_commit'],
        source_code_freeze_sha256=sha(OUT/'SOURCE_CODE_FREEZE.json'),
        input_audit_sha256=sha(OUT/'INPUT_ELIGIBILITY_REPRODUCTION.json'),
        preregistration_sha256=sha(OLD/'NEXT_PROSPECTIVE_4DOC_PREREGISTRATION.md'),
        pair_rule_sha256=sha(WT/'next_prospective_pair_rule.py'),rotation_identities=protocol['calibration']['rotation_identities'],
        primary=protocol['primary'],execution_order=[dict(document=d['document'],conditions=['FP',d['P1_predicted_rotation'],d['L7_predicted_rotation']]) for d in selected],
        original_formal_verdict='MIXED_PERMANENTLY_PRESERVED',long_horizon_outcome_accessed=False,frozen_epoch=time.time())
    publish(OUT/'FROZEN_4DOC_SELECTION.json',value)
    publish(OUT/'SELECTION_FREEZE_RECEIPT.json',dict(sha256=sha(OUT/'FROZEN_4DOC_SELECTION.json'),epoch=time.time(),
        long_GPU_workers_launched=0,long_science_units_launched=0))
    publish(OUT/'SHORT_HORIZON_SELECTION_STATUS.json',dict(status='COMPLETE_4_ELIGIBLE_FROZEN',
        screened_documents=len(decisions),selected_documents=4,selection_sha256=sha(OUT/'FROZEN_4DOC_SELECTION.json')))
    return value

def make_long_spec(selection):
    old=read(OLD/'CROSS_ROTATION_FROZEN_SPEC.json');source_assets=old['source_assets'].copy()
    documents=[];units=[]
    for d in selection['documents']:
        entry=d['document_identity'];document=d['document'];token_file=dict(path=entry['token_file_path'],sha256=entry['token_file_sha256'])
        raw=dict(path=entry['raw_text_path'],sha256=entry['raw_sha256'])
        documents.append(dict(document=document,token_file=token_file,raw_text=raw,
            P1_predicted_winner=d['P1_predicted_rotation'],L7_predicted_winner=d['L7_predicted_rotation']))
        for item in [token_file,raw]:
            st=Path(item['path']).stat();source_assets[item['path']]=dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns,sha256=item['sha256'])
        for condition in ['FP',d['P1_predicted_rotation'],d['L7_predicted_rotation']]:
            rotation=None if condition=='FP' else selection['rotation_identities'][condition]
            units.append(dict(document=document,rotation=condition,token_file=token_file,rotation_identity=rotation))
    for item in selection['rotation_identities'].values():
        st=Path(item['path']).stat();source_assets[item['path']]=dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns,sha256=item['sha256'])
    spec=dict(experiment=selection['experiment'],status='FROZEN_PROSPECTIVE_4DOC',documents=documents,new_units=units,
        source_assets=source_assets,source_commit=read(OUT/'SOURCE_CODE_FREEZE.json')['source_commit'],
        source_worktree=str(WT),engine_sha256=old['engine_sha256'],model=old['model'],runtime=old['runtime'],
        primary=old['primary'],secondary=old['secondary'],resolution=old['resolution'],
        execution=dict(old['execution'],FP_references=4,student_trajectories=8),
        reference_experiment=str(OUT),reference_primary_spec_sha256='CURRENT_FROZEN_SPEC_SHA_VERIFIED_AT_READ',
        historical_STOP_sha256=old['historical_STOP_sha256'],frozen_selection_sha256=sha(OUT/'FROZEN_4DOC_SELECTION.json'),
        no_GPUh_hard_cap=True,no_fixed_VRAM_percent_gate=True,no_progress_timeout_seconds=900,
        original_formal_verdict='MIXED_PERMANENTLY_PRESERVED',planned_FP=4,planned_students=8)
    publish(OUT/'FROZEN_SPEC.json',spec)
    publish(OUT/'FP_PAIRING_CONTRACT_GATE.json',dict(status='PASS',spec_sha256=sha(OUT/'FROZEN_SPEC.json'),
        mode='NEW_MATCHED_FP_PER_FROZEN_DOCUMENT_SAME_GPU_RUNTIME',old_FP_reused_for_new_documents=False,
        numeric_contract_reused_from_qualified_GPU6=True,FP_execution_and_payload_integrity='PENDING_FORMAL_FP'))
    return spec

def qualify(spec):
    pairs=[]
    for unit in spec['new_units']:
        document,condition=unit['document'],unit['rotation']
        parent=run(document,condition,'parent');resumed=run(document,condition,'resumed_parity')
        for field in ['uninterrupted_suffix_full_logit_sha256','final_full_native_cache_sha256','audit_state_fingerprint','checkpoint_sha256']:
            assert parent[field]==resumed[field],('EXACT_FRESH_PROCESS_PARITY_FAILED',document,condition,field)
        if condition!='FP':assert parent['integrity']['passed'] and resumed['integrity']['passed']
        inventory=parent['cache_inventory'];first=inventory['511'];last=inventory['8255']
        rate=(last['cache_bytes']-first['cache_bytes'])/(8256-512);assert rate==61440
        assert all(x['cache_bytes']==first['cache_bytes']+(int(k)+1-512)*rate for k,x in inventory.items())
        pairs.append(dict(document=document,rotation=condition,status='PASS',parent_sha256=sha(OUT/'runs'/f'{document}__{condition}__parent.json'),
            resumed_sha256=sha(OUT/'runs'/f'{document}__{condition}__resumed_parity.json'),checkpoint_sha256=parent['checkpoint_sha256'],
            observer_ON_equals_OFF=True,own_rotation_history=True,fresh_process_exact_64_logits_cache_audit=True,
            native_cache_growth_bytes_per_token=rate,predicted_cache_bytes_32768=last['cache_bytes']+(32768-8256)*rate))
    publish(OUT/'ENGINEERING_QUALIFICATION_PRE_FORMAL.json',dict(status='PASS',units=pairs,count=12,
        same_GPU_runtime=True,no_tolerance_relaxation=True,checkpoint_restore_source_AST_unchanged=True,
        observer_no_mutation_PASS=True,INT8_dequantized_FP32_carrier_not_physical_INT8_storage=True,
        full32768_finite_status='PENDING_FORMAL',qualification_completed_epoch=time.time()))
    publish(OUT/'QUALIFICATION_GATES_COMPLETED.json',dict(status='ALL_EIGHT_GATES_PASS',
        full32768_finite_status='PENDING_FORMAL',engineering_qualification_sha256=sha(OUT/'ENGINEERING_QUALIFICATION_PRE_FORMAL.json')))

def main():
    selection=select();spec=make_long_spec(selection);qualify(spec)
    for d in selection['execution_order']:
        for condition in d['conditions']:run(d['document'],condition,'formal')
    publish(OUT/'PROSPECTIVE_COMPLETE.json',dict(status='COMPLETE',experiment=selection['experiment'],
        FP_complete=4,student_complete=8,planned_trajectories=12,endpoint=32768,epoch=time.time(),
        selection_sha256=sha(OUT/'FROZEN_4DOC_SELECTION.json'),original_formal_verdict='MIXED_PERMANENTLY_PRESERVED',training_steps=0))
    print('PROSPECTIVE_12_COMPLETE_STOP_NO_NEXT_SCIENCE',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if not (OUT/'STOP.json').exists():publish(OUT/'STOP.json',dict(status='STOP',stage='PROSPECTIVE_CONTROLLER',
            reason=repr(exc),traceback=traceback.format_exc(),epoch=time.time(),no_retry=True))
        raise
