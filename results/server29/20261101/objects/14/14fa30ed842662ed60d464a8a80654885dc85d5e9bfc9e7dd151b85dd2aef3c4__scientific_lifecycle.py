"""Fail-closed preregistration, document result sealing and exact shard release.

No GPU computation. Historical files are only read. Release is solely of new
completed document shards, by an explicit prevalidated manifest, never a tree.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from capture_original_drive import AMEND,ROOT,REPO,read,save,sha
from isolated_replay_engineering import GATES,RULES,LAYERS,CONDITIONS,require
sys.path.insert(0,str(ROOT/'scripts'))
import prepare_local_documents as A

PHASES=['factorization','schema','fp','invariance','decomposition','isolation','determinism','streaming']
HORIZONS=[2048,8192,32768]

def write_text(path,text):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='') as f:f.write(text);f.flush();os.fsync(f.fileno())
    path.chmod(0o444)

def historical_integrity():
    history=A.history();assert A.seal(ROOT,'SHA256SUMS')['all_exact']
    for directory in [ROOT/'amendments/STORAGE_LIFECYCLE_AMENDMENT_V1',AMEND]:
        for line in (directory/'SHA256SUMS').read_text().splitlines():
            expected,relative=line.split(None,1);path=(directory/relative).resolve()
            assert path.is_relative_to(ROOT.resolve()) and sha(path)==expected
    return history

def preregister():
    control=read(AMEND/'continuation_control_freeze.json')
    for relative,expected in control['code_sha256'].items():assert sha(AMEND/relative)==expected,'Prospective controller code drift'
    for phase in PHASES:require(phase)
    assert not list((AMEND/'scientific').rglob('drive_manifest.json')),'Scientific capture already exists before preregistration'
    history=historical_integrity();documents=read(ROOT/'protocol/frozen_discovery_documents.json')
    anchors=read(ROOT/'protocol/frozen_anchor_conditions.json');rules=read(RULES)
    for row in documents['documents']:
        assert sha(ROOT/row['token_path'])==row['token_sha256'] and sha(ROOT/row['text_path'])==row['text_sha256']
    assert len(documents['documents'])==4 and anchors['ANCHOR_CONDITION_COUNT']==4
    budget=read(AMEND/'engineering2048_storage_supplement.json')
    st=os.statvfs(ROOT);available=st.f_bavail*st.f_frsize
    assert available>=budget['PEAK_REQUIRED_STORAGE_BYTES'],'Current capacity failed before preregistration'
    frozen_files=[ROOT/'protocol/frozen_discovery_documents.json',ROOT/'protocol/frozen_anchor_conditions.json',
        ROOT/'protocol/tokenizer_identity.json',ROOT/'protocol/candidate_pool_manifest.json']
    # Candidate-pool location is inherited. Never select/materialize candidates
    # here; freeze every existing prospective input/protocol file as evidence.
    frozen_files=[p for p in frozen_files if p.exists()]
    frozen_files.extend(p for p in sorted((ROOT/'protocol').rglob('*')) if p.is_file())
    frozen_files.extend(p for p in sorted((ROOT/'inputs').rglob('*')) if p.is_file())
    frozen_files.extend(p for p in sorted((AMEND/'protocol').rglob('*')) if p.is_file())
    frozen_files.extend(p for p in sorted((AMEND/'scripts').glob('*')) if p.is_file() and p.suffix in ['.py','.sh','.txt'])
    frozen_files.extend([AMEND/'long_qualification_freeze.json',AMEND/'extended_capture_freeze.json',
        AMEND/'engineering2048_storage_supplement.json',AMEND/'storage_lifecycle_report.json'])
    frozen_files.extend(GATES/(phase+'.json') for phase in PHASES)
    frozen_files.append(AMEND/'engineering/capture_runtime_variability.json')
    frozen_files=sorted(set(frozen_files))
    manifest=dict(files=[dict(path=str(p.relative_to(REPO)),sha256=sha(p),bytes=p.stat().st_size) for p in frozen_files],
        role='FULL_SCIENTIFIC_PREMEASUREMENT_FREEZE; all gates passed, no32K capture/replay yet')
    prereg=dict(EXPERIMENT='QWEN_GDN_FROZEN_DRIVE_ANCHOR_MECHANISM_V1',
        scientific_type='PROSPECTIVE_CONTROLLED_MECHANISM_DISCOVERY',relation_to_parent='NEW_EXPERIMENT_NOT_PARENT_RESUME',
        document_manifest_sha256=sha(ROOT/'protocol/frozen_discovery_documents.json'),
        anchor_manifest_sha256=sha(ROOT/'protocol/frozen_anchor_conditions.json'),documents=documents,anchors=anchors,
        horizons=HORIZONS,metric_rules=rules,metric_window='inclusive1024..T',epsilon=1e-12,
        weighting='Equal layer/token weighting; whole32-head state Frobenius energies',
        persistent_trajectory=True,reset512=False,quantizer='Unmodified formal INT8-C128 quantize_c128; persistent writeback',
        capture_policy_sha256=sha(AMEND/'protocol/extended_capture_policy.json'),
        canonical_capture='FIRST_SUCCESSFUL_VALID_CAPTURE',capture_strict_determinism=False,replay_strict_determinism=True,
        release_policy='Only exact enumerated new bulky drive_shards after capture+all4anchors+all3horizons+all result seals verified',
        keep_forever='documents/tokens/tokenizer/source/model/drive manifests/per-shard hashes/runtime/anchors/metrics/results/release records/sparse FP references',
        no_failed_drive_deletion=True,storage_strategy='SEQUENTIAL_PER_DOCUMENT_CAPTURE_REPLAY_RELEASE',
        maximum_concurrent_full_document_drives=1,storage_formula='BASE+max(5GiB,20%BASE)<=AVAILABLE; margin once',
        lifecycle_peak_required_bytes=budget['PEAK_REQUIRED_STORAGE_BYTES'],available_at_final_freeze=available,
        primary='C6R-minus-H',secondary=['C7-minus-C6R','Native-minus-H'],
        interpretation='Descriptive mechanism discovery, n_docs4; no universal surrogate rank or end-to-end reasoning causality',
        statistics=['per_doc','median_paired_difference','sign_count','range','horizon_consistency'],
        confirmation_rule='Continue the original frozen hash ranking after the four Discovery documents; NOT_EXECUTED',
        prohibited=dict(training=True,optimizer_step=True,AIME=True,free_generation=True,confirmation=True,intervention=True,rotation_search=True),
        all_engineering_gates='PASS',any_new_failure='FAIL_AND_STOP; no retry/relaxation; preserve incomplete drive',
        historical_integrity=history)
    save(AMEND/'preregistration.json',prereg)
    save(AMEND/'PREMEASUREMENT_FREEZE_MANIFEST.json',manifest)
    write_text(AMEND/'PREREGISTRATION.md','# Full scientific premeasurement preregistration\n\n'+
        'All engineering gates passed. The same frozen four documents and four anchors remain unchanged. '+
        'The original double-counted stop is preserved. '+
        'This append-only freeze governs sequential capture → Native/H/C6R/C7 persistent replay → seal → exact shard release.\n\n'+
        'FP64 formulas, common FP-state normalization, inclusive1024..T averaging,2K/8K/32K horizons, '+
        'original BF16 drive/FP32 state, original recurrence and formal C128 QDQ are unchanged.\n\n'+
        'Scientific execution is blocked until this complete manifest and preregistration are committed and the actual SHA is recorded.\n')
    freeze_paths=frozen_files+[AMEND/'preregistration.json',AMEND/'PREMEASUREMENT_FREEZE_MANIFEST.json',AMEND/'PREREGISTRATION.md']
    write_text(AMEND/'PREMEASUREMENT_SHA256SUMS',''.join(sha(p)+'  '+str(p.relative_to(REPO))+'\n' for p in sorted(freeze_paths)))
    for p in freeze_paths+[AMEND/'PREMEASUREMENT_SHA256SUMS']:
        subprocess.run(['git','add','-f','--',str(p.relative_to(REPO))],cwd=REPO,check=True)
    subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@local','commit','-m',
        'Complete scientific premeasurement freeze after G0-G12 PASS; no scientific capture yet'],cwd=REPO,check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    for p in freeze_paths:
        committed=subprocess.check_output(['git','show',commit+':'+str(p.relative_to(REPO))],cwd=REPO)
        assert hashlib.sha256(committed).hexdigest()==sha(p)
    save(AMEND/'scientific_execution_authorization.json',dict(DISCOVERY_EXECUTION_AUTHORIZED='YES',
        PREMEASUREMENT_COMMIT_SHA=commit,manifest_sha256=sha(AMEND/'PREMEASUREMENT_FREEZE_MANIFEST.json'),
        preregistration_sha256=sha(AMEND/'preregistration.json'),all_gates_pass=True,
        scientific_capture_count_at_freeze=0,confirmation_authorized=False))
    print('SCIENTIFIC_PREMEASUREMENT_FROZEN_AND_AUTHORIZED',flush=True)

def verify_freeze():
    authorization=read(AMEND/'scientific_execution_authorization.json')
    assert authorization['DISCOVERY_EXECUTION_AUTHORIZED']=='YES'
    for row in read(AMEND/'PREMEASUREMENT_FREEZE_MANIFEST.json')['files']:
        assert sha(REPO/row['path'])==row['sha256'],'Scientific freeze drift'
    assert sha(AMEND/'preregistration.json')==authorization['preregistration_sha256']
    return authorization

def seal_release(document):
    verify_freeze();policy=read(AMEND/'protocol/extended_capture_policy.json')
    path=Path(policy['documents'][document-1]['output_directory']);m=read(path/'drive_manifest.json');ms=sha(path/'drive_manifest.json')
    seal=read(path/'capture_seal.json');assert seal['DOCUMENT_DRIVE_SEAL_GATE']=='PASS' and seal['drive_manifest_sha256']==ms
    assert m['token_count']==32768 and m['all_complete_finite'] and m['doc_id']==policy['documents'][document-1]['doc_id']
    capture=read(AMEND/'engineering/capture_runtime_variability.json');assert capture['DRIVE_RELEASE_AFTER_SEAL_ALLOWED']=='YES'
    results=[]
    for condition in CONDITIONS:
        p=path/'results'/condition/'replay_result.json';v=read(p);s=read(p.with_name('result_seal.json'))
        assert v['status']=='PASS' and s['RESULT_SEAL_GATE']=='PASS' and sha(p)==s['replay_result_sha256']
        assert v['identity']['drive_manifest_sha256']==ms==s['drive_manifest_sha256']
        assert v['identity']['conditions']==[condition] and v['identity']['length']==32768
        assert set(v['identity']['outputs'][condition]['horizons'])==set(map(str,HORIZONS))
        assert v['identity']['calls']['blocked_cumsum']==v['identity']['calls']['full_model_forward']==0
        results.append(dict(condition=condition,path=str(p),sha256=sha(p),seal_sha256=sha(p.with_name('result_seal.json'))))
    save(path/'all_anchor_results_seal.json',dict(RESULT_SEAL_GATE='PASS',drive_manifest_sha256=ms,
        all4anchors_complete=True,all3horizons_complete=True,results=results))
    shards=[];target=(path/'drive_shards').resolve()
    for row in m['shards']:
        p=(path/row['relative_path']).resolve()
        assert p.parent==target and p.is_relative_to((AMEND/'scientific').resolve())
        assert p.is_file() and not (path/row['relative_path']).is_symlink()
        assert p.name.startswith('tokens_') and p.suffix=='.pt'
        assert p.stat().st_size==row['file_size_bytes'] and sha(p)==row['file_sha256']
        shards.append(dict(path=str(p),sha256=row['file_sha256'],bytes=row['file_size_bytes']))
    assert len(shards)==512 and len(set(r['path'] for r in shards))==512
    record=dict(doc_id=m['doc_id'],drive_manifest_sha256=ms,released_paths=[r['path'] for r in shards],
        released_bytes=sum(r['bytes'] for r in shards),all_anchor_results_complete=True,
        all3horizons_complete=True,all_result_hashes_verified=True,exact_shards=shards,
        result_seal_sha256=sha(path/'all_anchor_results_seal.json'))
    save(path/'drive_release_intent.json',record)
    for row in shards:Path(row['path']).unlink()
    assert all(not Path(row['path']).exists() for row in shards)
    for row in results:assert sha(row['path'])==row['sha256']
    record.update(release_complete=True,release_time=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        release_policy='Only large new completed drive shards; no older artifact/directory/reference deletion')
    save(path/'drive_release_record.json',record)
    print('DOC_%02d_RESULT_SEAL_AND_RELEASE_PASS BYTES=%d'%(document,record['released_bytes']),flush=True)

def csv_write(path,rows):
    assert rows;fields=list(rows[0]);buffer=io.StringIO(newline='');writer=csv.DictWriter(buffer,fieldnames=fields)
    writer.writeheader();writer.writerows(rows)
    write_text(path,buffer.getvalue())
    with path.open(newline='',encoding='utf-8') as f:
        imported=list(csv.DictReader(f));assert len(imported)==len(rows) and list(imported[0])==fields
        assert len({tuple(r.items()) for r in imported})==len(rows)
        for row in imported:
            for name in fields:
                if name.startswith('J_') or name.startswith('delta_J_'):assert math.isfinite(float(row[name]))

def finalize():
    auth=verify_freeze();history=historical_integrity();policy=read(AMEND/'protocol/extended_capture_policy.json');rules=read(RULES)
    names=rules['scientific_metric_names'];unit_rows=[];layer_rows=[];lookup={};releases=[];closure=[];disk=[]
    for document,item in enumerate(policy['documents'],1):
        path=Path(item['output_directory']);m=read(path/'drive_manifest.json');ms=sha(path/'drive_manifest.json')
        release=read(path/'drive_release_record.json');assert release['release_complete'] and release['all_result_hashes_verified']
        assert release['drive_manifest_sha256']==ms and all(not Path(p).exists() for p in release['released_paths'])
        releases.append(dict(document=document,bytes_released=release['released_bytes'],manifest_sha256=ms))
        disk.append(dict(document=document,minimum_available_bytes=m['minimum_observed_available_bytes'],
            observed_capture_disk_consumption_bytes=m['observed_disk_consumption_bytes']))
        for c in CONDITIONS:
            p=path/'results'/c/'replay_result.json';v=read(p);s=read(p.with_name('result_seal.json'))
            assert sha(p)==s['replay_result_sha256'] and v['identity']['drive_manifest_sha256']==ms
            output=v['identity']['outputs'][c]
            closure.append(dict(document=document,condition=c,vector_max=output['max_vector_closure_residual'],energy_max=output['max_energy_closure_residual']))
            for horizon in HORIZONS:
                h=output['horizons'][str(horizon)];assert h['token_count']==horizon-1024+1
                provenance=dict(document_index=document,doc_id=m['doc_id'],condition=c,horizon_tokens=horizon,
                    averaging_first_token=1024,averaging_token_count=h['token_count'],doc_sha256=m['doc_text_sha256'],
                    token_file_sha256=m['fixture_token_ids_file_sha256'],drive_manifest_sha256=ms,
                    anchor_manifest_sha256=v['identity']['anchor_manifest_sha256'],
                    metric_definition_sha256=sha(RULES),PREMEASUREMENT_COMMIT_SHA=auth['PREMEASUREMENT_COMMIT_SHA'])
                row=dict(provenance);row.update(dict(zip(names,h['normalized_mean'])))
                unit_rows.append(row);lookup[document,c,horizon]=row
                for i,l in enumerate(LAYERS):
                    r=dict(provenance,layer=l);r.update(dict(zip(names,h['per_layer'][i])));layer_rows.append(r)
    assert len(unit_rows)==48 and len(layer_rows)==1152
    contrasts=[];summaries={};comparisons=[('C6R','H','PRIMARY'),('C7','C6R','SECONDARY'),('Native','H','SECONDARY')]
    for a,b,role in comparisons:
        label=a+'_minus_'+b;summary={}
        for horizon in HORIZONS:
            collected={n:[] for n in names}
            for document in range(1,5):
                left=lookup[document,a,horizon];right=lookup[document,b,horizon]
                row=dict(document_index=document,doc_id=left['doc_id'],comparison=label,role=role,horizon_tokens=horizon,
                    drive_manifest_sha256=left['drive_manifest_sha256'],metric_definition_sha256=sha(RULES),
                    PREMEASUREMENT_COMMIT_SHA=auth['PREMEASUREMENT_COMMIT_SHA'])
                assert left['drive_manifest_sha256']==right['drive_manifest_sha256']
                for n in names:
                    delta=left[n]-right[n];row['delta_'+n]=delta;collected[n].append(delta)
                contrasts.append(row)
            summary[str(horizon)]={n:dict(per_doc=values,median=statistics.median(values),minimum=min(values),maximum=max(values),
                negative=sum(v<0 for v in values),zero=sum(v==0 for v in values),positive=sum(v>0 for v in values)) for n,values in collected.items()}
        consistency={n:[dict(document=d,signs=[0 if lookup[d,a,h][n]==lookup[d,b,h][n] else (1 if lookup[d,a,h][n]>lookup[d,b,h][n] else -1) for h in HORIZONS]) for d in range(1,5)] for n in names}
        summaries[label]=dict(comparison=label,role=role,horizons=summary,horizon_signs=consistency,n_docs=4,
            descriptive_only=True,no_p_value_or_universal_ranking=True)
    csv_write(AMEND/'results/per_unit_horizon_metrics.csv',unit_rows)
    csv_write(AMEND/'results/per_layer_metrics.csv',layer_rows)
    csv_write(AMEND/'results/anchor_pair_contrasts.csv',contrasts)
    save(AMEND/'analysis/decomposition_summary.json',dict(closure=closure,ratios_are_not_probabilities=True,alignment_signed=True))
    ratios=[]
    for row in unit_rows:
        acc=row['J_ACC_FD'];ratios.append(dict(document_index=row['document_index'],condition=row['condition'],horizon_tokens=row['horizon_tokens'],
            HISTORY_over_ACC=row['J_HISTORY_FD']/acc if acc else None,
            INJECTION_over_ACC=row['J_INJ_PATH_FD']/acc if acc else None,
            ALIGNMENT_over_ACC=row['J_ALIGN_FD']/acc if acc else None))
    save(AMEND/'analysis/horizon_summary.json',dict(horizons=HORIZONS,ratios=ratios,definition='Ratios of aggregated common-denominator energies, not probability shares'))
    for label,filename in [('C6R_minus_H','c6r_vs_hadamard'),('C7_minus_C6R','c7_vs_c6r'),('Native_minus_H','native_vs_hadamard')]:
        save(AMEND/'analysis'/(filename+'.json'),summaries[label])
    save(AMEND/'storage_execution_report.json',dict(releases=releases,capture_disk_observations=disk,
        max_concurrent_full_drives=1,all_completed_drive_shards_released=True,failed_drive_shards_deleted=0,
        no_old_artifacts_deleted=True,formula='Safety margin counted once',
        minimum_observed_available_bytes=min(r['minimum_available_bytes'] for r in disk)))
    gates={field:read(GATES/(phase+'.json'))['gate'] for phase,field in [
        ('factorization','DRIVE_FACTORIZATION_GATE'),('schema','CAPTURE_SCHEMA_GATE'),('fp','FP_REPLAY_PARITY_GATE'),
        ('invariance','NO_QDQ_ROTATION_INVARIANCE_GATE'),('decomposition','COUNTERFACTUAL_DECOMPOSITION_GATE'),
        ('isolation','REPLAY_ISOLATION_GATE'),('determinism','REPLAY_DETERMINISM_GATE'),('streaming','STREAMING_REDUCER_PARITY_GATE')]}
    budget=read(AMEND/'engineering2048_storage_supplement.json');baseline=read(AMEND/'storage_lifecycle_report.json')
    status=dict(EXPERIMENT='QWEN_GDN_FROZEN_DRIVE_ANCHOR_MECHANISM_V1',
        STORAGE_LIFECYCLE_AMENDMENT='SEQUENTIAL_PER_DOCUMENT_CAPTURE_REPLAY_RELEASE',OLD_TOTAL_STORAGE_ESTIMATE_GB=45.4,
        MAX_CONCURRENT_FULL_DOCUMENT_DRIVES=1,OLD_EVIDENCE_INTEGRITY_GATE='PASS',
        FROZEN_DISCOVERY_DOCUMENT_INTEGRITY_GATE='PASS',DISCOVERY_DOCUMENT_SELECTION_GATE='PASS',DISCOVERY_DOC_COUNT=4,
        ANCHOR_CONDITION_COUNT=4,ANCHOR_IDENTITY_GATE='PASS',CAPTURE_EMPIRICAL_REPRODUCIBILITY_GATE='PASS',
        DRIVE_RELEASE_AFTER_SEAL_ALLOWED='YES',ESTIMATED_SINGLE_DOC_DRIVE_BYTES=19478347776,
        PEAK_REQUIRED_STORAGE_BYTES=budget['PEAK_REQUIRED_STORAGE_BYTES'],BASE_PEAK_BYTES=budget['BASE_PEAK_BYTES'],
        SAFETY_MARGIN_BYTES=budget['SAFETY_MARGIN_BYTES'],STORAGE_CAPACITY_GATE='PASS',
        ORIGINAL_CORRECTED_PEAK_REQUIRED_STORAGE_BYTES=baseline['PEAK_REQUIRED_STORAGE_BYTES'],
        ORIGINAL_AVAILABLE_DATA_BYTES=baseline['AVAILABLE_DATA_BYTES'],ORIGINAL_HEADROOM_BYTES=baseline['HEADROOM_BYTES'],
        PREMEASUREMENT_COMMIT_SHA=auth['PREMEASUREMENT_COMMIT_SHA'],DISCOVERY_EXECUTION_AUTHORIZED='YES',
        SCIENTIFIC_MEASUREMENT_COUNT=16,DISCOVERY_COMPLETE='16/16',HORIZON_2048_COMPLETE='16/16',
        HORIZON_8192_COMPLETE='16/16',HORIZON_32768_COMPLETE='16/16',COMPLETED_DOC_DRIVES_RELEASED='4/4',
        CONFIRMATION_RULE_FROZEN='YES',CONFIRMATION_EXECUTED=0,FULL_MODEL_CLOSED_LOOP_PROPAGATION='NOT_EVALUATED',
        NEW_C8_TRAINING_AUTHORIZED='NO',NEW_AIME_GENERATION=0,NEW_FREE_GENERATION=0,OPTIMIZER_STEP=0,
        OLD_PARENT_STOP_PRESERVED='YES',OLD_CUMSUM_CLOSURE_FAIL_PRESERVED='YES',OLD_PREFLIGHT_FAIL_PRESERVED='YES',
        OLD_INPUT_FREEZE_AMENDMENT_FAIL_PRESERVED='YES',OLD_DOUBLE_COUNT_STORAGE_FAIL_PRESERVED='YES',
        REPLAY_BLOCKED_CUMSUM_CALL_COUNT=0,REPLAY_FULL_MODEL_FORWARD_COUNT=0,FINAL_ACTION='DISCOVERY_COMPLETE_REPORT_ONLY')
    status.update(gates);save(AMEND/'final_verdict.json',status)
    lines=['# Frozen-drive anchor mechanism Discovery report','',
        'The storage double-counting correction was append-only. Previous FAIL evidence remains byte-identical. All four frozen documents and anchors are unchanged.',
        '', '## Q1–Q6: qualification','',
        'The prospective outcome-independent document protocol, unique four-document freeze and authoritative four-anchor identity passed. Replay used the unchanged cumsum-free production recurrence and formal QDQ/rotation. FP capture parity, no-QDQ invariance, vector/energy identities, strict fresh-process determinism and streaming parity passed.',
        '', '## Q7–Q9: paired mechanism contrasts','',
        'Values below are median paired differences across the four documents. Lower accumulated error is not an endpoint accuracy result. Each component is shown without attributing an end-to-end causal explanation.',
        '', '| Contrast | Horizon | '+ ' | '.join(names)+' |','|---|---:|'+ '|'.join(['---:']*6)+'|']
    for a,b,role in comparisons:
        label=a+'_minus_'+b
        for h in HORIZONS:lines.append('| '+label+' | '+str(h)+' | '+' | '.join('%.9g'%summaries[label]['horizons'][str(h)][n]['median'] for n in names)+' |')
    lines.extend(['','## Q10: horizon trends','',
        'The per-document horizon sign consistency and exact energy ratios are recorded in the paired-comparison JSON and horizon_summary.json. Alignment may be negative: these ratios are not probability shares.',
        '', '## Q11: interpretation boundary','',
        'These results concern intrinsic recurrent error dynamics under a frozen external FP drive. They do not evaluate full-model closed-loop propagation or reasoning causality, and do not validate a universal surrogate ranking.',
        '', '## Q12: future confirmation','',
        'The paired component and horizon patterns may motivate an independently preregistered confirmation hypothesis on untouched subsequent hash-ranked documents. No confirmation, intervention, new rotation training or new AIME/free generation was run.',
        '', '## Required final status','', '```json',json.dumps(status,indent=2),'```',
        '', 'Underlying per-document/per-layer values, sign counts, ranges, drive/source/anchor provenance and exact release records are included in the sealed artifacts.', ''])
    write_text(AMEND/'FINAL_REPORT.md','\n'.join(lines))
    print('DISCOVERY_COMPLETE_REPORT_ONLY',flush=True)
    seal_final()

def seal_final():
    files=[p for p in sorted(AMEND.rglob('*')) if p.is_file() and p.name!='FINAL_SHA256SUMS' and '__pycache__' not in p.parts]
    write_text(AMEND/'FINAL_SHA256SUMS',''.join(sha(p)+'  '+str(p.relative_to(AMEND))+'\n' for p in files))
    for line in (AMEND/'FINAL_SHA256SUMS').read_text().splitlines():
        expected,rel=line.split(None,1);assert sha(AMEND/rel)==expected

def stop(phase):
    assert not (AMEND/'FINAL_REPORT.md').exists()
    errors=[p for p in AMEND.rglob('*.error.json') if '__pycache__' not in p.parts]
    assert errors or phase!='unspecified','Failure evidence required'
    status=dict(EXPERIMENT='QWEN_GDN_FROZEN_DRIVE_ANCHOR_MECHANISM_V1',FINAL_ACTION='FAIL_AND_STOP',failed_phase=phase,
        error_files=[dict(path=str(p),sha256=sha(p),evidence=read(p)) for p in errors],
        old_evidence_preserved=historical_integrity(),STORAGE_CAPACITY_GATE='PASS',
        original_accounting_correction=read(AMEND/'storage_lifecycle_report.json'),
        retry_started=False,tolerance_changed=False,failed_drive_deleted=False,confirmation_executed=0,
        training_started=False,AIME_generation=0,free_generation=0)
    status['gates']={p.stem:read(p).get('gate') for p in sorted(GATES.glob('*.json')) if read(p).get('gate')}
    status['completed_units']=len(list((AMEND/'scientific').rglob('replay_result.json')))
    status['drive_release_count']=len(list((AMEND/'scientific').rglob('drive_release_record.json')))
    auth=AMEND/'scientific_execution_authorization.json';status['PREMEASUREMENT_COMMIT_SHA']=read(auth)['PREMEASUREMENT_COMMIT_SHA'] if auth.exists() else None
    field_map=dict(factorization='DRIVE_FACTORIZATION_GATE',schema='CAPTURE_SCHEMA_GATE',fp='FP_REPLAY_PARITY_GATE',
        invariance='NO_QDQ_ROTATION_INVARIANCE_GATE',decomposition='COUNTERFACTUAL_DECOMPOSITION_GATE',
        isolation='REPLAY_ISOLATION_GATE',determinism='REPLAY_DETERMINISM_GATE',streaming='STREAMING_REDUCER_PARITY_GATE')
    for key,field in field_map.items():
        status[field]=status['gates'].get(key,'NOT_EVALUATED')
        if (GATES/(key+'.error.json')).exists():status[field]='FAIL'
    if any('repeatability' in str(p) for p in errors):status['REPLAY_DETERMINISM_GATE']='FAIL'
    baseline=status['original_accounting_correction'];capture=read(AMEND/'engineering/capture_runtime_variability.json')
    status.update(STORAGE_LIFECYCLE_AMENDMENT='SEQUENTIAL_PER_DOCUMENT_CAPTURE_REPLAY_RELEASE',
        MAX_CONCURRENT_FULL_DOCUMENT_DRIVES=1,OLD_TOTAL_STORAGE_ESTIMATE_GB=45.4,
        OLD_EVIDENCE_INTEGRITY_GATE='PASS',FROZEN_DISCOVERY_DOCUMENT_INTEGRITY_GATE='PASS',DISCOVERY_DOC_COUNT=4,
        ANCHOR_CONDITION_COUNT=4,ANCHOR_IDENTITY_GATE='PASS',DISCOVERY_DOCUMENT_SELECTION_GATE='PASS',
        CAPTURE_EMPIRICAL_REPRODUCIBILITY_GATE=capture['CAPTURE_EMPIRICAL_REPRODUCIBILITY_GATE'],
        DRIVE_RELEASE_AFTER_SEAL_ALLOWED=capture['DRIVE_RELEASE_AFTER_SEAL_ALLOWED'],
        BASE_PEAK_BYTES=baseline['BASE_PEAK_BYTES'],SAFETY_MARGIN_BYTES=baseline['SAFETY_MARGIN_BYTES'],
        PEAK_REQUIRED_STORAGE_BYTES=baseline['PEAK_REQUIRED_STORAGE_BYTES'],AVAILABLE_DATA_BYTES=baseline['AVAILABLE_DATA_BYTES'],
        HEADROOM_BYTES=baseline['HEADROOM_BYTES'],DISCOVERY_EXECUTION_AUTHORIZED='NO',
        DISCOVERY_COMPLETE=str(status['completed_units'])+'/16',
        HORIZON_2048_COMPLETE=str(status['completed_units'])+'/16',
        HORIZON_8192_COMPLETE=str(status['completed_units'])+'/16',
        HORIZON_32768_COMPLETE=str(status['completed_units'])+'/16',
        COMPLETED_DOC_DRIVES_RELEASED=str(status['drive_release_count'])+'/4',
        CONFIRMATION_RULE_FROZEN='YES',CONFIRMATION_EXECUTED=0,FULL_MODEL_CLOSED_LOOP_PROPAGATION='NOT_EVALUATED',
        NEW_C8_TRAINING_AUTHORIZED='NO',NEW_AIME_GENERATION=0,NEW_FREE_GENERATION=0,
        OLD_PARENT_STOP_PRESERVED='YES',OLD_CUMSUM_CLOSURE_FAIL_PRESERVED='YES',OLD_PREFLIGHT_FAIL_PRESERVED='YES',
        OLD_INPUT_FREEZE_AMENDMENT_FAIL_PRESERVED='YES',OLD_DOUBLE_COUNT_STORAGE_FAIL_PRESERVED='YES')
    save(AMEND/'final_verdict.json',status)
    write_text(AMEND/'FINAL_REPORT.md','# Resumed engineering/scientific workflow stopped\n\n'+
        'The storage correction passed and remains append-only. A subsequent gate/unit failed; all dependent work stopped. '+
        'No retries, threshold relaxation or deletion of an incomplete drive occurred.\n\n```json\n'+json.dumps(status,indent=2,ensure_ascii=False)+'\n```\n')
    seal_final()

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['preregister','release','finalize','stop'],required=True)
    p.add_argument('--document',type=int);p.add_argument('--failed-phase',default='unspecified');a=p.parse_args()
    try:
        if a.phase=='preregister':preregister()
        elif a.phase=='release':seal_release(a.document)
        elif a.phase=='finalize':finalize()
        else:stop(a.failed_phase)
    except Exception as e:
        error=AMEND/('lifecycle_'+a.phase+'.error.json')
        if not error.exists():save(error,dict(status='FAIL_AND_STOP',phase=a.phase,document=a.document,
            exception=type(e).__name__,message=str(e),traceback=traceback.format_exc(),retry_authorized=False))
        raise

if __name__=='__main__':main()
