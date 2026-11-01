"""CPU-only scoring of actual retained C6R trajectories. No GPU or reruns.

The original reducer is imported byte for byte. Only identity/I/O and labels
are new. Archived scores are fetched one document at a time and checked before
loading. Missing or failed trajectories never become zero-valued observations.
"""
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OLD = Path('/data/zypan/analyses/qwen_gdn_dual_axis_prospective_replication_5doc_v1/experiments/qwen_gdn/quantizer/QWEN_GDN_DUAL_AXIS_PROSPECTIVE_REPLICATION_5DOC_V1')
SSH = ['ssh', '-S', '/tmp/qwen_replication_archive29.sock', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'zypan@202.38.247.29']
CONDITIONS = ('C6R_C128', 'C6R_DUAL_M')


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(16 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def need(v, message):
    if not v:
        raise ValueError(message)


def save(rel, value):
    p = ROOT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def csv_save(rel, rows):
    if not rows:
        return  # No shell CSV for an unrun scientific analysis.
    with (ROOT / rel).open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def location(doc, relative):
    p = ROOT / f'archive/DOC_{doc:02d}_CURRENT_ASSET_LOCATIONS.json'
    m = read(p)
    need(m['status'] == 'PASS', 'Unverified new score archive')
    entries = [a for a in m['assets'] if a['relative_path'] == relative]
    need(len(entries) == 1, 'Ambiguous archived score location')
    return entries[0]


def get_score(doc, phase, scratch):
    rel = f'scores/{phase}.pt'
    a = location(doc, rel)
    source = ROOT / rel
    if source.is_file():
        p = source
    else:
        p = scratch / (phase + '.pt')
        need(not p.exists(), 'Unexpected stale analysis cache')
        # Stream a single known tensor into a new task-only temporary path.
        with p.open('xb') as f:
            subprocess.run(SSH + [shlex.join(['cat', a['archive_path']])], stdout=f, check=True, timeout=600)
    need(p.stat().st_size == a['bytes'] and sha(p) == a['sha256'], 'Current score SHA/bytes mismatch')
    return p, a


def normalized_winner(d):
    result = dict(d)
    result['Legacy_reducer_Winner'] = d['Winner']
    result['Winner'] = {'H_DUAL_M': 'D_WIN', 'H_C128': 'C_WIN', 'NUMERICALLY_UNRESOLVED': 'NUMERICALLY_UNRESOLVED'}[d['Winner']]
    return result


def descriptive(docs):
    if not docs:
        return dict(document_count=0, status='NOT_RUN')
    ka = math.fsum(d['K_A'] for d in docs) / len(docs)
    kd = math.fsum(d['K_D'] for d in docs) / len(docs)
    ds = [d['Delta'] for d in docs]
    gains = [d['RelativeGain'] for d in docs]
    den_error = math.fsum(d['K_A_arithmetic_error']+d['time_mean_subtraction_roundoff'] for d in docs)/len(docs)
    nu = (8*len(docs)+32)*2.0**-53
    den_error += nu/(1-nu)*(1+math.fsum(abs(d['K_A']) for d in docs)/len(docs)+den_error)
    return dict(document_count=len(docs), mean_K_C6R_C128=ka, mean_K_C6R_DUAL_M=kd,
                mean_Delta_C6R=math.fsum(ds) / len(ds), median_Delta_C6R=statistics.median(ds),
                mean_per_document_RelativeGain=math.fsum(gains) / len(gains) if all(x is not None for x in gains) else None,
                relative_gain_of_mean_K=(ka-kd)/ka if ka > den_error else None,
                mean_K_denominator_arithmetic_error=den_error,
                equal_weight_unit='document; historical two-doc pilot excluded')


def interaction_description(rows):
    if not rows:
        return dict(status='NOT_RUN', document_count=0)
    values=[d['Interaction'] for d in rows]
    return dict(document_count=len(rows),mean_Interaction=math.fsum(values)/len(values),
                median_Interaction=statistics.median(values),
                directions={k:sum(d['Interaction_direction']==k for d in rows) for k in ('POSITIVE','NEGATIVE','NUMERICALLY_UNRESOLVED')},
                interpretation='descriptive historically observed documents; do not pool A/B as confirmatory')


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.path.insert(0, str(ROOT/'src'))
    import torch
    import reducer as R
    R.configure_cpu()
    need(not torch.cuda.is_initialized(), 'CPU analyzer initialized CUDA')
    need(sha(ROOT/'src/reducer.py') == '7e94c95803acb7ae47f0e3ec44fa97faf7cc8d93c4ecc5791a56ea2cf305b84f', 'Original reducer changed')
    started = time.time()
    pre = read(ROOT/'PREREGISTRATION.json')
    old_docs = read(OLD/'FROZEN_DOCUMENTS.json')['selected_documents']
    history = read(OLD/'SCIENCE_RESULTS.json')['documents']
    catalog = {a['path']: a for a in read(OLD/'hashes/PAYLOAD_CATALOG.json')['assets']}
    manifest_sha = sha(ROOT/'SOURCE_MANIFEST.json')
    hist = read(ROOT/'HISTORICAL_ASSET_AUDIT.json')
    rotation = read(ROOT/'C6R_ROTATION_IDENTITY.json')
    need(hist['status'] == 'PASS', 'Historical reuse was not qualified')
    need(rotation['status'] == 'PASS', 'Frozen C6R was not qualified')
    new_source=read(ROOT/'SOURCE_MANIFEST.json')
    need(new_source['inherited_historical_source_manifest_sha256']==sha(OLD/'SOURCE_MANIFEST.json'), 'Historical source binding drifted')
    postrun=read(ROOT/'analysis/POSTRUN_SOURCE_VERIFICATION.json')
    need(postrun['status']=='PASS', 'Postrun content audit absent')
    verified={a['path']:(a['sha256'],a['bytes']) for a in postrun['actual_stream_hashed_dependencies']}
    required=list(hist['current_model_runtime_hashes'])+list(rotation['assets'].values())
    required += [dict(path=str(Path(hist['tokenizer_identity']['path'])/a['name']),sha256=a['sha256'],bytes=(Path(hist['tokenizer_identity']['path'])/a['name']).stat().st_size) for a in hist['tokenizer_identity']['files']]
    common_content_contract=bool(required) and all(verified.get(a['path'])==(a['sha256'],a['bytes']) for a in required)
    need(common_content_contract,'Exact shared weight/tokenizer/runtime/C6R content qualification incomplete')
    qualification=read(ROOT/'GPU_QUALIFICATION.json')
    need(qualification['status']=='PASS','Actual four-card C6R GPU qualification absent')
    ledger=read(ROOT/'GPU_RESOURCE_LEDGER.json')
    ledger_phases={a['phase']:a for a in ledger['phases']}
    science_source={a['path']:a['sha256'] for a in new_source['execution_sources']}
    old_source={a['path']:a['sha256'] for a in read(OLD/'SOURCE_MANIFEST.json')['execution_sources']}
    need(all(science_source['src/'+n]==old_source['src/'+n] for n in ('core.py','worker.py','codec.py','reducer.py')), 'Scientific core byte reuse contract changed')
    gpu = read(ROOT/'FINAL_GPU_STATUS.json')
    need(gpu['no_further_GPU_work'] and gpu['GPU_release_verified'], 'GPU/CPU boundary not released')
    scratch = Path('/tmp/qwen_c6r_interaction_cpu_score_cache_20261016')
    need(not scratch.exists(), 'Analysis cache already exists; no implicit resume')
    scratch.mkdir()
    paired, interactions, strata, windows, assets, missing, runtime = [], [], [], [], [], [], []
    score_index = []
    for doc, frozen in enumerate(old_docs, 1):
        phases = [f'science_doc{doc}_{c}' for c in CONDITIONS]
        if not all((ROOT/f'analysis/RECEIPT_{p}.json').is_file() and read(ROOT/f'analysis/RECEIPT_{p}.json')['status'] == 'PASS' for p in phases):
            missing.append(dict(document_index=doc, status='NOT_RUN_OR_INCOMPLETE', conditions={c: 'COMPLETE' if (ROOT/f'analysis/RECEIPT_{p}.json').is_file() and read(ROOT/f'analysis/RECEIPT_{p}.json').get('status')=='PASS' else 'NOT_RUN_OR_INCOMPLETE' for c,p in zip(CONDITIONS,phases)}))
            continue
        identity = dict(document_index=doc, doc_id=frozen['document_id'], raw_text_sha256=frozen['raw_text_sha256'],
                        input_ids_sha256=frozen['artifacts']['input_ids_32768']['canonical_json_sha256'],
                        source_manifest_sha256=manifest_sha, historical_source_manifest_sha256=sha(OLD/'SOURCE_MANIFEST.json'),
                        historical_preregistration_sha256=sha(OLD/'PRE_REGISTRATION.json'),
                        C6R_identity_sha256=sha(ROOT/'C6R_ROTATION_IDENTITY.json'),
                        GPU_UUID=pre['GPU_UUIDS'][str(pre['doc_GPU_assignment'][str(doc)])],
                        reference_score_file_sha256=catalog[f'scores/science_doc{doc}_FP_NATIVE.pt']['sha256'])
        fp_path = OLD/f'scores/science_doc{doc}_FP_NATIVE.pt'
        need(sha(fp_path) == identity['reference_score_file_sha256'], 'Original common FP reference changed')
        fp = torch.load(fp_path, map_location='cpu', weights_only=True)
        ids = read(OLD/frozen['artifacts']['input_ids_32769']['path'])
        need(fp['condition'] == 'FP_NATIVE' and fp['doc_id'] == identity['doc_id'] and fp['input_ids_sha256'] == identity['input_ids_sha256'] and fp['source_manifest_sha256']==identity['historical_source_manifest_sha256'], 'Original FP identity mismatch')
        need(fp['consumed_tokens'] == 32768 and tuple(fp['logits']) == R.SCORE_POSITIONS, 'Original FP score positions mismatch')
        arms = []
        envs = []
        for condition, phase in zip(CONDITIONS, phases):
            p, asset = get_score(doc, phase, scratch)
            payload = torch.load(p, map_location='cpu', weights_only=True)
            need(payload['condition'] == condition and payload['doc_id'] == identity['doc_id'] and payload['input_ids_sha256'] == identity['input_ids_sha256'], 'C6R score identity mismatch')
            need(payload['source_manifest_sha256'] == manifest_sha and payload['consumed_tokens'] == 32768 and tuple(payload['logits']) == R.SCORE_POSITIONS, 'C6R score provenance/count mismatch')
            env = read(ROOT/f'analysis/ENV_{phase}.json')
            need(env['UUID'] == identity['GPU_UUID'] and env['strict'] is True and env['TF32'] is False, 'C6R device/numerical configuration mismatch')
            need(env['C6R']['rotation_sha256']==rotation['rotation_SHA256'] and env['C6R']['deployment_manifest_sha256']==rotation['deployment_manifest_SHA256'] and env['C6R']['direct_frozen_load'] and not env['C6R']['theta_reconstruction'], 'Actual C6R environment rotation binding failed')
            need(env['actual_conditions']==list(CONDITIONS) and env['physical_GPU_ID']==pre['doc_GPU_assignment'][str(doc)], 'Actual condition/device identity changed')
            need(env['rotation_adapter']['scientific_sources']=={n:science_source['src/'+n] for n in ('core.py','worker.py','codec.py','reducer.py')}, 'Actual runtime mathematical source qualification mismatch')
            receipt_path=ROOT/f'analysis/RECEIPT_{phase}.json'
            receipt=read(receipt_path)
            need(receipt['result']==dict(status='PASS',condition=condition,doc=doc,consumed=32768,scores=384), 'New completed trajectory receipt mismatch')
            phase_record=ledger_phases[phase]
            need(phase_record['returncode']==0 and phase_record['released'] and phase_record['GPU_UUID']==identity['GPU_UUID'] and phase_record['receipt_sha256']==sha(receipt_path), 'Actual reservation/receipt/device pairing failed')
            envs.append(env)
            rows = []
            raw_path = ROOT/f'analysis/{phase}_KL_ROWS.jsonl'
            with raw_path.open('x') as f:
                for t in R.SCORE_POSITIONS:
                    a, b = fp['logits'][t], payload['logits'][t]
                    need(a.device.type == b.device.type == 'cpu' and a.dtype == b.dtype == torch.float32 and a.numel() == b.numel() == 248320 and tuple(a.shape) == tuple(b.shape), 'Full vocab FP32 shape/dtype mismatch')
                    row = R.score_pair(a, b, t, ids[t+1], identity)
                    rows.append(row)
                    f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                    score_index.append(dict(doc=doc, condition=condition, t=t, sha256=row['student_logits_sha256'], reference_sha256=row['reference_logits_sha256']))
            arms.append(rows)
            assets.append(dict(document_index=doc, condition=condition, **asset, actual_vocabulary=248320, actual_score_count=len(rows)))
            actual = read(ROOT/f'analysis/{phase}.json')
            need(actual['condition']==condition and actual['doc_id']==identity['doc_id'] and actual['consumed_tokens']==32768 and actual['writebacks']==32768*24 and actual['next_token_consumption_checks']==32767*24 and actual['rotation_calls']==32768*24 and actual['score_file_sha256']==asset['sha256'], 'Actual persistent C6R history/completion/score binding failed')
            times = [v['full_model_plus_QDQ_seconds'] for v in actual['performance'] if v['t'] >= 8192]
            risk_keys=('raw_row_bound_hits','raw_column_bound_hits','row_floor_hits','column_floor_hits','row_ceiling_hits','column_ceiling_hits')
            risk_totals={k:sum(v.get('scale_risk',{}).get(k,0) for v in actual['performance']) for k in risk_keys} if condition=='C6R_DUAL_M' else None
            runtime.append(dict(document_index=doc, condition=condition, wall_seconds=actual['wall_seconds'],
                                post_warmup_sample_count=len(times), post_warmup_median_seconds=statistics.median(times),
                                post_warmup_tokens_per_second=1/statistics.median(times), peak_allocated=actual['peak_allocated'],
                                peak_reserved=actual['peak_reserved'], mechanism=actual['mechanism'],
                                all_token_scale_risk_counts=risk_totals,
                                scale_risk_count_scope='raw hit/floor calls from all32768 actual Dual-M steps; not unique elements across time' if risk_totals is not None else 'Canonical C128 metrics sampled at384 score positions; no invented all-token FP16scale counters',
                                mechanism_scope='own student history local-QDQ; not common-input codec causal effect'))
            del payload
            if p.parent == scratch:
                need(sha(p) == asset['sha256'], 'Analysis cache changed before retirement')
                p.unlink()
            gc.collect()
        need(envs[0]['load']['architecture_sha256'] == envs[1]['load']['architecture_sha256'] and envs[0]['load']['parameter_dtypes'] == envs[1]['load']['parameter_dtypes'], 'Within-C6R model/runtime mismatch')
        result = R.summarize_document(*arms)
        result = normalized_winner(result)
        result['document_index'] = doc
        paired.append(result)
        h = history[doc-1]
        need(h['identity']['doc_id'] == identity['doc_id'] and h['identity']['input_ids_sha256'] == identity['input_ids_sha256'], 'Historical H pairing mismatch')
        old_env = read(OLD/f'analysis/ENV_science_doc{doc}_H_C128.json')
        runtime_equal = common_content_contract and qualification['status']=='PASS' and all(old_env['load'][k] == e['load'][k] for e in envs for k in ('tokenizer_class','parameter_dtypes','architecture_sha256','recurrent_source'))
        device_equal = h['identity']['GPU_UUID'] == identity['GPU_UUID']
        stratum = 'A' if runtime_equal and device_equal else 'B' if runtime_equal else 'C'
        strata.append(dict(document_index=doc, stratum=stratum, historical_GPU_UUID=h['identity']['GPU_UUID'], new_GPU_UUID=identity['GPU_UUID'], same_physical_GPU=device_equal, qualified_runtime_identity_match=runtime_equal, content_contract_audit_SHA256=sha(ROOT/'analysis/POSTRUN_SOURCE_VERIFICATION.json'), qualification_receipt_SHA256=sha(ROOT/'GPU_QUALIFICATION.json'), scope='same-device exploratory; no32K forward bound' if stratum=='A' else 'cross-device descriptive; no32K forward bound' if stratum=='B' else 'excluded from valid interaction'))
        if stratum != 'C':
            interaction = result['Delta'] - h['Delta']
            bound = result['Delta_resolution'] + h['Delta_resolution'] + R.gamma64(16)*(1+abs(result['Delta'])+abs(h['Delta']))
            temporal=h['time_mean_subtraction_roundoff']+result['time_mean_subtraction_roundoff']
            interactions.append(dict(document_index=doc, stratum=stratum, K_H_C128=h['K_A'], K_H_DualM=h['K_D'], K_C6R_C128=result['K_A'], K_C6R_DualM=result['K_D'], Delta_H=h['Delta'], Delta_C6R=result['Delta'], Interaction=interaction, Interaction_arithmetic_resolution=bound, Interaction_direction='POSITIVE' if interaction>bound else 'NEGATIVE' if interaction < -bound else 'NUMERICALLY_UNRESOLVED', LearnedGain_C128=h['K_A']-result['K_A'], LearnedGain_DualM=h['K_D']-result['K_D'], LearnedGain_C128_arithmetic_resolution=h['K_A_arithmetic_error']+result['K_A_arithmetic_error']+temporal+R.gamma64(16)*(1+abs(h['K_A'])+abs(result['K_A'])), LearnedGain_DualM_arithmetic_resolution=h['K_D_arithmetic_error']+result['K_D_arithmetic_error']+temporal+R.gamma64(16)*(1+abs(h['K_D'])+abs(result['K_D']))))
        for w, z in result['windows'].items():
            windows.append(dict(document_index=doc, window=w, **{k:normalized_winner(z)[k] for k in ('K_A','K_D','Delta','RelativeGain','Winner','Delta_resolution','score_count')}))
        print(json.dumps(dict(document_index=doc, status='CPU_AUDITED', score_points=768)), flush=True)
        del fp, arms
        gc.collect()
    scratch.rmdir()
    counts = {name:sum(d['Winner']==name for d in paired) for name in ('D_WIN','C_WIN','NUMERICALLY_UNRESOLVED')}
    complete = len(paired)==5 and not missing and not (ROOT/'FIRST_FAILURE.json').exists()
    primary = 'NOT_RUN' if not paired else 'INCOMPLETE' if not complete else 'POSITIVE_WITHIN_C6R' if counts['D_WIN']==5 else 'NEGATIVE_WITHIN_C6R' if counts['C_WIN']==5 else 'MIXED_OR_UNRESOLVED'
    sensitivity = dict(leave_one_out=[dict(omitted_document_index=i, **descriptive([d for d in paired if d['document_index']!=i])) for i in range(1,6)] if complete else [], excluding_historical_Doc1=descriptive([d for d in paired if d['document_index']!=1]) if complete else dict(status='NOT_RUN_INCOMPLETE_PRIMARY'), Interaction_by_stratum={s:interaction_description([d for d in interactions if d['stratum']==s]) for s in ('A','B')}, Interaction_all_descriptive=interaction_description(interactions) if complete else dict(status='NOT_RUN_INCOMPLETE'), Interaction_leave_one_out=[dict(omitted_document_index=i,**interaction_description([d for d in interactions if d['document_index']!=i])) for i in range(1,6)] if complete else [], Interaction_excluding_Doc1=interaction_description([d for d in interactions if d['document_index']!=1]) if complete else dict(status='NOT_RUN_INCOMPLETE'))
    science = dict(TASK=ROOT.name, status='COMPLETE' if complete else 'INCOMPLETE_OR_NOT_RUN', PRIMARY_RESULT=primary, documents=paired, missing_documents=missing, direction_counts=counts, summary=descriptive(paired) if complete else dict(status='NOT_RUN_COMPLETE_N5_REQUIRED'), sensitivity=sensitivity, statistical_unit='document', N_fixed=5, PROJECT_INDEPENDENCE='UNKNOWN', analysis_design='POSTHOC_ON_HISTORICALLY_OBSERVED_DOCUMENTS', no_confirmatory_significance=True, historical_two_doc_pilot_pooled=False, conditional_arithmetic_resolution_scope=R.REDUCTION_POLICY['resolution_scope'], CPU_wall_seconds=time.time()-started, CUDA_initialized=torch.cuda.is_initialized())
    save('SCIENCE_RESULTS.json',science)
    save('INTERACTION_RESULTS.json',dict(status='COMPLETE_EXPLORATORY' if complete and len(interactions)==5 else 'INCOMPLETE_OR_NOT_RUN', documents=interactions, device_strata=strata, evidence_scope='Never pool A/B as strictly same-environment confirmatory2x2'))
    save('SCORE_ARTIFACT_VERIFICATION.json',dict(status='PASS' if paired else 'NOT_RUN', assets=assets, full_vocabulary_logit_hashes=score_index, original_reducer_SHA256=sha(ROOT/'src/reducer.py'), no_GPU=True))
    save('RUNTIME_COST.json',dict(status='MEASURED_ACTUAL_ONLY', GPU_RESOURCE_LEDGER_SHA256=sha(ROOT/'GPU_RESOURCE_LEDGER.json'), trajectories=runtime, metadata_bytes=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file() and p.suffix!='.pt'), local_QDQ_not_common_history=True))
    csv_save('SCIENCE_RESULTS.csv',[dict(document_index=d['document_index'], K_C6R_C128=d['K_A'], K_C6R_DUAL_M=d['K_D'], Delta_C6R=d['Delta'], RelativeGain_C6R=d['RelativeGain'], ArithmeticResolution=d['Delta_resolution'], Winner=d['Winner']) for d in paired])
    csv_save('INTERACTION_RESULTS.csv',interactions)
    csv_save('INTERACTION_DEVICE_STRATA.csv',strata)
    csv_save('WINDOW_RESULTS.csv',windows)


if __name__ == '__main__':
    main()
