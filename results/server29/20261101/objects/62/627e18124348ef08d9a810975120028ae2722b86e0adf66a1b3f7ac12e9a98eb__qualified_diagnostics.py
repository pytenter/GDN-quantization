"""CPU delivery report from actual final receipts only; never launches GPU work.

Run after postprocess_scores.py and verified release of every GPU reservation.
--self-test checks synthetic control branches in memory and writes no outputs.
This independent analysis file is outside the frozen GPU execution manifest.
"""
import argparse
import collections
import hashlib
import json
import math
import pathlib
import statistics
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
TASK = 'QWEN_GDN_HADAMARD_DUAL_AXIS_DIRECT_32K_PILOT_V1'
CONDITIONS = ('FP_NATIVE', 'H_C128', 'H_DUAL_M')
REQUIRED_FIELDS = (
    'TASK', 'GPU_AUTHORIZATION', 'SCIENTIFIC_PROTOCOL', 'OLD_FP_NATIVE_H_FAILURE',
    'CPU_V1_SEAL', 'HADAMARD_STRUCTURAL_CORRECTNESS', 'CANONICAL_H_C128_PARITY',
    'DUAL_M_GPU_PARITY', 'PERSISTENT_QDQ', 'FRESH_PROCESS_DETERMINISM',
    'NUMERICAL_SAFETY', 'RESOURCE_ADMISSION', 'DOCS_FROZEN', 'FP_NATIVE_32K',
    'H_C128_32K', 'H_DUAL_M_32K', 'VALID_PAIRED_DOCS',
    'PRIMARY_384_POSITION_CHECK', 'DUAL_M_FULLMODEL_RESULT', 'GPU_HOURS_USED',
    'MAX_GPU_HOURS', 'PHYSICAL_INT8_STORAGE', 'AIME', 'C6R_INTERACTION',
    'C8_TRAINING', 'FINAL_EXECUTION_STATUS',
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(pathlib.Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def optional(root, relative):
    p = root / relative
    return read(p) if p.exists() else None


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def complete_run(run, tokens=128):
    return bool(run and run.get('status') == 'PASS' and run.get('consumed_tokens') == tokens
                and len(run.get('performance', [])) == tokens)


def comparable(run, helpers=False):
    rows = run.get('rows', [])
    require(len(rows) == 128 and [r['t'] for r in rows] == list(range(128)),
            'canary record is missing exact128 sequential token records')
    keys = ('t', 'input_id', 'logits', 'cache', 'cache_cursor')
    if helpers:
        keys += ('helper', 'QDQ')
    return [{k: row[k] for k in keys} for row in rows]


def exact_pair(a, b, helpers=False):
    if not (complete_run(a) and complete_run(b)):
        return 'NOT_RUN'
    return 'PASS' if comparable(a, helpers) == comparable(b, helpers) else 'FAIL'


def official_dual_parity(run):
    if not complete_run(run):
        return 'NOT_RUN'
    token0 = [x for x in run.get('mechanism', []) if x.get('token_index') == 0]
    if len(token0) != 1 or not isinstance(token0[0].get('stats'), dict):
        return 'NOT_RUN'
    stats = token0[0]['stats']
    return 'PASS' if (stats.get('official_exact') == 'PASS_TORCH_EQUAL_CODES_SCALES_DEQUANT'
                      and stats.get('batched_vs_individual') == 'BYTE_EXACT') else 'FAIL'


def qdq_counts(run):
    if not run or run.get('condition') not in ('H_C128', 'H_DUAL_M'):
        return None
    n = run.get('consumed_tokens')
    valid = (type(n) is int and n > 0 and run.get('writebacks') == n * 24
             and run.get('next_token_consumption_checks') == (n-1) * 24)
    return dict(status='PASS' if valid else 'FAIL', consumed_tokens=n,
                writebacks=run.get('writebacks'), next_token_consumption_checks=run.get('next_token_consumption_checks'))


def classification(documents):
    if not documents:
        return 'INCONCLUSIVE'
    if len(documents) == 1:
        return 'PARTIAL'
    require(len(documents) == 2, 'scientific target must remain two documents')
    winners = [d['Winner'] for d in documents]
    if winners == ['H_DUAL_M'] * 2:
        return 'POSITIVE_EXPLORATORY'
    if winners == ['H_C128'] * 2:
        return 'NEGATIVE_EXPLORATORY'
    if set(winners) == {'H_DUAL_M', 'H_C128'} or ('H_DUAL_M' in winners and 'NUMERICALLY_UNRESOLVED' in winners):
        return 'MIXED_EXPLORATORY'
    return 'INCONCLUSIVE'


def final_execution_status(coordinator_status, paired_count):
    if paired_count == 2 and coordinator_status == 'COMPLETE_AND_STOP':
        return 'COMPLETE_AND_STOP'
    if paired_count == 1:
        return 'PARTIAL_STOP'
    if coordinator_status in ('FAIL_AND_STOP', 'BLOCKED_AND_STOP', 'PARTIAL_STOP'):
        return coordinator_status
    return 'PARTIAL_STOP'


def safety_summary(runs):
    selected = []
    for tag, run in runs.items():
        n = run.get('consumed_tokens')
        if run.get('condition') != 'H_DUAL_M' or not complete_run(run, n):
            continue
        perf = run['performance']
        require(all(isinstance(p.get('scale_risk'), dict) and 'raw_row_bound_hits' in p['scale_risk'] for p in perf),
                'a completed Dual-M run lacks all-token scale risk records: ' + tag)
        keys = ('raw_row_bound_hits', 'raw_column_bound_hits', 'row_floor_hits', 'row_ceiling_hits',
                'column_floor_hits', 'column_ceiling_hits')
        sums = {k: sum(p['scale_risk'].get(k, 0) for p in perf) for k in keys}
        selected.append(dict(tag=tag, condition=run['condition'], doc_id=run['doc_id'], consumed_tokens=n,
            **sums, max_affected_energy_fraction=max(p['scale_risk']['max_affected_energy_fraction'] for p in perf),
            max_affected_relative_mse=max(p['scale_risk']['max_affected_relative_mse'] for p in perf),
            per_layer_head_severe_events=0, evidence='Actual completed worker trajectory; frozen per-layer/head safety check executed on every token'))
    if not selected:
        return None
    return dict(status='PASS', scope='Completed Dual-M trajectories listed here only; no claim about unexecuted tokens or documents',
        trajectories=selected, completed_trajectories=len(selected), consumed_tokens_total=sum(x['consumed_tokens'] for x in selected),
        LOW=2**-24, HIGH=65504.0, row_mean_floor=1e-20, LS_den_floor=1e-30,
        affected_mask='Union of every raw preclamp row and initial/final column outside LOW/HIGH, intersect original S!=0',
        unit='doc/token/layer/head; no cross-layer dilution',
        affected_energy_fraction='FP64 sum(S^2 on affected mask)/sum(S^2); zero total energy maps to0',
        affected_relative_mse='FP64 sum((decode-S)^2 on mask)/sum(S^2 on mask); zero affected energy maps to0',
        severe_stop='affected_energy_fraction >=1e-4 AND affected_relative_MSE >1e-2',
        maximum_warning='The two reported maxima need not belong to the same head; they are not combined into a new risk criterion',
        GPU_reconstruction_scale_dtype='FP16 rows and FP16 columns; no changed floor/dtype/iteration')


def timing_summary(tag, run):
    perf = run.get('performance', [])
    n = run.get('consumed_tokens')
    require(complete_run(run, n), 'timing source must be a complete actual trajectory')
    def stats(values):
        values = [float(v) for v in values if number(v)]
        if not values:
            return None
        return dict(count=len(values), mean=math.fsum(values)/len(values), median=statistics.median(values),
                    min=min(values), max=max(values), p95=sorted(values)[math.ceil(len(values)*0.95)-1])
    warmed = perf[16:] if len(perf) > 16 else perf
    times = stats([p['full_model_plus_QDQ_seconds'] for p in warmed])
    codec = {name: stats([p.get('codec', {}).get(name) for p in perf])
             for name in ('encode_ms', 'decode_ms', 'writeback_ms', 'safety_and_metric_audit_ms')}
    layer_sums = collections.defaultdict(lambda: collections.defaultdict(float))
    metrics = []
    for m in run.get('mechanism', []):
        raw = m.get('stats')
        rows = raw if isinstance(raw, list) else [raw] if isinstance(raw, dict) else []
        for row in rows:
            metrics.append(row)
            for key in ('encode_ms', 'decode_and_finite_check_ms', 'writeback_ms'):
                if number(row.get(key)):
                    layer_sums[m['token_index']][key] += row[key]
    mechanism = {}
    for field in ('relative_local_QDQ_mse', 'scale_min', 'scale_max', 'effective_scale_min', 'effective_scale_max',
                  'rounded_clamp_fraction', 'unrounded_over127_fraction', 'zero_code_fraction', 'saturation_fraction'):
        s = stats([m.get(field) for m in metrics])
        if s is not None:
            mechanism[field] = s
    return dict(tag=tag, condition=run['condition'], doc_id=run['doc_id'], consumed_tokens=n,
        scope='SCIENCE_FULL32768' if n == 32768 else 'CANARY_ONLY', wall_seconds=run.get('wall_seconds'),
        wall_tokens_per_second=n/run['wall_seconds'] if number(run.get('wall_seconds')) and run['wall_seconds'] > 0 else None,
        postwarm_full_model_plus_QDQ_seconds=times,
        postwarm_tokens_per_second=1/times['median'] if times and times['median'] > 0 else None,
        forward_seconds=stats([p.get('forward_seconds') for p in perf]),
        qdq_seconds=stats([p.get('qdq_seconds') for p in perf]), qdq_gpu_ms=stats([p.get('qdq_gpu_ms') for p in perf]),
        batched_DualM_per_token_cuda_event_ms=codec,
        sampled_C128_24_layer_sum_cuda_event_ms={k: stats([v[k] for v in layer_sums.values() if k in v])
            for k in ('encode_ms', 'decode_and_finite_check_ms', 'writeback_ms')},
        peak_allocated_bytes=run.get('peak_allocated'), peak_reserved_bytes=run.get('peak_reserved'),
        checkpoints=run.get('checkpoints', []), mechanism_summary=mechanism,
        mechanism_interpretation='Local QDQ error uses each student own pre-QDQ state; A/D inputs differ. No matched FP_H state capture; cannot substitute for paired fullmodel Future-KL or shared-input state MSE')


def verify_inputs(root):
    manifest = read(root/'source_manifest.json'); prereg = read(root/'preregistration.json')
    require(manifest['task'] == prereg['TASK_ID'] == TASK and manifest['frozen_before_any_new_GPU_forward'] is True,
            'wrong frozen prospective task')
    require(prereg['GPU_EXECUTION_AUTHORIZED'] == 'YES' and prereg['SCIENTIFIC_PROTOCOL'] == 'NEW_PROSPECTIVE_CODEC_COMPARISON',
            'new prospective GPU authorization missing')
    for item in manifest['execution_sources']:
        p = (root/item['path']).resolve()
        require(p.is_relative_to(root.resolve()) and p.is_file(), 'frozen source path invalid')
        require(p.stat().st_size == item['bytes'] and sha(p) == item['sha256'], 'frozen source changed: ' + item['path'])
    docs = read(root/'frozen_documents.json')
    require(sha(root/'frozen_documents.json') == prereg['documents_manifest_sha256'], 'frozen documents changed')
    require(docs.get('DOCS_FROZEN') == len(docs.get('selected_documents', [])) == 2, 'two documents not frozen')
    structure = read(root/'qualification_cpu/CPU_STRUCTURE_001.json')
    model = read(root/'qualification_cpu/MODEL_CONTENT_IDENTITY.json')
    require(structure.get('status') == model.get('status') == 'PASS', 'mandatory CPU identities did not pass')
    require(sha(root/'qualification_cpu/MODEL_CONTENT_IDENTITY.json') == manifest['model_identity_receipt_sha256'],
            'model identity receipt changed')
    gate = read(root/'GATE_STATUS_FINAL.json'); ledger = read(root/'gpu_resource_ledger.json')
    post = read(root/'analysis/POSTPROCESS_STATUS.json')
    require(gate['task_id'] == ledger['task_id'] == post['TASK'] == TASK, 'final task identity mismatch')
    require(gate.get('no_further_GPU_work') is True and gate.get('GPU_release_verified') is True
            and ledger.get('final_GPU_release_verified') is True and all(p.get('released') is True for p in ledger['phases']),
            'report forbidden while any GPU reservation lacks verified release')
    require(gate['GPU_HOURS_USED'] == ledger['GPU_HOURS_USED'] == post['GPU_HOURS_USED']
            and 0 <= ledger['GPU_HOURS_USED'] <= 24.0, 'resource accounting mismatch or cap exceeded')
    require(ledger.get('final_status')==gate['status']==post.get('final_GPU_execution_status')
            and ledger.get('GPU_UUID')==prereg['G1']['UUID'], 'final execution/GPU identity mismatch')
    require(post.get('CPU_ONLY') is True and post.get('CUDA_initialized') is False
            and post.get('source_manifest_sha256') == sha(root/'source_manifest.json'), 'CPU score postprocess receipt absent/stale')
    require(prereg['old_failure']['classification'] == 'FAIL_AND_STOP' and prereg['old_failure']['new_gate'] is False,
            'historical failure may not be reclassified')
    return manifest, prereg, docs, structure, model, gate, ledger, post


def load_actual_runs(root, gate, ledger):
    runs = {}; available = {}
    for tag in ('N_OFF','A_REFERENCE','A_OFF','A_ON','D_OFF','D_ON','N_FRESH','A_FRESH','D_FRESH'):
        value = optional(root, 'analysis/'+tag+'.json')
        if value:
            require(value.get('tag') == tag and complete_run(value), 'invalid actual canary receipt: ' + tag)
            runs[tag] = value
    for doc in (1, 2):
        for condition in CONDITIONS:
            phase = f'science_doc{doc}_{condition}'
            available[phase] = False
            if gate['gates'].get(phase) != 'PASS':
                continue
            receipt_path = root/f'analysis/RECEIPT_{phase}.json'
            receipt = read(receipt_path); result = read(root/f'analysis/{phase}.json')
            entries = [p for p in ledger['phases'] if p['phase'] == phase]
            require(len(entries) == 1 and entries[0].get('returncode') == 0 and entries[0].get('released') is True
                    and entries[0].get('receipt_sha256') == sha(receipt_path), 'completed phase reservation/receipt mismatch')
            require(receipt.get('status') == receipt.get('result', {}).get('status') == 'PASS'
                    and receipt['result']['consumed'] == 32768 and receipt['result']['scores'] == 384
                    and complete_run(result, 32768) and result['condition'] == condition and result['tag'] == phase,
                    'invalid complete32768/384 phase receipt')
            available[phase] = True; runs[phase] = result
    return runs, available


def checked_science(root, post, available):
    results = optional(root, 'science_results.json')
    docs = [] if results is None else results.get('documents', [])
    require(len(docs) == post.get('complete_paired_documents'), 'postprocess complete-document count mismatch')
    if not docs:
        require(results is None and post.get('status') == 'NOT_RUN'
                and post.get('scientific_outputs_created') is False, 'no-pair scientific output must remain NOT_RUN')
        return None, [], 'NOT_RUN'
    require(results['status'] in ('COMPLETE','PARTIAL') and results['source_manifest_sha256'] == post['source_manifest_sha256'],
            'scientific analysis source/status identity mismatch')
    require(results['status']==('COMPLETE' if len(docs)==2 else 'PARTIAL')
            and (len(docs)==2 or results.get('two_document_means') is None),
            'partial results must not contain two-document means or a complete status')
    evidence = read(root/'analysis/SCORE_ARTIFACT_VERIFICATION.json')
    require(evidence.get('status') == 'PASS' and evidence.get('full384_position_check') == 'PASS'
            and evidence.get('all_actual_full_vocab_FP32_logits_finite') is True, '384 full-vocab score qualification absent')
    indices = set()
    for d in docs:
        i = d['document_index']; require(i in (1,2) and i not in indices, 'invalid duplicate scientific document'); indices.add(i)
        require(all(available[f'science_doc{i}_{c}'] for c in CONDITIONS), 'scientific document lacks three full trajectories')
        require(d.get('complete') is True and d.get('score_count') == 384
                and d['K_A'] == d['sum384_A']/384 and d['K_D'] == d['sum384_D']/384
                and d['Delta'] == d['K_A']-d['K_D'], 'Primary is not actual exact fsum384/384')
        require(all(number(d[x]) for x in ('K_A','K_D','Delta','Delta_resolution')) and d['Delta_resolution'] >= 0,
                'nonfinite scientific Primary/resolution')
        winner = 'H_DUAL_M' if d['Delta'] > d['Delta_resolution'] else 'H_C128' if d['Delta'] < -d['Delta_resolution'] else 'NUMERICALLY_UNRESOLVED'
        require(d['Winner'] == winner, 'winner did not honor actual new arithmetic resolution')
        require(set(d['windows']) == {'Early','Middle','Late'} and all(w['score_count'] == 128 for w in d['windows'].values()),
                'three128-point windows absent')
        a=read(root/f'analysis/score_rows_doc{i}_A.json')
        other=read(root/f'analysis/score_rows_doc{i}_D.json')
        audit_pair_rows(a,other,d)
    if len(docs)==2:
        means=results.get('two_document_means');require(isinstance(means,dict),'complete two-document means absent')
        require(means['K_A']==math.fsum(d['K_A'] for d in docs)/2
                and means['K_D']==math.fsum(d['K_D'] for d in docs)/2
                and means['mean_absolute_Delta']==math.fsum(d['Delta'] for d in docs)/2,
                'reported two-document absolute means differ from actual document values')
        gains=[d['RelativeGain'] for d in docs]
        expected_gain=math.fsum(gains)/2 if all(g is not None for g in gains) else None
        expected_ratio=(math.fsum(d['Delta'] for d in docs)/math.fsum(d['K_A'] for d in docs)) if all(g is not None for g in gains) else None
        require(means['mean_per_doc_RelativeGain']==expected_gain and means['relative_gain_of_mean_K']==expected_ratio,
                'reported relative means differ from frozen denominator rules')
    require(classification(docs) == results['DUAL_M_FULLMODEL_RESULT'] == post['DUAL_M_FULLMODEL_RESULT'], 'scientific classification mismatch')
    return results, docs, 'PASS'


def recompute_pair_arithmetic(a,d):
    """Independent stdlib audit of the frozen reducer's time/uncertainty math.

    Full-vocabulary scalar KL was already independently recomputed by the
    postprocessor. This function checks its saved raw per-position observations,
    aggregation, arithmetic resolution, denominator and direction again.
    """
    n=len(a);require(n==len(d) and n>0,'paired arithmetic needs nonempty equal rows')
    ka=math.fsum(r['KL_FP32'] for r in a)/n;kd=math.fsum(r['KL_FP32'] for r in d)/n
    ea=math.fsum(r['per_arm_arithmetic_error'] for r in a)/n
    ed=math.fsum(r['per_arm_arithmetic_error'] for r in d)/n
    nu=(8*n+32)*2.0**-53;gamma=nu/(1.0-nu)
    temporal=gamma*(1.0+math.fsum(abs(r['KL_FP32']) for r in a+d)/n+ea+ed)
    resolution=ea+ed+temporal;delta=ka-kd
    return dict(K_A=ka,K_D=kd,Delta=delta,RelativeGain=delta/ka if ka>ea+temporal else None,
                Winner='H_DUAL_M' if delta>resolution else 'H_C128' if delta<-resolution else 'NUMERICALLY_UNRESOLVED',
                Delta_resolution=resolution,K_A_arithmetic_error=ea,K_D_arithmetic_error=ed,
                time_mean_subtraction_roundoff=temporal,score_count=n)


def audit_pair_rows(a,d,summary):
    positions=list(8192+64*k for k in range(384))
    require(len(a)==len(d)==384 and [r['token_index'] for r in a]==[r['token_index'] for r in d]==positions,
            'saved paired raw rows do not contain exactly384 increasing frozen positions')
    for ra,rd in zip(a,d):
        for key in ('identity','token_index','consumed_tokens','predicted_token_index','next_token_id',
                    'full_vocabulary_size','reference_logits_sha256','reducer_policy_sha256'):
            require(ra[key]==rd[key],'saved raw row reference/input identity mismatch: '+key)
        for r in (ra,rd):
            require(r['consumed_tokens']==r['predicted_token_index']==r['token_index']+1,
                    'saved raw row cursor is off by one')
            require(r['identity']==summary['identity'],'saved raw row and document identity differ')
            require(all(number(r[k]) for k in ('KL_FP32','KL_FP64','FP64_reference_roundoff','FP32_vs_FP64_abs_error','per_arm_arithmetic_error')),
                    'nonfinite saved per-position arithmetic audit')
            require(r['FP64_reference_roundoff']>=0 and r['FP32_vs_FP64_abs_error']==abs(r['KL_FP32']-r['KL_FP64'])
                    and r['per_arm_arithmetic_error']==r['FP32_vs_FP64_abs_error']+r['FP64_reference_roundoff'],
                    'saved per-position error differs from frozen resolution rule')
    calculated=recompute_pair_arithmetic(a,d)
    require(all(summary[k]==v for k,v in calculated.items()),'Primary/time mean/resolution differs from saved raw score rows')
    require(summary['sum384_A']==math.fsum(r['KL_FP32'] for r in a)
            and summary['sum384_D']==math.fsum(r['KL_FP32'] for r in d),'saved score sums differ from384 raw observations')
    for name,lo,hi in (('Early',8192,16384),('Middle',16384,24576),('Late',24576,32768)):
        aw=[r for r in a if lo<=r['token_index']<hi];dw=[r for r in d if lo<=r['token_index']<hi]
        calculated=recompute_pair_arithmetic(aw,dw)
        require(len(aw)==len(dw)==128 and all(summary['windows'][name][k]==v for k,v in calculated.items()),
                'window mean/resolution differs from its128 saved raw score rows: '+name)


def derive_status(prereg, docs_manifest, structure, gate, ledger, runs, available, scientific_docs, primary, failure, scale):
    canonical = exact_pair(runs.get('A_REFERENCE'), runs.get('A_ON'), True)
    observer_a = exact_pair(runs.get('A_OFF'), runs.get('A_ON'))
    observer_d = exact_pair(runs.get('D_OFF'), runs.get('D_ON'))
    dual = official_dual_parity(runs.get('D_ON'))
    fresh_pairs = {k:exact_pair(runs.get(x),runs.get(y),h) for k,x,y,h in (
        ('FP_NATIVE','N_OFF','N_FRESH',False), ('H_C128','A_ON','A_FRESH',True), ('H_DUAL_M','D_ON','D_FRESH',True))}
    fresh = 'PASS' if all(v == 'PASS' for v in fresh_pairs.values()) else 'FAIL' if 'FAIL' in fresh_pairs.values() else 'NOT_RUN'
    checks = {tag: qdq_counts(r) for tag,r in runs.items() if r['condition'] in ('H_C128','H_DUAL_M')}
    qdq = ('FAIL' if any(v and v['status']=='FAIL' for v in checks.values()) or observer_a=='FAIL' or observer_d=='FAIL'
           else 'PASS' if canonical == dual == observer_a == observer_d == 'PASS' and all(checks[t]['status']=='PASS' for t in ('A_ON','D_ON')) else 'NOT_RUN')
    failure_text = json.dumps(failure or {}).lower()
    if fresh=='NOT_RUN' and 'same gpu fresh-process exact parity failed' in failure_text:
        fresh='FAIL'
        failed_condition=(failure or {}).get('context',{}).get('condition')
        if failed_condition in fresh_pairs:fresh_pairs[failed_condition]='FAIL'
    numeric = 'FAIL' if (scale and scale.get('status') == 'FAIL') or any(x in failure_text for x in ('nonfinite','scale risk stop','state dtype/shape','reducer fp64 crosscheck failed')) else 'PASS' if scale and scale.get('status')=='PASS' else 'NOT_RUN'
    if canonical == 'NOT_RUN' and any(x in failure_text for x in ('original h/c128','c128 calls/writebacks')): canonical='FAIL'
    if dual == 'NOT_RUN' and any(x in failure_text for x in ('official all8/ones gpu parity','batched layer encode differs')): dual='FAIL'
    if qdq == 'NOT_RUN' and any(x in failure_text for x in ('next token did not consume qdq','qdq bytes changed','not exactly one gdn','missing gdn calls')): qdq='FAIL'
    counts = {c:sum(bool(available.get(f'science_doc{i}_{c}')) for i in (1,2)) for c in CONDITIONS}
    status = dict(TASK=TASK,GPU_AUTHORIZATION='YES',SCIENTIFIC_PROTOCOL='NEW_PROSPECTIVE_CODEC_COMPARISON',
        OLD_FP_NATIVE_H_FAILURE='PRESERVED',CPU_V1_SEAL='PASS' if structure.get('seals',{}).get('inherited_cpu_v1',{}).get('status')=='PASS_UNCHANGED' else 'FAIL',
        HADAMARD_STRUCTURAL_CORRECTNESS='PASS' if structure.get('fixed_tensor_structure',{}).get('status')=='PASS' else 'FAIL',
        CANONICAL_H_C128_PARITY=canonical,DUAL_M_GPU_PARITY=dual,PERSISTENT_QDQ=qdq,FRESH_PROCESS_DETERMINISM=fresh,
        NUMERICAL_SAFETY=numeric,RESOURCE_ADMISSION='NOT_RUN',DOCS_FROZEN=docs_manifest['DOCS_FROZEN'],
        FP_NATIVE_32K=f'{counts["FP_NATIVE"]}/2',H_C128_32K=f'{counts["H_C128"]}/2',H_DUAL_M_32K=f'{counts["H_DUAL_M"]}/2',
        VALID_PAIRED_DOCS=f'{len(scientific_docs)}/2',PRIMARY_384_POSITION_CHECK=primary,DUAL_M_FULLMODEL_RESULT=classification(scientific_docs),
        GPU_HOURS_USED=ledger['GPU_HOURS_USED'],MAX_GPU_HOURS=24.0,
        PHYSICAL_INT8_STORAGE='NOT_QUALIFIED' if any(r['condition'] in ('H_C128','H_DUAL_M') for r in runs.values()) else 'NOT_RUN',
        AIME='NOT_RUN',C6R_INTERACTION='NOT_RUN',C8_TRAINING='NOT_AUTHORIZED',
        FINAL_EXECUTION_STATUS=final_execution_status(gate['status'],len(scientific_docs)))
    require(tuple(status) == REQUIRED_FIELDS, 'all26 final user fields must appear exactly once and in order')
    return status, dict(canonical_H_C128_original_reference=canonical,observer_OFF_ON=dict(H_C128=observer_a,H_DUAL_M=observer_d),
        official_DualM_all8_ones=dual,fresh_process_pairs=fresh_pairs,per_completed_student_QDQ_counts=checks,
        official_DualM_token0_observations=[m['stats'] for m in runs.get('D_ON',{}).get('mechanism',[]) if m.get('token_index')==0],
        official_comparison_semantics='Original torch.equal codes/scales/dequant; observed signed-zero differences are disclosed in token0 observations, not silently called byte-exact',
        checkpoint_restore64=[x for x in runs.get('A_ON',{}).get('checkpoints',[]) if x.get('cursor')==64 and x.get('restored_exact') is True])


def fmt(value):
    if value is None:
        return 'N/A'
    if isinstance(value, float):
        return f'{value:.12g}'
    return str(value)


def render_report(root, status, prereg, docs_manifest, gate, ledger, post, results, scientific_docs,
                  parity, runtime, numeric, scale, failure, input_hashes):
    lines = [f'# {TASK} 科学报告', '', f'最终执行状态：**{status["FINAL_EXECUTION_STATUS"]}**；科学分类：**{status["DUAL_M_FULLMODEL_RESULT"]}**。',
        f'实际完整配对文档 {status["VALID_PAIRED_DOCS"]}，本任务累计占用 {fmt(ledger["GPU_HOURS_USED"])} GPUh / 24 GPUh；最终GPU释放已由协调器和资源账本核验。',
        '', '本轮是独立前瞻性codec替换探索，比较共同实际FP_NATIVE目标下的H_C128与H_DUAL_M。旧execution_02仍为FAIL_AND_STOP：FP_NATIVE/FP_H relative-L2=0.006718461729707917，旧上限0.002；本次没有解决或重新判定该失败。',
        '', '两篇文档在新GPU forward前冻结，统计单位为文档，目标N=2。repository/project独立性UNKNOWN；不同raw SHA不构成项目级独立性证明。两种学生都采用原正式FP32 tensor承载QDQ值：逐token真实量化写回语义与物理packed INT8资格分别报告，PHYSICAL_INT8_RUNTIME=NOT_QUALIFIED。',
        '', '## 实际科学结果', '']
    if scientific_docs:
        lines += ['| Doc | K_A | K_D | Delta | Relative Gain | 新算术分辨率 | Winner |',
                  '|---|---:|---:|---:|---:|---:|---|']
        for d in scientific_docs:
            lines.append('| '+ ' | '.join(fmt(v) for v in (d['document_index'],d['K_A'],d['K_D'],d['Delta'],d['RelativeGain'],d['Delta_resolution'],d['Winner']))+' |')
        lines += ['', 'K_A、K_D均由同篇384个真实位置的full-vocabulary FP32 KL通过固定math.fsum/384得到。Delta=K_A-K_D；RelativeGain=Delta/K_A仅在分母高于其合法算术分辨范围时提供。', '',
                  '| Doc | 窗口 | K_A | K_D | Delta | Relative Gain | 窗口算术分辨率 | Winner |', '|---|---|---:|---:|---:|---:|---:|---|']
        for d in scientific_docs:
            for name in ('Early','Middle','Late'):
                w=d['windows'][name]
                lines.append('| '+' | '.join(fmt(v) for v in (d['document_index'],name,w['K_A'],w['K_D'],w['Delta'],w['RelativeGain'],w['Delta_resolution'],w['Winner']))+' |')
    else:
        lines += ['正式32K Future-KL：**NOT_RUN**。没有文档同时具备三条合格完整32768轨迹与384个真实配对评分位置；canary诊断不填入正式科学表。没有将未运行或缺失值填0。']
    lines += ['', '新数值分辨规则事前固定：每个保存logit对的FP32 KL与独立标量FP64参考之差，加条件FP64算术界，再按两臂误差的fsum/384及均值/相减舍入界汇总；未复用NoInflation的旧固定分辨率。该界依赖IEEE64/libm假设，约束reducer算术，不证明32K模型forward的完整数值不确定性；fresh-process资格限于真实128-token重复。小于分辨范围记NUMERICALLY_UNRESOLVED。',
        '', '## 用户要求的八项回答', '']
    n=len(scientific_docs); classification_name=status['DUAL_M_FULLMODEL_RESULT']
    if n==2:
        means=results.get('two_document_means') or {}
        q1={'POSITIVE_EXPLORATORY':'两篇均出现超过本轮算术分辨率的正向Delta，支持这两篇受测条件下Dual-M更优的探索性判断。',
            'NEGATIVE_EXPLORATORY':'两篇均出现超过分辨率的负向Delta，当前受测Hadamard+Dual-M在两篇上均劣于C128。',
            'MIXED_EXPLORATORY':'两篇方向不同，或一篇正向、一篇未分辨，不能给出一致优于C128的结论。',
            'INCONCLUSIVE':'真实差异未形成一致可分辨方向，当前结果不能建立Dual-M优于C128。'}[classification_name]
        q2=f'等权文档平均绝对Delta={fmt(means.get("mean_absolute_Delta"))}；等权每doc RelativeGain平均={fmt(means.get("mean_per_doc_RelativeGain"))}；均值KL的相对改善={fmt(means.get("relative_gain_of_mean_K"))}。二者分母不同，均保留每doc原值。'
    elif n==1:
        q1='仅一篇完成真实配对，结果为PARTIAL；不能宣称两文档科学结论。'
        q2='两篇平均绝对与相对改善尚不能计算；唯一完成文档的真实值见上表。'
    else:
        q1='没有完整正式配对，无法回答两篇32K上是否优于C128；INCONCLUSIVE。'
        q2='NOT_RUN：没有正式384点Primary，不计算或外推两篇平均收益。'
    if scientific_docs:
        parts=[]
        for d in scientific_docs:
            order=sorted(d['windows'],key=lambda k:d['windows'][k]['Delta'],reverse=True)
            parts.append(f'Doc{d["document_index"]}的最大原始Delta窗口为{order[0]}，其winner={d["windows"][order[0]]["Winner"]}')
        q3='；'.join(parts)+'。这只是各128点窗口描述，整篇384点Primary决定文档方向；不将Late或改善位置比例替代Primary。'
        if len(scientific_docs)==2:
            shared_d=[w for w in ('Early','Middle','Late') if all(d['windows'][w]['Winner']=='H_DUAL_M' for d in scientific_docs)]
            shared_a=[w for w in ('Early','Middle','Late') if all(d['windows'][w]['Winner']=='H_C128' for d in scientific_docs)]
            q3+=' 两篇一致可分辨支持D的窗口：'+(', '.join(shared_d) if shared_d else '无')+'；一致支持A的窗口：'+(', '.join(shared_a) if shared_a else '无')+'。其余窗口为方向不一致或含未分辨。'
        else:q3+=' 只有一篇完成，不能判断两篇窗口方向是否一致。'
    else:q3='NOT_RUN：Early/Middle/Late各128点窗口尚无完整正式配对；不沿用旧NoInflation的Late发现。'
    if scale and scale.get('status')=='FAIL':
        q4='真实触发冻结scale安全STOP，原始scale_safety.json保持不变。具体doc/token/layer/head与受影响能量、误差见该首个风险证据；未以改FP16 dtype、floor或阈值补救。'
    elif scale:
        q4=f'已完成的{scale["completed_trajectories"]}条Dual-M轨迹、合计{scale["consumed_tokens_total"]} token未触发逐layer/head严重scale规则；有效scale、clipping与越界的实际范围见runtime_cost.json。该结论只覆盖列明已完成轨迹，不能证明未执行32K的安全。'
    else:q4='Dual-M未完成可汇总的真实轨迹，scale风险状态NOT_RUN；不能用合成CPU数据宣称真实模型安全。'
    if runtime.get('G4') and runtime['G4'].get('status')!='PASS':
        q5='实际吞吐/显存的保守G4预测不满足本任务24GPUh或24GB门禁，当前预算下不能接受两篇完整32K执行，因此未以额外GPU或缩短科学长度绕过。'
    elif runtime.get('trajectories'):
        q5='已记录实际整轨迹、逐token forward/QDQ与codec CUDA事件成本及峰值显存，见下方资源与runtime_cost.json。G4通过仅说明保守预测可在预算内执行；没有预注册部署延迟阈值，不能据此宣称生产在线开销可接受。'
        if runtime.get('matched_runtime_ratios'):
            q5+=' 同文档实际D/A post-warmup整token中位耗时比：'+ '；'.join(r['scope']+'='+fmt(r['D_over_A_median_full_token_seconds']) for r in runtime['matched_runtime_ratios'])+'。'
    else:q5='真实在线开销NOT_RUN，无法判断可接受性。'
    if n==2 and classification_name=='POSITIVE_EXPLORATORY':
        q6='值得另行预注册、独立授权的多文档验证；本任务到此STOP，不自行启动。N=2不支持显著性、广泛泛化或AIME提升。'
    else:q6='当前证据不足以据此安排有收益前提的多文档验证；只有后续独立决策与授权才可启动新实验，本任务不追加样本。'
    if n==2 and classification_name=='NEGATIVE_EXPLORATORY':
        q7='建议停止当前受测Hadamard+Dual-M配置的推进；这只约束本实现与两篇文档，不能否定所有Dual-Axis量化器。'
    elif n==2 and classification_name=='MIXED_EXPLORATORY':
        q7='应停止将当前路线当作已确认改进来推进。混合结果本身不证明普遍无效；本轮不做事后调参或无限修补。'
    else:q7='本轮按完成或失败/资源门禁到此停止。资格不足、未分辨或单篇结果不能等同于路线已被科学否定；不自动开始新attempt。'
    q8='尚未验证。没有运行C6R/C7、learned rotation交互或C8训练；当前结果不能支持互补性，也不能声称互补性已被否定。'
    questions=[('Hadamard+Dual-M是否在两篇真实32K上更优？',q1),('平均绝对与相对Future-KL改善是多少？',q2),
        ('Early/Middle/Late哪个窗口受益、是否一致？',q3),('真实scale、clipping和数值范围是否有新增风险？',q4),
        ('Dual-M真实在线开销是否可接受？',q5),('如改善，是否值得独立多文档验证？',q6),('混合或负面时是否应停止当前路线？',q7),('是否支持与learned rotation互补？',q8)]
    for i,(q,a) in enumerate(questions,1):lines += [f'{i}. **{q}** {a}', '']
    lines += ['## 工程门禁与资源', '', '| 项目 | 实际状态 |', '|---|---|']
    for k in ('CPU_V1_SEAL','HADAMARD_STRUCTURAL_CORRECTNESS','RESOURCE_ADMISSION','CANONICAL_H_C128_PARITY','DUAL_M_GPU_PARITY','PERSISTENT_QDQ','FRESH_PROCESS_DETERMINISM','NUMERICAL_SAFETY','PRIMARY_384_POSITION_CHECK'):
        lines.append(f'| {k} | {status[k]} |')
    lines += ['', f'协调器原始结束状态：{gate["status"]}。首个失败文件：'+('FIRST_FAILURE.json（原件不修改）' if failure else '无')+'。',
        f'GPU UUID：{ledger.get("GPU_UUID", "UNKNOWN")}；累计{fmt(ledger["GPU_HOURS_USED"])} GPUh，含加载、资格、forward、codec、IO、checkpoint与释放前占用。',
        '', '| 阶段 | 占用秒 | GPUh | 峰值NVML MiB（采样） | 释放核验 |', '|---|---:|---:|---:|---|']
    for p in ledger['phases']:
        lines.append('| '+' | '.join(fmt(v) for v in (p['phase'],p.get('reservation_seconds'),p.get('GPU_hours'),p.get('peak_sampled_NVML_MiB'),p.get('released')))+' |')
    if failure:
        lines += ['', '首个失败原始门禁/阶段：'+str(failure.get('phase','UNKNOWN'))+'；错误：'+str(failure.get('error','UNKNOWN'))+'。',
                  '首个失败context：`'+json.dumps(failure.get('context',{}),ensure_ascii=False,sort_keys=True)+'`。']
    if runtime.get('G4'):
        g4=runtime['G4']
        lines += ['',f'G4实际保守预测：status={g4.get("status")}，六条32K剩余预测={fmt(g4.get("projected_remaining_GPU_seconds"))}秒，预计峰值reserved={fmt(g4.get("projected_peak_reserved_bytes"))} bytes，可接受设备上限={fmt(g4.get("admissible_memory_bytes"))} bytes，安全余量={fmt(g4.get("safety_exit_reserve_seconds"))}秒。',
                  '预测包括post-warmup p95×1.6、每轨迹加载、四次checkpoint投影和score IO；最终reducer在释放GPU后只用CPU。']
    if runtime.get('trajectories'):
        lines += ['', '| 真实轨迹 | token数 | wall秒 | wall token/s | post-warmup token/s | peak reserved bytes |', '|---|---:|---:|---:|---:|---:|']
        for r in runtime['trajectories']:
            lines.append('| '+' | '.join(fmt(v) for v in (r['tag'],r['consumed_tokens'],r['wall_seconds'],r['wall_tokens_per_second'],r['postwarm_tokens_per_second'],r['peak_reserved_bytes']))+' |')
        mechanisms=[r for r in runtime['trajectories'] if r.get('mechanism_summary')]
        if mechanisms:
            lines += ['', '| 实际轨迹 | scale观测最小值 | scale观测最大值 | rounded clamp均值 | unrounded越界均值 | local-QDQ相对MSE均值 |',
                      '|---|---:|---:|---:|---:|---:|']
            for r in mechanisms:
                m=r['mechanism_summary'];lo=m.get('effective_scale_min',m.get('scale_min',{})).get('min')
                hi=m.get('effective_scale_max',m.get('scale_max',{})).get('max')
                lines.append('| '+' | '.join(fmt(v) for v in (r['tag'],lo,hi,m.get('rounded_clamp_fraction',{}).get('mean'),
                    m.get('unrounded_over127_fraction',{}).get('mean'),m.get('relative_local_QDQ_mse',{}).get('mean')))+' |')
            lines += ['', 'D的scale列为effective_scale[i,j]=row_scale[i]×column_scale[j]；A为column scale，未把D的row scale单独与A比较。表中为固定采样位置的观测汇总，per-layer/head完整风险门禁仍在每个token执行。']
    lines += ['', 'C128的分段事件在固定采样位置观察原函数，包括decode与有限性检查；Dual-M事件覆盖batched24层encode/decode/writeback及单独安全/指标审计。CPU观察与隐式同步可影响wall耗时，完整实际reservation账本优先于把CUDA事件之和当作全部成本。',
        '未采集匹配FP_H整条state轨迹，因此共同FP state relative-MSE为NOT_MEASURED。A/D各自在自身输入状态上的local-QDQ误差仅为探索性诊断，不能归因于单独row scale或Query importance。',
        '', '## 冻结文档与身份', '', '| Doc | document ID | raw SHA256 | input32768 canonical SHA256 | project |', '|---|---|---|---|---|']
    for d in docs_manifest['selected_documents']:
        lines.append('| '+' | '.join(str(v) for v in (d['document_index'],d['document_id'],d['raw_text_sha256'],d['artifacts']['input_ids_32768']['canonical_json_sha256'],d.get('project_identity','UNKNOWN')))+' |')
    lines += ['', '评分位置为0-based t=8192+64k，k=0..383；消费ids[t]后当前位置logits预测ids[t+1]。每轨迹消费32768 token；两种学生都从token0建立自己的QDQ历史，不使用旧anchor。',
        '', '| 文件 | 当前SHA256 |', '|---|---|']
    for name,digest in sorted(input_hashes.items()):lines.append(f'| {name} | {digest} |')
    lines += ['', '完整源码与依赖身份见source_manifest.json；模型14文件全流SHA见qualification_cpu/MODEL_CONTENT_IDENTITY.json。score文件和每个384位置logit身份由analysis/SCORE_ARTIFACT_VERIFICATION.json在实际存在时绑定。checkpoint版本SHA见checkpoints/CATALOG.jsonl，只有滚动CURRENT.pt为当前保留资产；退役版本按事前保留规则替换，不伪称全部仍在磁盘。最终交付seal由独立交付核验写入。',
        '', '## 证据类别', '', '**VERIFIED_ENGINEERING_RESULT**：仅上表真实PASS的CPU身份、结构、资源与GPU数值/执行门禁。未运行阶段保持NOT_RUN。',
        '**EXPLORATORY_SCIENTIFIC_RESULT**：仅完整真实384点配对结果。N=2；不作confirmatory显著性宣称。',
        '**NOT_ESTABLISHED**：此前FP_NATIVE/FP_H全模型数值等价、广泛泛化、AIME收益、learned-rotation互补、physical packed INT8部署与C8训练均未建立。',
        '', '## 用户要求的最终字段', '', '```text']
    lines += [f'{k} = {status[k]}' for k in REQUIRED_FIELDS]
    lines += ['```','',f'报告绝对路径：{root/"FINAL_SCIENTIFIC_REPORT.md"}',
        '本任务到此STOP；不运行后续confirmatory实验、AIME、C8、自动后台任务或git push。','']
    return '\n'.join(lines)


def process(root):
    root=pathlib.Path(root).resolve(); started=time.time()
    manifest,prereg,docs,structure,model,gate,ledger,post=verify_inputs(root)
    runs,available=load_actual_runs(root,gate,ledger)
    results,scientific_docs,primary=checked_science(root,post,available)
    failure=optional(root,'FIRST_FAILURE.json')
    original_failure_sha=sha(root/'FIRST_FAILURE.json') if failure else None
    scale_existing=optional(root,'scale_safety.json')
    derived_scale=safety_summary(runs)
    if scale_existing and scale_existing.get('status')=='FAIL':
        scale=scale_existing
    elif scale_existing:
        require(derived_scale is not None and scale_existing==derived_scale,
                'existing non-failure scale summary does not match actual completed Dual-M trajectories')
        scale=scale_existing
    else:
        scale=derived_scale
    status,parity=derive_status(prereg,docs,structure,gate,ledger,runs,available,scientific_docs,primary,failure,scale)
    admission=optional(root,'gpu_admission.json')
    status['RESOURCE_ADMISSION']=('PASS' if admission and admission.get('status')=='PASS'
        else 'FAIL' if admission and admission.get('status')=='FAIL' else 'NOT_RUN')
    if not admission and failure and not ledger['phases']:
        status['RESOURCE_ADMISSION']='FAIL' if 'admission' in json.dumps(failure).lower() or 'gpu0' in json.dumps(failure).lower() else 'NOT_RUN'
    # A later document's safety STOP does not erase an earlier completed,
    # fully qualified paired document. The postprocessor counts only completed
    # PASS phases; global NUMERICAL_SAFETY still truthfully remains FAIL.
    require(not scientific_docs or all(status[x]=='PASS' for x in ('CANONICAL_H_C128_PARITY','DUAL_M_GPU_PARITY','PERSISTENT_QDQ','FRESH_PROCESS_DETERMINISM')),
            'scientific results lack mandatory canary GPU qualification')
    g4=optional(root,'G4_RUNTIME_BUDGET.json') or optional(root,'analysis/G4_RUNTIME_BUDGET.json')
    runtime=dict(status='MEASURED' if runs else 'NOT_RUN',GPU_HOURS_USED=ledger['GPU_HOURS_USED'],MAX_GPU_HOURS=24.0,
        GPU_UUID=ledger['GPU_UUID'],GPU_release_verified=True,resource_ledger_sha256=sha(root/'gpu_resource_ledger.json'),
        phase_reservations=ledger['phases'],G4=g4,trajectories=[timing_summary(k,r) for k,r in runs.items()],
        load_measurements={p['phase']:optional(root,'analysis/ENV_'+p['phase']+'.json') for p in ledger['phases']},
        measured_deployment_latency_threshold='NOT_PREREGISTERED',PHYSICAL_INT8_RUNTIME='NOT_QUALIFIED',
        current_checkpoint_catalog='checkpoints/CATALOG.jsonl' if (root/'checkpoints/CATALOG.jsonl').exists() else None,
        final_reducer_CPU_wall_seconds=post.get('CPU_wall_seconds'))
    timing_by_tag={x['tag']:x for x in runtime['trajectories']};ratios=[]
    for scope,a_tag,d_tag in [('CANARY_ONLY','A_OFF','D_OFF')]+[(f'Doc{i}_SCIENCE_FULL32768',f'science_doc{i}_H_C128',f'science_doc{i}_H_DUAL_M') for i in (1,2)]:
        if a_tag in timing_by_tag and d_tag in timing_by_tag:
            a=timing_by_tag[a_tag]['postwarm_full_model_plus_QDQ_seconds'];d=timing_by_tag[d_tag]['postwarm_full_model_plus_QDQ_seconds']
            if a and d and a['median']>0:
                ratios.append(dict(scope=scope,A_tag=a_tag,D_tag=d_tag,D_over_A_median_full_token_seconds=d['median']/a['median']))
    runtime['matched_runtime_ratios']=ratios
    current=root/'checkpoints/CURRENT.pt';catalog=root/'checkpoints/CATALOG.jsonl'
    if current.exists():
        require(catalog.is_file(), 'current checkpoint has no retention/version catalog')
        versions=[json.loads(line) for line in catalog.read_text().splitlines() if line.strip()]
        require(bool(versions), 'checkpoint catalog is empty')
        latest=versions[-1];current_sha=sha(current)
        require(current_sha==latest['sha256'] and current.stat().st_size==latest['bytes'], 'current rolling checkpoint differs from last catalog receipt')
        runtime['current_checkpoint_identity']=dict(path='checkpoints/CURRENT.pt',sha256=current_sha,bytes=current.stat().st_size,
            cursor=latest['cursor'],condition=latest['condition'],doc_id=latest['doc_id'],fresh_stream_hash=True,
            historical_versions_in_catalog=len(versions),prior_versions_retained=False)
    numerical=dict(status='PASS' if status['NUMERICAL_SAFETY']=='PASS' and primary=='PASS' else 'PARTIAL_QUALIFICATION' if status['NUMERICAL_SAFETY']=='PASS' else status['NUMERICAL_SAFETY'],
        Hadamard_structural_correctness=status['HADAMARD_STRUCTURAL_CORRECTNESS'],
        FP_NATIVE_H_fullmodel_equivalence='OLD_FAIL_PRESERVED_NOT_REQUALIFIED',
        canary_fresh_process=status['FRESH_PROCESS_DETERMINISM'],canary_reducer=optional(root,'analysis/CANARY_REDUCER.json'),
        full384_position_check=primary,full_paired_documents=len(scientific_docs),
        resolution_rule=prereg['numerical_resolution'],no_old_resolution_reused=True,
        arithmetic_resolution_is_conditional=True,long_forward_uncertainty_established=False,
        FP16_scale_safety_status=status['NUMERICAL_SAFETY'],scale_safety_evidence='scale_safety.json' if scale else None,
        first_failure_sha256=original_failure_sha)
    if any(status[k]=='FAIL' for k in ('CANONICAL_H_C128_PARITY','DUAL_M_GPU_PARITY','PERSISTENT_QDQ','FRESH_PROCESS_DETERMINISM','NUMERICAL_SAFETY')) or any(
        word in json.dumps(failure or {}).lower() for word in ('reducer','fp64 crosscheck','state dtype/shape')):
        numerical['status']='FAIL'
    parity.update(status='PASS' if all(status[k]=='PASS' for k in ('CANONICAL_H_C128_PARITY','DUAL_M_GPU_PARITY','PERSISTENT_QDQ','FRESH_PROCESS_DETERMINISM'))
        else 'FAIL' if any(status[k]=='FAIL' for k in ('CANONICAL_H_C128_PARITY','DUAL_M_GPU_PARITY','PERSISTENT_QDQ','FRESH_PROCESS_DETERMINISM')) else 'PARTIAL_QUALIFICATION',
        scope='Actual128-token safety canary plus exact completed production counters; no substitute for full32K science',
        source_manifest_sha256=sha(root/'source_manifest.json'),physical_INT8='NOT_QUALIFIED',
        H_C128_direct_original_function=True,DualM_inherited_official_all8_ones=True)
    hash_names=['source_manifest.json','preregistration.json','frozen_documents.json','qualification_cpu/CPU_STRUCTURE_001.json',
        'qualification_cpu/MODEL_CONTENT_IDENTITY.json','src/core.py','src/worker.py','src/codec.py','src/reducer.py',
        'src/coordinator.py','analysis/POSTPROCESS_STATUS.json','gpu_resource_ledger.json','GATE_STATUS_FINAL.json',
        'FIRST_FAILURE.json','gpu_admission.json','G4_RUNTIME_BUDGET.json','analysis/CANARY_REDUCER.json',
        'science_results.json','science_results.csv','window_results.csv','analysis/SCORE_ARTIFACT_VERIFICATION.json','checkpoints/CATALOG.jsonl','scale_safety.json']
    hash_names += ['analysis/'+tag+'.json' for tag in runs]
    hash_names += ['analysis/'+prefix+p['phase']+'.json' for p in ledger['phases'] for prefix in ('RECEIPT_','ENV_')]
    input_hashes={n:sha(root/n) for n in hash_names if (root/n).is_file()}
    if 'current_checkpoint_identity' in runtime:
        input_hashes['checkpoints/CURRENT.pt']=runtime['current_checkpoint_identity']['sha256']
    input_hashes['analysis_scripts/finalize_report.py']=sha(__file__)
    evidence=dict(final_status_fields=list(REQUIRED_FIELDS),postprocess_status_sha256=input_hashes['analysis/POSTPROCESS_STATUS.json'],
        source_manifest_sha256=input_hashes['source_manifest.json'],first_failure_sha256=original_failure_sha,
        GPU_release_verified=True,coordinator_original_status=gate['status'],project_independence=docs['project_independence'],
        scientific_unit='document',target_N=2,actual_complete_paired_N=len(scientific_docs),
        complete_scientific_values_only=True,canary_scores_never_substituted=True,
        report_absolute_path=str(root/'FINAL_SCIENTIFIC_REPORT.md'),script_sha256=sha(__file__),
        input_artifact_sha256=input_hashes,report_started_unix=started,report_completed_unix=time.time())
    report=render_report(root,status,prereg,docs,gate,ledger,post,results,scientific_docs,parity,runtime,numerical,scale,failure,input_hashes)
    outputs={'FINAL_STATUS.json':dict(status,delivery_evidence=evidence),'runtime_cost.json':runtime,
        'numerical_qualification.json':numerical,'codec_parity.json':parity}
    if scale is not None and scale_existing is None:
        outputs['scale_safety.json']=scale
    for name in list(outputs)+['FINAL_SCIENTIFIC_REPORT.md']:
        require(not (root/name).exists(), 'refuse to overwrite final evidence: '+name)
    for name,value in outputs.items():
        with (root/name).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    with (root/'FINAL_SCIENTIFIC_REPORT.md').open('x') as f:f.write(report)
    require((sha(root/'FIRST_FAILURE.json') if failure else None)==original_failure_sha, 'first failure changed during report')
    return dict(status='PASS',FINAL_EXECUTION_STATUS=status['FINAL_EXECUTION_STATUS'],DUAL_M_FULLMODEL_RESULT=status['DUAL_M_FULLMODEL_RESULT'],
                scientific_documents=len(scientific_docs),required_user_fields=len(REQUIRED_FIELDS),
                report=str(root/'FINAL_SCIENTIFIC_REPORT.md'),GPU_work_performed=False,scientific_CSV_created=False)


def self_test():
    require(len(REQUIRED_FIELDS)==26 and len(set(REQUIRED_FIELDS))==26,'required fields count')
    assert classification([])=='INCONCLUSIVE'
    assert classification([{'Winner':'H_DUAL_M'}])=='PARTIAL'
    for wins,expected in [(['H_DUAL_M']*2,'POSITIVE_EXPLORATORY'),(['H_C128']*2,'NEGATIVE_EXPLORATORY'),
        (['H_DUAL_M','H_C128'],'MIXED_EXPLORATORY'),(['H_DUAL_M','NUMERICALLY_UNRESOLVED'],'MIXED_EXPLORATORY'),
        (['H_C128','NUMERICALLY_UNRESOLVED'],'INCONCLUSIVE'),(['NUMERICALLY_UNRESOLVED']*2,'INCONCLUSIVE')]:
        assert classification([dict(Winner=w) for w in wins])==expected
    a=dict(status='PASS',condition='H_C128',doc_id='SYNTHETIC_METADATA_ONLY',tag='A_ON',consumed_tokens=128,
        performance=[dict(t=i,scale_risk={}) for i in range(128)],writebacks=3072,next_token_consumption_checks=3048,
        rows=[dict(t=i,input_id=i,logits='HASH',cache='HASH',cache_cursor=i+1,helper={},QDQ={}) for i in range(128)])
    assert exact_pair(a,a,True)=='PASS' and qdq_counts(a)['status']=='PASS'
    bad=dict(a,writebacks=3071);assert qdq_counts(bad)['status']=='FAIL'
    badrows=[dict(r) for r in a['rows']];badrows[1]['cache']='DIFFERENT'
    assert exact_pair(a,dict(a,rows=badrows))=='FAIL'
    assert exact_pair(a,None)=='NOT_RUN'
    d=dict(a,condition='H_DUAL_M',mechanism=[dict(token_index=0,stats=dict(official_exact='PASS_TORCH_EQUAL_CODES_SCALES_DEQUANT',batched_vs_individual='BYTE_EXACT'))])
    assert official_dual_parity(d)=='PASS'
    assert safety_summary({}) is None
    d['performance']=[dict(t=i,scale_risk=dict(raw_row_bound_hits=0,raw_column_bound_hits=0,row_floor_hits=0,row_ceiling_hits=0,
        column_floor_hits=0,column_ceiling_hits=0,max_affected_energy_fraction=0,max_affected_relative_mse=0)) for i in range(128)]
    s=safety_summary({'D_ON':d});assert s['consumed_tokens_total']==128 and s['status']=='PASS'
    assert final_execution_status('COMPLETE_AND_STOP',2)=='COMPLETE_AND_STOP'
    assert final_execution_status('FAIL_AND_STOP',1)=='PARTIAL_STOP'
    assert final_execution_status('BLOCKED_AND_STOP',0)=='BLOCKED_AND_STOP'
    assert final_execution_status('FAIL_AND_STOP',0)=='FAIL_AND_STOP'
    meta=dict(DOCS_FROZEN=2,selected_documents=[],project_independence='UNKNOWN')
    struct=dict(seals=dict(inherited_cpu_v1=dict(status='PASS_UNCHANGED')),fixed_tensor_structure=dict(status='PASS'))
    gate=dict(status='BLOCKED_AND_STOP',gates={});ledger=dict(GPU_HOURS_USED=0.1,GPU_UUID='SYNTHETIC',phases=[])
    available={f'science_doc{i}_{c}':False for i in (1,2) for c in CONDITIONS}
    st,parity=derive_status({},meta,struct,gate,ledger,{'A_ON':a,'D_ON':d},available,[],'NOT_RUN',None,s)
    assert len(st)==26 and st['VALID_PAIRED_DOCS']=='0/2' and st['DUAL_M_FULLMODEL_RESULT']=='INCONCLUSIVE'
    assert st['PRIMARY_384_POSITION_CHECK']=='NOT_RUN' and st['FP_NATIVE_32K']=='0/2'
    failed=dict(error='RuntimeError("Same GPU fresh-process exact parity failed")',context=dict(condition='H_C128'))
    st_failed,parity_failed=derive_status({},meta,struct,gate,ledger,{'A_ON':a,'D_ON':d},available,[],'NOT_RUN',failed,s)
    assert st_failed['FRESH_PROCESS_DETERMINISM']=='FAIL' and parity_failed['fresh_process_pairs']['H_C128']=='FAIL'
    reducer_failed=dict(error='RuntimeError("Canary reducer FP64 crosscheck failed")')
    st_failed,_=derive_status({},meta,struct,gate,ledger,{'A_ON':a,'D_ON':d},available,[],'NOT_RUN',reducer_failed,s)
    assert st_failed['NUMERICAL_SAFETY']=='FAIL'
    # Synthetic raw arithmetic fixtures only: no scores/models/files are read.
    ar=[];dr=[];ident=dict(doc_id='SYNTHETIC_METADATA_ONLY')
    for t in [8192+64*k for k in range(384)]:
        shared=dict(identity=ident,token_index=t,consumed_tokens=t+1,predicted_token_index=t+1,
            next_token_id=0,full_vocabulary_size=2,reference_logits_sha256='SYNTHETIC',reducer_policy_sha256='SYNTHETIC')
        for target,value in ((ar,0.5),(dr,0.25)):
            target.append(dict(shared,KL_FP32=value,KL_FP64=value,FP64_reference_roundoff=1e-12,
                FP32_vs_FP64_abs_error=0.0,per_arm_arithmetic_error=1e-12))
    summary=dict(recompute_pair_arithmetic(ar,dr),identity=ident,
        sum384_A=math.fsum(r['KL_FP32'] for r in ar),sum384_D=math.fsum(r['KL_FP32'] for r in dr))
    summary['windows']={name:recompute_pair_arithmetic([r for r in ar if lo<=r['token_index']<hi],
        [r for r in dr if lo<=r['token_index']<hi]) for name,lo,hi in (('Early',8192,16384),('Middle',16384,24576),('Late',24576,32768))}
    audit_pair_rows(ar,dr,summary)
    summary['windows']['Late']['Delta_resolution']+=0.01
    try:audit_pair_rows(ar,dr,summary)
    except ValueError:pass
    else:raise AssertionError('tampered window arithmetic was accepted')
    report=render_report(pathlib.Path('/SYNTHETIC_METADATA_ONLY'),st,{},meta,gate,ledger,{},None,[],parity,
        dict(trajectories=[],G4=None),{},s,None,{})
    assert '正式32K Future-KL：**NOT_RUN**' in report and all(f'{i}. **' in report for i in range(1,9))
    summary['windows']['Late']=recompute_pair_arithmetic(ar[-128:],dr[-128:])
    summary.update(document_index=1,complete=True)
    partial_status=dict(st,VALID_PAIRED_DOCS='1/2',PRIMARY_384_POSITION_CHECK='PASS',
        DUAL_M_FULLMODEL_RESULT='PARTIAL',FINAL_EXECUTION_STATUS='PARTIAL_STOP')
    partial_report=render_report(pathlib.Path('/SYNTHETIC_METADATA_ONLY'),partial_status,{},meta,gate,ledger,{},
        dict(two_document_means=None),[summary],parity,dict(trajectories=[],G4=None),{},s,None,{})
    assert '两篇平均绝对与相对改善尚不能计算' in partial_report and all(f'{i}. **' in partial_report for i in range(1,9))
    second=dict(summary,document_index=2)
    full_status=dict(st,VALID_PAIRED_DOCS='2/2',PRIMARY_384_POSITION_CHECK='PASS',
        DUAL_M_FULLMODEL_RESULT='POSITIVE_EXPLORATORY',FINAL_EXECUTION_STATUS='COMPLETE_AND_STOP')
    full_report=render_report(pathlib.Path('/SYNTHETIC_METADATA_ONLY'),full_status,{},meta,
        dict(gate,status='COMPLETE_AND_STOP'),ledger,{},dict(two_document_means=dict(mean_absolute_Delta=0.25,
            mean_per_doc_RelativeGain=0.5,relative_gain_of_mean_K=0.5)),[summary,second],parity,
        dict(trajectories=[],G4=None),{},s,None,{})
    assert '窗口算术分辨率' in full_report and '事前固定' in full_report and all(f'{i}. **' in full_report for i in range(1,9))
    return dict(status='PASS',scope='Synthetic metadata/control only; no real analysis/scientific scores read or outputs written',
                required_user_fields=26,classification_cases=8,zero_science_status_and_8_answer_render='PASS',
                fresh_failure_not_mislabelled_NOT_RUN=True,raw384_and_window_resolution_audit='PASS',
                complete_partial_failure_26fields_8answer_render='PASS',
                first_failure_never_written=True,GPU_work_performed=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(ROOT));p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    print(json.dumps(self_test() if a.self_test else process(a.root),ensure_ascii=False),flush=True)
