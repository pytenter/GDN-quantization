"""Independent CPU delivery analysis; never imports a model or edits frozen files.

Run only after the final coordinator receipts prove GPU work has stopped.
The --self-test mode exercises synthetic metadata/control branches in memory;
it never reads actual scores or writes scientific outputs.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import csv
import hashlib
import json
import math
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONDITIONS = ('FP_NATIVE', 'H_C128', 'H_DUAL_M')
TASK = 'QWEN_GDN_DUAL_AXIS_PROSPECTIVE_REPLICATION_5DOC_V1'
TARGET_N = 5
SCORE_STORAGE_ROOT = pathlib.Path('/tmp/qwen_5doc_scores_20261016')


def require(ok, why):
    if not ok:
        raise ValueError(why)


def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read(path):
    return json.loads(pathlib.Path(path).read_text())


def write_new(path, value):
    path = pathlib.Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')


def safe_path(root, relative):
    root = pathlib.Path(root).resolve(); relative = pathlib.Path(relative)
    require(not relative.is_absolute() and '..' not in relative.parts, 'unsafe relative artifact path')
    p = (root / relative).resolve()
    if p.is_relative_to(root):
        return p
    # The sole authorized out-of-root I/O binding is the fixed score directory.
    allowed = SCORE_STORAGE_ROOT.resolve()
    link = root / 'scores'
    require(relative.parts and relative.parts[0] == 'scores' and len(relative.parts) > 1
            and link.is_symlink() and link.resolve() == allowed and p.is_relative_to(allowed),
            'artifact escapes experiment root or fixed score storage: ' + str(relative))
    return p


def winner_direction(raw):
    mapping = {'H_DUAL_M': 'D_WIN', 'H_C128': 'A_WIN',
               'D_WIN': 'D_WIN', 'A_WIN': 'A_WIN',
               'NUMERICALLY_UNRESOLVED': 'NUMERICALLY_UNRESOLVED'}
    require(raw in mapping, 'unknown frozen reducer Winner label')
    return mapping[raw]


def annotate_summary(summary):
    # Preserve the actual frozen reducer's Winner and all numerical outputs.
    summary['Winner_direction'] = winner_direction(summary['Winner'])
    summary['RelativeGain_unavailable_reason'] = (
        'K_A_NOT_ABOVE_ITS_FROZEN_ARITHMETIC_DENOMINATOR_BOUND'
        if summary['RelativeGain'] is None else None)
    for window in summary.get('windows', {}).values():
        window['Winner_direction'] = winner_direction(window['Winner'])
        window['RelativeGain_unavailable_reason'] = (
            'K_A_NOT_ABOVE_ITS_FROZEN_ARITHMETIC_DENOMINATOR_BOUND'
            if window['RelativeGain'] is None else None)
    return summary


def scientific_classification(results):
    require(len(results) <= TARGET_N, 'more than the five frozen replication documents')
    if len(results) != TARGET_N:
        return 'INCOMPLETE_NOT_CONFIRMATORY'
    directions = [winner_direction(r['Winner']) for r in results]
    mean_delta = math.fsum(r['Delta'] for r in results) / TARGET_N
    if directions.count('D_WIN') == 5 and mean_delta > 0:
        return 'POSITIVE_PROSPECTIVE_REPLICATION'
    if directions.count('D_WIN') == 4 and mean_delta > 0:
        return 'DIRECTIONALLY_SUPPORTIVE_MIXED'
    if directions.count('A_WIN') >= 3 and mean_delta < 0:
        return 'REPORT_NEGATIVE_EVIDENCE'
    return 'MIXED_OR_INCONCLUSIVE'


def document_means(results):
    if len(results) != TARGET_N:
        return None
    gains = [s['RelativeGain'] for s in results]
    valid = all(g is not None for g in gains)
    sum_ka = math.fsum(s['K_A'] for s in results)
    mean_delta = math.fsum(s['Delta'] for s in results) / TARGET_N
    reason = None if valid else 'AT_LEAST_ONE_DOCUMENT_K_A_DENOMINATOR_NOT_ARITHMETICALLY_QUALIFIED'
    return {'K_A': sum_ka / TARGET_N,
            'K_D': math.fsum(s['K_D'] for s in results) / TARGET_N,
            'mean_absolute_Delta': mean_delta,
            'mean_Delta': mean_delta,
            'mean_per_doc_RelativeGain': math.fsum(gains) / TARGET_N if valid else None,
            'mean_per_doc_RelativeGain_unavailable_reason': reason,
            'relative_gain_of_mean_K': math.fsum(s['Delta'] for s in results) / sum_ka if valid and sum_ka > 0 else None,
            'relative_gain_of_mean_K_unavailable_reason': reason if not valid else None if sum_ka > 0 else 'NONPOSITIVE_MEAN_K_A_DENOMINATOR',
            'document_count': TARGET_N,
            'weighting': 'Each of the five new documents has equal weight; historical Pilot documents excluded'}


def validate_score_metadata(payload, *, condition, doc_id, input_sha, source_sha, positions):
    require(payload.get('condition') == condition, 'score condition identity mismatch')
    require(payload.get('doc_id') == doc_id, 'score document identity mismatch')
    require(payload.get('consumed_tokens') == 32768, 'score artifact is not a full32768 trajectory')
    require(payload.get('input_ids_sha256') == input_sha, 'score consumed-input identity mismatch')
    require(payload.get('source_manifest_sha256') == source_sha, 'score source manifest identity mismatch')
    require(payload.get('score_positions') == list(positions), 'score metadata exact384 ordering mismatch')
    logits = payload.get('logits')
    require(isinstance(logits, dict) and tuple(logits) == tuple(positions), 'score tensor keys incomplete, duplicated, or reordered')


def verify_frozen(root):
    manifest_path = root / 'source_manifest.json'; manifest = read(manifest_path)
    require(manifest['task'] == TASK and manifest['frozen_before_any_new_GPU_forward'] is True, 'wrong prospective source manifest')
    for item in manifest['execution_sources']:
        p = safe_path(root, item['path'])
        require(p.stat().st_size == item['bytes'] and sha(p) == item['sha256'], 'frozen source identity mismatch: ' + item['path'])
    prereg = read(root / 'preregistration.json'); documents = read(root / 'frozen_documents.json')
    require(prereg['TASK_ID'] == TASK and prereg['SCIENTIFIC_PROTOCOL'] == 'NEW_PROSPECTIVE_CODEC_COMPARISON', 'wrong scientific protocol')
    require(sha(root / 'frozen_documents.json') == prereg['documents_manifest_sha256'], 'preregistered document manifest changed')
    model_receipt = root / 'qualification_cpu/MODEL_CONTENT_IDENTITY.json'
    require(sha(model_receipt) == manifest['model_identity_receipt_sha256'] and read(model_receipt)['status'] == 'PASS', 'model-content receipt identity failed')
    require(prereg['TARGET_N'] == TARGET_N and documents['DOCS_FROZEN'] == TARGET_N and len(documents['selected_documents']) == TARGET_N, 'five prospective replication documents were not frozen')
    return manifest, prereg, documents, sha(manifest_path)


def gpu_mapping(prereg, ledger):
    actual = ledger.get('GPU_UUIDS')
    expected = prereg.get('G1', {}).get('GPU_UUIDS', prereg.get('GPU_UUIDS'))
    require(isinstance(actual, dict) and isinstance(expected, dict), 'four-card preregistered and actual GPU_UUIDS mappings are required')
    actual = {str(k):v for k,v in actual.items()}; expected = {str(k):v for k,v in expected.items()}
    require(set(actual) == set(expected) == {'0','1','2','3'} and actual == expected,
            'four-card resource ledger differs from preregistered physical UUIDs')
    require(len(set(actual.values())) == 4 and all(isinstance(v, str) and v.startswith('GPU-') for v in actual.values()), 'GPU UUID mapping is not four distinct actual devices')
    return actual


def doc_gpu_expectation(prereg, ledger, document_index):
    mapping = gpu_mapping(prereg, ledger); assignments = prereg.get('doc_GPU_assignment')
    require(isinstance(assignments, dict) and {str(k) for k in assignments} == {str(i) for i in range(1,TARGET_N+1)}, 'all five document-to-GPU assignments must be preregistered')
    assignments = {str(k):v for k,v in assignments.items()}; value = assignments[str(document_index)]
    card = value.get('physical_GPU_ID') if isinstance(value,dict) else value
    require(type(card) is int and str(card) in mapping, 'invalid preregistered document GPU assignment')
    if isinstance(value,dict) and 'GPU_UUID' in value:
        require(value['GPU_UUID'] == mapping[str(card)], 'document GPU assignment UUID disagrees with preregistered GPU roster')
    return card, mapping[str(card)]


def verify_cpu_boundary(root, prereg):
    gate = read(root / 'GATE_STATUS_FINAL.json'); ledger = read(root / 'gpu_resource_ledger.json')
    require(gate.get('task_id') == TASK and gate.get('no_further_GPU_work') is True, 'final no-further-GPU receipt missing')
    require(gate.get('GPU_release_verified') is True and ledger.get('final_GPU_release_verified') is True, 'GPU release was not verified')
    uuid_map = gpu_mapping(prereg, ledger)
    for entry in ledger['phases']:
        card = entry.get('physical_GPU_ID')
        require(type(card) is int and str(card) in uuid_map and entry.get('GPU_UUID') == uuid_map[str(card)], 'phase resource ledger physical GPU/UUID identity mismatch')
    require(math.isclose(math.fsum(p['GPU_hours'] for p in ledger['phases']), ledger['GPU_HOURS_USED'], rel_tol=0, abs_tol=1e-9), 'four-card summed reservations differ from task GPUh')
    require(all(p.get('released') is True for p in ledger['phases']), 'a reservation still has unverified GPU release')
    require(gate['GPU_HOURS_USED'] == ledger['GPU_HOURS_USED'] and 0 <= ledger['GPU_HOURS_USED'] <= 24.0, 'GPU accounting identity/cap failed')
    require('ended_unix' in ledger and ledger['ended_unix'] <= time.time(), 'coordinator final end time missing')
    own_pids = {int(p['pid']) for p in ledger['phases'] if 'pid' in p}
    probe = {'mode': 'final coordinator NVML receipt plus read-only current own-PID query', 'own_worker_pids': sorted(own_pids), 'GPU_UUIDS': uuid_map}
    try:
        raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name', '--format=csv,noheader,nounits'], text=True, timeout=10)
        apps = []
        for line in raw.splitlines():
            parts = [x.strip() for x in line.split(',')]
            if len(parts) >= 2 and parts[1].isdigit():
                apps.append({'uuid': parts[0], 'pid': int(parts[1]), 'name': ','.join(parts[2:])})
        require(not any(a['pid'] in own_pids for a in apps), 'an own scientific GPU worker is still resident')
        probe.update(current_query='PASS_NO_OWN_GPU_WORKER', current_apps=apps)
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        # Local read-only delivery analysis may lack NVML. This never upgrades an
        # absent release receipt; the two mandatory final receipts above remain.
        probe.update(current_query='UNAVAILABLE_USING_VERIFIED_FINAL_RECEIPTS', query_error=repr(exc))
    return gate, ledger, probe


def model_vocabulary_from_verified_config(model_receipt):
    entries = model_receipt['model_weight_and_config_current_content_hashes']
    configs = [r for r in entries if pathlib.Path(r['path']).name == 'config.json']
    require(len(configs) == 1, 'model config identity ambiguous')
    p = pathlib.Path(configs[0]['path'])
    if not p.exists():
        return None, {'status': 'CONFIG_NOT_LOCAL', 'expected_sha256': configs[0]['sha256'],
                      'note': 'actual vocabulary must match all complete saved vectors; model config was verified by the frozen G0 receipt'}
    require(sha(p) == configs[0]['sha256'], 'live small model config differs from the frozen model receipt')
    cfg = read(p); cfg = cfg.get('text_config', cfg)
    return int(cfg['vocab_size']), {'status': 'PASS', 'path': str(p), 'sha256': configs[0]['sha256']}


def phase_is_complete(root, gate, ledger, phase, prereg):
    if gate['gates'].get(phase) != 'PASS':
        return False, {'phase': phase, 'status': gate['gates'].get(phase, 'NOT_RUN')}
    rp = root / ('analysis/RECEIPT_' + phase + '.json')
    ap = root / ('analysis/' + phase + '.json')
    receipt = read(rp); result = read(ap)
    entry = [r for r in ledger['phases'] if r['phase'] == phase]
    require(len(entry) == 1 and entry[0].get('returncode') == 0 and entry[0].get('released') is True, 'phase ledger completion mismatch: ' + phase)
    require(sha(rp) == entry[0]['receipt_sha256'] and receipt.get('status') == 'PASS' and result.get('status') == 'PASS', 'phase PASS receipt identity failed: ' + phase)
    require(receipt['result']['status'] == 'PASS' and receipt['result']['consumed'] == 32768 and receipt['result']['scores'] == 384, 'phase did not complete full32768/384: ' + phase)
    require(result['tag'] == phase and result['consumed_tokens'] == 32768, 'phase result cursor/tag mismatch')
    parts=phase.split('_',2)
    require(receipt['result']['doc']==int(parts[1][3:]) and receipt['result']['condition']==parts[2] and result['condition']==parts[2], 'phase receipt document/condition mismatch')
    card, uuid = doc_gpu_expectation(prereg, ledger, int(parts[1][3:]))
    require(entry[0].get('physical_GPU_ID') == card and entry[0].get('GPU_UUID') == uuid,
            'completed document trajectory ran on a different physical GPU than preregistered')
    require(entry[0].get('worker_phase') == phase and entry[0].get('output_root_relative') == '.',
            'formal phase did not use the frozen main-root science output binding')
    ep = root / ('analysis/ENV_' + phase + '.json'); env = read(ep)
    require(env.get('UUID') == uuid and str(env.get('CUDA_VISIBLE_DEVICES')) == str(card),
            'completed phase ENV disagrees with physical device/UUID assignment')
    return True, {'phase': phase, 'status': 'PASS', 'receipt_sha256': sha(rp), 'analysis_sha256': sha(ap),
                  'ENV_sha256': sha(ep), 'physical_GPU_ID': card, 'GPU_UUID': uuid, 'result': result}



def load_scores(root, info, d, condition, source_sha, expected_vocab, torch, R):
    result = info['result']; p = safe_path(root, result['score_file'])
    file_sha = sha(p)
    require(file_sha == result['score_file_sha256'], 'score payload SHA differs from completed trajectory receipt')
    require(result['condition'] == condition and result['doc_id'] == d['document_id'], 'phase condition/doc result mismatch')
    payload = torch.load(p, map_location='cpu', weights_only=True)
    validate_score_metadata(payload, condition=condition, doc_id=d['document_id'],
                            input_sha=d['artifacts']['input_ids_32768']['canonical_json_sha256'],
                            source_sha=source_sha, positions=R.SCORE_POSITIONS)
    vocabulary = None; vector_hashes = []
    for t, x in payload['logits'].items():
        require(isinstance(x, torch.Tensor) and x.dtype == torch.float32 and x.device.type == 'cpu', 'saved full logits are not CPU FP32')
        require(x.ndim >= 1 and all(s == 1 for s in x.shape[:-1]) and x.shape[-1] >= 2, 'saved logits must have exactly one full-vocabulary vector')
        n = x.numel()
        if vocabulary is None:
            vocabulary = n
        require(n == vocabulary and (expected_vocab is None or n == expected_vocab), 'actual vocabulary shape mismatch')
        require(bool(torch.isfinite(x).all()), 'nonfinite saved full-vocabulary logits')
        vector_hashes.append({'token_index': t, 'consumed_tokens': t+1, 'predicted_token_index': t+1,
                              'full_vocabulary_size': n, 'FP32_logits_sha256': R.tensor_sha256(x)})
    return payload['logits'], {'phase': info['phase'], 'condition': condition, 'path': result['score_file'],
                              'sha256': file_sha, 'bytes': p.stat().st_size, 'actual_vocabulary': vocabulary,
                              'score_count': len(vector_hashes), 'full384_logit_hashes': vector_hashes,
                              'phase_receipt_sha256': info['receipt_sha256'], 'phase_analysis_sha256': info['analysis_sha256'],
                              'physical_GPU_ID': info['physical_GPU_ID'], 'GPU_UUID': info['GPU_UUID'], 'ENV_sha256': info['ENV_sha256']}


def frozen_ids(root, d):
    vectors = {}
    for name in ('input_ids_32768', 'input_ids_32769'):
        entry = d['artifacts'][name]; p = safe_path(root, entry['path']); value = read(p)
        require(sha(p) == entry['file_sha256'] and canonical(value) == entry['canonical_json_sha256'], 'frozen input file/content hash failed')
        require(len(value) == entry['count'] and all(type(x) is int and x >= 0 for x in value), 'frozen input token vector invalid')
        vectors[name] = value
    require(vectors['input_ids_32768'] == vectors['input_ids_32769'][:32768], '32769 labels do not extend the consumed32768 input')
    return vectors['input_ids_32769']


def write_csv(path, columns, rows):
    require(rows, 'refuse to create an empty scientific CSV')
    with pathlib.Path(path).open('x', newline='') as f:
        w=csv.DictWriter(f, fieldnames=columns);w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in columns})


def process(root):
    root = pathlib.Path(root).resolve(); start = time.time()
    manifest, prereg, documents, source_sha = verify_frozen(root)
    gate, ledger, boundary = verify_cpu_boundary(root, prereg)
    sys.path.insert(0, str(root / 'src'))
    import torch
    import reducer as R
    require(not torch.cuda.is_initialized(), 'postprocessor must never initialize CUDA')
    require(R.__file__ == str(root / 'src/reducer.py') and sha(R.__file__) == manifest['KL_reducer_source_sha256'], 'wrong frozen reducer module')
    R.configure_cpu()
    require(tuple(prereg['score_positions']) == R.SCORE_POSITIONS, 'preregistered score positions differ from reducer')
    uuid_map = gpu_mapping(prereg, ledger)
    model_receipt = read(root / 'qualification_cpu/MODEL_CONTENT_IDENTITY.json')
    actual_vocab, config_verification = model_vocabulary_from_verified_config(model_receipt)
    availability = []; complete_docs = []; phase_infos = {}
    for doc in range(1, TARGET_N + 1):
        infos = []
        for condition in CONDITIONS:
            phase = f'science_doc{doc}_{condition}'
            complete, info = phase_is_complete(root, gate, ledger, phase, prereg)
            availability.append({k:v for k,v in info.items() if k != 'result'})
            if complete:
                infos.append(info);phase_infos[(doc,condition)] = info
        if len(infos) == 3:
            card, uuid = doc_gpu_expectation(prereg, ledger, doc)
            require(all(gate['gates'].get(key) == 'PASS' for key in ('canary_initial','canary_fresh','cross_gpu_canary_exact',f'gpu{card}_canary_initial',f'gpu{card}_canary_fresh')), 'completed scientific document lacks its device canary/fresh/cross-GPU gates')
            require({info['GPU_UUID'] for info in infos} == {uuid} and {info['physical_GPU_ID'] for info in infos} == {card}, 'three paired document arms did not use the identical physical GPU')
            complete_docs.append(doc)
    # Audit every completed score artifact, including otherwise unpaired phases.
    # No KL is computed unless the entire three-condition document is complete.
    audited_assets={}
    for (doc,condition),info in phase_infos.items():
        d=documents['selected_documents'][doc-1]
        vectors,asset=load_scores(root,info,d,condition,source_sha,actual_vocab,torch,R)
        audited_assets[(doc,condition)]=dict(document_index=doc,**asset)
        del vectors
    require(len({a['actual_vocabulary'] for a in audited_assets.values()}) <= 1, 'completed trajectories have inconsistent actual vocabulary sizes')
    base_status = {'TASK': TASK, 'CPU_ONLY': True, 'CUDA_initialized': False,
                   'CPU_started_unix': start, 'GPU_coordinator_ended_unix': ledger['ended_unix'],
                   'postprocessor_source_sha256': sha(__file__), 'source_manifest_sha256': source_sha,
                   'frozen_execution_entries_verified': len(manifest['execution_sources']),
                   'final_GPU_execution_status': gate['status'], 'GPU_HOURS_USED': ledger['GPU_HOURS_USED'],
                   'GPU_CPU_boundary': boundary, 'GPU_UUIDS': uuid_map, 'doc_GPU_assignment': prereg['doc_GPU_assignment'], 'phase_availability': availability,
                   'target_documents': TARGET_N, 'complete_paired_documents': len(complete_docs),
                   'project_independence': documents['project_independence'],
                   'conditional_reducer_resolution_scope': R.REDUCTION_POLICY['resolution_scope'],
                   'reference_roundoff_assumptions': R.REFERENCE_ASSUMPTIONS,
                   'old_FP_NATIVE_H_failure': 'PRESERVED_NOT_RECLASSIFIED',
                   'reducer_policy_sha256': R.REDUCTION_POLICY_SHA256,
                   'model_vocabulary_config_verification': config_verification}
    if not complete_docs:
        require(not any((root/x).exists() for x in ('science_results.json','science_results.csv','window_results.csv')), 'scientific outputs already exist despite no complete paired document')
        status = dict(base_status, status='NOT_RUN', reason='No document has three completed qualified full32768 trajectories and384 real paired scores',
                      completed_unpaired_score_artifacts=list(audited_assets.values()),
                      scientific_outputs_created=False, DUAL_M_FULLMODEL_RESULT='INCOMPLETE_NOT_CONFIRMATORY', CPU_wall_seconds=time.time()-start)
        write_new(root / 'analysis/POSTPROCESS_STATUS.json', status)
        return status
    outputs = ['science_results.json','science_results.csv','window_results.csv','analysis/POSTPROCESS_STATUS.json','analysis/SCORE_ARTIFACT_VERIFICATION.json']
    for doc in complete_docs:
        outputs += [f'analysis/score_rows_doc{doc}_{arm}.json' for arm in ('A','D')]
    require(not any((root / x).exists() for x in outputs), 'postprocess outputs already exist; refuse to overwrite scientific evidence')
    summaries=[];score_assets=list(audited_assets.values());all_rows={}
    for doc in complete_docs:
        d=documents['selected_documents'][doc-1]; ids=frozen_ids(root,d); vectors={}; assets={}
        for condition in CONDITIONS:
            vectors[condition], assets[condition] = load_scores(root, phase_infos[(doc,condition)], d, condition, source_sha, actual_vocab, torch, R)
        require(len({a['actual_vocabulary'] for a in assets.values()}) == 1, 'three trajectories use different vocabulary dimensions')
        reference_identity=canonical({'phase':phase_infos[(doc,'FP_NATIVE')]['phase'],
                                      'score_file_sha256':assets['FP_NATIVE']['sha256'],
                                      'receipt_sha256':assets['FP_NATIVE']['phase_receipt_sha256']})
        ident={'doc_id':d['document_id'], 'document_index':doc, 'raw_text_sha256':d['raw_text_sha256'],
               'input_ids_sha256':d['artifacts']['input_ids_32768']['canonical_json_sha256'],
               'target_ids_32769_sha256':d['artifacts']['input_ids_32769']['canonical_json_sha256'],
               'source_manifest_sha256':source_sha, 'preregistration_sha256':sha(root/'preregistration.json'),
               'model_identity_receipt_sha256':manifest['model_identity_receipt_sha256'],
               'GPU_UUID':phase_infos[(doc,'FP_NATIVE')]['GPU_UUID'],
               'physical_GPU_ID':phase_infos[(doc,'FP_NATIVE')]['physical_GPU_ID'],
               'reference_trajectory_identity_sha256':reference_identity,
               'reference_score_file_sha256':assets['FP_NATIVE']['sha256']}
        arms={}
        for condition, arm in [('H_C128','A'),('H_DUAL_M','D')]:
            rows=[]
            for t in R.SCORE_POSITIONS:
                require(ids[t+1] < assets[condition]['actual_vocabulary'], 'target token outside actual vocabulary')
                row=R.score_pair(vectors['FP_NATIVE'][t],vectors[condition][t],t,ids[t+1],ident)
                row['student_condition']=condition;row['student_score_file_sha256']=assets[condition]['sha256']
                row['student_phase_receipt_sha256']=assets[condition]['phase_receipt_sha256'];rows.append(row)
            arms[arm]=rows;all_rows[(doc,arm)]=rows
        summary=R.summarize_document(arms['A'],arms['D']);summary['document_index']=doc
        annotate_summary(summary)
        summary['sum384_A']=math.fsum(r['KL_FP32'] for r in arms['A']);summary['sum384_D']=math.fsum(r['KL_FP32'] for r in arms['D'])
        require(summary['K_A']==summary['sum384_A']/384 and summary['K_D']==summary['sum384_D']/384, 'Primary is not exact fsum/384')
        require(summary['Delta']==summary['K_A']-summary['K_D'], 'Primary Delta arithmetic mismatch')
        summaries.append(summary)
        del vectors
        print(json.dumps({'CPU_postprocess_doc':doc,'status':'384_PAIRED_SCORES_AUDITED','elapsed_seconds':time.time()-start}),flush=True)
    classification=scientific_classification(summaries)
    means=document_means(summaries)
    directions=[winner_direction(s['Winner']) for s in summaries]
    result=dict(base_status,status='COMPLETE' if len(summaries)==TARGET_N else 'PARTIAL',
                DUAL_M_FULLMODEL_RESULT=classification, documents=summaries, document_means=means,
                direction_counts={key:directions.count(key) for key in ('D_WIN','A_WIN','NUMERICALLY_UNRESOLVED')},
                statistical_unit='document or independently verified project; never score positions',
                small_N_limitation='N=5 prospective replication. PROJECT_INDEPENDENCE=UNKNOWN forbids formal project-level confirmatory significance; only conditional statistical interpretation. Historical two-doc Pilot excluded from inference.',
                metric_target='Actual FP_NATIVE common target; does not establish old FP_NATIVE/FP_H equivalence',
                score_file_assets=[{k:v for k,v in a.items() if k!='full384_logit_hashes'} for a in score_assets],
                CPU_wall_seconds=time.time()-start)
    for (doc,arm),rows in all_rows.items():
        write_new(root/f'analysis/score_rows_doc{doc}_{arm}.json',rows)
    write_new(root/'analysis/SCORE_ARTIFACT_VERIFICATION.json',dict(status='PASS',assets=score_assets,full384_position_check='PASS',all_actual_full_vocab_FP32_logits_finite=True,source_manifest_sha256=source_sha,CUDA_initialized=False))
    write_new(root/'science_results.json',result)
    columns=['Doc','Document_ID','K_A','K_D','Delta','RelativeGain','Winner','Winner_direction','Delta_resolution','RelativeGain_unavailable_reason','Score_count']
    write_csv(root/'science_results.csv',columns,[dict(Doc=s['document_index'],Document_ID=s['identity']['doc_id'],Score_count=s['score_count'],**{k:s[k] for k in columns if k in s}) for s in summaries])
    window_rows=[]
    for s in summaries:
        for name,w in s['windows'].items():
            window_rows.append(dict(Doc=s['document_index'],Document_ID=s['identity']['doc_id'],Window=name,Score_count=w['score_count'],**w))
    write_csv(root/'window_results.csv',['Doc','Document_ID','Window','K_A','K_D','Delta','RelativeGain','Winner','Winner_direction','Delta_resolution','RelativeGain_unavailable_reason','Score_count'],window_rows)
    require(not torch.cuda.is_initialized(), 'CPU postprocess unexpectedly initialized CUDA')
    status=dict(base_status,status=result['status'],scientific_outputs_created=True,DUAL_M_FULLMODEL_RESULT=classification,CPU_wall_seconds=time.time()-start)
    write_new(root/'analysis/POSTPROCESS_STATUS.json',status)
    return status


def self_test():
    positions=tuple(8192+64*k for k in range(384))
    metadata={'condition':'H_C128','doc_id':'SYNTHETIC_METADATA_ONLY','consumed_tokens':32768,
              'input_ids_sha256':'INPUT','source_manifest_sha256':'SOURCE','score_positions':list(positions),
              'logits':dict.fromkeys(positions,None)}
    kwargs=dict(condition='H_C128',doc_id='SYNTHETIC_METADATA_ONLY',input_sha='INPUT',source_sha='SOURCE',positions=positions)
    validate_score_metadata(metadata,**kwargs)
    rejected=[]
    for key,value in [('consumed_tokens',32767),('doc_id','WRONG'),('source_manifest_sha256','WRONG'),
                      ('score_positions',list(positions[:-1])),('logits',dict.fromkeys(positions[:-1],None))]:
        bad=dict(metadata);bad[key]=value
        try:validate_score_metadata(bad,**kwargs)
        except ValueError:rejected.append(key)
        else:raise AssertionError('bad metadata accepted: '+key)
    assert scientific_classification([])=='INCOMPLETE_NOT_CONFIRMATORY'
    assert scientific_classification([{'Winner':'H_DUAL_M','Delta':1.0}])=='INCOMPLETE_NOT_CONFIRMATORY'
    cases=[(['H_DUAL_M']*5,[1.0]*5,'POSITIVE_PROSPECTIVE_REPLICATION'),
           (['H_DUAL_M']*4+['H_C128'],[1.0]*4+[-1.0],'DIRECTIONALLY_SUPPORTIVE_MIXED'),
           (['H_DUAL_M']*4+['NUMERICALLY_UNRESOLVED'],[1.0]*4+[0.0],'DIRECTIONALLY_SUPPORTIVE_MIXED'),
           (['H_C128']*3+['H_DUAL_M']*2,[-1.0]*3+[1.0]*2,'REPORT_NEGATIVE_EVIDENCE'),
           (['H_C128']*5,[-1.0]*5,'REPORT_NEGATIVE_EVIDENCE'),
           (['H_C128']*3+['H_DUAL_M']*2,[-0.1]*3+[1.0]*2,'MIXED_OR_INCONCLUSIVE'),
           (['H_DUAL_M']*4+['H_C128'],[0.1]*4+[-1.0],'MIXED_OR_INCONCLUSIVE'),
           (['NUMERICALLY_UNRESOLVED']*5,[0.0]*5,'MIXED_OR_INCONCLUSIVE')]
    for winners,deltas,expected in cases:
        assert scientific_classification([{'Winner':w,'Delta':d} for w,d in zip(winners,deltas)])==expected
    assert scientific_classification([{'Winner':'H_DUAL_M','Delta':1.0}]*4)=='INCOMPLETE_NOT_CONFIRMATORY'
    summary={'Winner':'H_DUAL_M','RelativeGain':None,'windows':{'Early':{'Winner':'NUMERICALLY_UNRESOLVED','RelativeGain':None}}}
    annotate_summary(summary)
    assert summary['Winner']=='H_DUAL_M' and summary['Winner_direction']=='D_WIN' and summary['RelativeGain_unavailable_reason'] is not None
    synthetic=[dict(K_A=1.0,K_D=0.8,Delta=0.2,RelativeGain=0.2) for _ in range(5)]
    means=document_means(synthetic)
    assert means['K_A']==1.0 and means['K_D']==0.8 and means['mean_absolute_Delta']==0.2 and means['mean_per_doc_RelativeGain']==0.2
    synthetic[0]['RelativeGain']=None;means=document_means(synthetic)
    assert means['mean_per_doc_RelativeGain'] is None and means['relative_gain_of_mean_K'] is None and means['mean_per_doc_RelativeGain_unavailable_reason']
    assert document_means(synthetic[:4]) is None
    uuids={str(i):'GPU-SYNTHETIC-'+str(i) for i in range(4)}
    preg={'G1':{'GPU_UUIDS':uuids},'doc_GPU_assignment':{'1':0,'2':1,'3':2,'4':3,'5':0}}
    synthetic_ledger={'GPU_UUIDS':dict(uuids)}
    assert doc_gpu_expectation(preg,synthetic_ledger,1)==(0,uuids['0'])
    assert doc_gpu_expectation(preg,synthetic_ledger,5)==(0,uuids['0'])
    assert doc_gpu_expectation(preg,synthetic_ledger,4)==(3,uuids['3'])
    bad=dict(synthetic_ledger,GPU_UUIDS=dict(uuids,**{'2':'GPU-WRONG'}))
    try:gpu_mapping(preg,bad)
    except ValueError:pass
    else:raise AssertionError('changed physical GPU UUID accepted')
    import tempfile
    with tempfile.TemporaryDirectory(prefix='qwen_N5_postprocessor_metadata_selftest_') as td:
        root=pathlib.Path(td);(root/'scores').symlink_to(SCORE_STORAGE_ROOT,target_is_directory=True)
        assert safe_path(root,'scores/metadata_only.pt')==SCORE_STORAGE_ROOT.resolve()/'metadata_only.pt'
        assert safe_path(root,'analysis/metadata_only.json')==root.resolve()/'analysis/metadata_only.json'
        (root/'other').symlink_to(SCORE_STORAGE_ROOT,target_is_directory=True)
        for relative in ('../outside','other/metadata_only.pt','/etc/passwd'):
            try:safe_path(root,relative)
            except ValueError:pass
            else:raise AssertionError('unauthorized escape accepted: '+relative)
        (root/'scores').unlink();(root/'scores').symlink_to(root.parent/'unauthorized_scores',target_is_directory=True)
        try:safe_path(root,'scores/metadata_only.pt')
        except ValueError:pass
        else:raise AssertionError('nonfrozen score symlink target accepted')
    return dict(status='PASS',scope='synthetic metadata/control only; no actual score/model/GPU or scientific outputs',
                TASK=TASK,TARGET_N=TARGET_N,invalid_metadata_rejected=rejected,
                classification_branches=len(cases)+3,raw_Winner_preserved=True,
                equal_document_means=True,null_relative_gain_with_reason=True,
                fixed_score_symlink_allowlist=True,other_path_escapes_rejected=True,
                per_doc_three_arm_GPU_binding=True,four_card_UUID_mapping_checked=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',default=str(ROOT));parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    print(json.dumps(self_test() if args.self_test else process(args.root),ensure_ascii=False),flush=True)
