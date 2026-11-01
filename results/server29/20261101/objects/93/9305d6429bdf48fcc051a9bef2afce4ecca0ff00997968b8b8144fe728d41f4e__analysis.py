"""CPU-only analysis and final sealing for the single frozen Ling Dual-M task."""
import csv
import hashlib
import json
import math
import os
import pathlib
import statistics
import time
from metrics_cpu import aggregate, resolved_difference

TASK = 'LING_KDA_ROTATION_DUAL_AXIS_FUNCTIONAL_V1'
CONDITIONS = ['FP_NATIVE', 'H_R128', 'H_DUAL_M', 'L6_R128', 'L6_DUAL_M']
METRICS = ['Delta_L6', 'Delta_H', 'Interaction', 'LearnedGain_R128', 'LearnedGain_DualM']

def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('x') as stream:
        stream.write(text); stream.flush(); os.fsync(stream.fileno())
    if path.exists():
        assert path.read_text() == text, ('FINAL_ARTIFACT_ALREADY_DIFFERS', str(path))
        temporary.unlink()
    else:
        os.link(temporary, path); temporary.unlink()

def csv_text(rows, columns):
    import io
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(row.get(k), ensure_ascii=False) if isinstance(row.get(k), (dict, list)) else row.get(k) for k in columns})
    return stream.getvalue()

def score_result(result, document, condition, uuid):
    assert result['status'] == 'PASS'
    assert result['kind'] == 'formal' and result['document'] == document and result['condition'] == condition
    assert result['UUID'] == uuid and result['begin'] == 0 and result['endpoint'] == 8448
    rows = result['scores']
    assert len(rows) == 256 and [x['position'] for x in rows] == list(range(8192, 8448))
    for row in rows:
        assert row['vocabulary_size'] == 157184
        assert row['arithmetic_scope'] == 'same_FP32_logits_scoring_only_not_forward_error_bound'
        for key in ['kl_fp32', 'kl_reference_extended', 'arithmetic_error_radius']:
            assert type(row[key]) in (int, float) and math.isfinite(row[key])
        assert row['arithmetic_error_radius'] >= 0
        assert abs(row['kl_fp32'] - row['kl_reference_extended']) <= row['arithmetic_error_radius']
    actual = aggregate(rows)
    saved = result.get('aggregate')
    assert isinstance(saved, dict) and saved == actual, 'SAVED_REDUCER_OR_REFERENCE_OR_COUNT_DIFFERS'
    if condition == 'FP_NATIVE':
        assert all(x['kl_fp32'] == x['kl_reference_extended'] == 0 for x in rows)
    return actual

def derive(values):
    out = {}
    pairs = {'Delta_L6': ('L6_R128', 'L6_DUAL_M'),
             'Delta_H': ('H_R128', 'H_DUAL_M'),
             'LearnedGain_R128': ('H_R128', 'L6_R128'),
             'LearnedGain_DualM': ('H_DUAL_M', 'L6_DUAL_M')}
    for name, (left, right) in pairs.items():
        out[name] = resolved_difference(values[left], values[right]) if left in values and right in values else None
    if out['Delta_L6'] is not None and out['Delta_H'] is not None:
        a, b = out['Delta_L6'], out['Delta_H']
        value = a['value'] - b['value']
        radius = a['arithmetic_error_radius'] + b['arithmetic_error_radius'] + 4 * math.ulp(value)
        out['Interaction'] = dict(value=value, arithmetic_error_radius=radius,
                                  label='POSITIVE' if value > radius else 'NEGATIVE' if value < -radius else 'NUMERICALLY_UNRESOLVED')
    else:
        out['Interaction'] = None
    for name, baseline in [('Delta_L6', 'L6_R128'), ('Delta_H', 'H_R128')]:
        if out[name] is not None:
            denom = values[baseline]['K']
            out[name]['relative_gain'] = out[name]['value'] / denom if denom > 0 else None
            out[name]['relative_gain_status'] = 'DEFINED' if denom > 0 else 'UNDEFINED_NONPOSITIVE_BASELINE'
    if len(values) == 5:
        students = {key: value for key, value in values.items() if key != 'FP_NATIVE'}
        best = min(students, key=lambda key: students[key]['K'])
        unresolved = [key for key in students if key != best and resolved_difference(students[key], students[best])['label'] == 'NUMERICALLY_UNRESOLVED']
        out['actual_best_student_by_point_estimate'] = best
        out['best_separation'] = 'NUMERICALLY_UNRESOLVED' if unresolved else 'RESOLVED_FROM_ALL_OTHER_STUDENTS'
        out['unresolved_with_best'] = unresolved
    else:
        out['actual_best_student_by_point_estimate'] = None
        out['best_separation'] = 'INCOMPLETE'
    return out

def local_probe_summary(result):
    """Own-history observations, deliberately not a matched-state codec effect."""
    values = []
    def collect(value):
        if isinstance(value, list):
            for item in value:
                collect(item)
        else:
            assert type(value) in (int, float) and math.isfinite(value) and value >= 0
            values.append(value)
    for row in result.get('local_diagnostics', []):
        collect(row['native_local_QDQ_relative_mse_per_head'])
    return dict(mean_local_relative_MSE=math.fsum(values)/len(values) if values else None,
                recorded_layer_head_probe_values=len(values),
                scope='Each condition own-history states at frozen probes; not the same input state across codecs; exploratory only')

def summarize(documents):
    stats = {}
    for name in METRICS:
        rows = [x['metrics'][name] for x in documents if x['metrics'][name] is not None]
        values = [x['value'] for x in rows]
        stats[name] = dict(paired_documents=len(rows), mean=math.fsum(values) / len(values) if values else None,
                           median=statistics.median(values) if values else None,
                           positive=sum(x['label'] == 'POSITIVE' for x in rows),
                           negative=sum(x['label'] == 'NEGATIVE' for x in rows),
                           numerically_unresolved=sum(x['label'] == 'NUMERICALLY_UNRESOLVED' for x in rows),
                           missing=6-len(rows))
    complete = len(documents) == 6 and all(x['complete_pairing'] for x in documents)
    primary = stats['Delta_L6']
    classification = ('INCOMPLETE' if not complete else 'PRIMARY_DIRECTIONALLY_REPLICATED' if primary['positive'] == 6
                      else 'PRIMARY_DIRECTIONALLY_NEGATIVE' if primary['negative'] == 6 else 'MIXED_OR_NUMERICALLY_UNRESOLVED')
    return dict(classification=classification, complete_six_document_pairing=complete, metrics=stats,
                statistical_unit='document; project independence UNKNOWN', project_independence='UNKNOWN',
                confirmatory_sign_test_p=None, multiplicity='Primary Delta_L6 only; secondary effects descriptive, no confirmatory tests',
                no_token_layer_head_pseudoreplication=True)

def seal(directory):
    d = pathlib.Path(directory)
    if (d/'FINAL_SEAL_RECEIPT.json').exists():
        return read(d/'FINAL_SEAL_RECEIPT.json')
    config = read(d/'PREREGISTRATION.json', {})
    frozen = read(d/'FROZEN_DOCUMENTS.json', {})
    ledger = read(d/'GPU_RESOURCE_LEDGER.json', {})
    assert not ledger.get('active_workers'), 'SEAL_REQUIRES_NO_ACTIVE_OWN_GPU_WORKERS'
    first = read(d/'FIRST_FAILURE.json')
    uuid = config.get('identity', {}).get('target_uuid', 'GPU-de53f243-b271-521c-d875-d96611ed8574')
    documents, science_rows, interaction_rows = [], [], []
    completed = 0
    errors = []
    for entry in frozen.get('formal_documents', []):
        name = entry['document']; values = {}; per_condition = {}; local = {}
        for condition in CONDITIONS:
            path = d/'formal'/f'{name}__{condition}.formal.json'
            result = read(path)
            status = 'NOT_RUN' if result is None else 'FAILED_OR_INCOMPLETE'
            if result is not None:
                try:
                    tag = f'{name}__{condition}.formal'
                    jobdir = d/'jobs'/tag
                    receipt = read(jobdir/'EXIT.json')
                    assert receipt and receipt['status'] == 'PASS' and receipt['qualified_completion'] is True
                    assert receipt['result_path'] == str(path) and receipt['result_sha256'] == sha(path)
                    assert receipt['source_manifest_sha256'] == sha(d/'SOURCE_MANIFEST.json')
                    for filename, key in [('ENV.json', 'environment_sha256'),
                                          ('ASSET_BINDING.json', 'asset_binding_sha256'),
                                          ('TERMINAL_RESOURCE.PASS.json', 'terminal_gate_sha256')]:
                        assert sha(jobdir/filename) == receipt[key], ('JOB_PROOF_BINDING_FAILED', filename)
                    assert sha(path.with_suffix('.PROFILE.json')) == receipt['profile_sha256']
                    value = score_result(result, name, condition, uuid)
                    # Match the actual fullV FP reference, tokens and scientific identity.
                    assert result['token_file_sha256'] == entry['token_file_sha256']
                    assert result['canonical_prefix_token_sha256'] == entry['canonical_prefix_token_sha256']
                    diagnostic = local_probe_summary(result) if condition != 'FP_NATIVE' else None
                    values[condition] = value; completed += 1; status = 'PASS'
                    if diagnostic is not None:
                        local[condition] = diagnostic
                except Exception as error:
                    errors.append(dict(document=name, condition=condition, error=repr(error), result_path=str(path)))
            per_condition[condition] = dict(status=status, result_path=str(path) if path.exists() else None,
                                             result_sha256=sha(path) if path.exists() else None,
                                             aggregate=values.get(condition))
            value = values.get(condition, {})
            science_rows.append(dict(document=name, condition=condition, status=status,
                                      K=value.get('K'), reference_K=value.get('reference_K'),
                                      arithmetic_error_radius=value.get('arithmetic_error_radius'),
                                      consumed_tokens=8448 if status == 'PASS' else None,
                                      scored_positions=256 if status == 'PASS' else None))
        metrics = derive(values)
        local_pairs = {}
        for label, base, dual in [('L6', 'L6_R128', 'L6_DUAL_M'), ('H', 'H_R128', 'H_DUAL_M')]:
            if base in local and dual in local and local[base]['mean_local_relative_MSE'] is not None and local[dual]['mean_local_relative_MSE'] is not None:
                delta = local[base]['mean_local_relative_MSE'] - local[dual]['mean_local_relative_MSE']
                functional = metrics['Delta_' + label]
                local_pairs[label] = dict(own_history_local_MSE_difference=delta,
                    functional_KL_difference=functional['value'],
                    directional_concordance=(delta * functional['value'] > 0) if delta != 0 and functional['label'] != 'NUMERICALLY_UNRESOLVED' else None,
                    interpretation='Descriptive own-history probe concordance only; not causal matched-state codec reconstruction evidence')
        row = dict(document=name, complete_pairing=len(values)==5,
                   conditions=per_condition, metrics=metrics, project_independence='UNKNOWN',
                   local_mechanism_observations=dict(conditions=local, own_history_codec_pairs=local_pairs))
        documents.append(row)
        flattened = dict(document=name, complete_pairing=row['complete_pairing'], actual_best_student=metrics['actual_best_student_by_point_estimate'],
                         best_separation=metrics['best_separation'])
        for name_metric in METRICS:
            metric = metrics[name_metric]
            flattened[name_metric] = metric['value'] if metric else None
            flattened[name_metric+'_label'] = metric['label'] if metric else 'NOT_RUN'
            flattened[name_metric+'_arithmetic_radius'] = metric['arithmetic_error_radius'] if metric else None
        for name_metric in ['Delta_L6', 'Delta_H']:
            flattened[name_metric+'_relative_gain'] = metrics[name_metric].get('relative_gain') if metrics[name_metric] else None
        interaction_rows.append(flattened)
    stats = summarize(documents)
    qualification = read(d/'GPU_QUALIFICATION.json', {'status':'NOT_RUN'})
    qualification_pass = str(qualification.get('status','')).startswith('PASS')
    actual_gpu_hours = ledger.get('charged_GPU_hours', ledger.get('GPU_hours_used', ledger.get('actual_GPU_hours', 0.0)))
    assert actual_gpu_hours <= 16.0, 'ABSOLUTE_GPU_BUDGET_EXCEEDED'
    final_status = 'COMPLETE_AND_STOP' if completed == 30 and len(documents) == 6 and not first and not errors else 'FAIL_CLOSED_STOP'
    status = dict(TASK_ID=TASK, FINAL_EXECUTION_STATUS=final_status,
                  SCIENTIFIC_CLASSIFICATION=stats['classification'], GPU_HOURS_USED=actual_gpu_hours,
                  FORMAL_TRAJECTORIES_COMPLETED=completed, FORMAL_TRAJECTORIES_PLANNED=30,
                  COMPLETE_DOCUMENTS=sum(x['complete_pairing'] for x in documents), DOCUMENTS_PLANNED=6,
                  NUMERICAL_QUALIFICATION_COMPLETE=qualification_pass,
                  FORMAL_SCIENCE_COMPLETE=completed==30 and not first and not errors,
                  PROJECT_INDEPENDENCE='UNKNOWN', PHYSICAL_PACKED_INT8_VALIDATED=False,
                  FIRST_FAILURE=first, CPU_ANALYSIS_ERRORS=errors)
    full = dict(status=status, documents=documents, analysis=stats,
                scoring_arithmetic_scope='FP32 logits scoring arithmetic only; not full-model forward uncertainty',
                interpretation='operational codec effects under frozen rotation and native-coordinate KDA recurrence',
                full_model_future_KL_only=True, forbidden_claims=['32K benefit','AIME benefit','reasoning accuracy improvement','deployable physical packed INT8','new rotation loss effectiveness','same mechanism as GDN'])
    write(d/'SCIENCE_RESULTS.json', full)
    write(d/'SCIENCE_RESULTS.csv', csv_text(science_rows, ['document','condition','status','K','reference_K','arithmetic_error_radius','consumed_tokens','scored_positions']))
    columns = ['document','complete_pairing','actual_best_student','best_separation']
    for name in METRICS:
        columns += [name, name+'_label', name+'_arithmetic_radius']
    columns += ['Delta_L6_relative_gain','Delta_H_relative_gain']
    write(d/'INTERACTION_ANALYSIS.csv', csv_text(interaction_rows, columns))
    write(d/'GATE_STATUS_FINAL.json', status)
    cost = dict(GPU_HOURS_USED=actual_gpu_hours, MAX_NEW_GPU_HOURS=16.0, PROTECTIVE_GPU_HOURS=15.5,
                budget_scope='New independent task only; historical Ling36GPUh and RTX4090 accounts unchanged',
                ledger=ledger, measurement_scope='CUDA peaks refer to the recorded worker reset scope; NVML is periodic sampling, not instantaneous lifetime peak')
    write(d/'RUNTIME_COST.json', cost)
    lines = ['# Ling/KDA Rotation × Dual-Axis 最终科学报告', '',
             f'TASK_ID：`{TASK}`', f'FINAL_EXECUTION_STATUS：`{final_status}`',
             f'SCIENTIFIC_CLASSIFICATION：`{stats["classification"]}`', f'GPU_HOURS_USED：`{actual_gpu_hours!r}`',
             f'ABSOLUTE_REPORT_PATH：`{d / "FINAL_SCIENTIFIC_REPORT.md"}`', '',
             f'实际完成文档 {status["COMPLETE_DOCUMENTS"]}/6；完整正式轨迹 {completed}/30。缺失是NOT_RUN/INCOMPLETE，绝不作为0或平局。', '',
             '## 冻结方法和解释范围', '',
             '原native KDA model_step、FLA、R128与H/L6矩阵保持原值。旋转只在Value侧codec坐标中进行，QDQ后逆变换回原生state，下一token消费真实写回；不在任意旋转Key坐标中保留原对角gate。',
             '数学结构已审计，但冻结矩阵与FP32往返含有限精度差异，因此仅讨论实际系统operational codec effect，不声称严格纯量化误差分解。实际无量化H/L6 canary差异见GPU_QUALIFICATION及资格worker记录。',
             '正式输入为官方PG19 validation新split，未将旧PG19 test100当未见样本；未找到可信RepoBench-P历史池，不能虚称数据源。所有候选、排除、原始SHA、U32LE8449token前缀SHA与确定性seed0排序已前瞻冻结。项目独立性UNKNOWN。',
             '每条从0建立自己的8448-token历史；评分t8192..8447，fullV157184原FP32 KL(FP_NATIVE||student)在CPU执行同运算顺序，math.fsum/256。独立扩展精度CPU参考产生本轮算术分辨范围，不套旧Ling/Qwen阈值，不是完整forward误差上界。', '',
             '## 每篇功能质量和交互', '',
             '|文档|H_R128|H_DUAL_M|L6_R128|L6_DUAL_M|Delta_L6|Delta_H|Interaction|',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for doc in documents:
        vals = [doc['conditions'][c]['aggregate'] for c in CONDITIONS[1:]]
        cells = [repr(v['K']) if v else 'NOT_RUN' for v in vals]
        cells += [repr(doc['metrics'][m]['value']) if doc['metrics'][m] else 'NOT_RUN' for m in ['Delta_L6','Delta_H','Interaction']]
        lines.append('|'+doc['document']+'|'+'|'.join(cells)+'|')
    lines += ['', '完整精度、两种LearnedGain、relative gain、mean/median/方向计数与unresolved标签见SCIENCE_RESULTS.json和INTERACTION_ANALYSIS.csv。', '',
              '## 科学判断', '',
              f'Primary：{stats["metrics"]["Delta_L6"]}。仅当6篇完整配对且可分辨Delta_L6全部正向时为PRIMARY_DIRECTIONALLY_REPLICATED。',
              f'关键Secondary Interaction：{stats["metrics"]["Interaction"]}，不升格Primary。',
              'L6+DualM是否绝对最佳须同时比较H+DualM及其分辨范围，不能从Delta_L6正向直接推出。实际每篇最佳条件与分辨标签已单列。',
              'local-QDQ MSE来自各自历史state，只是轻量探索性诊断；与Future-KL的并行变化不能被称为同输入codec因果效应或唯一KDA中介。没有 matched-state干预时不作该因果推断。',
              '每篇local_mechanism_observations列出实际probe平均local-MSE、各codec自身历史的MSE差及与功能KL方向是否一致；未记录/不可分辨时为null，不把方向一致解释成唯一中介。',
              '项目独立性UNKNOWN，因此不报告确认性sign-test p值，不将256token/layer/head当独立样本。Incomplete不回答正负科学结论。',
              '只有本次完整功能证据可以作为后续新rotation-loss研究的探索性依据；不自动授权新loss训练。没有32K、AIME、reasoning accuracy、部署packedINT8或KDA/GDN相同机制结论。', '',
              '## 资源、失败与未运行项', '',
              'FP32 persistent state载体保存的是实际QDQ写回值，不是物理packedINT8部署。INT8 codes与FP16 row/column metadata成本仅作为codec逻辑payload记录。',
              f'独立新任务预算16GPUh，15.5保护；本轮账本用量{actual_gpu_hours!r}GPUh，含模型加载、forward、评分、checkpoint、GPU占用期间IO与cleanup证明。',
              'CUDA/NVML的真实scope、峰值和unknown以RUNTIME_COST/GPU_RESOURCE_LEDGER及worker记录为准，不将周期采样冒充瞬时峰值。',
              '首失败保留原记录，未自动重试、更换GPU或降低2GiB余量。CPU准备fixture/TLS机械修正的原失败与限定变更证据保留在audits，不冒称GPU科学重试。',
              '```json', json.dumps(status, ensure_ascii=False, indent=2, allow_nan=False), '```', '',
              '**完成后STOP：未启动新实验、AIME、训练、16K/32K或git push。**', '']
    write(d/'FINAL_SCIENTIFIC_REPORT.md', '\n'.join(lines))
    files, payloads = {}, {}
    # The controller is still returning from seal(); stdout/locks are mutable
    # process plumbing. Scientific events/job/raw/EXIT evidence is already static.
    excluded = {'SHA256_MANIFEST.json','SHA256SUMS','FINAL_SEAL_RECEIPT.json','controller.log','controller.lock','runner.lock','CONTROLLER.lock'}
    for path in sorted(d.rglob('*')):
        if not path.is_file() or path.name.endswith('.partial') or path.name in excluded:
            continue
        relative = str(path.relative_to(d))
        entry = dict(bytes=path.stat().st_size, sha256=sha(path))
        first_component = pathlib.PurePosixPath(relative).parts[0]
        if first_component in ['payloads', 'checkpoints'] or (path.suffix in ['.npy','.pt','.safetensors','.bin'] and first_component != 'audits'):
            entry['evidence_role'] = 'PARTIAL_UNQUALIFIED_PAYLOAD' if '.partial' in path.name else 'SERVER_ONLY_MODEL_STATE_OR_FULLV_LOGITS_PAYLOAD'
            payloads[relative] = entry
        else:
            if path.suffix == '.pt' and first_component == 'audits':
                entry['evidence_role'] = 'CPU_TEST_FIXTURE_NOT_SCIENTIFIC_CHECKPOINT'
            files[relative] = entry
    manifest = dict(TASK_ID=TASK, status='SEALED_ACTUAL_EVIDENCE', files=files,
                    large_payloads_server_only=payloads,
                    self_exclusion='Manifest and SHA256SUMS excluded from manifest; SHA256SUMS includes manifest itself; seal receipt binds SHA list.')
    write(d/'SHA256_MANIFEST.json', manifest)
    sums = ''.join(entry['sha256']+'  '+name+'\n' for name,entry in sorted(files.items()))
    sums += sha(d/'SHA256_MANIFEST.json')+'  SHA256_MANIFEST.json\n'
    write(d/'SHA256SUMS', sums)
    receipt = dict(status='CPU_FINAL_SEALED_AND_STOP', completed_epoch=time.time(), TASK_ID=TASK,
                   FINAL_EXECUTION_STATUS=final_status, SCIENTIFIC_CLASSIFICATION=stats['classification'],
                   GPU_HOURS_USED=actual_gpu_hours, manifest_sha256=sha(d/'SHA256_MANIFEST.json'),
                   sums_sha256=sha(d/'SHA256SUMS'), small_file_count=len(files), server_only_payload_count=len(payloads),
                   report_sha256=sha(d/'FINAL_SCIENTIFIC_REPORT.md'))
    write(d/'FINAL_SEAL_RECEIPT.json', receipt)
    return receipt
