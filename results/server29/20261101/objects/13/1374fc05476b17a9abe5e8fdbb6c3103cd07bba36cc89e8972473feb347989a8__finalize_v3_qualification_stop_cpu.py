"""Append-only CPU delivery. Never launches CUDA or edits measurement evidence."""
import collections
import json
import math
import shutil
import subprocess
import time
import zipfile
from pathlib import Path
import sys

CODE = Path('/data/zypan/worktrees/ling-kda-p1-l7-fast-recovery-and-formal-32k-v3')
sys.path.insert(0, str(CODE))
from v3_common import OUT, RUNS, FREEZE, ENGINE_SHA, checked, digest, emit, ledger
from v2_common import fingerprint
from v3_guard import proc_identity

COMMIT = '31646483521ce6732be80b7cd64675b3a7b0663a'
FREEZE_SHA = '21712a6c8516661176423384ad4d9893ddc2c18acb7e9084c6fce41298bae88f'
KEYS = ('endpoint', 'fp_reference_full_logit_sha256', 'fp_replay_parity_tokens',
        'hybrid_cache_inventories', 'results', 'full_model_teacher_forced',
        'every_token_qdq_including_after_anchors', 'same_gpu_live_fp_reference')


def differences(a, b, path=''):
    if type(a) is not type(b):
        return [dict(path=path, fresh0=a, fresh1=b)]
    if isinstance(a, dict):
        out = []
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                out.append(dict(path=path+'/'+str(key), fresh0=a.get(key), fresh1=b.get(key)))
            else:
                out.extend(differences(a[key], b[key], path+'/'+str(key)))
        return out
    if isinstance(a, list):
        assert len(a) == len(b), ('LENGTH_MISMATCH', path)
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out.extend(differences(x, y, path+'/'+str(i)))
        return out
    return [] if a == b else [dict(path=path, fresh0=a, fresh1=b)]


def gone(identity):
    try:
        return proc_identity(identity['pid']) != identity
    except FileNotFoundError:
        return True


def main():
    assert not (OUT/'RESULT_AND_DELIVERY_HASHES.json').exists(), 'ALREADY_FINALIZED_NO_OVERWRITE'
    # Native results, including the native status-report bug, are not overwritten.
    before = {str(p.relative_to(OUT)): digest(p) for p in OUT.rglob('*') if p.is_file()}
    stop = checked(OUT/'V3_FAIL_CLOSED_STOP.json')
    assert 'FRESH_PROCESS_EXACT_REPEAT_FAIL' in stop['reason'] and stop['no_retry']
    freeze = checked(FREEZE)
    assert digest(FREEZE) == FREEZE_SHA and freeze['code_commit'] == COMMIT
    assert subprocess.check_output(['git', '-C', str(CODE), 'rev-parse', 'HEAD'], text=True).strip() == COMMIT
    for path, sha in freeze['code_sha256'].items():
        assert digest(path) == sha, ('FROZEN_CODE_CHANGED', path)
    assert digest(CODE/'persistent_engine_v1.py') == ENGINE_SHA
    identities = checked(OUT/'SOURCE_AND_FROZEN_SELECTION_IDENTITY.json')
    weight_count = 0
    for path, row in identities['assets'].items():
        p = Path(path)
        st = p.stat()
        assert st.st_size == row['bytes'] and st.st_mtime_ns == row['mtime_ns'], ('ASSET_METADATA_CHANGED', path)
        if p.suffix == '.safetensors':
            weight_count += 1  # Prior full SHA audit reused; no redundant weight deep audit.
        else:
            assert digest(p) == row['sha256'], ('ASSET_SHA_CHANGED', path)
    for path, sha in identities['historical_STOP_sha256'].items():
        assert digest(path) == sha, ('HISTORICAL_STOP_CHANGED', path)

    lifecycle = []
    for p in sorted(RUNS.glob('*.PROCESS_IDENTITY.json')):
        item = checked(p)
        assert gone(item['identity']), ('WORKER_STILL_ACTIVE', p.name)
        lifecycle.append(dict(source=str(p.relative_to(OUT)), identity=item['identity'],
                              no_matching_active_process=True))
    supervisor = checked(OUT/'SUPERVISOR_STARTED.json')
    assert not Path(f"/proc/{supervisor['pid']}").exists(), 'SUPERVISOR_STILL_ACTIVE'
    sessions = ['ling_p1l7_v3_pipeline']
    for p in RUNS.glob('*.LAUNCH_AUTHORIZATION.json'):
        receipt = checked(p)
        if 'tmux' in receipt:
            sessions.append(receipt['tmux'])
        if 'session' in receipt:
            sessions.append(receipt['session'])
    tmux_checks = {}
    for name in sorted(set(sessions)):
        query = subprocess.run(['tmux', 'has-session', '-t', name], capture_output=True, text=True)
        assert query.returncode == 1, ('TMUX_NOT_NATURALLY_GONE', name, query.returncode)
        tmux_checks[name] = dict(returncode=query.returncode, stderr=query.stderr)
    hardware = {}
    for gpu, uuid in [(6, 'GPU-de53f243-b271-521c-d875-d96611ed8574'),
                      (3, 'GPU-759526ed-efb2-e7f6-881c-ab098e0b05ae')]:
        physical = subprocess.check_output(['nvidia-smi', '-i', uuid,
            '--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'], text=True).strip()
        apps = subprocess.check_output(['nvidia-smi', '-i', uuid,
            '--query-compute-apps=gpu_uuid,pid,used_gpu_memory', '--format=csv,noheader,nounits'], text=True).strip()
        fields = [s.strip() for s in physical.split(',')]
        assert fields[0] == str(gpu) and fields[1] == uuid
        hardware[str(gpu)] = dict(physical=physical, compute=apps)
        # Other later jobs would not be killed; current terminal checkpoint is idle.
        assert not apps, ('NEW_EXTERNAL_OCCUPANCY_TERMINAL_CHECK', gpu)
    account = checked(OUT/'RESOURCE_USAGE_LEDGER.json')
    recalculated = ledger()
    assert account['status'] == recalculated['status'] == 'COMPLETE'
    assert account['v3_gpu_hours'] == recalculated['v3_gpu_hours']
    assert len(account['jobs']) == 6
    assert not list(RUNS.glob('micro_*.LAUNCH_AUTHORIZATION.json'))
    assert not list(RUNS.glob('formal_*.LAUNCH_AUTHORIZATION.json'))
    assert not (OUT/'FORMAL_NUMERIC_TABLES.json').exists()
    stage_points = [checked(p)['epoch'] for p in OUT.glob('V3_STAGE_STATUS_*.json') if '.sha256.' not in p.name]
    terminal_exit_bounds = []
    for j in account['jobs']:
        candidates = [x for x in stage_points if x > j['end_epoch']] + [stop['epoch']]
        exit_bound = min(x for x in candidates if x > j['end_epoch'])
        terminal_exit_bounds.append(dict(tag=j['tag'],gpu=j['gpu'],canonical_end_epoch=j['end_epoch'],
            first_subsequent_stage_or_STOP_after_supervisor_checked_worker_gone_epoch=exit_bound,
            upper_device_hours_including_receipt_to_confirmed_exit=(exit_bound-j['launch_epoch'])/3600))
    conservative_exit_upper_GPUh = sum(x['upper_device_hours_including_receipt_to_confirmed_exit'] for x in terminal_exit_bounds)

    docs, all_canon = [], []
    for doc, gpu in [('pg19-test-860', 6), ('pg19-test-5956', 3)]:
        paths = [RUNS/f'qualification_{doc}_gpu{gpu}_fresh{i}.json' for i in range(2)]
        values = [checked(p) for p in paths]
        a, b = values
        bindings = [checked(RUNS/f'qualification_{doc}_gpu{gpu}_fresh{i}.PROCESS_IDENTITY.json') for i in range(2)]
        assert bindings[0]['identity']['pid'] != bindings[1]['identity']['pid']
        for v in values:
            assert v['status'] == 'PASS' and v['endpoint'] == 512
            assert v['code_commit'] == COMMIT and v['engine_sha256'] == ENGINE_SHA
            assert v['freeze_sha256'] == FREEZE_SHA
            assert v['stream']['fp_replay_parity_tokens'] == 512
            assert fingerprint(v['stream']) == v['numeric_fingerprint']
            assert all(x['passed'] for x in v['persistent_integrity'].values())
            for r in v['runtime_telemetry']:
                assert r['deterministic_algorithms_enabled'] and not r['warn_only']
                assert not r['matmul_tf32'] and not r['cudnn_tf32']
                assert r['cublas_workspace_config'] == ':4096:8' and r['kernel_selector'] == 'BT16_W8'
            all_canon.append(v)
        dd = differences({k:a['stream'][k] for k in KEYS}, {k:b['stream'][k] for k in KEYS})
        diagnostics = [checked(RUNS/f'qualification_{doc}_gpu{gpu}_fresh{i}.REDUCER_DIAGNOSTIC.json') for i in range(2)]
        dx = differences(diagnostics[0]['samples'], diagnostics[1]['samples'])
        condition_summaries = []
        for x, y in zip(a['stream']['results'], b['stream']['results']):
            assert x['condition'] == y['condition'] and len(x['kl_trajectory']) == len(y['kl_trajectory']) == 512
            deltas = [abs(xx-yy) for xx, yy in zip(x['kl_trajectory'], y['kl_trajectory'])]
            assert all(math.isfinite(v) for v in deltas)
            changed = [i for i, v in enumerate(deltas) if v != 0]
            condition_summaries.append(dict(condition=x['condition'],
                total_qualification_token_count=512, changed_KL_scalar_count=len(changed),
                first_changed_zero_based_trajectory_index=changed[0] if changed else None,
                first_changed_teacher_forced_position=changed[0] if changed else None,
                max_absolute_KL_scalar_difference=max(deltas),
                max_difference_zero_based_position=deltas.index(max(deltas)) if changed else None,
                full_logit_sha256=[x['full_logit_sha256'], y['full_logit_sha256']],
                persistent_integrity=[x['persistent_integrity'], y['persistent_integrity']],
                qualification_only_NOT_formal_scientific_effect=True))
        docs.append(dict(document=doc, gpu=gpu, gpu_uuid=a['gpu_uuid'],
            canonical_sha256=[digest(p) for p in paths],
            canonical_sidecar_sha256=[digest(p.with_suffix('.sha256.json')) for p in paths],
            numeric_fingerprint=[v['numeric_fingerprint'] for v in values],
            recorded_fingerprints_independently_recomputed=True,
            fresh_process_512x2_status='PASS' if not dd and not dx else 'FAIL',
            per_process_mechanical_status=['PASS','PASS'],
            same_process_FP_replay_token_counts=[v['stream']['fp_replay_parity_tokens'] for v in values],
            FP_full_logit_sha256=[v['stream']['fp_reference_full_logit_sha256'] for v in values],
            FP_full_logit_fresh_process_equal=a['stream']['fp_reference_full_logit_sha256'] == b['stream']['fp_reference_full_logit_sha256'],
            fingerprint_fields_compared=list(KEYS), numeric_payload_differences=dd,
            numeric_payload_difference_count=len(dd), conditions=condition_summaries,
            cpu_same_logits_fp64_diagnostic_sample_differences=dx,
            logits_archived=False, full_logit_per_token_difference_not_recoverable_from_digest=True))
    assert docs[0]['fresh_process_512x2_status'] == 'PASS'
    assert docs[1]['fresh_process_512x2_status'] == 'FAIL'
    assert docs[1]['numeric_payload_difference_count'] == 1072
    assert all(x['changed_KL_scalar_count'] == 509 for x in docs[1]['conditions'])
    raw_guard_summary = []
    nvml_samples = []
    for p in sorted(RUNS.glob('*.GPU_EXCLUSIVITY.jsonl')):
        rows = [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
        if p.name.startswith(('guardprobe_', 'qualification_')):
            assert all(r['exclusive'] and not r['failure_conditions'] for r in rows)
            assert all(r['target_compute_pids'] == [r['binding']['pid']] for r in rows)
        for r in rows:
            if r.get('physical', {}).get('stdout'):
                fields = [s.strip() for s in r['physical']['stdout'].strip().split(',')]
                if len(fields) == 8:
                    nvml_samples.append(dict(gpu=r['gpu'], memory_used_mib=float(fields[6]), epoch=r['epoch']))
        raw_guard_summary.append(dict(file=str(p.relative_to(OUT)),sha256=digest(p),row_count=len(rows),
            exclusive_rows=sum(r['exclusive'] for r in rows),
            non_self_fd_diagnostic_rows=sum(x['classification']=='NON_SELF_FD_DIAGNOSTIC_NOT_COMPUTE_AUTHORIZATION'
                for r in rows for x in r['pid_classifications'])))
    peak_allocated = max(v['gpu_memory']['peak_allocated_bytes'] for v in all_canon)
    peak_reserved = max(v['gpu_memory']['peak_reserved_bytes'] for v in all_canon)
    analysis = dict(status='FAIL', classification='REAL_NUMERICAL_FRESH_PROCESS_REPEAT_FAILURE',
        first_failure=stop, epoch=time.time(), source_commit=COMMIT,
        CPU_only_posthoc_diagnosis_not_a_new_measurement=True,
        exact_repeat_rule_unchanged=True, metadata_only_fingerprint_failure=False,
        documents=docs, root_cause='UNDETERMINED_FROM_CURRENT_RECORDED_EVIDENCE',
        not_attributed_to_external_occupancy_or_guard=True,
        external_wait_not_identified_as_failure_cause=True,
        observation='GPU3 FP/student full-logit digests, KL scalar trajectories and states differ across fresh processes despite strict runtime flags and per-process replay PASS.',
        failed_stage_only='qualification', formal_P1_vs_L7_NOT_evaluated=True,
        no_GPU_reexecution_no_patch_no_tolerance_relaxation=True)
    emit(OUT/'V3_FRESH_PROCESS_REPEAT_FAILURE_ANALYSIS.json', analysis)
    statuses = dict(ENGINEERING_STATUS='STOP', NUMERICAL_QUALIFICATION_STATUS='FAIL',
        MICROBENCHMARK_STATUS='NOT_RUN', FORMAL_32K_STATUS='NOT_RUN',
        P1_VS_L7_SCIENTIFIC_VERDICT='NOT_EVALUATED_NO_QUALIFIED_32K_RESULT')
    correction = dict(status='FINAL', **statuses, completed_student_32k=0, completed_same_GPU_FP_32k=0,
        per_gpu_512x2={'6':'PASS','3':'FAIL'}, actual_v3_gpu_hours=account['v3_gpu_hours'],
        conservative_receipt_to_worker_exit_upper_GPUh=conservative_exit_upper_GPUh,
        GPU_cost_method='Native receipt-to-canonical ledger plus separate conservative bound through first recorded supervisor-confirmed natural exit; exact process-exit timestamps not recorded.',
        native_status_preserved_sha256=digest(OUT/'V3_FINAL_STATUS.json'),
        native_qualification_preserved_sha256=digest(OUT/'GPU_512X2_QUALIFICATION.json'),
        native_status_reporting_defect='Native terminal_report computed NOT_RUN before appending aggregate STOP placeholder; qualification actually executed and failed exact fresh-process repeat.',
        authoritative_append_only_correction=True, no_native_file_overwrite=True,
        analysis_sha256=digest(OUT/'V3_FRESH_PROCESS_REPEAT_FAILURE_ANALYSIS.json'),
        raw_guard_summary=raw_guard_summary, source_commit=COMMIT, epoch=time.time())
    emit(OUT/'V3_TERMINAL_STATUS_CORRECTION.json', correction)
    disk = shutil.disk_usage(OUT)
    audit = dict(status='PASS', scope='terminal evidence and guard repair validation, NOT numerical qualification PASS',
        CPU_regression_sha256=digest(OUT/'CPU_REGRESSION_RESULTS.json'),
        CPU_regression_status=checked(OUT/'CPU_REGRESSION_RESULTS.json')['status'],
        tiny_CUDA_probe_GPU6_and_GPU3='PASS', all_four_512_process_sampled_target_compute_guard='PASS',
        no_cross_gpu_peer_allowlist=True, shared_fd_diagnostic_only=True,
        target_NVML_context_bound_to_current_PID_starttime_boot_UID_NSpid=True,
        no_exact_historical_V2_failure_snapshot_available=True,
        V2_defect='V2 combined target NVML compute PIDs and fuser device holders, allowing only worker PID. A held FD is not by itself a target CUDA compute context.',
        historical_V2_instant_root_cause='UNDETERMINED',
        scheduling='SERIAL_GPU6_THEN_GPU3', dual_guard_and_parallel_micro_formal='NOT_RUN',
        current_physical_UUID_queries=hardware, all_bound_workers_naturally_exited=lifecycle,
        supervisor_pid=supervisor['pid'], supervisor_gone=True, tmux_terminal_checks=tmux_checks,
        frozen_code_commit=COMMIT, original_engine_sha256=ENGINE_SHA,
        frozen_code_all_hashes_PASS=True, assets_metadata_PASS_count=len(identities['assets']),
        non_weight_assets_fresh_SHA_PASS_count=len(identities['assets'])-weight_count,
        weight_shards_metadata_PASS_prior_full_SHA_reused=weight_count,
        historical_STOP_hashes=identities['historical_STOP_sha256'],
        disk_total_bytes=disk.total, disk_free_bytes=disk.free,
        maintenance_reboot_required=Path('/var/run/reboot-required').exists(),
        scheduled_shutdown=Path('/run/systemd/shutdown/scheduled').exists(),
        no_signals_used=True, final_ledger_COMPLETE=True, final_ledger_sha256=digest(OUT/'RESOURCE_USAGE_LEDGER.json'),
        CUDA_peak_allocated_bytes=peak_allocated, CUDA_peak_reserved_bytes=peak_reserved,
        sampled_NVML_memory_not_continuous_peak=nvml_samples,
        actual_GPUh_by_device={str(g):sum(j['device_hours'] for j in account['jobs'] if j['gpu']==g) for g in (6,3)},
        receipt_to_natural_worker_exit_cost_upper_GPUh=conservative_exit_upper_GPUh,
        natural_exit_cost_bound_rows=terminal_exit_bounds,
        GPU_HOURS_CAP=None, FIXED_VRAM_PERCENT_CAP=None, epoch=time.time())
    emit(OUT/'GPU_GUARD_REPAIR_VALIDATION.json', audit)

    # Portable numerical assets only. No SSH transport/credential helper is archived.
    copies = {}
    for path, row in identities['assets'].items():
        p = Path(path)
        selected = p.suffix in ('.pt', '.txt') or p.name in {
            'pg19-test-860.json','pg19-test-5956.json','PHASE_B_CALIBRATION_CONFIG.json',
            'PROSPECTIVE_DOCUMENT_AND_PAIR_SELECTION.json','PHASE_B_PAIR_METHODS_FREEZE.json',
            'PROSPECTIVE_DOCUMENT_SELECTION.json','config.json','configuration_bailing_moe_v3.py',
            'modeling_bailing_moe_v3.py','tokenizer.json','tokenizer_config.json',
            'model.safetensors.index.json','fused_recurrent.py','cache_utils.py','capture.py',
            'common.py','replay_core.py','transition.py','pytorch_kda_reference.py',
            'run_ling_kda_query_readout_causal_closure_v1.py','run_direction_causal.py',
            'run_short_horizon_proxy_v2.py','recurrent_dense_train.py','phase2_final_hidden_512_smoke.py','v4_common.py'}
        if not selected:
            continue
        target = OUT/'portable_frozen_assets'/row['sha256']/p.name
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as f:
            f.write(p.read_bytes())
        assert digest(target) == row['sha256']
        copies[path] = dict(relative_path=str(target.relative_to(OUT)), sha256=row['sha256'])
    emit(OUT/'PORTABLE_ASSET_COPY_MANIFEST.json', dict(status='PASS', files=copies,
        model_weight_shards_not_copied=True, model_weight_original_SHA_in_source_identity=True,
        measurement_code_archive_sha256=digest(OUT/'FROZEN_V3_MEASUREMENT_CODE.tar.gz')))

    answers = [
        ('V2独占检查的问题', '源码把目标GPU compute PID与fuser全部holder合并，仅允许worker自身PID，可能把非compute fd持有者误判为目标计算占用。历史失败瞬间未保存硬件快照，因此不能确认历史具体触发者。'),
        ('V3运行中独占', '按物理UUID查询NVML compute context，并绑定PID/starttime/boot_id/UID/NSpid；逐次保存raw查询、fuser与分类。四次512 worker的采样guard均通过。退出后快照不能代替运行时证据。'),
        ('最小修复是否有效', 'CPU regression及两卡tiny real CUDA probe PASS；运行时guard通过，V2 worker/runtime及数值engine字节不变。但guard有效不等于整体数值资格通过。'),
        ('串行还是并行', 'GPU6 probe→GPU6两个fresh 512→GPU3 probe→GPU3两个fresh 512，全部串行。并行轻量probe/micro/formal均未执行。GPU3启动前短暂外部占用仅等待自然释放，未发信号。'),
        ('两卡512×2资格', 'GPU6 PASS；GPU3 FAIL。GPU3两次各自512过程机械门禁及同进程FP replay通过，但fresh-process FP/student logit SHA与数值轨迹不同；整体资格FAIL，不能用各单次canonical的PASS替代。'),
        ('8448 micro', 'NOT_RUN。资格失败后未启动。'),
        ('32K真实物理可行性', 'NOT_ESTABLISHED。仅512显存记录不能证明8192/32768缓存增长可行；未执行8448或physical resource gate。没有16 GPUh/85%行政上限，也不能凭此宣称物理可行。'),
        ('4/4 student trajectories', '0/4正式32K。资格期的两条512 student不是正式科学单元。'),
        ('2条same-GPU FP', '0/2正式32K FP。资格期512 live FP和同进程replay存在，但GPU3跨fresh process FP字节不一致，不能据此批准32K。'),
        ('文档860的P1/L7 KL', '正式primary[8192,32768)与secondary[2048,32768)均未测得（missing，不是0）。不把512资格KL当作正式KL。'),
        ('文档5956的P1/L7 KL', '正式primary/secondary均未测得（missing，不是0）。资格期已有真实数值差异，仅用于失败诊断，不支持科学排名。'),
        ('两文档支持P1还是L7', '均NOT_EVALUATED；没有合格长期结果。未改变document/pair/rotation或容差，没有重跑凑结果。'),
        ('独立预测增量证据', '本轮没有新增合格32K增量证据；不推翻或扩张既有Phase A/B的有限结论，不宣称统计确认、梯度等价或训练收益。'),
        ('GPUh与显存', f"原ledger按receipt→canonical计{account['v3_gpu_hours']:.14f} GPUh，含启动/loading/JIT/probe/FP。精确process exit时刻未记录；另按下一stage/STOP已确认worker退出计保守上界{conservative_exit_upper_GPUh:.14f} GPUh，避免漏计canonical后的短暂退出开销，CPU与外部等待不计。CUDA allocated峰值{peak_allocated} bytes ({peak_allocated/2**30:.6f} GiB)，reserved {peak_reserved} bytes ({peak_reserved/2**30:.6f} GiB)。两卡分别求和，历史V1/V2成本另列。NVML只提供采样显存，不冒充连续峰值。"),
        ('是否值得做P1梯度可优化性', '本轮不足以支持直接进入梯度验证。下一项有依据的工作是单独定位GPU3 fresh-process确定性问题，需新指令批准；本轮不自动修复/retry、不训练或optimizer.step。')]
    lines = ['# Ling/KDA V3 最终资格停止报告', '',
        '本轮真实终态：GPU6完整512×2 PASS；GPU3 fresh-process exact-repeat FAIL。流水线已fail-closed，micro/32K均未启动。', '',
        '## 五项状态', '', '```json', json.dumps(statuses,ensure_ascii=False,indent=2), '```', '',
        '## 已定位的数值差异', '',
        'GPU3两次资格canonical的原数值fingerprint分别为：', '',
        '`'+docs[1]['numeric_fingerprint'][0]+'`', '', '`'+docs[1]['numeric_fingerprint'][1]+'`', '',
        '原fingerprint不含PID/时间，独立重算与记录一致。FP和两个student的full-logit SHA均不一致。两个student分别有509/512个KL scalar不同，首个差异为零基位置3；最大绝对差分别为'+
        repr(docs[1]['conditions'][0]['max_absolute_KL_scalar_difference'])+'和'+
        repr(docs[1]['conditions'][1]['max_absolute_KL_scalar_difference'])+'，均在位置218。原数值payload共1072处差异。', '',
        '这些是资格期失败诊断量，不是正式32K P1/L7效果。原逐token FP logits未归档，无法由SHA还原每token logit误差。具体kernel/硬件根因尚未确定，不能把deterministic=True当作fresh-process确定性已被证明。', '',
        '## 原状态报告的明确更正', '',
        '原生V3_FINAL_STATUS.json误记NUMERICAL_QUALIFICATION_STATUS=NOT_RUN：terminal_report先读取缺失的aggregate文件，再写STOP占位。这是状态汇报顺序缺陷，不改变数值失败。原文件与STOP永久保留；以V3_TERMINAL_STATUS_CORRECTION.json追加声明FAIL，未覆盖原证据。原生FINAL_STOP_REPORT.md中对应NOT_RUN及未冻结numerical uncertainty的概述也不能视为有效资格或已执行阈值冻结。', '',
        '## 十五问', '']
    for i,(q,a) in enumerate(answers,1):
        lines.extend([f'{i}. {q}：{a}', ''])
    lines.extend(['## 复现与封存', '', f'测量commit：`{COMMIT}`。', '',
        f'原数值engine SHA256：`{ENGINE_SHA}`。', '',
        '所有6条launch receipt、STARTED/process binding、canonical+sidecar、raw GPU_EXCLUSIVITY、日志及冻结代码archive均保留。相关非权重输入重新SHA核验；32权重分片仅新鲜metadata核验并复用已有全SHA资格，不重复全部历史深度审计。V1/V2 STOP哈希未变。', '',
        '追加交付程序仅CPU分析与打包，未修改测量HEAD、engine、worker、容差或冻结科学选择。未生成未执行的科学CSV、FP manifest、32K run manifest或空壳结果。核验所有本轮worker/supervisor自然退出后封存，未使用signal/kill。', ''])
    with (OUT/'FINAL_QUALIFICATION_STOP_REPORT.md').open('x',encoding='utf-8') as f:
        f.write('\n'.join(lines))
    emit(OUT/'FINAL_QUALIFICATION_STOP_REPORT.sha256.json', dict(
        path=str(OUT/'FINAL_QUALIFICATION_STOP_REPORT.md'),sha256=digest(OUT/'FINAL_QUALIFICATION_STOP_REPORT.md')))
    for rel,sha in before.items():
        assert digest(OUT/rel) == sha, ('PREEXISTING_EVIDENCE_CHANGED', rel)
    emit(OUT/'DELIVERY_FINALIZATION_AUDIT.json', dict(status='PASS',
        prior_evidence_files_byte_identical=len(before), all_prior_evidence_sha256=before,
        finalizer_path=str(Path(__file__)), finalizer_sha256=digest(Path(__file__)),
        source_HEAD_and_measurement_frozen=True, CPU_only=True, epoch=time.time()))
    files = {str(p.relative_to(OUT)): digest(p) for p in sorted(OUT.rglob('*')) if p.is_file()
             and p.name not in {'RESULT_AND_DELIVERY_HASHES.json','RESULT_AND_DELIVERY_HASHES.sha256.json',
                               'SHA256SUMS.txt','V3_QUALIFICATION_STOP_DELIVERY.zip','V3_QUALIFICATION_STOP_DELIVERY.zip.sha256.json'}}
    emit(OUT/'RESULT_AND_DELIVERY_HASHES.json', dict(status='FINAL_STOP_DELIVERY',
        statuses=statuses, files=files, actual_v3_gpu_hours=account['v3_gpu_hours'],
        source_commit=COMMIT, no_formal_CSV_because_not_run=True,
        manifest_self_and_checksum_and_archive_excluded_to_avoid_recursive_hashes=True,
        authoritative_status_file='V3_TERMINAL_STATUS_CORRECTION.json'))
    sums = dict(files)
    for name in ['RESULT_AND_DELIVERY_HASHES.json','RESULT_AND_DELIVERY_HASHES.sha256.json']:
        sums[name]=digest(OUT/name)
    with (OUT/'SHA256SUMS.txt').open('x',encoding='utf-8') as f:
        f.write(''.join(f'{sha}  {name}\n' for name,sha in sorted(sums.items())))
    archive = OUT/'V3_QUALIFICATION_STOP_DELIVERY.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in sorted(set(sums)|{'SHA256SUMS.txt'}):
            z.write(OUT/name,arcname=name)
    emit(OUT/'V3_QUALIFICATION_STOP_DELIVERY.zip.sha256.json',dict(
        path=str(archive),sha256=digest(archive),bytes=archive.stat().st_size,
        archived_files=len(sums)+1, file_hash_manifest_sha256=digest(OUT/'RESULT_AND_DELIVERY_HASHES.json'),
        checksum_file_sha256=digest(OUT/'SHA256SUMS.txt')))
    print(json.dumps(dict(status='FINAL_STOP_DELIVERY_SEALED',statuses=statuses,
        archive_sha256=digest(archive),archive_bytes=archive.stat().st_size,
        file_count=len(sums)+1,v3_gpu_hours=account['v3_gpu_hours'],per_GPU_hours=audit['actual_GPUh_by_device'],
        peak_allocated_GiB=peak_allocated/2**30,peak_reserved_GiB=peak_reserved/2**30)))


if __name__ == '__main__':
    main()
