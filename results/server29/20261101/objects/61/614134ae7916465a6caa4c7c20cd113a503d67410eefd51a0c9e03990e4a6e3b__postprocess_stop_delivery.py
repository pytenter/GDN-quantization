"""CPU-only terminal delivery clarification. Never import or execute GPU workers."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time
import datetime
import ast
import pathlib

sys.dont_write_bytecode = True
R = Path('/data/zypan/experiment_runs/LING_KDA_P1_L7_PROSPECTIVE_EXTENSION_4DOC_V1')
D = R / 'execution_04'
W = Path('/data/zypan/worktrees/ling-kda-prospective-execution04')
HEAD = 'd3cd88d4643a460e027cae6ba6989628c21f60bf'
REV = R / 'delivery_cpu_revisions/execution04_stop_original_seal.tar.gz'

def read(p):
    return json.loads(p.read_text())

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()

def write(p, obj):
    s = json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n'
    t = p.with_name(p.name + '.cpu_revision_tmp')
    with t.open('x') as f:
        f.write(s)
        f.flush()
        os.fsync(f.fileno())
    t.replace(p)

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
assert not (D / 'DELIVERY_CPU_CORRECTIONS.json').exists(), 'NO_REPEATED_DELIVERY_MUTATION'
assert sha(REV) == 'caae6326aabc5b6ea6ca56b9d193f5d7c7f4650ba4ec378a5259b41d27524f82'
assert sha(D/'RESULT_AND_DELIVERY_HASHES.json') == '859af8ed2a4d069bd0d426148bd75e059aea930ef0f206eadd4ef3b45ec25961'
old = read(D/'RESULT_AND_DELIVERY_HASHES.json')
for n, v in old['files'].items():
    assert sha(D/n) == v['sha256'] and (D/n).stat().st_size == v['bytes'], n
stop_sha = sha(D/'STOP.json')
stop = read(D/'STOP.json')
assert stop['no_retry'] and 'EXACT_DEVICE_RESUME_PARITY_FAILED' in stop['reason']
assert not (D/'PROSPECTIVE_COMPLETE.json').exists()
assert subprocess.check_output(['git', '-C', str(W), 'rev-parse', 'HEAD'], text=True).strip() == HEAD
assert subprocess.check_output(['git', '-C', str(W), 'status', '--porcelain'], text=True) == ''
freeze = read(D/'SOURCE_CODE_FREEZE.json')
for n, h in freeze['files'].items():
    assert sha(W/n) == h, n

# Every exact worker PID identity is gone; do not signal or terminate any process.
process_rows = []
for p in sorted(D.rglob('*.LAUNCH.json')):
    ident = read(p)['worker']
    proc = Path('/proc') / str(ident['pid'])
    current = None
    try:
        stat = (proc/'stat').read_text().rsplit(')', 1)[1].split()
        current = dict(pid=ident['pid'], uid=proc.stat().st_uid, start_ticks=int(stat[19]))
    except (OSError, ValueError):
        pass
    same = current is not None and all(current[k] == ident[k] for k in current)
    assert not same, ('WORKER_STILL_ALIVE', ident)
    process_rows.append(dict(launch=str(p.relative_to(D)), original_worker=ident,
                             same_PID_UID_starttime_alive=same, current_identity=current))
smi = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                      '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
targets = set(read(D/'EXECUTION_ROUTING.json')['authorized_devices'])
assert not any(row.split(',')[0].strip() in targets for row in smi.stdout.splitlines() if row.strip())
assert subprocess.run(['tmux', 'has-session', '-t', 'ling_prospective_execution04'], capture_output=True).returncode != 0
write(D/'TERMINAL_PROCESS_AUDIT.json', dict(status='PASS_ALL_13_OWNED_GPU_WORKERS_EXITED', epoch=time.time(),
      worker_count=len(process_rows), workers=process_rows, target_GPU_compute_processes=[],
      SMI_compute_snapshot=smi.stdout, tmux_session_exists=False, process_signals_sent=0))

fields = ['uninterrupted_suffix_full_logit_sha256', 'final_full_native_cache_sha256',
          'audit_state_fingerprint', 'checkpoint_sha256']
pa = D/'runs/pg19-test-26183__FP__parent.json'
fr = D/'runs/pg19-test-26183__FP__resumed_parity.json'
a, b = read(pa), read(fr)
cmp = {k: dict(parent=a[k], fresh=b[k], exact_equal=a[k] == b[k]) for k in fields}
assert [k for k in fields if not cmp[k]['exact_equal']] == fields[:2]
assert a['host']['gpu_uuid'] == b['host']['gpu_uuid'] == 'GPU-759526ed-efb2-e7f6-881c-ab098e0b05ae'
assert a['begin'] == 0 and b['begin'] == 8192 and a['endpoint'] == b['endpoint'] == 8256
assert a['observer_enabled'] is True and b['observer_enabled'] is False
runtime_keys = ['kernel','driver','torch','cuda_runtime','selector']
assert all(a['host'][k] == b['host'][k] for k in runtime_keys)
write(D/'NUMERICAL_PARITY_FAILURE_AUDIT.json', dict(status='FAIL_EXACT_PARITY_NO_RETRY',
      document=a['document'], rotation='FP', parent_manifest_sha256=sha(pa), fresh_manifest_sha256=sha(fr),
      STOP_sha256=stop_sha, same_GPU_UUID=True, same_reported_runtime=True,
      parent_observer_ON=True, fresh_observer_OFF=True, fields=cmp,
      FP_audit_state='NOT_APPLICABLE_NULL_IN_BOTH', worker_exit_status='BOTH_NATURAL_EXIT',
      underlying_cause='UNKNOWN',
      limits='Aggregate full-64 logits and full-native-cache fingerprints differ. No magnitude, first divergent token, specific tensor, kernel, GPU fault or causal observer attribution is established. No extra GPU run or tolerance relaxation.',
      numerical_qualification='FAIL', formal_science='NOT_RUN'))

matrix = read(D/'QUALIFICATION_MATRIX.json')
for u in matrix['units']:
    if u['document'] == 'pg19-test-26183' and u['rotation'] == 'FP':
        u.update(qualification_status='FAIL_EXACT_PARITY', target_fresh64_status='FAIL_EXACT_PARITY',
                 fresh64_worker_exit_status='PASS_NATURAL_EXIT', failure_audit='NUMERICAL_PARITY_FAILURE_AUDIT.json')
matrix.update(status='STOP_9_PASS_1_FAIL_2_NOT_RUN', failed_conditions=1, not_run_conditions=2,
              clarification='Worker execution success is distinct from parent/fresh exact qualification.')
write(D/'QUALIFICATION_MATRIX.json', matrix)

qualified = []
for q in sorted((D/'qualification_units').glob('*.json')):
    u = read(q); key = u['document']+'__'+u['rotation']
    p, f = read(D/'runs'/(key+'__parent.json')), read(D/'runs'/(key+'__resumed_parity.json'))
    assert p['status'] == f['status'] == 'PASS'
    assert all(p[k] == f[k] for k in fields)
    assert f['host']['gpu_uuid'] == u['qualified_gpu_uuid']
    assert all(p['host'][k] == f['host'][k] for k in runtime_keys)
    if u['rotation'] != 'FP':
        for z in [p, f]:
            i = z['integrity']
            assert i['passed'] and i['every_token_layer_qdq_count'] == 8256*18
            assert i['actual_next_kernel_equal_post_qdq_count'] == 8255*18
    qualified.append(dict(document=u['document'], rotation=u['rotation'], receipt_sha256=sha(q)))
assert len(qualified) == 9
checkpoints = []
for cp in read(D/'CHECKPOINT_MANIFEST.json')['checkpoints']:
    assert sha(Path(cp['path'])) == cp['sha256']
    assert Path(cp['path']).stat().st_size == cp['bytes']
    checkpoints.append(dict(path=cp['path'], sha256=cp['sha256'], qualified_GPU_UUID=cp['qualified_GPU_UUID']))
assert len(checkpoints) == 10 and checkpoints[-1]['qualified_GPU_UUID'] == 'NOT_QUALIFIED'
admissions = []
# Reapply the unchanged frozen classifier to recorded samples. Epoch spans
# include capture scheduling jitter; do not invent a new duration threshold.
namespace = dict(json=json, os=os, pathlib=pathlib)
os.environ['LING_PROSPECTIVE_RESULTS'] = str(D)
tree = ast.parse((W/'gpu_admission.py').read_text())
functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name in ['graphics_allowed','classify','classify_window']]
assert len(functions) == 3
exec(compile(ast.Module(body=functions, type_ignores=[]), str(W/'gpu_admission.py'), 'exec'), namespace)
for p in sorted((D/'admission').glob('*.window.jsonl')):
    rows = [json.loads(x) for x in p.read_text().splitlines()]
    assert len(rows) == 13
    assert namespace['classify_window'](rows) == 'AVAILABLE_VERIFIED'
    for j in rows[-3:]:
        assert j['target'][0]['uuid'] == j['uuid']
        assert float(j['target'][0]['GPU_util']) == 0
        assert float(j['target'][0]['used_MiB']) <= 64
        assert j['direct_NVML']['status'] == 'PASS'
        assert not j['direct_NVML_processes']['compute']['processes']
        assert j['direct_NVML']['memory_v2']['used'] <= 64*1024**2
    admissions.append(dict(path=str(p.relative_to(D)), samples=13,
                      snapshot_epoch_span_seconds=rows[-1]['epoch']-rows[0]['epoch'],
                      original_classifier='AVAILABLE_VERIFIED',
                      duration_scope='Original unchanged scheduled 60-second observer; snapshot epoch span is capture timing, not a new scientific gate', sha256=sha(p)))
assert len(admissions) == 13
ledger = read(D/'RESOURCE_REPORT.json')
stages = {}
for r in ledger['natural_exit_workers']:
    v = stages.setdefault(r['stage'],dict(workers=0,wall_seconds=0.0,GPU_hours=0.0))
    v['workers'] += 1; v['wall_seconds'] += r['wall_seconds']; v['GPU_hours'] += r['gpu_hours']
assert sum(x['workers'] for x in stages.values()) == 13
assert abs(sum(x['GPU_hours'] for x in stages.values())-ledger['GPU_hours_new_execution_04']) < 1e-14
stages['formal'] = dict(workers=0, wall_seconds=0.0, GPU_hours=0.0, status='NOT_RUN_NO_LAUNCH')
samples = []
for p in D.rglob('*.GPU_EXCLUSIVITY.jsonl'):
    for line in p.read_text().splitlines():
        for row in json.loads(line).get('pid_classifications', []):
            if row.get('classification') == 'VERIFIED_THIS_GPU_WORKER':
                try: samples.append(float(row['used_gpu_memory_mib']))
                except (ValueError, TypeError): pass
assert len(samples) == ledger['NVML_samples_new_execution_04']
write(D/'RESOURCE_STAGE_BREAKDOWN.json', dict(status='PASS_RECEIPT_SUM', stages=stages,
      new_execution_04_GPU_hours=ledger['GPU_hours_new_execution_04'],
      new_execution_04_NVML_sample_peak_MiB=max(samples) if samples else None,
      new_execution_04_NVML_sample_count=len(samples),
      formal_resource_plan='NOT_RUN_12_OF_12_QUALIFICATION_BARRIER_FAILED',
      scope=ledger['GPU_hours_method'], CUDA_scope=ledger['CUDA_peak_scope'], NVML_scope=ledger['NVML_scope']))
write(D/'FINAL_EVIDENCE_AUDIT.json', dict(status='PASS_ACTUAL_PARTIAL_STOP_DELIVERY', epoch=time.time(),
      source_commit=HEAD, all_frozen_source_files_unchanged=len(freeze['files']),
      qualified_conditions=qualified, exact_failures=1, not_run_conditions=2,
      checkpoints_actual_bytes_rehashed=checkpoints, worker_admissions=admissions,
      formal_FP=0, formal_student=0, Primary_384_sum_checks='NOT_RUN_NO_FORMAL_STUDENT',
      FP_payloads='NOT_RUN_NO_FORMAL_FP', frozen_selection_sha256=sha(D/'FROZEN_4DOC_SELECTION.json'),
      STOP_sha256=stop_sha, science_assets_or_first_STOP_modified=False))
machine = read(D/'FINAL_MACHINE_STATUS.json')
machine.update(FAILED_QUALIFICATION_CONDITIONS=1, NOT_RUN_QUALIFICATION_CONDITIONS=2,
               NUMERICAL_FAILURE='pg19-test-26183/FP: full-64 logits and final native cache SHA mismatch',
               ALL_OWNED_GPU_WORKERS_EXITED=True, EXTRA_GPU_RETRIES=0,
               ENGINEERING_COMPLETE_SCOPE='Frozen startup environment, true CPU launch chain, both GPU runtime canaries only')
write(D/'FINAL_MACHINE_STATUS.json', machine)

report = (D/'FINAL_SCIENTIFIC_REPORT.md').read_text()
report = report.replace('# Ling/KDA 四文档前瞻实验 qualified completion', '# Ling/KDA execution_04 最小修复与前瞻验证：终态交付')
start = report.index('## B. GPU稳定性和使用证据')
end = report.index('## C. 资格', start)
report = report[:start] + '''## B. 启动修复、真实链路测试与GPU证据

冻结源码断言和历史成功manifest共同确认三个环境要求：`PYTHONHASHSEED=0`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`、`LING_GATED_RMSNORM_FIXED_CONFIG=BT16_W8`。历史原始进程环境未完整捕获，hash seed的历史证据是冻结断言加成功worker；没有将其冒充原始env快照。本轮launcher在新解释器启动前构造env，不在worker内部补seed，未删除或改写initializer断言。

新增真实CPU链路6项全部PASS：parent/fresh64各2正例和1个缺seed负例。正式launcher同一builder、原master lane表达式和未改unit_runner实际启动子解释器，再调用冻结initializer。正例env逐项等于launcher预期、启动hash_randomization=0、全新解释器hash一致；两个负例在原断言处正确失败。CPU-only未加载模型或执行CUDA科学计算。原37项工程测试保留，13项恢复测试重新PASS；CPU合格没有代替GPU资格。

execution04实际预检文件为GPU_PREFLIGHT_RESULT_01.json和GPU_PREFLIGHT_WINDOW_01.jsonl。13个实际GPU worker各有独立原60秒、13点准入窗口，最后3点util严格0、allocated不超过64MiB、无compute占用，UUID及归属按冻结规则检查；终态CPU复核见FINAL_EVIDENCE_AUDIT.json。GPU2/3各一次runtime canary均正常退出、实际环境与启动hash seed正确、冻结initializer和CUDA初始化成功。Canary仅最小1-byte分配，不加载模型、不计数值资格或科学轨迹。

系统Xorg约4MiB保留其PID/UID/starttime及SMI/NVML图形基线。NVML v1含驱动reserved，v2 used与SMI allocated分别核对，不能混用。主机PID namespace和不可读驱动日志仍UNKNOWN。未使用非授权GPU或终止其他用户进程。

''' + report[end:]
report = report.replace('第四篇只计算其原缺3个parent及fresh。', '第四篇实际只完成FP parent及fresh各1个worker；二者正常退出但精确资格FAIL，另2个student parent/fresh均NOT_RUN。')
report = report.replace('逐条件状态见QUALIFICATION_MATRIX.json；parent不等于完整PASS。', '逐条件状态为9 PASS、1 FAIL、2 NOT_RUN，见QUALIFICATION_MATRIX.json；worker自然退出和parent完成都不等于精确资格PASS。')
report = report.replace('实际完整student逐点有限且384齐全，严格原Python sum/384（没有新求和算法）。', '本轮没有正式student，384点有限性和sum/384检查均NOT_RUN；若正式运行，必须逐点齐全并使用严格原Python sum/384。')
report = report.replace('FP真实payload见各formal manifest。', '正式FP payload为NOT_RUN，没有可交付的32K FP logits；第10个26183/FP checkpoint仅保存原始资产，未通过恢复资格。')
report = report.replace('四篇仅有限比较证据，不能证明', '四篇前瞻Primary均未运行，本轮没有P1-vs-L7科学比较结果，不能证明')
report = report.replace('若混合，后续可另行设计error propagation/readout observability研究；若方向一致，仍需更大独立预注册confirmation。建议不自动执行。', '本轮到此停止，不安排后续科学实验；任何新执行需要新的用户指令。')
report = report.replace('读取原成功、历史execution01/02 manifest与科学预注册', '读取原成功、历史execution01/02/03 manifest与科学预注册')
report = report.replace('完整精度KL、Delta见E与真实CSV，未完成值NOT_RUN。', '四篇KL、Delta和胜负均NOT_RUN；没有生成空科学CSV，未运行项不填0。')
report += '''
## CPU终态核验与首个失败字段

本轮启动工程修复已通过；随后遇到独立的数值恢复资格失败。26183/FP的parent与fresh64都在GPU3同一UUID和相同已记录runtime上运行，parent observer ON、fresh observer OFF。不能称为该文档的跨GPU误差。

| 比较字段 | parent | fresh64 | 结论 |
| --- | --- | --- | --- |
'''
for k in fields:
    v = cmp[k]
    report += f"| {k} | {v['parent']} | {v['fresh']} | {'相同' if v['exact_equal'] else '不同，FAIL'} |\n"
report += f'''
FP的audit字段均为null，表示不适用；不是额外有效state证据。两个完整指纹不同已足以触发原精确门禁，但不能推出误差大小、首个分歧token、具体tensor、kernel或硬件故障。底层原因UNKNOWN；没有额外GPU重试或新公式。首STOP北京时间{datetime.datetime.fromtimestamp(stop['epoch'], datetime.timezone(datetime.timedelta(hours=8))).isoformat()}，原STOP字节保持不变。

13个worker均自然退出：2个canary、9个GPU2 fresh64、1个GPU3 parent、1个GPU3 fresh64；worker exit无失败不表示资格无失败。12条正式32768轨迹全部NOT_RUN；384位置Primary、四篇KL/Delta及文档胜负全部NOT_RUN。正式前资源估算/交付能力门禁尚未执行，不能把资格worker的显存测量冒充正式32K资源PASS。

实际新增卡小时按stage见RESOURCE_STAGE_BREAKDOWN.json：

```json
{json.dumps(stages, indent=2, ensure_ascii=False)}
```

本轮NVML exact-owned定期采样峰值{max(samples) if samples else 'UNKNOWN'}MiB，共{len(samples)}个有效采样；RESOURCE_REPORT原NVML峰值为历史与本轮合并口径。CUDA峰值仅原loop reset scope，不包括加载全生命周期峰值。所有10个checkpoint实际字节SHA再次核验，9个旧parent没有重算；第10个不获资格标签。终态进程已退出，tmux已结束，未发送任何进程终止信号。

原自动封存报告保存在独立原始archive中（见DELIVERY_CPU_CORRECTIONS.json）。本次只澄清自动报告引用旧预检文件名、将fresh worker正常退出误作资格PASS、未运行384点措辞等CPU元数据问题。未修改worktree、原STOP、worker结果、checkpoint或任何科学门禁。当前报告覆盖原A-H及15问；机器状态如下。

```json
{json.dumps(machine, indent=2, ensure_ascii=False)}
```
'''
(D/'FINAL_SCIENTIFIC_REPORT.md').write_text(report)
changes = {n: dict(original_sha256=v['sha256'], corrected_sha256=sha(D/n))
           for n,v in old['files'].items() if sha(D/n) != v['sha256']}
assert set(changes) == {'QUALIFICATION_MATRIX.json','FINAL_MACHINE_STATUS.json','FINAL_SCIENTIFIC_REPORT.md'}
write(D/'DELIVERY_CPU_CORRECTIONS.json', dict(status='CPU_METADATA_CLARIFICATION_ONLY', epoch=time.time(),
      original_seal_archive=str(REV), original_seal_archive_sha256=sha(REV),
      original_manifest_sha256='859af8ed2a4d069bd0d426148bd75e059aea930ef0f206eadd4ef3b45ec25961',
      changed_files=changes, reason='Separate worker exit from exact qualification; report actual preflight files and NOT_RUN Primary, first failure fields, and measured resource scopes.',
      CPU_delivery_attempt_01='A newly added CPU snapshot-epoch-span>=60 assertion rejected 13 scheduled samples with capture timing jitter. It was removed in favor of the unchanged frozen classifier; original scientific admission and STOP were never changed. The three derived reports were restored from the original archive before this successful CPU-only rerun.',
      original_STOP_sha256=stop_sha, STOP_modified=False, frozen_worktree_modified=False,
      science_assets_modified=False, new_GPU_units=0, script_sha256=sha(Path(__file__))))
assert sha(D/'STOP.json') == stop_sha
for n,h in freeze['files'].items(): assert sha(W/n) == h
# Final seal: all small files and source symlinks, no model/large FP/checkpoint payload.
files = {}
for p in sorted(D.rglob('*')):
    rel = p.relative_to(D).as_posix()
    if not p.is_file() or rel in ['RESULT_AND_DELIVERY_HASHES.json','SHA256SUMS.txt']:
        continue
    if p.suffix == '.pt' or p.stat().st_size > 16*1024**2:
        continue
    if p.name.endswith('.cpu_revision_tmp'):
        continue
    files[rel] = dict(bytes=p.stat().st_size, sha256=sha(p))
manifest = dict(execution='execution_04',experiment='LING_KDA_P1_L7_PROSPECTIVE_EXTENSION_4DOC_V1',
                status='FAIL_CLOSED_STOP_CPU_DELIVERY_CLARIFIED', source_commit=HEAD, files=files,
                large_payload_binding='CHECKPOINT_MANIFEST.json; no formal FP payload was run')
write(D/'RESULT_AND_DELIVERY_HASHES.json', manifest)
lines = [f"{v['sha256']}  {n}\n" for n,v in files.items()]
lines.append(f"{sha(D/'RESULT_AND_DELIVERY_HASHES.json')}  RESULT_AND_DELIVERY_HASHES.json\n")
(D/'SHA256SUMS.txt').write_text(''.join(lines))
for n,v in files.items(): assert sha(D/n) == v['sha256']
print(json.dumps(dict(status='PASS_CPU_DELIVERY_CLARIFICATION', sealed_files=len(files),
      manifest_sha256=sha(D/'RESULT_AND_DELIVERY_HASHES.json'),STOP_sha256=stop_sha,
      new_GPU_hours=ledger['GPU_hours_new_execution_04'],stages=stages),indent=2))
