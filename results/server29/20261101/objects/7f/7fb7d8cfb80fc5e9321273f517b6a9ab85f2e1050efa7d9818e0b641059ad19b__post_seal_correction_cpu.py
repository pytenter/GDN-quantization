"""CPU delivery correction only; preserve the scientific STOP and all raw assets."""
import ast
import csv
import hashlib
import io
import json
import os
import pathlib
import shutil
import subprocess
import time

D = pathlib.Path('/data/zypan/experiment_runs/LING_KDA_P1_L7_PROSPECTIVE_EXTENSION_4DOC_V1/execution_01')
WT = pathlib.Path('/data/zypan/worktrees/ling-kda-p1-l7-prospective-extension-4doc-v1')
COMMIT = '7713fd20068479ccd7594b47b56fe540a93bde3f'
OUT = D

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def read(p):
    return json.loads(p.read_text())

def atomic(p, content):
    tmp = p.with_name(p.name + '.cpu_revision_01.partial')
    with tmp.open('x') as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)

def publish_text(p, content):
    assert not p.exists()
    atomic(p, content)

def js(p, value):
    atomic(p, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n')

assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=WT, text=True).strip() == COMMIT
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=WT, text=True).strip()
sessions = subprocess.run(['tmux', 'list-sessions', '-F', '#{session_name}'], capture_output=True, text=True).stdout.splitlines()
assert not any(n in sessions for n in ['ling_prospective_4doc_v1', 'ling_prospective_delivery_v1'])
ps = subprocess.check_output(['ps', '-u', 'zypan', '-o', 'args='], text=True)
assert not any(str(WT / n) in line for line in ps.splitlines() for n in ['prospective_pipeline.py', 'short_worker.py', 'functional_worker.py', 'formal_worker.py'])
assert (D / 'STOP.json').exists() and not (D / 'PROSPECTIVE_COMPLETE.json').exists()
assert not (D / 'CPU_DELIVERY_CORRECTION.json').exists()
original = read(D / 'RESULT_AND_DELIVERY_HASHES.json')
for n, info in original['files'].items():
    assert sha(D / n) == info['sha256'] and (D / n).stat().st_size == info['bytes'], n
archive = D.parent / 'delivery_cpu_revisions/revision_00_original_seal.tar.gz'
assert sha(archive) == 'dbbf7628738366a45b5075a6654547b298295f0e41a5b25c5b0a0124c3008e25'
backup = D / 'cpu_delivery_audit/original'
backup.mkdir(parents=True, exist_ok=False)
for n in ['FINAL_SCIENTIFIC_REPORT.md', 'RESULT_AND_DELIVERY_HASHES.json', 'SHA256SUMS.txt']:
    shutil.copyfile(D / n, backup / n)

decisions = sorted((read(f) for f in (D / 'screening').glob('*.SELECTION_DECISION.json')), key=lambda d: d['scan_index'])
assert len(decisions) == 3 and [d['scan_index'] for d in decisions] == [1, 2, 3]
assert [d['document'] for d in decisions] == read(D / 'INPUT_ELIGIBILITY_REPRODUCTION.json')['eligible_order'][:3]
assert all(not (D / n).exists() for n in ['FROZEN_4DOC_SELECTION.json', 'SELECTION_FREEZE_RECEIPT.json', 'ENGINEERING_QUALIFICATION_PRE_FORMAL.json', 'FOUR_DOC_PRIMARY_RESULTS.csv', 'POSITIONAL_KL.csv'])
for folder in ['formal', 'runs']:
    assert not (D / folder).exists() or not list((D / folder).glob('*.STARTED.json'))
ns = {}
exec(compile((D / 'next_prospective_pair_rule.py').read_text(), 'frozen_pair_rule', 'exec'), ns)
for d in decisions:
    r = read(D / 'screening' / (d['document'] + '.json'))
    assert sha(D / 'screening' / (d['document'] + '.json')) == d['short_result_sha256']
    assert r['status'] == 'PASS' and r['valid'] and r['all_finite'] and len(r['results']) == 16
    assert not r['gradients_or_optimizer_used'] and not r['long_horizon_outcome_generated_or_used']
    assert r['endpoint'] == 1152 and r['source_commit'] == COMMIT
    assert {x['condition']: {k: x['aggregate'][k] for k in ['P1', 'L7_FUNCTIONAL_ORIGINAL', 'L7_STATE_ORIGINAL', 'L7_TOTAL_ORIGINAL']} for x in r['results']} == d['scores']
    for x in r['results']:
        g = x['mechanical_gates']
        assert g['passed'] and g['every_token_layer_qdq_count'] == 20736
        assert g['actual_next_kernel_equal_post_qdq_count'] == g['expected_consumption_count'] == 20718
    for rt in r['runtime']:
        assert rt['seed'] == 0 and rt['torch'] == '2.7.1+cu128' and rt['cuda_runtime'] == '12.8'
        assert rt['kernel_selector'] == 'BT16_W8' and rt['deterministic_algorithms_enabled'] and not rt['warn_only']
        assert not rt['matmul_tf32'] and not rt['cudnn_tf32']
    av = [d['scores'][c]['P1'] for c in d['roster']]
    bv = [d['scores'][c]['L7_TOTAL_ORIGINAL'] for c in d['roster']]
    assert dict(zip(d['roster'], ns['midranks'](av))) == d['P1_midranks']
    assert dict(zip(d['roster'], ns['midranks'](bv))) == d['L7_midranks']
    assert ns['opposite_pairs'](d['roster'], av, bv, 3) == d['eligible_pairs']
    assert d['selected_pair'] == d['eligible_pairs'][0]

# Execute just the already frozen, pure-CPU CSV exporter, without importing any worker.
tree = ast.parse((D / 'prospective_pipeline.py').read_text())
exporter = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'export_selection_csv')
exec(compile(ast.Module(body=[exporter], type_ignores=[]), 'frozen_csv_exporter', 'exec'), globals())
export_selection_csv(decisions)
with (D / 'SHORT_HORIZON_SELECTION.csv').open() as f:
    assert len(list(csv.DictReader(f))) == 48

resource = read(D / 'RESOURCE_REPORT.json')
status = read(D / 'FINAL_STATUS.json')
guard = json.loads((D / 'CONTROLLER_GPU_ADMISSION.jsonl').read_text().splitlines()[-1])
assert not guard['exclusive'] and guard['failure_conditions'] == ['UNEXPLAINED_TARGET_MEMORY_OR_UTILIZATION']
assert guard['target_compute_pids'] == []
pair_lines = []
for d in decisions:
    p, l = d['P1_predicted_rotation'], d['L7_predicted_rotation']
    pair_lines.append(f"| {d['scan_index']} | {d['document']} | {p} | {l} | {d['P1_gap']:g} | {d['L7_gap']:g} | {d['P1_midranks'][p]:g}/{d['P1_midranks'][l]:g} | {d['L7_midranks'][p]:g}/{d['L7_midranks'][l]:g} | {len(d['eligible_pairs'])} |")
pair_table = '\n'.join(pair_lines)
report = f'''# Ling/KDA P1 vs Original L7 四文档前瞻验证：实际部分交付

实验 ID：LING_KDA_P1_L7_PROSPECTIVE_EXTENSION_4DOC_V1。终态：FAIL_CLOSED_STOP。
北京时间 2026-10-14 01:39:01，前三篇短程 worker 正常退出后，第四篇 pg19-test-26183 的启动前 GPU admission 门禁失败。记录为 UNEXPLAINED_TARGET_MEMORY_OR_UTILIZATION：GPU6 为 RTX3090，NVML 显存 15 MiB、利用率 100%，compute-process 与 fuser 查询没有发现目标进程。日志不能确定原因，不能据此认定他人占用、OOM 或数值错误。第四篇没有启动；未修改门禁、未重试、未切换 GPU。

1. **正式身份及历史 manifest。** 读取并验证原 FUNCTIONAL_VALIDATION 成功 delivery/DOCUMENT_VERDICT.json 及 COMPLETED_ARTIFACT_AUDIT、MIXED_OUTCOME_RESOLUTION 的 RESULT_AND_DELIVERY_HASHES.json，以及原 NEXT_PROSPECTIVE_4DOC_PREREGISTRATION.md 和关联 freeze、输入、scorer、rotation 资产。Phase0 为 PASS。预注册 SHA256：54d6d24fbc3142a692ef1a6c04a5617cf6b0abe6702a662722f34fc1b6141722。原成功 verdict SHA：5c49fab645e13ce7240d57b4a15388741abcbc910b51a9ea5bf177de8ee86dc4；MIXED manifest SHA：ab4b85f3cc5f3f347ff9a22aae2322b09da41cb5908a599267a3c2d29281ac95。绝对原路径、完整 source/model/input/rotation SHA 和各项验证范围见 PREREGISTRATION_RECONCILIATION.json。模型权重复用既有合格 SHA 并核对大小/mtime，其余指定资产执行 fresh SHA；不把历史权重复用称为本轮重新哈希全部权重。

2. **100→58 输入资格。** 完全复现全部 100 篇原文、完整 token 重编码、前缀 token SHA、去重与 seed=0 顺序：历史排除 20，长度不足 22，可用 58。首次新增 CPU 审计错误保留 CRLF，而原程序使用 Path.read_text 的换行语义；恢复原审计读取语义后全部 SHA 一致，原输入没有改变。根目录首次 STOP 保留，当前科学 STOP 仅为 execution_01/STOP.json。

3. **实际筛选。** 按冻结顺序完成 3 篇，每篇全部 16 个旋转，共 48 条短程分数。三篇均存在 eligible pair，数量分别为 2、6、4；第四篇启动前停止。尚未取得四篇，不能标记为前 40 篇不足四篇；这是物理 admission 中断，最大 40 的规则未改变。

4. **最终四篇。** NOT_FROZEN。只有过程记录中的 pg19-test-41603、pg19-test-2544、pg19-test-7987 三篇合格，不能把它们冒充完成的四文档 cohort，也不能把未运行的第四篇视为已选定合格文档。

5. **已完成三篇的过程 pair。** 下表完全转录原 SELECTION_DECISION.json，属于短程筛选进度，尚无四篇共同 freeze。排名从 1 开始且低分更优；两项“排名”列均按 P1 所选旋转 / L7 所选旋转顺序。

| 扫描序号 | 文档 | P1 预测旋转 | Original L7 预测旋转 | P1 gap | L7 gap | P1 排名 | L7 排名 | eligible pairs |
|---|---|---|---|---:|---:|---|---|---:|
{pair_table}

SHORT_HORIZON_SELECTION.csv 包含真实 48 行完整精度 P1、原 L7 Functional/State/Total、两套 midranks、过程预测与 gap；全部 eligible pairs、input/rotation/source 身份见对应 screening/*.SELECTION_DECISION.json 和原短程结果。CSV 的 selected_document 表示已进入筛选过程候选列表，不表示四篇共同 freeze。没有重新选择、重新计算 GPU 分数或创造 Secondary 公式。

6. **Selection freeze 与时间先后。** NOT_RUN。FROZEN_4DOC_SELECTION.json 和 SELECTION_FREEZE_RECEIPT.json 均不存在。所有新文档 long-horizon qualification、32K worker 和 Primary outcome 均为 0；因此没有在 freeze 前偷跑长程，也没有以结局调整候选。

7. **工程资格。** 不是全部 PASS。历史 engine/scorer AST/SHA、实际相同 GPU UUID/runtime 和短程 helper 检查合格；48 条短程旋转均记录真实每 token/layer QDQ 20736 次、下一 kernel 消费 20718 次，状态有限。QDQ 后量化值使用 FP32 tensor 承载，不能称为 physical resident INT8 kernel storage。本轮自己的 0→8192 checkpoint、fresh 8192→8256 exact64、observer ON/OFF 和 32K 完整状态尚未运行；历史 PASS 不能代替这次资格。前三个启动门禁通过，第四次 GPU admission FAIL；未出现本轮 OOM 或 numerical parity 失败证据。

8. **轨迹完成数。** 正式 FP 0/4，persistent student 0/8，计划总 12 条均 NOT_RUN。三个短程 FP teacher 只执行到 1152，不属于四个正式 32K FP reference。没有本轮长程 checkpoint 或正式 FP payload。

9. **K_P1 / K_L7 / Delta。** 全部 NOT_RUN，非 0。没有任何可用的 384 点 Primary，不能执行 sum/384 校验或汇总。原合同仍为 V157184、FP32 KL(FP||student)、消费 token[t] 后的 next-token logits、8192+64k 共 384 点至 32704、原 Python sum/384；经验算术分辨率 3.3300064494019496e-6 未变。没有生成空 FOUR_DOC_PRIMARY_RESULTS.csv 或 POSITIONAL_KL.csv。

10. **文档胜负。** 有效文档 n=0；P1 胜 0、L7 胜 0、已观测 tie/unresolved 0，计划缺失 4。这些零是无观察的计数，不是四篇平局或方法表现。没有 sign-test 或其他确认性统计。

11. **独立前瞻证据。** 尚未获得长程方向性证据，结论为 INCOMPLETE_NOT_CONFIRMATORY。三个短程分歧 pair 不能证明 P1 优于 L7，也不能证明 P1 无效。不能宣称独立增量预测信息、统计确认、可训练 Loss、未来 Query 完整因果链或 reasoning/AIME 改善。若未来另有完整验证，其解释也仅适用于预定义的短程分歧且 gap≥3 的条件分布，统计单位为文档。

12. **旧 MIXED / posthoc 现象。** 本轮没有正式 KL，因此无法检验是否复制旧候选的真实长期排序，也没有可识别的预测失败文档或效应大小分布。原 pg19-test-860 的 P1 胜、pg19-test-5956 的 L7 胜和正式 MIXED 永久保留；旧 L7-a50<L6-a50<L7-a25 的两文档三旋转排序仍是 posthoc exploratory。新短程 pair 不与旧面板合并作确认性样本。误差传播、历史状态方向与未来 readout observability 的因果机制仍 NOT_IDENTIFIABLE。

13. **真实资源。** 实际启动 GPU worker 3，均自然退出，失败退出 worker 0；第四次启动前 gate 失败不计为 worker。逐 worker launch-to-exit 时间含加载及退出，不含 CPU/空等，不按利用率加权：41603 为 2664.2788355350494 s，2544 为 2772.289976835251 s，7987 为 2730.296783208847 s，合计 {resource['GPU_hours']!r} GPUh。CUDA allocated 峰值 16971451904 bytes（15.805896282196045 GiB），reserved 17087594496 bytes（15.9140625 GiB）；此为短程 teacher 与每条件 reset scope 的已保存峰值。NVML 目标 worker 周期采样峰值 16636 MiB，共 114 个有效进程采样，不是连续瞬时高水位。单 GPU6 RTX3090 24GB、单缓存串行，UUID GPU-de53f243-b271-521c-d875-d96611ed8574；torch 2.7.1+cu128、CUDA 12.8、driver 580.178.04、BT16_W8、seed0、determinism 且无 TF32。完整账本见 RESOURCE_REPORT.json。没有 85% 显存或人为 GPUh cap。

14. **源码、SHA、目录。** 冻结 source commit {COMMIT}，worktree /data/zypan/worktrees/ling-kda-p1-l7-prospective-extension-4doc-v1 仍干净。服务器目录 {D}；Mac 交付目录 /Users/pytenter/Documents/临时项目1/ling_prospective_extension_4doc_v1/delivery/。SOURCE_CODE_FREEZE.json、FORMAL_CODE_FREEZE.json、CODE_FREEZE.json、MODEL_WEIGHT_SHA256.txt 以及逐短程输入/rotation manifest 绑定实际身份；RESULT_AND_DELIVERY_HASHES.json 和 SHA256SUMS.txt 绑定小型封存文件。本轮未产生大型长程张量，原模型/旋转及历史大型资产保留服务器。

15. **STOP 与授权。** 当前科学 first failure 保留且 no_retry=true。所有本轮 GPU worker、控制器和 CPU封存 tmux 均退出，没有运行后续科学任务。TRAINING_AUTHORIZATION=NOT_AUTHORIZED；梯度/optimizer 步骤 0，没有新 L8/L9、P1+Cayley、AIME/free generation、额外32K或 Qwen。既有 L8 冻结旋转只是原 16 个短程 roster 的成员，不代表新训练。最终交付后暂停当前跟进。

Phase5 补充：没有成功/失败方向数据、没有可评估的大效应文档、没有复制检验结果；P1 是否具有独立于 L7 的增量信息尚不能判断。本轮中断没有提供选择后续科研方向或启动 P1 gradient trainability 的新证据。任何资源问题调查及后续执行均需另行明确范围；本次不自动开展。E_future、Hidden KL 和新未来功能代理均 NOT_FROZEN_NOT_COMPUTED。

CPU 交付修正：首次 CPU 报告使用了完整运行模板，在部分 STOP 下引用尚不存在的 freeze/CSV 文件。此次仅修正交付说明并调用原冻结纯 CPU CSV exporter，输出实际48行；首次报告、manifest、SHA清单保留在 cpu_delivery_audit/original/，完整首次封存 archive 的路径/SHA 见 CPU_DELIVERY_CORRECTION.json。科学输入、分数、pair、源码、所有 gate、RESOURCE_REPORT、FINAL_STATUS 和当前 STOP 字节保持不变。

机器状态：

```json
{json.dumps(status, ensure_ascii=False, indent=2)}
```
'''
atomic(D / 'FINAL_SCIENTIFIC_REPORT.md', report)
unchanged = {n: info['sha256'] for n, info in original['files'].items() if n != 'FINAL_SCIENTIFIC_REPORT.md'}
for n, h in unchanged.items():
    assert sha(D / n) == h, n
correction = dict(status='CPU_DELIVERY_CORRECTION_ONLY', revision=1, epoch=time.time(),
    reason='Partial STOP report template referenced nonexistent complete-cohort files; frozen exporter was not called after pre-launch interruption.',
    original_archive_path=str(archive), original_archive_sha256=sha(archive),
    original_manifest_sha256=sha(backup / 'RESULT_AND_DELIVERY_HASHES.json'),
    original_report_sha256=sha(backup / 'FINAL_SCIENTIFIC_REPORT.md'),
    actual_partial_rows=48, completed_short_documents=3,
    science_or_gates_changed=False, GPU_work_launched=False, new_formulas=False,
    immutable_original_files_sha256=unchanged, source_commit=COMMIT,
    final_status_sha256=sha(D / 'FINAL_STATUS.json'), scientific_STOP_sha256=sha(D / 'STOP.json'),
    corrected_report_sha256=sha(D / 'FINAL_SCIENTIFIC_REPORT.md'),
    CSV_sha256=sha(D / 'SHORT_HORIZON_SELECTION.csv'),
    exact_exporter_AST_sha256=hashlib.sha256(ast.dump(exporter).encode()).hexdigest())
js(D / 'CPU_DELIVERY_CORRECTION.json', correction)
manifest = dict(original)
manifest['CPU_delivery_revision'] = dict(revision=1, correction_path='CPU_DELIVERY_CORRECTION.json',
    scope='Actual partial CSV and report only; scientific STOP, gates, raw scores, source, resource ledger and final status unchanged')
manifest['required_artifacts'] = dict(original['required_artifacts'])
manifest['required_artifacts']['SHORT_HORIZON_SELECTION.csv'] = 'COMPLETE_ACTUAL_PARTIAL_48_ROWS_3_DOCS_NOT_4DOC_FREEZE'
files = dict(original['files'])
for f in sorted(D.rglob('*')):
    if not f.is_file() or f.name in ['RESULT_AND_DELIVERY_HASHES.json', 'SHA256SUMS.txt', 'delivery_wait.log'] or f.suffix in ['.pt', '.partial'] or '__pycache__' in f.parts:
        continue
    n = str(f.relative_to(D))
    files[n] = dict(sha256=sha(f), bytes=f.stat().st_size)
# Nested originals are intentionally included even though their basenames match live seal files.
for f in backup.iterdir():
    files[str(f.relative_to(D))] = dict(sha256=sha(f), bytes=f.stat().st_size)
manifest['files'] = dict(sorted(files.items()))
js(D / 'RESULT_AND_DELIVERY_HASHES.json', manifest)
lines = [f"{info['sha256']}  {n}" for n, info in sorted(files.items())]
lines.append(f"{sha(D / 'RESULT_AND_DELIVERY_HASHES.json')}  RESULT_AND_DELIVERY_HASHES.json")
atomic(D / 'SHA256SUMS.txt', '\n'.join(lines) + '\n')
check = subprocess.run(['sha256sum', '-c', 'SHA256SUMS.txt'], cwd=D, capture_output=True, text=True)
assert check.returncode == 0, check.stdout + check.stderr
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=WT, text=True).strip()
print(json.dumps(dict(status='CPU_DELIVERY_REVISION_01_VERIFIED', sealed_files=len(files), SHA_records=len(lines), manifest_SHA=sha(D / 'RESULT_AND_DELIVERY_HASHES.json'), report_SHA=sha(D / 'FINAL_SCIENTIFIC_REPORT.md'), STOP_SHA=sha(D / 'STOP.json'))))
