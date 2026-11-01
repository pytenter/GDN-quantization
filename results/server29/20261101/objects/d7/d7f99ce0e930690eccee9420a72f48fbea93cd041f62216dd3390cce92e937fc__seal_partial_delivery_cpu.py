"""CPU-only partial-delivery audit; scientific assets, gates and STOP stay immutable."""
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

D=Path('/data/zypan/experiment_runs/LING_KDA_P1_L7_PROSPECTIVE_EXTENSION_4DOC_V1/execution_02')
W=Path('/data/zypan/worktrees/ling-kda-prospective-engineering-recovery')
COMMIT='dfcbb7e8497a585f8ef94c0eec6d4146ade860b0'
UUID='GPU-de53f243-b271-521c-d875-d96611ed8574'
sys.path.insert(0,str(W));sys.dont_write_bytecode=True
from recovery_state import sha,atomic_json
from gpu_admission import classify,classify_window
def read(p):return json.loads(Path(p).read_text())
def put(p,j):atomic_json(p,j)
def replace_json(p,j):atomic_json(p,j,replace=True)
def replace_text(p,s):
    tmp=Path(str(p)+'.CPU_DELIVERY_REVISION_01.partial')
    with tmp.open('x') as f:f.write(s);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)

assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()==COMMIT
assert not subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True).strip()
ps=subprocess.check_output(['ps','-u','zypan','-o','args='],text=True)
assert not any(str(W/n) in line for line in ps.splitlines() for n in ['recovery_controller.py','unit_runner.py','short_worker.py','functional_worker.py','formal_worker.py','finalize_cpu.py'])
assert (D/'STOP.json').exists() and not (D/'PROSPECTIVE_COMPLETE.json').exists()
assert sha(D/'RESULT_AND_DELIVERY_HASHES.json')=='9c54da8905cac81b00a495298bdcb53821322b31c98a5c122c183dee0d924b6e'
initial=read(D/'RESULT_AND_DELIVERY_HASHES.json')
for n,info in initial['files'].items():assert sha(D/n)==info['sha256'],n
archive=D.parent/'delivery_cpu_revisions/execution_02_revision_00_original_seal.tar.gz'
assert sha(archive)=='8b5163c7a94bcf21fc56821602d1fb97a82ef53ad9de72d6fa59638b69f0bc07'
backup=D/'cpu_delivery_audit/original'
if backup.exists():
    for n in ['FINAL_SCIENTIFIC_REPORT.md','RESULT_AND_DELIVERY_HASHES.json','SHA256SUMS.txt']:assert sha(backup/n)==sha(D/n)
else:
    backup.mkdir(parents=True)
    for n in ['FINAL_SCIENTIFIC_REPORT.md','RESULT_AND_DELIVERY_HASHES.json','SHA256SUMS.txt']:shutil.copyfile(D/n,backup/n)
selection=read(D/'FROZEN_4DOC_SELECTION.json');spec=read(D/'FROZEN_SPEC.json');fsha=sha(D/'FROZEN_4DOC_SELECTION.json')
assert fsha==read(D/'SELECTION_FREEZE_RECEIPT.json')['sha256']=='a5c227d15918a249f8ae2059539c15f94668cb2afddf3aecd8a9f7e13851354b'
assert [d['document'] for d in selection['documents']]==read(D/'INPUT_ELIGIBILITY_REPRODUCTION.json')['eligible_order'][:4]
assert spec['primary']['positions']==[8192+64*k for k in range(384)]
assert len(spec['new_units'])==12 and len(selection['documents'])==4
assert not list((D/'formal').glob('*.STARTED.json')) and not list((D/'formal').glob('*.json'))
with (D/'SHORT_HORIZON_SELECTION.csv').open() as f:score_rows=list(csv.DictReader(f))
assert len(score_rows)==64
for d in selection['documents']:
    raw=D/'screening'/f"{d['document']}.json";r=read(raw)
    assert sha(raw)==d['short_result_sha256'] and r['status']=='PASS' and len(r['results'])==16
    assert r['long_horizon_outcome_generated_or_used'] is False and not r['gradients_or_optimizer_used']
    for row in (x for x in score_rows if x['document']==d['document']):
        c=row['rotation'];assert float(row['P1'])==d['scores'][c]['P1'] and float(row['L7_TOTAL_ORIGINAL'])==d['scores'][c]['L7_TOTAL_ORIGINAL']
units=[];checkpoints=[];pairs=0
for u in spec['new_units']:
    doc,c=u['document'],u['rotation'];p=D/'runs'/f'{doc}__{c}__parent.json';r=D/'runs'/f'{doc}__{c}__resumed_parity.json'
    item=dict(document=doc,rotation=c,parent_status='NOT_RUN',fresh64_status='NOT_RUN',qualification_status='NOT_RUN',formal32K_status='NOT_RUN')
    if p.exists():
        parent=read(p);assert parent['status']=='PASS' and parent['document']==doc and parent['condition']==c and parent['begin']==0 and parent['endpoint']==8256
        assert parent['observer_enabled'] is True and not parent['science_outcomes_read']
        started=read(p.with_name(p.name.replace('.json','.STARTED.json')));assert started['started']>selection['frozen_epoch']
        assert started['spec_sha256']==sha(D/'FROZEN_SPEC.json')
        cp=Path(parent['checkpoint_path']);assert cp==D/'runs'/f'{doc}__{c}__8192.pt' and sha(cp)==parent['checkpoint_sha256']
        b=parent['cache_inventory'];first=b['511']['cache_bytes'];assert set(b)=={'511','8191','8255'}
        for k,v in b.items():assert v['cache_bytes']==first+(int(k)+1-512)*61440
        for k in ['gpu_uuid','kernel','driver','torch','cuda_runtime','selector']:assert parent['host'][k]==(UUID if k=='gpu_uuid' else spec['runtime'][k])
        for rt in parent['runtime']:
            assert rt['seed']==0 and rt['torch']=='2.7.1+cu128' and rt['cuda_runtime']=='12.8' and rt['kernel_selector']=='BT16_W8'
            assert rt['deterministic_algorithms_enabled'] and not rt['warn_only'] and not rt['matmul_tf32'] and not rt['cudnn_tf32']
        if c!='FP':
            q=parent['integrity'];assert q['passed'] and q['every_token_layer_qdq_count']==8256*18 and q['actual_next_kernel_equal_post_qdq_count']==8255*18
        item.update(parent_status='PASS_COMPLETE_0_TO_8256',parent_manifest_path=str(p),parent_manifest_sha256=sha(p),checkpoint_sha256=parent['checkpoint_sha256'],qualification_status='PARENT_COMPLETE_FRESH64_NOT_RUN')
        checkpoints.append(dict(document=doc,rotation=c,path=str(cp),bytes=cp.stat().st_size,sha256=parent['checkpoint_sha256'],
            parent_manifest_path=str(p),parent_manifest_sha256=sha(p),frozen_spec_sha256=sha(D/'FROZEN_SPEC.json'),
            token_file_sha256=u['token_file']['sha256'],rotation_sha256=None if c=='FP' else u['rotation_identity']['sha256'],
            source_commit=COMMIT,GPU_UUID=UUID,own_rotation_history=True,saved_next_position=8192))
        if r.exists():
            resumed=read(r);assert resumed['status']=='PASS' and resumed['begin']==8192 and resumed['endpoint']==8256 and resumed['observer_enabled'] is False
            assert resumed['document']==doc and resumed['condition']==c and not resumed['science_outcomes_read']
            for k in ['uninterrupted_suffix_full_logit_sha256','final_full_native_cache_sha256','audit_state_fingerprint','checkpoint_sha256']:assert parent[k]==resumed[k],(doc,c,k)
            assert resumed['host']['gpu_uuid']==UUID
            if c!='FP':assert resumed['integrity']['passed'] and resumed['integrity']['every_token_layer_qdq_count']==8256*18 and resumed['integrity']['actual_next_kernel_equal_post_qdq_count']==8255*18
            item.update(fresh64_status='PASS',qualification_status='PASS_OWN_HISTORY_FRESH64_OBSERVER_EXACT',fresh_manifest_path=str(r),fresh_manifest_sha256=sha(r),
                full_logits_cache_audit_checkpoint_exact=True,observer_ON_equals_OFF=True)
            pairs+=1
    units.append(item)
assert pairs==8 and len(checkpoints)==9
assert sum(x['qualification_status']=='PARENT_COMPLETE_FRESH64_NOT_RUN' for x in units)==1
assert sum(x['qualification_status']=='NOT_RUN' for x in units)==3
put(D/'PARTIAL_LONG_HORIZON_QUALIFICATION.json',dict(status='INCOMPLETE_8_OF_12_PASS_1_PARENT_ONLY_3_NOT_RUN',planned_conditions=12,qualified_conditions=8,parent_complete=9,
    remaining_fresh64=4,units=units,science32K_units_started=0,no_tolerance_relaxation=True,source_commit=COMMIT))
put(D/'CHECKPOINT_MANIFEST.json',dict(status='PASS_ACTUAL_9_CHECKPOINT_SHA_BINDINGS',count=9,checkpoints=checkpoints,
    limitations='One parent-only checkpoint has not passed fresh64/observer restoration; no 32K qualification or formal completion claimed. Large tensors remain server-only.'))
window=D/'admission/pg19-test-7987__H_to_L7R_a25__resumed_parity.window.jsonl';samples=[json.loads(l) for l in window.read_text().splitlines()]
states=[classify(s) for s in samples];assert states==['AVAILABLE_VERIFIED']*11+['BUSY_UNATTRIBUTED','AVAILABLE_VERIFIED']
assert classify_window(samples)=='TELEMETRY_INCONSISTENT'
s=samples[11];assert s['target'][0]['GPU_util']=='1' and s['direct_NVML']['utilization']['gpu']==1 and not s['compute']['stdout'].strip()
assert s['target'][0]['used_MiB']=='15' and s['direct_NVML']['pstate']==8
diag=dict(status='GPU_ADMISSION_UNRESOLVED_STOP',controller_window_classification='TELEMETRY_INCONSISTENT',reason='LAST_THREE_SAMPLES_NOT_ALL_AVAILABLE',
    API_disagreement_at_trigger=False,SMI_utilization_percent=1,direct_NVML_utilization_percent=1,SMI_memory_used_MiB=15,
    visible_target_compute_pids=[],root_cause='UNKNOWN',Xid_log_access='UNKNOWN_PERMISSION_LIMITED',
    window_path=str(window),window_sha256=sha(window),states=states,trigger_sample=s,
    failing_next_unit=dict(document='pg19-test-7987',rotation='H_to_L7R_a25',stage='resumed_parity',worker_launched=False),
    no_GPU_reset=True,no_other_process_kill=True,no_retry=True,capacity_OOM_or_numerical_parity_failure_observed=False)
put(D/'PHYSICAL_GATE_FAILURE_DIAGNOSIS.json',diag)
put(D/'ENGINEERING_TERMINAL_STATUS.json',dict(legacy_FINAL_EXECUTION_STATUS='FAIL_CLOSED_STOP',mapped_engineering_status='GPU_ADMISSION_UNRESOLVED_STOP',
    admission_root_cause_resolved=False,GPU_changed=False,alternative_GPU_numerical_qualification='NOT_RUN',synthetic_tests_pass=24,
    recovered_old_scores=48,selected_documents=4,qualified_conditions=8,parent_only_conditions=1,not_run_conditions=3,
    formal32K_started=0,independent_directional_scientific_evidence='NOT_AVAILABLE',training_steps=0))
ledger=read(D/'RESOURCE_REPORT.json');assert ledger['new_GPU_worker_count']==18 and ledger['actual_GPU_worker_count']==21
assert ledger['GPU_hours_reused_execution_01']==2.268573776549763
assert ledger['GPU_hours_new_execution_02']==sum(x['gpu_hours'] for x in ledger['natural_exit_workers'] if not x['reused_from_execution_01'])
assert ledger['GPU_hours']==sum(x['gpu_hours'] for x in ledger['natural_exit_workers'])
assert not ledger['failed_exit_workers'] and not ledger['unreceipted_duration_upper_bounds']
clock_metadata=[]
import_audit=read(D/'CONTROLLER_RESUME_AUDIT.json')
for f in D.rglob('*.NATURAL_EXIT.json'):
    rec=read(f);rp=f.with_name(f.name.replace('.NATURAL_EXIT.json','.json'));assert sha(rp)==rec['result_sha256']
    interval=rec['completed_epoch']-rec['started_epoch']
    assert interval>=0 and rec['wall_seconds']>=0
    if rec['source_commit']==COMMIT:assert rec['wall_seconds']==interval
    else:
        assert rec['source_commit']==import_audit['original_source_commit'] and str(f.relative_to(D)) in import_audit['imported_files_sha256']
        assert rec['wall_seconds']>=interval
        clock_metadata.append(dict(path=str(f),offset_seconds=rec['wall_seconds']-interval,method='Separate original time.time reads; saved launch-to-exit wall_seconds retained unchanged'))
status=read(D/'FINAL_STATUS.json')
pairtable='\n'.join(f"|{d['document']}|{d['P1_predicted_rotation']}|{d['L7_predicted_rotation']}|{d['P1_gap']:g}|{d['L7_gap']:g}|" for d in selection['documents'])
unit_table='\n'.join(f"|{x['document']}|{x['rotation']}|{x['parent_status']}|{x['fresh64_status']}|NOT_RUN|" for x in units)
report=f'''# Ling/KDA 四文档前瞻验证：工程恢复后的实际部分交付

实验：LING_KDA_P1_L7_PROSPECTIVE_EXTENSION_4DOC_V1；执行身份：execution_02；终态：FAIL_CLOSED_STOP，工程映射：GPU_ADMISSION_UNRESOLVED_STOP。

北京时间2026-10-14 13:55:23，pg19-test-7987 / H_to_L7R_a25 的 parent 已自然退出；fresh64 worker 启动前60秒/13点准入窗口未取得最后三个连续合格空闲样本。前11点AVAILABLE_VERIFIED，第12点SMI与直接NVML均报告1% GPU utilization、显存15MiB、P8、无可见compute进程，第13点恢复0%。故窗口分类TELEMETRY_INCONSISTENT；这里指稳定性条件不足，不是两接口在触发点相互矛盾。根因UNKNOWN，不能据此认定GPU损坏、他人占用、NVML误报、OOM或数值不一致。此fresh64 worker未启动，没有重试、放宽门禁或切换GPU。

1. **正式预注册及成功manifest。** 原FUNCTIONAL_VALIDATION成功delivery与COMPLETED_ARTIFACT_AUDIT、MIXED_OUTCOME_RESOLUTION成功manifest、原NEXT_PROSPECTIVE_4DOC_PREREGISTRATION及关联freeze/scorer/model/tokenizer/rotation身份核验PASS。预注册SHA为54d6d24fbc3142a692ef1a6c04a5617cf6b0abe6702a662722f34fc1b6141722；execution_01最终manifest SHA为202bf8f64d7a2bfeab93451a607d575f7cc7052ab96692d88de02176f536516b，119个清单文件及120条SHA记录再核验，历史失败目录未改。详细原路径及验证范围见PREREGISTRATION_RECONCILIATION.json、RECOVERY_ASSET_AUDIT.json。

2. **100→58。** execution_01已按原Path.read_text语义重现100篇重编码和20历史排除/22长度排除/58合格；execution_02重新核验100原文与100 token文件SHA，复用精确的既有重编码及排序结果，没有重新抽样或改变tokenizer。原首次CPU换行审计错误与STOP永久保留。

3. **短程筛选。** 按seed0冻结顺序实际扫描4篇，每篇完整16旋转，共64条分数；前3篇48条直接由原manifest字节恢复，新GPU只评分第四篇。4篇均合格，在40篇上限内选齐。窗口256/1024、H128、18层16heads、两窗等权、原P1/L7及原midrank/pair优先规则均未改。

4. **四篇文档。** pg19-test-41603、pg19-test-2544、pg19-test-7987、pg19-test-26183；均为冻结池的新文档，不与历史两篇合并。

5. **冻结旋转对。** 全16分数及排名见SHORT_HORIZON_SELECTION.csv；全部eligible pairs、模型、旋转、输入和原评分source身份见FROZEN_4DOC_SELECTION.json。

|文档|P1预测旋转|Original L7预测旋转|P1 gap|L7 gap|
|---|---|---|---:|---:|
{pairtable}

6. **冻结时间。** 四篇共同freeze于10:44:26，SHA={fsha}，先于所有新长程资格worker。所有parent启动记录均晚于freeze；正式32K与Primary均未启动。SELECTION_FREEZE_RECEIPT.json保留原时刻，未以结局换文档或候选。

7. **资格。** 不是12/12 PASS：8条件自身0→8192 checkpoint、parent0→8256与fresh8192→8256的完整logit/cache/audit/checkpoint SHA exact以及observer ON/OFF通过；第9条件仅parent完整，fresh64因物理准入未启动；第四篇3条件全部NOT_RUN。9个真实checkpoint在服务器重新流式核验SHA并绑定，未将未资格checkpoint冒充PASS。INT8学生parent记录每token/layer QDQ 148608次、下一kernel消费148590次；量化值由FP32 tensor承载，不宣称physical resident INT8 storage。全部正式32K有限性与语义门禁尚未完成。见PARTIAL_LONG_HORIZON_QUALIFICATION.json与CHECKPOINT_MANIFEST.json。

|文档|条件|parent|fresh64|正式32K|
|---|---|---|---|---|
{unit_table}

8. **轨迹。** 正式FP 0/4，student 0/8，共0/12。9个parent和8个fresh64是工程资格worker，不是完整32K轨迹；没有正式FP full-vocabulary payload。成功资格和checkpoint保留，未运行单元标为NOT_RUN。

9. **K_P1/K_L7/Delta。** 四篇全部NOT_RUN，非0。没有实际384点Primary，故sum/384校验为NOT_RUN；没有空Primary或逐位置科学CSV。原V157184 FP32 KL(FP||student)、消费当前token后的next-token logits、8192+64k共384点至32704、完整32768及原sum/384合同不变。经验算术分辨率3.3300064494019496e-6不当作统计显著性阈值。

10. **方向计数。** 有效文档n=0；P1胜0、L7胜0、已观测tie/unresolved0、计划缺失4。无观察计数不是四篇平局，也不是任何方法胜负证据。

11. **科学解释。** INCOMPLETE_NOT_CONFIRMATORY。没有长程方向性证据，不能支持或否定P1相对L7的独立增量价值，不能宣称统计确认、可训练Loss、因果机制或reasoning/AIME改善。未来若完成验证，解释仍仅限预先定义的短程分歧且gap≥3条件分布，独立单位是文档；本轮不做确认性p-value。

12. **旧MIXED/posthoc现象。** 原860 P1胜、5956 L7胜及正式MIXED永久保留。新冻结文档2544的pair在候选身份上与旧L6-a50/L7-a50对相同，但没有新正式KL，无法验证是否复制真实长期排序；其余新对也没有正式结果。没有可识别的新预测失败文档、效应方向或大效应主导现象。旧三旋转面板仍为posthoc；未来readout与误差传播因果链NOT_IDENTIFIABLE。

13. **真实资源。** execution_02新增18个GPU worker：1个第四篇短程+17个资格（9 parent、8 fresh64），全部自然退出、失败退出0；复用execution_01的3个短程worker后研究账本21。新增{ledger['GPU_hours_new_execution_02']!r} GPUh，复用{ledger['GPU_hours_reused_execution_01']!r} GPUh，研究合计{ledger['GPU_hours']!r} GPUh。时间是worker launch-to-exit，含加载/退出，排除CPU、准入等待及闲置，不按利用率加权。CUDA allocated峰值16971451904 bytes（15.805896282196045 GiB）、reserved17855152128 bytes（16.62890625 GiB）；NVML周期采样峰值17368MiB，共313个已归属worker采样，不是连续瞬时峰值。CUDA是原loop reset及short teacher/per-condition统计scope，不宣称整个生命周期峰值。单GPU6 RTX3090 24GB、同UUID、torch2.7.1+cu128/CUDA12.8/driver580.178.04/BT16_W8/seed0，无TF32；没有人为4/16 GPUh或85%显存上限。

14. **源码与交付。** 冻结commit={COMMIT}，worktree={W}仍干净；原数学helper AST/SHA不变，新增控制/准入/退出账本/CPU交付代码及实际diff见SOURCE_DIFF.patch、NUMERICAL_SOURCE_COMPATIBILITY_AUDIT.json。24项CPU synthetic测试在Linux全PASS，包含资源故障、48分数恢复、controller锁、原子提交、成功unit幂等复用、freeze不可改写、资格失败阻止正式阶段和独立runner退出记录；这些fixture不是科学数据。服务器目录={D}；Mac交付目录=/Users/pytenter/Documents/临时项目1/ling_prospective_extension_4doc_v1/engineering_recovery_execution_02/delivery/。最终manifest及SHA256SUMS绑定全部小文件，9个checkpoint用真实绝对路径/SHA在服务器保留，不下载模型或大型张量。

15. **严格停止。** 本轮所有GPU worker、controller、runner及封存tmux已退出；首个STOP/no_retry保持原字节，没有恢复失败资格、换GPU、添加旋转、梯度/optimizer/P1+Cayley、新L8/L9、AIME/free-generation或Qwen。TRAINING_AUTHORIZATION=NOT_AUTHORIZED，训练步数0。最终交付后暂停跟进。

工程结果：结构性控制器改造、既有48分数恢复、四篇freeze、24项mock测试及8个真实自身恢复资格已完成；GPU6历史根因仍UNKNOWN，新的准入稳定性仍失败，因此不能宣称整个工程资源问题已经解决。没有实际切换GPU或取得替代GPU数值资格。允许未来恢复的必要条件是另行证明资源稳定与归属，并在独立授权执行身份下核验已有科学资产和checkpoint合同；本次不会自行重试。没有数据支持自动推进更大科学验证或P1 gradient trainability；E_future、Hidden KL和新未来功能代理均NOT_FROZEN_NOT_COMPUTED。

CPU交付修正：首次自动封存报告使用通用完成模板，未充分展开部分资格，也未直接列出已有9个checkpoint。此修订仅补齐真实部分状态、大型资产绑定、资源分类细节和中文说明；首次报告/manifest/SHA保留在cpu_delivery_audit/original/，完整原封存archive及SHA见CPU_DELIVERY_CORRECTION.json。所有原始scientific output、source、selection、gate、RESOURCE_REPORT、FINAL_STATUS及STOP字节不变。

机器状态：

```json
{json.dumps(status,ensure_ascii=False,indent=2)}
```
'''
replace_text(D/'FINAL_SCIENTIFIC_REPORT.md',report)
unchanged={n:info['sha256'] for n,info in initial['files'].items() if n!='FINAL_SCIENTIFIC_REPORT.md'}
for n,h in unchanged.items():assert sha(D/n)==h,n
put(D/'CPU_DELIVERY_CORRECTION.json',dict(status='CPU_PARTIAL_DELIVERY_AUDIT_ONLY',revision=1,epoch=time.time(),scientific_assets_or_gates_changed=False,GPU_worker_launched=False,
    original_archive_path=str(archive),original_archive_sha256=sha(archive),original_manifest_sha256=sha(backup/'RESULT_AND_DELIVERY_HASHES.json'),
    original_report_sha256=sha(backup/'FINAL_SCIENTIFIC_REPORT.md'),immutable_original_file_hashes=unchanged,
    corrected_report_sha256=sha(D/'FINAL_SCIENTIFIC_REPORT.md'),source_commit=COMMIT,qualification_counts=dict(complete=8,parent_only=1,not_run=3),actual_checkpoint_count=9,imported_receipt_clock_metadata=clock_metadata,CPU_first_metadata_audit_failure_path=str(D/'CPU_PARTIAL_AUDIT_FAILURE_01.json')))
manifest=dict(initial);manifest['CPU_delivery_revision']=dict(revision=1,correction_path='CPU_DELIVERY_CORRECTION.json',scope='CPU derived report and actual partial/checkpoint bindings only; first STOP and all science unchanged')
manifest['required_artifacts']=dict(initial['required_artifacts'],PARTIAL_LONG_HORIZON_QUALIFICATION='COMPLETE_ACTUAL_PARTIAL',CHECKPOINT_MANIFEST='COMPLETE_ACTUAL_9',PHYSICAL_GATE_FAILURE_DIAGNOSIS='COMPLETE',ENGINEERING_TERMINAL_STATUS='COMPLETE')
manifest['large_remote_payload_bindings']=dict(FP_manifest_paths=[],checkpoint_qualification_path=None,partial_qualification_path=str(D/'PARTIAL_LONG_HORIZON_QUALIFICATION.json'),checkpoint_manifest_path=str(D/'CHECKPOINT_MANIFEST.json'),checkpoint_count=9)
files=dict(initial['files'])
for f in sorted(D.rglob('*')):
    if not f.is_file() or '__pycache__' in f.parts or f.suffix=='.pt' or '.partial' in f.name:continue
    n=str(f.relative_to(D))
    if n in ['RESULT_AND_DELIVERY_HASHES.json','SHA256SUMS.txt','controller.log','controller.lock'] or f.name.endswith('.unit.lock'):continue
    files[n]=dict(bytes=f.stat().st_size,sha256=sha(f))
manifest['files']=dict(sorted(files.items()));replace_json(D/'RESULT_AND_DELIVERY_HASHES.json',manifest)
sums=''.join(v['sha256']+'  '+n+'\n' for n,v in sorted(files.items()))+sha(D/'RESULT_AND_DELIVERY_HASHES.json')+'  RESULT_AND_DELIVERY_HASHES.json\n'
replace_text(D/'SHA256SUMS.txt',sums)
check=subprocess.run(['sha256sum','-c','SHA256SUMS.txt'],cwd=D,capture_output=True,text=True);assert check.returncode==0,check.stderr
assert not subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True).strip()
print(json.dumps(dict(status='CPU_PARTIAL_DELIVERY_VERIFIED',files=len(files),SHA_records=len(files)+1,manifest_sha256=sha(D/'RESULT_AND_DELIVERY_HASHES.json'),STOP_sha256=sha(D/'STOP.json'),qualified=8,parent_only=1,checkpoints=9)))
