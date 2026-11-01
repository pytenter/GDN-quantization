"""CPU-only delivery after natural pipeline exit; no additional GPU work."""
import collections
import csv
import hashlib
import io
import json
import math
import os
import pathlib
import subprocess
import sys
import time

D=pathlib.Path(os.environ['LING_PROSPECTIVE_RESULTS'])
OLD=pathlib.Path('/data/zypan/experiment_runs/LING_KDA_P1_L7_MIXED_OUTCOME_RESOLUTION_V1')
WT=pathlib.Path('/data/zypan/worktrees/ling-kda-p1-l7-prospective-extension-4doc-v1')
TASK='LING_KDA_P1_L7_PROSPECTIVE_EXTENSION_4DOC_V1'
sys.dont_write_bytecode=True

def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def text(p,s):
    tmp=p.with_name(p.name+'.partial')
    with tmp.open('x') as f:f.write(s);f.flush();os.fsync(f.fileno())
    os.link(tmp,p);tmp.unlink()
def publish(p,j):text(p,json.dumps(j,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n')
def export(p,rows):
    assert rows,'NO_FAKE_EMPTY_SCIENCE_CSV'
    b=io.StringIO();w=csv.DictWriter(b,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);text(p,b.getvalue())

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU_ONLY_REQUIRED'
    complete=(D/'PROSPECTIVE_COMPLETE.json').exists();stop=(D/'STOP.json').exists()
    assert complete!=stop,'NEEDS_UNAMBIGUOUS_TERMINAL_RECEIPT'
    assert subprocess.run(['tmux','has-session','-t','ling_prospective_4doc_v1'],capture_output=True).returncode!=0,'WRITERS_STILL_ACTIVE'
    freeze=read(D/'SOURCE_CODE_FREEZE.json');head=freeze['source_commit']
    assert subprocess.check_output(['git','-C',str(WT),'rev-parse','HEAD'],text=True).strip()==head
    assert subprocess.check_output(['git','-C',str(WT),'status','--porcelain'],text=True)==''
    for n,h in freeze['files'].items():assert sha(WT/n)==h,n
    a=read(D/'PREREGISTRATION_RECONCILIATION.json');assert a['status']=='PASS'
    for n,h in a['frozen_protocol_files'].items():assert sha(OLD/n)==h,n
    for name,h in a['historical_STOP_preserved'].items():assert sha(pathlib.Path(name))==h,name
    old_failure=read(D.parent/'CPU_AUDIT_CORRECTION.json')
    assert sha(pathlib.Path(old_failure['first_failure_path']))==old_failure['first_failure_sha256']
    for target,source in [('CPU_AUDIT_CORRECTION.json',D.parent/'CPU_AUDIT_CORRECTION.json'),
                          ('INITIAL_PHASE0_AUDITOR_STOP.json',D.parent/'STOP.json')]:
        text(D/target,source.read_text())
    receipts=[];all_results={};failed=[]
    for f in sorted(D.rglob('*.NATURAL_EXIT.json')):
        r=read(f);assert r['natural_exit'] and r['source_commit']==head
        result_path=f.with_name(f.name.replace('.NATURAL_EXIT.json','.json'))
        assert sha(result_path)==r['result_sha256']
        result=read(result_path);assert result['status']=='PASS'
        all_results[r['tag']]=result
        mem=result.get('memory',[])
        receipts.append(dict(tag=r['tag'],stage=r.get('stage'),wall_seconds=r['wall_seconds'],gpu_hours=r['wall_seconds']/3600,
            receipt_sha256=sha(f),result_sha256=r['result_sha256'],
            peak_allocated=max((m['peak_allocated'] for m in mem),default=0),
            peak_reserved=max((m['peak_reserved'] for m in mem),default=0)))
    for f in sorted(D.rglob('*.FAILED_EXIT.json')):
        r=read(f);failed.append(dict(tag=r['tag'],returncode=r['returncode'],wall_seconds=r['wall_seconds'],gpu_hours=r['wall_seconds']/3600,receipt_sha256=sha(f)))
    unreceipted=[]
    if stop:
        failure=read(D/'STOP.json');end=failure.get('stopped_epoch',failure.get('epoch'))
        for f in sorted(D.rglob('*.STARTED.json')):
            stem=f.name.removesuffix('.STARTED.json')
            if (f.parent/(stem+'.NATURAL_EXIT.json')).exists() or (f.parent/(stem+'.FAILED_EXIT.json')).exists():continue
            r=read(f);begin=r.get('started_epoch',r.get('started'))
            if begin is not None and end is not None:
                unreceipted.append(dict(tag=stem,started_record_sha256=sha(f),GPU_hours_upper_bound=max(0,end-begin)/3600,
                    method='worker STARTED through first STOP; no exact natural/failed-exit receipt, excluded from exact receipted total'))
    samples=[]
    for f in D.rglob('*.GPU_EXCLUSIVITY.jsonl'):
        for line in f.read_text().splitlines():
            j=json.loads(line)
            if complete:assert j['exclusive'],j['failure_conditions']
            for row in j.get('pid_classifications',[]):
                if row.get('classification')=='VERIFIED_THIS_GPU_WORKER':
                    value=row['used_gpu_memory_mib']
                    if str(value).replace('.','',1).isdigit():samples.append(float(value))
    ledger=dict(status='COMPLETE' if complete else 'STOP_PARTIAL',experiment=TASK,source_commit=head,
        actual_GPU_worker_count=len(receipts)+len(failed)+len(unreceipted),natural_exit_worker_count=len(receipts),failed_exit_worker_count=len(failed),
        GPU_hours=sum(w['gpu_hours'] for w in receipts)+sum(w['gpu_hours'] for w in failed),
        GPU_hours_method='serial worker launch to natural/failed exit wall time, including model loading and context teardown; CPU and controller waits excluded; not utilization-weighted kernel time',
        peak_CUDA_allocated=max((w['peak_allocated'] for w in receipts),default=0),peak_CUDA_reserved=max((w['peak_reserved'] for w in receipts),default=0),
        CUDA_peak_scope='original loop reset statistics for qualification/formal; short teacher and per-condition peaks separately retained; failed worker peak may be unavailable',
        NVML_sample_peak_MiB=max(samples) if samples else None,NVML_samples=len(samples),NVML_scope='periodic exact-UUID verified process samples, not continuous instantaneous peak',
        GPU_UUID='GPU-de53f243-b271-521c-d875-d96611ed8574',physical_GPU=6,hardware='single RTX3090 24GB; single-cache serial',
        natural_exit_workers=receipts,failed_exit_workers=failed,unreceipted_duration_upper_bounds=unreceipted,
        no_GPUh_cap=True,no_85percent_gate=True,training_steps=0,
        STOP=read(D/'STOP.json') if stop else None)
    publish(D/'RESOURCE_REPORT.json',ledger)
    selection=read(D/'FROZEN_4DOC_SELECTION.json') if (D/'FROZEN_4DOC_SELECTION.json').exists() else None
    rows=[];position_rows=[];unit_ids=[];counts=collections.Counter();fp_complete=student_complete=0
    if selection:
        receipt=read(D/'SELECTION_FREEZE_RECEIPT.json');assert sha(D/'FROZEN_4DOC_SELECTION.json')==receipt['sha256']
        assert receipt['long_GPU_workers_launched']==0
        spec=read(D/'FROZEN_SPEC.json');positions=[8192+64*k for k in range(384)]
        assert spec['primary']['positions']==positions and spec['resolution']['threshold']==3.3300064494019496e-6
        qualification=read(D/'ENGINEERING_QUALIFICATION_PRE_FORMAL.json') if (D/'ENGINEERING_QUALIFICATION_PRE_FORMAL.json').exists() else None
        if qualification:
            assert qualification['status']=='PASS' and len(qualification['units'])==12
            for unit in qualification['units']:
                doc,rot=unit['document'],unit['rotation']
                parent=read(D/'runs'/f'{doc}__{rot}__parent.json');resumed=read(D/'runs'/f'{doc}__{rot}__resumed_parity.json')
                for key in ['uninterrupted_suffix_full_logit_sha256','final_full_native_cache_sha256','audit_state_fingerprint','checkpoint_sha256']:
                    assert parent[key]==resumed[key],(doc,rot,key)
                assert sha(pathlib.Path(parent['checkpoint_path']))==parent['checkpoint_sha256']
        if complete:
            import torch
            assert not torch.cuda.is_initialized()
        for d in selection['documents']:
            doc=d['document'];token_file=pathlib.Path(d['document_identity']['token_file_path']);tokens=read(token_file)
            paths={c:D/'formal'/f'{doc}__{c}__formal.json' for c in ['FP',d['P1_predicted_rotation'],d['L7_predicted_rotation']]}
            results={c:read(f) for c,f in paths.items() if f.exists()}
            if complete:assert len(results)==3
            for condition,j in results.items():
                assert j['status']=='PASS' and j['all_finite'] and j['endpoint']==32768
                assert j['sample_positions']==positions and j['vocabulary_dimension']==157184
                assert j['token_sha256']==sha(token_file) and j['primary_spec_sha256']==sha(D/'FROZEN_SPEC.json')
                assert j['formal_observer_uninterrupted_parity_PASS'] and j['observer_qualified_before_first_primary_reduction']
                assert j['final_cache_inventory']['cache_bytes']==2033025024
                assert j['gpu_uuid']==ledger['GPU_UUID']
                for rt in j['runtime']:
                    assert rt['torch']==spec['runtime']['torch'] and rt['cuda_runtime']==spec['runtime']['cuda_runtime']
                    assert rt['kernel_selector']==spec['runtime']['selector'] and rt['seed']==0
                    assert rt['deterministic_algorithms_enabled'] and not rt['warn_only'] and not rt['matmul_tf32'] and not rt['cudnn_tf32']
                assert j['E_future']==j['Hidden_KL']=='NOT_FROZEN_NOT_COMPUTED'
                parent=read(D/'runs'/f'{doc}__{condition}__parent.json')
                assert j['checkpoint_sha256']==parent['checkpoint_sha256']
                assert j['started_epoch']>selection['frozen_epoch']
                if condition=='FP':
                    fp_complete+=1;assert len([r for r in j['references'] if r['logits']])==384
                    assert [r['position'] for r in j['references']]==sorted(set(positions)|{8192,16383,24575,32767})
                    digest=hashlib.sha256()
                    for ref in j['references']:
                        path=pathlib.Path(ref['path']);assert sha(path)==ref['sha256'] and path.stat().st_size==ref['bytes']
                        if complete:
                            payload=torch.load(path,map_location='cpu',weights_only=False)
                            assert payload['position']==ref['position'] and payload['token']==tokens[ref['position']]
                            if ref['logits']:
                                logits=payload['logits'];assert tuple(logits.shape)==(1,1,157184) and logits.dtype==torch.float32
                                assert bool(torch.isfinite(logits).all())
                                digest.update(ref['position'].to_bytes(8,'little'));digest.update(logits.detach().contiguous().numpy().tobytes())
                            del payload
                    if complete:assert digest.hexdigest()==j['sampled_logits_content_sha256']
                else:
                    student_complete+=1;v=j['primary_samples']
                    assert [x['position'] for x in v]==positions and all(math.isfinite(x['KL']) and x['vocabulary_dimension']==157184 for x in v)
                    assert sum(x['KL'] for x in v)/384==j['primary_mean_KL']
                    q=j['persistent_integrity'];assert q['passed'] and q['every_token_layer_qdq_count']==32768*18 and q['actual_next_kernel_equal_post_qdq_count']==32767*18
                    assert set(j['state_diagnostics'])=={'8192','16383','24575','32767'}
                    for x in v:position_rows.append(dict(document=doc,rotation=condition,position=x['position'],token=tokens[x['position']],KL=x['KL'],vocabulary_dimension=157184,token_file_sha256=sha(token_file),formal_result_sha256=sha(paths[condition])))
                unit_ids.append(dict(document=doc,condition=condition,status='COMPLETE',formal_result_path=str(paths[condition]),formal_result_sha256=sha(paths[condition]),
                    document_sha256=d['document_identity']['raw_sha256'],token_file_sha256=sha(token_file),rotation_sha256=None if condition=='FP' else selection['rotation_identities'][condition]['sha256'],
                    model_identity=spec['model'],source_commit=head,runtime=j['runtime'],GPU_UUID=j['gpu_uuid'],checkpoint_sha256=j['checkpoint_sha256'],
                    scoring_contract_sha256=sha(D/'FROZEN_SPEC.json'),FP_reference_sha256=sha(paths['FP']) if paths['FP'].exists() else None))
            if len(results)!=3:counts['MISSING']+=1;continue
            p,l=d['P1_predicted_rotation'],d['L7_predicted_rotation']
            kp,kl=results[p]['primary_mean_KL'],results[l]['primary_mean_KL'];delta=kp-kl;threshold=spec['resolution']['threshold']
            winner='P1' if delta < -threshold else 'L7' if delta > threshold else 'UNRESOLVED';counts[winner]+=1
            rows.append(dict(document=doc,P1_rotation=p,L7_rotation=l,P1_gap=d['P1_gap'],L7_gap=d['L7_gap'],
                P1_score_P1_rotation=d['scores'][p]['P1'],P1_score_L7_rotation=d['scores'][l]['P1'],
                L7_score_P1_rotation=d['scores'][p]['L7_TOTAL_ORIGINAL'],L7_score_L7_rotation=d['scores'][l]['L7_TOTAL_ORIGINAL'],
                P1_rank_P1_rotation=d['P1_midranks'][p],P1_rank_L7_rotation=d['P1_midranks'][l],
                L7_rank_P1_rotation=d['L7_midranks'][p],L7_rank_L7_rotation=d['L7_midranks'][l],
                K_P1=kp,K_L7=kl,Delta=delta,relative_delta_over_K_L7=delta/kl if kl!=0 else 'UNDEFINED_ZERO_DENOMINATOR',
                winner=winner,empirical_resolution=threshold,numerical_and_physical_gates='PASS',status='COMPLETE'))
    if complete:assert fp_complete==4 and student_complete==8 and len(rows)==4 and len(position_rows)==8*384
    if rows:export(D/'FOUR_DOC_PRIMARY_RESULTS.csv',rows)
    if position_rows:export(D/'POSITIONAL_KL.csv',position_rows)
    if unit_ids:publish(D/'SCIENTIFIC_UNIT_IDENTITIES.json',dict(status='COMPLETE' if complete else 'PARTIAL',units=unit_ids))
    if selection and (D/'ENGINEERING_QUALIFICATION_PRE_FORMAL.json').exists():
        q=read(D/'ENGINEERING_QUALIFICATION_PRE_FORMAL.json')
        publish(D/'ENGINEERING_QUALIFICATION.json',dict(pre_formal=q,status='PASS_COMPLETE_RUN' if complete else 'PRE_FORMAL_PASS_FORMAL_INCOMPLETE',
            full32768_all12_finite='PASS' if complete else 'INCOMPLETE',formal_FP=fp_complete,formal_students=student_complete))
    verdict=('DIRECTIONALLY_SUPPORTIVE_SMALL_PROSPECTIVE_EXTENSION' if counts['P1']==4 else
        'NOT_SUPPORTIVE_FOR_P1_FOUR_L7_WINS' if counts['L7']==4 else
        'UNRESOLVED_SMALL_PROSPECTIVE_EXTENSION' if counts['UNRESOLVED']==4 else 'MIXED_SMALL_PROSPECTIVE_EXTENSION') if complete else 'INCOMPLETE_NOT_CONFIRMATORY'
    status=dict(PREREGISTRATION_IDENTITY_STATUS='PASS',INPUT_ELIGIBILITY_STATUS='PASS_EXACT_100_20_22_58',
        SHORT_HORIZON_SELECTION_STATUS=read(D/'SHORT_HORIZON_SELECTION_STATUS.json')['status'] if (D/'SHORT_HORIZON_SELECTION_STATUS.json').exists() else 'INCOMPLETE',
        FROZEN_SELECTION_STATUS='PASS_FROZEN_BEFORE_LONG_OUTCOMES' if selection else 'NOT_RUN',
        NUMERICAL_QUALIFICATION_STATUS='PASS' if (D/'ENGINEERING_QUALIFICATION_PRE_FORMAL.json').exists() else 'NOT_COMPLETED',
        PERSISTENT_INT8_SEMANTICS_STATUS='PASS_8_COMPLETE_QDQ_WRITEBACK_CONSUMPTION' if complete else 'INCOMPLETE_OR_NOT_RUN',
        FP_REFERENCE_STATUS=f'COMPLETE_{fp_complete}_OF_4',STUDENT_32K_STATUS=f'COMPLETE_{student_complete}_OF_8',
        PROSPECTIVE_PRIMARY_STATUS='COMPLETE_4_VALID_DOCUMENTS' if complete else f'INCOMPLETE_{len(rows)}_VALID_OF_4',
        DOCUMENT_LEVEL_VERDICT=verdict,SCIENTIFIC_INTERPRETATION_STATUS='CONDITIONAL_SMALL_PROSPECTIVE_EXTENSION_NO_STATISTICAL_CONFIRMATION' if complete else 'INCOMPLETE_NO_CONFIRMATORY_INFERENCE',
        TRAINING_AUTHORIZATION='NOT_AUTHORIZED',FINAL_EXECUTION_STATUS='COMPLETE_AND_STOP' if complete else 'FAIL_CLOSED_STOP')
    publish(D/'FINAL_STATUS.json',status)
    decisions=list((D/'screening').glob('*.SELECTION_DECISION.json')) if (D/'screening').exists() else []
    selected_text=', '.join(d['document'] for d in selection['documents']) if selection else 'NOT_FROZEN'
    pair_text='；'.join(f"{d['document']}: P1={d['P1_predicted_rotation']}, L7={d['L7_predicted_rotation']}, gaps={d['P1_gap']}/{d['L7_gap']}" for d in selection['documents']) if selection else 'NOT_RUN'
    table='|Document|P1 rotation|L7 rotation|K_P1|K_L7|Delta|Winner|\n|---|---|---|---:|---:|---:|---|\n'
    for r in rows:table+=f"|{r['document']}|{r['P1_rotation']}|{r['L7_rotation']}|{r['K_P1']:.17g}|{r['K_L7']:.17g}|{r['Delta']:.17g}|{r['winner']}|\n"
    effects=sorted(((abs(r['Delta']),r['document']) for r in rows),reverse=True)
    effect_text='；'.join(f'{doc}: |Delta|={value:.17g}' for value,doc in effects) or 'NOT_RUN'
    p1_failed=[r['document'] for r in rows if r['winner']=='L7']
    l7_failed=[r['document'] for r in rows if r['winner']=='P1']
    old_pairs={frozenset(['H_to_L6_a50','H_to_L7_a25']),frozenset(['H_to_L6_a50','H_to_L7_a50'])}
    reused_pair_docs=[r['document'] for r in rows if frozenset([r['P1_rotation'],r['L7_rotation']]) in old_pairs]
    report=f'''# Ling/KDA P1 vs Original L7 四文档前瞻验证

实验：{TASK}。最终执行：{status['FINAL_EXECUTION_STATUS']}。原两篇正式1胜1负、MIXED永久保留；旧交叉面板仅为事后探索，本轮不与历史合并。

1. 读取的正式资产：原FUNCTIONAL_VALIDATION成功delivery及COMPLETED_ARTIFACT_AUDIT，MIXED_OUTCOME_RESOLUTION成功RESULT_AND_DELIVERY_HASHES，正式NEXT_PROSPECTIVE_4DOC_PREREGISTRATION及协议freeze。SHA与原scorer身份PASS，具体绝对路径/SHA见PREREGISTRATION_RECONCILIATION.json。

2. PG19输入：100篇原文、完整token重新编码、前缀token、MD5/SHA及去重/排序逐项复现。历史排除20、长度不足22、输入资格58，与冻结记录完全一致。首次新增CPU审计脚本错误保留CRLF，导致假失败；对照原Path.read_text读取后复现全部原SHA。首次STOP和修复证据保留，科学输入未改、当时GPU科学单元0。

3. 实际完整短程筛选{len(decisions)}篇，上限40。选择状态：{status['SHORT_HORIZON_SELECTION_STATUS']}。每篇完整16个冻结旋转、原P1和L7_TOTAL、两窗256/1024 H128、18层16heads；筛选阶段没有读取新文档长程结果。

4. 最终四篇：{selected_text}。

5. 每篇冻结预测与midrank gap：{pair_text}。完整16分数、两套排名、全部eligible pairs、固定优先规则及过程见FROZEN_4DOC_SELECTION.json和SHORT_HORIZON_SELECTION.csv；精确相等才并列，双方gap≥3，最大min-gap后字典序。

6. Selection freeze：{status['FROZEN_SELECTION_STATUS']}。实际freeze时间和SHA记录在SELECTION_FREEZE_RECEIPT.json，任何新长程worker及Primary outcome之前共同冻结；没有根据结局换文档或旋转。

7. numerical/observer/fresh-process resume/persistent QDQ/physical资格：{status['NUMERICAL_QUALIFICATION_STATUS']}。自身旋转0→8192完整KDA/MLA/conv/position与RNG checkpoint，fresh8192→8256 exact64 logits/cache/audit、ON/OFF observer，原helper AST/SHA、相同GPU UUID/runtime及原reducer合同。INT8代码经QDQ后以FP32承载量化值写回并由下一kernel消费，不宣称physical resident INT8 kernel storage，不存在高精度shadow绕过量化。

8. 正式完整轨迹：FP {fp_complete}/4，persistent student {student_complete}/8，计划总12。完成的单元均32768执行，384固定位置齐全；未完成或未启动单元不填零、不替换。planned document分母4，实际valid={len(rows)}。

9. 完整精度逐文档Primary：

{table if rows else 'NOT_RUN：没有可合法汇总的完整文档Primary；未创建空科学结果表。'}
原V157184 FP32 KL(FP||student)，消费token[t]后的next-token logits，positions8192+64k至32704，Python sum/384；模型运行至32768。阈值3.3300064494019496e-6原样保留，只是同合格合同的经验数值分辨率。相对差异(Delta/K_L7)在CSV明确分母，仅描述，非新Primary/Secondary。

10. 文档级方向：P1={counts['P1']}，L7={counts['L7']}，UNRESOLVED={counts['UNRESOLVED']}，MISSING={4-len(rows)}。方向判定以冻结经验分辨率为界，不将小于分辨率的符号算成胜利。

11. 独立前瞻解释：{verdict}。各方向来自新文档和预先冻结pair；只适用于短程P1/L7分歧且双方rank gap≥3的条件分布。独立统计单位是document，不把384位置、18层16heads或6个contrast当独立样本。本轮按预注册不报告确认性p-value。即使4/4也不宣称统计确认、可训练Loss、完整未来Query因果机制或reasoning/AIME改善。

12. 与旧候选选择现象：本轮冻结pair与旧两种pair重合的文档为{reused_pair_docs}。只测所冻两个student，没有额外L7-a50对照；未测第三候选时不能断定复制了旧三旋转排序或L7-a50普遍最优。P1预测失败文档={p1_failed}，L7预测失败文档={l7_failed}，其余UNRESOLVED不计胜负。效应大小排序：{effect_text}；文档方向计数不按少数大效应加权，不凭 pooled mean 宣称优势。四篇不足以判断效应分布的总体代表性。

13. 实际资源：GPU worker {ledger['actual_GPU_worker_count']}，自然退出{len(receipts)}，失败退出{len(failed)}；实际{ledger['GPU_hours']:.12g} GPUh。CUDA allocated峰值{ledger['peak_CUDA_allocated']} bytes，reserved峰值{ledger['peak_CUDA_reserved']} bytes；NVML采样峰值{ledger['NVML_sample_peak_MiB']} MiB。逐worker墙钟含加载/退出，排除CPU/空等；CUDA为原统计scope，NVML不是连续瞬时高水位。详见RESOURCE_REPORT.json，单GPU6 RTX3090串行单缓存。

14. source commit={head}。实际服务器交付目录={D}，clean worktree={WT}。模型/旋转/输入/FP/checkpoint/runtime/评分身份及source SHA逐项绑定；大型张量留服务器，可用manifest核验。RESULT_AND_DELIVERY_HASHES.json和SHA256SUMS.txt封存交付身份。

15. 已{status['FINAL_EXECUTION_STATUS']}。TRAINING_AUTHORIZATION=NOT_AUTHORIZED。没有启动梯度、optimizer/Cayley、新L8/L9、AIME/free generation、额外32K或Qwen。本轮仅完成已授权4-doc扩展；后续建议只写报告。

方向预测结果可以提供P1相对于L7在这一条件分布中的有限比较证据，但不能据4篇识别统计条件独立的增量信息。更大独立验证可另行预注册；若仍MIXED，可优先error propagation与future readout observability机制研究；若不支持P1，降低其作为独立训练Loss的优先级。任何P1 gradient trainability资格研究仍需独立设计与授权，本轮结果不自动推进训练。具体误差方向、Query敏感性与历史累积的因果链仍NOT_IDENTIFIABLE。

E_future=Hidden KL=NOT_FROZEN_NOT_COMPUTED。原L7已H128 recurrent-aware；P1仍原K3局部readout-aware，并非全部未来Query。原AIME背景未重测。

机器状态：

```json
{json.dumps(status,indent=2,ensure_ascii=False)}
```
'''
    text(D/'FINAL_SCIENTIFIC_REPORT.md',report)
    required=['FINAL_SCIENTIFIC_REPORT.md','PREREGISTRATION_RECONCILIATION.json','SHORT_HORIZON_SELECTION.csv','FROZEN_4DOC_SELECTION.json',
        'ENGINEERING_QUALIFICATION.json','FOUR_DOC_PRIMARY_RESULTS.csv','POSITIONAL_KL.csv','RESOURCE_REPORT.json','FINAL_STATUS.json']
    files={}
    for f in sorted(D.rglob('*')):
        if not f.is_file() or '__pycache__' in f.parts or f.suffix=='.pt' or '.partial' in f.name or f.name in ['delivery_wait.log','RESULT_AND_DELIVERY_HASHES.json','SHA256SUMS.txt']:continue
        files[str(f.relative_to(D))]=dict(bytes=f.stat().st_size,sha256=sha(f))
    manifest=dict(status='COMPLETE' if complete else 'STOP_WITH_ACTUAL_PARTIAL_ASSETS',experiment=TASK,source_commit=head,
        original_formal_verdict='MIXED_PERMANENTLY_PRESERVED',files=files,
        required_artifacts={n:'COMPLETE' if (D/n).exists() else 'NOT_RUN' for n in required},
        large_remote_payload_bindings=dict(FP_manifest_paths=[str(f) for f in (D/'formal').glob('*__FP__formal.json')] if (D/'formal').exists() else [],
            checkpoint_qualification_path=str(D/'ENGINEERING_QUALIFICATION_PRE_FORMAL.json') if (D/'ENGINEERING_QUALIFICATION_PRE_FORMAL.json').exists() else None),
        initial_CPU_auditor_failure_preserved=old_failure,live_wait_log_excluded=True,self_reference_exclusions=['manifest excludes itself and SHA256SUMS; SHA256SUMS includes manifest but excludes itself'])
    publish(D/'RESULT_AND_DELIVERY_HASHES.json',manifest)
    sums=''.join(v['sha256']+'  '+n+'\n' for n,v in sorted(files.items()))+sha(D/'RESULT_AND_DELIVERY_HASHES.json')+'  RESULT_AND_DELIVERY_HASHES.json\n'
    text(D/'SHA256SUMS.txt',sums)
    print(json.dumps(dict(status='CPU_DELIVERY_SEALED',complete=complete,files=len(files),valid_documents=len(rows),GPU_hours=ledger['GPU_hours'])),flush=True)

if __name__=='__main__':main()
