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
WT=pathlib.Path(__file__).resolve().parent
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
    from gpu_admission import proc_identity
    for launch in D.rglob('*.LAUNCH.json'):
        ident=read(launch)['worker']
        try:actual=proc_identity(ident['pid'])
        except (OSError,ValueError):actual=None
        assert actual!=ident,'SCIENTIFIC_WORKER_STILL_ACTIVE'
    freeze=read(D/'SOURCE_CODE_FREEZE.json');head=freeze['source_commit'];routing=read(D/'EXECUTION_ROUTING.json')
    assert subprocess.check_output(['git','-C',str(WT),'rev-parse','HEAD'],text=True).strip()==head
    assert subprocess.check_output(['git','-C',str(WT),'status','--porcelain'],text=True)==''
    for n,h in freeze['files'].items():assert sha(WT/n)==h,n
    phase0=read(D/'PHASE0_HISTORICAL_ASSET_AUDIT.json');assert phase0['status']=='PASS'
    for historical in phase0['manifests']:
        base=pathlib.Path(historical['path']);assert sha(base/'RESULT_AND_DELIVERY_HASHES.json')==historical['manifest_sha256']
        for n,v in read(base/'RESULT_AND_DELIVERY_HASHES.json')['files'].items():assert sha(base/n)==v['sha256'],n
    assert sha(D/'FROZEN_SPEC.json')==routing['original_checkpoint_contract_sha256']
    a=read(D/'PREREGISTRATION_RECONCILIATION.json');assert a['status']=='PASS'
    for n,h in a['frozen_protocol_files'].items():assert sha(OLD/n)==h,n
    for name,h in a['historical_STOP_preserved'].items():assert sha(pathlib.Path(name))==h,name
    receipts=[];failed=[];samples=[]
    for f in sorted(D.rglob('*.NATURAL_EXIT.json')):
        r=read(f);assert r['natural_exit'] and r['source_commit']==head
        assert r['wall_seconds']==r['completed_epoch']-r['started_epoch']
        result_path=f.with_name(f.name.replace('.NATURAL_EXIT.json','.json'))
        assert sha(result_path)==r['result_sha256'];result=read(result_path);assert result['status']=='PASS'
        mem=result.get('memory',[]);ident=r['unit_identity']
        gpu=result.get('gpu_uuid',result.get('host',{}).get('gpu_uuid'))
        assert gpu==ident['target_gpu_uuid']==routing['document_devices'][ident['document']]
        receipts.append(dict(tag=r['tag'],stage=r['stage'],GPU_UUID=gpu,wall_seconds=r['wall_seconds'],gpu_hours=r['wall_seconds']/3600,receipt_sha256=sha(f),result_sha256=r['result_sha256'],peak_allocated=max((m['peak_allocated'] for m in mem),default=0),peak_reserved=max((m['peak_reserved'] for m in mem),default=0)))
    for f in sorted(D.rglob('*.FAILED_EXIT.json')):
        r=read(f);failed.append(dict(tag=f.name.removesuffix('.FAILED_EXIT.json'),returncode=r['returncode'],wall_seconds=r['wall_seconds'],gpu_hours=r['wall_seconds']/3600,receipt_sha256=sha(f),GPU_UUID=r['unit_identity']['target_gpu_uuid']))
    unreceipted=[]
    for launch in D.rglob('*.LAUNCH.json'):
        base=launch.name.removesuffix('.LAUNCH.json')
        if not (launch.parent/(base+'.NATURAL_EXIT.json')).exists() and not (launch.parent/(base+'.FAILED_EXIT.json')).exists():
            end=read(D/'STOP.json').get('epoch',read(D/'STOP.json').get('stopped_epoch')) if stop else None
            begin=read(launch)['started_epoch'];unreceipted.append(dict(tag=base,launch_sha256=sha(launch),GPU_hours_upper_bound_to_first_STOP=max(0,end-begin)/3600 if end else None,exact_GPU_hours='UNKNOWN_NO_EXIT_RECEIPT'))
    for f in D.rglob('*.GPU_EXCLUSIVITY.jsonl'):
        for line in f.read_text().splitlines():
            j=json.loads(line)
            if complete:assert j['exclusive'],j['failure_conditions']
            for row in j.get('pid_classifications',[]):
                if row.get('classification')=='VERIFIED_THIS_GPU_WORKER' and str(row['used_gpu_memory_mib']).replace('.','',1).isdigit():samples.append(float(row['used_gpu_memory_mib']))
    oldresource=read(pathlib.Path(routing['original_execution'])/'RESOURCE_REPORT.json')
    old1=oldresource['GPU_hours_reused_execution_01'];old2=oldresource['GPU_hours_new_execution_02']
    new=sum(r['gpu_hours'] for r in receipts)+sum(r['gpu_hours'] for r in failed)
    ledger=dict(status='COMPLETE' if complete else 'STOP_PARTIAL',experiment=TASK,source_commit=head,actual_GPU_worker_count=len(receipts)+len(failed)+len(unreceipted),natural_exit_worker_count=len(receipts),failed_exit_worker_count=len(failed),GPU_hours_new_execution_04=new,GPU_hours_reused_execution_01=old1,GPU_hours_reused_execution_02=old2,GPU_hours=new+old1+old2,GPU_hours_method='sum of actual per-worker launch-to-exit wall times; includes load/teardown, excludes admission/CPU/waits; concurrent lanes summed separately, no utilization weighting',peak_CUDA_allocated=max([oldresource['peak_CUDA_allocated']]+[r['peak_allocated'] for r in receipts]),peak_CUDA_reserved=max([oldresource['peak_CUDA_reserved']]+[r['peak_reserved'] for r in receipts]),peak_CUDA_allocated_new_execution_04=max((r['peak_allocated'] for r in receipts),default=0),peak_CUDA_reserved_new_execution_04=max((r['peak_reserved'] for r in receipts),default=0),CUDA_peak_scope='original loop-reset qualification/formal counters; excludes model-load whole-lifetime high-water mark; historical short scope separately preserved',NVML_sample_peak_MiB=max(samples+[oldresource['NVML_sample_peak_MiB']]) if samples else oldresource['NVML_sample_peak_MiB'],NVML_samples_new_execution_04=len(samples),NVML_scope='periodic exact-UUID owned process sample; not continuous instantaneous peak',device_mapping=routing['document_devices'],natural_exit_workers=receipts,failed_exit_workers=failed,unreceipted_duration_upper_bounds=unreceipted,no_GPUh_cap=True,no_85percent_gate=True,training_steps=0,STOP=read(D/'STOP.json') if stop else None)
    ledger['runtime_canary_worker_count']=sum(r['stage']=='runtime_canary' for r in receipts)
    ledger['science_qualification_and_formal_worker_count']=sum(r['stage']!='runtime_canary' for r in receipts)+len(failed)
    ledger['runtime_canary_GPU_hours']=sum(r['gpu_hours'] for r in receipts if r['stage']=='runtime_canary')
    ledger['historical_execution_03_failed_start_GPU_hours']=read(D.parent/'execution_03/RESOURCE_REPORT.json')['GPU_hours_new_execution_03']
    ledger['GPU_hours_all_execution_attempts']=ledger['GPU_hours']+ledger['historical_execution_03_failed_start_GPU_hours']
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
                assert j['gpu_uuid']==routing['document_devices'][doc]
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
                        if True:
                            payload=torch.load(path,map_location='cpu',weights_only=False)
                            assert payload['position']==ref['position'] and payload['token']==tokens[ref['position']]
                            if ref['logits']:
                                logits=payload['logits'];assert tuple(logits.shape)==(1,1,157184) and logits.dtype==torch.float32
                                assert bool(torch.isfinite(logits).all())
                                digest.update(ref['position'].to_bytes(8,'little'));digest.update(logits.detach().contiguous().numpy().tobytes())
                            del payload
                    assert digest.hexdigest()==j['sampled_logits_content_sha256']
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
    spec=read(D/'FROZEN_SPEC.json');matrix=[];checkpoints=[]
    for u in spec['new_units']:
        key=u['document']+'__'+u['rotation'];p=D/'runs'/(key+'__parent.json');f=D/'runs'/(key+'__resumed_parity.json');cp=D/'runs'/(key+'__8192.pt');qp=D/'qualification_units'/(key+'.json')
        row=dict(document=u['document'],rotation=u['rotation'],target_GPU_UUID=routing['document_devices'][u['document']],parent_status='PASS' if p.exists() else 'NOT_RUN',target_fresh64_status='PASS' if f.exists() else 'NOT_RUN',qualification_status='PASS' if qp.exists() else 'NOT_COMPLETED',formal_status='COMPLETE' if (D/'formal'/(key+'__formal.json')).exists() else 'NOT_RUN')
        for stage,field in [('parent','parent_status'),('resumed_parity','target_fresh64_status')]:
            failed_path=D/'runs'/(key+'__'+stage+'.FAILED_EXIT.json')
            if failed_path.exists():
                row[field]='FAILED';row['qualification_status']='FAILED_OR_NOT_QUALIFIED';row['failed_exit_sha256']=sha(failed_path)
        if qp.exists():
            q=read(qp)
            from qualified_completion import compare_exact
            compare_exact(read(p),read(f),row['target_GPU_UUID'])
            assert sha(p)==q['parent_sha256'] and sha(f)==q['resumed_sha256'] and sha(cp)==q['checkpoint_sha256']
            row['qualification_manifest_sha256']=sha(qp)
        if cp.exists():
            parent=read(p);assert sha(cp)==parent['checkpoint_sha256']
            checkpoints.append(dict(document=u['document'],rotation=u['rotation'],path=str(cp),resolved_path=str(cp.resolve()),bytes=cp.stat().st_size,sha256=sha(cp),parent_manifest_path=str(p),parent_manifest_sha256=sha(p),parent_GPU_UUID=parent['host']['gpu_uuid'],qualified_GPU_UUID=row['target_GPU_UUID'] if qp.exists() else 'NOT_QUALIFIED',next_position=8192,checkpoint_contract_sha256=sha(D/'FROZEN_SPEC.json')))
        matrix.append(row)
    qualified=sum(r['qualification_status']=='PASS' for r in matrix)
    publish(D/'QUALIFICATION_MATRIX.json',dict(status='COMPLETE' if qualified==12 else 'PARTIAL',target_device_qualified=qualified,planned=12,historical_GPU6_complete=8,units=matrix))
    publish(D/'CHECKPOINT_MANIFEST.json',dict(status='PASS_ACTUAL_BYTE_SHA',count=len(checkpoints),checkpoints=checkpoints))
    status.update(ENGINEERING_COMPLETE=(D/'GPU_RUNTIME_CANARY.json').exists() and read(D/'GPU_RUNTIME_CANARY.json')['status']=='PASS_BOTH_AUTHORIZED_DEVICES',NUMERICAL_QUALIFICATION_COMPLETE=qualified==12,FORMAL_SCIENCE_COMPLETE=complete and fp_complete==4 and student_complete==8,QUALIFIED_CONDITIONS=qualified,FORMAL_FP_COMPLETE=fp_complete,FORMAL_STUDENT_COMPLETE=student_complete,ACTUAL_VALID_DOCUMENTS=len(rows))
    # FINAL_STATUS above is written by the inherited checks; the final status
    # file is committed only once below after these independent counts.
    publish(D/'FINAL_MACHINE_STATUS.json',status)
    table='|文档|P1旋转|原L7旋转|K_P1|K_L7|Delta|判定|\n|---|---|---|---:|---:|---:|---|\n'
    for r in rows:table+=f"|{r['document']}|{r['P1_rotation']}|{r['L7_rotation']}|{r['K_P1']:.17g}|{r['K_L7']:.17g}|{r['Delta']:.17g}|{r['winner']}|\n"
    blocker=json.dumps(read(D/'STOP.json'),ensure_ascii=False,indent=2) if stop else 'NONE'
    pairs='；'.join(f"{r['document']}: {r['P1_predicted_rotation']} vs {r['L7_predicted_rotation']} (gap={r['P1_gap']}/{r['L7_gap']})" for r in selection['documents'])
    report=f'''# Ling/KDA 四文档前瞻实验 qualified completion

科学ID：{TASK}；本轮独立身份：execution_04。结论：{status['FINAL_EXECUTION_STATUS']}。ENGINEERING_COMPLETE={status['ENGINEERING_COMPLETE']}；NUMERICAL_QUALIFICATION_COMPLETE={status['NUMERICAL_QUALIFICATION_COMPLETE']}；FORMAL_SCIENCE_COMPLETE={status['FORMAL_SCIENCE_COMPLETE']}。原两文档正式 MIXED 永久保留，旧 posthoc 面板不变。

## A. 原资产审计

原预注册 SHA={phase0['pre_registration_sha256']}；四篇 freeze SHA={phase0['selection_sha256']}；freeze 北京时间2026-10-14 10:44:26。Phase0逐文件核验原成功manifest、execution01/02 manifest和SHA清单、64条评分、9个真实checkpoint、8个完整PASS和第9 parent。旧目录全程只读。修复仅在解释器启动前由launcher构造冻结三变量环境；未改或删除initializer断言。新科学文档和旋转未改：{pairs}。79项源资产含模型权重实际字节SHA已复核。execution03原环境启动错误与全部旧修复记录只读保留。本轮真实启动链路CPU回归见CPU_LAUNCH_CHAIN_REGRESSION.json；parent/fresh64正例与缺失seed负例均调用冻结initializer。GPU2/3各一次runtime canary见GPU_RUNTIME_CANARY.json，不加载模型且科学轨迹计数0。数学helper、原quantizer/forward/scale/KL reducer不变，实际新源码commit={head}，diff与AST核验见SOURCE_DIFF.patch和NUMERICAL_SOURCE_COMPATIBILITY_AUDIT.json。

## B. GPU稳定性和使用证据

用户最新明确授权GPU2/3。当前目标设备绑定为{routing["authorized_devices"]}；本轮实际60秒预检结果见GPU_PREFLIGHT_RESULT_02/03，逐worker另有原冻结准入记录。昨日准备窗口的GPU4/5仅是历史准备证据；系统Xorg图形约4MiB显式保留PID/UID/starttime及双API身份。最初新增审计将任何graphics一概判busy，超过原冻结compute准入定义；原窗口原样保留，解释修正在GPU_PREFLIGHT_INTERPRETATION.json，未放宽util/内存/60秒/final3条件。启动时非目标GPU占用见本轮只读inventory，没有使用或终止非目标进程。主机PID namespace及不可读驱动日志仍UNKNOWN；实际每个worker要求SMI/NVML compute exact自有PID/UID/starttime/UUID，图形身份保持冻结基线。所有worker（包括canary）都另经一次原60秒准入；预检不冒充正式worker准入。Canary仅初始化CUDA和最小分配，无模型推理，不计资格或科学轨迹；卡小时在账本按stage单列。

## C. 资格

目标设备真实合格 {qualified}/12；历史GPU6 8/12保持原标签。9个已完成parent没有重算，跨设备fresh64须逐条件比较全部64 full-V logits SHA、完整native cache、audit state、checkpoint SHA，恢复tensor指纹/RNG及observer ON/OFF exact；新设备标签只有真实通过后才生成。第四篇只计算其原缺3个parent及fresh。数学与数值门禁、精确比较方式未改。逐条件状态见QUALIFICATION_MATRIX.json；parent不等于完整PASS。

## D. 正式32K

FP {fp_complete}/4；student {student_complete}/8；valid文档 {len(rows)}/4；planned12轨迹。资格全部12通过且资源PASS后才正式运行，原冻结文档順序逐篇FP→P1→L7，全局串行；双卡只加速资格阶段。同文档FP/P1/L7在同一目标UUID/runtime配对。原8192 checkpoint是其自身token0量化/FP历史的真实合法前缀；迁移资格只有在真实fresh64 bitwise证明完整状态/缓存兼容后才算PASS，再由正式first64 observer复验；本轮实际资格/轨迹计数以前述字段为准，绝不把8256当32768。前3篇prefix历史GPU6、正式continuation设备={routing["document_devices"]["pg19-test-41603"]}，第四篇设备={routing["document_devices"]["pg19-test-26183"]}；不会声称整条轨迹始终在同UUID。INT8 code经QDQ后由FP32载体承载量化值，写回并被下一kernel消费；不称physical resident INT8 storage。

## E. Primary结果

{table if rows else 'NOT_RUN：没有完整有效四文档Primary；未执行项不填0、不生成空科学CSV。'}
V=157184，原FP32 KL(FP||student)，消费ids[t]后logits预测ids[t+1]，t=8192+64k，k0..383，末32704，模型实际endpoint32768。实际完整student逐点有限且384齐全，严格原Python sum/384（没有新求和算法）。Delta=K_P1-K_L7，阈值3.3300064494019496e-6仅原经验数值分辨率。P1={counts['P1']}，L7={counts['L7']}，UNRESOLVED={counts['UNRESOLVED']}，MISSING={4-len(rows)}。原协议不报告confirmatory p-value；即使独立无平局且p0=.5下4/4同方向，单侧exact sign .0625也不构成5%确认。本轮仅短程分歧及gap≥3条件下的小样本前瞻解释，document为单位，不将384点/层/heads当独立n，不合并旧2篇为n6。E_future/Hidden KL=NOT_FROZEN_NOT_COMPUTED。

## F. 实际资源及交付

execution01复用 {old1:.17g} GPUh；execution02复用 {old2:.17g} GPUh；execution04新增 {new:.17g} GPUh；累计 {ledger['GPU_hours']:.17g} GPUh。本轮自然退出worker {len(receipts)}（含runtime canary {ledger["runtime_canary_worker_count"]}）、失败退出 {len(failed)}；历史execution03失败启动另计 {ledger["historical_execution_03_failed_start_GPU_hours"]:.17g} GPUh，全部尝试合计 {ledger["GPU_hours_all_execution_attempts"]:.17g} GPUh；未获退出receipt的时段仅独立上界，不能伪装精确计量。总计按每worker launch-to-exit求和，含加载/退出，排除CPU/准入空等；并行卡小时分别相加。本轮CUDA allocated峰值 {ledger['peak_CUDA_allocated_new_execution_04']} bytes，reserved {ledger['peak_CUDA_reserved_new_execution_04']} bytes；历史+本轮NVML定期owned采样峰值 {ledger['NVML_sample_peak_MiB']} MiB。CUDA原loop reset scope、NVML periodic scope有明确限制，未声称whole-lifetime或连续峰值。真实checkpoint {len(checkpoints)}个，逐路径、bytes、SHA见CHECKPOINT_MANIFEST.json；FP真实payload见各formal manifest。服务器 {D}；工作树 {WT}；本地交付 /Users/pytenter/Documents/临时项目1/ling_prospective_extension_4doc_v1/execution_04_minimal_recovery/delivery/。大型FP/checkpoint留服务器绑定SHA，模型不下载。manifest与SHA256SUMS含manifest自身的SHA验证。

## G. 首个blocker与复用资产

```json
{blocker}
```

实际成功单元/NOT_RUN见QUALIFICATION_MATRIX及机器状态。任何失败首条保留no_retry，不放宽门禁、不重选、不用fixture作新科学证据。已有成功字节可复用的范围仅已核验的相同科学key、source/runtime及实际已合格设备；新设备/新的恢复仍须资格，不自动沿用旧PASS。

## H. 科学解释与后续建议

{verdict}。四篇仅有限比较证据，不能证明P1梯度可优化性、全部future Query因果机制、reasoning/AIME改善或总体稳定优势。若混合，后续可另行设计error propagation/readout observability研究；若方向一致，仍需更大独立预注册confirmation。建议不自动执行。训练授权NOT_AUTHORIZED；P1 Loss、L8/L9、STE/Cayley/optimizer、AIME/free generation/Qwen均未启动，交付后停止。

## 原15问逐项状态

1. 读取原成功、历史execution01/02 manifest与科学预注册：PASS，路径/SHA见Phase0。
2. 输入资格原100/20/22/58与真实tokens/raw SHA保留，新执行不重复编码或选文档。
3. 原完整4篇16旋转=64分数全部复用，未重筛。
4. 四篇41603/2544/7987/26183，完全冻结。
5. 预测与midrank gap：{pairs}；全部候选、排名、规则原样。
6. freeze在所有新长程前；原时间和receipt字节未改。
7. 目标GPU numerical/fresh/observer/QDQ资格：{qualified}/12，身份及设备迁移证据逐项保留。
8. 正式FP {fp_complete}/4，student {student_complete}/8，valid {len(rows)}/4。
9. 完整精度KL、Delta见E与真实CSV，未完成值NOT_RUN。
10. 文档计数P1={counts['P1']}/L7={counts['L7']}/UNRESOLVED={counts['UNRESOLVED']}，未完成不算负。
11. 只作条件性小样本前瞻，不做confirmatory p-value，不伪重复。
12. 没有额外第三候选轨迹，不能据冻结两候选推断L7-a50普遍最优或复现所有旧三旋转排序。
13. 实际资源分执行账本及真实scope见F/RESOURCE_REPORT。
14. 实际source commit、路径、模型/输入/旋转/CP/FP/runtime/UUID/scoring SHA见封存清单。
15. 终态{status['FINAL_EXECUTION_STATUS']}；未授权训练与额外科学任务均未执行。

机器状态：

```json
{json.dumps(status,ensure_ascii=False,indent=2)}
```
'''
    text(D/'FINAL_SCIENTIFIC_REPORT.md',report)
    from recovery_controller import event
    event('COMPLETE_AND_STOP' if complete else 'STOP_DELIVERY_COMPLETE',CPU_verification='PASS',training_steps=0)
    required=['FINAL_SCIENTIFIC_REPORT.md','FINAL_MACHINE_STATUS.json','PHASE0_HISTORICAL_ASSET_AUDIT.json','FROZEN_4DOC_SELECTION.json','SELECTION_FREEZE_RECEIPT.json','QUALIFICATION_MATRIX.json','CHECKPOINT_MANIFEST.json','RESOURCE_REPORT.json','NUMERICAL_SOURCE_COMPATIBILITY_AUDIT.json','LONG_HORIZON_QUALIFICATION.json','PRE_FORMAL_RESOURCE_ESTIMATE.json','FOUR_DOC_PRIMARY_RESULTS.csv','POSITIONAL_KL.csv']
    files={}
    for f in sorted(D.rglob('*')):
        if not f.is_file() or '__pycache__' in f.parts or f.suffix=='.pt' or '.partial' in f.name or f.name.endswith('.lock') or f.name=='controller.log' or f.name.startswith('lane_GPU-') and f.name.endswith('.log') or f.name in ['RESULT_AND_DELIVERY_HASHES.json','SHA256SUMS.txt']:continue
        files[str(f.relative_to(D))]=dict(bytes=f.stat().st_size,sha256=sha(f))
    manifest=dict(status='COMPLETE' if complete else 'STOP_WITH_ACTUAL_PARTIAL_ASSETS',experiment=TASK,execution='execution_04',source_commit=head,original_formal_verdict='MIXED_PERMANENTLY_PRESERVED',files=files,required_artifacts={n:'COMPLETE' if (D/n).exists() else 'NOT_RUN' for n in required},large_remote_payload_bindings=dict(checkpoint_manifest=str(D/'CHECKPOINT_MANIFEST.json'),FP_manifest_paths=[str(p) for p in (D/'formal').glob('*__FP__formal.json')]),self_reference_exclusions=['manifest excludes itself and SHA256SUMS; SHA256SUMS includes manifest and excludes itself'])
    publish(D/'RESULT_AND_DELIVERY_HASHES.json',manifest)
    text(D/'SHA256SUMS.txt',''.join(v['sha256']+'  '+n+'\n' for n,v in sorted(files.items()))+sha(D/'RESULT_AND_DELIVERY_HASHES.json')+'  RESULT_AND_DELIVERY_HASHES.json\n')
    print(json.dumps(dict(status='CPU_DELIVERY_SEALED',complete=complete,files=len(files),qualified=qualified,valid_documents=len(rows),new_GPU_hours=new)),flush=True)

if __name__=='__main__':main()
