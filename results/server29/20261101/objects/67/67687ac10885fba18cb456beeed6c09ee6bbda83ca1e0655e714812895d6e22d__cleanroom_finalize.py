"""CPU verification and result delivery; no additional GPU execution."""
import csv,hashlib,io,json,math,os,statistics,subprocess,time
from pathlib import Path
from cleanroom_common import D,W,TASK,read,sha,publish

def text(path,value):
 with Path(path).open('x') as f:f.write(value);f.flush();os.fsync(f.fileno())
def csv_export(path,rows):
 if not rows:return
 out=io.StringIO();writer=csv.DictWriter(out,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows);text(path,out.getvalue())
def main():
 assert os.environ['CUDA_VISIBLE_DEVICES']==''
 complete=(D/'COMPLETE.json').exists();assert complete!=(D/'STOP.json').exists()
 from gpu_admission import proc_identity
 receipts=[read(p) for p in sorted((D/'jobs').glob('*.EXIT.json'))]
 for launch in (D/'jobs').glob('*.LAUNCH.json'):
  ident=read(launch)['identity']
  try:live=proc_identity(ident['pid'])==ident
  except (OSError,ValueError):live=False
  assert not live,'OWN_GPU_WORKER_STILL_ACTIVE'
  assert launch.with_name(launch.name.replace('.LAUNCH.json','.EXIT.json')).exists(),'EXIT_RECEIPT_MISSING'
 freeze=read(D/'SOURCE_CODE_FREEZE.json');assert subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True)==''
 for n,h in freeze['files'].items():assert sha(W/n)==h,n
 status_files={r['tag']:r for r in receipts}
 for r in receipts:
  assert r['wall_seconds']==r['completed_monotonic']-r['started_monotonic']
  if r['status']=='PASS':
   assert sha(r['result_path'])==r['result_sha256']
   assert sha(Path(r['result_path']).with_suffix('.PROFILE.json'))==r['profile_sha256']
 ledger=read(D/'gpu_resource_ledger.json');assert not ledger['active_workers']
 assert ledger['actual_new_GPU_hours']<=36
 positions=[8192+64*k for k in range(384)]
 spec=read(D/'FROZEN_SPEC.json');selection=read(D/'FROZEN_4DOC_SELECTION.json');routing=read(D/'EXECUTION_ROUTING.json')
 assert spec['primary']['positions']==positions and positions[-1]==32704
 qrows=[];FP=students=0;units=[];docrows=[];posrows=[];CP=[];resolution=(D/'resolution_contract.json').exists()
 for u in spec['new_units']:
  key=u['document']+'__'+u['rotation'];p=D/'qualification_units'/(key+'.json');cp=D/'runs'/(key+'__8192.pt')
  if p.exists():
   q=read(p)
   from cleanroom_common import compare_exact
   compare_exact(read(D/'runs'/(key+'__parent.json')),read(D/'runs'/(key+'__resumed_parity.json')),q['qualified_gpu_uuid'])
   assert sha(cp)==q['checkpoint_sha256'] and sha(D/'runs'/(key+'__parent.json'))==q['parent_sha256'] and sha(D/'runs'/(key+'__resumed_parity.json'))==q['resumed_sha256']
   qrows.append(q)
  else:qrows.append(dict(document=u['document'],rotation=u['rotation'],status='NOT_COMPLETED',INT8_QDQ='INCOMPLETE' if u['rotation']!='FP' else 'N/A_FP'))
  if cp.exists():CP.append(dict(document=u['document'],condition=u['rotation'],cursor=8192,path=str(cp),resolved_path=str(cp.resolve()),bytes=cp.stat().st_size,sha256=sha(cp),prefix_source='REUSED_VERIFIED_FP' if cp.is_symlink() else 'NEW_OWN_HISTORY'))
  long=D/'checkpoints'/(key+'__16384.pt')
  if long.exists():
   metadata=read(long.with_suffix('.json'));assert metadata['sha256']==sha(long)
   CP.append(dict(document=u['document'],condition=u['rotation'],cursor=16384,path=str(long),bytes=long.stat().st_size,sha256=sha(long),logical_reducer_and_native_state=True))
 import torch
 assert not torch.cuda.is_initialized()
 for doc in selection['documents']:
  name=doc['document'];valid={};token_path=Path(doc['document_identity']['token_file_path']);tokens=read(token_path)
  for condition in ['FP',doc['P1_predicted_rotation'],doc['L7_predicted_rotation']]:
   tag=name+'__'+condition+'__formal2';receipt=status_files.get(tag)
   if not receipt or receipt['status']!='PASS':continue
   j=read(receipt['result_path']);assert j['status']=='PASS' and j['all_finite'] and j['endpoint']==32768
   assert j['sample_positions']==positions and j['vocabulary_dimension']==157184
   assert j['token_sha256']==sha(token_path) and j['primary_spec_sha256']==sha(D/'FROZEN_SPEC.json')
   assert j['gpu_uuid']==routing['document_devices'][name]
   assert j['formal_observer_uninterrupted_parity_PASS'] and j['observer_qualified_before_first_primary_reduction']
   assert j['final_cache_inventory']['cache_bytes']==2033025024
   proof=read(D/'segments'/(name+'__'+condition+'__long_resume_exact.json'));assert proof['status']=='PASS_EXACT_64_LOGITS_NATIVE_CACHE_AUDIT_RNG'
   if condition=='FP':
    assert len([r for r in j['references'] if r['logits']])==384
    assert [r['position'] for r in j['references']]==sorted(set(positions)|{8192,16383,24575,32767})
    digest=hashlib.sha256()
    for r in j['references']:
     path=Path(r['path']);assert sha(path)==r['sha256'] and path.stat().st_size==r['bytes']
     payload=torch.load(path,map_location='cpu',weights_only=False);assert payload['position']==r['position'] and payload['token']==tokens[r['position']]
     if r['logits']:
      x=payload['logits'];assert tuple(x.shape)==(1,1,157184) and x.dtype==torch.float32 and bool(torch.isfinite(x).all())
      digest.update(r['position'].to_bytes(8,'little'));digest.update(x.contiguous().numpy().tobytes())
     del payload
    assert digest.hexdigest()==j['sampled_logits_content_sha256'];FP+=1
   else:
    samples=j['primary_samples'];assert len(samples)==384 and [v['position'] for v in samples]==positions
    assert all(math.isfinite(v['KL']) and v['vocabulary_dimension']==157184 for v in samples)
    assert sum(v['KL'] for v in samples)/384==j['primary_mean_KL']
    integrity=j['persistent_integrity'];assert integrity['passed'] and integrity['every_token_layer_qdq_count']==32768*18 and integrity['actual_next_kernel_equal_post_qdq_count']==32767*18
    students+=1
    for x in samples:posrows.append(dict(document=name,condition=condition,position=x['position'],KL=x['KL'],vocabulary_dimension=157184,result_sha256=sha(receipt['result_path'])))
   units.append(dict(document=name,condition=condition,status='COMPLETE',UUID=j['gpu_uuid'],result_path=receipt['result_path'],sha256=sha(receipt['result_path']),checkpoint_sha256=j['checkpoint_sha256'],long_resume_proof_sha256=sha(D/'segments'/(name+'__'+condition+'__long_resume_exact.json'))))
   valid[condition]=j
  p,l=doc['P1_predicted_rotation'],doc['L7_predicted_rotation'];base=dict(Doc=name,P1_Rotation=p,Original_L7_Rotation=l,K_P1=None,K_L7=None,Delta=None,Winner='NOT_RUN',status='NOT_RUN')
  if len(valid)==3:
   kp,kl=valid[p]['primary_mean_KL'],valid[l]['primary_mean_KL'];delta=kp-kl
   threshold=3.3300064494019496e-6
   winner='P1' if delta < -threshold else 'L7' if delta>threshold else 'NUMERICALLY_UNRESOLVED'
   base.update(K_P1=kp,K_L7=kl,Delta=delta,Winner=winner,status='COMPLETE',relative_delta_over_K_L7=delta/kl if kl!=0 else None,numerical_resolution=threshold,resolution_evidence='new exact replay, frozen profile and historical arithmetic threshold; no new32K error calibration')
  docrows.append(base)
 valid_docs=[r for r in docrows if r['status']=='COMPLETE'];P1=sum(r['Winner']=='P1' for r in valid_docs);L7=sum(r['Winner']=='L7' for r in valid_docs);ties=len(valid_docs)-P1-L7
 qcount=sum(r['status']=='PASS' for r in qrows);int8q=[r for r in qrows if r['rotation']!='FP']
 classification='P1_DIRECTIONALLY_SUPPORTED' if len(valid_docs)==4 and P1>=3 else 'P1_INCREMENTAL_VALUE_NOT_SUPPORTED' if len(valid_docs)==4 and L7>=3 else 'MIXED_OR_INCONCLUSIVE'
 if complete:assert qcount==12 and FP==4 and students==8 and len(valid_docs)==4 and len(posrows)==8*384
 totals=dict(TASK=TASK,SCIENTIFIC_PROTOCOL_IDENTITY='PASS',ENGINEERING_IMPLEMENTATION='CLEANROOM_CONTROL_ONLY',GPU_RESOURCE_ADMISSION='PASS' if len([p for p in (D/'admission').glob('*.json') if not p.name.endswith('.PASS.json')])==len(list((D/'admission').glob('*.PASS.json'))) and (complete or read(D/'STOP.json').get('stage')!='RUNNING_RESOURCE_SAFETY') else 'FAIL',FLA_NUMERICAL_PROFILE='PASS' if (D/'G1_CANARY_COMPLETE.json').exists() else 'INCOMPLETE',MODEL_AND_ROTATION_IDENTITY='PASS' if read(D/'audits/G0_ACTUAL_MODEL_INPUT_ROTATION_SHA.json')['status']=='PASS_ACTUAL_BYTES' else 'FAIL',FRESH64_QUALIFICATION=f'{qcount}/12',INT8_PERSISTENT_QDQ_QUALIFICATION='PASS' if len(int8q)==8 and all(r['status']=='PASS' for r in int8q) else 'INCOMPLETE',FORMAL_FP_COMPLETE=f'{FP}/4',FORMAL_STUDENT_COMPLETE=f'{students}/8',ACTUAL_VALID_DOCUMENTS=f'{len(valid_docs)}/4',PRIMARY_384_SCORE_REPRODUCTION='PASS' if students else 'NOT_RUN',P1_WINS=P1,L7_WINS=L7,TIE_UNRESOLVED=ties,MISSING_DOCUMENTS=4-len(valid_docs),NEW_GPU_HOURS_USED=ledger['actual_new_GPU_hours'],MAX_GPU_HOURS=36.,FINAL_SCIENTIFIC_CLASSIFICATION=classification,FINAL_EXECUTION_STATUS='COMPLETE_AND_STOP' if complete else 'PARTIAL_STOP' if qcount or units else 'FAIL_CLOSED_STOP')
 deltas=[r['Delta'] for r in valid_docs]
 judgment=('本次没有完成四篇，尚不能回答P1增量预测价值；工程/资源停止不作为selector科学负例。' if len(valid_docs)<4 else '三篇或四篇支持P1，仅是有限文档方向性信号；应结合平均/中位Delta和每篇效应，不等于总体平均优势或泛化确认。' if P1>=3 else '三篇或四篇Original L7胜出，本次不支持当前P1 selector的稳定增量；建议暂停该版本推进。' if L7>=3 else '结果混合或数值未分辨，未建立稳定增量；不支持据此扩大selector训练，也不能证明P1普遍无效。')
 next_science=('若继续，先以独立更大预注册文档样本确认方向与效应稳定性。' if len(valid_docs)==4 and P1>=3 else '当前不自动开展训练或新实验；不足或混合的证据应先明确未回答问题，避免把工程失败或n4小样本当普遍结论。')
 summary=dict(machine_status=totals,documents=docrows,valid_n=len(valid_docs),mean_Delta=statistics.mean(deltas) if deltas else None,median_Delta=statistics.median(deltas) if deltas else None,exact_sign_test='NOT_REPORTED: document independence and sign-symmetry are not established; no confirmatory p-value',four_of_four_theoretical_one_sided_p_if_independent_symmetric_null=0.0625,old_two_document_MIXED='PERMANENTLY_PRESERVED_NOT_COMBINED_N6')
 publish(D/'final_scientific_results.json',summary);csv_export(D/'final_scientific_results.csv',valid_docs);csv_export(D/'positional_KL.csv',posrows)
 publish(D/'qualification_matrix.json',dict(qualified=qcount,planned=12,units=qrows))
 publish(D/'checkpoint_manifest.json',dict(checkpoints=CP,large_payloads_remote_only=True))
 publish(D/'run_manifest.json',dict(task=TASK,source_commit=freeze['source_commit'],formal_units=units,GPU_workers=receipts,planned_formal_trajectories=12,actual_GPU_workers=len(receipts),extra_overlap_tokens_per_formal_trajectory=64,unique_logical_consumed_tokens_per_complete_trajectory=32768,actual_formal_worker_calls_per_complete_trajectory=2))
 peaks=[];NVML=[]
 for r in receipts:
  if r.get('NVML_periodic_peak_owned_bytes') is not None:NVML.append(r['NVML_periodic_peak_owned_bytes'])
  if r['status']=='PASS':
   profile=read(Path(r['result_path']).with_suffix('.PROFILE.json'));peaks.append(profile['measured_CUDA_peak'])
   j=read(r['result_path']);memory=j.get('memory',[])
   if isinstance(memory,dict):memory=[memory]
   peaks.extend(memory)
 resource=dict(new_GPU_hours=ledger['actual_new_GPU_hours'],max_GPU_hours=36.,actual_GPU_worker_count=len(receipts),peak_measured_CUDA_allocated=max((v['peak_allocated'] for v in peaks),default=None),peak_measured_CUDA_reserved=max((v['peak_reserved'] for v in peaks),default=None),CUDA_scope='canary model-load+8token peak; frozen workers reset at loop; segment measurements aggregated; no universal lifetime peak claim',NVML_periodic_peak_owned_bytes=max(NVML) if NVML else None,NVML_scope='approximately10s exact owned PID samples, not instantaneous peak',launch_to_exit_includes_load_restore_teardown=True)
 publish(D/'gpu_resource_summary.json',resource)
 table='|Doc|P1 Rotation|Original L7 Rotation|K_P1|K_L7|Delta|Winner|\n|---|---|---|---:|---:|---:|---|\n'
 for r in docrows:
  v=[r[k] for k in ['Doc','P1_Rotation','Original_L7_Rotation','K_P1','K_L7','Delta','Winner']]
  table+='|'+'|'.join('NOT_RUN' if x is None else format(x,'.17g') if isinstance(x,float) else str(x) for x in v)+'|\n'
 blocker=json.dumps(read(D/'STOP.json'),ensure_ascii=False,indent=2) if not complete else 'NONE'
 report=f'''# Ling/KDA 四文档 Clean-Room Final

独立TASK：{TASK}；原科学ID：LING_KDA_P1_L7_PROSPECTIVE_EXTENSION_4DOC_V1。
终态：{totals['FINAL_EXECUTION_STATUS']}。本轮仅更换控制层，旧execution01至05全部保留，未覆盖STOP/模型/CP，不启动下一轮尝试。

## 科学身份与执行

原prereg54d6d24fbc3142a692ef1a6c04a5617cf6b0abe6702a662722f34fc1b6141722，四篇freeze a5c227d15918a249f8ae2059539c15f94668cb2afddf3aecd8a9f7e13851354b，FROZEN_SPEC 8ae8927ed37209efccbf19c672b9baeb46c0d7f272601f8f54d1f9609b64f3d1均实际SHA PASS。100/20/22/58输入流程与4篇×16=64冻结评分不变，无重排、重选或新训练。32份权重、raw/token、旋转及数学源码实际SHA见G0/source manifest。源码commit：{freeze['source_commit']}。

冻结数学worker文件字节保持；正式编译入口仅添加已登记的checkpoint/cursor/输出原子化控制节点，逐项可逆还原整个原始源码，未改model_step/QDQ/rotation/observer/KL/原Python sum384。旧执行05的双零13点规则不追溯改变；新任务使用3次间隔采样，≤2%差异只提示，前提是无外部compute/身份异常、显存稳定且按历史max(reserved,NVML峰值)+至少2GiB空余准入。实际资源安全仍独立监督，每个模型进程来源/profile/UUID验证。

每卡仅一次真实模型8-token canary，非科学轨迹。数值资格{qcount}/12；INT8资格{totals['INT8_PERSISTENT_QDQ_QUALIFICATION']}，FP audit=None为N/A。同一所选UUID上的既有FP8192 CP只在当前GPU fresh64合同实际PASS后复用，设备不同时重建其FP前缀；INT8各有自己的从token0 persistent-QDQ历史。FLA Conv BD8/W8、Gated RMSNorm BT16/W8固定，五个启动变量在新解释器前设置。8192原fresh64、observer/nativecache/RNG精确合同比较未放宽。

正式逻辑顺序41603→2544→7987→26183，每篇FP→P1→OriginalL7，全局串行。新增一次计划16384 checkpoint，原cache/RNG/QDQ state及完整已处理评分列表/原reducer状态原子保存。第一段继续64步收集未中断对照，第二段fresh恢复16384，完整64 logits/cache/audit/RNG必须exact，重放评分不重复计入384数组。每条完整轨迹唯一逻辑consumed tokens32768，计划恢复验证额外64token单列计时；两个正式segment不冒充两条科学样本。非预期进程退出无静默重试。

## 实际结果

{table}

原V157184，t=8192+64k (k0..383、末32704)，消费ids[t]预测ids[t+1]；原FP32 full-vocabulary KL(FP||student)，原Python sum/384。未完成项NOT_RUN，不填0、无空壳科学CSV。实际有效文档{len(valid_docs)}/4，FP{FP}/4、student{students}/8。P1胜{P1}、L7胜{L7}、数值未分辨{ties}、缺失{4-len(valid_docs)}。平均Delta={summary['mean_Delta']}，中位Delta={summary['median_Delta']}。每篇相对效应在真实JSON/CSV中，独立单位document，最多n4；不把384点/层/head伪重复，不合并旧2doc为n6。

原阈值3.3300064494019496e-6为历史经验分辨率。正式结果揭示前依据冻结数学/软件/实际FLA profile、所选GPU self-KL零和各条件fresh64精确重复性冻结适用证据；这不是新32K误差上界校准。阈值内标NUMERICALLY_UNRESOLVED。原始完整精度Delta始终保留。未计算确认性sign-test；独立/对称假设未建立，即使4/4同向在该假设下单侧理论p也为0.0625。

## 资源与文件

{json.dumps(resource,ensure_ascii=False,indent=2)}

预算硬上限36新增GPUh；含canary、加载、parent、fresh64、正式两段和恢复的每worker launch-to-exit，双GPU分别累计；历史GPUh不计本轮额度。运行前和第一批真实parent后更新实际测量预测，保护性35.8GPUh提前停止留退出余量，没有提高36上限。所有实际CP和FP payload由manifest真实SHA绑定，大模型/CP只留服务器，不下载模型。最多两卡，未杀其他用户/重置GPU。

服务器目录：{D}；工作树：{W}。source_manifest、engineering_change_manifest、resource_admission、qualification_matrix、run_manifest、gpu_resource_ledger和hashes保存真实记录。完整8×384逐点校验仅在实际完成时为PASS，部分状态详见机器字段与原始数组。

## 首失败及证据边界

```json
{blocker}
```

出现实质身份/数值/资源失败后保留首STOP及已合格资产，不扩大诊断，不修数学或改变容差，不启动第二轮Clean-Room。E_future/Hidden KL及新proxy不计算；FP32载体仅声明persistent-QDQ/writeback，不声称physical resident INT8 storage。

## 五项科研判断

1. P1是否在四篇预先冻结文档上更好？本轮实际分类 **{classification}**，胜负、完整Delta与缺失如上；不以工程通过代替科学答案。
2. 与原2doc MIXED 1:1的关系？旧MIXED和posthoc永久保留。新四篇是同一冻结问题的独立前瞻扩展，条件筛选且n小，不合并确认性n6。
3. 是否有足够增量值得继续？{judgment}
4. 没有稳定增量是否停止当前路线？若完整结果多为L7胜，应暂停当前P1 selector推进；若混合或不完整，尚不能作普遍无效结论，但没有继续训练授权。
5. 下一步科学验证？{next_science} 仅科研建议，未执行任何后续训练/新数据/旋转/L8/L9/STE/Cayley/AIME/free generation/Qwen。

```json
{json.dumps(totals,ensure_ascii=False,indent=2)}
```
'''
 text(D/'FINAL_SCIENTIFIC_REPORT.md',report)
 text(D/'FAILURE_HISTORY.md','# Failure history\n\nHistorical executions01-05 are read-only, including their original STOPs. New Clean-Room first STOP:\n\n```json\n'+blocker+'\n```\nNo second Clean-Room attempt launched.\n')
 files={}
 for p in sorted(D.rglob('*')):
  if not p.is_file() or p.suffix=='.pt' or p.name.endswith('.lock') or '.partial' in p.name or p.name in ['artifact_sha256.txt','DELIVERY_VERIFICATION.json'] or p.name=='controller.log':continue
  files[str(p.relative_to(D))]=dict(bytes=p.stat().st_size,sha256=sha(p))
 manifest=dict(status='COMPLETE' if complete else 'STOP_ACTUAL_PARTIAL',task=TASK,source_commit=freeze['source_commit'],files=files,large_payloads=dict(checkpoints=str(D/'checkpoint_manifest.json'),FP_manifests=[u['result_path'] for u in units if u['condition']=='FP']),self_exclusion='manifest and SHA list excluded from manifest; SHA list includes manifest itself')
 publish(D/'hashes/DELIVERY_VERIFICATION.json',manifest)
 text(D/'hashes/artifact_sha256.txt',''.join(v['sha256']+'  '+n+'\n' for n,v in sorted(files.items()))+sha(D/'hashes/DELIVERY_VERIFICATION.json')+'  hashes/DELIVERY_VERIFICATION.json\n')
 print(json.dumps(dict(status='CPU_DELIVERY_SEALED',final=totals,small_files=len(files))),flush=True)

if __name__=='__main__':main()
