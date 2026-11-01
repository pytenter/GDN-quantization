"""Independent terminal reduction; preserves science files and original stops."""
import json,hashlib,csv,math,shutil
from pathlib import Path
from collect_terminal import R
D=R/'delivery';read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,v):(D/name).write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
plan=read(D/'GPU_PLAN.json');parent=D/plan['parent_delivery_relative'];oldplan=read(parent/'GPU_PLAN.json');s=read(D/'execution/SESSION_RESULT.json');ps=read(parent/'execution/SESSION_RESULT.json')
terminal=read(R/'operations/TERMINAL_ADMISSION.json');archive=read(D/'archive/FINAL_ARCHIVE_VERIFICATION.json')
results={};locations={};unitmap={u['id']:u for u in oldplan['units']}
for base,permit in ((parent,sha(parent/'GPU_PERMIT.json')),(D,sha(D/'GPU_PERMIT.json'))):
 for p in sorted((base/'execution').glob('*/RESULT.json')):
  v=read(p);name=p.parent.name;assert name not in results and v['unit']==name and v['permit_sha256']==permit
  results[name]=v;locations[name]=p.parent
micro=results['CODEC_MICRO'];methods={};pairs={};nextrows=[];measurements=[]
for name,v in results.items():
 u=unitmap[name]
 if u['kind']=='canary' and v['status']=='PASS':
  a=v['audit'];assert a['tokens']==u['tokens'] and a['next_consumption_checks']==(u['tokens']-1)*24 and a['qdq_layer_updates']==u['tokens']*24
  assert a['token0_QDQ'] and a['actual_values_and_bytes_compared'] and not a['unquantized_shadow'] and not a['packed_INT8_cache']
  assert len(list(locations[name].glob('TOKEN_*.json')))==u['tokens'];nextrows.append(dict(unit=name,**a))
 if u['kind']=='benchmark' and v['status']=='PASS':
  assert v['warmup']==v['measured']==128 and not v['observer'] and not v['profile'];assert math.isfinite(v['TPS']) and v['TPS']>0 and v['TPS']==128/v['seconds']
  measurements.append(dict(condition=u['method'],TPS=v['TPS'],ms_per_token=v['ms_per_token'],peak_allocated_bytes=v['peak_allocated'],peak_reserved_bytes=v['peak_reserved'],unit=name,warmup=128,measurement=128))
for m in ('B0','B1','B2'):
 paired=[]
 for n in (32,128,512):
  names=[f'{m}_{n}_R{i}' for i in (0,1)]
  if all(k in results and results[k]['status']=='PASS' for k in names):
   for t in range(n):
    a,b=[read(locations[k]/f'TOKEN_{t:04d}.json') for k in names];assert b['fresh_actual_bytes_compared'] is True and a['tensors']==b['tensors'] and a['raw']==b['raw']
   paired.append(n)
  else:continue
 maximum=max(paired,default=0);codec=micro['conditions'][m]['status'];has_tps=any(x['condition']==m for x in measurements)
 methods[m]=dict(CPU='CPU_CODEC_QUALIFIED',GPU_codec=codec,passed_fresh_lengths=paired,maximum_fresh_paired_tokens=maximum,persistent_9B='PERSISTENT_QDQ_CANARY_QUALIFIED' if maximum==512 else f'PASS_{maximum}_TOKEN_PREFIX_ONLY',fresh_process='FRESH_PROCESS_QUALIFIED' if maximum else 'NOT_RUN',short_runtime='INT8_SHORT_RUNTIME_QUALIFIED' if maximum==512 and has_tps else 'NOT_RUN',Future_KL_eligible=maximum==512 and codec=='GPU_CODEC_QUALIFIED')
 pairs[m]=dict(passed_lengths=paired,scope='Same GPU independent processes, actual full raw tensor values/bytes/dtypes/shapes/index')
save('GPU_CODEC_ONLY_QUALIFICATION.json',dict(status='GPU_CODEC_QUALIFIED',source='Immutable parent execution/CODEC_MICRO',conditions=micro['conditions'],parent_manifest_sha256=sha(parent/'SHA256_MANIFEST.json'),not_rerun=True,H_GPU_orthogonality_threshold=2e-6,H_CPU_GPU_transform_bit_proof='NOT_CLAIMED'))
save('GPU_FULL_MODEL_CANARY.json',dict(conditions=methods,session=s['status'],protocol=plan['source_protocol'],all_24_GDN_layers=True,scientific_worker_and_math_unchanged=True))
save('NEXT_CONSUMPTION_AUDIT.json',dict(status='PASS_COMPLETED_CANARIES_ONLY',units=nextrows,total_actual_kernel_input_checks=sum(x['next_consumption_checks'] for x in nextrows),total_layer_QDQ_writebacks=sum(x['qdq_layer_updates'] for x in nextrows),actual_value_and_bytes=True,unquantized_shadow=False,carrier='native_FP32_QDQ_not_packed_INT8'))
save('FRESH_PROCESS_DETERMINISM.json',dict(conditions=pairs,same_GPU_UUID=plan['device']['uuid'],parent32_reused_without_forward=True,no_long_context_claim=True))
save('INT8_THROUGHPUT_STATUS.json',dict(status='COMPLETE_MATCHED_MEASUREMENTS' if len(measurements)==3 else 'MEASURED_SUBSET_OR_NOT_RUN',measurements=measurements,unmeasured={m:'NOT_RUN' for m in methods if not any(x['condition']==m for x in measurements)},canary_audit_wall_is_not_TPS=True))
if measurements:
 with (D/'INT8_THROUGHPUT_RESULTS.csv').open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(measurements[0]));w.writeheader();w.writerows(measurements)
events=[json.loads(x) for x in (D/'execution/RESOURCE_EVENTS.jsonl').read_text().splitlines()];dispatch={x['unit'] for x in events if x['event']=='dispatch'};usage={x['unit'] for x in s['usage']}
assert dispatch==usage and not s['cleanup_usage_incomplete']
for name in dispatch:
 for event in ('dispatch','first_process_exit','first_owned_resource_release'):assert sum(x['event']==event and x.get('unit')==name for x in events)==1
assert s['own_all_UUID_release'] is True
lower=ps['reservation_GPUh_lower']+s['reservation_GPUh_lower'];upper=ps['reservation_GPUh_upper']+s['reservation_GPUh_upper']
save('RESOURCE_LEDGER.json',dict(status='TERMINAL_OWNED_GPU_RELEASE_VERIFIED',EXACT_GPUh='UNRESOLVED',parent_reservation_GPUh=[ps['reservation_GPUh_lower'],ps['reservation_GPUh_upper']],continuation_reservation_GPUh=[s['reservation_GPUh_lower'],s['reservation_GPUh_upper']],cumulative_reservation_GPUh=[lower,upper],no_time_budget=True,parent_budget_stop_preserved=True,parent_cleanup_flag_explanation='Immutable parent had a loop-name reporting flag bug; independent parent review resolved all seven dispatch intervals. No history modified.',current_all_dispatch_intervals_present=True,independent_release=terminal,session=s))
status='INT8_SHORT_RUNTIME_QUALIFIED' if s['status']=='FINITE_ROSTER_TERMINAL' and len(s['completed'])==15 and all(x['short_runtime']=='INT8_SHORT_RUNTIME_QUALIFIED' for x in methods.values()) else 'INCOMPLETE_OR_ENGINEERING_STOP'
save('TASK_RESULT.json',dict(status=status,conditions=methods,session_status=s['status'],inherited_units=plan['inherited_units'],new_complete=s['completed'],not_run=s['skipped'],raw_archive=archive['catalog'],raw_objects=archive['object_count'],raw_bytes=archive['total_bytes'],no_new_research=True))
failures=[dict(path=str(p.relative_to(D)),receipt=read(p)) for p in (D/'execution').rglob('*FAILURE*.json')]
save('FAILURE_HISTORY_INDEX.json',dict(failures=failures,parent_budget_stop=ps['status'],parent_seal_preserved=True,no_patch_retry=True))
for n in ('CPU_REFERENCE_PARITY.json','CPU_CODEC_QUALIFICATION.json'):
 if not (D/n).exists():shutil.copyfile(parent/n,D/n)
table=['| 条件 | 配对长度 | GPU Codec | TPS | ms/token | 测速峰allocated GiB |','|---|---|---|---:|---:|---:|']
for m,x in methods.items():
 z=next((v for v in measurements if v['condition']==m),None)
 table.append(f"| {m} | {x['passed_fresh_lengths']} | {x['GPU_codec']} | {z['TPS'] if z else 'NOT_RUN'} | {z['ms_per_token'] if z else 'NOT_RUN'} | {z['peak_allocated_bytes']/2**30 if z else 'NOT_RUN'} |")
writes=sum(x['qdq_layer_updates'] for x in nextrows);checks=sum(x['next_consumption_checks'] for x in nextrows)
report=f'''# Qwen/GDN Group32 资格验证实际报告

整体状态：**{status}**。新控制器终态 `{s['status']}`。人类“没有预算限制啊”取消本任务GPU时间预算；原0.5GPUh运行及其预算停止封存保持不变。原微型Codec和6条32-token轨迹只读复用，新运行仅原剩余15个单位，同GPU0 UUID `{plan['device']['uuid']}`，不重跑旧前缀。

{chr(10).join(table)}

1. **B0 canonical是否复现？** CPU全24层×32heads共12,582,912元素值和字节一致；原7项canonical AST不变。已完成GPU微型37 fixture×2严格字段比对。FP32 scalar除127保留CUDA reciprocal语义及Sym负零flags，未换为另一个舍入数学。
2. **B1 Group32 Sym是否通过？** GPU Codec为{methods['B1']['GPU_codec']}；实际9B persistent与同卡fresh完成长度见表。仅Key连续32分组，每Value列4组，不跨head/layer。没有将未运行长度补造PASS。
3. **B2 Asym资格？** GPU Codec为{methods['B2']['GPU_codec']}。FP32 scale、UINT8 zero-point/codes、int16减zp和decode严格参考通过；include-zero extrema、Tensor/Tensor除255和RNE顺序不变。真实模型范围见表；没有失败后改dtype/阈值。
4. **是否统一Runtime和时序？** 三者相同HISTORICAL_PORT、模型BF16、输入、H128、种子和设备，仅Codec不同。Worker及所有科学src与旧资格逐字节相同；本次仅资源准入、生命周期和新I/O root变化。历史完整二进制EXACT仍UNKNOWN。
5. **是否token0真实persistent？** PASS canary从token0逐token×24层QDQ并写回，共{writes}次层写回。query长度1；多token chunk被拒绝。未量化state没有参与下一次递归。载体是native FP32 QDQ state，48MiB，不是物理packed INT8。
6. **是否真实消费QDQ state？** 已完成canary共{checks}次native helper initial_state真实值及完整字节检查，和上一token整数编码的临时CPU decode比较；不是指针或hash替代。每完整N-token轨迹检查(N-1)×24次。
7. **fresh稳定性？** 表中每个长度在同UUID两独立模型进程运行，真实raw全tensor值/字节、dtype/shape/index一致，包括state/preQDQ/codes/metadata/decode/cache/logits及下一kernel输入。短512不能当32K/81K稳定性或误差上界。
8. **真实TPS？** 仅表中实际完成的128warmup+128measurement、CUDA同步、无Profiler、无逐token观测的匹配窗口。测速保留QDQ与finite检查；完整writeback/next-consumption值与字节观察仅在Canary开启，测速窗口关闭该观察，不能称其逐token重复审计。未测字段NOT_RUN，不借历史9.064610或FP13.774973填入；full-raw canary的wall包含观察、hash和归档，不能当推理TPS。
9. **显存/Clipping/数值风险？** 峰值为实际worker在模型常驻后reset统计，非加载峰；不把理论编码字节当显存节省。B0 scales384KiB，B1 scales1.5MiB，B2另zp384KiB；codes各12MiB，Sym临时bool负零flags12MiB。持久载体仍48MiB。已完成范围finite通过；canary完整round前clipping统计未单独保留，不能声称长程零overflow/saturation。H GPU正交阈值2e-6通过，不声称另有CPU/GPU H变换bit证明。
10. **后续32K资格？** 具备完整512 fresh及GPU Codec资格者：{[m for m,x in methods.items() if x['Future_KL_eligible']]}。这些仅为后续匹配Future-KL的短程前置资格，不证明功能精度改善。仍需独立文档、同协议FP、32768输入/384位置/全248320词表FP32 KL及新用户授权；没有启动32K、AIME、训练或新研究。

资源预留观察GPUh：原运行[{ps['reservation_GPUh_lower']}, {ps['reservation_GPUh_upper']}]，本次[{s['reservation_GPUh_lower']}, {s['reservation_GPUh_upper']}]，累计[{lower}, {upper}]。**EXACT_GPUh=UNRESOLVED**，轮询区间不是精确物理计算时间。所有实际dispatch均独立核验首次进程退出和全UUID释放；失败数量{len(failures)}。30分钟仅无真实phase/token进展watchdog，120秒仅清理观察，不是GPU运行预算。

归档：当前及所引用原raw共{archive['object_count']}对象、{archive['total_bytes']}字节，.29独立fresh流式SHA/bytes核验；原catalog只读，新增catalog原子发布。孤儿tmp {len(archive['orphan_tmp'])}。模型和大tensor未下载Mac，原CPU/FP/GPU封存未改写，失败与预算停止历史全部保留。
'''
(D/'FINAL_SCIENTIFIC_REPORT.md').write_text(report)
(D/'SOURCE_AND_RUNTIME_IDENTITY.md').write_text((parent/'SOURCE_AND_RUNTIME_IDENTITY.md').read_text()+'\n\n## 本次续跑实际身份\n\n'+f"500项源/控制绑定，SOURCE_FREEZE {sha(D/'SOURCE_FREEZE.json')}。相较原497项，仅GPU_PLAN、gpu_control、gpu_controller为资源/生命周期修订；3项新增为资源授权、continuation_control与CPU控制测试。scientific worker和全部其他原494项逐字节不变。新permit显式NO_GPU_TIME_BUDGET；同GPU0。原预算停止和当时状态仅作历史，不是当前预算。\n")
(D/'GROUP32_FUTURE_KL_NEXT_STAGE_PLAN.md').write_text((parent/'GROUP32_FUTURE_KL_NEXT_STAGE_PLAN.md').read_text()+f"\n\n## 最新续跑后资格\n\n当前合法短程候选：{[m for m,x in methods.items() if x['Future_KL_eligible']]}。独立32K任务仍未授权/未启动；当前以本目录TASK_RESULT为准，原文的NOT_RUN属于历史时间点。\n")
(D/'EXECUTION_PLAN.md').write_text('本次原剩余15个单位，无GPU时间预算；完整实际结果见TASK_RESULT.json及FINAL_SCIENTIFIC_REPORT.md。旧微型/32只读复用，不重跑。原数学、种子、输入、设备及失败即停保持。\n')
(D/'README_CURRENT_STATUS.md').write_text('当前结果以本目录TASK_RESULT、FINAL_SCIENTIFIC_REPORT及新execution为准。inputs/parent_gpu_delivery是原预算停止交付，所有旧报告/STOP/seal保持原样。SOURCE_FREEZE中的执行前CPU/协议状态不回写。independent_analysis为只读分析和交付工具，不参与模型计算。\n')
print(json.dumps(dict(status=status,conditions=methods,TPS_measurements=len(measurements),raw_objects=archive['object_count']),ensure_ascii=False))

root=read(D/'ROOT.json');root.update(status=status,actual_result='TASK_RESULT.json',actual_report='FINAL_SCIENTIFIC_REPORT.md',GPU_dispatches=len(dispatch),GPU_queries='ACTUAL_RECEIPTS',Driver_API_calls='ACTUAL_PER_UNIT_IDENTITY_RECEIPTS');save('ROOT.json',root)
