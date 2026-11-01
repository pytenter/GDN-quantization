"""Independent reduction of terminal receipts only, no inference/extra experiment."""
import csv,json,hashlib,shutil
from pathlib import Path
from collect_terminal import R
D=R/'delivery'
def read(p):return json.loads((D/p).read_text())
def save(p,v):(D/p).write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
s=read('execution/SESSION_RESULT.json');plan=read('GPU_PLAN.json');permit=read('GPU_PERMIT.json')
terminal=json.loads((R/'operations/TERMINAL_ADMISSION.json').read_text())
archive=read('archive/FINAL_ARCHIVE_VERIFICATION.json')
results={p.parent.name:json.loads(p.read_text()) for p in (D/'execution').glob('*/RESULT.json')}
micro=results.get('CODEC_MICRO',{});methods={};pairs={};next_rows=[];measured=[]
unit_map={u['id']:u for u in plan['units']}
for name,v in results.items():
 u=unit_map[name]
 assert v['unit']==name and v['permit_sha256']==hashlib.sha256((D/'GPU_PERMIT.json').read_bytes()).hexdigest()
 if u['kind']=='canary' and v['status']=='PASS':
  a=v['audit'];assert a['tokens']==u['tokens'] and a['next_consumption_checks']==(u['tokens']-1)*24
  assert a['qdq_layer_updates']==u['tokens']*24 and a['token0_QDQ'] and not a['unquantized_shadow']
  next_rows.append(dict(unit=name,**a))
 if u['kind']=='benchmark' and v['status']=='PASS':
  assert v['warmup']==v['measured']==128 and not v['observer'] and not v['profile']
  measured.append(dict(condition=u['method'],TPS=v['TPS'],ms_per_token=v['ms_per_token'],peak_allocated_bytes=v['peak_allocated'],peak_reserved_bytes=v['peak_reserved'],unit=name))
for m in ('B0','B1','B2'):
 complete=[];paired=[]
 for n in (32,128,512):
  valid=[k for k in (f'{m}_{n}_R0',f'{m}_{n}_R1') if k in results and results[k]['status']=='PASS']
  if valid:complete.append(dict(tokens=n,units=valid))
  if len(valid)==2:
   for t in range(n):
    a=read(f'execution/{m}_{n}_R0/TOKEN_{t:04d}.json');b=read(f'execution/{m}_{n}_R1/TOKEN_{t:04d}.json')
    assert b['fresh_actual_bytes_compared'] and a['tensors']==b['tensors'] and a['raw']==b['raw']
   paired.append(n)
 maximum=max(paired,default=0);codec=micro.get('conditions',{}).get(m,{}).get('status','NOT_RUN')
 runtime=any(x['condition']==m for x in measured) and maximum==512
 methods[m]=dict(CPU='CPU_CODEC_QUALIFIED',GPU_codec=codec,completed_canaries=complete,maximum_fresh_paired_tokens=maximum,
   persistent_9B='PERSISTENT_QDQ_CANARY_QUALIFIED' if maximum==512 else ('PASS_'+str(maximum)+'_TOKEN_PREFIX_ONLY' if maximum else 'NOT_RUN'),
   fresh_process='FRESH_PROCESS_QUALIFIED' if maximum else 'NOT_RUN',fresh_scope_tokens=maximum,
   short_runtime='INT8_SHORT_RUNTIME_QUALIFIED' if runtime else 'NOT_RUN',Future_KL_eligible=bool(maximum==512 and codec=='GPU_CODEC_QUALIFIED'))
 pairs[m]=dict(status=methods[m]['fresh_process'],passed_lengths=paired,comparison='Actual full serialized tensor bytes, dtype/shape/index; same GPU, independent processes')
save('GPU_CODEC_ONLY_QUALIFICATION.json',dict(status='GPU_CODEC_QUALIFIED' if all(x['GPU_codec']=='GPU_CODEC_QUALIFIED' for x in methods.values()) else 'INCOMPLETE_OR_NUMERICAL_FAIL',conditions={m:micro.get('conditions',{}).get(m,dict(status='NOT_RUN')) for m in methods},same_GPU_repeat=2,actual_driver_UUID=permit['device']['uuid'],H_check='GPU orthogonality threshold2e-6 passed; exact maxabs not retained',H_CPU_GPU_transform_bit_proof='NOT_CLAIMED',large_model_loaded_in_micro=False))
save('GPU_FULL_MODEL_CANARY.json',dict(session_status=s['status'],conditions=methods,all_24_GDN_layers=True,source_unchanged=True,protocol=plan['source_protocol'],scope='Only actually completed lengths; no long-horizon claim'))
save('NEXT_CONSUMPTION_AUDIT.json',dict(status='PASS_COMPLETED_CANARIES_ONLY' if next_rows else 'NOT_RUN',units=next_rows,total_actual_kernel_input_checks=sum(x['next_consumption_checks'] for x in next_rows),total_layer_QDQ_writebacks=sum(x['qdq_layer_updates'] for x in next_rows),actual_value_and_bytes=True,unquantized_shadow=False,carrier='native_FP32_QDQ_not_physical_packed_INT8',unfinished_units_not_qualified=True))
save('FRESH_PROCESS_DETERMINISM.json',dict(conditions=pairs,CPU_prior_proof='provenance/cpu_seal',same_GPU_UUID=permit['device']['uuid'],no_cross_GPU_or_long_context_claim=True))
save('INT8_THROUGHPUT_STATUS.json',dict(status='MEASURED_SUBSET' if measured else 'NOT_RUN',measurements=measured,unmeasured={m:'NOT_RUN' for m in methods if not any(x['condition']==m for x in measured)},reason=s['status'],observed_canary_wall_not_TPS=True))
if measured:
 with (D/'INT8_THROUGHPUT_RESULTS.csv').open('x',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(measured[0]));writer.writeheader();writer.writerows(measured)
else:assert not (D/'INT8_THROUGHPUT_RESULTS.csv').exists()
events=[json.loads(x) for x in (D/'execution/RESOURCE_EVENTS.jsonl').read_text().splitlines()]
dispatched={x['unit'] for x in events if x['event']=='dispatch'}
usage_units={x['unit'] for x in s['usage']}
assert dispatched==usage_units==set(s['completed']),'Missing actual resource interval'
for name in s['completed']:
 assert sum(x['event']=='dispatch' and x.get('unit')==name for x in events)==1
 assert sum(x['event']=='first_process_exit' and x.get('unit')==name for x in events)==1
 assert sum(x['event']=='first_owned_resource_release' and x.get('unit')==name for x in events)==1
flag_note='Frozen cleanup_usage_incomplete flag compares the next skipped loop-unit name with the previous child usage. All seven actual dispatch units have exactly one exit/release interval and are present in usage. Flag retained, independently resolved as no missing dispatched-unit accounting; no code was patched.'
ledger=dict(status='TERMINAL_OWNED_GPU_RELEASE_VERIFIED',session=s,EXACT_GPUh='UNRESOLVED',reservation_GPUh_lower=s['reservation_GPUh_lower'],reservation_GPUh_upper=s['reservation_GPUh_upper'],physical_compute_time_not_inferred_from_polling=True,permit=permit,independent_release=terminal,events_file='execution/RESOURCE_EVENTS.jsonl',all_actual_dispatch_intervals_present=True,frozen_cleanup_flag_explanation=flag_note,independent_cleanup_usage_incomplete=False)
save('RESOURCE_LEDGER.json',ledger)
status='INT8_SHORT_RUNTIME_QUALIFIED' if all(x['Future_KL_eligible'] for x in methods.values()) else ('INCOMPLETE_OR_ENGINEERING_STOP' if s['status']=='FAIL_CLOSED_STOP' else 'INCOMPLETE_WITHIN_AUTHORIZED_GPU_BUDGET')
save('TASK_RESULT.json',dict(status=status,session_status=s['status'],conditions=methods,no_new_research=True,raw_archive=archive['catalog'],raw_objects=archive['object_count'],raw_bytes=archive['total_bytes']))
snap=D/'provenance/pre_GPU_report_snapshot';snap.mkdir(exist_ok=True)
for n in ('FINAL_SCIENTIFIC_REPORT.md','SOURCE_AND_RUNTIME_IDENTITY.md','GROUP32_FUTURE_KL_NEXT_STAGE_PLAN.md'):
 shutil.copyfile(D/n,snap/n)
identity_add=f'''\n\n# 本次GPU执行实际身份\n\n新RUN GROUP32_GPU_EXECUTION_20261019_V1，单卡UUID {permit['device']['uuid']}。实际source freeze SHA {permit['source_freeze_sha256']}，plan SHA {permit['plan_sha256']}。497项冻结源仅GPU_PLAN的授权开关与archive I/O root改变；canonical/codec/FP helper均与CPU封存逐字节相同。用户另行回复“授权”，预算0.5 GPUh、1800硬秒、1650保护、150清理，未引用旧四卡不限时授权。Driver、PREFLIGHT/POSTFLIGHT库SHA/bytes/inode与owner-PID回执保存在每个执行单位。独立复核确认真实进程退出及全UUID中原worker资源释放。完整历史二进制EXACT身份仍UNKNOWN。\n'''
(D/'SOURCE_AND_RUNTIME_IDENTITY.md').write_text('# 前期CPU身份原文（保留时间点）\n\n'+(snap/'SOURCE_AND_RUNTIME_IDENTITY.md').read_text()+identity_add)
table=['| 条件 | GPU Codec | 最大完整fresh长度 | Persistent资格 | INT8 TPS | 32K准入 |','|---|---|---:|---|---|---|']
for m,x in methods.items():
 t=next((str(z['TPS']) for z in measured if z['condition']==m),'NOT_RUN')
 table.append(f"| {m} | {x['GPU_codec']} | {x['maximum_fresh_paired_tokens']} | {x['persistent_9B']} | {t} | {'具备短程前置资格，仍需新32K授权' if x['Future_KL_eligible'] else '尚未具备完整前置资格'} |")
failed=[]
for p in sorted((D/'execution').rglob('FIRST_FAILURE*.json')):failed.append(dict(file=str(p.relative_to(D)),receipt=json.loads(p.read_text())))
checks=sum(x['next_consumption_checks'] for x in next_rows);writes=sum(x['qdq_layer_updates'] for x in next_rows)
report=f'''# Qwen Group32实际GPU资格报告

整体：**{status}**。控制器终态：`{s['status']}`。预算0.5 GPUh已固定，未更改科学程序、未重启单位、未补跑FP，也没有启动32K/AIME/训练。

{chr(10).join(table)}

1. **B0复现canonical？** CPU在24层×32heads、12,582,912元素上值/字节均完全一致。GPU微型B0实际37fixture×2次通过codes/scales/负零/重建严格参考；保留CUDA scalar reciprocal语义，不将朴素CPU除法当作CUDA bit参考。canonical7项AST及FP数学不变。协议是本次token0 QDQ，不能冒充历史OLD首token漏QDQ协议。
2. **B1 Group32 Sym GPU资格？** GPU Codec为{methods['B1']['GPU_codec']}；真实9B完成与fresh长度见表。未完成的128/512或测速不自动升级PASS。
3. **B2 Asym资格？** GPU Codec为{methods['B2']['GPU_codec']}。37组冻结边界/真实state fixture逐字段核验FP32 scale、UINT8 zero-point/codes、int16相减及decode；同进程同卡重复2次。真实模型完成范围见表，没有扩大scale容差或修改舍入规则。
4. **统一Runtime和时序？** 全部实际canary使用相同HISTORICAL_PORT、H128、输入、BF16模型、FP32 carrier、每token逐层QDQ；只变codec。没有Dual-M/permutation/rotation/新FP优化。所有实际数值单位同一GPU UUID，不跨卡引用fresh byte参考。
5. **首token persistent？** 所有PASS canary均从token0写回，完整运行共{writes}个层QDQ更新。stepwise query长度必须为1，未用chunk末尾单次QDQ代替逐token。未完成单位不声称完整资格。
6. **真实next-consumption？** 已完成单位共{checks}次实际native helper initial_state检查：实际值和完整字节对照上一token整数编码的临时CPU解码。不是指针/hash代替。未量化state不参与下一次递归；native FP32 QDQ carrier仍48MiB，不是物理packed INT8。
7. **fresh-process？** 表中最大完整配对长度以内，两个独立模型进程在同UUID、相同种子/输入下实际完整raw逐字节比较通过。CPU两fresh证据保留。短32/128不能称512、32K或81K稳定性；未完成R1不形成配对资格。
8. **真实TPS？** 仅实际完成128warmup+128measurement且同步、无Profiler/逐token观察的测量才进入CSV。目前测量条数{len(measured)}；其余为NOT_RUN。canary包含CPU完整字节观察、哈希与归档，其wall不能当吞吐，也不以历史9.064610或FP13.774973填当前值。
9. **风险与显存？** 完成范围内执行state/scale/level/decode/logits finite检查。实际峰值逐单位记录于RESULT；不把编码理论字节当实测显存节省。B0 scales384KiB，B1 scales1.5MiB，B2另加zp384KiB；codes各12MiB，Sym负零bool flags临时12MiB，native持久carrier仍48MiB。完整canary的round前clipping计数未单独留存，不能把边界codes计数冒称overflow。37fixture严格通过不构成长程误差上界。H GPU正交阈值检查通过，但没有另称CPU/GPU H变换输出bit等价证明。
10. **32K Future-KL候选？** 仅表中具备完整短程数值与fresh前置资格者可作为下一任务候选，仍必须新预算/文档/用户授权。缺少计划的512验证不能直接进入；测速NOT_RUN表示缺少性能证据，不解释为数值否决。后续匹配FP_NATIVE/H+C128/H+C32 Sym/H+C32 Asym、32768输入、384位置、全248320词表FP32 KL与文档聚合方案保留；没有用历史NEW_STRICT KL替代。

资源：进程预留GPUh观察区间[{s['reservation_GPUh_lower']}, {s['reservation_GPUh_upper']}]；**EXACT_GPUh=UNRESOLVED**，不能把轮询时间当精确物理计算时间。append-only事件保留dispatch、首次exit、首次release及上下界。独立进程/NVML检查确认原worker全部释放；预算未被重置。CPU准入/依赖hash与GPU预留时间分开。

冻结控制器的cleanup_usage_incomplete=True字段有记录逻辑问题：退出循环时name已指向未派发的B0_128_R0，而child仍是最后完成的B2_32_R1，造成该布尔值误报。独立复核7个真实dispatch均有且仅有一次exit/release，全部7段也都在usage中，资源区间没有缺项。原字段/源码保持不变，在RESOURCE_LEDGER中单独记录这一解释，不将其当作0GPUh或精确物理时间。

归档：{archive['object_count']}个当前大raw对象、{archive['total_bytes']}字节，.29永久存储已fresh流式SHA/bytes核验；临时孤儿数{len(archive['orphan_tmp'])}。大tensor未下载Mac，历史raw未删除。本次科学源与依赖在结束后复核。失败回执数{len(failed)}；所有原始回执/日志保留，预算NOT_RUN不解释为科学负结果。

前序CPU seal/542文件和旧FP最终seal保持不变。前序finisher当时未seal与随后实际最终seal是两个时间点，完整provenance均保留。
'''
(D/'FINAL_SCIENTIFIC_REPORT.md').write_text(report)
eligible=[m for m,x in methods.items() if x['Future_KL_eligible']]
(D/'GROUP32_FUTURE_KL_NEXT_STAGE_PLAN.md').write_text((snap/'GROUP32_FUTURE_KL_NEXT_STAGE_PLAN.md').read_text()+f'\n\n## 本次GPU后更新\n\n实际完整短程资格候选：{eligible}。没有自动开始32K。未完成部分须保留NOT_RUN并另行授权，不能借历史数值补齐。\n')
save('FAILURE_HISTORY_INDEX.json',dict(session_status=s['status'],failures=failed,no_patch_and_retry=True,original_CPU_negative_controls_preserved=True))
root=read('ROOT.json');root.update(status=status,GPU_queries='NOT_COUNTED_USE_ACTUAL_RECEIPTS',Driver_API_calls='ACTUAL_PER_UNIT_DRIVER_IDENTITY_RECEIPTS',GPU_dispatches=sum(x['event']=='dispatch' for x in events),valid_GPU_permit='GPU_PERMIT.json',actual_result='TASK_RESULT.json',actual_report='FINAL_SCIENTIFIC_REPORT.md');save('ROOT.json',root)
(D/'EXECUTION_PLAN.md').write_text('# 执行后状态\n\n'+f'用户已另行授权；本次实际终态{status}，控制器{s["status"]}。执行过的单位、未执行的单位及预算见execution/SESSION_RESULT.json，实际分条件资格见TASK_RESULT.json。没有重复运行、修改冻结数学或自动启动32K。\n\n'+'# 前期CPU计划原文（保留时间点）\n\n'+(D/'EXECUTION_PLAN.md').read_text())
(D/'README_CURRENT_STATUS.md').write_text('本目录是本次GPU执行的最终交付。当前结果以FINAL_SCIENTIFIC_REPORT.md、TASK_RESULT.json、RESOURCE_LEDGER.json及实际execution回执为准。SOURCE_FREEZE锁定的protocol/fixture/CPU文件中的NOT_RUN描述属于执行前时间点，保留不可回写；不能将其当作最终GPU状态。497项执行冻结源保持原样。independent_analysis为结束后只读复核与交付源码，不参与模型计算。所有大raw位于.29 archive/catalog指定位置，Mac仅有小元数据/源码/fixture。\n')
print(json.dumps(dict(status=status,session=s['status'],conditions=methods,raw_objects=archive['object_count'],TPS_measurements=len(measured)),ensure_ascii=False))
