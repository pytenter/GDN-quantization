#!/usr/bin/env python3
"""Prepared CPU-only closeout, usable ONLY after explicit root CaseC decision.
No GPU imports or execution, candidate development, scoring or automatic decision.
This script does not seal or claim final delivery verification; separate checks do.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile

TASK="QWEN_GDN_DUAL_M_EQUIVALENT_SPEEDUP_V1"


def require(ok,message):
    if not ok:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(4*1024**2),b''):h.update(chunk)
    return h.hexdigest()


def text_number(value,format_spec='.6f'):
    return '未知/未建立' if value is None else format(value,format_spec)


def json_bytes(value):return (json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--analysis',type=Path)
    parser.add_argument('--decision',type=Path,required=True,help='Explicit root-authored decision binding complete profile SHA')
    parser.add_argument('--output',type=Path,help='Default root/profile_only_closeout; does not overwrite any prior report')
    args=parser.parse_args();root=args.root.resolve();analysis=(args.analysis or root/'phase0_analysis').resolve();output=(args.output or root/'profile_only_closeout').resolve()
    profile_path=analysis/'BASELINE_RUNTIME_PROFILE.json';profile=read(profile_path);decision=read(args.decision)
    require(profile['task_id']==TASK and profile['status']=='ACTUAL_PHASE0_PROFILE_REDUCED_NO_OPTIMIZATION_DECISION','Actual complete Phase0 reduction required')
    require(decision.get('task_id')==TASK and decision.get('status')=='ROOT_DECISION_RECORDED','Missing root decision')
    require(decision.get('decision')=='CASE_C_PROFILE_ONLY_STOP' and decision.get('candidate_created') is False,'Only no-candidate root CaseC closure accepted')
    require(decision.get('BASELINE_RUNTIME_PROFILE_sha256')==sha(profile_path),'Root decision must bind actual profile SHA')
    require(isinstance(decision.get('reason'),str) and decision['reason'].strip(),'Root decision must state its actual evidence/reason')
    require(profile['release_verification']['status']=='PASS','True process/release proof required')
    require(sha(root/'GPU_RESOURCE_LEDGER.json')==profile['phase0_ledger_sha256'],'Ledger changed after Phase0 reduction; refresh independent proof rather than fabricate closure')
    ledger=read(root/'GPU_RESOURCE_LEDGER.json')
    require(ledger['all_own_GPUs_released'] and ledger['cumulative_GPU_hours']<=2,'Release/budget mismatch')
    require(not (root/'FIRST_FAILURE.json').exists(),'A real execution failure needs FAIL_CLOSED_STOP delivery, not CaseC closure')
    sources=read(root/'SOURCE_MANIFEST.json')
    for entry in sources['execution_sources']:
        p=root/entry['path'];require(p.is_file() and p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256'],'Frozen execution source changed')
    require(sha(root/'SOURCE_MANIFEST.json')==profile['source_manifest_sha256'],'Source manifest differs')
    prior_tests=root/'audit_reference/REFERENCE_CPU_CODEC_TESTS.json';prior_diag=root/'audit_reference/REFERENCE_CPU_CODEC_DIAGNOSTICS.json'
    require(prior_tests.exists() and prior_diag.exists(),'Retain initial official strictbyte failure and diagnostics')
    cpu=read(prior_tests);diag=read(prior_diag)
    history=dict(initial_CPU_official_strict_comparison=dict(path=str(prior_tests),sha256=sha(prior_tests),status=cpu['status'],scope=cpu['scope']),
                 independent_CPU_difference_diagnostics=dict(path=str(prior_diag),sha256=sha(prior_diag),status=diag['status'],scope=diag['scope']),
                 interpretation='OriginalDualM vs officialfit_state strict signedzero byte differences are retained. Nooptimizedcandidate exists; these are not candidatefailure and never silently changed toPASS.')
    histories=[]
    for case in diag.get('cases',[]):
        d=case.get('dequant_difference_diagnostics')
        if d:histories.append(dict(name=case['name'],status=case['status'],**d))
    history['actual_CPU_difference_cases']=histories
    require({row['condition'] for row in profile['short_noninstrumented_baselines']}=={'FP_NATIVE','H_C128','H_DUAL_M_REFERENCE'},'Three actualbaseline conditions required')
    ref=next(r for r in profile['short_noninstrumented_baselines'] if r['condition']=='H_DUAL_M_REFERENCE')
    micro=profile['codec_micro'];ls_us=micro['LS_grouped_device_attributed_us'];ls_ratio=micro['LS_fraction_of_exclusive_named_stage_device_attribution']
    prof=next(r for r in profile['separate_instrumented_profiles'] if r['condition']=='H_DUAL_M_REFERENCE')
    stages=prof['original_disjoint_CPU_phase_scopes']
    ordered=sorted(stages,key=lambda r:r['device_total_us'],reverse=True)
    common=dict(task_id=TASK,status='NOT_RUN_NO_CANDIDATE',candidate_created=False,reason=decision['reason'],root_decision_sha256=sha(args.decision),phase0_profile_sha256=sha(profile_path),scope='Nooptimizedimplementation or old-vs-new numerical/performancecomparison executed')
    artifacts={}
    for name in ('GPU_CODEC_PARITY.json','RECURRENT_STATE_PARITY.json','FRESH_PROCESS_QUALIFICATION.json','OPTIMIZED_RUNTIME_PROFILE.json'):
        artifacts[name]=json_bytes(dict(common,gate_scope=name.removesuffix('.json')))
    artifacts['CPU_CODEC_PARITY.json']=json_bytes(dict(common,retained_reference_history=history))
    artifacts['SOURCE_DIFF_MANIFEST.json']=json_bytes(dict(task_id=TASK,status='REFERENCE_UNCHANGED_NO_OPTIMIZED_CANDIDATE',candidate_files=[],candidate_code_changes=0,
        frozen_execution_manifest_sha256=sha(root/'SOURCE_MANIFEST.json'),frozen_execution_sources_verified_unchanged=len(sources['execution_sources']),
        added_measurement_python_sources=[entry for entry in sources['execution_sources'] if entry['path'].endswith('.py')],
        added_postexecution_analysis='Separate stdlibCPU reducer/report scripts; not changes tohistoricalcodec/GDN/Hmath; root seal separately hashes actualscripts',
        reference_history=history))
    artifacts['GATE_STATUS_FINAL.json']=json_bytes(dict(task_id=TASK,status='PROFILE_ONLY_COMPLETE_AND_STOP',root_decision=decision,
        reference_source_identity='PASS',Phase0_actual_measurements='PASS_9SHORT_BASELINE_PLUS3_PROFILE',real_single_layer_stage_instrumentation_parity='BYTE_EXACT_INSTRUMENTED_VS_ORIGINAL_ONLY',
        old_official_CPU_byte_contract=cpu['status'],old_official_byte_contract_scope='REFERENCE_DIAGNOSTIC_NOT_OPTIMIZED_CANDIDATE',
        optimized_codec_CPU_parity='NOT_RUN_NO_CANDIDATE',optimized_codec_GPU_parity='NOT_RUN_NO_CANDIDATE',old_new_recurrent_parity='NOT_RUN_NO_CANDIDATE',fresh_process_optimized_parity='NOT_RUN_NO_CANDIDATE',
        optimized_performance='NOT_RUN_NO_CANDIDATE',formal_AIME='NOT_RUN',new32Kscience='NOT_RUN',new_rotation_training='NOT_RUN',
        actual_GPU_hours=ledger['cumulative_GPU_hours'],all_own_GPU_released=True,final_seal_and_download_verification='SEPARATE_REQUIRED_AFTER_THIS_CLOSEOUT'))
    comparison=dict(condition='H_DUAL_M_REFERENCE_vs_OPTIMIZED',measurement_scope='128fixeddecodeafter128retainedwarmup_samePhase0GPU',reference_actual_repeats=3,
        reference_decode_window_wall_median_seconds=ref['decode_window_wall_seconds']['p50'],reference_decode_tokens_per_second_median=ref['decode_tokens_per_second']['p50'],
        candidate_status='NOT_RUN_NO_CANDIDATE',candidate_decode_window_wall_median_seconds='',candidate_decode_tokens_per_second_median='',wall_speedup='',
        reason='Nooptimizedcandidate implemented ormeasured; missingnumbersremainblank, notzero orfakeequivalence.')
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(comparison));writer.writeheader();writer.writerow(comparison);artifacts['SPEEDUP_COMPARISON.csv']=stream.getvalue().encode()
    artifacts['OPTIMIZATION_DESIGN.md']=('# 优化状态\n\nNOT_RUN_NO_CANDIDATE。根Agent核验完整Phase0数据后明确决定Case C，未实现优化候选，未自动扩展GPU阶段。\n\n根决定：'+decision['reason']+'\n\n原codec/H/GDN数学、FP16scale、INT8位宽、LS、最终re-encode及persistent QDQ均未改变；不选择额外blocksize/packed runtime/SGLang方案。此状态不是候选数值失败，也不声称速度提升。\n').encode()
    phases='\n'.join(f"|{r['name']}|{r['count']}|{r['cpu_total_us']:.3f}|{r['device_total_us']:.3f}|" for r in ordered)
    artifacts['BOTTLENECK_ANALYSIS.md']=(f"# 完整Phase0后的根决定\n\n{decision['reason']}\n\n同一次instrumented H_DualM的三个原始非嵌套owner范围如下；设备归属量和CPU范围耗时不是相同口径，不能相加为wall，也不能再加其ATen子项。\n\n|范围|count|CPU total us|device attributed us|\n|---|---:|---:|---:|\n{phases}\n\n单层真实state的LS grouped device归属 {text_number(ls_us,'.3f')} us；占八个命名stage归属总量 {text_number(None if ls_ratio is None else 100*ls_ratio,'.3f')}%。该局部比例不是24层完整模型wall占比，不能乘24替代实测。严格非instrumented Amdahl ceiling仍按BASELINE_RUNTIME_PROFILE记录UNKNOWN；instrumented删除整个QDQ owner的条件估计不是候选可达保证。\n\n性能记录保留CPU发射、CUDA活动区间union、内核名/count和同步信息；wall减GPU active union不被当成纯CPU overhead。任务ledger包括加载、warmup、采集、CPU Kineto解析、Chrome trace导出、真实释放前全部GPU占用，不能只报告采集窗口耗时。\n").encode()
    rows='\n'.join(f"|{r['condition']}|3|{r['decode_tokens_per_second']['p50']:.6f}|{r['decode_tokens_per_second']['min']:.6f}–{r['decode_tokens_per_second']['max']:.6f}|{r['decode_window_wall_seconds']['p50']:.6f}|" for r in profile['short_noninstrumented_baselines'])
    report=f'''# Dual-M 等价加速：Profile-only 技术报告

最终状态：**PROFILE_ONLY_COMPLETE_AND_STOP**。这是根Agent在完整、真实Phase0报告后作出的明确决定，不是汇总器自动选择的方向。没有实现优化候选，没有宣称正向加速，也没有优化候选失败。

根决定及实际依据：{decision['reason']}

Phase0实际完成9次非instrumented128-token基线（三条件×三重复）和另3次instrumented128-token profile；每次都继承128-token warmup cache，不重跑测量前缀。真实单层GDN state codec micro有3次有效重复及一次独立stage诊断，其instrumentation输出codes、FP16行列scale和QDQ均与原codec逐字节一致。没有执行候选、512/2048/8192扩展、32K科学实验或AIME生成评分。

冻结worker原派生字段把CUDA user_annotation混入kernel/active统计，原JSON保留为RAW_DERIVED_UNCORRECTED。本次独立CPU分析逐SHA验证Chrome trace，按cat==kernel、gpu_memcpy/memset和gpu_user_annotation分开并核对correlation；CPU phase只取cpu_total_us>0的行。修正kernel/active不改变真实wall或冻结数值执行。单层micro未导出Chrome分类trace，其真实kernel数/active union保持UNKNOWN，LS只用CPU owner对应的设备归属量。

|条件|实际重复数|decode token/s中位|最小–最大|128-token decode wall中位秒|
|---|---:|---:|---:|---:|
{rows}

上述为同固定token非AIME回放、同卡、同测量窗；不是历史AIME自由生成速度比。每token host iteration p50/p95包含参考内部检查/等待，但不等于GPU完整token latency；CUDA stream event间隔包括idle gaps，实际device active union另报。instrumented profile有额外开销，不用其TPS冒充基线。

## 用户12问

1. **实际最耗时操作？** 根决定依据如上。完整profile保存真实kernel名称/count及三个原始phase范围，前三设备活动与CPU self操作分别列在BASELINE_RUNTIME_PROFILE。设备活动sum、CPU owner归属与wall不同，不能混合求和。
2. **Column LS占多少？** 单层真实state的LS grouped device归属 {text_number(ls_us,'.3f')} us，占八个命名stage设备归属合计 {text_number(None if ls_ratio is None else 100*ls_ratio,'.3f')}%。这是一次诊断的局部活动比例，包含独立trace观察合同，不是端到端wall比例；若归属不可用则保持未知。
3. **额外开销来自哪里？** 以根决定和profile的CPU launch/scalar/sync、kernel活动、model/codec/finite原始范围为证据。原DualM包含一次LS、批量cat与多轮中间state、finite/scale安全标量和原event等待。不能只凭源码宣布LS主导，也不把wall−deviceunion称为纯CPU发射时间。
4. **官方融合哪些适用？** Row/column融合边界、减少中间tensor与coalesced访存可参考。官方packed GDN、异步pool、SGLang或mixedprecision/pivot runtime不是本任务可直接替换路径；快速倒数/FMA/归约重排不自动bitwise exact。七指定官方源固定commit f2deac58131219e6aecff274f8eabe88e08d2e3c，缓存manifest及commit raw字节已核。
5. **最终改哪些代码？** 新增有界测量controller/worker、只读审计和CPU分析收口脚本；原DualM/H/GDN冻结源未修改。没有新增优化codec/kernel，SOURCE_DIFF_MANIFEST列明实际冻结执行源与候选空集合。
6. **codes/scales/QDQ exact吗？** 原codec对自身stage instrumentation在真实fixture上四项byte-exact，这只证明诊断观察没有改数值。旧vs新候选CPU/GPU parity均NOT_RUN，因为没有候选。初次旧参考vs官方fit_state严格字节测试FAIL永久保留，CPU诊断将部分差异定位signed zero；不能改成PASS，也不是候选FAIL，详见CPU_CODEC_PARITY内原记录SHA。
7. **下一token消费正确？** 原参考短回放每token使用原Observe检查实际initial_state.data_ptr与上一token QDQ carrier，实际次数随12记录保存。优化臂不存在，old-vs-new状态/输出/采样序列及fresh重复资格均NOT_RUN。FP_NATIVE/H历史门禁失败未修补或放宽。
8. **Codec-only加速多少？** 未知/NOT_RUN_NO_CANDIDATE。只报告原单层codec实际3重复和stage诊断，没有优化测速，不能给1.0×或0%冒充结果。
9. **全模型加速多少？** 未知/NOT_RUN_NO_CANDIDATE。原H_DualM三次128-token测量中位 {ref['decode_tokens_per_second']['p50']:.6f} token/s；没有优化臂。SPEEDUP_COMPARISON仅一行明确原值与候选缺失，speedup留空。
10. **是否值得下一步32K？** 本轮没有合格优化候选或≥10%同条件完整decode提升证据，不授权自动长程验证。根停止原因如上；不从局部CUDA活动加速推断32K功能收益。
11. **显存与临时tensor下降吗？** 没有候选对照，下降量未知。原24层reconstructed FP32 persistent carrier为50,331,648B；原编码INT8与FP16scale payload另算，不是physicalpackedINT8常驻。各实际CUDA allocated/reserved峰值已记录，不能把多个不同时存活临时tensor静态相加当峰值。
12. **还有独立下一阶段瓶颈吗？** 源审计和实际profile保留可供判断，但本次不自动追加kernel、packed runtime、训练或benchmark。任何后续候选应独立授权、保持原codec数学/QDQ，先bitwise数值与真实next-token/fresh资格，再测同窗wall加速。

本任务累计GPU占用 **{ledger['cumulative_GPU_hours']:.12f} GPUh**，包含加载、warmup、全部回放、profiler采集、CPU key_averages/events解析、trace导出、codec诊断及进程退出/NVML释放前全部占用。`decode_window_wall`仅采集窗口，不能当作总profile成本；host解析/export未逐项单独计时，不虚构其分解。所有本任务进程已退出、NVML释放，释放证明SHA和ledger SHA见BASELINE_RUNTIME_PROFILE。

报告状态不等于32K/AIME功能验证，不更新旧科学seal、不训练、不租A800、不git push。此CPU收口后仍须独立复核源码/全部当前资产、生成SHA256_MANIFEST与逐SHA/bytes本地交付核验，才可宣称交付完成。
'''
    artifacts['FINAL_TECHNICAL_REPORT.md']=report.encode()
    for filename in ('BASELINE_RUNTIME_PROFILE.json','PER_KERNEL_COST.csv','FULL_MODEL_BASELINE.csv'):
        p=analysis/filename;require(p.is_file() and p.stat().st_size>0,'Required actual Phase0 output missing');artifacts[filename]=p.read_bytes()
    output.mkdir(parents=True,exist_ok=True)
    require(not any((output/name).exists() for name in artifacts),'Refuse overwriting prior closeout assets')
    for name,data in artifacts.items():
        fd,temp=tempfile.mkstemp(prefix='.closeout_',dir=output)
        with os.fdopen(fd,'wb') as handle:handle.write(data)
        os.rename(temp,output/name)
    print(json.dumps(dict(status='PROFILE_ONLY_COMPLETE_AND_STOP',scope='CPU_CLOSEOUT_ONLY_FINAL_SEAL_STILL_REQUIRED',output=str(output),
        artifacts=[dict(path=str(output/name),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()) for name,data in artifacts.items()]),ensure_ascii=False))


if __name__=='__main__':main()
