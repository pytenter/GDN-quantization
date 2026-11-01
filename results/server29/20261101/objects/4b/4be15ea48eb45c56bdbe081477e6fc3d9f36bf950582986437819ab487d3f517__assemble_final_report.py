#!/usr/bin/env python3
"""CPU-only final report assembly from complete, SHA-bound actual Phase0 results."""
import hashlib
import json
from pathlib import Path

TASK = 'QWEN_GDN_DUAL_M_EQUIVALENT_SPEEDUP_V1'

def read(p):
    return json.loads(p.read_text())

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    root = Path(__file__).resolve().parents[1]
    profile = read(root / 'phase0_analysis/BASELINE_RUNTIME_PROFILE.json')
    ls = read(root / 'phase0_analysis/BATCHED_LS_ATTRIBUTION.json')
    decision = read(root / 'ROOT_PROFILE_DECISION.json')
    assert decision['decision'] == 'CASE_C_PROFILE_ONLY_STOP' and not decision['candidate_created']
    assert sha(root / 'phase0_analysis/BASELINE_RUNTIME_PROFILE.json') == decision['BASELINE_RUNTIME_PROFILE_sha256']
    assert sha(root / 'phase0_analysis/BATCHED_LS_ATTRIBUTION.json') == decision['BATCHED_LS_ATTRIBUTION_sha256']
    assert ls['status'] == 'PASS_UNIQUE_EXISTING_TRACE_GPU_ACTIVITY_ATTRIBUTION_ONLY' and ls['matched_QDQ_regions'] == 128
    cond = {x['condition']:x for x in profile['short_noninstrumented_baselines']}
    h = next(x for x in profile['separate_instrumented_profiles'] if x['condition'] == 'H_DUAL_M_REFERENCE')
    phases = {x['name']:x for x in h['original_disjoint_CPU_phase_scopes']}
    wall = h['decode_window_wall_seconds']
    model_fraction = phases['REFERENCE/model_forward']['cpu_total_us'] / 1e6 / wall
    qdq_fraction = phases['REFERENCE/QDQ_and_original_safety']['cpu_total_us'] / 1e6 / wall
    micro = profile['codec_micro']
    lines = []
    for name in ['FP_NATIVE', 'H_C128', 'H_DUAL_M_REFERENCE']:
        row = cond[name]; t = row['decode_tokens_per_second']; w = row['decode_window_wall_seconds']
        dispatch = row['pooled_host_iteration_dispatch_seconds']
        lines.append(f"|{name}|{t['p50']:.6f}|{t['min']:.6f}–{t['max']:.6f}|{w['p50']:.6f}|{w['p50']*1000/128:.3f}|{dispatch['p50']*1000:.3f}/{dispatch['p95']*1000:.3f}|{row['peak_allocated_bytes']['p50']}|{row['peak_reserved_bytes']['p50']}|")
    baseline_table = '\n'.join(lines)
    phase_table = '\n'.join(f"|{x['name']}|{x['cpu_total_us']/1e6:.9f}|{x['cpu_total_us']/1e6/wall*100:.5f}%|{x['device_total_us']/1e6:.9f}|" for x in sorted(phases.values(), key=lambda x:x['cpu_total_us'], reverse=True))
    cpu_table = '\n'.join(f"|{x['name']}|{x['count']}|{x['self_cpu_us']/1e6:.9f}|" for x in h['top3_CPU_operators_by_self_time'])
    kernel_table = '\n'.join(f"|`{x['name']}`|{x['count']}|{x['summed_duration_us']/1e6:.9f}|" for x in h['top3_device_activity_by_summed_duration'])
    micro_table = '\n'.join(f"|{x['name'].split('/')[-1]}|{x['cpu_total_us']:.3f}|{x['device_total_us']:.3f}|" for x in micro['instrumented_original_stage_costs'])
    event_rows = []
    for i, trial in enumerate(cond['H_DUAL_M_REFERENCE']['batch24_original_codec_event_spans_by_trial']):
        v = trial['per_token_ms']
        event_rows.append(f"|{i}|{v['encode_ms']['p50']:.6f}|{v['decode_ms']['p50']:.6f}|{v['safety_and_metric_audit_ms']['p50']:.6f}|{v['writeback_ms']['p50']:.6f}|{trial['per_token_sum_event_span_ms']['p50']:.6f}|")
    event_table = '\n'.join(event_rows)
    historical = read(root / 'REFERENCE_SOURCE_AUDIT.json')['historical_profiles']
    ref_tps = cond['H_DUAL_M_REFERENCE']['decode_tokens_per_second']['p50']
    hc_tps = cond['H_C128']['decode_tokens_per_second']['p50']
    amdahl = profile['conservative_Amdahl_analysis'][0]
    report = f'''# Qwen/GDN Dual-M 等价加速技术报告

最终状态：**PROFILE_ONLY_COMPLETE_AND_STOP**。已完成授权的有界定位，按任务情况 C 停止。未开发优化候选，未得到可核验的等价端到端加速实现；优化 Gate A/B/C、候选测速及 fresh-process 资格均为 **NOT_RUN_NO_CANDIDATE**。

主要发现：本次 batch=1、BF16 模型、原 32K 确定性路径下，主要耗时落在模型 forward 及其大量 CPU/PyTorch 发射中。H+Dual-M 的非instrumented中位数为 **{ref_tps:.6f} token/s**；profile 的模型 forward CPU 范围占 {model_fraction*100:.5f}%，整个 QDQ 与原安全检查范围占 {qdq_fraction*100:.5f}%。实际全24层 LS 设备活动中位数约 **{ls['LS_device_activity_duration_sum_us']['median']/1000:.6f} ms/token**，不支持将 LS 当作整机首要瓶颈。

## 真实测量与身份

仅使用 .55 的 GPU0，UUID `GPU-a78176ed-f97f-c54f-6201-5a01b570f422`，RTX3090 24GB；单次 dispatch。原 32K `core.py/worker.py/codec.py` 分别绑定 aa338b8b83cde62b97c94fd205d1c62999f783e7ec3242cbb813dc877e01f575、e04ec85092ee4f2207699366b2aaa61e1e71038e728c9234477b5b2c21794773、ece859b901145c6d9d646b8befd01916037c9117ef4a80f7b801f27f79f8c23f。原 prospective 与 C6R 使用的这三份源逐字节一致。保留 strict deterministic=True、TF32=False、原 compensated chunk、H128、FP16行列scale、all8、w=1、一次LS、最终重编码及FP32 persistent QDQ carrier；未修改历史源与seal。

输入为原 prospective 非AIME Doc1 固定 tokens，JSON SHA18faa0ca311362181c5c3340d19d72ac39422d1ee07846e516bf34074a51703d，canonical token SHA8ccd76d9556130c4dac436ac8d99aa7fef98a5d7c767bbd8299a79c4e276dde3。每个 trial 独立cache/RNG，128-token warmup 后在同一cache上测量接下来的128token。三条件×三次普通回放，共9条；另三条独立 profiler 采集，不用其TPS冒充基线。共3072个实际模型token，没有新AIME或32K科学评分。未在每token添加全局CUDA synchronize；保留参考内部等待与安全门禁。

|配置|token/s中位数|范围|128token墙钟中位秒|完整测量窗平均ms/token|host iteration p50/p95 ms|allocated峰值中位B|reserved峰值中位B|
|---|---:|---:|---:|---:|---:|---:|---:|
{baseline_table}

host iteration是发射及原内部等待所耗的CPU循环时间，不等于逐token GPU完成延迟。完整测量窗平均ms/token包含测量窗末尾等待。512/2048/8192未运行，依冻结PLAN在情况C停止；这些短窗数据不是长上下文吞吐预测或历史AIME比较。

## 用户12问

### 1. Dual-M实际最耗时的操作是什么？

完整H+Dual-M instrumented窗为 {wall:.9f} 秒，三个原始owner范围按CPU elapsed排序如下。device attributed是归属于该范围的GPU子活动，不能和CPU范围、ATen子项相加为墙钟。

|执行范围|CPU elapsed秒|占profile完整墙钟|device attributed秒|
|---|---:|---:|---:|
{phase_table}

CPU self前三项：

|操作/范围|次数|CPU self秒|
|---|---:|---:|
{cpu_table}

实际设备kernel累计活动前三项：

|实际kernel名称|次数|活动sum秒|
|---|---:|---:|
{kernel_table}

模型范围含Q/K Hadamard与原生GDN；此次未在其内部添加新的分段observer，不能进一步伪造GDN/MLP/attention各自独占wall。CPU发射/同步是主要工程线索，不能仅凭GPU活动较短断言所有间隙都是CPU overhead，或声称设备内存带宽已被测定饱和。

### 2. Column LS究竟占多少时间？

从真实全24层 `[24,32,128,128]` trace 的全部128个QDQ范围，逐个唯一匹配原88个顶层ATen操作，以完整 runtime correlation 归属LS。每token LS设备活动中位 **{ls['LS_device_activity_duration_sum_us']['median']:.4f} us**，范围 {ls['LS_device_activity_duration_sum_us']['min']:.3f}–{ls['LS_device_activity_duration_sum_us']['max']:.3f} us，p95 {ls['LS_device_activity_duration_sum_us']['p95']:.3f} us；128次合计 {ls['LS_device_activity_duration_sum_us']['sum']:.3f} us。占encode设备活动sum **{ls['aggregate_LS_fraction_of_encode_summed_device_activity']*100:.5f}%**。对应CPU dispatch区间中位 {ls['LS_CPU_dispatch_interval_us']['median']:.3f} us，p95 {ls['LS_CPU_dispatch_interval_us']['p95']:.3f} us。此LS组包含v、分母、分子、guarded fit、FP16列scale和原raw-scale clone；不是只计一个sum kernel。

**完整LS墙钟仍UNKNOWN**：没有为LS插入独立完成事件，GPU活动sum与CPU发射区间均不能当作完整LS wall，更不能相加或当作可实现收益。没有单层×24外推。

三次普通全模型回放中，原24层codec已有事件区间的每token中位数（ms）为：

|重复|encode|decode|安全/metric审计|writeback|逐token四段合计的中位数|
|---|---:|---:|---:|---:|---:|
{event_table}

这些区间包括设备空隙，且不包括event0之前的cat和部分host准备，不能称完整独占codec wall。另真实单层 `[1,32,128,128]` fixture的codec完整encode/decode/finite墙钟中位为 {micro['encode_decode_finite_complete_wall_seconds']['p50']*1000:.6f} ms（三次，范围 {micro['encode_decode_finite_complete_wall_seconds']['min']*1000:.6f}–{micro['encode_decode_finite_complete_wall_seconds']['max']*1000:.6f}）。独立stage诊断如下，观察开销独立于三次普通micro：

|单层诊断stage|CPU elapsed us|device attributed us|
|---|---:|---:|
{micro_table}

单层micro没有导出分类Chrome trace，真实kernel数与active union保持UNKNOWN；owner device归属不能补造为其物理kernel/active指标。fixture没有保存tensor文件，仅保留真实来源、shape、SHA和byte parity证据。

### 3. 额外开销主要来自哪里？

本次Dual-M相对H+C128的完整墙钟没有观察到稳定新增惩罚：三次中位token/s分别 {ref_tps:.6f} 与 {hc_tps:.6f}，约 {100*(ref_tps/hc_tps-1):.3f}%描述性差别，重复范围相邻且N=3，不作显著性或加速候选结论。Dual-M批量cat24层后编码，增加完整FP32临时张量和设备活动，但减少原C128逐层Python/标量调用。

实际H+Dual-M每128token有 {h['actual_kernel_activity_count']} 个物理kernel活动（{h['actual_kernel_activities_per_measured_token']:.0f}/token），host `cudaLaunchKernel` 同为482176次；H+C128为525952次（4109/token），FP_NATIVE为455296次（3557/token）。H+Dual-M有2688次item/局部scalar与2688次stream synchronize、128次原event synchronize；scalar和其子调用是嵌套，不能重复相加。FP_NATIVE整条回放链（含原有限性检查）已有3557 kernels/token，不能全称纯主干活动；codec优化不能消除该链路。

H+Dual-M实际CUDA活动union为 {h['CUDA_device_active_interval_union_us']/1e6:.9f} 秒，profile wall为 {wall:.9f} 秒。低设备活动覆盖与大量CPU self/launch共同支持主干执行与发射为优先瓶颈线索；wall减active不是纯CPU时间。未测硬件带宽/SM利用率，不能精确裁定compute-bound或memory-bound。

### 4. STEPQuant官方融合设计有哪些适用？

固定commit [f2deac58131219e6aecff274f8eabe88e08d2e3c](https://github.com/Dreamer-Toby/STEPQuant/tree/f2deac58131219e6aecff274f8eabe88e08d2e3c) 的七指定文件已只读核字节。row统计、column初始编码/LS/最终重编码融合、减少中间FP32访存可以作为独立候选思路；当前数据不支持把它升级为最主要优化方向。官方packed GDN异步pool/writeback依赖不同运行时与Triton/Gluon合同，不能整体搬进来；mixedbits、pivots、SGLang、fast reciprocal/FMA或归约重排不自动满足本任务byte-exact。详见REFERENCE_IMPLEMENTATION_AUDIT及CODEC_EXECUTION_GRAPH。

### 5. 最终改动了哪些代码？

新增独立的有界测量worker/controller、源码审计、CPU归约/LS归属/报告/封存下载脚本；没有修改原Dual-M/H/GDN、reducer、位宽、scale或QDQ时序。独立工作树基于622df3543c85f72fe700046110a0da35ca214af7；服务器Git1.8不支持git-worktree，采用隔离shared-clone工作目录，未push。冻结执行manifest共27文件，148小外部依赖。优化候选文件数=0。

冻结worker的原profile_rows误将CUDA user_annotation混入kernel/active字段，原JSON永久保留RAW_DERIVED_UNCORRECTED。独立CPU脚本按真实Chrome `cat==kernel`、memcpy/memset与gpu_user_annotation分类，并核对correlation；修正统计不改GPU执行、数值或真实wall，不重跑GPU。

### 6. INT8 codes、FP16 scales、QDQ是否exact一致？

真实单层fixture上的独立stage观察与原codec codes、行scale、列scale、QDQ四项逐字节一致；本地三类stage测试也一致。这仅证明测量观察本身的输出一致，**不是优化候选资格**。候选CPU/GPU Gate A未运行。

初次原codec对官方fit_state的10类CPU参考诊断中，codes/FP16 scales exact、dequant torch.equal且maxabs0，但6类存在signed-zero字节差异；严格字节FAIL与诊断SHA保留，没有改标PASS，也不把它当作不存在的候选失败。保留此前FP_NATIVE/FP_H历史门禁失败，未放宽规则。

### 7. 真实next-token消费是否正确？

12条独立回放共3072token、73728个native GDN调用。8条量化回放共2048次QDQ、49152次writeback、48960次真实next-token initial_state指针身份核验；每条首token chunk24、之后recurrent6120次。原链路安全/finite/scale/指针门禁通过。指针身份不是逐token全值hash；不扩张成全模型logits、采样token、独立新候选值相等或fresh-process资格。优化Gate B/C均NOT_RUN。

### 8. Codec-only加速多少？

**UNKNOWN / NOT_RUN_NO_CANDIDATE**。只有原codec实测与诊断，没有优化臂，不能填1.0倍或0%制造比较。

### 9. Full-model吞吐加速多少？

**UNKNOWN / NOT_RUN_NO_CANDIDATE**。原H+Dual-M中位 {ref_tps:.6f} token/s，不存在old-vs-new速度比。SPEEDUP_COMPARISON保留真实参考值，候选/加速数字留空并写明原因。

本次instrumented若假设整个QDQ owner独占、完全可删除且不新增依赖/代价，条件理想上限为 **{amdahl['conditional_ideal_delete_entire_owner_region_ceiling']:.6f}倍**（约4.45%提升）。这甚至删除了真实数学和安全检查，不能作为可实现方案；严格非instrumented state-only加速上限 **UNKNOWN**，因为没有独占完整codec wall。不能把约4.4%的原事件跨度比换算为生产硬上限，也不从FP差值归因LS。

### 10. 是否值得后续32K科学验证？

本轮没有候选，更没有≥10%同条件端到端吞吐提升和候选数值资格证据，不建议直接追加32K/AIME。按情况C和止损规则在短profile后结束，不用剩余GPU预算盲试kernel。不是预算耗尽、显存不足或GPU故障停止。

### 11. 实际显存和临时tensor成本下降吗？

没有候选，下降量UNKNOWN。原24层persistent reconstructed FP32 carrier为50,331,648B=48MiB；瞬时INT8+FP16 row/column payload为12,976,128B=12.375MiB，不能称physical packed INT8常驻推理。

代码上cat bank和dequant bank各48MiB，另有scaled、levels、v和安全审计FP64临时量及3份rawscale clone（合计1,179,648B）；不把非同时存活的量机械相加成allocator峰值。实际普通回放H+Dual-M allocated峰值中位18,524,252,160B、reserved18,811,453,440B；H+C128 allocated中位18,011,731,456B、reserved18,018,729,984B。没有优化后的实测内存下降。

### 12. 是否仍有独立下一阶段瓶颈？

优先研究原确定性主干/GDN的大量CPU发射与小kernel执行，以及H变换的调用组织，可能比LS融合更有价值；这只是方向，不是已实现优化。任何CUDA graph、减少observer或kernel替换均须独立授权、保留安全/QDQ/确定性并重新做old-vs-new codes/scales/state/logits/next-consumption/fresh资格。本次不改权重精度、不重写GDN、不训练、不租A800、不启动新GPU研究。

## 成本、历史与交付

新增实际占用 **0.418113451952 GPUh**（1505.208427秒），只有一张GPU；包括加载、warmup、回放、profiler采集、CPU Kineto解析/Chrome导出时GPU仍驻留的全部时间，直到进程退出与NVML释放，低于2GPUh上限。host解析/导出的单独成本未逐项计时，保持UNKNOWN，不从仅26–30秒的采集窗假装总成本。独立最终129项、归约52项、runtime合同123项核验均PASS。资源已释放，无新GPU失败。

历史只作参考：原prospective D_OFF记录forward/QDQ/full中位为{historical['prospective_gpu0_D_OFF']['median_forward_s']:.9f}/{historical['prospective_gpu0_D_OFF']['median_QDQ_s']:.9f}/{historical['prospective_gpu0_D_OFF']['median_full_s']:.9f}秒，约{historical['prospective_gpu0_D_OFF']['median_tokens_per_second']:.6f} token/s；原C6R D_OFF约{historical['C6R_gpu0_D_OFF']['median_tokens_per_second']:.6f} token/s。它们含原observer/event/同步，当前AIME native cumsum/deterministicFalse路径又不同；历史FP13.38和AIME H6.83不得用作本轮正式加速比。

根决定原文中94.621%是人工转录误差；本报告按原值27.217554568/28.763418689重新计算为 **94.6255897532%**，原决定、原profile和初始收口报告均保留，不改原始数据或GPU执行。源码/分析脚本、报告、全部当前trace SHA/bytes见SHA256_MANIFEST及hashes/FINAL_REMOTE_DELIVERY_VERIFICATION；本地逐SHA/bytes交付记录见LOCAL_DELIVERY_VERIFICATION。三份共4,016,371,720B的Chrome traces留永久/data远端，小文件才下载本地。模型、大tensor未下载，旧资产未删除。
'''
    p = root / 'FINAL_TECHNICAL_REPORT.md'
    with p.open('x') as handle:
        handle.write(report)
    correction = {'task_id': TASK, 'status': 'TRANSCRIPTION_CORRECTION_ONLY',
                  'original_root_decision_sha256': sha(root / 'ROOT_PROFILE_DECISION.json'),
                  'original_text_percent': 94.621, 'actual_profile_derived_percent': model_fraction*100,
                  'source_profile_sha256': sha(root / 'phase0_analysis/BASELINE_RUNTIME_PROFILE.json'),
                  'original_decision_and_raw_results_preserved': True, 'GPU_rerun_or_source_change': False}
    with (root / 'ROOT_DECISION_PERCENT_CORRECTION.json').open('x') as handle:
        json.dump(correction, handle, ensure_ascii=False, indent=2); handle.write('\n')
    print(json.dumps({'status':'FINAL_REPORT_ASSEMBLED_FROM_ACTUAL_DATA', 'bytes':p.stat().st_size, 'sha256':sha(p)}))

if __name__ == '__main__':
    main()
