#!/usr/bin/env python3
"""CPU-only report from sealed records; no new performance workload."""
from pathlib import Path
import json,math,hashlib
B=Path(__file__).resolve().parents[1]
def rd(n):return json.loads((B/n).read_bytes())
j=rd('MATCHED_RUNTIME_BENCHMARK.json');h=rd('HISTORICAL_TPS_RECOMPUTATION.json');p=rd('PERFORMANCE_PROFILE_COMPARISON.json');g=j['groups']
assert j['batch_status']=='PASS' and all(g[k]['N']==3 for k in g)
def fmt(x,n=6):return f'{x:.{n}f}'
def med(c):return g[c]['TPS']['median']
def ms(c):return g[c]['wall_ms_per_token']['median']
table='| 条件 | 三次TPS中位数 | TPS最小–最大 | 中位wall ms/token |\n|---|---:|---:|---:|\n'
for c in g:table+=f"| {c} | {med(c):.6f} | {g[c]['TPS']['min']:.6f}–{g[c]['TPS']['max']:.6f} | {ms(c):.3f} |\n"
ht='| 历史配置 | N | 总token/总耗时 TPS | TPS算术均值 | TPS中位数 | EOS / 达上限 |\n|---|---:|---:|---:|---:|---:|\n'
for c,x in h['methods'].items():ht+=f"| {c} | {x['N']} | {x['weighted_TPS']:.9f} | {x['mean_TPS']:.9f} | {x['median_TPS']:.9f} | {x['EOS_count']} / {x['cap_count']} |\n"
pt='| 独立Profiler，128 tokens | OLD_FP | NEW_FP |\n|---|---:|---:|\n';pp={x['condition']:x for x in p['profiles']};o=pp['OLD_FP'];n=pp['NEW_FP']
for label,key in [('窗口wall秒','profiled_wall_seconds'),('真实Kernel数/token','kernel_launches_per_token'),('GPU Kernel activity sum ms/token','GPU_kernel_sum_ms_per_token'),('GPU active union ms/token','GPU_active_union_ms_per_token')]:pt+=f'| {label} | {o[key]:.6f} | {n[key]:.6f} |\n'
for name in ['aten::mm','aten::isfinite','aten::all','aten::item','cudaLaunchKernel','cudaStreamSynchronize','cudaEventSynchronize']:
 for metric,label in [('count_per_token','次数/token'),('self_CPU_ms_per_token','CPU self ms/token')]:
  def v(x):
   z=x['selected_CPU_ops'][name];return f'{z[metric]:.6f}' if metric in z else 'NOT_OBSERVED'
  pt+=f'| {name} {label} | {v(o)} | {v(n)} |\n'
# Artifact is an actual evidence-derived technical report, not an optimization proposal.
s=f'''历史FP约13.38 token/s，而当前FP约7.93 token/s，主要差异来自已经复现的新旧runtime执行路径差异，最显著的Profiler证据集中在矩阵乘法主机侧调用范围，但底层主因尚无法归因；证据是同一3090、同输入、同窗口下旧FP为{med('OLD_FP'):.2f}、新FP为{med('NEW_FP'):.2f} token/s，关闭确定性开关也只达到{med('DIAG_NEW_DETERMINISM_OFF'):.2f}；仍未确认的是新增安全检查、Python包装、后端设置及其交互分别贡献多少。

TASK_ID：QWEN_GDN_HISTORICAL_THROUGHPUT_RECONCILIATION_V1。终态：**RUNTIME_DIFFERENCE_CONFIRMED_AND_STOP**。执行路径差异已确认，独占的主要微观原因仍为UNRESOLVED；没有把未完成的因果分解标成PASS。本轮一次集中批次，GPU0一张RTX3090，全部12主试验、6单项诊断及2独立Profiler完成并真实释放，累计{j['cumulative_GPU_hours']:.9f} GPUh，小于0.75。此费用包括模型加载、warmup、Profiler及其CPU后处理到进程退出/NVML释放。

## Q1. 历史13.38和9.12能否重算？

能。8个原始JSONL分片共120条，逐文件raw SHA/bytes等于原PROVENANCE，FP和H各60条且problem×seed唯一。每条保存TPS逐浮点等于N/T；token ID列表长度也等于N，无runtime error或nonfinite。没有重评分、导出答案或重写历史科学结果。

{ht}
FP总token1854451、总耗时138554.8332631588秒；H总token1818351、总耗时199485.0360198021秒。因此按两位小数确实分别13.38、9.12。来源为原执行commit `f552b9c527246c6dfe23205c0f1dc8ed1efd6546`，runner raw SHA `9ff56d00b6b2dbaf9be3c3186bd30be8d3b9cc9d9b580a680316bcc8c01247dd`；git show、原路径及迁移副本逐字节一致。当前公共仓库只是参考，未替代真实历史执行证据。

## Q2. 统计口径是否相同、能否直接比？

历史13.38/9.12是Σ生成token/Σ逐条耗时，是4个独立单卡worker的时间加权单请求速率，**不是四卡并发总吞吐**。历史计时含patch设置、45–373 token批量prefill、N−1个单token cached forward、N次采样、QDQ与安全检查；排除加载、分词/输入传卡、最后文本解码、评分和JSON写盘。历史未在计时末尾显式全局同步，但每token有logits有限性和采样scalar读取；没有证据把差距解释成异步GPU漏计。

近期7.932661/6.662540/6.801314为3次普通128-token窗的中位速率：1-token prefill、总128次warmup forward后保留cache，接着计时位置128–255，首尾同步，排除sampler/加载/warmup。名字中的“32K”是科学参考路径，**该速度并非在32K cache位置测量**。两者不能充当同协议算法加速比；但历史均值/中位数仍约13.4–13.5，统计聚合差异不足以解释40.7%的表面下降。本轮统一窗仍复现差距，所以也不能只归因于采样、长输出、prefill长度或四卡聚合。

原记录的输出长度FP为9572–81920、H为9125–64764；没有≤8192样本，长度CSV中N=0的组明确没有历史证据。FP/H的输出长度与TPS描述性Pearson为0.03675/−0.27398；prefill长度为−0.00654/−0.14027，均不是因果证明。BY_GPU按原物理index0–3各15条，历史UUID未保存，未反推为今天的UUID。

## Q3. 实际代码、Kernel及数值设置有何不同？

两者Qwen模型主源码raw SHA均为 `90d929129ffc835d2652c604925c4f3842bc6e401e174ec6f0db2285dfb8f85a`，历史2026-09-17模型源码manifest、snapshot与当前文件一致；Transformers commit为 `a353632607c59463e6ced86a44c2de3c2cd62d5e`。Py3.10.18、Torch2.5.1+cu121、CUDA build12.1、Transformers5.16.0.dev0匹配。BF16模型权重，24层GDN state为FP32[1,32,128,128]，FP不表示整模型FP32权重。

旧runner/shell没有显式指定deterministic、TF32、OMP/MKL或CPU线程，历史实际值UNKNOWN；今天独立旧路径实际记录det=False、24/24线程及loader选择的attention后端。新路径明确det=True、TF32=False、1/1线程、CUBLAS_WORKSPACE_CONFIG=:4096:8、compensated chunk。新路径原Observe(on=False)依旧包含24个state有限性归约、layer计数/shape/dtype检查和H时的next-consumption指针检查；没有CPU tensor哈希。旧FP主要为全logits有限性检查（历史AIME还含sampler安全检查）。这些差异源码可核，不代表其独占耗时份额已确定。

六个H相关函数AST相同，但旧路径每次重建H128，新路径缓存H矩阵；旧H不在首次prefill后QDQ，新H在每次forward后QDQ。故Secondary为真实路径性能比较，**没有声称H数值等价**。Column LS和Dual-M在四组主试验中均未执行，不能作为本次FP/H+C128差距的原因。

同一模型源码不等于已证明物理Kernel完全一致：GDN helper带hub/fallback装饰器，inspect.unwrap仅确认原Python函数，历史optional package完整清单缺失。已另存IDENTITY_INTERPRETATION_ADDENDUM。实际Kernel名称/计数以本轮Chrome精确cat==kernel为证据，gpu_user_annotation不冒充Kernel。没有torch.compile/CUDA Graph或SGLang调用证据，未开发任何优化。

## Q4. 主要已验证解释是什么？

已验证的是**runtime路径差异具有真实性能影响**，不是单纯汇总口径错误。相同今天权重、同一GPU UUID、同固定非AIME输入、同计时和cache条件，三次FP范围不重叠，旧/新中位数比为{med('OLD_FP')/med('NEW_FP'):.6f}。新FP比旧低{100*(1-med('NEW_FP')/med('OLD_FP')):.3f}%，中位wall差{ms('NEW_FP')-ms('OLD_FP'):.3f} ms/token。

**最显著的已定位操作范围是ATen矩阵乘法的主机侧调用，不是GPU矩阵计算耗时的大幅增加。** 两条独立Profiler中aten::mm都是249次/token，CPU self由7.076增至55.828 ms/token；对应operator关联device耗时20.928/20.939 ms/token接近，真实全GPU active union为26.803/28.092 ms/token。mm的host self差48.751 ms/token是Profiler中的直接观测，不表示普通推理55.804 ms差额已被独占解释，也不能排除未被展开的库调用或等待。

两trace旧57种Kernel名称全部存在于新64种中；新增847次Kernel/token的计数差全部来自fill和有限性/布尔归约等操作类型，没有观察到新增矩阵乘法次数。isfinite由1变25次/token，all由1变26；aten::fill_由24变823。scalar item与cudaStreamSynchronize均为1次/token，故没有证据称新路径增加24次全局标量同步。完整真实名称和差分已封存KERNEL_NAME_DELTA_AUDIT。名称/次数相同仍不能证明数学输出逐字节等价。

**尚未证明哪一个底层配置是主要触发原因。** 单项det开关和CPU intraop线程诊断没有恢复历史速度；不能把剩余差额全记到CPU overhead、安全检查或某个GDN Kernel。CUBLAS环境、interop线程、cudnn配置、state检查/包装及其交互没有分别消融，本轮唯一GPU批次完成后不追加试验。

## Q5. deterministic/compensated是否导致Decode降速？

仅把新路径deterministic切为False（其余包括CUBLAS环境、compensated chunk、1/1线程和检查保留），三次中位数{med('DIAG_NEW_DETERMINISM_OFF'):.6f}，比新原路径{100*(med('DIAG_NEW_DETERMINISM_OFF')/med('NEW_FP')-1):.3f}%；这是独立诊断的观测速率差，不是严格数值等价或独占的因果百分比，也没有恢复到旧FP。仅改intraop为24而interop仍1，结果{med('DIAG_NEW_INTRAOP24'):.6f}，同样未恢复。不能由这两项断言所有确定性后端约束完全没有成本，也不能用它们替换原科学基线。

compensated只改chunk中g.cumsum的调用点。每次256次真实全模型forward恰有首token24次chunk，全部发生在计时前；随后255×24次recurrent，计时窗128×24=3072次。recurrent源码没有该cumsum。故“每token反复执行compensated chunk导致稳定Decode变慢”被调用证据排除；初始state、初始化和后端设置的间接影响未单独拆分。

独立Profiler的实际指标如下，**不用于普通TPS中位数或加速比**：

{pt}
完整CPU self/top Kernel及实际CUDA runtime API见PERFORMANCE_PROFILE_COMPARISON.json和MATCHED_RUNTIME_BENCHMARK.json。CPU elapsed含嵌套范围，CPU self、GPU activity sum、GPU active union、wall是不同量；不得相加，也不能把wall−union全当可消除CPU开销。实际Profiler对旧路径本身也有显著扰动，不能照搬其耗时比例到普通推理。

## Q6. 同条件分别多少TPS，是否真实全Forward？

{table}
前四行为Primary/Secondary，后两行只是预声明的独立单设置诊断，每组3次，无Profiler。RTX3090 GPU0 UUID `GPU-a78176ed-f97f-c54f-6201-5a01b570f422`。固定Doc1全token canonical SHA `8ccd76d9556130c4dac436ac8d99aa7fef98a5d7c767bbd8299a79c4e276dde3`，fixture raw SHA `18faa0ca311362181c5c3340d19d72ac39422d1ee07846e516bf34074a51703d`。

各trial独立模型进程和cache，共256全模型forward、8192 decoder-layer调用、24 chunk+6120 recurrent。实际原生RoPE position逐token为0..255，全部8个attention KV逐token从1增长到256；warmup结束cache128被保留，最终256；记录每一步shape/dtype。没有重置cache冒充继续，没有采样或部分模型空转。通用额外计数/shape元数据hook不增加tensor数学或逐token全局同步；RoPE小tensor仅计时后读回CPU。旧路径使用原始函数但统一为非padding单token固定replay，省略历史显式ones attention mask，不声称精确重建长AIME生成。

普通18次测量窗的SM时钟中位数均为1695 MHz，未观察到新路径换卡或降SM时钟；FP旧GPU利用率中位39–39.5%、新22–24%，两者worker CPU约单核100%。这些是今天窗口的采样描述，不是独占因果份额，也不补全历史clocks。

CSV逐trial包含wall、p50/p95 host迭代、allocated/reserved峰值、UUID、测量窗SM时钟/功率/utilization、CPU利用率、调用和cache计数、source SHA；JSON含逐token位置/KV、完整environment及2秒采样的覆盖数、范围。没有用GPU编号代替UUID，也没有把没有采到的值填成历史遥测。

## Q7. 能解释多少，多少UNKNOWN？

历史数值来源和聚合、当前真实短cache窗、同卡重复差异、chunk调用范围、额外安全检查代码及单设置诊断均已澄清。同卡FP观察到{ms('NEW_FP')-ms('OLD_FP'):.3f} ms/token总差，但这不是一个已经完成独占分解的时间预算。现有证据不足以把该差分成确定性X%、CPU Y%、检查Z%；三个重复和短窗也不构成长上下文/跨机器统计。

历史实际UUID、clocks/power/CPU负载、optional kernels完整清单、全权重与tokenizer逐文件SHA没有捕获，保持UNKNOWN。今天四个共享checkpoint分片CPU原始SHA已经核验，postrun stat guard另核；它证明本轮共享身份，不能倒推历史权重未变。今天同卡复核排除了当前组间换卡/不同输入/cache长度/计时范围作为本轮差距的解释，但不能补造历史环境。

## Q8. 后续应如何做？

保留现有科学runtime和原数值资格，修正历史与短窗性能对照口径。若之后另行授权优化，优先定位aten::mm主机侧调用中的库开销/等待及后端配置，再拆分严格runtime的检查/包装与发射路径，保持next-consumption、finite、dtype等必要语义并重新做数值资格；当前证据不足以预定CUDA Graph、修改GDN数学或关闭确定性。只靠改线程或关闭det开关无法解决本轮差距。没有证据宣称必须更换GDN Kernel，也没有证据宣称所有性能问题已经解决。

本任务已停止新增GPU工作，不自动启动优化、新32K、AIME评测或训练。所有历史source/STOP/seal不变；本轮2个大型trace保留在55永久/data，不下载Mac、不删除原资产。详细SHA/bytes与当前进程/NVML释放由REMOTE_DELIVERY_VERIFICATION及本地下载回执核验。

## 交付与边界

冻结GPU执行SOURCE_SHA256_MANIFEST保持 `b59ee3a7417ebdae7f4f857bc3e8eb2f1644c373d6d28bf8d31da2871671c9f3`，包含7个执行资产与194个外部依赖。额外CPU分析/交付源独立封存。14项必需交付齐全；FIRST_FAILURE表示无真实GPU失败，同时永久保留启动前shell日志目录缺失事件及冻结前draft修订历史，未将它们冒称GPU重试。
'''
(B/'FINAL_TECHNICAL_REPORT.md').write_text(s)
print(json.dumps({'report_bytes':len(s.encode()),'FP_median_ratio':med('OLD_FP')/med('NEW_FP'),'FP_gap_ms_token':ms('NEW_FP')-ms('OLD_FP')}))
