# Dual-M 等价加速：Profile-only 技术报告

最终状态：**PROFILE_ONLY_COMPLETE_AND_STOP**。这是根Agent在完整、真实Phase0报告后作出的明确决定，不是汇总器自动选择的方向。没有实现优化候选，没有宣称正向加速，也没有优化候选失败。

根决定及实际依据：Case C：真实128-token H_DUAL_M profile中，模型forward CPU owner范围27.217554568秒，占完整28.763418689秒的94.621%；QDQ与原安全检查owner为1.226533067秒（4.264%）。真实24层batched LS设备活动中位599.5825微秒/token，仅占encode设备活动17.5767%，不能当作完整wall；完整模型存在3767个真实kernel活动/token，主要CPU self耗时在aten::mm、model_forward与cudaLaunchKernel。三次非instrumented H_DualM中位6.801314154 token/s，与H_C128的6.662540440接近且未观察到LS主导的额外端到端耗时。故不实施codec融合候选，不更改模型/GDN数学，按授权止损；512/2048扩展与候选A/B/C资格均NOT_RUN。删除整个instrumented QDQ owner仅有条件估计1.044541倍，非instrumented严格加速上限仍UNKNOWN，不能承诺生产加速。

Phase0实际完成9次非instrumented128-token基线（三条件×三重复）和另3次instrumented128-token profile；每次都继承128-token warmup cache，不重跑测量前缀。真实单层GDN state codec micro有3次有效重复及一次独立stage诊断，其instrumentation输出codes、FP16行列scale和QDQ均与原codec逐字节一致。没有执行候选、512/2048/8192扩展、32K科学实验或AIME生成评分。

冻结worker原派生字段把CUDA user_annotation混入kernel/active统计，原JSON保留为RAW_DERIVED_UNCORRECTED。本次独立CPU分析逐SHA验证Chrome trace，按cat==kernel、gpu_memcpy/memset和gpu_user_annotation分开并核对correlation；CPU phase只取cpu_total_us>0的行。修正kernel/active不改变真实wall或冻结数值执行。单层micro未导出Chrome分类trace，其真实kernel数/active union保持UNKNOWN，LS只用CPU owner对应的设备归属量。

|条件|实际重复数|decode token/s中位|最小–最大|128-token decode wall中位秒|
|---|---:|---:|---:|---:|
|FP_NATIVE|3|7.932661|7.784859–7.964258|16.135822|
|H_C128|3|6.662540|6.648197–6.797635|19.211891|
|H_DUAL_M_REFERENCE|3|6.801314|6.785842–6.908012|18.819892|

上述为同固定token非AIME回放、同卡、同测量窗；不是历史AIME自由生成速度比。每token host iteration p50/p95包含参考内部检查/等待，但不等于GPU完整token latency；CUDA stream event间隔包括idle gaps，实际device active union另报。instrumented profile有额外开销，不用其TPS冒充基线。

## 用户12问

1. **实际最耗时操作？** 根决定依据如上。完整profile保存真实kernel名称/count及三个原始phase范围，前三设备活动与CPU self操作分别列在BASELINE_RUNTIME_PROFILE。设备活动sum、CPU owner归属与wall不同，不能混合求和。
2. **Column LS占多少？** 单层真实state的LS grouped device归属 55.008 us，占八个命名stage设备归属合计 18.814%。这是一次诊断的局部活动比例，包含独立trace观察合同，不是端到端wall比例；若归属不可用则保持未知。
3. **额外开销来自哪里？** 以根决定和profile的CPU launch/scalar/sync、kernel活动、model/codec/finite原始范围为证据。原DualM包含一次LS、批量cat与多轮中间state、finite/scale安全标量和原event等待。不能只凭源码宣布LS主导，也不把wall−deviceunion称为纯CPU发射时间。
4. **官方融合哪些适用？** Row/column融合边界、减少中间tensor与coalesced访存可参考。官方packed GDN、异步pool、SGLang或mixedprecision/pivot runtime不是本任务可直接替换路径；快速倒数/FMA/归约重排不自动bitwise exact。七指定官方源固定commit f2deac58131219e6aecff274f8eabe88e08d2e3c，缓存manifest及commit raw字节已核。
5. **最终改哪些代码？** 新增有界测量controller/worker、只读审计和CPU分析收口脚本；原DualM/H/GDN冻结源未修改。没有新增优化codec/kernel，SOURCE_DIFF_MANIFEST列明实际冻结执行源与候选空集合。
6. **codes/scales/QDQ exact吗？** 原codec对自身stage instrumentation在真实fixture上四项byte-exact，这只证明诊断观察没有改数值。旧vs新候选CPU/GPU parity均NOT_RUN，因为没有候选。初次旧参考vs官方fit_state严格字节测试FAIL永久保留，CPU诊断将部分差异定位signed zero；不能改成PASS，也不是候选FAIL，详见CPU_CODEC_PARITY内原记录SHA。
7. **下一token消费正确？** 原参考短回放每token使用原Observe检查实际initial_state.data_ptr与上一token QDQ carrier，实际次数随12记录保存。优化臂不存在，old-vs-new状态/输出/采样序列及fresh重复资格均NOT_RUN。FP_NATIVE/H历史门禁失败未修补或放宽。
8. **Codec-only加速多少？** 未知/NOT_RUN_NO_CANDIDATE。只报告原单层codec实际3重复和stage诊断，没有优化测速，不能给1.0×或0%冒充结果。
9. **全模型加速多少？** 未知/NOT_RUN_NO_CANDIDATE。原H_DualM三次128-token测量中位 6.801314 token/s；没有优化臂。SPEEDUP_COMPARISON仅一行明确原值与候选缺失，speedup留空。
10. **是否值得下一步32K？** 本轮没有合格优化候选或≥10%同条件完整decode提升证据，不授权自动长程验证。根停止原因如上；不从局部CUDA活动加速推断32K功能收益。
11. **显存与临时tensor下降吗？** 没有候选对照，下降量未知。原24层reconstructed FP32 persistent carrier为50,331,648B；原编码INT8与FP16scale payload另算，不是physicalpackedINT8常驻。各实际CUDA allocated/reserved峰值已记录，不能把多个不同时存活临时tensor静态相加当峰值。
12. **还有独立下一阶段瓶颈吗？** 源审计和实际profile保留可供判断，但本次不自动追加kernel、packed runtime、训练或benchmark。任何后续候选应独立授权、保持原codec数学/QDQ，先bitwise数值与真实next-token/fresh资格，再测同窗wall加速。

本任务累计GPU占用 **0.418113451952 GPUh**，包含加载、warmup、全部回放、profiler采集、CPU key_averages/events解析、trace导出、codec诊断及进程退出/NVML释放前全部占用。`decode_window_wall`仅采集窗口，不能当作总profile成本；host解析/export未逐项单独计时，不虚构其分解。所有本任务进程已退出、NVML释放，释放证明SHA和ledger SHA见BASELINE_RUNTIME_PROFILE。

报告状态不等于32K/AIME功能验证，不更新旧科学seal、不训练、不租A800、不git push。此CPU收口后仍须独立复核源码/全部当前资产、生成SHA256_MANIFEST与逐SHA/bytes本地交付核验，才可宣称交付完成。
