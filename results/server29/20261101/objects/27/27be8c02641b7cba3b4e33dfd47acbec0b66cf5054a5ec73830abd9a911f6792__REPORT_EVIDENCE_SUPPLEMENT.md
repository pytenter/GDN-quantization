# Phase 0 报告证据补充：只读 CPU 复核

本补充仅从已完成且已释放 GPU 的实际记录、Chrome trace 分类重算及冻结源码审计中整理证据，不修改任何历史科学程序、原始性能记录、原始失败或决策文件。根协调者已依据完整 Phase 0 选择 Case C；最终状态为 `PROFILE_ONLY_COMPLETE_AND_STOP`。没有实现优化候选，也没有新增 GPU 执行。

## 1. 测量口径和原始证据

Qwen3.5-9B、BF16 权重、batch=1，固定非 AIME teacher-forced 输入；GPU0 `GPU-a78176ed-f97f-c54f-6201-5a01b570f422`，RTX3090 24GB，冻结 H128、原 compensated GDN、deterministic=True、TF32=False。每条回放独立从空 cache 开始，128 token warmup 后保留状态，再计量后续 128 token；三条件各三次非 profiler 计时，按预先平衡顺序执行。窗口 wall 以 CPU 单调时钟记录，并在窗口边界等候 CUDA 完成；加载和 warmup 不计入 decode 吞吐，计入任务 GPU 费用。

128-token 回放不是 32K 或 AIME 的速度、精度结论；512、2048、可选 8192 token 均未执行。每 token host iteration p50/p95 包含原检查/等待，不是逐 token GPU 完成延迟。原 Dual-M 内部事件等待保留；没有为了测量新增每 token 全局同步。

| 条件 | 有效重复 | 128-token wall 中位数 / 范围（s） | token/s 中位数 / 范围 | 总 token / 总 wall 加权 token/s | peak allocated 中位数（B） | peak reserved 中位数（B） |
|---|---:|---:|---:|---:|---:|---:|
| FP_NATIVE | 3 | 16.135821864 / 16.071804255–16.442174032 | 7.932660702 / 7.784858605–7.964258273 | 7.893146504 | 18,011,698,688 | 18,018,729,984 |
| H_C128 | 3 | 19.211890891 / 18.830078706–19.253340170 | 6.662540440 / 6.648197085–6.797634890 | 6.702119276 | 18,011,731,456 | 18,018,729,984 |
| H_DUAL_M_REFERENCE | 3 | 18.819892317 / 18.529210299–18.862802267 | 6.801314154 / 6.785842219–6.908011617 | 6.831293136 | 18,524,252,160 | 18,811,453,440 |

H_DUAL_M_REFERENCE 的中位吞吐稍高于 H_C128，只是本次三个短重复的描述；不能从三个回放推断所有长度的开销顺序，也不能把它称为优化加速。它与 FP_NATIVE 的差异同时包含 Hadamard、原生 forward 调用和 QDQ/安全合同，不能全归因于 Column LS。

## 2. 最耗时的三个操作以及 CPU / GPU 区分

独立 H_DUAL_M profiler 窗口为 28.763418689 s，窗口吞吐不能替代非 profiler 的 6.801314154 token/s。CPU owner `REFERENCE/model_forward` 为 27.217554568 s，占窗口 **94.62558975320%**；`REFERENCE/QDQ_and_original_safety` 为 1.226533067 s，占 **4.26421170670%**；有限性门禁 0.290216460 s。model_forward 范围包括原生 GDN、Hadamard、模型主干及其 CPU 发射/框架开销，不能将 94.6256% 称为 GPU 计算百分比。

原 `ROOT_PROFILE_DECISION.json` reason 文本把 model owner 比例写成 94.621%；保留该原始决策不改。依据原始 27.217554568 / 28.763418689 重新计算的正确值为 **94.62558975320465%**；这是文字算术更正，不改变选择。

| CPU self 时间前 3 项 | 次数 | self CPU（s） | 含义 |
|---|---:|---:|---|
| `aten::mm` | 38,016 | 8.229430342 | ATen 矩阵乘操作的主机 self 开销；不是同量 GPU 计算时间 |
| `REFERENCE/model_forward` | 128 | 5.627263586 | forward owner 的 self/framework 范围；不把它与子算子 total 相加 |
| `cudaLaunchKernel` | 482,176 | 4.157395605 | 主机 CUDA launch API 开销；不是 GPU kernel 时长 |

| 真实 GPU kernel 设备活动累计时长前 3 项 | 次数 | 设备活动累计（s） |
|---|---:|---:|
| `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x5_tn` | 8,192 | 1.065580728 |
| `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn` | 11,264 | 0.913404068 |
| `void cutlass::Kernel<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8>(cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8::Params)` | 4,096 | 0.332020686 |

上述 GPU 前三项均为 BF16 GEMM。真实 `cat==kernel` 为 **482,176 次 / 128 token = 3767 次/token**；另外 memcpy/memset 24,960 次。真实 kernel/copy/set interval union 为 **4.238565554 s**，kernel-only union 4.184156288 s；这些是采集窗口内被 trace 观察到的活动，不能据 wall 减 union 精确分解全部 CPU 成本，也不是无 profiler 的设备利用率。CPU self、owner total、嵌套 ATen、GPU 累计和 wall 属于不同口径，禁止相加冒充完整时长。

原冻结 worker 的 DeviceType 判据把 GPU user annotation scope 当成 kernel，污染了原始 kernel 数与 active union。本次仅在独立 CPU 分析中按 Chrome 精确 category 修正；384 条 GPU scope annotation 被排除，482176 个真实 kernel 的 correlation 均与 cuda_runtime 匹配。原始 JSON 仍作为 `RAW_DERIVED_UNCORRECTED` 保留，不能将其约 28.68 s union 当作 GPU active。

真实同步证据：`aten::item` / `_local_scalar_dense` 各 2688 次（21 次/token，两个名字为同一嵌套路径，不相加），`cudaStreamSynchronize` 2688 次，self 0.456402965 s；`cudaEventSynchronize` 128 次；全窗口 `cudaDeviceSynchronize` 2 次。观察到的是大量细粒度 kernel 发射及原安全合同的主机标量/同步开销，主要 owner 仍在模型 forward。未采集硬件 DRAM 带宽或完整 stall 计数，因此“显存带宽受限”的定量比例为 UNKNOWN。

## 3. Column LS 的真实证据边界

**真实整模型 batch24 归因**不是把单层微基准乘以 24。对 H_DUAL_M 原始 profiler 中 128 个 QDQ 区域，以冻结原始 ATen 结构严格匹配并关联真实 kernel/copy/set 活动；全部 128 区域各有唯一匹配，实际 state batch 为 `[24,32,128,128]`。LS 组包括 v 构造、分母、分子、guarded fit、FP16 fitted scale 与原 trace clone。

LS 设备活动累计 **76758.137 μs / 128 token**，每 token 中位 **599.5825 μs**，范围 597.470–602.398 μs，p95 601.343 μs；占 encode 累计设备活动 **17.576735%**。这是 encode 子活动比例，不是完整模型 wall 百分比。

LS 主机 dispatch interval 中位 966.135864 μs，范围 957.725098–2236.447021 μs，p95 974.349121 μs。它不是已完成设备执行的 LS wall。**LS complete wall = UNKNOWN**，不能把 17.5767% 称为 LS 占端到端 17.5767%，也不能据此承诺同等加速。

**真实单层微基准**使用 warmup 后一个 GDN 层的 `[1,32,128,128]` state；三次 encode+decode+finite 完整 wall 中位 5.347386 ms，范围 5.246162–5.481690 ms；encode host 中位 4.706740 ms。独立分段诊断的 LS device attribution 为 55.008 μs，占其八个已命名阶段设备 attribution 的 18.813808%；有 tracing 开销，且微基准没有导出 category trace，其真实 kernel 数与 active union 仍 UNKNOWN。该数不得外推成 24 层 1.32 ms 或整模型 wall。

## 4. 原整模型 codec event spans 与加速上限

原 source 已在同一 CUDA stream 记录整 batch24 encode、decode、原安全/指标审计和 writeback 边界。三次无 profiler 的每 token 平均 event span 如下（ms）：

| 重复 | encode | decode | safety/metric audit | writeback | 四段合计 | 合计 / 完整 decode wall |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 4.254987240 | 0.458309001 | 1.349034248 | 0.357217750 | 6.419548239 | 4.356204% |
| 1 | 4.230319746 | 0.457507251 | 1.326955250 | 0.350084749 | 6.364866995 | 4.396858% |
| 2 | 4.254937492 | 0.460064249 | 1.352472250 | 0.358006251 | 6.425480243 | 4.370171% |

这些 event span 包含相邻边界间的设备空闲/发射间隙，cat 在 event0 之前，其余部分 host setup 也不在四段内。它们不是经过证明可从总 wall 中删除的独占成本；不能使用 event 合计/完整 wall 直接给出严格 Amdahl 上限。

在**同一个有 profiler 的窗口**内，若额外假设整个 4.2642117067% QDQ owner 区域都独占、完全可删、没有新增依赖/成本，则条件理想比值为 `1/(1−0.0426421170670) = 1.0445414591839×`。该模型连原安全审计都理想删除，且含 profiler 开销，只用于说明“只动 LS 的收益空间不能从局部微基准推成大幅端到端加速”。它不是当前非 profiler 执行的严格上限，也不是可以实现的 4.45% 性能承诺。**非 profiler state-only 严格可移除 wall fraction 与严格加速上限均为 UNKNOWN**。

## 5. 内存与源码适用性

三个条件的实际 resident recurrent-state carrier 都为 **50,331,648 B = 48 MiB FP32**；INT8 QDQ 后仍回写该 carrier，不是物理 packed INT8 驱动原 GDN。Dual-M 原 encode 表示（24 层）为 12,976,128 B，单层为 540,672 B，只是编码 tensor，不能替代 runtime resident bytes。

H_DUAL_M 的 peak allocated 中位 18,524,252,160 B（17.252054 GiB），reserved 18,811,453,440 B（17.519531 GiB），高于该窗口的 FP/H_C128 基线；没有候选，因此没有证明内存下降。原代码会 cat 24 层形成额外 48 MiB FP32 全状态，scaled、initial/final code、中间 v、decode reconstruction 等也可能为完整状态级张量；raw scale tracing clone 约 1,179,648 B，条件 safety 分支还可能临时物化 FP64 完整状态。生命周期不同且 allocator 重用，禁止把所有静态张量尺寸相加冒称实测峰值，也不把峰值差额完全归因于某一个张量。

只读参考的 STEPQuant 固定 commit 为 `f2deac58131219e6aecff274f8eabe88e08d2e3c`：row statistics + column kernel 的融合边界、initial scale / first levels / LS / final re-encode 同 tile 处理、合并读写、避免中间全状态物化值得借鉴。直接移植仍不合格：其 fast divide、FP fusion、归约和关联次序不能保证旧 Dual-M byte exact；packed pool、异步 writeback / CUDA graph 依赖固定布局、event/stream 合同，且官方 fused GDN recurrence 不在本任务允许范围。当前固定 all8、w=1、zero pivot 不允许移入 mixed bits、FP16 pivots、SGLang 或修改 GDN 数学。

## 6. 用户 12 问的结果边界

1. **最耗时操作：** 完整模型的 forward owner 主导；CPU self 前三项为 aten::mm、model_forward owner、cudaLaunchKernel；真实 GPU 累计前 3 均 BF16 GEMM。LS 不是本证据中的端到端主瓶颈。
2. **Column LS 占时：** batch24 每 token 设备活动中位 599.5825 μs，占 encode 累计设备活动 17.576735%，完整 LS wall 未测得。单层数另列，不代替 batch24。
3. **额外成本来自哪里：** 原生模型细粒度操作与主机发射/框架范围突出，保留的标量/同步存在。profile 指向 forward 调用链与 launch，不足以给出无 profiler 的纯 CPU、纯 GPU、DRAM 带宽三者精确因果占比；不能把 Python owner 当纯 Python 时间。
4. **官方设计适用性：** row/column 统计融合、同 tile initial/LS/final encode、减少全状态中间张量为可借鉴结构；官方 packed recurrence、异步合同、fast math 不可自动移植或声称 exact。
5. **改了哪些代码：** 新增独立准入、只读性能回放/诊断、分类纠错、trace 归因与报告程序。旧 core/codec/worker 和数学未变；没有 optimized codec，没 hotpatch frozen worker。
6. **codes/scales/QDQ 是否 exact：** 优化候选不存在，old-vs-new Gate A 为 NOT_RUN；不能标 EXACT。原分段诊断包裹原语句的输出 byte parity PASS，只证明测量没有改变其输入输出。更早“原 reference 对官方纯 CPU 参考”的 signed-zero strict-byte FAIL 永久保留，不能冒充候选 FAIL 或被诊断 PASS 覆盖。
7. **next-token 消费：** 原 reference 复用了已冻结的真实 QDQ/writeback，12 条记录核验原生调用、pointer 消费与 QDQ 计数；该 PASS 范围是原 reference 的测量链。新候选 next-token、logits、固定 seed 生成和 fresh-process Gate B/C 均 NOT_RUN。
8. **Codec-only 加速：** NOT_RUN_NO_CANDIDATE，不是 0 倍、1 倍或“验证不加速”；只有原 codec 基线。
9. **完整模型加速：** NOT_RUN_NO_CANDIDATE，无同条件 old/new speedup；6.801314154 token/s 是 reference，不能当 optimized。
10. **是否值得立即 32K 验证：** 根依据本次证据选择 Case C 停止 codec 优化，没有满足数值资格且端到端 >=10% 的候选，本任务不启动 32K 或 AIME。短前缀不证明长程精度不变。
11. **显存/临时成本下降：** 未证实；只有 reference 的实际 peak 与静态生命周期审计，优化后为 NOT_RUN，48 MiB FP32 carrier 未改变。
12. **下一独立阶段：** 原生模型/compensated GDN 的细粒度 launch 与 framework 调用链、保持严格数值/安全合同的调度优化值得另行研究。当前数据不授权取消安全检查、改模型权重、改 GDN 数学、启动新 kernel sweep 或后续实验。

## 7. 完整性、成本与证据绑定

实际 GPU 费用为 **0.41811345195190774 GPUh**，覆盖 dispatch 至真进程退出与 NVML 释放；不是只计 128-token profiler 窗口。H 的 whole replay 至 profile_rows 记录为 363.465589896 s，包含 warmup 与 host 解析；Chrome export 在该字段之后，费用 ledger 仍计入，未独立测得纯解析/export 开销。worker/controller PASS，跟踪进程 25490/25493 已退出且不在 NVML。

Phase 0 是 9 条无 profiler 短回放与 3 条独立 profiler，共 12 条，未把资格、trace 解析或微基准另充作长程科学样本。实际输出是性能诊断结果，不产生准确率、Future-KL 或空科学 CSV。最终 manifest 应同时绑定本补充、root 决策、LS 归因及原 reducer，不修改已封存历史资产。

| 证据 | SHA256 |
|---|---|
| `phase0_analysis/BASELINE_RUNTIME_PROFILE.json` | `85a91783ac8aa2ed9139a7b6dfdd1a7eb71e539c05998f53f6b8e4d8497fe86c` |
| `phase0_analysis/BATCHED_LS_ATTRIBUTION.json` | `c0684858b32a7e1b562cf0f31220cc9871a678c3c46f6861ee0ddc23af5bc69d` |
| `ROOT_PROFILE_DECISION.json` | `c4768c02677b75fbdfabcf86bddc8994f144bb6a1f5e13f492d76f938afc83f6` |
| `SOURCE_MANIFEST.json` | `d0069f87aafa9ab461827000e4322c0d4758c61a04d1d296da2b495e1c9ba90d` |
| `REFERENCE_BINDING.json` | `10ce89a4453ea64de41f1e067fbb9c414263fe545b2190bd05b411a2146e20f8` |
| `source/baseline_worker.py` | `0961d96bdafa92d6c3bca7d1980b74e9d70fe3aabff404417878ced20c7f1e41` |
| `audit_reference/PROFILE_FINAL_INDEPENDENT_VERIFICATION.json` | `b681f0857f3e61a9f5822abad85e6ca69fe8ecb9bc676fac44b4aac6711485e7` |
| `audit_reference/POST_PROFILE_REFERENCE_DEPENDENCY_VERIFICATION.json` | `c226abf92b949135aa88b038fd286e3e2be6b616b520effb15fc435737eb419a` |
| `audit_official/REFERENCE_IMPLEMENTATION_AUDIT.json` | `aeafd9665de3d4f34125f9687e9f205f31faa4da1355df2437cd87475ebd45d9` |
| `audit_official/OFFICIAL_FROZEN_SOURCE_IDENTITY.json` | `b7c505897a480cfa2f036eac417b3d05ec1856d10c6779de51a69f4686718543` |
| `audit_reference/REFERENCE_CPU_CODEC_TESTS.json` | `73f684bec46d45f53230d3e0587c9c6ff42c51015e9d08e8682dc75311c8d75c` |
| `audit_reference/REFERENCE_CPU_CODEC_DIAGNOSTICS.json` | `a37b765572dddb7923ef9cc131b697ac7578e147c6587e51d372327cedbcd9b0` |
| `analysis_scripts/reduce_phase0.py` | `cb78c257e1dc33dd91ccaa8d56b0828f5005ebac89df04d748027117eb31d326` |
| `analysis_scripts/chrome_trace_categories.py` | `01dc868866742af3163e09e2097f2ec9efaee64620cfa2ecbf8f5034578c12b9` |
| `analysis_scripts/attribute_batched_ls_trace.py` | `b844d7b72b2db830d89263973855a2e1e5170a57e5d746bf469bb76a55805dbc` |
| `analysis_scripts/finalize_profile_only.py` | `017fb354cb799de3415442ff4d41d05a9890baf4932da0addb48895653f40791` |

原始 H trace 保留在远端 `/data/zypan/analyses/qwen_gdn_dual_m_equivalent_speedup_v1/profiles/H_DUAL_M_REFERENCE_L128_R0.trace.json`，1,326,443,226 B，SHA256 `945cb4d8aae403d3d8c815380f5ab4c51c69579e453fca24548aeaa0f7be733a`；本补充不下载大 trace 或任何模型/tensor。
