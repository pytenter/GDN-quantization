# QWEN_GDN_HADAMARD_DUAL_AXIS_REAL_STATE_RUNTIME_GATE_V1

本轮按GPU_CANARY_AUTHORIZATION=NO执行，仅CPU前置工作及预注册已完成。G0/G1/G2 PASS，GPU资源准入、GateA和全部GateB NOT_RUN。GPU_HOURS_USED=0.0；FINAL_CLASSIFICATION=PARTIAL_STOP。停止在GPU之前，不声称runtime canary合格。

## 实际完成的资格

G0：本地及远端CPU V1共70文件SHA/bytes、seal、回执、attempt001反例/原代码和attempt002结果恢复核验；远端完整失败panel的SHA也核验，未下载大tensor。当前实际22份源码/小索引已采集SHA、5个工作树HEAD/branch/dirty状态已记录。历史资产未编辑。

G1：12项静态边界检查PASS。当前代码是经资格的V3 FP32 compensated路径，旧数值FAIL保留。Hadamard只作用Q/K Key坐标；state=[B,32,128,128]，K=V=128；updated state先产生当前readout，然后完成原norm/gate/out_proj及model forward，外部调用原C128写回，下一token消费QDQ状态。旧AIME initial-prefill边界与后续每token harness不相同，未混称端到端复现。完整审计见RUNTIME_BOUNDARY_AUDIT.md；GPU实际dispatch和新H-only baseline仍未验证。

G2：不修改V1 codec，以原封存tests/reference重新执行。Canonical9案例×4输入dtype及原正式CPU fake-cache24层exact；Dual-M官方all8/ones/pivots0的9案例code/row/column/decode exact；49项roundtrip/scale/非法输入/存储检查；两个独立进程code digest以及一个独立进程recurrent digest完全一致，且与旧V1封存结果一致。连续128步在1/8/32/128记录，每步只从上一encoded bytes恢复，丢弃瞬时未QDQ FP state。GPU fresh-process determinism依然NOT_RUN。

## 人类澄清后的文本冻结

只使用现有缓存合法RepoBench-P/P_e test来源。按事前固定seed20261015和SHA排序，从775个符合已知排除及128-token资格的候选中取前2；没有读取D的真实state/KL结果，也没有任何校准或参数拟合。排除已知历史ID/rawSHA、旧8-doc及人类提供的四个Ling ID。源文件内容SHA、行号、原文SHA与dataset ID实际核对，使用原本地Qwen2Tokenizer，无特殊token和chat template，冻结每篇前128 token。

| 文档ID | consumed tokens | canonical token hash |
|---|---:|---|
| `long_bench/repobench-p/test/27812d6ea02f9e98707f2f03c0e379417a196e4de5c96493` | 128 | `e8b4bd1b698e1f77d24f8a705c6cc8b1b7a703c1d472c4c2fbfb1a5a960e1d5b` |
| `long_bench/repobench-p/test/844f86363fc751fbf98b1d3b5e1d608e9701d51eeed872e7` | 128 | `d59307230ee4f81a922ab0d3e62092643227cc2ea29ff33ffd39b6c06db29c3c` |

两份raw文档不同、128token输入不同；repository/project独立性UNKNOWN，不能把去重当项目独立证明。Ling原manifest未提供，核验来源是本轮人类直接ID指示，没有伪造路径或manifest验证。preregistration.json保留原始pending状态，**effective协议由其与preregistration_addendum_001.json共同组成**，后者冻结真实文档身份并解除Ling路径依赖。设备ID/UUID未准入，权重新content rehash、H-only GPU adapter和数值/runtime资格仍待未来单独授权完成；GPU_execution_ready=false。

## 所请求真实模型数据的状态

| 条件 | 真实state误差 | 8/32/128 KL/readout/state指标 | GPU token/codec开销与吞吐 | 实测GPU常驻cache/显存 |
|---|---|---|---|---|
| A C128_CANONICAL | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| B C128_FP16_SCALE | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| C C128_LS_FP16 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| D DUAL_M_INT8 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |

没有真实snapshot、logits或模型forward，因此不生成real_state_distribution.csv、real_state_codec_comparison.csv或short_canary_kl.csv。无FP_H-native等价或C128 runtime parity、GPU跨进程、full-model QDQ/physical INT8资格；HISTORICAL_END_TO_END_REPRODUCTION=NOT_RUN。上轮synthetic MSE/timing不填入本轮真实表。

## scale与物理风险

CPU finite/roundtrip继续PASS，但V1 stress风险未消失：±1e10时B/C relativeMSE约0.99833689，D约3.0463e-9；FP32_MAX时A原正式语义fail-closed，B/C/D relativeMSE1。官方reference exact不代表真实state安全。本轮real scale/clipping分布NOT_RUN。预注册冻结raw scale floor/ceiling识别、nonzero affected-energy及affected relativeMSE门禁（energy≥1e-4且affected error>1e-2即STOP），阈值在任何真实state观察前确定；不能分清scale量化与clipping则NOT_IDENTIFIABLE。

旧runtime是FP32 tensor持有QDQ值，按state元素计算单层2097152bytes、24层50331648bytes；这是静态state字节，不是本轮GPU内存实测，也不含attention/conv/weights/allocator。V1 CPU int8对象表示复验通过，但未集成GPU常驻cache。PHYSICAL_INT8_RUNTIME=NOT_RUN。

## GPU账本与停止

完整账本analysis/gpu_resource_ledger.json：GPU授权NO、设备准入NOT_RUN、分配[]、模型加载0、forward0、profiling0、GPUh0.0；预算上限2.0GPUh/单3090_24GB，新增归档上限10GiB、raw模型snapshot上限1GiB。未初始化CUDA，没有需要释放的本轮GPU context；NVML峰值/资源恢复采样未运行，不伪造“已测空闲”。CPU Torch编译CUDA版本不等于占用GPU。

## 失败、来源和科学边界

本轮无新的first-failure、无重试/修复；既有失败历史详见reports/failure_history.md。新源码、原始来源、文档及交付SHA见source_manifest.json、source_manifest_addendum_001.json和hashes/artifact_sha256.txt。两端封存回执见hashes/DELIVERY_VERIFICATION.json。

当前不能据此建议进入独立2K探索：真实Hadamard FP状态、FP_H-native等价、A的runtime parity和短canary数值/资源门禁尚未完成。未创建NEXT_2K_EXPLORATORY_PREREG_DRAFT.md。未来若收到明确YES，只能先完成单卡资源准入与≤128-token GateA/B；仍不授权2K/32K/AIME/C8，也不自动启动。局部MSE、短程KL、长期功能收益和与C6R互补性均不能相互替代。本轮CPU交付封存后STOP。
