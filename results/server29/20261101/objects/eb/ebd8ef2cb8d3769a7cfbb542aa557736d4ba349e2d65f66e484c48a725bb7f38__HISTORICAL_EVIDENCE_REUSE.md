# 历史证据复用与本轮边界

前序根因任务`QWEN_GDN_RUNTIME_THROUGHPUT_ROOT_CAUSE_V1`终态`DIRECTIONALLY_SUPPORTED_ONLY`。本轮先重新计算其全部80个文件SHA/bytes和81条SHA256SUMS（多一条是manifest自身），全部PASS；不是仅阅读终态。原预登记完整raw SHA `404e2b2e07519434e589acfe71a2a1355b224a038d3d8d707f2eb94885ea1884`已精确复制到`evidence/previous/ROOTCAUSE_GPU_DIAGNOSTIC_PREREGISTRATION.json`，没有修改原任务。证明见[contract_prior_evidence_verification.json](/Users/pytenter/Documents/临时项目1/QWEN_GDN_FAST_RUNTIME_QUALIFICATION_V1/analysis/contract_prior_evidence_verification.json)。

## 已闭合事实直接复用

|历史观察|OLD|NEW|资格与限制|
|---|---:|---:|---|
|FP无Profiler TPS，3次中位|13.742036|7.776711|历史同卡/固定输入/128+128短缓存；本轮新测量NOT_RUN|
|H+C128无Profiler TPS，3次中位|9.064610|6.650598|有prefill QDQ/H缓存差异，不是Codec严格单变量|
|mm CPU self，ms/token|7.076412|55.827730|历史instrumented统计，非纯CPU算力或独占墙钟份额|
|mm开始至首次launch，ms/token|6.807047|55.391320|历史trace249ordinal普遍增大；具体native库/调度帧UNKNOWN|
|关联GPU GEMM，ms/token|20.928374|20.939048|同名/同数GPU活动；不能证明全部数学/参数或整机critical path等价|
|kernel/token|2710|3557|额外847已对账：451FP32fill+271BF16fill+2longfill+96finite组件+25boolall+2cat/and|

已有detFalse诊断中位7.996658 TPS，仅比NEW +2.828%；intraop24中位7.655933，−1.553%。这些干预未恢复旧速度、未证明可采用的严格等价优化。NEW的CUBLAS/OMP/interop/safety等在detOFF保持；不能将单开关观察扩大成所有确定性或线程因素已排除。

已有源码/trace共同反对“新增24个逐层主机同步”；两FP都是每token一次合并标量读取。补偿chunk在prefill、计时窗口中零chunk调用，不直接解释稳态降速。额外724fill具有确定性初始化来源机制，但历史fill动态flag未显式记录，也无单变量fill-off因果/完整全写后读证明，不批准全局关闭。

## 来源与复制策略

- 复用前序201+175源/依赖fresh SHA记录：全部376 PASS；本轮没有重复访问全部源或重新解析815MB/978MB原trace。原完整proof作为精确小文件复制，SHA与原seal绑定；这叫复用前次fresh资格，不声称本轮每一源又fresh了一遍。
- 原大trace永久留55；原trace完整SHA已由前序stream核验。本轮复用小归约、kernel delta与callpath，不下载大trace/model、不重新占GPU。
- 原历史任务本地369个资产和Speedup270个资产的SHA验证记录随前序proof保留；本轮重点验证前序root80文件的传递身份，没有无限重复所有底层文件读取。
- 原Strict目前权重4片SHAs来自调和共享checkpoint资格；未来GPU必须fresh全部4片。Tokenizer/config等现行文件另外CPU-only stream SHA供后续同文件绑定；不能追认更早历史全Tokenizer SHA。
- 当前CPU已补齐28条Torch/CUDA/cuBLAS/cuDNN物理库路径SHA（22个独立物理文件）；明确区分磁盘文件与实际loaded选择，pip/conda不同cuBLAS副本同时存在，需未来实际mapped路径资格；历史driver/全so身份仍UNKNOWN。本轮不导入torch、不读NVML、不操作进程。

## 前序预登记与本请求

原完整cuBLAS合同真实可访问，但包括OLD/NEW先裸速和2条数值单位；最新请求强调NEW内部B0/B1、数值资格先行及两边fresh重复性，因此本任务以独立addendum明确覆盖顺序/重复矩阵。原文件raw保持不变，单卡1800秒、1650保护+150清理不扩大。新的严格矩阵及其CPU审查不构成GPU授权。第二fill假设要独立预登记、预算、人类授权，不偷并。

本轮所有候选结果均`NOT_RUN_GPU_NOT_AUTHORIZED`，`NEW_GPUh=0`。历史7.776711/6.650598/6.801314是历史参考，不能放入当前新STRICT/FAST成对测量列冒充新结果。没有Fast validated脚本，也没有已验证可采用的Workspace或fill收益。
