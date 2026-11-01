# 源码、环境与测量口径审计

本轮仅离线读取历史文件并用CPU标准库计算SHA/统计；没有导入PyTorch、加载模型、查询GPU/NVML、操作进程或执行forward。本报告的“当前共同条件”指已经结束的历史调和批次，不指本轮重新测量。新旧FP裸速差异可靠重现，但版本是多项设置与包装的组合差异，现有环境证据尚不足以将差距归到某一个因素。

## 1. 证据身份与可访问性

- 调和任务`SOURCE_SHA256_MANIFEST.json`：`b59ee3a7417ebdae7f4f857bc3e8eb2f1644c373d6d28bf8d31da2871671c9f3`；7执行源+194外部依赖，本轮对55原文件201项逐一before/after-stat、bytes/SHA复核，全部PASS。
- Speedup任务`SOURCE_MANIFEST.json`：`d0069f87aafa9ab461827000e4322c0d4758c61a04d1d296da2b495e1c9ba90d`；27执行源+148外部依赖，175项同法复核全部PASS。
- 调和任务175个小文件及194个依赖的本地原始字节全部369/369符合封存catalog；Speedup的122个任务文件及148个小依赖共270/270符合封存SHA。Speedup旧manifest声明的dependency delivery_path被已封存下载器转换为`by_path_sha256`位置；本轮只解析该实际只读位置，并保留原声明路径、path-hash输入与真实SHA，没有改旧manifest。
- 调和20个单位的RESULT与批次回执SHA/bytes逐一相符，所有ENV/STARTED及相关元数据75项原端fresh SHA/bytes亦PASS。大trace不下载，本轮另由timeline分析记录原端stream SHA。
- 清单、每个文件的期望/实际SHA、原路径/实际本地路径均在`analysis/environment_local_asset_verification.json`、`environment_remote_source_verification.json`与`environment_remote_metadata_verification.json`，不是仅相信“PASS”字符串。
- 旧原AIME入口是manifest绑定的f552源码raw SHA，而不是当前HEAD推断。原历史AIME的完整驱动、硬件遥测、所有权重/Tokenizer原始SHA与可选kernel包清单并未恢复；这些缺失不能由同路径或今天共同权重SHA补全。

相关绑定源码及部分继承清单依赖（full SHA；逐项fresh核验；出现在清单不表示当前FP执行所有模块）：

|实际源路径|bytes|SHA256|
|---|---:|---|
|`source/matched_worker.py`|12661|`08b62839e88fa6bb49f10388a19e152e04beb0ab36b347ffcfdf2d9259e2ff8c`|
|`source/matched_controller.py`|9259|`c16ae53e7fb2c4ea6c1c02d798d79f09310d62cb6ce5bf41b4b2035a354b46da`|
|`/data/zypan/analyses/qwen_gdn_dual_axis_prospective_replication_5doc_v1/experiments/qwen_gdn/quantizer/QWEN_GDN_DUAL_AXIS_PROSPECTIVE_REPLICATION_5DOC_V1/src/codec.py`|3553|`ece859b901145c6d9d646b8befd01916037c9117ef4a80f7b801f27f79f8c23f`|
|`/data/zypan/analyses/qwen_gdn_dual_axis_prospective_replication_5doc_v1/experiments/qwen_gdn/quantizer/QWEN_GDN_DUAL_AXIS_PROSPECTIVE_REPLICATION_5DOC_V1/src/core.py`|5596|`aa338b8b83cde62b97c94fd205d1c62999f783e7ec3242cbb813dc877e01f575`|
|`/data/zypan/analyses/qwen_gdn_dual_axis_prospective_replication_5doc_v1/experiments/qwen_gdn/quantizer/QWEN_GDN_DUAL_AXIS_PROSPECTIVE_REPLICATION_5DOC_V1/src/worker.py`|22456|`e04ec85092ee4f2207699366b2aaa61e1e71038e728c9234477b5b2c21794773`|
|`/data/zypan/analyses/qwen_gdn_dual_m_equivalent_speedup_v1/source/baseline_worker.py`|31170|`0961d96bdafa92d6c3bca7d1980b74e9d70fe3aabff404417878ced20c7f1e41`|
|`/data/zypan/analyses/qwen_hadamard_dual_axis_real_state_runtime_gate_v1/experiments/qwen_gdn/quantizer/QWEN_GDN_HADAMARD_DUAL_AXIS_REAL_STATE_RUNTIME_GATE_V1/inherited_cpu_v1/reference_sources/historical_numerical_recovery.py`|35065|`b13152f81ecd87428936dc5eced5a2a9f1e06ff992e8ba5bbb81703206dec26c`|
|`/data/zypan/analyses/qwen_hadamard_dual_axis_real_state_runtime_gate_v1/experiments/qwen_gdn/quantizer/QWEN_GDN_HADAMARD_DUAL_AXIS_REAL_STATE_RUNTIME_GATE_V1/inherited_cpu_v1/reference_sources/modeling_qwen3_5.py`|88546|`90d929129ffc835d2652c604925c4f3842bc6e401e174ec6f0db2285dfb8f85a`|
|`/data/zypan/analyses/qwen_hadamard_dual_axis_real_state_runtime_gate_v1/experiments/qwen_gdn/quantizer/QWEN_GDN_HADAMARD_DUAL_AXIS_REAL_STATE_RUNTIME_GATE_V1/inherited_cpu_v1/reference_sources/stepquant_core.py`|1741|`3947d82c2f90d248a68e82773927f2b7b04d4044f678f17dfd7a4bfb931883b0`|
|`/data/zypan/analyses/qwen_hadamard_dual_axis_real_state_runtime_gate_v1/experiments/qwen_gdn/quantizer/QWEN_GDN_HADAMARD_DUAL_AXIS_REAL_STATE_RUNTIME_GATE_V1/inherited_cpu_v1/src/dual_axis_codec.py`|6234|`cd494824c242b2ce967dc470b50160aff25cf1776e1d45344d547a30fb6253e6`|
|`/data/zypan/analyses/qwen_hadamard_dual_axis_real_state_runtime_gate_v1/experiments/qwen_gdn/quantizer/QWEN_GDN_HADAMARD_DUAL_AXIS_REAL_STATE_RUNTIME_GATE_V1/runtime_sources/current_modeling_qwen3_5.py`|88546|`90d929129ffc835d2652c604925c4f3842bc6e401e174ec6f0db2285dfb8f85a`|
|`/data/zypan/analyses/qwen_hadamard_dual_axis_real_state_runtime_gate_v1/experiments/qwen_gdn/quantizer/QWEN_GDN_HADAMARD_DUAL_AXIS_REAL_STATE_RUNTIME_GATE_V1/runtime_sources/formal_c128_runner.py`|34744|`77b1e1f6dad8a0dd53a27cf5a47c02be86c26d2cf30732604ba6d3747e178f69`|
|`/data/zypan/analyses/qwen_hadamard_dual_axis_real_state_runtime_gate_v1/experiments/qwen_gdn/quantizer/QWEN_GDN_HADAMARD_DUAL_AXIS_REAL_STATE_RUNTIME_GATE_V1/runtime_sources/numerical_recovery.py`|35065|`b13152f81ecd87428936dc5eced5a2a9f1e06ff992e8ba5bbb81703206dec26c`|
|`/data/zypan/transformers-qwen35/src/transformers/models/qwen3_5/modeling_qwen3_5.py`|88546|`90d929129ffc835d2652c604925c4f3842bc6e401e174ec6f0db2285dfb8f85a`|
|`/data/zypan/worktrees/aime26-sglang-rotation-v1/reference_snapshot/qwen/modeling_qwen3_5.py`|88546|`90d929129ffc835d2652c604925c4f3842bc6e401e174ec6f0db2285dfb8f85a`|
|`/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/qwen_gdn/aime26/81920/run_qwen_aime26_formal.py`|26510|`9ff56d00b6b2dbaf9be3c3186bd30be8d3b9cc9d9b580a680316bcc8c01247dd`|
|`/data/zypan/worktrees/qwen-gdn-c6r-c7-canonical60-v1/experiments/QWEN_GDN_C6R_C7_CANONICAL60_V1/scripts/canonical60_worker.py`|7763|`73b0852b76b28402dac3510cbb03ecb5e4fcb30701d452b50d2a09b9c7bbfd18`|
|`/data/zypan/worktrees/qwen-gdn-deterministic-scale-discovery-v3/experiments/QWEN_GDN_DETERMINISTIC_RECOVERY_TO_SCALE_DISCOVERY_V3/scripts/numerical_recovery.py`|35065|`b13152f81ecd87428936dc5eced5a2a9f1e06ff992e8ba5bbb81703206dec26c`|
|`/data/zypan/worktrees/qwen-gdn-functional-causal-replication-v1/experiments/QWEN_GDN_FUNCTIONAL_CAUSAL_REPLICATION_V1/replication_worker.py`|1546|`ad7106a864729326e830c44ac982d15b0b9320ccfe9b7819bdf35e146827b3b6`|
|`/data/zypan/worktrees/qwen-gdn-scale-feedback-fullmodel-functional-causal-pilot-v1/experiments/QWEN_GDN_SCALE_FEEDBACK_FULLMODEL_FUNCTIONAL_CAUSAL_PILOT_V1/scripts/formal_worker.py`|19502|`c4e66288680bc0d2398a711097a16c824928ddca9057e5449cc13ca100314aca`|

旧入口对应历史commit`f552b9c527246c6dfe23205c0f1dc8ed1efd6546`；架构源是`90d929…`原始文件，历史源快照与调和批次均绑定此SHA；Transformers源commit`a353632607c59463e6ced86a44c2de3c2cd62d5e`。这支持Python源码身份，不单凭`inspect.unwrap`宣布物理CUDA kernel相同。模块级差异详见[CALLPATH_DIFF.md](/Users/pytenter/Documents/临时项目1/QWEN_GDN_RUNTIME_THROUGHPUT_ROOT_CAUSE_V1/analysis/CALLPATH_DIFF.md)。

## 2. 同一调和批次实际环境

下表来自每个单位加载后保存的`ENVIRONMENT.json`和RESULT；3重复及各自FP profile的设置一致，两个diagnostic只有指定字段不同。

|项目|OLD_FP/OLD_H_C128|NEW_FP/NEW_H_C128|证据资格与剩余混杂|
|---|---|---|---|
|Python/Torch/CUDA build/Transformers|3.10.18 /2.5.1+cu121 /12.1 /5.16.0.dev0|相同|归档版本相同；不等于所有so二进制/运行时动态库逐字节相同|
|GPU|GPU0 RTX3090 24GiB，UUID `GPU-a78176ed-f97f-c54f-6201-5a01b570f422`|同UUID|每单位真实PID/UUID准入及回执；不同CUDA_DEVICE_ORDER未造成换卡|
|模型参数|全模型cuda:0、BF16|全模型cuda:0、BF16|代码显式逐参数资格，不是CPU offload比较|
|GDN state|FP32 `[1,32,128,128]`、24层|相同|实际wrapper核验dtype/shape；浮点state不等于所有路径数值已证明相等|
|共享权重|4 safetensors原始SHA与bytes共同绑定|相同|证明本批次共同checkpoint；原旧AIME没有保留完整4片SHA，历史完全同权重资格UNKNOWN|
|attention配置|sdpa|sdpa|物理trace均有同名、同数的PyTorch flash splitkv/combine；8 attention层×128。配置字符串本身不是kernel证明|
|matmul TF32|False|False|这一项相同，不能归因“旧开TF32”|
|cuDNN TF32|True|False|真实差异；但需具体cuDNN调用成本/单变量干预才能归因。GDN depthwise CUDA kernel同名同数，不能仅开关推断主因|
|deterministic algorithms|False|True|真实组合差异；diagnostic单关全局开关的结果见下文|
|cuDNN deterministic|False|False|均False；不可将全局detTrue误写成此字段True|
|intraop / interop|24 /24|1 /1|intraop24对照仍保持interop1与OMP/MKL1，没有恢复旧完整线程状态|
|CUBLAS_WORKSPACE_CONFIG|未设置|`:4096:8`|未做独立开关；即使globaldetFalse仍保留它，残余cost UNKNOWN|
|OMP_NUM_THREADS / MKL_NUM_THREADS|未设置/未设置|1 /1|没有独立OMP/MKL干预；Torch线程值并不能替代所有库线程状态|
|CUDA_DEVICE_ORDER|未设置|PCI_BUS_ID|UUID核验可排除实际换卡，环境仍如实记录|
|完整driver/cuBLAS/cuDNN runtime版本|NOT_RETAINED|NOT_RETAINED|ENV、admission和trace元数据不含这些版本；相同batch降低版本切换可能，不构成完整二进制审计|
|CPU affinity/频率/进程竞争/调度|NOT_RETAINED|NOT_RETAINED|归档仅lscpu+被监控worker的/proc/stat及2s GPU采样，没有机器级CPU负载、cgroup配额、per-thread调度/CPU频率时间线|
|可选包/fused dispatch原历史完整清单|UNKNOWN|UNKNOWN|实际调和trace可比较所观察kernel；不能外推原AIME全部实现选择|

`torch.utils.deterministic.fill_uninitialized_memory`的历史值没有显式ENV回执；新增fill现象需按trace父算子/实际ATen分派分析，不能凭今天安装源码默认值替代历史实测。现在重新读取的Profiler Python源只说明本轮解释采用的定义；若没有历史清单匹配，不补写为历史执行已锁定文件。

共同checkpoint原始SHA（来自归档shared-checkpoint资格，本轮未重复读取20GB权重、未下载模型）：

|分片|bytes|SHA256|
|---|---:|---|
|1/4|5276436216|`db6f444b43d318c92f360a13a25561a6a65b10c0631b8ed305a426dbaa6c380e`|
|2/4|5335161512|`31c7d7e2dd5d207840b31cc59083c8f4c4718959149e0358c0364052bb9a0330`|
|3/4|5368717440|`7ec36ba3a4176a44c3c0876ad80c56a2f70c84bf008d82e9501df642f17dadec`|
|4/4|3325995712|`b62b0c4cd7e44edee103ee8f4fe225f246d5e768e07bfd5f25b63a8aa1fdd0c6`|

## 3. workload与计时资格

调和批次固定同一非AIME 32768输入JSON（rawSHA`18faa0ca311362181c5c3340d19d72ac39422d1ee07846e516bf34074a51703d`），只消费前256个token。每次fresh process、B=1、单token第一次prefill，前128次forward作为warmup；缓存保留，再计128次forward（RoPE128–255、8路KV从128增至256）。每单位完整256个forward、8192层调用、24次chunk仅首次prefill、6120 recurrent，其中计时3072 recurrent。不能称为32K cache速度。

128窗口入口`torch.cuda.synchronize()`；profile若启用则在入口之后创建/进入，然后开始monotonic wall；窗口结束统一synchronize后停wall，再退出profile及作key_averages/trace导出。普通窗口没有另加逐token全局同步；但两路径都保留各自真实finite gate并发生逐token标量读取。计时包含原路径安全检查、Python包装与共同计数/metadata hook，排除模型load、warmup、采样、评分、字节hash及trace导出。共同metadata hook也不是“零成本”；这是同一受控测量协议，不是原AIME无hook吞吐。

旧/新Hadamard+C128仍有prefill QDQ与Hadamard矩阵缓存差异，因此次要H对照不称数值严格同路径。FP是首要比较，未调用Dual-M/Column-LS。

所有无Profiler重复的原始数值重新从回执计算：

|条件|TPS R0,R1,R2|中位TPS|范围TPS|中位ms/token|
|---|---|---:|---|---:|
|OLD_FP|13.640066, 13.742036, 13.742683|13.742036|13.640066–13.742683|72.769421|
|NEW_FP|7.913233, 7.479526, 7.776711|7.776711|7.479526–7.913233|128.589070|
|OLD_H_C128|8.999410, 9.064610, 9.740766|9.064610|8.999410–9.740766|110.319143|
|NEW_H_C128|6.650598, 6.721773, 6.575868|6.650598|6.575868–6.721773|150.362422|
|DIAG_NEW_DETERMINISM_OFF|7.996658, 8.029660, 7.922476|7.996658|7.922476–8.029660|125.052233|
|DIAG_NEW_INTRAOP24|7.516472, 7.655933, 7.693205|7.655933|7.516472–7.693205|130.617658|

NEW比OLD中位TPS低43.409327%，OLD/NEW倍率1.767076；裸速范围完全不重叠，runtime组合差异可靠。该事实不等于已识别其中某一机制的独占因果份额。

## 4. 已有两个单变量diagnostic究竟说明什么

1. `DIAG_NEW_DETERMINISM_OFF`：只在完成new reference setup后调用`torch.use_deterministic_algorithms(False)`；compensated chunk、CUBLAS `:4096:8`、1/1线程、TF32 False、有限性包装和输入均保留。中位7.996658 TPS，相对NEW +2.828287%；范围7.922476–8.029660，高于NEW本次3点范围，但远未回到OLD 13.742036。这个干预反对“单独globaldet就是全部43.4%退化”的解释；不能排除CUBLAS约束/其它确定性设置或它们的交互。未留逐token输出等价证据，不能将它计为可采用优化。
2. `DIAG_NEW_INTRAOP24`：只改`torch.set_num_threads(24)`；interop仍1、OMP/MKL仍1。中位7.655933 TPS，相对NEW −1.553074%，范围交叠；不支持“只把intraop1改回24即可恢复”。不能由此否定线程亲和、库线程、CPU频率或竞争问题。
3. 六个diagnostic排在全部12个main单位之后；这是预先冻结的不同时间fresh进程，不是同一状态随机交叉/相邻配对。R0/R1/R2标签不自动构成统计配对。3点中位差为描述性观测，未给不存在的精确因果CI，也不把profile差值当normal wall可相加成本。

完整main顺序：OLD_FP0→NEW_FP0→OLD_H0→NEW_H0→NEW_FP1→OLD_H1→NEW_H1→OLD_FP1→OLD_H2→NEW_H2→OLD_FP2→NEW_FP2。之后detOFF0→intra24_0→intra24_1→detOFF1→detOFF2→intra24_2，最后OLD_FP_PROFILE0→NEW_FP_PROFILE0。OLD_FPR1/R2在较晚热状态依然13.742，反对仅时间漂移即可解释全差距，但没有排除全部环境混杂。

## 5. Profiler口径与环境观察

调和两个profile使用`activities=[CPU,CUDA]`、`record_shapes=False`、`profile_memory=False`、`with_stack=False`；只在测量128token窗口开启。没有完整Python调用栈或tensor-shape记录，不能从缺失栈伪造精确GEMM源行。新旧均有相同`MATCH/model_forward`、`MATCH/finite_and_QDQ`范围标注。两个profile在所有普通试验后按OLD再NEW顺序单次执行，没有profile条件重复，计时会受instrumentation影响。

|独立profile单位|窗口wall秒|TPS|相对各自普通中位wall|
|---|---:|---:|---:|
|OLD_FP_PROFILE_R0|17.456808|7.332383|1.874157×|
|NEW_FP_PROFILE_R0|25.980477|4.926776|1.578458×|

这两个倍率只是分开试次的instrumented/noninstrumented观测，不是可减去的固定Profiler开销。原Speedup profile的`profile_memory=True`，而调和profile为False；两任务的profiler数据不能直接混成统一cost估计。Speedup的裸128窗中位FP/H/HDual-M为7.932661/6.662540/6.801314 TPS，与调和NEW路径量级相符，但前者多条件共享一个worker模型/独立缓存循环，后者每单位fresh进程，不能以小数差当根因。

归档每2秒GPU遥测按RESULT实际decode起止筛选（本轮没有查询NVML）：

- 18个普通窗口的所有采样SM=1695MHz、显存时钟=9501MHz、power limit=350W。每窗口仅4–10个点；不能保证点间恒定，也没有retained throttle reason/P-state完整时间线。
- OLD_FP三窗util中位39%、39.5%、39%；NEW为24%、22%、24%。功耗分别中位172.165/179.645/178.710W vs151.525/152.380/153.820W。这与新路径更少GPU忙时间相容；不能把较低功耗反向定义为已证实原因。
- OLD_FP温度采样41–57°C，NEW46–55°C；OLD后两窗仍快，因此没有支持“新窗热降频导致43%差距”的直接证据，但仍保留稀疏采样限制。
- 所有decode采样未见非本worker的GPU PID；不能据此排除CPU竞争、点间瞬时GPU占用或采样未覆盖的事件。
- `/proc/pid/stat`的utime+stime增长约100 ticks/s，两FP均近一个core的活跃量（若USER_HZ=100则约100%；历史HZ未单独保留，结构化输出保留ticks/s与条件假设）。CPU算力工作/驱动busy等待/自旋不能凭此区分。
- last CPU字段在OLD_FP R0/1/2分别为5/19/16，NEW为4/14/21；只是采样时最后运行CPU，不是affinity。没有sched/frequency/NUMA memory placement时间线。profile_NEW采样CPU16/25/41，也不足以归因其mm host时间异常。

因此，同卡/输入/参数dtype/缓存长度/主架构源/ATen mm次数/所观察GPU kernel名称与频率已有较强控制；driver完整二进制、CPU频率/竞争/亲和、线程库状态、backend配置交互及profiler扰动仍有缺口。不能写“环境完全相同，已排除所有混杂”。

## 6. 审计结论与允许的推断

- **已证实**：在同GPU、同固定输入/短缓存协议下，旧与新runtime组合有稳定裸速差；mm关联GPU计算时间近似，但CPU侧记账与发射存在差异；单关globaldet或单改intraop24均未恢复旧速度。
- **没有证实**：CUBLAS workspace、finite包装、空tensor确定性fill、interop/cudnn或CPU竞争中的哪一个是43.4%下降的主触发原因；也没有具备严格数值/状态/确定性等价资格的优化。
- **不能补全**：原旧AIME完整硬件/driver/权重SHA状态；没有采集的调用栈/CPU scheduler；不把当前installed source/default配置自动倒填为历史。
- 环境证据支持继续围绕new包装/后端host提交路径建立最小单变量因果验证，但本轮GPU授权为NO，以上均为现有历史观测。优化与GPU方案由根报告在相同门禁下单独界定。

## 可复核输出

- [local assets SHA](/Users/pytenter/Documents/临时项目1/QWEN_GDN_RUNTIME_THROUGHPUT_ROOT_CAUSE_V1/analysis/environment_local_asset_verification.json)
- [376 frozen sources fresh SHA](/Users/pytenter/Documents/临时项目1/QWEN_GDN_RUNTIME_THROUGHPUT_ROOT_CAUSE_V1/analysis/environment_remote_source_verification.json)
- [75 metadata fresh SHA](/Users/pytenter/Documents/临时项目1/QWEN_GDN_RUNTIME_THROUGHPUT_ROOT_CAUSE_V1/analysis/environment_remote_metadata_verification.json)
- [20 receipts、全部ENV、18裸窗与诊断计算](/Users/pytenter/Documents/临时项目1/QWEN_GDN_RUNTIME_THROUGHPUT_ROOT_CAUSE_V1/analysis/environment_historical_measurements.json)
- [归档遥测逐窗统计与限制](/Users/pytenter/Documents/临时项目1/QWEN_GDN_RUNTIME_THROUGHPUT_ROOT_CAUSE_V1/analysis/environment_archived_telemetry.json)
- [仅标准库审计脚本](/Users/pytenter/Documents/临时项目1/QWEN_GDN_RUNTIME_THROUGHPUT_ROOT_CAUSE_V1/scripts/environment_evidence_audit.py)
