# Codex 正式科研任务：Ling/KDA 四文档32K Clean-Room Final

**TASK_ID：** `LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1`

**执行目标：** 一次性完成已冻结的4篇文档、12条32K轨迹，回答P1是否比Original L7具有额外长期预测价值。

**FULL_32K_EXECUTION_AUTHORIZED = YES**

**MAX_CONCURRENT_GPUS = 2**

**MAX_TOTAL_NEW_GPUH = 36.0**

**TARGET_GPUS = 两张已知RTX3090 24GB，优先GPU2/GPU3**

以上授权仅对本独立Clean-Room任务有效，不追溯修改execution_05，也不授权其他实验。

---

## 0. 最高优先级：得到科学答案，而不是继续修补工程

你是一名负责Ling/KDA recurrent-state INT8量化的高级科研工程Agent。

此前 `LING_KDA_P1_L7_PROSPECTIVE_EXTENSION_4DOC_V1` 已经多次尝试正式32K前瞻验证，但长期停留在launcher、FLA配置、数值恢复与GPU遥测等工程问题上。

用户明确要求：

**停止围绕旧execution_01至execution_05反复修补，采用一条独立、简洁、可复现的执行路线，尽可能高效地完成冻结的科学问题。**

本轮不再修复旧execution_05，不再扩展复杂的工程诊断系统，不得自动启动execution_06/07/08式的修补循环。

本轮成功的最终判据是：

- 4条真实FP 32768-token轨迹全部完成。
- 8条真实persistent INT8 student 32768-token轨迹全部完成。
- 四篇文档各有完整的K_P1、K_L7和Delta。
- 384个评分位置的Full-vocab FP32 KL与原冻结评分协议一致。
- 所有必须的数值与QDQ资格通过。
- 保留独立科学结果、资源账本和可核验的SHA。
- 最后停止，不启动训练或新的实验。

工程测试只服务于这些目标。

禁止为了追求工程流程完美而增加无关资格、重复诊断或新的科研分支。

但绝不能为追求完成率而牺牲科学数值正确性、GPU资源安全或真实INT8语义。

## 1. 必须保留的科学协议

原科学ID：

`LING_KDA_P1_L7_PROSPECTIVE_EXTENSION_4DOC_V1`

此次建立独立执行ID：

`LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1`

新执行ID仅代表工程运行实现不同。

不得改变原科学问题或将这次运行视为重新选择的前瞻实验。

### 1.1 冻结科学身份

必须读取原始预注册、selection freeze、FROZEN_SPEC、旋转manifest、token manifest和数学worker。

已知身份：

- 预注册SHA：`54d6d24fbc3142a692ef1a6c04a5617cf6b0abe6702a662722f34fc1b6141722`
- 四文档freeze SHA：`a5c227d15918a249f8ae2059539c15f94668cb2afddf3aecd8a9f7e13851354b`
- FROZEN_SPEC SHA：`8ae8927ed37209efccbf19c672b9baeb46c0d7f272601f8f54d1f9609b64f3d1`
- 原冻结时间：2026-10-14 10:44:26北京时间

原先100篇候选、20篇历史排除、22篇长度不够、58篇合格pool的流程保持不变。

此前4篇×16旋转的64个评分保持冻结。

不得重新排序、重新计算候选排名、修改P1公式或选择新的旋转。

### 1.2 四篇文档和12条轨迹

| 文档 | P1选择 | Original L7选择 |
|---|---|---|
| pg19-test-41603 | H_to_L7_a50 | H |
| pg19-test-2544 | H_to_L6_a50 | H_to_L7_a50 |
| pg19-test-7987 | H_to_L7_a50 | H_to_L7R_a25 |
| pg19-test-26183 | H_to_L6_a50 | H_to_L7R_a25 |

每篇必须执行：

1. FP reference
2. P1选择的INT8 rotation
3. Original L7选择的INT8 rotation

总计4 FP + 8 INT8 = 12条轨迹。

具体checkpoint、旋转参数、Hadamard乘法顺序、Cayley矩阵、quantizer坐标与模型输入，均以原冻结manifest为准。

如果上表和真正的冻结资产发生冲突，停止并报告身份冲突，不得自行选择其中一版。

### 1.3 长度和评分

每条正式轨迹：

32768 consumed tokens。

评分位置：

`t = 8192 + 64*k, k=0,...,383`

共384个位置，最后一个位置32704。

遵循原冻结的：

消费 `ids[t]` 后预测 `ids[t+1]`。

完整词表大小：

157184。

原评分：

`KL(FP || student)`

使用原FP32完整词表logits、原softmax/KL数值实现及原Python `sum / 384`。

绝对禁止：
- 改成Top-K KL。
- 改成sampled-vocabulary KL。
- 改用NLL代替KL。
- 改变评分位置。
- 改变平均方法。
- 用新的近似reducer替代原合同。
- 事后扩大数值容差。

最终：

`Delta = K_P1 - K_L7`

Delta小于0表示P1选择更好；Delta大于0表示Original L7更好。

### 1.4 Persistent INT8原则

每个token都必须执行真实recurrent-state INT8 QDQ/writeback。

下一token消费上一次量化写回状态。

禁止：
- 延迟requantization
- FP16/FP32 persistent compensation
- 跳过QDQ
- 在INT8学生条件中使用FP完整persistent state
- 改变INT8 R128量化协议
- 修改旋转
- 模型微调
- 新Loss或训练

如原实现使用FP32 tensor承载QDQ后的数值，只能声明persistent-QDQ/writeback semantics，不能声明physical resident INT8 storage。

## 2. Clean-Room：只更换控制层，不改数学层

这是本任务最重要的设计要求。

**Clean-Room不意味着重新实现KDA数学。**

重新写数学kernel会引入新的数值风险，违背用户希望高效得到答案的目标。

正确做法：

- 保留冻结的数学worker。
- 保留KDA量化与QDQ代码。
- 保留原旋转与参数加载。
- 保留FP32 KL/reducer。
- 保留原模型权重与token输入。
- 重新实现最简launcher、scheduler、resource admission、checkpoint管理和停止监督。

### 2.1 历史代码只读复用

优先读取：

`/data/zypan/worktrees/ling-kda-prospective-execution05`

原实验输出：

`/data/zypan/experiment_runs/LING_KDA_P1_L7_PROSPECTIVE_EXTENSION_4DOC_V1/execution_05`

以及原冻结scientific assets。

原execution_05修复commit：

`902d7974c89465df882403e96cc8dcc8dc1c0e85`

基于：

`d3cd88d4643a460e027cae6ba6989628c21f60bf`

请实际确认commit与文件SHA，不假定路径一定存在。

建立全新、干净的工作树和独立输出目录。

禁止修改execution_01至execution_05。

禁止重写旧STOP。

禁止覆盖历史checkpoint。

### 2.2 数学源码资格

确认新旧数学模块完全一致，包括：

- 模型前向
- FLA kernel选择
- recurrent state update
- persistent INT8 QDQ
- rotation应用
- checkpoint状态定义
- observer
- FP32 KL/reducer

使用文件SHA和必要AST比较，不必重复之前全部115文件的人工式长篇审计。

如果实际数学实现无法保持冻结等价，直接STOP。

仅允许修改控制层：
- launcher
- env setup
- resource admission
- scheduler
- logging
- checkpoint orchestration
- atomic outputs
- failure handling

所有控制层改动必须在GPU执行前记录并冻结。

## 3. 固定正确的FLA和运行环境

本轮必须从原execution_05已验证的数值路径起步。

进程启动之前设置：

`PYTHONHASHSEED=0`

`CUBLAS_WORKSPACE_CONFIG=:4096:8`

`LING_GATED_RMSNORM_FIXED_CONFIG=BT16_W8`

`FLA_CACHE_MODE=default`

显式指定原冻结的FLA配置目录和6份JSON。

使用原始配置，不使用自动选择的新profile。

已知正确fingerprint：

- Conv BD8 / warps8
- Gated RMSNorm BT16 / warps8

必须以真实源码配置与运行profile复核。

历史BD128/warps4对应旧失败fingerprint。

不能为了加速切回该配置。

禁止：
- 改变Torch/FLA数值kernel
- 重新编译未经资格的替代kernel
- 修改batching数学语义
- 通过更宽松容差强行PASS
- 在worker启动后才补环境变量

可以使用已有经过验证的软件环境。

如果当前环境发生漂移，优先恢复已验证环境，而不是研究一套全新的依赖版本。

## 4. 重新设计GPU资源准入：简单、安全、明确

优先使用原已知两张RTX3090：

GPU2：
`GPU-7d6246ed-c3d1-75bc-c2c7-d02eb2882cca`

GPU3：
`GPU-759526ed-efb2-e7f6-881c-ab098e0b05ae`

必须实测设备ID与UUID映射。

不自动使用GPU6。

不自动改用4090或A800。

如果必须更换设备，当前执行STOP，不得静默迁移。

### 4.1 不再复刻旧13点双零遥测门禁

旧execution_05在GPU3出现：

SMI utilization = 1%
NVML utilization = 0%

虽然内存算术与进程身份一致，却因为遥测不一致直接终止全部任务。

**新的Clean-Room执行不再把这种单次微小采样差异自动解释为资源冲突。**

这是前瞻性的新工程准入规则，不追溯改变旧execution_05的FAIL。

新的准入依据：

1. 设备UUID正确。
2. GPU上没有未授权的外部compute进程。
3. 当前显存真实剩余量足够。
4. 可归属显存与进程信息没有异常增长。
5. 计算占用率没有持续明显升高。
6. 不会影响其他GPU用户。
7. 本任务的模型实际峰值在安全范围内。

执行前用3次间隔采样检查资源，保存时间戳及SMI/NVML读数。

可以将双方均不超过2%的低利用率微小差异视为非阻断提示，前提是无未知compute进程、显存稳定且内存余量满足预算。

单个1%/0%样本不能直接导致全局STOP。

但是：
- 发现未知外部compute进程
- 显存不足
- 不可归属的显存持续增长
- 明显持续GPU占用
- 设备身份不一致

必须STOP，不能强行抢占。

不能杀他人进程。
不能reset GPU。
不能偷偷忽略真正资源冲突。

### 4.2 根据真实已测显存设置headroom

execution_05已测：

CUDA allocated峰值约16.49GB。

CUDA reserved峰值约20.13GB。

NVML采样峰值19536MiB。

这些不是完整生命周期绝对峰值。

新的resource admission在此基础上保留额外至少2GiB可用空间余量，并根据实际GPU可用容量判断是否能安全运行。

不得仅检查CUDA allocated而忽略reserved与其他进程。

记录实际设备峰值，但不重新开展与科学运行无关的长期遥测诊断。

## 5. 最小数值资格：必须通过，但不无限重复

目标是用最少必要测试证明12条科学轨迹可运行。

在正式32768-token continuation前，必须有：

### G0：身份与环境

- 预注册SHA PASS。
- 四文档输入SHA PASS。
- 旋转和模型SHA PASS。
- 数学源码SHA PASS。
- FLA配置身份PASS。
- 资源准入PASS。

### G1：真实模型启动canary

每张GPU各执行一次最小模型启动与短forward。

验证：
- 实际FLA profile
- 实际RMSNorm profile
- kernel加载
- 模型加载
- GPU峰值
- 自然退出

只做一次必要的GPU canary，不反复执行数十次环境诊断。

### G2：FP和INT8数值资格

必须完成原12个科学key对应的数值资格。

使用原冻结fresh64合同。

检查：
- 64-step完整词表logits
- native cache
- checkpoint/resume
- RNG恢复
- observer ON/OFF
- fresh-process deterministic replay

对于INT8条件还必须额外检查：
- recurrent QDQ code/scale
- writeback
- 下一token消费量化状态
- 全部实际INT8路径资格

FP audit_state=None只能标记N/A。

不能冒充INT8 QDQ证明。

同一个条件的数值资格和正式运行应使用相同GPU UUID与运行配置。

不强制要求不同GPU的bitwise完全一致；必须保证每个条件自身的fresh-process一致性及合法配对数值协议。

**正式科学执行必须等12/12资格全部完成并PASS。**

如果原冻结合同要求更严格的资格顺序，优先遵守原合同。

如果本轮核心数值资格失败，停止，不自动修数学kernel。

## 6. 优先复用两个已验证FP checkpoint

execution_05已有两个8192-token真实FP checkpoint。

### Doc41603

路径：

`/data/zypan/experiment_runs/LING_KDA_P1_L7_PROSPECTIVE_EXTENSION_4DOC_V1/execution_05/runs/pg19-test-41603__FP__8192.pt`

SHA256：

`367ad2083285fd5a74c69ead6b84e9dbe443dfd22e628ab038cf0966cc10a0a7`

原对应GPU2。

### Doc26183

路径：

`/data/zypan/experiment_runs/LING_KDA_P1_L7_PROSPECTIVE_EXTENSION_4DOC_V1/execution_05/runs/pg19-test-26183__FP__8192.pt`

SHA256：

`2e6abbe74fc5266fd26c3b74da4977cdd3849382534852503fb68a10e0f77f4f`

原对应GPU3。

两份checkpoint此前已经重新计算且与历史对应CP完整SHA一致，并通过各自fresh64资格。

允许复用的条件：

- 文件完整SHA一致。
- 源码及模型身份一致。
- FLA数值路径一致。
- 输入及token cursor一致。
- checkpoint含完整所需cache/RNG状态。
- 当前GPU上的fresh64 resume达到原资格。
- FP评分与恢复时序不产生缺失位置。

如合格，直接复用，避免重复0→8192 FP计算。

如果不能证明某checkpoint符合新执行条件，则在本轮预算内重新计算该FP前缀，不允许未经验证强行复用。

**不能把FP checkpoint用于INT8学生条件。**

每条学生轨迹必须有自身从起始token开始的真实persistent QDQ历史。

不允许以共同FP前缀替代学生前缀，除非原冻结科学合同明确允许；本次按原合同执行，不引入新的共同anchor设计。

其他历史10个checkpoint如无法证明原FLA配置及数值资格，不复用。

## 7. 最短科学执行调度

目标：让两张GPU持续执行有效科学任务，减少重复模型加载和前缀计算。

先完成身份、资源及全部12项数值资格，然后执行完整32K。

### 7.1 设备亲和性

优先安排：

GPU2：
- pg19-test-41603
- pg19-test-2544

GPU3：
- pg19-test-26183
- pg19-test-7987

这样可优先复用Doc41603与Doc26183各自已合格的FP checkpoint。

但是必须在正式运行前核对原冻结任务顺序要求。

原四文档逻辑顺序及每篇：
FP→P1→Original L7
必须保持。

如果原协议允许跨文档并行，则两张GPU可并行处理不同文档，但同一文档三个条件保持固定顺序且固定GPU。

如果原协议严格冻结全局串行顺序，则遵守原协议，不擅自以并行为理由修改。

最终冻结实际调度和设备亲和性。

不得根据部分科学结果重新排列测试顺序。

### 7.2 前缀执行

为12个科学条件建立合法8192-token起点。

合格FP checkpoint优先复用。

其余条件正常执行0→8192。

这12个parent及其fresh64资格应当尽量共用已经实际完成的合格计算，避免重复前缀。

但资格fresh64不能自动冒充正式Primary结果，除非原冻结协议明确允许且评分位置、checkpoint和reducer状态完整一致。

### 7.3 正式continuation

每条轨迹完成：

8192→32768。

按原协议记录384个评分位置。

每个文档的FP reference只计算一次，并供该文档两个student配对使用。

可以保存经过SHA验证的FP32 full-vocab参考logits，避免重复FP forward。

Student KL计算必须读取相同文档的真正FP参考。

所有正式文档必须保留：
- 原始384点评分值
- 正确的平均
- score position hash
- input hash
- rotation hash
- checkpoint identity
- resume identity
- 运行设备身份

### 7.4 断点恢复

为每条轨迹使用原合同兼容的原子checkpoint。

至少保留8192边界及合理的长程恢复点。

必须包含：
- 当前token cursor
- recurrent cache
- RNG
- 原量化state
- 已处理评分位置
- reducer累加状态
- model/source/rotation identity

恢复必须证明：
连续运行与分段运行在原冻结数值合同下等价。

checkpoint间隔根据真实state大小和IO成本在正式执行前确定，不得创建无上限的大量checkpoint。

意外进程退出不能静默重试。

预定的checkpoint阶段切换可以按资格恢复；数值FAIL、资源冲突或身份失败必须停止。

## 8. GPU预算与资源策略

本轮GPU预算为：

`MAX_TOTAL_NEW_GPUH = 36.0`

这是硬上限，不是要求必须消耗36小时。

实际计入：
- GPU canary
- 模型加载
- 8192-token parent
- fresh64
- 正式32768-token运行
- checkpoint恢复
- GPU性能资格

两张GPU同时运行时分别累计卡小时。

使用历史实测性能作为初始参考，但不能直接将以前的不完整GPUh除以token数量作为精确预测。

在第一批真实合格轨迹取得吞吐后，计算12条完整科学轨迹的剩余GPUh预计值。

如果预测总成本超过36GPUh，必须在超支前STOP并提交真实预算证据，不自动提高额度。

如果预算足够，直接连续完成冻结任务。

不要每完成一篇就停下来等待人工再次授权。

本次已授权的是整个受限四文档正式执行。

但如果发生数学或数值资格失败，本次授权不能用于无界修复和反复重启。

## 9. 最终统计与科学判定

必须计算四篇文档的：

`K_P1`

`K_L7`

`Delta = K_P1 - K_L7`

生成正式表格：

| Doc | P1 Rotation | Original L7 Rotation | K_P1 | K_L7 | Delta | Winner |
|---|---|---|---|---|---|---|

对于四篇，每一条Delta必须来源于完整384个实际合法KL数值。

不能将没跑完的结果填0。

不能将历史短程排序填入正式32K结果。

不能把fresh64 KL当成正式Primary。

### 9.1 数值分辨率

原经验阈值：

`3.3300064494019496e-6`

该数值属于历史资格经验，不能直接假定新环境自动合格。

应在正式结果揭示前，根据原冻结数值合同和独立的repeatability证据确认是否适用。

如果新环境无法合法确认分辨率，仍然报告完整原始Delta，但对微小差异标记NUMERICALLY_UNRESOLVED，不得伪造确定胜负。

### 9.2 文档层面统计

统计单位为文档。

样本量最多4。

报告：
- P1胜出篇数
- Original L7胜出篇数
- 数值无法分辨篇数
- 缺失篇数
- 平均Delta
- 中位Delta
- 每篇效应大小
- 方向一致性
- 如适用，exact sign test

不把384个位置当成384个独立文档。

不把layer/head当成独立样本。

不将新4doc与原MIXED 2doc合并为未经预注册的n=6 confirmatory实验。

注意，即使4/4篇全是P1胜出，四个独立配对在无平局、零假设对称条件下的单侧sign-test p也只有0.0625。

因此不能凭4/4直接宣称达到p<0.05的普遍显著性。

### 9.3 科学终态分类

完整执行之后，必须客观给出：

- `P1_DIRECTIONALLY_SUPPORTED`
- `MIXED_OR_INCONCLUSIVE`
- `P1_INCREMENTAL_VALUE_NOT_SUPPORTED`

以及实际统计证据。

上述分类只是研究决策，不得取代原始数值。

如出现3/4或4/4支持P1，应明确是有限文档上的方向性证据，而不是泛化能力完全确认。

如果P1没有稳定增量，直接如实报告，不增加其他旋转、新Loss或新训练来挽救结果。

## 10. 工程失败策略：不再无限缝补

本次必须采用明确的停止原则。

### 允许的工程处理

- 修复新launcher的路径或配置错误。
- 补齐正确的启动环境变量。
- 恢复已验证的FLA配置。
- 修复纯控制层的明显缺陷。
- 使用经过验证的checkpoint进行正常resume。
- 合理处理新的低利用率遥测采样差异。

所有改动必须在相应运行前登记。

### 禁止的行为

- 为了通过数值门禁修改数学kernel。
- 为了通过验收放宽数值容差。
- 为了节省时间跳过INT8 QDQ。
- 静默重试失败的正式科学轨迹。
- 无限重写GPU admission。
- 引入新的profiling实验来研究已经不影响安全的轻微遥测差异。
- 改变FP或INT8状态语义。
- 改变32K长度或384评分位置。
- 先看结果再修改rotation。
- 添加新的训练目标。
- 扩充科学样本或重选文档。

如果出现一次实质性数学身份、数值资格或资源安全失败：

保留证据，立即停止相关依赖执行。

不要自行开启新一轮execution attempt。

在完成已授权的最小工程修复后仍无法解决核心资格的情况下，分类为STOP，不再继续扩展诊断。

这里的目标不是保证无条件跑出结果，而是在一个可信且有界的执行方案内，尽可能高效地完成结果。

## 11. 数据和输出组织

建立独立目录，例如：

`/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1/`

新工作树也必须独立。

建议交付：

- `preregistration.json`
- `source_manifest.json`
- `engineering_change_manifest.json`
- `resource_admission.json`
- `qualification_matrix.json`
- `run_manifest.json`
- `gpu_resource_ledger.json`
- `final_scientific_results.csv`
- `final_scientific_results.json`
- `FINAL_SCIENTIFIC_REPORT.md`
- `FAILURE_HISTORY.md`
- `hashes/artifact_sha256.txt`
- `hashes/DELIVERY_VERIFICATION.json`

各篇文档另行存储实际checkpoint、384点评分数组及运行摘要。

只保存真实生成的科学资产。

未执行条件标记NOT_RUN。

最终不需要保存大量逐token全模型trace；只保留评分合同、必要数值资格与复现所需的最小证据。

大checkpoint保存在服务器，使用SHA引用，不必复制到本地交付目录。

不自动git push。

不覆盖历史任何文件。

## 12. 最终必须输出的结果

Codex最终必须输出：

`TASK = LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1`

`SCIENTIFIC_PROTOCOL_IDENTITY = PASS/FAIL`

`ENGINEERING_IMPLEMENTATION = CLEANROOM_CONTROL_ONLY`

`GPU_RESOURCE_ADMISSION = PASS/FAIL`

`FLA_NUMERICAL_PROFILE = PASS/FAIL`

`MODEL_AND_ROTATION_IDENTITY = PASS/FAIL`

`FRESH64_QUALIFICATION = x/12`

`INT8_PERSISTENT_QDQ_QUALIFICATION = PASS/FAIL/INCOMPLETE`

`FORMAL_FP_COMPLETE = x/4`

`FORMAL_STUDENT_COMPLETE = x/8`

`ACTUAL_VALID_DOCUMENTS = x/4`

`PRIMARY_384_SCORE_REPRODUCTION = PASS/FAIL/NOT_RUN`

`P1_WINS = 实际数量`

`L7_WINS = 实际数量`

`TIE_UNRESOLVED = 实际数量`

`MISSING_DOCUMENTS = 实际数量`

`NEW_GPU_HOURS_USED = 实际值`

`MAX_GPU_HOURS = 36.0`

`FINAL_SCIENTIFIC_CLASSIFICATION = 实际分类`

`FINAL_EXECUTION_STATUS = COMPLETE_AND_STOP / PARTIAL_STOP / FAIL_CLOSED_STOP`

如果4doc完整完成，必须提供如下表格，不能只提供总体结论：

| Doc | K_P1 | K_L7 | Delta | Winner |
|---|---|---|---|---|
| 41603 | 实测 | 实测 | 实测 | 实测 |
| 2544 | 实测 | 实测 | 实测 | 实测 |
| 7987 | 实测 | 实测 | 实测 | 实测 |
| 26183 | 实测 | 实测 | 实测 | 实测 |

如果未完成，准确说明缺哪些轨迹，不把缺失当作平局或科学失败。

最终科研报告回答：

1. P1在这4篇预先冻结的文档上是否比Original L7选择更好？
2. 这个结论与原2doc MIXED 1:1是什么关系？
3. P1是否表现出足够的增量预测价值，值得继续研究？
4. 如果没有，是否应该停止当前P1 selector路线？
5. 如果有，下一步最合理的科学验证是什么？

第4、5题只提出科研判断，不得自动启动后续实验。

## 13. 最终停止边界

本任务只授权：

- 新Clean-Room控制层。
- 原冻结数学与数值资格。
- 最多两张合格RTX3090。
- 最多36新增GPUh。
- 原四篇文档。
- 原4FP+8student完整32K。
- 原384位置Full-vocab KL。
- 必要checkpoint与恢复。
- 完整科学结果封存。

不授权：

- 新P1训练。
- L8/L9训练。
- 新STE/Cayley实验。
- 修改rotation。
- 修改quantizer。
- 新AIME。
- Free generation。
- Qwen实验。
- 更换科学数据。
- 自动进行第二轮Clean-Room实验。

**最终原则：工程控制尽量简单，数学正确性绝不妥协，GPU资源有界，科学目标明确。**

请先给出简短执行计划并立即按照门禁实施已授权任务，不要因为非必要的工程细节反复请求确认。

优先拿到完整四文档32K结果。

如果全部科学门禁通过，执行到底。

如果出现无法在本次有界方案内解决的核心失败，保存证据后STOP，不开展新的修补研究。

完成后提交可核验的最终科学报告。
