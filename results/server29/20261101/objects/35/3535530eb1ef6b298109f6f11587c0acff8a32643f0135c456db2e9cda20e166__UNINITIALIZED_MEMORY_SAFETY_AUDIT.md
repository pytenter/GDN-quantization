# 未初始化内存与 deterministic fill 安全审计

任务 `QWEN_GDN_FAST_RUNTIME_QUALIFICATION_V1`；结论 **FILL_OPT_SAFETY_UNPROVEN**。本轮只完成 CPU 源码与历史活动审计，未导入 torch/模型、未查询 GPU/NVML、未执行 forward 或任何关闭 fill 的实验。**不批准全局 `fill_uninitialized_memory=False`，没有生成或应用该补丁。**

已找到多类“先分配、后完整写入逻辑输出”的局部源码，仍未建立全部实际 kernel、所有 padding/掩码、跨 stream 读写和后续消费者的全程序证明。即使将来短程输出逐字节相同，也不能代替未初始化读取安全性证明。

## 证据身份与证明范围

前序任务 manifest 核验后的 architecture、core、Observe、matched worker、profile 计数摘要和安装头文件只读复制到本任务。实际 architecture SHA=`90d929129ffc835d2652c604925c4f3842bc6e401e174ec6f0db2285dfb8f85a`；历史执行依赖中的 cache_utils SHA=`b63732985b520a62aa4b06fc2af8ca6e506912679e0f0479b706534d041fb1bd`，其103241字节也已重新核验。

本轮用标准库读取安装包 `torch/version.py`，版本2.5.1+cu121、git_version=`a8d6afb511a69687bbb2b7e88a3cf67917e1697e`。随后只读获取[该不可变官方 PyTorch 源树](https://github.com/pytorch/pytorch/tree/a8d6afb511a69687bbb2b7e88a3cf67917e1697e)，逐文件记录 URL、SHA与bytes。它使实现分析可复核；**不证明实际已安装二进制未被补丁修改，也不证明历史动态 dispatch 覆盖了源树中全部分支。** 未找到的初始候选路径保留404历史；正确路径另行列出，没有把缺失资产标PASS。

来源清单：[fill_source_catalog.json](analysis/fill_source_catalog.json)；采集脚本：[fill_collect_sources.py](scripts/fill_collect_sources.py)；源码/计数检查：[fill_source_checks.json](analysis/fill_source_checks.json)，63项CPU检查PASS。这只是源码内容和计数检查，不是内存安全或GPU数值资格。

## 724 个新增 fill 的计数身份

复用已封存 profile 归约，不重新解析大 trace；每份均128个测量token。

|实际 FillFunctor kernel|OLD/token|NEW/token|增量/token|
|---|---:|---:|---:|
|float|24|475|451|
|BF16|0|271|271|
|long|0|2|2|
|合计|24|748|724|

新增724是实际kernel事件的对账量。CPU `aten::fill_` 从24到823，增量799；差75没有被伪造为空tensor或无发射的某个具体原因。其余新增123是96个finite组件、25个bool归约和cat/and各1，总计847。

float 的已保留父路径在NEW中为：355/token关联 `empty_strided←_to_copy←to`；24关联 `empty←zeros`；24为 `zero_←zeros`；24关联 `empty←resize_←abs`；24关联 `resize_←abs←isfinite`；24关联 `empty←flash_attention`。这些475中包含原本就存在的24次显式zeros填充，不能将全部475称为可删除保护。

BF16 的摘要明确列出的父路径为：81/token关联type_as转换；74关联to转换；48关联repeat_interleave/clone；24+24关联depthwise输出resize；8关联reshape/clone；8关联flash输出empty_like，总计267。实际BF16总数271，**其余4/token的更细父路径未在前序截断摘要中保留，保持UNRESOLVED**；不为凑数补造归属。long的2/token关联arange/resize。父链只保留ATen层次，不等于Python栈、allocation ID或动态buffer shape。

因此，**“由fill开关单独造成的精确数量”仍为UNPROVEN**。源码直接显示相应分配函数受到该开关控制，trace父链高度相容；但没有关闭该开关的匹配trace、每次分配身份和完整已编译调用栈，不能把724自动当作该开关的干预效应或可安全消除数量。更不能用它推算可恢复TPS。

## 分配、写入和读取覆盖矩阵

以下“局部全写”只针对所列源码的逻辑输出元素，前提是输入已初始化、正确的TensorIterator/尺寸/stride/dispatch和stream依赖成立；不覆盖任意storage padding或第三方二进制。完整结构化结论见 [fill_safety_matrix.json](analysis/fill_safety_matrix.json)。

|类别/实际来源|分配后写入与首次读取顺序|分支、掩码与异步风险|CPU审计结论|
|---|---|---|---|
|通用 `empty_cuda` / `empty_strided_cuda`|官方TensorFactories.cu54–60、80–86：先分配，再在deterministic&&fill同时为true时填充；关闭后只返回分配结果，没有普适的后继写入保证|strided storage可能有逻辑外孔洞；仅分配函数无法证明全部消费者不读取旧内容；越界也不由fill修复|**通用安全性未证明**，不能全局取消保护|
|`to/_to_copy/type_as` 目标|TensorConversions.cpp344–386：empty_strided/empty之后立即copy_(source)；Copy.cu173–240：同dtype连续用numel×itemsize memcpy，否则identity/conversion kernel写逻辑目标，未先读取旧目标|同设备当前stream提交；cross-device使用双向event。实际各转换shape/stride未采集，特殊dtype/别名和外部stream不全覆盖|**局部逻辑全写有源码支持**；不是全程序安全PASS|
|`clone`|TensorFactories.cpp1739–1759：dense保持stride分配，其他empty_like；随后zero_或copy_，然后才return|源必须已初始化；保存format/非稠密布局和所有消费者需要证明；copy内核vector尾部要遵守边界|**局部逻辑全写有支持**；不据此批准关闭全局fill|
|`repeat_interleave` / `reshape` 物化|Repeat.cpp101–123：unsqueeze/expand后contiguous clone再flatten；TensorShape.cpp1683–1719：view能成立则不分配，否则clone后unsafe_view|expand读源有零stride；clone创建新逻辑输出。实际view分支/物理stride未记录，unsafe_view正确性依赖前置clone|**所列物化路径局部支持**；不把view当新分配，也不证明全部布局|
|通用 `resize_` 扩容|Resize.cpp17–46只复制min(old,new)字节；49–71条件调用fill_resize；ResizeCommon.h55–71只填新增storage尾部|关闭保护时新增尾部会保留未知数据；必须证明后继kernel完整写入后所有读取才发生；不是所有resize都天然安全|**存在通用未初始化增长风险**；逐调用证明缺失|
|finite的 `abs` / unary TensorIterator|AbsKernel.cu11–47输入值→abs；Loops.cuh43–66先加载输入、计算inbounds项、再store；MemoryAccess.cuh199–230对尾部有remaining检查|输出不作为unary输入；逻辑范围能解释全写，但编译vectorized分支、完整Iterator地址/别名和实际二进制未逐次证明|**逻辑输出局部支持**，不提供所有padding/alias全程序证明|
|`arange`|RangeFactories.cu40–65以idx<N写data[idx]=f(idx)，213–270 resize后运行该kernel，必要时copy回非连续输出|N=0没有元素；有效idx全写且不读旧目标。resize helper自身有scalar empty和tail view，可能产生多次fill，不能仅凭两次事件反推每次动态分配|**有效逻辑元素局部全写支持**；实际分配对应身份未保留|
|native GDN recurrent 输出/初始state|architecture355–379：core_attn_out=zeros；循环i=0..T−1整片写output[:,:,i]；初始state不存在则zeros，否则消费previous FP32 state；更新state后才写readout，再返回|原始zeros具有明确数学初始化语义。当前T=1输出覆盖看似完整，但初始state在更新表达式中被读取，不能删state的显式zero；window外chunk也有初始化|**原生逻辑顺序可读**；不批准替换zeros或改变递推|
|conv/recurrent cache|cache_utils1014–1086：conv/recurrent用zeros/zeros_like初始化；prefill短于kernel则显式pad0，conv窗口整片copy_，recurrent更新整片copy_；architecture210–216使用conv缓存后再更新|卷积历史零padding是数学输入；cache复用与copy完成顺序必须保留；mark_static_address只在compile分支，不等于已经capture graph|**显式初始化与writeback不可省略**；不证明任意恢复/第三方cache路径|
|native depthwise卷积输出|DepthwiseConv2d.cu37–137、145–212：每linearIndex<totalElements用bias或0初始化局部acc，仅读合法input/weight，最终output[index]=value；forward_out360–445 resize后当前stream发射|边界分支跳过input乘积但不跳过该输出赋值。能支持该native前向模板的逻辑全写；cuDNN/替代实现/二进制workspace不能由此覆盖|**当前可见native输出局部支持**；所有卷积工作区安全仍未证明|
|FlashAttention输出O/LSE|flash_api.cpp437–486先empty O、LSE及可选splitKV scratch，再以同current stream调用forward；flash_fwd_kernel.h59–109有效块写0/INFINITY早返回；430–473按有效row/head掩码写O/LSE|越界tile直接return；尾部/OOB列故意不写global。必须证明所有后继消费者只读取有效区域。header中K shared tile故意不clear、scores后mask，不能仅凭masked结果有限断言无未初始化读取|**关键masked/共享内存覆盖未完成**，FILL_OPT_SAFETY_UNPROVEN|
|FlashAttention splitKV/临时量|flash_api.cpp312–321对num_splits>1分配LSE_accum和O_accum；launch_template103–158先splitKV再combine；kernel518–555空分片写0/−inf，正常块masked写入；同stream顺序可见|并非所有head/block/tail组合和combine读域都已形式核验；实际num_splits/指针/shape未保留。dropout0下seed/offset还可能empty后返回，不能推广至autograd/capture或其他消费者|**关键scratch和consumer覆盖未证明**|
|可选FLA/Triton/causal_conv实现|hub_kernels.py792–836：优先hub kernel/可用原包，失败才torch；functools.wraps可让inspect.unwrap指回torch源而实际call选其他implementation|只验证unwrap文件名不能证明编译实现不会被选择。历史trace形态与torch fallback一致但无全部可选包二进制/配置与kernel源码闭包|**可选dispatch/编译buffer覆盖UNKNOWN**；不能宣称全部自定义buffer安全|
|CUDA Graph/重用buffer|审计core、baseline_worker、matched_worker未显式使用CUDAGraph/capture/replay/torch.compile；历史归约API无cudaGraph条目，存在stream-capture状态查询只说明库检查|cache_utils的mark_static_address及flash的capture分支确实存在；没有dispatch/全部内部stream证明，不能把“本范围未观察”写成运行时完全不可能|**本测量路径未观察显式Graph**；不授予未来Graph模式安全资格|

## 必须保留的初始化与顺序

1. `zeros`/`zero_`的显式数学语义和deterministic empty保护是不同工作。源GDN初始state、卷积左侧padding、cache初始化依赖真实零值；删除这些零写入会改变数学输入。关闭fill并不授权删zero，也不自动证明一切zero之前的保护冗余可安全消除。
2. 一般to/clone/abs/arange的逻辑全写可成为**局部候选研究依据**，但要把每个allocation、后继消费者、未覆盖padding/stride、stream事件和实际编译dispatch对齐。当前trace无shape/stride/stack/allocation ID，不能建立这种完整对应。
3. 同current stream上分配→写kernel→读kernel提供源码顺序；它不是对所有自定义stream、异步allocator、capture重用或第三方算子的全局证明。本轮未执行sanitizer或GPU测试；相关资格为NOT_RUN。
4. FlashAttention中主动保留未clear的共享tile、masked store及splitKV combine读域是明确的证明工作缺口。全局开关控制的是分配tensor的初始化，不直接“修复”所有shared memory；不能将该差异混为安全证明或性能承诺。

## 结论与后续准入

可解释新增fill来自哪些分配家族，并发现部分局部逻辑输出确有完整写入路径；**没有证明新版运行中所有未初始化内存都不会被读**，也没有证明724个fill都能由单一开关安全消除。因此：

- global fill disable：**REJECTED_FOR_USE_SAFETY_UNPROVEN**。
- fill GPU实验：**NOT_RUN / NOT_AUTHORIZED**；需独立假设、预算、seal和新授权，不能混入cuBLAS资格。
- 局部冗余分配/初始化优化：**NO_PATCH_PROPOSED_OR_APPLIED**；若未来提出，须单独证明覆盖和完整数值/确定性资格。
- 输出parity、fresh重复、nonfinite门禁：本轮均非GPU证明；即使未来通过，也仍不能替代未初始化读检测/源码覆盖。
- Strict启动必须保留deterministic及fill=True、原数学和全部正确性门禁；本审计不给任何Codec、32K/81K或CUDA Graph模式自动继承Fast资格。
