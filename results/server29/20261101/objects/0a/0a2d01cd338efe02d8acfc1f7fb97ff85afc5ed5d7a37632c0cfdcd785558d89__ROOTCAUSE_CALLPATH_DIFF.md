# 新旧实际 FP Runtime 调用路径审计（CPU-only）

本分析只解析已冻结源码、历史执行回执及已安装头文件，不导入模型/torch，不运行 CUDA。状态为 `STATIC_CALLPATH_AUDIT_COMPLETE`；具体退化根因仍需综合时间线，不能以源码差异宣称墙钟因果。

## 身份与边界

历史实际测量入口为 `source/matched_worker.py`，SHA `08b62839e88fa6bb49f10388a19e152e04beb0ab36b347ffcfdf2d9259e2ff8c`，历史 source manifest 对其以及外部依赖作绑定。入口第58–69行按 OLD/NEW 路由导入源，第59–60行另对实际 OLD runner 内容作 SHA 验证。历史 delivery 的内容均按该 manifest 重新计算 SHA 后复制至本任务 evidence；完整明细为 [SOURCE_IDENTITY_AUDIT.json](SOURCE_IDENTITY_AUDIT.json)。

|角色|SHA256|实际绑定方式|
|---|---|---|
|OLD runner `run_qwen_aime26_formal.py`|`9ff56d00b6b2dbaf9be3c3186bd30be8d3b9cc9d9b580a680316bcc8c01247dd`|matched worker 中实际导入路径 SHA 检查；manifest 中另一历史副本相同 SHA|
|NEW `baseline_worker.py`|`0961d96bdafa92d6c3bca7d1980b74e9d70fe3aabff404417878ced20c7f1e41`|matched worker 导入其 reference；历史 manifest 固定|
|NEW core|`aa338b8b83cde62b97c94fd205d1c62999f783e7ec3242cbb813dc877e01f575`|reference binding 固定，原 core.configure/load|
|NEW Observe 来源 worker|`e04ec85092ee4f2207699366b2aaa61e1e71038e728c9234477b5b2c21794773`|baseline worker 只提取原 Observe/dual AST|
|实际模型 architecture|`90d929129ffc835d2652c604925c4f3842bc6e401e174ec6f0db2285dfb8f85a`|两个 profile ENV 及 matched worker 检查同一个实际模块|
|模型 config|`d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05`|历史 manifest 固定模型 config；32层 `(L,L,L,A)*8`|
|NEW 补偿 chunk recovery|`b13152f81ecd87428936dc5eced5a2a9f1e06ff992e8ba5bbb81703206dec26c`|原 AST 提取，并且只替换 chunk 的一个 cumsum 调用|
|NEW formal H/C128 runner|`77b1e1f6dad8a0dd53a27cf5a47c02be86c26d2cf30732604ba6d3747e178f69`|原 AST 提取；本次 FP 不进入 H/QDQ 路径|

这里的“旧版”是历史 matched OLD 复现执行，不将本轮可见环境倒推为更早原 AIME 执行环境。安装 PyTorch 头文件在本轮只读采集，具有本轮 SHA，而非当时预冻结依赖；其作用是解释安装包机制，不补造历史动态标志。

## 调用路径差异表

简称 M=matched_worker；O=旧 runner；B=baseline_worker；K=NEW core；W=原 worker/Observe；A=原 architecture；N=numerical_recovery。行号均为复制的原始文本行号。

|模块/调用点|旧版实现|新版实现|每 token 调用数差异（测量窗口）|潜在机制与限定|代码位置|
|---|---|---|---|---|---|
|加载|原 loader，BF16，device_map={"":0}，eval|同血缘 loader；随后 requires_grad=False、配置检查|加载不在128-token测量窗口|不是每步加载；无 offload 证据|O170–187；K54–62；M58–70|
|forward 入口|model(input_ids=GPU切片,cache,use_cache=True)|相同公共入口|均1/token；采样均不运行|公共 Python 元数据/计数 wrapper 存在于两臂|M80–112|
|decoder 与矩阵乘|32层原 native architecture|同一实际 architecture|均32层及249 aten::mm/token|静态形状相同；trace未记录实际shape/stride；不能把CPU self等同GPU GEMM时间|A446–453,543,668–701,716–718,1785|
|GDN recurrent|缓存后 seq_len=1→原 recurrent helper|同一 helper 外增加 Observe wrapper|均24核心/token|原 helper 本身无矩阵乘；FP32逐元素递推/归约相同|A330–380,505–535；W65–76|
|chunk/prefill|原 chunk cumsum|只替换 chunk cumsum 为 compensated_scan|测量窗口两臂均0；整个256输入都只在t0调用24次|补偿扫描不能直接解释稳态t128–255耗时；可能影响数值历史，但不是窗口内重复算子|N327–336；K61；M130–144|
|GDN state finite|没有新增逐层state finite|每层 `torch.isfinite(s).all()`，保存GPU bool|新增24 isfinite、24 all/token|会发射GPU工作；此处没有 `.item()`、没有CUDA Tensor Python条件；不是24次主机同步|W74|
|末尾 finite|logits isfinite→all→item|24 state bool stack→all；logits isfinite→all；bitwise_and→K.check.item|各1 Tensor item/token；NEW另有stack1、all1、and1|唯一末尾标量读取等待结果；不能声称新增24次sync|M112–114；K24–26|
|Observe关闭后的Python工作|无Observe|24 prehook更新 active，层集合/Counter、dtype/shape检查|新增24 wrap及各24 metadata检查/token|这些表达式消费Python元数据；FP previous={}，没有data_ptr或字节hash检查|W56–76；M119–120|
|hash/CPU读tensor|公共小RoPE位置读在计时后|相同；Observe(on=False) q/k/state hash关闭|计时内新增大Tensor hash=0|不能把tsha中的.cpu()按函数存在就计入每token|W72,75；M133–134|
|通用 dtype/layout|原native `.float/to/type_as`、contiguous/clone/cat|相同数学源码，但deterministic设置不同|trace新增fill关联多个empty/转换父路径|相同Python源码可有不同后台初始化行为；不能将全部fill归于finite|A各转换位点；K45–52|
|同步|末尾finite item；窗口边界synchronize|同上|profile显示各约1 stream synchronize/token；无24次增长|边界显式同步只在t128前和窗口末；profile的item等待与这些边界分别记录|M101,113–114,127|
|日志/telemetry|每128输入公共PROGRESS；结果结束写|相同公共频率|无额外逐token日志；测量窗口内仅t255公共进度打印|公共日志及层计数属于测量协议；不能据此将数十ms归于NEW|M126,153；controller为进程外|
|H/C128（次要）|每次构建Hadamard；第一prefill不QDQ|Hadamard memoize；每forward均QDQ并检查下一次消费指针|measured decode两臂均C128；prefill契约不同|H臂不是数值匹配对照；当前FP主差异不能归于Dual-M|O76–167；K31–41；M115–119|

## 249 个 GEMM 的静态重建

实际配置 hidden=4096、MLP=12288；GDN key=16×128=2048、value=32×128=4096；全注意力16×256=4096、KV=4×256=1024；词表248320。全套249个 ordinal 的模板与预期形状存于 [SOURCE_CALLPATH_INVENTORY.json](SOURCE_CALLPATH_INVENTORY.json)，由 [source_callpath_inventory.py](../scripts/source_callpath_inventory.py) 只用 AST/JSON 构建。

|层类型|按源码执行的 nn.Linear 顺序|推断 mm M×N×K（单token）|每层次数|
|---|---|---|---:|
|GDN（24层）|in_proj_qkv；in_proj_z；in_proj_b；in_proj_a；out_proj；MLP gate/up/down|1×8192×4096；1×4096×4096；两次1×32×4096；1×4096×4096；两次1×12288×4096；1×4096×12288|8|
|full_attention（8层）|q_proj；k_proj；v_proj；o_proj；MLP gate/up/down|1×8192×4096；两次1×1024×4096；1×4096×4096；两次1×12288×4096；1×4096×12288|7|
|lm_head|lm_head|1×248320×4096|1|

`24×8+8×7+1=249` 与实际 profile 数量吻合。参数实际BF16已在ENV保留；native norm与block输出回到输入dtype，因此上表线性输入/输出预期BF16。GDN递推state实际FP32 [1,32,128,128]；q/k重复至32头后，helper将q/k/v/beta/g转FP32；输出回到初始query dtype，再经norm返回native数据流。**这是静态调用顺序重建，不是 trace 直接记录了这些模块、形状、dtype或stride。** profile 配置 `record_shapes=False,with_stack=False`；实际 per-call contiguous、leading dimension、CUBLAS内部选择不可由该表证明。

## finite、fill 与新增 847 个 kernel

与本任务 CPU trace 解析交叉核对后，源码支持的计数如下：新增24state finite各产生 abs、ne、eq、bool mul 四类工作，各24/token；24state all加stack后的all共25个新增bool reduction；stack带来1个bool cat，组合state/logits检查带来1个bool and。合计123；其余724为新增float/BF16/long fill（451+271+2）。详细核对由 [trace_kernel_delta.json](trace_kernel_delta.json) 保留。

安装包 `torch/include/ATen/native/TensorFactories.h` 101–115明确说明 empty* 在确定性算法开启时填充浮点NaN或整数最大值；其实现调用tensor.fill_。安装 `Context.h`390保留 `_deterministic_fill_uninitialized_memory=true` 默认值；`utils/deterministic.py`接口说明该开关配合 use_deterministic_algorithms。原NEW未修改fill开关。trace关联fill父路径包括：to/_to_copy/empty_strided、repeat_interleave→clone、reshape→clone、abs的resize、conv depthwise workspace/output、flash_attention empty、arange resize，并非只在GDN外围检查中。

这支持“确定性内存初始化是新增fill的来源机制”方向，但有两条限制：历史没有单独记录fill开关动态值/所执行的完整C++栈；已有detOFF裸计时仅7.996658 TPS，未恢复13.742036 TPS，不能证明724fill解释整个退化。CPU `aten::fill_`计数823−24=799，实际新增fill kernel724；差75是操作事件与实际发射不一一对应，未靠源码猜成具体空tensor/无GPU调用原因。

本轮采集安装头文件 SHA：Context.h `de052023d7f01f0659921f727adc8f157dd040a2082598a587c3469a83996db3`；TensorFactories.h `860f2884ca1515cb762e84bdec0fd429e3d7610c21e2e0bf470c582230f93ee4`。它们为已安装包主源证据，不是同名网络文档的泛化解释。

## mm 异常：源码能够排除和不能确认的部分

native recurrent helper330–380只有逐元素、sum、拷贝等，且新旧使用同一源码。新增finite出现在helper之后，不会改变线性矩阵形状；249个mm数量不变。trace agent恢复的mm前发射间隙在所有249ordinal普遍扩大，而不是集中某个GDN投影。这反对把问题直接归给某一个量化器或某一类投影，但并未排除后台状态对所有GEMM的系统影响。

CPU self内仍可能包括未单独标记的C++/库逻辑、分配、检查、runtime等待或调度停顿；没有C++/Python stack、CPU调度trace和cuBLAS内部trace，因此不能把空段命名成已证实的函数。CUDA API可见时间稳定、关联GPU kernel时间稳定，都不意味着该空段已经证明是纯Python或cuBLAS workspace。

原NEW在导入core时设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`；detOFF对照是模型加载以后只调用 use_deterministic_algorithms(False)，该环境变量、OMP/MKL及其他加载时选择保留。安装Context.h326–330明确接受CUDA>=10.2确定性所需的 `:16:8` 或 `:4096:8` 两种配置。因此合法配置间差异可以作为未来单变量可证伪假设；源码不保证它们的性能或数值逐字节相同，不能先应用或宣称优化成立。

## 结论等级

- 调用路径、检查消费方式与新增工作来源：`STATIC_CALLPATH_AUDIT_COMPLETE`。
- “24state finite导致24个新增主机同步”：源码和trace不支持。
- “补偿cumsum支配测量窗口”：窗口内零chunk调用，不支持。
- “新增fill/检查导致主要裸wall下降”：`DIRECTIONALLY_SUPPORTED_ONLY`，没有满足重复干预与语义控制的因果闭环。
- “mm前发射空段具体来自何处”：`UNRESOLVED`；只能据此准备最小后续门禁，不能修改冻结源码或去掉正确性/确定性要求。
