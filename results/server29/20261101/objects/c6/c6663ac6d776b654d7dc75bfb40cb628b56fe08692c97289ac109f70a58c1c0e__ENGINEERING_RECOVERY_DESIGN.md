# Ling/KDA execution_02 工程恢复设计

本设计在修改任何旧控制器或 worker 前形成。execution_01 全部只读，当前 manifest SHA 为 202bf8f64d7a2bfeab93451a607d575f7cc7052ab96692d88de02176f536516b；服务器120条封存SHA校验PASS。新执行采用 execution_02 与独立 worktree，科学预注册、输入、旋转与公式不变。

## A. 实际审计及根因边界

1. admission 入口是 prospective_pipeline.run → v3_guard.device；short/functional/formal worker 另有同类启动检查，均硬编码 GPU6。nvidia-smi 基础状态和 compute-apps 查询分别进行；fuser 是 FD 诊断，不能充当计算授权。UUID固定检查防止错设备。
2. 旧 evaluate 在没有 worker binding 时要求显存≤64MiB、GPU util严格为0；任意单次非0即 UNEXPLAINED_TARGET_MEMORY_OR_UTILIZATION。第三篇 natural exit epoch 1791913141.0905108；失败采样1791913141.1228518，相差0.032341秒（STOP在1791913141.4239182）。日志确实有100%/15MiB/空compute-apps，不能推断他人占用、硬件故障或NVML误报。此前报告所说0.33秒是退出到STOP，而非退出到采样；这里保留精确时间。
3. 本次初始60秒观察7张真实3090，每张13个样本。GPU6全部13次SMI=15MiB/0%/P8，直接NVML亦0%，没有compute-apps；GPU0/1忙且有进程，GPU2窗口末段发生临时显存使用，不能用初始快照作长期空闲证明。GPU3/4/5当前空闲不意味着数值兼容。
4. 直接NVML v1 used包含系统预留；本机v2实测 total=25769803776，reserved=473759744，free=25280643072，used=15400960 bytes，v1 used=reserved+v2 used。因此v1约466.5MiB与SMI约15MiB并非同一字段语义。分类应比较v2 allocated与SMI，保留reserved/free，不能把差额默认为可忽略。ABI与语义来源：https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html 和 https://docs.nvidia.com/deploy/nvml-api/api/structnvmlMemory__v2__t.html 。
5. 自身PID namespace可读；/proc/1/ns/pid无权限，容器/host一致性UNKNOWN。dmesg拒绝权限；journalctl提示不可见system日志，空日志不是没有Xid。历史GPU6异常根因仍UNKNOWN，仅确认本轮连续观察资源恢复。诊断程序NVML打开的FD不等于模型计算PID，必须明确归属自身诊断身份。
6. 旧select从空列表开始且run拒绝已有log，无法幂等恢复48分数。publish先写固定.partial再hard-link，遗留partial会阻塞；需要在单写者锁下用唯一临时文件、fsync文件及目录和no-clobber提交。旧checkpoint保存8192且仅在parent完成manifest绑定后允许fresh restore；不能把未绑定的.pt当作恢复资格。正式worker从自己已资格的8192恢复，没有任意中途checkpoint机制。
7. 历史跨GPU/fresh process不一致记录保留；不能凭同型号替代。当前GPU6同UUID/runtime，无需改数学引擎；所有12条件仍各自执行parent0→8256、8192 checkpoint、fresh64/observer exact资格。

## B. 集中修改模块与不变量

* gpu_telemetry.py：只读SMI+直接NVML v2+process+power/clock/pstate/temperature/权限证据。未知保留UNKNOWN。NVML shutdown释放诊断FD。无CUDA context。
* gpu_admission.py：统一分类与准入；每个worker前至多60秒/5秒观察，最后至少3个连续有效idle样本才可AVAILABLE_VERIFIED。早期瞬态完整保留，持续100%无进程必BUSY_UNATTRIBUTED；不删除util检查、不提高64MiB旧idle门限。v2字段不支持或API错误不能宣告可用。UUID而非历史index选设备。容量24GB和原driver合同不变。
* 所有worker的pre-context资源入口统一到gpu_admission.device；bound context继续用原v3_guard exact PID/starttime/UID/namespace所有权检查，其数学、QDQ及cache判断原样保留。启动使用有SHA、UUID、完成时间和execution identity的短期admission receipt，再实时检查防止观察后被占用。
* 优先GPU6同UUID；其他设备物理PASS仍须独立冻结数值兼容资格PASS。没有资格receipt的替代设备不得使用。该门禁及失败路径接受synthetic测试；本轮只要GPU6合格就不产生额外替代设备科学计算。
* recovery_state.py：稳定unit identity、no-clobber原子提交、flock排他锁、completed SHA校验、状态事件及checkpoint admissibility。controller重启必须复用成功单元并校验SHA；数据损坏不重算掩盖。
* recovery_controller.py：复用旧select/make_long_spec/qualify的科学规则，恢复原3篇decision和原result/receipt，从cursor=3继续；四篇共同freeze前无长程。幂等生成freeze/spec/gates，已存在且合法时复用；若冲突STOP。
* unit_runner.py：独立薄进程记录模型worker launch-to-exit和返回码，即使controller中断仍完成receipt；controller重启只接管同execution已验证runner，不二次启动。worker真实失败保留首STOP，不自动放宽或重试数值失败。未知死亡或未绑定checkpoint即EXECUTION_RECOVERY_FAIL_STOP。
* finalize_cpu.py：作为controller最后的CPU阶段，不另起接力科学shell。复用既有逐点sum/384、FP payload/checkpoint/hash验证；正确处理部分STOP、原3篇import来源与本轮新增资源时间。phase文档/测试报告/源码diff一并封存，原始science assets不修改。

## C. 状态机与恢复路径

AUDITED → GPU_QUALIFIED（资源已验证，同设备历史合同；新12条件numerical仍pending）→ SHORT_SELECTION_RESUMED → FOUR_DOCS_ELIGIBLE → SELECTION_FROZEN → LONG_HORIZON_QUALIFIED → FULLMODEL_RUNNING → RESULTS_VERIFIED → DELIVERY_COMPLETE → COMPLETE_AND_STOP。

恢复48条分数使用execution_01 manifest验证字节拷贝或只读引用，记录原source commit/scorer/input/rotation与原执行身份，不能伪称新代码产生旧分数。前三篇不得GPU重跑。unit ID包括文档、rotation、模型/tokenizer、输入、quantizer/scoring合同和原source身份；物理attempt另有execution_02身份。

Controller中断后成功单元SHA通过则skip，live runner PID/starttime一致则接管；坏SHA或未知PID不重启掩盖。fresh64和正式轨迹仅从自身绑定8192 checkpoint恢复；已成功正式结果不得重跑。任何未完成模型失败均依原恢复合同判断，未经资格的任意中途状态不可恢复，保存NOT_RUN并STOP。现有8192接口支持资格与正式路径，不新增未经验证的中途cache schema。

## D. 一次性测试与数值风险

资源fixture覆盖idle、visible busy、unattributed100%、接口不一致、查询失败、容量不足、idle→busy、UUID/index变化、无合格GPU、替代设备数值FAIL以及瞬态→连续idle。有界60秒，不无限重试。系统保留现有模型engine、full_kl、original P1/L7及checkpoint/observer helper SHA/AST；只变控制与资源入口。进口48分数逐项对原manifest、排名及pair helper复算CPU验证，不重新评分。

状态机fixture覆盖48分数恢复、第四篇中断、freeze后重启、FP完成后重启、成功unit重复进入、8192 checkpoint合法/坏SHA/错rotation、原子提交中断、重复controller锁、失败不补零、384 positions和实际sum/384、manifest独立验证。真实GPU资格只执行原冻结的12条件parent/fresh64及必要边界；不为测试重跑32K。

科学门禁失败与实际资源不明分别保留first failure并映射终态。旧MIXED与posthoc身份不变；不训练、不启动额外scientific units、不改变阈值或科学公式。
