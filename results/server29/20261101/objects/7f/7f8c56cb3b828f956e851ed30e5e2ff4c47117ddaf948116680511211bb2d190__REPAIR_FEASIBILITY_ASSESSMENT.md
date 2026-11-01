# 隔离修复可行性（审计提案，不是修复执行）

最终判定：`ELIGIBLE_FOR_ISOLATED_REPAIR`。

首个实际失败已定位为 frozen worker 第63行的全局 Torch deterministic-algorithms flag 断言。GPU6先失败，GPU3随后同一门禁失败；两条stream均未进入。原始STOP/log及对应f1fa提交文件均已重哈希相符。因此当前证据支持小范围 runtime 初始化/guard配套问题的隔离修复候选，不支持把失败归为预算耗尽或P1/L7科研失效。

## 预计修改范围及风险

可提议在新worker初始化、首次CUDA模型构造/forward前显式设置确定性算法模式，并在继承Capture/seed助手前后记录模式、TF32与CUBLAS配置；保留原断言，不能删门禁或把FAIL改成PASS。具体设置位置须在新代码审查中确认。旧源码无显式enable，不等于已证明所有未展开的legacy助手都不会设置/重置；原失败直接证明assert时flag为False。

这只是有明确根因的工程候选，不能保证一行修复即可通过：installed Torch源码说明开启该模式可能切换算法或对不支持的operation抛RuntimeError，可能改变性能和数值路径。若后续要求换kernel/运行库/量化语义或弱化门禁，本候选即不再是本审计建议的小范围修复，必须停止并重新评估。

| 科学项 | 修复计划允许变化？ |
| --- | --- |
| dataset、IDs、token内容 | 不允许 |
| 16-rotation roster、tensor/权重 | 不允许 |
| frozen pairs/predicted winners | 不允许 |
| 原P1、L7 Functional/State/Total数学定义 | 不允许 |
| 每token INT8-R128 QDQ/inverse/writeback语义 | 不允许 |
| 同GPU live FP配对原则 | 不允许；新代码需新同GPUreference，不混旧不合格reference |
| endpoint32768、primary/secondary窗口、方向 | 不允许 |
| deterministic初始化/flag telemetry、新code hash | 可另行授权、隔离冻结；本轮未改 |

原L7本就H128 recurrent-aware。Phase A/B的matched-window deployment损失应用并非历史512-token训练采样完整重演，排名冗余也不是gradient等价或训练优势。本轮不重新定义P1，不训练。

## 资源和预算

已发生device-time=2.089558591710197 GPUh；保守release bound=2.104673161506653 GPUh，历史cap=16。按保守界限剩余13.895326838493347 GPUh。两指定3090本次只读快照无compute apps/fuser holders，驱动580.178.04、kernel7.0.0-38-generic、Torch2.7.1+cu128匹配冻结字段。结果/默认临时/缓存目录当前容量充足，但没有未来资源预约。

微基准为0，完整四条32K+两FP成本与32Khybrid-cache容量仍未可验证预估。剩余额度是算术余量，不是正式长程资格或授权。修复不许直接续旧receipt/清latch，不许先跑部分有利样本。

## 下一阶段必须满足的条件（全部未在本轮执行）

1. 另行获得工程修复与资格执行授权；新隔离目录/新提交/新 freeze，不删除或清除旧 STOP/latch，不复用旧启动标签。
2. 只改 runtime 初始化与可观察性：显式设置并记录确定性配置，保留原 assert，不使用 warn_only 绕过，不更换 Torch/CUDA/FLA 或固定 kernel selector。
3. 验证 frozen document/token/rotation/model/selection SHA、全部配置及同一固定 GPU UUID；空闲与独占要在新启动前再次检查。
4. 两卡分别完整重做新的512-token×2 GPU qualification：同 GPU 新 FP reference/replay与 full-logit parity、真实INT8 codes/inverse/native-coordinate writeback、next-kernel consumption、storage separation、全18KDA/6MLA/conv缓存、finite/repeat。CPU PASS 不替代这些门禁。
5. 两篇都通过后才允许同引擎8448 micro；必须跨2048/8192边界继续每token QDQ，量测loading/JIT/FP+两student总耗时、峰值VRAM、native cache growth、storage与连续性。
6. 累计全部历史+新资格+micro开销，用原1.5安全系数与增长包络计算完整四student32K+两同GPU FP成本；总保守值<=16、VRAM<=85%及存储/连续性都PASS才可申请/执行正式长程。不得用旧速度或部分roster替代。
7. 继续使用已冻结RouteA/pairs/winners，primary KL(FP||student)[8192,32768)，secondary[2048,32768)同轨迹；anchor后持续QDQ，不要求原协议没有的跨GPU/driver FP bitwise identity。

## 授权边界

本轮在审计结果处停止。没有实施上述设置、没有forward/backward/optimizer.step、没有新FP/32K标签、没有安装/升级/清理；所有旧FAIL/STOP、模型/rotation、冻结CSV/JSON/hash manifest保持原样。`ELIGIBLE_FOR_ISOLATED_REPAIR`只表示可提交隔离工程方案，不表示GPU qualification已PASS，更不授权重启实验。

NO_NEW_GPU_SCIENTIFIC_RUN = TRUE
NO_NEW_32K_LABELS = TRUE
NO_TRAINING = TRUE
