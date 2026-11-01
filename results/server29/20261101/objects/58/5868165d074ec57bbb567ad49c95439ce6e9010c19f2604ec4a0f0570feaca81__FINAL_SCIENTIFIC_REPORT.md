# QWEN_GDN_SHORT_HORIZON_SCALE_INFLATION_SURROGATE_VALIDATION_V1

## 结论

P0 = AUDIT_BLOCKED_IDENTITY；P1 = NOT_STARTED；PRIMARY_STATUS = ENGINEERING_OR_BUDGET_STOP。
这是输入资产/预算阻断，不是 S1/S2 的科学否定。旧实验 COMPLETE/PASS/FAIL 身份不变。

## 1. P0 审计

冻结 quantizer、model 文件、原始 helper 和文档身份重新核验：FAIL。Direction POSTFINAL 原哈希及全部 269 个条目重新核验：PASS。
独立文档4篇；旧32/32是四文档×四条件×两个干预，不是32篇独立文档。

## 2. 历史资产可复用性

原始 Parent 连续 G 和 FP 状态端点可复用；原始 G 是无量纲 J_ACC_FD(32768)/J_ACC_FD(8192)。J采用每层每token物理空间相对误差，再按24层、1024..T等权均值，FP64累计。G值有限，未对记录的G截断；是否发生内部动态饱和不能仅由G有限推出。
正式 drive 及学生8192锚点张量已按旧release释放；仅保存SHA/资格记录不是保存张量。工程2048输入仅覆盖Doc1早期，不能替代8192后的窗。
时间分辨 signed scale/onset 来自 NO_SCALE_INFLATION/NO_SCALE_DEFLATION 干预轨迹，不得充当自然 Parent 短期信号。原始 Parent只有2048/8192/32768累计观测，不能恢复精确首次onset。

## 3. 实际执行

新增正式科学单元0；GPU资格测试0；没有模型forward、replay或量化观测执行。所有科学CSV标记NOT_PRODUCED_UNEXECUTED，没有生成空壳CSV。

## 4. S1/S2 H32/H64

全部NOT_MEASURED。用户给出的S1/S2/D1/D2定义记录为候选；epsilon/aggregation尚未冻结，不把方案当作测量。S2必须用同一student pre-QDQ u，观察分支不得写回；本次未实现/验证该hook。

## 5. 两个Track

TrackA：NOT_RUN_INPUTS_RELEASED。8192状态是已消费1-based8192 token后的状态；0-based[8192,8224)/[8192,8256)对应1-based8193..8224/8256。原G numerator与短窗重叠；没有8256累计边界，不能精确剔除重叠，亦不能声称pre-onset预测。
TrackB：NOT_RUN_PREREQUISITES_NOT_QUALIFIED。原helper确认1024tokens、8 captures、H64 TBPTT和every-token QDQ/writeback；不做8192 warmup。WikiText训练teacher不能错配RepoBench长期G；本次没有改变文档或capture规则。

## 6. H/C6R/C7区分

已有长期G的从小到大顺序：Doc1 H<C6R<C7；Doc2 H<C6R<C7；Doc3 C6R<C7<H；Doc4 H<C7<C6R。它说明旧连续label有排序信息，不说明surrogate能预测排序。Native仅作negativecontrol；未计算任何surrogate Spearman或显著性。

## 7. 原C5/C6比较

原始helper可定位并通过SHA核验。B0是每head矩阵相对误差均值，分母clamp_min(1e-12)，再层均值；不是旧Parent按整层Frobenius归一化的J。B1是实际out_proj output与teacher output的global relative-MSE，再层均值；B2=B1+0.1B0。只用drive内q/k/v/g/beta或out_proj权重不能冒充原functional loss。原matched training确实有recurrent feedback。未计算B0/B1/B2与S1的matched comparison。

## 8. 预测分歧

NOT_EVALUATED。既不能报有增量，也不能报NO_ELIGIBLE_INCREMENTAL_DISCORDANCE；没有短程分数。

## 9. 成本与资源

新增GPUh=0；4GPUh余额=4。当前/data可用49643806720bytes（46.23GiB）；4张3090状态见runtime记录。磁盘不是主要阻断。按四篇历史capture的实测速率，重建到8256tokens的capture-only估算4.1679GPUh，加25%余量5.2099GPUh；尚未计入学生anchor重建、qualification、TrackB和functional baseline。四卡并行可缩短墙钟时间，但不降低总GPUh。该估算不是新microbenchmark或精确下界。

## 10. 后续与授权边界

没有证据支持直接启动独立full-model KL pilot；本次没有执行该pilot，也没有C8/训练/AIME/自由生成。恢复需先获得身份可核验的自然Parent短窗drive、学生锚点及原functional pairing，或另行授权一个预算与原生产capture确定性闭环均合格的最小输入重建方案。当前旧capture使用deterministic_enforcement=False、replay使用True；不能静默改cumsum或关闭本轮determinism来恢复输入。

不能把本次工程停止解释为NO_STABLE_ROTATION_DISCRIMINATION、HISTORY_CONDITIONED_ONLY_NOT_TRAIN_READY或LONG_HORIZON_LABEL_INVALID_FOR_RANKING。
