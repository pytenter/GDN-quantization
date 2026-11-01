# V3科学报告：数值合同未解决，新增生成0/60

本次终态为 **V3_NUMERICAL_CONTRACT_UNRESOLVED_AND_STOP**。新增GPU派发0、C1/S3 AIME生成0/60；没有AIME准确率、完整128-token真实9B PASS或持续INT8 GPU PASS可报告。

## 实际完成

| 检查 | 实际结果 | 范围 |
|---|---|---|
| P1不可变性 | 622源、61外部依赖、17模型stat核验 | 仅只读；运行日志/raw未改 |
| Query资产 | 24层×16独立头，各4096样本 | 共享复制不计新增独立样本 |
| M0–M3 | 全384独立头通过结构及解释预算 | 使用原U/A，无重拟合 |
| 独立FP64递归 | 全384矩阵一步，3组各128步通过 | 独立稠密/秩一实现；不是9B GPU |
| 真实Torch helper CPU算术 | 128步通过局部及固定输入传播预算 | 最大componentwise残差/预算比约0.7323；不是CUDA证明 |
| 第22 token首差异 | 原SHA/bytes和指标真实复现 | 唯一上游原因UNRESOLVED |
| 完整9B功能接受范围 | UNRESOLVED | 不以4H规则或略增常数替代 |

M0最大相对对角偏差9.551248320045331e-15；M1（FP64构造后单次FP32转换）1.6890838568648173e-7；M2（FP32构造）2.7452575492717835e-6；M3（FP32检查器）2.1366414188318343e-6。A32最大谱正交残差7.680727240586925e-8，最大部署矩阵表示谱偏差5.207700416169671e-8。所有路径统一计算，没有按旧门槛挑选路径。

## 首差异说明

旧停止在零基token22（第23个输入token）、layer14、有效Key张量。最大位置为recurrent head18、Key坐标78，真实共享Query/Key head9。Native为0.56640625，UH回投后为0.5898437745114653。实际MaxAbs为0.023437524511465302，Relative L2为1.3733534416246003%；原门限为0.023437473855633995，只超5.065583130736684e-8。

但实际偏差不是5e-8：该位置与相隔6个BF16步长一致。44个元素误差超过该数值范围一个BF16步长；最大元素贡献约9.1%的总平方误差。小的回投舍入量可以解释末尾若干小数，不能唯一解释完整模型传播。原保存没有此前完整误差轨迹、原始Q/K/gate/value、对应Native/H state或full-vocab logits。因此不能声称找到了唯一根因，也不能声称不存在早期功能漂移。

## 为什么没有继续GPU

P1仍合法占用三张授权3090，旧INT8占用GPU0，V3没有抢占或迁移。更关键的是：固定同输入递归的数值界不等于完整9B网络功能界；后者的输入、BF16舍入与非线性传播尚无可信预登记接受范围。用户明确要求无法建立时一次性fail-closed，本次据此停止，而不是等资源释放后拿目标128-token结果反向定门限。

完整128-token9B、512canary、持续QDQ GPU资格、60条正式生成、FrozenV4评分和四配置交互全部NOT_RUN。计划manifest可以核验60条同题同UUID安排，但不可执行。本次不生成空CSV/JSONL，不把历史29题写成30题，不读取P1未封存的答案做推断。

## 历史和运行任务

V1保持QUERY_EIGENBASIS_QUALIFICATION_FAIL_CLOSED_STOP；旧V2共享RAM及失败处理错误保持原STOP0/90；当前V2的C1/S3真实FP门禁停止也不重判。P1继续其独立30条；旧INT8继续其独立120条。本次不覆盖源码/STARTED/回执、不改共享模块/动态库、不删除任何运行任务跟进，也不自动创建更多Recovery项目。

两个CPU审计工具错误（函数dispatch装饰器未绑定、局部变量遮蔽JSON读取函数）连同原脚本和traceback保留。矩阵及递归通过结果未被覆盖；独立CPU工具完成未执行的首差异分析并复核全部384头、128步CPU回执及保存Tensor。没有GPU重试或调整数学门限。

数学推导、数值范围和证据缺口见FP32_ROUNDING_AND_RECURRENCE_ERROR_BUDGET.md；原始Tensor、Cq及U/A保存在55/29，Mac仅交付小报告/源码/JSON与remote-only清单。最终逐文件SHA/bytes、原子无覆盖封存及独立本地核验见FINAL_ARCHIVE_VERIFICATION.json、FINAL_SEAL_RECEIPT.json和operations/LOCAL_DELIVERY_VERIFICATION.json。
