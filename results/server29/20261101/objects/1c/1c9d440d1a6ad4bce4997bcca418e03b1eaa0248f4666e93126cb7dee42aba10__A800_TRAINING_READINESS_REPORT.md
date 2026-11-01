# A800租用前资格结论

任务：QWEN_GDN_DUALM_AWARE_ROTATION_TRAINING_READINESS_V1。

最终结论：**FAIL_CLOSED_STOP；当前不应支付A800租金。** 数据身份资格未完成，本地GPU资格未运行，完整9B新训练入口未获运行资格，8小时配对训练也没有成本保证。本轮只做准备与CPU检查，没有租卡、正式旋转训练、32K或AIME执行。

| 门禁 | 实际状态 | 证据和边界 |
|---|---|---|
| TRAINING_PROVENANCE_GATE | PASS | 原已完成A800 C6R源码、训练记录、配置、冻结导出和17个模型/tokenizer文件内容身份恢复 |
| CPU_CODEC_QUALIFIED | PASS | 冻结Dual-M与训练包装在已测真实shape及边界fixtures的codes、两组FP16 scales、重构字节相同；不是输入空间穷举 |
| TRAINING_GRADIENT_QUALIFIED | CPU_LOCAL_OPERATOR_ONLY | 206项CPU检查，两臂24层参数均有有限非零梯度；单层单head H64三步及合法边界恢复通过 |
| DATA_ISOLATION_QUALIFIED | FAIL_CLOSED_STOP / NOT_QUALIFIED | 第二遍raw/fresh-token复核在CRLF文本身份检查停止，不能将partial64/16/2作为冻结数据集 |
| LOCAL_MICRO_TRAINING_QUALIFIED | NOT_RUN | 离线前置条件未全满足，没有GPU dispatch；新增0 GPUh |
| FULL_MODEL_A800_TRAINING | NOT_TESTED | 本地CPU局部计算图及源码闭包不替代真实9B forward/backward |
| A800_8H_COST_GATE | NOT_QUALIFIED | 历史数值和明确假设构成的场景约10.24h；未来Dual-M实际耗时、价格UNKNOWN |

## 数据STOP及失败保留

首轮数据收集器把旧8doc清单中的未选候选当成已使用，误将6篇合法长候选排除；保留首失败后，仅进行一次有界CPU分类修正，未改变seed、排名、长度阈值及真实已用集合。修正后27篇长候选中21篇有历史使用/选定证据，6篇仍符合候选筛选。

该次修正在物化64 TRAIN、16 DEV及2 FUTURE候选后，于第二遍身份复核真实失败。只读诊断确认82份raw原始字节与预期SHA相同，84份token文件的文件/内容hash相同；9篇TRAIN及1篇DEV含CRLF，`Path.read_text()`的universal-newline行为使复核文本变成LF，触发SHA不一致。它是本轮I/O验证程序缺陷，不是物理文件损坏，也不是Dual-M数学失败。全体第二遍fresh-token核验未完成，不能只凭已落盘hash给出数据PASS。

两次失败、原规则、原脚本、分类修正源、59来源分类表和只读诊断均保留。本轮不继续修补重试，不跳过门禁进入GPU。`PAIRED_TRAINING_PREREGISTRATION.json`明确frozen=false，保存可审查的科学计划但不冒充已准入的可执行训练包。

## 数学与梯度资格范围

原Loss是24层GDN完整out_proj输出的relative MSE，包含readout→RMSNorm（先norm后gate）→out_proj；辅助项为恢复原坐标的previous recurrent state relative MSE，总目标Functional+0.1 State。复用原Cayley、原H/Delta坐标与原helper AST，不增加loss或改recurrence。

Arm A保留原C128 STE。Arm D硬前向直接调用冻结Dual-M encode/decode，反向采用整个codec的identity STE；row/column估计、LS、FP16、round、clamp全部detach，饱和区也传identity cotangent。这是明确的surrogate，不是真实hard rounding导数。

Arm A的legacy与hard C128不是通用逐位等价：真实full32head fixture数值不等0处，但有551个signed-zero字节差异。A包装保留历史legacy字节/梯度，不改其算术表达式。单一新backend仅CPU导入/闭包检查；完整9B phase orchestration仍未资格，不能将源码复用或默认dry-run当作完整可运行训练包PASS。

CPU实际fixture来自历史Doc5的C6R+Dual-M存储坐标；诊断q/k/v/gate、norm及projection权重由固定seed生成。它们不是原生clean FP teacher或真实9B功能轨迹。CPU证明原helper的局部图和rotation/Adam/Python及Torch CPU RNG的有限步恢复；本轮CUDA未初始化，CUDA RNG恢复仍未测试。它不证明完整训练稳定、GPU确定性或32K改善。微测产物不是正式R_A/R_D。

## 资源与未来付费准入要求

历史原9B H64最小梯度图peak allocated为31,035,885,568 bytes（约28.90GiB），高于单张3090的24GiB；不通过改dtype/Loss/递归规则或故意OOM来声称本地全模型可行。代表性单层GPU脚本只完成离线编译/默认拒绝检查，未运行。

历史125次train_document合计3.4348266小时/臂。假设新A/D每臂同耗时，另计setup、admission、一次两臂最终验证、teacher准备、export和25%余量的场景约10.2426小时。这是预算场景，不是新A800预测；Dual-M可能更慢，实例实际价格UNKNOWN。金额公式为实际付费小时×实际小时单价。8h只是未授权提案。

未来即使另行获授权，也须先完成数据资格和完整启动依赖，随后在真实A800核验GPU型号/UUID、80GB级实际可用显存、驱动/源码/环境、全模型两臂短forward/backward、梯度覆盖、QDQ消费、数值/确定性、完整checkpoint恢复及真实step耗时。只有保守预计两臂同updates、验证、导出及余量都能容纳于授权预算，才可进入正式训练；不能先完成一臂后放弃另一臂却声称配对实验成立。

启动入口默认dry-run，当前execute必须拒绝。完成后释放实例需人工在提供商控制台停止/释放compute，确认计费停止及独立存储费用。这里没有租卡API或后续自动执行。

本轮关闭：新增本地GPU 0 GPUh，A800 0小时/0租用动作；不自动启动修复、租卡、训练或评价。
