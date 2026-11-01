# 未来 32K 评价协议（本轮不执行）

Task: QWEN_GDN_DUALM_AWARE_ROTATION_TRAINING_READINESS_V1

本文件只冻结未来评价定义，不授权训练或评价。最终文档身份须引用 DATA_SPLIT_MANIFEST.json；身份/泄漏门禁未通过时不允许启动。

- 模型为原冻结 Qwen3.5-9B，权重/tokenizer/source 内容身份与正式 persistent QDQ 合同一致。
- 至少两个全新、未观察模型结果的 32769-token 文档，输入前 32768 token，按冻结选样规则选择；训练和开发验证不得包含这些 raw/token 身份。项目身份无法证明时 PROJECT_INDEPENDENCE=UNKNOWN。
- 单张独立 RTX3090；同文档所有条件必须同物理 GPU UUID、同 runtime/dtype/数值配置。未来真实 GPU 资格仍须另行执行。
- 条件为 FP_NATIVE、H+DUAL_M、冻结 C6R+DUAL_M、R_A+DUAL_M、R_D+DUAL_M。原 H/C6R 只读加载；R_A/R_D 均采用未来配对训练的同一预冻结最终 update125 导出，不以测试结果选择。
- 每个 student 必须从 token0 独立建立自己的 persistent QDQ 历史，不共享 FP/其他 codec/其他旋转前缀。严格 writeback→next token consumption。
- t=8192+64*k，k=0..383。消费 ids[t] 后预测 ids[t+1]。每位置完整248320词表 FP32 KL(FP_NATIVE||student)，不使用 top-k/NLL/MSE 代替。
- 逐 doc K=math.fsum(384个原FP32 KL)/384；沿用原 reducer 与独立 CPU FP64 算术审计。实际保存 logits 决定各比较算术分辨范围，不复制历史固定阈值；该范围不控制完整模型前向误差。
- 唯一未来 Primary 建议并固定为 G_specific=K(R_A,DualM)-K(R_D,DualM)。Secondary：G_adapt=K(C6R,DualM)-K(R_D,DualM)；关键性能目标 G_H=K(H,DualM)-K(R_D,DualM)。正值表示 R_D 更好。
- 报告每 doc 三项原始差、相对收益、算术分辨率/方向、mean/median与完整 Early/Middle/Late；统计单位 doc，固定 N=2。未分辨或缺失如实报告，不替换样本、不以位置/head伪增N。
- N=2仅探索性，不能宣称统计显著或广泛泛化。项目独立性和完整32K数值不确定性限制保留。
- 首次资源/数学/身份失败保首失败并停止后续 GPU；不自动重试、扩样、训练、AIME或git push。

本轮所有未来 G_specific/G_adapt/G_H 数值均 NOT_RUN，不创建科学结果 CSV。

当前数据清单为 FAIL_CLOSED_STOP/NOT_QUALIFIED，已有两篇长文仅是物化候选，不能称最终测试集已经冻结。拟议新配对训练数据由历史 WikiText 改为 RepoBench；若未来另行完成数据资格，A/D必须共同使用相同数据与顺序。G_specific的配对设计只改变codec；G_adapt相对旧C6R还混有数据来源、最终125步对历史验证选择94步及硬件/runtime差别，不能单独解释为Dual-M适配的因果效果。三窗口固定为k=0..127、128..255、256..383，各128位置，不按结果改变边界。
