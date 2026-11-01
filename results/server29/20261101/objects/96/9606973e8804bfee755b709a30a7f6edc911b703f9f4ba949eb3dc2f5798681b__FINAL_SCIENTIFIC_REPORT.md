# Ling/KDA 四文档前瞻验证：工程恢复后的实际部分交付

实验：LING_KDA_P1_L7_PROSPECTIVE_EXTENSION_4DOC_V1；执行身份：execution_02；终态：FAIL_CLOSED_STOP，工程映射：GPU_ADMISSION_UNRESOLVED_STOP。

北京时间2026-10-14 13:55:23，pg19-test-7987 / H_to_L7R_a25 的 parent 已自然退出；fresh64 worker 启动前60秒/13点准入窗口未取得最后三个连续合格空闲样本。前11点AVAILABLE_VERIFIED，第12点SMI与直接NVML均报告1% GPU utilization、显存15MiB、P8、无可见compute进程，第13点恢复0%。故窗口分类TELEMETRY_INCONSISTENT；这里指稳定性条件不足，不是两接口在触发点相互矛盾。根因UNKNOWN，不能据此认定GPU损坏、他人占用、NVML误报、OOM或数值不一致。此fresh64 worker未启动，没有重试、放宽门禁或切换GPU。

1. **正式预注册及成功manifest。** 原FUNCTIONAL_VALIDATION成功delivery与COMPLETED_ARTIFACT_AUDIT、MIXED_OUTCOME_RESOLUTION成功manifest、原NEXT_PROSPECTIVE_4DOC_PREREGISTRATION及关联freeze/scorer/model/tokenizer/rotation身份核验PASS。预注册SHA为54d6d24fbc3142a692ef1a6c04a5617cf6b0abe6702a662722f34fc1b6141722；execution_01最终manifest SHA为202bf8f64d7a2bfeab93451a607d575f7cc7052ab96692d88de02176f536516b，119个清单文件及120条SHA记录再核验，历史失败目录未改。详细原路径及验证范围见PREREGISTRATION_RECONCILIATION.json、RECOVERY_ASSET_AUDIT.json。

2. **100→58。** execution_01已按原Path.read_text语义重现100篇重编码和20历史排除/22长度排除/58合格；execution_02重新核验100原文与100 token文件SHA，复用精确的既有重编码及排序结果，没有重新抽样或改变tokenizer。原首次CPU换行审计错误与STOP永久保留。

3. **短程筛选。** 按seed0冻结顺序实际扫描4篇，每篇完整16旋转，共64条分数；前3篇48条直接由原manifest字节恢复，新GPU只评分第四篇。4篇均合格，在40篇上限内选齐。窗口256/1024、H128、18层16heads、两窗等权、原P1/L7及原midrank/pair优先规则均未改。

4. **四篇文档。** pg19-test-41603、pg19-test-2544、pg19-test-7987、pg19-test-26183；均为冻结池的新文档，不与历史两篇合并。

5. **冻结旋转对。** 全16分数及排名见SHORT_HORIZON_SELECTION.csv；全部eligible pairs、模型、旋转、输入和原评分source身份见FROZEN_4DOC_SELECTION.json。

|文档|P1预测旋转|Original L7预测旋转|P1 gap|L7 gap|
|---|---|---|---:|---:|
|pg19-test-41603|H_to_L7_a50|H|6|4|
|pg19-test-2544|H_to_L6_a50|H_to_L7_a50|7|6|
|pg19-test-7987|H_to_L7_a50|H_to_L7R_a25|6|4|
|pg19-test-26183|H_to_L6_a50|H_to_L7R_a25|5|6|

6. **冻结时间。** 四篇共同freeze于10:44:26，SHA=a5c227d15918a249f8ae2059539c15f94668cb2afddf3aecd8a9f7e13851354b，先于所有新长程资格worker。所有parent启动记录均晚于freeze；正式32K与Primary均未启动。SELECTION_FREEZE_RECEIPT.json保留原时刻，未以结局换文档或候选。

7. **资格。** 不是12/12 PASS：8条件自身0→8192 checkpoint、parent0→8256与fresh8192→8256的完整logit/cache/audit/checkpoint SHA exact以及observer ON/OFF通过；第9条件仅parent完整，fresh64因物理准入未启动；第四篇3条件全部NOT_RUN。9个真实checkpoint在服务器重新流式核验SHA并绑定，未将未资格checkpoint冒充PASS。INT8学生parent记录每token/layer QDQ 148608次、下一kernel消费148590次；量化值由FP32 tensor承载，不宣称physical resident INT8 storage。全部正式32K有限性与语义门禁尚未完成。见PARTIAL_LONG_HORIZON_QUALIFICATION.json与CHECKPOINT_MANIFEST.json。

|文档|条件|parent|fresh64|正式32K|
|---|---|---|---|---|
|pg19-test-41603|FP|PASS_COMPLETE_0_TO_8256|PASS|NOT_RUN|
|pg19-test-41603|H_to_L7_a50|PASS_COMPLETE_0_TO_8256|PASS|NOT_RUN|
|pg19-test-41603|H|PASS_COMPLETE_0_TO_8256|PASS|NOT_RUN|
|pg19-test-2544|FP|PASS_COMPLETE_0_TO_8256|PASS|NOT_RUN|
|pg19-test-2544|H_to_L6_a50|PASS_COMPLETE_0_TO_8256|PASS|NOT_RUN|
|pg19-test-2544|H_to_L7_a50|PASS_COMPLETE_0_TO_8256|PASS|NOT_RUN|
|pg19-test-7987|FP|PASS_COMPLETE_0_TO_8256|PASS|NOT_RUN|
|pg19-test-7987|H_to_L7_a50|PASS_COMPLETE_0_TO_8256|PASS|NOT_RUN|
|pg19-test-7987|H_to_L7R_a25|PASS_COMPLETE_0_TO_8256|NOT_RUN|NOT_RUN|
|pg19-test-26183|FP|NOT_RUN|NOT_RUN|NOT_RUN|
|pg19-test-26183|H_to_L6_a50|NOT_RUN|NOT_RUN|NOT_RUN|
|pg19-test-26183|H_to_L7R_a25|NOT_RUN|NOT_RUN|NOT_RUN|

8. **轨迹。** 正式FP 0/4，student 0/8，共0/12。9个parent和8个fresh64是工程资格worker，不是完整32K轨迹；没有正式FP full-vocabulary payload。成功资格和checkpoint保留，未运行单元标为NOT_RUN。

9. **K_P1/K_L7/Delta。** 四篇全部NOT_RUN，非0。没有实际384点Primary，故sum/384校验为NOT_RUN；没有空Primary或逐位置科学CSV。原V157184 FP32 KL(FP||student)、消费当前token后的next-token logits、8192+64k共384点至32704、完整32768及原sum/384合同不变。经验算术分辨率3.3300064494019496e-6不当作统计显著性阈值。

10. **方向计数。** 有效文档n=0；P1胜0、L7胜0、已观测tie/unresolved0、计划缺失4。无观察计数不是四篇平局，也不是任何方法胜负证据。

11. **科学解释。** INCOMPLETE_NOT_CONFIRMATORY。没有长程方向性证据，不能支持或否定P1相对L7的独立增量价值，不能宣称统计确认、可训练Loss、因果机制或reasoning/AIME改善。未来若完成验证，解释仍仅限预先定义的短程分歧且gap≥3条件分布，独立单位是文档；本轮不做确认性p-value。

12. **旧MIXED/posthoc现象。** 原860 P1胜、5956 L7胜及正式MIXED永久保留。新冻结文档2544的pair在候选身份上与旧L6-a50/L7-a50对相同，但没有新正式KL，无法验证是否复制真实长期排序；其余新对也没有正式结果。没有可识别的新预测失败文档、效应方向或大效应主导现象。旧三旋转面板仍为posthoc；未来readout与误差传播因果链NOT_IDENTIFIABLE。

13. **真实资源。** execution_02新增18个GPU worker：1个第四篇短程+17个资格（9 parent、8 fresh64），全部自然退出、失败退出0；复用execution_01的3个短程worker后研究账本21。新增3.611504961715804 GPUh，复用2.268573776549763 GPUh，研究合计5.880078738265568 GPUh。时间是worker launch-to-exit，含加载/退出，排除CPU、准入等待及闲置，不按利用率加权。CUDA allocated峰值16971451904 bytes（15.805896282196045 GiB）、reserved17855152128 bytes（16.62890625 GiB）；NVML周期采样峰值17368MiB，共313个已归属worker采样，不是连续瞬时峰值。CUDA是原loop reset及short teacher/per-condition统计scope，不宣称整个生命周期峰值。单GPU6 RTX3090 24GB、同UUID、torch2.7.1+cu128/CUDA12.8/driver580.178.04/BT16_W8/seed0，无TF32；没有人为4/16 GPUh或85%显存上限。

14. **源码与交付。** 冻结commit=dfcbb7e8497a585f8ef94c0eec6d4146ade860b0，worktree=/data/zypan/worktrees/ling-kda-prospective-engineering-recovery仍干净；原数学helper AST/SHA不变，新增控制/准入/退出账本/CPU交付代码及实际diff见SOURCE_DIFF.patch、NUMERICAL_SOURCE_COMPATIBILITY_AUDIT.json。24项CPU synthetic测试在Linux全PASS，包含资源故障、48分数恢复、controller锁、原子提交、成功unit幂等复用、freeze不可改写、资格失败阻止正式阶段和独立runner退出记录；这些fixture不是科学数据。服务器目录=/data/zypan/experiment_runs/LING_KDA_P1_L7_PROSPECTIVE_EXTENSION_4DOC_V1/execution_02；Mac交付目录=/Users/pytenter/Documents/临时项目1/ling_prospective_extension_4doc_v1/engineering_recovery_execution_02/delivery/。最终manifest及SHA256SUMS绑定全部小文件，9个checkpoint用真实绝对路径/SHA在服务器保留，不下载模型或大型张量。

15. **严格停止。** 本轮所有GPU worker、controller、runner及封存tmux已退出；首个STOP/no_retry保持原字节，没有恢复失败资格、换GPU、添加旋转、梯度/optimizer/P1+Cayley、新L8/L9、AIME/free-generation或Qwen。TRAINING_AUTHORIZATION=NOT_AUTHORIZED，训练步数0。最终交付后暂停跟进。

工程结果：结构性控制器改造、既有48分数恢复、四篇freeze、24项mock测试及8个真实自身恢复资格已完成；GPU6历史根因仍UNKNOWN，新的准入稳定性仍失败，因此不能宣称整个工程资源问题已经解决。没有实际切换GPU或取得替代GPU数值资格。允许未来恢复的必要条件是另行证明资源稳定与归属，并在独立授权执行身份下核验已有科学资产和checkpoint合同；本次不会自行重试。没有数据支持自动推进更大科学验证或P1 gradient trainability；E_future、Hidden KL和新未来功能代理均NOT_FROZEN_NOT_COMPUTED。

CPU交付修正：首次自动封存报告使用通用完成模板，未充分展开部分资格，也未直接列出已有9个checkpoint。此修订仅补齐真实部分状态、大型资产绑定、资源分类细节和中文说明；首次报告/manifest/SHA保留在cpu_delivery_audit/original/，完整原封存archive及SHA见CPU_DELIVERY_CORRECTION.json。所有原始scientific output、source、selection、gate、RESOURCE_REPORT、FINAL_STATUS及STOP字节不变。

机器状态：

```json
{
  "DOCUMENT_LEVEL_VERDICT": "INCOMPLETE_NOT_CONFIRMATORY",
  "FINAL_EXECUTION_STATUS": "FAIL_CLOSED_STOP",
  "FP_REFERENCE_STATUS": "COMPLETE_0_OF_4",
  "FROZEN_SELECTION_STATUS": "PASS_FROZEN_BEFORE_LONG_OUTCOMES",
  "INPUT_ELIGIBILITY_STATUS": "PASS_EXACT_100_20_22_58",
  "NUMERICAL_QUALIFICATION_STATUS": "NOT_COMPLETED",
  "PERSISTENT_INT8_SEMANTICS_STATUS": "INCOMPLETE_OR_NOT_RUN",
  "PREREGISTRATION_IDENTITY_STATUS": "PASS",
  "PROSPECTIVE_PRIMARY_STATUS": "INCOMPLETE_0_VALID_OF_4",
  "SCIENTIFIC_INTERPRETATION_STATUS": "INCOMPLETE_NO_CONFIRMATORY_INFERENCE",
  "SHORT_HORIZON_SELECTION_STATUS": "COMPLETE_4_ELIGIBLE_FROZEN",
  "STUDENT_32K_STATUS": "COMPLETE_0_OF_8",
  "TRAINING_AUTHORIZATION": "NOT_AUTHORIZED"
}
```
