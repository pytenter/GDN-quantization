# Ling/KDA P1 vs Original L7 四文档前瞻验证

实验：LING_KDA_P1_L7_PROSPECTIVE_EXTENSION_4DOC_V1。最终执行：FAIL_CLOSED_STOP。原两篇正式1胜1负、MIXED永久保留；旧交叉面板仅为事后探索，本轮不与历史合并。

1. 读取的正式资产：原FUNCTIONAL_VALIDATION成功delivery及COMPLETED_ARTIFACT_AUDIT，MIXED_OUTCOME_RESOLUTION成功RESULT_AND_DELIVERY_HASHES，正式NEXT_PROSPECTIVE_4DOC_PREREGISTRATION及协议freeze。SHA与原scorer身份PASS，具体绝对路径/SHA见PREREGISTRATION_RECONCILIATION.json。

2. PG19输入：100篇原文、完整token重新编码、前缀token、MD5/SHA及去重/排序逐项复现。历史排除20、长度不足22、输入资格58，与冻结记录完全一致。首次新增CPU审计脚本错误保留CRLF，导致假失败；对照原Path.read_text读取后复现全部原SHA。首次STOP和修复证据保留，科学输入未改、当时GPU科学单元0。

3. 实际完整短程筛选4篇，上限40。选择状态：COMPLETE_4_ELIGIBLE_FROZEN。每篇完整16个冻结旋转、原P1和L7_TOTAL、两窗256/1024 H128、18层16heads；筛选阶段没有读取新文档长程结果。

4. 最终四篇：pg19-test-41603, pg19-test-2544, pg19-test-7987, pg19-test-26183。

5. 每篇冻结预测与midrank gap：pg19-test-41603: P1=H_to_L7_a50, L7=H, gaps=6.0/4.0；pg19-test-2544: P1=H_to_L6_a50, L7=H_to_L7_a50, gaps=7.0/6.0；pg19-test-7987: P1=H_to_L7_a50, L7=H_to_L7R_a25, gaps=6.0/4.0；pg19-test-26183: P1=H_to_L6_a50, L7=H_to_L7R_a25, gaps=5.0/6.0。实际完成分数、排名及eligible pairs见screening/*.SELECTION_DECISION.json；未freeze时FROZEN_4DOC_SELECTION.json为NOT_RUN；精确相等才并列，双方gap≥3，最大min-gap后字典序。

6. Selection freeze：PASS_FROZEN_BEFORE_LONG_OUTCOMES。若未freeze则SELECTION_FREEZE_RECEIPT.json为NOT_RUN，尚无新长程outcome；没有根据结局换文档或旋转。

7. numerical/observer/fresh-process resume/persistent QDQ/physical资格：NOT_COMPLETED。自身旋转0→8192完整KDA/MLA/conv/position与RNG checkpoint，fresh8192→8256 exact64 logits/cache/audit、ON/OFF observer，原helper AST/SHA、相同GPU UUID/runtime及原reducer合同。INT8代码经QDQ后以FP32承载量化值写回并由下一kernel消费，不宣称physical resident INT8 kernel storage，不存在高精度shadow绕过量化。

8. 正式完整轨迹：FP 0/4，persistent student 0/8，计划总12。完成的单元均32768执行，384固定位置齐全；未完成或未启动单元不填零、不替换。planned document分母4，实际valid=0。

9. 完整精度逐文档Primary：

NOT_RUN：没有可合法汇总的完整文档Primary；未创建空科学结果表。
原V157184 FP32 KL(FP||student)，消费token[t]后的next-token logits，positions8192+64k至32704，Python sum/384；模型运行至32768。阈值3.3300064494019496e-6原样保留，只是同合格合同的经验数值分辨率。相对差异(Delta/K_L7)在CSV明确分母，仅描述，非新Primary/Secondary。

10. 文档级方向：P1=0，L7=0，UNRESOLVED=0，MISSING=4。方向判定以冻结经验分辨率为界，不将小于分辨率的符号算成胜利。

11. 独立前瞻解释：INCOMPLETE_NOT_CONFIRMATORY。各方向来自新文档和预先冻结pair；只适用于短程P1/L7分歧且双方rank gap≥3的条件分布。独立统计单位是document，不把384位置、18层16heads或6个contrast当独立样本。本轮按预注册不报告确认性p-value。即使4/4也不宣称统计确认、可训练Loss、完整未来Query因果机制或reasoning/AIME改善。

12. 与旧候选选择现象：本轮冻结pair与旧两种pair重合的文档为[]。只测所冻两个student，没有额外L7-a50对照；未测第三候选时不能断定复制了旧三旋转排序或L7-a50普遍最优。P1预测失败文档=[]，L7预测失败文档=[]，其余UNRESOLVED不计胜负。效应大小排序：NOT_RUN；文档方向计数不按少数大效应加权，不凭 pooled mean 宣称优势。四篇不足以判断效应分布的总体代表性。

13. 实际资源：GPU worker 21，自然退出21，失败退出0；实际5.88007873827 GPUh。CUDA allocated峰值16971451904 bytes，reserved峰值17855152128 bytes；NVML采样峰值17368.0 MiB。逐worker墙钟含加载/退出，排除CPU/空等；CUDA为原统计scope，NVML不是连续瞬时高水位。详见RESOURCE_REPORT.json，单GPU6 RTX3090串行单缓存。

14. source commit=dfcbb7e8497a585f8ef94c0eec6d4146ade860b0。实际服务器交付目录=/data/zypan/experiment_runs/LING_KDA_P1_L7_PROSPECTIVE_EXTENSION_4DOC_V1/execution_02，clean worktree=/data/zypan/worktrees/ling-kda-prospective-engineering-recovery。模型/旋转/输入/FP/checkpoint/runtime/评分身份及source SHA逐项绑定；大型张量留服务器，可用manifest核验。RESULT_AND_DELIVERY_HASHES.json和SHA256SUMS.txt封存交付身份。

15. 已FAIL_CLOSED_STOP。TRAINING_AUTHORIZATION=NOT_AUTHORIZED。没有启动梯度、optimizer/Cayley、新L8/L9、AIME/free generation、额外32K或Qwen。本轮仅完成已授权4-doc扩展；后续建议只写报告。

方向预测结果可以提供P1相对于L7在这一条件分布中的有限比较证据，但不能据4篇识别统计条件独立的增量信息。更大独立验证可另行预注册；若仍MIXED，可优先error propagation与future readout observability机制研究；若不支持P1，降低其作为独立训练Loss的优先级。任何P1 gradient trainability资格研究仍需独立设计与授权，本轮结果不自动推进训练。具体误差方向、Query敏感性与历史累积的因果链仍NOT_IDENTIFIABLE。

E_future=Hidden KL=NOT_FROZEN_NOT_COMPUTED。原L7已H128 recurrent-aware；P1仍原K3局部readout-aware，并非全部未来Query。原AIME背景未重测。

机器状态：

```json
{
  "PREREGISTRATION_IDENTITY_STATUS": "PASS",
  "INPUT_ELIGIBILITY_STATUS": "PASS_EXACT_100_20_22_58",
  "SHORT_HORIZON_SELECTION_STATUS": "COMPLETE_4_ELIGIBLE_FROZEN",
  "FROZEN_SELECTION_STATUS": "PASS_FROZEN_BEFORE_LONG_OUTCOMES",
  "NUMERICAL_QUALIFICATION_STATUS": "NOT_COMPLETED",
  "PERSISTENT_INT8_SEMANTICS_STATUS": "INCOMPLETE_OR_NOT_RUN",
  "FP_REFERENCE_STATUS": "COMPLETE_0_OF_4",
  "STUDENT_32K_STATUS": "COMPLETE_0_OF_8",
  "PROSPECTIVE_PRIMARY_STATUS": "INCOMPLETE_0_VALID_OF_4",
  "DOCUMENT_LEVEL_VERDICT": "INCOMPLETE_NOT_CONFIRMATORY",
  "SCIENTIFIC_INTERPRETATION_STATUS": "INCOMPLETE_NO_CONFIRMATORY_INFERENCE",
  "TRAINING_AUTHORIZATION": "NOT_AUTHORIZED",
  "FINAL_EXECUTION_STATUS": "FAIL_CLOSED_STOP"
}
```

工程恢复：execution_01只读保留，原48条分数经manifest验证后直接复用；本轮新增部分从第四篇按原顺序继续。历史GPU异常根因仍UNKNOWN，实际连续准入结果见GPU_ADMISSION_DIAGNOSIS.json；GPU6同UUID，未更换设备。资源准入、控制器中断与恢复的synthetic测试见GPU_ADMISSION_TEST_REPORT.json、CONTROLLER_RESUME_TEST_REPORT.json。数学helper不变及实际源码diff见NUMERICAL_SOURCE_COMPATIBILITY_AUDIT.json和SOURCE_DIFF.patch。12条件自身checkpoint/fresh64/observer资格见LONG_HORIZON_QUALIFICATION.json（未运行则NOT_RUN）。资源账本分列execution_01复用GPUh和execution_02新增GPUh。没有以提高阈值或删除利用率检查取得PASS。
