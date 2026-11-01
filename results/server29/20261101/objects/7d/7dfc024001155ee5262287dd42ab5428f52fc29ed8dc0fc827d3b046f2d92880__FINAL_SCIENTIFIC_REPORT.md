# Ling/KDA execution_04 最小修复与前瞻验证：终态交付

科学ID：LING_KDA_P1_L7_PROSPECTIVE_EXTENSION_4DOC_V1；本轮独立身份：execution_04。结论：FAIL_CLOSED_STOP。ENGINEERING_COMPLETE=True；NUMERICAL_QUALIFICATION_COMPLETE=False；FORMAL_SCIENCE_COMPLETE=False。原两文档正式 MIXED 永久保留，旧 posthoc 面板不变。

## A. 原资产审计

原预注册 SHA=54d6d24fbc3142a692ef1a6c04a5617cf6b0abe6702a662722f34fc1b6141722；四篇 freeze SHA=a5c227d15918a249f8ae2059539c15f94668cb2afddf3aecd8a9f7e13851354b；freeze 北京时间2026-10-14 10:44:26。Phase0逐文件核验原成功manifest、execution01/02 manifest和SHA清单、64条评分、9个真实checkpoint、8个完整PASS和第9 parent。旧目录全程只读。修复仅在解释器启动前由launcher构造冻结三变量环境；未改或删除initializer断言。新科学文档和旋转未改：pg19-test-41603: H_to_L7_a50 vs H (gap=6.0/4.0)；pg19-test-2544: H_to_L6_a50 vs H_to_L7_a50 (gap=7.0/6.0)；pg19-test-7987: H_to_L7_a50 vs H_to_L7R_a25 (gap=6.0/4.0)；pg19-test-26183: H_to_L6_a50 vs H_to_L7R_a25 (gap=5.0/6.0)。79项源资产含模型权重实际字节SHA已复核。execution03原环境启动错误与全部旧修复记录只读保留。本轮真实启动链路CPU回归见CPU_LAUNCH_CHAIN_REGRESSION.json；parent/fresh64正例与缺失seed负例均调用冻结initializer。GPU2/3各一次runtime canary见GPU_RUNTIME_CANARY.json，不加载模型且科学轨迹计数0。数学helper、原quantizer/forward/scale/KL reducer不变，实际新源码commit=d3cd88d4643a460e027cae6ba6989628c21f60bf，diff与AST核验见SOURCE_DIFF.patch和NUMERICAL_SOURCE_COMPATIBILITY_AUDIT.json。

## B. 启动修复、真实链路测试与GPU证据

冻结源码断言和历史成功manifest共同确认三个环境要求：`PYTHONHASHSEED=0`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`、`LING_GATED_RMSNORM_FIXED_CONFIG=BT16_W8`。历史原始进程环境未完整捕获，hash seed的历史证据是冻结断言加成功worker；没有将其冒充原始env快照。本轮launcher在新解释器启动前构造env，不在worker内部补seed，未删除或改写initializer断言。

新增真实CPU链路6项全部PASS：parent/fresh64各2正例和1个缺seed负例。正式launcher同一builder、原master lane表达式和未改unit_runner实际启动子解释器，再调用冻结initializer。正例env逐项等于launcher预期、启动hash_randomization=0、全新解释器hash一致；两个负例在原断言处正确失败。CPU-only未加载模型或执行CUDA科学计算。原37项工程测试保留，13项恢复测试重新PASS；CPU合格没有代替GPU资格。

execution04实际预检文件为GPU_PREFLIGHT_RESULT_01.json和GPU_PREFLIGHT_WINDOW_01.jsonl。13个实际GPU worker各有独立原60秒、13点准入窗口，最后3点util严格0、allocated不超过64MiB、无compute占用，UUID及归属按冻结规则检查；终态CPU复核见FINAL_EVIDENCE_AUDIT.json。GPU2/3各一次runtime canary均正常退出、实际环境与启动hash seed正确、冻结initializer和CUDA初始化成功。Canary仅最小1-byte分配，不加载模型、不计数值资格或科学轨迹。

系统Xorg约4MiB保留其PID/UID/starttime及SMI/NVML图形基线。NVML v1含驱动reserved，v2 used与SMI allocated分别核对，不能混用。主机PID namespace和不可读驱动日志仍UNKNOWN。未使用非授权GPU或终止其他用户进程。

## C. 资格

目标设备真实合格 9/12；历史GPU6 8/12保持原标签。9个已完成parent没有重算，跨设备fresh64须逐条件比较全部64 full-V logits SHA、完整native cache、audit state、checkpoint SHA，恢复tensor指纹/RNG及observer ON/OFF exact；新设备标签只有真实通过后才生成。第四篇实际只完成FP parent及fresh各1个worker；二者正常退出但精确资格FAIL，另2个student parent/fresh均NOT_RUN。数学与数值门禁、精确比较方式未改。逐条件状态为9 PASS、1 FAIL、2 NOT_RUN，见QUALIFICATION_MATRIX.json；worker自然退出和parent完成都不等于精确资格PASS。

## D. 正式32K

FP 0/4；student 0/8；valid文档 0/4；planned12轨迹。资格全部12通过且资源PASS后才正式运行，原冻结文档順序逐篇FP→P1→L7，全局串行；双卡只加速资格阶段。同文档FP/P1/L7在同一目标UUID/runtime配对。原8192 checkpoint是其自身token0量化/FP历史的真实合法前缀；迁移资格只有在真实fresh64 bitwise证明完整状态/缓存兼容后才算PASS，再由正式first64 observer复验；本轮实际资格/轨迹计数以前述字段为准，绝不把8256当32768。前3篇prefix历史GPU6、正式continuation设备=GPU-7d6246ed-c3d1-75bc-c2c7-d02eb2882cca，第四篇设备=GPU-759526ed-efb2-e7f6-881c-ab098e0b05ae；不会声称整条轨迹始终在同UUID。INT8 code经QDQ后由FP32载体承载量化值，写回并被下一kernel消费；不称physical resident INT8 storage。

## E. Primary结果

NOT_RUN：没有完整有效四文档Primary；未执行项不填0、不生成空科学CSV。
V=157184，原FP32 KL(FP||student)，消费ids[t]后logits预测ids[t+1]，t=8192+64k，k0..383，末32704，模型实际endpoint32768。本轮没有正式student，384点有限性和sum/384检查均NOT_RUN；若正式运行，必须逐点齐全并使用严格原Python sum/384。Delta=K_P1-K_L7，阈值3.3300064494019496e-6仅原经验数值分辨率。P1=0，L7=0，UNRESOLVED=0，MISSING=4。原协议不报告confirmatory p-value；即使独立无平局且p0=.5下4/4同方向，单侧exact sign .0625也不构成5%确认。本轮仅短程分歧及gap≥3条件下的小样本前瞻解释，document为单位，不将384点/层/heads当独立n，不合并旧2篇为n6。E_future/Hidden KL=NOT_FROZEN_NOT_COMPUTED。

## F. 实际资源及交付

execution01复用 2.2685737765497631 GPUh；execution02复用 3.6115049617158039 GPUh；execution04新增 0.50427797138690944 GPUh；累计 6.3843567096524767 GPUh。本轮自然退出worker 13（含runtime canary 2）、失败退出 0；历史execution03失败启动另计 0.0030530354711744518 GPUh，全部尝试合计 6.3874097451236516 GPUh；未获退出receipt的时段仅独立上界，不能伪装精确计量。总计按每worker launch-to-exit求和，含加载/退出，排除CPU/准入空等；并行卡小时分别相加。本轮CUDA allocated峰值 16935037440 bytes，reserved 17349738496 bytes；历史+本轮NVML定期owned采样峰值 17368.0 MiB。CUDA原loop reset scope、NVML periodic scope有明确限制，未声称whole-lifetime或连续峰值。真实checkpoint 10个，逐路径、bytes、SHA见CHECKPOINT_MANIFEST.json；正式FP payload为NOT_RUN，没有可交付的32K FP logits；第10个26183/FP checkpoint仅保存原始资产，未通过恢复资格。服务器 /data/zypan/experiment_runs/LING_KDA_P1_L7_PROSPECTIVE_EXTENSION_4DOC_V1/execution_04；工作树 /data/zypan/worktrees/ling-kda-prospective-execution04；本地交付 /Users/pytenter/Documents/临时项目1/ling_prospective_extension_4doc_v1/execution_04_minimal_recovery/delivery/。大型FP/checkpoint留服务器绑定SHA，模型不下载。manifest与SHA256SUMS含manifest自身的SHA验证。

## G. 首个blocker与复用资产

```json
{
  "epoch": 1792027914.700743,
  "no_retry": true,
  "reason": "AssertionError('EXACT_DEVICE_RESUME_PARITY_FAILED')",
  "stage": "RECOVERY_CONTROLLER",
  "status": "STOP",
  "traceback": "Traceback (most recent call last):\n  File \"/data/zypan/worktrees/ling-kda-prospective-execution04/recovery_controller.py\", line 123, in <module>\n    main()\n  File \"/data/zypan/worktrees/ling-kda-prospective-execution04/recovery_controller.py\", line 110, in main\n    run_main()\n  File \"/data/zypan/worktrees/ling-kda-prospective-execution04/qualified_completion.py\", line 111, in run_main\n    if os.environ.get('LING_EXECUTION_LANE'):return lane_main()\n                                                    ^^^^^^^^^^^\n  File \"/data/zypan/worktrees/ling-kda-prospective-execution04/qualified_completion.py\", line 64, in lane_main\n    done=[qualification_unit(u,uuid) for u in units]\n         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/data/zypan/worktrees/ling-kda-prospective-execution04/qualified_completion.py\", line 64, in <listcomp>\n    done=[qualification_unit(u,uuid) for u in units]\n          ^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/data/zypan/worktrees/ling-kda-prospective-execution04/qualified_completion.py\", line 50, in qualification_unit\n    compare_exact(parent,fresh,uuid)\n  File \"/data/zypan/worktrees/ling-kda-prospective-execution04/qualified_completion.py\", line 21, in compare_exact\n    assert all(parent[k]==fresh[k] for k in FIELDS),'EXACT_DEVICE_RESUME_PARITY_FAILED'\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\nAssertionError: EXACT_DEVICE_RESUME_PARITY_FAILED\n"
}
```

实际成功单元/NOT_RUN见QUALIFICATION_MATRIX及机器状态。任何失败首条保留no_retry，不放宽门禁、不重选、不用fixture作新科学证据。已有成功字节可复用的范围仅已核验的相同科学key、source/runtime及实际已合格设备；新设备/新的恢复仍须资格，不自动沿用旧PASS。

## H. 科学解释与后续建议

INCOMPLETE_NOT_CONFIRMATORY。四篇前瞻Primary均未运行，本轮没有P1-vs-L7科学比较结果，不能证明P1梯度可优化性、全部future Query因果机制、reasoning/AIME改善或总体稳定优势。本轮到此停止，不安排后续科学实验；任何新执行需要新的用户指令。训练授权NOT_AUTHORIZED；P1 Loss、L8/L9、STE/Cayley/optimizer、AIME/free generation/Qwen均未启动，交付后停止。

## 原15问逐项状态

1. 读取原成功、历史execution01/02/03 manifest与科学预注册：PASS，路径/SHA见Phase0。
2. 输入资格原100/20/22/58与真实tokens/raw SHA保留，新执行不重复编码或选文档。
3. 原完整4篇16旋转=64分数全部复用，未重筛。
4. 四篇41603/2544/7987/26183，完全冻结。
5. 预测与midrank gap：pg19-test-41603: H_to_L7_a50 vs H (gap=6.0/4.0)；pg19-test-2544: H_to_L6_a50 vs H_to_L7_a50 (gap=7.0/6.0)；pg19-test-7987: H_to_L7_a50 vs H_to_L7R_a25 (gap=6.0/4.0)；pg19-test-26183: H_to_L6_a50 vs H_to_L7R_a25 (gap=5.0/6.0)；全部候选、排名、规则原样。
6. freeze在所有新长程前；原时间和receipt字节未改。
7. 目标GPU numerical/fresh/observer/QDQ资格：9/12，身份及设备迁移证据逐项保留。
8. 正式FP 0/4，student 0/8，valid 0/4。
9. 四篇KL、Delta和胜负均NOT_RUN；没有生成空科学CSV，未运行项不填0。
10. 文档计数P1=0/L7=0/UNRESOLVED=0，未完成不算负。
11. 只作条件性小样本前瞻，不做confirmatory p-value，不伪重复。
12. 没有额外第三候选轨迹，不能据冻结两候选推断L7-a50普遍最优或复现所有旧三旋转排序。
13. 实际资源分执行账本及真实scope见F/RESOURCE_REPORT。
14. 实际source commit、路径、模型/输入/旋转/CP/FP/runtime/UUID/scoring SHA见封存清单。
15. 终态FAIL_CLOSED_STOP；未授权训练与额外科学任务均未执行。

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
  "FINAL_EXECUTION_STATUS": "FAIL_CLOSED_STOP",
  "ENGINEERING_COMPLETE": true,
  "NUMERICAL_QUALIFICATION_COMPLETE": false,
  "FORMAL_SCIENCE_COMPLETE": false,
  "QUALIFIED_CONDITIONS": 9,
  "FORMAL_FP_COMPLETE": 0,
  "FORMAL_STUDENT_COMPLETE": 0,
  "ACTUAL_VALID_DOCUMENTS": 0
}
```

## CPU终态核验与首个失败字段

本轮启动工程修复已通过；随后遇到独立的数值恢复资格失败。26183/FP的parent与fresh64都在GPU3同一UUID和相同已记录runtime上运行，parent observer ON、fresh observer OFF。不能称为该文档的跨GPU误差。

| 比较字段 | parent | fresh64 | 结论 |
| --- | --- | --- | --- |
| uninterrupted_suffix_full_logit_sha256 | b5396ddb9e8b25a190b69067262064d68898952890cde13da568020125347182 | 4659e4c25505f8533b1a0d85fc6ccf432c281564423d0b123d8470f45657a69b | 不同，FAIL |
| final_full_native_cache_sha256 | 19607dc044c592d00a597662733b0f7c10089e46f68f36f3add3dd7ae289320f | 90a05ecdf48fa8c4ce3c3ab34f81bc86ff43b2afec4b1fd29fc26792cdd9b362 | 不同，FAIL |
| audit_state_fingerprint | None | None | 相同 |
| checkpoint_sha256 | 2e6abbe74fc5266fd26c3b74da4977cdd3849382534852503fb68a10e0f77f4f | 2e6abbe74fc5266fd26c3b74da4977cdd3849382534852503fb68a10e0f77f4f | 相同 |

FP的audit字段均为null，表示不适用；不是额外有效state证据。两个完整指纹不同已足以触发原精确门禁，但不能推出误差大小、首个分歧token、具体tensor、kernel或硬件故障。底层原因UNKNOWN；没有额外GPU重试或新公式。首STOP北京时间2026-10-15T09:31:54.700743+08:00，原STOP字节保持不变。

13个worker均自然退出：2个canary、9个GPU2 fresh64、1个GPU3 parent、1个GPU3 fresh64；worker exit无失败不表示资格无失败。12条正式32768轨迹全部NOT_RUN；384位置Primary、四篇KL/Delta及文档胜负全部NOT_RUN。正式前资源估算/交付能力门禁尚未执行，不能把资格worker的显存测量冒充正式32K资源PASS。

实际新增卡小时按stage见RESOURCE_STAGE_BREAKDOWN.json：

```json
{
  "resumed_parity": {
    "workers": 10,
    "wall_seconds": 547.9189898967743,
    "GPU_hours": 0.15219971941577065
  },
  "parent": {
    "workers": 1,
    "wall_seconds": 1258.3612127304077,
    "GPU_hours": 0.34954478131400213
  },
  "runtime_canary": {
    "workers": 2,
    "wall_seconds": 9.120494365692139,
    "GPU_hours": 0.002533470657136705
  },
  "formal": {
    "workers": 0,
    "wall_seconds": 0.0,
    "GPU_hours": 0.0,
    "status": "NOT_RUN_NO_LAUNCH"
  }
}
```

本轮NVML exact-owned定期采样峰值16886.0MiB，共29个有效采样；RESOURCE_REPORT原NVML峰值为历史与本轮合并口径。CUDA峰值仅原loop reset scope，不包括加载全生命周期峰值。所有10个checkpoint实际字节SHA再次核验，9个旧parent没有重算；第10个不获资格标签。终态进程已退出，tmux已结束，未发送任何进程终止信号。

原自动封存报告保存在独立原始archive中（见DELIVERY_CPU_CORRECTIONS.json）。本次只澄清自动报告引用旧预检文件名、将fresh worker正常退出误作资格PASS、未运行384点措辞等CPU元数据问题。未修改worktree、原STOP、worker结果、checkpoint或任何科学门禁。当前报告覆盖原A-H及15问；机器状态如下。

```json
{
  "ACTUAL_VALID_DOCUMENTS": 0,
  "DOCUMENT_LEVEL_VERDICT": "INCOMPLETE_NOT_CONFIRMATORY",
  "ENGINEERING_COMPLETE": true,
  "FINAL_EXECUTION_STATUS": "FAIL_CLOSED_STOP",
  "FORMAL_FP_COMPLETE": 0,
  "FORMAL_SCIENCE_COMPLETE": false,
  "FORMAL_STUDENT_COMPLETE": 0,
  "FP_REFERENCE_STATUS": "COMPLETE_0_OF_4",
  "FROZEN_SELECTION_STATUS": "PASS_FROZEN_BEFORE_LONG_OUTCOMES",
  "INPUT_ELIGIBILITY_STATUS": "PASS_EXACT_100_20_22_58",
  "NUMERICAL_QUALIFICATION_COMPLETE": false,
  "NUMERICAL_QUALIFICATION_STATUS": "NOT_COMPLETED",
  "PERSISTENT_INT8_SEMANTICS_STATUS": "INCOMPLETE_OR_NOT_RUN",
  "PREREGISTRATION_IDENTITY_STATUS": "PASS",
  "PROSPECTIVE_PRIMARY_STATUS": "INCOMPLETE_0_VALID_OF_4",
  "QUALIFIED_CONDITIONS": 9,
  "SCIENTIFIC_INTERPRETATION_STATUS": "INCOMPLETE_NO_CONFIRMATORY_INFERENCE",
  "SHORT_HORIZON_SELECTION_STATUS": "COMPLETE_4_ELIGIBLE_FROZEN",
  "STUDENT_32K_STATUS": "COMPLETE_0_OF_8",
  "TRAINING_AUTHORIZATION": "NOT_AUTHORIZED",
  "FAILED_QUALIFICATION_CONDITIONS": 1,
  "NOT_RUN_QUALIFICATION_CONDITIONS": 2,
  "NUMERICAL_FAILURE": "pg19-test-26183/FP: full-64 logits and final native cache SHA mismatch",
  "ALL_OWNED_GPU_WORKERS_EXITED": true,
  "EXTRA_GPU_RETRIES": 0,
  "ENGINEERING_COMPLETE_SCOPE": "Frozen startup environment, true CPU launch chain, both GPU runtime canaries only"
}
```
