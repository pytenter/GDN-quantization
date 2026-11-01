# Ling/KDA 四文档前瞻验证 execution_05

科学ID：LING_KDA_P1_L7_PROSPECTIVE_EXTENSION_4DOC_V1。终态：FAIL_CLOSED_STOP。
ENGINEERING_COMPLETE=True；NUMERICAL_QUALIFICATION_COMPLETE=False；FORMAL_SCIENCE_COMPLETE=False。

## A. 历史、根因和最小修复

execution01/02/03/04及其首STOP只读保留。原两文档MIXED与posthoc身份永久保留。预注册SHA 54d6d24fbc3142a692ef1a6c04a5617cf6b0abe6702a662722f34fc1b6141722，共同四篇freeze SHA a5c227d15918a249f8ae2059539c15f94668cb2afddf3aecd8a9f7e13851354b，freeze北京时间2026-10-14 10:44:26。原100篇输入、20历史排除、22不足长度、58资格顺序及4×16=64个短评分未重算、未重选。源资产、模型权重、tokens/raw、旋转SHA与历史manifest逐项复核见Phase0。

engineering_repair_05进行了5个预先有界GPU诊断；科学轨迹/资格计数均0。同GPU、同进程、同8192 checkpoint/RNG/observer的A→B→A干预证明convolution配置BD128/warps4会产生失败fingerprint、原冻结BD8/warps8产生旧parent fingerprint、切回A再次复现。execution04当时实际kernel配置未捕获仍UNKNOWN，不捏造历史观测；底层驱动日志与host PID namespace未知项保留。完整证据见CONFIRMED_ROOT_CAUSE.json和诊断封存。

修复只恢复原冻结FLA启动profile及缺失即失败门禁：新解释器前注入原3变量PYTHONHASHSEED=0、CUBLAS_WORKSPACE_CONFIG=:4096:8、LING_GATED_RMSNORM_FIXED_CONFIG=BT16_W8，并绑定FLA_CACHE_MODE=default和本工作树内6个逐字节原配置JSON。原initializer断言、数学helper、量化器、rotation、forward、full-V KL及容差不变。新CPU真实链路10case（4正、6缺seed/FLA负例）与11项profile/envelope检查全部PASS，均无模型/CUDA；CPU通过不等于GPU数值资格。每个科学worker实际/proc启动环境、原FLA loader加载值、实际执行conv BD8/warps8及gated norm BT16/warps8经独立envelope只读检查后，才能生成NATURAL_EXIT。源码commit=902d7974c89465df882403e96cc8dcc8dc1c0e85，diff/AST/依赖/6配置SHA见SOURCE与FORMAL_CODE_FREEZE、NUMERICAL_SOURCE_COMPATIBILITY_AUDIT。

## B. 资源准入与canary

仅用户授权GPU2/3：{'GPU-759526ed-efb2-e7f6-881c-ab098e0b05ae': 3, 'GPU-7d6246ed-c3d1-75bc-c2c7-d02eb2882cca': 2}。实际预检见GPU_PREFLIGHT_RESULT_*.json，每worker仍另经未改60秒13点、最后3点严格util0/allocated≤64MiB、SMI/NVML v2一致性与UUID/PID UID/starttime归属门禁；系统Xorg约4MiB以真实身份显式绑定，UNKNOWN不作空闲。未杀他人进程、reset GPU或放宽任何门禁。两卡各一次1-byte CUDA runtime canary不加载模型、无token，不计数值资格或科学轨迹；stage资源单列。canary实际状态见GPU_RUNTIME_CANARY.json，缺失时为未完成。

## C. checkpoint复用边界与资格

旧10个checkpoint实际SHA及合法科学key保留，但旧0→8192过程中完整FLA配置没有可验证启动记录；fresh64兼容不能倒推旧prefix运行profile合规。因此本轮不继承旧parent或旧PASS标签，按相同12个科学key在目标设备重新建立12个自身0→8192 parent并执行12个fresh64；这不是新增科学条件。见CHECKPOINT_REUSE_DECISION.json。本轮真实合格2/12；每项完整64步V157184 logits、native cache、state/audit、checkpoint/RNG、observer ON/OFF exact，INT8 QDQ写回与下一kernel消费均沿用原比较和阈值。历史GPU6 8/12及execution04 9PASS/1FAIL/2NOT_RUN仅历史标签。逐项真实PASS/FAILED/NOT_RUN见QUALIFICATION_MATRIX.json。

## D. 原冻结正式32K

固定预测：pg19-test-41603: H_to_L7_a50 vs H (gap=6.0/4.0)；pg19-test-2544: H_to_L6_a50 vs H_to_L7_a50 (gap=7.0/6.0)；pg19-test-7987: H_to_L7_a50 vs H_to_L7R_a25 (gap=6.0/4.0)；pg19-test-26183: H_to_L6_a50 vs H_to_L7R_a25 (gap=5.0/6.0)。原文档顺序41603→2544→7987→26183，每篇FP→P1→OriginalL7，全局串行；双卡仅并行资格阶段。前三篇GPU2，第四篇GPU3；本轮每个parent、fresh和正式continuation同目标UUID、同恢复后冻结runtime。只有12/12资格和原物理显存/磁盘/交付门禁PASS，才进入正式。当前FP=0/4，student=0/8，完整有效文档=0/4。8256资格parent绝不冒充32768正式轨迹。INT8采用QDQ后的量化值写回FP32载体并被下一kernel消费，不称physical resident INT8 storage。

## E. 实际Primary

NOT_RUN：尚无完整合法文档Primary；未运行数值不填0。

原V157184 FP32 KL(FP||student)，消费ids[t]后预测ids[t+1]。固定t=8192+64k，k0..383，末32704，endpoint32768。实际已计入的完整student=0/8，其384点逐点finite且原Python sum/384严格相等；未计入项没有声称通过。Delta=K_P1−K_L7，负值P1更好，正值L7更好；原经验分辨率3.3300064494019496e-6不改。文档胜负P1=0、L7=0、UNRESOLVED=0、缺失=4。完整精度值与逐点核验在真实CSV；未完成时不生成空科学CSV。Secondary只用原V4 recurrent-state L2/MSE与scale/clipping/QDQ诊断；原V4未导出的layer-wise指标不新增；E_future/Hidden KL/new proxy均NOT_FROZEN_NOT_COMPUTED。

## F. 实际资源与交付

历史各执行GPUh：{"execution_01": 2.268573776549763, "execution_02": 3.611504961715804, "execution_03": 0.003053035471174452, "execution_04": 0.5042779713869094, "engineering_repair_05": 0.42219087839126584}；execution05新增0.85848787903785717 GPUh；所有执行与5个工程诊断合计7.6680885025527745 GPUh。按每worker launch-to-exit求和，包含加载/退出，排除准入/CPU/空等，并行卡分别相加，不做util加权；无退出receipt只报告单独上界，不伪造精确值。各stage见RESOURCE_REPORT.json；新自然退出6、失败退出1、canary自然退出2。CUDA allocated/reserved新测量峰值分别16486794240/20128464896 bytes，为原loop-reset计数scope，不声称模型加载全生命周期峰值。NVML新owned定期采样峰值19536.0 MiB，非连续瞬时峰值；无测量为null。历史峰值各原账本保留。未添加GPUh或显存百分比硬上限。

真实checkpoint=2，逐文件路径/bytes/SHA见CHECKPOINT_MANIFEST；真实FP payload/387位置refs见各正式manifest。服务器/data/zypan/experiment_runs/LING_KDA_P1_L7_PROSPECTIVE_EXTENSION_4DOC_V1/execution_05，工作树/data/zypan/worktrees/ling-kda-prospective-execution05。本地小文件交付目录/Users/pytenter/Documents/临时项目1/ling_prospective_extension_4doc_v1/engineering_repair_05/execution_05/delivery/；模型、大型FP/checkpoint只服务器manifest绑定。RESULT_AND_DELIVERY_HASHES与SHA256SUMS含manifest自身记录并本地逐项核验。

## G. 首失败与部分状态

```json
{
  "epoch": 1792039241.7216046,
  "no_retry": true,
  "reason": "RuntimeError(('GPU_ADMISSION_STOP', {'status': 'TELEMETRY_INCONSISTENT', 'uuid': 'GPU-759526ed-efb2-e7f6-881c-ab098e0b05ae', 'index': '3', 'completed_epoch': 1792039241.713792, 'sample_count': 13, 'states': ['AVAILABLE_VERIFIED', 'AVAILABLE_VERIFIED', 'AVAILABLE_VERIFIED', 'AVAILABLE_VERIFIED', 'AVAILABLE_VERIFIED', 'AVAILABLE_VERIFIED', 'AVAILABLE_VERIFIED', 'AVAILABLE_VERIFIED', 'AVAILABLE_VERIFIED', 'AVAILABLE_VERIFIED', 'AVAILABLE_VERIFIED', 'AVAILABLE_VERIFIED', 'TELEMETRY_INCONSISTENT'], 'execution': '/data/zypan/experiment_runs/LING_KDA_P1_L7_PROSPECTIVE_EXTENSION_4DOC_V1/execution_05', 'driver': '580.178.04', 'window_sha256': 'e6e760e9c98f800323db92dd70c0f9b2e8152932b278a64855887bed8c815d5a', 'window_path': '/data/zypan/experiment_runs/LING_KDA_P1_L7_PROSPECTIVE_EXTENSION_4DOC_V1/execution_05/admission/pg19-test-26183__H_to_L6_a50__parent.window.jsonl'}))",
  "stage": "RECOVERY_CONTROLLER",
  "status": "STOP",
  "traceback": "Traceback (most recent call last):\n  File \"/data/zypan/worktrees/ling-kda-prospective-execution05/recovery_controller.py\", line 123, in <module>\n    main()\n  File \"/data/zypan/worktrees/ling-kda-prospective-execution05/recovery_controller.py\", line 110, in main\n    run_main()\n  File \"/data/zypan/worktrees/ling-kda-prospective-execution05/qualified_completion.py\", line 111, in run_main\n    if os.environ.get('LING_EXECUTION_LANE'):return lane_main()\n                                                    ^^^^^^^^^^^\n  File \"/data/zypan/worktrees/ling-kda-prospective-execution05/qualified_completion.py\", line 64, in lane_main\n    done=[qualification_unit(u,uuid) for u in units]\n         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/data/zypan/worktrees/ling-kda-prospective-execution05/qualified_completion.py\", line 64, in <listcomp>\n    done=[qualification_unit(u,uuid) for u in units]\n          ^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/data/zypan/worktrees/ling-kda-prospective-execution05/qualified_completion.py\", line 49, in qualification_unit\n    parent=run(doc,rot,'parent');fresh=run(doc,rot,'resumed_parity')\n           ^^^^^^^^^^^^^^^^^^^^^\n  File \"/data/zypan/worktrees/ling-kda-prospective-execution05/recovery_controller.py\", line 58, in run\n    admit(cfg,admission_path/(tag+'.window.jsonl'))\n  File \"/data/zypan/worktrees/ling-kda-prospective-execution05/gpu_admission.py\", line 85, in admit\n    if state!='AVAILABLE_VERIFIED':raise RuntimeError(('GPU_ADMISSION_STOP',report))\n                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\nRuntimeError: ('GPU_ADMISSION_STOP', {'status': 'TELEMETRY_INCONSISTENT', 'uuid': 'GPU-759526ed-efb2-e7f6-881c-ab098e0b05ae', 'index': '3', 'completed_epoch': 1792039241.713792, 'sample_count': 13, 'states': ['AVAILABLE_VERIFIED', 'AVAILABLE_VERIFIED', 'AVAILABLE_VERIFIED', 'AVAILABLE_VERIFIED', 'AVAILABLE_VERIFIED', 'AVAILABLE_VERIFIED', 'AVAILABLE_VERIFIED', 'AVAILABLE_VERIFIED', 'AVAILABLE_VERIFIED', 'AVAILABLE_VERIFIED', 'AVAILABLE_VERIFIED', 'AVAILABLE_VERIFIED', 'TELEMETRY_INCONSISTENT'], 'execution': '/data/zypan/experiment_runs/LING_KDA_P1_L7_PROSPECTIVE_EXTENSION_4DOC_V1/execution_05', 'driver': '580.178.04', 'window_sha256': 'e6e760e9c98f800323db92dd70c0f9b2e8152932b278a64855887bed8c815d5a', 'window_path': '/data/zypan/experiment_runs/LING_KDA_P1_L7_PROSPECTIVE_EXTENSION_4DOC_V1/execution_05/admission/pg19-test-26183__H_to_L6_a50__parent.window.jsonl'})\n"
}
```

首个STOP保留no_retry，失败后不自动重算或放宽门禁。只有自然退出、实际profile合格且原数值资格通过的结果才计合法单元；raw worker PASS不等于数值资格PASS。实际完成与NOT_RUN/FAILED逐项如矩阵，历史诊断和fixture不作为正式结果。

## H. 科学解释与停止

INCOMPLETE_NOT_CONFIRMATORY。独立单位为文档，本轮仅gap≥3短程分歧筛选条件下4文档小样本前瞻，不将384点、层或head伪重复，不将旧2文档并为确认性n6，不做confirmatory p-value。结果不能证明P1梯度可优化性、总体稳定优势或reasoning/AIME改善。完整交付后停止；新Loss/L8/L9/STE/Cayley/optimizer/AIME/free generation/Qwen未授权且未运行。

## 原15问

1. 原成功与execution01/02/03/04 manifest、预注册和首STOP：Phase0实际SHA核验PASS。
2. 输入100/20/22/58资格与raw/token身份保留，无新tokenization或择样。
3. 完整64短评分复用，未重筛或重算。
4. 四篇41603/2544/7987/26183不变。
5. 各候选分数/midrank/gap和固定预测：pg19-test-41603: H_to_L7_a50 vs H (gap=6.0/4.0)；pg19-test-2544: H_to_L6_a50 vs H_to_L7_a50 (gap=7.0/6.0)；pg19-test-7987: H_to_L7_a50 vs H_to_L7R_a25 (gap=6.0/4.0)；pg19-test-26183: H_to_L6_a50 vs H_to_L7R_a25 (gap=5.0/6.0)；完整值见原selection。
6. 四篇共同freeze早于新长程，原receipt字节保留。
7. 新目标数值、fresh64、observer和QDQ资格2/12，未继承旧标签。
8. 正式4FP/8student实际0/0，有效文档0/4。
9. 合法完整KL/Delta见E与真实CSV；其余NOT_RUN。
10. 文档计数P1=0、L7=0、UNRESOLVED=0，未运行不当0或负例。
11. 仅小样本条件性前瞻，不做确认统计与伪重复。
12. 无额外第三候选，不推断L7-a50普遍最优或旧三旋转全排序。
13. 分执行/stage资源、launch-to-exit和CUDA/NVML真实scope见F。
14. commit/diff/数学AST/6配置/模型/输入/旋转/FP/CP/runtime/UUID/计分身份均SHA绑定。
15. 机器终态如下；交付后停止额外工作。

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
  "QUALIFIED_CONDITIONS": 2,
  "FORMAL_FP_COMPLETE": 0,
  "FORMAL_STUDENT_COMPLETE": 0,
  "ACTUAL_VALID_DOCUMENTS": 0
}
```
