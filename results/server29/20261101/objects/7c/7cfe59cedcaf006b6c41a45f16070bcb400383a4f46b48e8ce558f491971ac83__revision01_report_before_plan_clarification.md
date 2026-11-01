# Ling/KDA 四文档前瞻实验 qualified completion

科学ID：LING_KDA_P1_L7_PROSPECTIVE_EXTENSION_4DOC_V1；本轮独立身份：execution_03。结论：RUNTIME_ENVIRONMENT_GATE_STOP。ENGINEERING_COMPLETE=False；NUMERICAL_QUALIFICATION_COMPLETE=False；FORMAL_SCIENCE_COMPLETE=False。原两文档正式 MIXED 永久保留，旧 posthoc 面板不变。

## 本次启动失败：已核实原因与真实状态

本次未能进入模型加载或GPU逐token科学计算。启动器遗漏了冻结runtime要求的 `PYTHONHASHSEED=0`，两个worker均在 `v2_runtime.initialize()` 第9行环境断言处退出。GPU2/3物理准入均PASS，故障来自启动配置；不是OOM、GPU占用或KL/量化数值失败。首失败STOP SHA `094a5e22dfa5a18e8eec62a126cbb0982f4ad9f9afd7b0493a28fd7ce1adb72d` 原样保留，`no_retry=true`，没有覆盖STOP或重跑。

37项CPU synthetic测试未覆盖真实启动器到冻结initializer的完整环境传递，不能代替GPU资格。修复候选现已补齐三个冻结环境变量并通过真实冻结initializer的CPU子进程正例、重复解释器hash一致性和缺失seed反例检查，CUDA_VISIBLE_DEVICES为空、无模型推理。这只是下次启动的工程修复证据，未应用为本轮GPU重试，未生成新的科学PASS。

ENGINEERING_COMPLETE=false；NUMERICAL_QUALIFICATION_COMPLETE=false；FORMAL_SCIENCE_COMPLETE=false。目标设备资格0/12、FP0/4、student0/8、valid0/4。历史GPU6完整8/12和9个parent/checkpoint保持原身份。两次实际初始化尝试在资格矩阵中标为FAILED_PRE_MODEL_RUNTIME_ENVIRONMENT_GATE；其他尚未启动项为NOT_RUN。没有Primary KL/Delta、没有新文档胜负，也没有空科学CSV。

新增资源按冻结worker launch-to-exit合同为 `0.0030530354711744518` GPUh，即 `10.990927696228027` 秒，两条均为启动到退出的墙钟，含CUDA前CPU导入/初始化；不能理解为GPU kernel计算时间。新CUDA allocated/reserved峰值没有runtime测量，标UNKNOWN/未测；保留历史已测峰值及其scope。两worker和tmux均已退出。下一次若获用户明确重启指令，应使用新的独立execution，保留execution03失败证据并重做资源准入和受影响资格，不能在本执行静默重试。

本修订仅更正CPU交付中的工程完成标签、已尝试失败状态及未测峰值表达，未改任何科学输入、checkpoint、资格阈值、数学、worker记录或原STOP。原自动封存与差异SHA均可核验。

## A. 原资产审计

原预注册 SHA=54d6d24fbc3142a692ef1a6c04a5617cf6b0abe6702a662722f34fc1b6141722；四篇 freeze SHA=a5c227d15918a249f8ae2059539c15f94668cb2afddf3aecd8a9f7e13851354b；freeze 北京时间2026-10-14 10:44:26。Phase0逐文件核验原成功manifest、execution01/02 manifest和SHA清单、64条评分、9个真实checkpoint、8个完整PASS和第9 parent。旧目录全程只读。新科学文档和旋转未改：pg19-test-41603: H_to_L7_a50 vs H (gap=6.0/4.0)；pg19-test-2544: H_to_L6_a50 vs H_to_L7_a50 (gap=7.0/6.0)；pg19-test-7987: H_to_L7_a50 vs H_to_L7R_a25 (gap=6.0/4.0)；pg19-test-26183: H_to_L6_a50 vs H_to_L7R_a25 (gap=5.0/6.0)。79项源资产含模型权重实际字节SHA已复核。新增CPU审计字段错误与原修复记录均保留。数学helper、原quantizer/forward/scale/KL reducer不变，实际新源码commit=55509063131e88f5040fbdf85452eef3145e41eb，diff与AST核验见SOURCE_DIFF.patch和NUMERICAL_SOURCE_COMPATIBILITY_AUDIT.json。

## B. GPU稳定性和使用证据

用户最新明确授权GPU2/3。当前目标设备绑定为{'GPU-759526ed-efb2-e7f6-881c-ab098e0b05ae': 3, 'GPU-7d6246ed-c3d1-75bc-c2c7-d02eb2882cca': 2}；本轮实际60秒预检结果见GPU_PREFLIGHT_RESULT_02/03，逐worker另有原冻结准入记录。昨日准备窗口的GPU4/5仅是历史准备证据；系统Xorg图形约4MiB显式保留PID/UID/starttime及双API身份。最初新增审计将任何graphics一概判busy，超过原冻结compute准入定义；原窗口原样保留，解释修正在GPU_PREFLIGHT_INTERPRETATION.json，未放宽util/内存/60秒/final3条件。启动时非目标GPU占用见本轮只读inventory，没有使用或终止非目标进程。主机PID namespace及不可读驱动日志仍UNKNOWN；实际每个worker要求SMI/NVML compute exact自有PID/UID/starttime/UUID，图形身份保持冻结基线。所有worker都另经一次原60秒准入；预检不冒充正式worker准入。

## C. 资格

目标设备真实合格 0/12；历史GPU6 8/12保持原标签。9个已完成parent没有重算，跨设备fresh64须逐条件比较全部64 full-V logits SHA、完整native cache、audit state、checkpoint SHA，恢复tensor指纹/RNG及observer ON/OFF exact；新设备标签只有真实通过后才生成。第四篇只计算其原缺3个parent及fresh。数学与数值门禁、精确比较方式未改。逐条件状态见QUALIFICATION_MATRIX.json；parent不等于完整PASS。

## D. 正式32K

FP 0/4；student 0/8；valid文档 0/4；planned12轨迹。资格全部12通过且资源PASS后才正式运行，原冻结文档順序逐篇FP→P1→L7，全局串行；双卡只加速资格阶段。同文档FP/P1/L7在同一目标UUID/runtime配对。原8192 checkpoint是其自身token0量化/FP历史的真实合法前缀；已授权设备迁移且fresh64 bitwise证明完整状态/缓存兼容，再由正式first64 observer复验，绝不把8256当32768。前3篇prefix历史GPU6、正式continuation设备=GPU-7d6246ed-c3d1-75bc-c2c7-d02eb2882cca，第四篇设备=GPU-759526ed-efb2-e7f6-881c-ab098e0b05ae；不会声称整条轨迹始终在同UUID。INT8 code经QDQ后由FP32载体承载量化值，写回并被下一kernel消费；不称physical resident INT8 storage。

## E. Primary结果

NOT_RUN：没有完整有效四文档Primary；未执行项不填0、不生成空科学CSV。
V=157184，原FP32 KL(FP||student)，消费ids[t]后logits预测ids[t+1]，t=8192+64k，k0..383，末32704，模型实际endpoint32768。实际完整student逐点有限且384齐全，严格原Python sum/384（没有新求和算法）。Delta=K_P1-K_L7，阈值3.3300064494019496e-6仅原经验数值分辨率。P1=0，L7=0，UNRESOLVED=0，MISSING=4。原协议不报告confirmatory p-value；即使独立无平局且p0=.5下4/4同方向，单侧exact sign .0625也不构成5%确认。本轮仅短程分歧及gap≥3条件下的小样本前瞻解释，document为单位，不将384点/层/heads当独立n，不合并旧2篇为n6。E_future/Hidden KL=NOT_FROZEN_NOT_COMPUTED。

## F. 实际资源及交付

execution01复用 2.2685737765497631 GPUh；execution02复用 3.6115049617158039 GPUh；execution03新增 0.0030530354711744518 GPUh；累计 5.883131773736741 GPUh。本轮自然退出worker 0、失败退出 2；未获退出receipt的时段仅独立上界，不能伪装精确计量。总计按每worker launch-to-exit求和，含加载/退出，排除CPU/准入空等；并行卡小时分别相加。本轮CUDA峰值未记录（在运行环境初始化前退出），不能以缺失测量冒充已测0 bytes；历史+本轮NVML定期owned采样峰值 17368.0 MiB。CUDA原loop reset scope、NVML periodic scope有明确限制，未声称whole-lifetime或连续峰值。真实checkpoint 9个，逐路径、bytes、SHA见CHECKPOINT_MANIFEST.json；FP真实payload见各formal manifest。服务器 /data/zypan/experiment_runs/LING_KDA_P1_L7_PROSPECTIVE_EXTENSION_4DOC_V1/execution_03；工作树 /data/zypan/worktrees/ling-kda-prospective-qualified-completion；本地交付 /Users/pytenter/Documents/临时项目1/ling_prospective_extension_4doc_v1/qualified_completion_execution_03/delivery/。大型FP/checkpoint留服务器绑定SHA，模型不下载。manifest与SHA256SUMS含manifest自身的SHA验证。

## G. 首个blocker与复用资产

```json
{
  "condition": "FP",
  "document": "pg19-test-26183",
  "no_new_training": true,
  "no_retry": true,
  "physical_resource_stop": false,
  "reason": "AssertionError()",
  "science_verdict": "INCOMPLETE_NOT_EVALUATED",
  "stage": "parent",
  "started_epoch": 1792024783.672028,
  "status": "STOP",
  "stopped_epoch": 1792024789.7382743,
  "traceback": "Traceback (most recent call last):\n  File \"/data/zypan/worktrees/ling-kda-prospective-qualified-completion/functional_worker.py\", line 172, in main\n    runtime = [initialize(torch, cfg['seed'])]\n               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/data/zypan/worktrees/ling-kda-p1-l7-efficient-recovery-v4/v2_runtime.py\", line 9, in initialize\n    assert os.environ.get('PYTHONHASHSEED')==str(seed)\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\nAssertionError\n"
}
```

实际成功单元/NOT_RUN见QUALIFICATION_MATRIX及机器状态。任何失败首条保留no_retry，不放宽门禁、不重选、不用fixture作新科学证据。已有成功字节可复用的范围仅已核验的相同科学key、source/runtime及实际已合格设备；新设备/新的恢复仍须资格，不自动沿用旧PASS。

## H. 科学解释与后续建议

INCOMPLETE_NOT_CONFIRMATORY。四篇仅有限比较证据，不能证明P1梯度可优化性、全部future Query因果机制、reasoning/AIME改善或总体稳定优势。若混合，后续可另行设计error propagation/readout observability研究；若方向一致，仍需更大独立预注册confirmation。建议不自动执行。训练授权NOT_AUTHORIZED；P1 Loss、L8/L9、STE/Cayley/optimizer、AIME/free generation/Qwen均未启动，交付后停止。

## 原15问逐项状态

1. 读取原成功、历史execution01/02 manifest与科学预注册：PASS，路径/SHA见Phase0。
2. 输入资格原100/20/22/58与真实tokens/raw SHA保留，新执行不重复编码或选文档。
3. 原完整4篇16旋转=64分数全部复用，未重筛。
4. 四篇41603/2544/7987/26183，完全冻结。
5. 预测与midrank gap：pg19-test-41603: H_to_L7_a50 vs H (gap=6.0/4.0)；pg19-test-2544: H_to_L6_a50 vs H_to_L7_a50 (gap=7.0/6.0)；pg19-test-7987: H_to_L7_a50 vs H_to_L7R_a25 (gap=6.0/4.0)；pg19-test-26183: H_to_L6_a50 vs H_to_L7R_a25 (gap=5.0/6.0)；全部候选、排名、规则原样。
6. freeze在所有新长程前；原时间和receipt字节未改。
7. 目标GPU numerical/fresh/observer/QDQ资格：0/12，身份及设备迁移证据逐项保留。
8. 正式FP 0/4，student 0/8，valid 0/4。
9. 完整精度KL、Delta见E与真实CSV，未完成值NOT_RUN。
10. 文档计数P1=0/L7=0/UNRESOLVED=0，未完成不算负。
11. 只作条件性小样本前瞻，不做confirmatory p-value，不伪重复。
12. 没有额外第三候选轨迹，不能据冻结两候选推断L7-a50普遍最优或复现所有旧三旋转排序。
13. 实际资源分执行账本及真实scope见F/RESOURCE_REPORT。
14. 实际source commit、路径、模型/输入/旋转/CP/FP/runtime/UUID/scoring SHA见封存清单。
15. 终态RUNTIME_ENVIRONMENT_GATE_STOP；未授权训练与额外科学任务均未执行。

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
  "FINAL_EXECUTION_STATUS": "RUNTIME_ENVIRONMENT_GATE_STOP",
  "ENGINEERING_COMPLETE": false,
  "NUMERICAL_QUALIFICATION_COMPLETE": false,
  "FORMAL_SCIENCE_COMPLETE": false,
  "QUALIFIED_CONDITIONS": 0,
  "FORMAL_FP_COMPLETE": 0,
  "FORMAL_STUDENT_COMPLETE": 0,
  "ACTUAL_VALID_DOCUMENTS": 0
}
```
