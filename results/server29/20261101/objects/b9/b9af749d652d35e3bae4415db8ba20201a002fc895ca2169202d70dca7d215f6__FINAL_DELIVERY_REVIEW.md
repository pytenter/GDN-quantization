# Ling/KDA execution_05 最终交付复核（STOP）

科学ID：`LING_KDA_P1_L7_PROSPECTIVE_EXTENSION_4DOC_V1`。本报告为终态后的附加CPU复核，原`FINAL_SCIENTIFIC_REPORT.md`、根manifest、源码、首STOP和科学资产均保持原字节。

|状态|实际结果|
|---|---|
|ENGINEERING_COMPLETE|true：CPU启动/配置检查及双GPU canary通过|
|NUMERICAL_QUALIFICATION_COMPLETE|false：2/12，且仅FP条件|
|FORMAL_SCIENCE_COMPLETE|false：FP 0/4，student 0/8，有效文档0/4|
|最终执行状态|FAIL_CLOSED_STOP，未重试，未自动启动后续执行|

## A. 原资产及修复身份

预注册SHA `54d6d24fbc3142a692ef1a6c04a5617cf6b0abe6702a662722f34fc1b6141722`，四篇共同freeze SHA `a5c227d15918a249f8ae2059539c15f94668cb2afddf3aecd8a9f7e13851354b`；FROZEN_SPEC SHA `8ae8927ed37209efccbf19c672b9baeb46c0d7f272601f8f54d1f9609b64f3d1`。原freeze北京时间2026-10-14 10:44:26，早于新长程与任何本轮32K outcome（实际32K未运行）。原100→20历史排除/22不足长度/58资格顺序、实际4篇×16旋转=64评分、固定pair、所有原source/model/input/rotation身份均复用，未择样或改公式。Phase0重新验证历史manifest、逐文件SHA及其SHA清单：

|资产|manifest SHA|核验|
|---|---|---|
|execution_01|`202bf8f64d7a2bfeab93451a607d575f7cc7052ab96692d88de02176f536516b`|PASS|
|execution_02|`3d30ed6066e4f4be6c8db5302f53cf8a7c721c9f76bfc9f5dc62c50966209954`|PASS|
|LING_KDA_P1_L7_MIXED_OUTCOME_RESOLUTION_V1|`ab4b85f3cc5f3f347ff9a22aae2322b09da41cb5908a599267a3c2d29281ac95`|PASS|
|execution_03|`1bd32a34923b52310c81762a45f92dbbb2ef13c05aa6e51ab552d52a7ee8128b`|PASS|
|execution_04|`1a483c637621113b7e8b2ecdb75d69f5612a303b6695e359b52ba1b7d0d5ce8f`|PASS|

execution01/02/03/04及旧MIXED/posthoc历史只读保留。独立修复commit `902d7974c89465df882403e96cc8dcc8dc1c0e85`，基于`d3cd88d4643a460e027cae6ba6989628c21f60bf`。源码冻结115文件含6个原FLA JSON和provenance，98个未涉及原Python文件逐字节不变；数学worker、helper、量化器、rotation、KL/reducer及容差未改，资格比较AST与CPU Primary校验原字节保持。变更范围为launcher/env/envelope/退出监督、测试与CPU交付、worker计划计数事件；diff及SHA见`SOURCE_DIFF.patch`、`NUMERICAL_SOURCE_COMPATIBILITY_AUDIT.json`。

根因定位进行了5个预先有界诊断（科学单元0）。同GPU、同进程、同checkpoint/RNG/observer的A→B→A干预证明conv配置BD128/warps4直接产生旧失败fingerprint、原BD8/warps8产生旧parent fingerprint、回A再复现。旧execution04当时实际kernel配置未捕获仍UNKNOWN。修复恢复原冻结6个FLA配置的`FLA_CACHE_MODE=default`及明确配置目录，并保留解释器启动前的`PYTHONHASHSEED=0`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`、`LING_GATED_RMSNORM_FIXED_CONFIG=BT16_W8`；没有删initializer断言或在worker内补seed。

37项工程检查、10个真实launcher→master/lane→unit_runner→冻结initializer CPU用例（4正6负）、11项profile/envelope检查及4项交付身份检查均PASS，无模型/CUDA。旧CPU fixture最初两个过期元数据/路径错误的FAIL日志保留，修正只涉及测试fixture身份与显式历史fixture路径，不是科学重试。

## B. GPU与首个停止门禁

仅GPU2 `GPU-7d6246ed-c3d1-75bc-c2c7-d02eb2882cca`和GPU3 `GPU-759526ed-efb2-e7f6-881c-ab098e0b05ae`。两卡canary实际环境、UUID、冻结initializer、真实6项FLA loader与正常退出均PASS，canary无模型/token，不算数值资格或32K科学轨迹。所有worker仍单独经过原60秒13点/final3严格0%util/allocated≤64MiB及SMI/NVML/进程归属门禁；Xorg约4MiB的PID4040/UID120/starttime3403显式绑定。

首STOP时间：2026-10-15T12:40:41.721605+08:00。GPU3准备启动`pg19-test-26183 × H_to_L6_a50 parent`，13个采样中前12个AVAILABLE_VERIFIED，最后一个SMI利用率**1%**、直接NVML利用率**0%**，原分类器判`TELEMETRY_INCONSISTENT`，也未满足原双零利用率条件。显存SMI15MiB与NVMLv2 15400960 bytes差0.3125MiB，内存算术一致，compute进程列表为空，已知图形身份未变；不能据此解释不一致的底层原因，该原因仍UNKNOWN。没有重复采样来覆盖失败、没有降低阈值或自动重试。原窗口SHA `e6e760e9c98f800323db92dd70c0f9b2e8152932b278a64855887bed8c815d5a`，首STOP SHA `390b6062be8ff2dca0ded48c9e9823477fc980460c0a535eee6d5851e56b588c`。

GPU2当时加载中的`41603 × H_to_L7_a50 parent`随后在安全token边界读到全局STOP并退出`SAFE_TOKEN_STOP`；这是首资源失败后的联动中止，不是额外观测到的INT8数值或resume失败。该GPU3条件未实际launch，不算一个GPU worker。所有本轮GPU worker、runner和lane身份已退出，tmux已结束，未杀他人或reset GPU。可读取驱动/host namespace的历史限制仍UNKNOWN，不推断硬件故障。

## C. 资格与checkpoint复用

原10个CP实际SHA保留，但旧完整0→8192 FLA配置缺乏运行记录，因此本轮计划在原12科学key下新建12 parent+12fresh；没有继承旧GPU6/04 PASS。实际只完成**2 parent+2fresh64**，均为FP：41603/GPU2和26183/GPU3。它们通过全部64步完整词表logits、最终native cache、同checkpoint及恢复状态/RNG/observer ON/OFF的原严格合同；FP audit_state=None是N/A，不是INT8 QDQ证明。4个实际model worker均有实际conv BD8/warps8和norm BT16/warps8的post-forward profile、/proc启动环境及NATURAL_EXIT绑定。包括此前26183/FP的失败恢复路径，本轮实际exact PASS；不能把两个FP通过扩展为8个INT8条件通过。

**本轮没有完成任何INT8条件资格**；persistent QDQ写回/下一kernel消费的当前整组资格仍INCOMPLETE，历史结果只保持历史身份。

|文档|条件|实际资格阶段状态|正式32768|
|---|---|---|---|
|pg19-test-41603|FP|PASS|NOT_RUN|
|pg19-test-41603|H_to_L7_a50|ABORTED_BY_GLOBAL_STOP|NOT_RUN|
|pg19-test-41603|H|NOT_RUN|NOT_RUN|
|pg19-test-2544|FP|NOT_RUN|NOT_RUN|
|pg19-test-2544|H_to_L6_a50|NOT_RUN|NOT_RUN|
|pg19-test-2544|H_to_L7_a50|NOT_RUN|NOT_RUN|
|pg19-test-7987|FP|NOT_RUN|NOT_RUN|
|pg19-test-7987|H_to_L7_a50|NOT_RUN|NOT_RUN|
|pg19-test-7987|H_to_L7R_a25|NOT_RUN|NOT_RUN|
|pg19-test-26183|FP|PASS|NOT_RUN|
|pg19-test-26183|H_to_L6_a50|ADMISSION_STOP_WORKER_NOT_LAUNCHED|NOT_RUN|
|pg19-test-26183|H_to_L7R_a25|NOT_RUN|NOT_RUN|

## D/E. 正式轨迹和Primary

4FP/8student均未运行，12条正式轨迹均NOT_RUN，无实际384位置Primary，无sum/384校验对象，不生成空壳科学CSV或把未运行项填0。固定原文档顺序及每篇FP→P1→OriginalL7保持。

|文档|P1旋转|Original L7旋转|P1 midrank gap|L7 midrank gap|K_P1|K_L7|Delta|
|---|---|---|---:|---:|---|---|---|
|pg19-test-41603|H_to_L7_a50|H|6|4|NOT_RUN|NOT_RUN|NOT_RUN|
|pg19-test-2544|H_to_L6_a50|H_to_L7_a50|7|6|NOT_RUN|NOT_RUN|NOT_RUN|
|pg19-test-7987|H_to_L7_a50|H_to_L7R_a25|6|4|NOT_RUN|NOT_RUN|NOT_RUN|
|pg19-test-26183|H_to_L6_a50|H_to_L7R_a25|5|6|NOT_RUN|NOT_RUN|NOT_RUN|

原评分合同仍为V157184、t=8192+64k (k0..383，末32704)、消费ids[t]后预测ids[t+1]、原FP32 KL(FP||student)、原Python sum/384，Delta=K_P1−K_L7。阈值3.3300064494019496e-6只保留为原经验分辨率，本次没有实际新环境Primary来检验它。P1/L7/tie-unresolved三个已观测计数均0，**缺失4**，不能称四个平局。没有新的独立前瞻科学方向证据，未复现或否定旧MIXED/posthoc候选选择现象。document为独立单位，不并n6，不将384点/层/head伪重复，不做confirmatory p。E_future/Hidden KL/new proxy不计算，未导出的layerwise指标不新造。

## F. 资源、模型与交付

本轮science execution05实际GPU worker **7**：canary2、FP parent2、FP fresh2、全局STOP中止的student parent1；自然退出6、失败退出1。物理admission未launch的条件不计worker。按实际launch-to-exit卡小时求和，含加载/退出，排除CPU/准入/等待，并行卡分别计入。

|stage|实际GPUh|自然退出|失败退出|
|---|---:|---:|---:|
|runtime canary|0.010547742578718398|2|0|
|parent（含中止）|0.82534228450722169|2|1|
|fresh64|0.022597851951917013|2|0|
|本轮新增合计|0.85848787903785717|6|1|

历史分账：execution01 2.268573776549763、execution02 3.611504961715804、execution03 0.003053035471174452、execution04 0.5042779713869094、5个诊断0.42219087839126584 GPUh；全部尝试合计7.6680885025527745 GPUh，未把历史/诊断算为本轮科学成功。

本轮已测CUDA allocated峰值16486794240 bytes、reserved峰值20128464896 bytes，为原loop-reset scope；中止加载worker没有完整loop峰值记录，不声称覆盖整个生命周期。NVML owned定期样本38个，采样峰值19536MiB，非连续瞬时峰值。5诊断未测CUDA reset峰值为UNKNOWN。未施加人为GPUh或显存百分比硬上限。

两个真实own-history8192 checkpoint（大型文件留服务器，不下载）：

- `/data/zypan/experiment_runs/LING_KDA_P1_L7_PROSPECTIVE_EXTENSION_4DOC_V1/execution_05/runs/pg19-test-41603__FP__8192.pt`；523116845 bytes；SHA `367ad2083285fd5a74c69ead6b84e9dbe443dfd22e628ab038cf0966cc10a0a7`。
- `/data/zypan/experiment_runs/LING_KDA_P1_L7_PROSPECTIVE_EXTENSION_4DOC_V1/execution_05/runs/pg19-test-26183__FP__8192.pt`；523116845 bytes；SHA `2e6abbe74fc5266fd26c3b74da4977cdd3849382534852503fb68a10e0f77f4f`。

它们的完整实际FPprefix均重新计算，不是旧CP软链接，完整文件SHA恰与历史对应CP一致；这个相等只证明这两个实际已复算FP条件，不外推其他条件。正式FP payload数量0；没有虚构full32K参考。模型权重仍服务器原路径，由`MODEL_WEIGHT_SHA256.txt`/原spec与source资产绑定。

服务器：`/data/zypan/experiment_runs/LING_KDA_P1_L7_PROSPECTIVE_EXTENSION_4DOC_V1/execution_05`；冻结工作树：`/data/zypan/worktrees/ling-kda-prospective-execution05`；本地全部小型封存文件：`/Users/pytenter/Documents/临时项目1/ling_prospective_extension_4doc_v1/engineering_repair_05/execution_05/delivery/`。原根manifest SHA `214f57ff93ac0208018f23af0c666172f4d625db58521fa87d0b0dd0afc44a8f`，283个原封存小文件/284条SHA记录（含manifest自身）已复核；本次新增复核报告、终态/证据审计及退出后稳定日志另以`delivery_review/DELIVERY_REVIEW_HASHES.json`和`DELIVERY_REVIEW_SHA256SUMS.txt`封存，原根manifest/SHA/STOP保持不变。下载后按两份清单逐项本地验证，详见本地`LOCAL_FINAL_DELIVERY_VERIFICATION.json`。

## G. 已保留资产与停止边界

首blocker是GPU3资源遥测门禁，非新观察到的resume数值失败或P1/L7胜负。可保留原64评分/四文档freeze、全部历史身份、原数学source、诊断因果证据、修复commit，以及本轮两个具完整启动profile和fresh64 exact证据的FP checkpoint/资格。未完成条件不继承PASS。资源门禁失败不能靠复用这些资产免除；后续若恢复，须独立执行身份、适用的新启动授权和原物理/数值门禁，不能在本STOP目录续跑。本次没有开始下一轮。

## H. 科学建议与15问

当前仅能说已修复的运行配置在两个FP条件上通过真实恢复资格，不能声称persistent INT8/P1优于Original L7、总体优势、可训练性或reasoning/AIME改善。下一步先解决并证实遥测稳定性与可归属，再讨论在同冻结问题下恢复未完成条件；这是建议，未执行新GPU诊断、训练或科学条件。

1. **读了哪些预注册/成功manifest，SHA是否通过？** 原NEXT_PROSPECTIVE_4DOC_PREREGISTRATION.md及正式协议freeze、原成功MIXED与execution01/02/03/04封存均读取并SHA PASS，详见A与Phase0。预注册与四文档完整SHA如A。
2. **100→58输入资格完全复现？** 历史真实retokenization的100/20/22/58、raw/token SHA与顺序保留并复核；本轮未再编码或择样。不能把历史复现冒称本轮重新tokenization。
3. **实际短程筛多少篇，是否≤40？** 原实际4篇，每篇16旋转，共64分数，在最多40篇内选齐四篇；本轮新增短程scoring0，合法复用64原分数，未重筛。
4. **最终四篇？** 41603、2544、7987、26183。
5. **P1/L7旋转与midrank gap？** 逐篇完整映射见D/E表（6/4、7/6、6/4、5/6），全部分数/ranks原样封存。
6. **freeze早于32K outcome？** 是，原freeze早于任何新长程；本轮正式32K没有开始，原receipt字节未改。
7. **所有门禁通过？** 否。启动工程/双canary通过；数值/observer/fresh资格只有两个FP PASS，INT8资格未完成；GPU3后续资源门禁FAIL。不能称12/12或persistent INT8整组合格。
8. **4FP+8student全部完成？** 否，正式FP0/4、student0/8、valid0/4；两个8256资格parent不计完整32K。
9. **四篇完整精度K_P1/K_L7/Delta？** 全部NOT_RUN，无合法Primary值，不填0。
10. **P1/L7/tie-unresolved各多少？** 已观测胜负计数0/0/0，缺失4；不是四个平局。
11. **有方向性独立前瞻证据？** 没有；新完整文档科学n=0，不能宣称P1增量预测价值或稳定优势。
12. **复制旧MIXED/Posthoc的候选选择现象？** 未检验，当前无新32K outcome，既不支持也不否定；旧MIXED/posthoc永久保留、不并n6。
13. **真实worker/GPUh/峰值？** 7 worker、0.85848787903785717新增GPUh；历史分账、CUDA16486794240/20128464896 bytes与38样本NVML19536MiB及其scope见F。无退出时段UNKNOWN未伪装0，本轮所有launch均已有退出receipt。
14. **最终commit/SHA/manifest/绝对目录？** commit `902d7974c89465df882403e96cc8dcc8dc1c0e85`；首STOP `390b6062be8ff2dca0ded48c9e9823477fc980460c0a535eee6d5851e56b588c`；原root manifest `214f57ff93ac0208018f23af0c666172f4d625db58521fa87d0b0dd0afc44a8f`；本地绝对目录 `/Users/pytenter/Documents/临时项目1/ling_prospective_extension_4doc_v1/engineering_repair_05/execution_05/delivery/`；新增review清单单独封存且SHA包含其manifest自身。
15. **严格STOP，无训练/L8/L9/AIME？** 是，全部自有worker/runner/lane退出，tmux结束；无梯度、新Loss/L8/L9/STE/Cayley/optimizer/AIME/free generation/Qwen或下一执行。交付后暂停本跟进。

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
  "QUALIFIED_CONDITIONS": 2,
  "SCIENTIFIC_INTERPRETATION_STATUS": "INCOMPLETE_NO_CONFIRMATORY_INFERENCE",
  "SHORT_HORIZON_SELECTION_STATUS": "COMPLETE_4_ELIGIBLE_FROZEN",
  "STUDENT_32K_STATUS": "COMPLETE_0_OF_8",
  "TRAINING_AUTHORIZATION": "NOT_AUTHORIZED"
}
```
