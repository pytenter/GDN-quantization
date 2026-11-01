# QWEN_GDN_HADAMARD_DUAL_AXIS_DIRECT_32K_PILOT_V1 科学报告

最终执行状态：**FAIL_AND_STOP**；科学分类：**INCONCLUSIVE**。
实际完整配对文档 0/2，本任务累计占用 1.5062975197 GPUh / 24 GPUh；最终GPU释放已由协调器和资源账本核验。

本轮是独立前瞻性codec替换探索，比较共同实际FP_NATIVE目标下的H_C128与H_DUAL_M。旧execution_02仍为FAIL_AND_STOP：FP_NATIVE/FP_H relative-L2=0.006718461729707917，旧上限0.002；本次没有解决或重新判定该失败。

两篇文档在新GPU forward前冻结，统计单位为文档，目标N=2。repository/project独立性UNKNOWN；不同raw SHA不构成项目级独立性证明。两种学生都采用原正式FP32 tensor承载QDQ值：逐token真实量化写回语义与物理packed INT8资格分别报告，PHYSICAL_INT8_RUNTIME=NOT_QUALIFIED。

## 实际科学结果

正式32K Future-KL：**NOT_RUN**。没有文档同时具备三条合格完整32768轨迹与384个真实配对评分位置；canary诊断不填入正式科学表。没有将未运行或缺失值填0。

新数值分辨规则事前固定：每个保存logit对的FP32 KL与独立标量FP64参考之差，加条件FP64算术界，再按两臂误差的fsum/384及均值/相减舍入界汇总；未复用NoInflation的旧固定分辨率。该界依赖IEEE64/libm假设，约束reducer算术，不证明32K模型forward的完整数值不确定性；fresh-process资格限于真实128-token重复。小于分辨范围记NUMERICALLY_UNRESOLVED。

## 用户要求的八项回答

1. **Hadamard+Dual-M是否在两篇真实32K上更优？** 没有完整正式配对，无法回答两篇32K上是否优于C128；INCONCLUSIVE。

2. **平均绝对与相对Future-KL改善是多少？** NOT_RUN：没有正式384点Primary，不计算或外推两篇平均收益。

3. **Early/Middle/Late哪个窗口受益、是否一致？** NOT_RUN：Early/Middle/Late各128点窗口尚无完整正式配对；不沿用旧NoInflation的Late发现。

4. **真实scale、clipping和数值范围是否有新增风险？** 已完成的3条Dual-M轨迹、合计384 token未触发逐layer/head严重scale规则；有效scale、clipping与越界的实际范围见runtime_cost.json。该结论只覆盖列明已完成轨迹，不能证明未执行32K的安全。

5. **Dual-M真实在线开销是否可接受？** 已记录实际整轨迹、逐token forward/QDQ与codec CUDA事件成本及峰值显存，见下方资源与runtime_cost.json。G4通过仅说明保守预测可在预算内执行；没有预注册部署延迟阈值，不能据此宣称生产在线开销可接受。 同文档实际D/A post-warmup整token中位耗时比：CANARY_ONLY=0.975529750731。

6. **如改善，是否值得独立多文档验证？** 当前证据不足以据此安排有收益前提的多文档验证；只有后续独立决策与授权才可启动新实验，本任务不追加样本。

7. **混合或负面时是否应停止当前路线？** 本轮按完成或失败/资源门禁到此停止。资格不足、未分辨或单篇结果不能等同于路线已被科学否定；不自动开始新attempt。

8. **是否支持与learned rotation互补？** 尚未验证。没有运行C6R/C7、learned rotation交互或C8训练；当前结果不能支持互补性，也不能声称互补性已被否定。

## 工程门禁与资源

| 项目 | 实际状态 |
|---|---|
| CPU_V1_SEAL | PASS |
| HADAMARD_STRUCTURAL_CORRECTNESS | PASS |
| RESOURCE_ADMISSION | PASS |
| CANONICAL_H_C128_PARITY | PASS |
| DUAL_M_GPU_PARITY | PASS |
| PERSISTENT_QDQ | PASS |
| FRESH_PROCESS_DETERMINISM | PASS |
| NUMERICAL_SAFETY | PASS |
| PRIMARY_384_POSITION_CHECK | NOT_RUN |

协调器原始结束状态：FAIL_AND_STOP。首个失败文件：FIRST_FAILURE.json（原件不修改）。
GPU UUID：GPU-a78176ed-f97f-c54f-6201-5a01b570f422；累计1.5062975197 GPUh，含加载、资格、forward、codec、IO、checkpoint与释放前占用。

| 阶段 | 占用秒 | GPUh | 峰值NVML MiB（采样） | 释放核验 |
|---|---:|---:|---:|---|
| canary_initial | 671.269097716 | 0.186463638254 | 18284 | True |
| canary_fresh | 413.102344438 | 0.114750651233 | 18282 | True |
| science_doc1_FP_NATIVE | 4338.29962875 | 1.20508323021 | 24176 | True |

首个失败原始门禁/阶段：science_doc1_FP_NATIVE；错误：BrokenPipeError(32, 'Broken pipe')。
首个失败context：`{}`。

G4实际保守预测：status=PASS，六条32K剩余预测=47123.2967044秒，预计峰值reserved=20422066176 bytes，可接受设备上限=25169690624 bytes，安全余量=900秒。
预测包括post-warmup p95×1.6、每轨迹加载、四次checkpoint投影和score IO；最终reducer在释放GPU后只用CPU。

| 真实轨迹 | token数 | wall秒 | wall token/s | post-warmup token/s | peak reserved bytes |
|---|---:|---:|---:|---:|---:|
| N_OFF | 128 | 37.8185263127 | 3.38458455366 | 7.71663121433 | 18016632832 |
| A_REFERENCE | 128 | 173.309227392 | 0.738564252614 | 0.835892524747 | 18039701504 |
| A_OFF | 128 | 42.7325665504 | 2.9953735601 | 6.54287741237 | 18039701504 |
| A_ON | 128 | 171.402794391 | 0.746778956871 | 0.849986092657 | 18079547392 |
| D_OFF | 128 | 40.8125466555 | 3.13629044226 | 6.70699935852 | 18811453440 |
| D_ON | 128 | 170.414621502 | 0.751109258535 | 0.856396618012 | 18811453440 |
| N_FRESH | 128 | 37.5356649011 | 3.41009011928 | 7.81047652744 | 18016632832 |
| A_FRESH | 128 | 169.915322781 | 0.753316404344 | 0.854586543719 | 18039701504 |
| D_FRESH | 128 | 169.208203793 | 0.756464504268 | 0.857679351715 | 18809356288 |
| science_doc1_FP_NATIVE | 32768 | 4303.01387843 | 7.61512765837 | 7.65879518413 | 25052577792 |

| 实际轨迹 | scale观测最小值 | scale观测最大值 | rounded clamp均值 | unrounded越界均值 | local-QDQ相对MSE均值 |
|---|---:|---:|---:|---:|---:|
| A_REFERENCE | 7.87401573658e-15 | 0.201626420021 | 0 | 0.00142045021057 | 3.73539664793e-05 |
| A_OFF | 7.87401573658e-15 | 0.201626420021 | 0 | 0.00142045021057 | 3.73539664793e-05 |
| A_ON | 7.87401573658e-15 | 0.201626420021 | 0 | 0.00142045021057 | 3.73539664793e-05 |
| D_OFF | 5.47473177903e-12 | 0.202273607254 | 9.92774963464e-05 | 0.00411275243387 | 1.21742965348e-05 |
| D_ON | 5.47473177903e-12 | 0.202273607254 | 9.92774963464e-05 | 0.00411275243387 | 1.21742965348e-05 |
| A_FRESH | 7.87401573658e-15 | 0.201626420021 | 0 | 0.00142045021057 | 3.73539664793e-05 |
| D_FRESH | 5.47473177903e-12 | 0.202273607254 | 9.92774963464e-05 | 0.00411275243387 | 1.21742965348e-05 |

D的scale列为effective_scale[i,j]=row_scale[i]×column_scale[j]；A为column scale，未把D的row scale单独与A比较。表中为固定采样位置的观测汇总，per-layer/head完整风险门禁仍在每个token执行。

C128的分段事件在固定采样位置观察原函数，包括decode与有限性检查；Dual-M事件覆盖batched24层encode/decode/writeback及单独安全/指标审计。CPU观察与隐式同步可影响wall耗时，完整实际reservation账本优先于把CUDA事件之和当作全部成本。
未采集匹配FP_H整条state轨迹，因此共同FP state relative-MSE为NOT_MEASURED。A/D各自在自身输入状态上的local-QDQ误差仅为探索性诊断，不能归因于单独row scale或Query importance。

## 冻结文档与身份

| Doc | document ID | raw SHA256 | input32768 canonical SHA256 | project |
|---|---|---|---|---|
| 1 | long_bench/repobench-p/test/82835656a00f4dec3b750a7d2b33452f34c511d6ac6d56fa | 124c9386adaad5b335f1b6c49cfa0c0b8acb0ddaab28756980f8ed37f4680a17 | 84b031ca7a2d44e16f5618e93c6ce4f2a1ada936daed0d98852682b1a79bcbd4 | UNKNOWN_NOT_IN_SNAPSHOT_SCHEMA |
| 2 | long_bench/repobench-p/test/8aea8877e306036cae02eb7b51698639fc1965690e3a96d6 | 256363460cbdefd4ef8dbc0c82f676a201fe246a7872efae4a30eaf4e027d4fa | 29ebf4c6e8276b4d451b10b42138d41395511a559501d56dc585bb284feb32d8 | UNKNOWN_NOT_IN_SNAPSHOT_SCHEMA |

评分位置为0-based t=8192+64k，k=0..383；消费ids[t]后当前位置logits预测ids[t+1]。每轨迹消费32768 token；两种学生都从token0建立自己的QDQ历史，不使用旧anchor。

| 文件 | 当前SHA256 |
|---|---|
| FIRST_FAILURE.json | 76a43b0929a82bce1f713e42d507e8567015e791faffcfa1911642a7d8de7d47 |
| G4_RUNTIME_BUDGET.json | 833a63d75d715ebe6534c1ac8da89a9bfe1368006727d057d7bea178eab413dd |
| GATE_STATUS_FINAL.json | 3e4636819d983884be0b2f61c41763131e95e39bfe70d47b845c9b0446e99ccb |
| analysis/A_FRESH.json | 78478abaa81e9ac1c333eb93b00e7ea55703ae102412cae1f9f603d302fa6670 |
| analysis/A_OFF.json | 0a45c1beb130d5a96f2305b1bfb366e271c1a6f3b3ce8292e1756ec1e993c787 |
| analysis/A_ON.json | b76246989915c68aea6941b11a38bf5aecfed96b630af5f8e204101197bd06df |
| analysis/A_REFERENCE.json | 49c52e60867eec87735aee718a5cfe5a2d52d4a042fcf1982ab52ff1d2980cc3 |
| analysis/CANARY_REDUCER.json | 1b1d1f55e066d720a4a7fcaf23850ec28d31f7daefaa5df35725d803760611c2 |
| analysis/D_FRESH.json | 04c4fc4827d048807a738486a215600a75937b650987fefe4cc7f5177711a681 |
| analysis/D_OFF.json | a45c922f8ae591eca66fa2b776f1295cc112aaa1ee3f7b97a00d4b8edafaa9c5 |
| analysis/D_ON.json | fabe9894a45d88b3b60a32f79fa2038f711a42162852a48d393433d8dfc24c81 |
| analysis/ENV_canary_fresh.json | 7af45b37753d11984a6d34cee46199d51a82f7fc3f1226319b614a9cbfe5b394 |
| analysis/ENV_canary_initial.json | 89abf58cf476ced8dabc4d206d9f763690f529d1370b3f42034a0a86507de48a |
| analysis/ENV_science_doc1_FP_NATIVE.json | 3617fc39505b93f509a0b03ddffecbf038abc5aceb48fc2b6db067765e17e0ee |
| analysis/N_FRESH.json | fc08b0dc31588682f564a5129bf9fd226c9c9e95e6f2ac98800adddf157a5b0f |
| analysis/N_OFF.json | 9fe6deab74798d79fa264a9208abbc267a1d5fa839028b553c061932278c399e |
| analysis/POSTPROCESS_STATUS.json | 7abdb70252a31fbb5b2cb2b6b09565b9e15aaf571af6fa1e7c8124d7f1bbadea |
| analysis/RECEIPT_canary_fresh.json | d4f2408fdac712552559f6cbab4496dd722344095d29a0a042d96c2a376af5a2 |
| analysis/RECEIPT_canary_initial.json | 71f6e38a6cdf0b7c2a6f8e98e63546a8cd2789e8c439d4cd0ce41d7ad7fd99f3 |
| analysis/RECEIPT_science_doc1_FP_NATIVE.json | a8d79776284ba76852941385e901ceace00d5c566b8805252df6d8735257292f |
| analysis/science_doc1_FP_NATIVE.json | 30f3b799c994d24ed6685d2b680285fbc8641cb9aab6d18d2d4c29e2888e980c |
| analysis_scripts/finalize_report.py | 627e18124348ef08d9a810975120028ae2722b86e0adf66a1b3f7ac12e9a98eb |
| checkpoints/CATALOG.jsonl | 5995bf0a3aad538345676c7b44a8edc0314f7e34fa14a753d326da7940b6c5d1 |
| checkpoints/CURRENT.pt | cb937eac15179cad3918b9881f4f4ed6e487be8ea4e610d7ac940b76db39c9a4 |
| frozen_documents.json | 9abebe06044341b829acea095e18dcc1571197f8a66f178f6721c8b6085bc02f |
| gpu_admission.json | a3195eddff88dacf1fdacfb20fcd1580cd66ef85dbf4c528feae4c02bcd396e2 |
| gpu_resource_ledger.json | de8d6f667448387e66aee660d2f18efd3ac2b6b209d5a5b1552c7899b362963d |
| preregistration.json | 544da3293e5f819cbfa4953ed8381e30126342dfd47ebd52bfbcde77ea04fb5b |
| qualification_cpu/CPU_STRUCTURE_001.json | f4236fcfc4a5f0a6c389af63a7d1091373eaddaf2d0fc2f872559a9e29b888ee |
| qualification_cpu/MODEL_CONTENT_IDENTITY.json | 43c2a981fe79a5a00d41a5d2cf8bb250595d523b85f24f616ef60f5d227ea222 |
| source_manifest.json | eaf0395728a0c57efbdbc28cb6d062803896c8c132cc4c5dd35687267f6eb1d4 |
| src/codec.py | ece859b901145c6d9d646b8befd01916037c9117ef4a80f7b801f27f79f8c23f |
| src/coordinator.py | 28647e7d03b9d64f88c8e882bc27ae296053187874ee8431036dfeddbd99e065 |
| src/core.py | aa338b8b83cde62b97c94fd205d1c62999f783e7ec3242cbb813dc877e01f575 |
| src/reducer.py | 7e94c95803acb7ae47f0e3ec44fa97faf7cc8d93c4ecc5791a56ea2cf305b84f |
| src/worker.py | e04ec85092ee4f2207699366b2aaa61e1e71038e728c9234477b5b2c21794773 |

完整源码与依赖身份见source_manifest.json；模型14文件全流SHA见qualification_cpu/MODEL_CONTENT_IDENTITY.json。score文件和每个384位置logit身份由analysis/SCORE_ARTIFACT_VERIFICATION.json在实际存在时绑定。checkpoint版本SHA见checkpoints/CATALOG.jsonl，只有滚动CURRENT.pt为当前保留资产；退役版本按事前保留规则替换，不伪称全部仍在磁盘。最终交付seal由独立交付核验写入。

## 证据类别

**VERIFIED_ENGINEERING_RESULT**：仅上表真实PASS的CPU身份、结构、资源与GPU数值/执行门禁。未运行阶段保持NOT_RUN。
**EXPLORATORY_SCIENTIFIC_RESULT**：仅完整真实384点配对结果。N=2；不作confirmatory显著性宣称。
**NOT_ESTABLISHED**：此前FP_NATIVE/FP_H全模型数值等价、广泛泛化、AIME收益、learned-rotation互补、physical packed INT8部署与C8训练均未建立。

## 用户要求的最终字段

```text
TASK = QWEN_GDN_HADAMARD_DUAL_AXIS_DIRECT_32K_PILOT_V1
GPU_AUTHORIZATION = YES
SCIENTIFIC_PROTOCOL = NEW_PROSPECTIVE_CODEC_COMPARISON
OLD_FP_NATIVE_H_FAILURE = PRESERVED
CPU_V1_SEAL = PASS
HADAMARD_STRUCTURAL_CORRECTNESS = PASS
CANONICAL_H_C128_PARITY = PASS
DUAL_M_GPU_PARITY = PASS
PERSISTENT_QDQ = PASS
FRESH_PROCESS_DETERMINISM = PASS
NUMERICAL_SAFETY = PASS
RESOURCE_ADMISSION = PASS
DOCS_FROZEN = 2
FP_NATIVE_32K = 1/2
H_C128_32K = 0/2
H_DUAL_M_32K = 0/2
VALID_PAIRED_DOCS = 0/2
PRIMARY_384_POSITION_CHECK = NOT_RUN
DUAL_M_FULLMODEL_RESULT = INCONCLUSIVE
GPU_HOURS_USED = 1.5062975196954276
MAX_GPU_HOURS = 24.0
PHYSICAL_INT8_STORAGE = NOT_QUALIFIED
AIME = NOT_RUN
C6R_INTERACTION = NOT_RUN
C8_TRAINING = NOT_AUTHORIZED
FINAL_EXECUTION_STATUS = FAIL_AND_STOP
```

报告绝对路径：/data/zypan/analyses/qwen_hadamard_dual_axis_direct_32k_pilot_v1/experiments/qwen_gdn/quantizer/QWEN_GDN_HADAMARD_DUAL_AXIS_DIRECT_32K_PILOT_V1/FINAL_SCIENTIFIC_REPORT.md
本任务到此STOP；不运行后续confirmatory实验、AIME、C8、自动后台任务或git push。
