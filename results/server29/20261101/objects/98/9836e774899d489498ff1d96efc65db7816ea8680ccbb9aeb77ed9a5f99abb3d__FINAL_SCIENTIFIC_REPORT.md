# QWEN_GDN_HADAMARD_DUAL_AXIS_DIRECT_32K_PILOT_V1 科学报告

最终执行状态：**COMPLETE_AND_STOP**；科学分类：**POSITIVE_EXPLORATORY**。
实际完整配对文档 2/2，本任务累计占用 8.32214502533 GPUh / 24 GPUh；最终GPU释放已由协调器和资源账本核验。

本轮是独立前瞻性codec替换探索，比较共同实际FP_NATIVE目标下的H_C128与H_DUAL_M。旧execution_02仍为FAIL_AND_STOP：FP_NATIVE/FP_H relative-L2=0.006718461729707917，旧上限0.002；本次没有解决或重新判定该失败。

两篇文档在新GPU forward前冻结，统计单位为文档，目标N=2。repository/project独立性UNKNOWN；不同raw SHA不构成项目级独立性证明。两种学生都采用原正式FP32 tensor承载QDQ值：逐token真实量化写回语义与物理packed INT8资格分别报告，PHYSICAL_INT8_RUNTIME=NOT_QUALIFIED。

## 实际科学结果

| Doc | K_A | K_D | Delta | Relative Gain | 新算术分辨率 | Winner |
|---|---:|---:|---:|---:|---:|---|
| 1 | 0.0318188729701 | 0.0243192953821 | 0.00749957758799 | 0.235695890142 | 1.64759748877e-05 | H_DUAL_M |
| 2 | 0.0520338261948 | 0.0300131244169 | 0.0220207017779 | 0.423199741174 | 1.79368617063e-05 | H_DUAL_M |

K_A、K_D均由同篇384个真实位置的full-vocabulary FP32 KL通过固定math.fsum/384得到。Delta=K_A-K_D；RelativeGain=Delta/K_A仅在分母高于其合法算术分辨范围时提供。

| Doc | 窗口 | K_A | K_D | Delta | Relative Gain | 窗口算术分辨率 | Winner |
|---|---|---:|---:|---:|---:|---:|---|
| 1 | Early | 0.0260655539925 | 0.0300999445549 | -0.00403439056234 | -0.154778623293 | 1.63536411168e-05 | H_C128 |
| 1 | Middle | 0.023409654312 | 0.0129348169251 | 0.0104748373869 | 0.447458012293 | 1.46799675417e-05 | H_DUAL_M |
| 1 | Late | 0.0459814106057 | 0.0299231246663 | 0.0160582859394 | 0.349234304208 | 1.83943152843e-05 | H_DUAL_M |
| 2 | Early | 0.0211550845223 | 0.0162685150781 | 0.00488656944417 | 0.230987942356 | 9.86006832374e-06 | H_DUAL_M |
| 2 | Middle | 0.0728308957275 | 0.0475907560019 | 0.0252401397256 | 0.346558139557 | 2.72908429997e-05 | H_DUAL_M |
| 2 | Late | 0.0621154983348 | 0.0261801021707 | 0.035935396164 | 0.578525442561 | 1.66596730573e-05 | H_DUAL_M |

新数值分辨规则事前固定：每个保存logit对的FP32 KL与独立标量FP64参考之差，加条件FP64算术界，再按两臂误差的fsum/384及均值/相减舍入界汇总；未复用NoInflation的旧固定分辨率。该界依赖IEEE64/libm假设，约束reducer算术，不证明32K模型forward的完整数值不确定性；fresh-process资格限于真实128-token重复。小于分辨范围记NUMERICALLY_UNRESOLVED。

## 用户要求的八项回答

1. **Hadamard+Dual-M是否在两篇真实32K上更优？** 两篇均出现超过本轮算术分辨率的正向Delta，支持这两篇受测条件下Dual-M更优的探索性判断。

2. **平均绝对与相对Future-KL改善是多少？** 等权文档平均绝对Delta=0.014760139683；等权每doc RelativeGain平均=0.329447815658；均值KL的相对改善=0.352049244209。二者分母不同，均保留每doc原值。

3. **Early/Middle/Late哪个窗口受益、是否一致？** Doc1的最大原始Delta窗口为Late，其winner=H_DUAL_M；Doc2的最大原始Delta窗口为Late，其winner=H_DUAL_M。这只是各128点窗口描述，整篇384点Primary决定文档方向；不将Late或改善位置比例替代Primary。 两篇一致可分辨支持D的窗口：Middle, Late；一致支持A的窗口：无。其余窗口为方向不一致或含未分辨。

4. **真实scale、clipping和数值范围是否有新增风险？** 已完成的5条Dual-M轨迹、合计65920 token未触发逐layer/head严重scale规则；有效scale、clipping与越界的实际范围见runtime_cost.json。该结论只覆盖列明已完成轨迹，不能证明未执行32K的安全。

5. **Dual-M真实在线开销是否可接受？** 已记录实际整轨迹、逐token forward/QDQ与codec CUDA事件成本及峰值显存，见下方资源与runtime_cost.json。G4通过仅说明保守预测可在预算内执行；没有预注册部署延迟阈值，不能据此宣称生产在线开销可接受。 同文档实际D/A post-warmup整token中位耗时比：CANARY_ONLY=0.975529750731；Doc1_SCIENCE_FULL32768=0.960936201204；Doc2_SCIENCE_FULL32768=1.00867911778。

6. **如改善，是否值得独立多文档验证？** 值得另行预注册、独立授权的多文档验证；本任务到此STOP，不自行启动。N=2不支持显著性、广泛泛化或AIME提升。

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
| PRIMARY_384_POSITION_CHECK | PASS |

协调器原始结束状态：COMPLETE_AND_STOP。首个失败文件：无。
GPU UUID：GPU-a78176ed-f97f-c54f-6201-5a01b570f422；累计8.32214502533 GPUh，含加载、资格、forward、codec、IO、checkpoint与释放前占用。

| 阶段 | 占用秒 | GPUh | 峰值NVML MiB（采样） | 释放核验 |
|---|---:|---:|---:|---|
| canary_initial | 671.269097716 | 0.186463638254 | 18284 | True |
| canary_fresh | 413.102344438 | 0.114750651233 | 18282 | True |
| science_doc1_FP_NATIVE | 4338.29962875 | 1.20508323021 | 24176 | True |
| science_doc1_H_C128 | 5234.42236245 | 1.45400621179 | 24234 | True |
| science_doc1_H_DUAL_M | 4949.33031252 | 1.3748139757 | 20228 | True |
| science_doc2_FP_NATIVE | 4240.39446327 | 1.17788735091 | 24176 | True |
| science_doc2_H_C128 | 5097.49488892 | 1.41597080248 | 24176 | True |
| science_doc2_H_DUAL_M | 5015.40899314 | 1.39316916476 | 20228 | True |

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
| science_doc1_H_C128 | 32768 | 5199.5414668 | 6.30209417681 | 6.59761783721 | 25052577792 |
| science_doc1_H_DUAL_M | 32768 | 4913.09322384 | 6.66952539003 | 6.86582296404 | 20849885184 |
| science_doc2_FP_NATIVE | 32768 | 4204.65210122 | 7.79327259692 | 7.83503990497 | 25052577792 |
| science_doc2_H_C128 | 32768 | 5062.56191429 | 6.47261219808 | 6.82468286832 | 25052577792 |
| science_doc2_H_DUAL_M | 32768 | 4978.46133378 | 6.58195329903 | 6.76596030196 | 20849885184 |

| 实际轨迹 | scale观测最小值 | scale观测最大值 | rounded clamp均值 | unrounded越界均值 | local-QDQ相对MSE均值 |
|---|---:|---:|---:|---:|---:|
| A_REFERENCE | 7.87401573658e-15 | 0.201626420021 | 0 | 0.00142045021057 | 3.73539664793e-05 |
| A_OFF | 7.87401573658e-15 | 0.201626420021 | 0 | 0.00142045021057 | 3.73539664793e-05 |
| A_ON | 7.87401573658e-15 | 0.201626420021 | 0 | 0.00142045021057 | 3.73539664793e-05 |
| D_OFF | 5.47473177903e-12 | 0.202273607254 | 9.92774963464e-05 | 0.00411275243387 | 1.21742965348e-05 |
| D_ON | 5.47473177903e-12 | 0.202273607254 | 9.92774963464e-05 | 0.00411275243387 | 1.21742965348e-05 |
| A_FRESH | 7.87401573658e-15 | 0.201626420021 | 0 | 0.00142045021057 | 3.73539664793e-05 |
| D_FRESH | 5.47473177903e-12 | 0.202273607254 | 9.92774963464e-05 | 0.00411275243387 | 1.21742965348e-05 |
| science_doc1_H_C128 | 4.30038560317e-10 | 568.720336914 | 0 | 0.00141761079431 | 1.92973734513e-05 |
| science_doc1_H_DUAL_M | 3.88524767914e-11 | 0.281166553497 | 9.52639538957e-07 | 0.00402482097343 | 1.00713156156e-05 |
| science_doc2_H_C128 | 8.32737878653e-10 | 8322.52832031 | 0 | 0.00141791337066 | 1.85965610737e-05 |
| science_doc2_H_DUAL_M | 3.41344730259e-11 | 0.309987545013 | 1.04991101113e-06 | 0.00403496368745 | 9.64481975529e-06 |

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
| G4_RUNTIME_BUDGET.json | 833a63d75d715ebe6534c1ac8da89a9bfe1368006727d057d7bea178eab413dd |
| GATE_STATUS_FINAL.json | e4df72c824c7737985948fdf95e6be68117e76e9e9e420b352b79a01e7a54836 |
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
| analysis/ENV_science_doc1_H_C128.json | c9e90e997919763e59617e83a5f0f1c0d04cb0fe20cb8cda5fb1ff72389f58e1 |
| analysis/ENV_science_doc1_H_DUAL_M.json | e8d493f906e3aacaf494da410d3f46e0c5ce6a4ea6cd7c04c0ac442a7d46f1f4 |
| analysis/ENV_science_doc2_FP_NATIVE.json | a8b9d1e5e034e9f9f6a3672418bc4861e530293c4a8bcd880f9530e7ce7dfd91 |
| analysis/ENV_science_doc2_H_C128.json | b38e6e4299337fbbdba635bba0d032d65e8333865ac1d56a6f6e0bba2313a8f8 |
| analysis/ENV_science_doc2_H_DUAL_M.json | 391d2ddf1180a59956857f0c8e7899473f1460128617f0af6700770c7e13b6c5 |
| analysis/N_FRESH.json | fc08b0dc31588682f564a5129bf9fd226c9c9e95e6f2ac98800adddf157a5b0f |
| analysis/N_OFF.json | 9fe6deab74798d79fa264a9208abbc267a1d5fa839028b553c061932278c399e |
| analysis/POSTPROCESS_STATUS.json | ff552631a8ca30979de356a0d1e4484d927cbdca8cc39cdacc452f1427adbac3 |
| analysis/RECEIPT_canary_fresh.json | d4f2408fdac712552559f6cbab4496dd722344095d29a0a042d96c2a376af5a2 |
| analysis/RECEIPT_canary_initial.json | 71f6e38a6cdf0b7c2a6f8e98e63546a8cd2789e8c439d4cd0ce41d7ad7fd99f3 |
| analysis/RECEIPT_science_doc1_FP_NATIVE.json | a8d79776284ba76852941385e901ceace00d5c566b8805252df6d8735257292f |
| analysis/RECEIPT_science_doc1_H_C128.json | d98b3d43753075f6c089078042d98e8523f8e70e3057aaa5545784264e3fcb20 |
| analysis/RECEIPT_science_doc1_H_DUAL_M.json | 43bc376587181b445cecd44b0635595b07dc414cf58e833046d0bd8f7f86784f |
| analysis/RECEIPT_science_doc2_FP_NATIVE.json | e6a0460c15c8c073a1faab2b5b240d9444050a45e7c2789bffab3b22f13bd1a8 |
| analysis/RECEIPT_science_doc2_H_C128.json | 4bc4e93a6ea02e55a83d4d5f3592a5ac86235a6b3cf04cdce97cc17fd781a794 |
| analysis/RECEIPT_science_doc2_H_DUAL_M.json | 58d4456a65d1a55c85351f6425763cc2fb25d1b03876722c4a4d138d1c2f959b |
| analysis/SCORE_ARTIFACT_VERIFICATION.json | fd8e5eb0b78fd94dca4e8cf3082ac5d7238077dd0ab8a7527b499f634e82bed7 |
| analysis/science_doc1_FP_NATIVE.json | 30f3b799c994d24ed6685d2b680285fbc8641cb9aab6d18d2d4c29e2888e980c |
| analysis/science_doc1_H_C128.json | 98570854f8447b7b577dc70730dd53215c120499990faf231958adf306e277d8 |
| analysis/science_doc1_H_DUAL_M.json | 602c1a752a7816fb2f53cba10ae26144ae2d1e9be3b1eef83c686c5bc6d5b4ea |
| analysis/science_doc2_FP_NATIVE.json | cec5860c8cf55384d008ad1a2f80b6ed8a8d440e1460b0f5d30edcccd836eeea |
| analysis/science_doc2_H_C128.json | 7ab9699e32106cbcaf55acdf55dbb411f67d7c2043963bcd7c98e5ac29fda64d |
| analysis/science_doc2_H_DUAL_M.json | f34efec4373d0f2bca2d7654af7f1d6be4af375d03d4651430d2da0c93dc620f |
| analysis_scripts/finalize_report.py | 627e18124348ef08d9a810975120028ae2722b86e0adf66a1b3f7ac12e9a98eb |
| checkpoints/CATALOG.jsonl | aaa9c01f512b837528eb7a83a2c6502bcd7a1cadd94da73791069b50b7061a8c |
| checkpoints/CURRENT.pt | 9384a86d4463d736c9927a3816db1aebe5ef147c9b6b1467385cd42eb86cc4d1 |
| frozen_documents.json | 9abebe06044341b829acea095e18dcc1571197f8a66f178f6721c8b6085bc02f |
| gpu_admission.json | bcf8b67eb16a2d7e7fb0eb4a6f30d9c0f811d5c1901d2e43fd4f598491cbfb40 |
| gpu_resource_ledger.json | 77b290a19abc1d44001c37ab6f2fdf593387d4c2acdc4534f7daac6d871b84a1 |
| preregistration.json | 544da3293e5f819cbfa4953ed8381e30126342dfd47ebd52bfbcde77ea04fb5b |
| qualification_cpu/CPU_STRUCTURE_001.json | f4236fcfc4a5f0a6c389af63a7d1091373eaddaf2d0fc2f872559a9e29b888ee |
| qualification_cpu/MODEL_CONTENT_IDENTITY.json | 43c2a981fe79a5a00d41a5d2cf8bb250595d523b85f24f616ef60f5d227ea222 |
| science_results.csv | 0bad60352bf8dba02617f262fe5828d033ed76870bbf7edcd912d63554df5d0c |
| science_results.json | fa65a570771fad9746e0624b928e50f448ef1fa230f3389b3266b99fa69ec8be |
| source_manifest.json | eaf0395728a0c57efbdbc28cb6d062803896c8c132cc4c5dd35687267f6eb1d4 |
| src/codec.py | ece859b901145c6d9d646b8befd01916037c9117ef4a80f7b801f27f79f8c23f |
| src/coordinator.py | 28647e7d03b9d64f88c8e882bc27ae296053187874ee8431036dfeddbd99e065 |
| src/core.py | aa338b8b83cde62b97c94fd205d1c62999f783e7ec3242cbb813dc877e01f575 |
| src/reducer.py | 7e94c95803acb7ae47f0e3ec44fa97faf7cc8d93c4ecc5791a56ea2cf305b84f |
| src/worker.py | e04ec85092ee4f2207699366b2aaa61e1e71038e728c9234477b5b2c21794773 |
| window_results.csv | df8d601d221c503cb00fc24435a1d197bfe49af1fc42485055fe8353d57395fc |

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
FP_NATIVE_32K = 2/2
H_C128_32K = 2/2
H_DUAL_M_32K = 2/2
VALID_PAIRED_DOCS = 2/2
PRIMARY_384_POSITION_CHECK = PASS
DUAL_M_FULLMODEL_RESULT = POSITIVE_EXPLORATORY
GPU_HOURS_USED = 8.322145025332768
MAX_GPU_HOURS = 24.0
PHYSICAL_INT8_STORAGE = NOT_QUALIFIED
AIME = NOT_RUN
C6R_INTERACTION = NOT_RUN
C8_TRAINING = NOT_AUTHORIZED
FINAL_EXECUTION_STATUS = COMPLETE_AND_STOP
```

报告绝对路径：/data/zypan/analyses/qwen_hadamard_dual_axis_direct_32k_pilot_v1/continuation_01/experiments/qwen_gdn/quantizer/QWEN_GDN_HADAMARD_DUAL_AXIS_DIRECT_32K_PILOT_V1/FINAL_SCIENTIFIC_REPORT.md
本任务到此STOP；不运行后续confirmatory实验、AIME、C8、自动后台任务或git push。
