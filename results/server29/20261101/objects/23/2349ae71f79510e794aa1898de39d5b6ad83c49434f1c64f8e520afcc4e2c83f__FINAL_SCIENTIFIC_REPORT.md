# Ling/KDA 四文档 Clean-Room Final

独立TASK：LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1；本次单卡执行身份：LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU6_SINGLE_V1；原科学ID：LING_KDA_P1_L7_PROSPECTIVE_EXTENSION_4DOC_V1。
终态：PARTIAL_STOP。本轮仅更换控制层，旧execution01至05及前次Clean-Room STOP全部保留，未覆盖STOP/模型/CP。本次由用户新明确授权仅GPU6串行运行，不自动增加后续尝试。

## 科学身份与执行

原prereg54d6d24fbc3142a692ef1a6c04a5617cf6b0abe6702a662722f34fc1b6141722，四篇freeze a5c227d15918a249f8ae2059539c15f94668cb2afddf3aecd8a9f7e13851354b，FROZEN_SPEC 8ae8927ed37209efccbf19c672b9baeb46c0d7f272601f8f54d1f9609b64f3d1均实际SHA PASS。100/20/22/58输入流程与4篇×16=64冻结评分不变，无重排、重选或新训练。32份权重、raw/token、旋转及数学源码实际SHA见G0/source manifest。源码commit：ab7cdaa4a57065f630b5aa399f1cf0806c46ca3e。

冻结数学worker文件字节保持；正式编译入口仅添加已登记的checkpoint/cursor/输出原子化控制节点，逐项可逆还原整个原始源码，未改model_step/QDQ/rotation/observer/KL/原Python sum384。旧执行05的双零13点规则不追溯改变；新任务使用3次间隔采样，≤2%差异只提示，前提是无外部compute/身份异常、显存稳定且按历史max(reserved,NVML峰值)+至少2GiB空余准入。实际资源安全仍独立监督，每个模型进程来源/profile/UUID验证。

每卡仅一次真实模型8-token canary，非科学轨迹。数值资格12/12；INT8资格PASS，FP audit=None为N/A。同一所选UUID上的既有FP8192 CP只在当前GPU fresh64合同实际PASS后复用，设备不同时重建其FP前缀；INT8各有自己的从token0 persistent-QDQ历史。FLA Conv BD8/W8、Gated RMSNorm BT16/W8固定，五个启动变量在新解释器前设置。8192原fresh64、observer/nativecache/RNG精确合同比较未放宽。

正式逻辑顺序41603→2544→7987→26183，每篇FP→P1→OriginalL7，全局串行。新增一次计划16384 checkpoint，原cache/RNG/QDQ state及完整已处理评分列表/原reducer状态原子保存。第一段继续64步收集未中断对照，第二段fresh恢复16384，完整64 logits/cache/audit/RNG必须exact，重放评分不重复计入384数组。每条完整轨迹唯一逻辑consumed tokens32768，计划恢复验证额外64token单列计时；两个正式segment不冒充两条科学样本。非预期进程退出无静默重试。

## 实际结果

|Doc|P1 Rotation|Original L7 Rotation|K_P1|K_L7|Delta|Winner|
|---|---|---|---:|---:|---:|---|
|pg19-test-41603|H_to_L7_a50|H|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|
|pg19-test-2544|H_to_L6_a50|H_to_L7_a50|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|
|pg19-test-7987|H_to_L7_a50|H_to_L7R_a25|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|
|pg19-test-26183|H_to_L6_a50|H_to_L7R_a25|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|


原V157184，t=8192+64k (k0..383、末32704)，消费ids[t]预测ids[t+1]；原FP32 full-vocabulary KL(FP||student)，原Python sum/384。未完成项NOT_RUN，不填0、无空壳科学CSV。实际有效文档0/4，FP0/4、student0/8。P1胜0、L7胜0、数值未分辨0、缺失4。平均Delta=None，中位Delta=None。每篇相对效应在真实JSON/CSV中，独立单位document，最多n4；不把384点/层/head伪重复，不合并旧2doc为n6。

原阈值3.3300064494019496e-6为历史经验分辨率。正式结果揭示前依据冻结数学/软件/实际FLA profile、所选GPU self-KL零和各条件fresh64精确重复性冻结适用证据；这不是新32K误差上界校准。阈值内标NUMERICALLY_UNRESOLVED。原始完整精度Delta始终保留。未计算确认性sign-test；独立/对称假设未建立，即使4/4同向在该假设下单侧理论p也为0.0625。

## 资源与文件

{
  "new_GPU_hours": 4.296229118945387,
  "max_GPU_hours": 36.0,
  "actual_GPU_worker_count": 26,
  "peak_measured_CUDA_allocated": 16935037440,
  "peak_measured_CUDA_reserved": 21791506432,
  "successful_PROFILE_records": [
    {
      "tag": "pg19-test-2544__FP__canary",
      "kind": "canary",
      "profile_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU6_SINGLE_V1/canary/gpu6.PROFILE.json",
      "profile_sha256": "1d29c5f8b093af124d5afa0ab0a9023da96a7f8e3f3965394188d3472d3cc0d5",
      "scope": "last frozen loop reset through completion including checkpoint operations; canary resets before model load",
      "peak_allocated": 15856368640,
      "peak_reserved": 15864954880
    },
    {
      "tag": "pg19-test-2544__FP__fresh",
      "kind": "fresh",
      "profile_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU6_SINGLE_V1/runs/pg19-test-2544__FP__resumed_parity.PROFILE.json",
      "profile_sha256": "48d4d0ac515e37c6dfd4dc13f1f47f3ff31f01860ffabe65bde6a7f803d205ec",
      "scope": "last frozen loop reset through completion including checkpoint operations; canary resets before model load",
      "peak_allocated": 16486794240,
      "peak_reserved": 16783507456
    },
    {
      "tag": "pg19-test-2544__FP__parent",
      "kind": "parent",
      "profile_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU6_SINGLE_V1/runs/pg19-test-2544__FP__parent.PROFILE.json",
      "profile_sha256": "5c4479005ad1cd5081172d47875db2292d68ad8b09f0d01b1cad0dc18193e55b",
      "scope": "last frozen loop reset through completion including checkpoint operations; canary resets before model load",
      "peak_allocated": 16486794240,
      "peak_reserved": 20128464896
    },
    {
      "tag": "pg19-test-2544__H_to_L6_a50__fresh",
      "kind": "fresh",
      "profile_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU6_SINGLE_V1/runs/pg19-test-2544__H_to_L6_a50__resumed_parity.PROFILE.json",
      "profile_sha256": "fa583dc61e54fd027194a4f743f39ef9a461e9e994057d95963dc8137549770e",
      "scope": "last frozen loop reset through completion including checkpoint operations; canary resets before model load",
      "peak_allocated": 16935037440,
      "peak_reserved": 17265852416
    },
    {
      "tag": "pg19-test-2544__H_to_L6_a50__parent",
      "kind": "parent",
      "profile_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU6_SINGLE_V1/runs/pg19-test-2544__H_to_L6_a50__parent.PROFILE.json",
      "profile_sha256": "ea57edaa0986aef80a0869c80cf297379d0fb758cb2bd80465d1dbce24f303ea",
      "scope": "last frozen loop reset through completion including checkpoint operations; canary resets before model load",
      "peak_allocated": 16935037440,
      "peak_reserved": 20667432960
    },
    {
      "tag": "pg19-test-2544__H_to_L7_a50__fresh",
      "kind": "fresh",
      "profile_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU6_SINGLE_V1/runs/pg19-test-2544__H_to_L7_a50__resumed_parity.PROFILE.json",
      "profile_sha256": "ce39f2e1e8514986bb3a0d1b2e92a5774ed0c6227e887818b3b08d8c9bebc403",
      "scope": "last frozen loop reset through completion including checkpoint operations; canary resets before model load",
      "peak_allocated": 16935037440,
      "peak_reserved": 17265852416
    },
    {
      "tag": "pg19-test-2544__H_to_L7_a50__parent",
      "kind": "parent",
      "profile_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU6_SINGLE_V1/runs/pg19-test-2544__H_to_L7_a50__parent.PROFILE.json",
      "profile_sha256": "ed80e993ba95ee4acb03124ae8115ea6c8136deadf28714a5aa9490df244a66c",
      "scope": "last frozen loop reset through completion including checkpoint operations; canary resets before model load",
      "peak_allocated": 16935037440,
      "peak_reserved": 20667432960
    },
    {
      "tag": "pg19-test-26183__FP__fresh",
      "kind": "fresh",
      "profile_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU6_SINGLE_V1/runs/pg19-test-26183__FP__resumed_parity.PROFILE.json",
      "profile_sha256": "f0195e6adcc85506f94a936ad29b3e6639813077e43a198672f1e6140131ae23",
      "scope": "last frozen loop reset through completion including checkpoint operations; canary resets before model load",
      "peak_allocated": 16486794240,
      "peak_reserved": 16783507456
    },
    {
      "tag": "pg19-test-26183__FP__parent",
      "kind": "parent",
      "profile_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU6_SINGLE_V1/runs/pg19-test-26183__FP__parent.PROFILE.json",
      "profile_sha256": "ed25b5f52c8ad0d73dee4169c92bd875e6ee13302017eca2f01e7c43852fcf1e",
      "scope": "last frozen loop reset through completion including checkpoint operations; canary resets before model load",
      "peak_allocated": 16486794240,
      "peak_reserved": 20128464896
    },
    {
      "tag": "pg19-test-26183__H_to_L6_a50__fresh",
      "kind": "fresh",
      "profile_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU6_SINGLE_V1/runs/pg19-test-26183__H_to_L6_a50__resumed_parity.PROFILE.json",
      "profile_sha256": "e3343a0dda699f13e212878285e618676fd7c70f54b7d9c678938dc2dc397204",
      "scope": "last frozen loop reset through completion including checkpoint operations; canary resets before model load",
      "peak_allocated": 16935037440,
      "peak_reserved": 17265852416
    },
    {
      "tag": "pg19-test-26183__H_to_L6_a50__parent",
      "kind": "parent",
      "profile_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU6_SINGLE_V1/runs/pg19-test-26183__H_to_L6_a50__parent.PROFILE.json",
      "profile_sha256": "c9d03e0cffb74b6cb15d38580bf2412db003d87c86d60aa66ef5d1246eed8512",
      "scope": "last frozen loop reset through completion including checkpoint operations; canary resets before model load",
      "peak_allocated": 16935037440,
      "peak_reserved": 20667432960
    },
    {
      "tag": "pg19-test-26183__H_to_L7R_a25__fresh",
      "kind": "fresh",
      "profile_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU6_SINGLE_V1/runs/pg19-test-26183__H_to_L7R_a25__resumed_parity.PROFILE.json",
      "profile_sha256": "1b045e8050e31e86383d29c35a82366838090e7339fed2911985ed6793cd26f1",
      "scope": "last frozen loop reset through completion including checkpoint operations; canary resets before model load",
      "peak_allocated": 16935037440,
      "peak_reserved": 17265852416
    },
    {
      "tag": "pg19-test-26183__H_to_L7R_a25__parent",
      "kind": "parent",
      "profile_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU6_SINGLE_V1/runs/pg19-test-26183__H_to_L7R_a25__parent.PROFILE.json",
      "profile_sha256": "2c7489f49ebfccd4e0dc2459f02514d235d807318dd833d1b1f9f93fe9ebf6d7",
      "scope": "last frozen loop reset through completion including checkpoint operations; canary resets before model load",
      "peak_allocated": 16935037440,
      "peak_reserved": 20667432960
    },
    {
      "tag": "pg19-test-41603__FP__fresh",
      "kind": "fresh",
      "profile_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU6_SINGLE_V1/runs/pg19-test-41603__FP__resumed_parity.PROFILE.json",
      "profile_sha256": "12b861447f97cd1d7b50e442febac054a20195fd3d59155b8f816646d3de2d6e",
      "scope": "last frozen loop reset through completion including checkpoint operations; canary resets before model load",
      "peak_allocated": 16486794240,
      "peak_reserved": 16783507456
    },
    {
      "tag": "pg19-test-41603__FP__parent",
      "kind": "parent",
      "profile_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU6_SINGLE_V1/runs/pg19-test-41603__FP__parent.PROFILE.json",
      "profile_sha256": "658632011ae7e6a3e00a2f1da44dd1bf2b105174e99af2c02ffde3858a8ace86",
      "scope": "last frozen loop reset through completion including checkpoint operations; canary resets before model load",
      "peak_allocated": 16486794240,
      "peak_reserved": 20128464896
    },
    {
      "tag": "pg19-test-41603__H__fresh",
      "kind": "fresh",
      "profile_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU6_SINGLE_V1/runs/pg19-test-41603__H__resumed_parity.PROFILE.json",
      "profile_sha256": "7aa342e7bc73388231c4bc603a19c48044fe10fe351928490431ffba504e39f9",
      "scope": "last frozen loop reset through completion including checkpoint operations; canary resets before model load",
      "peak_allocated": 16935037440,
      "peak_reserved": 17265852416
    },
    {
      "tag": "pg19-test-41603__H__parent",
      "kind": "parent",
      "profile_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU6_SINGLE_V1/runs/pg19-test-41603__H__parent.PROFILE.json",
      "profile_sha256": "e0e41854b1bb1f315dcbc51eff487e9e264666061ea86e6a59d2949d2cdde178",
      "scope": "last frozen loop reset through completion including checkpoint operations; canary resets before model load",
      "peak_allocated": 16935037440,
      "peak_reserved": 20667432960
    },
    {
      "tag": "pg19-test-41603__H_to_L7_a50__fresh",
      "kind": "fresh",
      "profile_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU6_SINGLE_V1/runs/pg19-test-41603__H_to_L7_a50__resumed_parity.PROFILE.json",
      "profile_sha256": "8dfdecb0bbef14db6c8e9ffa85076e73c3a30495d2b5b174b356d5ac01f5254b",
      "scope": "last frozen loop reset through completion including checkpoint operations; canary resets before model load",
      "peak_allocated": 16935037440,
      "peak_reserved": 17265852416
    },
    {
      "tag": "pg19-test-41603__H_to_L7_a50__parent",
      "kind": "parent",
      "profile_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU6_SINGLE_V1/runs/pg19-test-41603__H_to_L7_a50__parent.PROFILE.json",
      "profile_sha256": "5187d3bc92532f0df4318ad8ac4c9ff0c2e16ed26f4977e8d3f51e6c3ac0957b",
      "scope": "last frozen loop reset through completion including checkpoint operations; canary resets before model load",
      "peak_allocated": 16935037440,
      "peak_reserved": 20671627264
    },
    {
      "tag": "pg19-test-7987__FP__fresh",
      "kind": "fresh",
      "profile_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU6_SINGLE_V1/runs/pg19-test-7987__FP__resumed_parity.PROFILE.json",
      "profile_sha256": "7764f4af651a218e037735e989e1f255397d0c03cedf5ec7f4636ada7019c461",
      "scope": "last frozen loop reset through completion including checkpoint operations; canary resets before model load",
      "peak_allocated": 16486794240,
      "peak_reserved": 16783507456
    },
    {
      "tag": "pg19-test-7987__FP__parent",
      "kind": "parent",
      "profile_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU6_SINGLE_V1/runs/pg19-test-7987__FP__parent.PROFILE.json",
      "profile_sha256": "742af670546cf567cf0977dca0413b4f22b22ad2042012be3eb40262a92e1940",
      "scope": "last frozen loop reset through completion including checkpoint operations; canary resets before model load",
      "peak_allocated": 16486794240,
      "peak_reserved": 20128464896
    },
    {
      "tag": "pg19-test-7987__H_to_L7R_a25__fresh",
      "kind": "fresh",
      "profile_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU6_SINGLE_V1/runs/pg19-test-7987__H_to_L7R_a25__resumed_parity.PROFILE.json",
      "profile_sha256": "28476333afc4996f592ebcd475964cc107aa5364ec8f436ba27e524aaf9e17f7",
      "scope": "last frozen loop reset through completion including checkpoint operations; canary resets before model load",
      "peak_allocated": 16935037440,
      "peak_reserved": 17265852416
    },
    {
      "tag": "pg19-test-7987__H_to_L7R_a25__parent",
      "kind": "parent",
      "profile_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU6_SINGLE_V1/runs/pg19-test-7987__H_to_L7R_a25__parent.PROFILE.json",
      "profile_sha256": "38bbf4bcde347167cfb3fead1b91b71cfcc14612060ba342e6a97bc4a940266e",
      "scope": "last frozen loop reset through completion including checkpoint operations; canary resets before model load",
      "peak_allocated": 16935037440,
      "peak_reserved": 20667432960
    },
    {
      "tag": "pg19-test-7987__H_to_L7_a50__fresh",
      "kind": "fresh",
      "profile_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU6_SINGLE_V1/runs/pg19-test-7987__H_to_L7_a50__resumed_parity.PROFILE.json",
      "profile_sha256": "c7865880eb3bff869f2957638e22ba06f8c771305b7c2326358ff1df5d8308f0",
      "scope": "last frozen loop reset through completion including checkpoint operations; canary resets before model load",
      "peak_allocated": 16935037440,
      "peak_reserved": 17265852416
    },
    {
      "tag": "pg19-test-7987__H_to_L7_a50__parent",
      "kind": "parent",
      "profile_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU6_SINGLE_V1/runs/pg19-test-7987__H_to_L7_a50__parent.PROFILE.json",
      "profile_sha256": "29723817ef73d68cdc0f571ccc0c2dd4af6949150a8bbd9f19838e76987dea8e",
      "scope": "last frozen loop reset through completion including checkpoint operations; canary resets before model load",
      "peak_allocated": 16935037440,
      "peak_reserved": 20667432960
    }
  ],
  "artifact_only_PROFILE_records": [],
  "partial_progress_records": [
    {
      "tag": "pg19-test-2544__FP__canary",
      "kind": "canary",
      "document": "pg19-test-2544",
      "condition": "FP",
      "worker_status": "PASS",
      "failed_or_incomplete": false,
      "source_log_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU6_SINGLE_V1/jobs/pg19-test-2544__FP__canary.log",
      "source_log_sha256": "b3145fa49a16f20babd00307da1cb831eac5cee52f4af508e2b11af0573ae801",
      "progress_row_count": 0,
      "invalid_progress_line_numbers": [],
      "last_recorded_progress": null,
      "last_confirmed_consumed_tokens": null,
      "exact_terminal_cursor": null,
      "exact_terminal_cursor_status": "UNKNOWN_NOT_RECORDED_BY_LOG_AUDIT",
      "max_observed_peak_allocated_bytes": null,
      "max_observed_peak_reserved_bytes": null,
      "CUDA_scope": "logged cumulative CUDA peak counters at recorded frozen-loop positions; excludes pre-reset model load/restore; failed-worker values are lower bounds, not terminal or lifetime peaks",
      "unlogged_tokens_not_inferred": true,
      "science_qualification_claimed": false
    },
    {
      "tag": "pg19-test-2544__FP__fresh",
      "kind": "fresh",
      "document": "pg19-test-2544",
      "condition": "FP",
      "worker_status": "PASS",
      "failed_or_incomplete": false,
      "source_log_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU6_SINGLE_V1/jobs/pg19-test-2544__FP__fresh.log",
      "source_log_sha256": "ac8d7fac69e5fbc925c10f651c440f59402e49bbb702c42ae46bcd23a3ab3061",
      "progress_row_count": 1,
      "invalid_progress_line_numbers": [],
      "last_recorded_progress": {
        "line_number": 39,
        "event": "qualification_progress",
        "tag": "pg19-test-2544__FP__resumed_parity",
        "position": 8256,
        "elapsed": 11.610116387950256,
        "allocated": 16359982080,
        "reserved": 16783507456,
        "peak_allocated": 16486794240,
        "peak_reserved": 16783507456,
        "inactive_split": 117341184
      },
      "last_confirmed_consumed_tokens": 8256,
      "exact_terminal_cursor": null,
      "exact_terminal_cursor_status": "UNKNOWN_NOT_RECORDED_BY_LOG_AUDIT",
      "max_observed_peak_allocated_bytes": 16486794240,
      "max_observed_peak_reserved_bytes": 16783507456,
      "CUDA_scope": "logged cumulative CUDA peak counters at recorded frozen-loop positions; excludes pre-reset model load/restore; failed-worker values are lower bounds, not terminal or lifetime peaks",
      "unlogged_tokens_not_inferred": true,
      "science_qualification_claimed": false
    },
    {
      "tag": "pg19-test-2544__FP__parent",
      "kind": "parent",
      "document": "pg19-test-2544",
      "condition": "FP",
      "worker_status": "PASS",
      "failed_or_incomplete": false,
      "source_log_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU6_SINGLE_V1/jobs/pg19-test-2544__FP__parent.log",
      "source_log_sha256": "dfcc68223de14f69bed6a9bf5b904bbd969ccac1891849a5c5da9e4d894cb54c",
      "progress_row_count": 17,
      "invalid_progress_line_numbers": [],
      "last_recorded_progress": {
        "line_number": 55,
        "event": "qualification_progress",
        "tag": "pg19-test-2544__FP__parent",
        "position": 8256,
        "elapsed": 1123.295603322913,
        "allocated": 16359982080,
        "reserved": 20128464896,
        "peak_allocated": 16486794240,
        "peak_reserved": 20128464896,
        "inactive_split": 29260800
      },
      "last_confirmed_consumed_tokens": 8256,
      "exact_terminal_cursor": null,
      "exact_terminal_cursor_status": "UNKNOWN_NOT_RECORDED_BY_LOG_AUDIT",
      "max_observed_peak_allocated_bytes": 16486794240,
      "max_observed_peak_reserved_bytes": 20128464896,
      "CUDA_scope": "logged cumulative CUDA peak counters at recorded frozen-loop positions; excludes pre-reset model load/restore; failed-worker values are lower bounds, not terminal or lifetime peaks",
      "unlogged_tokens_not_inferred": true,
      "science_qualification_claimed": false
    },
    {
      "tag": "pg19-test-2544__H_to_L6_a50__fresh",
      "kind": "fresh",
      "document": "pg19-test-2544",
      "condition": "H_to_L6_a50",
      "worker_status": "PASS",
      "failed_or_incomplete": false,
      "source_log_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU6_SINGLE_V1/jobs/pg19-test-2544__H_to_L6_a50__fresh.log",
      "source_log_sha256": "bda153221d2297499da600a8968935fa5f2c384800764aaa70f13d6f25a75b4e",
      "progress_row_count": 1,
      "invalid_progress_line_numbers": [],
      "last_recorded_progress": {
        "line_number": 39,
        "event": "qualification_progress",
        "tag": "pg19-test-2544__H_to_L6_a50__resumed_parity",
        "position": 8256,
        "elapsed": 11.121971239103004,
        "allocated": 16380298240,
        "reserved": 17265852416,
        "peak_allocated": 16935037440,
        "peak_reserved": 17265852416,
        "inactive_split": 46693376
      },
      "last_confirmed_consumed_tokens": 8256,
      "exact_terminal_cursor": null,
      "exact_terminal_cursor_status": "UNKNOWN_NOT_RECORDED_BY_LOG_AUDIT",
      "max_observed_peak_allocated_bytes": 16935037440,
      "max_observed_peak_reserved_bytes": 17265852416,
      "CUDA_scope": "logged cumulative CUDA peak counters at recorded frozen-loop positions; excludes pre-reset model load/restore; failed-worker values are lower bounds, not terminal or lifetime peaks",
      "unlogged_tokens_not_inferred": true,
      "science_qualification_claimed": false
    },
    {
      "tag": "pg19-test-2544__H_to_L6_a50__parent",
      "kind": "parent",
      "document": "pg19-test-2544",
      "condition": "H_to_L6_a50",
      "worker_status": "PASS",
      "failed_or_incomplete": false,
      "source_log_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU6_SINGLE_V1/jobs/pg19-test-2544__H_to_L6_a50__parent.log",
      "source_log_sha256": "5ea29ee7a775f085085e97e3b510f8408ff1ec102ca3ee21a5cdd543b646c095",
      "progress_row_count": 17,
      "invalid_progress_line_numbers": [],
      "last_recorded_progress": {
        "line_number": 55,
        "event": "qualification_progress",
        "tag": "pg19-test-2544__H_to_L6_a50__parent",
        "position": 8256,
        "elapsed": 1252.1509723159252,
        "allocated": 16380298240,
        "reserved": 20667432960,
        "peak_allocated": 16935037440,
        "peak_reserved": 20667432960,
        "inactive_split": 32013312
      },
      "last_confirmed_consumed_tokens": 8256,
      "exact_terminal_cursor": null,
      "exact_terminal_cursor_status": "UNKNOWN_NOT_RECORDED_BY_LOG_AUDIT",
      "max_observed_peak_allocated_bytes": 16935037440,
      "max_observed_peak_reserved_bytes": 20667432960,
      "CUDA_scope": "logged cumulative CUDA peak counters at recorded frozen-loop positions; excludes pre-reset model load/restore; failed-worker values are lower bounds, not terminal or lifetime peaks",
      "unlogged_tokens_not_inferred": true,
      "science_qualification_claimed": false
    },
    {
      "tag": "pg19-test-2544__H_to_L7_a50__fresh",
      "kind": "fresh",
      "document": "pg19-test-2544",
      "condition": "H_to_L7_a50",
      "worker_status": "PASS",
      "failed_or_incomplete": false,
      "source_log_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU6_SINGLE_V1/jobs/pg19-test-2544__H_to_L7_a50__fresh.log",
      "source_log_sha256": "7449cea80ee7631d52edc43b33730b57c8217fba9ff65df2520422eab77f4fd6",
      "progress_row_count": 1,
      "invalid_progress_line_numbers": [],
      "last_recorded_progress": {
        "line_number": 39,
        "event": "qualification_progress",
        "tag": "pg19-test-2544__H_to_L7_a50__resumed_parity",
        "position": 8256,
        "elapsed": 11.073697676998563,
        "allocated": 16380298240,
        "reserved": 17265852416,
        "peak_allocated": 16935037440,
        "peak_reserved": 17265852416,
        "inactive_split": 46693376
      },
      "last_confirmed_consumed_tokens": 8256,
      "exact_terminal_cursor": null,
      "exact_terminal_cursor_status": "UNKNOWN_NOT_RECORDED_BY_LOG_AUDIT",
      "max_observed_peak_allocated_bytes": 16935037440,
      "max_observed_peak_reserved_bytes": 17265852416,
      "CUDA_scope": "logged cumulative CUDA peak counters at recorded frozen-loop positions; excludes pre-reset model load/restore; failed-worker values are lower bounds, not terminal or lifetime peaks",
      "unlogged_tokens_not_inferred": true,
      "science_qualification_claimed": false
    },
    {
      "tag": "pg19-test-2544__H_to_L7_a50__parent",
      "kind": "parent",
      "document": "pg19-test-2544",
      "condition": "H_to_L7_a50",
      "worker_status": "PASS",
      "failed_or_incomplete": false,
      "source_log_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU6_SINGLE_V1/jobs/pg19-test-2544__H_to_L7_a50__parent.log",
      "source_log_sha256": "d8d45fd6c2d6e9ecf509733365719f16e20a80f0473ee49a9e0a02634c16cb48",
      "progress_row_count": 17,
      "invalid_progress_line_numbers": [],
      "last_recorded_progress": {
        "line_number": 55,
        "event": "qualification_progress",
        "tag": "pg19-test-2544__H_to_L7_a50__parent",
        "position": 8256,
        "elapsed": 1204.5642549709883,
        "allocated": 16380298240,
        "reserved": 20667432960,
        "peak_allocated": 16935037440,
        "peak_reserved": 20667432960,
        "inactive_split": 32013312
      },
      "last_confirmed_consumed_tokens": 8256,
      "exact_terminal_cursor": null,
      "exact_terminal_cursor_status": "UNKNOWN_NOT_RECORDED_BY_LOG_AUDIT",
      "max_observed_peak_allocated_bytes": 16935037440,
      "max_observed_peak_reserved_bytes": 20667432960,
      "CUDA_scope": "logged cumulative CUDA peak counters at recorded frozen-loop positions; excludes pre-reset model load/restore; failed-worker values are lower bounds, not terminal or lifetime peaks",
      "unlogged_tokens_not_inferred": true,
      "science_qualification_claimed": false
    },
    {
      "tag": "pg19-test-26183__FP__fresh",
      "kind": "fresh",
      "document": "pg19-test-26183",
      "condition": "FP",
      "worker_status": "PASS",
      "failed_or_incomplete": false,
      "source_log_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU6_SINGLE_V1/jobs/pg19-test-26183__FP__fresh.log",
      "source_log_sha256": "ea7f76b2a2b55d347c202a6fb588aa89cc6c4069c7ecaaeda9efd1954b3f7e4f",
      "progress_row_count": 1,
      "invalid_progress_line_numbers": [],
      "last_recorded_progress": {
        "line_number": 39,
        "event": "qualification_progress",
        "tag": "pg19-test-26183__FP__resumed_parity",
        "position": 8256,
        "elapsed": 12.430987688014284,
        "allocated": 16359982080,
        "reserved": 16783507456,
        "peak_allocated": 16486794240,
        "peak_reserved": 16783507456,
        "inactive_split": 117341184
      },
      "last_confirmed_consumed_tokens": 8256,
      "exact_terminal_cursor": null,
      "exact_terminal_cursor_status": "UNKNOWN_NOT_RECORDED_BY_LOG_AUDIT",
      "max_observed_peak_allocated_bytes": 16486794240,
      "max_observed_peak_reserved_bytes": 16783507456,
      "CUDA_scope": "logged cumulative CUDA peak counters at recorded frozen-loop positions; excludes pre-reset model load/restore; failed-worker values are lower bounds, not terminal or lifetime peaks",
      "unlogged_tokens_not_inferred": true,
      "science_qualification_claimed": false
    },
    {
      "tag": "pg19-test-26183__FP__parent",
      "kind": "parent",
      "document": "pg19-test-26183",
      "condition": "FP",
      "worker_status": "PASS",
      "failed_or_incomplete": false,
      "source_log_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU6_SINGLE_V1/jobs/pg19-test-26183__FP__parent.log",
      "source_log_sha256": "ab732803d0ed264eac28932f5cf6bf17119cf6d0bd7f70f3c1108fb0a11691e8",
      "progress_row_count": 17,
      "invalid_progress_line_numbers": [],
      "last_recorded_progress": {
        "line_number": 55,
        "event": "qualification_progress",
        "tag": "pg19-test-26183__FP__parent",
        "position": 8256,
        "elapsed": 1040.778216567007,
        "allocated": 16359982080,
        "reserved": 20128464896,
        "peak_allocated": 16486794240,
        "peak_reserved": 20128464896,
        "inactive_split": 29260800
      },
      "last_confirmed_consumed_tokens": 8256,
      "exact_terminal_cursor": null,
      "exact_terminal_cursor_status": "UNKNOWN_NOT_RECORDED_BY_LOG_AUDIT",
      "max_observed_peak_allocated_bytes": 16486794240,
      "max_observed_peak_reserved_bytes": 20128464896,
      "CUDA_scope": "logged cumulative CUDA peak counters at recorded frozen-loop positions; excludes pre-reset model load/restore; failed-worker values are lower bounds, not terminal or lifetime peaks",
      "unlogged_tokens_not_inferred": true,
      "science_qualification_claimed": false
    },
    {
      "tag": "pg19-test-26183__H_to_L6_a50__fresh",
      "kind": "fresh",
      "document": "pg19-test-26183",
      "condition": "H_to_L6_a50",
      "worker_status": "PASS",
      "failed_or_incomplete": false,
      "source_log_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU6_SINGLE_V1/jobs/pg19-test-26183__H_to_L6_a50__fresh.log",
      "source_log_sha256": "ad31ff585a9fec88c32465543c0c283ce68e1119bd1270b1d2ed7cb3847e6d69",
      "progress_row_count": 1,
      "invalid_progress_line_numbers": [],
      "last_recorded_progress": {
        "line_number": 39,
        "event": "qualification_progress",
        "tag": "pg19-test-26183__H_to_L6_a50__resumed_parity",
        "position": 8256,
        "elapsed": 11.943986961967312,
        "allocated": 16380298240,
        "reserved": 17265852416,
        "peak_allocated": 16935037440,
        "peak_reserved": 17265852416,
        "inactive_split": 46693376
      },
      "last_confirmed_consumed_tokens": 8256,
      "exact_terminal_cursor": null,
      "exact_terminal_cursor_status": "UNKNOWN_NOT_RECORDED_BY_LOG_AUDIT",
      "max_observed_peak_allocated_bytes": 16935037440,
      "max_observed_peak_reserved_bytes": 17265852416,
      "CUDA_scope": "logged cumulative CUDA peak counters at recorded frozen-loop positions; excludes pre-reset model load/restore; failed-worker values are lower bounds, not terminal or lifetime peaks",
      "unlogged_tokens_not_inferred": true,
      "science_qualification_claimed": false
    },
    {
      "tag": "pg19-test-26183__H_to_L6_a50__parent",
      "kind": "parent",
      "document": "pg19-test-26183",
      "condition": "H_to_L6_a50",
      "worker_status": "PASS",
      "failed_or_incomplete": false,
      "source_log_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU6_SINGLE_V1/jobs/pg19-test-26183__H_to_L6_a50__parent.log",
      "source_log_sha256": "b2413c76a8e43fce7c684d2f40576c5819668521c0ecf5e4bdc9af0574e53fc5",
      "progress_row_count": 17,
      "invalid_progress_line_numbers": [],
      "last_recorded_progress": {
        "line_number": 55,
        "event": "qualification_progress",
        "tag": "pg19-test-26183__H_to_L6_a50__parent",
        "position": 8256,
        "elapsed": 1155.0499089659424,
        "allocated": 16380298240,
        "reserved": 20667432960,
        "peak_allocated": 16935037440,
        "peak_reserved": 20667432960,
        "inactive_split": 32013312
      },
      "last_confirmed_consumed_tokens": 8256,
      "exact_terminal_cursor": null,
      "exact_terminal_cursor_status": "UNKNOWN_NOT_RECORDED_BY_LOG_AUDIT",
      "max_observed_peak_allocated_bytes": 16935037440,
      "max_observed_peak_reserved_bytes": 20667432960,
      "CUDA_scope": "logged cumulative CUDA peak counters at recorded frozen-loop positions; excludes pre-reset model load/restore; failed-worker values are lower bounds, not terminal or lifetime peaks",
      "unlogged_tokens_not_inferred": true,
      "science_qualification_claimed": false
    },
    {
      "tag": "pg19-test-26183__H_to_L7R_a25__fresh",
      "kind": "fresh",
      "document": "pg19-test-26183",
      "condition": "H_to_L7R_a25",
      "worker_status": "PASS",
      "failed_or_incomplete": false,
      "source_log_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU6_SINGLE_V1/jobs/pg19-test-26183__H_to_L7R_a25__fresh.log",
      "source_log_sha256": "535ef3405e30cacfd8e7523620a7797e84a335c322a4037874758f27532e1ed1",
      "progress_row_count": 1,
      "invalid_progress_line_numbers": [],
      "last_recorded_progress": {
        "line_number": 39,
        "event": "qualification_progress",
        "tag": "pg19-test-26183__H_to_L7R_a25__resumed_parity",
        "position": 8256,
        "elapsed": 12.969454625970684,
        "allocated": 16380298240,
        "reserved": 17265852416,
        "peak_allocated": 16935037440,
        "peak_reserved": 17265852416,
        "inactive_split": 46693376
      },
      "last_confirmed_consumed_tokens": 8256,
      "exact_terminal_cursor": null,
      "exact_terminal_cursor_status": "UNKNOWN_NOT_RECORDED_BY_LOG_AUDIT",
      "max_observed_peak_allocated_bytes": 16935037440,
      "max_observed_peak_reserved_bytes": 17265852416,
      "CUDA_scope": "logged cumulative CUDA peak counters at recorded frozen-loop positions; excludes pre-reset model load/restore; failed-worker values are lower bounds, not terminal or lifetime peaks",
      "unlogged_tokens_not_inferred": true,
      "science_qualification_claimed": false
    },
    {
      "tag": "pg19-test-26183__H_to_L7R_a25__parent",
      "kind": "parent",
      "document": "pg19-test-26183",
      "condition": "H_to_L7R_a25",
      "worker_status": "PASS",
      "failed_or_incomplete": false,
      "source_log_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU6_SINGLE_V1/jobs/pg19-test-26183__H_to_L7R_a25__parent.log",
      "source_log_sha256": "5c01c54ecf90e17cef89d6f37ce9747f2d50f34d01dc8ffcd9d0290f02c161ff",
      "progress_row_count": 17,
      "invalid_progress_line_numbers": [],
      "last_recorded_progress": {
        "line_number": 55,
        "event": "qualification_progress",
        "tag": "pg19-test-26183__H_to_L7R_a25__parent",
        "position": 8256,
        "elapsed": 1303.0854179630987,
        "allocated": 16380298240,
        "reserved": 20667432960,
        "peak_allocated": 16935037440,
        "peak_reserved": 20667432960,
        "inactive_split": 32013312
      },
      "last_confirmed_consumed_tokens": 8256,
      "exact_terminal_cursor": null,
      "exact_terminal_cursor_status": "UNKNOWN_NOT_RECORDED_BY_LOG_AUDIT",
      "max_observed_peak_allocated_bytes": 16935037440,
      "max_observed_peak_reserved_bytes": 20667432960,
      "CUDA_scope": "logged cumulative CUDA peak counters at recorded frozen-loop positions; excludes pre-reset model load/restore; failed-worker values are lower bounds, not terminal or lifetime peaks",
      "unlogged_tokens_not_inferred": true,
      "science_qualification_claimed": false
    },
    {
      "tag": "pg19-test-41603__FP__formal1",
      "kind": "formal1",
      "document": "pg19-test-41603",
      "condition": "FP",
      "worker_status": "FAILED_OR_ABORTED",
      "failed_or_incomplete": true,
      "source_log_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU6_SINGLE_V1/jobs/pg19-test-41603__FP__formal1.log",
      "source_log_sha256": "65f3f8669dc44b153cdb9d262e2d63b94e7e61698892a1818c53c9c61bd352d4",
      "progress_row_count": 8,
      "invalid_progress_line_numbers": [],
      "last_recorded_progress": {
        "line_number": 46,
        "event": "formal_progress",
        "tag": "pg19-test-41603__FP__formal",
        "position": 12288,
        "elapsed": 543.3783228000393,
        "allocated": 16608086016,
        "reserved": 21791506432,
        "peak_allocated": 16683095040,
        "peak_reserved": 21791506432
      },
      "last_confirmed_consumed_tokens": 12288,
      "exact_terminal_cursor": null,
      "exact_terminal_cursor_status": "UNKNOWN_NOT_RECORDED_BY_LOG_AUDIT",
      "max_observed_peak_allocated_bytes": 16683095040,
      "max_observed_peak_reserved_bytes": 21791506432,
      "CUDA_scope": "logged cumulative CUDA peak counters at recorded frozen-loop positions; excludes pre-reset model load/restore; failed-worker values are lower bounds, not terminal or lifetime peaks",
      "unlogged_tokens_not_inferred": true,
      "science_qualification_claimed": false
    },
    {
      "tag": "pg19-test-41603__FP__fresh",
      "kind": "fresh",
      "document": "pg19-test-41603",
      "condition": "FP",
      "worker_status": "PASS",
      "failed_or_incomplete": false,
      "source_log_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU6_SINGLE_V1/jobs/pg19-test-41603__FP__fresh.log",
      "source_log_sha256": "909e39ffbba91b5e99cf4ce8338766f063989b319f46bd8a69df0cccb08915c9",
      "progress_row_count": 1,
      "invalid_progress_line_numbers": [],
      "last_recorded_progress": {
        "line_number": 39,
        "event": "qualification_progress",
        "tag": "pg19-test-41603__FP__resumed_parity",
        "position": 8256,
        "elapsed": 10.978746976994444,
        "allocated": 16359982080,
        "reserved": 16783507456,
        "peak_allocated": 16486794240,
        "peak_reserved": 16783507456,
        "inactive_split": 117341184
      },
      "last_confirmed_consumed_tokens": 8256,
      "exact_terminal_cursor": null,
      "exact_terminal_cursor_status": "UNKNOWN_NOT_RECORDED_BY_LOG_AUDIT",
      "max_observed_peak_allocated_bytes": 16486794240,
      "max_observed_peak_reserved_bytes": 16783507456,
      "CUDA_scope": "logged cumulative CUDA peak counters at recorded frozen-loop positions; excludes pre-reset model load/restore; failed-worker values are lower bounds, not terminal or lifetime peaks",
      "unlogged_tokens_not_inferred": true,
      "science_qualification_claimed": false
    },
    {
      "tag": "pg19-test-41603__FP__parent",
      "kind": "parent",
      "document": "pg19-test-41603",
      "condition": "FP",
      "worker_status": "PASS",
      "failed_or_incomplete": false,
      "source_log_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU6_SINGLE_V1/jobs/pg19-test-41603__FP__parent.log",
      "source_log_sha256": "0624edae2b7298f3c4c51be676f37e0ec0e69f31c4183c9b09c08a1bb226ea74",
      "progress_row_count": 17,
      "invalid_progress_line_numbers": [],
      "last_recorded_progress": {
        "line_number": 55,
        "event": "qualification_progress",
        "tag": "pg19-test-41603__FP__parent",
        "position": 8256,
        "elapsed": 1158.4292272389866,
        "allocated": 16359982080,
        "reserved": 20128464896,
        "peak_allocated": 16486794240,
        "peak_reserved": 20128464896,
        "inactive_split": 29260800
      },
      "last_confirmed_consumed_tokens": 8256,
      "exact_terminal_cursor": null,
      "exact_terminal_cursor_status": "UNKNOWN_NOT_RECORDED_BY_LOG_AUDIT",
      "max_observed_peak_allocated_bytes": 16486794240,
      "max_observed_peak_reserved_bytes": 20128464896,
      "CUDA_scope": "logged cumulative CUDA peak counters at recorded frozen-loop positions; excludes pre-reset model load/restore; failed-worker values are lower bounds, not terminal or lifetime peaks",
      "unlogged_tokens_not_inferred": true,
      "science_qualification_claimed": false
    },
    {
      "tag": "pg19-test-41603__H__fresh",
      "kind": "fresh",
      "document": "pg19-test-41603",
      "condition": "H",
      "worker_status": "PASS",
      "failed_or_incomplete": false,
      "source_log_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU6_SINGLE_V1/jobs/pg19-test-41603__H__fresh.log",
      "source_log_sha256": "d429e706c2ec962432b165aed0909707079b536adb9fca7dc678fe76bc6d58a6",
      "progress_row_count": 1,
      "invalid_progress_line_numbers": [],
      "last_recorded_progress": {
        "line_number": 39,
        "event": "qualification_progress",
        "tag": "pg19-test-41603__H__resumed_parity",
        "position": 8256,
        "elapsed": 11.232737614074722,
        "allocated": 16380298240,
        "reserved": 17265852416,
        "peak_allocated": 16935037440,
        "peak_reserved": 17265852416,
        "inactive_split": 46693376
      },
      "last_confirmed_consumed_tokens": 8256,
      "exact_terminal_cursor": null,
      "exact_terminal_cursor_status": "UNKNOWN_NOT_RECORDED_BY_LOG_AUDIT",
      "max_observed_peak_allocated_bytes": 16935037440,
      "max_observed_peak_reserved_bytes": 17265852416,
      "CUDA_scope": "logged cumulative CUDA peak counters at recorded frozen-loop positions; excludes pre-reset model load/restore; failed-worker values are lower bounds, not terminal or lifetime peaks",
      "unlogged_tokens_not_inferred": true,
      "science_qualification_claimed": false
    },
    {
      "tag": "pg19-test-41603__H__parent",
      "kind": "parent",
      "document": "pg19-test-41603",
      "condition": "H",
      "worker_status": "PASS",
      "failed_or_incomplete": false,
      "source_log_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU6_SINGLE_V1/jobs/pg19-test-41603__H__parent.log",
      "source_log_sha256": "519a4c9eebc0d373d5efa65bde11e6b35b2e9c038d651868a10e6ba649f9cfc2",
      "progress_row_count": 17,
      "invalid_progress_line_numbers": [],
      "last_recorded_progress": {
        "line_number": 55,
        "event": "qualification_progress",
        "tag": "pg19-test-41603__H__parent",
        "position": 8256,
        "elapsed": 1171.8665859170724,
        "allocated": 16380298240,
        "reserved": 20667432960,
        "peak_allocated": 16935037440,
        "peak_reserved": 20667432960,
        "inactive_split": 32013312
      },
      "last_confirmed_consumed_tokens": 8256,
      "exact_terminal_cursor": null,
      "exact_terminal_cursor_status": "UNKNOWN_NOT_RECORDED_BY_LOG_AUDIT",
      "max_observed_peak_allocated_bytes": 16935037440,
      "max_observed_peak_reserved_bytes": 20667432960,
      "CUDA_scope": "logged cumulative CUDA peak counters at recorded frozen-loop positions; excludes pre-reset model load/restore; failed-worker values are lower bounds, not terminal or lifetime peaks",
      "unlogged_tokens_not_inferred": true,
      "science_qualification_claimed": false
    },
    {
      "tag": "pg19-test-41603__H_to_L7_a50__fresh",
      "kind": "fresh",
      "document": "pg19-test-41603",
      "condition": "H_to_L7_a50",
      "worker_status": "PASS",
      "failed_or_incomplete": false,
      "source_log_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU6_SINGLE_V1/jobs/pg19-test-41603__H_to_L7_a50__fresh.log",
      "source_log_sha256": "d0ee072e4a729b89478847bbba917365f26db8ada894fab6fabc5a2f1b071bce",
      "progress_row_count": 1,
      "invalid_progress_line_numbers": [],
      "last_recorded_progress": {
        "line_number": 39,
        "event": "qualification_progress",
        "tag": "pg19-test-41603__H_to_L7_a50__resumed_parity",
        "position": 8256,
        "elapsed": 12.503550583962351,
        "allocated": 16380298240,
        "reserved": 17265852416,
        "peak_allocated": 16935037440,
        "peak_reserved": 17265852416,
        "inactive_split": 46693376
      },
      "last_confirmed_consumed_tokens": 8256,
      "exact_terminal_cursor": null,
      "exact_terminal_cursor_status": "UNKNOWN_NOT_RECORDED_BY_LOG_AUDIT",
      "max_observed_peak_allocated_bytes": 16935037440,
      "max_observed_peak_reserved_bytes": 17265852416,
      "CUDA_scope": "logged cumulative CUDA peak counters at recorded frozen-loop positions; excludes pre-reset model load/restore; failed-worker values are lower bounds, not terminal or lifetime peaks",
      "unlogged_tokens_not_inferred": true,
      "science_qualification_claimed": false
    },
    {
      "tag": "pg19-test-41603__H_to_L7_a50__parent",
      "kind": "parent",
      "document": "pg19-test-41603",
      "condition": "H_to_L7_a50",
      "worker_status": "PASS",
      "failed_or_incomplete": false,
      "source_log_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU6_SINGLE_V1/jobs/pg19-test-41603__H_to_L7_a50__parent.log",
      "source_log_sha256": "d9aae6b87f922aa0eb4bd6b0bc5aabaa61458238d24a4c957b598202acea6c6b",
      "progress_row_count": 17,
      "invalid_progress_line_numbers": [],
      "last_recorded_progress": {
        "line_number": 55,
        "event": "qualification_progress",
        "tag": "pg19-test-41603__H_to_L7_a50__parent",
        "position": 8256,
        "elapsed": 1243.9690449599875,
        "allocated": 16380298240,
        "reserved": 20671627264,
        "peak_allocated": 16935037440,
        "peak_reserved": 20671627264,
        "inactive_split": 29916160
      },
      "last_confirmed_consumed_tokens": 8256,
      "exact_terminal_cursor": null,
      "exact_terminal_cursor_status": "UNKNOWN_NOT_RECORDED_BY_LOG_AUDIT",
      "max_observed_peak_allocated_bytes": 16935037440,
      "max_observed_peak_reserved_bytes": 20671627264,
      "CUDA_scope": "logged cumulative CUDA peak counters at recorded frozen-loop positions; excludes pre-reset model load/restore; failed-worker values are lower bounds, not terminal or lifetime peaks",
      "unlogged_tokens_not_inferred": true,
      "science_qualification_claimed": false
    },
    {
      "tag": "pg19-test-7987__FP__fresh",
      "kind": "fresh",
      "document": "pg19-test-7987",
      "condition": "FP",
      "worker_status": "PASS",
      "failed_or_incomplete": false,
      "source_log_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU6_SINGLE_V1/jobs/pg19-test-7987__FP__fresh.log",
      "source_log_sha256": "4f59f160999d2c374282d529ac3dad747dc7cc51c1868799b438769174bb5554",
      "progress_row_count": 1,
      "invalid_progress_line_numbers": [],
      "last_recorded_progress": {
        "line_number": 39,
        "event": "qualification_progress",
        "tag": "pg19-test-7987__FP__resumed_parity",
        "position": 8256,
        "elapsed": 10.198662390001118,
        "allocated": 16359982080,
        "reserved": 16783507456,
        "peak_allocated": 16486794240,
        "peak_reserved": 16783507456,
        "inactive_split": 117341184
      },
      "last_confirmed_consumed_tokens": 8256,
      "exact_terminal_cursor": null,
      "exact_terminal_cursor_status": "UNKNOWN_NOT_RECORDED_BY_LOG_AUDIT",
      "max_observed_peak_allocated_bytes": 16486794240,
      "max_observed_peak_reserved_bytes": 16783507456,
      "CUDA_scope": "logged cumulative CUDA peak counters at recorded frozen-loop positions; excludes pre-reset model load/restore; failed-worker values are lower bounds, not terminal or lifetime peaks",
      "unlogged_tokens_not_inferred": true,
      "science_qualification_claimed": false
    },
    {
      "tag": "pg19-test-7987__FP__parent",
      "kind": "parent",
      "document": "pg19-test-7987",
      "condition": "FP",
      "worker_status": "PASS",
      "failed_or_incomplete": false,
      "source_log_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU6_SINGLE_V1/jobs/pg19-test-7987__FP__parent.log",
      "source_log_sha256": "a30ed2204d8cca1513b04d15757454875ffa7fa50a07ce824d86868cc5481bf3",
      "progress_row_count": 17,
      "invalid_progress_line_numbers": [],
      "last_recorded_progress": {
        "line_number": 55,
        "event": "qualification_progress",
        "tag": "pg19-test-7987__FP__parent",
        "position": 8256,
        "elapsed": 1031.1909114250448,
        "allocated": 16359982080,
        "reserved": 20128464896,
        "peak_allocated": 16486794240,
        "peak_reserved": 20128464896,
        "inactive_split": 29260800
      },
      "last_confirmed_consumed_tokens": 8256,
      "exact_terminal_cursor": null,
      "exact_terminal_cursor_status": "UNKNOWN_NOT_RECORDED_BY_LOG_AUDIT",
      "max_observed_peak_allocated_bytes": 16486794240,
      "max_observed_peak_reserved_bytes": 20128464896,
      "CUDA_scope": "logged cumulative CUDA peak counters at recorded frozen-loop positions; excludes pre-reset model load/restore; failed-worker values are lower bounds, not terminal or lifetime peaks",
      "unlogged_tokens_not_inferred": true,
      "science_qualification_claimed": false
    },
    {
      "tag": "pg19-test-7987__H_to_L7R_a25__fresh",
      "kind": "fresh",
      "document": "pg19-test-7987",
      "condition": "H_to_L7R_a25",
      "worker_status": "PASS",
      "failed_or_incomplete": false,
      "source_log_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU6_SINGLE_V1/jobs/pg19-test-7987__H_to_L7R_a25__fresh.log",
      "source_log_sha256": "b2907c7fe23fb3425cafc58eb070fef8db62ecb9fc01db711d96e4f4c7836669",
      "progress_row_count": 1,
      "invalid_progress_line_numbers": [],
      "last_recorded_progress": {
        "line_number": 39,
        "event": "qualification_progress",
        "tag": "pg19-test-7987__H_to_L7R_a25__resumed_parity",
        "position": 8256,
        "elapsed": 11.170209154020995,
        "allocated": 16380298240,
        "reserved": 17265852416,
        "peak_allocated": 16935037440,
        "peak_reserved": 17265852416,
        "inactive_split": 46693376
      },
      "last_confirmed_consumed_tokens": 8256,
      "exact_terminal_cursor": null,
      "exact_terminal_cursor_status": "UNKNOWN_NOT_RECORDED_BY_LOG_AUDIT",
      "max_observed_peak_allocated_bytes": 16935037440,
      "max_observed_peak_reserved_bytes": 17265852416,
      "CUDA_scope": "logged cumulative CUDA peak counters at recorded frozen-loop positions; excludes pre-reset model load/restore; failed-worker values are lower bounds, not terminal or lifetime peaks",
      "unlogged_tokens_not_inferred": true,
      "science_qualification_claimed": false
    },
    {
      "tag": "pg19-test-7987__H_to_L7R_a25__parent",
      "kind": "parent",
      "document": "pg19-test-7987",
      "condition": "H_to_L7R_a25",
      "worker_status": "PASS",
      "failed_or_incomplete": false,
      "source_log_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU6_SINGLE_V1/jobs/pg19-test-7987__H_to_L7R_a25__parent.log",
      "source_log_sha256": "2b9bfcfc55ac356f930b60a6dbd20d290c56c72a62c53a686379dc5bd4853f3b",
      "progress_row_count": 17,
      "invalid_progress_line_numbers": [],
      "last_recorded_progress": {
        "line_number": 55,
        "event": "qualification_progress",
        "tag": "pg19-test-7987__H_to_L7R_a25__parent",
        "position": 8256,
        "elapsed": 1132.7596247240435,
        "allocated": 16380298240,
        "reserved": 20667432960,
        "peak_allocated": 16935037440,
        "peak_reserved": 20667432960,
        "inactive_split": 32013312
      },
      "last_confirmed_consumed_tokens": 8256,
      "exact_terminal_cursor": null,
      "exact_terminal_cursor_status": "UNKNOWN_NOT_RECORDED_BY_LOG_AUDIT",
      "max_observed_peak_allocated_bytes": 16935037440,
      "max_observed_peak_reserved_bytes": 20667432960,
      "CUDA_scope": "logged cumulative CUDA peak counters at recorded frozen-loop positions; excludes pre-reset model load/restore; failed-worker values are lower bounds, not terminal or lifetime peaks",
      "unlogged_tokens_not_inferred": true,
      "science_qualification_claimed": false
    },
    {
      "tag": "pg19-test-7987__H_to_L7_a50__fresh",
      "kind": "fresh",
      "document": "pg19-test-7987",
      "condition": "H_to_L7_a50",
      "worker_status": "PASS",
      "failed_or_incomplete": false,
      "source_log_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU6_SINGLE_V1/jobs/pg19-test-7987__H_to_L7_a50__fresh.log",
      "source_log_sha256": "1a0c921c51f0e4d9cd18081da4e66073395d4fd68bdc34214f32eb576668d815",
      "progress_row_count": 1,
      "invalid_progress_line_numbers": [],
      "last_recorded_progress": {
        "line_number": 39,
        "event": "qualification_progress",
        "tag": "pg19-test-7987__H_to_L7_a50__resumed_parity",
        "position": 8256,
        "elapsed": 11.242552413023077,
        "allocated": 16380298240,
        "reserved": 17265852416,
        "peak_allocated": 16935037440,
        "peak_reserved": 17265852416,
        "inactive_split": 46693376
      },
      "last_confirmed_consumed_tokens": 8256,
      "exact_terminal_cursor": null,
      "exact_terminal_cursor_status": "UNKNOWN_NOT_RECORDED_BY_LOG_AUDIT",
      "max_observed_peak_allocated_bytes": 16935037440,
      "max_observed_peak_reserved_bytes": 17265852416,
      "CUDA_scope": "logged cumulative CUDA peak counters at recorded frozen-loop positions; excludes pre-reset model load/restore; failed-worker values are lower bounds, not terminal or lifetime peaks",
      "unlogged_tokens_not_inferred": true,
      "science_qualification_claimed": false
    },
    {
      "tag": "pg19-test-7987__H_to_L7_a50__parent",
      "kind": "parent",
      "document": "pg19-test-7987",
      "condition": "H_to_L7_a50",
      "worker_status": "PASS",
      "failed_or_incomplete": false,
      "source_log_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU6_SINGLE_V1/jobs/pg19-test-7987__H_to_L7_a50__parent.log",
      "source_log_sha256": "45f7c0c945f7893bb9b60fcfcd0b0a551565eb9cc0dbe74489d45d9c301caf70",
      "progress_row_count": 17,
      "invalid_progress_line_numbers": [],
      "last_recorded_progress": {
        "line_number": 55,
        "event": "qualification_progress",
        "tag": "pg19-test-7987__H_to_L7_a50__parent",
        "position": 8256,
        "elapsed": 1124.2266107379692,
        "allocated": 16380298240,
        "reserved": 20667432960,
        "peak_allocated": 16935037440,
        "peak_reserved": 20667432960,
        "inactive_split": 32013312
      },
      "last_confirmed_consumed_tokens": 8256,
      "exact_terminal_cursor": null,
      "exact_terminal_cursor_status": "UNKNOWN_NOT_RECORDED_BY_LOG_AUDIT",
      "max_observed_peak_allocated_bytes": 16935037440,
      "max_observed_peak_reserved_bytes": 20667432960,
      "CUDA_scope": "logged cumulative CUDA peak counters at recorded frozen-loop positions; excludes pre-reset model load/restore; failed-worker values are lower bounds, not terminal or lifetime peaks",
      "unlogged_tokens_not_inferred": true,
      "science_qualification_claimed": false
    }
  ],
  "max_observed_CUDA_peak_allocated_bytes": 16935037440,
  "max_observed_CUDA_peak_reserved_bytes": 21791506432,
  "CUDA_scope": "maximum observed recorded counters across separately scoped verified successful or artifact-only PROFILEs and all actual progress logs; failed-worker observations are lower bounds through last logged position, not terminal or lifetime peaks; model loading/restore excluded after frozen-loop resets; canary has its separate pre-model-load reset; no full32K peak measured",
  "NVML_periodic_peak_owned_bytes": 23246929920,
  "NVML_scope": "periodic exact-owned PID samples plus exact PID/UID/starttime owned-memory observations from the failed raw evidence; foreign/child/graphics memory excluded; not an instantaneous peak",
  "STOP_resource_evidence": {
    "status": "INCOMPLETE_OR_FAILED_RESOURCE_EVIDENCE",
    "first_STOP": {
      "epoch": 1792138737.031479,
      "evidence_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU6_SINGLE_V1/FAILED_RESOURCE_SNAPSHOT.json",
      "evidence_sha256": "09bfa67d5981360af61acaef31563850e1a8d6baa112ace890f156b027901397",
      "forbiddenPIDs": [],
      "no_retry": true,
      "original_resource_error": {
        "repr": "AssertionError('INSUFFICIENT_FREE_MEMORY_AND_2GIB_HEADROOM')",
        "traceback": "Traceback (most recent call last):\n  File \"/data/zypan/worktrees/ling-kda-4doc-cleanroom-gpu6-single-v1/resource_evidence.py\", line 127, in assess_with_evidence\n    return assess_fn(snapshot,uuid,owner,baseline)\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/data/zypan/worktrees/ling-kda-4doc-cleanroom-gpu6-single-v1/cleanroom_run.py\", line 188, in checked\n    return assess_lifecycle_sample(s,u,o,b,poll_before,poll_after)\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/data/zypan/worktrees/ling-kda-4doc-cleanroom-gpu6-single-v1/lifecycle_guard.py\", line 65, in assess_lifecycle_sample\n    safety=strict_snapshot_safety(snapshot,uuid,owner,baseline)\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/data/zypan/worktrees/ling-kda-4doc-cleanroom-gpu6-single-v1/lifecycle_guard.py\", line 45, in strict_snapshot_safety\n    assert m['free']>=g.HEADROOM,'INSUFFICIENT_FREE_MEMORY_AND_2GIB_HEADROOM'\n           ^^^^^^^^^^^^^^^^^^^^^\nAssertionError: INSUFFICIENT_FREE_MEMORY_AND_2GIB_HEADROOM\n",
        "type": "AssertionError"
      },
      "reason": "ResourceEvidenceError('RESOURCE_ASSESSMENT_FAILED_WITH_PINNED_EVIDENCE: {\\'type\\': \\'AssertionError\\', \\'repr\\': \"AssertionError(\\'INSUFFICIENT_FREE_MEMORY_AND_2GIB_HEADROOM\\')\", \\'traceback\\': \\'Traceback (most recent call last):\\\\n  File \"/data/zypan/worktrees/ling-kda-4doc-cleanroom-gpu6-single-v1/resource_evidence.py\", line 127, in assess_with_evidence\\\\n    return assess_fn(snapshot,uuid,owner,baseline)\\\\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\\\n  File \"/data/zypan/worktrees/ling-kda-4doc-cleanroom-gpu6-single-v1/cleanroom_run.py\", line 188, in checked\\\\n    return assess_lifecycle_sample(s,u,o,b,poll_before,poll_after)\\\\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\\\n  File \"/data/zypan/worktrees/ling-kda-4doc-cleanroom-gpu6-single-v1/lifecycle_guard.py\", line 65, in assess_lifecycle_sample\\\\n    safety=strict_snapshot_safety(snapshot,uuid,owner,baseline)\\\\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\\\n  File \"/data/zypan/worktrees/ling-kda-4doc-cleanroom-gpu6-single-v1/lifecycle_guard.py\", line 45, in strict_snapshot_safety\\\\n    assert m[\\\\\\'free\\\\\\']>=g.HEADROOM,\\\\\\'INSUFFICIENT_FREE_MEMORY_AND_2GIB_HEADROOM\\\\\\'\\\\n           ^^^^^^^^^^^^^^^^^^^^^\\\\nAssertionError: INSUFFICIENT_FREE_MEMORY_AND_2GIB_HEADROOM\\\\n\\'}')",
      "stage": "RUNNING_RESOURCE_SAFETY",
      "status": "STOP",
      "tag": "pg19-test-41603__FP__formal1",
      "task": "LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1"
    },
    "first_STOP_sha256": "06ef4e6b443e5147c3bec4204df103a2dd3bc3ef700685670a895be36e0828be",
    "is_resource_gate_STOP": true,
    "artifacts": [
      {
        "path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU6_SINGLE_V1/FAILED_RESOURCE_SNAPSHOT.json",
        "bytes": 60518,
        "sha256": "09bfa67d5981360af61acaef31563850e1a8d6baa112ace890f156b027901397",
        "expected_SHA_from_STOP": "09bfa67d5981360af61acaef31563850e1a8d6baa112ace890f156b027901397"
      }
    ],
    "pinned_failures": [],
    "in_worker_failure_bindings": [],
    "admission_failure_bindings": [],
    "observed_process_records": [
      {
        "artifact_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU6_SINGLE_V1/FAILED_RESOURCE_SNAPSHOT.json",
        "epoch": 1792138736.6770148,
        "uuid": "GPU-de53f243-b271-521c-d875-d96611ed8574",
        "observation_status": "OBSERVED_COMPLETE_PROCESS_APIS",
        "NVML_compute_PIDs": [
          3334383
        ],
        "SMI_non_graphics_PIDs": [
          3334383
        ],
        "early_SMI_compute_source_present": true,
        "early_SMI_compute_PIDs": [
          3334383
        ],
        "early_SMI_compute_status": "PASS_TARGET_UUID_PID_LIST",
        "owned_NVML_usedGpuMemory_bytes": [
          23246929920
        ],
        "owned_process_snapshot_identity_status": "MATCH_PID_UID_STARTTIME",
        "foreign_process_memory_excluded": true,
        "unauthorized_PIDs_under_unchanged_exact_owner_policy": [],
        "unauthorized_process_metadata": [],
        "exact_owned_process_identity": {
          "boot_id": "9e80e411-1604-4937-803f-b78c74b28fe8",
          "nspid": [
            3334383
          ],
          "pid": 3334383,
          "start_ticks": 53712546,
          "uid": 1022
        },
        "posthoc_data_used": false
      }
    ],
    "exact_trigger_PID_or_ownership_not_captured": "SEE_LINKED_RAW_OBSERVATIONS_WITH_EXPLICIT_API_STATUS",
    "current_or_posthoc_samples_substituted": false,
    "errors": [
      {
        "stage": "resource_evidence",
        "reason": "STOP_LINKED_ARTIFACT_OR_RAW_BINDING_VERIFICATION_FAILED",
        "path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU6_SINGLE_V1/FAILED_RESOURCE_SNAPSHOT.json",
        "error": "AttributeError(\"'str' object has no attribute 'get'\")"
      },
      {
        "stage": "resource_evidence",
        "reason": "RESOURCE_STOP_WITHOUT_VERIFIED_PINNED_RAW_SNAPSHOT",
        "query_or_trigger_PID_identity": "UNKNOWN"
      }
    ]
  },
  "resource_evidence_errors": [
    {
      "stage": "resource_evidence",
      "reason": "STOP_LINKED_ARTIFACT_OR_RAW_BINDING_VERIFICATION_FAILED",
      "path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU6_SINGLE_V1/FAILED_RESOURCE_SNAPSHOT.json",
      "error": "AttributeError(\"'str' object has no attribute 'get'\")"
    },
    {
      "stage": "resource_evidence",
      "reason": "RESOURCE_STOP_WITHOUT_VERIFIED_PINNED_RAW_SNAPSHOT",
      "query_or_trigger_PID_identity": "UNKNOWN"
    }
  ],
  "launch_to_exit_includes_load_restore_teardown": true,
  "prior_same_task_RTX3090_GPUh": 1.007654354178602,
  "current_execution_RTX3090_GPUh": 4.296229118945387,
  "same_task_cumulative_RTX3090_GPUh": 5.303883473123989,
  "remaining_same_task_RTX3090_GPUh": 30.696116526876008,
  "MAX_SAME_TASK_RTX3090_GPUh": 36.0,
  "current_execution_cap_GPUh": 34.992345645821395,
  "protective_cumulative_GPUh": 35.8,
  "independent_RTX4090_qualification_GPUh": 3.447347587855119,
  "other_task_resources": {
    "rtx4090_qualification": {
      "FORMAL_SCIENCE_COMPLETE": false,
      "TASK": "LING_KDA_P1_L7_4DOC_RTX4090_QUALIFICATION_V1",
      "charged_to_current_3090_budget": false,
      "cumulative_RTX4090_GPUh": 3.447347587855119,
      "current_execution_GPUh": 3.320084693548787,
      "manifest_sha256": "4ff204601805bca0f685c480bcc6d942dcdf2b651cc0f0f4c151ebc1272e0099",
      "prior_RTX4090_GPUh": 0.1272628943063319,
      "status": "12/12_NUMERICAL_QUALIFICATION_COMPLETE"
    }
  },
  "independent_RTX4090_budget_scope": "separate qualification-only task; reported for provenance, not charged against this3090 CleanRoom cap"
}

预算硬上限36同任务3090新增GPUh；此前同任务3090实测1.007654354178602GPUh。独立4090资格任务已报告3.447347587855119GPUh仅单列历史身份，不扣本任务3090额度。含canary、加载、parent、fresh64、正式两段和恢复的每worker launch-to-exit，仅GPU6按真实launch-to-exit累计；旧V2 GPUh计入同任务累计额度（本轮剩余34.992345645821395GPUh）。运行前和第一批真实parent后更新实际测量预测，保护性35.8GPUh提前停止留退出余量，没有提高36上限。所有实际CP和FP payload由manifest真实SHA绑定，大模型/CP只留服务器，不下载模型。仅一张GPU6，未杀其他用户/重置GPU。

服务器目录：/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU6_SINGLE_V1；工作树：/data/zypan/worktrees/ling-kda-4doc-cleanroom-gpu6-single-v1。source_manifest、engineering_change_manifest、resource_admission、qualification_matrix、run_manifest、gpu_resource_ledger和hashes保存真实记录。完整8×384逐点校验仅在实际完成时为PASS，部分状态详见机器字段与原始数组。

## 首失败及证据边界

失败partial worker的最后确认消费位置与counter下界见TERMINAL_PARTIAL_PROGRESS_AUDIT；未记录的实际终止cursor为UNKNOWN，不按耗时推断。资源摘要max_observed合并成功PROFILE和失败日志累计counter，不能冒充完整生命周期峰值。失败pin与精确raw字节offset/line SHA、原worker或唯一admission关联见TERMINAL_RESOURCE_EVIDENCE_AUDIT；查询失败/丢失保持UNKNOWN及error，不用posthoc替代。

```json
{
  "epoch": 1792138737.031479,
  "evidence_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU6_SINGLE_V1/FAILED_RESOURCE_SNAPSHOT.json",
  "evidence_sha256": "09bfa67d5981360af61acaef31563850e1a8d6baa112ace890f156b027901397",
  "forbiddenPIDs": [],
  "no_retry": true,
  "original_resource_error": {
    "repr": "AssertionError('INSUFFICIENT_FREE_MEMORY_AND_2GIB_HEADROOM')",
    "traceback": "Traceback (most recent call last):\n  File \"/data/zypan/worktrees/ling-kda-4doc-cleanroom-gpu6-single-v1/resource_evidence.py\", line 127, in assess_with_evidence\n    return assess_fn(snapshot,uuid,owner,baseline)\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/data/zypan/worktrees/ling-kda-4doc-cleanroom-gpu6-single-v1/cleanroom_run.py\", line 188, in checked\n    return assess_lifecycle_sample(s,u,o,b,poll_before,poll_after)\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/data/zypan/worktrees/ling-kda-4doc-cleanroom-gpu6-single-v1/lifecycle_guard.py\", line 65, in assess_lifecycle_sample\n    safety=strict_snapshot_safety(snapshot,uuid,owner,baseline)\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/data/zypan/worktrees/ling-kda-4doc-cleanroom-gpu6-single-v1/lifecycle_guard.py\", line 45, in strict_snapshot_safety\n    assert m['free']>=g.HEADROOM,'INSUFFICIENT_FREE_MEMORY_AND_2GIB_HEADROOM'\n           ^^^^^^^^^^^^^^^^^^^^^\nAssertionError: INSUFFICIENT_FREE_MEMORY_AND_2GIB_HEADROOM\n",
    "type": "AssertionError"
  },
  "reason": "ResourceEvidenceError('RESOURCE_ASSESSMENT_FAILED_WITH_PINNED_EVIDENCE: {\\'type\\': \\'AssertionError\\', \\'repr\\': \"AssertionError(\\'INSUFFICIENT_FREE_MEMORY_AND_2GIB_HEADROOM\\')\", \\'traceback\\': \\'Traceback (most recent call last):\\\\n  File \"/data/zypan/worktrees/ling-kda-4doc-cleanroom-gpu6-single-v1/resource_evidence.py\", line 127, in assess_with_evidence\\\\n    return assess_fn(snapshot,uuid,owner,baseline)\\\\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\\\n  File \"/data/zypan/worktrees/ling-kda-4doc-cleanroom-gpu6-single-v1/cleanroom_run.py\", line 188, in checked\\\\n    return assess_lifecycle_sample(s,u,o,b,poll_before,poll_after)\\\\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\\\n  File \"/data/zypan/worktrees/ling-kda-4doc-cleanroom-gpu6-single-v1/lifecycle_guard.py\", line 65, in assess_lifecycle_sample\\\\n    safety=strict_snapshot_safety(snapshot,uuid,owner,baseline)\\\\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\\\n  File \"/data/zypan/worktrees/ling-kda-4doc-cleanroom-gpu6-single-v1/lifecycle_guard.py\", line 45, in strict_snapshot_safety\\\\n    assert m[\\\\\\'free\\\\\\']>=g.HEADROOM,\\\\\\'INSUFFICIENT_FREE_MEMORY_AND_2GIB_HEADROOM\\\\\\'\\\\n           ^^^^^^^^^^^^^^^^^^^^^\\\\nAssertionError: INSUFFICIENT_FREE_MEMORY_AND_2GIB_HEADROOM\\\\n\\'}')",
  "stage": "RUNNING_RESOURCE_SAFETY",
  "status": "STOP",
  "tag": "pg19-test-41603__FP__formal1",
  "task": "LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1"
}
```

出现实质身份/数值/资源失败后保留首STOP及已合格资产，不扩大诊断，不修数学或改变容差，本次单卡授权以外不启动后续Clean-Room尝试。E_future/Hidden KL及新proxy不计算；FP32载体仅声明persistent-QDQ/writeback，不声称physical resident INT8 storage。

## 五项科研判断

1. P1是否在四篇预先冻结文档上更好？本轮实际分类 **MIXED_OR_INCONCLUSIVE**，胜负、完整Delta与缺失如上；不以工程通过代替科学答案。
2. 与原2doc MIXED 1:1的关系？旧MIXED和posthoc永久保留。新四篇是同一冻结问题的独立前瞻扩展，条件筛选且n小，不合并确认性n6。
3. 是否有足够增量值得继续？本次没有完成四篇，尚不能回答P1增量预测价值；工程/资源停止不作为selector科学负例。
4. 没有稳定增量是否停止当前路线？若完整结果多为L7胜，应暂停当前P1 selector推进；若混合或不完整，尚不能作普遍无效结论，但没有继续训练授权。
5. 下一步科学验证？当前不自动开展训练或新实验；不足或混合的证据应先明确未回答问题，避免把工程失败或n4小样本当普遍结论。 仅科研建议，未执行任何后续训练/新数据/旋转/L8/L9/STE/Cayley/AIME/free generation/Qwen。

```json
{
  "TASK": "LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1",
  "EXECUTION_ID": "LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU6_SINGLE_V1",
  "MAX_CONCURRENT_GPUS": 1,
  "SCIENTIFIC_PROTOCOL_IDENTITY": "PASS",
  "ENGINEERING_IMPLEMENTATION": "CLEANROOM_CONTROL_ONLY",
  "GPU_RESOURCE_ADMISSION": "FAIL",
  "FLA_NUMERICAL_PROFILE": "PASS",
  "MODEL_AND_ROTATION_IDENTITY": "PASS",
  "FRESH64_QUALIFICATION": "12/12",
  "INT8_PERSISTENT_QDQ_QUALIFICATION": "PASS",
  "FORMAL_FP_COMPLETE": "0/4",
  "FORMAL_STUDENT_COMPLETE": "0/8",
  "ACTUAL_VALID_DOCUMENTS": "0/4",
  "PRIMARY_384_SCORE_REPRODUCTION": "NOT_RUN",
  "P1_WINS": 0,
  "L7_WINS": 0,
  "TIE_UNRESOLVED": 0,
  "MISSING_DOCUMENTS": 4,
  "NEW_GPU_HOURS_USED": 4.296229118945387,
  "PRIOR_SAME_TASK_RTX3090_GPUH": 1.007654354178602,
  "INDEPENDENT_RTX4090_QUALIFICATION_GPUH": 3.447347587855119,
  "MAX_GPU_HOURS": 36.0,
  "FINAL_SCIENTIFIC_CLASSIFICATION": "MIXED_OR_INCONCLUSIVE",
  "FINAL_EXECUTION_STATUS": "PARTIAL_STOP"
}
```
