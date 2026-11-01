# Ling/KDA 四文档 Clean-Room Final

独立TASK：LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1；本次单卡执行身份：LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU3_SINGLE_V2；原科学ID：LING_KDA_P1_L7_PROSPECTIVE_EXTENSION_4DOC_V1。
终态：PARTIAL_STOP。本轮仅更换控制层，旧execution01至05及前次Clean-Room STOP全部保留，未覆盖STOP/模型/CP。本次由用户新明确授权仅GPU3串行运行，不自动增加后续尝试。

## 科学身份与执行

原prereg54d6d24fbc3142a692ef1a6c04a5617cf6b0abe6702a662722f34fc1b6141722，四篇freeze a5c227d15918a249f8ae2059539c15f94668cb2afddf3aecd8a9f7e13851354b，FROZEN_SPEC 8ae8927ed37209efccbf19c672b9baeb46c0d7f272601f8f54d1f9609b64f3d1均实际SHA PASS。100/20/22/58输入流程与4篇×16=64冻结评分不变，无重排、重选或新训练。32份权重、raw/token、旋转及数学源码实际SHA见G0/source manifest。源码commit：3c5837dcf61701661a935f3bbb3d2170a00f8d59。

冻结数学worker文件字节保持；正式编译入口仅添加已登记的checkpoint/cursor/输出原子化控制节点，逐项可逆还原整个原始源码，未改model_step/QDQ/rotation/observer/KL/原Python sum384。旧执行05的双零13点规则不追溯改变；新任务使用3次间隔采样，≤2%差异只提示，前提是无外部compute/身份异常、显存稳定且按历史max(reserved,NVML峰值)+至少2GiB空余准入。实际资源安全仍独立监督，每个模型进程来源/profile/UUID验证。

每卡仅一次真实模型8-token canary，非科学轨迹。数值资格2/12；INT8资格INCOMPLETE，FP audit=None为N/A。同一所选UUID上的既有FP8192 CP只在当前GPU fresh64合同实际PASS后复用，设备不同时重建其FP前缀；INT8各有自己的从token0 persistent-QDQ历史。FLA Conv BD8/W8、Gated RMSNorm BT16/W8固定，五个启动变量在新解释器前设置。8192原fresh64、observer/nativecache/RNG精确合同比较未放宽。

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
  "new_GPU_hours": 1.007654354178602,
  "max_GPU_hours": 36.0,
  "actual_GPU_worker_count": 6,
  "peak_measured_CUDA_allocated": 16935037440,
  "peak_measured_CUDA_reserved": 20667432960,
  "successful_PROFILE_records": [
    {
      "tag": "pg19-test-2544__FP__canary",
      "kind": "canary",
      "profile_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU3_SINGLE_V2/canary/gpu3.PROFILE.json",
      "profile_sha256": "7e14d385024d3f717cbebb0e157d71e6983f61e43eb75ebccfccc46fba2820e2",
      "scope": "last frozen loop reset through completion including checkpoint operations; canary resets before model load",
      "peak_allocated": 15856368640,
      "peak_reserved": 15864954880
    },
    {
      "tag": "pg19-test-41603__FP__fresh",
      "kind": "fresh",
      "profile_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU3_SINGLE_V2/runs/pg19-test-41603__FP__resumed_parity.PROFILE.json",
      "profile_sha256": "c95ac1a1b29b1b561b8acf8abbe3bed3bdad5fd7c0bcb9322bff21013eaa64e8",
      "scope": "last frozen loop reset through completion including checkpoint operations; canary resets before model load",
      "peak_allocated": 16486794240,
      "peak_reserved": 16783507456
    },
    {
      "tag": "pg19-test-41603__FP__parent",
      "kind": "parent",
      "profile_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU3_SINGLE_V2/runs/pg19-test-41603__FP__parent.PROFILE.json",
      "profile_sha256": "ba16d70952172889ca67ff13d54b336cc65093298fa3342fa88d0451ae7db680",
      "scope": "last frozen loop reset through completion including checkpoint operations; canary resets before model load",
      "peak_allocated": 16486794240,
      "peak_reserved": 20128464896
    },
    {
      "tag": "pg19-test-41603__H__parent",
      "kind": "parent",
      "profile_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU3_SINGLE_V2/runs/pg19-test-41603__H__parent.PROFILE.json",
      "profile_sha256": "0399f30d97c2a4d4b1895091948d313e9ad8a0fab39e8bc8100f95d9791e9fd1",
      "scope": "last frozen loop reset through completion including checkpoint operations; canary resets before model load",
      "peak_allocated": 16935037440,
      "peak_reserved": 20667432960
    },
    {
      "tag": "pg19-test-41603__H_to_L7_a50__fresh",
      "kind": "fresh",
      "profile_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU3_SINGLE_V2/runs/pg19-test-41603__H_to_L7_a50__resumed_parity.PROFILE.json",
      "profile_sha256": "3497dec7ad3997fcacba926ef3ff9366ba0444825ea82d668334ba8c3256692a",
      "scope": "last frozen loop reset through completion including checkpoint operations; canary resets before model load",
      "peak_allocated": 16935037440,
      "peak_reserved": 17265852416
    },
    {
      "tag": "pg19-test-41603__H_to_L7_a50__parent",
      "kind": "parent",
      "profile_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU3_SINGLE_V2/runs/pg19-test-41603__H_to_L7_a50__parent.PROFILE.json",
      "profile_sha256": "445ba423c69258e421bcb9989039341cc86ced3a40cf9cb8dcc4257b86ed66fc",
      "scope": "last frozen loop reset through completion including checkpoint operations; canary resets before model load",
      "peak_allocated": 16935037440,
      "peak_reserved": 20667432960
    }
  ],
  "partial_progress_records": [
    {
      "tag": "pg19-test-2544__FP__canary",
      "kind": "canary",
      "document": "pg19-test-2544",
      "condition": "FP",
      "worker_status": "PASS",
      "failed_or_incomplete": false,
      "source_log_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU3_SINGLE_V2/jobs/pg19-test-2544__FP__canary.log",
      "source_log_sha256": "47efc52e976b67554cf4e2ac1fdc43a8622985a9b9be3bee56319fdfbcd9a8db",
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
      "tag": "pg19-test-41603__FP__fresh",
      "kind": "fresh",
      "document": "pg19-test-41603",
      "condition": "FP",
      "worker_status": "PASS",
      "failed_or_incomplete": false,
      "source_log_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU3_SINGLE_V2/jobs/pg19-test-41603__FP__fresh.log",
      "source_log_sha256": "15d721486d4b68b9d739cfad0b84f62e20f6f61645b95870cdab060a4ece5b5d",
      "progress_row_count": 1,
      "invalid_progress_line_numbers": [],
      "last_recorded_progress": {
        "line_number": 39,
        "event": "qualification_progress",
        "tag": "pg19-test-41603__FP__resumed_parity",
        "position": 8256,
        "elapsed": 10.476819265983067,
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
      "source_log_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU3_SINGLE_V2/jobs/pg19-test-41603__FP__parent.log",
      "source_log_sha256": "f46488ba7da5308343ccde5541a58676626f703845f60f63cb913e5f237b12ed",
      "progress_row_count": 17,
      "invalid_progress_line_numbers": [],
      "last_recorded_progress": {
        "line_number": 55,
        "event": "qualification_progress",
        "tag": "pg19-test-41603__FP__parent",
        "position": 8256,
        "elapsed": 1127.2118417340098,
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
      "tag": "pg19-test-41603__H__parent",
      "kind": "parent",
      "document": "pg19-test-41603",
      "condition": "H",
      "worker_status": "PASS",
      "failed_or_incomplete": false,
      "source_log_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU3_SINGLE_V2/jobs/pg19-test-41603__H__parent.log",
      "source_log_sha256": "54562811cf13a8f7a103ffe78f2d9225d16aaeeda4ab6530970e7bb937c0bb3d",
      "progress_row_count": 17,
      "invalid_progress_line_numbers": [],
      "last_recorded_progress": {
        "line_number": 55,
        "event": "qualification_progress",
        "tag": "pg19-test-41603__H__parent",
        "position": 8256,
        "elapsed": 1162.154730549024,
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
      "source_log_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU3_SINGLE_V2/jobs/pg19-test-41603__H_to_L7_a50__fresh.log",
      "source_log_sha256": "b849dfcbdba6afacab2b3f0e1b12274254b174e992039527d9053fa0b3636eaf",
      "progress_row_count": 1,
      "invalid_progress_line_numbers": [],
      "last_recorded_progress": {
        "line_number": 39,
        "event": "qualification_progress",
        "tag": "pg19-test-41603__H_to_L7_a50__resumed_parity",
        "position": 8256,
        "elapsed": 11.418847665016074,
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
      "source_log_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU3_SINGLE_V2/jobs/pg19-test-41603__H_to_L7_a50__parent.log",
      "source_log_sha256": "22988d3122c5b10168541d54cbfa9c85790bb96c324ba2cd26fcd0004dc1f8c5",
      "progress_row_count": 17,
      "invalid_progress_line_numbers": [],
      "last_recorded_progress": {
        "line_number": 55,
        "event": "qualification_progress",
        "tag": "pg19-test-41603__H_to_L7_a50__parent",
        "position": 8256,
        "elapsed": 1145.6260352139943,
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
  "max_observed_CUDA_peak_reserved_bytes": 20667432960,
  "CUDA_scope": "maximum observed recorded counters across separately scoped successful PROFILEs and all actual progress logs; failed-worker observations are lower bounds through last logged position, not terminal or lifetime peaks; model loading/restore excluded after frozen-loop resets; canary has its separate pre-model-load reset; no full32K peak measured",
  "NVML_periodic_peak_owned_bytes": 21023948800,
  "NVML_scope": "periodic exact-owned PID samples plus exact PID/UID/starttime owned-memory observations from the failed raw evidence; foreign/child/graphics memory excluded; not an instantaneous peak",
  "STOP_resource_evidence": {
    "status": "PASS_FIRST_STOP_EVIDENCE_BOUND",
    "first_STOP": {
      "epoch": 1792118206.72933,
      "evidence_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU3_SINGLE_V2/FAILED_RESOURCE_SNAPSHOT.json",
      "evidence_sha256": "22cd8873acee4ed2f17732cca03bdb4bf612890bebc6f6e08ee05c67fa9d14a9",
      "forbiddenPIDs": [],
      "no_retry": true,
      "original_resource_error": {
        "repr": "AssertionError('MATERIAL_MEMORY_WITHOUT_ATTRIBUTABLE_COMPUTE_PID')",
        "traceback": "Traceback (most recent call last):\n  File \"/data/zypan/worktrees/ling-kda-4doc-cleanroom-gpu3-single-v2/resource_evidence.py\", line 114, in assess_with_evidence\n    return assess_fn(snapshot,uuid,owner,baseline)\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/data/zypan/worktrees/ling-kda-4doc-cleanroom-gpu3-single-v2/gpu_admission.py\", line 43, in assess\n    if m['used']>64*2**20:assert owner['pid'] in union,'MATERIAL_MEMORY_WITHOUT_ATTRIBUTABLE_COMPUTE_PID'\n                                 ^^^^^^^^^^^^^^^^^^^^^\nAssertionError: MATERIAL_MEMORY_WITHOUT_ATTRIBUTABLE_COMPUTE_PID\n",
        "type": "AssertionError"
      },
      "reason": "ResourceEvidenceError('RESOURCE_ASSESSMENT_FAILED_WITH_PINNED_EVIDENCE: {\\'type\\': \\'AssertionError\\', \\'repr\\': \"AssertionError(\\'MATERIAL_MEMORY_WITHOUT_ATTRIBUTABLE_COMPUTE_PID\\')\", \\'traceback\\': \\'Traceback (most recent call last):\\\\n  File \"/data/zypan/worktrees/ling-kda-4doc-cleanroom-gpu3-single-v2/resource_evidence.py\", line 114, in assess_with_evidence\\\\n    return assess_fn(snapshot,uuid,owner,baseline)\\\\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\\\n  File \"/data/zypan/worktrees/ling-kda-4doc-cleanroom-gpu3-single-v2/gpu_admission.py\", line 43, in assess\\\\n    if m[\\\\\\'used\\\\\\']>64*2**20:assert owner[\\\\\\'pid\\\\\\'] in union,\\\\\\'MATERIAL_MEMORY_WITHOUT_ATTRIBUTABLE_COMPUTE_PID\\\\\\'\\\\n                                 ^^^^^^^^^^^^^^^^^^^^^\\\\nAssertionError: MATERIAL_MEMORY_WITHOUT_ATTRIBUTABLE_COMPUTE_PID\\\\n\\'}')",
      "stage": "RUNNING_RESOURCE_SAFETY",
      "status": "STOP",
      "tag": "pg19-test-41603__H__parent",
      "task": "LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1"
    },
    "first_STOP_sha256": "0acd47e7c2317207751e42f487be5dc2bdf251807bf3c497ecd667084b126b41",
    "is_resource_gate_STOP": true,
    "artifacts": [
      {
        "path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU3_SINGLE_V2/FAILED_RESOURCE_SNAPSHOT.json",
        "bytes": 55292,
        "sha256": "22cd8873acee4ed2f17732cca03bdb4bf612890bebc6f6e08ee05c67fa9d14a9",
        "expected_SHA_from_STOP": "22cd8873acee4ed2f17732cca03bdb4bf612890bebc6f6e08ee05c67fa9d14a9"
      },
      {
        "path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU3_SINGLE_V2/jobs/pg19-test-41603__H__parent.RESOURCE_RAW.jsonl",
        "bytes": 6034313,
        "sha256": "e82851125948e4708d76029ac12a136280aea08f026b7ac15cc98e85cb37db12",
        "expected_SHA_from_STOP": null
      }
    ],
    "pinned_failures": [
      {
        "path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU3_SINGLE_V2/FAILED_RESOURCE_SNAPSHOT.json",
        "sha256": "22cd8873acee4ed2f17732cca03bdb4bf612890bebc6f6e08ee05c67fa9d14a9",
        "status": "FAIL",
        "snapshot_sha256": "dc688dba45a0226b746656806d086dfde67cb7b02d545d430276e9d9b74b8f94",
        "raw_reference": {
          "bytes": 53931,
          "line_sha256": "a0cd142b0777c5362014ddb8123349020c9371b5804be7121cc259a66c6b593d",
          "offset": 5980382,
          "path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU3_SINGLE_V2/jobs/pg19-test-41603__H__parent.RESOURCE_RAW.jsonl"
        },
        "raw_line_and_snapshot_SHA_verified": true,
        "original_error": {
          "repr": "AssertionError('MATERIAL_MEMORY_WITHOUT_ATTRIBUTABLE_COMPUTE_PID')",
          "traceback": "Traceback (most recent call last):\n  File \"/data/zypan/worktrees/ling-kda-4doc-cleanroom-gpu3-single-v2/resource_evidence.py\", line 114, in assess_with_evidence\n    return assess_fn(snapshot,uuid,owner,baseline)\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/data/zypan/worktrees/ling-kda-4doc-cleanroom-gpu3-single-v2/gpu_admission.py\", line 43, in assess\n    if m['used']>64*2**20:assert owner['pid'] in union,'MATERIAL_MEMORY_WITHOUT_ATTRIBUTABLE_COMPUTE_PID'\n                                 ^^^^^^^^^^^^^^^^^^^^^\nAssertionError: MATERIAL_MEMORY_WITHOUT_ATTRIBUTABLE_COMPUTE_PID\n",
          "type": "AssertionError"
        },
        "diagnostic_classification": {
          "acceptance_predicate_unchanged": true,
          "allowed_owner_pid": 2974124,
          "ancestry_or_same_UID_never_authorizes": true,
          "compute_union": [],
          "forbidden_pids": [],
          "parse_errors": [],
          "processes": [],
          "status": "COMPLETE_EXACT_COMPUTE_UNION"
        },
        "forbidden_pids": [],
        "owner": {
          "boot_id": "9e80e411-1604-4937-803f-b78c74b28fe8",
          "nspid": [
            2974124
          ],
          "pid": 2974124,
          "start_ticks": 51603262,
          "uid": 1022
        },
        "ancestry_or_same_UID_does_not_authorize": true
      }
    ],
    "in_worker_failure_bindings": [],
    "admission_failure_bindings": [],
    "observed_process_records": [
      {
        "artifact_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU3_SINGLE_V2/FAILED_RESOURCE_SNAPSHOT.json",
        "epoch": 1792118206.239932,
        "uuid": "GPU-759526ed-efb2-e7f6-881c-ab098e0b05ae",
        "observation_status": "OBSERVED_COMPLETE_PROCESS_APIS",
        "NVML_compute_PIDs": [],
        "SMI_non_graphics_PIDs": [],
        "owned_NVML_usedGpuMemory_bytes": null,
        "owned_process_snapshot_identity_status": "UNKNOWN_METADATA_NOT_CAPTURED",
        "foreign_process_memory_excluded": true,
        "unauthorized_PIDs_under_unchanged_exact_owner_policy": [],
        "unauthorized_process_metadata": [],
        "exact_owned_process_identity": {
          "boot_id": "9e80e411-1604-4937-803f-b78c74b28fe8",
          "nspid": [
            2974124
          ],
          "pid": 2974124,
          "start_ticks": 51603262,
          "uid": 1022
        },
        "posthoc_data_used": false
      }
    ],
    "exact_trigger_PID_or_ownership_not_captured": "SEE_LINKED_RAW_OBSERVATIONS_WITH_EXPLICIT_API_STATUS",
    "current_or_posthoc_samples_substituted": false,
    "errors": []
  },
  "resource_evidence_errors": [],
  "launch_to_exit_includes_load_restore_teardown": true,
  "prior_same_task_RTX3090_GPUh": 0.0,
  "current_execution_RTX3090_GPUh": 1.007654354178602,
  "same_task_cumulative_RTX3090_GPUh": 1.007654354178602,
  "remaining_same_task_RTX3090_GPUh": 34.992345645821395,
  "MAX_SAME_TASK_RTX3090_GPUh": 36.0,
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

预算硬上限36同任务3090新增GPUh；此前同任务3090实测0GPUh。独立4090资格任务已报告3.447347587855119GPUh仅单列历史身份，不扣本任务3090额度。含canary、加载、parent、fresh64、正式两段和恢复的每worker launch-to-exit，仅GPU3按真实launch-to-exit累计；历史GPUh不计本轮额度（前次Clean-Room新增0GPUh）。运行前和第一批真实parent后更新实际测量预测，保护性35.8GPUh提前停止留退出余量，没有提高36上限。所有实际CP和FP payload由manifest真实SHA绑定，大模型/CP只留服务器，不下载模型。仅一张GPU3，未杀其他用户/重置GPU。

服务器目录：/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU3_SINGLE_V2；工作树：/data/zypan/worktrees/ling-kda-4doc-cleanroom-gpu3-single-v2。source_manifest、engineering_change_manifest、resource_admission、qualification_matrix、run_manifest、gpu_resource_ledger和hashes保存真实记录。完整8×384逐点校验仅在实际完成时为PASS，部分状态详见机器字段与原始数组。

## 首失败及证据边界

失败partial worker的最后确认消费位置与counter下界见TERMINAL_PARTIAL_PROGRESS_AUDIT；未记录的实际终止cursor为UNKNOWN，不按耗时推断。资源摘要max_observed合并成功PROFILE和失败日志累计counter，不能冒充完整生命周期峰值。失败pin与精确raw字节offset/line SHA、原worker或唯一admission关联见TERMINAL_RESOURCE_EVIDENCE_AUDIT；查询失败/丢失保持UNKNOWN及error，不用posthoc替代。

```json
{
  "epoch": 1792118206.72933,
  "evidence_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU3_SINGLE_V2/FAILED_RESOURCE_SNAPSHOT.json",
  "evidence_sha256": "22cd8873acee4ed2f17732cca03bdb4bf612890bebc6f6e08ee05c67fa9d14a9",
  "forbiddenPIDs": [],
  "no_retry": true,
  "original_resource_error": {
    "repr": "AssertionError('MATERIAL_MEMORY_WITHOUT_ATTRIBUTABLE_COMPUTE_PID')",
    "traceback": "Traceback (most recent call last):\n  File \"/data/zypan/worktrees/ling-kda-4doc-cleanroom-gpu3-single-v2/resource_evidence.py\", line 114, in assess_with_evidence\n    return assess_fn(snapshot,uuid,owner,baseline)\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/data/zypan/worktrees/ling-kda-4doc-cleanroom-gpu3-single-v2/gpu_admission.py\", line 43, in assess\n    if m['used']>64*2**20:assert owner['pid'] in union,'MATERIAL_MEMORY_WITHOUT_ATTRIBUTABLE_COMPUTE_PID'\n                                 ^^^^^^^^^^^^^^^^^^^^^\nAssertionError: MATERIAL_MEMORY_WITHOUT_ATTRIBUTABLE_COMPUTE_PID\n",
    "type": "AssertionError"
  },
  "reason": "ResourceEvidenceError('RESOURCE_ASSESSMENT_FAILED_WITH_PINNED_EVIDENCE: {\\'type\\': \\'AssertionError\\', \\'repr\\': \"AssertionError(\\'MATERIAL_MEMORY_WITHOUT_ATTRIBUTABLE_COMPUTE_PID\\')\", \\'traceback\\': \\'Traceback (most recent call last):\\\\n  File \"/data/zypan/worktrees/ling-kda-4doc-cleanroom-gpu3-single-v2/resource_evidence.py\", line 114, in assess_with_evidence\\\\n    return assess_fn(snapshot,uuid,owner,baseline)\\\\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\\\n  File \"/data/zypan/worktrees/ling-kda-4doc-cleanroom-gpu3-single-v2/gpu_admission.py\", line 43, in assess\\\\n    if m[\\\\\\'used\\\\\\']>64*2**20:assert owner[\\\\\\'pid\\\\\\'] in union,\\\\\\'MATERIAL_MEMORY_WITHOUT_ATTRIBUTABLE_COMPUTE_PID\\\\\\'\\\\n                                 ^^^^^^^^^^^^^^^^^^^^^\\\\nAssertionError: MATERIAL_MEMORY_WITHOUT_ATTRIBUTABLE_COMPUTE_PID\\\\n\\'}')",
  "stage": "RUNNING_RESOURCE_SAFETY",
  "status": "STOP",
  "tag": "pg19-test-41603__H__parent",
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
  "EXECUTION_ID": "LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU3_SINGLE_V2",
  "MAX_CONCURRENT_GPUS": 1,
  "SCIENTIFIC_PROTOCOL_IDENTITY": "PASS",
  "ENGINEERING_IMPLEMENTATION": "CLEANROOM_CONTROL_ONLY",
  "GPU_RESOURCE_ADMISSION": "FAIL",
  "FLA_NUMERICAL_PROFILE": "PASS",
  "MODEL_AND_ROTATION_IDENTITY": "PASS",
  "FRESH64_QUALIFICATION": "2/12",
  "INT8_PERSISTENT_QDQ_QUALIFICATION": "INCOMPLETE",
  "FORMAL_FP_COMPLETE": "0/4",
  "FORMAL_STUDENT_COMPLETE": "0/8",
  "ACTUAL_VALID_DOCUMENTS": "0/4",
  "PRIMARY_384_SCORE_REPRODUCTION": "NOT_RUN",
  "P1_WINS": 0,
  "L7_WINS": 0,
  "TIE_UNRESOLVED": 0,
  "MISSING_DOCUMENTS": 4,
  "NEW_GPU_HOURS_USED": 1.007654354178602,
  "PRIOR_SAME_TASK_RTX3090_GPUH": 0.0,
  "INDEPENDENT_RTX4090_QUALIFICATION_GPUH": 3.447347587855119,
  "MAX_GPU_HOURS": 36.0,
  "FINAL_SCIENTIFIC_CLASSIFICATION": "MIXED_OR_INCONCLUSIVE",
  "FINAL_EXECUTION_STATUS": "PARTIAL_STOP"
}
```
