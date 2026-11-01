# Ling/KDA 四文档 Clean-Room Final

独立TASK：LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1；原科学ID：LING_KDA_P1_L7_PROSPECTIVE_EXTENSION_4DOC_V1。
终态：FAIL_CLOSED_STOP。本轮仅更换控制层，旧execution01至05全部保留，未覆盖STOP/模型/CP，不启动下一轮尝试。

## 科学身份与执行

原prereg54d6d24fbc3142a692ef1a6c04a5617cf6b0abe6702a662722f34fc1b6141722，四篇freeze a5c227d15918a249f8ae2059539c15f94668cb2afddf3aecd8a9f7e13851354b，FROZEN_SPEC 8ae8927ed37209efccbf19c672b9baeb46c0d7f272601f8f54d1f9609b64f3d1均实际SHA PASS。100/20/22/58输入流程与4篇×16=64冻结评分不变，无重排、重选或新训练。32份权重、raw/token、旋转及数学源码实际SHA见G0/source manifest。源码commit：d03bd1b1122e93f89c5bc06b212b0b52b4974eed。

冻结数学worker文件字节保持；正式编译入口仅添加已登记的checkpoint/cursor/输出原子化控制节点，逐项可逆还原整个原始源码，未改model_step/QDQ/rotation/observer/KL/原Python sum384。旧执行05的双零13点规则不追溯改变；新任务使用3次间隔采样，≤2%差异只提示，前提是无外部compute/身份异常、显存稳定且按历史max(reserved,NVML峰值)+至少2GiB空余准入。实际资源安全仍独立监督，每个模型进程来源/profile/UUID验证。

每卡仅一次真实模型8-token canary，非科学轨迹。数值资格0/12；INT8资格INCOMPLETE，FP audit=None为N/A。同一所选UUID上的既有FP8192 CP只在当前GPU fresh64合同实际PASS后复用，设备不同时重建其FP前缀；INT8各有自己的从token0 persistent-QDQ历史。FLA Conv BD8/W8、Gated RMSNorm BT16/W8固定，五个启动变量在新解释器前设置。8192原fresh64、observer/nativecache/RNG精确合同比较未放宽。

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
  "new_GPU_hours": 0.0,
  "max_GPU_hours": 36.0,
  "actual_GPU_worker_count": 0,
  "peak_measured_CUDA_allocated": null,
  "peak_measured_CUDA_reserved": null,
  "CUDA_scope": "canary model-load+8token peak; frozen workers reset at loop; segment measurements aggregated; no universal lifetime peak claim",
  "NVML_periodic_peak_owned_bytes": null,
  "NVML_scope": "approximately10s exact owned PID samples, not instantaneous peak",
  "launch_to_exit_includes_load_restore_teardown": true
}

预算硬上限36新增GPUh；含canary、加载、parent、fresh64、正式两段和恢复的每worker launch-to-exit，双GPU分别累计；历史GPUh不计本轮额度。运行前和第一批真实parent后更新实际测量预测，保护性35.8GPUh提前停止留退出余量，没有提高36上限。所有实际CP和FP payload由manifest真实SHA绑定，大模型/CP只留服务器，不下载模型。最多两卡，未杀其他用户/重置GPU。

服务器目录：/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1；工作树：/data/zypan/worktrees/ling-kda-4doc-cleanroom-final-v1。source_manifest、engineering_change_manifest、resource_admission、qualification_matrix、run_manifest、gpu_resource_ledger和hashes保存真实记录。完整8×384逐点校验仅在实际完成时为PASS，部分状态详见机器字段与原始数组。

## 首失败及证据边界

```json
{
  "epoch": 1792043978.5036623,
  "no_retry": true,
  "reason": "AssertionError('UNAUTHORIZED_EXTERNAL_COMPUTE_PROCESS')",
  "stage": "DEPENDENCY_FAIL_CLOSED",
  "status": "STOP",
  "task": "LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1",
  "traceback": "Traceback (most recent call last):\n  File \"/data/zypan/worktrees/ling-kda-4doc-cleanroom-final-v1/cleanroom_run.py\", line 9, in protected\n    try:return fn(*args)\n               ^^^^^^^^^\n  File \"/data/zypan/worktrees/ling-kda-4doc-cleanroom-final-v1/cleanroom_run.py\", line 33, in job\n    admitted=g.admit(uuid,tag);assert admitted['graphics_baseline']==read(D/'resource_baselines.json')[uuid]\n             ^^^^^^^^^^^^^^^^^\n  File \"/data/zypan/worktrees/ling-kda-4doc-cleanroom-final-v1/gpu_admission.py\", line 64, in admit\n    result=assess_window(rows,uuid);result.update(epoch=time.time(),sample_path=str(path))\n           ^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/data/zypan/worktrees/ling-kda-4doc-cleanroom-final-v1/gpu_admission.py\", line 48, in assess_window\n    values=[assess(s,uuid) for s in rows];base=values[0]['graphics']\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/data/zypan/worktrees/ling-kda-4doc-cleanroom-final-v1/gpu_admission.py\", line 48, in <listcomp>\n    values=[assess(s,uuid) for s in rows];base=values[0]['graphics']\n            ^^^^^^^^^^^^^^\n  File \"/data/zypan/worktrees/ling-kda-4doc-cleanroom-final-v1/gpu_admission.py\", line 40, in assess\n    assert not (union-allowed),'UNAUTHORIZED_EXTERNAL_COMPUTE_PROCESS'\n           ^^^^^^^^^^^^^^^^^^^\nAssertionError: UNAUTHORIZED_EXTERNAL_COMPUTE_PROCESS\n"
}
```

出现实质身份/数值/资源失败后保留首STOP及已合格资产，不扩大诊断，不修数学或改变容差，不启动第二轮Clean-Room。E_future/Hidden KL及新proxy不计算；FP32载体仅声明persistent-QDQ/writeback，不声称physical resident INT8 storage。

## 五项科研判断

1. P1是否在四篇预先冻结文档上更好？本轮实际分类 **MIXED_OR_INCONCLUSIVE**，胜负、完整Delta与缺失如上；不以工程通过代替科学答案。
2. 与原2doc MIXED 1:1的关系？旧MIXED和posthoc永久保留。新四篇是同一冻结问题的独立前瞻扩展，条件筛选且n小，不合并确认性n6。
3. 是否有足够增量值得继续？本次没有完成四篇，尚不能回答P1增量预测价值；工程/资源停止不作为selector科学负例。
4. 没有稳定增量是否停止当前路线？若完整结果多为L7胜，应暂停当前P1 selector推进；若混合或不完整，尚不能作普遍无效结论，但没有继续训练授权。
5. 下一步科学验证？当前不自动开展训练或新实验；不足或混合的证据应先明确未回答问题，避免把工程失败或n4小样本当普遍结论。 仅科研建议，未执行任何后续训练/新数据/旋转/L8/L9/STE/Cayley/AIME/free generation/Qwen。

```json
{
  "TASK": "LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1",
  "SCIENTIFIC_PROTOCOL_IDENTITY": "PASS",
  "ENGINEERING_IMPLEMENTATION": "CLEANROOM_CONTROL_ONLY",
  "GPU_RESOURCE_ADMISSION": "FAIL",
  "FLA_NUMERICAL_PROFILE": "INCOMPLETE",
  "MODEL_AND_ROTATION_IDENTITY": "PASS",
  "FRESH64_QUALIFICATION": "0/12",
  "INT8_PERSISTENT_QDQ_QUALIFICATION": "INCOMPLETE",
  "FORMAL_FP_COMPLETE": "0/4",
  "FORMAL_STUDENT_COMPLETE": "0/8",
  "ACTUAL_VALID_DOCUMENTS": "0/4",
  "PRIMARY_384_SCORE_REPRODUCTION": "NOT_RUN",
  "P1_WINS": 0,
  "L7_WINS": 0,
  "TIE_UNRESOLVED": 0,
  "MISSING_DOCUMENTS": 4,
  "NEW_GPU_HOURS_USED": 0.0,
  "MAX_GPU_HOURS": 36.0,
  "FINAL_SCIENTIFIC_CLASSIFICATION": "MIXED_OR_INCONCLUSIVE",
  "FINAL_EXECUTION_STATUS": "FAIL_CLOSED_STOP"
}
```
