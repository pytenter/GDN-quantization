# Ling/KDA Rotation × Dual-Axis 最终科学报告

TASK_ID：`LING_KDA_ROTATION_DUAL_AXIS_FUNCTIONAL_V1`
FINAL_EXECUTION_STATUS：`FAIL_CLOSED_STOP`
SCIENTIFIC_CLASSIFICATION：`INCOMPLETE`
GPU_HOURS_USED：`1.2779262566580696`
ABSOLUTE_REPORT_PATH：`/data/zypan/experiment_runs/ling-kda-dual-axis-resume-01/LING_KDA_ROTATION_DUAL_AXIS_FUNCTIONAL_V1/FINAL_SCIENTIFIC_REPORT.md`

实际完成文档 0/6；完整正式轨迹 0/30。缺失是NOT_RUN/INCOMPLETE，绝不作为0或平局。

## 冻结方法和解释范围

原native KDA model_step、FLA、R128与H/L6矩阵保持原值。旋转只在Value侧codec坐标中进行，QDQ后逆变换回原生state，下一token消费真实写回；不在任意旋转Key坐标中保留原对角gate。
数学结构已审计，但冻结矩阵与FP32往返含有限精度差异，因此仅讨论实际系统operational codec effect，不声称严格纯量化误差分解。实际无量化H/L6 canary差异见GPU_QUALIFICATION及资格worker记录。
正式输入为官方PG19 validation新split，未将旧PG19 test100当未见样本；未找到可信RepoBench-P历史池，不能虚称数据源。所有候选、排除、原始SHA、U32LE8449token前缀SHA与确定性seed0排序已前瞻冻结。项目独立性UNKNOWN。
每条从0建立自己的8448-token历史；评分t8192..8447，fullV157184原FP32 KL(FP_NATIVE||student)在CPU执行同运算顺序，math.fsum/256。独立扩展精度CPU参考产生本轮算术分辨范围，不套旧Ling/Qwen阈值，不是完整forward误差上界。

## 每篇功能质量和交互

|文档|H_R128|H_DUAL_M|L6_R128|L6_DUAL_M|Delta_L6|Delta_H|Interaction|
|---|---:|---:|---:|---:|---:|---:|---:|
|pg19-validation-44896|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|
|pg19-validation-38214|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|
|pg19-validation-54098|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|
|pg19-validation-42067|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|
|pg19-validation-53682|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|
|pg19-validation-51859|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|

完整精度、两种LearnedGain、relative gain、mean/median/方向计数与unresolved标签见SCIENCE_RESULTS.json和INTERACTION_ANALYSIS.csv。

## 科学判断

Primary：{'paired_documents': 0, 'mean': None, 'median': None, 'positive': 0, 'negative': 0, 'numerically_unresolved': 0, 'missing': 6}。仅当6篇完整配对且可分辨Delta_L6全部正向时为PRIMARY_DIRECTIONALLY_REPLICATED。
关键Secondary Interaction：{'paired_documents': 0, 'mean': None, 'median': None, 'positive': 0, 'negative': 0, 'numerically_unresolved': 0, 'missing': 6}，不升格Primary。
L6+DualM是否绝对最佳须同时比较H+DualM及其分辨范围，不能从Delta_L6正向直接推出。实际每篇最佳条件与分辨标签已单列。
local-QDQ MSE来自各自历史state，只是轻量探索性诊断；与Future-KL的并行变化不能被称为同输入codec因果效应或唯一KDA中介。没有 matched-state干预时不作该因果推断。
每篇local_mechanism_observations列出实际probe平均local-MSE、各codec自身历史的MSE差及与功能KL方向是否一致；未记录/不可分辨时为null，不把方向一致解释成唯一中介。
项目独立性UNKNOWN，因此不报告确认性sign-test p值，不将256token/layer/head当独立样本。Incomplete不回答正负科学结论。
只有本次完整功能证据可以作为后续新rotation-loss研究的探索性依据；不自动授权新loss训练。没有32K、AIME、reasoning accuracy、部署packedINT8或KDA/GDN相同机制结论。

## 资源、失败与未运行项

FP32 persistent state载体保存的是实际QDQ写回值，不是物理packedINT8部署。INT8 codes与FP16 row/column metadata成本仅作为codec逻辑payload记录。
独立新任务预算16GPUh，15.5保护；本轮账本用量1.2779262566580696GPUh，含模型加载、forward、评分、checkpoint、GPU占用期间IO与cleanup证明。
CUDA/NVML的真实scope、峰值和unknown以RUNTIME_COST/GPU_RESOURCE_LEDGER及worker记录为准，不将周期采样冒充瞬时峰值。
首失败保留原记录，未自动重试、更换GPU或降低2GiB余量。CPU准备fixture/TLS机械修正的原失败与限定变更证据保留在audits，不冒称GPU科学重试。
```json
{
  "TASK_ID": "LING_KDA_ROTATION_DUAL_AXIS_FUNCTIONAL_V1",
  "FINAL_EXECUTION_STATUS": "FAIL_CLOSED_STOP",
  "SCIENTIFIC_CLASSIFICATION": "INCOMPLETE",
  "GPU_HOURS_USED": 1.2779262566580696,
  "FORMAL_TRAJECTORIES_COMPLETED": 0,
  "FORMAL_TRAJECTORIES_PLANNED": 30,
  "COMPLETE_DOCUMENTS": 0,
  "DOCUMENTS_PLANNED": 6,
  "NUMERICAL_QUALIFICATION_COMPLETE": false,
  "FORMAL_SCIENCE_COMPLETE": false,
  "PROJECT_INDEPENDENCE": "UNKNOWN",
  "PHYSICAL_PACKED_INT8_VALIDATED": false,
  "FIRST_FAILURE": {
    "active_job": {
      "NVML_owned_periodic_peak_bytes": 17641242624,
      "condition": "H_R128",
      "document": "pg19-validation-34016",
      "kind": "qual_fresh",
      "launch_epoch": 1792164105.070241,
      "launch_monotonic": 563117.31868359,
      "result_path": "/data/zypan/experiment_runs/ling-kda-dual-axis-resume-01/LING_KDA_ROTATION_DUAL_AXIS_FUNCTIONAL_V1/qualification/pg19-validation-34016__H_R128.qual_fresh.json",
      "tag": "pg19-validation-34016__H_R128.qual_fresh"
    },
    "details": {
      "returncode": 1,
      "worker_error_path": "/data/zypan/experiment_runs/ling-kda-dual-axis-resume-01/LING_KDA_ROTATION_DUAL_AXIS_FUNCTIONAL_V1/qualification/pg19-validation-34016__H_R128.qual_fresh.ERROR.json"
    },
    "epoch": 1792164151.5141666,
    "evidence": null,
    "no_retry": true,
    "owner": {
      "boot_id": "9e80e411-1604-4937-803f-b78c74b28fe8",
      "nspid": [
        3674662
      ],
      "pid": 3674662,
      "start_ticks": 56311731,
      "uid": 1022
    },
    "reason": "WORKER_NONZERO_EXIT",
    "status": "FAIL_CLOSED_STOP",
    "task_id": "LING_KDA_ROTATION_DUAL_AXIS_FUNCTIONAL_V1"
  },
  "CPU_ANALYSIS_ERRORS": []
}
```

**完成后STOP：未启动新实验、AIME、训练、16K/32K或git push。**
