# 完整Phase0后的根决定

Case C：真实128-token H_DUAL_M profile中，模型forward CPU owner范围27.217554568秒，占完整28.763418689秒的94.621%；QDQ与原安全检查owner为1.226533067秒（4.264%）。真实24层batched LS设备活动中位599.5825微秒/token，仅占encode设备活动17.5767%，不能当作完整wall；完整模型存在3767个真实kernel活动/token，主要CPU self耗时在aten::mm、model_forward与cudaLaunchKernel。三次非instrumented H_DualM中位6.801314154 token/s，与H_C128的6.662540440接近且未观察到LS主导的额外端到端耗时。故不实施codec融合候选，不更改模型/GDN数学，按授权止损；512/2048扩展与候选A/B/C资格均NOT_RUN。删除整个instrumented QDQ owner仅有条件估计1.044541倍，非instrumented严格加速上限仍UNKNOWN，不能承诺生产加速。

同一次instrumented H_DualM的三个原始非嵌套owner范围如下；设备归属量和CPU范围耗时不是相同口径，不能相加为wall，也不能再加其ATen子项。

|范围|count|CPU total us|device attributed us|
|---|---:|---:|---:|
|REFERENCE/model_forward|128|27217554.568|3626552.529|
|REFERENCE/QDQ_and_original_safety|128|1226533.067|607834.942|
|REFERENCE/finite_state_and_logits_gate|128|290216.460|3958.741|

单层真实state的LS grouped device归属 55.008 us；占八个命名stage归属总量 18.814%。该局部比例不是24层完整模型wall占比，不能乘24替代实测。严格非instrumented Amdahl ceiling仍按BASELINE_RUNTIME_PROFILE记录UNKNOWN；instrumented删除整个QDQ owner的条件估计不是候选可达保证。

性能记录保留CPU发射、CUDA活动区间union、内核名/count和同步信息；wall减GPU active union不被当成纯CPU overhead。任务ledger包括加载、warmup、采集、CPU Kineto解析、Chrome trace导出、真实释放前全部GPU占用，不能只报告采集窗口耗时。


最终精确数值和根决定94.621%转录勘误见 FINAL_TECHNICAL_REPORT.md、ROOT_DECISION_PERCENT_CORRECTION.json 及 REPORT_EVIDENCE_SUPPLEMENT.md。真实全24层LS 128范围唯一关联的设备活动中位599.5825us/token，encode活动占17.576735%；不是完整LS wall。
