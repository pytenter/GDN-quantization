# 独立资源与终态核验

本轮真实 FAIL_CLOSED_STOP；仅四卡各派发一次 B0_FP_Q0，未启动其余单位。提交 token 数为 [252, 253, 253, 256]。GPU3 完成 256 token 的 reference index 后实际 RESULT 为失败；它没有做 fresh 字节比较，不能标作数值资格通过。GPU0–2 没有 RESULT，保持部分结果原貌。

冻结事件 reducer 逐卡重算与四份最终 live ledger 完全一致：派发资源预留为 1.922697234406–1.923192716634 卡时。该范围从 worker 派发/启动计入，包括 CUDA 之前的源/权重 hash；不是精确物理 GPU 执行卡时。物理执行卡时 UNKNOWN。

首次至末次 NVML 进程存在观测跨度合计 1.765317892383 卡时，仅为描述性观测跨度，不能替代 GPU 忙碌积分。四卡派发至首次 NVML 进程出现合计 566.585999 秒。

| 卡 | 提交 token | 预留下界(h) | 预留上界(h) | 进程退出码 | 单位 RESULT |
|---|---:|---:|---:|---:|---|
|gpu0|252|0.480425875895|0.480546945925|-15|NO_RESULT_RECEIPT_PARTIAL|
|gpu1|253|0.480920398032|0.481052803923|-15|NO_RESULT_RECEIPT_PARTIAL|
|gpu2|253|0.480427895263|0.480547821129|-15|NO_RESULT_RECEIPT_PARTIAL|
|gpu3|256|0.480923065216|0.481045145657|-15|FAIL_CLOSED_STOP|

在 2026-10-19T05:51:53.394586+00:00 独立只读核验：NVML compute-apps 为空，任务 cmdline/旧 worker 与 coordinator PID 均不存在，任务 tmux session 已结束。初始实际 PID/starttick/cmdline、四次唯一首次退出/释放观测和 current release 证据已逐项绑定。未改冻结 ledger 或源程序，也未重试。
