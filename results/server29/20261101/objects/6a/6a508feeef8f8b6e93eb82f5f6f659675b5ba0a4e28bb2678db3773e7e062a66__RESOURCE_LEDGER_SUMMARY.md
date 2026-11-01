# 最终独立资源与进程核验

本恢复执行真实 STOP：实际12次派发，每卡B0_Q0、B0_Q1、B1_Q0各一次；12次首次退出与释放回执均唯一，四份终态ledger与冻结reducer逐项完全一致。8个B0完整单位PASS；GPU3的B1在token0出现严格字节差异；其余3个B1在模型加载/前置库核验阶段被共享STOP终止，未进入qualify，RESULT缺失继续保留缺失。没有B1_Q1、测速、H或Dual-M。

本次worker/设备预留范围：3.994544229197–3.997691420023 GPUh。此前失败预留仍单列：1.922697234406–1.923192716634 GPUh。累计预留范围：5.917241463603–5.920884136657 GPUh。全部采用同一55主机monotonic事件；范围含CUDA之前的CPU核验与初始化，不是精确物理GPU计算成本，物理精确GPUh保持UNRESOLVED。

在2026-10-19T07:53:06.561868+00:00只读核验，NVML compute-apps为空，四卡UUID匹配；所有本任务worker/controller/wrapper PID和cmdline均不存在，任务tmux已结束。其他tmux未触及。12个worker均有STARTED自报PID/starttick；8个完整单位还保留controller捕获的实际argv，4个B1无终态controller实际argv回执，明确标作未保留，不补造。

本审计未发起GPU工作、重启、修复、原始tensor复哈希或归档发布，未改冻结source/ledger。
