原根因任务的OLD_FP/NEW_FP中位TPS为13.742036/7.776711；OLD_H+C128/NEW_H+C128为9.064610/6.650598。历史mm CPU self约7.0764/55.8277ms/token，首次launch前约6.8070/55.3913ms，直接GPU GEMM约20.9284/20.9390ms。支持主机launch前开销方向，唯一根因仍UNRESOLVED。

这些数据均为历史参考，不是本轮四卡新测量；H的新旧prefill/cache协议不同，Dual-M的6.801314154 TPS也来自另一冻结任务，不能拼成当前严格同条件Codec比较。`PERFORMANCE_COMPARISON.csv`仅保留六条有来源SHA的历史实测记录。本轮测速0条，不计算提速率，不把保存参考轨迹的耗时当作解码TPS。完整历史审计保留于 `parent_cpu/HISTORICAL_EVIDENCE_REUSE.md`。
