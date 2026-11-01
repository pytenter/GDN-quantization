历史OLD_FP/NEW_FP中位TPS13.742036/7.776711，OLD_H/NEW_H为9.064610/6.650598；另一个Dual-M任务约6.801314154。旧新H存在prefill/cache协议差异。这些仅历史参考，不能拼成本轮同条件Codec因果对照。

原Profiler每token249个mm，CPU self约7.0764→55.8277ms、首次launch前6.8070→55.3913ms，而直接GPU GEMM约20.9284/20.9390ms；支持主机launch前开销方向，唯一内部根因仍UNKNOWN。本轮在数值门禁停止，新增可合法比较吞吐为0条。PERFORMANCE_COMPARISON.csv仅六条带原来源SHA的历史记录，没有把逐token保存/读回、CPU哈希或整体预留耗时当模型TPS。
